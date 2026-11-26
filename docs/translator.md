# Machine code for sm_75 made machine code for sm_70

The translator turns a cubin compiled for Turing (sm_75) into one that runs on
Volta (sm_70), function by function. It also translates a subset of stripped
Ampere (sm_80) cubins (section 1, "What is translated"). This document says what
it does, why each rule is there, and where to look in the code.

For the instructions it emulates (the tensor-core products, the conversions, the
arithmetic on bf16), see `docs/emulations.md`. For what translation costs in
time, see `docs/performance.md`, "Where the time goes". For how a wrong
translation would be found, see `docs/validation.md`.


## 1. Why machine code

CUDA 13 no longer makes code for sm_70, and many programs ship only machine code
for sm_75 and later. A stock PyTorch 2.14 wheel for CUDA 13 carries 475 cubins
of sm_75 machine code and no PTX at all. NVIDIA's own libraries for CUDA 13
(cuBLAS, cuBLASLt, cuSOLVER, cuSPARSE, cuRAND, NPP, nvJPEG) do the same: their
sm_75 cubins are the only code they have for a device that is told it is 7.5.
When a program has PTX, the layer's PTX lowering (`crates/vf-ptx`) rewrites
what Volta lacks and the driver compiles the result. When it has only machine
code, the translator is the only path.

**What is translated.** VF translates Turing (sm_75) executables and a subset of
Ampere (sm_80) executables. A program that also carries compatible PTX uses the
PTX path by default. `VF_TRANSLATE=first` gives machine code precedence.

For sm_80, an instruction passes to the Turing translator as it is when the two architectures'
instruction descriptions give it the same reading, by one of two tables that
`reference/ampere_forms.py` derives from them:

- **Audited forms** (`reference/ampere-forms.txt`): a class with its modifiers fixed, of which
  the generator checks the opcode, the modifiers and the positions of the variable operands on
  both sides.
- **Whole classes, in the forms that ran right** (since 2026-11-25). `whole_classes` finds the
  classes of sm_80 for which every bit of the instruction holds, in the sm_75 class of the same
  name, the same thing: the same fixed value, or the same bit of the same operand, of a type
  whose values read the same on both (every row of an encoding table of sm_80 a row of sm_75's;
  a register, an immediate of the same width and sign; an enumeration whose values have the same
  names, or else only the values that do, one form for each). An operand that one side has and
  the other fixes must be at its default there (ISETP's `EX` and its third predicate, which
  sm_80 splits into two classes). A class whose words could be of another class that is not
  kept whole, on either side, is left out, so that both decoders name the same instruction for
  every word of a kept class: 680 of sm_80's 962 classes (1,444 forms). That is necessary and
  **was not enough**: with every such class taken, three double-precision kernels that PyTorch
  compiles for sm_80 while it runs (erfc, erfinv, exp2) were translated and gave wrong results
  on a V100 (`tests/gpu/torch.sh` told 8.0 with `VF_CAPABILITY_ALL=1`: 3 of 925 cases
  different), although every word of them reads the same as sm_80 and as sm_75
  (`reference/ampere_check.py`): what differs between the two architectures there is not an
  instruction but when its result may be read (section 7, "A result of double precision";
  `docs/validation.md`, section 8). So a word passes by its class only in a form listed in
  `reference/ampere-evidence.txt`: the class with the values of all its modifiers (registers,
  predicates and immediates free), each seen in translated sm_80 code that gave the right
  results on a V100 against a native build or a host computation
  (`reference/ampere_evidence.py` prints the forms of a kernel). 74 forms of 43 classes at
  2026-11-25, from the kernels of `tests/kernels/k_told.cu`, vLLM's bf16 norm and the float32
  kernels of erfc, erfinv and exp2; none of double precision, which the generator refuses
  until a translated kernel with it ran right. A form is added with the kernel that shows it.

The translator lowers global loads and stores through a memory descriptor (every size), paired
FP32-to-FP16 conversions, and 16-byte asynchronous copies. Each copy uses a four-register
temporary for a global load followed by a shared store. Both instructions preserve the original
predicate. Global memory accesses require a proven system descriptor: on every path of the
function, or, for a descriptor that nothing but its own load ever writes anywhere in the
function, on every path of a thread (a thread reaches the place where the threads of a `BSSY`
meet again by the `BSYNC` before it or by a branch, not from the `BSSY`, so a `BSSY` placed
before the load does not undo it). The supported GPU-scope 32-bit atomic
add preserves the previous value. A shared counter increment with a discarded result becomes a
scaled address calculation and a predicated Volta shared atomic add. VF checks each repeated
warp convergence barrier against the branch target on every incoming path. Two instructions of
sm_80 for bfloat16 are rewritten (`docs/emulations.md`, "What is not in machine code"):
`HFMA2.BF16_V2` with a multiplier of one in both halves (the sum of two pairs of bf16) and
`F2FP.BF16.PACK_AB` (two f32 rounded to bf16), each in three registers of the translation's own.

A kernel can call one private helper at the end of its section when each direct call establishes
the return address and the helper preserves its return register, and returns through it or through
an unguarded copy of it that no branch can skip. VF recognizes local helpers and
the weak functions that CUDA 12’s assembler places in the local portion of the kernel’s symbol
table. A call invalidates any descriptor or convergence barrier the helper can change.
Supported functions then pass through the existing register allocator and Volta scheduler.
A function outside this subset is refused at lookup. The supported forms are
listed in `reference/ampere-forms.txt` and `reference/ampere_forms.py`.

The sm_80 path is `crates/vf-sass/src/translate/ampere.rs`, on the forms that
`reference/generate.py` writes into `crates/vf-tables/src/ampere.rs`, and
`reference/ampere.py` in the reference. The independent validator
(`reference/validate.py`) does not check it: it reads every original with the
sm_75 descriptions. What checks the sm_80 path is the agreement of the two
translators (`tests/sass/compare.sh`, whose test kernels include sm_80 cubins),
`reference/ampere_check.py` (every word that passes as it is, read by NVIDIA's disassembler as
sm_80 and as sm_75), unit tests (with a host model of the bf16 sequences), the sm_80 cases of
`tests/gpu/run.sh`, and engines run by hand.
`docs/validation.md`, section 8, says what this leaves out.

**Where the translation happens.** The proxy (`crates/vf-proxy`, by the
decisions of `crates/vf-adapt`) calls the translator when a program loads a
module whose only usable code is supported machine code. The translation is done on the host CPU, before the driver sees the
module. It is also available as a command:

```
vf translate INPUT.cubin OUTPUT.cubin [MAP [REPORT]]
```

MAP gets one line per instruction of the result (function, new offset, old
offset, what was done), and `MAP.lives` beside it what `reference/validate.py`
reads with the map (where the uniform predicates live, what was borrowed where),
as `reference/translate.py` writes them. REPORT gets one line per function
(`ok`, or why it was refused). `VF_BORROW`, `VF_PAIRS` and `VF_ALIGNED_ROWS`
apply as in the layer.


## 2. The two instruction sets

Volta (sm_70) and Turing (sm_75) share the same 128-bit instruction encoding
and much of the same instruction set. In PyTorch's 475 cubins, 94.9 % of the
29.9 million sm_75 instructions (2.9 million distinct instruction words) are,
bit for bit, instructions of the sm_70 description: nvdisasm prints the same
text for them under either architecture (`reference/both.py`, 2026-11-20).
These instructions pass through the translator unchanged.

What sm_75 has and sm_70 lacks:

| Feature | Share of sm_75 instructions | What the translator does |
|---|---:|---|
| Uniform registers (UR0-UR63) and predicates (UP0-UP6): instructions on them (`UIADD3`, `ULDC`, `UPLOP3` ...) and ordinary instructions with a uniform operand | 2.0 % | Each has a rule to its sm_70 twin; uniform registers placed in ordinary registers |
| `IMAD.HI` (the upper half of a product-sum) | 2.4 % | `IMAD.WIDE` into two registers, then `MOV` of the upper half |
| Memory at `[register * 4, 8 or 16]` in shared memory | 0.5 % | `SHF.L` of the register into a kept copy, then the plain address form |
| `IMNMX` (integer minimum and maximum) | 0.12 % | `ISETP` into a dead predicate, then `SEL` |
| Turing's tensor-core instructions (`HMMA.1688`, `LDSM`) | 0.02 % | Volta's own products (`HMMA.884`) and shuffles; see section 6 |
| A cache hint on loads (`LDG.E.LTC128B`) | 0.02 % | The load is rebuilt with sm_70's table, so the hint is dropped; a zero-fill predicate (`Pnz`) is kept (section 2) |

The shares are of all the instructions of PyTorch's 475 cubins (by nvdisasm's
text of each word that is no sm_70 instruction; an instruction with a uniform
operand is counted with the uniform registers, whatever else it is). The one
other instruction that sm_70 lacks there is `I2IP`, 384 times (section 8).


### Uniform registers and predicates

Turing added a uniform register file (UR0 to UR63) and uniform predicates (UP0
to UP6) with a dedicated ALU, for values that are the same across a warp.
The compiler uses them for kernel parameters, loop counters, block dimensions
and addresses derived from them.

A V100 has no uniform register file. The translator gives each uniform register
an ordinary register above those the function uses (a 64-bit pair stays on an
even register and the next), and each uniform predicate an ordinary predicate
that holds nothing of the program during its life. Each instruction on uniform
registers is then rewritten to its ordinary twin using `reference/rules-sm75.tsv`.
For each rule, nvdisasm must read the translated word on sm_70 exactly as it read
the original on sm_75, with the uniform registers renamed. PyTorch's 475 cubins
exercise 75 rules across 393,803 instructions. The Triton gated delta rule
kernel used by vLLM exercises two further forms, `USGXT.U32` with an immediate
and `LEA.HI` with a uniform source. Its unified attention kernel also uses
`ULEA.HI.SX32`, which maps to `LEA.HI.SX32` on ordinary registers. SGLang's
all-reduce kernel for four ranks uses `ULEA.HI` with an immediate
(`ULEA.HI UR4, UR4, 0x1, URZ, 0x1e`), which maps to `LEA.HI` with the same
operands. No cubin kept here has a `USGXT` (387 Triton cubins were searched,
2026-11-22). The words of these kernels, and seven more with other fields (a
third register that is not `URZ`, a negated operand, a guard, the other forms of
`USGXT`), each as nvdisasm reads it on both sides, are the unit test of these
rules (`remap_uniform_operands_in_triton_kernels` in `crates/vf-sass/src/rule.rs`);
a `USGXT` put into a test kernel goes through the whole translator
(`a_uniform_sign_extension_becomes_sgxt` in `crates/vf-sass/src/translate/tests.rs`).
`UIMAD` with an immediate addend (`UIMAD UR4, UR4, UR5, 0x240`, the form of
issue #8) and the forms with a carry in, `UIMAD.X` and `UIMAD.WIDE.X`, map to
`IMAD`, `IMAD.X` and `IMAD.WIDE.X` with the same operands (pull request #15).
Seven words of these forms,
each as nvdisasm reads it on both sides, are the unit test
`remap_uimad_and_carry_variants` in `crates/vf-sass/src/rule.rs`;
`tests/kernels/translate/k_uimad.cu` (the immediate form and `UIMAD.WIDE`) goes
through the whole translator and runs on a V100, and a `UIMAD.X` and a
`UIMAD.WIDE.X` put into test kernels go through the whole translator
(`crates/vf-sass/src/translate/tests.rs`).

An instruction with a uniform operand (a load at `[R4 + UR6 + 0x10]`, a
comparison against UR2) is rewritten to the sm_70 form with an ordinary
register in its place.


### Instructions that differ in encoding

`IMAD.HI` and `IMNMX` exist on sm_70 by their semantics but have different
binary encodings on sm_75. The translator decomposes them:

- `IMAD.HI Rd, Ra, Rb, Rc` becomes `IMAD.WIDE Ra, Rb, Rc` into two registers
  of the translation, then `MOV Rd` from the upper half. (One that gives its
  carry into the predicate that guards it is refused: the move, under that
  guard, would go by the carry. No code at hand has one: of the 103 million
  instructions of the two corpora, 323 write the predicate that guards them,
  all of them comparisons, sums and address computations on ordinary
  registers, each of which stays one instruction.)
- `IMNMX Rd, Ra, Rb, Pp` becomes `ISETP.LT Ps, Ra, Rb` into a predicate that
  holds nothing at that point, then `SEL Rd, Ra, Rb, Ps` (or `!Ps` for the
  maximum). The dead predicate is found by a liveness analysis over the function
  (`reference/translate.py`, `Function.stand_ins`).

Memory instructions with a scaled index (`[R4.X4 + 0x10]` in shared or local
memory) are decomposed into a shift (`SHF.L Rk, R4, 2`) into a register kept
for that operand, then the load or store at the plain address `[Rk + 0x10]`.

A `ULDC` (load from the constant bank into a uniform register) becomes one or
two `MOV` from the same constant bank entry into the ordinary register that
stands for it.

A Turing global or local load (`LDG`, `LD`) carries a cache and scope table that
sm_70 has no field for (`LTC128B` and the like). It is rebuilt with sm_70's table
(the extended address, size, weak or constant state, scope and cop), its data,
guard and scoreboards unchanged, and the batch part of its scheduling table
cleared: an instruction the translation changed names no batch (the wait hint
stays). A load that names `Pnz`, Turing's zero-fill predicate, is done where the
predicate holds, and its destinations are set to zero under its complement:
`@Pnz LDG` and `@!Pnz MOV R, RZ` for each destination, which is what ptxas writes
for the same construct on Volta itself. A load with both an instruction guard and
a `Pnz` is refused: sm_70 has one predicate per instruction.


## 3. The instruction tables

The translator decodes and assembles instructions from exact descriptions of
the sm_70 and sm_75 instruction sets, not from examples or heuristics.

**Where they come from.** The denvdis project
(https://github.com/redplait/denvdis) extracted from NVIDIA's nvdisasm the
descriptions NVIDIA's tools are built from: the opcode, the assembly format, and
which bits of the 128-bit word hold which operand, for each class of
instructions (648 classes for sm_70, 1,229 for sm_75). The decoder
(`reference/md.py`, `crates/vf-sass/src/decode.rs`) reads those descriptions
and identifies any instruction word by its class and operands. Against nvdisasm
(`reference/check.py`, 2026-11-20), on every distinct instruction word of
PyTorch's 475 cubins (2,935,225 words of sm_75) and of the 418 cubins of the
last PyTorch wheel with code for Volta (3,166,930 words of sm_70): one class
each and the same mnemonic for all of them, the same registers and predicates
for all but 871 and 885 words of two forms of `BMOV`, where nvdisasm names a
barrier register (`B0`) that the comparison does not read from the word.

**What is generated.** `reference/generate.py` reads the descriptions and the
rules and writes the Rust tables of `crates/vf-tables/src/` (1.9 MB
of source; 1,229 sm_75 classes and 648 sm_70 classes, each with its opcode, its
fields, its operands, what it reads and writes and its scheduling kind; the 79
rules; templates for the 44 classes of instructions the translator builds; 73
distances; the sm_80 forms of `reference/ampere-forms.txt`, which
`reference/ampere_forms.py` checks against the sm_75 and sm_80 descriptions;
`cargo fmt -p vf-tables` then formats the output). The generator
checks what the translator takes for granted of the descriptions instead of
looking it up: a distance in the code is one operand in one scaled field, and
the values of two enumerations.

**Licence.** The descriptions are distributed by denvdis under the GNU Lesser
General Public License, version 2.1. The tables generated from them are
distributed under the same licence, in a crate of their own (`NOTICE`;
`crates/vf-tables/LICENSE`). The rest of the translator is under the Apache
License 2.0.


## 4. A function, step by step

The translator works on a cubin (an ELF file of machine code), function by
function. Functions that call one another within the same section share their
uniform register placement. What it cannot do it refuses with a reason, and a
refused function is left with a trap at its start (and its exits where its
attributes say they are, since the driver visits them): the proxy refuses the
kernels that need it at the lookup.

The steps for one function (the entry point is `translate()` in
`crates/vf-sass/src/translate/`, matching `Function.translate` in
`reference/translate.py`):

### Parsing

The cubin is read as an ELF file (`crates/vf-sass/src/elf.rs`,
`reference/elf.py`). Each section of code (`.text.NAME`) holds one function.
Every 128-bit word is decoded by the sm_75 tables: its class, its operands,
what it reads and writes, its scheduling fields (stall count, scoreboards,
yield hint). In sm_80 code, each word must first match one of the supported
forms, which `normalize` (`crates/vf-sass/src/translate/ampere.rs`;
`reference/ampere.py`) turns into a word the sm_75 tables read; a word that
matches none refuses the function. The container's attributes (`.nv.info` and
`.nv.info.NAME` sections) are read: they record, for each function, properties the driver needs
(register count, shared memory, the lists of exits, shuffles, votes, calls
through a register, stack accesses, some shared-memory loads, and 16-bit
atomics, each by offset in the code). An attribute the translator does not
know causes the function to be refused.

### Control flow

Branches and calls establish a control-flow graph. The translator needs it for
liveness of predicates, liveness of registers, and to know where code may come
from. `BRA.DIV` and `BRA.CONV` (branches on whether the warp is converged) are
taken as branches to both sides, as are conditional branches. An unconditional
`BRA` goes one way. A branch through a register (`BRX`) goes to every target
listed for it in the container. The flow is compared with nvdisasm's listing by
`reference/flow_check.py`.

### Lives of registers and predicates

**Predicates.** A uniform predicate needs an ordinary predicate that holds
nothing of the program while it lives. The translator works out where each of
the seven predicates holds a value of the program, from the control flow and
from what each instruction reads and writes, `P2R` and `R2P` by their masks.
The validator works the same lives out again by itself, from nvdisasm's reading
of each instruction (`docs/validation.md`, "The independent validator");
nvdisasm's own chart of lives (`-plr`) is not enough for this: it ends every
life at a call, while code keeps predicates across calls within its section.

**Registers.** The lives of ordinary registers (which are dead at which
instruction) are those of `reference/pairing.py`'s flow, with what the code
called within the section reads and writes. They are compared exhaustively with
nvdisasm's chart of register lives on PyTorch's 475 cubins (23,683 of 23,693 functions,
29.8 million instructions) and on the held-out corpus (3,108 cubins, 71,426
functions): where the translator says dead and nvdisasm says alive, the places
are of three known kinds (the called code's convention, stores nobody reads,
branches always taken), and where the translation needs its two scratch
registers, no pair the translator calls dead has a mark in nvdisasm's chart
(7.9 million pairs in PyTorch, 33.0 million in the held-out corpus).

### Placing uniform registers and predicates

Each uniform register is given an ordinary register. Where a register of the
program that holds nothing through the whole life of a uniform register exists,
it is used (a "home": level 1); otherwise a register above the program's own.
A 64-bit pair stays on an even register and the next. All uniform registers of
all functions in a section share the same placement (a function called from
another may use the caller's uniform registers).

Each life of a uniform predicate (there may be several per uniform predicate:
carries born and dead within two instructions, many times) takes the lowest
ordinary predicate that holds nothing of the program along it. Where all seven
predicates hold something, one is saved in a register before the life (`P2R`),
used as the stand-in, and given back after it (`R2P`), as native code does when
it runs out.

Where that cannot be either (the program touches all seven during the life, or
the life goes through branches), the uniform predicate is held in a register of
the translation for that life: written there after each instruction that writes
it (`P2R`), and read back into an ordinary predicate before each one that reads
it (`ISETP.NE`). At level 3 with room made (section 5, "Room"), a held predicate
for which no register holds nothing through its lives is kept in a word of the
frame instead: loaded, through a register taken at the instruction, before an
instruction that reads it, and stored after one that writes it (`held-slot` in
`MAP.lives`; `reference/validate.py` checks these moves).

### Rewriting each instruction

Each instruction of the original becomes one or more instructions in the
result:

- **Same**: an instruction that is already sm_70 passes through, with its
  scheduling fields, branch targets adjusted for the new offsets.
- **By a rule**: an instruction with uniform operands is rewritten to its sm_70
  twin by the rule, with the ordinary registers that stand for the uniform ones.
- **By a sequence**: `IMAD.HI`, `IMNMX`, `ULDC`, memory with a scaled index,
  memory at a uniform address, a conversion with a uniform operand, and
  `LDSM` and `HMMA.1688` (section 6) each become a sequence of sm_70
  instructions. In sm_80 code, so do a 16-byte asynchronous copy, a paired
  conversion to halves and a shared counter increment (section 1).

A memory instruction with a uniform address part (`LDG [R4 + UR6]`) becomes the
sum of the two into the pool registers (`IADD3` or `IMAD.WIDE.U32`), then the
plain load. The instruction that reads the pool says on a scoreboard when it has
read its operands, and the very next instruction waits for that scoreboard.
This is what native sm_70 code does (measured in 45,183 loads from global memory
in a quarter of PyTorch's sm_70 build and cuBLAS: `reference/late.py`), and it
lets one pair of registers serve every sequence (section 5).

### Stall counts and the schedule

Each instruction carries a stall count: how many cycles before the next
instruction issues. The same scheduling model holds on both architectures
(measured: in 24.2 million writer-reader pairs of native sm_70 code and 21.4
million of sm_75, the shortest distance by kind of writer and of reader; of 170
pairs with 200 or more samples on each side, 155 are equal, 13 longer in sm_75,
2 shorter; `reference/latency.py`, `reference/distances.py`).

Instructions that are not rewritten keep their stall counts. What is rewritten
is spaced by the minimum distances measured on native sm_70 code
(`reference/distances-sm70.tsv`): 4 cycles from an integer instruction to the
next, 5 to a multiplication, 8 from arithmetic to a memory instruction, 12 from
a comparison to an instruction it guards, and so on. Section 7 gives each rule
with what it prevents.

### Relocations

The driver writes addresses into instructions when it loads a module
(relocations). The translator carries each relocation with its instruction: when
an instruction moves, its relocation entry is updated. A function whose
relocations cannot be carried (a relocation on an instruction that becomes
several, or on an instruction that is not the same in sm_70) is refused.
cuFFT's kernels for double precision have such relocations.

### Jump tables and return addresses

A call gives its callee a return address as an immediate: `MOV R26, 0x180`
before `CALL.REL`, or `LEPC` (read the program counter) with two immediates
added to it before an indirect call. When the translator inserts or removes
instructions, those numbers must follow. The translator finds them by their
structure (the `MOV` whose value matches the return point and whose register
reaches the call; the two terms of the `LEPC` addition), and refuses the
function when it cannot.

Of PyTorch's cubins: every relative call (`CALL.REL`) has a `MOV` with the
return address before it, and every indirect call has its `LEPC` immediates in
one of two recognized arrangements. Four functions are refused for a return
address that is not given the usual way.

A branch through a register (`BRX`) is a jump table. The container lists its
targets; the translator adjusts each target offset. Two functions of PyTorch are
refused for a jump table the translator cannot handle.

### Attributes of the function

The container records lists of offsets in the code (exits, shuffles, votes,
calls through a register, stack accesses, some shared-memory loads, 16-bit
atomics). The translator moves each offset with its instruction. A function that
is refused keeps only its exit offsets, so the driver can visit them.

The register count in the section header and the `.nv.info` attributes is set
to the translation's count. The cubin is tagged as machine code for sm_70; the
field that says which architecture it was compiled for stays the original's, 7.5
(8.0 for sm_80 code; what `cudaFuncGetAttributes` gives as `ptxVersion`): a
program chooses by it among
what it was built with, and the algorithms of Thrust and CUB answer "invalid
device function" to an architecture below the lowest of theirs.


## 5. Registers

A translated function needs registers of its own:

- One for each uniform register (two for a 64-bit pair).
- Two to compute into (the "pool"): every sequence (the upper half of a product,
  a sum for an address, a shift for a stride) writes its result into the same
  two registers, says on a scoreboard when it has read them, and the next
  instruction waits.
- For a function with tensor-core instructions: one for the lane of the thread
  (read once at the entry), and scratch registers for the sequences (section 6).

Where those registers come from is the subject of the borrowing levels. The
choice is made by `VF_BORROW` (default 7).


### Level 0: above the program's

The simplest: the translation's registers start where those of the function end
(the first register the function does not use; native code declares two more
than its highest). This adds to the function's declared register count.

A V100 gives each thread its registers in groups of eight, out of 65,536 per
multiprocessor divided into four sub-partitions of 16,384 each. The formula
(measured, matches all 112 test points on a V100; `crates/vf-proxy/src/profile.rs`):

```
allocated = ceil(register_count / 8) * 8
max_threads_per_block = min(floor(16384 / (allocated * 32)) * 128, 1024)
```

A kernel compiled with 64 registers has max_threads of 1,024. At 65 the
allocation jumps to 72 and the maximum drops to 896. A kernel launched with
1,024 threads that now declares more than 64 registers fails with error 701
("too many resources requested for launch"). With level 0 alone, 1,012 of
PyTorch's 23,693 functions cross that boundary, and five cases of the operation
battery cannot be launched (`avg_pool2d`, `interpolate`, `unfold`).


### Level 1: homes in the program's registers

A uniform register whose value is alive over some instructions may find an
ordinary register of the program that is neither alive nor written at any of
those instructions (a "home"). The uniform register lives there instead of above
the program's registers, adding nothing to the count.

Tried on every function of PyTorch (an experiment: the default does this only
for a kernel that would be refused), it brought 35 % of the functions that had
more registers back to the original's count, and 54 % of those pushed over 64
back under it.


### Level 2: the pool at the instruction

The pool (the two registers every sequence computes into) need not be the same
two throughout the function. At each instruction that needs them, the two
lowest registers that hold nothing there are taken. This saves two registers
in functions where no pair is dead everywhere.


### Level 3: nothing kept between instructions

For a kernel that names nearly every register of the machine and has no register
free through a whole life (Triton's `_topk_topp_kernel` at 255 registers, which
stops vLLM; FlashInfer's attention kernels at 246 to 248 registers):

- Uniform registers go into a frame in the thread's local memory. A word at
  each instruction that names one: loaded before (`LDL`, on a scoreboard;
  the next instruction waits), stored after (`STL`; says when it has read, and
  the next waits). The store is under no guard, whatever guards the
  instruction: an instruction may write the predicate that guards it
  (`@UP0 UIADD3 UR6, UP0, UR4, URZ, URZ` leaves its carry there), and a store
  under that guard would go by what the instruction left in it. What a guarded
  instruction writes is loaded before it, so that the store puts back what was
  there when the instruction did not run. An instruction that reads a register
  so loaded some time after it is issued (a population count) says when it has
  read, and the next instruction waits. The frame is on top of the program's
  own: the kernel attributes that say how far under the stack pointer the
  thread may go are raised by its size, and the stack pointer is lowered by it
  after the first instruction (which must be the one that sets R1 from
  `c[0x0][0x28]`). A kernel that does not use R1 at all needs none of that: its
  frame is addressed from `c[0x0][0x28]`, loaded into a register of the
  translation at each instruction that reads or writes it, and the stack pointer
  is not lowered. Room is made only in the first form: what is put away and
  brought back is addressed from the stack pointer.
- The pool, the base register, and the scratch registers of a tensor-core
  sequence are taken among the program's registers that hold nothing at that
  instruction.
- The lane is read where it is used: an `S2R` on a scoreboard before the
  sequence, waited for.

Level 3 adds no registers of its own. Tried first on every kernel of PyTorch (an
experiment, `VF_BORROW=3`), 16,052 of the 23,630 kernels that translate alone
kept their own register count (15,950 exactly, 102 with one or two more for
saved predicates), 5,577 of them with uniform registers in memory. The default
uses it only for a kernel that would otherwise be refused: it is the slowest
code.

A kernel that is entered from outside and is refused at level 3 for want of
registers may take the registers above the ones the program names, up to R252:
no caller keeps values there. That is a second level-3 attempt, made only after
the first is refused, so a kernel that fits in its own registers keeps its layout
and its count; it is not made for a kernel that declares its block size, whose
CTA is preserved. The section's register count follows the highest register
taken.


### Room

At level 3, an instruction may find no register that holds nothing (or no two
side by side, or no eight in a row for a matrix load). Then registers that the
instruction neither reads nor writes are put in the frame for the duration of
its sequence (`STL` before, `LDL` after), and the sequence computes in them.
Only at an instruction that always goes on to the next one and that does not
write the stack pointer.

Where the program moves R1 (730 of PyTorch's 23,693 functions), the depth under
the initial stack pointer is tracked at each instruction by a forward pass,
so that slots are addressed as `[R1 + depth + 4 * slot]` without a register
or a load.


### The budget

A kernel loses no thread as long as its register count stays in its group of
eight: its budget is `min(max(ceil8(declared), 32), 255)`. With `VF_BORROW=4`
or `5` the translator keeps to it: it tries levels 0, 1, 2, 3 in order and
takes the first whose count stays within the budget (when none does, the one
that takes the fewest). Level 3 then uses the registers under the budget for
what it takes at each instruction.

This is not what the layer does by itself. Keeping every kernel within its
budget was measured on PyTorch's operations and is not faster overall: a
borrowed register that must be saved and brought back costs more than a
register above (`docs/performance.md`, "Where the time goes"). The budget is
used where it is needed: for the second translation of a kernel that the driver
refuses to launch (below).

On PyTorch's 475 cubins, with the default (`VF_BORROW=7`): 23,664 of 23,693
functions are translated (99.9 %), 29 refused, one of them for registers (2026-11-23). The
usual translation takes 1,012 kernels from 64 registers or fewer to more than
64: those cannot be launched in blocks of 1,024 threads without the second
translation.


### What `VF_BORROW` chooses

| Value | What happens |
|---|---|
| `0` | Level 0 only. A kernel that names nearly every register is refused. |
| `4` | The budget: for every kernel, the first of levels 0 to 3 whose count stays within the kernel's own group of eight registers. |
| `5` | The same, with room made at level 3. What the second translation at launch uses (`crates/vf-proxy/src/fit.rs`). |
| `6` | Level 0 wherever that translates, the same bytes as with `0`; for a kernel refused for want of registers, levels 1, 2, then 3 with room made. A kernel with tensor-core products is refused at level 3 (the default until 2026-11-20). |
| `7` (default) | The same, and a kernel with tensor-core products is translated at level 3 too. |
| `9` | What `7` does. Then, for a kernel whose translation takes more registers than would let it be launched with every block size its own register count allowed on a V100 (up to the block size its attributes declare): levels 1, 2, then 3 within that count, and the first that fits is taken when it keeps every offset of the first translation. A kernel with relocations on its code or a jump table keeps the first. For CUDA graphs, where a launch is recorded before the driver counts registers (`crates/vf-sass/src/translate/block_registers.rs`; `docs/architecture.md`, "Translation control"). |

The other values are for tests of the levels themselves: `1` tries levels 2, 1,
then 0 for every kernel; `2` the same, then level 3 for a kernel still refused;
`3` and `8` level 3 first with every uniform register in memory (`8` kernels
with products included), then levels 2, 1, 0 when that is refused.


### The second translation at launch

A kernel whose translation exceeds the launch limit (error 701) is not refused.
Instead, the layer translates the module again at the moment the launch is
refused, within each kernel's own declared register count (`VF_FIT=1`, the
default; `crates/vf-proxy/src/fit.rs`). This second translation uses the
budget chain (`VF_BORROW=5`: levels up to 3 with room), where every kernel stays
at or below its declared count. A kernel with tensor-core products goes up to
level 2 only: level 3 refuses it there ("a matrix product in a function at
level 3"), and when no lower level keeps within its count, it is not launched
from the second translation, and the program gets the driver's answer to the
first. It is slower code (uniform registers in memory) and is made
once, at the first refused launch. The independent validator finds nothing in
PyTorch's 475 cubins translated this way, as in their usual translation
(`docs/validation.md`, "The corpora", says what it objected to until 2026-11-20
and what each objection was).

`docs/architecture.md`, "Launch", says what it does not cover (a kernel that
can reach data of its module, among others). In PyTorch's test battery,
`avg_pool2d` and `interpolate` in float32 and float64 and `unfold` in float32
need it for blocks of 1,024 threads. `VF_FIT=0` turns it off.


### A kernel that requires its block size

A kernel may require one block size (the attribute `EIATTR_REQNTID`, from
PTX's `.reqntid`: Triton writes it for every kernel, CUDA C never): the driver
launches it with that block and no other. The second translation at launch does
not help such a kernel: a translation with more registers than that block may
have never runs, and a program that reads the largest block the loaded kernel
allows (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, which the layer answers as
the driver does, from the translation's count) stops before any launch. Triton
does: `OutOfResources: threads, Required: 512, Hardware limit: 384`.

So when the translation chosen by `VF_BORROW` takes more registers than the
required block allows (128 for 512 threads, 64 for 1,024), the translator takes
the first of levels 1, 2 and 3 (with room) that keeps within that count; when
none does (or with `VF_BORROW=0`), the kernel is refused for its registers
("more registers than its required block of 512 threads allows (128)"), and the
layer runs the PTX that the cubin keeps, as for any kernel whose machine code is
refused; without PTX the kernel is refused when the program asks for it, which
is what its launch would end in. A kernel whose block is only a maximum
(`EIATTR_MAX_THREADS`, `__launch_bounds__`) or that declares none keeps the
usual translation and the second translation at launch: a smaller block may fit
the faster code. In the corpus of Triton's kernels (4,308 sm_75 cubins of
`tests/gpu/triton_corpus.py`, 347 of vLLM's and SGLang's kernels and 334 that
Triton made in this site's other runs) this changes three functions: the gather
of 16 warps of the corpus' sort kernel (two cubins of it, from two versions of
the corpus' source; 128 registers, 137 translated at every level) is refused and
runs as PTX, and SGLang's MoE router for 128 experts (128 registers, 136
translated the usual way) is translated at level 3 within 128. PyTorch's and the
held-out cubins have no required block.


## 6. Tensor-core products in machine code

Turing's tensor-core instructions (`HMMA.1688` for the m16n8k8 half-precision
product, `LDSM` for the matrix load from shared memory) have no equivalent on
Volta. The translator replaces them with Volta's own tensor-core product
(`HMMA.884`, the m8n8k4 product) and warp shuffles.


### LDSM (matrix load)

`LDSM` loads one, two, or four 8x8 matrices from shared memory, cooperatively
across the warp, into registers that a tensor-core product reads. Volta has no
such instruction. Each matrix is replaced by:

- For a non-transposed load of one matrix: the address of the thread's row by a
  shuffle from the lane that has it, the offset added, a load of 4 bytes.
  2 + 3 or 4 instructions per matrix (17 for four matrices).
- For a transposed load of one matrix: two rows by two shuffles, two loads of 2
  bytes, the two halves assembled. 3 + 9 instructions per matrix (38 for four).

The shuffles bring the addresses into the destination registers: the original
code often has the address in one of them (`LDSM.16.M88.4 R36, [R36]`), and
that one is shuffled last.

The lanes the shuffles read from are computed in registers of the translation,
not in the destinations. A store just before the load may not have read a
destination yet: a store reads its registers some time after it is issued,
and the compiler counts on the load, a memory instruction like the store, to
write them only after that. Native code writes such a register at once with a
memory instruction (`ptxas` does it 4 cycles after a store, with a load from
shared memory or a matrix load), and with arithmetic only after a barrier or a
wait for a scoreboard of a later memory instruction (none otherwise among
44,052 such writes in 821 sm_75 cubins: PyTorch's 475 and 346 of Triton). So
arithmetic writes a destination only
after waiting for the shuffles; in a transposed load, the lanes of the first
rows go into the destinations once the shuffles of the second rows are there.
Until 2026-11-23 the lanes went into the destinations first, and a store of
one of them just before the load could store the lane: a Triton kernel whose
epilogue stores registers and then reloads them with `ldmatrix` gave wrong,
changing results on a V100, and the test kernel `k_stored.cu` gave 616,576 and
325,152 wrong words of 62,914,560, plain and transposed (`tests/gpu/run.sh`,
`stored-machine-code`; 0 since).


### HMMA.1688 (the product of m16n8k8)

One Turing product of 16x8x8 halves becomes two of Volta's products of 8x8x4,
each in four steps (eight `HMMA.884` steps in all). The operands must be
rearranged:

- **A**: the halves of each register are swapped (`SHF.L.W`), giving a second
  pair of A registers with even-k and odd-k halves exchanged.
- **B**: the four elements each thread needs are gathered from other lanes by
  five shuffles. The intermediate values are assembled with shifts and bitwise
  selections (`LOP3`), and four `SHFL.IDX` operations fetch the final values
  from the lanes computed at the function's entry.
- **Selection**: after the eight steps, the thread's four results are selected
  from the two pairs of partial sums by a bitwise choice (`LOP3`) with a mask
  derived from the lane.

The sequence for accumulators in floats: 25 instructions when B is gathered, 14
when it is not (below); 31 for accumulators in halves when gathered, 20 when
not, plus exchanges with the thread two lanes away.

**Example.** The kernel `t_mma_f32` of `tests/kernels/k_tensor.cu` has one
Turing product of halves with float accumulators,
`HMMA.1688.F32 R8, R4, R6, R8`, among 24 instructions. Translated
(`vf translate tests/kernels/k_tensor.135.sm75.cubin OUT.cubin`), the function has 57:
nine at its entry, and 25 in place of the product. As nvdisasm prints them:

```
; at the entry: the lane, and the four values kept from it (R25 to R28)
S2R R24, SR_LANEID
SHF.R.U32.HI R25, RZ, 0x1, R24
LOP3.LUT R25, R25, 0x8, RZ, 0xc0, !PT
LOP3.LUT R25, R25, 0x3, R24, 0xf8, !PT              ; low: a lane that B is gathered from
LOP3.LUT R26, R25, 0x10, RZ, 0xfc, !PT              ; high: the other one
SHF.L.U32 R27, R24, 0x1c, RZ
SHF.R.S32.HI R27, RZ, 0x1f, R27                     ; half: a mask of the lane's bit 3
SHF.L.U32 R28, R24, 0x1f, RZ
SHF.R.S32.HI R28, RZ, 0x1f, R28                     ; odd: a mask of the lane's bit 0

; in place of HMMA.1688.F32 R8, R4, R6, R8
SHFL.BFLY PT, R16, R6, 0x4, 0x1f                    ; B as the lane four away has it
SHF.L.U32 R17, R6, 0x10, RZ
SHF.L.W.U32.HI R20, R4, 0x10, R4                    ; A with the halves of each register exchanged
SHF.L.W.U32.HI R21, R5, 0x10, R5
SHF.R.U32.HI R18, RZ, 0x10, R16
SHF.R.W.U32 R17, R17, 0x10, R16
SHF.L.W.U32.HI R18, R6, 0x10, R18
LOP3.LUT R16, R18, R17, R27, 0xe4, !PT              ; the two words of B that this half of the warp holds
LOP3.LUT R17, R17, R18, R27, 0xe4, !PT
SHFL.IDX PT, R12, R16, R25, 0x1f                    ; B gathered: R12 to R15
SHFL.IDX PT, R13, R16, R26, 0x1f
SHFL.IDX PT, R14, R17, R25, 0x1f
SHFL.IDX PT, R15, R17, R26, 0x1f
HMMA.884.F32.F32.STEP0 R16, R4.COL, R12.ROW, R8     ; a first Volta product
HMMA.884.F32.F32.STEP1 R18, R4.COL, R12.ROW, R10
HMMA.884.F32.F32.STEP2 R8, R4.COL, R12.ROW, R8
HMMA.884.F32.F32.STEP3 R10, R4.COL, R12.ROW, R10
HMMA.884.F32.F32.STEP0 R16, R20.COL, R14.ROW, R16   ; a second one, on the exchanged A
HMMA.884.F32.F32.STEP1 R18, R20.COL, R14.ROW, R18
HMMA.884.F32.F32.STEP2 R8, R20.COL, R14.ROW, R8
HMMA.884.F32.F32.STEP3 R10, R20.COL, R14.ROW, R10
LOP3.LUT R8, R8, R16, R28, 0xe4, !PT                ; the thread's four results, by the odd mask
LOP3.LUT R9, R9, R17, R28, 0xe4, !PT
LOP3.LUT R10, R10, R18, R28, 0xe4, !PT
LOP3.LUT R11, R11, R19, R28, 0xe4, !PT
```

(`tests/gpu/run.sh` holds the translator to these counts: 33 instructions more
than the original in `t_mma_f32`, 39 in `t_mma_f16`.) One Turing product is as
much arithmetic as one of Volta's instructions; the sequence runs two and uses
half of what each computes. Pairs, below, are what uses all of it.


### Kept values

Four values computed at the entry of a function with tensor-core products depend
only on the lane of the thread and do not change:

- `low` and `high`: the two lane indices from which B is gathered.
- `half`: a mask that selects even or odd rows.
- `odd`: a mask for the final selection.

These are computed once into four registers that nothing else writes (`kept()`
in `reference/translate.py`), from the lane read by an `S2R SR_LANEID` at the
function's entry. The eight instructions that compute them are checked by
`reference/validate.py` against their recipe; no other instruction of the
function writes those registers or the lane register.

A product that gathers B: 25 instructions (accumulators in floats). A product
that takes B from where the product before left it: 14 instructions (32 of the
64 products of PyTorch's forward attention kernels do). The
registers of the gathered B are kept for the next product on the same B
register, as long as nothing writes B in between and no other path leads to the
product. The check for a reuse goes back through the control flow to the last
product that gathered, verifying that nothing writes the gathered-B registers
and that no other entry point exists between the two (`reference/validate.py`).


### Pairs of products on whole rows

Two products that read the same A and the two B registers of one pair, with
operands that matrix loads brought from shared memory, are computed together:
eight `HMMA.884` steps for the two (where each by itself takes eight), on rows
that each thread loads directly in the layout Volta's product wants. The loads
that feed them become plain loads of whole rows, their addresses fetched from
the lanes that hold them by two shuffles a load; no B is gathered. The
accumulators stay in their own registers, permuted between lanes, and are
brought back to Turing's layout (one `SHFL.IDX` a register) only where something
other than such a product reads them. `docs/pairs.md` is the specification:
which products may be paired, where a conversion may stand, the schedule.

The planner (`reference/pairing.py`, `crates/vf-sass/src/pairing.rs`)
identifies which products form pairs and which matrix loads provide their
operands; `VF_PAIRS=0` turns pairs off. An independent checker
(`reference/qcheck.py`, `reference/qflow.py`) verifies every plan against the
design's conditions: the pairs' conditions, who reads what a load gives, the
states of the accumulators on every path, the places that are whole, the sharing
of registers. No plan the planner makes is objected to on either corpus (2,158
functions with products); fifteen kinds of mutated plans are objected to.

With pairs, PyTorch's attention in half precision takes 1.64 times the native
time (the benchmark of `tests/gpu/torch-bench.sh`, 2026-11-20, on a Tesla V100,
driver 580.178.04). `docs/performance.md`, "Where the time goes", has the
effect of pairs on a loop and on that kernel.


### Products at level 3

A kernel translated at level 3 (section 5) has no register kept between
instructions. A tensor-core product at level 3 takes everything it needs at the
instruction, among the registers that hold nothing there: 16 registers at once.
The lane is read where it is used (an `S2R` on a scoreboard before the
sequence). The kept values (`low`, `high`, `half`, `odd`) are computed within
the sequence and freed after. B is gathered every time (no reuse across
products). Pairs of products are not done at level 3.

The scratch registers are taken in pieces (each pair from an even register)
rather than as one block of eight, so that they fit into the gaps of a function
whose registers are nearly all in use.

On a V100: FlashInfer's `BatchPrefillWithRaggedKVCacheKernel` (246 registers,
272 tensor-core products, a frame of 96 bytes) translates at level 3 and runs
correctly through SGLang's engine (the engine's greedy text is the CPU's token
for token, 150 of 150). PyTorch's attention-gradient kernels of 242 to 250
registers (160 products each) likewise translate and pass the battery.


## 7. Timing rules

Every timing rule of the translator was either measured on a V100 with
hand-made kernels or observed in NVIDIA's own compiled code and confirmed by
nvdisasm. The minimum distances are recorded in `reference/distances-sm70.tsv`
(73 writer-reader pairs, measured with `reference/latency.py` on native sm_70
code of PyTorch and cuBLAS).


### Stall counts

Each instruction carries a stall count in its scheduling field. The translator
keeps the stall count of an instruction that is not rewritten. For an
instruction that becomes several, the first gets the original's stall count
(what it waited for, it still waits for), and the added instructions are spaced
by the distances of `reference/distances-sm70.tsv`.


### Scoreboards and waits

A V100 has six scoreboards (numbered 0 to 5) per warp. An instruction names a
scoreboard for its result ("write barrier") and optionally for having read its
operands ("read barrier"). Another instruction waits for a scoreboard ("wait
mask"). A scoreboard counts: up to 12 instructions pending on one in the 24
cycles before a wait (measured in native code: `reference/pending.py`).

**A wait must come two cycles or more after the instruction that names the
scoreboard.** In native code of both architectures this always holds (some
700,000 pairs; `reference/scoreboards.py`). A wait one cycle after does not
wait: measured on a V100.

For an instruction that reads a register of the translation (a stand-in of a
uniform register, or the pool), the instruction says on a scoreboard when it
has read, and the very next instruction waits for that scoreboard. This is what
lets one pair of registers serve every sequence without copies.


### Slow operations: results written late

Not all results arrive at the same time after an instruction is issued. Measured
on a V100 with 510 hand-made sm_70 kernels (three runs each, one warp):

| Instruction kind | When its result is written |
|---|---|
| Integer, float, half arithmetic (`FMAI`, `FXU`, `FMALITE` for halves) | In the order issued; a write one cycle later replaces it |
| **Arithmetic on doubles** (`DADD`, `DMUL`, `DFMA`) | **Two cycles later** than the integer and float kinds: an instruction issued one or two cycles after the double is overwritten by it |
| **Volta's matrix product** (`HMMA.884`) | **Two cycles later**, as doubles |
| **Conversions** (`I2F`, `F2I`, `F2F`, `MUFU`) without a scoreboard | **9 to 11 cycles** after issue, whether a scoreboard is named or not |
| Loads, special registers (`S2R`) on a scoreboard | When announced on the scoreboard (may be many cycles) |

**What this means for borrowed registers.** Before the translation writes into
a register it borrows, every instruction of the program that wrote that register
on a scoreboard must have been waited for (the first write waits for all six
scoreboards). A register that some instruction of the function writes late under
no scoreboard, or by arithmetic on doubles, or by a matrix product, is never
borrowed: nothing can be waited for, and the distance across a branch is not
known. On PyTorch, 161 homes fewer of 12,109 (`reference/translate.py`,
`unannounced`).


### Predicates written late

A comparison of doubles (`DSETP`) writes its predicate two cycles later than
integer, float, and half comparisons. `FCHK` writes its predicate 13 to 16
cycles after issue. Measured on a V100 with hand-made sm_70 kernels (one warp,
three runs each).

A predicate that no `DSETP` and no late-kind instruction writes anywhere in
the function is taken wherever there is one. Where none exists, a slow one is
taken, and the instruction whose translation writes or saves it begins with
instructions that do nothing: the first waits for every scoreboard, and together
they last as long as native code keeps a plain reader from the slowest writer of
that predicate under no scoreboard (13 cycles for `DSETP`, 26 for `FCHK`; one
cycle when every late writer names a scoreboard, which the wait covers). The
delays and their correctness were measured on a V100 for `DSETP` (stalls 1 to
15, in a straight line and after a taken branch) and for `FCHK`.


### A result of double precision

An instruction of double precision (DADD, DMUL, DFMA, DSETP) names a scoreboard for its
result, and its reader waits for it. That wait is not all that Volta's own code keeps: a
reader of such a result never comes closer than 8 cycles when it is of the same pipe (double
or half precision), 9 otherwise, and a predicate of DSETP 15 cycles as a guard or to a branch
(`docs/validation.md`, section 8, has the measurements). Code for sm_75 keeps the same
distances, so its instructions are right as the compiler had them; code for sm_80 has 6 and
14. The schedule gives these distances wherever the writer or the reader is the translation's
(`Machine.waited`, `distance::waited`): in a function of sm_80 code, everywhere. The schedule
sees straight-line code, and the reader may be the first instruction where a branch goes (the
reciprocal of a double-precision division: a DFMA, a branch five cycles later, a DMUL at its
target). So in a function of sm_80 code a result of double precision is also as old at the end
of straight-line code as any reader needs, 9 cycles for a register and 15 for a predicate
(`Machine.waited_longest`, `Item.anew`); code for sm_75 keeps Volta's distances along its
branches too, and nothing is added there. No such function is translated yet (no form of these
instructions passes: section 1).

### R2P reads all seven predicates

`R2P` reads all seven predicates and writes them all back: those of its mask
from its register, the others as they were. Its read is an ordinary read: what
was written less than 4 cycles before the `R2P` (5 for another `R2P`) is not
read yet. So:

- A predicate outside the mask of an `R2P` that was written 1 to 3 cycles
  before it gets its former value back.
- Of two `R2P` of one predicate each, 1 to 4 cycles apart, the first is lost.

Both were measured on a V100 with 400 hand-made sm_70 kernels (one warp, three
runs each). The translator spaces `R2P` instructions
at least 5 cycles apart and keeps a plain reader at least 4 cycles from a
writer.

**An instruction between two others costs more than its stall count.** An
`ISETP` with a stall of 1, then a `NOP` with a stall of 1, then `SEL`: the
`SEL` sees the new value. An `ISETP` with a stall of 2 directly followed by
`SEL`: it does not. The translator's distances are those of one instruction
directly to the next, without intermediate instructions; with instructions
between, the same sum of stall counts is more time, never less.


### After a tensor-core product

After the last step of a Volta product, eight cycles must pass before its result
is read by anything but another product (thirteen for accumulators in halves).
A wait on a scoreboard alone is not enough: read sooner, the registers hold
what they held before (374 of 1,536 values wrong in a test kernel on a V100).
The translation keeps the compiler's schedule, eight cycles of other work
between the last step and the selections, and `reference/validate.py` checks
that distance.

In native code (cuBLAS and PyTorch's sm_70 code), no result of a product is
read by anything but another product less than 32 cycles after the last step
(`reference/pending.py`).


## 8. What is refused

A function the translator cannot handle is refused with a reason. Its code is
replaced by a trap at its start, with its exits left where its attributes say
they are (the driver visits them), and the proxy refuses the kernels that need
it at the lookup (error 218, with the reason on standard error). Other kernels
of the same module work.

**PyTorch 2.14.0** (475 cubins, 23,693 sm_75 functions): 23,664 translated
(99.9 %), 29 refused (`vf translate` of 0.1.3, 2026-11-23). The reasons and counts (each
function counted by the first reason met):

| Reason | Count |
|---|---|
| A branch through a uniform register | 10 |
| `I2IP` (a conversion the translator does not have) | 6 |
| A call whose return address is not given the usual way | 4 |
| A conversion of a uniform register (a form not written) | 4 |
| `IMAD.HI.X` (a carry form not handled) | 2 |
| A jump table the translator cannot handle | 2 |
| Registers (no room at any level) | 1 |

Of 925 operations tested against the CPU on a V100 (`tests/gpu/torch.sh`): 924
give the same result, 1 is skipped (the CPU does not do it), none is refused,
none is wrong. Of 32 small whole models (`tests/gpu/torch-models.sh`): 32 pass.

**FlashInfer** (SGLang's default attention): of 18 prefill kernels, 13 are
translated. Five are refused: three that name 255 registers (no room at any
level), two for `LEA.HI` with a uniform register (a form not written). These
are the counts of 2026-11-20, before the rule for `LEA.HI` with a uniform source
(section 2) was added.

**The held-out corpus** (3,108 cubins of NVIDIA's CUDA 13.5 libraries, cuDNN,
NCCL and others; not programs that were run): 3,105 translated in whole or in
part. Three cubins are not read at all (NCCL, NVSHMEM: a section table that
points outside the file, two sections of code of the same name).

**cuFFT double precision**: the kernels have relocations that cannot be carried
with their instructions; those functions are refused.


## 9. Two implementations

The translator exists as a Python reference (`reference/translate.py`) and a
Rust implementation (`crates/vf-sass/src/translate/`). The rule is that they
give the same bytes: for every cubin at hand, the translated bytes, the map of
instructions, the lives written beside it (`MAP.lives`), and the refusals with
their reasons must be identical.

**How it is checked.** `tests/sass/compare.sh` runs both translators on every
cubin and compares the cubin, the map, `MAP.lives` and the refusals, byte for
byte. On PyTorch's 475 cubins (23,693 functions): all 475 the same, the same
refusals for the same reasons, and since 2026-11-22 the same lives. On the test
kernels: all the same. On the held-out corpus of 3,108 cubins: all the same
(2026-11-20, before the lives were compared).
`tests/sass/fuzz.py` sets the two against each other on cubins damaged at random
(fields flipped, instructions removed, sections truncated): 200,000 mutants
(2026-11-20) with no difference in bytes, map or refusals. A change to what one
translator does is a change to both.

The Rust translator is what runs in the layer. The Python reference is what the
independent validator (`reference/validate.py`) is tested against, what the
planner of pairs (`reference/pairing.py`) works from, what the hand-made
controls (`reference/controls.py`) are tried on, and what the corpora are
checked with. Both write the map and `MAP.lives` that the validator reads, so
the validator also checks a translation made by `vf translate IN OUT MAP`
directly. The two exist so that neither is trusted by itself.

The Rust translator takes two and a half minutes for PyTorch's 475 cubins (626
MB) one after the other on one core, and 2.4 s for the largest of them (22 MB,
358 functions); the Python reference, about a minute with 48 processes
(2026-11-20). In the layer a translation is made once and kept in the cache
(`docs/architecture.md`, "The path of a module").
