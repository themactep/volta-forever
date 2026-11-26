# Instructions of later GPUs on Volta

What a Tesla V100 (compute capability 7.0) computes in place of an instruction it does not
have. Each section says what the sequence is, why, what is exact and what is not, and where
to look in the code. The emulations apply to PTX (program text) adapted for Volta by the
lowering pass (`crates/vf-ptx/src/lower/`); section 10 says which of them also exist in
translated machine code (`crates/vf-sass/src/translate/`).

Programs that carry PTX for sm_75 or later have their PTX rewritten for sm_70 before the
driver compiles it. Programs that carry only sm_75 machine code have it translated function
by function. In both cases the instructions a V100 lacks are replaced by sequences of
instructions it has. A kernel that runs such sequences is announced on the console; a
kernel whose instructions have no sequence is refused when the program asks for it.

The test suite (`tests/gpu/run.sh`) checks every emulated form against a host model of the
instruction. The host models are the definitions of the PTX manual, and the test values are
those where an answer is decided: halfway points of a rounding, limits of a range, sums
that are exact in any order. The tests are in `tests/kernels/`.

Where a result depends on the order or the width of the sums (a product of floats on
arbitrary values), no manual says what the hardware does, and "the hardware's answer" is not
one thing: Volta's own product, Turing's and Blackwell's do not add alike. For those cases the
sections below give what was measured against an RTX 5090.

## 1. What Volta's tensor cores do

A V100 has one tensor-core instruction: `mma.sync.aligned.m8n8k4` on halves (f16), with
f16 or f32 accumulators. It computes four independent 8x8x4 matrix products per warp, one
per octet (lanes `4q..4q+3` and `16+4q..16+4q+3`). Each thread holds:

- **A**: 4 half values, its row of the octet's 8x4 input matrix.
- **B**: 4 half values, its column of the octet's 4x8 input matrix.
- **C/D (f32)**: 8 values, scattered positions in the octet's 8x8 output.
- **C/D (f16)**: 8 values, one complete row of the 8x8 output.

Every product of two f16 values is exact as an f32; the instruction adds four of them and
the accumulator in f32 with truncation: the terms are aligned to the largest, the bits shifted
out are dropped, and the sum is cut to f32, not rounded (measured by Fasi, Higham, Mikaitis and
Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ Computer Science, 2021). Its results
differ in the last bits from a sum of `fma.rn.f32`, and long sums of terms of one sign come out
slightly low.

Volta has no tensor-core instruction for int8, bf16, tf32, f64, FP8 or the 4-bit formats.
It does have `dp4a` (the dot product of four pairs of int8, accumulated into an int32),
which is used by the integer emulation.

Programs compiled for Turing (sm_75) and later ask for products of other shapes (`m16n8k8`,
`m16n8k16`, `m16n8k32`, `m8n8k16`, ...) and other element types. Each is built from what
Volta has, as the sections below describe.

## 2. Products of halves and floats of later shapes

### FP16 products: m16n8k8, m16n8k16

The instruction `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` computes a 16x8x8
product over the full warp (32 threads). A thread in group g (lane = 4g + t, t in 0..3)
holds two A registers (rows g and g+8), one B register, and four D registers (the results
for its rows and columns).

**Decomposition.** Each Turing product becomes two of Volta's `mma.m8n8k4`, one for even k
and one for odd k (a product of a chain is computed otherwise: "Chains of products" below):

1. The A registers of the thread are used as column-major operands. The first product takes
   them as they are; the second takes them with their halves swapped (`shf.l.wrap.b32` by
   16), which exchanges even and odd elements of k.
2. The B operand must be redistributed. A thread provides four elements of one value of k
   for four columns. Five warp shuffles (`shfl.sync`) bring each thread the elements it
   needs from the lanes that hold them.
3. With f32 accumulators, a thread's four results are in registers 0..3 (even thread number)
   or 4..7 (odd), selected by `selp`. The other four registers hold sums of unrelated terms
   and are discarded.
4. With f16 accumulators, the result is in Volta layout: half of what the thread needs and
   half of what the thread two lanes away needs. One shuffle each way converts between the
   two layouts.

The `m16n8k16` shape is two `m16n8k8` products on the two halves of k, the second
accumulating on the first.

**Exactness.** The products are computed on Volta's tensor cores. Each term (f16 x f16) is
exact as an f32. What differs from a later GPU is how the eight terms of k and the
accumulator are added: here four at a time, by Volta's instruction (the even k, then the odd
k), each sum truncated to f32 (section 1); later hardware adds otherwise, in an accumulator wider than f32.

- On the inputs of the test suite, where every sum is exact whatever its order, every result
  is the host model's, and an RTX 5090 gives the same (all shapes of the FP16 product).
- On random values of mixed magnitudes the last bits differ. Of 327,680 results of one
  `m16n8k8` (zero accumulators) through the machine-code translation, which adds in the same
  order, compared with an RTX 5090's: 47.6 % identical, 41.2 % one unit of the last place
  away, 10.2 % two to four, 1.1 % further, the largest where terms cancel (2026-11-18).

A program built for Volta computes its products with the same instruction and has the same
kind of difference from later hardware.

**Example.** A product of constant halves, before and after adaptation (`bin/vf adapt`):

Before (PTX for sm_75):
```
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3}, {%a0, %a1}, {%b0}, {%c0, %c1, %c2, %c3};
```

After (PTX for sm_70, the core of the sequence):
```
shf.l.wrap.b32 %vf_r0, %a0, %a0, 16;         // A with halves swapped
shf.l.wrap.b32 %vf_r1, %a1, %a1, 16;
shfl.sync.bfly.b32 %vf_r2, %b0, 4, 31, -1;   // five shuffles gather B
  ... (4 shifts and 2 selp to assemble the upper/lower halves) ...
shfl.sync.idx.b32 %vf_r9, %vf_r7, %vf_s0, 31, -1;
shfl.sync.idx.b32 %vf_r10, %vf_r7, %vf_s1, 31, -1;
shfl.sync.idx.b32 %vf_r11, %vf_r8, %vf_s0, 31, -1;
shfl.sync.idx.b32 %vf_r12, %vf_r8, %vf_s1, 31, -1;
mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 // first Volta product (even k)
    {...}, {%a0, %a1}, {%vf_r9, %vf_r10}, {...};
mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 // second (odd k)
    {...}, {%vf_r0, %vf_r1}, {%vf_r11, %vf_r12}, {...};
selp.b32 %d0, %vf_r25, %vf_r21, %vf_p1;  // choose this thread's results
selp.b32 %d1, %vf_r26, %vf_r22, %vf_p1;
selp.b32 %d2, %vf_r27, %vf_r23, %vf_p1;
selp.b32 %d3, %vf_r28, %vf_r24, %vf_p1;
```

The preamble (computed once per function, 10 instructions) reads the lane and derives the
indices, masks and predicates the sequences use. The sequence above uses 29 scratch registers and no
register of the program beyond the operands.

### Ampere shapes: m16n8k16 on FP16

`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` has four A registers (the first two
for k 0..7, the other two for k 8..15) and two B registers. It becomes two `m16n8k8`
products, the second accumulating on the first: four of Volta's `mma.m8n8k4`.

### Remembered operands

The lowering remembers what it derives from the operands of a product: the A registers with
their halves swapped, the gathered B, an f16 accumulator in Volta's layout. When several
products use the same register with nothing writing it in between, it is derived once.

### Chains of products

Four or more `m16n8k8` products with f32 accumulators that follow one another on the same
four accumulator registers, in straight-line code, are a chain, and are computed otherwise:
one `mma.m8n8k4.row.col` for each product instead of two. The accumulator is kept in eight
registers of the rewriting, in Volta's layout and split: the two halves of k of a product are
computed by different octets of the warp, and each lane holds partial sums that only the end
of the chain adds up. The direct load for a pair of A fragments requires consecutive matrices
in the source load. Other pairs gather their two registers by index. The registers of the
program get the accumulator back before any
statement that names one of them, before a label or a branch, and at the bounds of a block.

A register is followed by its name, and it has one: a member of a family written with zeros
before its number (`%d00` for `%d0` of `.reg .b32 %d<4>`, the same register to the
assembler) is written as the family declares it before anything else is done
(`lower/names.rs`). A function that writes a member so is refused when the name could mean
something else, under either spelling: a register declared on its own under the name of a
member, two families with a name in common, a variable (`.shared .b32 %d00;` in a block is
that variable there, and `mov.u32 %d1, %d00;` its address), a label, a parameter. Which of
two declarations the assembler follows depends on the blocks, on their order and on where the
name is written; the rewriting does not guess. The registers and labels that the rewriting
adds have names that begin with `%vf_` and `$vf_`: a function that has such a name
already is refused too. (The BF16 switches of section 4 add names that begin with
`%bfgroup_` and `$bfgroup_`; a function that has such a name already keeps the scalar sequence.)

The statements of a body are read as text, and a comment is not code: the comments of a body
that is rewritten are written over with spaces before anything reads it (a body in which
nothing is rewritten is left as it came). A family declared as `%d<4> /* four */` is the same
family, and a name that only a comment mentions is declared by nothing.

Measured on products whose operands are held in registers (what `vf adapt` makes of
`tests/kernels/ptx-cases/padded-chain-read.ptx`, and of the same with fewer products):
alone, a product is two Volta products, five shuffles and 17 statements in all; in a chain,
one Volta product, six shuffles and 9 statements. The chain itself costs 8 statements to
begin from zero (42, with 16 shuffles, to begin from values that the program computed) and
28, with 12 shuffles, to end. Timings are in `docs/performance.md` ("Flash attention").

### Operands loaded from shared memory

Most tensor-core operands come from shared memory through `ldmatrix` (section 3). A first
pass over the function body finds operand registers that one `ldmatrix` loads and nothing
else writes. That `ldmatrix` then loads them a second time in the layout Volta takes, so
that the product uses rows and columns directly instead of gathering by shuffles:

- Both operands of an FP16 product from plain `ldmatrix`: the product is split along halves
  of k instead of even and odd. The form is `mma.m8n8k4.row.col` (not `.col.row`).
- The B operand from `ldmatrix.trans`: four words of two rows of the loaded matrix.

A register used only this way is never loaded in Turing layout.

### Pairs of products

Two such products (both operands from plain `ldmatrix`, f32 accumulators in place) with the same A
are computed together where the lowering can show that it gives the same bits (`docs/pairs.md`,
section 13; `VF_PAIRS=0` turns it off): for each `m16n8k8`, two of Volta's `mma.m8n8k4.row.col` on
both products' eight accumulator registers instead of two for each product, half of whose results
are dropped. **Exactness:** each result is the same sum as the product by itself computes,
`(C + sum of k 0..3) + sum of k 4..7`, each by Volta's instruction on the same four terms in the
same slots, so the same bits, NaN, infinities and subnormals included; only which lane holds which
result differs, and one shuffle a register gives each reader Turing's layout. Compared on a V100
with each product by itself (`VF_PAIRS=0`): the `pairs-adversarial` case's words for 1, 2 and 37
turns (subnormals, infinities, NaN, the largest half, cancelling turns), and Triton's GEMMs and the
dot and attention cases of its corpus at capability 8.0 (2026-11-23). Products whose B comes from
`ldmatrix.trans` are not paired: there a product by itself adds the even k, then the odd k, and the
halves of k would round otherwise.

### What it costs

A single `m16n8k8` on halves with f32 accumulators: 19 instructions, after a 10-instruction
preamble computed once per function. When B is already gathered: 8 instructions (6 when A
is also reused). The same product in machine code: 25 instructions, or 14 when B is kept
from the product before (section 10). What this costs in time is in `docs/performance.md`,
"Where the time goes".

**Test cases.** `tests/gpu/run.sh`: `tensor-core` (37 kernels, 0 wrong), `chains`
(7 kernels, 0 wrong), `kept` (5 kernels, 0 wrong), and the PTX-only cases
(`ptx-family-declared-again`, `ptx-accumulator-as-operand`, and others).

## 3. Matrix loads and moves

### ldmatrix

`ldmatrix.sync.aligned.m8n8.x4.b16` loads four 8x8 matrices from shared memory, one row
per thread, into four registers. It is a warp-cooperative load. Volta lacks it.

**On Volta.** For each of the 1, 2 or 4 matrices: the address of the thread's row is
obtained by a shuffle from the lane that holds it, the offset of the thread's pair of
elements is added, and 4 bytes are loaded from shared memory. For the transposed form
(`ldmatrix.trans`), each matrix needs two rows fetched by shuffles and two 2-byte loads,
the two halves assembled. Other shapes and element types (`m16n16`, `m8n16`, `.b8`) are
refused. The address is a register, or, in the `.shared` form, a variable of shared memory
(`[global_smem]`, `[global_smem+16]`, as Triton writes it for a layout conversion), whose
address in the shared window then stands for every lane's row; the machine code `ptxas`
makes of that (`LDSM [RZ]`) is refused by the machine-code translator, and the layer runs
the PTX that the cubin keeps when there is one.

A 16-byte load from shared memory faults on a V100 when its address is not a multiple of
16. The PTX manual requires every address to be aligned
to its access size. By default (`VF_ALIGNED_ROWS` on, `Options::aligned_rows`) the layer
takes the manual at its word and loads 16 bytes. A program whose rows are elsewhere stops
with a misaligned-address fault (error 716), and `VF_ALIGNED_ROWS=0` makes the layer
test the addresses at runtime: if a row is not at a multiple of 16 the same bytes are
loaded word by word. The test costs 17 % to a factor of two on a program's emulated
tensor-core kernels; every program tested keeps its rows aligned.

**Instruction counts.** Plain `ldmatrix` of 4 matrices: 14 instructions (2 for the
address, one more when it has an offset, and 3 per matrix), besides the lane values of the
preamble. Transposed: 38 (2 + 9 per matrix).

**movmatrix.** `movmatrix.sync.aligned.m8n8.trans.b16` transposes a matrix fragment across
lanes. It becomes two warp shuffles.

**stmatrix.** For `stmatrix.sync.aligned.m8n8` stores of `.b16` in `.shared` or `.shared::cta`, `.x1`, `.x2` and `.x4` store one, two and four matrices. Each consecutive group of eight lanes supplies the row addresses for one matrix. Each group of four lanes stores one row of every matrix, using one source register per matrix. With `.trans`, each lane stores the low and high halves of its register in successive rows of one column.

**Test cases.** `tests/gpu/run.sh`: the `tensor-core` and `ampere` cases load every
operand with `ldmatrix` and check every register of every thread against the host model.
The `tensor-core-tested` case runs the same kernels with `VF_ALIGNED_ROWS=0`, including
those whose rows are deliberately elsewhere: 42 kernels, 0 wrong. The `stmatrix-x1`,
`stmatrix-multi` and `stmatrix-transposed` cases check `stmatrix`.

## 4. bf16

A bf16 (bfloat16) is the upper half of a 32-bit float: the same 8-bit exponent, 7 bits of
mantissa instead of 23. A V100 has no instruction for bf16. The layer emulates arithmetic,
comparisons, conversions and atomic additions.

### Products

By default, bf16 tensor-core products (`mma.sync.aligned.m16n8k16` and `m16n8k8` on bf16;
`m16n8k8` and `m16n8k4` on tf32; `m8n8k4` on f64) get no help from Volta's tensor cores: a
half does not hold the exponent range of a bf16 or an f64. Each thread gathers its rows of
A and its columns of B from the other lanes (shuffles, as for the integer products) and
computes its sums with `fma.rn.f32`, one term after the other in the order of k. A bf16 is
made an f32 by a shift of 16 (exact). All sums are complete before a result is written, so
that the instruction's atomicity is preserved.

Four switches, off by default, change this for `m16n8k16` on bf16 (`lower/bf16_groups/`;
`INSTALL.md`, "Environment variables"). With `VF_BF16_TC_GROUPS=1`, a group of consecutive
products, and with `VF_BF16_TC_TILES=1`, a two-by-two tile of them, runs on Volta's FP16
tensor cores when a warp vote finds that every multiplicand converts exactly to a normal half
or a signed zero; otherwise the warp takes the scalar sequence. On the tensor cores the f32
sums are truncated (section 1) and can differ from the scalar sequence in the last bits. `VF_BF16_COLUMN` and `VF_BF16_ROW` (`ordered` or `tree`) specialize
products where, as a test finds when the code runs, only column 0 of B, or only the first
row of A, contributes: `ordered` keeps the scalar order of the sums, `tree` adds partial
sums and can round differently.

**Determinism.** With `VF_BF16_TC_GROUPS` and `VF_BF16_TC_TILES`, the path of a group (or a
tile) is chosen by one `vote.sync.any` over every multiplicand of the group in the 32 lanes,
including rows and columns whose results the program discards. A result of row i and column j
then depends, in its last bits, on other rows of A, other columns of B and other products of
the group, where the hardware of later GPUs and the default path make it of row i, column j and
its accumulator only. A program that leaves unused operands of a product uninitialised (stale
shared memory in the rows that FlashAttention-2 masks out of its last tile) gets results that
are not bit-reproducible from run to run; where a tile mixes the tokens of several requests (a
mixture of experts), a result of one request can change with the others.
The differences are of the size of rounding: both paths are within what the PTX manual allows.
The `tree` modes of `VF_BF16_ROW` and `VF_BF16_COLUMN` have the same property, since a vote
over the other rows or columns (which must be zero) and over the bounds of every operand
chooses their path. The `ordered` modes give the same bits on both paths (the same order, the
same `fma.rn`), and the default path has no vote: both are reproducible.

**A scaled form on the FP16 tensor cores, tried and not kept.** A lowering of bf16 `m16n8k16`
that scales each row of A and each column of B by a power of two into the range of a half, checks
that every value is then a normal half or zero, computes the product with four `m8n8k4` and
applies the scales in one `fma` a result, with the scalar product as the fallback of a result
whose row or column does not fit, was made by hand for Triton's bf16 GEMM (both operands by
`ldmatrix`, tiles 64x64x32 and 128x128x32, one stage, 2048 x 2048 x 2048) and run on a V100
(2026-11-24). Its results were right against a float64 product on the host (on terms of one
sign, a mean error of -0.17 units in the last place, K = 512), and it was slower: 5.35 and 5.63
TFLOPS against 9.79 and 9.92 for the scalar products, 0.54 to 0.57 times. The conversion and the
check cost more instructions than the `fma` they replace (one turn of the loop of the 64x64
kernel, per warp: 2,804 instructions, 256 of them `HMMA`, against 1,725 for the scalar products,
1,024 of them `FFMA`), and the converted rows stay in registers (the 64x64 kernel's 125 registers
become 216 without the check and the fallback, 255 with them). The layer keeps the scalar
products.
`VF_BF16_TC_GROUPS=1` stays the opt-in for programs whose bf16 operands are known to be halves:
on the same GEMMs, with one and with three stages, 1.33 to 1.51 times the scalar products when
every operand converts, 0.84 to 0.98 times when some do not, and on terms of one sign a mean error
of -28.6 to -29.7 units in the last place (K = 512; the scalar products: -0.0015), the truncation
of Volta's sums (section 1).

**Instruction counts.** One `m16n8k16` on bf16: 199 instructions (64 `fma`, 32 shuffles).
One `m16n8k8` on bf16: 116. One `m16n8k8` on tf32: 103 (plus operand truncation, below).
One `m16n8k4` on tf32: 68. One `m8n8k4` on f64: 63. These are whole-kernel counts;
roughly 30 of each read the operands and store the results.

**Accuracy.** The layer adds the terms one at a time with `fma.rn.f32` (23-bit mantissa,
round-to-nearest-even). An RTX 5090 uses a wider internal accumulator (~25-bit mantissa
with truncation to f32). The difference (measured 2026-11-17 on an RTX 5090 with 768,000
random results):

| Type | Results that differ from RTX 5090 | Magnitude |
|---|---|---|
| bf16 m16n8k16 | ~20% of random values | 1 to 2 ULP |
| bf16 m16n8k8 | ~20% | 1 to 2 ULP |
| tf32 m16n8k8 | not measured since the operands are cut as the hardware cuts them (before: ~40 %) | |
| f64 m8n8k4 | 0% | exact |

For tf32, the layer cuts the operands to 10 bits of mantissa (`and.b32 reg, reg, 0xFFFFE000`),
as the hardware was measured to (the low 13 bits dropped, not rounded). What is left should
be the difference in the sums, as for bf16; it was not measured again.

Both rounding behaviors (the layer's sequential f32 `fma` and the hardware's wider
accumulator) are valid implementations of the PTX specification, which requires
"accumulation with at least single precision". The products of bf16 operands are exact (a
bf16 x bf16 has at most 14 significant bits, well within f32).

### Arithmetic

| Instruction | On Volta |
|---|---|
| `neg.bf16`, `abs.bf16`, `abs.bf16x2` | Flip or clear the sign bit (of each half) |
| `cvt.f32.bf16` | Exact: the bf16 as the upper half of the f32 (a shift) |
| `add{.rn}.bf16`, `add{.rn}.bf16x2` | Operands as f32 (a shift), `add.rn.f32`, rounded to bf16 |
| `sub{.rn}.bf16`, `sub{.rn}.bf16x2` | Subtract widened operands and round each result to bf16 |
| `mul{.rn}.bf16`, `mul{.rn}.bf16x2` | Multiply widened operands and round each result to bf16 |
| `add.rn.f32.bf16`, `sub.rn.f32.bf16` | Widen the first operand and add or subtract in f32 |
| `min.NaN.bf16`, `max.NaN.bf16` | Compare the widened inputs, propagating a NaN from either operand |
| `{min,max}.bf16x2`, `{min,max}.NaN.bf16x2` | Each half widened and compared on its own; with `.NaN`, a NaN in either operand gives a NaN in that half |
| `min.bf16`, `max.bf16` | Compare the widened inputs with `min.f32`/`max.f32`: a NaN input gives the other input, two NaN give 0x7fff (the manual says a NaN; 0x7fff measured on sm_120) |
| `cvt.rn.bf16.f16` | Widen exactly to f32, then round to bf16 |
| `fma.rn.bf16`, `fma.rn.bf16x2` | Rounded to odd in f32 first (from `fma.rm.f32` and `fma.rp.f32`), then to bf16. Packed lanes with proven exact cancellation use a single f32 add per lane. A scalar `fma` whose addend is a register set to -0 (`mov.b16` of `0x8000`) in the same block is the `mul` above |
| `cvt.rn.bf16.f64` | f32 rounded to odd (from `cvt.rm.f32.f64` and `cvt.rp.f32.f64`), then to bf16 |
| `cvt.rn.f16.bf16` | Widen exactly to f32, then round to f16 |
| `cvt.rn.bf16x2.e4m3x2`, `cvt.rn.bf16x2.e5m2x2` | Decode each FP8 byte through an exact f16 intermediate, then pack both bf16 values |
| `setp.<cmp>{.and,.or,.xor}.bf16` | Comparison of the two f32, ordered or unordered, with the same Boolean operation |
| `set.<cmp>.bf16x2.bf16x2` | Two comparisons, 1.0 or 0.0 as bf16 in each half |

For a packed FMA, VF tracks the bits set by moves, masks and bitwise OR. When both multiplicands
are constant, their product can sometimes be represented exactly in BF16. If the addend has the
opposite sign and its magnitude lies between half and twice the product magnitude, subtraction is
exact in BF16 (Sterbenz’s lemma). When both lanes meet these conditions, VF adds each constant
product in f32 and packs the high halves. This occurs when TensorFold unpacks affine 4-bit weights.

**One rounding, not two.** Computing an `fma` in f32 and then rounding to bf16 would round
twice, which can give the wrong answer: a first rounding can land on a halfway point, and
the second rounding (ties to even) goes the wrong way. This happens for 14% of `fma`
operands whose result is near a bf16 halfway point (measured on 300,000 each of `fma`,
`add`, `mul` and `cvt`). The fix:
the f32 result is rounded to odd first (the value whose last mantissa bit is 1, chosen from
the two directed roundings), which preserves the information the second rounding needs.
Cost: two `fma` and three instructions instead of one `fma`.

For `add` and `mul` the double rounding is harmless. The product of two significands of 8
bits has at most 16 bits: it is exact as an f32. The f32 sum of two bf16 is not always exact
(1 + 2^-30: both are bf16 values, and their f32 sum is 1), but rounding it to bf16 gives
what one rounding of the exact sum gives. The two could differ only for a sum that is within
half a unit of f32 of a point halfway between two bf16 values without being on it, and no
sum of two numbers of 8 significant bits is. This was computed rather than argued: for every
pair of significands, both signs and the 44 distances between the exponents from 0 to 43
(further apart, the smaller term changes neither result), on exact integers: 1,433,536 sums,
of which 852,096 are not exact as f32, and the two ways of rounding agree in all of them.
(A sum small enough to be subnormal as an f32 is exact, and bf16 has the exponents of f32.)

### Atomic additions

`atom.add.noftz.bf16` and `atom.add.noftz.bf16x2`: a compare-and-swap loop. The thread
reads the word, adds as above, and gives both the old and new values to `atom.cas`, which
writes the sum only if the word still holds the original value. If another thread
intervened, the loop retries with the new value. The `.sem` and `.scope` qualifiers of the
original instruction are placed on the `atom.cas` (`.cluster`, which Volta does not have,
becomes `.gpu`). The form without a result, `red.add.noftz.bf16{x2}`, is refused.

Under a predicate the entire loop is skipped: a thread whose predicate is false does not
touch memory. The bf16x2 form uses `atom.cas.b32` on the whole word. `atom.cas.b16` is
available from sm_70.

**Test cases.** `tests/gpu/run.sh`: `bf16` (15 kernels, 0 wrong). The arithmetic tests
use operands at and near bf16 halfway points, subnormals, NaN and infinity. The atomic
tests run 64 threads adding to the same address. The PTX cases `packed-bf16-*`,
`bf16-product` and `xla-sort-bounds` check more forms of the table, and the `bf16-mma-*`
cases run the products under each switch above (groups and tiles off and on, column and row
`ordered` and `tree`). The bounds: `packed-bf16-bounds`, `scalar-bf16-bounds` and `signed-zero-bounds` (their words agree with a native run on an RTX 5090).

## 5. Integer products

### Turing: m8n8k16

`mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` computes two 32-bit sums per thread: the
dot product of the thread's 16-byte row of A with two 16-byte columns of B. Volta has no
integer tensor-core instruction.

**On Volta.** Each thread's row of A (one register, four bytes) and its columns of B are
gathered from other lanes with shuffles. The B columns (eight registers, four for each of
the two columns) come from eight lanes, one shuffle each. Each sum is four `dp4a.s32.s32`
instructions, one for each group of 4 consecutive k-elements. The two sums are interleaved: three steps of the first,
three of the second, then the last step of each (which writes the destination register).
The accumulator is passed through.

Signed, unsigned and mixed forms are all supported (`dp4a.s32.s32` for signed, `.u32.u32`
for unsigned, `.u32.s32` or `.s32.u32` for mixed). Products on 4-bit integers (`.s4`, `.u4`)
and on bits (`.b1`) are refused.

**Example.** Before:
```
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
    {%d0, %d1}, {%a0}, {%b0}, {%c0, %c1};
```

After:
```
shfl.sync.idx.b32 %vf_r0, %a0, %vf_ia0, 31, -1;  // gather row of A (4 shuffles)
shfl.sync.idx.b32 %vf_r1, %a0, %vf_ia1, 31, -1;
shfl.sync.idx.b32 %vf_r2, %a0, %vf_ia2, 31, -1;
shfl.sync.idx.b32 %vf_r3, %a0, %vf_ia3, 31, -1;
shfl.sync.idx.b32 %vf_r4, %b0, %vf_ib0, 31, -1;  // gather columns of B (8 shuffles)
  ...
shfl.sync.idx.b32 %vf_r11, %b0, %vf_ib7, 31, -1;
dp4a.s32.s32 %vf_r12, %vf_r0, %vf_r4, %c0;     // three dp4a of sum 1
dp4a.s32.s32 %vf_r12, %vf_r1, %vf_r5, %vf_r12;
dp4a.s32.s32 %vf_r12, %vf_r2, %vf_r6, %vf_r12;
dp4a.s32.s32 %vf_r13, %vf_r0, %vf_r8, %c1;     // three dp4a of sum 2
dp4a.s32.s32 %vf_r13, %vf_r1, %vf_r9, %vf_r13;
dp4a.s32.s32 %vf_r13, %vf_r2, %vf_r10, %vf_r13;
dp4a.s32.s32 %d0, %vf_r3, %vf_r7, %vf_r12;     // last dp4a of each sum
dp4a.s32.s32 %d1, %vf_r3, %vf_r11, %vf_r13;
```

**Exactness.** Each `dp4a` computes the exact dot product of four int8 pairs; the sum of
16 such products cannot overflow an int32 (maximum 16 * 128 * 128 = 262,144). The emulation
gives the hardware's answer: 0 differences on an RTX 5090 (measured 2026-11-17).

**`.satfinite`.** The qualifier saturates the final sum to the int32 range on overflow. The
layer computes the product sum from zero (which cannot overflow) and adds the accumulator
last with `add.sat.s32`.

### Ampere shapes: m16n8k16, m16n8k32

Each is a composition of Turing products. `m16n8k16` is two Turing products (rows 0-7 and
8-15) with the same B. `m16n8k32` is four: for each half of the rows, the first 16 columns
then the last 16, the second accumulating on the first. The decomposition follows from the
fragment layouts of the PTX manual.

### A operands loaded from shared memory

Kernels keep many A operands in registers through long unrolled loops, four registers for
each. When the A operand was loaded by `ldmatrix` and nothing between the load and the
product can change the memory (no label, no store to shared memory, no call, no barrier),
the layer keeps only the address of the row and loads the 16 bytes at each product with
`ld.volatile.shared.v4.u32` (marked `volatile` so the compiler does not keep the result).
This frees the four gathered registers and eliminates stack spills: the integer kernels of
llama.cpp's q6_K format dropped from 3.95 times the native `dp4a` kernel's time to 1.88
(measured 2026-11-16 with llama.cpp's `test-backend-ops perf`, `tests/gpu/llamacpp-ops.sh`).
It takes `VF_ALIGNED_ROWS` on (the default, section 3); with `VF_ALIGNED_ROWS=0` the row is
gathered by shuffles.

**Test cases.** `tests/gpu/run.sh`: `tensor-core` (integer kernels among the 37),
`ampere` (31 kernels, 0 wrong).

## 6. FP8

### Conversions

**Decode (FP8 to f16).** Every value of `e4m3` and `e5m2` is exactly a half. `e5m2` has the
exponent of a half: its byte is the upper byte of the half. `e4m3` has a narrower exponent,
but its seven bits below the sign, placed in bits 13..7 of a half, give the value divided
by 256 (the exponent fields differ by exactly 8). Multiplication by 256 (a half) recovers
the value, exactly. The NaN byte `0x7f` of `e4m3` (which would decode to 480) is detected:
bits 14..9 are set to make a NaN of the same sign.

| Form | Instructions | Exact |
|---|---|---|
| `cvt.rn.f16x2.e4m3x2` | 30 (in the test kernel) | yes |
| `cvt.rn.f16x2.e5m2x2` | 14 | yes |

**Encode (f32 to FP8).** The value (divided by 256 for `e4m3`) is cut to a half towards
zero (`cvt.rz.f16.f32`), a bit records whether the cut dropped anything, and the byte is
that half without its low bits, rounded to nearest even on them and on that bit. What is
beyond the largest finite value becomes that value (`.satfinite`: 448 for `e4m3`, 57,344
for `e5m2`). A NaN becomes the NaN of the format. Rounding to a half first and then to the
byte would round twice and be wrong at the halfway points; the layer rounds once.

| Form | Instructions |
|---|---|
| `cvt.rn.satfinite.e4m3x2.f32` | 47 (for two values) |
| `cvt.rn.satfinite.e5m2x2.f32` | 45 |
| `cvt.rn.satfinite.e4m3x2.f16x2` | (the same technique on the f16 input) |
| `cvt.rn.satfinite.e5m2x2.f16x2` | |
| `cvt.rn.satfinite.e4m3x2.bf16x2` | (the same on the two bf16, widened to f32) |
| `cvt.rn.satfinite.e5m2x2.bf16x2` | |

The `.relu` forms of the FP8 conversions, and encodings with another rounding (`.rz`), are
refused.

### Products

`mma.sync.aligned.m16n8k32` on `e4m3` and `e5m2`, with f32 accumulators. The k = 32
layout gives a thread four consecutive k-elements in each register. Each pair of bytes
becomes a pair of halves (exact, as above), and the product is four FP16 `m16n8k8` in a
row (eight of Volta's `mma.m8n8k4`). No lane is asked for anything to decode, and the
halves of an operand used by several products are made once.

**Instruction count.** 225 to 353 instructions per product (the range is across the three
type combinations of the test kernels: e4m3xe4m3, e4m3xe5m2, e5m2xe4m3). An e5m2 operand
costs less to decode: e5m2xe5m2 takes 64 instructions fewer than e5m2xe4m3.

**Exactness.** The conversion to halves is exact, and the products are FP16 products on
Volta's tensor cores, with what section 2 says of them: on the inputs of the test suite,
where the sums are exact, no difference from the host model nor from an RTX 5090 (measured
2026-11-17); on arbitrary values, last bits that depend on how the sums are made (the
MXF8F6F4 row of section 8 shows how much on E4M3 operands).

**Test cases.** `tests/gpu/run.sh`: `fp8` (19 kernels, 0 wrong). Tests cover every byte
pair decoded, conversions at and around every value and halfway point of both formats,
products of the three type combinations, and operands from `ldmatrix`.

## 7. FP4 and exponent-only formats

### E2M1 (4-bit float)

A 4-bit float with 2 exponent bits (bias 1) and 1 mantissa bit. The 16 values: 0, 0.5, 1,
1.5, 2, 3, 4, 6, and their negatives. Every value is an exact f16 half.

**Conversions.** `cvt.rn.f16x2.e2m1x2` (decode): exact, lowered with `prmt.b32` byte
lookup (`0x3E3C3800`, `0x46444240` for the high bytes of magnitudes `0..7`, zero low bytes)
and `prmt.b32` + `lop3.b32` sign placement (without `HMUL2`). Because generic `prmt`
sign-extends bit 7 (0) of those LUT bytes whenever selector bit 3 is set (`8..15`), the
`.relu` variant clamps negative inputs including `-0.0` to `+0.0` in two `prmt.b32`
instructions. The upper nibble gives the upper half, the lower nibble the lower half.
`cvt.rn.satfinite.e2m1x2.f32` (encode): round to nearest, ties to even. NaN becomes +6
(the format has no NaN); values beyond +-6 become +-6. The `.relu` variant sets negative
inputs to +0.

### UE8M0 (unsigned 8-bit exponent)

An 8-bit format representing powers of two: 2^(e - 127) for bytes 1 to 254. Byte 0 is
2^(-127) (not zero). Byte 0xFF is NaN.

**Decode.** `cvt.rn.bf16x2.ue8m0x2`: exact. A byte e from 1 to 254 becomes the bf16 whose
exponent field is e, with no mantissa; bytes 0 and 0xFF become what the hardware gives.

**What the hardware does with byte 0 and 0xFF** (measured on an RTX 5090, 2026-11-17,
`tests/kernels/blackwell/ref/decode_ue8m0.bin`):
- Byte 0 decodes to bf16 `0x0040`, which is the bf16 subnormal for 2^(-127).
- Byte 0xFF decodes to bf16 `0x7FFF` (a quiet NaN, all mantissa bits set).

**Encode.** `cvt.rp.satfinite.ue8m0x2.f32` (round toward +infinity) and
`cvt.rz.satfinite.ue8m0x2.f32` (round toward zero). For subnormals: `.rp` gives byte 0
when the mantissa is at or below `0x400000` (which is exactly 2^(-127)) and byte 1 above
it; `.rz` gives byte 0 for all subnormals. Zero goes to byte 0. NaN to 0xFF. Infinity to
0xFE. The sign is ignored. The `.rn` modifier is rejected by the assembler (ptxas 13.5).
The from-bf16 forms (`cvt.{rz,rp}.satfinite.ue8m0x2.bf16x2`) follow the same rules.

### UE4M3 (the scale of NVFP4)

A 7-bit unsigned E4M3 (bit 7 is ignored): 4 exponent bits (bias 7), 3 mantissa bits. Byte
0 is zero. Byte `0x7f` is NaN. The 128 distinct values range from 2^(-9) (the smallest
subnormal) to 448. Measured exhaustively on an RTX 5090 (all 256 byte values,
`tests/kernels/blackwell/ref_scaled/`).

### Agreement with the RTX 5090

Every conversion of every value of the formats matches the RTX 5090's output byte for byte
(measured 2026-11-17 on the RTX 5090 and on a V100 through the layer: 201,010 records in 9
reference files, `tests/gpu/run.sh` `fp4-hardware`). The reference files are in
`tests/kernels/blackwell/ref/` and cover every byte of E2M1 (with and without `.relu`),
every pair of bytes of UE8M0, and every rounding boundary of the encode forms.

**Test cases.** `tests/gpu/run.sh`: `fp4` (10 kernels, 0 wrong), `fp4-hardware` (9
reference files, the same byte for byte).

## 8. Block-scaled products

Blackwell (sm_120) introduces block-scaled tensor-core products: `mma.sync.aligned` with
`.block_scale`, where the operands are E2M1, E4M3 or E5M2 and each block of k-elements has
its own scale factor. The layer emulates three forms:

| Kind | Operands | Scale type | Scale_vec | K |
|---|---|---|---|---|
| `kind::mxf4` (MXFP4) | E2M1 | UE8M0 | 2X | 64 |
| `kind::mxf4nvf4` (NVFP4) | E2M1 | UE4M3 | 4X | 64 |
| `kind::mxf8f6f4` (MXF8F6F4) | E4M3, E5M2 | UE8M0 | 1X | 32 |

The formula: D[r][c] = C[r][c] + sum over blocks b of (scale_A[r][b] * scale_B[c][b] *
sum over k in block b of A[r][k] * B[k][c]).

### What the layer computes

For each block:

1. For FP4 (`kind::mxf4` and `kind::mxf4nvf4`), packed 32-bit E2M1 operand registers are
   first shuffled to the split-octet `mma.m8n8k4.row.col` arrangement and then decoded in
   place to exact f16x2 halves via the `prmt`/`lop3` lookup (8 nibbles per `.b32` word,
   zero `HMUL2`). For E4M3 and E5M2 (`kind::mxf8f6f4`), bytes are decoded to halves as in
   section 6.
2. Each block is computed on Volta's `mma.m8n8k4` (section 2) into a zero-initialized f32
   accumulator. Both FP4 forms use the single-instruction `mma.m8n8k4.row.col` split-octet
   layout (8 Volta `mma.m8n8k4` instructions = 32 HMMA steps total for K=64, reducing across
   octets with four `bfly 8` exchanges per block). The f32 block sum is exact for E2M1
   (every term is a quarter, and sums reach at most 576 for NVFP4 or 1,152 for MXFP4).
3. Each block sum is multiplied by `scale_A * scale_B` and added to the `f64` accumulator
   (kept in split-octet layout across blocks for FP4). For NVFP4, UE4M3 scales are decoded
   to `value * 2^-8` in f32 (exact normal values) with `* 2^16` folded once onto `scale_A` per
   block, then combined and applied in f32 before widening to f64 (exact for the small
   integer-of-quarters block sums); MXFP4 and MXF8F6F4 keep UE8M0 scales in f64.
4. The final f64 sum is converted to f32 (truncation toward zero for `scale_vec::4X`,
   round-to-nearest-even for `scale_vec::2X` and `1X`, `0x7fffffff` on NaN) and, for FP4,
   gathered back to Turing's `D` layout once at the end.

### Scale distribution

The scales are read from specific threads and bytes of the scale register, controlled by
the `thread_id_a`, `thread_id_b` and `byte_id` selectors of the instruction.

**scale_A** (`thread_id_a = 0`): thread 0 of the group provides the scale for the upper
row (rows 0..7), thread 1 for the lower (rows 8..15).

**scale_B** (`thread_id_b = 0`): for column c, the scale comes from lane 4c.

**Blocks of k.** For `scale_vec::2X` (MXFP4): 2 blocks of 32, using scale bytes 0 and 1
(or 2 and 3 for `byte_id = 2`). For `scale_vec::4X` (NVFP4): 4 blocks of 16, split by
thread pair within the group: threads 0,1 for blocks 0 and 2, threads 2,3 for blocks 1
and 3. For `scale_vec::1X` (MXF8F6F4): 1 block of 32.

**NaN scales.** A NaN in scale_A makes every result of the row it controls `0x7fffffff`. A
NaN in scale_B makes every result of its column `0x7fffffff`.

### Selectors implemented and refused

The layer implements `thread_id_a` 0 and 1, `thread_id_b` 0 and 1, `byte_id` 0 and 2
(the two values the assembler accepts for `scale_vec::2X`). `thread_id_b` 2 and 3, and
`byte_id` 1 or 3 with a single scale, are lowered but have no reference vectors.
Selectors the instruction does not have (a `byte_id` above 3, a `thread_id_a` above 1, a
`thread_id_b` above 3, `byte_id` 1 or 3 with two scales, a `byte_id` other than 0 with
four) are refused when the program is read. So are the other operand types of
`kind::mxf8f6f4` (E3M2, E2M3, E2M1). `kind::mxf4nvf4` with UE8M0
scales is refused (not verified against the hardware).

### Agreement with the RTX 5090

Measured 2026-11-18 on an RTX 5090 and a V100. The counts are what `tests/gpu/run.sh`
(`block-scaled`) holds the layer to today:

| Kind | Cases | Total results | Identical | 1-ULP | Beyond 1-ULP | NaN mismatch |
|---|---|---|---|---|---|---|
| **NVFP4** (`kind::mxf4nvf4`, 4X) small | 284 | 36,352 | **36,352 (100.0%)** | 0 | 0 | 0 |
| **NVFP4** (`kind::mxf4nvf4`, 4X) large | 3,179 | 406,912 | **405,861 (99.74%)** | 1,051 (0.26%) | 0 | 0 |
| **MXFP4** (`kind::mxf4`, 2X) small | 284 | 36,352 | 36,242 (99.70%) | 106 | 4 | 0 |
| **MXFP4** (`kind::mxf4`, 2X) large | 3,079 | 394,112 | 371,635 (94.30%) | 21,636 (5.49%) | 841 | 0 |
| **MXF8F6F4** (`kind::mxf8f6f4`, 1X) small | 284 | 36,352 | 24,272 (66.8%) | 8,059 (22.2%) | 4,021 | 0 |
| **MXF8F6F4** (`kind::mxf8f6f4`, 1X) large | 3,029 | 387,712 | 308,475 (79.6%) | 52,538 (13.6%) | 26,699 | 0 |

The "small" sets are the original 284-case reference vectors; the "large" sets add
rounding discriminators, random accumulators and one-block tests.

**Where the forms differ and why.**

- **NVFP4 (4X).** The best agreement. Each block has at most 16 E2M1 values: the block
  sums are small (max 576) and fit exactly in f32. The f64 scale arithmetic adds no
  rounding. The 1,051 results at 1-ULP on the large set are from the truncation-to-f32
  boundary: the hardware's wider accumulator (~25 bits) rounds differently from the f64
  model at the edge.
- **MXFP4 (2X).** Larger block sums (max 1,152 per block). The same f32 tensor-core sums
  as NVFP4, but the rounding model (round-to-nearest-even) accounts for 5.5% at 1-ULP.
  The 841 results beyond 1-ULP on the large set (and 4 on the small set) are from
  extreme-scale cases (scales near 2^127 and large accumulators).
- **MXF8F6F4 (1X).** The differences come from the sums of the FP8 product itself (Volta's
  product adds four terms at a time in f32, the hardware in a wider accumulator: section 2),
  not from the scale arithmetic. With all scales at 1 (UE8M0 byte 127), the
  block-scaled form gives identical results to the plain FP8 product (measured on the
  RTX 5090, 12,800 results).

Extra forms tested against the RTX 5090 (300 cases each): `thread_id_a = 1`,
`thread_id_b = 1`, `byte_id = 2` with two scales, and E5M2 operands (the `kind::mxf8f6f4`
form). All consistent with the main forms, no NaN mismatches.

**Test cases.** `tests/gpu/run.sh`: `block-scaled` (10 forms checked, counts as above).
The reference vectors are in `tests/kernels/blackwell/ref_scaled/`.

## 9. Other instructions

### tanh

`tanh.approx.f32` is expanded to `1 - 2 / (exp(2x) + 1)`, using `ex2.approx` and
`rcp.approx`, and `x` itself where `|x| < 0.004`. These are instructions every target has.

### Instructions with a plain equivalent

| Instruction | On Volta |
|---|---|
| `mbarrier.init.shared.b64`, `mbarrier.arrive.shared.b64`, `mbarrier.test_wait.parity.shared.b64`, `cp.async.mbarrier.arrive.noinc.shared.b64` | A fixed arrival count and a wrapping, CTA-scoped atomic counter in the original 8-byte shared object. An acquire load checks the phase. Each synchronous copy finishes before its arrival. `.shared::cta` is also accepted. |
| `cp.async.ca.shared.global [dst], [src], size` (4, 8 or 16 bytes), `cp.async.cg` (16 bytes) | A plain load from global memory and a store to shared memory, under the guard of the copy. An operand that says how much of the source is read (the rest is zeros), or a predicate that says to ignore it, is followed: nothing or all takes one predicated load, and so does an amount in a register that the function writes only once, with a choice between two numbers that are each nothing or all (`selp.b32 %r, 16, 0, %p`, as Triton writes its pipelined loads); another amount known only when the kernel runs takes a byte loop behind a branch. A prefetch size (`.L2::64B` to `.L2::256B`) is dropped. `cp.async.commit_group`, `cp.async.wait_group` and `cp.async.wait_all` become no-ops (the copies are synchronous). |
| `ld.global.v8.b32` (256-bit load, and the others: `.v8` of 32-bit types, `.v4` of 64-bit types) | Two `ld.global.v4.b32` (128-bit loads). The second half is loaded first when the base register is a destination of the first half. |
| `st.global.v4.b64` (256-bit store, and the others) | Two `st.global.v2.b64` (128-bit stores). |
| `cvt.{rn,rz}.f16x2.f32` (pack two f32 to f16x2) | `cvt.rn.f16.f32` (or `.rz`) for each, then a shift and an `or` to pack, the first operand in the upper half. |
| `cvt.{rn,rz}.bf16.f32`, `cvt.{rn,rz}.bf16x2.f32` | Integer arithmetic on the bits: the upper half of the f32, rounded to nearest even on the lower half for `.rn`, cut for `.rz`; a NaN gives `0x7fff`. For bf16x2, the first operand in the upper half. |
| `cvt.rn.tf32.f32` | The 10-bit mantissa (zero the low 13 bits), ties to even: 9 instructions. |
| `redux.sync.add.{s32,u32}`, `redux.sync.{min,max}.{s32,u32,f32}`, `redux.sync.{and,or,xor}.b32` | Unguarded, with a member mask of the whole warp: a constant, or a register that holds all ones where the instruction reads it (the only statement of the function that writes it is an unguarded `mov` of all ones, or such a `mov` comes before the instruction with nothing between that writes the register or that a branch can reach: labels that no statement names, as compilers put for their line tables, do not count, a branch out does not). A ballot identifies the lanes that have not exited. A full warp uses five butterfly shuffles. A partial warp gathers the input of each participating lane once, from the highest lane down: in the same order in every lane. Other forms are refused: a type the PTX ISA does not give the operation (`redux.sync.add.f32`, `redux.sync.and.u32`), `.abs`, a guard, another member mask. |
| `redux.sync.{min,max}.NaN.f32` | The same reduction, and a warp vote to propagate a NaN from any participating lane. |
| `{min,max}{.ftz}.NaN.f32` | Compare two or three operands and propagate a NaN from any operand. |
| `{min,max}{.ftz}.f32` with three operands | Two of the instruction with two operands. |
| `{min,max}.relu.s32` | The operation, then `max.s32` with 0. Other `.relu` forms are refused, except those of the E2M1 conversions (section 7). |
| `{add,sub,mul,fma}{.rnd}{.ftz}.f32x2` | Arithmetic on pairs of f32, packed in a 64-bit register. Each is two scalar operations. Other `.f32x2` operations are refused. |
| `ld` and `st` with `.L2::cache_hint`, `.L2::evict_*` or a prefetch size; `createpolicy` | The hints and the cache-policy operand are dropped; the access is otherwise unchanged. `createpolicy` writes 0. `.volatile` on `.local` memory, which Volta's compiler refuses, is dropped too. |
| `min{.NaN}.f16`, `max{.NaN}.f16` and their `.f16x2` forms | Each half widened exactly (`cvt.f32.f16`), `min.f32`/`max.f32`, converted back (`cvt.rn.f16.f32`, exact: the result is an input). A NaN input gives the other input, two NaN give 0x7fff; with `.NaN`, 0x7fff. +0 > -0, subnormals kept. As CUDA's `__hmax`/`__hmin` below sm_80. The `.ftz` and `.xorsign.abs` forms are refused. |
| `griddepcontrol` | Dropped (a Hopper/Blackwell instruction for grid-level dependencies, which Volta's driver does not support). |

### What is refused

An instruction that has no Volta sequence makes its function trap; the kernels that need
that function are refused when the program asks for them. The refusal names the kernel and
the reason (on the console and in the error returned to the program, error 218).

**Instructions without a sequence** (from the PTX census of NVIDIA's libraries): clusters
of blocks (`barrier.cluster`, `mapa`, `getctarank`, `clusterlaunchcontrol`, memory of
`::cluster`, the registers `%cluster_*`, and a kernel whose directives ask for a cluster:
`.explicitcluster`, `.reqnctapercluster`, `.maxclusterrank`), the `mbarrier` forms the table
above does not list (`mbarrier.try_wait`, `mbarrier.expect_tx`, `mbarrier.arrive_drop`, ...),
bulk copies (`cp.async.bulk`) and proxy fences (`fence.proxy.async`, `fence.proxy.tensormap`,
`fence.mbarrier_init`), tensor maps (`tensormap`), multicast memory (`multimem`, but see
below), `wmma` on integers, bf16, tf32 or f64, `cvt.pack`, and other `stmatrix` shapes and
layouts. A form on a type of later targets (bf16, tf32, FP8, E2M1, UE8M0, UE4M3) that the
sections above do not list is refused too (`neg.bf16x2`, `cvt.rna.tf32.f32`). These
forms are refused on Volta.

**Reported traps: the multimem family.** `multimem.ld_reduce`, `multimem.st`, `multimem.red` and
the other `multimem.*` instructions operate on a multimem address, and the PTX manual leaves them
undefined on any other ("Address operand a must be a multimem address. Otherwise, the behavior is
undefined", PTX ISA 9.3, sections 8.2.3 and 9.7.10.15). Such an address is the mapping of a
multicast object, whose team can only hold GPUs that have `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`;
a V100 has not (the 580 driver answers 0, and `cuMulticastGetGranularity` and `cuMulticastCreate`
return 801, `CUDA_ERROR_NOT_SUPPORTED`, without the layer and through it). So in a process where
the driver answers 0 for every GPU, no such instruction can be reached with a defined result, and
the layer replaces each one, not its function, by a reported trap: under the instruction's guard,
a call of `__assertfail` (the file `volta-forever`, the number of the trap in its function, the
function's name, the message "FORM cannot run on this GPU: it needs a multicast address, which no
GPU of this process can have; volta-forever put this trap in its place") and a `trap` after it.
What the instruction would have written is not written: nothing of the thread runs after it. The
rest of the function is lowered as any other, and its kernels are given, said in one line for
their image (`LIMITS.md`, "Several GPUs and NCCL"). Reached all the same, the kernel stops with
`CUDA_ERROR_ASSERT` (710), the driver writes the assertion with the block and the thread, and the
context ends, as for any trap. Where a GPU of the process has multicast, with `VF_REPORTED_TRAPS=0`,
in modules of 32-bit addresses and in what `vf ptxas` and the nvJitLink adapter make, the function
traps and its kernels are refused, as for the forms above. No other form is replaced so: a program
can execute any of them.

**What the driver's compiler refuses.** The lowering rewrites or traps the instructions it
knows. The driver may still refuse what remains: a function whose compiler error is
collected is replaced by a trap, and its kernels are refused at the lookup
(`crates/vf-ptx/src/refused.rs`). The other kernels of the module work.

**Test cases.** `tests/gpu/run.sh`: `later` (3 kernels, 0 wrong) and the PTX cases
`cta-mbarrier`, `ptx-reduction-exited`, `ptx-reduction-exited-f32`, `guarded-copy` and
`chosen-copy` for the table above; `refuse-library`, `refuse-module` (through both driver
interfaces: a refused kernel is refused at the lookup, the others run; an emulated kernel
is announced); `reported-library`, `reported-module`, `reported-library-lazy-off`, `reported-off`
for the reported traps (`tests/kernels/test_reported.c`). Half bounds: `f16-bounds` and
`signed-zero-bounds`.

## 10. The same emulations in machine code

The machine-code translator (`crates/vf-sass/src/translate/`) rewrites sm_75 cubins, and a
subset of sm_80 ones (`docs/translator.md`), to sm_70, function by function. The tensor-core
instructions it replaces are those of Turing:

| sm_75 instruction | Replaced by | Notes |
|---|---|---|
| `LDSM` (ldmatrix), 1/2/4 matrices | Shuffles, loads (same logic as section 3) | 17 instructions for 4 matrices |
| `LDSM` transposed | Shuffles, 2-byte loads, assembly | 38 instructions for 4 matrices |
| `HMMA.1688` (mma.m16n8k8), f32 accumulators | Two of Volta's `HMMA.884` (8 steps), B gathered by 5 shuffles | 25 instructions; 14 when B is kept |
| `HMMA.1688`, f16 accumulators | Same with 4 steps and exchanges | 31 instructions; 20 when B is kept |

Differences from the PTX path:
- The B operand is **kept** across products that use the same register: a product after one
  that gathered from the same B, with nothing writing that register in between, skips the
  gathering (11 instructions fewer). This is what `tests/gpu/run.sh` `kept` and
  `kept-machine-code` test.
- Products are done **in pairs on whole rows** where two products read the same A and the two
  B registers of a pair, with operands that unconditional `LDSM` brought: the two are computed
  by two of Volta's products on rows loaded directly in Volta's layout, with no gathering at
  all (`docs/pairs.md`). The `pairs`, `pairs-machine-code`, `groups` and
  `groups-machine-code` test cases exercise this form.
- Masks and selections replace predicates (the translator must not use a predicate of the
  program).
- The lane is read once at the function's entry.

**What is not in machine code.** Integer products (`IMMA`; emulated with `dp4a` in PTX
only), the transposition of a fragment (`MOVM`, movmatrix), FP8 products, block-scaled
products, bf16 and tf32 products, bf16 arithmetic, conversions of FP8/E2M1/UE8M0, and the
other instructions of section 9: these appear only in PTX and are handled by the PTX
lowering. Two exceptions come with sm_80 machine code: the 16-byte asynchronous copy from
global to shared memory becomes a load and a store under its predicate, and the conversion
of two f32 to f16x2, `F2FP.PACK_AB`: each f32 is `cvt.rn.f16.f32`, a single rounding to nearest
even, `Ra` in the high half and `Rb` in the low one. The translation is two `F2F.F16.F32`, the
high one into a register of the translation, and a `PRMT` (selector `0x5410`) that packs their
low halves into the destination; `Rd` may be `Rb`. Three instructions for one. The CUDA 13.5
cuBLAS half GEMM uses it: on a V100 all 65,536 outputs of the probe's GEMM are the CPU
reference's, and the same probe run natively on an RTX 3060 (where the F2FP is the hardware's)
gives the same (`runtime-13.5-ptx` of `tests/gpu/run.sh`). And two for bfloat16, which vLLM's
vectorized bf16 norm uses (2026-11-25):

- `HFMA2.BF16_V2 Rd, Ra, 1, 1, Rc`, what NVIDIA's assembler makes of `fma.rn.bf16x2` with a
  multiplier of 1.0 in both halves (checked by assembling that PTX for sm_80), the sum of two
  pairs of bf16: each half of each operand widened to an f32 (a shift, or a mask), `FADD`, and
  the sum rounded to bf16 as below; the two results packed with `PRMT`. One rounding of the
  exact sum and this give the same bf16 (section 4, "One rounding, not two": computed for every
  pair of significands). With any other multiplier the instruction is not translated (a product
  and a sum rounded twice can differ).
- `F2FP.BF16.PACK_AB Rd, Ra, Rb`: each f32 rounded to bf16, Rb in the low half.

The rounding, in integers, without a predicate: the upper half of the f32 plus the carry of
`0x7fff + the lowest bit that stays` on its lower half (to nearest, ties to even; a carry out
of the significand raises the exponent, and gives infinity at the top), and `0x7fff` in place
of that when the magnitude is above `0x7f800000` (a NaN). 23 instructions for the sum, 17 for
the pack. The expected values are the arithmetic's (the PTX manual's `fma.rn`: the exact
result rounded once to nearest even): on a V100, 16,384 sums of pairs chosen at and around the
points halfway between two bf16, zeros, subnormals, infinities and at random, and 16,384
conversions, give the host's exact rounding, every one (`told-8.0-*` of `tests/gpu/run.sh`),
and the sequence gives it for every one of the 2^32 f32 on a host model. For a NaN the layer
gives `0x7fff`, as its PTX lowering of the bf16 instructions does; what an sm_80 GPU gives for
the bits of a NaN from these two instructions was not measured. Of the tensor-core instructions,
the machine-code translator handles Turing's matrix load and product on halves; a function
with `IMMA` or `MOVM` is refused.

The mechanism of the translator (register allocation, timing rules, what is refused, the
two implementations) is described in `docs/translator.md`.

## 11. cuBLAS bf16 paths (opt-in)

cuBLAS 12.9 computes a bf16 product with fp32 accumulation (`cublasGemmEx`,
`cublasGemmStridedBatchedEx`, `cublasLtMatmul` with `CUBLAS_COMPUTE_32F`) on a V100 with scalar
kernels. Two opt-in paths compute some of these calls otherwise (`INSTALL.md`, "The bf16 cuBLAS
variants"; `docs/architecture.md`, section 6). Every other call, and every call they decline, goes to
cuBLAS 12.9 with its arguments as given. The runtime is `crates/vf-fwdgen/bf16tc/runtime.c`, its
kernels `crates/vf-fwdgen/bf16tc/kernels.cu`.

### The skinny path (`VF_CUBLAS_BF16_SKINNY=1`)

For a product with few columns (2 <= n <= 16, or n < 32 when m k < 2^20), one kernel computes every
element as one fp32 fused multiply-add chain over l = 0, 1, ..., k-1, from +0:

- each product of two bf16 values is exact inside the fused multiply-add (also when it lies below
  2^-126 or above the largest fp32: `fma(2^64, 2^64, -2^104)` gives the largest fp32, not inf); one
  rounding to nearest per step; no flush of subnormals;
- then `alpha * acc`, rounded; without a bias, +0 added to it, so that a -0 becomes +0 (with alpha < 0
  and a sum that is exactly zero, `alpha * acc` is -0, and cuBLAS 12.9 gives +0 even when `beta * C`
  is -0); then `beta * C` added by one fused multiply-add when beta is not zero (when beta is +0 or -0,
  C is not read); then the bias of a cuBLASLt bias epilogue, added and rounded. cuBLAS 12.9 always
  adds a bias term: the bias element, or +0 without a bias; so with a bias of -0, a zero result is -0
  when `alpha * acc` and `beta * C` are -0 too;
- a NaN in fp32 output is the GPU's own (0x7fffffff); in bf16 output every NaN is 0x7fff; bf16 output
  is rounded to nearest even.

This is what cuBLAS 12.9's sequential kernel on this GPU (`magma_sgemmEx_kernel`) computes, bit for
bit, on every input that `tests/gpu/forward/bf16-arith.c` puts to both: subnormal products, a product
brought back from overflow, beta 0.3 and 1/3 with alpha 0.7, the bias with alpha 0.7 and beta 0.3,
device-mode alpha = 0 with NaN, rows of -0 products, NaN payloads, and signed zeros (alpha -1 and a
zero sum with beta * C = -0: beta 1 or 0.5 with C = -0, beta -1 with C = +0, in both pointer modes and
both output types; and through cuBLASLt with no bias, a bias of +0 and a bias of -0, with alpha -1, 1,
-0 and +0, beta 0, -0, 1, -1 and 0.5 and C = +0 or -0). But 12.9 does not always run that
kernel: on some shapes it runs `gemmSN_TN_kernel` or `gemmSN_NN_kernel`, which sum in another order,
and its choice depends on the shape, the layout, the alignment of the operands and the workspace of
the handle. The skinny path therefore serves only where 12.9 was seen to run its sequential kernel
(`tests/gpu/forward/bf16-skinny.c` reads the names of the kernels of 12.9's captured calls, over
266,195 configurations):

| op(A) op(B) | served when |
|---|---|
| NT, TT | always (m >= 2): 12.9 ran its sequential kernel in all 130,096 such configurations, whatever the alignment, the workspace or the interface |
| TN | A, B, C at 16-byte addresses, lda, ldb, ldc, m, k multiples of 8, a workspace of cuBLAS's default or of at least 4 MiB at a 256-byte aligned address, 448 <= m <= 16384, 256 <= k <= 4096 |
| NN | the same conditions, with 8 <= m <= 9728, 896 <= k <= 16384 |

(A workspace of none, 4 KiB, 64 KiB, 256 KiB, 1 MiB or 2 MiB changed 12.9's choice on 202, 202,
100, 100, 158 and 158 of 1,120 configurations; 4, 8 and 32 MiB on none. cuBLAS gives a handle its
default workspace when it is made and at every `cublasSetStream`; with `CUBLAS_WORKSPACE_CONFIG`
set, the default does not count.) Applied as coded, with the bounds below, the rule serves 101,595 of
the 266,195 configurations (the grid and 18,961 random ones not on it), all where 12.9 ran its
sequential kernel alone; it forwards every configuration where 12.9 ran `gemmSN` and some where it
did not. Executed on 1,578 configurations in
these regions, the path served 1,206 (the others lie outside its bounds, below), each with 12.9's
bits in every element; at the rule's edges (each bound just inside and just outside), with operands
at unaligned addresses, padded leading dimensions and each class of workspace (160 configurations,
both output types, n = 8 and 16), the 76 it served had 12.9's bits too, and neither it nor 12.9
wrote outside the m rows of a column. Where 12.9 runs `gemmSN`, that kernel is the more accurate
one (at m = 777, TN, k = 1536: 0.35-0.45 units of 2^-24 sum |a b| against 1.45-2.38 for the chain),
and the skinny path does not serve. The rule rests on samples of one GPU and one driver: it is not a property that
cuBLAS documents. `VF_CUBLAS_BF16_VERIFY=1` checks it on a program's own calls: each call the path serves is also
made by 12.9 and the two outputs compared bit for bit (`INSTALL.md`, "The bf16 cuBLAS variants").

The path also has bounds of its own, a rule of performance and not of correctness (inside them and
outside, the bits are 12.9's where the rule above serves), from the time it takes against 12.9's (G1,
the same tool, timing both on each shape the rule serves): k at most 16384; with op(B) transposed,
where it reads op(B) by elements, either 256 <= k and m <= 4608, or 64 <= k and 128 <= m <= 1536
(outside, at large m or very short k, it was slower than 12.9, down to 0.16 times); with NN,
m >= 128. On the 1,206 configurations above, executed with the rule and these bounds in place, it
was 1.16 to 20 times as fast as 12.9 (median 3.2 for NT, 3.9 for TT, 4.8 for NN, 6.1 for TN, the
layout of a framework's linear layers), and also where 12.9's sequential kernel is slow, at small m
and long k (m = 2 to 1024, k = 256 to 16384: 2.8 to 8.3 times with op(B) transposed, 4.9 to 19
times for TN).

### The tensor-core path (`VF_CUBLAS_BF16_TC=1`)

For the other products in the TN layout (m k >= 2^20 or n >= 32, with 768 <= k <= 16384, outside the shapes
that the rules of speed below leave to 12.9, by class of shapes: the gate below), when the
call is not captured in a graph, each row of op(A) and each column of
op(B) is scaled by a power of two, 2^(15 - e) for e the exponent of its largest magnitude, and
converted to fp16; a row or column is kept only when every element converts exactly (the exponent of
its maximum in [-24, 31], no infinity, NaN or subnormal maximum, no element that would round or become
zero). cuBLAS 12.9's fp16 GEMM with fp32 output then computes the product over fixed slices of k of
256 (a strided batch, on a cuBLASLt handle of the runtime's own, without workspace), and the
epilogue sums the slices' partials in fp32 in slice order from +0, multiplies by 2^(e_i + f_j - 30)
(exact), and applies alpha, beta and the bias as the skinny path does. The products are exact; the
tensor cores add with truncation (section 1), so long sums of one sign can come out low: in the
design's measurements of this arithmetic (before the runtime existed), over 21 shapes the worst error
against fp64 was 24.6 units of 2^-24 sum |a b| for k <= 16384, with a mean bias of -13 units, whatever
kernel cuBLAS chose for a slice; the same arithmetic reached 32.8 units at k = 65536, so longer
products are forwarded, and an RTX 5090's own bf16 product gave 37.9 to 42.1 units on the same
same-sign data at k = 1024. On this runtime, with both operands uniform in (0, 1], over 256 sampled
elements, the largest error was 19.4 units at TN 4096 x 256 x 16384, 16.0 at 7344 x 512 x 4096 and
17.6 at 2048 x 64 x 1024, and the mean error -13.1, -13.1 and -13.2 units: such sums come out low by
about 13 units, the truncation's bias (cuBLAS 12.9 on the same data: largest 96.5, 36.8 and 19.2
units, mean -1.2, -1.9 and 0.1). With both operands in one binade, uniform in [0.5, 1), a slice's
256-term sum of products is exact (16-bit products and 8 bits of growth fit in fp32's 24) and the
truncation does not act: largest 5.8, 3.2 and 1.8 units, mean 0.3, -0.1 and 0.0
(`tests/gpu/forward/bf16-cost.c`). Sums of products of one sign spread over many binades come out lower: by
36 to 40 units on average (log-normal magnitudes) and 49 to 70 units (an adversarial positive construction) at
every k from 768 to 16384, single elements up to 90 and 216 units (the acceptance test's lognpos and advpos,
"The tensor-core path's gate"). The GPU tests hold every tensor-core result to 32 units.

The elements of the rows and columns that fail the check are computed by the skinny path's chain (a
tiled kernel, one fp32 fused multiply-add chain per element), with the same epilogue, in one launch
after the tensor-core elements of every chunk: they are 12.9's where 12.9 runs its sequential kernel
(also measured at TN 1536 x n x 8960 for n = 64, 512 and 2048, `tests/gpu/forward/bf16-fallback.c`).
Every element has one writer; an element's value depends on
its row and column of op(A) and op(B) and on the call's shape (the gates, the chunks of at most
32 MiB of workspace, and the kernel cuBLAS chooses for a chunk's slices), never on other rows, other
calls or what ran before. With beta = 0, C is not read. Two runs give the same bits.

A tensor-core call captured in a graph goes to cuBLAS 12.9 (its workspace would be a node of the
program's graph: `LIMITS.md`, "The bf16 cuBLAS variants"), so a replay has 12.9's bits, not those of
the same call made outside a capture; so does a call whose workspace cannot be allocated. The variants do not change cuBLAS's workspace contract: a forwarded call uses the
handle's workspace as 12.9 does, with the same rules for threads and streams.

### The tensor-core path's gate

The tensor-core path takes a TN product with n >= 2 and k from 768 (`VF_TC_MIN_K`) to 16384 when m k >=
2^20 or n >= 32, after the skinny path has declined it; a product it does not take goes to cuBLAS 12.9. Three
rules of speed leave more products to 12.9, the classes of shapes in which the path's median speed was below
1.10 times 12.9's or its slowest below 0.90 times, or with no layer of a real model measured, decided before the
shortest k, so that they hold whatever
`VF_TC_MIN_K` is: the layout (TN only: in the other layouts the path was slower than 12.9 on the backward
products of training, input gradients NN and weight gradients NT), narrow products of tall matrices and long
products of many rows (m >= 8192 with n <= 64, or m >= 6144 with k > 9216: 12.9 fills the GPU and was as fast or
faster at some of these shapes; no layer of a real model was measured among the long ones), wide down projections
(k > 8192 with n > 256: their gated inputs fail the exact check often enough on real text that the fallback cost
more than the path saved) (`LIMITS.md`; `docs/performance.md`, section 7); a product they serve can still run below
12.9's speed: the classes' floor of 0.9 times holds for the products measured, and a product whose input has many
columns failing the exact check can run well below it, at long k at about half of 12.9's speed on synthetic inputs
(`LIMITS.md`). The acceptance test's NN, NT and TT calls therefore go
to 12.9.
The gate depends on n, so one element's bits depend on the batch: the same row of op(A) times the same
column of op(B) has 12.9's bits in a narrow product (n = 1, or within the skinny path's rule) and the
tensor-core path's in a wide one (`LIMITS.md`). n = 1 is never taken: 12.9 computes it faster. Were
it taken, its internal fp16 GEMM would give column 0 other bits than at n >= 2 (measured on 15
shapes, where n = 2 to 512 agreed); a zero second column given to that GEMM is the expected remedy
(not run on a GPU).

The shortest k is 768. Real layers pass from the lowest k measured, with wide margins: on 1,728 operand
pairs taken from the linear layers of four language models (k = 896 to 12288), against the exact product,
every element's error was at most 0.045 times the bound A below and every matrix's bias at most 0.22 units
(B below); on the layers of k = 768 and 896 of five models (Qwen2.5-0.5B, T5-base, OPT-125m,
bge-base-en-v1.5 and whisper-small's encoder: 990 operand pairs of up to 512 tokens), every element's
error at most 0.040 times A in fp32 output (in bf16 output, within the rounding of the output itself: 0.98
at most) and every matrix's bias at most 0.82 units (0.046 times B read with the slope of samesign, the
smallest of the test's families of one sign: no newer GPU's product was measured on these operands); and
at the level of the models no gate from 4096 down to none could be told apart from cuBLAS 12.9 beyond a
noise control of one unit in the last place. Two of the acceptance test's synthetic families, of positive
data spread over many binades, exceed its bias bound below 1024 (below). A model gains on its layers with
768 in-features or more.

The acceptance test (`tests/gpu/forward/bf16-accept.c`, run by `tests/gpu/forward-bf16.sh`) uses 10
families of data (normal, squared ReLU, outliers, cancelling pairs, spreads over about 30 binades, one
sign, log-normal, an adversarial positive construction, and two built against the path's 256-element
slices, with large values at the start of every slice) and the configurations of an adversarial search,
at k = 256 to 16384 (1024 x 128, TN, both output types; NN, NT and TT at 2048; alpha and beta with a
general C, the cuBLASLt bias epilogue). A call is accepted when every element's class and NaN bits are
12.9's, two runs, a handle without workspace and a call of half the columns give the same bits (S1, S2),
each element's error is at most the larger of 12.9's own error and (k - 1) u sum |a b| + h (A; u = 2^-24,
h half an ulp of the output at the exact value; with alpha, beta or a bias, (k + 2) u S' + h, S' adding
|beta c| and |bias|), and, per matrix with fp32 output, the mean error towards zero in units of u sum |a b|
is at most the larger of 12.9's and 0.8 times that of an RTX 5090's own accumulation on the same data (B;
its native bias slope per family, measured once at k = 1024 to 2048, where that card does not split its
sums; the constants are in the test). The error bound held from k = 384 on (at k = 256 one family reached
1.54 times it, the search 1.29).

Four of the test's data sets exceed the bias bound from the gate on, and the test judges their bias from a
larger k (documented exceptions; `VF_ACCEPT_STRICT=1` removes them, and the test then fails): two
constructions against the path's slices, and two families of positive data spread over many binades, below
k = 1024:

- **craftsub**, built against the path's 256-element slices: a product of 2^20 at the start of every slice,
  then products of half to one ulp of it. Its bias is judged from k = 4096, where it is 0.85 times its bound;
  it is 1.50 times its bound at 1024, 1.29 at 1536 and 1.20 at 2048 (-382.32 units against 318.32, there
  12.9's own bias).
- **The search's configuration 1653006390**: non-negative operands with a 2^21 activation at the start of
  every slice. Its bias is judged from k = 2048, where it is 0.95 times its bound (-183.65 units against
  192.7); it is 1.27 times its bound at 1536 and 1.90 times at 1024.
- **lognpos**, log-normal magnitudes of one sign: its bias is judged from k = 1024; it is 1.16 times its bound
  at 768 (-39.51 units against 34.0) and 0.91 times at 896.
- **advpos**, an adversarial construction of positive operands (log-normal magnitudes with a scale per row of
  op(A) and per column of op(B), one element of op(A) in 1,000 made 2,048 times larger, and the first element
  of every 256-element slice of op(B)'s columns made 8 times larger): its bias is judged from k = 1024; it is
  1.25 times its bound at 768 (-52.83 units against 42.3), 1.003 times at 896 (-49.45 units against 49.3,
  within the half unit that the test adds to every bias bound) and 0.98 times at 1024 (-55.48 units against
  56.4).

lognpos has nothing aligned with the path's slices, advpos only the larger first element of each slice of
op(B), and that element is not what makes it exceed: moved to position 128 or 37 of each slice, or removed,
its bias at k = 768 stays at -51.8 to -52.5 units (measured in review). Both exceed for their data, products
of one sign spread over many binades: what a program whose operands are all positive over a wide range should
expect at k = 768 to 1023 (`LIMITS.md`). Their bias grows little with k, the bound in proportion to it:
lognpos's is -36.1 to -39.6 units and advpos's -49.5 to -70.0 (its largest at 16384) at every k from 768 to
16384, against bounds of 34.0 and 42.3 at 768 that double by 1536. Their error stays within the bound A at every k the path serves (at most 0.25 times
it in fp32 output).

Every call from the gate on must be served by the path (a call left to 12.9 there would not be judged), and
the test fails with the path off; it fails when the gate is moved to 512 or 256 (a test build's
`VF_BF16_TC_MIN_K`; on other families than lognpos and advpos, whose bias it judges from 1024), and with
`VF_SAFE=1` it requires that no call be served by the tensor-core path and every result be 12.9's.
