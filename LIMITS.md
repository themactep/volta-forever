# volta-forever: known limits

What does not work through the layer, what is slower, and what was not tried. Everything was
measured on one Tesla V100 with driver 580.178.04 and no switch set; a figure carries its date
where it is not from the build of this release. `docs/` says how each figure was obtained:
`docs/validation.md` for what was compared with what, `docs/performance.md` for the timings,
`docs/emulations.md` for what is computed in place of the instructions that a V100 lacks.

## The limits of 0.1.6 to know first

- **Triton's `tl.dot` uses no tensor cores at the default capability, unless asked.** Told 7.5,
  which is what Triton is told by default in a program with PyTorch, Triton 3.8 multiplies with
  float32 `fma`: its float16 and bf16 matrix products ran 3 to 150 times slower than the same
  kernels told 8.0 ([Speed](#speed)). `VF_TRITON_TC=1` (opt-in, through the launcher) compiles
  Triton's float16, bfloat16 and int8 products for sm_80 while everyone is still told 7.5, at the
  default capability only: its float16 products are then summed by Volta's tensor cores, which
  truncate, and its bfloat16 and int8 products are the layer's emulated ones ([Triton's matrix
  products](#tritons-matrix-products)). Otherwise, in a program with PyTorch,
  `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` is the way to the tensor cores, and PyTorch's own
  attention is then refused ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **Triton's machine code is validated on a corpus, not kernel by kernel.** Kernels that Triton
  makes while a program runs, and that run from their sm_75 machine code (the default for a
  program told 7.5), were run on a V100 in 3,516 configurations of 20 kernels and 241 cases of
  vLLM's and SGLang's kernels (`tests/gpu/triton.sh` runs 180 of them): every one that Triton
  compiles gives what Triton's interpreter computes where it computes it (bit for bit, or within
  the error the case states) and what the PTX of the same kernel gives, and the independent
  validator finds no problem in their 4,308 cubins nor in the 334 of this site's runs
  (2026-11-23, with the translator and the rules of 0.1.4). A translation that stored
  wrong values in Triton's fp32 `tl.dot` kernels was fixed in 0.1.3. The kernels that Triton
  makes for other programs, shapes and settings were not looked at (`docs/validation.md`,
  sections 4, 6 and 8).
- **The translation of sm_80 machine code is not independently validated**: the validator reads
  sm_75 code only ("Cubins for sm_80 and later", below; `docs/validation.md`, section 8). Told 8.0,
  a kernel that only the sm_80 machine code of its image has is now given from that code,
  translated, where its lookup failed in 0.1.5: compare its results; `VF_TOLD_CODE=0` makes the
  lookup fail again ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **`VF_CAPABILITY_ALL=1` no longer reaches the libraries that ask for themselves.** It tells the
  value asked for to a CUDA runtime and to the code that uses it (PyTorch's library) only; cuDNN,
  cuFFT and cuSOLVER, which ask the driver without that runtime, are told what their own code
  allows, 7.5 for code that runs from sm_75 machine code ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **A refused kernel that a program obtains without a lookup is refused where it is used, and
  some uses still trap.** The layer refuses a kernel when the program looks it up; a handle that
  the program holds all the same (cuFFT enumerates the functions of its modules through a private
  table of the driver) is refused at a launch, in a kernel node of a graph made or changed with
  it, and in a question about it or a setting of it: error 218, and a line names the kernel. What
  still traps: a kernel launched by another kernel (dynamic parallelism), a function reached by an
  indirect call or through a call into another linker input, and a launch through a private
  interface of the driver ([Kernels refused](#kernels-refused)).
- **A kernel whose code is a trap and nothing else is given, and its launch refused.** A kernel
  built for some architectures only is emitted for the others with a body that traps; launched,
  it would end the context (Xid 43). Such a kernel of PTX that the layer retargets or of sm_75
  machine code that it translates is given at the lookup, as NVIDIA's driver gives it, but every
  launch of it and a kernel node of a graph with it return 218, and a line names the kernel and
  its image: a program that chooses such a kernel for the V100 stops at that error, without a GPU
  fault. Strata's CUDA 13 engine 0.1.66 does, with its default settings, at its first prompt of a
  few thousand tokens (4,021 here); its guide says which setting avoids it ([its
  guide](docs/guides/strata.md)). Not recognised: such a kernel in an input of the driver's
  linker (`cuLinkAddData`, `cuLinkAddFile`) or of nvJitLink, which traps as without the layer,
  and code that an image has for sm_70, which is loaded as it is. A kernel whose body is empty
  is not refused: launched, it does nothing and returns no error, as without the layer;
  `VF_LOG_LEVEL=2` counts them in each image adapted ([Kernels refused](#kernels-refused)).
- **The bf16 cuBLAS variants are opt-in.** `VF_CUBLAS_BF16_SKINNY=1` and `VF_CUBLAS_BF16_TC=1`
  are off by default. The tensor-core path's results are not cuBLAS 12.9's bits. It takes TN
  products with k from 768 to 16384 only, but for the classes of shapes that its rules of speed
  leave to 12.9, so that the products of a training step's backward pass go to 12.9 (a bf16
  fine-tuning step ran at 1.12 to 1.24 times its speed without it), and a model whose layers have
  fewer than 768 in-features gets it only on its wider layers. A product whose input has many
  columns that do not convert exactly to fp16 can run slower than with 12.9 (about half its speed
  at the slowest point measured). `VF_SAFE=1` turns the path off, and `VF_CUBLAS_BF16_VERIFY=1`
  checks the skinny path's calls against 12.9's own ([The bf16 cuBLAS
  variants](#the-bf16-cublas-variants)).
- **Under eager loading (`CUDA_MODULE_LOADING=EAGER`), PyTorch takes very long to start.**
  cuBLAS, cuBLASLt, cuSPARSE, cuSOLVER, cuFFT, cuRAND and cuDNN 9.10 start under it, as on
  NVIDIA's driver, but every image that a runtime registered is adapted and compiled at the start:
  with the layer's cache off, cuBLAS 12.9 started in 30 s (2 s without the layer), and the start
  of PyTorch 2.14 had not ended after ten minutes (2026-11-23); a later start, from the caches, was
  not measured ([Kernels refused](#kernels-refused), "Eager loading").
- **The CUDA 13 build of NCCL runs, but its first start takes about 15 minutes.** Where no GPU of
  the process can take part in multicast, as with V100s, the layer replaces each instruction of
  multicast memory (`multimem.*`) by a reported trap and gives the kernels that hold one, which it
  refused in 0.1.5: the wheel's NCCL then runs its collectives (the CPU's values on two V100s), but
  its first start compiles its adapted code for about 15 minutes in each process, and the later
  ones take about half a minute more than with NCCL's CUDA 12 build (`vf fetch --only libnccl`),
  which stays the one to use. A reported trap that a kernel reaches all the same stops it with
  `CUDA_ERROR_ASSERT` (710) and ends its context; `VF_REPORTED_TRAPS=0` refuses those kernels
  again ([Several GPUs and NCCL](#several-gpus-and-nccl)).
- **A lookup table asked for by name can be refused** (error 801), where NVIDIA's driver gives
  it, once a kernel that the layer compiled alone holds a copy of it ([Kernels
  refused](#kernels-refused), "A variable asked for by name can be refused too").
- **A cubin that the layer's nvJitLink links, loaded without the layer, keeps its kernels that
  trap**: the warning that nvJitLink writes when it links them is all there is. Likewise, the
  image that the driver's linker makes, saved and loaded in another process, gives its kernels
  that trap to the program ([Kernels refused](#kernels-refused)).
- **A kernel of one linker input that calls a function of another input that cannot run still
  traps** when it is launched: through the driver's linker and through the layer's nvJitLink,
  what traps is decided input by input ([Kernels refused](#kernels-refused)).
- **NVML reports the capability asked for** with `VF_CAPABILITY` to every caller, while a caller
  whose code runs from sm_75 machine code is told 7.5 by the driver: a program that reads both
  (vLLM does) gets two answers ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).

## Programs tested

| Program | Version | Status | Test |
|---|---|---|---|
| PyTorch | 2.14.0+cu130 and 2.14.1+cu130 (stock wheels: machine code for sm_75 and later, no PTX) | works, with the limits below | `tests/gpu/torch.sh`, `torch-models.sh`, `torch-bench.sh` |
| llama.cpp | b12418 (official build for CUDA 13.5) | works, with flash attention on and off, with quantized and NVFP4 weights | `tests/gpu/llamacpp.sh`, `llamacpp-ops.sh`, `llamacpp-nvfp4.sh` |
| ZML | upstream at `8898051c7b` with one fix of its own (`tests/gpu/zml/`) | works, told capability 7.5 and 8.0 | `tests/gpu/zml.sh` |
| vLLM | 0.34.0, installed with pip | works: the engine's greedy text is the CPU's token for token, sampling answers, the HTTP server gives the CPU's text | `tests/gpu/vllm.sh` |
| SGLang | 0.5.23, installed with pip | works, with its default attention (FlashInfer) and with `--attention-backend triton`: the engine's greedy text is the CPU's token for token | `tests/gpu/sglang.sh` |

- **vLLM** (Qwen2.5-0.5B, three prompts): every stage of the test; greedy generation at 84 to 99
  tokens a second over the runs of 2026-11-20 and 21 (83 to 97 with sampling); the HTTP server
  (the OpenAI API) with the model's repetition penalty gives the CPU's text. Its sampling kernel
  by itself against the same kernel run on the CPU by Triton's interpreter: 18 of 18 cases keep
  the same positions with the same values.
- **SGLang** (Qwen2.5-0.5B, three prompts; it makes the model float16 on a GPU that reports a
  capability under 8, whatever it is asked). With its default attention (FlashInfer): its
  engine's greedy text is the CPU's (float32) token for token, 150 of 150, at 45 to 52 tokens a
  second, one prompt at a time, and its HTTP server's too. With `--attention-backend triton`: the
  same from the engine at 55 to 59 tokens a second; the server leaves the CPU's text at one token
  of 150, where the CPU's own lead is 0.021 in logit (a near tie: `tests/gpu/greedy_compare.py`).
  Its log-probabilities differ from the CPU's by 1e-2 to 7e-2, which is what float16 does. Five
  of FlashInfer's eighteen prefill kernels are refused (below): another model or batch may ask
  for one, and the error then names it.
- **The engines were run with one small model each** and a few prompts. A program that works
  here with one model may launch, with another, a kernel that no test ran.

## Kernels refused

A kernel that the layer cannot run is refused when the program asks for it: the lookup
(`cuModuleGetFunction`, `cuLibraryGetKernel`) returns `CUDA_ERROR_INVALID_PTX` (218) and leaves
NULL in the handle, and a line on standard error names the kernel and says why. The other
kernels of its module load and run. The same holds for the image that the driver's linker makes
(`cuLinkComplete`) of PTX, cubins and fat binaries given to it in memory (`cuLinkAddData`) or in
files (`cuLinkAddFile`), with two exceptions: a kernel that calls a function of another input of
the link that cannot run is not refused, and traps when it is launched; and an input of another
type (a host object, or a library of them) is not read: the driver gets it as it is.

**Eager loading (`CUDA_MODULE_LOADING=EAGER`).** The CUDA 13 runtime asks for each kernel by
itself, and goes on without one that is refused, under eager loading as under lazy loading. The
CUDA 12 runtime, which the CUDA 12.9 libraries that the layer forwards to carry inside them,
takes the kernels of each of its fat binaries from the module of its library
(`cuLibraryGetModule`, then `cuModuleGetFunction` for each); under eager loading it does so for
every fat binary at its start, and gives up altogether at the first lookup that fails. NVIDIA's
driver gives no module for an image without code for the GPU (209,
`CUDA_ERROR_NO_BINARY_FOR_GPU`), and the runtime then does without that image. Under eager
loading the layer gives no module either (its kernels stay given by `cuLibraryGetKernel`) for an
image with a kernel that it refuses, and for an image that it made from one without code for
the V100 when the caller has code for the V100 of its own, as the CUDA 12.9 libraries do, which
choose that code here: the module would have every kernel of the image compiled at the start,
for nothing (cuBLAS 12.9 and cuBLASLt have hundreds of such images, kernels for sm_80 and later;
one took more than nine minutes to compile). So cuBLAS, cuBLASLt, cuSPARSE, cuSOLVER, cuFFT,
cuRAND and cuDNN 9.10 (its CUDA 12 build) start under eager loading as on NVIDIA's driver (until
2026-11-23 the handles of the first four could not be created, and cuDNN 9.10's first operation
failed: one of cuBLAS 12.9's images has ten grouped GEMM kernels for sm_80 that the layer does
not translate). A program of the CUDA 12 runtime whose own fat binary has a refused kernel
starts without the kernels of that fat binary under eager loading; under lazy loading it is
given the others.

What eager loading costs through the layer: every image of machine code that a runtime
registered is translated, and every kernel of retargeted PTX that it asks for is compiled by the
driver, at its start, and every refused kernel is named then on standard error. With the layer's
cache off (`VF_PROXY_CACHE=0`), cuBLAS 12.9 started in 30 s (2 s without the layer); the start of
PyTorch 2.14 (3,358 images of PyTorch, cuDNN, cuBLAS and the others adapted, 1,411 kernels
named) had not ended after ten minutes, nor that of cuDNN 9.28 (its CUDA 13 build) after two, by
which it had compiled 1,441 kernels one by one (2026-11-23). A later start takes what it can from
the layer's cache and the driver's (`CUDA_CACHE_PATH`); that was not measured.

A program can hold a handle to a refused kernel without having asked for it: cuFFT enumerates
the functions of each module it loads through an entry of a private table of the driver
(`cuGetExportTable`), not by their names, and an enumeration made before the driver's compiler
refused something of a deferred library still holds what it gave. The function is in the module
that the driver loaded, made to trap. Such a handle is refused where it is used: a launch
(`cuLaunchKernel`, `cuLaunchKernelEx` and their forms for the per-thread default stream,
`cuLaunchCooperativeKernel` and `cuLaunchCooperativeKernelMultiDevice`, the old `cuLaunch`,
`cuLaunchGrid`, `cuLaunchGridAsync`), a kernel node of a graph made or changed with it, a
question about it or a setting of it (`cuFuncGetAttribute`, `cuFuncGetParamInfo`, the occupancy
functions and the like) and `cuKernelGetFunction` return 218, the driver is not called, and a
line names the kernel and the reason once. What still traps: a kernel launched by another kernel
(dynamic parallelism), a function reached from a kernel by an indirect call or through a call
into another linker input (above), a launch through a private interface of the driver (cuFFT
launches through `cuLaunchKernel`; no other is known).

**A kernel without code is given, and its launch refused.** A kernel built for some
architectures only is emitted for the others too, with a body that is a trap and nothing else.
Launched, it ends the context on any GPU: with NVIDIA's driver and the sm_70 code of one, the
launch returned 0, the next synchronisation 719 (`CUDA_ERROR_LAUNCH_FAILED`), and the GPU wrote an
Xid 43 line (2026-11-24). Through the layer, such a kernel of PTX that it retargets or of sm_75
machine code that it translates is given when the program asks for it and answered about, as
NVIDIA's driver does, but a launch of any kind and a kernel node of a graph return 218 without
calling the driver, and a line names the kernel and its image; the context goes on. Code that the
image already has for sm_70 is loaded as it is and not read: a stub there traps as without the
layer. Nor is what was found carried through a link: a kernel without code in an input of
`cuLinkAddData` or `cuLinkAddFile`, or of nvJitLink, is launched from the image that the link
makes, and traps, as without the layer (at `VF_LOG_LEVEL=2` the line of such an input of the
driver's linker says that its launch is refused, which it is not). A kernel with an empty body is not refused: launched, it does nothing, as
without the layer, and nothing says so but `VF_LOG_LEVEL=2`, which counts both kinds in each
image adapted. A program that chooses, for the V100 it detects, kernels that its CUDA 13 build
carries only as such stubs now stops with that error and no GPU fault: Strata's CUDA 13 engine
0.1.66 ([its guide](docs/guides/strata.md)).

A kernel can be refused after it was given, though it runs: the layer learns a refusal after it
gave a kernel only when the driver's compiler refuses a function of a deferred library loaded
whole, of which the kernel was given compiled alone (the compiler accepted it then: that code holds
no trap). From then on the kernel is refused at every use, while a kernel node of a graph made with
it before goes on launching it (a graph is not looked at when it is instantiated or launched): the
answers are not the same before and after, but nothing that traps is launched.

**PyTorch 2.14.0.** Of the 23,693 functions for sm_75 in the 475 units of machine code of its
library, 23,664 are translated (99.9 %) and 29 are refused: 10 for a branch through a uniform
register, 6 for the instruction `I2IP`, 4 for a call whose return address is not given the
usual way, 4 for a conversion of a uniform register, 2 for `IMAD.HI.X`, 2 for a jump table, 1
for want of registers (`vf translate` of 0.1.3 on each unit, 2026-11-23). None of them is
reached by the tests below.

Of 925 operations compared with the CPU (`tests/gpu/torch.sh`): **924 give the same result**,
1 is skipped (the CPU does not do it), none is refused, none is wrong. Of 32 small whole models
(`tests/gpu/torch-models.sh`): 32 pass.

**PyTorch 2.14.1**, which is what `pip install torch` brought from PyTorch's index for CUDA 13
on 2026-11-21, installed in an environment made by `vf venv`: the same 925 operations and 32
models, with the same outcome (924 and 1; 32). Its machine code was not taken out and counted
as that of 2.14.0 was: the figures above and the timings below are those of 2.14.0.

**FlashInfer** (SGLang's default attention). Of its eighteen prefill kernels
(`BatchPrefillWithPagedKVCache` and `BatchPrefillWithRaggedKVCache`, nine each), thirteen are
translated and five are refused: three that name 255 registers, two for an instruction that
the translator did not have (`LEA.HI` with a uniform register). These are the counts of
2026-11-20; the rule for `LEA.HI` with a uniform register came on 2026-11-21 (in 0.1.2), and
these kernels were not counted again.

**Other libraries.** The machine code of 42 libraries of CUDA 13.5, cuDNN, NCCL and others was
translated and checked instruction by instruction, not run (`docs/validation.md`, "The corpora"):
of 3,108 units, 3,105 are translated in whole or in part; three are not read at all (a section
table that points outside the file; two sections of code of the same name). In these libraries
more functions are refused than in PyTorch (10,665 of 70,975, `vf translate` of 0.1.3,
2026-11-23); cuFFT's kernels for double
precision are among them (relocations that the translator cannot carry with their instructions).
The CUDA 13 libraries are not meant to be translated: the layer puts the CUDA 12.9 builds, which
have code for Volta, behind their names (`INSTALL.md`).

**Cubins for sm_80 and later.** VF can use the PTX that NVIDIA's assembler retains in
`.nv_debug_ptx_txt` with line information. It checks the architecture and the kernel names
before compiling that PTX for Volta. This covers standalone cubins from Triton and cubins in
fat binaries. Calls that the executable leaves to the driver (`__assertfail`, `vprintf`,
`malloc`, `free`) are accepted when the retained source declares them `.extern .func`. For sm_75, VF also tries retained PTX when its default machine-code translation
refuses every kernel in the image. The line information also tells which additions written
without a rounding modifier ptxas fused with a multiplication: one that it did not fuse
(`--fmad=false`, a debug build) is given the `.rn` form, so that the driver's compiler does not
fuse it either, and a source of which that cannot be told is not used.

Since 0.1.2, VF also translates a subset of stripped `sm_80` executables. VF checks each function's
instruction forms, branches and memory addressing. Global loads and stores, including 128-bit
accesses, require the CUDA system descriptor on every path. VF lowers the supported 16-byte
global-to-shared copy to a guarded Volta load and store and waits for the completed group.
SGLang uses this path for its top-eight expert sum and its wheel’s 32-bit expert counting and
alignment kernels. The router uses shared counter increments and GPU-scope 32-bit global atomic adds.
FlashInfer uses the same translator for 2D RMSNorm, fused residual RMSNorm and 3D query/key
normalization. For compiler-generated warp helpers, VF checks the call target, return register
and descriptor writes. Unsupported functions are refused individually. Other stripped targets above
`sm_75`, including `sm_86`, still require compatible PTX or a second supported image. Of 31,207
functions in 1,193 sm_80 cubins of PyTorch, cuBLAS, cuSOLVER and cuFFT, 900 (2.9 %) were
translated (2026-11-22). **This translation is not checked by the independent validator**, which
reads sm_75 code only: what checks it, and what does not, is in `docs/validation.md`, section 8.

**PTX.** A kernel that uses an instruction with no equivalent on Volta is refused: clusters of
blocks, Hopper's barriers, tensor maps (`docs/emulations.md`, "Other instructions").
For an indirect call within a single PTX module, an explicit `.calltargets` list identifies the
possible callees. With only a `.callprototype`, VF follows each device function that
could match the declared input and return lists. It distinguishes ordinary 32-bit and
64-bit scalars. Unfamiliar declarations, including array parameters, leave all targets
possible. A kernel is refused if one of those paths reaches a trapped function.
When linking separate device objects, an indirect call specified only by a prototype
can still be refused if any linked object contains a trapped device function.
The instructions of multicast memory (`multimem.*`) are not trapped so where no GPU of the process
can take part in multicast (every V100): each is replaced by a reported trap and its kernels are
given ("Several GPUs and NCCL", below; `docs/emulations.md`, "What is refused").
The image that the driver's linker makes (`cuLinkComplete`) is known by its bytes, as it is or
inside a fat binary, for the life of the process only: saved to a file and loaded in another
process, its kernels that trap are given to the program, and launched, end its context.
The cubin that the layer's nvJitLink (`libnvJitLink.so.13`) makes from inputs of which some
kernels cannot run (and others can: when none can, the link is refused) names them in a note
(`.note.vf.refused`). The layer reads it wherever the cubin is loaded, in any process: as a cubin,
as an input of the driver's linker, or stored uncompressed in a fat binary. Stored compressed in a
fat binary, it is not read (that would take decompressing the code of every library at every
load), except in a process whose links through the driver had kernels to refuse. A cubin loaded without the layer keeps its kernels that trap: the warning that nvJitLink
writes when it links them is then all there is.

**A variable asked for by name can be refused too.** In a library (`cuLibraryLoadData`, which
the CUDA runtime uses), a kernel compiled on its own (`VF_LAZY`, the default) that reads a
lookup table of its module (a variable with an initializer and no `.visible` or `.weak` name,
which no function of the module can change, as a `static const __device__` array or the grids
of the i-quants in llama.cpp and Strata) is given a copy of the table. NVIDIA's driver gives
such a table by its name in the PTX (`table`, or the name that nvcc mangles in a namespace), and
a kernel then reads what the program writes there. Through the layer, once a kernel holds a
copy, a lookup of the table by name returns `CUDA_ERROR_NOT_SUPPORTED` (801) instead of an
address, and a line on standard error says why, since a write through the address would not
reach the copy: `cuLibraryGetGlobal`, `cuModuleGetGlobal` on a module of the library, and with
them the CUDA runtime's `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol` and
`cudaGetSymbolAddress` (`cudaErrorNotSupported`, measured with CUDA 13.5). The answer depends on
the order: a table that the program asks for before any kernel that reads it is given, and its
kernels then come from the whole module, where the program's writes are. A name that the module
does not have is `CUDA_ERROR_NOT_FOUND` (500), as without the layer. `VF_LAZY=0` compiles every
module whole, with one copy of each variable, and gives every table. Neither Strata on two
V100s (issue #3) nor any case of `tests/gpu/run.sh` but those that test this asked for such a
table (2026-11-23).

**cuBLAS 13.5's half kernels for sm_75.** The CUDA 13.5 cuBLAS half GEMM runs on a V100: the
kernel cuBLAS selects for a 256×256 NN product
(`turing_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_stages_32x1_nn`) translates, and all 65,536
outputs are the CPU reference's (`tests/gpu/run.sh`, `runtime-13.5-ptx`). Six of the image's 100
kernels are refused: the `128x128` variants of
`turing_fp16_s1688gemm_fp16_128x128_ldg8_{f2f,relu_f2f}_{nn,tn,tt}`, whose level-3 translation
finds no 8 registers in a row for a matrix load even with those above the program's own. cuBLAS
did not ask for them for the tested shapes; another shape may. Two CUTLASS GELU kernels of the
same image (`cutlass_75_tensorop_f16_s1688gemm_gelu_f16_64x64_nn_align8` and its `dgelu`) are
refused for `MUFU` (`no translation for mufu__RRR_RR`); the GEMM does not use them.

## Kernels launched from a second translation

A kernel compiled for blocks of 1,024 threads may have 64 registers at most on a V100, and the
usual translation takes some kernels over that. When the driver refuses a launch for it, the
layer translates the module again within each kernel's own count and launches the kernel from
there (`VF_FIT=0` turns this off). In PyTorch's tests, `avg_pool2d` and `interpolate` in
float32 and float64, `unfold` in float32 and three of the whole models need it.

- The independent validator finds nothing in PyTorch's 475 units of machine code translated
  that way, as in their usual translation (`docs/validation.md`, "The corpora": until
  2026-11-20 it objected to 55 of them; five kinds of objection were rules of the validator
  that did not fit a translation within a kernel's own registers, and one was a fault of the
  translation, corrected). The other corpus of machine code, 42 libraries, is at 0 problems
  translated that way too.
- That second translation is slower code; its speed was not measured against a native build.
- It is not used for a kernel that can reach data of its module, nor for a kernel of a module
  from which the program asked for a variable or a texture or surface reference by name
  (`cuModuleGetGlobal`, `cuModuleGetGlobal_v2`, `cuModuleGetTexRef`, `cuModuleGetSurfRef`,
  `cuLibraryGetGlobal`, `cuLibraryGetManaged`, and what `cuGetProcAddress` gives for
  them): such a launch is refused to the program, as the driver refused it. A library and the
  module that the driver gives for it (`cuLibraryGetModule`) are one image: a variable asked
  for through either excludes the kernels taken through the other.
- It covers the four launch entries (`cuLaunchKernel`, `cuLaunchKernelEx` and their forms for
  the per-thread default stream), called or taken from `cuGetProcAddress`, and kernels taken
  with `cuModuleGetFunction`, `cuLibraryGetKernel`, `cuLibraryEnumerateKernels`, or
  `cuModuleGetFunction` on the module of a library.
- What the program sets on a kernel (`cuFuncSetAttribute`, `cuKernelSetAttribute`, the cache
  configuration) is set on its second translation too. A library's kernel and the function of
  the same name in the module that the driver gives for the library are one kernel to the
  driver (measured on a V100: a setting made with `cuKernelSetAttribute` holds for the
  function, unless the function has one of its own for that attribute, whatever the order of
  the calls): the function's second translation is given the library kernel's settings for
  the device of its context, then its own. The other way round is not done: a setting made
  with `cuFuncSetAttribute` on such a function is not carried to the second translation of
  the library's kernel, and a launch through the library's kernel that needs it (more dynamic
  shared memory than the default allows) is refused there as the first was.
- A launch whose second translation was being made when the program asked for data of the
  module by name gets the driver's refusal, like the launches after it: none starts from a
  second translation once the program has been given such data.
- It does not cover a launch recorded in a CUDA graph, a kernel launched through a function
  handle made from a library's kernel (`cuKernelGetFunction`), nor the old launch functions
  (`cuLaunch`, `cuLaunchGrid`).

## After a kernel faults

A kernel that faults (an illegal address, a misaligned access, a `trap`: errors 700, 716, 719 and
the others of their kind) ends its CUDA context, and the program stops at a later call than the
launch, often saying no more than "unspecified launch failure". The first time a call that the layer
routes returns such an error in a context, the layer says, at every log level, which kernels may
have faulted: those launched in that context since their last synchronisation that succeeded (of
the context, of their stream, the thread's own default stream included, or a `cuStreamQuery` that
found their stream idle), the last first, each with its grid, block, stream and thread, what the
layer made of its image (loaded as the program gave it, PTX retargeted, the PTX kept in a cubin
retargeted, machine code translated, with its tensor-core products computed in pairs on whole rows
unless `VF_PAIRS=0` or `VF_SAFE=1` turned them off; compiled alone or with its whole module; set up
from the layer's cache; launched from its second translation), the size and XXH64 of the image given to the driver, the program's image as the line
of its load named it (its XXH64 from `VF_LOG_LEVEL=2`), and the instructions it runs emulated. The
report at exit (`VF_SUMMARY`) says it again, with the same caveats. The lines do not say that a
kernel faulted: a copy or a memset, which the layer does not see, may be what failed; and errors
214 (an uncorrectable ECC error of the GPU's memory), 220 (of an NVLink) and 911 (of an external
device) are the hardware's, said as such, with the launches that ran then. A graph's launch is named
with the kernel nodes the layer saw added to it, and the line says what it does not follow.
`VF_FAULT_TRACE=1` waits for every launch and names the launch whose synchronisation met the fault,
at a cost (`INSTALL.md`).

What the layer cannot say:

- which of the candidates faulted, without `VF_FAULT_TRACE=1`; nor why. With it, the launch whose
  synchronisation met the fault is named, alone when no other launch of the context is uncovered;
  a copy, a memset or a host function enqueued before it on its stream, which the layer does not
  see, may have failed instead. A candidate is not a
  culprit: a program's own code faults through the layer as it does without it, and only a
  comparison with a native build or a newer GPU tells (`AGENTS.md`, step 8).
- a kernel that the layer does not see launched: by another kernel, through a private interface of
  the driver, by the driver from a CUDA graph. A graph's launch is named with the kernel nodes that
  the layer saw it made of (launches captured on a stream, nodes added with `cuGraphAddKernelNode`
  or `cuGraphAddNode`): a child graph, a clone, a graph changed after it was instantiated
  (`cuGraphExecUpdate`, a node given other parameters) or one instantiated by a device is not known
  as it is. Under `VF_FAULT_TRACE=1` the graph is named, not its node: run without graphs (an
  engine's eager mode) to name the kernel.
- launches older than the last 256 of their thread, when no synchronisation of the whole context
  bounds them; and the launches of a thread whose first launch found the records of 128 threads
  held (each thread holds one from its first launch to its end): it keeps none, even after records
  come free (the line says so of both, with the number of such threads).
- in which order the launches of two threads were made between the same two synchronisations:
  each thread's are in their order.
- a fault that only calls that the layer does not route met (a copy, a memset, an allocation,
  which go to NVIDIA's driver without passing through it) is said only when a later call that the
  layer routes returns the error (a launch, a synchronisation, a query, a load). When none does,
  the layer does not report it at all: it asks the driver nothing of its own to find one, neither
  before a context is destroyed or the primary context released, nor at exit. (A query of each
  context at exit hung a program that called `exit` from a signal handler during a call of the
  driver, and could ask a context that another thread was destroying.) The report at exit lists
  the faults that were said.
- with `VF_ADAPT=0` and `VF_FIT=0`, the launches that a program makes through the entry points the
  driver gives it (the CUDA runtime's) go straight to the driver, unless `VF_FAULT_TRACE=1` is
  set: none of them is known. A synchronisation of an event bounds nothing (which stream an event
  was recorded in is not known), and a fault that a wait for an event meets first, or a call made
  with no context current, is said of a context that the layer cannot tell: the launches of every
  context are named, each with its context, and the fault is said again, of its context, at the
  first call that meets it there. Faults of a context that the layer cannot tell are said once
  until a context is destroyed, and again only when a later one has candidates launched since the
  reports before read them, at most 4 times more in a process: a fault of another context met the
  same way with no launch since, or after those 4, is said only at a call that meets it in its own
  context. The context of a stream's synchronisation or query that meets a
  fault is that of the launches kept on the stream; on a stream with none of a context the layer
  knows, it cannot be told either, nor when the context of its last such launch was destroyed since
  (said "a stream of a context since destroyed": a stream that went with its context, whose handle
  the driver may give to a stream of another). A kernel of a library launched by its own handle runs in whatever
  context is current: it is a candidate in every context (in a process of several, the line says
  how many such). A kernel is named as the program asked for it; one of which it holds a handle that
  it did not take by name through the layer is named by the driver if it is a library's kernel,
  else by its address. A kernel whose module was unloaded (or went with its context) after its
  launch is named as it was, without what the layer made of its image. Whether its products are computed in pairs is said of machine code that the
  layer translated, not of PTX (whose pairs give the results of each product by itself).
- under `VF_FAULT_TRACE=1`, a launch made while a graph is being captured anywhere in the process
  is not waited for (a wait could end that capture), nor a cooperative launch on several devices.
  A program whose work waits for what it does later never ends under it: kernels that wait for one
  another, a host function (`cuLaunchHostFunc`) that waits for something the launching thread
  does after the launch, a `cuStreamWaitValue32`/`64` of a value written later (the layer does not
  tell them apart: the wait after the next launch on that stream lasts for ever).

## Speed

`docs/performance.md` has the method and the detail. In short:

**PyTorch.** 89 operations timed against the wheel of the same PyTorch that has code for sm_70
(2026-11-20 and 21, five runs): from 0.79 to 1.65 times its time, median 1.00, geometric mean
1.02 to 1.03; 69 to 72 of the 89 within 3 %. The slowest: attention in float16 (1.65: Turing's tensor-core
products are emulated), `std` along a dimension (1.3), `argmax` and `max` along a dimension,
`cumsum` (1.10 or 1.24, from one run to the next), a 3x3 convolution and a small CNN (1.2).
The convolutions are slower by the algorithm that the newer cuDNN picks, not by the layer.

**llama.cpp.** With a 0.5B model (2026-11-20): generation one token at a time at 410 to 470
tokens a second, 97.5 to 102 % of the native build's speed; a prompt of 2,048 tokens at 9,700
to 11,900 tokens a second, 40 to 53 % of the native build's speed (for a batch the official
build takes kernels with integer tensor-core products, which are emulated, where the native
build calls cuBLAS). Flash attention, 196 cases of llama.cpp's own backend test (2026-11-18): 1.60
times the native build's time over all of them; most of the worst cases compare two different
kernels, since llama.cpp picks another kernel on a Volta than on the Turing it is told.

**Told 7.5, llama.cpp takes its MMQ kernels for every quantized batch.** ggml computes the
products of quantized weights by prompt batches of any size with its MMQ kernels, which make the
activations 8-bit integers and use integer tensor-core products, emulated on Volta (eight `dp4a`
a product, `docs/emulations.md`). Built for sm_70, it keeps its MMQ kernels (on `dp4a`) for
batches of fewer than 64 rows and sends the others to cuBLAS on the V100's fp16 tensor cores,
the weights made fp16. On dense layers the layer's way is the slower: a 4096 x 512 x 14336
product took 3,112 µs through the layer against 1,084 µs in a native sm_70 build (cuBLAS) and
1,752 µs in a native build with its MMQ kernels forced (Q8_0 weights; Q4_K: 3,308, 1,284 and
2,128 µs; Q6_K: 3,693, 1,162 and 1,950 µs), and prompts of 512 tokens of a 0.5B model in Q8_0 ran
at 0.47 times the native build's speed. On mixtures of experts it is the faster, the sm_70 build
calling cuBLAS expert by expert: 128 experts of which 8 are used, a batch of 512, 2,674 µs
through the layer against 4,217 µs natively (Q4_K; Q8_0: 1,839 against 3,950 µs) (2026-11-24, one
V100). An outside report on Ollama 0.40.4 says the same of whole models (`docs/reports/`): prompts
at 1.3 to 1.9 times the native runner's speed on two mixture-of-experts models, 0.40 to 0.67
times on two dense ones, generation at the same speed. For a dense model, a native sm_70 build
of llama.cpp is the faster.

**llama.cpp with NVFP4 weights** (Qwen3.8-27B, 2026-11-20): generation at 98.5 % of the native
build's speed; prompts at 299 tokens a second, 1.33 times faster than the native build as it
is (224) and 1.95 times slower than the native build with its quantized kernels forced (582).

**vLLM, SGLang, ZML**: no native build exists to compare with. vLLM 83 to 99 tokens a second
and SGLang 45 to 59 (Qwen2.5-0.5B); ZML 27 to 29 (Qwen3.5-9B). Runs of 2026-11-20 and 21.

**vLLM told 8.0 to keep a bfloat16 model.** `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1` is for a
model that refuses float16 (below 8.0 vLLM makes a bfloat16 model float16). For an ordinary dense
model it is the slow setting: Qwen2.5-0.5B-Instruct (stored in bfloat16) on stock vLLM 0.34.0,
compilation and CUDA graphs on, 200 tokens a request, one V100, ran at 359 tokens a second at one
request and 7,445 at 32 in total at the default capability, and at 81.5 and 742 told 8.0, 4.4 and
10 times slower; the text began the same in every run (2026-11-24; one small model). Told 8.0, its
bfloat16 products run in FlashAttention-2 and in cuBLAS's bfloat16 kernels without tensor cores,
which a V100 has for float16 only. `VF_TRITON_TC=1` acts at the default capability only (there: 356
to 365 and 7,901, 6 % more at 32), and the bfloat16 cuBLAS variants gave nothing told 8.0 (81.6
and 745: vLLM replays its forward from CUDA graphs). `docs/performance.md`, "vLLM", has every run.

**Triton's matrix products** (`tl.dot`). Told 7.5, which is what Triton is told by default in a
program with PyTorch, Triton 3.8 uses no tensor-core instruction: it multiplies with float32
`fma` (its own choice below 8.0; it says so with `MLIR_ENABLE_DIAGNOSTICS=remarks`). Its results
are right and slow: a float16 GEMM at 0.23 to 3.2 TFLOPS, vLLM's mixture-of-experts kernel at
0.1 to 0.15, its attention at 2.6 (prefill) and 0.54 (decode); the same kernels told 8.0 (with
`VF_CAPABILITY_ALL=1`, and its hazards: [The capability told with
`VF_CAPABILITY`](#the-capability-told-with-vf_capability)) at 26 to 33, 4.4 to 12.9, 13.6 and
1.8 (2026-11-23, one session). The layer's translation adds 11 instructions to the 5,952 of
Triton's GEMM kernel: the time is in that code (`docs/performance.md`, "Triton's matrix products").
With `VF_TRITON_TC=1` (off by default; [Triton's matrix products](#tritons-matrix-products)),
everyone still told 7.5, the same float16 GEMMs run at 17 to 38 TFLOPS, those of 16 rows at 4.6
to 5.2 against 0.35 to 1.5; the bfloat16 GEMMs at 8.6 to 11.8 (16 rows: 1.9 to 2.8) against 0.22
to 3.3 (0.34 to 1.5); vLLM's mixture of experts at 0.42 to 14.9 TFLOPS in float16 (13 to 98 times
faster) and 0.23 to 6.4 in bfloat16 (7 to 43 times), its attention at 13.6 and 1.83 in float16
and 5.2 and 1.07 in bfloat16 (2026-11-23, one session, through the launcher).

**Blackwell's block-scaled products** (NVFP4, MXFP4): a matrix-vector product written with them
runs at 2.7 times the time of a kernel written by hand for Volta (2026-11-18).

**The first run** of a program translates its machine code, rewrites its PTX and lets the
driver compile it: seconds to a minute for the programs above, kept in a cache for the next
runs (`INSTALL.md`). The translator's time grows faster than the size of a function: one unit
of machine code of vLLM's compiled extension for mixtures of experts (5 MB, 224,000
instructions in 36 functions; not among what the tests here make the layer translate) takes
it more than seven minutes (2026-11-20).

## Emulated instructions

A V100 lacks these; the layer computes them otherwise (`docs/emulations.md`). A kernel that
runs emulated instructions is announced on standard error.

| Instructions | How | Result | Speed |
|---|---|---|---|
| Turing's tensor-core products on FP16 | Volta's products, operands gathered from other threads | the host model's and an RTX 5090's where the sums are exact; on arbitrary values the last bits differ from an RTX 5090's (47.6 % of random results identical, 41.2 % one unit of the last place away), as a program built for Volta differs | slower: attention in float16 at 1.64 times native |
| Products on bf16, tf32, f64 | each thread computes with `fma`: no tensor core of a V100 takes them | sums added in another order than the hardware's: for bf16 about one random result in five differs from an RTX 5090's, by 1 or 2 units of the last place | slow |
| Integer tensor-core products | `dp4a` on bytes gathered from other threads | exact | 1.95 times a kernel written with `dp4a` (above) |
| FP8 products | exact conversion to halves, then Volta's products | as FP16 products | |
| Block-scaled products (NVFP4, MXFP4, MXF8F6F4) | each block through Volta's products, the scales applied after | NVFP4: of 406,912 results compared with an RTX 5090's, 405,861 identical, 1,051 one unit of the last place away, none further. The other forms agree less (`docs/emulations.md`, "Block-scaled products") | 2.7 times a kernel written by hand |
| Conversions of bf16, tf32, FP8, E2M1, UE8M0 | bit manipulation | every value of the small formats gives what an RTX 5090 gives, byte for byte | |
| `F2FP.PACK_AB` (two f32 to two f16 in one register) | two `F2F.F16.F32` and a `PRMT` | each half is the single rounding to nearest even that the hardware's `F2F.F16.F32` gives; the CUDA 13.5 half GEMM's 65,536 outputs are the CPU's, on a V100 and on an RTX 3060's own cuBLAS | one conversion and a pack: three instructions for one |
| bf16 arithmetic and atomic additions | `fma`; a compare-and-swap loop | | |
| `cp.async`, 256-bit loads | plain loads | | |

The layer takes it that a program keeps the rows that `ldmatrix` reads at multiples of 16
bytes, as the PTX manual requires. A program that does not stops with a misaligned-address
fault (error 716); `VF_ALIGNED_ROWS=0` makes the layer test the addresses, with slower
emulated kernels.

## Triton's matrix products

Told 7.5, Triton computes `tl.dot` with float32 `fma` ([Speed](#speed)). `VF_TRITON_TC=1` (the
launcher; `INSTALL.md`) makes it compile a kernel whose products all multiply float16, bfloat16
or int8 operands for sm_80, as on an Ampere GPU, while PyTorch, NVML and every other reader of the
capability are still told 7.5; the layer then runs the PTX that each such cubin keeps. It is off
by default (one line of the launcher, `triton_tc_default`, decides that), and `VF_SAFE=1` turns it
off whatever it is set to. It acts at the default capability only: with `VF_CAPABILITY` set to
another value than 7.5 it says so in one line and changes nothing, so a bfloat16 model that a
program must be told 8.0 for (vLLM's) gains nothing from it. On vLLM's engine at the default
capability, a bfloat16 model made float16, it gave 6 % more at 32 requests ([Speed](#speed)).
What changes for a program, by type:

- **float16**: the products run on Volta's tensor cores (`docs/emulations.md`, section 2).
  Their sums are truncated, four terms at a time, where Triton's `fma` rounds each: on random
  data of both signs the largest error of a float32 result against float64 was 6.8e-6 of the
  largest value against 2.3e-6 (K = 4,096; 1.0e-5 against 4.8e-6 at K = 16,384); on long sums of
  terms of one sign the results come out low, 8.5e-5 against 3.8e-6 at K = 4,096 and 3.3e-4
  against 7.2e-6 at K = 16,384. Written as float16, the same results differ from float64 by
  3.14e-4 against 3.13e-4 (the rounding of float16). This is what Volta's tensor cores give to
  every program (PyTorch's own float16 products, through cuBLAS, are computed so). Up to 170
  times faster in the GEMMs ([Speed](#speed)), 1.6 to 61 times in the kernels of vLLM and SGLang
  timed with CUDA graphs (below).
- **bfloat16**: the layer's scalar products (float32 `fma` in the order of k), which gave the
  same bits as Triton's `fma` path in the GEMMs and in vLLM's mixture of experts; in vLLM's
  attention, other bits in 56 of 146 shapes of a sweep (the others the same), with the same
  largest error against the reference (4.7e-3 of the largest value, on as off); 1.4 to 50 times
  faster.
- **int8**: the layer's integer products, exact. Triton 3.3 to 3.9 cannot compile an int8
  `tl.dot` for 7.5 at all (its compilation fails), nor Triton 3.3 one of float16 or bfloat16;
  with `VF_TRITON_TC=1` they run.
- **float32, float64, FP8, scaled products**, and a kernel with any of these beside products of
  the three types: unchanged (compiled for 7.5, the same cubin). A float32 product compiled for
  8.0 would be tf32, less exact than today: it is not.
- A kernel whose pipelined sm_80 code needs more shared memory than a V100 has (96 KiB) is
  compiled with one stage fewer, down to one: Triton's configurations tuned for Ampere (164 KiB)
  run where they would stop with `OutOfResources`. The last bits of a float16 kernel's results can
  depend on its number of stages: with B stored N x K, a float16 GEMM compiled for sm_80 gave
  other last bits with one stage than with two to four (random data; the same bits where the sums
  are exact; B stored K x N and bfloat16: the same bits for any number), where for 7.5 every
  number gives the same bits. A kernel compiled with fewer stages for its shared memory can so
  give other last bits than the configuration asked would give on an Ampere GPU. Of the 92 cases
  of vLLM's and SGLang's kernels compiled for sm_80 below, one needed it. Three MLA decode
  attentions of theirs, which ask 102,400 bytes for 7.5 and stop with `OutOfResources` without the
  switch, run with it.
- **Skinny products left to 7.5.** A bfloat16 product of a tile of at most 32 rows and 16 x 128 x
  64 elements whose B is stored by rows (K x N, as `a @ b` with `b` contiguous), and a float16
  product with float16 sums of a tile of 16 rows and 128 columns or more with B stored so, ran
  slower for sm_80 than Triton's `fma` code for 1 or 16 rows: down to 0.53 times its speed in a
  sweep of 432 such products (Volta's products read such a B with transposing loads). A kernel
  all of whose products are of these kinds is compiled for 7.5 (`Triton kernel NAME: compiled
  for sm_75: ... (measured slower for sm_80 on a V100)`). It gives up what these products gain at
  256 rows (up to 2.6 times), which a kernel compiled once for any number of rows cannot know.
  The products of the sweep that stay at sm_80 ran 0.96 to 32 times as fast as Triton's; B
  stored by columns (N x K), as PyTorch keeps its weights, is never left so.
- A kernel that the layer's tool would refuse, or whose sm_80 compilation fails in any way, is
  compiled for 7.5, as without the switch; the line `Triton kernel NAME: compiled for sm_75: WHY`
  (`VF_LOG_LEVEL=2`) says so. The sm_80 compilation runs in a **helper**, a child process of the
  program's that compiles one kernel at a time, so that an error of LLVM or MLIR that ends it does
  not take the program down: a helper ended by SIGSEGV, SIGILL or SIGFPE, a compilation that does
  not end in `VF_TRITON_TC_TIMEOUT` seconds, and an exception of Triton's compiler leave that kernel
  at 7.5, and Triton's cache keeps it so (`ptxas` that crashes too). A helper ended by SIGABRT or
  SIGBUS, which a lack of resources gives too (LLVM aborts when an allocation fails, at `RLIMIT_AS`
  or with memory overcommit off; a library truncated under the helper by a reinstall gives SIGBUS),
  has the kernel compiled again at once in a new helper, once (said at `VF_LOG_LEVEL=2`): a second
  such end is the kernel's, so that a kernel whose compilation aborts every time is compiled twice,
  once per cache. A helper ended from outside (SIGKILL as by the out-of-memory killer, SIGTERM,
  SIGINT or SIGHUP) or stopped from outside past the time-out (SIGSTOP, a debugger, a frozen
  control group), `ptxas` ended so, or failing on a full disk or quota, no helper of the process
  free in `VF_TRITON_TC_WAIT` seconds, the tool that did not run: the kernel is compiled for 7.5
  this time and nothing is kept in Triton's cache. What Triton's compiler wrote then is written at
  `VF_LOG_LEVEL=3` (the last 20,000 bytes of it, for a helper that crashed).
- **What a helper is, and how long it lives.** A process starts one at its first kernel that may be
  compiled for sm_80, and more, up to `VF_TRITON_TC_JOBS`, when its threads compile at once (an
  Inductor process that precompiles its choices in threads); each is about 105 MB resident, 150 MB
  while it compiles, and counts against the memory limit of the program's control group. A helper
  ends with its process (`PR_SET_PDEATHSIG`, and the end of its socket), after `VF_TRITON_TC_IDLE`
  seconds (10) with nothing to do, and after half a second with nothing to do while a process of
  the user waits for a place: a server after its warm-up keeps none, and a kernel compiled later,
  at a request that needs a new kernel, starts one again (about 0.6 s). Before it ends it kills
  what it still runs and reaps its own children. A thread of the program waits for it (the wait of
  its pid alone: the layer takes no other child of the program), and then reaps what of its group a
  program that is a child subreaper inherits (Python as PID 1 of a container is given the `ptxas`
  of a helper killed while it ran; not on PyPy, which has no `waitid`). A child made by `fork()`
  does not use its parent's helpers (it closes its copies of their sockets and starts its own when
  it needs one); `spawn` and `forkserver` workers are new processes.
- **A helper is a child of the program.** A program that waits for any child (`os.wait()`,
  `waitpid(-1)`, a `SIGCHLD` handler that reaps every child, a supervisor that counts its children)
  can be given a helper's pid and status in place of one of its own; Python's `subprocess`,
  `multiprocessing`, `concurrent.futures` and asyncio's child watchers wait by pid, and are not
  affected. A program that ignores `SIGCHLD` or reaps every child itself takes the helper's status
  from the layer, which then compiles that kernel for 7.5 and keeps nothing (a crash is compiled
  twice, never kept wrongly). A program that replaces itself with `exec()` while a helper lives
  does not pass the helper's socket on: the helper ends at once (one that compiles, when that
  compilation ends) and stays a zombie of the new program, which does not know it, until that
  program waits for any child or ends.
- **The places.** The user's helpers on one host are `VF_TRITON_TC_JOBS` at most, all programs that
  share Triton's cache directory together, each holding one of that many lock files (`flock`)
  from its start to its end; PyTorch
  Inductor's compile workers included, `torch.compile` with max-autotune and 32 workers then holds
  at most that many helpers (`docs/performance.md`). The lock files are in `volta-forever-EUID` of
  Triton's cache directory as Triton finds it when the process compiles its first kernel that may
  be compiled for sm_80: with PyTorch's Inductor and `TRITON_CACHE_DIR` unset, Inductor's own
  cache's `triton/0`, which Inductor sets before it compiles anything. Processes whose Triton
  caches differ count their places apart: vLLM, unless `TRITON_CACHE_DIR` is set, gives each model
  and each rank of an engine a Triton cache of its own, so that an engine of four ranks may have
  four times that many helpers. The cap is a best effort: where a lock file cannot be
  opened or locked for another reason than another holder (a file system whose locks do not work,
  as Lustre without its `flock` option or NFSv3 without a lock manager; a directory that cannot be
  written, that is not one, or a link), the process does not count places (said once, at
  `VF_LOG_LEVEL=2`), and its helpers start without one, `VF_TRITON_TC_JOBS` of its own at most;
  no kernel waits. A process waits for a place `VF_TRITON_TC_WAIT` seconds (60) once in its life
  (meanwhile an idle helper of another process gives its place up after half a second): when none
  came, as when places are held by helpers that do not give them up (stopped by a debugger or a
  freezer, also after their program died, or stuck in a compilation after their program died until
  their own alarm, a minute after `VF_TRITON_TC_TIMEOUT`), it says so once, waits for none again,
  and compiles with one helper of its own without a place when none is free, that kernel included
  (and the kernels of its other threads that waited with it, which then wait for that helper as
  for a busy helper of the process).
- Every wait has a bound: a request is taken in 30 s, an answer comes in `VF_TRITON_TC_TIMEOUT`
  seconds (the time during which the program itself was stopped not counted), the first helper's
  import of Triton is waited for `VF_TRITON_TC_START` seconds once (then each kernel goes to 7.5,
  not kept, until it is ready, said once; an import that does not end ends the helper after
  `VF_TRITON_TC_TIMEOUT` seconds), a place `VF_TRITON_TC_WAIT` seconds once in a process (above),
  and a kernel whose process has all its helpers busy `VF_TRITON_TC_WAIT` seconds (then it goes to
  7.5, not kept). A program with no descriptor left, or that can start no process (`RLIMIT_NPROC`,
  a pids control group), gets that kernel at 7.5, not kept, said once in the process, and its next
  kernels try again; a program that closes its descriptors (the helper's socket among them) gets a
  new helper at its next kernel. The layer closes no descriptor that is no longer its own, also in
  a child made by `fork()`: each is kept with what it is (device and inode) and closed only while
  it still is that. It waits with `poll`, for any number of descriptors.
- Triton's cache, and PyTorch Inductor's caches of compiled graphs and autotuning, keep what was
  made with the switch apart from what was made without it.

What was run with it (2026-11-24, 0.1.4 with the work after it, Triton 3.8.0, PyTorch 2.14, at the
default capability, one V100): the engines of `tests/gpu/vllm.sh` (stages b to d: the engine greedy
and sampling, the HTTP server) and `tests/gpu/sglang.sh` (Triton's attention, stages b to d) through
the launcher, the CPU's text as without it, their attention kernels compiled for sm_80 (5 of 39 and
2 of 16 kernels in Triton's cache); with the 0.5B model of the tests, decoding is not limited by
those kernels: stage c at 85.5 tokens a second (vLLM; 86 to 92 off and on in the cold starts of
`docs/performance.md`) and 6.8 (SGLang, one prompt at a time). `tests/gpu/triton.sh`: 164 PASS, 1
PASS-PTX (a gather of 16 warps in float32, as without it) and 15 without a reference of 180 (13
compiled for sm_80); `tests/gpu/triton-tc.sh`: 51 PASS. The kernels of vLLM and SGLang of
`tests/gpu/triton_vendor.py` (full level) against Triton's interpreter, both routes: vLLM 148 PASS,
2 PASS-PTX and one FP8 kernel that Triton refuses for 7.5, of 151 (65 compiled for sm_80), where
without it (2026-11-23, before 0.1.4) 135 PASS, 10 PASS-PTX and 6 that Triton cannot run for 7.5
(the same FP8 one, two MLA decode attentions that ask 102,400 bytes of shared memory, more than the
V100 has, and three int8 `scaled_mm`, whose compilation fails); SGLang 87 PASS and 3 PASS-PTX of 90
(27 compiled for sm_80), where without it 83 PASS, 5 PASS-PTX, one MLA decode attention that Triton
cannot run for 7.5 (102,400 bytes of shared memory) and the MoE router of 128 experts, which 0.1.4
runs (it passes with the switch and without it). Those runs were made when the compilation ran in a
compile service shared by the user's processes; with the helpers of a process, as they are now (their
temporary files in the temporary directory, the places a best effort; 2026-11-24, 0.1.5 with the
work after it, the same Triton and PyTorch): `tests/gpu/triton-tc.sh` 51 PASS,
`tests/gpu/triton.sh` 164 PASS, 1 PASS-PTX and 15 without a reference of 180 (13 compiled for
sm_80), as before, `tests/gpu/vllm.sh` stages b to d PASS (5 of 39 kernels in Triton's cache
compiled for sm_80), a part of `tests/gpu/torch.sh` (`lib.`, `attention`, `softmax`: 56 of 56),
`tests/gpu/run.sh` without the switch 202 PASS; the cubins of eleven configurations byte for byte
those of the same sm_80 compilation in the program's process; `torch.compile` with max-autotune and
`TRITON_CACHE_DIR` unset: the places in Inductor's own `triton/0`, the kernels compiled for sm_80.

The same kernels timed with CUDA graphs (the launches of a case captured and replayed: the best
of 7 replays of 20 launches, each after products that raise the GPU's clocks, in four processes
off, on, on, off; 2026-11-24, one V100): every case with a kernel compiled for sm_80 ran faster
but one. vLLM's float16 cases 1.6 to 61 times (39, median 4.5), bfloat16 1.2 to 16 (12),
products of quantized weights 0.94 to 3.9 (11; an AWQ product of 5 rows at 0.94, 35.5 us against
33.7), int8 only with it (3); SGLang's float16 1.9 to 45 times (17), bfloat16 1.8 to 15 (7),
quantized weights 1.1 to 1.2 (3). The kernels that the switch does not change ran at 0.97 to 1.14
times. Two bfloat16 kernels of vLLM are left to 7.5 by the rule of skinny products (its paged
attention, a chunk state of Mamba): their cases run 2.7 and 6.6 times faster than without the
switch, 1 and 2 % faster than with those kernels compiled for sm_80. Timed eagerly, as before,
`kernel_unified_attention` for 30 query tokens of 3 sequences with heads of 96 and a window of 127
seemed as fast or slower (0.71 to 1.01): its 28 to 56 us hide behind the time Python takes to
launch it; with CUDA graphs it runs 2.0 times faster, and every case of that kernel 1.2 to 4.5
times. Compiling those kernels with one pipeline stage instead of the stages asked made some
faster and others slower (timed eagerly): it is not done. `torch.compile` with Inductor's Triton
template alone: 28 template kernels compiled for sm_80, the product 287 us against 702.

What it does not cover:

- It works through the launcher (or `PYTHONPATH` set by hand: `INSTALL.md`, "Triton's products,
  by hand"); the layer's libraries alone do not do it. A Python started with `-I`, `-E` or `-S`
  does not read it, nor one whose program sets `PYTHONPATH` without the launcher's directory.
- **A helper runs the program's interpreter**, with `-I -S`: no `PYTHON*` setting, no user site, no
  `sitecustomize`, no `.pth` file, nothing of the program's `sys.path` but the directory that holds
  its Triton, after the standard library (a Triton of an editable installation whose import hook
  finds `triton` elsewhere is found there only when the package and its `_C` are below the
  directory that holds `triton/`; otherwise the helper's import fails, and the process compiles as
  without the switch, said once); `-B` when the program writes no bytecode, and the program's
  `-X pycache_prefix` when it writes its bytecode under a prefix (`PYTHONPYCACHEPREFIX`): a helper
  writes none into the program's Triton where the program writes none. The process checks that
  the Triton it imported is the program's (its directory and version, and the device, inode, size
  and time of its library, `libtriton`, and of its NVIDIA backend's module, as they were when the
  program imported Triton: a Triton reinstalled under a running program with the same version, or
  whose library or NVIDIA backend's module was edited in place, is not taken, said once) and uses
  it only then. Its other files are not compared: another of its modules edited in place, as an
  editable installation allows, is what later helpers import (the libdevice and `ptxas` that it
  finds are those of the program's own later compilations too); and the two files are recorded a
  few milliseconds after the program loaded them, as its import ends, so that one replaced in
  that moment would be taken for the program's. Its
  environment is not the program's: the variables that a compilation reads (Triton's `TRITON_*`
  but those of its caches and dumps, `MLIR_*`, `LLVM_*`, `NVPTX_*`, `PTXAS_*`, `DISABLE_*`, the
  layer's `VF_*`) and `PATH`, `HOME`, `USER`, the locale, `LD_LIBRARY_PATH`, and `TMPDIR` the
  program's temporary directory (below); no other (no
  secret of the program's, no `LD_PRELOAD` or `LD_AUDIT`). Its working directory is `/`, it has its
  own session and process group, and of the program's descriptors only its socket and its two
  locks; it is started only with the interpreter that runs the process (a frozen application, or
  one whose `sys.executable` is not that interpreter, does not use the switch, said once). A process
  that cannot use a helper at all (its Triton cannot be imported there, or its import does not
  end, or its files changed, or the helper's temporary directory cannot be made) says so once and then compiles as without the switch: its kernels' keys lose the mark, and
  Triton's cache keeps them as it does without the switch, but for the first eligible kernel,
  whose key had it (compiled for 7.5, not kept); what does not depend on a kernel (the
  interpreter) is decided before any key has the mark, and the places' directory turns nothing off
  (above).
- **What a helper compiles**: what its process sends it, a TTIR and Triton's options, each value
  checked: `ptx_options` (`PTXAS_OPTIONS`, or a kernel's) only of the options that Triton itself
  gives `ptxas` (`-v`, `-lineinfo`, `--fmad`, `--opt-level N`, `--regAllocOptLevel=N`, `-g`,
  `-suppress-debug-info`), the count of registers and the warnings, none that names a file;
  `extern_libs` only files of Triton's own directory or the libdevice Triton links by default; no
  IR given in place of a stage, no instrumentation; every other value a number or a short plain
  text; 64 MiB a request. A kernel whose options are not such is compiled for 7.5, said once.
- **Its temporary files, and whom it trusts.** A helper writes Triton's PTX, the cubin that
  `ptxas` writes and the tool's files in a directory of its own (mode 0700, `mkdtemp`) in the
  temporary directory where Triton's own compilation in the program writes the same files
  (Python's `tempfile.gettempdir()`: `TMPDIR`, else `TEMP` or `TMP`, else `/tmp`), and reads the
  cubin through that directory's descriptor, not following a link and only from a regular file:
  the exposure of these files is Triton's own. It removes that directory when it ends, the
  program's waiter after a crash; that of a helper killed with its program (both by SIGKILL) is
  left to the system's cleaning of temporary files, as Triton's own files of such a run are.
  `volta-forever-EUID` (mode 0700) of Triton's cache directory holds the lock files of the places
  alone: opened once without following a link, then used by its descriptor (a link or another
  directory put at its path afterwards is not followed); nothing else is made there, and nothing
  there is removed. Whoever may write Triton's cache directory, or rename what is in it or above
  it, can already give the program kernels of their choice through Triton's cache, without the
  layer; with the lock files they can keep places taken (a process then waits once) or have the
  places not counted, and nothing more. No other process can reach a helper (its socket has no
  path). Nothing more is asked of these directories than Triton asks of its own (not their owner:
  a network file system may give root's files to another uid).
- **What a process takes from its helper.** Only an answer to the request it sent (the digest of
  what it asked, given back) whose cubin is one for sm_80 (ELF, machine, SM, at most 64 MiB) that
  keeps its PTX, the PTX it keeps being the answer's (compared line by line without comments,
  `.loc` and `.file` lines and DWARF sections, as `ptxas` keeps it: a few ms), whose PTX is for
  sm_80 and whose metadata is that kernel's (its name; its number of stages; no field of the
  request changed but those that Triton's stages set). Anything else is refused, said once, and
  the process compiles as without the switch. Against a bug or a Triton that makes something else
  than the layer expects: no other process can answer in a helper's place.
- Only at the default capability (`VF_CAPABILITY` unset or 7.5), on a machine whose GPUs are all
  Voltas, and for Triton 3.3 to 3.9 (the versions tried). The hook replaces stages of Triton's
  internal compiler (`CUDABackend.add_stages` and its stage methods, `triton_key`,
  `FileCacheManager.put_group`), which it recognizes by their names and signatures: a Triton whose
  methods differ is told so once and left as it is. A fork of Triton under another name (vLLM's `tokenspeed_triton`) is
  not touched.
- A kernel that the layer's tool accepts and the driver's compiler then refuses would fail at its
  first launch; none did among the kernels run (2026-11-23).
- Cubins made with it are sm_80 machine code that keeps its PTX: an ahead-of-time artifact
  (AOTInductor, a saved Triton cache, a packaged kernel) built on a V100 with the switch is sm_80
  code, which runs on a V100 through the layer and on Ampere and later GPUs, not on a Turing GPU.
- The float16 sums of these kernels are Volta's tensor cores' truncated ones (above), wherever the
  program asked for float32 accumulation: no setting gives Triton's rounded `fma` sums back but the
  switch off.
- A Python started through the launcher with the switch on runs the layer's `sitecustomize` at
  its start, compiled each time in an installation of the archive (whose `python/` keeps no
  bytecode, so that nothing is written into it): about 3.2 ms more per start; a process that
  imports Triton compiles `volta_forever_triton.py` too, about 6.5 ms more, and one that compiles a
  kernel for sm_80 the client of its helpers, about 7 ms more, and starts a helper (about 0.6 s, of
  CPU time beside the program's; more from a cold disk) (`docs/performance.md`).
- The first run after the switch is turned on, or off, compiles again every Triton kernel
  (and Inductor's graphs) that the caches have only from the other way; they keep both. A first
  run with the switch costs more than one without it: a cold `torch.compile` with max-autotune took
  6 to 18 % longer in single jobs, about 13 % over the pooled runs (12 and 12, in six jobs of two
  sessions; the same with Triton's cache on a network disk and in memory), of which about 8.8 s is
  the driver compiling the PTX of each sm_80 kernel at its first launch (any compilation for sm_80
  pays it, and the driver's cache keeps what it compiled) and 4 to 5 s the starts of the helpers;
  later runs take the caches (`docs/performance.md`).
- The rule that leaves skinny products to 7.5 (`SLOWER`) was fitted to one sweep on one V100 SXM2:
  cells near its line (0.97 of off) measured on either side of it in another sweep (the method's
  noise), vLLM's AWQ product of 5 rows stays at 0.95, and other V100s (PCIe, other clocks) and the
  TITAN V were not measured.

## The capability told with `VF_CAPABILITY`

`VF_CAPABILITY=8.0` (or later) is for programs whose kernels are PTX, which the layer then takes
for that capability: llama.cpp, TensorFold, ZML's own library, and the engines whose guides set
it. A caller whose code the layer can only give from its sm_75 machine code (PyTorch, cuDNN,
cuFFT, cuSOLVER, and the CUDA runtime that such a library uses) is still told 7.5, with one line
at the default log level: told 8.0, it would choose kernels that its sm_75 code has as stubs,
which write nothing, and told 9.0, some that stop the GPU (`docs/architecture.md`, section 4).
What that does not cover:

- `VF_CAPABILITY_ALL=1` tells the value asked for all the same to a CUDA runtime and to the code
  that uses it (PyTorch's library), the program; a library that asks for itself without that
  runtime (cuDNN, cuFFT, cuSOLVER) is still told what its own code allows (told 8.0, cuDNN's convolutions gave zeros, cuFFT stopped the GPU
  and cuSOLVER's Cholesky factorization failed). A kernel that the program then chooses for that
  capability, and whose sm_75 code is a stub (it writes no memory, where its code for the
  capability told does: PyTorch's attention kernels for sm_80), is refused, and the operation
  fails. It is only for a program that does not launch those kernels, as SGLang with the
  settings of its guide, or vLLM for a model that refuses float16: for an ordinary dense model
  vLLM was 4.4 to 10 times faster at the default capability ([Speed](#speed)).
- **A kernel that only the code for the capability told has.** The layer gives an image that has
  machine code for sm_75 and for sm_80, and no PTX, its sm_75 code, translated, whatever the
  capability told: its translation of sm_80 machine code covers fewer instructions and is not
  independently validated (`docs/architecture.md`, "Fat binaries and what is chosen"). A library
  compiled for several architectures can have a kernel in its code for sm_80 only, or under
  another name in its code for sm_75 (a template whose `enable_if` takes another overload there).
  Told 8.0 (with `VF_CAPABILITY_ALL=1` for a library that uses a CUDA runtime), its host code asks
  for the kernel by its name in the sm_80 code, and the driver has no kernel of that name in the
  code given (500; the CUDA runtime says `named symbol not found`). Since 2026-11-25 the layer
  then gives that one kernel from the image's sm_80 code, translated, in a module or library of
  its own beside the program's, and says so once for each kernel: `kernel NAME is given from the
  machine code for sm_80 of its image (FILE: SIZE bytes, XXH64 DIGEST), translated, in a library
  of its own: ...`. Only at such a lookup: every kernel that the sm_75 code has is still given
  from there, and a program whose lookups all succeed is given the same images as before. vLLM
  0.34.0's vectorized bf16 `fused_add_rms_norm` is such a kernel (it is only in the sm_80 code of
  its `_C_stable_libtorch.abi3.so`): with a bfloat16 model, told 8.0 with `VF_CAPABILITY_ALL=1`
  and its compilation off (`--enforce-eager`, or `TORCH_COMPILE_DISABLE=1`), vLLM stopped at its
  first forward, and now runs (measured on a V100, 2026-11-25: the kernel alone on 81,920 bf16
  values of five shapes against the same computation in float32 on the CPU, rounded to bf16: its
  residual the CPU's bits for every one, and its norm the CPU's bits for all but 2, which are
  one bf16 unit away, as 4 of its float16 values are from the sm_75 code: the kernel adds the
  squares in another order and takes the reciprocal square root from the GPU's instruction, so
  that the float32 value before the last rounding differs in its last bits, and falls on the
  other side of a bf16 halfway point about once in 30,000; `tests/gpu/vllm.sh`, stage e. The
  engine then generates text, which leaves the CPU's float32 text where the CPU led by a few
  tenths in logit: compare before relying on it). What this does not cover, where the lookup still fails
  with 500 and a line says why (`kernel NAME is not in the code that the layer gives for its
  image ...: it is only in the image's machine code for sm_80 ... The kernel alone is not given
  from the code for sm_80 either: REASON`):
  - a kernel that can reach data of its image: a `__device__` variable or a table (addressed
    through the bank of the image's addresses, or by a relocation), a `__constant__` variable
    (read from a constant bank of the module with no relocation), a function of other code.
    The module of the layer's own has another copy of that data than the one the program reads
    and writes through the module it holds, so such a kernel is not given from it: every
    constant bank that the kernel reads must be its own alone (its parameters and the launch's
    sizes, stack pointer and memory descriptor in bank 0, each at a fixed place, and no other
    word of bank 0: the driver puts the address of the image's `__constant__` data there too;
    or a bank that the image has for that kernel only, with no name in it), a kernel that uses
    a texture or a surface is not given, and what the layer cannot read of a
    kernel counts as data (until 2026-11-25, for a few hours of an unreleased build, a kernel
    that read a `__constant__` variable was given, and read the image's initial values);
  - a function of a module asked for while another context than the module's is current: the
    layer would load the kernel's code in that other context;
  - a kernel that the layer does not translate from sm_80 machine code (at 2026-11-25 it
    translates 55 of the 3,652 functions of vLLM 0.34.0's sm_80 code and 464 of the 23,708 of
    PyTorch 2.14.0's; 23 and 429 before: an instruction passes only in a form that was seen to
    run right on a V100, `docs/translator.md`, section 1), or whose image has no code that the
    layer translates for the capability told (sm_86 and later);
  - an image linked from relocatable code: nothing is said either;
  - under eager loading with the CUDA 12 runtime's interface (the module of a library), an
    image of whose sm_80 code some kernel is refused: no module is given for it.
  What a program gets is then a mix: that kernel from the sm_80 code, the others from the sm_75
  code, where a GPU of 8.0 runs the sm_80 code of every one; a kernel present in both under one
  name is not looked at (PyTorch's attention kernels for sm_80, stubs in its sm_75 code, are
  still refused). The kernels that a program is given when it lists those of a library
  (`cuLibraryEnumerateKernels`) are those of the sm_75 code; `cuKernelGetLibrary` answers the
  program's library for such a kernel, `cuFuncGetModule` the layer's module (a program that
  unloads that module, or asks it for a variable, acts on the layer's copy). The translation of
  sm_80 machine code is still not independently validated (`docs/validation.md`, section 8, says
  what checks it); two instructions of sm_80 for bfloat16 are rewritten (`docs/emulations.md`,
  "What is not in machine code"): the results are those of one rounding, as on sm_80, and a NaN
  is given as 0x7fff, where what an sm_80 GPU gives for the bits of a NaN was not measured.
  `VF_TOLD_CODE=0` turns this off (the lookup fails as before, with the line); `VF_SAFE=1` does
  not (below). `VF_TRANSLATE=first` gives the machine code for the capability told precedence in
  every image, PyTorch's too.
- NVML, answered by the audit library, reports the value asked for to every caller: told 8.0, a
  program that reads NVML (vLLM does) and a CUDA runtime that is told 7.5 give it two answers.
- Triton compiles for the capability that PyTorch reports (`torch.cuda.get_device_capability`)
  whenever PyTorch is installed: Triton 3.8 imports PyTorch while it looks for a GPU, also in a
  program that does not import it. With `VF_CAPABILITY=8.0` alone, Triton is then told 7.5, as
  PyTorch's CUDA runtime is, and its `tl.dot` runs without tensor cores ([Speed](#speed)).
  `VF_CAPABILITY_ALL=1` gives it 8.0 and the tensor cores, and gives PyTorch 8.0 too, with the
  refusals of the first item (`VF_TRITON_TC` gives Triton the tensor cores at the default
  capability: [Triton's matrix products](#tritons-matrix-products)). Told 8.0, Triton also
  computes a float32 `tl.dot` of its default precision in tf32 (10 bits of mantissa), as on an
  Ampere GPU, where told 7.5 it computes it in float32; a bf16 `tl.dot` told 8.0 runs on the
  layer's scalar bf16 products, unless a `VF_BF16_TC_*` switch is set (`docs/emulations.md`,
  "bf16"); the `tf32x3` precision is refused. With `VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1`,
  vLLM's engine (greedy and sampling) and HTTP server gave the CPU's text with the small model of
  `tests/gpu/vllm.sh` (its stages b to d, 2026-11-23); `transformers` with its default attention,
  which is PyTorch's, does not.
  A program that does not itself import PyTorch can keep Triton from importing it with
  `TRITON_DEFAULT_BACKEND=nvidia`: Triton then asks the driver itself, and with
  `VF_CAPABILITY=8.0` it is told 8.0 while the libraries whose code runs from sm_75 machine code
  are still told 7.5.
- `TRITON_OVERRIDE_ARCH=sm80` does not make Triton 3.8 use tensor cores on a GPU it is told is
  7.5: Triton applies it to its own passes, but writes the PTX for, and calls ptxas with, the
  capability it is told. ptxas refuses the result (`Feature '.m16n8k16' requires .target sm_80
  or higher`), and a kernel with more than one stage (Triton's default is three) stops the
  process inside Triton (`LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.cp.async.wait.group`).
- What a caller is told is decided when it first asks. A library loaded after that (an extension
  opened once CUDA is in use) is not looked at, nor a library that opens the CUDA runtime with
  `dlopen` instead of naming it among the libraries it needs.
- A value that is not `<major>.<minor>` (`8`, `80`, `sm_80`), or that is below 7.0, is not used,
  and the layer says so once.
- **The architecture a kernel's PTX was written for** (`ptxVersion` of `cudaFuncGetAttributes`,
  `CU_FUNC_ATTRIBUTE_PTX_VERSION`), by which CUB and Thrust choose their code among the
  architectures a program was built for ("invalid device function" when it is none of them): for
  code that the layer retargeted from PTX for a later architecture, the layer answers that
  architecture, at most what the caller is answered for the machine code (the capability told),
  as a GPU that compiled that PTX answers; native sm_70 code, PTX for sm_70 or earlier and
  translated machine code (whose header keeps its PTX's architecture) are answered what the
  driver answers. What that does not cover: the image that the driver's linker makes from PTX
  inputs (`cuLinkAddData`, `cuLinkAddFile`) and the cubins of the layer's nvJitLink are answered
  7.0, the PTX that the driver compiled, so that CUB's algorithms in such code still answer
  "invalid device function"; a kernel of a library loaded whole from PTX text, asked about as a
  function in a context on a later card of the same machine, is answered what the driver answers
  there; the functions of an image linked from relocatable objects (`-rdc`) are all answered the
  lowest architecture of its objects; and a launcher client without GPU code of its own, told 7.5
  for the device, is answered 7.0 for both its kernels' PTX and machine code, the driver's
  answers, where a GPU of 7.5 says 7.5 for PTX for sm_75 (no device-wide CUB algorithm can be in
  such a client: its host side is in the file with its GPU code). A program whose PTX is only for
  an architecture above the one it is told (PTX for sm_80 alone with `VF_CAPABILITY=7.5`; by
  default such a program is told 8.0) is answered the capability told, never more: CUB then finds
  none of its architectures and answers "invalid device function" (`cub_sort.cu`, 2026-11-24),
  where a GPU of that capability has no image of its code at all and fails at the first call with
  "no kernel image is available"; the program's other kernels run through the layer. Until 0.1.5
  the layer answered 7.0 for all the code that it retargeted from PTX, the PTX that it gave the
  driver, so that CUB's device-wide algorithms failed with "invalid device function" in a program
  whose code is PTX only for a later architecture: in an outside run on 0.1.5, llama.cpp v1.0.0
  built as PTX for compute_75 only (as Ollama's `cuda_v13` runner is) stopped in `top_k_cub` with
  `-bs`, and with `draft-mtp` and its default backend sampling of drafts;
  `--no-spec-draft-backend-sampling` avoided it, and `draft-simple` was not affected
  (`docs/reports/`). That is fixed in 0.1.6: CUB's sort, sum and scan from such a build give the
  right results (`tests/gpu/run.sh`, the `cub-*` cases), and so did the ARGSORT (up to 1,048,576
  columns), TOP_K, CUMSUM, SUM and MEAN cases of llama.cpp b11521's `test-backend-ops` built so
  (2026-11-24). In the same outside tester's run with the fix (`e3c870b`, the code of 0.1.6,
  2026-11-25), the same llama.cpp v1.0.0 build ran with `-bs` and with `draft-mtp` and its default
  backend sampling of drafts to the end, and its other runs gave the tokens of 0.1.5, with the same
  log-probabilities of the five first candidates at every position where they were read
  (`docs/reports/`).

## CUDA graphs

A Tesla V100 can capture and replay translated kernels. The official SGLang AWQ server used CUDA graphs for both prefill and decode. A graph can report error 701 when it is instantiated or replayed if the translated kernel needs too many registers for the recorded block size. For translated machine code, `VF_BORROW=9 vf …` can preserve a source block size that would otherwise be lost. Shared-memory limits still apply. [SGLang measurements](docs/guides/benchmarks/sglang-default-operators.json) · [GPU validation](docs/validation.md#cuda-graph-replay)

## The bf16 cuBLAS variants

What `VF_CUBLAS_BF16_TC=1` and `VF_CUBLAS_BF16_SKINNY=1` change is in `INSTALL.md` and
`docs/emulations.md` (section 11). Both are off by default. Their limits:

- **The tensor-core path's results are not cuBLAS 12.9's bits**: they differ in the last bits
  (`docs/emulations.md`, section 11), in inference and in training. What it gains depends on the program's shapes.
  Measured on one model of each kind on one V100 (`docs/performance.md`, "bf16 products through cuBLAS: the opt-in
  variants on real programs"): eager generation at batch 8 and 32 ran 1.40 to 1.43 and 1.49 to 1.52 times as fast
  with it, prompts and scoring 1.33 to 1.74 times, an embedding encoder 1.53 to 1.77 times, Whisper's encoder 1.31
  times and its transcription at batch 16 1.26 to 1.29 times, the generation model under `torch.compile` 1.46 to
  1.92 times, and generation at batch 1 at 1.00 to 1.04 times its speed without the switch, unchanged within the
  noise of a case bound by the host (its steps' one column is never served; the prompt's products, served, take
  under 1 % of its time). A bf16 fine-tuning step (LoRA and full fine-tuning of the same 0.5B model) ran at 1.12
  to 1.24 times its speed without the switch: its forward products are served, those of its backward pass go to
  12.9 (below, the TN layout). `VF_SAFE=1` keeps every call at 12.9's bits whatever `VF_CUBLAS_BF16_TC` says
  ([Safe mode](#safe-mode-vf_safe1)).
- **Graph captures.** A tensor-core call captured in a graph is computed by cuBLAS 12.9: its
  workspace would be an allocation node of the program's graph, and with one, measured on driver
  580.178.04 (`tests/gpu/forward/bf16-graphs.c` with the test build's control
  `VF_BF16_CONTROL=capture-tc`, which serves captured calls), a second instantiation of the graph is
  refused, it cannot be embedded as a child graph or instantiated for device launch,
  `cudaGraphExecUpdate` from a recapture is refused, and a graph captured into a conditional node's
  body cannot be instantiated at all, while the same graph without the node works; the layer cannot
  tell any of these apart when it is called. A program that replays its steps from graphs gains from
  the tensor-core path only in what runs outside them: vLLM replays its decode steps and its prefill
  chunks up to the largest captured size (512 and 2048 tokens for the two models measured), and with
  `VF_CUBLAS_BF16_TC=1` its prefill was 6 % faster on one model and unchanged on the other (measured
  before the gate on k). A bfloat16 model on stock vLLM 0.34.0 told 8.0, CUDA graphs on, ran at 81.6
  tokens a second at one request and 745 at 32 with it, against 81.5 and 742 without (2026-11-24,
  Qwen2.5-0.5B-Instruct, `docs/performance.md`, "vLLM"). The skinny
  path serves captured calls (one kernel node, nothing allocated). A replayed tensor-core call has
  12.9's bits, which differ in their last bits from the same call made outside a capture: a program
  that compares a graph's replay with an eager run of the same step sees it.
- **One element's bits depend on the batch size, and on capture.** With the tensor-core path on, the
  same row of op(A) times the same column of op(B) has cuBLAS 12.9's bits when the product is narrow
  (n = 1, or within the skinny path's rule: n <= 16, or n < 32 when m k < 2^20) and the tensor-core
  path's when it is wide (m k >= 2^20 or n >= 32, outside the skinny path's rule, with k from 768 to
  16384) and made outside a graph capture; captured, the same call has 12.9's bits again. `VF_SAFE=1`
  removes what the variants add: it gives cuBLAS 12.9's bits for that call, whatever n and whatever the
  capture ([Safe mode](#safe-mode-vf_safe1)). 12.9's own results depend on n at n = 1 and, on some
  shapes, up to n = 16. Measured with
  12.9 alone on 18 TN shapes of linear layers (each column against the same column at n = 64): every
  shape gave other bits at n = 1 (in 77 to 95 % of the elements); TN 384 x 1536 and 256 x 2048, where
  12.9 runs another kernel up to n = 16, gave other bits in about 90 % of the elements at every n from
  2 to 16 and the same bits from n = 17; the 16 others gave the same bits at every n from 2 to 63.
- **The tensor-core path takes k >= 768 only** (`docs/emulations.md`, "The tensor-core path's gate"): a model
  gains on its layers with 768 in-features or more; a shorter product goes to the skinny path (n <= 16, within its
  rule) or to cuBLAS 12.9. On real operands at k = 768 and 896 (the layers of k = 768 and 896 of five models:
  Qwen2.5-0.5B, T5-base, OPT-125m, bge-base-en-v1.5 and whisper-small's encoder, 990 pairs against the exact
  product) every element's error was within 0.040 times the acceptance test's bound in fp32 output (35.8 units of
  2^-24 sum |a b| at most, 12.9's own up to 36.0), and every matrix's mean error towards zero was 0.82 units at
  most (0.046 times the test's bias bound read with the slope of samesign, the smallest of its families of one
  sign: no newer GPU's product was measured on these operands). None of these layers had non-negative inputs
  (42 to 59 % of each input's values were positive) and at most 51 % of a layer's products a b were positive
  (measured in review).
  Products of two operands of one sign spread over many binades fare worse: the acceptance test's two such
  families, log-normal magnitudes and an adversarial positive construction, come out low by 36 to 40 and 49 to 70
  units on average at every k from 768 to 16384, about 2.2e-6 to 4.2e-6 of their value (in fp32 output; far below
  bf16's own rounding), with single elements up to 90 and 216 units, each element still within the error bound;
  the same at k >= 1024, which the gate of 1024 served. That bias grows little with k (lognpos's is 36 to 40
  units at every k, advpos's 49 to 70, its largest at 16384); the test's bias bound, 0.8 times an RTX 5090's
  own on the same data, grows in proportion to k, and is exceeded below k = 1024: 1.16 and 1.25 times at
  k = 768 (where the path's bias is 0.93 and 1.00 times the RTX 5090's, from its slope on that data), 0.91 and
  1.003 at 896. The test judges these two families' bias from k = 1024 (documented exceptions, as the two
  constructions aligned with the path's k-slices are from 2048 and 4096). The data concerned are products of two
  non-negative (or sign-correlated) operands spread over many binades, with 768 to 1023 terms: for example Gram or
  similarity matrices of counts or of post-ReLU features, probabilities or attention weights times non-negative
  values, non-negative filterbanks applied to power spectra, NMF or Markov-chain products; a linear layer with
  zero-mean weights is not among them, whatever its inputs. There a mean error towards zero above the test's bound
  is to be expected at k = 768 to 1023. `VF_SAFE=1` keeps every call at 12.9's bits
  ([Safe mode](#safe-mode-vf_safe1)). The skinny path is unaffected.
- **The tensor-core path serves the TN layout only** (op(A) = A^T, op(B) = B: a framework's linear layer,
  y = x W^T, as cuBLAS gets it), a rule of speed. Inference keeps the path on the linear layers it gates in,
  and training on its forward products; the products of a backward pass, input gradients (NN) and weight
  gradients (NT, whose k is the batch's tokens), go to cuBLAS 12.9, and so do TT products. In the other
  layouts the path reads strided vectors, 64 to a block of its scale kernel and again for every chunk of
  columns, and a gradient's rows and columns often fail the exact check and go to its generic fallback. Every
  product of four fine-tuning steps (adapters of rank 16 and full fine-tuning, 2,048 and 4,096 tokens, a
  model of hidden size 896), timed in place with the gate at 1024, each call of a real step waited for: with
  the rule removed (a test build's control), 20 of the 22 NN and NT products the path would serve ran at 0.24
  to 0.92 times 12.9's speed (the other two, of 16 rows, at 1.01 and 1.35), and the steps' products together
  at 0.61 to 0.84 times; with the rule, at 1.02 to 1.08 times, the TN products it serves at 1.45 to 3.88
  times. With the gate at 768, which also serves the steps' forward products of 896 in-features, the steps'
  products together ran at 1.16 to 1.29 times 12.9's speed, those TN products at 1.45 to 2.21 times
  (`docs/performance.md`, section 7).
- **Narrow products of tall matrices go to cuBLAS 12.9**: TN with m >= 8192 is served only with n > 64, and TN
  with m >= 6144 only with k <= 9216 (an output layer over a vocabulary, or a wide MLP's up projection, at a batch
  of up to 64 tokens; and long products of many rows). With so many rows 12.9 fills the GPU. Up to 64 columns, its
  tile's width, it computed such products about as fast as the path or faster from m = 8256 on (1.014 to 1.03
  times as fast at m = 8256 to 8320, as fast or faster from 8384 on, down to 0.65 times its speed on output
  layers): on the layers of the models measured that class ran at a median of 1.07 times 12.9's speed and at 0.66
  at worst, below the class criterion of the rules of speed (`docs/performance.md`, section 7, "The rules of
  speed"). From 65 to 128 columns the path was faster at most points measured (up to 1.9 times), and within 3 % of
  12.9 or slower at some in an earlier sweep (down to 0.95 times, at 112 to 128 columns on output layers and at m
  = 10240): that class passes the criterion (a median of 1.24 and 1.04 at worst on the layers measured; on a sweep
  of that tier at m = 8192 to 248320 and k = 1024 to 4096, a median of 1.40 and 1.05 at worst) and is served. The
  long products of many rows (m from 6144 on with k beyond 9216) have no layer of a real model measured, and a
  class without one goes to 12.9: with operands that convert exactly, 12.9 was within 3 % of the path or faster in
  narrow bands of m and n in which its 64 x 64 tiles fill its waves (48 of the 951 points measured, down to 0.94
  times) and the path faster at the other widths of the same shapes (up to 1.9 times; the median of the 951
  points, 1.30); with 30 to 60 % of the input's columns failing, on synthetic operands, such shapes ran at 0.52 to
  0.89 times 12.9's speed (measured in review). With k up to 9216 the path was at least 1.05 times as fast at
  every point measured.
- **Wide down projections at many columns go to cuBLAS 12.9**: TN with k > 8192 is served only with n <= 256. Such
  products are the down projections of SiLU-gated MLPs, k their intermediate size: their input, the gated
  activation, holds values far below each column's largest, and on real text up to 75 % of a layer's columns fail
  the exact check; the fallback computes them, one fused multiply-add chain over the whole k each, which costs more
  than the path saves from a few hundred columns on. Whether a layer loses depends on its data, which the rule
  cannot see: of the down projections of four models measured on real text (k = 4864 to 12288, n = 32 to 2048),
  the layer that lost (k = 8960, 53 to 70 % of its columns failing) ran at 1.15 to 1.17 times 12.9's speed at 256
  columns and below 1.0 at every n from 384 on (down to 0.62 times; at 320, 0.85 or 1.08 by the choice of
  columns), while the other down projections of k > 8192 ran at 1.05 to 2.15 times 12.9's speed from 320
  columns on; the class goes to 12.9, and with it those gains. No model
  measured has a k between 4864 and 8960; the down projections of k = 4864 kept at least 1.08 times 12.9's speed
  at every n. On four ReLU and GELU models no token vector failed the check.
- **Calls not served**: one column (n = 1), batches of more than one matrix, the other entry points,
  explicit algorithms of `cublasGemmEx` and `cublasGemmStridedBatchedEx`, a `cublasLtMatmul`
  algorithm on the skinny path other than the one 12.9's heuristic gives there (id 16 with every
  configuration attribute as `cublasLtMatmulAlgoInit` leaves it, k in one split),
  pedantic modes, k above 16384 or below 768 on the tensor cores, products in another layout than TN on the
  tensor cores, and TN with m >= 8192 and n <= 64, with m >= 6144 and k > 9216, or with k > 8192 and n > 256
  there, cuBLASLt descriptors with any attribute outside
  the supported set, a `cublasLtMatmul` without a C descriptor, with a workspace size and no
  workspace, or with a workspace that is not 256-byte aligned (12.9 refuses that), a bias epilogue
  whose bias overlaps D, and every call of a context that is not a V100's. A legacy workspace set at
  an address that is not 256-byte aligned counts as smaller than 4 MiB (with 4 MiB 16 bytes past an
  aligned address, 12.9 ran its gemmSN kernel at TN 448 x 8 x 256). The tensor-core path does not follow
  a `cublasLtMatmul` algorithm: it computes the product its own way, whatever the algorithm says.
- **Speed that depends on the data.** A row of op(A) or a column of op(B) that does not convert
  exactly to fp16 is computed by the fallback, which is slower: with every column failing at
  n = 480, a tensor-core call took 32 to 71 % longer than cuBLAS 12.9's on six shapes of linear
  layers, TN m x k = 1536 x 1536, 4608 x 1536, 1536 x 4608, 7344 x 1536, 9728 x 896 and 896 x 4864
  (`tests/gpu/forward/bf16-cost.c`), against 1.7 to 2.8 times as fast when nothing fails (measured
  before the gate on k, which sends 9728 x 896 to 12.9). From `VF_LOG_LEVEL=2` the report at exit says
  how many tensor-core calls had rows or columns failing the check, how many of them more than a quarter
  of their rows or of their columns, and the share of the calls' elements that the fallback computed;
  `VF_PROFILE` gives the fallback kernel's own time (`vf_bf16_fallback_tn`). These three counts are kept on
  the GPU and read at exit only: `VF_SUMMARY`'s file, which a process killed without exiting leaves, has the
  counts of each entry point but not them, and says so. In a program whose first CUDA call is cuBLAS's (the
  variants' runtime then loaded before the driver), the report at exit comes after the driver's own teardown
  and says that they could not be read.
- **A served product can run below cuBLAS 12.9's speed**, whatever the rules of speed above. They serve a class of
  shapes when it has real members and, over every layer and n measured in it, its median speed was 1.10 times
  12.9's or more and its slowest 0.90 or more, and no model measured had its products' summed time above 12.9's at
  any n (`docs/performance.md`, section 7, "The rules of speed", gives each class). That floor holds for the
  products measured, not for every product of a served class: on the real operands of the eight models measured
  every product the rules serve ran at 1.04 times 12.9's speed or more but in one layer, and on the sweeps of the
  served classes at 0.95 or more. What takes a product below it is its data, not its shape: an input with many
  columns failing the exact check, which the fallback computes, one fused multiply-add chain over the whole k for
  each element of a failing column. At long k such a product can run well below the floor, at about half of
  12.9's speed at the slowest point measured, and more failing columns cost more: on synthetic operands with 30
  and 60 % of their input's columns failing (measured in review; 12.9 and the path alternated in one process, 3
  rounds; a failing column made by one element at 2^-40), served shapes ran at 0.83 and 0.74 times 12.9's speed
  (TN 4096 x 256 x 12288, the 9B hybrid's down projection's shape), 0.67 and 0.55 (TN 5120 x 256 x 13824), the
  same two shapes at 128 columns at 0.77 and 0.70 with 60 % failing, and the 4B hybrid's 2560 x 256 x 9216 at
  0.84 with 60 %; where the path's margin with converting inputs is wide it stayed faster (the 1.5B model's 1536 x
  256 x 8960, 1.18 with 60 %). The real layers measured at long k had few failing columns (at most
  2 % in the 9B hybrid's down projections, 3 % in the 4B hybrid's; the 1.5B model's layer with 53 to 70 %, of k =
  8960, is in the wide classes and goes to 12.9 from 257 columns), and the report at exit counts the calls with
  more than a quarter of their rows or columns failing (above; in review it counted each of the 338 synthetic
  calls). The one real layer measured below the floor is the documented exception, its own row in the class table
  (its model's summed step stays at 1.488 times 12.9's speed or more at every n measured with it served): a
  product whose input has a quarter or more of its columns failing, near the top of one of 12.9's waves.
  12.9 computes a bf16 product in tiles of 64 x 64, four per SM, 320 at a time on a V100:
  its time is flat over ranges of n and jumps where one more of its tiles lands on the busiest SM (at m =
  4096: at n = 129, 193, and 321, where its second wave begins). The tensor-core path's own time grows
  evenly with n, and the fallback adds one fused multiply-add chain over the whole k for every element of
  a failing column, less efficient per column than 12.9's tiles. With a quarter or more of the columns
  failing, the sum can pass 12.9's time on its flat stretches, most before its second wave. Measured on a
  layer of TN 4096 x n x 4096 whose input had 26 to 38 % of its columns failing, 12.9 and the path
  alternated in one process per n (2 passes): at worst 0.86 times 12.9's speed (n = 320), 0.86 to 0.97 at
  n = 128, 192 and 256 to 320, 1.00 to 1.03 at n = 96, 160 and 640, 1.05 to 1.44 at the other n from 32 to 1024; an attention
  layer of the same model and shape, whose columns all converted, at 1.33 to 2.46 (`docs/performance.md`,
  section 7, gives the table per n with 12.9's waves). No rule of shape tells such a layer from the others
  of its shape, and a choice made from the data would need a synchronization with the host for every call
  (measured: 11 % of the speed of eager decoding at batch 32) and would make an element's bits depend on
  other columns: the path serves it. The report at exit counts the calls with more than a quarter of their
  rows or columns failing (above).
- **Verify mode** (`VF_CUBLAS_BF16_VERIFY=1`) checks the calls a program makes, not more. Calls that several
  threads make on one stream are verified one after the other: a call holds its stream's scratch buffer from
  its first launch to its last (the copy of C, the path's kernels, 12.9's call, the comparison), so the
  threads enqueue their calls on that stream in turn; the per-thread default stream (`cudaStreamPerThread`)
  has a buffer for each thread (two threads, 300 calls each, on a stream of the program's, the legacy stream
  and the per-thread default stream: none reported as differing, where before this lock 260 to 599 of the
  600 calls were, in two runs; `tests/gpu/forward/bf16-verify-threads.c`). The graphs captured on one stream
  share that stream's scratch buffer: two of them replayed at the same time on different streams can report
  differences that are not there, or miss some (each graph replayed on the stream it was captured on, or one
  at a time, does not). The tensor-core path's fallback elements equal 12.9's where 12.9 computes each
  element in one chain over k, as its kernels for these products do on a V100
  (`tests/gpu/forward/bf16-fallback.c`, `bf16-verify.c`); a product for which 12.9 splits k would be
  reported as differing there. A call is not verified when no stream or memory could be made for it, when
  12.9 refuses it, or when a launch fails (the report counts them). The report reads its counters at exit,
  with each context made current: a program that destroys its context first loses them (said).
- **What a program sees of the layer's own work**: cuBLAS's logging (`CUBLASLT_LOG_LEVEL`) shows the
  layer's internal fp16 products; a framework's allocator does not see the layer's workspace, which
  the pool of the context keeps up to 64 MiB between eager calls.
- **cuBLAS's workspace contract is unchanged.** A forwarded call uses the handle's workspace as 12.9
  does: threads that share a handle and its workspace on different streams race in cuBLAS itself, as
  they do without the layer (`tests/gpu/forward/bf16-threads.c` gives each calling thread a context
  of its own; calls the layer serves use workspace of their own).
- **The skinny path's rule** was measured with CUDA 12.9.2's cuBLAS (build 12.9.2.10) on one V100 and
  one driver. The runtime checks that cuBLAS reports 12.9.2 (the build number cannot be read at run
  time); with another cuBLAS the skinny path serves nothing.
- **Contexts and processes.** A CUDA context gets the runtime's module at the first call either path
  serves in it, and a memory pool and a cuBLASLt handle at the first call the tensor-core path serves
  there; none is destroyed: a program that makes and destroys many contexts keeps one of each per
  context where they were made (the pool up to 64 MiB). That first call can come late, when device
  memory is nearly full: a module that fails to load then is not loaded again, and that context forwards
  every call for the rest of the process, with one line at level 1 (a pool or cuBLASLt handle that
  cannot be made: its tensor-core calls). A handle used from another context than its own is forwarded (by
  the code; not tested on a GPU). Two V100s in one process, and a V100 beside another GPU, were not
  run on real hardware: the host test, with a stand-in driver whose two devices have different compute
  capabilities, shows the decision taken per context's device at each call, every call of the other
  device's context forwarded. After a `fork`, a child that calls through the variants gets status 13
  and then `cudaErrorInitializationError`, as a child that calls cuBLAS 12.9 directly does, and the
  parent's next call is served with 12.9's bits (`tests/gpu/forward/bf16-default.c`, mode `fork`, run
  by hand).

## Several GPUs and NCCL

Any use of several GPUs from PyTorch goes through NCCL: `torch.distributed` (a process for each GPU), `torch.nn.DataParallel` (one process with several GPUs) and SGLang's tensor parallelism (below). Two builds of NCCL run on V100s through the layer: the CUDA 13 build of the wheels, since 0.1.6, and the CUDA 12 build, which has code for Volta and starts at once: fetch it with `vf fetch --only libnccl` and start the program through `vf`, which gives it that build in place of the wheel's. The CUDA 12 build stays the one to use where the start matters (below).

**The CUDA 13 build (`nvidia-nccl-cu13` 2.30.7, measured).** Its code is PTX for sm_120 and machine code for sm_75 and later; the layer cannot translate that machine code (its relocatable objects carry relocations in code: 1,196 of 1,258 functions refused), so it adapts the PTX. Nearly every function of its collectives holds instructions of multicast memory (`multimem.*`: 7,896 in 373 functions), behind tests of flags that only its NVLink-SHARP transport sets: the step counters of its simple protocol, for one, are read with `multimem.ld_reduce` only on a connection made for multicast. The PTX manual leaves these instructions undefined without a multicast address, and a V100 cannot have one: the 580 driver answers 0 to `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` and 801 (`CUDA_ERROR_NOT_SUPPORTED`) to `cuMulticastGetGranularity` and `cuMulticastCreate`, without the layer and through it, and NCCL says `NVLS multicast support is not available`. Until 0.1.5 the layer refused every kernel that might reach such an instruction: 41 of the 53 kernels of NCCL's device-linked image, and 18 of its 82 kernels for symmetric memory; the first collective failed. Since 0.1.6, where the driver says that no GPU of the process can take part in multicast, each such instruction, and not its function, is replaced by a reported trap, and the kernels are given (`docs/emulations.md`, "What is refused"): none of the 53 is refused, and of the 82 only the 17 that copy with `cp.async.bulk` and wait on `mbarrier` transaction counts, which NCCL 2.30 chooses on sm_100 and later only, with `NCCL_SYM_TMA_ENABLE=1`, and goes on without (it ignores a lookup that fails). The kernels given so are said at the default level in one line for each image (how many, the forms, the first three names; each kernel from `VF_LOG_LEVEL=2`), and the report at exit counts them. Reached all the same, a reported trap stops the kernel with an assertion that the driver writes (``volta-forever:1: FUNCTION: block: [...], thread: [...] Assertion `FORM cannot run on this GPU: ...` failed.``), the program gets `CUDA_ERROR_ASSERT` (710), the layer's report of the fault names the kernel with what it was given with, and the context ends; in the one run that reached one on purpose (`tests/gpu/run.sh` with `VF_TEST_FAULTS=1`, case `reported-fault`), the GPU wrote one Xid 43 line, and the next process ran on it. Where a GPU of the process can take part in multicast (a later card beside the V100: a library compiled kernel by kernel gives it the V100's PTX), where the driver cannot be asked, and with `VF_REPORTED_TRAPS=0`, those kernels are refused as before 0.1.6, and the first refusal of a kernel of NCCL's collectives says that the CUDA 12 build is the remedy. In the first two cases the line after the first of these refusals gives the driver's answer, which is otherwise said from `VF_LOG_LEVEL=2` only (a program without such a kernel hears nothing of multicast at the default level).

What it costs. The adapted code is 248 MB of PTX (136 MB as NCCL gives it), which the driver compiles when NCCL loads it in each process: on the first start, about 15 minutes of one core in each process, all at once (two ranks: a run of 545 s cut off by its time limit, then 411 s more for one process; `ptxas` of CUDA 12.9 took 1,100 CPU-seconds for the same objects), and 20 to 23 s of the layer's adaptation in each process at every start (the layer's cache does not keep what it makes of the relocatable objects of a device-linked image). Before the driver compiles an image given with reported traps that has 16 MiB of PTX or more, the layer says so once in the process, at the default level: the image, how much PTX, that the first compilation can take many minutes and is then kept in the driver's cache, and, for kernels of NCCL's collectives, that NCCL's CUDA 12 build starts in seconds (`vf fetch --only libnccl`). The driver keeps the compiled code in its cache (`CUDA_CACHE_PATH`, by default `~/.nv/ComputeCache`): about 383 MB, which its default limit of 1 GiB holds (measured: with the limit unset, nothing of NCCL's was evicted when other code was compiled; with 256 MiB, 134 MB of it was). That cache is shared by all the code that the driver compiles, the oldest used first out, so where other programs fill it or a smaller limit is set, raise `CUDA_CACHE_MAXSIZE` (up to 4294967296) and keep `CUDA_CACHE_PATH` on a directory that lasts. With that cache, on two Tesla V100-SXM2-32GB over NVLink (P2P/CUMEM, driver 580.178.04, 2026-11-25), `tests/gpu/nccl.sh` of 24410aa (all-reduce of sums in five types and max and min in int32, broadcast, all-gather, reduce-scatter, reduce, send and receive, all-to-all, on 1 to 4,194,304 elements of small integers, every value equal to the CPU's) passed in 38 to 46 s with the CUDA 13 build and 10 to 11 s with the CUDA 12 build 2.32.6, in one job, alternated: an all-reduce of 64 MiB of float32 reached a bus bandwidth of 33.3 GB/s (median of 3 runs; 32.9 to 33.8) against 34.8 (34.7 to 35.5), one of 8 bytes took 77.8 µs against 80.9 (the two builds are not the same version of NCCL). Since then `tests/gpu/nccl.sh` compares floats exactly too (float64, float32, float16 and bfloat16, sums of values that every order adds exactly, max and min), and pins each of NCCL's algorithms and protocols in a pass of its own (Ring and Tree with Simple, LL and LL128; NCCL's lines say what ran): 388 cases passed with each build, in 83 s with the CUDA 13 build and 37 s with the CUDA 12 build, which ran the same algorithms and protocols. SGLang 0.5.23 with `--tp-size 2 --disable-custom-all-reduce` (its all-reduces through NCCL) and Qwen2.5-0.5B-Instruct in float16 was ready in 142 s with the CUDA 13 build and 109 s with the CUDA 12 build, and gave the CPU's greedy tokens (48 for each of 3 prompts, three times) in the same 0.14 to 0.20 s a request.

SGLang starts one process per GPU for tensor parallelism. A user tested two and four V100s with NCCL's CUDA 12 build mapped through VF; with the CUDA 13 build and VF 0.1.1 the first all-reduce failed, its kernels refused for the multicast instructions that their indirect calls could reach. Four ranks also needed the uniform-address translation merged in PR #1. [Issue #2 and reproduction](https://github.com/nbald/volta-forever/issues/2) · [Address translation change](https://github.com/nbald/volta-forever/pull/1)

We separately tested this mapping with two Tesla V100-SXM2-32GBs. One process requested NCCL by soname and the other used the CUDA 13 library's absolute path. In a control with NCCL 2.30.7 for CUDA 13 (before 0.1.6), VF refused `AllReduce_Sum_f16_RING_LL` on both ranks and the run timed out during initialization. With the map, both ranks loaded NCCL 2.32.6 for CUDA 12. Each rank received the exact expected integer sums for 1, 257 and 65,536 elements. [Commands for the SGLang environment](docs/guides/sglang.md#6-additional-notes)

Four PyTorch ranks with the CUDA 12.9 build of NCCL also passed SUM and MAX reductions on four V100s. All 1,052,704 output values matched a CPU oracle, and NCCL reported peer access in all twelve directions. The full SGLang 35B run completed on two GPUs. The four-GPU model startup stopped before generation. [Collective and model checks](docs/guides/benchmarks/sglang-tensor-parallel.json)

## Peer copies over NVLink

One process enabled direct peer access between two V100-SXM2-32GBs connected by two NVLinks (`NV2`). It copied a 64 MiB buffer and verified every byte in each direction. Each rate below is the median of three sets of 32 copies after warm-up, measured one direction at a time on driver 580.178.04.

| Bulk peer copy | Native driver (GB/s) | With VF (GB/s) |
|---|---:|---:|
| GPU 0 → GPU 2 | 48.44 | 48.45 |
| GPU 2 → GPU 0 | 48.44 | 48.45 |

The timer covers the copies through stream synchronization. The driver did not expose per-link traffic counters on these cards. These rates describe application payload on this GPU pair. [Test source](tests/gpu/peer-copy.c)

Build with the CUDA headers and run on a pair listed as `NV2` by `nvidia-smi topo -m`. Set `CUDA_HOME` to your toolkit directory.

```sh
gcc -std=gnu11 -O2 -I"$CUDA_HOME/include" tests/gpu/peer-copy.c -lcuda -o /tmp/vf-peer-copy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 /tmp/vf-peer-copy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2 vf /tmp/vf-peer-copy
```

## What was not tried

- A program on several GPUs in one process, beyond the test kernels. `tests/gpu/multi-gpu.sh`
  (pull request #6) runs a test kernel on two GPUs at once from two threads of one process (native,
  retargeted from PTX and translated from machine code, under lazy and eager loading, with CUDA
  graphs and copies between the GPUs); the review of 2026-11-22 ran PyTorch on four V100s in one
  process, and the driver API from four threads on four devices, with no wrong result. No script of
  the repository runs a program on several GPUs in one process.
- A machine with a V100 and a later GPU. Modules loaded into a context on the later card are
  given to the driver as the program gave them; this is tested with a stand-in for the driver
  only. A library (`cuLibraryLoadData`) is loaded for every card: its fat binary keeps the entries
  that are replaced for the V100 beside their replacements, and the later card takes its own. A
  kernel refused for the V100 is given in the module that the driver gives for the library in a
  context on the later card, and refused when it is taken from the library (`cuLibraryGetKernel`,
  which is for every card); a handle to it that the program holds all the same is launched in a
  context on the later card and refused on the V100. A library whose PTX is compiled kernel by
  kernel (deferred) gives the later card the V100's PTX.
- MPS (the Multi-Process Service), and several processes on one V100, by the project. An outside
  tester ran the two runner processes of one Ollama 0.40.4 through the layer, adapting, compiling
  and writing the layer's and the driver's caches at once, each reply the same text as alone; and
  the same two runners as clients of one MPS server, with the same text and the same
  log-probabilities as without MPS, with both caches empty and with them warm (`docs/reports/`).
- Long runs: one was tried, for one model on one card. An outside tester ran Ollama 0.40.4's CUDA
  13 runner through the layer for 4 h 10 min, 4,000 requests to Qwen3.5-35B-A3B, and nothing
  drifted from the first hour to the last (the runner's memory, the layer's and the driver's
  caches, the speed; no Xid line) (`docs/reports/`).
- Drivers of the 580 branch other than 580.178.04, with which the project ran everything. Outside
  reports (`docs/reports/`) have `tests/gpu/run.sh` pass its 200 cases of 0.1.5, at the default
  and at `VF_CAPABILITY=8.0`, on 580.173.02, 580.178.04 and 580.196.07, with the V100 passed through
  to a virtual machine (Proxmox VFIO), and its 165 cases of 0.1.4 on 580.196.07, where an engine
  also ran until a bug of its own, on long prompts. The drivers after 580.196.07 were not tried.
- Distributions other than Ubuntu 24.04 with a GPU, by the project. Outside reports have the
  release's installer, `vf doctor`, the build from source and `tests/gpu/run.sh` (its 200 cases of
  0.1.5, both ways) on Rocky Linux 9.9 (glibc 2.34, the oldest that the layer takes), Ubuntu 26.04,
  Debian 13 and Fedora 45, with driver 580.178.04, SELinux enforcing on Rocky Linux and Fedora
  (`docs/reports/`).
- Ollama, by the project. An outside tester ran Ollama 0.40.4's stock CUDA 13 runner through the
  layer (told 7.5) against its stock CUDA 12 runner (sm_70), on four models: no kernel refused, no
  Xid line, each runner's text the same from one run to the next (`docs/reports/`; its speed:
  [Speed](#speed)).
- Windows; ARM
- CUDA 14 and later
- Nsight Compute and Nsight Systems (CUPTI's sampling of the program counter was used through
  the layer)
- Coverage across all model architectures. The engine guides give the checkpoints and workloads tested.

## Safe mode (`VF_SAFE=1`)

What `VF_SAFE=1` turns off, and how it is read, is in `INSTALL.md` ("Environment variables"). It
gives the bits of the conservative reference for what the layer chooses, not results that are the
same everywhere:

- **The reference has dependences of its own.** cuBLAS 12.9 chooses its kernel by the shape of a
  product, so the bits of one column can change with the number of columns (the batch: at n = 1,
  and on some shapes up to n = 16, measured in [The bf16 cuBLAS
  variants](#the-bf16-cublas-variants)); the
  layer's emulated products round otherwise than a later GPU's ([Numerical
  accuracy](#numerical-accuracy)). Safe mode keeps these: a model's outputs can still depend on
  its batch size, and differ from another GPU's.
- **What a program is told is unchanged.** A program told 8.0 (`VF_CAPABILITY`,
  `VF_CAPABILITY_ALL`) chooses its code for 8.0 and computes what that code computes: Triton's
  float32 `tl.dot` in tf32, for one ([The capability told with
  `VF_CAPABILITY`](#the-capability-told-with-vf_capability)).
- **Pairs of products in machine code are not made.** In sm_75 machine code, two products on one A
  computed together (a pair) add the halves of k, (C + k 0..3) + k 4..7, where a product translated
  by itself adds the even k, then the odd k (`docs/pairs.md`, section 9), and their last bits
  differ (every word of the `pair-bits` cases of `tests/gpu/run.sh`, with B loaded plain and
  transposed). By default a kernel gets pairs where it has the registers for them, and none at the
  third level of borrowing (`docs/translator.md`, "Registers"): a kernel that names nearly every
  register, one that requires a block (`.reqntid`) that its translation with pairs cannot have, one
  over its count with `VF_BORROW=9`. Safe mode translates each product by itself at every level,
  which gives these one result (the same bits at levels 0 and 3, measured), whatever `VF_BORROW`
  and `VF_FIT` say. (The second translation at launch takes no kernel with products to the third
  level: such a kernel keeps the pairs of its first translation, or its large block is refused,
  with safe mode or without.) In the PTX lowering a pair gives the bits of each product by itself
  (`docs/pairs.md`, section 13): those pairs stay. Measured on 2026-11-24 on one V100, in three
  rounds alternated after a warm-up: the fp16 GEMMs of Triton told 8.0 (PTX) and of cuBLAS ran at
  the same speed with `VF_SAFE=1` as without, Triton's with the same bits, while PyTorch's fp16
  attention (`scaled_dot_product_attention`, sm_75 machine code whose products pair) took 5.24 ms a
  call instead of 2.80, 1.9 times as long.
- **A kernel given from the sm_80 code of its image stays given.** A kernel that a program told
  8.0 asks for, and that only its image's sm_80 machine code has, is given from that code with
  safe mode too ([The capability told with `VF_CAPABILITY`](#the-capability-told-with-vf_capability)):
  it has no other code on this GPU, so there are no other bits to prefer; it is what the program
  chose for the capability it is told, which safe mode does not change. `VF_TOLD_CODE=0` makes
  the lookup fail instead.
- **Triton compiles for 7.5.** `VF_TRITON_TC`, set or by the launcher's default, is off: Triton's
  float16 products are then computed with float32 `fma` as without the switch, not summed by
  Volta's tensor cores ([Triton's matrix products](#tritons-matrix-products)); the kernels it
  compiles and Triton's cache entries are those of a run without the switch.
- **It is read once.** `libcuda.so.1` reads it when it is loaded, the nvJitLink adapter and the
  bf16 cuBLAS variants at their first use: a program that sets `VF_SAFE` while it runs does not get
  safe mode from `libcuda.so.1`, and may get it from the others. The layer's Python module reads it
  at the start of each Python and again when Triton is imported: a program that sets it before it
  imports Triton gets Triton's own compilation.

## Numerical accuracy

- **Machine-code translation** does not change arithmetic, Turing's tensor-core products apart
  (the table above): a translated kernel runs the program's own operations in the program's
  order, on other registers. The tests compare its results with the native sm_70 build of the
  same source where there is one (60 kernels, the same results), and with the CPU otherwise.
- **PTX**: integer products are exact. FP16 products are Volta's own, which add four terms
  at a time in float32: they give the host model's results where the sums are exact, and on
  arbitrary values differ from a later GPU's in the last bits, as a program built for Volta
  does (`docs/emulations.md`, "Products of halves and floats of later shapes"). bf16 and tf32
  products are computed by each thread with successive `fma` in float32, where the hardware
  keeps a wider sum and cuts it at the end: about 20 % of random bf16 results differ from an
  RTX 5090's by 1 or 2 units of the last place.
- **The last bits of a Triton FP16 product can depend on `num_stages`** (Triton told 8.0, whose
  `tl.dot` reaches the layer as PTX). A product whose two operands come straight from plain
  `ldmatrix` is added on the halves of k (two by two on one A where they pair, with the same
  bits), one whose operands are gathered from other threads on the even, then the odd k
  (`docs/emulations.md`, "Operands loaded from shared memory"). In a pipelined kernel
  (`num_stages` 2 or more) the operands of the first products of each turn of the loop are
  register copies of what the turn before loaded, which the layer gathers; with one stage none
  are. With B stored K x N (`ldmatrix.trans`), every product adds the even, then the odd k,
  whatever the stages. Both orders are valid sums
  of the same terms, by Volta's instruction four terms at a time; neither is wrong. Measured on
  2026-11-23 on one V100, Triton 3.8 through the layer, float16 inputs and outputs, float32
  accumulators, on 2048 x 7344 x 1536, 256 x 151936 x 1536 and 4096 x 4096 x 4096, with and
  without `VF_PAIRS`: with B stored N x K (K contiguous, as a linear layer's weight), the three
  1-stage configurations (tiles 128 x 128 x 32 with 8 or 4 warps, 64 x 128 x 32) gave one result,
  which was cuBLAS 12.9's bit for bit; the two 3-stage ones (128 x 128 x 32, 64 x 64 x 32)
  another, and 128 x 128 x 64 with 2 stages a third, each differing from cuBLAS's in 0.09 to
  0.15 % of the outputs, by at most 0.03125. With B stored K x N, all six configurations gave the
  same bits (0.16 to 0.20 % of the outputs off cuBLAS's).
- **Triton's products with `VF_TRITON_TC`**: float16 products on Volta's tensor cores, whose
  truncated sums are less exact than Triton's `fma` path on long sums of one sign; bfloat16
  products the same bits as without it in the GEMMs and the mixtures of experts measured, other
  bits with the same largest error in 56 of 146 shapes of vLLM's attention; int8 products exact,
  where without it Triton 3.3 to 3.9 cannot compile them for 7.5, so that there is no result
  without it to compare with ([Triton's matrix products](#tritons-matrix-products)).
- **Told 7.5, llama.cpp computes some operations otherwise than an sm_70 build does** (its choice,
  not the layer's): the products of quantized prompt batches on 8-bit activations
  ([Speed](#speed)), where an sm_70 build's cuBLAS takes fp16 ones (their errors against the CPU
  are about those of a native build with its MMQ kernels forced, on the same inputs), and the scan
  of the state-space layers of hybrid models (`SSM_SCAN`, Mamba-2) over chunks of more than 128
  tokens, which at 7.5 and later is made of matrix products with fp16 intermediates, where at 7.0
  it is a sequential scan in fp32: a normalized mean squared error against the CPU of about 8.5e-8
  instead of 2e-14 (llama.cpp's own test allows 2e-7 for that way). Through the layer, its results
  are bit-identical to those of a native sm_70 build changed to take the same way. So a hybrid
  model's tokens differ more from an sm_70 build's than a dense model's do: granite4:tiny-h
  (Q4_K_M), over 2,048 tokens, at a mean KL divergence of 0.0083 from the native build, where the
  native build with its MMQ kernels forced is at 0.0079, and with that scan forced at 0.0017
  (2026-11-24).
- **With greedy decoding, a near tie can end in a loop.** A near tie that the PTX build resolves
  the other way can send a text into a repetition loop that Ollama's repeat limit ends with HTTP
  500: 2 of the 4,000 requests of an outside long run of Ollama 0.40.4's CUDA 13 runner through the
  layer, both again on a new server and with the code of 0.1.6 (`e3c870b`), where its CUDA 12
  runner completes them; a third request loops with both runners (`docs/reports/`).
- **Conversions of the 4-bit and 8-bit formats**: identical to an RTX 5090's on every value of
  the formats.
- **Block-scaled products**: see the table above.
