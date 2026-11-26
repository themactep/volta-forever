#!/bin/bash
# GPU tests of the proxy, on a machine with a V100:
#   tests/gpu/run.sh [GPU]
# GPU is the PCI-order index of the GPU to use (default 0). Builds the test programs once into
# $VF_OUT (default $VF_TEST_ROOT), then runs each case and prints PASS or FAIL; exit status 1
# on any FAIL. The tree tested is the one this script is in; the build tested is its dist/lib, or
# the directory $VF_PROXY (a build staged beside the installed one). Each case sets the layer's
# settings it needs; of the others it takes VF_CAPABILITY only (a release is checked at 8.0 too,
# tests/release-check.sh), except in the cases of what a program is told by default.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
. "$W/tests/gpu/env.sh"
vf_clear_settings VF_CAPABILITY
G=$W/tests/kernels
T=$W/toolkits
out=${VF_OUT:-$VF_TEST_ROOT}
proxy=${VF_PROXY:-$W/dist/lib}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${1:-0}
. "$W/tests/gpu/cache.sh"
. "$W/scripts/nvcc-host.sh"

mkdir -p "$out/bin" "$out/logs"
[ -x "$out/bin/drvrun" ] && [ "$out/bin/drvrun" -nt "$G/drvrun.c" ] || gcc -O2 -Wall -o "$out/bin/drvrun" "$G/drvrun.c" -ldl -lm || exit 2
[ -x "$out/bin/tcrun" ] && [ "$out/bin/tcrun" -nt "$G/tcrun.c" ] || gcc -O2 -Wall -o "$out/bin/tcrun" "$G/tcrun.c" -ldl -lm || exit 2
[ -x "$out/bin/ptxrun" ] && [ "$out/bin/ptxrun" -nt "$G/ptxrun.c" ] || gcc -O2 -Wall -o "$out/bin/ptxrun" "$G/ptxrun.c" -ldl || exit 2
[ -x "$out/bin/rt_probe.135.sm75ptx" ] || "$G/build-rt-probe.sh" "$out/bin" >/dev/null || exit 2
rdc=$out/bin/rdc.135.sm120a
if [[ ! -x $rdc || $G/rdc_main.cu -nt $rdc || $G/rdc_other.cu -nt $rdc ]]; then
    nvcc_host "$T/cuda-13.5.0/bin/nvcc" -O2 -rdc=true --cudart=shared \
        -gencode 'arch=compute_120a,code=[sm_120a,compute_120a]' "$G/rdc_main.cu" "$G/rdc_other.cu" -o "$rdc" || exit 2
fi
[ -e "$W/dist/forward/libcublas.so.13" ] || { echo "missing: $W/dist/forward (run scripts/build-forwarders.sh)"; exit 2; }

failed=0
# check NAME EXPECTED-REGEX... : the case's output is in $log; every regex must match a line of it.
check() {
    local name=$1 ok=1; shift
    for want in "$@"; do grep -Eq -- "$want" "$log" || { ok=0; echo "    missing: $want"; }; done
    if [ $ok = 1 ]; then echo "PASS  $name"; else echo "FAIL  $name (output: $log)"; failed=1; fi
}
drv() {  # drv NAME MODULE EXPECTED... : load MODULE through the proxy with the driver-API harness
    local name=$1 module=$2; shift 2
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=3 "$out/bin/drvrun" "$module" >"$log" 2>&1
    check "$name" "$@"
}
tc() {  # tc NAME MODULE EXPECTED... : run the tensor-core kernels of MODULE through the proxy and check them against the host models
    local name=$1 module=$2; shift 2
    log=$out/logs/$name.log
    # The proxy takes the rows that ldmatrix reads to be at multiples of 16 and does not test them:
    # the kernels whose rows are deliberately elsewhere would fault, and are left out (-a), except
    # with VF_ALIGNED_ROWS=0, which makes the proxy test.
    local only=-a
    [ "${VF_ALIGNED_ROWS:-1}" = 0 ] && only=
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/tcrun" $only "$module" >"$log" 2>&1
    check "$name" "$@"
}
px() {  # px NAME MODULE WORD... : run the kernel `k` of a PTX module through the proxy; every lane of a warp must store these words
    local name=$1 module=$2; shift 2
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" "$module" k "$@" >"$log" 2>&1
    check "$name" "^PASS k of"
}
rt() {  # rt NAME PROGRAM LIBRARY-PATH EXPECTED... : run a CUDA-runtime program with the proxy first in the path
    local name=$1 program=$2 libs=$3; shift 3
    log=$out/logs/$name.log
    LD_LIBRARY_PATH=$proxy:$libs VF_LOG_LEVEL=2 "$out/bin/$program" >"$log" 2>&1
    check "$name" "$@"
}
by_default() {  # by_default CASE... : run the case without the caller's VF_CAPABILITY, then give it back
    local told=${VF_CAPABILITY-} was=${VF_CAPABILITY+set}
    unset VF_CAPABILITY
    "$@"
    [ -z "$was" ] || export VF_CAPABILITY="$told"
}
all7=("saxpy +ok" "mathk +ok" "block_sum +ok" "warp_sum +ok" "histo +ok" "half_dot +ok" "dbl +ok")

drv native-cubin "$G/k_basic.128.sm70.cubin" "${all7[@]}"
drv ptx-13.5-text "$G/k_basic.135.sm75.ptx" "PTX text without sm_70 code: PTX for sm_75 retargeted" "${all7[@]}"
drv fatbin-elf75-ptx75 "$W/crates/vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin" "fat binary without sm_70 code: PTX for sm_75 retargeted \(1 module\)" "${all7[@]}"
drv fatbin-with-sm70 "$W/crates/vf-fatbin/tests/data/elf70-elf75-ptx70-lz4.fatbin" "fat binary with code for this GPU" "${all7[@]}"
drv ptx-tanh "$G/k_tanh.135.sm75.fast.ptx" "PTX for sm_75 retargeted \(1 module, 1 tanh expanded\)" "tanhk +ok"
tc tensor-core "$G/k_tensor.135.sm75.ptx" "PTX for sm_75 retargeted \(1 module, 106 tensor-core instructions rewritten\)" "37 kernels, 0 wrong"
tc chains "$G/k_chains.135.sm75.ptx" "PTX for sm_75 retargeted \(1 module, 77 tensor-core instructions rewritten\)" "7 kernels, 0 wrong"
log=$out/logs/cta-mbarrier.log
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -t 256 "$G/ptx-cases/cta-mbarrier.ptx" k 0 28 >"$log" 2>&1
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -t 1 "$G/ptx-cases/cta-mbarrier.ptx" counts 0 108 >>"$log" 2>&1
check cta-mbarrier "256 of 256 lanes as expected" "1 of 1 lanes as expected"

# Positive powers of two take the shifted loop. The other divisors keep native signed division.
while read -r divisor w0 w1 w2 w3; do
    name=uniform-divisor-$divisor
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/ptxrun" -i "$divisor" "$G/ptx-cases/uniform-divisor.ptx" k "$w0" "$w1" "$w2" "$w3" >"$log" 2>&1
    check "$name" "^PASS k of .*: 32 of 32 lanes as expected"
done <<'DIVISORS'
8 fffffffc 00000000 fe000000 01ffffff
3 ffffffe4 ffffffff f1c71c72 0e38e38e
-3 ffffffe4 ffffffff f1c71c72 0e38e38e
1 fffffeff fffffff0 80000000 7fffffff
DIVISORS

# The x1 warp store uses eight independent row addresses and leaves the padding intact.
px stmatrix-x1 "$G/ptx-cases/stmatrix-x1.ptx" a9c31357 a9c31357 0
px stmatrix-multi "$G/ptx-cases/stmatrix-multi.ptx" 0 0
px stmatrix-transposed "$G/ptx-cases/stmatrix-transposed.ptx" 0 0

# Each half of a packed BF16 bound propagates its own NaNs and preserves signed zero.
px xla-sort-bounds "$G/ptx-cases/xla-sort-bounds.ptx" c0400000 80000000 00000000 0000000f
px packed-bf16-exact-sum "$G/ptx-cases/packed-bf16-exact-sum.ptx" 0 00018001 12345678 0
px same-line-debug-write "$G/ptx-cases/same-line-location.ptx" 43004300 43004300 43004300 43004300
px packed-bf16-convert "$G/ptx-cases/packed-bf16-convert.ptx" 40203f88 c13cc238 7fff0001 ff80ff80
px bf16-product "$G/ptx-cases/bf16-product.ptx" 00008000 00020000 7f807fff 0
px packed-bf16-bounds "$G/ptx-cases/packed-bf16-bounds.ptx" 00000002 80008001 7fff0002 40007fff
px scalar-bf16-bounds "$G/ptx-cases/scalar-bf16-bounds.ptx" 80010002 3f803f80 7fff7fff c040c000
px f16-bounds "$G/ptx-cases/f16-bounds.ptx" 80010002 7fff3c00 c0007fff 80017fff
px signed-zero-bounds "$G/ptx-cases/signed-zero-bounds.ptx" 00000000 80008000 80000000 80000000
px bf16-mma-branches "$G/ptx-cases/bf16-cases.ptx" 180 0 0 0
# An exactly representable BF16 tile and a tile with one out-of-range lane.
VF_BF16_TC_GROUPS=0 px bf16-mma-group-off "$G/ptx-cases/bf16-group.ptx" 42400000 5aa55aa5 12345678 0
VF_BF16_TC_GROUPS=1 px bf16-mma-group-on "$G/ptx-cases/bf16-group.ptx" 42400000 5aa55aa5 12345678 0
VF_BF16_TC_TILES=0 px bf16-mma-tiles-off "$G/ptx-cases/bf16-tiles.ptx" 42700000 c2200000 0 0
VF_BF16_TC_TILES=1 px bf16-mma-tiles-on "$G/ptx-cases/bf16-tiles.ptx" 42700000 c2200000 0 0
VF_BF16_COLUMN=ordered px bf16-column-ordered "$G/ptx-cases/bf16-column.ptx" 0 0 0 0
VF_BF16_COLUMN=tree px bf16-column-tree "$G/ptx-cases/bf16-column.ptx" 0 0 0 0
VF_BF16_ROW=ordered px bf16-row-ordered "$G/ptx-cases/bf16-row.ptx" 4b800000 4b800000 41800000 1
VF_BF16_ROW=tree px bf16-row-tree "$G/ptx-cases/bf16-row.ptx" 4b800008 4b800008 41800000 1
# Safe mode (VF_SAFE=1, INSTALL.md): a switch whose results are not the default's bits is not used,
# and one line says so. Four bf16 products whose terms round up one at a time with the default's
# fma.rn (3f800004), which Volta's tensor cores truncate (3f800000 with VF_BF16_TC_GROUPS=1): with
# VF_SAFE=1 the default's words; with VF_SAFE=true (not 1) the switch's. The same for
# VF_BF16_ROW=tree, whose words differ from the default's above. (Before safe mode, the cases with
# VF_SAFE=1 gave the switch's words: 2026-11-24.)
safe_px() {  # safe_px NAME SAID MODULE WORD...: px, and a line of the log matches SAID
    local name=$1 said=$2 module=$3; shift 3
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" "$module" k "$@" >"$log" 2>&1
    check "$name" "^PASS k of" "$said"
}
px bf16-group-rounding "$G/ptx-cases/bf16-group-rounding.ptx" 3f800004 3f800004 3f800004 3f800004
VF_BF16_TC_GROUPS=1 px bf16-group-rounding-on "$G/ptx-cases/bf16-group-rounding.ptx" 3f800000 3f800000 3f800000 3f800000
VF_SAFE=1 VF_BF16_TC_GROUPS=1 safe_px safe-bf16-group-rounding "VF_SAFE=1: the layer keeps its conservative paths: VF_BF16_TC_GROUPS=1, VF_PAIRS in machine code not used" \
    "$G/ptx-cases/bf16-group-rounding.ptx" 3f800004 3f800004 3f800004 3f800004
VF_SAFE=true VF_BF16_TC_GROUPS=1 safe_px safe-not-one-bf16-group-rounding "VF_SAFE='true' is not 1: safe mode stays off" \
    "$G/ptx-cases/bf16-group-rounding.ptx" 3f800000 3f800000 3f800000 3f800000
VF_SAFE=1 VF_BF16_ROW=tree safe_px safe-bf16-row-tree "VF_SAFE=1: the layer keeps its conservative paths: VF_BF16_ROW=tree, VF_PAIRS in machine code not used" \
    "$G/ptx-cases/bf16-row.ptx" 4b800000 4b800000 41800000 1

# Kernels that exist as PTX text only (tests/kernels/ptx-cases): four products
# on accumulators that hold 1.0, after a scope that declares their family again and sets its own
# registers to zero (33 in every register: 32 until 2026-11-19, the 1.0 was lost), the same with
# another name in the scope, and a product that takes a register of its own accumulator as its A
# operand, on f32 accumulators (34 in the upper rows, 32 below: 24 until then) and on f16 ones
# (72 and 16: 8 until then).
px ptx-family-declared-again "$G/ptx-cases/zeros-shadow.ptx" 42040000 42040000 42040000 42040000
px ptx-family-of-another-name "$G/ptx-cases/zeros-renamed.ptx" 42040000 42040000 42040000 42040000
px ptx-accumulator-as-operand "$G/ptx-cases/alias-a.ptx" 42080000 42080000 42000000 42000000
px ptx-accumulator-as-operand-f16 "$G/ptx-cases/alias-a-f16.ptx" 54805480 4c004c00
# Each warp computes 16 ones per destination, plus the specified accumulator fragment.
px ptx-mma-constant-accumulator "$G/ptx-cases/mma-immediate.ptx" 41980000 41580000 41a00000 418c0000
# A member of a family of registers written with zeros before its number (%d00 for %d0 of
# %d<4>: the same register to the assembler) reads the accumulator in the middle of a chain of
# products: 32.0 after four of the eight, and 64.0 in the others at the end.
px ptx-padded-chain-read "$G/ptx-cases/padded-chain-read.ptx" 42000000 42800000 42800000 42800000
# The same with comments inside the declaration of the family and between the operands of the
# statement that reads: a comment is not code (read with its comment, the declaration was not
# taken for a family until 2026-11-21, and what was read was 0).
px ptx-padded-chain-read-commented "$G/ptx-cases/padded-chain-read-commented.ptx" 42000000 42800000 42800000 42800000
px ptx-chain-nonconsecutive "$G/ptx-cases/chain-nonconsecutive.ptx" 42000000 42000000 42800000 42800000
px ptx-reduction-exited "$G/ptx-cases/reduction-exited.ptx" 5 1 4 5
# The same over floats, a form of sm_100a: the sign of a zero (+0 > -0) and the canonical NaN, in
# every surviving lane. (Each lane reduces in the same order since 2026-11-22; Volta's minimum
# and maximum give the same for both orders of their operands, so this passed before too.)
px ptx-reduction-exited-f32 "$G/ptx-cases/reduction-exited-f32.ptx" 80000000 00000000 7fffffff 7fffffff
# One lane alone: without .NaN, the least and the greatest of one NaN are the canonical NaN, as of
# several (until 2026-11-23 the NaN of the lane was given as it was, 7fc00001 and ff800001); the
# least of -0 alone and the greatest of -1 alone are what they are. (Not run on later hardware:
# redux.f32 is of sm_100a and sm_103a, not of the RTX 5090, whose sm_120 ptxas refuses it.)
px ptx-reduction-one-lane-f32 "$G/ptx-cases/reduction-one-lane-f32.ptx" 7fffffff 7fffffff 80000000 bf800000
# The member mask in a register set to all ones before labels that nothing branches to and before
# a branch out, the register a loop counter afterwards (Triton's code): refused until 2026-11-23.
px ptx-reduction-mask-reaching "$G/ptx-cases/reduction-mask-reaching.ptx" 210 1 20 2d
px ptx-copy-octal "$G/ptx-cases/copy-octal.ptx" 12345678 abcdef01 0 0
px guarded-copy "$G/ptx-cases/guarded-copy.ptx" 0 0 0 0
# An amount chosen once between nothing and all (Triton's pipelined loads): one predicated load.
px chosen-copy "$G/ptx-cases/chosen-copy.ptx" 0 0 0 0
# Pairs of FP16 products on one A (docs/pairs.md, section 13) on values that make the order of the
# additions show (subnormals, infinities, NaN, the largest half, cancellation), in a loop of 1, 2
# and 37 turns: each product computed by itself (VF_PAIRS=0) and the pairs give the same words. (A
# build that only adds the two halves of k in the other order gives other words at every count.)
while read -r turns w0 w1 w2 w3; do
    for pairs in 0 1; do
        name=pairs-adversarial-$turns-$pairs
        log=$out/logs/$name.log
        VF_PAIRS=$pairs DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -i "$turns" "$G/ptx-cases/pairs-adversarial.ptx" k "$w0" "$w1" "$w2" "$w3" >"$log" 2>&1
        check "$name" "^PASS k of .*: 32 of 32 lanes as expected"
    done
done <<'TURNS'
1 a556a958 d66347fa 814664c2 0000000c
2 dd7fa33d cf83424e 6537527c 00000032
37 858142b0 54074692 63b15f97 00000032
TURNS
# The same kernel with readers of the accumulators after the loop where not every lane runs: under
# a branch that some lanes take, and in a loop whose number of turns depends on the lane (the pair
# whose register they read must be dropped); at the join of such a branch right before an ldmatrix
# (every lane gets there: that pair stays, with its conversions there). The words of each product
# computed by itself (VF_PAIRS=0). (A lowering that took every place for one where the 32 lanes run
# together fails the six with pairs, the GPU stopping the kernel: 2026-11-23.)
while read -r case turns w0 w1 w2 w3; do
    for pairs in 0 1; do
        name=$case-$turns-$pairs
        log=$out/logs/$name.log
        VF_PAIRS=$pairs DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -i "$turns" "$G/ptx-cases/$case.ptx" k "$w0" "$w1" "$w2" "$w3" >"$log" 2>&1
        check "$name" "^PASS k of .*: 32 of 32 lanes as expected"
    done
done <<'TURNS'
pairs-divergent-branch 1 a556a958 d66347fa e3f42f26 0000000c
pairs-divergent-branch 2 dd7fa33d cf83424e 5733154a 00000032
pairs-divergent-branch 37 858142b0 54074692 0fe4e688 00000032
pairs-divergent-loop 1 a556a958 d66347fa a3963f76 0000000c
pairs-divergent-loop 2 dd7fa33d cf83424e 67688f20 00000032
pairs-divergent-loop 37 858142b0 54074692 bddcf416 00000032
TURNS
# The same product with B by plain ldmatrix (summed on k 0..3, then 4..7) and by ldmatrix.trans
# (even k, then odd k): 52, 53 and 78 of the 128 results differ in their bits. Why products with B
# by ldmatrix.trans are not paired; a lowering that summed them in the halves of k would give 0.
while read -r turns w0 w1 w2 w3; do
    name=trans-order-$turns
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -i "$turns" "$G/ptx-cases/trans-order.ptx" k "$w0" "$w1" "$w2" "$w3" >"$log" 2>&1
    check "$name" "^PASS k of .*: 32 of 32 lanes as expected"
done <<'TURNS'
1 00000034 b28c04cc 8117e4ae 00000001
2 00000035 c21ec7db a2c0de75 00000002
37 0000004e b1a314a3 cc4bf5eb 00000025
TURNS
# Built from the same sources with CUDA 12.9 ptxas -lineinfo -arch=sm_80.
px cubin-embedded-ptx "$G/ptx-cases/copy-octal.sm80.cubin" 12345678 abcdef01 0 0
px cubin-embedded-division "$G/ptx-cases/signed-division.sm80.cubin" 0 0 0 0
px cubin-embedded-import "$G/ptx-cases/assert-import.sm80.cubin" 12345678 9abcdef0 0badf00d 00c0ffee
# Built with CUDA 13.5 ptxas -lineinfo -arch=sm_80: the PTX compiled again rounds as the machine
# code did. (1 + 2^-23) * (1 - 2^-23) - 1 is 0 rounded twice (--fmad=false: FMUL and FADD, with and
# without a driver import) and -2^-46 rounded once (--fmad=true: FFMA). (Until 2026-11-23 the
# driver's compiler contracted the first two too, and they gave a8800000.)
px cubin-embedded-separate-product "$G/ptx-cases/product-sum.fmad-false.sm80.cubin" 00000000 3f800001 3f7ffffe bf800000
px cubin-embedded-separate-product-import "$G/ptx-cases/product-sum-assert.fmad-false.sm80.cubin" 00000000 3f800001 3f7ffffe bf800000
px cubin-embedded-fused-product "$G/ptx-cases/product-sum.fmad-true.sm80.cubin" a8800000 3f800001 3f7ffffe bf800000
# CUDA 12.9 ptxas -lineinfo -arch=sm_75 predicates the HMMA; the PTX keeps its uniform branch.
px cubin-turing-ptx-retry "$G/ptx-cases/guarded-mma.sm75.cubin" 41300000 41400000 41500000 41600000
# ldmatrix at a variable of shared memory, no register in the address (Triton's [global_smem]): as
# PTX, and as the sm_75 machine code that CUDA 12.9 ptxas -lineinfo makes of it (LDSM [RZ], which the
# translator refuses: the layer runs the PTX kept in the cubin). Both refused until 2026-11-23.
px ptx-ldmatrix-variable "$G/ptx-cases/ldmatrix-variable.ptx" a0b0c000 a0b0c003 a0b0c005 a0b0c006
px cubin-ldmatrix-variable "$G/ptx-cases/ldmatrix-variable.sm75.cubin" a0b0c000 a0b0c003 a0b0c005 a0b0c006
# Kernels that require blocks of 1,024 threads (.reqntid, as Triton writes it), at 64 registers on
# sm_75, whose usual translation takes more than such a block may have: ptxrun reads the largest
# block that the loaded kernel allows before it launches, as Triton does, and stops when it is
# smaller. The first is translated within 64 registers, the second, which no translation keeps
# within them, runs from the PTX that its cubin keeps (both made by CUDA 13.5 ptxas -lineinfo). Until
# 2026-11-23 both were given translated, at 66 and 73 registers: blocks of 896 and 768 threads at most.
rb() {  # rb NAME ROUTE WORD... : kernel `k` of tests/kernels/translate/k_required_block_NAME.sm75.cubin, 1,024 threads
    local name=$1 route=$2; shift 2
    log=$out/logs/cubin-required-block-$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -t 1024 -i 5 \
        "$G/translate/k_required_block_$name.sm75.cubin" k "$@" >"$log" 2>&1
    check "cubin-required-block-$name" "^PASS k of" "$route"
}
rb within "machine code for sm_75 translated \(1 functions\)" 082eedf0 a2a5da74 e4681800 00000005
rb over "PTX for sm_75 retargeted" 00000000 40b1ac30 00000054 00000005
tc ampere "$G/k_ampere.135.sm80.ptx" "PTX for sm_80 retargeted \(1 module, 25 tensor-core instructions rewritten\)" "31 kernels, 0 wrong"
# FP8: the conversions both ways, and the products, which run on the tensor cores as products of halves.
tc fp8 "$G/k_fp8.135.sm89.ptx" "PTX for sm_89 retargeted \(1 module, 8 tensor-core instructions rewritten\)" "19 kernels, 0 wrong"
# The conversions of the 4-bit floats (E2M1) and of the exponents alone (UE8M0), against host models.
tc fp4 "$G/k_fp4.135.sm120a.ptx" "PTX for sm_120 retargeted \(1 module\)" "10 kernels, 0 wrong"
# The same conversions against what an RTX 5090 computed: on the inputs of the
# reference files (every value of the small formats; for floats every midpoint with a float on each
# side), the layer's results must be those files, byte for byte.
B=$G/blackwell
[ -x "$out/bin/fp4_vectors" ] && [ "$out/bin/fp4_vectors" -nt "$B/fp4_vectors.c" ] || gcc -O2 -o "$out/bin/fp4_vectors" "$B/fp4_vectors.c" -ldl -lm || exit 2
log=$out/logs/fp4-hardware.log
mkdir -p "$out/logs/fp4-hardware"
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=1 "$out/bin/fp4_vectors" "$B/k_fp4_vectors.135.sm120a.ptx" "$out/logs/fp4-hardware" >"$log" 2>&1
for f in "$B"/ref/*.bin; do
    cmp -s "$f" "$out/logs/fp4-hardware/$(basename "$f")" && echo "the same: $(basename "$f")" || echo "NOT THE SAME: $(basename "$f")"
done >>"$log"
check fp4-hardware "^the same: decode_e2m1.bin" "^the same: decode_e2m1_relu.bin" "^the same: decode_ue8m0.bin" "^the same: encode_e2m1.bin" \
    "^the same: encode_e2m1_relu.bin" "^the same: encode_ue8m0_rp.bin" "^the same: encode_ue8m0_rz.bin" \
    "^the same: encode_ue8m0_bf16_rp.bin" "^the same: encode_ue8m0_bf16_rz.bin"
# Block-scaled MMA (NVFP4, MXFP4, MXF8F6F4): against reference vectors that an RTX 5090 computed
# (`docs/emulations.md`, "Block-scaled products"), form by form. The results are compared with the hardware's, one by one: identical,
# one unit of the last place away, further away, a NaN against a number. The layer does not give the
# hardware's results for every form (its sums are cut or rounded to f32 where the hardware's wider
# accumulator does otherwise), so each form is held to what it gives today: no fewer identical
# results, no more further away, no NaN mismatch. NVFP4 (kind::mxf4nvf4, 4X, UE4M3) is held to
# more: nothing further than one unit, on every vector set. A form that gets worse fails.
[ -x "$out/bin/mma_scaled_check" ] && [ "$out/bin/mma_scaled_check" -nt "$B/mma_scaled_check.c" ] || gcc -O2 -o "$out/bin/mma_scaled_check" "$B/mma_scaled_check.c" -ldl -lm || exit 2
log=$out/logs/block-scaled.log
: >"$log"
scaled_run() {  # scaled_run MODULE SUFFIX [KERNEL...]
    local module=$1 suffix=$2; shift 2
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/mma_scaled_check" "$B/$module" "$B/ref_scaled" "$suffix" "$@" >>"$log" 2>&1 || true
}
scaled_run k_mma_scaled_forms.135.sm120a.ptx ""
scaled_run k_mma_scaled_forms.135.sm120a.ptx _large
scaled_run k_mma_scaled_extra.135.sm120a.ptx "" s_mxf4_tid_a1 s_mxf4_tid_b1 s_mxf4_bid_a2 s_mxf8_e5m2_1x
scaled_form() {  # scaled_form NAME TOTAL MIN-IDENTICAL MAX-FURTHER MAX-ONE-UNIT (counts measured 2026-11-18)
    local name=$1 total=$2 min_identical=$3 max_further=$4 max_one=$5 line identical one further nanmm launch tot
    line=$(grep -E "^RESULT $name " "$log") || { echo "    missing: RESULT $name"; return 1; }
    identical=$(sed -E 's/.* identical=([0-9]+).*/\1/' <<<"$line"); one=$(sed -E 's/.* one_unit=([0-9]+).*/\1/' <<<"$line")
    further=$(sed -E 's/.* further=([0-9]+).*/\1/' <<<"$line"); nanmm=$(sed -E 's/.* nan_mismatch=([0-9]+).*/\1/' <<<"$line")
    launch=$(sed -E 's/.* launch_fail=([0-9]+).*/\1/' <<<"$line"); tot=$(sed -E 's/.* total=([0-9]+).*/\1/' <<<"$line")
    local ok=0
    [ "$tot" = "$total" ] && [ "$launch" = 0 ] && [ "$nanmm" = 0 ] && [ "$identical" -ge "$min_identical" ] &&
        [ "$further" -le "$max_further" ] && [ "$one" -le "$max_one" ] && ok=1
    echo "    $name: identical $identical (>= $min_identical), one unit $one (<= $max_one), further $further (<= $max_further), NaN mismatches $nanmm (0), of $tot"
    [ $ok = 1 ]
}
scaled_forms() {
    local bad=0
    # NVFP4: nothing further than one unit.
    scaled_form s_nvf4_4x 36352 36352 0 0 || bad=1
    scaled_form s_nvf4_4x_large 406912 405861 0 1051 || bad=1
    # The other forms: no worse than today.
    scaled_form s_mxf4_2x 36352 36242 4 106 || bad=1
    scaled_form s_mxf4_2x_large 394112 371635 841 21636 || bad=1
    scaled_form s_mxf8_e4m3_1x 36352 24272 4021 8059 || bad=1
    scaled_form s_mxf8_e4m3_1x_large 387712 308475 26699 52538 || bad=1
    # Other thread_id and byte_id values, E5M2 operands.
    scaled_form s_mxf4_tid_a1 38400 36488 143 1769 || bad=1
    scaled_form s_mxf4_tid_b1 38400 36365 186 1849 || bad=1
    scaled_form s_mxf4_bid_a2 38400 36429 164 1807 || bad=1
    scaled_form s_mxf8_e5m2_1x 38400 36359 578 1463 || bad=1
    return $bad
}
if scaled_forms >"$log.forms"; then ok=PASS; else ok=FAIL; failed=1; fi
cat "$log.forms"
echo "$ok  block-scaled (output: $log)"
check block-scaled-loaded "PTX for sm_120 retargeted \(1 module, 3 tensor-core instructions rewritten\)" \
    "PTX for sm_120 retargeted \(1 module, 4 tensor-core instructions rewritten\)"
# Instructions of the GPUs after Ampere with a plain equivalent: arithmetic on pairs of floats, cache policies.
tc later "$G/k_later.135.sm100.ptx" "PTX for sm_100 retargeted \(1 module\)" "3 kernels, 0 wrong"
# bf16 outside the products: arithmetic, comparisons, the conversion from f64, atomic additions.
tc bf16 "$G/k_bf16.135.sm90.ptx" "PTX for sm_90 retargeted \(1 module\)" "15 kernels, 0 wrong"
# The same with the addresses of the rows tested when the code runs, for programs that do not keep
# them at multiples of 16: the kernels with such rows are run too.
VF_ALIGNED_ROWS=0 tc tensor-core-tested "$G/k_tensor.135.sm75.ptx" "PTX for sm_75 retargeted \(1 module, 106 tensor-core instructions rewritten\)" "42 kernels, 0 wrong"
VF_ALIGNED_ROWS=0 tc ampere-tested "$G/k_ampere.135.sm80.ptx" "PTX for sm_80 retargeted \(1 module, 25 tensor-core instructions rewritten\)" "35 kernels, 0 wrong"
# The tensor-core kernels as machine code for sm_75, without program text: Turing's matrix load
# and product on halves are replaced in the machine code (`docs/translator.md`, "Tensor-core products in machine code"). 28 of the 37 functions
# are translated (integer products and MOVM are not, and are refused at the lookup); the rows of
# a matrix may be at any address, so the kernels with rows elsewhere than at multiples of 16 run too.
VF_ALIGNED_ROWS=0 tc tensor-core-machine-code "$G/k_tensor.135.sm75.cubin" "machine code for sm_75 translated \(37 functions\)" "9 of these functions are not translated" \
    "emulated in t_mma_f32: 33 instructions added in translated machine code" "emulated in t_mma_f16: 39 instructions added in translated machine code" "32 kernels, 0 wrong"
# Products that follow each other on one register of B: in machine code the elements of B are
# gathered for the first and kept for the next ones, until something writes the register
# (the same section). The same kernels as program text, where each product is rewritten by itself.
tc kept-machine-code "$G/k_kept.135.sm75.cubin" "machine code for sm_75 translated \(5 functions\)" "5 kernels, 0 wrong"
tc kept "$G/k_kept.135.sm75.ptx" "PTX for sm_75 retargeted \(1 module, 12 tensor-core instructions rewritten\)" "5 kernels, 0 wrong"
# GROUP patterns: two A x four B = eight products per group, the accumulators chained along loops.
# Each kernel exercises a different way of entering, leaving or interrupting such a chain.
# Machine code: the translator takes 9 of the 11 functions; it refuses the one whose products are
# predicated and the one whose accumulators are f16. (Until 2026-11-20 it refused three more,
# g_once, g_partial and g_shared: a product of theirs writes its results over the register of its
# B, which is no harm: `matrix_product`. FlashInfer's attention kernels have such products.)
tc groups-machine-code "$G/k_groups.135.sm75.cubin" "machine code for sm_75 translated \(11 functions\)" "2 of these functions are not translated" "22 kernels, 0 wrong"
tc groups "$G/k_groups.135.sm75.ptx" "PTX for sm_75 retargeted" "26 kernels, 0 wrong"
# PAIRS patterns: unconditional loads in loops, testing the pairs-of-products-on-whole-rows arrangement.
# Machine code: the translator takes 11 of 12 functions (p_uniform is refused: predicated HMMA).
# (p_basic's count says the pairs were made: 234 instructions are added when each product is translated by itself.)
tc pairs-machine-code "$G/k_pairs.135.sm75.cubin" "machine code for sm_75 translated \(12 functions\)" "1 of these functions is not translated" \
    "emulated in p_basic: 91 instructions added in translated machine code" "34 kernels, 0 wrong"
tc pairs "$G/k_pairs.135.sm75.ptx" "PTX for sm_75 retargeted" "37 kernels, 0 wrong"
# The same machine code with each product translated by itself (VF_PAIRS=0), as it was before
# pairs were the default: that path stays in use where a function has no register left for pairs.
VF_PAIRS=0 tc pairs-machine-code-unpaired "$G/k_pairs.135.sm75.cubin" "machine code for sm_75 translated \(12 functions\)" "1 of these functions is not translated" \
    "emulated in p_basic: 234 instructions added in translated machine code" "34 kernels, 0 wrong"
VF_PAIRS=0 tc groups-machine-code-unpaired "$G/k_groups.135.sm75.cubin" "machine code for sm_75 translated \(11 functions\)" "2 of these functions are not translated" "22 kernels, 0 wrong"
# Safe mode and the pairs of machine code (tests/kernels/k_pair_bits.cu): eight FP16 products a turn
# on two A, with B by plain and by transposed ldmatrix, on halves of many magnitudes, in a loop of 1
# and 37 turns. A pair adds the halves of k, a product translated by itself the even k, then the odd:
# the words differ (the control: the pairs by default, each product by itself with VF_PAIRS=0). The
# same code requiring blocks of 1,024 threads (.reqntid) is translated at level 3, which makes no
# pairs: by default, at that block, each product by itself. With VF_SAFE=1, the words of each product
# by itself in every case, and one line says that the pairs are not used; with VF_SAFE=true (not 1),
# the words of the pairs. (Before safe mode turned the pairs off, the cases with VF_SAFE=1 and no
# required block gave the words of the pairs: 2026-11-24.)
pb() {  # pb NAME THREADS TURNS CUBIN KERNEL SAID WORD...: ptxrun of KERNEL; a line of the log matches SAID when it is not empty
    local name=$1 threads=$2 turns=$3 cubin=$4 kernel=$5 said=$6; shift 6
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ptxrun" -t "$threads" -i "$turns" "$G/$cubin" "$kernel" "$@" >"$log" 2>&1
    check "$name" "^PASS $kernel of .*: $threads of $threads lanes as expected" "machine code for sm_75 translated \(2 functions\)" ${said:+"$said"}
}
said_safe="VF_SAFE=1: the layer keeps its conservative paths: VF_PAIRS in machine code not used"
while read -r kernel turns p0 p1 p2 p3 u0 u1 u2 u3; do
    pb pair-bits-$kernel-$turns 32 "$turns" k_pair_bits.135.sm75.cubin "$kernel" "" "$p0" "$p1" "$p2" "$p3"
    VF_PAIRS=0 pb pair-bits-$kernel-$turns-unpaired 32 "$turns" k_pair_bits.135.sm75.cubin "$kernel" "" "$u0" "$u1" "$u2" "$u3"
    VF_SAFE=1 pb safe-pair-bits-$kernel-$turns 32 "$turns" k_pair_bits.135.sm75.cubin "$kernel" "$said_safe" "$u0" "$u1" "$u2" "$u3"
    pb pair-bits-required-$kernel-$turns 1024 "$turns" k_pair_bits.required.135.sm75.cubin "$kernel" "" "$u0" "$u1" "$u2" "$u3"
    VF_SAFE=1 pb safe-pair-bits-required-$kernel-$turns 1024 "$turns" k_pair_bits.required.135.sm75.cubin "$kernel" "$said_safe" \
        "$u0" "$u1" "$u2" "$u3"
done <<'WORDS'
plain 1 333c0907 3c09f41b 841b5a4b 00000000 3c70c519 e56cf81a 6bf993f0 00000000
plain 37 a0a4f340 1a3b77a7 abcbaaea 00000000 828556d8 faaaff15 bc9006aa 00000000
trans 1 cd2dd65b 6176d54a 2b0f0c1e 00000000 290a56f3 8f67fe00 5313ebd2 00000000
trans 37 76284c2c ca585eaf 54451aa2 00000000 5f962557 db039578 2a22b8c8 00000000
WORDS
VF_SAFE=true pb safe-not-one-pair-bits-plain-37 32 37 k_pair_bits.135.sm75.cubin plain "VF_SAFE='true' is not 1: safe mode stays off" \
    a0a4f340 1a3b77a7 abcbaaea 00000000
# Matrix loads into registers that stores just before them still read (k_stored.cu, plain and
# transposed), on 640 blocks that keep the memory pipeline full: what stands for a load writes a
# destination with arithmetic only after waiting for its shuffles. Until 2026-11-23 it wrote the
# lanes of its rows there first, and the stores stored lanes (on a V100, in two runs: 625,408 and
# 616,576 words wrong of 62,914,560 in t_stored_plain, 335,584 and 325,152 in t_stored_trans).
[ -x "$out/bin/ldsmrun" ] && [ "$out/bin/ldsmrun" -nt "$G/ldsmrun.c" ] || gcc -O2 -Wall -o "$out/bin/ldsmrun" "$G/ldsmrun.c" -ldl || exit 2
log=$out/logs/stored-machine-code.log
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/ldsmrun" "$G/k_stored.135.sm75.cubin" >"$log" 2>&1
check stored-machine-code "machine code for sm_75 translated \(2 functions\)" "^t_stored_plain: .* 0 wrong$" "^t_stored_trans: .* 0 wrong$" "^2 kernels, 0 wrong$"
# Machine code alone: translated when it is loaded. Its seven kernels run; two have instructions added.
translated=("machine code for sm_75 translated \(7 functions\)" "${all7[@]}"
    "emulated in warp_sum: 1 instructions added in translated machine code"
    "emulated in block_sum: 3 instructions added in translated machine code")
by_default drv cubin-sm75 "$G/k_basic.135.sm75.cubin" "cubin without sm_70 code: machine code" "${translated[@]}" "machine code of saxpy: for 7.0"
# A program that is told a capability and asks which architecture the machine code of its kernel
# is for is answered that capability, not the 7.0 of the translation (PyTorch's backward attention
# asks, and stops when the answer is below the variant it chose).
VF_CAPABILITY=7.5 drv cubin-sm75-told-7.5 "$G/k_basic.135.sm75.cubin" "machine code of saxpy: for 7.5" "saxpy +ok"
# The same cubin with the first instruction of one kernel overwritten by a word that is no
# instruction (16 bytes of 0xff at 0x2980): that kernel is in a function the translator refuses,
# and is refused at the lookup, with the reason; the six others run.
drv cubin-sm75-damaged "$G/k_basic.135.sm75.damaged.cubin" "machine code for sm_75 translated \(7 functions\)" "saxpy +ok" "mathk +ok" "block_sum +ok" "histo +ok" "half_dot +ok" "dbl +ok" \
    "kernel warp_sum cannot run on this GPU and is not given to the program: its machine code for sm_75 is not translated \(the instruction ffffffffffffffffffffffffffffffff at 0x0 is not one of the sm_75 description\)"
# Both entry points belong to the same stripped Ampere image. Only the supported one may be launched.
DRVRUN_KERNELS=memory_check drv cubin-sm80-strided-half "$G/translate/k_ampere.sm80.cubin" \
    "machine code for sm_80 translated" "memory_check +ok" "lookup unsupported_atomic -> 218 CUDA_ERROR_INVALID_PTX"
DRVRUN_KERNELS=guarded_copy drv cubin-sm80-predicated-copy "$G/translate/k_guarded_copy.sm80.cubin" \
    "machine code for sm_80 translated" "guarded_copy +ok"
DRVRUN_KERNELS=checked_add drv cubin-sm80-private-helper "$G/translate/k_ampere_helper.sm80.cubin" \
    "machine code for sm_80 translated" "checked_add +ok"
DRVRUN_KERNELS=histogram drv cubin-sm80-histogram "$G/translate/k_histogram.sm80.cubin" \
    "machine code for sm_80 translated" "histogram +ok"
DRVRUN_KERNELS=warp_scan drv cubin-sm80-warp-scan "$G/translate/k_warp_scan.sm80.cubin" \
    "machine code for sm_80 translated" "warp_scan +ok"
# The same code in a fat binary that also has PTX, which is what is taken unless told otherwise.
VF_TRANSLATE=first drv fatbin-elf75-first "$W/crates/vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin" "fat binary without sm_70 code: machine code" "${translated[@]}"
VF_TRANSLATE=0 drv cubin-sm75-as-is "$G/k_basic.135.sm75.cubin" "cubin with machine code for sm_75 and later only" "-> 209 CUDA_ERROR_NO_BINARY_FOR_GPU"
# The translator's own test kernels (tests/kernels/translate), one kind of rewriting each: every
# source compiled for sm_70 by CUDA 12.9.2 and for sm_75 by CUDA 13.5; the sm_75 cubin goes through
# the proxy, and each kernel must give what the sm_70 build gives on the same inputs.
mkdir -p "$out/translate"
[ -x "$out/bin/diffrun" ] && [ "$out/bin/diffrun" -nt "$G/translate/diffrun.c" ] || gcc -O2 -Wall -o "$out/bin/diffrun" "$G/translate/diffrun.c" -ldl -lm || exit 2
mc() {  # mc FILE KERNELS : FILE.cu has KERNELS kernels that the translator takes
    local built=$out/translate/$1
    if ! [ "$built.sm75.cubin" -nt "$G/translate/$1.cu" ]; then
        # (What scripts/nvcc-host.sh adds is said here: the compiles below hide what they write.)
        nvcc_host_args "$T/cuda-12.9.2/bin/nvcc" && nvcc_host_args "$T/cuda-13.5.0/bin/nvcc"
        nvcc_host "$T/cuda-12.9.2/bin/nvcc" -arch=sm_70 -O3 -cubin -o "$built.sm70.cubin" "$G/translate/$1.cu" 2>/dev/null || exit 2
        nvcc_host "$T/cuda-13.5.0/bin/nvcc" -arch=sm_75 -O3 -cubin -o "$built.sm75.cubin" "$G/translate/$1.cu" 2>/dev/null || exit 2
    fi
    log=$out/logs/translated-$1.log
    # The native build against itself first: a kernel whose result depends on the order of its
    # threads would fail here, and proves nothing below.
    if ! "$out/bin/diffrun" "$built.sm70.cubin" "$built.sm70.cubin" 2>&1 | grep -Eq "SUMMARY: $2 PASS, 0 FAIL"; then
        echo "FAIL  translated-$1 (its native build does not give the same result twice)"; failed=1; return
    fi
    LD_LIBRARY_PATH=$proxy VF_LOG_LEVEL=1 "$out/bin/diffrun" "$built.sm70.cubin" "$built.sm75.cubin" >"$log" 2>&1
    check "translated-$1" "SUMMARY: $2 PASS, 0 FAIL"
}
mc k_translate 20
mc k_calls 4
mc k_loop_only 2
mc k_simple_minmax 1
mc k_uniform 4
mc k_params 3
mc k_uniform_math 2
mc k_uimad 2
mc k_switch 12
mc k_memory 7
mc k_convert 2
mc k_guards 3
# The memory kernels again with another, equivalent schedule (waits_moved.py: where a load is
# followed by a ULDC and then by its first reader, the wait is on the ULDC instead of the reader;
# 6 waits of k_memory.sm75.cubin as CUDA 13.5 compiled it). Whatever an instruction becomes, its
# wait must be kept: until 2026-11-17 the ULDC lost it and the two atomic kernels gave wrong sums
# here.
log=$out/logs/translated-waits-moved.log
LD_LIBRARY_PATH=$proxy VF_LOG_LEVEL=1 "$out/bin/diffrun" "$G/translate/k_memory.sm70.cubin" "$G/translate/k_memory.waits-moved.sm75.cubin" >"$log" 2>&1
check translated-waits-moved "SUMMARY: 7 PASS, 0 FAIL"
# The guarded kernels with every uniform register kept in the thread's local memory (VF_BORROW=8:
# what a kernel with no register to spare gets, here for all three), and with their guarded sums
# and logical operations giving a predicate into the one that guards them (guard_written.py: 28
# instructions of k_guards.sm75.cubin as CUDA 13.5 compiled it; the program computes the same).
# What a translation adds after an instruction must not go by the guard of that instruction: until
# 2026-11-20 the store of the uniform register did, and k_guarded_count and k_guarded_state gave
# wrong results here.
log=$out/logs/translated-guard-written.log
VF_BORROW=8 LD_LIBRARY_PATH=$proxy VF_LOG_LEVEL=1 "$out/bin/diffrun" "$G/translate/k_guards.sm70.cubin" "$G/translate/k_guards.guard-written.sm75.cubin" >"$log" 2>&1
check translated-guard-written "SUMMARY: 3 PASS, 0 FAIL" "3 kernels with emulated instructions were given to the program"


for lazy in 0 1; do
    log=$out/logs/indirect-lookup-$lazy.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LAZY=$lazy python3 "$W/tests/gpu/lookup.py" \
        "$W/crates/vf-ptx/tests/data/indirect-calls.ptx" dispatch=218 bounded=0 unrelated=0 \
        different_signature=0 different_return=0 transitive_dispatch=218 >"$log" 2>&1
    check "indirect-lookup-$lazy" "^all lookup results match$"
done
# The cubin that the layer's nvJitLink makes from PTX in which one kernel cannot run and one can
# names the first in a note (.note.vf.refused): NVIDIA's driver loads it and gives both kernels;
# through the layer, the one that cannot run is refused (218 and NULL), by a module and a library,
# eager and deferred. (Until 2026-11-23 the layer gave it to the program.)
log=$out/logs/jitlink-marked.log
VF_JITLINK_REAL=$T/cuda-12.9.2/lib/libnvJitLink.so.12 python3 "$W/tests/gpu/jitlink-marked.py" \
    "$proxy/libnvJitLink.so.13" "$out/bin/jitlink-marked.cubin" >"$log" 2>&1
check jitlink-marked "^marked cubin written$"
log=$out/logs/jitlink-marked-driver.log
DRVRUN_LIBCUDA=libcuda.so.1 python3 "$W/tests/gpu/lookup.py" "$out/bin/jitlink-marked.cubin" bad=0 good=0 >"$log" 2>&1
check jitlink-marked-driver "^all lookup results match$"
for lazy in 0 1; do
    log=$out/logs/jitlink-marked-$lazy.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LAZY=$lazy python3 "$W/tests/gpu/lookup.py" \
        "$out/bin/jitlink-marked.cubin" bad=218 good=0 >"$log" 2>&1
    check "jitlink-marked-$lazy" "^all lookup results match$"
done
# Lookup tables of internal linkage (k_tables.cu: `static const __device__`, `static __constant__`,
# and one in a namespace, whose name nvcc mangles): NVIDIA's driver gives each by its name in the
# PTX, before and after the kernel that reads it, by the library, by its module and by the module of
# the kernel's function, and the kernel reads what the program writes there; a name the PTX does not
# have is 500. Through the layer the same, except in a deferred library whose kernel that reads them
# was compiled alone with copies of them: the tables are then refused by name with 801
# (CUDA_ERROR_NOT_SUPPORTED, until 2026-11-23 500, which says that the module has no such name),
# and given when the program asks for them first, or with VF_LAZY=0.
tables() {  # tables NAME LIBCUDA IMAGE INTERFACE STEP...
    local name=$1 libcuda=$2 image=$3 interface=$4; shift 4
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$libcuda VF_LOG_LEVEL=2 python3 "$W/tests/gpu/tables.py" "$G/$image" "$interface" "$@" >"$log" 2>&1
}
mangled=_ZN40_INTERNAL_61e6d877_11_k_tables_cu_reader2ns6ntableE
read=run=166,288,410,532 written=run=1155,1267,1379,1491
given=("get:table=0" "get:ctable=0" "get:$mangled=0" "get:ntable=500")
in_modules=("mod:table=0" "fn:table=0" "mod:$mangled=0")
tables tables-driver libcuda.so.1 k_tables.129.sm70.ptx library $read "${given[@]}" "${in_modules[@]}" set:table $written
check tables-driver "^all results match$"
tables tables-driver-first libcuda.so.1 k_tables.129.sm70.ptx library "${given[@]}" "${in_modules[@]}" set:table $written
check tables-driver-first "^all results match$"
tables tables-module "$proxy/libcuda.so.1" k_tables.135.sm75.ptx module $read "${given[@]}" set:table $written
check tables-module "^all results match$"
tables tables-asked-first "$proxy/libcuda.so.1" k_tables.135.sm75.ptx library "${given[@]}" "${in_modules[@]}" set:table $written
check tables-asked-first "^all results match$" "lazy: module with 2 entries deferred"
VF_LAZY=0 tables tables-lazy-off "$proxy/libcuda.so.1" k_tables.135.sm75.ptx library $read "${given[@]}" "${in_modules[@]}" set:table $written
check tables-lazy-off "^all results match$"
tables tables-copied "$proxy/libcuda.so.1" k_tables.135.sm75.ptx library $read \
    get:table=801 get:ctable=801 "get:$mangled=801" get:ntable=500 mod:table=801 fn:table=801 "mod:$mangled=801"
check tables-copied "^all results match$" "lazy: compiling sub-module for kernel reader" \
    "the variable table is not given to the program \(error 801\)" "the variable $mangled is not given to the program \(error 801\)"

rt runtime-12.9 rt_probe.129.sm70 "$T/cuda-12.9.2/lib" "own kernel: 0/65536 wrong" "sgemm check: 0 wrong" "half gemm check: 0 wrong"
by_default rt runtime-13.5-ptx rt_probe.135.sm75ptx "$T/cuda-13.5.0/lib" "runtime 13050, driver API 13000" "rt_probe.135.sm75ptx is told compute capability 7.5: its code starts there" "device 0: .*sm_75" "PTX for sm_75 retargeted" "own kernel: 0/65536 wrong" "machine code for sm_75 translated" "sgemm check: 0 wrong" "cublasCreate -> 0 " "cublasSgemm -> 0 " "cublasGemmEx\(half\) -> 0 " "half gemm check: 0 wrong"
rt runtime-13.5-cublas-forwarded rt_probe.135.sm75ptx "$W/dist/forward:$T/cuda-12.9.2/lib" "device 0: .*sm_75" "own kernel: 0/65536 wrong" "libcublas.so.12[0-9.]* is told the real compute capability" "cuBLAS version 120902" "sgemm check: 0 wrong" "half gemm check: 0 wrong"
VF_CAPABILITY=12.0 rt rdc-cross-object rdc.135.sm120a "$T/cuda-13.5.0/lib" "compiling sub-module for kernel set_five" "kernel using another CUDA object: 42" "cross-object cluster kernel refused" "unresolved indirect kernel refused"
VF_CAPABILITY=12.0 VF_LAZY=0 rt rdc-cross-object-eager rdc.135.sm120a "$T/cuda-13.5.0/lib" "kernel using another CUDA object: 42" "cross-object cluster kernel refused" "unresolved indirect kernel refused"
VF_CAPABILITY=7.0 rt capability-forced rt_probe.135.sm75ptx "$T/cuda-13.5.0/lib" "is told compute capability 7.0, as VF_CAPABILITY asks" "device 0: .*sm_70" "own kernel: 0/65536 wrong" "cublasCreate -> 8 "
VF_STATS=1 rt stats rt_probe.129.sm70 "$T/cuda-12.9.2/lib" "half gemm check: 0 wrong" "calls: .*cuLaunchKernel=[0-9]+"
VF_ADAPT=0 rt adapt-off rt_probe.135.sm75ptx "$T/cuda-13.5.0/lib" "cudaGetLastError\(\) +-> 209"
VF_LAZY=0 rt lazy-off rt_probe.135.sm75ptx "$T/cuda-13.5.0/lib" "own kernel: 0/65536 wrong" "lazy off,"

# CUB's device-wide algorithms (tests/kernels/cub_sort.cu: a radix sort of pairs, a sum, an
# inclusive scan) choose among the architectures that the program was built for by the PTX version
# of a kernel of the program (cudaFuncGetAttributes), and answer "invalid device function" when it
# is none of them. A program whose code is PTX for sm_75 only (as Ollama's cuda_v13 runner) is
# answered 7.5, as a Turing or later GPU answers for PTX for sm_75, and its machine code 7.5 (the
# capability told); told 8.0 (VF_CAPABILITY_ALL=1, its CUDA runtime a shared library), its machine
# code 8.0 and its PTX still 7.5. Its sm_75 machine code, translated, is answered 7.5 for both; a
# program with code for sm_70 is answered the truth, 7.0. (Until 2026-11-24 the PTX of the first was
# said to be for sm_70, and its algorithms answered "invalid device function".)
cub() {  # cub NAME TOOLKIT NVCC-FLAGS...
    local program=$out/bin/cub_sort.$1 tk=$2
    shift 2
    [ -x "$program" ] && [ "$program" -nt "$G/cub_sort.cu" ] ||
        nvcc_host "$T/cuda-$tk/bin/nvcc" -O2 "$@" "$G/cub_sort.cu" -o "$program" || exit 2
}
cub 135.ptx75 13.5.0 -gencode arch=compute_75,code=compute_75
cub 135.ptx75-shared 13.5.0 --cudart=shared -gencode arch=compute_75,code=compute_75
cub 135.sm75 13.5.0 -gencode arch=compute_75,code=sm_75
cub 129.sm70 12.9.2 -arch=sm_70 -Wno-deprecated-gpu-targets
sorted=("SortPairs: no error, right" "Sum: no error, right" "InclusiveSum: no error, right" "ALL CHECKS PASSED")
by_default rt cub-ptx75 cub_sort.135.ptx75 "$T/cuda-13.5.0/lib" "ptxVersion 75, binaryVersion 75" \
    "PTX for sm_75 retargeted" "${sorted[@]}"
VF_CAPABILITY=8.0 rt cub-ptx75-told-8.0 cub_sort.135.ptx75 "$T/cuda-13.5.0/lib" "ptxVersion 75, binaryVersion 75" \
    "is told compute capability 7.5, not the 8.0 that VF_CAPABILITY asks" "${sorted[@]}"
VF_CAPABILITY=8.0 VF_CAPABILITY_ALL=1 rt cub-ptx75-all-8.0 cub_sort.135.ptx75-shared "$T/cuda-13.5.0/lib" \
    "ptxVersion 75, binaryVersion 80" "${sorted[@]}"
by_default rt cub-sm75 cub_sort.135.sm75 "$T/cuda-13.5.0/lib" "ptxVersion 75, binaryVersion 75" \
    "machine code for sm_75 translated" "${sorted[@]}"
rt cub-sm70 cub_sort.129.sm70 "$T/cuda-12.9.2/lib" "ptxVersion 70, binaryVersion 70" "${sorted[@]}"

# A fat binary of the program with a kernel that the layer refuses beside two that it runs, under
# the runtimes of CUDA 13.5 and 12.9 and both loading modes. The CUDA 13 runtime takes each kernel
# by itself (cuLibraryGetKernel) and goes on without the refused one; so does the CUDA 12 runtime
# under lazy loading, which takes them from the module of the library. Under eager loading
# (CUDA_MODULE_LOADING=EAGER) the CUDA 12 runtime takes every kernel of every image at its start
# and gives up at the first lookup that fails: it is given no module for an image with a refused
# kernel, as NVIDIA's driver gives none for an image without code for the GPU, and starts without
# that image's kernels (until 2026-11-23 it did not start: cudaMalloc -> cudaErrorInvalidPtx).
for v in 12.9.2:129 13.5.0:135; do
    program=$out/bin/rt_refuse.${v#*:}.ptx90
    [ -x "$program" ] && [ "$program" -nt "$G/rt_refuse.cu" ] || nvcc_host "$T/cuda-${v%%:*}/bin/nvcc" -O2 \
        -gencode arch=compute_90,code=compute_90 "$G/rt_refuse.cu" -o "$program" || exit 2
done
ran=("good kernels -> cudaSuccess, value 43" "bad launch -> cudaErrorInvalidPtx" "good after bad -> cudaSuccess, value 42")
CUDA_MODULE_LOADING=LAZY rt rt-refuse-13.5-lazy rt_refuse.135.ptx90 "$T/cuda-13.5.0/lib" "${ran[@]}"
CUDA_MODULE_LOADING=EAGER rt rt-refuse-13.5-eager rt_refuse.135.ptx90 "$T/cuda-13.5.0/lib" "${ran[@]}"
CUDA_MODULE_LOADING=LAZY rt rt-refuse-12.9-lazy rt_refuse.129.ptx90 "$T/cuda-12.9.2/lib" "${ran[@]}"
CUDA_MODULE_LOADING=EAGER rt rt-refuse-12.9-eager rt_refuse.129.ptx90 "$T/cuda-12.9.2/lib" "cudaMalloc -> cudaSuccess" \
    "cuLibraryGetModule: 1 kernel of this library cannot run on this GPU .* no module is given for it" \
    "bad launch -> cudaError[A-Za-z]+$"
# cuBLAS 12.9 under eager loading, called by a program of the CUDA 12.9 runtime and through the
# forwarder by one of CUDA 13.5. Its images without code for the V100 (and cuBLASLt's) are given
# no module: those with a refused kernel, and the others too, since cuBLAS has code for the V100 of
# its own and chooses it (the module would have every kernel of them compiled at cublasCreate, for
# minutes each the first time). Until 2026-11-23 cublasCreate failed under eager loading.
eager=("cublasCreate -> 0 " "cuBLAS version 120902" "sgemm check: 0 wrong" "half gemm check: 0 wrong"
    "cuLibraryGetModule: under eager loading no module is given to a caller with code for this GPU of its own")
CUDA_MODULE_LOADING=EAGER rt runtime-12.9-eager rt_probe.129.sm70 "$T/cuda-12.9.2/lib" "${eager[@]}"
CUDA_MODULE_LOADING=EAGER rt runtime-13.5-cublas-forwarded-eager rt_probe.135.sm75ptx "$W/dist/forward:$T/cuda-12.9.2/lib" \
    "own kernel: 0/65536 wrong" "${eager[@]}"

# Kernels that cannot run on the GPU are refused when the program asks for them and the others of
# their module work, through both interfaces; a kernel with emulated instructions is announced. At
# the default log level: this is what the user sees.
[ -x "$out/bin/test_refuse" ] && [ "$out/bin/test_refuse" -nt "$G/test_refuse.c" ] || gcc -O2 -Wall -o "$out/bin/test_refuse" "$G/test_refuse.c" -ldl -lm || exit 2
refuse() {  # refuse NAME INTERFACE EXPECTED...
    local name=$1 interface=$2; shift 2
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_refuse" "$interface" >"$log" 2>&1
    check "$name" "$@"
}
told=("ALL CHECKS PASSED"
    "kernel refused_by_compiler cannot run on this GPU and is not given to the program: .*Unknown symbol '%never_declared'"
    "kernel refused_by_layer cannot run on this GPU and is not given to the program: barrier.cluster.arrive: no Volta sequence"
    "kernel caller_of_bad cannot run on this GPU .*Unknown symbol '%never_declared' \(in bad_helper, which it calls\)"
    "warning: this GPU lacks instructions that kernels of the program were compiled with"
    "emulated in soft: 1 tanh"
    "1 kernel with emulated instructions was given to the program: 1 tanh")
refuse refuse-library library "${told[@]}"
refuse refuse-module module "${told[@]}"
VF_LAZY=0 refuse refuse-library-lazy-off library "${told[@]}"
# Through the driver's linker: the image that the link makes refuses the kernel too, and announces
# the emulated one (until 0.1.2 both were given to the program, the refused one as a trap).
refuse refuse-link link "ALL CHECKS PASSED" \
    "kernel refused_by_layer cannot run on this GPU and is not given to the program: barrier.cluster.arrive: no Volta sequence" \
    "emulated in soft: 1 tanh" "1 kernel with emulated instructions was given to the program: 1 tanh"
# Files given to the driver's linker (cuLinkAddFile, cuLinkAddFile_v2) are adapted as the same bytes
# in memory are, and the image that the link makes runs: PTX for sm_75; relocatable sm_75 machine
# code (built here: the driver's linker takes no other), of which the translator refuses three
# kernels (a relocation in their code), refused at the lookup; a fat binary of sm_75 machine code
# and PTX. NVIDIA's driver alone refuses the first (222, the PTX's version) and the second (209),
# and links the third into an image without these kernels. (Until 2026-11-26 the layer gave the
# driver such files as they are.)
rdc_cubin=$out/bin/k_basic.135.sm75.rdc.cubin
[ "$rdc_cubin" -nt "$G/k_basic.cu" ] || nvcc_host "$T/cuda-13.5.0/bin/nvcc" -arch=sm_75 -O3 -rdc=true -cubin \
    -o "$rdc_cubin" "$G/k_basic.cu" || exit 2
DRVRUN_LINK=cuLinkAddFile drv link-file-ptx "$G/k_basic.135.sm75.ptx" \
    "cuLinkAddFile of .*: PTX text without sm_70 code: PTX for sm_75 retargeted" "link: complete -> 0 " "${all7[@]}"
DRVRUN_LINK=cuLinkAddFile_v2 drv link-file-cubin "$rdc_cubin" \
    "cuLinkAddFile of .*: cubin without sm_70 code: machine code for sm_75 translated \(10 functions\)" \
    "link: complete -> 0 " "saxpy +ok" "warp_sum +ok" "histo +ok" "half_dot +ok" \
    "kernel mathk cannot run on this GPU and is not given to the program: its machine code for sm_75 is not translated" \
    "kernel block_sum cannot run on this GPU and is not given to the program: its machine code for sm_75 is not translated" \
    "kernel dbl cannot run on this GPU and is not given to the program: its machine code for sm_75 is not translated"
DRVRUN_LINK=cuLinkAddFile drv link-file-fatbin "$W/crates/vf-fatbin/tests/data/elf75-ptx75-zstd.fatbin" \
    "cuLinkAddFile of .*: fat binary without sm_70 code: PTX for sm_75 retargeted \(1 module\)" "link: complete -> 0 " "${all7[@]}"

# A fat binary whose machine code for sm_80 has a kernel that its code for sm_75 has not
# (tests/kernels/k_sections.cu; vLLM 0.34's library has its bf16 norm so). The layer gives it its
# sm_75 code, whatever it is told. Told 8.0, a program asks for the kernel of the sm_80 code, which
# the driver does not find in the code given (500): the layer then gives that kernel from the sm_80
# code, translated, in a module of its own, and says so once, whatever the interface; the kernel
# runs and stores what the sm_80 code stores (until 2026-11-25 the lookup failed, with a line since
# 2026-11-24; `VF_TOLD_CODE=0` is that again). Told 7.5 (by default) the lookup fails and nothing is
# said: a GPU of 7.5 has no such kernel either. With VF_TRANSLATE=first the whole sm_80 code is
# translated, and has the kernel.
[ -x "$out/bin/test_sections" ] && [ "$out/bin/test_sections" -nt "$G/test_sections.c" ] || gcc -O2 -Wall -o "$out/bin/test_sections" "$G/test_sections.c" -ldl || exit 2
[ -x "$out/bin/test_told" ] && [ "$out/bin/test_told" -nt "$G/test_told.c" ] || gcc -O2 -Wall -o "$out/bin/test_told" "$G/test_told.c" -ldl -lm || exit 2
sections() {  # sections NAME PROGRAM INTERFACE EXPECTED...
    local name=$1 program=$2 interface=$3 image=k_sections; shift 3
    [ "$program" = test_told ] && image=k_told
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=1 "$out/bin/$program" "$G/$image.135.sm75-sm80.fatbin" \
        "$interface" >"$log" 2>&1
    echo "lines that say a kernel is not in the code given: $(grep -c 'is not in the code that the layer gives' "$log")" >>"$log"
    echo "lines that say a kernel is given from the sm_80 code: $(grep -c 'is given from the machine code for sm_80 of its image' "$log")" >>"$log"
    echo "of them for k_from_80: $(grep -c 'kernel k_from_80 is given from the machine code' "$log")" >>"$log"
    echo "lookups that failed otherwise than 500: $(grep -E ': lookup [0-9]+' "$log" | grep -Evc ': lookup (0|500)( |$)')" >>"$log"
    check "$name" "$@"
}
from_80=("kernel k_from_80 is given from the machine code for sm_80 of its image \("
    "[0-9]+ bytes, XXH64 [0-9a-f]{16}\), translated, in a (module|library) of its own: the code that the layer gives for the image, its machine code for sm_75, has no kernel of that name, and a GPU of the capability told \(8\.0\) loads the code for sm_80\. Its translation is not independently validated")
given=("k_both: ran the code for sm_75" "k_before_80: ran the code for sm_75" "k_from_80: lookup 0$"
    "k_from_80: ran the code for sm_80" "k_nowhere: lookup 500$" "lines that say a kernel is not in the code given: 0$")
once=("lines that say a kernel is given from the sm_80 code: 1$" "of them for k_from_80: 1$")
VF_CAPABILITY=8.0 sections sections-told-8.0-module test_sections module "${given[@]}" "${from_80[0]}${from_80[1]}" "${once[@]}" "^8 lookups"
VF_CAPABILITY=8.0 sections sections-told-8.0-file test_sections file "${given[@]}" \
    "${from_80[0]}[^ ]*/k_sections\.135\.sm75-sm80\.fatbin: ${from_80[1]}" "${once[@]}" "^8 lookups"
VF_CAPABILITY=8.0 sections sections-told-8.0-library test_sections library "${given[@]}" "${from_80[0]}${from_80[1]}" \
    "k_from_80: lookup 0 \(in the library's module\)" "${once[@]}" "^12 lookups"
by_default sections sections-told-7.5 test_sections library "k_both: ran the code for sm_75" \
    "k_before_80: ran the code for sm_75" "k_from_80: lookup 500$" "k_nowhere: lookup 500$" \
    "lines that say a kernel is not in the code given: 0$" "lines that say a kernel is given from the sm_80 code: 0$" "^12 lookups"
VF_CAPABILITY=8.0 VF_TRANSLATE=first sections sections-told-8.0-first test_sections library "k_both: ran the code for sm_80" \
    "k_from_80: ran the code for sm_80" "k_before_80: lookup 500$" "lines that say a kernel is not in the code given: 0$" \
    "lines that say a kernel is given from the sm_80 code: 0$" "^12 lookups"
# The same with the switch off: the lookup fails as the driver answers, and the line says which
# code has the kernel (what 2026-11-24 did).
VF_CAPABILITY=8.0 VF_TOLD_CODE=0 sections sections-told-8.0-off test_sections library "k_both: ran the code for sm_75" \
    "k_from_80: lookup 500$" "k_from_80: lookup 500 \(in the library's module\)" \
    "kernel k_from_80 is not in the code that the layer gives for its image \([0-9]+ bytes, XXH64 [0-9a-f]{16}\), its machine code for sm_75 translated: it is only in the image's machine code for sm_80, which a GPU of the capability told \(8\.0\) loads\. The lookup fails as the driver answers for a name that it does not have \(error 500\): the layer gives an image that has machine code for sm_75 and sm_80 its code for sm_75, whatever the capability told, as its translation of machine code for sm_80 covers fewer instructions and is not independently validated \(LIMITS\.md, \"The capability told with VF_CAPABILITY\"\)$" \
    "lines that say a kernel is not in the code given: 1$" "lines that say a kernel is given from the sm_80 code: 0$"

# What such a kernel computes, and which are not given (tests/kernels/k_told.cu, test_told.c). Told
# 8.0: the sums of pairs of bf16 and the pairs of bf16 nearest to f32 that only the sm_80 code has
# (two instructions of sm_80 that Volta has not, rewritten) are the host's, every one, also through
# the library's module and after the image is loaded again; the quotients of a division whose slow
# path is a function inside the kernel are the host's; the kernel's library is the one the program
# loaded. Not given, their lookups failing as before with a line that says why for each: a kernel
# that reads a variable of the image, which the program sets through the module it holds, and one
# that reads a table of the image (both through the bank of the image's addresses), one that adds
# a `__constant__` variable (read from a constant bank of the module, with no relocation: the
# program sets it to 20 where the image has 1.5, and a module of the layer's own would add 1.5),
# one that stores the address of that variable (which the driver puts in a word of the kernel's
# own bank 0: every instruction of it is translated, and it has no relocation),
# and one that the layer does not translate (a sum of pairs of f16). In safe mode the same (the
# kernel has no other code here). Told 7.5 none is given, and nothing is said.
bf16=("k_bf16_sum: lookup 0$" "k_bf16_sum: 16384 sums, 0 differ from the sum rounded once to bf16"
    "k_bf16_sum: 16384 conversions, 0 differ from the nearest bf16"
    "k_bf16_sum, loaded again: 16384 sums, 0 differ" "k_bf16_sum, loaded again: 16384 conversions, 0 differ"
    "k_divides: lookup 0$" "k_divides: 8192 quotients, 0 differ from the host's"
    "k_both: ran the code for sm_75" "k_before_80: ran the code for sm_75"
    "kernel k_bf16_sum is given from the machine code for sm_80 of its image" "kernel k_divides is given from the machine code for sm_80 of its image"
    "lines that say a kernel is given from the sm_80 code: 2$"
    "k_reads_data: lookup 500$" "k_not_translated: lookup 500$" "k_nowhere: lookup 500$"
    "k_reads_table: lookup 500$" "k_adds_const: lookup 500$" "k_const_address: lookup 500$"
    "kernel k_reads_data is not in the code that the layer gives for its image .*\(LIMITS\.md, \"The capability told with VF_CAPABILITY\"\)\. The kernel alone is not given from the code for sm_80 either: it can reach data of its image, of which a (module|library) of its own would have another copy than the one that the program reads and writes: constant bank 4, which holds addresses of the image's variables"
    "kernel k_reads_table is not in the code .* either: it can reach data of its image, .*: constant bank 4, which holds addresses of the image's variables"
    "kernel k_adds_const is not in the code .* either: it can reach data of its image, .*: constant bank 3, which is not this kernel's alone"
    "kernel k_const_address is not in the code .* either: it can reach data of its image, .*: the word of constant bank 0 at 0x50, which is not known to be the kernel's own"
    "kernel k_not_translated is not in the code that the layer gives for its image .*\. The kernel alone is not given from the code for sm_80 either: its machine code for sm_80 is not translated \(unsupported sm_80 instruction [0-9a-f]{32} at 0x[0-9a-f]+\)$"
    "lines that say a kernel is not in the code given: 5$" "lookups that failed otherwise than 500: 0$"
    "^unload: 0$" "the context goes on: 7$")
VF_CAPABILITY=8.0 sections told-8.0-module test_told module "${bf16[@]}" "told_scale set to 3: 0" "told_bias set to 10, 20, 30, 40: 0" "^20 lookups"
VF_CAPABILITY=8.0 sections told-8.0-file test_told file "${bf16[@]}" "^20 lookups"
VF_CAPABILITY=8.0 sections told-8.0-library test_told library "${bf16[@]}" "k_bf16_sum: its library is the one loaded" \
    "k_divides: its library is the one loaded" "k_bf16_sum: lookup 0 \(in the library's module\)" \
    "k_bf16_sum \(in the library's module\): 16384 sums, 0 differ" "k_reads_data: lookup 500 \(in the library's module\)" "k_adds_const: lookup 500 \(in the library's module\)" "^30 lookups"
VF_CAPABILITY=8.0 VF_SAFE=1 sections told-8.0-safe test_told library "${bf16[@]}" "^30 lookups"
by_default sections told-7.5 test_told library "k_both: ran the code for sm_75" "k_before_80: ran the code for sm_75" \
    "k_bf16_sum: lookup 500$" "k_divides: lookup 500$" "k_reads_data: lookup 500$" "k_bf16_sum, loaded again: lookup 500$" \
    "lines that say a kernel is not in the code given: 0$" "lines that say a kernel is given from the sm_80 code: 0$" \
    "lookups that failed otherwise than 500: 0$" "^30 lookups"
VF_CAPABILITY=8.0 VF_TOLD_CODE=0 sections told-8.0-off test_told library "k_bf16_sum: lookup 500$" "k_divides: lookup 500$" \
    "lines that say a kernel is not in the code given: 7$" "lines that say a kernel is given from the sm_80 code: 0$" "^30 lookups"

# An image that the driver's linker makes from relocatable code (two units built with -rdc=true,
# tests/kernels/k_told_rdc.cu and k_told_rdc2.cu), one of whose kernels only the sm_80 code of
# its unit has: the module is of the linked image, the lookup fails with 500 as the driver
# answers, and the layer loads nothing for it and says nothing of the code for sm_80. (The kernel
# that both codes have is refused today: a call that a relocation resolves is not translated.)
[ -x "$out/bin/test_told_rdc" ] && [ "$out/bin/test_told_rdc" -nt "$G/test_told_rdc.c" ] || gcc -O2 -Wall -o "$out/bin/test_told_rdc" "$G/test_told_rdc.c" -ldl || exit 2
log=$out/logs/told-8.0-rdc.log
VF_CAPABILITY=8.0 DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=1 "$out/bin/test_told_rdc" \
    "$G/k_told_rdc.135.sm75-sm80.fatbin" "$G/k_told_rdc2.135.sm75-sm80.fatbin" >"$log" 2>&1
echo "lines about the code for sm_80: $(grep -Ec 'is given from the machine code for sm_80|is not in the code that the layer gives' "$log")" >>"$log"
check told-8.0-rdc "link: complete 0$" "^load: 0$" "k_rdc_both: lookup (0|218)$" "k_rdc_calls_out: lookup 500, handle null$" \
    "lines about the code for sm_80: 0$" "^unload: 0$" "the context goes on: 0$"

# A refused kernel of which the program holds a handle that the layer did not give (taken here from
# NVIDIA's driver itself, as cuFFT takes the functions of its modules from a private table) is not
# launched: every launch function, a kernel node of a graph and the questions about it return 218,
# the context still runs a kernel afterwards, and the GPU did not stop (no line more with "Xid" in
# the kernel's log, where it can be read). Until 2026-11-23 the first launch trapped (Xid 43).
[ -x "$out/bin/test_guard" ] && [ "$out/bin/test_guard" -nt "$G/test_guard.c" ] || gcc -O2 -Wall -o "$out/bin/test_guard" "$G/test_guard.c" -ldl || exit 2
# The kernel's log is read from dmesg, else from `journalctl -k`, else from `sudo -n dmesg`
# (tests/gpu/kernel-log.sh), whichever can be read here first.
. "$W/tests/gpu/kernel-log.sh"
xids() { kernel_log | grep -c Xid; }
xid_unchecked=  # a guard or nocode case whose check of the kernel's log could not be made: said at the end
guard() {  # guard NAME INTERFACE
    local name=$1 interface=$2 before after
    log=$out/logs/$name.log
    before=$(xids)
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_guard" "$interface" >"$log" 2>&1
    after=$(xids)
    if [ -n "$kernel_log_source" ]; then
        echo "lines with Xid in the kernel's log ($kernel_log_source): $before before, $after after" >>"$log"
        [ "$before" = "$after" ] && echo "no Xid line more" >>"$log"
    else
        echo "no Xid line more (the kernel's log cannot be read here)" >>"$log"
        xid_unchecked=1
    fi
    check "$name" "ALL CHECKS PASSED" "no Xid line more" \
        "kernel refused_by_layer cannot run on this GPU: barrier.cluster.arrive: no Volta sequence.*\. The program holds a handle to it, which a lookup by name refuses"
}
guard guard-module module
guard guard-library library
VF_LAZY=0 guard guard-library-lazy-off library
# The function of the refused kernel in its context is refused there, also after it was asked
# about while no context, or another, was current (only questions: nothing that traps is launched).
VF_LAZY=0 guard guard-contexts contexts

# A kernel without code (its body in the image is a trap and nothing else: what a kernel built for
# other architectures only is, in the code for the others) is given at the lookup and answered
# about, as NVIDIA's driver does, but a launch of it is refused with 218 (said once, with its
# image), a kernel node of it is not made, a capture in progress goes on, the context survives and
# the kernels that are code run, one that traps behind a test and an empty one among them; no Xid
# line more. From PTX (retargeted) and from machine code for sm_75 (translated). Until 2026-11-24
# the launch trapped and the GPU stopped the context (Xid 43): Strata's CUDA 13 engine 0.1.66
# launched such a kernel (issue #3).
[ -x "$out/bin/test_nocode" ] && [ "$out/bin/test_nocode" -nt "$G/test_nocode.c" ] || gcc -O2 -Wall -o "$out/bin/test_nocode" "$G/test_nocode.c" -ldl || exit 2
nocode() {  # nocode NAME IMAGE INTERFACE
    local name=$1 image=$2 interface=$3 before after
    log=$out/logs/$name.log
    before=$(xids)
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_nocode" "$image" "$interface" >"$log" 2>&1
    after=$(xids)
    if [ -n "$kernel_log_source" ]; then
        echo "lines with Xid in the kernel's log ($kernel_log_source): $before before, $after after" >>"$log"
        [ "$before" = "$after" ] && echo "no Xid line more" >>"$log"
    else
        echo "no Xid line more (the kernel's log cannot be read here)" >>"$log"
        xid_unchecked=1
    fi
    echo "lines that say no_code is not launched: $(grep -c 'kernel no_code has no code in its image' "$log")" >>"$log"
    check "$name" "ALL CHECKS PASSED" "no Xid line more" "lines that say no_code is not launched: 1$" \
        "kernel no_code has no code in its image \([0-9]+ bytes, XXH64 [0-9a-f]{16}\): its body is a trap and nothing else; cuLaunchKernel with it is refused \(error 218\)"
}
nocode nocode-ptx-module "$G/k_nocode.135.sm75.ptx" module
nocode nocode-ptx-library "$G/k_nocode.135.sm75.ptx" library
VF_LAZY=0 nocode nocode-ptx-library-lazy-off "$G/k_nocode.135.sm75.ptx" library
nocode nocode-cubin-module "$G/k_nocode.135.sm75.cubin" module
nocode nocode-cubin-library "$G/k_nocode.135.sm75.cubin" library

# Instructions that need a multicast address (multimem.*), which no GPU of a process of V100s can
# have, are replaced by a reported trap (tests/kernels/test_reported.c): a kernel that calls through
# a table a function that holds one, one that holds one behind a test of a flag (as NCCL's step
# counters are read), one with two under a guard, run with the guard false, and one that holds
# nothing else are given, and the first three compute right on their paths; one line says them for
# their image at the default level. A module with comments before its header loads and runs. The same call that may reach
# barrier.cluster.arrive, which a program can really execute, is still refused, and so are the
# first ones with VF_REPORTED_TRAPS=0 (until 0.1.5 they were all refused: the CUDA 13 build of NCCL
# could not run its collectives). No case reaches a trap; what the driver says of multicast is read
# without the layer and through it. A trap reached is the opt-in case at the end (VF_TEST_FAULTS=1).
[ -x "$out/bin/test_reported" ] && [ "$out/bin/test_reported" -nt "$G/test_reported.c" ] || gcc -O2 -Wall -o "$out/bin/test_reported" "$G/test_reported.c" -ldl || exit 2
reported() {  # reported NAME MODE EXPECTED...
    local name=$1 mode=$2 before after; shift 2
    log=$out/logs/$name.log
    before=$(xids)
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_reported" "$mode" >"$log" 2>&1
    after=$(xids)
    if [ -n "$kernel_log_source" ]; then
        echo "lines with Xid in the kernel's log ($kernel_log_source): $before before, $after after" >>"$log"
        [ "$before" = "$after" ] && echo "no Xid line more" >>"$log"
    else
        echo "no Xid line more (the kernel's log cannot be read here)" >>"$log"
        xid_unchecked=1
    fi
    check "$name" "ALL CHECKS PASSED" "no Xid line more" "$@"
}
# (One line for each image at the default level, which names it by the call, its size and the file
# of the caller; each kernel from VF_LOG_LEVEL=2.)
given_with=("(cuLibraryLoadData|cuModuleLoadData) of [0-9]+ bytes in [^ ]*/test_reported: 4 kernels are given with instructions that cannot run on this GPU replaced by reported traps \(at most 2 in what one kernel may reach; multimem\.ld_reduce\.acquire\.sys\.global\.min\.u32, multimem\.ld_reduce\.relaxed\.sys\.global\.add\.u32, multimem\.st\.global\.b32\): pick, polled, stored and 1 more\. Such an instruction needs a multicast address"
    "(cuLibraryLoadData|cuModuleLoadData) of [0-9]+ bytes in [^ ]*/test_reported: 1 kernel is given with instructions that cannot run on this GPU replaced by reported traps \(at most 1 in what one kernel may reach; multimem\.st\.global\.b32\): stored\."
    "kernel pick cannot run on this GPU and is not given to the program: .*barrier\.cluster\.arrive: no Volta sequence")
reported reported-library library "${given_with[@]}"
reported reported-module module "${given_with[@]}"
VF_LAZY=0 reported reported-library-lazy-off library "${given_with[@]}"
VF_REPORTED_TRAPS=0 reported reported-off refused \
    "kernel pick cannot run on this GPU and is not given to the program: .*multimem\.ld_reduce\.relaxed\.sys\.global\.add\.u32: no Volta sequence" \
    "kernel polled cannot run on this GPU and is not given to the program: multimem\.ld_reduce\.acquire\.sys\.global\.min\.u32: no Volta sequence"
if grep -Eq "replaced by (a reported trap|reported traps)" "$out/logs/reported-off.log"; then
    echo "FAIL  reported-off-silent: VF_REPORTED_TRAPS=0 gave a kernel with reported traps ($out/logs/reported-off.log)"; failed=1
else
    echo "PASS  reported-off-silent"
fi
for how in native layer; do
    log=$out/logs/multicast-$how.log
    if [ $how = native ]; then "$out/bin/test_reported" multicast >"$log" 2>&1; else
        DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_reported" multicast >"$log" 2>&1; fi
    check "multicast-$how" "^GPU 0: CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED -> 0, value 0$" "^cuMulticastCreate -> [1-9][0-9]*$"
done
# A program without a multimem instruction, whose image the layer adapts (the driver is asked then),
# hears nothing of multicast at the default level; the driver's answer is said from VF_LOG_LEVEL=2.
log=$out/logs/multicast-quiet.log
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/drvrun" "$G/k_basic.135.sm75.ptx" >"$log" 2>&1
if grep -q "multicast" "$log"; then
    echo "    a line of multicast at the default level"; echo "FAIL  multicast-quiet (output: $log)"; failed=1
else
    check multicast-quiet "${all7[@]}"
fi
log=$out/logs/multicast-said.log
DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/drvrun" "$G/k_basic.135.sm75.ptx" >"$log" 2>&1
check multicast-said "no GPU of this process can take part in multicast \([0-9]+ asked\): instructions that need a multicast address are replaced by reported traps" "${all7[@]}"

# A kernel whose usual translation takes more registers than a block of 1,024 threads can have is
# launched from a second translation that keeps to the kernel's own count (`docs/architecture.md`, "Launch"): taken
# from a module and from a library, with an attribute that the program set on its kernel, in a
# second context, after an unload; and never a kernel that reads data of its module. The
# control: with the mechanism off the same blocks are refused.
[ -x "$out/bin/fitrun" ] && [ "$out/bin/fitrun" -nt "$G/fitrun.c" ] || gcc -O2 -Wall -o "$out/bin/fitrun" "$G/fitrun.c" -ldl -lm || exit 2
fit() {  # fit NAME OPTIONS EXPECTED...
    local name=$1 options=$2; shift 2
    log=$out/logs/$name.log
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 "$out/bin/fitrun" $options "$G/k_fit_test.135.sm75.cubin" >"$log" 2>&1
    check "$name" "$@"
}
fit second-translation "" "37 checks, 0 wrong" \
    "fit: k_fit_regs: blocks of 1024 threads take more registers than there are with the usual translation" \
    "fit: k_fit_reads_global is not launched from a second translation: it can reach data of its module"
VF_FIT=0 fit second-translation-off -x "37 checks, 0 wrong"
VF_FIT_SHARED=1 fit second-translation-shared -s "37 checks, 0 wrong"
gcc -O2 -Wall -I"$T/cuda-13.5.0/include" "$G/fit_shared.c" -L"$proxy" -l:libcuda.so.1 -o "$out/bin/fit_shared" || exit 2
for mode in 0 1; do
    log="$out/logs/fit-shared-$mode.log"
    VF_FIT_SHARED="$mode" LD_LIBRARY_PATH="$proxy${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out/bin/fit_shared" >"$log" 2>&1
    check "fit-shared-$mode" "shared opt-in checks passed"
done

# Proxy cache: a second run with a warm cache must skip adaptation and analysis and give the same
# results, including the same refused kernels and the same emulation warnings.
# In a local directory, whose files belong to whoever runs: the proxy refuses a cache that does not
# (under $TMPDIR, /tmp when it is not set).
cache_dir=${TMPDIR:-/tmp}/vf-cache-test-$$
rm -rf "$cache_dir"
cache_case() {  # cache_case RUN-NAME EXPECTED...
    local name=$1; shift
    log=$out/logs/$name.log
    VF_PROXY_CACHE=$cache_dir DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=2 VF_TIMING=1 \
        "$out/bin/test_refuse" library >"$log" 2>&1
    check "$name" "$@"
}
cache_case cache-cold "${told[@]}" "timing: cuLibraryLoadData: adapt"
cache_case cache-warm "${told[@]}" "timing: cuLibraryLoadData: cache hit"
# The warm run must NOT have adapted: "adapt" must not appear in its timing lines.
if grep -q "timing: cuLibraryLoadData: adapt" "$out/logs/cache-warm.log"; then
    echo "FAIL  cache-warm: adapted despite a cache hit ($out/logs/cache-warm.log)"; failed=1
else
    echo "PASS  cache-no-adapt: warm run did not adapt"
fi
# The translation of machine code is kept too: a second run takes it from the cache and gives the
# same kernels, the refused one included, with the same messages.
tcache() {  # tcache RUN-NAME EXPECTED...
    local name=$1; shift
    log=$out/logs/$name.log
    VF_PROXY_CACHE=$cache_dir DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_LOG_LEVEL=3 VF_TIMING=1 \
        "$out/bin/drvrun" "$G/k_basic.135.sm75.damaged.cubin" >"$log" 2>&1
    check "$name" "$@"
}
kept=("machine code for sm_75 translated \(7 functions\)" "saxpy +ok" "mathk +ok" "block_sum +ok" "histo +ok" "half_dot +ok" "dbl +ok"
    "kernel warp_sum cannot run on this GPU and is not given to the program: its machine code for sm_75 is not translated \(the instruction"
    "emulated in block_sum: 3 instructions added in translated machine code")
tcache translation-cold "${kept[@]}" "timing: .*: machine code translated in"
tcache translation-warm "${kept[@]}" "timing: .*: translation from the cache"
if grep -q "machine code translated in" "$out/logs/translation-warm.log"; then
    echo "FAIL  translation-warm: translated despite the cache ($out/logs/translation-warm.log)"; failed=1
else
    echo "PASS  translation-kept: warm run did not translate"
fi
rm -rf "$cache_dir"

# A kernel that faults on purpose (tests/kernels/test_fault.c: a write through a null pointer),
# launched after kernels that run: the layer names it among the kernels launched in the context since
# its last synchronisation that succeeded, and not those before; alone under VF_FAULT_TRACE=1; as a
# node of a CUDA graph that faulted; and when only a copy, which the layer does not route, met the
# fault, at the next launch. The report at exit (VF_SUMMARY) says it too.
# EACH OF THE FOUR (AND reported-fault BELOW) STOPS THE GPU'S CONTEXT ON PURPOSE and adds lines with "Xid" to the kernel's log
# (one to three each), each in a process of its own (after a fault no other context of the process
# runs): they run only with VF_TEST_FAULTS=1 (exactly), never by default. The control, the same
# kernels without the one that faults, runs always: nothing is said, at exit either.
[ -x "$out/bin/test_fault" ] && [ "$out/bin/test_fault" -nt "$G/test_fault.c" ] || gcc -O2 -Wall -o "$out/bin/test_fault" "$G/test_fault.c" -ldl || exit 2
fault() {  # fault NAME MODE EXPECTED...
    local name=$1 mode=$2 before after; shift 2
    log=$out/logs/$name.log
    rm -f "$out/logs/$name.summary".*
    before=$(xids)
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 VF_SUMMARY=$out/logs/$name.summary "$out/bin/test_fault" "$mode" >"$log" 2>&1
    after=$(xids)
    cat "$out/logs/$name.summary".* >>"$log" 2>/dev/null
    echo "lines with Xid in the kernel's log: $before before, $after after ($(date -u +%T))" >>"$log"
    check "$name" "$@"
}
fault fault-none nofault "^NO FAULT$"
if grep -Eq "can run nothing more|on the GPU failed|faulted" "$log"; then
    echo "FAIL  fault-none-silent: a fault was said where none was ($log)"; failed=1
else
    echo "PASS  fault-none-silent"
fi
if [ "${VF_TEST_FAULTS:-0}" = 1 ]; then
    named="PTX for sm_75 retargeted by the layer; the image given to the driver: [0-9]+ bytes, XXH64 [0-9a-f]{16}; the program's image: cuModuleLoadData of [0-9]+ bytes"
    fault fault-candidates fault \
        "work on the GPU failed: (cuCtxSynchronize|cuLaunchKernel) returned 700 \(CUDA_ERROR_ILLEGAL_ADDRESS\)" \
        "[23] launches were made in this context since its last synchronisation that succeeded" \
        "^\[volta-forever\]   write_null \(cuLaunchKernel, grid 1x1x1, block 32x1x1, the null stream\): $named" \
        "VF_FAULT_TRACE=1 synchronises every launch" \
        "^\[volta-forever\] work on the GPU failed: .* returned 700 .*the candidates, the last launched first: (add_one, )?write_null, add_one$"
    VF_FAULT_TRACE=1 fault fault-exact fault \
        "the synchronisation right after the launch of write_null met the fault: the synchronisation of the stream after cuLaunchKernel \(VF_FAULT_TRACE=1\) returned 700" \
        "^launch write_null -> 700$" \
        "^\[volta-forever\] work on the GPU failed: .*; the launch whose synchronisation met it: write_null \(work not seen on its stream before it not ruled out\)$"
    fault fault-graph fault-graph \
        "work on the GPU failed: cuStreamSynchronize returned 700" \
        "^after the fault: cuStreamGetCtx -> " \
        "a CUDA graph \(cuGraphLaunch, stream 0x[0-9a-f]+\): its kernels ran without passing through the layer; the layer saw 3 kernel nodes added to it, of 2 kernels" \
        "^\[volta-forever\]     write_null \(grid 1x1x1, block 32x1x1\): $named"
    fault fault-copy fault-copy \
        "^copy -> 700$" \
        "^launch add_one -> 700$" \
        "work on the GPU failed: cuLaunchKernel returned 700" \
        "^\[volta-forever\]   write_null \(cuLaunchKernel"
    # A reported trap reached (tests/kernels/test_reported.c, mode fault): the call of pick takes the
    # function with the multimem instruction. The kernel stops with the assertion that the trap
    # makes, which the driver writes with the function and the thread; the layer's report names pick
    # with what it was given with.
    log=$out/logs/reported-fault.log
    before=$(xids)
    DRVRUN_LIBCUDA=$proxy/libcuda.so.1 "$out/bin/test_reported" fault >"$log" 2>&1
    after=$(xids)
    echo "lines with Xid in the kernel's log: $before before, $after after ($(date -u +%T))" >>"$log"
    check reported-fault "^pick through the multimem function -> 0xdead02c6$" \
        "volta-forever:1: gathered: block: \[0,0,0\], thread: \[0,0,0\] Assertion .multimem\.ld_reduce\.relaxed\.sys\.global\.add\.u32 cannot run on this GPU" \
        "work on the GPU failed: cuCtxSynchronize returned 710 \(CUDA_ERROR_ASSERT\)" \
        "pick \(cuLaunchKernel, grid 1x1x1, block 32x1x1, the null stream\): .*given with 1 instruction that cannot run on this GPU replaced by a reported trap"
fi

# The guard and nocode cases above pass their check of the kernel's log where it cannot be read
# (dmesg, the journal and sudo need privileges that the user may not have): said once, at the end,
# so that it is seen.
[ -z "$xid_unchecked" ] || echo "Xid lines could not be checked (the kernel's log could not be read: dmesg, journalctl -k and sudo -n dmesg were tried)"
exit $failed
