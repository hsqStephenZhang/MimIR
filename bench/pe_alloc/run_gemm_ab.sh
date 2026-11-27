#!/usr/bin/env bash
# run_gemm_ab.sh OUT M:N:K...: F32 GEMMs compiled by build-rel (with Alloc) and by $MIM_BASE, timed in one process.
# MALLOC_TUNED=1 sets the glibc thresholds the other benches use; by default malloc runs with its defaults.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MIM=${MIM:-$HERE/../../build-rel/bin/mim}
BASE=${MIM_BASE:?set MIM_BASE to the baseline mim}
out=$1; shift
mkdir -p "$out"
for s in "$@"; do
    IFS=: read -r m n k <<< "$s"
    name=g${m}x${n}x${k}
    for v in new base; do
        cat > "$out/$name.$v.mim" <<MIM
plugin core;
plugin tensor;
plugin math;
let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
extern fun gemm_$v (a: tensor.Tensor 2 ($m, $k) F32, b: tensor.Tensor 2 ($k, $n) F32): tensor.Tensor 2 ($m, $n) F32 =
    return (tensor.product_2d R (a, b));
MIM
    done
    "$MIM" -p opt "$out/$name.new.mim" -p ll -X ll:o="$out/$name.new.ll" -o /dev/null
    "$BASE" -p opt "$out/$name.base.mim" -p ll -X ll:o="$out/$name.base.ll" -o /dev/null
    clang++ -O3 -march=native -Wno-override-module -DM=$m -DN=$n -DK=$k "$HERE/gemm_ab_main.cpp" "$out/$name.new.ll" "$out/$name.base.ll" -o "$out/$name.exe"
    if [ "${MALLOC_TUNED:-0}" = 1 ]; then
        MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=100000000000 taskset -c ${CORE:-5} "$out/$name.exe" "$name"
    else
        taskset -c ${CORE:-5} "$out/$name.exe" "$name"
    fi
done
