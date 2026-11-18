#!/bin/bash
# gemm_levels.sh <levels> [M=1024] [K=M]: the F16 GEMM of `gemm_tiled_wg_tma.mim` with an M×M result over K, scheduled by
# the level chain <levels>, compiled for sm_90a and run end to end; prints the median of three nsys kernel averages in µs.
# The driver's reference assumes K = M, so its verdict only counts then.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
B=$(cd "$(dirname "$0")" && pwd); R=$(cd "$B/../.." && pwd)
levels=$1; M=${2:-1024}; K=${3:-$M}
O=${OUT:-$B/out}; mkdir -p $O && cd $O
n=lv_$(echo "$levels $M $K" | md5sum | cut -c1-10)
sed "s/^let levels = .*/let levels = $levels;/; s/gemm1024/gemmX/; s/(a b: tensor.Tensor 2 (1024, 1024) H)/(a: tensor.Tensor 2 ($M, $K) H, b: tensor.Tensor 2 ($K, $M) H)/; s/tensor.Tensor 2 (1024, 1024) F32/tensor.Tensor 2 ($M, $M) F32/; s/((1024, 1024), (1024, 1024, 1024))/(($M, $M), ($M, $M, $K))/" $B/gemm_tiled_wg_tma.mim > $n.mim
$R/${MIM_BUILD:-build-h200-debug}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $n.mim -p nvptx -X nvptx:sm=90a ${MIM_FLAGS} ${MIM_XOPT:+-X "nvptx:Xopt=$MIM_XOPT"} > $n.log 2>&1 || { echo "mim failed: $(grep -m1 error $n.log | cut -c1-160)"; exit 1; }
clang++ -O2 -DN=$M -DFN=gemmX $B/gemm_driver16.cpp $n.ll -lcuda -o $n.exec -Wno-override-module 2> /dev/null || exit 1
check=$(./$n.exec 1 | tail -1)
t=$(for rep in 1 2 3; do
    nsys profile -o $n --force-overwrite true ./$n.exec 5 > /dev/null 2>&1
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $n.nsys-rep 2> /dev/null | awk -F, '$1+0>0 {s+=$4} END {printf "%.2f\n", s/1e3}'
done | sort -n | sed -n 2p)
echo "M=$M K=$K [$check] $t us  $levels"
