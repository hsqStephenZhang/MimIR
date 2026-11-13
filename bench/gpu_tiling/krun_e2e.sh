#!/bin/bash
# krun_e2e.sh <name.mim> [extra mim flags...]: compiles a 16-bit gpu_tiling GEMM (N=<n> resizes it), runs it end to end, and
# sums its kernel time with nsys; for kernels that take more than their buffers, such as TMA descriptors.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_tiling
mkdir -p $B/out && cd $B/out
f=$1; shift
n=$(basename $f .mim)_${N:-1024}
sed "s/1024/${N:-1024}/g" $B/$f > $n.mim
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $n.mim -p nvptx "$@" > $n.log 2>&1 || { echo "$n: mim failed"; grep -m3 -A3 error $n.log; exit 1; }
clang++ -O2 -DN=${N:-1024} -DFN=gemm${N:-1024} $B/gemm_driver16.cpp $n.ll -lcuda -o $n.exec -Wno-override-module || exit 1
check=$(./$n.exec 1 | tail -1)
nsys profile -o $n --force-overwrite true ./$n.exec 5 > /dev/null 2>&1
kern=$(nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $n.nsys-rep 2>/dev/null | python3 -c "
import csv,sys
rows=[r for r in csv.reader(sys.stdin) if len(r)>8 and r[0][:1].isdigit()]
print('%.4f ms/call' % (sum(float(r[3]) for r in rows)/1e6))")
echo "$n [$*] $check: $kern"
