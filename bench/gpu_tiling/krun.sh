#!/bin/bash
# krun.sh <name.mim> [extra mim flags...]: compile a gpu_tiling bench and time its kernel with kbench; N=<n> resizes it,
# TY=<f16f32|bf16f32|...> times a 16-bit one with kbench16.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_tiling
mkdir -p $B/out && cd $B/out
[ kbench -nt $B/kbench.cpp ] || clang++ -O2 -std=c++20 $B/kbench.cpp -I$HOME/opt/cuda/include -L/.singularity.d/libs -lcuda -o kbench
[ kbench16 -nt $B/kbench16.cpp ] || clang++ -O2 -std=c++20 $B/kbench16.cpp -I$HOME/opt/cuda/include -lcuda -o kbench16
f=$1; shift
n=$(basename $f .mim)
src=$B/$f
if [ -n "$N" ] && [ "$N" != 1024 ]; then n=${n}_$N; sed "s/1024/$N/g" $B/$f > $n.mim; src=$n.mim; fi
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $src -p nvptx -o - "$@" > $n.log 2>&1 || { echo "$n: mim failed"; tail -3 $n.log; exit 1; }
k=$(grep -oE 'c"(tileKernel|kernel)_[0-9]*' $n.ll | head -1 | cut -c3-)
read g i smem <<< $(grep -o 'cuLaunchKernel_ptsz(ptr %[^,]*, i32 [0-9]*, i32 1, i32 1, i32 [0-9]*, i32 1, i32 1, i32 [0-9]*' $n.ll | head -1 | sed -E 's/.*i32 ([0-9]+), i32 1, i32 1, i32 ([0-9]+), i32 1, i32 1, i32 ([0-9]+)/\1 \2 \3/')
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*' | head -1)
echo "$n [$*] $regs grid=$g block=$i smem=$smem: $(if [ -n "$TY" ]; then ./kbench16 ${n}_dev.cubin $k $g $i $TY ${ITERS:-200} ${N:-1024} $smem; else ./kbench ${n}_dev.cubin $k $g $i ${ITERS:-200} ${N:-1024} $smem; fi)"
