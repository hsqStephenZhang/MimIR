#!/bin/bash
# krun.sh <name.mim> [extra mim flags...]: compile a gpu_tiling bench and time its kernel with kbench; N=<n> resizes it.
export PATH=/usr/lib/llvm-20/bin:$HOME/opt/cuda/bin:$PATH
export LIBRARY_PATH=/.singularity.d/libs LD_LIBRARY_PATH=/.singularity.d/libs
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_tiling
mkdir -p $B/out && cd $B/out
[ kbench -nt $B/kbench.cpp ] || clang++ -O2 -std=c++20 $B/kbench.cpp -I$HOME/opt/cuda/include -L/.singularity.d/libs -lcuda -o kbench
f=$1; shift
n=$(basename $f .mim)
src=$B/$f
if [ -n "$N" ] && [ "$N" != 1024 ]; then n=${n}_$N; sed "s/1024/$N/g" $B/$f > $n.mim; src=$n.mim; fi
$R/build-gcc15/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $src -p nvptx -o - -X 'nvptx:Xopt=-passes="default<O2>,function(nvvm-reflect)"' "$@" > $n.log 2>&1 || { echo "$n: mim failed"; tail -3 $n.log; exit 1; }
k=$(grep -oE 'c"(tileKernel|kernel)_[0-9]*' $n.ll | head -1 | cut -c3-)
read g i <<< $(grep -o 'cuLaunchKernel_ptsz(ptr %[^,]*, i32 [0-9]*, i32 1, i32 1, i32 [0-9]*' $n.ll | head -1 | sed -E 's/.*i32 ([0-9]+), i32 1, i32 1, i32 ([0-9]+)/\1 \2/')
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*' | head -1)
echo "$n [$*] $regs grid=$g block=$i: $(./kbench ${n}_dev.cubin $k $g $i ${ITERS:-200} ${N:-1024})"
