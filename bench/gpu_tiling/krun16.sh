#!/bin/bash
# krun16.sh <n> <f16|bf16> [extra mim flags...]: generates gemm_h16.py's kernel, compiles it and times it with kbench16.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_tiling
mkdir -p $B/out && cd $B/out
[ kbench16 -nt $B/kbench16.cpp ] || clang++ -O2 -std=c++20 $B/kbench16.cpp -I$HOME/opt/cuda/include -lcuda -o kbench16
N=$1 TY=$2; shift 2
n=gemm_h16_${TY}_$N
python3 $B/gemm_h16.py $N $TY > $n.mim
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt $n.mim -p nvptx -o - "$@" > $n.log 2>&1 || { echo "$n: mim failed"; tail -3 $n.log; exit 1; }
k=$(grep -oE 'c"(tileKernel|kernel)_[0-9]*' $n.ll | head -1 | cut -c3-)
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*' | head -1)
echo "$n [$*] $regs: $(./kbench16 ${n}_dev.cubin $k $(( (N / 128) * (N / 128) )) 256 $TY ${ITERS:-200} $N)"
