#!/bin/bash
# krun16.sh <n> <f16|bf16> [generator args...]: generates a 16-bit GEMM kernel with $GEN.py (default gemm_h16), compiles it for
# sm_$SM (default 90a) and times it with kbench16.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_tiling
mkdir -p $B/out && cd $B/out
[ kbench16 -nt $B/kbench16.cpp ] || clang++ -O2 -std=c++20 $B/kbench16.cpp -I$HOME/opt/cuda/include -lcuda -o kbench16
N=$1 TY=$2; shift 2
GEN=${GEN:-gemm_h16}
n=${GEN}_${TY}_${N}$(printf '_%s' "$@")
python3 $B/$GEN.py $N $TY "$@" > $n.mim
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt $n.mim -p nvptx -X nvptx:sm=${SM:-90a} -o - > $n.log 2>&1 || { echo "$n: mim failed"; grep -m3 -A3 error $n.log; exit 1; }
k=$(grep -oE 'c"(tileKernel|kernel)_[0-9]*' $n.ll | head -1 | cut -c3-)
read g i smem <<< $(grep -o 'cuLaunchKernel_ptsz(ptr %[^,]*, i32 [0-9]*, i32 1, i32 1, i32 [0-9]*, i32 1, i32 1, i32 [0-9]*' $n.ll | head -1 | sed -E 's/.*i32 ([0-9]+), i32 1, i32 1, i32 ([0-9]+), i32 1, i32 1, i32 ([0-9]+)/\1 \2 \3/')
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*' | head -1)
echo "$n $regs smem=$smem: $(./kbench16 ${n}_dev.cubin $k $g $i $TY ${ITERS:-200} $N $smem)"
