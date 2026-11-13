#!/bin/bash
# kflash16.sh <BH> <L> <f16|bf16> [generator args...]: generates a hand-written 16-bit attention kernel with $GEN.py (default flash_h16), checks it and sums its kernel times
# with nsys.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_attn
BH=$1 L=$2 TY=$3; shift 3
GEN=${GEN:-flash_h16}
n=${GEN}_${TY}_${BH}_${L}$([ $# -gt 0 ] && printf '_%s' "$@")
mkdir -p $B/out && cd $B/out
python3 $B/$GEN.py $BH $L $TY "$@" > $n.mim
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt -p gpu $n.mim -p nvptx -X nvptx:sm=${SM:-90a} $MIMFLAGS > $n.log 2>&1 || { echo "$n: mim failed"; grep -m3 -A3 error $n.log; exit 1; }
clang++ -O2 -DH=$BH -DL=$L $([ $TY = bf16 ] && echo -DBF16) $B/flash_driver16.cpp $n.ll -lcuda -o $n.exec -Wno-override-module || exit 1
check=$(./$n.exec 1 | tail -1)
nsys profile -o $n --force-overwrite true ./$n.exec 5 > /dev/null 2>&1
kern=$(nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $n.nsys-rep 2>/dev/null | python3 -c "
import csv,sys
rows=[r for r in csv.reader(sys.stdin) if len(r)>8 and r[0][:1].isdigit()]
print(len(rows), 'kernels', '%.3f ms/call' % (sum(float(r[3]) for r in rows)/1e6))")
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*\|SHARED:[0-9]*' | head -2 | tr '\n' ' ')
echo "$n $regs $check: $kern"
