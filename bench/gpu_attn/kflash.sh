#!/bin/bash
# kflash.sh <BH> <L> [extra mim flags...] (BC=<keys per block>): generates the hand-written FA2 kernel, checks it and sums its kernel times
# with nsys.
export PATH=$HOME/opt/llvm-23/bin:/usr/lib/llvm-20/bin:$HOME/opt/cuda/bin:$PATH
export LIBRARY_PATH=/.singularity.d/libs LD_LIBRARY_PATH=/.singularity.d/libs
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_attn
BH=$1 L=$2 BC=${BC:-64}; shift 2
n=flash_tc_${BH}_${L}_${BC}
mkdir -p $B/out && cd $B/out
python3 $B/flash_tc.py $BH $L $BC > $n.mim
$R/build-gcc15/bin/mim -p opt -p gpu $n.mim -p nvptx "$@" > $n.log 2>&1 || { echo "$n: mim failed"; grep -m3 -A3 error $n.log; exit 1; }
clang++ -O2 -DH=$BH -DL=$L $B/flash_driver.cpp $n.ll -lcuda -o $n.exec -Wno-override-module || exit 1
check=$(./$n.exec 1 | tail -1)
nsys profile -o $n --force-overwrite true ./$n.exec 5 > /dev/null 2>&1
kern=$(nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $n.nsys-rep 2>/dev/null | python3 -c "
import csv,sys
rows=[r for r in csv.reader(sys.stdin) if len(r)>8 and r[0][:1].isdigit()]
print(len(rows), 'kernels', '%.3f ms/call' % (sum(float(r[3]) for r in rows)/1e6))")
regs=$(cuobjdump -res-usage ${n}_dev.cubin 2>/dev/null | grep -o 'REG:[0-9]*\|SHARED:[0-9]*' | head -2 | tr '\n' ' ')
echo "$n [$*] $regs $check: $kern"
