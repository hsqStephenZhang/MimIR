#!/bin/bash
# kattn.sh <name.mim> <H> <L> <D> [extra mim flags...]: compiles an attention bench for that shape, checks it and sums
# its kernel times with nsys.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=/home/chzh00001/workspaces/MimIR
B=$R/bench/gpu_attn
f=$1 H=$2 L=$3 D=$4; shift 4
BM=${BM:-64} BK=${BK:-32}
n=$(basename $f .mim)_${H}_${L}_${D}
mkdir -p $B/out && cd $B/out
scale=$(python3 -c "print(f'{1/$D**0.5:.6f}')")
sed -e "s/\bHH\b/$H/g; s/\bLL\b/$L/g; s/\bDD\b/$D/g; s/\bBM\b/$BM/g; s/\bBK\b/$BK/g; s/SCALE/$scale/g" $B/$f > $n.mim
$R/${MIM_BUILD:-build-h200}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $n.mim -p nvptx "$@" > $n.log 2>&1 || { echo "$n: mim failed"; tail -3 $n.log; exit 1; }
clang++ -O2 -DH=$H -DL=$L -DD=$D $B/attn_driver.cpp $n.ll -lcuda -o $n.exec -Wno-override-module || exit 1
check=$(./$n.exec 1 | tail -1)
nsys profile -o $n --force-overwrite true ./$n.exec 5 > /dev/null 2>&1
kern=$(nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv $n.nsys-rep 2>/dev/null | python3 -c "
import csv,sys
rows=[r for r in csv.reader(sys.stdin) if len(r)>8 and r[0][:1].isdigit()]
print(len(rows), 'kernels', '%.3f ms/call' % (sum(float(r[3]) for r in rows)/1e6), ' '.join('%s=%.3f' % (r[-1][:24], float(r[3])/1e6) for r in rows))")
echo "$n [$*] $check: $kern"
