#!/bin/bash
# compile_all.sh [out-dir]: compiles every GPU benchmark for sm_90a, its shape placeholders set to small defaults, and
# lists the ones that fail, and the `reject_*` examples that do not; a benchmark's own script sets the shapes it is timed at.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
R=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
O=${1:-$(mktemp -d)}
mkdir -p $O
fails=0
for f in $R/bench/{gpu_attn,gpu_fusion,gpu_tiling,typed_tiles}/*.mim; do
    grep -qE 'tile_req|law\.|tensor\.|gpu\.' $f || continue
    n=$(basename $(dirname $f))_$(basename $f .mim)
    sed -e 's/\bHH\b/2/g; s/\bLL\b/128/g; s/\bDD\b/64/g; s/\bBM\b/64/g; s/\bBK\b/64/g; s/SCALE/0.125/g; s/\bPIPE\b/2/g; s/\bCL\b/2/g' \
        -e "s|import \"\([^/\"]*\.mim\)\"|import \"$(dirname $f)/\1\"|; s|import \"\.\./|import \"$(dirname $f)/../|" $f > $O/$n.mim
    if (cd $O && timeout 900 $R/${MIM_BUILD:-build-debug}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $n.mim -p nvptx -X nvptx:sm=90a -o /dev/null > $n.log 2>&1); then
        [[ $n == *_reject_* ]] && { echo "$n: compiles, but it shows a rejected schedule"; fails=$((fails + 1)); }
    elif [[ $n != *_reject_* ]]; then
        echo "$n: $(grep -m1 error $O/$n.log | cut -c1-150)"
        fails=$((fails + 1))
    fi
done
echo "$fails failed"
