#!/bin/bash
# cmp_device_ir.sh <base-root> <new-root> <out-dir> <file.mim>...: compiles each file (a path relative to the roots, so
# each tree compiles its own copy, or absolute) for sm_90a with both trees' build-debug and reports whether the
# renumbered device IR is the same.
export PATH=$HOME/opt/llvm-23/bin:$HOME/opt/tools/bin:$HOME/opt/src/llvm-build/bin:$HOME/opt/cuda/bin:$PATH
T=$(cd "$(dirname "$0")" && pwd)
base=$1 new=$2 out=$3; shift 3
for rel in "$@"; do
    n=$(basename "$rel" .mim)
    for side in base new; do
        root=$([ $side = base ] && echo $base || echo $new)
        src=$([[ $rel = /* ]] && echo $rel || echo $root/$rel)
        mkdir -p $out/$side && cd $out/$side && cp $src $n.mim && rm -f $n.ll ${n}_dev.ll
        timeout 900 $root/${MIM_BUILD:-build-debug}/bin/mim -p opt -p gpu -X tensor:tile-fuse=1 $n.mim -p nvptx -X nvptx:sm=90a > $n.log 2>&1 \
            && python3 $T/canon_ir.py ${n}_dev.ll > $n.canon || { echo "$n: $side failed: $(grep -m1 error $n.log | cut -c1-120)"; rm -f $n.canon; }
    done
    if [ -s $out/base/$n.canon ] && [ -s $out/new/$n.canon ]; then
        cmp -s $out/base/$n.canon $out/new/$n.canon && echo "$n: same" \
            || echo "$n: differs ($(diff $out/base/$n.canon $out/new/$n.canon | grep -c '^[<>]') lines)"
    fi
done
