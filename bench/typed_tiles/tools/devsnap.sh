#!/bin/bash
# devsnap.sh <outdir>: from the repository root, the device IR of the wgmma lit tests as their nvptx RUN lines compile them, never executed.
out=$(realpath -m "$1"); mkdir -p "$out"; W=$PWD
for f in $W/lit/gpu/*wgmma*.mim; do
    t=$(basename $f .mim); [ -f $W/lit/nvptx/$t.mim ] || continue
    flags=$(grep -m1 "RUN:.*%mim.*nvptx" $W/lit/nvptx/$t.mim | grep -o "\-X [a-zA-Z:=0-9_-]*" | tr "\n" " ")
    mkdir -p $out/$t
    (cd $out/$t && timeout 300 $W/build/bin/mim -p opt $f -p nvptx $flags -o /dev/null >log 2>&1; rc=$?; d=${t}_dev.ll
     echo "$t rc=$rc lines=$([ -f $d ] && wc -l < $d) mbar_wait=$([ -f $d ] && grep -c 'mbarrier.try_wait' $d) tma=$([ -f $d ] && grep -c 'cp.async.bulk' $d)")
done
