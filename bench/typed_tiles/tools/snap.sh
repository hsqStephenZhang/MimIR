#!/bin/bash
# snap.sh <outdir>: from the repository root, compile every lit/gpu test to Mim and store the gid-normalized output.
out=$(realpath -m "$1"); mkdir -p "$out"
ls lit/gpu/*.mim | xargs -P 12 -I{} bash -c 'f={}; b=$(basename $f .mim); flags=$(grep -m1 "RUN:.*%mim" $f | grep -o "\-X [a-z:=0-9a-z_-]*" | tr "\n" " "); timeout 300 ./build/bin/mim -p opt $flags $f -p gpu -o - 2>/dev/null | sed -E "s/_[0-9]+//g" > '"$out"'/$b.out; echo "$? $(md5sum < '"$out"'/$b.out | cut -c1-12) $b"' | sort -k3 > "$out/SUMMARY"
