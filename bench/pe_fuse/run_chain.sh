#!/usr/bin/env bash
# run_chain.sh OUT "K..." M D TI TK [extra clang flags]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MIM=$HERE/../../build-rel/bin/mim
out=$1; Ks=$2; M=$3; D=$4; TI=$5; TK=$6; shift 6
mkdir -p "$out"
ms() { date +%s%N; }
for K in $Ks; do
    f=$out/chain$K
    stats=""
    for which in fused layered; do
        python3 "$HERE/gen_chain.py" "$K" "$M" "$D" "$TI" "$TK" "$which" > "$f.$which.mim"
        t0=$(ms)
        "$MIM" -p opt "$f.$which.mim" -p ll -X ll:o="$f.$which.ll" -o "$f.$which.out.mim"
        t1=$(ms)
        stats+=$(printf "%s: %.2fs %d ll  " "$which" "$(awk "BEGIN{print ($t1-$t0)/1e9}")" "$(wc -l < "$f.$which.ll")")
    done
    python3 "$HERE/gen_chain.py" "$K" "$M" "$D" "$TI" "$TK" both > "$f.mim"
    "$MIM" -p opt "$f.mim" -p ll -X ll:o="$f.ll" -o "$f.out.mim"
    clang++ -O2 -march=native "$@" "$HERE/chain_main.cpp" "$f.ll" -o "$f.exe" -Wno-override-module
    echo "$stats"
    "$f.exe" "$K" "$M" "$D"
done
