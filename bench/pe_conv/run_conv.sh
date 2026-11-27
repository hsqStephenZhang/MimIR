#!/usr/bin/env bash
# run_conv.sh OUT [layer...]: time F32 convolutions through run_levels (pe-cpu=1) and conv_schedule (pe-cpu=0).
# A layer is name:ci:h:w:co:k:stride:pad[:groups].
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MIM=$HERE/../../build-rel/bin/mim
out=$1; shift
mkdir -p "$out"
layers=("$@")
((${#layers[@]})) || layers=(vgg56:64:56:56:64:3:1:1 vgg28:128:28:28:128:3:1:1 vgg14:256:14:14:256:3:1:1
                             res1x1:256:56:56:64:1:1:0 stem:3:224:224:64:7:2:3 down:64:56:56:128:3:2:1)
for l in "${layers[@]}"; do
    IFS=: read -r name ci h w co k s p g <<< "$l"
    g=${g:-1}
    op=$([ "$g" = 1 ] && echo "tensor.conv R" || echo "tensor.conv_grouped R ($g,")
    args=$([ "$g" = 1 ] && echo "(($s, $s), (1, 1), ($p, $p))" || echo "($s, $s), (1, 1), ($p, $p))")
    oh=$(( (h + 2 * p - k) / s + 1 )); ow=$(( (w + 2 * p - k) / s + 1 ))
    for v in pe old; do
        cat > "$out/$name.$v.mim" <<MIM
plugin core;
plugin tensor;
plugin math;
let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
extern fun conv_$v (x: tensor.Tensor 4 (1, $ci, $h, $w) F32, w: tensor.Tensor 4 ($co, $((ci / g)), $k, $k) F32): tensor.Tensor 4 (1, $co, $oh, $ow) F32 =
    return ($op $args (x, w));
MIM
        flag=$([ $v = pe ] && echo 1 || echo 0)
        t0=$(date +%s%N)
        "$MIM" -p opt -X tensor:pe-cpu=$flag "$out/$name.$v.mim" -p ll -X ll:o="$out/$name.$v.ll" -o "$out/$name.$v.out.mim"
        t1=$(date +%s%N)
        echo "$name $v compile $(awk "BEGIN{printf \"%.2f\", ($t1-$t0)/1e9}")s ll $(wc -l < "$out/$name.$v.ll")" >> "$out/compile.txt"
    done
    clang++ -O3 -march=native -Wno-override-module -DCI=$ci -DH=$h -DW=$w -DCO=$co -DK=$k -DS=$s -DP=$p -DG=$g \
        "$HERE/conv_main.cpp" "$out/$name.pe.ll" "$out/$name.old.ll" -o "$out/$name.exe"
    MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=100000000000 "$out/$name.exe" "$name"
done
cat "$out/compile.txt"
