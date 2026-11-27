#!/usr/bin/env bash
# run_conv3.sh OUT [layer...]: one F32 convolution as an implicit im2col and as a direct convolution through run_levels,
# and through master's conv_schedule (pe-cpu=0, pe-generic=0). A layer is name:ci:h:w:co:k:stride:pad.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
MIM=${MIM:-$HERE/../../build-rel/bin/mim}
out=$1; shift
mkdir -p "$out"
layers=("$@")
((${#layers[@]})) || layers=(vgg56:64:56:56:64:3:1:1 vgg28:128:28:28:128:3:1:1 vgg14:256:14:14:256:3:1:1
                             res1x1:256:56:56:64:1:1:0 stem:3:224:224:64:7:2:3 down:64:56:56:128:3:2:1)
for l in "${layers[@]}"; do
    IFS=: read -r name ci h w co k s p <<< "$l"
    oh=$(( (h + 2 * p - k) / s + 1 )); ow=$(( (w + 2 * p - k) / s + 1 ))
    for v in im2col direct sched; do
        cat > "$out/$name.$v.mim" <<MIM
plugin core;
plugin tensor;
plugin math;
let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
extern fun conv_$v (x: tensor.Tensor 4 (1, $ci, $h, $w) F32, w: tensor.Tensor 4 ($co, $ci, $k, $k) F32): tensor.Tensor 4 (1, $co, $oh, $ow) F32 =
    return (tensor.conv R (($s, $s), (1, 1), ($p, $p)) (x, w));
MIM
        case $v in
            im2col) flags=() ;;
            direct) flags=(-X tensor:cpu-conv=1) ;;
            sched)  flags=(-X tensor:pe-cpu=0 -X tensor:pe-generic=0) ;;
        esac
        "$MIM" -p opt "${flags[@]}" "$out/$name.$v.mim" -p ll -X ll:o="$out/$name.$v.ll" -o /dev/null
    done
    clang++ -O3 -march=native -Wno-override-module -DCI=$ci -DH=$h -DW=$w -DCO=$co -DK=$k -DS=$s -DP=$p \
        "$HERE/conv3_main.cpp" "$out/$name.im2col.ll" "$out/$name.direct.ll" "$out/$name.sched.ll" -o "$out/$name.exe"
    MALLOC_MMAP_THRESHOLD_=4000000000 MALLOC_TRIM_THRESHOLD_=100000000000 taskset -c ${CORE:-5} "$out/$name.exe" "$name"
done
