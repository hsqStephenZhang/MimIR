#!/usr/bin/env bash
# tensor.product_2d gemm benchmark vs single-threaded OpenBLAS; shapes.txt lists name M K N reps.
# bench.mim is generated from shapes.txt (and shapes2.txt for the near-square set).
set -euo pipefail
cd "$(dirname "$0")"
MIM=${MIM:-../../build/bin/mim}
SHAPES=${1:-shapes.txt}
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$(uname -s)" == Darwin && -z "${DEVELOPER_DIR:-}" && -d /Library/Developer/CommandLineTools ]]; then
    # Prefer the standalone tools when Xcode's linker cannot read the active SDK.
    export DEVELOPER_DIR=/Library/Developer/CommandLineTools
fi

CXX=${CXX:-}
if [[ -z "$CXX" ]]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX=$(command -v clang++)
    elif command -v c++ >/dev/null 2>&1; then
        CXX=$(command -v c++)
    else
        echo "error: clang++ or c++ is required" >&2
        exit 1
    fi
fi

NATIVE_ARCH_FLAGS=()
if "$CXX" -march=native -x c++ -c /dev/null -o /dev/null >/dev/null 2>&1; then
    NATIVE_ARCH_FLAGS=(-march=native)
elif "$CXX" -mcpu=native -x c++ -c /dev/null -o /dev/null >/dev/null 2>&1; then
    # Some Apple Silicon clang versions expose native CPU tuning through -mcpu.
    NATIVE_ARCH_FLAGS=(-mcpu=native)
elif [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then
    # Older Apple clang releases do not expose a native CPU selector for arm64.
    NATIVE_ARCH_FLAGS=()
    echo "warning: $CXX lacks native arm64 tuning flags; using its macOS arm64 default" >&2
else
    echo "error: $CXX does not support -march=native or -mcpu=native" >&2
    exit 1
fi

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas; then
    OPENBLAS_CFLAGS_TEXT=$(pkg-config --cflags openblas)
    OPENBLAS_LIBS_TEXT=$(pkg-config --libs openblas)
    OPENBLAS_CFLAGS=()
    OPENBLAS_LIBS=()
    [[ -z "$OPENBLAS_CFLAGS_TEXT" ]] || read -r -a OPENBLAS_CFLAGS <<< "$OPENBLAS_CFLAGS_TEXT"
    [[ -z "$OPENBLAS_LIBS_TEXT" ]] || read -r -a OPENBLAS_LIBS <<< "$OPENBLAS_LIBS_TEXT"
elif [[ "$(uname -s)" == Darwin ]] && command -v brew >/dev/null 2>&1; then
    OPENBLAS_PREFIX=$(brew --prefix openblas)
    OPENBLAS_CFLAGS=(-I"$OPENBLAS_PREFIX/include")
    OPENBLAS_LIBS=(-L"$OPENBLAS_PREFIX/lib" -lopenblas -Wl,-rpath,"$OPENBLAS_PREFIX/lib")
else
    echo "error: OpenBLAS pkg-config metadata is required (install OpenBLAS and pkg-config)" >&2
    exit 1
fi

if [[ ! -x "$MIM" ]]; then
    echo "error: MimIR compiler not found or not executable: $MIM" >&2
    exit 1
fi
if [[ ! -f "$SHAPES" ]]; then
    echo "error: shape list not found: $SHAPES" >&2
    exit 1
fi

{ echo 'plugin tensor;
plugin core;
plugin math;

let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
'; while read -r n m k l _; do printf 'extern fun %s (a: «%s, %s; F32», b: «%s, %s; F32»): «%s, %s; F32» = return (tensor.product_2d R (a, b));\n' "$n" "$m" "$k" "$k" "$l" "$m" "$l"; done < "$SHAPES"; } > bench.mim
awk '{printf "S(%s, %s, %s, %s, %s)\n", $1, $2, $3, $4, $5}' "$SHAPES" > shapes.inc
"$MIM" -p opt bench.mim -p ll
"$CXX" -O3 -std=c++23 "${NATIVE_ARCH_FLAGS[@]}" -I. "${OPENBLAS_CFLAGS[@]}" bench2.cpp bench.ll \
    -Wno-override-module "${OPENBLAS_LIBS[@]}" -o bench
printf 'CXX: %s %s -O3\n' "$CXX" "${NATIVE_ARCH_FLAGS[*]}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./bench "$@"
