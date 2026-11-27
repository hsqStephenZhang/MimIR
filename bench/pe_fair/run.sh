#!/usr/bin/env bash
# Builds fair.mim into a shared library and times it against PyTorch eager and torch.compile in one process.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="${MIM_BUILD_DIR:-$here/../../build-rel}"
out="${OUT:-$here/out}"
py="${PY:-$HOME/workspaces/mimir-frontend/.venv/bin/python}"
mkdir -p "$out/inductor"
"$build/bin/mim" -p opt "$here/fair.mim" -p ll -X ll:o="$out/fair.ll" -o /dev/null
clang -O3 -march=native -shared -fPIC "$out/fair.ll" -o "$out/libfair.so" -Wno-override-module
USER="${USER:-bench}" TORCHINDUCTOR_CACHE_DIR="$out/inductor" "$py" "$here/fair.py" "$out/libfair.so"
