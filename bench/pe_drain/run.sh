#!/usr/bin/env bash
# Times drain.mim with the default fusion policy, with every fusion the mechanism can do (pe-fuse=2), with none (pe-fuse=0) and with consumer fusion (pe-drain=1), against PyTorch.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
build="${MIM_BUILD_DIR:-$here/../../build-rel}"
out="${OUT:-$here/out}"
py="${PY:-$HOME/workspaces/mimir-frontend/.venv/bin/python}"
mkdir -p "$out/inductor"
libs=()
for v in "policy:" "all:-X tensor:pe-fuse=2" "nofuse:-X tensor:pe-fuse=0" "drain:-X tensor:pe-drain=1"; do
    name="${v%%:*}"
    read -ra flags <<< "${v#*:}"
    "$build/bin/mim" -p opt "${flags[@]}" "$here/drain.mim" -p ll -X ll:o="$out/$name.ll" -o /dev/null
    clang -O3 -march=native -shared -fPIC "$out/$name.ll" -o "$out/lib$name.so" -Wno-override-module
    libs+=("$name=$out/lib$name.so")
done
USER="${USER:-bench}" TORCHINDUCTOR_CACHE_DIR="$out/inductor" "$py" "$here/drain.py" "${libs[@]}"
