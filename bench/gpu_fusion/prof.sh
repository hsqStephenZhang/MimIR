#!/bin/sh
# prof.sh <binary> [args...]: nsys kernel summary (ns) for one run.
set -e
bin=$1; shift
name=$(basename "$bin")
nsys profile -o "${name}_prof" --force-overwrite true "$bin" "$@" > /dev/null 2>&1
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "${name}_prof.nsys-rep" 2>/dev/null | grep -v "^$" | tail -n +2
