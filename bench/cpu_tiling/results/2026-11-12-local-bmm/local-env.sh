#!/bin/bash
set -euo pipefail
phase=$1
threads=$2
root=/tmp/mimir-bmm-local-20261112
case "$threads" in
 1) cpus=0;;
 4) cpus=0,2,4,6;;
 6) cpus=0,2,4,6,8,10;;
 8) cpus=0,2,4,6,8,10,12,13;;
 *) exit 2;;
esac
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="${cpus//,/ }" OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export OPENBLAS_THREAD_TIMEOUT=4 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 OPENBLAS_CORETYPE=Haswell
export DNNL_MAX_CPU_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_VERBOSE=0
unset OMP_PLACES
exec /workspaces/ml-compiler/gemm-bench/.venv/bin/python /workspaces/compile/MimIR/.worktrees/gpu-tiling/bench/cpu_tiling/remote_cpu.py "$phase" \
 --root "$root/t$threads" --threads "$threads" --cpus "$cpus" --cpu 0 --isa avx2 --mcpu haswell \
 --clang clang++ --jobs 5 --compile-cpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
 --calibrate-once --bind-openblas-workers \
 --openblas /workspaces/compile/OpenBLAS/libopenblas.so \
 --runtime-source /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/omp/rt/mim_omp_rt.c \
 --omp-runtime /workspaces/ml-compiler/gemm-bench/.venv/lib/python3.11/site-packages/torch/lib/libgomp.so.1
