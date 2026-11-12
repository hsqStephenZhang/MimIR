#!/bin/bash
set -euo pipefail
cd /home/sunqi/zc/mimir-bench/cpu-omp-20261112
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
export DNNL_MAX_CPU_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2
export MKL_VERBOSE=0 MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export OMP_PROC_BIND=TRUE OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 OPENBLAS_THREAD_TIMEOUT=4
unset OMP_PLACES
phase=$1
threads=$2
runroot=${3:-$PWD}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
cpus=12
if [ "$threads" = 4 ]; then cpus=12,13,14,15; fi
if [ "$threads" = 8 ]; then cpus=12,13,14,15,16,17,18,19; fi
export GOMP_CPU_AFFINITY="${cpus//,/ }"
exec /home/sunqi/zc/mimir-bench/gemm-bench-gpu-env/bin/python remote_cpu.py "$phase" \
 --root "$runroot/t$threads" --threads "$threads" --cpus "$cpus" --cpu 12 --isa avx2 --calibrate-once --jobs 6 \
 --openblas /home/sunqi/zc/mimir-bench/OpenBLAS-8e57c86-haswell-pthread/libopenblas.so \
 --runtime-source "$PWD/mim_omp_rt.c" \
 --omp-runtime /home/sunqi/anaconda3/lib/python3.12/site-packages/torch/lib/libgomp-a34b3233.so.1
