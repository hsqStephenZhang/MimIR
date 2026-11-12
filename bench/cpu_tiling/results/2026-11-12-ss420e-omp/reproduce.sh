#!/bin/bash
set -euo pipefail
cd /home/sunqi/zc/mimir-bench/cpu-omp-20261112
tar -xzf mimir-omp-static-all.tar.gz
ln -sfn ../mim_omp_rt.c static/mim_omp_rt.c
ln -sfn ../audit_omp.cpp static/audit_omp.cpp
for threads in 1 4 8; do
 mkdir -p static/t$threads/results
 ln -sfn /home/sunqi/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2/harness static/t$threads/harness
 ln -sfn /home/sunqi/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2/mimir static/t$threads/mimir
 ln -sfn /home/sunqi/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2/results/inputs static/t$threads/results/inputs
 bash mimir-omp-run.sh prepare $threads "$PWD/static" > static/t$threads/prepare.log 2>&1
 echo "prepared $threads"
done
for threads in 4 8; do
 env OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 OPENBLAS_CORETYPE=Haswell /home/sunqi/zc/mimir-bench/gemm-bench-gpu-env/bin/python mimir-omp-audit.py $threads "$PWD/static"
done
/home/sunqi/zc/mimir-bench/gemm-bench-gpu-env/bin/python mimir-omp-isa.py "$PWD/static"
for threads in 1 4 8; do
 bash mimir-omp-run.sh run $threads "$PWD/static" > static/t$threads/run.log 2>&1
 echo "measured $threads"
done
