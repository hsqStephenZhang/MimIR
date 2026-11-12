from pathlib import Path
import ctypes, json, os, subprocess, sys
threads=int(sys.argv[1]); cpus=set(range(12,12+threads));os.sched_setaffinity(0,cpus)
os.environ['OMP_PROC_BIND']='TRUE'
os.environ['GOMP_CPU_AFFINITY']=' '.join(map(str,sorted(cpus)))
root=Path(sys.argv[2]) if len(sys.argv)>2 else Path('/home/sunqi/zc/mimir-bench/cpu-omp-20261112')
sys.path.insert(0,str(root/f't{threads}/harness'))
import numpy as np
import torch
from mimir.benchmark import Kernel
base=root/f't{threads}/ir/mimir_multilevel_omp_6x16/256_256_256'
gomp=Path(torch.__file__).parent/'lib/libgomp-a34b3233.so.1'
subprocess.run(['gcc','-O3','-fPIC','-fopenmp','-Dmim_par_for=mim_par_for_impl','-c',str(root/'mim_omp_rt.c'),'-o',str(root/'audit-runtime.o')],check=True)
cmd=['/usr/lib/llvm-20/bin/clang++','-O3','-march=icelake-server','-mno-avx512f','-shared','-fPIC','-std=c++20','-Wno-override-module',str(base/'kernel.ll'),str(root/f't{threads}/mimir/native_timer.cpp'),str(root/'audit_omp.cpp'),str(root/'audit-runtime.o'),str(gomp),'-Wl,-rpath,'+str(gomp.parent),'-o',str(base/'audit.so')]
subprocess.run(cmd,check=True)
a=np.load(root/f't{threads}/results/inputs/256_256_256/A.npy'); b=np.load(root/f't{threads}/results/inputs/256_256_256/B.npy')
k=Kernel(base/'audit.so',a,b,(256,256),base,0,cmd[0],[],[])
ref=a.astype('float64')@b.astype('float64')
for i in range(20):np.testing.assert_allclose(k.run(),ref,rtol=.001,atol=.001)
lib=ctypes.CDLL(str(base/'audit.so'))
for name in ['audit_workers','audit_cpus']:getattr(lib,name).restype=ctypes.c_uint64
workers=lib.audit_workers();actualcpus=lib.audit_cpus();team=lib.audit_team();launches=lib.audit_launches()
print(dict(expected=threads,team=team,workers=workers,cpus=actualcpus,launches=launches),flush=True)
assert team==threads and workers==2**threads-1
assert actualcpus==sum(1<<i for i in cpus)
assert launches==20
record=dict(threads=threads,worker_ids=[i for i in range(64) if workers>>i&1],cpus=[i for i in range(64) if actualcpus>>i&1],team=team,launches=launches,validated_calls=20,compile=cmd)
(root/f't{threads}/results/worker-audit.json').write_text(json.dumps(record,indent=2));print(record)
