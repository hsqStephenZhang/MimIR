from pathlib import Path
import subprocess,os,json,ctypes
source=Path('/workspaces/compile/MimIR/.worktrees/gpu-tiling')
root=Path('/tmp/mimir-bmm-iree-20261112')
bench=Path('/tmp/gemm-bench-cpu-bmm')
gomp=Path('/workspaces/ml-compiler/gemm-bench/.venv/lib/python3.11/site-packages/torch/lib/libgomp.so.1')
if len(__import__('sys').argv)==1:
    for t,cpus in [(4,[0,2,4,6]),(6,[0,2,4,6,8,10]),(8,[0,2,4,6,8,10,12,13])]:
        r=root/f't{t}'
        obj=r/'audit-runtime.o'
        subprocess.run(['gcc','-O3','-fPIC','-fopenmp','-Dmim_par_for=mim_par_for_impl','-c',str(source/'src/mim/plug/omp/rt/mim_omp_rt.c'),'-o',str(obj)],check=True)
        for stem in ['8_256_256_256','3_128_128_128','1_128_128_128']:
            d=r/'ir/mimir_multilevel_omp_6x16'/stem
            subprocess.run(['clang++','-O3','-march=haswell','-mno-avx512f','-shared','-fPIC','-std=c++20','-Wno-override-module',str(d/'kernel.ll'),str(bench/'mimir/native_timer.cpp'),str(source/'bench/cpu_tiling/audit_omp.cpp'),str(obj),str(gomp),'-Wl,-rpath,'+str(gomp.parent),'-o',str(d/'audit.so')],check=True)
        env=dict(os.environ,OMP_NUM_THREADS=str(t),OMP_PROC_BIND='TRUE',GOMP_CPU_AFFINITY=' '.join(map(str,cpus)),OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',OPENBLAS_NUM_THREADS='1')
        subprocess.run(['/workspaces/ml-compiler/gemm-bench/.venv/bin/python',__file__,str(t),','.join(map(str,cpus))],env=env,check=True)
else:
    import sys
    t=int(sys.argv[1]); cpus=set(map(int,sys.argv[2].split(',')))
    os.sched_setaffinity(0,cpus)
    import numpy as np
    sys.path.insert(0,str(bench))
    from mimir.benchmark import Kernel
    r=root/f't{t}'; rows=[]
    for stem in ['8_256_256_256','3_128_128_128','1_128_128_128']:
        b,m,n,k=map(int,stem.split('_')); d=r/'ir/mimir_multilevel_omp_6x16'/stem
        inputs=r/'results/inputs'/stem
        a=np.load(inputs/'A.npy');bb=np.load(inputs/'B.npy')
        kernel=Kernel(d/'audit.so',a,bb,(b,m,n),d,0,'clang++',[],[])
        os.sched_setaffinity(0,{0})
        ref=a.astype('float64')@bb.astype('float64')
        for _ in range(20): np.testing.assert_allclose(kernel.run(),ref,rtol=1e-3,atol=1e-3)
        lib=kernel.library
        lib.audit_workers.restype=lib.audit_cpus.restype=ctypes.c_uint64
        workers=lib.audit_workers(); actual=lib.audit_cpus()
        rec=dict(shape=[b,m,n,k],calls=20,launches=lib.audit_launches(),team=lib.audit_team(),workers=[i for i in range(64) if workers>>i&1],cpus=[i for i in range(64) if actual>>i&1])
        assert rec['team']==t and rec['workers']==list(range(t)) and set(rec['cpus'])==cpus and rec['launches']==20,rec
        rows.append(rec)
    (r/'results/worker-audit.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(threads=t,rows=rows)),flush=True)
