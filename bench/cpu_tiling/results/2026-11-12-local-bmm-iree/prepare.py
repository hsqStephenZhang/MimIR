from pathlib import Path
import subprocess,concurrent.futures,json,shutil,resource
source=Path('/workspaces/compile/MimIR/.worktrees/gpu-tiling')
root=Path('/tmp/mimir-bmm-iree-20261112')
before=Path('/tmp/mimir-bmm-local-20261112')
bench=Path('/tmp/gemm-bench-cpu-bmm')
resource.setrlimit(resource.RLIMIT_STACK,(64*1024*1024,resource.getrlimit(resource.RLIMIT_STACK)[1]))
def lower(t):
    r=root/f't{t}'
    with open(f'/tmp/mimir-bmm-iree-lower{t}.log','w') as log:
        subprocess.run(['python3',str(source/'bench/cpu_tiling/prepare_omp.py'),'--bench',str(bench),'--build',str(source/'build'),'--cases',str(source/'bench/cpu_tiling/bmm-cases.txt'),'--root',str(r),'--threads',str(t),'--jobs','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest=json.loads((r/'manifest.json').read_text())
    for case in manifest['cases']:
        stem='_'.join(map(str,case['shape']))
        src=before/f't{t}/ir/mimir_multilevel_omp_6x16'/stem
        dst=r/'ir/mimir_before'/stem
        dst.mkdir(parents=True,exist_ok=True)
        for f in ['kernel.ll','kernel.mim','lower.json']:
            shutil.copy2(src/f,dst/f)
        manifest['files'][str((dst/'kernel.ll').relative_to(r))]=__import__('hashlib').sha256((dst/'kernel.ll').read_bytes()).hexdigest()
    manifest['variants']=['mimir_before','mimir_multilevel_omp_6x16']
    manifest['before_source_commit']='eabf14aabf'
    (r/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for name,target in [('harness',bench),('mimir',bench/'mimir')]:
        if not (r/name).exists():(r/name).symlink_to(target)
    (r/'results').mkdir(exist_ok=True)
    if not (r/'results/inputs').exists():(r/'results/inputs').symlink_to(before/'t1/results/inputs')
    print('lowered',t,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lower,[1,4,6,8]))
for t in [1,4,6,8]:
    with open(f'/tmp/mimir-bmm-iree-native{t}.log','w') as log:
        subprocess.run(['bash','/tmp/mimir-bmm-iree-env.sh','prepare',str(t)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('prepared',t,flush=True)
