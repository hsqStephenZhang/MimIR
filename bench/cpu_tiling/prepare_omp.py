"""Lower gemm-bench cases for one fixed CPU thread hint; native compilation runs remotely."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bench', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--cases', type=Path, required=True, help='Previous manifest containing gemm-bench cases')
p.add_argument('--root', type=Path, required=True)
p.add_argument('--threads', type=int, required=True)
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
sys.path.insert(0, str(a.bench.resolve()))
from mimir.operators import module
build = a.build.resolve()
source = build.parent
root = a.root.resolve()
if a.cases.suffix == '.json':
    cases = json.loads(a.cases.read_text())['cases']
else:
    from common.cases import load_cases
    cases = load_cases(a.cases)
variant = 'mimir_multilevel_omp_6x16'
flags = ['-X', 'btensor:registers=128', '-X', 'tensor:strict=1', '-X', 'omp:strict=1',
         '-X', f'omp:threads={a.threads}', '-X', 'll:rt=extern']
for key, value in dict(order=1, mr=6, nr=16, mc=320, nc=104512, kc=320).items():
    flags += ['-X', f'tensor:cpu-{key}={value}']

def lower(case):
    directory = root / 'ir' / variant / '_'.join(map(str, case['shape']))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'kernel.mim'
    path.write_text(module(case, nominal=True))
    command = [str(build / 'bin/mim'), '-p', 'opt', '-p', 'omp', str(path), '-p', 'll',
               *flags, '-X', 'll:o=' + str(directory / 'kernel.ll')]
    start = time.monotonic()
    with (directory / 'lower.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    record = dict(case=case['name'], command=command, seconds=time.monotonic()-start,
                  launches=(directory / 'kernel.ll').read_text().count('call void @mim_par_for'))
    (directory / 'lower.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)

with ThreadPoolExecutor(max_workers=a.jobs) as pool:
    list(pool.map(lower, cases))
manifest = dict(source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                source_dirty=bool(subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()),
                bench_revision=subprocess.check_output(['git', '-C', str(a.bench), 'rev-parse', 'HEAD'], text=True).strip(),
                cases=cases, variants=[variant], threads=a.threads, target='icelake-server -mno-avx512f',
                flags=flags, files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in root.glob('ir/**/kernel.ll')})
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
