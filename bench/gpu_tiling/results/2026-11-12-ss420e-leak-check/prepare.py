from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,json,re
source=Path('/workspaces/compile/MimIR/.worktrees/gpu-tiling')
root=Path('/tmp/mimir-gpu-leak-check')
cases=[('tile_pipeline',[16,16,16]),('tile_panel_tail',[32,32,24]),('tile_mma',[16,16,16])]
def run(case):
 name,shape=case;work=root/name;work.mkdir(exist_ok=True)
 commands=[[str(source/'build/bin/mim'),'-p','opt','-p','gpu','-X','tensor:tile-fuse=1','-X','tensor:strict=1',str(source/f'lit/gpu/{name}.mim'),'-p','nvptx','-X','nvptx:no-embed','-X','nvptx:sm=86'],['opt','-mtriple=nvptx64-nvidia-cuda','-passes=default<O2>,nvvm-reflect',f'{name}_dev.ll','-o','kernel.bc'],['llc','-march=nvptx64','-mcpu=sm_86','kernel.bc','-o','kernel.ptx']]
 with (work/'compile.log').open('w') as log:
  for cmd in commands:subprocess.run(cmd,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
 host=(work/f'{name}.ll').read_text();device=(work/f'{name}_dev.ll').read_text()
 kernels=set(re.findall(r'c"((?:tileKernel|kernel)_\d+)',host))
 launches=re.findall(r'cuLaunchKernel_ptsz\(ptr %[^,]+, i32 (\d+), i32 1, i32 1, i32 (\d+), i32 1, i32 1, i32 (\d+)',host)
 assert len(kernels)==len(launches)==1,(name,kernels,launches)
 heap_allocs=len(re.findall(r'\bcall\b[^\n]*@malloc\(',device));heap_frees=len(re.findall(r'\bcall\b[^\n]*@free\(',device))
 g,i,s=map(int,launches[0]);meta=dict(name=name,shape=shape,kernel=kernels.pop(),groups=g,items=i,smem=s,commands=commands,device_heap_alloc_sites=heap_allocs,device_heap_free_sites=heap_frees)
 (work/'launch.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,cases))
