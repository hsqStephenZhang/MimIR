from pathlib import Path
from collections import Counter
import hashlib,json,re,subprocess,sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/sunqi/zc/mimir-bench/cpu-omp-20261112')
binaries=sorted(root.glob('t*/ir/**/kernel.so'))+[Path('/home/sunqi/zc/mimir-bench/OpenBLAS-8e57c86-haswell-pthread/libopenblas.so')]
assert len(binaries)==67
records=[]
for path in binaries:
 text=subprocess.check_output(['objdump','-d','-w',str(path)],text=True)
 instructions=re.findall(r'^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*(.*)$',text,re.M)
 evex=sum(raw.split()[0]=='62' for raw,op in instructions)
 assert evex==0,(path,evex)
 fma=Counter('zmm' if '%zmm' in op else 'ymm' if '%ymm' in op else 'xmm' for raw,op in instructions if 'vfmadd' in op)
 records.append(dict(binary=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evex_instructions=evex,static_fma_instructions=dict(fma)))
data=dict(checked_binaries=len(records),all_evex_zero=True,records=records)
(root/'isa-audit.json').write_text(json.dumps(data,indent=2));print('AVX2 audit passed',len(records))
