from pathlib import Path
import csv,json,math
root=Path(__file__).resolve().parent
backend='mimir_multilevel_omp_6x16'
data={t:json.loads((root/f't{t}/paired-native.json').read_text()) for t in [1,4,8]}
byname={t:{r['name']:r for r in d['rows']} for t,d in data.items()}
assert all(len(d['rows'])==22 for d in data.values())
def gm(values):
 values=list(values);return math.exp(sum(map(math.log,values))/len(values))
summary={}
for t in [1,4,8]:
 rows=data[t]['rows']
 summary[t]={name:dict(speedup_vs_one=gm(byname[1][r['name']]['ms'][name]/r['ms'][name] for r in rows),
                        throughput_vs_openblas=gm(r['ms']['openblas']/r['ms'][name] for r in rows),
                        throughput_vs_pytorch=gm(r['ms']['pytorch']/r['ms'][name] for r in rows))
             for name in [backend,'openblas','pytorch']}
 large=[r for r in rows if r['name'] in ['sq512','sq1000','sq1024','sq2000','sq2048','odd2048','big_odd']]
 summary[t]['large']={name:dict(speedup_vs_one=gm(byname[1][r['name']]['ms'][name]/r['ms'][name] for r in large),
                               throughput_vs_openblas=gm(r['ms']['openblas']/r['ms'][name] for r in large),
                               throughput_vs_pytorch=gm(r['ms']['pytorch']/r['ms'][name] for r in large))
                        for name in [backend,'openblas','pytorch']}
(root/'summary.json').write_text(json.dumps(summary,indent=2))
with (root/'comparison.csv').open('w') as f:
 fields=['name','M','K','N']+[f'{name}_{t}_ms' for t in [1,4,8] for name in ['mimir','openblas','pytorch']]+[f'mimir_{t}_speedup' for t in [4,8]]+[f'mimir_{t}_vs_{name}' for t in [4,8] for name in ['openblas','pytorch']]
 writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
 for r in data[1]['rows']:
  m,n,k=r['shape'];line=dict(name=r['name'],M=m,K=k,N=n)
  for t in [1,4,8]:
   times=byname[t][r['name']]['ms']
   for key,name in [('mimir',backend),('openblas','openblas'),('pytorch','pytorch')]:line[f'{key}_{t}_ms']=times[name]
   if t>1:
    line[f'mimir_{t}_speedup']=r['ms'][backend]/times[backend]
    for name in ['openblas','pytorch']:line[f'mimir_{t}_vs_{name}']=times[name]/times[backend]
  writer.writerow(line)
print(json.dumps(summary,indent=2))
