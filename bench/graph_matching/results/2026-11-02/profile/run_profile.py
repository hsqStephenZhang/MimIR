import csv, json, os, statistics, subprocess
from pathlib import Path

root=Path('/tmp/mimir-pattern-profile')
reference=Path('/tmp/mimir-reference-bench')
corpora=reference/'factored-results'
mim=reference/'mim-release/bin/mim-portgraph-bench'
port=root/'port/target/release/mimir-port-bench'
stages=['analysis','prefilter','index_setup','enumerate','narrow','output','cleanup']
rows=[]
for workload in ['chain','shared-dag','wrong-port','many-chain']:
    corpus=corpora/f'{workload}-128.txt'
    expected_path=corpora/f'{workload}-128-mim-portgraph-bench-indexed.matches'
    expected={tuple(map(int,l.split())) for l in expected_path.read_text().splitlines()}
    for variant in ['baseline','profile','no-cache','no-metadata']:
        for repeat in range(7):
            log=root/f'{workload}-{variant}-{repeat}.jsonl'
            if log.exists(): log.unlink()
            env=dict(os.environ)
            if variant!='baseline':
                env.update(LD_PRELOAD=str(root/'libmatch-profile.so'),MIM_PROFILE_JSON=str(log))
                if variant=='no-cache':env['MIM_PROFILE_NO_CACHE']='1'
                if variant=='no-metadata':env['MIM_PROFILE_NO_METADATA']='1'
            output=subprocess.check_output(['taskset','-c','2',str(mim),str(corpus),str(root/'result.matches'),'indexed'],env=env,text=True)
            actual=[tuple(map(int,l.split())) for l in (root/'result.matches').read_text().splitlines()]
            assert len(actual)==len(expected) and set(actual)==expected,(workload,variant)
            fields=output.strip().split(',')
            row=dict(workload=workload,variant=variant,repeat=repeat,caller_ns=int(fields[3]),matches=int(fields[4]))
            if variant!='baseline':
                records=[json.loads(l) for l in log.read_text().splitlines()]
                row.update({key:sum(r[key] for r in records) for key in records[0] if key not in ['anchors','entries','nodes','operations','no_cache','no_metadata','matches']})
                row['nodes']=records[0]['nodes'];row['operations']=records[0]['operations']
                row['pattern_entries']=sum(r['entries'] for r in records)
                (root/f'{workload}-{variant}-last.json').write_text(json.dumps(records,indent=2)+'\n')
            rows.append(row)
        print(workload,variant,'ms=',round(statistics.median(r['caller_ns'] for r in rows if r['workload']==workload and r['variant']==variant)/1e6,3),flush=True)
    for variant in ['naive','many']:
        for repeat in range(7):
            log=root/f'{workload}-port-{variant}-{repeat}.jsonl'
            if log.exists():log.unlink()
            env=dict(os.environ,PORT_PROFILE_JSON=str(log))
            output=subprocess.check_output(['taskset','-c','2',str(port),str(corpus),str(root/'result.matches'),variant],env=env,text=True)
            actual=[tuple(map(int,l.split())) for l in (root/'result.matches').read_text().splitlines()]
            assert len(actual)==len(expected) and set(actual)==expected,(workload,variant)
            record=json.loads(log.read_text());rows.append(dict(workload=workload,variant='port-'+variant,repeat=repeat,matches=len(actual),**record))

(root/'stage-samples.json').write_text(json.dumps(rows,indent=2)+'\n')
medians=[]
for workload in ['chain','shared-dag','wrong-port','many-chain']:
    for variant in ['baseline','profile','no-cache','no-metadata','port-naive','port-many']:
        selected=[r for r in rows if r['workload']==workload and r['variant']==variant]
        med=dict(selected[0]);del med['repeat']
        for key in med:
            if isinstance(med[key],int):med[key]=int(statistics.median(r[key] for r in selected))
        medians.append(med)
(root/'stage-medians.json').write_text(json.dumps(medians,indent=2)+'\n')
for med in medians:
    print(med['workload'],med['variant'],{key:med[key] for key in ['caller_ns','ns','prefilter_ns','enumerate_ns','narrow_ns','output_ns','cleanup_ns','allocations','bytes'] if key in med})
