from pathlib import Path
import subprocess,time,json,hashlib
root=Path('/tmp/mimir-bmm-iree-20261112')
prototype=Path('/tmp/mimir-bmm-iree-prototype')
while 'prepared 8' not in Path('/tmp/mimir-bmm-iree-prepare-final.log').read_text():time.sleep(2)
while 'Total Discovered Tests' not in Path('/tmp/mimir-bmm-iree-lit.log').read_text():time.sleep(2)
assert 'Passed     : 821' in Path('/tmp/mimir-bmm-iree-lit.log').read_text()
assert '100% tests passed' in Path('/tmp/mimir-bmm-iree-unit.log').read_text()
previous_ir=json.loads(Path('/tmp/mimir-bmm-iree-prototype-ir-sha.json').read_text())
previous_timers=json.loads(Path('/tmp/mimir-bmm-iree-prototype-timer-sha.json').read_text())
changes=[];retime=[]
for t in [1,4,6,8]:
    old=json.loads((prototype/f't{t}/paired-native.json').read_text())
    binaries={}
    same=True
    for row in old['rows']:
        for variant,item in row['identities'].items():
            p=Path(item['binary']); sha=hashlib.sha256(p.read_bytes()).hexdigest()
            binaries[str(p.relative_to(root))]=dict(previous=item['sha256'],current=sha)
            same &= sha==item['sha256']
    for p in (root/f't{t}/results').glob('*_timer.so'):
        sha=hashlib.sha256(p.read_bytes()).hexdigest();key=str(p.relative_to(root))
        binaries[key]=dict(previous=previous_timers[key],current=sha)
        same &= sha==previous_timers[key]
    ir={}
    for p in (root/f't{t}/ir').glob('**/kernel.ll'):
        key=str(p.relative_to(root));sha=hashlib.sha256(p.read_bytes()).hexdigest()
        ir[key]=dict(previous=previous_ir[key],current=sha)
        if sha!=previous_ir[key]:changes.append(key)
    reuse=dict(reused=same,original_source_commit='35ff836d23',current_source_commit=json.loads((root/f't{t}/manifest.json').read_text())['source_commit'],binaries=binaries,ir=ir)
    (root/f't{t}/results/timing-reuse.json').write_text(json.dumps(reuse,indent=2))
    if not same or t==6:retime.append(t)
print(json.dumps(dict(changed_ir=changes,retime=retime)),flush=True)
with open('/tmp/mimir-bmm-iree-audit.log','w') as log:
    subprocess.run(['python3','/tmp/mimir-bmm-iree-audit.py'],stdout=log,stderr=subprocess.STDOUT,check=True)
for t in retime:
    print('timing',t,flush=True)
    with open(f'/tmp/mimir-bmm-iree-run{t}.log','w') as log:
        subprocess.run(['bash','/tmp/mimir-bmm-iree-env.sh','run',str(t)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('finished',t,flush=True)
print('FINAL TIMING COMPLETE',flush=True)
