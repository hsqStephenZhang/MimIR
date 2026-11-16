import subprocess, time, sys, statistics
W='.'
tests=['tile_outline_panel','tile_pipeline','tile_reduce','law_run','law_promote','law_fuse','tile_mma','tile_wgmma_tma','tile_resident','law_run_items','tile_attention','law_blocks']
mims={'base':sys.argv[1]+'/build/bin/mim','now':W+'/build/bin/mim'}
tot={k:0.0 for k in mims}
print(f"{'test':22s} {'base s':>8s} {'now s':>8s} {'ratio':>6s}")
for t in tests:
    row={}
    for k,m in mims.items():
        src=(sys.argv[1] if k=='base' else W)+f'/lit/gpu/{t}.mim'
        ts=[]
        for _ in range(3):
            s=time.perf_counter()
            r=subprocess.run([m,'-p','opt','-X','tensor:tile-fuse=1',src,'-p','gpu','-o','/dev/null'],capture_output=True)
            ts.append(time.perf_counter()-s)
        row[k]=(statistics.median(ts), r.returncode)
        tot[k]+=row[k][0]
    print(f"{t:22s} {row['base'][0]:8.2f} {row['now'][0]:8.2f} {row['now'][0]/row['base'][0]:6.2f}" + ("" if row['base'][1]==row['now'][1]==0 else f"  rc {row['base'][1]}/{row['now'][1]}"))
print(f"{'total':22s} {tot['base']:8.2f} {tot['now']:8.2f} {tot['now']/tot['base']:6.2f}")
