import os,sys,json,time
sys.path.insert(0,'/workspaces/compile/MimIR/.worktrees/gpu-tiling/bench/cpu_tiling')
from attention_cpu import MASKS
t=int(sys.argv[1]);os.sched_setaffinity(0,MASKS[t])
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
torch.set_num_threads(t);torch.set_num_interop_threads(1)
p='/tmp/mimir-attention-local/inputs/h8_s128_d64/'
q,k,v=[torch.from_numpy(np.load(p+n+'.npy')).unsqueeze(0) for n in ['q','k','v']]
os.sched_setaffinity(0,{0})
with torch.inference_mode():
 r=Timer('F.scaled_dot_product_attention(q,k,v)',globals=dict(F=F,q=q,k=k,v=v),num_threads=t).blocked_autorange(min_run_time=.3)
 print(json.dumps(dict(threads=t,median=r.median,iqr=r.iqr)))
