import ctypes as C,itertools,json,sys
from pathlib import Path
root=Path(sys.argv[1]);info=json.loads((root/'launch.json').read_text());m,n,k=info['shape']
lib=C.CDLL('libcuda.so.1');P=C.c_void_p;U=C.c_uint64;I=C.c_int;S=C.c_size_t
def api(name,args):
 f=getattr(lib,name);f.argtypes=args;f.restype=I
 def call(*values):
  code=f(*values)
  if code:raise RuntimeError(f'{name}: CUDA error {code}')
 return call
init=api('cuInit',[C.c_uint]);retain=api('cuDevicePrimaryCtxRetain',[C.POINTER(P),I]);release=api('cuDevicePrimaryCtxRelease_v2',[I]);setctx=api('cuCtxSetCurrent',[P]);sync=api('cuCtxSynchronize',[])
load=api('cuModuleLoad',[C.POINTER(P),C.c_char_p]);unload=api('cuModuleUnload',[P]);getfn=api('cuModuleGetFunction',[C.POINTER(P),P,C.c_char_p]);alloc=api('cuMemAlloc_v2',[C.POINTER(U),S]);free=api('cuMemFree_v2',[U]);htod=api('cuMemcpyHtoD_v2',[U,P,S]);dtoh=api('cuMemcpyDtoH_v2',[P,U,S]);memset=api('cuMemsetD8_v2',[U,C.c_ubyte,S]);meminfo=api('cuMemGetInfo_v2',[C.POINTER(S),C.POINTER(S)])
launch=api('cuLaunchKernel',[P]+[C.c_uint]*7+[P,C.POINTER(P),C.POINTER(P)])
a=(C.c_float*(m*k))(*[(i%7-3)*.5 for i in range(m*k)]);b=(C.c_float*(k*n))(*[(i%5-2)*.25 for i in range(k*n)]);c=(C.c_float*(m*n))();reference=[sum(a[i*k+t]*b[t*n+j] for t in range(k)) for i in range(m) for j in range(n)]
ctx=P();mod=P();fn=P();buffers=[]
init(0);retain(C.byref(ctx),0);setctx(ctx)
try:
 load(C.byref(mod),str(root/'kernel.cubin').encode());getfn(C.byref(fn),mod,info['kernel'].encode())
 for _ in range(3):
  ptr=U();alloc(C.byref(ptr),max(m*k,k*n,m*n)*4);buffers.append(ptr)
 orders=[[0,1,1,0,2]] if info['name']=='tile_panel_tail' else itertools.permutations(range(3))
 for order in orders:
  htod(buffers[0],a,C.sizeof(a));htod(buffers[1],b,C.sizeof(b));memset(buffers[2],0,C.sizeof(c))
  args=(U*len(order))(*[buffers[i].value for i in order]);params=(P*1)(C.addressof(args))
  launch(fn,info['groups'],1,1,info['items'],1,1,info['smem'],None,params,None);sync();dtoh(c,buffers[2],C.sizeof(c))
  if all(abs(v-r)<.001 for v,r in zip(c,reference)):break
 else:raise AssertionError('No argument ordering produced correct output')
 for _ in range(20):launch(fn,info['groups'],1,1,info['items'],1,1,info['smem'],None,params,None)
 sync();before=S();total=S();meminfo(C.byref(before),C.byref(total))
 for _ in range(1000):launch(fn,info['groups'],1,1,info['items'],1,1,info['smem'],None,params,None)
 sync();after=S();meminfo(C.byref(after),C.byref(total));dtoh(c,buffers[2],C.sizeof(c))
 assert all(abs(v-r)<.001 for v,r in zip(c,reference))
 assert after.value==before.value,(before.value,after.value)
 print(json.dumps(dict(case=info['name'],launches=1000,argument_map=list(order),correct=True,free_before=before.value,free_after=after.value,device_memory_growth=before.value-after.value)),flush=True)
finally:
 for ptr in buffers:free(ptr)
 if mod.value:unload(mod)
 release(0)
