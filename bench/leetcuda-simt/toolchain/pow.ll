target triple = "nvptx64-nvidia-cuda"

declare float @llvm.pow.f32(float, float)
declare double @llvm.pow.f64(double, double)

define ptx_kernel void @pow_probe(ptr addrspace(1) %out, float %x, float %y) {
    %value = call float @llvm.pow.f32(float %x, float %y)
    store float %value, ptr addrspace(1) %out
    ret void
}

define ptx_kernel void @pow_probe_f64(ptr addrspace(1) %out, double %x, double %y) {
    %value = call double @llvm.pow.f64(double %x, double %y)
    store double %value, ptr addrspace(1) %out
    ret void
}
