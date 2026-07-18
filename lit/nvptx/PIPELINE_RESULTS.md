# NVPTX pipeline result

- GPU: NVIDIA A100-PCIE-40GB (compute capability 8.0)
- Test: `lit/nvptx/pipelined_tuple_kernel.mim`
- Correctness: executable returned `41` (input `40`, one async-loaded value plus final increment)
- Cubin: generated for `sm_80`; `ptxas` completed successfully.
- PTX contains `cp.async.ca.shared.global`, `cp.async.commit_group`, and `cp.async.wait_group 0`.

Current TVM-style GEMM baseline (M=N=K=1024, 10 iterations, A100):

- Naive: 1.261 ms, 1703.007 GFLOPS (verification successful)
- Tiled 16x16: 0.851 ms, 2524.784 GFLOPS (verification successful)
- Double buffered: 0.994 ms, 2160.241 GFLOPS (verification successful)
- TVM-style: 0.450 ms, 4775.469 GFLOPS (verification successful)
