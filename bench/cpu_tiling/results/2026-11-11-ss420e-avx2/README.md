# ss420e CPU GEMM：禁用 AVX-512

与前一轮使用同一台 Xeon Silver 4310、同一 CPU 0 单线程、同一 Clang 20.1.8、同一
MimIR LLVM IR、同一 tiling 配置以及完全相同的 A/B 输入。输入目录复用前一轮文件，
二进制和结果保存在独立目录：
`sunqi@ss420e:~/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2`。

## ISA 限制与验证

- MimIR 保持 `-march=icelake-server`，增加 `-mno-avx512f`，没有改变调度参数。
- OpenBLAS 使用相同源码 `8e57c86` 的独立 `TARGET=HASWELL DYNAMIC_ARCH=0 NO_AVX512=1`
  单线程库；初始化后的 P/Q/R 为 320/320/104512。对照轮是 SKYLAKEX 448/448/74432。
- PyTorch 启动前设置 `ATEN_CPU_CAPABILITY=avx2`，运行时返回 `AVX2`；
  MKL 启动前设置 `MKL_ENABLE_INSTRUCTIONS=AVX2`，独立探针的 MKL verbose 头明确报告
  `Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2) enabled processors`。
  设置方法分别见 [PyTorch dispatch 文档](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/README.md)
  和 [Intel ISA dispatch 文档](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/instruction-set-specific-dispatch-on-intel-archs.html)。
- NumPy 参考计算使用 `OPENBLAS_CORETYPE=Haswell` 并关闭其 AVX-512 dispatch feature groups；
  `dispatch-probe.json` 记录其实际 Haswell 库身份和 feature 状态。
- 44 个 MimIR 内核加 Haswell OpenBLAS 共 **45 个二进制**均反汇编检查，EVEX 指令数全部为零。
  完整身份及静态 FMA 计数在 `isa-audit.json`。这是 benchmark 进程与内核的 ISA 限制，
  未改变系统或 BIOS 设置。

`default` 和 `haswell96` 分别保留原先 4×16 默认配置，以及
MR=6、NR=16、MC=KC=320、NC=104512、order=1、register budget=128 的固定配置。
全部 22 形状、四种实现都通过完整 FP64 参考校验（rtol=atol=1e-3）。
计时仍为 5 次 warmup、9 次随机交错批次、每批至少 100 ms，报告中位数。
所有编译与 ISA 探针在正式计时前完成。MimIR 包含输出和临时内存分配/释放，
OpenBLAS 与 PyTorch `mm.out` 使用预分配 C。

## 结果

22 个形状的几何平均：吞吐比以本轮 AVX2 OpenBLAS 为基准；加速比为上轮启用 AVX-512 的耗时除以本轮耗时，大于 1 表示本轮更快。

| 实现 | 相对 AVX2 OpenBLAS | 相对 AVX2 PyTorch | 禁用后的绝对加速比 |
|---|---:|---:|---:|
| MimIR 默认 | 83.25% | 81.28% | 1.255× |
| MimIR 6×16 | 94.64% | 92.40% | 1.214× |
| OpenBLAS | 100.00% | 97.63% | 0.656× |
| PyTorch / MKL | 102.42% | 100.00% | 0.587× |

时间单位 ms；每格为启用 AVX-512 → 禁用 AVX-512，形状按 M×K×N：

| 形状 | MimIR 6×16 | OpenBLAS | PyTorch |
|---|---:|---:|---:|
| 256×256×256 | 0.4804 → 0.3795 | 0.2235 → 0.3672 | 0.2014 → 0.3594 |
| 1024×1024×1024 | 29.6895 → 23.5540 | 13.5024 → 22.9868 | 12.1663 → 22.6133 |
| 8×2304×512 | 0.6663 → 0.5405 | 0.4760 → 0.4373 | 0.2010 → 0.3521 |
| 2000×2000×2000 | 189.9897 → 172.3199 | 97.0940 → 167.9422 | 88.9377 → 169.0915 |
| 2048×2048×2048 | 208.5525 → 186.1774 | 109.8915 → 180.6148 | 93.9872 → 189.4988 |
| 49×4608×512 | 3.4450 → 3.3449 | 2.1912 → 3.1308 | 1.8462 → 2.7263 |
| 97×383×257 | 0.3202 → 0.2664 | 0.1419 → 0.2281 | 0.1317 → 0.2277 |
| 1999×4095×2047 | 459.7388 → 366.3397 | 215.6671 → 358.9118 | 200.7006 → 363.7820 |

相对吞吐的改善同时包含 MimIR 本身的绝对耗时变化，以及 OpenBLAS/MKL 切换为 AVX2 内核后的基线变化。两轮未锁频，尚未用硬件计数器区分频率、累加依赖链和代码调度各自的贡献。

本轮 6×16 的绝对吞吐几何平均提高 **21.43%**，相对 OpenBLAS 从上一轮的 **51.14%**
提高到 **94.64%**。OpenBLAS 与 PyTorch 的耗时几何平均分别增加至原来的约
1.524×、1.703×。这支持当前固定微内核更适合 AVX2 的结论，同时说明相对比例的提高
并非全部来自 MimIR 的绝对加速；AVX-512 配置仍需单独优化。

## 复现

```sh
ssh sunqi@ss420e
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 OPENBLAS_CORETYPE=Haswell \
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL \
DNNL_MAX_CPU_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_VERBOSE=0 \
  ~/zc/mimir-bench/gemm-bench-gpu-env/bin/python \
  ~/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2/remote_cpu.py run --isa avx2 \
  --root ~/zc/mimir-bench/cpu-gemm-20261111-fc398e36-avx2 \
  --openblas ~/zc/mimir-bench/OpenBLAS-8e57c86-haswell/libopenblas.so
```

同一命令将 `run` 改为 `prepare` 可重新编译/验证。runner 会检查目标参数的缓存键和 ISA，
防止误用上一轮 AVX-512 二进制。构建选项和 runner 哈希见 `deployment.json`。
