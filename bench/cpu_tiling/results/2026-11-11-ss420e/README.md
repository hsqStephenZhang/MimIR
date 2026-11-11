# ss420e CPU GEMM（2026-11-11）

**基线说明：本报告的 `default`/“MimIR 默认”是已包含本次优化的分支默认参数，不是 `origin/master`；`haswell96` 是同一修复源码的显式参数配置。**

远端：`sunqi@ss420e:/home/sunqi/zc/mimir-bench/cpu-gemm-20261111-fc398e36`。
CPU 为 2× Xeon Silver 4310（Ice Lake），本轮仅测 **CPU 0 单线程**，SMT sibling 为 CPU 24，
不是双路/多线程吞吐。没有设置锁频或修改系统库。

使用 `feat/cpu-multilevel-tiling` 的 `fc398e36b7`（内核改动为 `124047b43e`），
复用其 CPU LLVM IR，由远端 Clang 20.1.8 按 `-O3 -march=icelake-server` 重新编译；
计时全部在远端原生执行。IR 哈希和配置在 `manifest.json`，二进制哈希在 `paired-native.json`。
原生循环、正确性调用及基线均使用 `gemm-bench` `feat/mimir-nominal-tensor` 的 adapters/timers。

- `default`：修复后的通用默认配置（与本地测量相同的 4×16 调度参数）。
- `haswell96`：迁移本地 6×16 配置，MR=6、NR=16、MC=KC=320、NC=104512、order=1，
  register budget=128。**本轮未为 AVX-512 重新调参。**
- `openblas`：与本地同一源码版本 `8e57c86`（0.3.33.dev），远端编译 `TARGET=SKYLAKEX`
  的单线程 SGEMM 库。初始化后的 P/Q/R 为 **448/448/74432**。
- `pytorch`：远端现有 PyTorch 2.2.1+cu121 的 CPU `mm.out`，MKL 2022.2，单线程。

系统原有 OpenBLAS 0.3.8 的动态检测返回 **Prescott**，因此未将它用作正式基线；
新库位于独立目录 `~/zc/mimir-bench/OpenBLAS-8e57c86-skylakex`，保留系统环境。
新库在第一次 GEMM 前导出 R=128，首次调用初始化后为 74432，元数据同时记录两者。

全部 22 个形状、四种实现都通过完整 FP64 matmul 参考校验（rtol=atol=1e-3）。
输入为 float32 uniform(-1,1)，seed 为 `np.random.SeedSequence([42,M,N,K])`；
所有实现在同一形状使用完全相同的 A/B，原始输入保存在远端。
编译及 prepare 验证全部完成后才开始测量；5 次 warmup、9 次随机交错批次，
每批至少 100 ms，报告中位数。MimIR 包含输出/内部临时分配及释放，OpenBLAS 和
PyTorch `mm.out` 使用预分配 C，排除 Python 和输入复制。

本地 i5-14600KF 使用 Clang 22、PyTorch 2.12/MKL，而本轮是 Clang 20、PyTorch 2.2/MKL。
因此这次检验的是上述固定调度配置迁移到 Ice Lake 的效果，不是仅更换 CPU 的受控实验，
也不能把本地 97.1% 的相对 OpenBLAS 吞吐直接推广到这台服务器。

## 结果

22 个形状的吞吐比几何平均，OpenBLAS=100%：

| 实现 | 相对 OpenBLAS | 相对 PyTorch |
|---|---:|---:|
| MimIR 优化分支默认 4×16 | 43.54% | 38.04% |
| MimIR 固定 6×16 | 51.14% | 44.69% |
| PyTorch / MKL | 114.44% | 100.00% |

单个形状中位时间，单位 ms，形状按 M×K×N：

| 形状 | MimIR 优化分支默认 | MimIR 6×16 | OpenBLAS | PyTorch |
|---|---:|---:|---:|---:|
| 256×256×256 | 0.6464 | 0.4804 | 0.2235 | 0.2014 |
| 1024×1024×1024 | 38.1648 | 29.6895 | 13.5024 | 12.1663 |
| 8×2304×512 | 0.6686 | 0.6663 | 0.4760 | 0.2010 |
| 2048×2048×2048 | 301.7038 | 208.5525 | 109.8915 | 93.9872 |
| 2048×512×128 | 4.9326 | 4.3154 | 1.8129 | 1.5167 |
| 49×4608×512 | 4.6635 | 3.4450 | 2.1912 | 1.8462 |
| 97×383×257 | 0.3516 | 0.3202 | 0.1419 | 0.1317 |
| 1999×4095×2047 | 589.4636 | 459.7388 | 215.6671 | 200.7006 |

固定 6×16 相对通用默认的吞吐几何平均加速为 1.175×。

## 代码生成核对与结论边界

1024³ 的远端二进制反汇编确认，默认和 6×16 的 FMA 都使用 **512-bit ZMM**；
检查到的 FMA 内循环没有 SIMD `rsp` 栈访问。默认主循环有 4 个 ZMM 累加器，
6×16 主循环有 6 个（4 行尾块为 4 个）。相同的 6×16 在 AVX2 目标下可拆成
12 个 YMM 累加器，迁移到 AVX-512 后独立累加链数量发生了变化，不能只按 cell 总数
判断微内核是否匹配。静态循环分析见 `codegen.json`。
因此这次慢速不能直接解释为没有启用 AVX-512，或沿用上次已修复的 SIMD spill。

本机 OpenBLAS SKYLAKEX 源码实际含 16×24 的计算路径（row-major 逻辑上对应 24×16），
其宏观 P/Q/R 也与本地 Haswell 不同。迁移的 6×16 配置尚未匹配这套 AVX-512 微内核与
cache 参数；本轮没有通过单变量实验把全部剩余差距归因于其中某一点。
结果说明通用选择器与本地固定参数尚不具备跨架构的接近 BLAS 性能，
不能用本地单台机器的 97.1% 作为跨 CPU 性能结论。

复现计时（需要远端现有 IR、二进制和输入）：

```sh
ssh sunqi@ss420e
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ~/zc/mimir-bench/gemm-bench-gpu-env/bin/python \
  ~/zc/mimir-bench/cpu-gemm-20261111-fc398e36/remote_cpu.py run \
  --root ~/zc/mimir-bench/cpu-gemm-20261111-fc398e36 \
  --openblas ~/zc/mimir-bench/OpenBLAS-8e57c86-skylakex/libopenblas.so
```

`remote_cpu.py prepare` 可复用 manifest 中的 IR 重新构建原生 timer、校验二进制及全部数值；
脚本位于本目录上两层。OpenBLAS 的构建选项及源码 archive 哈希在 `deployment.json`。
