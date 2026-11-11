# 真正的 origin/master 基线与优化分支对照

此前报告中的“MimIR 默认”指 **优化分支使用默认参数**，并非 `origin/master`。
本轮补齐真实上游基线，并在同一次交错测量中重新测量优化分支的两套参数。

| 结果键 | 源码 | 调度配置 |
|---|---|---|
| `origin_master` | `origin/master` `19d6c2de627cf060655c3a811d459cb4a039c406` | 原版默认，无本分支补丁或参数 override |
| `branch_default` | 优化源码 `124047b43e7d7c89e303899c5d97f9a92dbf7899` | multi-level/MC/packing、尾块与行 SSA 修复，默认参数 |
| `branch_6x16` | 同一优化源码 `124047b43e` | 显式 MR=6、NR=16、MC=KC=320、NC=104512、order=1、register budget=128 |

`origin/master` 提交已用 `git ls-remote origin refs/heads/master` 核对。
源码在独立 worktree 构建，`git status --porcelain` 为空；与优化编译器一样使用 Debug 构建。
它的 `kernel.mim` 与前面优化分支默认的 gemm-bench 输入文件逐字节相同。
IR 哈希和版本在 `manifest.json`，`kernel.mim` 程序哈希及原版编译命令在 `lowering-commands.json`。
**上游到本分支的累计差距包含此前已有 BLIS 路径和后续改进，不能全部归因于最近的 MC、尾块或 SSA 修复。**

远端目录：`sunqi@ss420e:~/zc/mimir-bench/cpu-master-comparison-20261111`。
仍是 Xeon Silver 4310 的 CPU 0 单线程；全部使用远端 Clang 20.1.8
`-O3 -march=icelake-server -mno-avx512f`，以及相同的输入文件。
OpenBLAS 是同一 Haswell 0.3.33.dev 库；PyTorch 2.2.1 的 MKL/ATen 均限制为 AVX2。
原生内核时间包含 MimIR 的输出/内部临时分配与释放，不包含编译、Python 调用或输入复制。

22 个形状、五种实现全部与完整 FP64 参考比较（rtol=atol=1e-3）。
5 次 warmup，9 个随机交错批次，每批至少 100 ms，报告中位数。
本轮按形状和实现先校准迭代次数，每批检查满足 100 ms 下限，以避免对慢速 master 内核重复进行每批校准。
该策略对同次测量的全部实现一致，计时函数和分配口径未变；`calibrate_once=true` 保存在结果 JSON 中。

## 结果

22 个形状的吞吐比几何平均：

| 实现 | 相对原版 master | 相对优化分支默认 | 相对 OpenBLAS |
|---|---:|---:|---:|
| 原版 origin/master | 1.000× | 0.312× | 26.06% |
| 优化分支，默认参数 | 3.204× | 1.000× | 83.50% |
| 优化分支，显式 6×16 等参数 | 3.652× | 1.140× | 95.18% |

中位时间，单位 ms，形状按 M×K×N：

| 形状 | origin/master | 优化分支默认 | 优化分支显式参数 | OpenBLAS | PyTorch |
|---|---:|---:|---:|---:|---:|
| 256×256×256 | 0.7788 | 0.5338 | 0.3783 | 0.3674 | 0.3599 |
| 1024×1024×1024 | 129.8511 | 28.1239 | 23.5305 | 23.0773 | 22.7127 |
| 8×2304×512 | 1.1130 | 0.4744 | 0.5098 | 0.4360 | 0.2967 |
| 2048×2048×2048 | 1143.4406 | 227.8844 | 187.4083 | 181.1807 | 190.3068 |
| 49×4608×512 | 13.5716 | 3.5289 | 3.3121 | 3.1242 | 2.6846 |
| 97×383×257 | 0.7359 | 0.2874 | 0.2639 | 0.2285 | 0.2279 |
| 1999×4095×2047 | 4005.1531 | 439.8140 | 365.9593 | 359.5188 | 365.0412 |

显式参数仍没有进入默认选择器；默认列已经包含源码优化，不能把默认到显式参数的提升理解为从 master 开始的全部收益。

## 复现

```sh
ssh sunqi@ss420e
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 OPENBLAS_CORETYPE=Haswell \
NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL \
DNNL_MAX_CPU_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_VERBOSE=0 \
  ~/zc/mimir-bench/gemm-bench-gpu-env/bin/python \
  ~/zc/mimir-bench/cpu-master-comparison-20261111/remote_cpu.py run \
  --isa avx2 --calibrate-once \
  --root ~/zc/mimir-bench/cpu-master-comparison-20261111 \
  --openblas ~/zc/mimir-bench/OpenBLAS-8e57c86-haswell/libopenblas.so
```

同一条命令把 `run` 改为 `prepare` 可重新编译/校验。
旧报告保留原始 JSON 键，已补上“优化分支默认参数”的明确说明；本轮键名直接区分源码与参数。
