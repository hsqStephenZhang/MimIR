# 参考 IREE 的本机 CPU batch GEMM 优化

日期：2026-11-12。本机 i5-14600KF，连续 FP32 NN，15 个形状，1/4/6/8 线程。
旧、新 MimIR 内核与 Torch、OpenBLAS batch API、OpenBLAS SGEMM 循环在同一进程交错计时。
优化继续使用现有多层 tiling、NC/KC/MC packing 和 6×16 向量微内核。

主要结果：singleton batch 已从标量慢路径恢复到向量微内核并支持并行；
6 线程的 13 个真实 batch 用例几何平均提升 8.9%，达到同轮 Torch 的约 93%。
`B=8, M=K=N=512` 的 6 线程耗时由 5.52 ms 降到 2.67 ms。

## IREE 参考与实际改动

参考 checkout `/workspaces/ml-compiler/iree`，提交
`7a8b0039c12d9df0e47cd5d6a9784b28655c2625`；引用文件的内容哈希见 `source-state.json`。

- `GlobalOptimization/Passes.cpp:156` 在调度前组织 unit-extent dimension folding；
  `DispatchCreation/FoldUnitExtentDims.cpp:400` 调用 Linalg 的单位维折叠模式，并配套更新 shape/映射。
  MimIR 的适配落在 **affine 逆像证明**：正向值域可以把长度 1 的轴折成零，
  但求逆像时，恒等投影仍需约束这个轴。此前将它误判成常量，再尝试证明零落在符号 box 内，
  使 tile-of 失去 exactness；现在保留直接投影的线性系数。真正的常量写映射仍保持原有保守判定。
  Tensor 的外部形状、融合 bias 与内存所有权均保持原有表示。
- `Codegen/LLVMCPU/KernelDispatch.cpp:1640` 联合分配 batch/M/N、将 batch tile 设为 1，
  分离 distribution/cache/vector 各层；`:476` 以约两倍 runtime threads 控制过细的任务分配。
  MimIR 的适配是在 batch 本身少于每线程两张矩阵时继续搜索 M/N 网格，考虑最坏线程的工作量，
  用输入遍历量打破平局。已有两轮及以上整矩阵任务时保留整矩阵，以控制 packing 代价。
  `B=8,T=6` 由 8 个整矩阵任务变为 24 个 M 子块任务，每线程 4 个任务。
  这是对 IREE 粒度控制思想的适配，并非复制它的 tile 参数或严格限制总任务数为 2T。
- IREE 的 `GlobalOptimization/ConvertBatchMatmulToMatmul.cpp:139` 还会将广播共享 RHS 的
  batch matmul 折为大 matmul。本次测量中的 RHS 在各 batch 独立，该变换的前提不成立；
  本次优化保留独立 batch 的语义。

源码提交：`35ff836d23`（单位轴逆像、batch 联合网格及数值/内存回归），
`bb909053b4`（长 batch 粒度限制及网格回归）。
benchmark 提交 `a4ea98d9d5` 修复了 IR 改变后错误复用旧 `.so` 的缓存判定，
`compile.json` 现在记录并检查 `ir_sha256`。

## 成对测量汇总

性能比 = 对照耗时 / 新 MimIR 耗时，越大越好；每个形状等权取几何平均。
“加速”相对同一 backend 的单线程，不是相对旧编译器。

| 线程 | 全部15例：新/旧 | 全部15例：新/Torch | batch>1：新/旧 | batch>1：新/Torch | batch>1：新/BLAS循环 | batch>1：新/BLAS batch | 新MimIR加速(batch>1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.544× | 0.868× | 1.000× | 0.858× | 0.985× | 2.483× | 1.00× |
| 4 | 1.780× | 0.869× | 1.000× | 0.864× | 1.558× | 2.545× | 2.87× |
| 6 | 2.005× | 0.930× | 1.089× | 0.934× | 1.719× | 2.809× | 3.68× |
| 8 (6P+2E) | 1.695× | 0.834× | 1.000× | 0.829× | 1.351× | 2.201× | 2.05× |

全部用例的改善受两个 singleton 用例大幅提升影响，故另列 batch>1 分组。
1/4/8 线程下，batch>1 的调度基本保持原状；本次额外的多核收益主要发生在 6 线程的非整除 batch。

## 代表用例

形状顺序 B×M×K×N，单位 ms。完整 60 行数据及 GFLOP/s 见 `summary.csv`。

| 线程 | B×M×K×N | 旧MimIR | 新MimIR | Torch | BLAS循环 | BLAS batch | 旧/新 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1×512×512×512 | 61.1626 | 1.8024 | 1.6919 | 1.6833 | 1.6883 | 33.93× |
| 4 | 1×512×512×512 | 61.2548 | 0.4635 | 0.4363 | 0.4481 | 1.6869 | 132.15× |
| 6 | 1×512×512×512 | 60.9598 | 0.3078 | 0.3029 | 0.3182 | 1.6871 | 198.03× |
| 6 | 8×256×256×256 | 0.4518 | 0.3268 | 0.4324 | 0.3846 | 0.4399 | 1.38× |
| 6 | 8×512×512×512 | 5.5170 | 2.6725 | 3.5768 | 2.6014 | 3.7498 | 2.06× |
| 6 | 64×16×16×16 | 0.0085 | 0.0085 | 0.0051 | 0.0091 | 0.0422 | 1.00× |
| 6 | 16×97×383×257 | 0.4337 | 0.4342 | 0.4061 | 0.5549 | 0.4087 | 1.00× |
| 6 | 16×8×2304×512 | 3.9224 | 3.9802 | 3.6453 | 3.9987 | 3.5637 | 0.99× |
| 8 | 1×512×512×512 | 61.6132 | 0.6986 | 0.6359 | 0.6605 | 1.6858 | 88.20× |

`1×512³` 单线程反汇编从 4096 条标量 FMA / 0 条 packed FMA，变为
0 条标量 FMA / 256 条 packed FMA（静态指令条数，不是运行时 FLOP 计数）。
4/6/8 线程下 singleton 每次 kernel 一次 launch，线程审计确认全部指定核参与。

## 粒度限制的消融

第一轮将 M/N 搜索扩展到所有 batch，6 线程部分长 batch 出现额外 packing 回退。
加入“已有两轮整矩阵任务即保留整矩阵”的限制后重测 6 线程。
以下列出第一轮与最终版中位数，单位 ms；两轮分别保留成对 baseline，
属于两次运行的消融参考，不宜将小幅差异当作稳定加速。

| 形状 | 不限制长batch拆分 | 最终策略 | 最终旧内核对照 |
|---|---:|---:|---:|
| 8×256×256×256 | 0.3280 | 0.3268 | 0.4518 |
| 8×512×512×512 | 2.6717 | 2.6725 | 5.5170 |
| 64×64×64×64 | 0.0544 | 0.0508 | 0.0507 |
| 16×97×383×257 | 0.4762 | 0.4342 | 0.4337 |
| 16×8×2304×512 | 4.2653 | 3.9802 | 3.9224 |
| 16×49×512×512 | 0.5632 | 0.6277 | 0.6306 |

第一轮原始 6 线程数据见 `ablation-t6-unbounded-split.json`，对应源码 `35ff836d23`。
最终策略中，13 例真正 batch 的新/旧性能比几何平均为 1.000×（1线程）、1.089×（6线程）。
粒度限制也放弃了部分长 batch 的潜在收益，例如上表的 `16×49×512×512`；
当前选择优先保留整矩阵复用，仍可继续改进逐形状成本模型。
瘦矩阵及很小矩阵仍落后于 Torch；本次没有优化它们的专用微内核或阈值。

## 测量、校验与版本

- 本机 i5-14600KF，不使用 SMT 同胞。1线程 CPU 0；4线程 0,2,4,6；
  6线程 0,2,4,6,8,10；8线程 0,2,4,6,8,10,12,13，即 **6P+2E**。
  当前任务按数量静态分配，不按 P/E 核速度加权，8线程仍可能慢于6个P核。
- Clang 22，Haswell AVX2/FMA，`-O3 -march=haswell -mno-avx512f`；
  MR=6、NR=16、MC=KC=320、NC=104512。
- Torch 2.12.0+cpu / MKL 2024.2，原生 `aten::bmm.out`。
  OpenBLAS 0.3.33.dev / Haswell / pthread，源码
  `8e57c8667da06cb55a777975b6c2435c9b304079`。
  两个 BLAS 对照分别调用 `cblas_sgemm_batch_strided` 和 C 循环 `cblas_sgemm`，
  后者每个 GEMM 使用同一线程预算。小矩阵的原生 batch API 慢路径原因见上一份报告，
  因此两个对照同时保留。
- MimIR/Torch 共用 Torch wheel 的 libgomp；所有数值库在导入前设置相同线程预算，
  OpenMP 明确绑核，OpenBLAS 逐 worker 设置并校验 affinity。
  PASSIVE/GOMP_SPINCOUNT=0/OPENBLAS_THREAD_TIMEOUT=4 防止闲置线程池竞争。
- 先完成构建、回归、lowering/native compilation、完整输出 FP64 校验和线程审计，再计时。
  5个 backend 各预热5次；9轮固定种子打乱顺序，每轮原生循环至少100 ms，中位数统计。
  60个形状/线程组合，300组 backend 测量，共2700个样本；不删除异常值。
  governor 为 powersave，保留正常睿频，未锁频；样本范围较大的组见 `timing-variation.json`。
- 旧/新 MimIR 均包含输出和内部 scratch 的分配/释放；Torch/BLAS 预分配输出。
  全部排除 Python 和输入/验证输出复制。输入与前一份报告完全相同，验证 rtol=atol=1e-3。
- 最终版重新生成全部 IR 并重新验证所有内核。1/4/8线程的30个 MimIR产物及两个 baseline
  timer逐线程与第一轮 **二进制哈希完全一致**，沿用该轮成对测量；6线程发生变化并完整重测。
  每个 tN 的 `timing-reuse.json` 保留 IR/二进制前后哈希，以及原始/当前 source commit。
- 最终源码状态 `a4ea98d9d5`，C++实现为 `35ff836d23` + `bb909053b4`；
  旧内核 IR 来自 `eabf14aabf`。gemm-bench 独立本地分支 `feat/cpu-bmm`，提交 `ffcc8af`。
  每个 tN 目录均保存 manifest、编译命令、IR哈希、实际binary哈希及原始9轮数据。
  `source-state.json` 记录源码/编译器/插件哈希。
- 260项单元测试通过；完整 lit **821通过、8不支持**。
  batch 回归含 NN/NT/TN、singleton batch、4D prefix中的单位轴、M=1/N=1、融合bias，
  serial/reference及 OpenMP 4/8线程 ASan/LSan 通过；仅抑制既有 libomp 176字节泄漏。
  新网格回归验证 B=8/12/16、T=6 时的任务数为24/12/16。
- 所有300组 backend 输出通过 FP64 校验；120个旧/新 MimIR `.so` 审计无 EVEX/ZMM。
  batch=1/3/8 的4/6/8线程各20次调用审计全部 worker 与CPU绑定，均为每调用1个launch。
  `B=64,M=N=K=16` 低于默认1,048,576 MAC阈值，仍保持串行。

本次性能范围是连续 FP32 NN、独立 RHS batch；未测 GPU、广播 batch、任意 stride 或转置布局性能。
这些结果不意味着已为所有 batch 大小及 CPU 型号选出最优参数。

## 复现

本报告的 `prepare.py`、`env.sh`、`audit.py` 和 `finish-final.py` 保留实际实验脚本与绝对路径。
复现依赖 `/tmp/gemm-bench-cpu-bmm`（提交 ffcc8af）以及
`/tmp/mimir-bmm-local-20261112` 的旧版 IR/输入；也可从旧提交 eabf14aabf 重建旧IR。

1. 将本机 MimIR 构建到上文最终源码状态，运行对应单元与 lit 测试。
2. 将 `env.sh` 放到 `/tmp/mimir-bmm-iree-env.sh`，运行 `python3 prepare.py`。
   该脚本使用64 MiB编译器栈，生成旧/新两个 variant，并运行 native compilation 和数值校验。
3. 按脚本路径运行 `audit.py`。确认所有编译/回归进程结束后，依次
   `bash /tmp/mimir-bmm-iree-env.sh run 1`、`run 4`、`run 6`、`run 8` 即可重新计时全部配置。
   `finish-final.py` 是本次检查哈希后仅重测受影响配置的记录，复现可直接重测全部配置。
