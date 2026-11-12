# ss420e CPU 多核 GEMM：multi-level tiling + OpenMP

源码为 `feat/cpu-multilevel-omp` 的 `3b066da3d404f6d12c0d659897985dedb4d66309`，
基于此前优化分支，移植 `origin/feature/omp` 的 `277874c4756a` outlining/runtime。
本轮测量固定 6×16、MC=KC=320、NC=104512、order=1、register budget=128。
这里没有把“分支默认参数”标为 `origin/master`；1 核是同一实现的新鲜基线。

## 结果

全部 22 形状的几何平均。相对吞吐 = 对照耗时 / MimIR 耗时，大于 1 表示 MimIR 更快；
加速比按每个实现自己的本轮单核计算。

| 核数 | MimIR 加速 | OpenBLAS 加速 | PyTorch 加速 | MimIR / OpenBLAS | MimIR / PyTorch |
|---|---:|---:|---:|---:|---:|
| 1 | 1.000× | 1.000× | 1.000× | 0.933× | 0.923× |
| 4 | 3.053× | 3.020× | 3.257× | 0.943× | 0.865× |
| 8 | 4.473× | 4.394× | 4.624× | 0.950× | 0.893× |

较大的 7 个 GEMM（sq512、sq1000、sq1024、sq2000、sq2048、odd2048、big_odd）：

| 核数 | MimIR 加速 | MimIR / OpenBLAS | MimIR / PyTorch |
|---|---:|---:|---:|
| 4 | 3.649× | 0.921× | 0.922× |
| 8 | 5.945× | 0.905× | 0.888× |

部分代表性耗时（ms）；形状按 M×K×N：

| 形状 | MimIR 1核 | MimIR 4核 | OpenBLAS 4核 | PyTorch 4核 | MimIR 8核 | OpenBLAS 8核 | PyTorch 8核 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128×128×128 | 0.0515 | 0.0257 | 0.0232 | 0.0226 | 0.0245 | 0.0236 | 0.0236 |
| 256×256×256 | 0.3790 | 0.1207 | 0.1148 | 0.1059 | 0.0847 | 0.0800 | 0.0739 |
| 1024×1024×1024 | 23.7545 | 6.6267 | 5.9831 | 5.9668 | 4.1362 | 3.6052 | 3.5610 |
| 8×2304×512 | 0.5831 | 0.2235 | 0.1544 | 0.1155 | 0.1359 | 0.1019 | 0.0641 |
| 2048×2048×2048 | 188.6226 | 51.8529 | 47.7449 | 49.3913 | 31.8227 | 28.3337 | 28.5444 |
| 49×4608×512 | 3.3343 | 0.9999 | 1.0114 | 0.8483 | 0.6129 | 0.6865 | 0.5038 |
| 97×383×257 | 0.2620 | 0.1087 | 0.0833 | 0.0846 | 0.0930 | 0.0645 | 0.0573 |
| 1999×4095×2047 | 372.4320 | 106.6145 | 94.4632 | 94.6808 | 67.0496 | 56.0423 | 56.1112 |

## 本次定位并修复的问题

初步接入直接把线程尾块 extent 作为动态 clamp 传入内层，破坏了静态 packing/微内核的专门化。
现在在每个线程 tile 入口分派 full/edge 路径，分别向串行层传入常量 extent。
增加了 LLVM 检查，保证测试中没有动态大小的临时 buffer 分配。

| 形状 | 核数 | 修复前 ms | 修复后 ms | 修复带来的加速 |
|---|---:|---:|---:|---:|
| 1999×4095×2047 | 4 | 3012.0517 | 106.6145 | 28.25× |
| 97×383×257 | 4 | 0.2874 | 0.1087 | 2.64× |
| 1999×4095×2047 | 8 | 1873.6173 | 67.0496 | 27.94× |
| 97×383×257 | 8 | 0.2081 | 0.0930 | 2.24× |

`initial/` 保留原始数据；`edge-codegen.json` 记录动态分配数从 6–12 降为 0。
修复后单核 LLVM IR 全部未变；性能变化来自线程尾块 lowering。

当前多核扩展已接近对照库，但不等于调度参数已经最优。默认网格按最坏 tile 工作量优先，
packing 量仅用于并列决胜，见 `thread-grids.json`；例如大奇数形状选择 N-only 切分，会复制
各线程的 A packing。后续可优化工作量/packing 的权衡、显式共享 panel、协作 packing 及
小矩阵的线程阈值。这些剩余项目的独立性能贡献尚未测量。

## 接入与测量方式

- 新增 `TileLevels#Threads` / `tensor.thread_par`，在串行 NC/KC/MC、packing、归约、寄存器层外切分 M/N 输出；K 不在线程间归约。
- 各线程写不相交的输出 tile，并拥有自己的临时 panel/fold buffer；不把 scratch hoist 到线程边界外。
- 自动策略仅覆盖静态 2D contraction；默认少于 1,048,576 MAC 保持串行，本轮 sq64 不启动线程组。显式 Threads 支持 capped 尾块，并在进入串行层前专门化 full/partial extent；动态输出 extent、嵌套线程层和线程/GPU 混合绑定尚未支持。
- 本轮是独立输出 tile 的并行调度。还未加入 BLAS 团队内共享 packed B、跨线程协作 packing 或独立的多核参数搜索。
- Xeon Silver 4310，CPU 12；4 核 CPU 12–15，8 核 CPU 12–19；同一 NUMA 节点，无 SMT。
- 所有实现均限制 AVX2。MimIR 使用 Clang 20 `-O3 -march=icelake-server -mno-avx512f`；OpenBLAS 使用源码 `8e57c86`、HASWELL、pthread 多线程独立构建；PyTorch 2.2.1 CPU `mm.out` / MKL 2022.2 设置 AVX2 dispatch。
- 使用 gemm-bench `feat/mimir-nominal-tensor`（基线 d562d65）的 CPU 数据、算子生成器及三个 native timer；保留原工作区 GPU 改动。远端目录 `sunqi@ss420e:~/zc/mimir-bench/cpu-omp-20261112/static`。
- MimIR 输出和内部 allocation/free 包含在计时内；OpenBLAS 与 PyTorch 使用预分配 C，均排除 Python 调用、输入复制和正确性结果复制。
- 5 次预热，9 轮随机交错，native 批次每批至少 100ms，报告中位数；1/4/8 核顺序运行。全部编译、worker/ISA 审计在计时前完成。机器未隔离、未锁频。
- 66 个 MimIR 内核和全部对照通过 FP64 完整输出校验（rtol=atol=1e-3）。`isa-audit.json` 检查 66 个内核及 OpenBLAS，EVEX 指令均为 0。

## Affinity 审计

仅设置进程允许的 CPU 集合不足以保证短任务分散执行。未绑定的初步审计中，
4/8 个 worker 的运行曾汇聚在一个核上，这些未用于正式计时。
正式设置 `OMP_PROC_BIND=TRUE` 和明确的 `GOMP_CPU_AFFINITY`；MimIR 与 PyTorch
共用 wheel 的 private libgomp。分别验证了 4/8 个 worker ID、完整的 4/8 个物理 CPU，
并对每组重复校验 20 次 GEMM。

OpenMP 初始化会收窄 primary thread 的 mask，因此在创建 OpenBLAS pthread pool 前
恢复整个核集合，随后把 primary thread 固定到 CPU 12。OpenBLAS 的独立 pthread
workers 继承完整核集合；metadata 中保留所有线程 mask。
`OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`、`OPENBLAS_THREAD_TIMEOUT=4`
用于减少非计时线程池的 idle spin 干扰。

## 测试与复现

248 项 CTest 全通过；10 项重点 lit 回归全部通过；全量 827 项 lit 仅此前已知的
`tensor/group_rules_exec.mim` reference 路径泄漏失败，没有新增失败。
新测试覆盖显式/自动线程 tiling、无 omp 的串行回退、尾块、带 padding/offset 的四种输入布局及
列主序输出；ASan/LSan 检查重复调用，单独抑制系统 libomp 的已知 176 字节全局分配。

见 `../../OMP.md`、`../../prepare_omp.py`、`../../remote_cpu.py`。
`summarize.py` 可从原始数据重算汇总和 CSV；`run.sh` 保存正式环境与远端命令，`reproduce.sh` 顺序执行准备、审计、计时；`audit.py`、`isa.py` 保存审计；每个 tN 子目录保存
manifest、metadata、FP64 校验、9 轮原始数据及 worker 审计。

t1/t8 manifest 的 `source_dirty=true` 来自 lowering 期间临时编辑 `bench/cpu_tiling/OMP.md`，随后已恢复。编译器 src/include/cmake 与记录的提交无差异，构建哈希见 `local-build.json`；单核 22 个 LLVM IR 与修复前逐一相同。
