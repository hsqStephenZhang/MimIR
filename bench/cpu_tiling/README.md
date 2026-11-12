# CPU multi-level tiling 与 OpenBLAS 排查

新增多核结果：[ss420e 4/8 核实验](results/2026-11-12-ss420e-omp/README.md)，
接入机制与复现方式见 [CPU thread tiling](OMP.md)。

CPU attention：[修复 OpenMP 缓冲区泄露后的本机 1/4/6/8 线程重测](results/2026-11-12-local-attention-fixed/README.md)，
含 Torch SDPA/显式组合对照、LSan 与持续内存检查。

新增：[Rule 识别后的 CPU attention 联合调度](results/2026-11-12-local-attention-rule/README.md)，
比较按 query 行块并行、归一化后移与原分解路径。

本目录保存 2026-11-11 在 i5-14600KF 上的 FP32 单线程 GEMM 排查。矩阵形状按
`M × K × N` 表述；JSON 的 `shape` 遵循 gemm-bench 的 `[M, N, K]`。

## 基线身份

此前表中的“MimIR 默认”指 **优化分支使用默认参数**，源码已包含 multi-level tiling、MC/packing 和尾块/SSA 修复，不是 `origin/master`。
“6×16”使用同一修复源码，只增加显式调度参数。
现有历史结果不能据此计算相对上游 master 的累计收益。

| 历史结果键 | 源码 | 含义 |
|---|---|---|
| `old` | `2d24a48949` | 已有 BLIS/multi-level 调度，无 MC；不是原版 master |
| `mc` | `e5e809f189` | 加入 MC 与独立 packing，保留默认参数 |
| `current` / 远端 `default` | `124047b43e` | 含尾块和行向量 SSA 修复的优化分支默认参数 |
| `rows96` / 远端 `haswell96` | `124047b43e` | 同一修复源码，显式 6×16/cache/order 参数 |

83.25% → 94.64% 是同一源码在 AVX2 下的约 **13.68% 选参收益**。
此前的 21.43% 是同一 6×16 配置禁用 AVX-512 后的吞吐变化。
当前默认选择器尚未自动采用该实测参数配置。

## 确认的差距与修复

1. **单行/单列尾块退化。** 原 `block_ok` 要求 MR、NR 都不为 1，使 peeled M=1
   尾块退回每个 C cell 单独遍历 K 的标量点积。49×4608×512 的最后一行仅占
   2.04% FLOPs，却占约 39.8% 的采样周期。现在只要面板有多个独立输出 cell，
   就保留 blocked/register 路径；修复后同形状的采样未再出现标量 FMA 内循环，
   见 `tail-before-samples.txt` / `tail-after-samples.txt`。`register_block_exec` 覆盖单行、单列和同时 peeled 的情况。
2. **累加器表示影响实际微内核。** 默认 4×16 保留 64 个 cells，而 OpenBLAS Haswell
   实际存在 16×6/8×12 的 96-cell 路径，不能仅根据 `SGEMM_UNROLL_M/N=8/4` 判断。
   对 row-major C，其 16×6 对应我们逻辑输出的 6×16。原来直接扩大扁平 SSA panel
   到 96 个 cells 会造成严重 spill；1024³ 的一个热循环有 36 处 SIMD 栈引用，
   另一处有 319 处，2048×512×128 因此显著变慢。
   将 accumulator 改为 MR 个 NR-wide 行向量后，这三个重点形状的全部 FMA 内循环
   都没有 SIMD 栈引用（见 `row-regs-codegen.json`）。前端输出独立行 PHI，LLVM 再拆为
   硬件宽度；`ll::emit_tuple` 同时修复 closed expression 含局部 SSA 值时错误生成常量 aggregate。
3. **宏观参数与方向此前未完全匹配。** OpenBLAS row-major CBLAS 会交换 M/N、A/B；
   相应 CPU 调度应使用 `cpu-order=1`。本机实际导出的 P/Q/R 是 320/320/104512，
   `mimir-openblas-96.toml` 在这个方向上使用 MR=6、NR=16、MC=KC=320、NC=104512，
   并显式允许 128 个 SSA cells。默认路径仍保留通用 64-cell 预算及原参数选择。

`comparison.csv` / `paired-native.json` 是加入 MC 后、此次尾块与寄存器修复前的对比；
`investigation-comparison.csv` / `investigation-paired.json` 是最终修复后的对比。
`focus-paired.json` 额外分离了“只修尾块”和“改行向量表示”的影响。

## 最终结果

22 个形状的吞吐比几何平均（OpenBLAS=100%）：

| 配置 | 相对 OpenBLAS | 相对修复前 MC 默认 |
|---|---:|---:|
| 多级 tiling 旧版，无 MC | 77.64% | 0.993× |
| 修复前 MC 默认 | 78.22% | 1.000× |
| 修复前宏观参数对齐，4×16 | 83.24% | 1.064× |
| 修复后通用默认，4×16 | 81.34% | 1.040× |
| 修复后本机配置，6×16 | 97.06% | 1.241× |

部分形状的中位时间，单位 ms，形状为 M×K×N：

| 形状 | 修复前对齐 4×16 | 修复后默认 | 修复后本机 6×16 | OpenBLAS | PyTorch |
|---|---:|---:|---:|---:|---:|
| 256×256×256 | 0.2498 | 0.2738 | 0.2214 | 0.2158 | 0.2128 |
| 1024×1024×1024 | 15.7236 | 16.6047 | 13.7628 | 13.3017 | 13.4914 |
| 8×2304×512 | 0.2494 | 0.3102 | 0.2588 | 0.2375 | 0.1595 |
| 2048×2048×2048 | 131.9013 | 140.4280 | 116.2176 | 111.1491 | 112.4578 |
| 2048×512×128 | 2.2108 | 2.2011 | 1.9354 | 1.7524 | 1.7724 |
| 49×4608×512 | 3.1929 | 2.1087 | 1.7458 | 1.7556 | 1.5133 |
| 7×1024×1024 | 0.2480 | 0.2888 | 0.2763 | 0.3097 | 0.1532 |
| 97×383×257 | 0.2205 | 0.1636 | 0.1412 | 0.1324 | 0.1329 |

本机 6×16 配置相对修复前 MC 默认的全形状几何平均加速为 1.241×，相对 PyTorch 的吞吐比为 89.30%。

## 排除与仍存在的差异

- malloc/free 调用耗时上界约占 1024³ 的 0.06%、49×4608×512 的 0.22%。这不包含
  packing 的实际读写成本，也不表示 packing 免费。
- 将 malloc 强制为 64-byte 对齐，大幅消除 split loads，但周期基本不变；只给 B
  增加 512/1024/2048-byte lookahead 的软件预取，收益均不足 1%。因此不能把这两点
  当作主要原因。原始交错结果在 `square-ablation.json`，硬件计数在 `*.stat`。
- 修复前 1024³ 约 94% 采样周期在向量 K 循环，OpenBLAS 约 97% 在 compute kernel；
  分配器不是大方阵差距的解释。两者静态 LLVM-MCA 理想吞吐均为 32 FLOP/cycle，
  不能据此认定实际执行周期相同。
- 仍有代码调度、K 展开与边界拆分差异。OpenBLAS 在剩余 K 介于 Q 和 2Q 之间时
  会平衡两块；例如 K=1024、Q=320 为 320/320/192/192，而当前固定分块是
  320/320/320/64。尚未通过单变量实验把修复后的剩余差距归因于这一点。
- 96-cell 配置是本机显式参数配置，不是跨架构自动选择；小形状、尾块和长宽比仍需选择策略。

## 布局支持的范围

`layout-correctness.json` 的 32 个测试全部通过完整 FP64 参考验证，包括 A/B 各自
row-major 或 column-major、带偏移及 padding 的静态 leading dimension、普通输出、
输出后 transpose、直接用 BᵀAᵀ 产生 column-major C。
`cpu_layout_exec` 另外使用 ASan 检查四种 padded 输入与直接 column-major 输出，并验证
两种 cache 循环方向。

当前 **静态输入布局表达和 packing 适配可用，自动布局选择与通用 stride ABI 尚不完善**：

- `cpu-order` 控制 cache 循环和打包复用方向，不是矩阵存储布局标志；当前 selector 不按
  输入物理 stride 自动选该方向。两种方向均能处理四种静态输入布局。
- column-major 输出可通过交换操作数/收缩轴直接生成，避免另一次矩阵复制；但是
  `transpose(product(...))` 目前额外分配并复制整个 C。128×256×192 样例中，普通输出
  为 3 次 malloc，transpose 输出为 4 次，新增一次 98,304-byte C 分配。
- 任意输出访问映射尚未规范化：直接把 `(m,n,k)` 写到 `(n,m)` 的 `map_out`，显式
  multi-level tiling 严格模式会拒绝，要求 root 写 loop d 到 output axis d。
  改写为 BᵀAᵀ 的 canonical 形式可用。失败探针和诊断已保存。
- `Tensor` 当前 ABI 使用 dense 物理形状；静态 padding 可以用 view 表达。
  运行时 `lda` 作为 inner extent 的实际编译探针被 bufferizer 拒绝：
  `cannot bufferize: runtime inner extent is not supported`。这还不能替代 CBLAS 的动态 lda/ldb/ldc 接口。

## 测量与复现

使用 `/workspaces/ml-compiler/gemm-bench-gpu` 的 `feat/mimir-nominal-tensor` adapter，
Clang 22 `-O3 -march=raptorlake`；CPU 0，一个线程，5 次 warmup，9 个随机交错批次，
每批至少 100 ms，报告中位数。所有输出与 FP64 matmul 比较，rtol=atol=1e-3。
MimIR 包含输出和内部临时内存分配/释放；OpenBLAS timer 使用预分配 C；排除 Python
调用和输入复制。该差异保留在原生接口测量中，未用估计数扣减。
OpenBLAS 明确链接本地同一库，避免 `-lopenblas` 意外选到另一版本；身份在 `openblas-library.json`。
PyTorch 使用单线程 MKL。此前完整 TVM 对比的 22 个 case 均正确，使用未调优 LLVM
初始候选；它不能代表 TVM 自动调优后的最高性能。

完整输入、IR、汇编及二进制保存在：
`/workspaces/ml-compiler/gemm-bench/results/cpu-openblas-mc-20261111`。
变体：`default`（修复前 MC 默认）、`aligned`（修复前宏观对齐）、`tail-fix`
（重点形状，只修尾块）、`current-default`（最终源码，通用参数）、`row-regs`
（最终源码，本机 6×16 参数）。

```sh
OPENBLAS_NUM_THREADS=1 /workspaces/ml-compiler/gemm-bench/.venv/bin/python \
  bench/cpu_tiling/compare_saved.py \
  --bench-repo /workspaces/ml-compiler/gemm-bench-gpu \
  --results-dir /workspaces/ml-compiler/gemm-bench/results/cpu-openblas-mc-20261111 \
  --previous-artifacts /workspaces/ml-compiler/gemm-bench/results/cpu-multilevel-20261111/mimir/artifacts \
  --openblas /workspaces/compile/OpenBLAS/libopenblas.so
```

`profile_driver.cpp` 用于 perf 中排除 Python 开销，参数为
`mim|blas library M N K iterations`；`profile_allocation.cpp` 配合重命名 LLVM 中的
malloc/free 统计调用成本，输出 C 的释放由 driver 完成，不计入内部分配计数。

验证：260/260 单元测试通过；823 个 lit 中 814 通过、8 unsupported、1 个此前已复现的
`tensor/group_rules_exec.mim` reference-tiles 内存泄漏失败。新增的 register/tail/layout 测试均通过。

## ss420e / Ice Lake CPU 补测

同一修复版本已在 2× Xeon Silver 4310 的 CPU 0 单线程完成全部 22 形状对照，
全部通过 FP64 校验。Clang 20.1.8 `-march=icelake-server`，新编译 OpenBLAS 0.3.33.dev
SKYLAKEX，远端 PyTorch 2.2.1/MKL。系统 OpenBLAS 0.3.8 错误检测为 Prescott，未用作正式基线。

吞吐比几何平均：默认 4×16 为 OpenBLAS 的 **43.54%**，迁移的固定 6×16 为 **51.14%**；
6×16 比默认快 **17.47%**。1024³ 的 6×16 / OpenBLAS / PyTorch 为
29.6895 / 13.5024 / 12.1663 ms，2048³ 为 208.5525 / 109.8915 / 93.9872 ms。
已确认生成 AVX-512 ZMM FMA；6×16 主循环只有 6 个 ZMM 累加器，仍需针对该架构选择
微内核和 cache 参数，并核对代码生成。此轮没有对剩余差距做完整因果归因。

这组结果不能沿用本地 97.1% 的结论。完整口径、版本差异和复现命令见
[ss420e 报告](results/2026-11-11-ss420e/README.md)，原始批次在同目录 `paired-native.json`。

## ss420e 禁用 AVX-512 对照

相同 Ice Lake 机器、CPU 0 单线程、Clang 20、IR、tiling 参数和输入，MimIR 仅增加
`-mno-avx512f`；OpenBLAS 与 MKL/ATen 也切换为 AVX2。44 个 MimIR 二进制及 Haswell
OpenBLAS 均核验没有 EVEX 指令，MKL verbose 与 PyTorch capability 也确认 AVX2。
全部 22 形状、四种实现通过完整 FP64 校验。

禁用后，默认配置达到 AVX2 OpenBLAS 的 **83.25%**，固定 6×16 达到 **94.64%**
（启用时分别为 43.54%、51.14%）。6×16 自身的吞吐几何平均提高 **21.43%**；
OpenBLAS 与 PyTorch 切到 AVX2 后的耗时几何平均分别增加到原来的约 1.524×、1.703×，
相对差距收敛包含双方的变化。
1024³ 的 6×16 / OpenBLAS / PyTorch 为 23.5540 / 22.9868 / 22.6133 ms，
2048³ 为 186.1774 / 180.6148 / 189.4988 ms。

这表明当前固定微内核在 AVX2 下更接近 BLAS，AVX-512 配置仍需单独优化；
尚未用硬件计数器区分频率、依赖链和代码调度的贡献。
[完整 AVX2 报告和前后对照](results/2026-11-11-ss420e-avx2/README.md)。

## 原版 origin/master 累计收益补测

此前“默认”列的命名容易混淆，现已补齐真正上游默认基线 `origin/master` `19d6c2de62`。
在 ss420e CPU 0、AVX2、Clang 20、同一输入的同次交错测量中，22 形状吞吐比几何平均：
优化分支默认参数为上游的 **3.204×**，同分支显式 6×16 等参数为上游的 **3.652×**。
后者相对优化分支默认的选参收益为 **1.140×**；两种优化分支配置相对 OpenBLAS 分别为
**83.50%**、**95.18%**。

1024³ 的上游 / 优化分支默认 / 优化分支显式参数为 129.8511 / 28.1239 / 23.5305 ms，
2048³ 为 1143.4406 / 227.8844 / 187.4083 ms。
累计差距包含此前已具备的 BLIS 路径和后续改进，不能全归因于最近的 MC、尾块和 SSA 修复。
默认选择器仍未自动采用这套实测参数；小 M 等形状还存在默认参数更快的情况，需要按形状和目标选择。
[完整 master 对照](results/2026-11-11-origin-master/README.md)。
