# 本机 CPU batch GEMM，2026-11-12

本次全部在本机 Intel Core i5-14600KF 上测量，FP32、连续 row-major NN：
`A[B,M,K] @ B[B,K,N] -> C[B,M,N]`。15 个形状，1/4/6/8 线程。

当前 batch>1 路径的单线程性能接近 OpenBLAS SGEMM 循环，多线程支持通过外层
`TileLevels#Threads` 组合 batch 与 M/N 分块，内层仍是 NC/KC/MC 和 6×16 向量微内核。
但 singleton batch 存在明确调度缺口：batch=1 跳过有效 tiling，生成标量 FMA；
设置更多线程也不会启动线程区。下面同时给出全部 15 例与 batch>1 的 13 例，避免掩盖该问题。

## 测量条件

- CPU：6 个 P 核、8 个 E 核；不使用 SMT 同胞。
- 1 线程 CPU 0；4 线程 0,2,4,6；6 线程 0,2,4,6,8,10；
  8 线程 0,2,4,6,8,10,12,13，即 **6P+2E**，不能视为 8 个同速 P 核。
- 所有 MimIR 内核：Clang 22，`-O3 -march=haswell -mno-avx512f`。
  CPU 参数 MR=6、NR=16、MC=320、KC=320、NC=104512。
- PyTorch 2.12.0+cpu，oneMKL 2024.2，GNU OpenMP，强制 AVX2。
  原生 C++ `aten::bmm.out`，计时中不调用 Python。
- OpenBLAS 0.3.33.dev，Haswell，pthread，MAX_THREADS=20；
  源码提交 `8e57c8667da06cb55a777975b6c2435c9b304079`。
- 两个 OpenBLAS 对照：原生 `cblas_sgemm_batch_strided`（JSON `openblas`），
  C 循环依次调用 `cblas_sgemm`（JSON `openblas_loop`）；后者每个 GEMM 均使用配置的线程预算。
- MimIR 与 Torch 共用 wheel 的同一份 libgomp；OpenMP 明确绑核，OpenBLAS pthread
  使用 `openblas_setaffinity/getaffinity` 逐线程绑定并校验。
  闲置池使用 PASSIVE / GOMP_SPINCOUNT=0 / OPENBLAS_THREAD_TIMEOUT=4。
- 先完成全部编译、回归、数值校验和线程审计，再顺序计时 1/4/6/8 线程。
  每个内核预热 5 次，9 轮固定种子打乱 backend 顺序；每轮原生循环至少 100 ms，取中位数。
  raw JSON 保留每轮数据、次数、顺序以及二进制哈希。
- 全部输出与 FP64 matmul 比较，rtol=atol=1e-3。
  MimIR 包含输出与内部 scratch 分配/释放；Torch/OpenBLAS 使用预分配输出。
  所有计时均排除 Python、输入复制及验证输出复制。
- CPU governor 为 powersave，保留系统正常睿频；未锁定频率。结果是本次运行的中位数。
  240 组形状/线程/backend 中有 10 组的 9 轮最大/最小值超过 1.3；主要是少量慢轮，
  保留全部样本，未删除异常值，见 `timing-variation.json`。

## 性能汇总

数值为各形状性能比的几何平均；性能比 = 对照耗时 / MimIR 耗时，越大越好。
“加速”均相对该 backend 自身单线程。每例等权，不按 FLOP 加权。

| 线程 | 全部15例：MimIR/Torch | batch>1：MimIR/Torch | batch>1：MimIR/BLAS循环 | batch>1：MimIR/BLAS batch | MimIR加速(batch>1) | Torch加速(batch>1) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.564× | 0.859× | 0.988× | 2.486× | 1.00× | 1.00× |
| 4 | 0.484× | 0.864× | 1.563× | 2.537× | 2.86× | 2.85× |
| 6 | 0.465× | 0.863× | 1.600× | 2.596× | 3.38× | 3.36× |
| 8 (6P+2E) | 0.495× | 0.831× | 1.359× | 2.207× | 2.05× | 2.12× |

## 代表形状

单位 µs，一次完整 batch GEMM；形状顺序是 B×M×K×N。
全部 60 个形状/线程组合见 `summary.csv`。

| 线程 | B×M×K×N | MimIR | BLAS batch | BLAS循环 | Torch | MimIR launch数/次 |
|---|---|---:|---:|---:|---:|---:|
| 1 | 1×512×512×512 | 61217.29 | 1682.16 | 1686.88 | 1686.56 | 0 |
| 1 | 2×256×256×256 | 442.95 | 432.33 | 433.36 | 426.09 | 0 |
| 1 | 8×256×256×256 | 1783.54 | 1745.97 | 1750.33 | 1722.09 | 0 |
| 1 | 64×16×16×16 | 7.38 | 67.35 | 7.85 | 5.52 | 0 |
| 1 | 16×8×2304×512 | 7171.65 | 6537.79 | 6544.48 | 6184.51 | 0 |
| 4 | 1×512×512×512 | 60865.84 | 1687.22 | 448.08 | 437.76 | 0 |
| 4 | 2×256×256×256 | 119.03 | 217.73 | 127.38 | 119.18 | 1 |
| 4 | 8×256×256×256 | 445.45 | 440.10 | 527.10 | 431.46 | 1 |
| 4 | 64×16×16×16 | 8.39 | 54.69 | 9.12 | 5.19 | 0 |
| 4 | 16×8×2304×512 | 3999.42 | 3486.55 | 4747.08 | 3646.81 | 1 |
| 6 | 1×512×512×512 | 61293.40 | 1679.27 | 319.83 | 301.65 | 0 |
| 6 | 2×256×256×256 | 84.79 | 216.53 | 97.83 | 88.01 | 1 |
| 6 | 8×256×256×256 | 443.12 | 438.06 | 395.68 | 429.77 | 1 |
| 6 | 64×16×16×16 | 8.37 | 43.62 | 8.87 | 5.22 | 0 |
| 6 | 16×8×2304×512 | 4035.11 | 3646.18 | 4014.07 | 3674.72 | 1 |
| 8 | 1×512×512×512 | 60944.15 | 1680.00 | 661.61 | 635.66 | 0 |
| 8 | 2×256×256×256 | 180.02 | 217.27 | 195.65 | 176.84 | 1 |
| 8 | 8×256×256×256 | 661.88 | 661.54 | 788.69 | 609.70 | 1 |
| 8 | 64×16×16×16 | 8.72 | 52.66 | 8.98 | 7.62 | 0 |
| 8 | 16×8×2304×512 | 4664.61 | 4511.15 | 4598.91 | 4631.04 | 1 |

## 对调度的判断

1. **batch=1 是实现缺口，无法靠调块大小修复。** 常量 batch 写映射不能通过当前 tile-preimage
   精确性证明，相关层被跳过。反汇编 `1×512³` 出现 4096 条标量 FMA、0 条 packed FMA；
   `2×256³`/`8×256³` 则有 80 条 packed FMA。单线程 61 ms 对比 Torch/BLAS 约 1.69 ms。
   这里的多线程配置确实是串行，审计 20 次调用为 0 launch。
   需修复 singleton 轴的映射/调度证明，并保留输出所有权及 epilogue 融合的正确性。
2. **batch>1 的并行表达已工作。** batch 足够时每个线程任务拥有完整矩阵；batch 不足时
   同一个线程网格进一步切 M/N。batch=3 与 batch=8 的 4/6/8 线程审计均验证实际所有 worker
   和所选核，无嵌套线程池；每次 kernel 一次 launch。batch=2 的多线程结果也能体现
   M/N 再切分相比只按 batch 分工的好处。
3. **不能只对比 OpenBLAS 原生 batch API 宣称领先。** 本地 Haswell 构建中，小矩阵的 batch
   API 明显慢于 SGEMM 循环。源码 `interface/gemm_batch_strided.c:358` 对单矩阵
   M*N*K<=1,000,000 直接选 small kernel；Haswell 的 small kernel 使用 generic 实现。
   普通 `interface/gemm.c` 则先调用 `GEMM_SMALL_MATRIX_PERMIT`。
   Haswell 使用的 generic permit 恒返回 0，因此普通 SGEMM 不走这条 generic small 路径。
   batch 驱动按每矩阵一个任务并分组执行，batch<线程数时不再切分单矩阵。
   这解释了需要保留两个 OpenBLAS 对照，不能把原生 batch 的慢路径视为 BLAS 性能上限。
4. **Torch 是实用的 BMM 对照。** 单独的 MKL_VERBOSE 探针确认 B=1 分派到 SGEMM，
   B=3/8/64 分派到 SGEMM_BATCH，且 NThr=4；见 `torch-dispatch.log`。
   该日志仅用于确认 dispatch，初始化时长不作为性能结果。
5. **小任务阈值与异构核负载仍需优化。** `64×16³` 小于默认 1,048,576 MAC 阈值，
   在所有线程配置中保持串行。更高线程数并非总有利；8 线程混入 E 核，当前静态任务分工
   不按核速度加权。瘦矩阵和很小矩阵也不能据大方阵结果推断性能。

本次性能数据只覆盖连续 FP32 NN，不代表转置、广播 batch、任意 strides 或 GPU batch。
NN/NT/TN、4D batch 前缀与 bias 的正确性和 ASan/LSan 由 omp/batch_exec 回归覆盖。
测试仅抑制既有 libomp 176 字节运行时泄漏，未抑制张量分配。

## 版本与复现

- MimIR 编译器实现提交：`eabf14aabf`，分支 `feat/cpu-multilevel-omp`。
  生成 IR 时的 source_dirty 包括本次 runner 文档/计时修正，C++ 调度实现与该提交一致；
  `source-state.json` 另存实际源码、编译器与插件哈希。
- gemm-bench：本地独立 worktree `/tmp/gemm-bench-cpu-bmm`，分支 `feat/cpu-bmm`，
  提交 `ffcc8af`（以 metadata 中完整哈希为准）。原有 GPU 工作目录保持原样。
- `cases.txt` 与 `local-env.sh` 保存本次形状和运行环境；每个 tN 目录中记录 lowering/
  compilation 命令、IR 哈希、运行时、baseline 版本与二进制哈希。
  早期 t1/t4 manifest 的 `target` 仍保留旧脚本写死的 icelake 标签；这不是实际编译选项，
  本次所有 `compile.json`/metadata 均为 Haswell AVX2，60 个产物已审计无 EVEX/ZMM。
  lowering 脚本已去掉该过期标签。
- 生成示例：`ulimit -s 65536`，随后运行 `bench/cpu_tiling/prepare_omp.py`
  `--bench /tmp/gemm-bench-cpu-bmm --build build --cases bench/cpu_tiling/bmm-cases.txt`
  `--root /tmp/mimir-bmm-local-20261112/t4 --threads 4 --jobs 2`。
  将该 root 的 `harness` 和 `mimir` 链接到上述 worktree 及其 mimir 子目录。
  按 `local-env.sh prepare 4` 先编译和校验，再 `local-env.sh run 4` 计时。
  64 MiB stack 只用于绕过当前递归编译器在 singleton 大矩阵上的默认栈溢出。
- 验证：260 项单元测试通过；lit 820 通过、8 不支持；新增 batch 回归 ASan 4/8 线程通过；
  gemm-bench operator 测试 6 项通过；60 个形状/线程组合、4 backends 全输出 FP64 校验通过；
  独立线程审计 3 个形状×3 个线程配置，每项 20 次调用。
