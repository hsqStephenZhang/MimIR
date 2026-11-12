# 本机 CPU attention：单线程与多线程

2026-11-12，Intel Core i5-14600KF，FP32。
测量给定 Q/K/V 的完整 SDPA 核心：`softmax(QKᵀ / sqrt(D) + mask) V`。
QKV 和输出投影不在本次范围内。H 表示折叠后的 batch×heads，
形状为 Q[H,M,D]、K[H,S,D]、V[H,S,N]，Torch 使用等价的 [1,H,...] 视图。

**发现当前默认 CPU attention 的多线程路径存在真实泄漏。**
H=8,M=S=128 时每次泄漏两个 score 大小的缓冲区，合计约1 MiB；
H=8,M=S=512 时约16 MiB/次。ASan/LSan 的20次调用报告、原生分配增长记录均一致。
因此默认路径的多线程数据是 **隔离进程中的受限短批次延迟，不是可持续吞吐指标**。
本次没有改变编译器或在计时中替它释放这些缓冲区。

## 对照与结果

- `mimir_default`：当前 multi-level tiling 的默认 SDPA 降低，`attention-online=0`。
- `mimir_online`：现有实验性 online SDPA，`attention-online=1`。
- `torch_sdpa`：原生 ATen `scaled_dot_product_attention`，对应 PyTorch 默认 API。
  profiler确认规则尺寸走 `_scaled_dot_product_flash_attention_for_cpu`；
  非整齐用例 D=32、N=48 回退 `_scaled_dot_product_attention_math`，详见 validation.json。
- `torch_explicit`：原生 ATen `bmm → scale/mask → softmax → bmm`，用于对照相同分解图。
  causal 的固定上三角 mask 在准备阶段建立；算子执行和中间分配均计时。

以 **H=8、M=S=128、D=N=64、non-causal** 为例，单位 ms：

| 配置线程 | 默认MimIR | online MimIR | Torch SDPA | Torch显式组合 | 默认MimIR launch数/次 | 默认MimIR分配增长/次 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.5188 | 4.8845 | 0.3116 | 0.4329 | 0 | 0.00 MiB |
| 4 | 0.3599 | 4.9062 | 0.0939 | 0.1851 | 2 | 1.00 MiB |
| 6 | 0.3482 | 4.9061 | 0.0730 | 0.1812 | 2 | 1.00 MiB |
| 8 (6P+2E) | 0.3888 | 4.9159 | 0.1245 | 0.2121 | 2 | 1.00 MiB |

**H=8、M=S=512、D=N=64、non-causal**，单位 ms：

| 线程 | 默认MimIR | online MimIR | Torch SDPA | Torch显式组合 | 默认MimIR分配增长/次 |
|---|---:|---:|---:|---:|---:|
| 1 | 8.344 | 78.748 | 4.504 | 7.164 | 0.00 MiB |
| 4 | 8.251 | 78.361 | 1.283 | 4.945 | 16.00 MiB |
| 6 | 7.692 | 78.724 | 0.896 | 3.204 | 16.00 MiB |
| 8 | 8.527 | 78.417 | 1.992 | 4.382 | 16.00 MiB |

11个形状的几何平均，性能比 = 对照耗时 / MimIR耗时；不是按FLOP加权。
默认多线程路径有9/11个用例泄漏，因此这一汇总只描述本次受限延迟，不能用于宣称长期扩展能力。

| 线程 | 默认MimIR/Torch SDPA | 默认MimIR/Torch显式组合 | 默认MimIR相对1线程加速 | Torch SDPA相对1线程加速 | online相对1线程加速 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.585× | 0.702× | 1.00× | 1.00× | 1.00× |
| 4 | 0.245× | 0.442× | 1.18× | 2.81× | 1.00× |
| 6 | 0.208× | 0.396× | 1.24× | 3.48× | 0.99× |
| 8 | 0.314× | 0.496× | 1.16× | 2.17× | 0.99× |

## 线程与内存观察

- 默认路径的代表用例每次有 **两个 OpenMP launch**。4/6/8线程审计均确认对应全部worker与指定CPU参与，
  不能将扩展差简单归因于 OMP_NUM_THREADS 未生效。
  当前自动线程级主要作用于两个 GEMM；其余 attention 处理没有独立的 OpenMP 并行层。
  本次没有做阶段级硬件计数器剖析，不能量化各阶段对总耗时的贡献。
- online路径在所有形状与配置中均为 **0 launch**，目前没有CPU多线程扩展。
  默认选择bk=8，行块由现有寄存器预算选择，不能据此代表充分优化的CPU online attention。
- decode(M=1,S=512)与非整齐(H=1,M=97,S=129,D=32,N=48)两个用例，
  默认路径各个GEMM低于自动并行阈值，在多线程配置中仍串行，也未发现相同泄漏。
- 规则尺寸下，多线程默认路径的 live-growth 中位数对应两份H×M×S的F32缓冲区，外加少量分配器元数据。
  例如H=8,S=128：1,048,608 bytes/次；H=8,S=512：16,777,248 bytes/次。
  LSan默认4线程20调用报告20,971,520 payload bytes、40个分配未释放，见 `leak-default-t4.log`。
  默认1线程和online4线程的同形状20调用检查均通过，未设置泄漏抑制。
- 代码层面的线索是 `mem/escape_analysis.cpp:95` 对未知调用保守保留传入指针，
  而 `omp/phase/outline.cpp:148` 把kernel捕获值放入同步launch环境。
  这是与缓冲区逃逸相关的排查方向；本次仅记录可复现现象，尚未完成所有权修复。
- 输出数值正确与生命周期正确是两项独立检查；176组输出均通过FP64验证，并不能消除上述泄漏。

## 测量方法

1. 11个形状覆盖单head、多head、S=128/256/512、D=64/128、cross attention、decode、
   非整齐维度以及causal mask；完整表见 `summary.csv` 和各tN/results.json。
   causal用例M=S，采用上左三角语义；decode用例为non-causal。
2. CPU掩码：1线程{0}；4线程{0,2,4,6}；6线程{0,2,4,6,8,10}；
   8线程{0,2,4,6,8,10,12,13}，即6P+2E，不使用SMT同胞。
   全部同一NUMA节点；8线程结果不等价于8个同速P核。
3. Clang 22，`-O3 -march=haswell -mno-avx512f`；CPU tiling参数与此前GEMM实验相同：
   MR=6、NR=16、MC=KC=320、NC=104512。88个MimIR产物均审计无EVEX/ZMM。
   Torch 2.12.0+cpu / MKL 2024.2，AVX2，MimIR与Torch使用wheel同一份libgomp。
4. 两条MimIR路径均包含结果与内部缓冲区分配/释放，Torch同样包含算子内部分配；
   计时为原生C++循环，排除Python、编译、输入加载和正确性输出复制。
   没有使用Python循环测量单次attention。
5. 每个无泄漏后端在独立worker中至少预热0.2秒，然后服务9轮请求；每个形状内固定种子打乱后端顺序，
   目标每轮30ms，取9轮中位数。闲置OpenMP线程使用PASSIVE、GOMP_SPINCOUNT=0，避免worker间争抢。
   原始样本保留实际迭代数、时长、预热次数、RSS峰值、原生分配增长及后端顺序。
   30ms是校准目标，不是保证的下限；校准波动或迭代限制使169/1584个样本短于10ms，
   最短约1.29ms，见 `short-samples.json`。这些样本同样保留并参与中位数统计。
6. 发现泄漏的默认多线程路径每轮新建进程，预热3次并做1次校准，再运行受限批次。
   依据每次16×H×M×S字节的保守预算，将估算的批次分配增长限制为256 MiB；
   这属于迭代数预算，不是OS层RSS限制。进程创建、输入读取和退出后的OS回收均不计时。
   计时区间没有替编译器释放任何中间缓冲区。它与充分预热的健康worker具有不同的生命周期条件，
   这是本次默认多线程数据的重要限制，修复泄漏后应重新测稳态性能。
7. 在确定预热方法时，最初每进程3次预热产生了明显偏高的Torch多线程值，该轮数据未用于最终结果。
   修正后原生SDPA(4线程约0.096ms)与标准 `torch.utils.benchmark.Timer` 一致：
   同形状1/4/6线程约0.310/0.097/0.072ms，见 `torch-timer-check.log`。
8. 核绑定通过MimIR原生worker审计和单独的Torch线程mask探针核对。
   cgroup各父层CPU配额为max、nr_throttled=0；governor为powersave，保留正常睿频，未锁频。
   单个样本最大/最小比超过1.3的组列于 `timing-variation.json`，没有删除样本。
9. 所有编译与数值验证均在正式计时前结束；原始输入固定种子Uniform(-1,1)转F32，
   `SeedSequence([42,H,M,S,D,N,causal])`。QKV raw文件哈希见 `source-state.json`。
   独立FP64参考使用稳定softmax，再乘V；rtol=atol=1e-3，实际最大绝对误差1.67e-7。

本次测试的编译器为 `e395e40c75`，没有修改编译器实现。
benchmark初版提交 `4b41b8f412`，最终预热/worker修正为 `b72b0cb28f`。
`metadata.json` 保留最初工具构建时的记录，最终驱动与脚本哈希见 `source-state.json`；
修正预热时没有改变MimIR kernel或ATen timer的计算实现。

## 复现

使用本机Python环境 `/workspaces/ml-compiler/gemm-bench/.venv/bin/python`，从MimIR仓库运行：

```sh
python bench/cpu_tiling/attention_cpu.py tools
python bench/cpu_tiling/attention_cpu.py prepare --threads 1
python bench/cpu_tiling/attention_cpu.py prepare --threads 4
python bench/cpu_tiling/attention_cpu.py prepare --threads 6
python bench/cpu_tiling/attention_cpu.py prepare --threads 8
```

validate阶段需在Python导入数值库前设置该线程数的OMP/MKL/OPENBLAS环境和核列表，
具体值由 `attention_cpu.environment(T)` 给出。逐一运行validate后运行
`python bench/cpu_tiling/attention_audit.py`，待全部编译/验证进程退出，再顺序运行：

```sh
python bench/cpu_tiling/attention_cpu.py run --threads 1
python bench/cpu_tiling/attention_cpu.py run --threads 4
python bench/cpu_tiling/attention_cpu.py run --threads 6
python bench/cpu_tiling/attention_cpu.py run --threads 8
```

默认工作目录为 `/tmp/mimir-attention-local`。重现LSan检查可将 `leak-probe.cpp`
与该目录对应kernel.ll及OpenMP runtime链接为ASan程序；该probe固定H=8、M=S=128、D=N=64、20次调用。
默认多线程检查应复现失败，不能把它当成已通过的内存安全测试。
