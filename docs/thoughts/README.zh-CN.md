# GPU 实验的整理：相关工作、贡献与方向

这组文档供作者逐条核对贡献和后续方向。五份分文档各管一个维度，本页给出总结论、跨文档的修正和合并后的清单。

| 文档 | 维度 | 内容 |
| --- | --- | --- |
| [A-infrastructure](A-infrastructure.zh-CN.md) | 生产级与研究级编译器基础设施 | MLIR、IREE、TVM、Triton（含 Gluon、ConSan）、Mojo/MAX、TileLang、cuTile、cubek、cuda-oxide、oneDNN、XLA、Inductor 的 IR 层次、异步表示与检查 |
| [B-graph-and-dsl](B-graph-and-dsl.zh-CN.md) | 图编译器与 kernel DSL | 谁决定融合、切块、流水、warp 特化与同步；fused attention 从哪里来；`mimir-frontend` 的实际能力 |
| [C-tile-and-fuse](C-tile-and-fuse.zh-CN.md) | tile 与 fuse | 切块表示、块大小由谁选、融合方式、合法性、流水生成；MimIR 的 tile law、`control`、环降级 |
| [D-types-and-verification](D-types-and-verification.zh-CN.md) | 类型系统与验证 | GPU 同步的类型与验证工作、一般的线性与 typestate 类型、分离逻辑、翻译验证；我们的信任链与缺口 |
| [E-contributions-and-directions](E-contributions-and-directions.zh-CN.md) | 我们的贡献清单 | 38 项贡献及其提交、文件、测试和状态；性能数字的出处；开放方向；核对命令 |

每份文档里，【本地】表示读过本地代码或论文，带 `路径:行号`；**未核实**表示只凭网页摘要或记忆。本地克隆都在 `/home/chzh00001/workspaces/` 下，各自的提交号写在相应文档开头。

agy（Gemini 4 Argon）核对过这组文档，报告在 [`../reviews/thoughts_agy_review.md`](../reviews/thoughts_agy_review.md)。它指出的问题，凡经汇总者核实的都已修正：见 §2；补充的相关工作见 §6。

## 1. 总结论

- **cgo27 论文的 GPU 路径只做显存管理和 kernel launch，原文写明 kernel fusion “is not implemented in MimIR yet”**（§8.3；E §0，汇总者在论文文本中核实）。所以 GPU 上的多级 tiling、tensor core、异步环、融合和协议类型都是论文之后的新工作。论文 §7 已有“调度在 IR 里、由部分求值展开”的原则，这一点不是新贡献。
- **新的多级 GPU 路径还没有接到 PyTorch 前端**（B §3，E §0）。
  - 论文 §8.3 的 GPU 评测确实从 PyTorch 出发，但走的是 E0 路径：一线程一输出、没有融合。
  - 现在的 `mimir-frontend` 作为 `torch.compile` 后端只生成 CPU 代码：float32、静态形状、只做推理。
  - 加载 `gpu` 插件时，group 和 attention 规则会关闭。
  - 多级 tiling、环和 WGMMA 的 benchmark 都来自手写的 tile 级 Mim 程序加调度（`law.of_blocks`）。
  - 因此“用户写 PyTorch”对新路径目前只是定位，不是已实现的事实。§4 的 P0 是对此做出选择。
- **在已查的张量编译器和 GPU DSL 编译栈内部**（MLIR、IREE、TVM、Triton 自动 WS、TileLang），**没有一个对自动生成的 cp.async、TMA、mbarrier、WGMMA 环协议做槽、代、相位层面的静态验证**（A §4，D §2）。已查到的检查只有下面这些：
  - MLIR 把流水正确性写成用户责任（`NVGPUTransformOps.td:97-100`）；
  - Triton 有动态插桩的 ConSan，aref 有缓冲区逃逸检查（`NVWS/IR/Ops.cpp:22-44`）；
  - TVM 只检查流水 stage 的顺序与依赖（`inject_software_pipeline.cc:1046-1050`）；
  - Mojo 在手写库里强制阶段句柄必须交还。

  但**编译器外部的验证器已经在做编译器生成 kernel 的验证**：
  - Volta（OOPSLA'26）在 PTX 层做等价和数据竞争验证；
  - RESOLVE（2026）把 Kuiper 的证明用到 Triton、CUTLASS、Gluon 的输出上；
  - WEFT（PLDI'15）早就静态验证 named barrier 的生产者/消费者同步，包括代和到达计数。

  所以新颖点必须写成“**在编译器 IR 内**、擦除前，由类型和逐 kernel 证书完成”，并与这三者正面比较（§6）。
- **要收窄或放弃的说法**（A、B、C，以及 agy 的核对）：
  - **线性阶段句柄连同 pipeline 身份和面板类型都已有先例**：Mojo 的 SM100 结构化 kernel 中，`InputProducerStage[origin, Payload, num_group_stages, …]` 是必须 `release()` 的线性句柄，类型里带着 pipeline 的 origin、环深和面板负载；只有槽号 `_stage` 和相位是运行时值（`modular/.../sm100_structured/structured_kernels/tile_pipeline.mojo:809-839`，汇总者核实）。我们能主张的差别是：**槽号进类型、两相环状态，以及对代与奇偶的逐 kernel 证书**。
  - **屏障 typestate 已有先例**：cuda-oxide。
  - **运行时并发检查已有先例**：Triton ConSan。
  - 多级切块、融合控制、按共享内存定级数、warp 特化、split-k 都有先例，重叠最强的是 IREE 的 7 种 GPU 切块级加 SMT 约束，以及 MLIR 的 `fusionControlFn`。自动 WS 与流水还有 Cypress（PLDI'25）、Twill（OSDI'26）、Tawa（CGO'26）。
  - “类型保证层级作用域”已不成立：`TileLevels` 合并后，改由 `check.walk` 在部分求值时判定。
  - “调度选择复用同一份合法性规则”**与代码不符**：`law.fastest`/`law.best` 只调用 `check.fits`（共享内存容量），不调用 `check.walk`（`tensor.mim:3805-3820`；`check.verdict` 只在 `:3936` 的 `tile_req` 处用到，汇总者核实）。
- **性能**（E §2，均转引自文档，未重测）：
  - H200 F16 GEMM 是 cuBLAS 的 94–100%（1024–8192）；
  - 编译出的 attention 是 cuDNN 的 0.73–0.88×，手写 FA3 结构为 0.92×；
  - CPU 上的 online attention 规则比不启用时慢 1.43–3.11×（`attention-schedule-experiment.zh-CN.md:1273-1360`）。

  性能用来说明“被认证的是生产级代码”，不作领先主张。

## 2. 修正记录

| 位置 | 原说法 | 修正 | 依据 |
| --- | --- | --- | --- |
| 本页、D、E、`linear-types-design` §22 | “51 个改变了生成代码的组合全部被拒，类型层 14、Lean 37” | 改为下面的口径。原来混淆了“生成代码改变”与“被拒”：静态拒绝时根本没有生成代码可比 | `bench/tools/mutate.py` 的 `verdict`；`erase_phases.cpp` 擦掉 `hold` 的槽号 |
| A、B、D、本页 | Mojo 的槽、环身份、面板都不在类型里 | 环身份（origin）、环深、面板负载都在类型里；只有槽号和相位不在 | `tile_pipeline.mojo:809-839` |
| C §5、本页 §3 | `law.fastest` 用 `check.walk` 排除不合法候选 | 它只用 `check.fits` | `tensor.mim:3805-3820` |
| E (f3) | CPU attention 实验“步骤 1 阻塞，尚未测量” | 步骤 1–5 都已完成；结论是比不启用规则慢 1.43–3.11×，堆分配为 3L²+132L+87 次 | `attention-schedule-experiment.zh-CN.md:208-1374` |
| B | `tensor.mim` 的行号来自删除解释器之前的旧树；`extra/torch/torch.mim` 不在本树 | 行号已更新；torch 插件在 `MimIR-fork/extra/torch/` | agy F-5，汇总者抽查 |
| B | Mojo 表项写“无检查”；IREE 仍写有 `GPUPipelining` | 已同步 A 的修正 | A §2.2 |
| B、D | 没有提到 ConSan | D 表中补了 ConSan 和 Mojo 两行 | A §2.4 |
| B | `pmlc-docker` 是 PlaidML | 是我们自己的 KernelBench 复现镜像 | A §1 |
| D G5 | 槽“按 fill 首次写到的缓冲区”命名 | 现在按 fold 循环初态里环数组的位置命名 | `ring_trace.cpp` 的 `Tracer::ring` |
| `gpu.mim` 中 `ring_trace` 的注释 | `-X gpu:ring-trace=1` 只追踪一个线程 | `-X gpu:ring-trace=N` 追踪线程 0、128、… | 代码已修正 |
| E §3 | 方向 1 是“核心线性性、环/槽/代进索引、删 `same_slot`” | 已被 §14、§17 和 RingState 报告否决，改走两相状态加残余验证；本页 §4 取代 E §3 的排序 | `linear-types-design` §14、§17，`reviews/ringstate_report.md` |
| `linear-types-design` §19 | 两个 §19；先写 0/11，后写 7/11 | 0/11 是旧分类器的结果，7/11 是 Lean 扩展之后的结果 | E §4.1 |
| `tile-law-design` 等两份文档 | 描述 Mim 里的 law 解释器 | 解释器已在 `0d85d5206` 删除 | E §4.1 |

**变异评测的正确口径**：15 种变异 × 7 个带环 kernel，共 105 对。

| 类别 | 对数 | 说明 |
| --- | ---: | --- |
| 生成代码不变 | 54 | 对该 kernel 是等价变异 |
| 只改了擦除前的幽灵下标 | 7 | `hold_slot`：生成代码不变，但协议库自身通不过类型检查 |
| 生成代码改变 | 44 | **全部被拒**，分布见下 |

那 44 对的分布：
- **6 对**被类型或 `check_sync` 拒：`read_other_slot` 5 对，`ws_release_current` 1 对；
- **1 对**被 `tensor.mim` 的有界协议模拟 `ring_safe` 拒：`ws_fold` 上的 `full_parity`；
- **37 对**被 Lean 检查器拒。其中 1 对（`attn_wg` 上的 `wgmma_wait`）是文本格式解析失败，不是语义规则拒绝。

## 3. 可主张的贡献

采纳 agy 的三项结构（§5.1），措辞经汇总者收窄：

1. **编译器生成的 GPU 异步协议在 IR 内的两层认证。**
   - 第一层：擦除式的线性与两相协议类型，加上 `check_sync`，在部分求值之后检查令牌线性、环身份、槽号与两相顺序。
   - 第二层：残余事件提取（`gpu.ring_trace`），加上 Lean 中证明可靠的逐角色检查器，逐 kernel 证明 T1/T2：读到本代，覆盖晚于上一代的最后一次读。
   - 已覆盖 11 个真实环中的 7 个；变异评测见 §2。
   - 应如实写出两层的分工：类型层只拦下了 6 对，时序和奇偶类错误目前全靠证书。
   - 证据：D §3，E (c)(d)(e)，`linear-types-design` §16–§22。
2. **协议骨架、F₂ 布局判定与调度合法性规则写在同一个高阶 IR 里，由同一个部分求值器特化。**
   - 协议骨架：`gpu.mim` 中的 `SyncOf`、`ring_fold`、`ws_fold`、`wgmma_batch`。
   - F₂ 布局判定：`tensor.mim:224-340` 中的 `lies_sw128`、`lies_tma128`。
   - 合法性规则：`check.walk`、`ring_safe`、`control`。
   - 新意在于这些都是普通的 IR 函数，部分求值后交给 C++ 降级，并且可以擦除。“调度在 IR 里”本身是论文 §7 已有的内容。
   - 要让“调度选择复用合法性规则”成立，需要让 `law.fastest` 调用 `check.verdict`（agy 估计约 3 行）。
3. **零运行时开销的擦除，以及生产级的多级代码生成。**
   - 从 SIMT 到 cp.async、mma.sync，再到 TMA、mbarrier、WGMMA、WS、cluster、persistent。
   - 加入类型与状态后，73 个 benchmark 的 device IR 中 71 个逐字节不变，2 个只差一个值名。
   - H200 GEMM 达 cuBLAS 的 94–100%。

降级为案例、不单列：
- 手写协议（是 1、2 的自然推论）；
- `of_rolling`（合并律由用户担保，解释器删除后只是薄包装）；
- CPU attention 规则（表达力案例，性能是负面结果）。

要避免的说法：
- “验证过的编译器”“kernel 被证明无竞争”；
- “类型系统保证同步安全”；
- “首个把异步同步放进类型”；
- “首个验证编译器生成的 GPU kernel”（Volta、RESOLVE 已在做）；
- “从 PyTorch 自动生成 GPU kernel”；
- “0 漏报”而不加限定；
- 性能领先。

## 4. 方向

**P0 是定位，由作者决定。** 两个选项：
- **(a) 改写定位**（agy 推荐）：输入界定为“Mim 的函数式张量规格加声明式多级调度项”，与 Halide、TensorIR、Exo 2、Cypress、MLIR Transform 同一定位；接通 PyTorch → GPU 列为进行中的前端工程。
- **(b) 接通 PyTorch → GPU**：要打开 `gpu` 插件下的 group 和 attention 规则，还要在 GPU 上自动选择多级调度，工作量接近一篇独立的论文。

技术方向，合并 agy §5.2 与 D §5 后排序：

| 优先级 | 方向 | 补的是什么 | 出处 |
| --- | --- | --- | --- |
| 1 | 把 `InFlight` 见证接到 `release` 上（`gpu.mim` 中 `ws_fold_at` 丢弃见证处）；`ring_idle` 由 `open` 产生并计入单次使用；`law.fastest` 调用 `check.verdict` | 让 `early_release`、`wgmma_wait` 在类型层被拒，明确两层的分工；堵住初始状态两次 await 的缺口；让贡献 2 名副其实 | agy §5.2，D G6/G7，`linear-types-design` §21.6 |
| 2 | 补叠放、persistent、cluster 的 Lean 模型和检查器 | 覆盖率从 7/11 提到 11/11；叠放对应 §29 的历史 bug，最快的 GEMM 配置正好在这几类里 | E §3，agy §5.2 |
| 3 | 从 PTX 提取事件，取代从 Mim 残余提取 | 把之后的各 phase、LLVM 移出信任基；还能检查 ptxas 越过 `wgmma.wait_group` 重排的问题 | D §5，`bench/gpu_tiling/RESULTS.zh-CN.md:769-771` |
| 4 | 扩大变异：回放历史 bug，变异 `LowerToMem` 的 C++ 代码；在 GPU 上用 compute-sanitizer、ConSan 式插桩做差分测试 | 现在的变异全是 `.mim`；检验手写的硬件模型 | D G9，A §5 |
| 5 | 用同一套检查器检查 Triton 自动 WS、Gluon、CUTLASS 的 PTX | 说明方法不只适用于 MimIR 自己的 lowering；依赖方向 3 | A §2.4 |
| 6 | 报告检查的编译期耗时 | 与 Koska（约 3.4 万行手写证明）、Volta/RESOLVE（SMT 或规约）对比代价 | agy §5.3 |
| 7 | 以后再做：硬件模型对接 Koska 的 PTX 语义；Lustig 的 PTX 内存模型；z3 写冲突检查；`SyncOf` 实例化到 AMD | 硬件模型有出处；去掉证书的一条假设；协议库的通用性 | D §5，A §5 |

E §3 的排序基于已被否决的设计，以本表为准。E §3.1 列出了已决定不做或已撤回的方向及原因。

## 5. 核对顺序建议

1. 先读 E §0、§1，对照提交号和 lit 测试核对每项贡献；E §4 给了命令。
2. 再读 D §3 和本页 §2 的变异口径，核对信任链的每一环。重点是提取器的假设：代表线程、`ws_fold` 的票号、步内 WGMMA 的顺序。
3. 然后决定 §4 的 P0。
4. A、C 和本页 §6 用来写相关工作。

## 6. 补充的相关工作

这些工作 A–E 漏掉了，其中大多已在仓库文档里调研过：`gpu-tiling-literature.zh-CN.md:25-275`、`next-directions.zh-CN.md:38-148`、`gpu-protocol-skeletons-survey-codex.zh-CN.md`。下表的摘要取自这些文档和 agy 报告 §3；汇总者只核实了它们在仓库文档中出现过，没有逐篇读原文。

| 工作 | 内容 | 对我们主张的影响 |
| --- | --- | --- |
| **WEFT**（Bauer、Sharma、Aiken，PLDI'15） | warp 特化生产者/消费者 kernel 中 named barrier 的操作语义；静态验证无竞争、无死锁，建模了屏障的代、到达计数和复用 | “前人不建模代”不成立。差别在于：WEFT 面向 named barrier 和手写 kernel；我们面向 mbarrier、TMA、cp.async、WGMMA 组，以及编译器输出和证书 |
| **Volta**（OOPSLA'26，arXiv 2511.12638） | 在 PTX 层做符号执行，验证 structured-CTA 类 ML kernel 的等价性、无竞争、无死锁 | 不能再说“没人验证编译器生成的 kernel”。差别在于：Volta 在编译器外部、PTX 层、用 SMT；我们在 IR 内、擦除前、用类型和证书。也可作为方向 3 的参照 |
| **RESOLVE**（2026，arXiv 2610.05683；Kuiper 团队） | 先测试，再把 kernel 规约，最后用 F\*/Pulse 证明；对象是 Triton、CUTLASS、Gluon 的融合 GEMM | D 说 Kuiper 只验证手写代码，需要补充：后续工作已经作用于编译器输出 |
| **Tatva**（NeurIPS'26 workshop）、**Mirage**（OSDI'25） | 前者用 Z3 检查 PyTorch/Triton 等价；后者是超优化器，用随机测试验证等价 | 属于翻译验证的相关工作 |
| **Prism / Typed Perspectives**（PLDI'26） | 用类型刻画线程、warp、tensor-core 组等观察视角，并有 `async` 作用域 | GPU 类型系统的直接对照 |
| **Cypress**（PLDI'25）、**Twill**（OSDI'26）、**Tawa**（CGO'26）、**Singe**（PPoPP'14） | 自动生成 TMA、mbarrier、warp 特化：Cypress 用 task 加映射规格；Twill 用约束求解求最优流水与 WS；Tawa 即 Triton 的 aref；Singe 是自动 WS 的早期工作 | 削弱我们在调度和 WS 生成上的新颖性；反过来也支持“自动生成的同步需要验证”这一动机 |
| **Lustig 等**（ASPLOS'19） | PTX 内存模型的形式化 | 用来界定我们手写硬件模型（G3）与真实 PTX 语义之间的差距 |
| **ATL**（POPL'22、PLDI'24）、**Exo / Exo 2**（PLDI'22、ASPLOS'25）、**Fehr 等**的 MLIR 验证方言（PLDI'25）、Lean-MLIR（ITP'24） | 经过验证的张量调度；保持语义的调度库；IR 内的验证契约 | 用于对比“在 IR 内表达验证契约”的不同路线 |
| **Linear Layouts**（ASPLOS'26）、CuTe 布局代数（arXiv 2603.02298） | F₂ 布局代数；分层形状与步长的布局代数 | 讨论贡献 2 的 F₂ 判定时必须引用 |
| **FlashAttention-3/4** | Hopper/Blackwell 上的 TMA、WGMMA、WS、ping-pong 重叠 | `gpu-protocol-skeletons-survey-codex` 已核实过 FA3；C 误写为未核实 |
