# D. GPU 同步的类型系统与验证：相关工作与自评

状态：调研笔记，2026-11-19。只读调研，未构建、未跑 lit。
对象：`~/workspaces/MimIR-thoughts`，HEAD `f12e45a50`。

**核实标记**（每条事实后一个）：
- 【原文】读过论文正文提取（`scratchpad/{ag,descend,kuiper}.txt`）；
- 【网页】本次在 arXiv 或论文主页核对过；
- 【本地】读过本地代码（`cuda-oxide`、本仓库）；
- 【报告】沿用 `docs/reviews/ringstate_report.md` 中标 [V] 的条目，本次没有重核；
- **未核实**：凭记忆，细节可能有误，引用前须查原文。

## 0. 结论先行

- **定位。** 我们的工作在相关工作里最接近的是**逐实例的翻译验证 / 证书编译**（Tristan–Leroy、Necula），而不是 GPU 类型系统。类型层是这套验证的廉价前置过滤，不是独立的安全保证。
- **真正的空白**（就本次检索而言，未作穷尽检索）：
  - GPU 方面，给 mbarrier 环的**槽、代、相位**以可检查的协议语义，并对**编译器自动生成**的 kernel 逐个检查，没有找到先例。
  - Async Graphene 和 Tawa 有 aref 抽象，但没有可靠性定理；Kuiper 和 Koska 有机器证明，但对象是手写程序，而且需要大量手工证明。
- **最大的缺口**在信任链中段：提取器未验证；它解释的是 `check_sync` 时刻的残余，之后的 phase、LLVM、ptxas 都不在范围内；每个角色只追踪一个代表线程。
- **最高价值的下一步**：从 PTX 提取事件，送进同一组 Lean 检查器。这样能把中段的编译器从信任基里移出去。

## 1. 相关工作一览

列说明：
- **对象**：被类型化或被验证的东西；
- **保证 / 信任基**：证明了什么，信任哪些部分；
- **自动**：自动完成，还是需要标注或手写证明；
- **作用于**：用户代码，还是编译器输出。

### 1.1 GPU 专用

| 工作 | 对象 | 保证 / 信任基 | 自动 | 作用于 | 核实 |
|---|---|---|---|---|---|
| **Async Graphene**（Hagedorn、Grover，CC'26） | 张量类型上的时间修饰：future `!T`（`!group`/`!mbar`）、aref `%T`（full/empty）、iterator `#[D].T`、tuple | 类型排除“未 wait 就用”。附录只有类型的构成规则，没有使用规则，也没有可靠性定理。“by construction”只是一句陈述。full/empty 是运行时 mbarrier 相位。 | 大部分推断；延迟释放（`delay_signal`）由人手写 | 用户写的 kernel DSL | 【原文】 |
| **Descend**（Köpcke、Gorlatch、Steuwer，PLDI'24） | 执行资源、内存视图、扩展的借用检查 | 静态保证视图访问不冲突、`sync` 不被遗漏或错放；有形式化类型系统。只有块级 `sync`，没有异步拷贝、mbarrier 和多缓冲。 | 函数需标注执行资源，其余由检查器完成 | 用户代码 | 【原文】（无 mbarrier 一条：【报告】） |
| **Kuiper**（Martínez 等，PLDI'26） | F\*/Pulse 中的程序逻辑：located resources；屏障令牌 `barrier_token(p,q,i,it)` 按**绝对**迭代号索引 | 无数据竞争和功能正确性；只证部分正确性，不证无死锁。信任基包括参考语义（作者明说）、验证工具，以及经 Karamel 生成的 CUDA 和 nvcc。`mma_sync` 当作原语。证明了带两级分块和 tensor core 的 GEMM。 | 手写规约与证明 | 用户代码 | 【原文】（没有 mbarrier 或异步流水：【报告】） |
| **Tawa**（Chen 等，CGO'26） | IR 抽象 aref，用于自动 warp specialization | 摘要里没有静态类型检查或验证；正确性依赖生成器。 | 全自动（编译器） | 编译器输出 | 【网页】仅摘要 |
| **Koska 等**（arXiv 2611.04447，2026-11） | Lean 4 中的 PTX 语义：逐线程小步语义加公理化的 PTX 内存模型（Lustig 等；省略 no-thin-air）、proxy 标签、`bar.sync`、mbarrier 相位完成、TMA、`cp.async`、`wgmma.commit/wait_group` | 对每个合法 launch 证明 Hopper GEMM 的功能正确性（部分正确性）。这个 GEMM 是 f16 输入、两级流水、`K = 16T`、`T ≥ 2`。另证 FlashAttention 和 3-kernel MoE。信任基包括：PTX 解析器与约 2.6k 行模型、Lean 内核、“无 UB”假设，以及 ptxas 和硬件。共 81 种指令形式，没有 tcgen05 和 DSMEM。 | 手写证明（LLM 辅助）：约 34k 行证明对应 454 条指令 | **手写 PTX** | 【网页】 |
| **CUTLASS `PipelineState`** | 运行时对象 `(index, phase, count)`，回绕时翻转 phase | 没有静态保证，两次 wait 或漏释放都可能发生 | — | 库和用户 | 【报告】 |
| **GPUVerify**（Betts 等，OOPSLA'12；Barrier Invariants，OOPSLA'13） | kernel 的数据竞争和屏障发散 | 用两线程归约加 Boogie/SMT 做验证；循环要不变式，可自动推断，也可手写。没有异步屏障。 | 基本自动 | 用户 kernel（源码或 IR） | OOPSLA'13：【报告】；其余未核实 |
| **VerCors**（Blom、Huisman 等，SCP 2014） | 基于权限的分离逻辑；屏障处交换权限 | 演绎验证，信任 VerCors 及其后端 | 需要大量标注 | 用户 kernel | 【报告】 |
| **Faial**（Cogumbreiro 等，CAV'21 及期刊版） | 从 CUDA kernel 推出访问内存协议，检查无数据竞争 | 用 SMT 判定；分析理论在 Coq 中机械化。报告验证了 227 个真实 kernel 中的 218 个。 | 自动 | 用户 kernel | 【网页】 |
| **Mojo 结构化 kernel 流水**（Modular） | `ProducerStage`/`ConsumerStage` 是 `not Deinitable` 的线性句柄，必须 `release()` 才能推进阶段；后端可插拔（`PipelineBackend`）。 | 编译器强制每个阶段句柄被交还一次；槽号 `_index` 与相位是运行时值，不进类型。 | 自动（类型检查） | 手写 kernel 库 | 【本地】`modular/max/kernels/src/structured_kernels/pipeline.mojo:495-540`（汇总者核实） |
| **Triton ConSan**（Concurrency Sanitizer） | 运行时插桩：追踪每个缓冲的可见写/读前沿，模拟 mbarrier 与 commit 计数，覆盖 cp.async、WGMMA、TMA。 | 动态、逐次执行，发现的是这次运行里出现的竞争，不是证明。 | 自动（插桩） | 编译器输出（Triton kernel） | 【本地】`triton/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md`（见 A 文档 §2.4） |
| **cuda-oxide**（NVlabs，alpha） | Rust → MIR → Pliron → LLVM → PTX。`ManagedBarrier<State, Kind, ID>` 的 typestate 是 `Uninit → Ready → Invalidated`；写入用 `DisjointSlice`；另有 typed launch contract。 | typestate 防住未初始化就用、重复初始化、失效后再用。`BarrierToken` 是 `Copy`；`try_wait_parity(u32)` 的奇偶由用户给；没有槽和代的追踪。 | Rust 类型检查，自动 | 用户代码 | 【本地】`crates/cuda-device/src/barrier.rs` |
| **Rust-GPU / Rust-CUDA** | 借用检查器覆盖主机语义；共享内存和屏障多为 `unsafe` | 不覆盖跨线程同步 | 自动 | 用户代码 | 未核实 |

### 1.2 线性、typestate、会话与依赖类型

| 工作 | 机制 | 与我们相关的一点 | 核实 |
|---|---|---|---|
| Linear Haskell（Bernardy 等，POPL'18） | 箭头上的重数 | 我们的“单次使用”在 PE 之后数图上的使用，而不是写在类型里；codex 建议的长期语义是 `Pi` 域重数 | 未核实 |
| ATS（Xi） | 带静态整数索引的线性 view；环缓冲在 `i mod n` 处切分 view | 它属于“绝对下标 + 约束求解”路线，我们没有走 | 未核实 |
| Mezzo（Pottier、Protzenko，ICFP'13） | 权限；循环写成递归，参数的权限就是不变式 | — | 未核实 |
| Alms（Tov、Pucella，POPL'11） | 仿射 typestate；循环入口和出口的状态相同 | 正好对应我们的两相 `Ring rid depth {ff, tt}`：循环不变，不依赖迭代号 | 未核实 |
| Granule / QTT / Idris 2 | 分级模态和数量类型；重数 0 的下标被擦除 | 我们的 `rid` 和槽号只出现在类型里，`erase_phases` 把它们擦掉，相当于重数 0 | 未核实 |
| 有界缓冲的会话类型（Lange–Yoshida k-MC，CAV'19；Gay–Vasconcelos，JFP'10） | k 有界的通道安全；类型里没有下标 | 深度为 d 的环相当于 k = d；它之所以可判定，正因为下标不在类型里 | k-MC：【报告】；其余未核实 |

### 1.3 幽灵状态与分离逻辑

| 工作 | 机制 | 关系 | 核实 |
|---|---|---|---|
| Iris / RustBelt（Jung 等） | 高阶并发分离逻辑、幽灵资源（权威/片段） | 我们的 Lean 层没有用分离逻辑，而是事件时间表加 `≺`。Meta XROS 用 Iris 证明环队列（逻辑“缎带”、绝对位置）【报告】，是“绝对幽灵下标、物理槽 = n mod D”的先例 | 未核实（XROS：【报告】） |
| Verus（Lattuada 等，OOPSLA'23） | 线性幽灵类型，编译时擦除 | 我们的 `Held`/`Vacant`/`Ring` 也是擦除的线性令牌，**但没有元理论**：`check_sync` 的规则没有可靠性证明 | 存在：【报告】 |

### 1.4 翻译验证与证书编译

| 工作 | 机制 | 与我们的差异 | 核实 |
|---|---|---|---|
| Pnueli 等（TACAS'98）提出“翻译验证”；Necula（PLDI'00）把它用于 gcc 的优化 | 每次编译后，检查输入与输出的语义等价 | 我们不查等价，只查一条**协议安全性质**（T1/T2） | 未核实 |
| Tristan–Leroy，“A simple, verified validator for software pipelining”（POPL'10） | 对每个实例做符号求值，核对序言、稳态核、尾声与展开后的原循环一致；验证器本身在 Coq 中证明 | 它是逐实例验证软件流水的直接先例。差异：它的验证器直接处理编译器自己的 IR，**没有提取这一层**；我们的提取器（`ring_trace`）未验证 | 【报告】 |
| CompCert 的其他验证器（指令调度、lazy code motion、寄存器分配等） | “未验证的启发式 + 已验证的检查器” | 我们的 Lean 检查器（`accepts`）及其可靠性（`fold_of_accepted`）属于这一模式，但输入要经过未验证的提取 | 未核实 |

## 2. 逐项与我们的关系

- **Async Graphene。**
  - 两边都把环、槽、完成机制放进类型，都在编译期擦除。
  - 它的类型不含游标，是“相对坐标”的先例；我们的两相 `Ring` 也是。
  - 它没有线性，没有可靠性定理，延迟释放也不检查。我们在这三处多做了：线性令牌、Lean 定理、证书检查。但我们的延迟释放**在类型层同样没有检查**，见 §3.2 的 `InFlight`。
- **Descend 与 Kuiper。**
  - 两者都面向用户代码，只有块级同步。
  - Kuiper 用绝对迭代号加“对所有 `it`”的证明义务，代价是手写证明。我们的代价是只覆盖有限的协议族，并且要求每个实例的事件序列可以被枚举出来。
- **Koska 等。**
  - 它的硬件语义比我们的细：有公理化内存模型和 proxy。我们的 `Machine`/`Timeline` 规则是对 PTX 文档的手工抽象。
  - 它证明的是一个手写 kernel 的功能正确性，并固定了流水深度；我们证明协议族对任意 `d`、`count` 成立，并且自动检查编译器的输出。
  - 两者可以互补：用它的语义作为我们硬件假设的下层模型（§5 的方向 3）。
- **Tawa、CUTLASS。** 它们是我们的编译器轴对照：都自动生成同步，都没有检查。我们能为“自动生成的同步可被检查”提供实证。
- **GPUVerify、Faial、VerCors。** 它们关心数据竞争，不建模异步代理、mbarrier 相位和 wgmma 组。我们的 T2 是一种受限的“读后写不竞争”，只针对环槽。
- **cuda-oxide。** 屏障生命周期的 typestate（`Uninit`/`Ready`/`Invalidated`）与我们正交：我们没有建模初始化和失效，它没有建模槽、代和相位。

## 3. 对我们工作的诚实评估

### 3.1 贡献与证据

| # | 贡献 | 证据 |
|---|---|---|
| C1 | PE 之后检查线性/仿射槽令牌：刚性比较、私有构造器、拒绝 `Hole`/`⊥`，闭包调用追踪 | `8c9340acc`、`201b2f5ac`、`3a36603f3`；`src/mim/plug/gpu/phase/check_sync.cpp`（505 行）；`lit/error/ring_token_*.mim`、`ring_slot_*.mim`，以及 `lit/gpu/ring_token_closure_*.mim` 这类正例 |
| C2 | 环身份 `rid`、类型化面板句柄 `Panels (rid, depth)`、两相 `Ring (rid, depth) Bool` | `b8322375e`、`961c8c9a6`、`d4b2e86f7`；`gpu.mim:868-912`；`lit/error/ring_token_other_ring.mim`、`ring_panels_other_ring.mim`、`ring_await_twice.mim`、`ring_state_alias.mim`。擦除后 73 个 benchmark 中 71 个的设备 IR 逐字节相同，另 2 个只差一个值名（设计文档 §21.6） |
| C3 | Lean 协议定理 T1（`read_current_generation`）和 T2（`overwrite_after_last_read`），覆盖四个模型：单消费者 TMA 小步模型、cp.async、合作 TMA、WS（empty mbarrier） | `proofs/ring/Ring/*.lean`；`822bf0149`、`db9aabb61`、`c2c79dd69`、`33ad70696`、`dbe26e329`。没有项目自定义公理（`#print axioms` 只列出 `propext`、`Classical.choice`、`Quot.sound`）。另有 `no_ABA`、`exists_drained_iff`，用来排除空真 |
| C4 | 可执行检查器，并证明“接受 ⇒ `Fold` ∧ `ReadCode` ⇒ T1/T2” | `AsyncCode.lean`（`391a25ebd`）；`TmaCode`/`CoopCode`/`WsCode.lean`（`82366ff0e`）；`fold_of_accepted`、`readCode_of_accepted`、`diagnose_none`/`diagnoseRole_none` |
| C5 | 逐 kernel 的证书检查：从残余里提取事件，覆盖 7/11 个真实环 | `ring_trace.cpp`（`f14560665`、`1176f89a1`）；`bench/tools/ring_trace_check.py`；`lit/gpu/ring_trace_async.mim`、`ring_trace_tma.mim`；`gemm_tiled_wg_tma` 有 181 个事件，`gemm_tiled_wg_ws` 有 107 个 |
| C6 | 变异评测：15 种变异 × 7 个 kernel，44 对改变了生成代码的组合全部被拒：类型/`check_sync` 8（含接上 `InFlight` 见证后的 `early_release` 2 对）、有界模拟 `ring_safe` 1、Lean 35（其中 1 对是文本格式解析失败）；另有 7 对 `hold_slot` 只改擦除前的幽灵下标，生成代码不变，被类型检查拒（口径见 README §2），7 个基线全部接受 | `bench/tools/mutate.py`（`2ce447bc1`）；设计文档 §22 |
| C7 | 覆盖分类（哪些环落在模型前提内） | `f8385cf39`、`2942730ec`；`bench/tools/ring_coverage.py`；`lit/gpu/ring_fold_coverage.mim` |

### 3.2 信任链与缺口

信任链：`gpu.mim` 协议 → PE → **类型检查与 `check_sync`** → **`ring_trace` 提取**（解释 `check_sync` 时刻的残余）→ **Lean 检查器** → T1/T2（在手写的硬件模型下）。此后还有 `erase_phases` → `mem.add_mem` → `affine.lower_for` → LLVM → ptxas → 硬件。

| 缺口 | 现状 | 后果 |
|---|---|---|
| G1 提取器未验证 | `ring_trace` 是 C++ `Rewriter`，借 `Lam::reduce` 在主机上解释残余；`ring_trace_check.py` 是胶水脚本；Lean 可执行文件的文本解析器也没有证明（`diagnose_none` 只覆盖解析之后） | 提取器的 bug 会被当作正确实例接受。这是 Tristan–Leroy 没有的一层 |
| G2 解释的是 `check_sync` 时刻 | `erase_phases`、`mem.add_mem`、`affine.lower_for`、LLVM 后端（含 `nuw nsw`）、ptxas 都不检查 | 设计文档 §13 记录过 ptxas 把读累加器的指令调度到 wgmma 等待之前，这类问题不在范围内 |
| G3 硬件模型手写 | mbarrier、TMA 字节计数、`wait_group`、proxy 规则都是按 PTX 文档写的 `Consistent` 规则。只有单消费者 TMA 有字节级小步模型，其他三个模型是事件时间表 | 模型可能比真实硬件更强。没有与 Koska 的 PTX 语义或真实 GPU 对照过 |
| G4 代表线程与收敛 | `gpu.bind` 循环只走一次迭代；每个 warpgroup 追踪一个代表线程（0、128、…）；warpgroup 收敛是假设 | 只发生在其他线程上的错误看不到：写入区间冲突、部分线程没到 `bar.sync` 等。“部分 CTA 当全组”这一真实 bug 类（设计文档 §2.3）恰好在盲区里 |
| G5 地址到槽的映射 | 槽按 fold 循环初态里环数组的位置命名（`Tracer::ring`）；TMA 每条加载的字节数取自 `buf_tensor_map`/`device_tensor_map` 的盒子尺寸 | 描述符和实际地址与槽之间的对应由提取器假定，没有证明 |
| G6 类型层没有元理论 | `check_sync` 是一组白名单规则，没有可靠性定理；`ring_idle` 不计使用次数，所以从初态出发的两次 `await` 不被拒（§21.6）；Nat 的语义（数学自然数还是 u64）未定（§14 P0-4） | 类型层只能当作“过滤器”，不能声称“良类型即安全” |
| G7 `InFlight` 见证被丢弃 | `ws_fold_at` 中 `let (m, a, _) = cps.cps2ds step …`（`gpu.mim` 约 1186 行） | `early_release` 只有 Lean 能抓到，类型层抓不到 |
| G8 覆盖范围 | 11 个环覆盖 7 个；cluster 2 个、persistent 1 个、epilogue 叠放 1 个不在模型内；`ws_fold` 假设 `nth i = i` | persistent 和 cluster 正是性能最好的配置 |
| G9 变异集的代表性 | 15 种变异全是手写的、只改 `.mim` 协议库；不改 `LowerToMem` 的 C++（例如 `rid` 分配、深度选择、面板叠放）；“same” 按规范化后的设备 IR 判定 | “0 MISSED” 只对这 51 对成立；有 54 对的变异没有体现在设备 IR 上 |
| G10 不证无死锁 | `exists_drained_iff` 只给出一条能排空的执行，不是对所有调度的进展性 | 与 Kuiper 和 Koska 相同，都是部分正确性 |

### 3.3 可以主张的与要避免的

**最强的可辩护主张**（建议原文）：

> 对基准中落在模型前提内的 7 个环（共 11 个），编译器在擦除令牌前的残余 IR 中，每个角色一个代表线程的同步事件序列，都被一个 Lean 中证明可靠的检查器接受。按 Lean 定理，在手写的 mbarrier/cp.async/WGMMA 硬件模型下，被接受的序列对任意一致的交错满足“读到本代”和“覆盖晚于上一代的最后一次读”。
> 另外，类型层（线性令牌、环身份、两相状态）在前端拒绝一类协议库错误。
> 15 种手写协议变异使 44 个（变异，kernel）组合的生成代码发生变化，全部被拒：类型层与 `check_sync` 8 个，有界协议模拟 1 个，Lean 检查器 35 个。

**要避免的主张：**
- “验证过的编译器”“端到端正确”“kernel 被证明无竞争”。原因：G1、G2、G4。
- “类型系统保证同步安全”。原因：G6、G7；类型层只抓到改变代码的 44 对中的 8 对。
- “检查任意 Hopper kernel”或“覆盖生产配置”。原因：G8，persistent 和 cluster 都不在内。
- “0 漏报”而不加限定。原因：G9。
- “首个给 mbarrier 定类型的系统”。新颖性检索不穷尽，只能写“据我们所知”。
- “与 Koska 的 PTX 语义一致”。两者尚未连接。

## 4. 与相关工作的差异小结

| 维度 | Async Graphene | Kuiper | Koska | Tawa | 我们 |
|---|---|---|---|---|---|
| mbarrier 的相位和代 | 运行时 | 不建模 | 建模（PTX 级） | 运行时 | 模型和检查器中按奇偶与票号 |
| 对象 | 用户 DSL | 用户 F\* | 手写 PTX | 编译器输出 | **编译器输出**（另可手写 Mim） |
| 定理 | 无 | 有（部分正确性） | 有（功能正确性） | 无 | 协议安全 T1/T2，对任意 d 和 count |
| 每个 kernel 的人工代价 | 低 | 高 | 很高 | 无 | 无（自动提取与检查） |
| 信任基中的编译器 | 全部 | Karamel、nvcc | ptxas | 全部 | 提取器、后续 phase、LLVM、ptxas |

## 5. 未来方向（按论文价值排序）

| 排名 | 方向 | 补哪个缺口 | 价值 / 代价 |
|---|---|---|---|
| 1 | **从 PTX 提取事件**：对最终 PTX 按代表线程做符号执行，把 `mbarrier.*`、`cp.async*`、`cp.async.bulk.tensor`、`wgmma.*`、`bar.sync`、`fence.proxy.async` 映射到同一种文本格式，送进同一组 Lean 检查器 | G2，部分 G5（地址按 PTX 求值） | 最高：把 `mem.add_mem`、`lower_for` 和 LLVM 移出信任基，只剩 ptxas 和硬件。代价中到高：需要 PTX 子集解释器和 `%tid` 代入；地址到槽的映射可以用共享内存符号基址解决 |
| 2 | **GPU 差分测试**：对 44 个改变了代码的变异 kernel 和基线跑 `compute-sanitizer`（racecheck、synccheck）和数值比对 | G3、G9 | 高：给“被拒的确实是 bug”“same 的确实等价”提供硬件证据，也可能暴露硬件模型过强的地方。racecheck 对 TMA/mbarrier 异步代理的支持程度**未核实**，可能需要靠数值差异和压力调度（改变 SM 占用、插入 nanosleep） |
| 3 | **把硬件模型接到 Koska 的 PTX 语义**：证明我们 `Consistent` 中的每条规则在其内存模型下成立 | G3 | 高，但代价大。它依赖 Mathlib 和 Lean 4.35（我们用的是 4.36，无外部依赖）；对方也没有展开 mbarrier 的奇偶语义（【网页】），可能要一起补 |
| 4 | **把 `InFlight` 见证接到 `release`**：`release` 要求一个证明“该槽的批次已退休”的见证；`ws_fold_at` 不再丢弃它 | G7 | 中：代价低，可把 `early_release` 一类挪到类型层，改变变异表中 static 与 lean 的比例，便于讲“分层” |
| 5 | **cluster 与 persistent 模型**：persistent 需要按槽实际填充次数编号的不变式，并去掉 `nth i = i`；cluster 需要 multicast 和远端 arrival | G8 | 中高：覆盖可从 7/11 升到 10/11，而且覆盖到性能最好的配置。代价高 |
| 6 | **线程覆盖**：对每个角色枚举全部线程（或每个 warp 的 lane 0 加谓词抽象），检查各线程的写区间不相交，并用一致性分析证明 warpgroup 收敛 | G4 | 中：直接对应真实 bug 类“部分 CTA 当全组” |
| 7 | **类型层收口**：`ring_idle` 只由 `open` 产生并带 mem；写出 `check_sync` 规则的形式化陈述，至少在 Lean 中证明“PE 后令牌图满足规则 ⇒ 事件序列满足检查器的某些局部条件” | G6 | 中：让类型层有一个可以陈述的意义 |
| 8 | **扩大变异空间**：变异 `LowerToMem` 的 C++、做随机变异、复现历史 bug（设计文档 §29 的跨轮叠放） | G9 | 中：增强评测的说服力，代价低 |
| 9 | **区域 typestate 与叠放模型**（设计文档 T3） | G8 的叠放环 | 低到中：只多覆盖 1 个环，但对应一个真实的历史 bug |

## 6. 资料

- 本地：`docs/linear-types-design.zh-CN.md` §13–§22；`docs/reviews/ringstate_report.md` §5；`proofs/ring/README.md`；`src/mim/plug/gpu/gpu.mim:868-1216`；`bench/tools/mutate.py`；`/home/chzh00001/workspaces/cuda-oxide/crates/cuda-device/src/barrier.rs`。
- 网页：
  - [Koska 等，arXiv 2611.04447](https://arxiv.org/abs/2611.04447)（[HTML 版](https://arxiv.org/html/2611.04447v1)）
  - [Tawa，arXiv 2510.14719](https://arxiv.org/abs/2510.14719)
  - [Faial，CAV'21](https://link.springer.com/chapter/10.1007/978-3-030-81685-8_19)
  - [Faial 期刊版](https://par.nsf.gov/biblio/10493403)

## 补记（汇总时，依据 agy 核对）

- 漏掉了 **WEFT**（PLDI'15）、**Volta**（OOPSLA'26）、**RESOLVE**（2026，Kuiper 团队，把 F\*/Pulse 证明用到 Triton、CUTLASS、Gluon 的输出上）、**Prism**（PLDI'26）、**Lustig 等**的 PTX 内存模型（ASPLOS'19）、**Fehr 等**的 MLIR 验证方言（PLDI'25）。它们对本文主张的影响见 [README §6](README.zh-CN.md)。
- 上面说 Kuiper“只作用于用户代码”，对 Kuiper 本身成立，但它的后续 RESOLVE 已经作用于编译器输出。
- Mojo 的阶段句柄在类型里带着 pipeline 的 origin、环深和面板负载；只有槽号和相位是运行时值（`modular/.../sm100_structured/structured_kernels/tile_pipeline.mojo:809-839`）。
