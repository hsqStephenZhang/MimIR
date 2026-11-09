# 文献调研：GPU 多级 tile-and-fuse 形式化的难点（2026-11-09）

由 codex-cli 0.167.0（`--search` 联网、只读沙箱）针对 `gpu-tiling-formalization.zh-CN.md` 的难点所做调研的原文，未经修改。
抽查核实：Kuiper（PLDI 2026，DOI 10.1145/3808280；原文确认"we do not model the collective aspect of the MMA operations"，且把 `mma_sync` 当作原语）、CuTe Layout Representation and Algebra（arXiv 2603.02298）、Twill（OSDI 2026）、Typed Perspectives / Prism（PLDI 2026 Distinguished Paper，arXiv 2511.11939）均存在且出处与正文一致；其余引用未逐一核对。

# 文献调研：GPU 多级 tile-and-fuse 的形式化基础

检索截止 **2026-11-09**。总体上，受限仿射域上的 tiling 覆盖、布局单射判定、结构化 GPU kernel 的数据竞争/普通 barrier 验证，以及纯张量语言中的调度重写，已经有相当成熟的方法；TensorIR、Exo、Descend、ATL、Faial、isl 等可以直接提供设计模板。相反，`cp.async`/TMA/`mbarrier` 与 warp specialization 的整程序验证、tensor-core collective 收敛性、浮点归约重组、跨 kernel 常驻与同步，以及把寄存器压力和 occupancy 纳入可证明的调度空间，仍主要依赖受限模型、差分测试或工程协议。2026 年的 Kuiper 已经证明了含两级 tiling、shared memory 和 tensor core 的 GEMM，但论文明确没有建模 MMA 的 collective 收敛性；这很好地界定了当前前沿。

以下将“形式化保证”严格区分为：

- **机械证明**：Coq/F\*/SMT 等检查过的定理；
- **静态验证**：在给定抽象和前提下可靠地拒绝非法程序；
- **IR verifier**：结构、作用域、shape 等一致性检查，不等于保义证明；
- **工程约定**：实现、文档和测试支持，但没有对应定理。

---

## 1. 多级调度的类型化表示与作用域保持

### 代表性工作

- Siyuan Feng 等，**TensorIR: An Abstraction for Automatic Tensorized Program Optimization**，ASPLOS 2023。[论文/元数据](https://arxiv.org/abs/2207.04296)
- Yuka Ikarashi 等，**Exocompilation for Productive Programming of Hardware Accelerators**，PLDI 2022。[论文页](https://pldb.kirancodes.me/p/conf/pldi/IkarashiBRGR22)
- Yuka Ikarashi 等，**Exo 2: Growing a Scheduling Language**，ASPLOS 2025。[论文页](https://dspace.mit.edu/entities/publication/a69989d9-d724-44a9-828b-5a0d0191ca8f)
- Bastian Köpcke 等，**Descend: A Safe GPU Systems Programming Language**，PLDI 2024。[论文](https://steuwer.info/files/publications/2024/PLDI-2024.pdf)
- Bastian Hagedorn 等，**Graphene: An IR for Optimized Tensor Computations on GPUs**，ASPLOS 2023。[作者论文页](https://mgarland.org/papers/2023/graphene/)
- Bastian Hagedorn 等，**Fireiron: A Scheduling Language for High-Performance Linear Algebra on GPUs**，arXiv 2020；未见同行评审会议信息，故按预印本计。[论文](https://arxiv.org/abs/2003.06324)
- Martin Paul Lücke 等，**The MLIR Transform Dialect: Your Compiler Is More Powerful Than You Think**，CGO 2025。[论文](https://michel-steuwer.github.io/files/publications/2025/CGO-2025-2.pdf)

### 核心方法与保证边界

**TensorIR** 把一个计算块的迭代变量、读写 region、初始化、归约和 block scope 显式放入 IR。调度原语在应用时检查 block 是否仍满足 complete/reduction block、dominance、scope 和 region 条件。其优点是合法性检查附着于结构化 IR，而不是只在 lowering 中凭上下文猜测。它主要是 **IR verifier 和调度前置条件检查**，并不是 GPU 并发语义下的完整保义证明。

**Exo/Exo 2** 的调度是受信任细粒度重写的组合。Exo 的 effect analysis 用符号化 location set 判断交换、删除、分裂等重写是否合法，目标包括程序等价和内存安全；Exo 2 允许用户在受信任 primitives 上定义调度库。这比 MLIR Transform dialect 更接近 MimIR 的 `Schedule = Tileable → Tileable`：组合子本身携带保义条件，而不是只报告“变换执行成功”。

**Descend** 把 grid、block、warp、thread 作为 *execution resources* 放入类型系统；值的所有权、borrow 和某条指令由哪个执行层次执行是类型的一部分。其 Rust 风格系统能静态排除越界生命周期和一类 GPU 数据竞争。这是防止“整块简化后 binding scope 消失”最直接的先例：scope 不应只是 loop annotation，而应成为不可无声擦除的 capability/resource。

**Graphene** 把多维线程和数据都表示为一等张量，并递归分解为 thread tile/data tile。它很擅长表达 grid→block→warp→lane 与 fragment 的层次映射，但论文重点是表达能力与性能，没有给出调度保义或并发安全定理。

**Fireiron** 用 decomposition 逐步细化 GEMM，把数据位置、线程映射和 tensor-core 指令作为一等决策；但后续研究明确指出 Fireiron 不能保证 schedule 保义。因此它适合借鉴语法，不适合作为 MimIR 的证明基础。

**MLIR Transform dialect** 有 typed handles、consume/invalidate 规则、transform pre/post-condition interface 和失败传播；这能避免变换继续引用已被替换的 payload op。不过某些变换只做粗略检查，文档也明确把独立性等条件留给用户，所以“transform 成功”不等于语义等价。[官方文档](https://mlir.llvm.org/docs/Dialects/Transform/)

### 对 MimIR T1–T7 的启发

可直接借鉴：

1. 为每一级引入不可擦除的作用域能力，例如：

   ```text
   Scope grid > block > warp > lane
   Region scope effects
   Dist scope owner-map
   ```

   整块 tiling 可以退化为 extent=1，但不得删除 `block`/`warp` scope。

2. 仿 TensorIR，把 region 的 iter-kind、read/write footprint、parent scope 和 reduction kind 放入可验证 descriptor。

3. 仿 Exo，把“调度选择”与“受信任的细粒度合法重写”分开；用户或搜索器只能组合已证明/已验证的 primitive。

4. Transform dialect 的 handle invalidation 可借给 T2 fusion：producer 被 clone、tile 或 erase 后，旧 region handle 必须失效。

不宜直接照搬：

- TensorIR/MLIR 的 verifier 主要保证结构合法，不足以证明 T5–T7 的 GPU 并发、collective 和数值语义。
- Descend 面向源语言 ownership；MimIR 是编译器 IR，若完整移植 Rust borrow checking，工程成本可能过大。更现实的是采用轻量 scope capability 加 footprint/ownership checker。

---

## 2. 布局代数与单射/双射判定

### 代表性工作

- Keren Zhou 等，**Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using \(\mathbb F_2\)**，arXiv 2025，预印本。[论文](https://arxiv.org/abs/2505.23819)
- Cris Cecka，**CuTe Layout Representation and Algebra**，arXiv 2026，预印本。[论文](https://arxiv.org/abs/2603.02298)
- Bastian Hagedorn 等，**Graphene**，ASPLOS 2023。[论文页](https://mgarland.org/papers/2023/graphene/)
- Sven Verdoolaege，**isl: Integer Set Library**；当前文档提供 Presburger set/map、injectivity 和 bijectivity 判定。[官方手册](https://libisl.sourceforge.io/user.html)
- Phitchaya Mangpo Phothilimthana 等，**Swizzle Inventor: Data Movement Synthesis for GPU Kernels**，ASPLOS 2019。[论文页](https://doi.org/10.1145/3297858.3304059)

### 核心方法

**Triton Linear Layouts** 将

```text
(reg-bit, lane-bit, warp-bit, block-bit) → tensor-coordinate-bit
```

表示为 \(\mathbb F_2\) 上的矩阵。组合是矩阵乘法；XOR swizzle 自然属于线性映射；单射、满射和逆可通过 rank 与高斯消元判定。论文还证明了该布局族在 Triton shape operators 下的闭包性，并用统一 layout conversion 取代大量 case-by-case lowering。

它的主要限制是维度通常要提升到 2 的幂并通过 masking 处理尾部；一般的整数 stride、非 2 幂 padding、`floorDiv/mod` rotation 不是纯 \(\mathbb F_2\) 线性映射。

**CuTe** 用 hierarchical shape/stride 表示布局，并定义 composition、complement、division、tiling、coalescing、inversion。它非常适合表达 memory tile、thread-value layout 与 tensor-core atom 的组合。2026 论文是数学规格与参考实现，不是 proof-assistant 机械证明；对 complement/logical division 的一般存在性和唯一性仍需谨慎，不能把 CUTLASS 的模板可实例化误认为一般判定定理。

**isl/Presburger** 对常数乘法、加法、条件分段、常数 `floorDiv/mod` 的整数关系可以精确表达。有限域上的单射条件可化为：

```text
¬∃x,y. x≠y ∧ Dom(x) ∧ Dom(y) ∧ L(x)=L(y)
```

即检查一个 Presburger set 是否为空。isl 已直接提供 `isl_map_is_injective` 和 `isl_map_is_bijective`。这正覆盖草案现有 perm/split/rotation 子集。

**Swizzle Inventor** 通过程序综合寻找满足规范的数据移动/swizzle；规范要求结果与原计算等价，并用求解器填充 shuffle holes。它说明布局选择可以留给搜索器，但搜索结果仍应接受独立的等价/单射检查。

### 对 T3/T7 的建议

建议把 `LayoutExpr` 分成至少两个可判定子代数：

```text
SemiAffineLayout:
  perm | split_const | merge_const | pad_stride
  | rotate_mod_const | compose

F2LinearLayout:
  xor_swizzle | bit_permute | bit_slice
  | product | compose
```

并分别提供：

- `InjectiveSA`：Presburger/isl 判定；
- `InjectiveF2`：矩阵 rank；
- `BijectionFrag`：有限 cardinality 相等加 injective，或直接检查方阵满秩；
- 字面量小域的 enumeration 仅作为后备检查和测试 oracle。

这比接受任意 Mim 函数后再尝试归一化可靠得多。`pad` 应主要进入 physical extent/stride，而非伪装成逻辑坐标置换。CuTe 可用于设计组合 API，Triton Linear Layouts 可用于 T7 的 lane/register fragment，isl 可覆盖现有 shared-memory rotation。

---

## 3. 线程到数据的分布与所有权

### 代表性工作

- Yaoyao Ding 等，**Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs**，ASPLOS 2023。[论文](https://arxiv.org/abs/2210.09603)
- Bastian Hagedorn 等，**Graphene**，ASPLOS 2023。[论文页](https://mgarland.org/papers/2023/graphene/)
- Bastian Köpcke 等，**Descend**，PLDI 2024。[论文](https://steuwer.info/files/publications/2024/PLDI-2024.pdf)
- Guido Martínez 等，**Kuiper: Correct and Efficient GPU Programming with Dependent Types and Separation Logic**，PLDI 2026。[论文页](https://pldi26.sigplan.org/details/pldi-2026-papers/37/Kuiper-Correct-and-Efficient-GPU-Programming-with-Dependent-Types-and-Separation-Log)
- Keren Zhou 等，**Linear Layouts**，2025 预印本。[论文](https://arxiv.org/abs/2505.23819)

### 核心方法与保证

**Hidet task mapping** 明确定义 `worker → ordered list of tasks`，并提供 spatial、repeat 和 mapping composition。组合是结合的但通常不交换。它非常适合表示：

- spatial：UniqueOwner；
- repeat：Replication；
- spatial×repeat：一个线程处理多个元素；
- 层次 composition：block/warp/lane 的分布。

但 Hidet 的代数主要是编程与生成模型，没有证明生成 kernel 的 DRF 或功能等价。

**Graphene** 用线程张量到数据张量的映射统一表达 cooperative copy、MMA fragment 和 fused kernels。它能作为 `Dist` 的表示设计参考，但不提供 ownership soundness theorem。

**Descend** 将 execution resource 与内存 object ownership/borrow 结合，能静态排除不同 execution resource 对同一对象的不安全访问。

**Kuiper** 使用 F\* 依赖类型与 Pulse 并发分离逻辑，将 GPU/global/block-local 资源表示为 located resources，并从每线程规格提升到 kernel 规格。它能证明 DRF、内存安全以及可选的功能正确性；这是 owner/partition witness 最强的现成先例。

### 与 tiling 覆盖证明的关系

草案应把 T1 的空间覆盖和 T4 的执行所有权拆成两个关系：

```text
TileCover:
  每个逻辑点属于哪些 tile

ExecutionCover:
  每个 (tile, logical point) 由哪些 processor instance 执行
```

然后为不同模式定义不同义务：

| 模式 | 覆盖义务 |
|---|---|
| `UniqueOwner` | total + 单值；不同 owner 的写 footprint 不交 |
| `Replicated` | total；多个副本只读，或写值可证明相同且语义允许 |
| `ReductionPartition` | 每个 reduction 点恰属一个 partial；partials 的 merge 覆盖完整 |
| `CooperativeTransfer` | 每个目标元素至少一次且无冲突；通常要求唯一写者 |
| predicated lane | active 集合与有效 tile 域恰好对应 |

R8 当前漏洞属于 `ReductionPartition` 的不全覆盖，而不是输出 box 的 `ExactCover`。这一点与 Hidet 的显式 worker→task map、Kuiper 的资源分割最一致。

---

## 4. GPU 同步正确性

### 代表性工作

- Adam Betts 等，**The Design and Implementation of a Verification Technique for GPU Kernels**，TOPLAS 2015（GPUVerify）。[论文页](https://doi.org/10.1145/2743017)
- José F. Santos 等，**Checking Data-Race Freedom of GPU Kernels, Compositionally**，CAV 2021（Faial）。[论文](https://cogumbreiro.github.io/assets/faial-cav21.pdf)
- Mingyuan Wu 等，**Simulee: Detecting CUDA Synchronization Bugs via Memory-Access Modeling**，ICSE 2020。[论文](https://lingming.cs.illinois.edu/publications/icse2020b.pdf)
- Guodong Li、Ganesh Gopalakrishnan，**Scalable SMT-based Verification of GPU Kernel Functions**，FSE 2010（PUG，早于重点年份，作为历史背景）。[论文信息](https://pldb.kirancodes.me/p/conf/sigsoft/LiG10)
- Guodong Li 等，**GKLEE: Concolic Verification and Test Generation for GPUs**，PPoPP 2012（历史背景）。[论文信息](https://pldb.kirancodes.me/p/conf/ppopp/LiLSGGR12)
- Daniel Lustig、Sameer Sahasrabuddhe、Olivier Giroux，**A Formal Analysis of the NVIDIA PTX Memory Consistency Model**，ASPLOS 2019。[论文页](https://research.nvidia.com/publication/2019-04_formal-analysis-nvidia-ptx-memory-consistency-model)
- Sourav Anand、Nadia Polikarpova，**Automatic Synchronization for GPU Kernels**，FMCAD 2018（AutoSync）。[论文](https://cseweb.ucsd.edu/~npolikarpova/publications/fmcad18.pdf)
- Michael Bauer、Rahul Sharma、Alex Aiken，**Verification of Producer-Consumer Synchronization in GPU Programs**，PLDI 2015（WEFT）。[论文页](https://research.nvidia.com/publication/2015-06_verification-producer-consumer-synchronization-gpu-programs)

### 核心方法与形式化程度

**GPUVerify** 使用同步、延迟可见性（SDV）语义和线程对归约，把大量线程的 race/barrier-divergence 问题转成 Boogie/SMT 验证。它验证：

- barrier 是否处于 uniform control flow；
- barrier 循环次数是否一致；
- barrier interval 之间是否存在冲突访问。

它是 sound static verification，但依赖受限 CUDA/OpenCL 子集、循环不变量和 launch 前提。

**Faial** 把 kernel 抽象成 access-memory protocols，将“两个访问能否并发”与“地址是否冲突”分离。其 DRF 判定对协议模型的 soundness/completeness 已在 Coq 中机械证明。它比直接移植 GPUVerify 更适合草案 T5，因为协议很接近 BSP superstep 的 read/write footprint。

**Simulee/GKLEE** 是动态或 concolic bug finding：能找到真实 race、barrier divergence 和冗余 barrier，但不能证明所有输入下无 bug。

**PTX memory-model 工作** 把官方 PTX 模型写成公理模型，使用 Alloy 检查 bounded cases，并在 Coq 中证明 scoped C++→PTX 映射 sound。它说明 T5 的 `⊑obs` 必须固定到具体 target memory model；裸写“DRF ⇒ 顺序化”不够。

HRF 系列的核心结论是：只有同步 scope 足够覆盖通信双方，才有类似 SC-for-DRF 的保证。Vulkan 同样显式区分 execution scope、memory scope、storage-class semantics 和 availability/visibility。[Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)

**AutoSync** 用 GPUVerify 作 correctness oracle，把 barrier placement 编码为 MaxSAT，能在其模型内合成最小/最优 barrier，且相对 oracle sound and complete。它直接支持草案“先从依赖推导必要边，再把 barrier placement 当调度选择”。

**WEFT** 为 named barrier 的 warp-specialized producer/consumer kernel给出 operational semantics，并对常见程序类证明验证算法 sound and complete；这比普通 CTA barrier 更接近后来的 `mbarrier`，但尚不含 TMA transaction byte count 与现代 proxy semantics。

### 对 T5 的具体建议

直接采用 Faial 风格的两层判定：

```text
MayParallel(event₁,event₂)
AddressConflict(event₁,event₂)
```

并显式构造：

```text
HB = program-order
   ∪ barrier-order(scope, instance)
   ∪ async-completion
   ∪ collective-order
```

只有 `MayParallel ∧ AddressConflict ∧ ¬HB` 才报 race。barrier checker 另证：

- scope 覆盖全部生产者/消费者；
- 所有参与线程到达同一动态 barrier instance；
- 控制条件 uniform；
- barrier 含正确 memory semantics，而不仅是 rendezvous。

普通 T5 子语言可采用 BSP/SDV；TMA、atomics、warp collective 必须作为显式扩展，不能隐含进普通 DRF 定理。

---

## 5. 异步拷贝与软件流水

### 代表性工作

- Rohan Yadav、Michael Garland、Alex Aiken、Michael Bauer，**Task-Based Tensor Computations on Modern GPUs**（Cypress），PLDI 2025。[论文](https://rohany.github.io/publications/pldi2025-cypress.pdf)
- Rupanshu Soi 等，**Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs**（Twill），OSDI 2026。[论文页](https://www.usenix.org/conference/osdi26/presentation/soi)
- Benjamin Spector 等，**ThunderKittens: Simple, Fast, and Adorable AI Kernels**，ICLR 2025。[论文](https://openreview.net/pdf?id=0fJfVOSUra)
- Lei Wang 等，**TileLang: A Composable Tiled Programming Model for AI Systems**，ICLR 2026。[论文](https://openreview.net/pdf/e851ad3eaca44bc18d67d753b03b3664ca2b9e9c.pdf)
- Rohan Yadav 等的 Cypress，以及 NVIDIA **CUTLASS Pipeline Types** 工程协议。[官方文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_pipelines.html)
- Michael Bauer 等，**WEFT**，PLDI 2015。[论文页](https://research.nvidia.com/publication/2015-06_verification-producer-consumer-synchronization-gpu-programs)

### 当前方案

**Cypress** 给程序员顺序 task 语义，mapping specification 决定 task 在哪些 warp/block 上执行、tensor 在何处物化。编译器从 task dependence 自动插入数据移动、event 和同步，再生成 TMA/tensor-core warp-specialized kernel。它非常接近草案“顺序参考 + 异步实现”，但论文给的是编程模型和编译算法，没有端到端 compiler-correctness 机械证明。

**Twill** 将软件流水和 warp specialization 联合建模为 constraint/modulo-scheduling 问题，并证明所求 schedule 在其机器模型和程序类中是吞吐最优的。这里的“proof optimal”是调度性能最优，不是异步 lowering 的功能正确性证明。

**CUTLASS/CuTe、ThunderKittens、TileLang、Triton** 普遍采用 stage ring 加 acquire/commit/wait/release 或 phase-bit `mbarrier` 协议。CUTLASS 已清晰区分：

- async thread store；
- `cp.async` producer；
- TMA transaction-byte completion；
- MMA/UMMA consumer；
- producer/consumer 各自的 stage counter。

但这些主要是模板约束、assert、类型和测试，并非异步语义的机械证明。

**WEFT** 证明了旧式 named-barrier producer/consumer 协议的一类正确性；它提示应把 barrier generation、arrival count、barrier reuse 作为状态机，而不是把 `mbarrier` 当普通 fence。

### 对 T6 的建议

草案的“完整 pipeline refinement”方向正确。建议以事件/token 语义表示：

```text
issue(copy, dst, bytes) -> token
commit(group, tokens)
complete(token)
wait(group, phase)
release(slot)
```

每个 ring slot 携带 typestate：

```text
Empty → Filling(g,phase) → Full(g,phase)
      → Reading → Empty(next-phase)
```

需要验证的不是 `store_async ≡ store`，而是：

1. 每次 `compute(b)` 前，对应 `fill(b)` 已完成且可见；
2. slot 重用前所有读已结束；
3. transaction byte count 与实际写入一致；
4. producer/consumer 对 phase 的推进一致；
5. `cp.async.wait_group` 的 per-thread 完成后，跨线程消费仍有 CTA barrier；
6. TMA proxy 与普通 shared access 间存在所需 proxy fence/order。

可以借 Cypress 的“顺序 task 参考”，借 CUTLASS 的 pipeline typestate，借 WEFT 的 barrier-reuse 验证。Triton/TileLang 的 pipeliner更适合作为差分测试对象，不应直接成为可信语义来源。

---

## 6. Tensor core 与 warp 级分布式值

### 代表性工作

- Guido Martínez 等，**Kuiper**，PLDI 2026。[论文](https://tyler-utah.github.io/files/kuiper.pdf)
- Bastian Hagedorn 等，**Graphene**，ASPLOS 2023。[论文页](https://mgarland.org/papers/2023/graphene/)
- Keren Zhou 等，**Linear Layouts**，2025 预印本。[论文](https://arxiv.org/abs/2505.23819)
- NVIDIA，**PTX ISA**，包括 `ldmatrix`、`mma`、TF32/BF16 fragment 表与 collective 前提。[官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- Manya Bansal 等，**Modular GPU Programming with Typed Perspectives**，PLDI 2026；论文页表明其核心是用类型表示 thread/warp collective 的观察粒度。[会议页](https://pldi26.sigplan.org/program/program-PLDI-2026)

### 核心方法与保证

**Graphene** 最早系统地把多维线程、warp tile 和 fragment 映射变成 IR 一等对象，适合表达数据到 lane/register 的复杂映射；但无证明。

**Linear Layouts** 直接把 `(reg,lane,warp) → tensor coordinate` 写成 \(\mathbb F_2\) 线性映射，可统一表达 MMA、shared swizzle、layout conversion 和 warp shuffle。rank/inverse 给出清晰的单射/满射判据。

**Kuiper** 引入 fragment type 和 fragment descriptor，并用 points-to relation 把物理 fragment 与数学矩阵关联；论文完成了含 tensor core 的两级 tiled GEMM 的端到端功能正确性证明。它还用 approximation class 表示浮点实现与理想实数矩阵之间的关系。

但 Kuiper 明确指出一个关键限制：它没有建模 `mma_sync` 的 collective aspect；若部分 lane 在 thread-dependent control flow 中缺席，硬件行为未定义，而其当前证明没有排除这一点。这恰好是草案 T7 仍需补的部分。

### 对 T7 的建议

`Frag(L,s,T)` 很合理，但需区分四种对象：

```text
LogicalFragLayout:
  (receiver lane, reg) → logical coordinate

LdMatrixAddressLayout:
  provider lane → shared address

LdMatrixTransfer:
  (provider lane, loaded element) → (receiver lane, reg)

PanelLayout:
  logical coordinate → shared offset
```

合法性定理应是上述关系的 composition 与 `LogicalFragLayout` 一致，而不是把 provider address map 误当成 fragment layout。

此外加入 capability：

```text
Collective<warp, active-mask, dynamic-instance>
```

`ldmatrix`/`mma.sync` 只有持有相同 collective capability 的全部 active lane 才能调用。PLDI 2026 的 typed perspectives 是这一设计的直接参考。

数值上必须区分：

- F16/BF16/TF32 输入转换；
- 乘法精度；
- accumulation 精度；
- 中间舍入或未指定顺序；
- flush-to-zero、NaN 等 target 行为。

因此 T7 的定理应是 `≈mode`，而非 `≡value`。TF32 转换应成为显式 IR op；不能依赖“传入 F32 bit pattern 后硬件自然处理”。

---

## 7. 归约的分块与重组

### 代表性工作

- Patricia Suriana、Andrew Adams，**Parallel Associative Reductions in Halide**，CGO 2017。[论文](https://andrew.adams.pub/par.pdf)
- Halide，**Factoring an Associative Reduction Using rfactor**。[官方说明](https://halide-lang.org/docs/tutorial/lesson_18_parallel_associative_reductions.html)
- Guido Martínez 等，**Kuiper**，PLDI 2026。[论文](https://tyler-utah.github.io/files/kuiper.pdf)
- Phitchaya Mangpo Phothilimthana 等，**Swizzle Inventor**，ASPLOS 2019。[论文](https://mangpo.net/papers/swizzle-inventor-asplos19.pdf)

### 核心方法

Halide 的 `rfactor` 将归约域切成 partial reductions，再生成 merge reduction。其实现为常见标量类型预计算大量基本归约模式，并用 Z3 检查/推导 associativity、identity 和必要时的 commutativity：

- 只改变括号结构：需要结合律；
- 改变元素次序：还需要交换律。

这类自动判定只对受限 expression grammar 可行；任意用户函数的结合律不可一般判定。

**Swizzle Inventor** 同样要求归约算子满足结合律和交换律，作为综合合法性的前提，而不是试图从任意 CUDA 代码自动推导。

**Kuiper** 对浮点归约不声称 bitwise 等价，而是使用 approximation relation，把实际 reduction 结果与理想实数 fold 联系起来。它是草案 `≈mode` 的直接先例。

### 对 T1/T5 的建议

把归约契约分成三档：

```text
OrderedChunk:
  保持完整字典序；bitwise/值精确

AssociativeFold:
  允许改括号，不允许任意置换

CommutativeMonoidFold:
  允许并行 partition、树形 merge、split-K
```

浮点 `+` 在 IEEE 精确语义下不满足结合律，因此只能：

- 保留 `OrderedChunk`；
- 要求显式 `fast/reassociate` 许可；
- 或给出 error-bound/approximation witness。

R8 和 split-K 还分别需要：

```text
PartitionCover(reduction-domain, partials)
MergeCover(partials, tree)
```

两者都不能由“算子结合”替代。CTA 间 split-K 还涉及 T0：第二 kernel、global atomic 或 cluster/global synchronization。

---

## 8. 跨 kernel 融合、常驻、同步与资源可行性

### 代表性工作

- Yining Shi 等，**WELDER: Scheduling Deep Learning Memory Access via Tile-graph**，OSDI 2023。[论文](https://www.usenix.org/system/files/osdi23-shi.pdf)
- Zhen Zheng 等，**FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads**，2020 预印本/后续系统工作。[论文](https://arxiv.org/abs/2009.10924)
- Rohan Yadav 等，**Composing Distributed Computations Through Task and Kernel Fusion**（Diffuse），ASPLOS 2025。[论文页](https://research.nvidia.com/publication/2025-03_composing-distributed-computations-through-task-and-kernel-fusion)
- Ghassan Shobaki、Austin Kerbow、Stanislav Mekhanoshin，**Optimizing Occupancy and ILP on the GPU Using a Combinatorial Approach**，CGO 2020。[论文信息](https://scholars.csus.edu/esploro/outputs/conferenceProceeding/Optimizing-occupancy-and-ILP-on-the/99257847147601671)
- Guido Martínez 等，**Kuiper**，PLDI 2026。[论文页](https://pldi26.sigplan.org/details/pldi-2026-papers/37/Kuiper-Correct-and-Efficient-GPU-Programming-with-Dependent-Types-and-Separation-Log)

### 当前方案

**WELDER** 的 tile-graph 同时传播 producer/consumer tile，估算 register/shared/global-memory traffic，在多层存储间权衡 intra-op 与 inter-op reuse，并决定融合边界。它非常接近 T2+T3+T0 的联合搜索，但合法性与成本主要由编译器分析和经验 cost model 保证，没有形式化保义证明。

**FusionStitching** 把 elementwise、reduction 和小 GEMM 融入大 kernel，以 ILP/启发式搜索融合计划，并显式考虑 shared/register 资源约束。其约束是工程 feasibility，不是证明系统。

**Diffuse** 在 task runtime 层表示分布式 task、跨库数据流和 kernel fusion，说明 T0 最自然的对象不是单个 `Tileable`，而是带 storage/event token 的 kernel DAG。

**Kuiper** 已形式化 CPU/GPU located resources、异步 kernel launch 和 kernel lifetime，能为 host/device ownership 与同步提供语义基础；但它不是融合/资源搜索系统。

**Shobaki 等** 联合优化 instruction-level parallelism 与 occupancy，把寄存器压力导致的 occupancy 下降纳入组合优化。这说明 T8 必须区分：

- 硬合法性：block size、shared bytes、架构限制；
- 性能可行性：register count、occupancy、spill、ILP；
- 成本估计：吞吐/latency。

后两类通常只有 lowering 后甚至 ptxas 后才精确可知。

### 对 T0/T8 的建议

为 T0 引入显式 kernel DAG：

```text
KernelNode {
  grid, block, effects,
  input/output allocations,
  stream/event dependencies
}
```

融合合法性至少需要：

- producer/consumer launch-domain 映射兼容；
- 跨 block 依赖不能被错误降成 CTA barrier；
- 未融合时由 event/stream 建立 kernel 间 happens-before；
- resident buffer 的 owner/lifetime 跨 kernel 明确；
- split-K 的 global communication 显式存在。

T8 可先实现 target-specific hard constraints：

```text
threads_per_block ≤ limit
static_shared + dynamic_shared ≤ limit
pipeline_buffers × panel_size ≤ shared limit
fragment/register-shape supported
```

寄存器和 occupancy 宜作为估算值与后端反馈，而不是语义证明前提。选择器可以启发式，但最终 schedule 必须携带资源 certificate 或编译后实测属性。

---

## 9. 已验证调度与变换

### 代表性工作

- Amanda Liu 等，**Verified Tensor-Program Optimization via High-Level Scheduling Rewrites**，POPL 2022。[论文](https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf)
- Amanda Liu、Gilbert Bernstein、Adam Chlipala、Jonathan Ragan-Kelley，**A Verified Compiler for a Functional Tensor Language**，PLDI 2024。[论文页](https://adam.chlipala.net/papers/AtlPLDI24/)
- Alex Reinking、Gilbert Bernstein、Jonathan Ragan-Kelley，**Formal Semantics for the Halide Language**，技术报告 2020、扩展预印本 2022。[论文](https://arxiv.org/abs/2210.15740)
- Yuka Ikarashi 等，**Exocompilation**，PLDI 2022。[论文页](https://pldb.kirancodes.me/p/conf/pldi/IkarashiBRGR22)
- Nuno Lopes 等，**Alive2: Bounded Translation Validation for LLVM**，PLDI 2021。[论文](https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf)
- Max Willsey 等，**egg: Fast and Extensible Equality Saturation**，POPL 2021。[论文页](https://popl21.sigplan.org/details/POPL-2021-research-papers/23/egg-Fast-and-Extensible-Equality-Saturation)
- Yichen Yang 等，**Equality Saturation for Tensor Graph Superoptimization**（TENSAT），MLSys 2021。[论文](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf)

### 核心方法与保证

**ATL/verified scheduling** 把 schedule 写成一串 Coq 已证明的高层等式重写。每一步产生新的程序和机械检查的等价证明。PLDI 2024 工作进一步验证从函数式 tensor language 到 imperative loop nests 的 lowering，并用类型系统保证 storage-index remapping 良构；验证过程中发现并修复了原 truncation reshape 算法的 soundness bug。

这是 MimIR 论文“调度是 IR 中的普通函数”最接近的现有形式化路线。差别是 ATL 的证明在 Coq 内完成，而 MimIR 当前依赖部分求值器、rewriter 和显式 checks 的可信性。

**Halide formal semantics** 对核心 schedule directives 给出算法无关语义和正确性论证，并形式化 bounds inference 问题；它证明的是一个核心模型，不代表生产 Halide 的全部 GPU lowering 已机械验证。

**Exo** 走轻量路线：受信任 primitive rewrite 加 effect-analysis side condition，适合实际调度库，但不像 ATL 那样每个 schedule 生成 proof term。

**Alive2** 做 per-compilation translation validation：检查 target 是否 refinement source。它支持 LLVM UB，但 loop 通过有界展开，因此是 bounded validation。其思想适合 MimIR 对 C++ lowering 的独立检查，尤其是 T1–T4 的小 tile 和 T7 固定 fragment。

**egg/TENSAT** 解决 rewrite phase ordering，但 equality saturation 本身只保证“在已声明等式生成的 congruence closure 中”。若 rewrite rule 错、浮点模式不匹配或 side condition 未编码，e-graph 不会自动带来正确性。因此要么规则先证明，要么 extraction 后做 translation validation。

### 对 MimIR 的建议

可信计算基可分三层：

1. **定义式层**：T1–T4 的 schedule combinator 写成 Mim 函数，经部分求值生成实现。
2. **受信重写层**：每个 primitive 有 reference theorem 或独立 validator；不要直接信任大型 C++ pass。
3. **翻译验证层**：对具体变换前后检查：
   - domain cover；
   - read/write footprint；
   - layout injectivity；
   - owner partition；
   - 固定 fragment 的 gather/scatter 等价。

T5–T7 不应仅用 Alive 风格的顺序函数等价；需把 event、memory model、collective 和 numerical mode 纳入 refinement relation。

Equality saturation可用于搜索 T1/T2 纯张量重写，但 T3–T7 应由 typed side conditions 限制，不能把 barrier、async copy 或 TF32 当普通等式。

---

# 难点 × 推荐借鉴的方法

| 难点 | 推荐借鉴 | 主要适用分面 | 不应误认为 |
|---|---|---|---|
| 1. 层次调度与作用域 | Descend execution resource；TensorIR block signature；Exo trusted primitives | T1、T2、T4、T5 | MLIR/TensorIR verifier 已证明 GPU 保义 |
| 2. 布局判定 | semi-affine→isl；XOR/fragment→Triton \(\mathbb F_2\) layout；CuTe 作组合 API | T3、T7 | 任意 Mim 函数的 injectivity 可判定 |
| 3. 分布与所有权 | Hidet worker→task map；Kuiper located resource/separation logic | T1、T4、T5 | `ExactCover` 能替代 owner/partition proof |
| 4. 同步 | Faial access protocol；GPUVerify uniform barrier；PTX scoped memory model；AutoSync | T5 | barrier 只是控制会合 |
| 5. 异步流水 | Cypress 顺序 task 参考；CUTLASS typestate；WEFT barrier reuse；Twill 作性能搜索 | T5、T6 | `store_async` 与 `store` 操作级等价 |
| 6. Tensor core | `Frag(L,s,T)`；Linear Layouts；Kuiper fragment points-to；PTX tables | T3、T4、T5、T7 | well-typed fragment 自动保证 warp convergence |
| 7. 归约 | Halide `rfactor` 律检查；Ordered/Associative/Commutative 三档契约；Kuiper approximation | T1、T5 | 浮点加法默认结合 |
| 8. 跨 kernel/资源 | kernel DAG+event tokens；Welder tile-graph；硬资源 certificate+后端反馈 | T0、T2、T3、T8 | occupancy 是语义合法性的充分条件 |
| 9. 已验证变换 | ATL proof-producing rewrite；Exo effect analysis；Alive2式翻译验证；受证明 e-graph rules | T1–T7 | equality saturation 本身证明规则正确 |

---

# 按草案 §5 路线的建议阅读顺序

## 第 1 步：修复 R8 树形合并覆盖

1. Suriana、Adams，**Parallel Associative Reductions in Halide**，CGO 2017。[论文](https://andrew.adams.pub/par.pdf)
2. Hidet 的 task-mapping composition，特别是 spatial/repeat 映射。[论文](https://arxiv.org/abs/2210.09603)
3. Kuiper 的 reduction 与 approximation relation。[论文](https://tyler-utah.github.io/files/kuiper.pdf)

重点：先把 `ReductionPartition` 与 merge-tree coverage 写成可枚举/SMT 检查，不要先做一般代数证明。

## 第 2 步：归约契约与 root descriptor

1. Halide `rfactor` 的 associativity/commutativity 判定。[教程](https://halide-lang.org/docs/tutorial/lesson_18_parallel_associative_reductions.html)
2. ATL 高层 verified scheduling rewrites。[论文](https://people.csail.mit.edu/lamanda/assets/documents/LiuPOPL2022.pdf)
3. Kuiper 的浮点近似契约。[论文](https://tyler-utah.github.io/files/kuiper.pdf)

重点：`OrderedChunk` 先做到精确；`ParallelFold` 后续再接 monoid witness 和 numerical mode。

## 第 3 步：作用域与 collective 保持

1. Descend execution resources。[论文](https://steuwer.info/files/publications/2024/PLDI-2024.pdf)
2. TensorIR block/region abstraction。[论文](https://arxiv.org/abs/2207.04296)
3. GPUVerify 的 barrier divergence 语义。[论文页](https://doi.org/10.1145/2743017)

重点：binding scope 成为不可静默擦除的 IR 项；warp collective 另有 convergence capability。

## 第 4 步：物理布局成为受控项

1. Triton **Linear Layouts**。[论文](https://arxiv.org/abs/2505.23819)
2. isl user manual 的 map/injective/bijective API。[手册](https://libisl.sourceforge.io/user.html)
3. Cecka，**CuTe Layout Representation and Algebra**。[论文](https://arxiv.org/abs/2603.02298)

重点：先支持草案现有 semi-affine rotation；随后再单独加入 \(\mathbb F_2\) XOR/fragment 子代数。

## 第 5 步：T7 fragment 参考定义与差分测试

1. Kuiper 的 fragment、tensor-core GEMM 与已声明限制。[论文](https://tyler-utah.github.io/files/kuiper.pdf)
2. Graphene 的 thread/data tile IR。[论文页](https://mgarland.org/papers/2023/graphene/)
3. NVIDIA PTX 的 `ldmatrix`/`mma`/TF32 规范。[官方文档](https://docs.nvidia.com/cuda/parallel-thread-execution/)

重点：分别实现 `Addr_LD`、`Transfer_LD`、`L_A/L_B/L_C`；TF32 conversion 显式化。

## 第 6 步：T5 受限 BSP 验证器

1. Faial access-memory protocols 与 Coq 证明。[论文](https://cogumbreiro.github.io/assets/faial-cav21.pdf)
2. GPUVerify 的 SDV 语义。[论文页](https://doi.org/10.1145/2743017)
3. Lustig 等的 PTX memory-model formalization。[论文](https://d1qx31qr3h6wln.cloudfront.net/publications/ASPLOS_2019_PTXMemoryModel.pdf)

重点：先限于 affine access、结构化 control、CTA shared、普通 barrier、无原子/async/collective。

## 第 7 步：T6 pipeline 变换子

1. Cypress 的顺序 task 语义与自动同步 lowering。[论文](https://rohany.github.io/publications/pldi2025-cypress.pdf)
2. WEFT 的 producer-consumer barrier operational semantics。[论文](https://legion.stanford.edu/pdfs/weft.pdf)
3. Twill 的 SWP+warp-specialization 约束模型。[论文](https://arxiv.org/abs/2512.18134)

重点：先证明固定 depth 的 ring-buffer refinement；性能最优搜索与正确性验证保持分离。

---

最值得 MimIR 立即采用的组合不是单一系统，而是：

```text
Exo/ATL 的“调度即受控重写”
+ Descend/Kuiper 的 scope 与 ownership
+ isl/Linear Layouts 的可判定布局
+ Faial 的 access protocol
+ Cypress 的顺序异步参考语义
+ Alive2 风格的具体 lowering 翻译验证
```

这样 T1–T4 可以较快形成可执行、可判定的合法性层；T5 先限制在普通 CTA BSP；T6、T7 则通过显式 event/collective/numerical contracts 渐进扩展，而不把尚未解决的 GPU 弱内存、异步代理和 tensor-core collective 问题隐藏在 C++ lowering 约定里。