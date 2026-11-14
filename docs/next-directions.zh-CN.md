# 重构之外的大改进方向（2026-11-13）

## 0. 第一性原理

一个张量编译器的价值，大致是三项的乘积再除以人力：

> 达到的性能 × 覆盖的工作负载 × 结果可信度 ÷ 每个 kernel 需要的人工

GPU 上的性能由几条物理约束决定，任何调度只是在它们之间取舍：

1. **算强与容量。** 张量核的峰值只有在算术强度足够时才吃得满；算术强度由 tile 尺寸决定，tile 尺寸又被 SMEM、寄存器（Blackwell 起还有 TMEM）的容量封顶。
2. **延迟只能靠重叠隐藏。** HBM→SMEM→寄存器/TMEM 的每一跳都有延迟，靠异步拷贝、流水和 warp 特化去重叠；重叠的正确性完全落在同步协议上。
3. **布局决定带宽利用率。** 合并访问、bank 冲突、swizzle、向量宽度，本质都是“线程/寄存器坐标 → 数据坐标”这个映射的性质。
4. **非对称扩展。** 张量核吞吐每代涨得比 SFU、FMA、SMEM 带宽快（FA4 的核心论点），所以 softmax 的 `exp`、rescale 这些非 matmul 运算会成为新的瓶颈，需要算法层面的改写配合调度。

可信度方面也有一条“第一性”事实：编译器里最大的可信基是 lowering 本身。
`lower_to_mem.cpp` 4782 行、`tile_fuse.cpp` 1869 行，类型系统和 Mim 规则只约束“请求是否合法”，不约束“lowering 是否忠实实现了请求”。

## 1. 现状（本仓库实测）

| 维度 | 现状 | 证据 |
|---|---|---|
| 性能：访存密集 | 已快于 torch.compile | softmax 4096² 0.0385 vs 0.043–0.055 ms；cross_entropy 1.168 vs 2.037 ms |
| 性能：GEMM | cuBLAS 的 85–91% | 4096³ 0.2247 vs 0.204 ms；persistent 8192 达 85% |
| 性能：attention（编译） | cuDNN 的 0.73–0.97× | `attn_wg` 128×4096 1.747 vs 1.283 ms；手写 FA3 结构 0.92× |
| 覆盖 | 只有前向；固定形状；H200 一种目标；每个算子一份带 `@占位符@` 的模板 | `bench/tritonbench/*.mim` |
| 调度 | 全部手选，结果对参数极敏感 | softmax item 32：0.297 ms，item 8：0.0405 ms（7.3×） |
| 可信度 | 类型 + Mim 规则（PE）约束请求；同步只有 `check_sync`；lowering 无验证 | `docs/gpu-tiling-formalization-gaps.zh-CN.md` G7、G9、G12 |

## 2. 方向

每个方向按“原理 → 我们的证据 → 文献 → 做法 → 形式化含义 → 代价”展开。

### D1 布局代数：把布局变成可判定的线性映射

- **原理**：约束 3。GPU 上几乎所有布局（行主/列主、swizzle、MMA fragment、TMA box、warp shuffle 的 lane 置换）都是比特层面的线性映射，即 F₂ 上的矩阵。
- **证据**：`panel_layout.cpp` 477 行按情况处理 swizzle、连续性和银行冲突；形式化空白 G1（box 合法与覆盖）、G3（读映射与 stage）、G10（tensor core 布局）都卡在“布局单射、可逆、复合”无法判定；`Fill::contiguous` 只抽查 64 个位置，没有论证（gaps §6）。
- **文献**：Linear Layouts（ASPLOS'26）把 Triton 的布局统一成 F₂ 矩阵，修掉了占 Triton 12% 的布局类 bug，平均 1.07×、最高 1.40× 加速；CuTe 的 layout algebra 是同一问题的整数版本。
- **做法**：在 Mim 里定义 `Layout = F₂ 矩阵`（`«m; «n; Bool»»`），复合、求逆、单射、秩都是高斯消元，可以直接写成 Mim 函数由 PE 判定；`panel_layout` 的决策改成“在线性布局空间里选”，正确性条件（单射、覆盖、向量化宽度 = 最低若干位的恒等块）变成可判定谓词。
- **形式化含义**：G1、G3、G10 的一部分从“未检查”变为“由判定过程保证”，而且不需要用户证明，正好落在“用户无法提供证明”的约束里。
- **代价**：中。非 2 的幂尺寸（尾块、padding）不在 F₂ 里，需要“线性 + 掩码”之类的扩展；Linear Layouts 怎么处理这一点还没核实（见 §4）。

### D2 异步协议类型：同步正确性按构造成立

- **原理**：约束 2。重叠越激进（TMA 环、warp 特化、ping-pong、persistent、Blackwell 的 TMEM 流水），同步协议越复杂，而这类义务按现有分析“不能由断言或 nominal 类型承担”（gaps §1 第 3 类）。
- **证据**：本轮 persistent、ping-pong、ring 的约束只能由 lower_to_mem 晚检；尾块写回屏蔽和 TMA 越界补零“是 lowering 的约定，相位类型里没有建模，`check_sync` 也不检查”。
- **文献**：Tawa（CGO'26）用 aref（异步引用：带 full/empty 协议的单槽通道）表达 warp 间通信，从未修改的 Triton 程序自动做 warp 特化，H100 上最高达 cuBLAS 的 1.1×；Descend（PLDI'24）用所有权和 view 在类型上排除数据竞争；Faial、GPUVerify 用成对线程推理检查竞争和 barrier 发散。
- **做法**：把 ring/mbarrier 抽象成 Mim 的 `Aref (T, slots)` 类型和**结构化组合子**（`produce`、`consume`、`release` 只能以固定形状出现，不暴露裸 token）。这样不需要线性类型就能保证“每次 consume 前有 produce、每个槽释放后才复用”，与“线性类型缓一缓、用 check 替代”的决定一致。lower_to_mem 生成 mbarrier 时只翻译组合子，不再自己推协议。
- **形式化含义**：G7、G9 从“未检查”变成“按构造成立 + 组合子的语义论证”（一次性的纸面证明）；也是做 Blackwell（TMEM 累加器在 MMA 与 softmax 角色之间传递）的前提。
- **代价**：高。牵涉 lower_to_mem 里流水、特化、persistent 的大段代码；但它也正是“C++ 里带状态的决策搬不走”（gaps §5）的那部分状态的 IR 表示。

### D3 逐 kernel 的翻译验证：给 lowering 一个独立的裁判

- **原理**：最大的可信基是 lowering。与其证明 4782 行 C++ 正确，不如每次输出后检查这一次的结果与规格等价。
- **证据**：本会话修过的回归（α 等价拷贝合并、fold 的 mem 链、尾块写回、配对写）都是 lowering 改错了语义，只能靠 exec 测试在少数形状上发现；Mim 规则与 lower_to_mem 是两份规则，一致性只靠 lit（`docs/tile-check-rules.md`）。
- **文献**：Volta（OOPSLA'26）对 PTX 做符号执行，对“structured-CTA”类 kernel（静态形状、静态控制流）的等价检查可靠且完备；Mirage（OSDI'25）在有限域上做随机测试，对其“Lax”片段有很强的概率保证；Tatva（NeurIPS'26）把 PyTorch/Triton 两侧降成 DAG 交给 Z3。
- **做法**：分两级。
  1. 低成本：`tile_req` 的规格就是 `map_reduce_post`，已有 CPU 参考语义；对每个编译出的 kernel 在小形状、有限域（模素数整数代替浮点）上随机比对，作为编译选项和 CI 步骤。
  2. 高成本：对我们的 kernel 类（静态形状、静态 tile 循环）在 Mim IR 层（lowering 之后、nvptx 之前）做符号执行等价检查，避开 PTX 解析。
- **形式化含义**：把 G12（“LowerToMem 承载正确性的旁表无表示”）从“信任”变成“逐次验证”；也顺带解决 Mim 规则与 C++ 两份规则可能不一致的问题。
- **代价**：第 1 级低、收益立刻可见；第 2 级是研究级工作，但我们的 kernel 恰好是 Volta 可判定的那一类。

### D4 规则即约束：从“检查调度”到“合成调度”

- **原理**：调度空间由 D1 的布局、`TileLevels` 的结构、D2 的协议共同定义；合法性已经写成了 Mim 谓词（`mod check`）。同一组谓词既能拒绝非法调度，也能作为求解器的约束。
- **证据**：调度全部手选、对参数极敏感（7.3×）；tritonbench 模板为每个算子手写层级链；GEMM 到 cuBLAS 的余差来自 tile、级数、特化组合的选择。
- **文献**：Twill（OSDI'26）把软件流水与 warp 特化建成一个联合约束问题，用现成求解器求最优，重新发现了 Hopper/Blackwell 上手写 FA 的调度；TileLang（ICLR'26）加入 tile 推断与推荐；Ansor/MetaSchedule 用学习的代价模型搜索。
- **做法**：从层级类型和 `check` 规则自动导出约束（容量、整除、协议），目标函数用解析代价（算强 × 带宽 × 重叠度），先做可枚举的小空间（tile 尺寸、级数、特化开关），再接 SMT/ILP。
- **形式化含义**：合法性由构造保证（只在规则允许的空间里搜），把“正确”和“快”分离成约束和目标两部分，这正是把形式化成果变现为性能的路径。
- **代价**：中到高。之前定过“手选和代价模型暂时不用管”，所以排在 D1–D3 之后；但 D1–D3 做完后，D4 的大部分输入已经齐了。

### D5 硬件层级参数化：Blackwell 与“目标描述”

- **原理**：约束 1 和 4 在每代硬件上的具体形态不同。Blackwell 多出 TMEM 一级存储、2-CTA MMA 和块缩放的 FP8/FP4，存储层级本身变了。
- **证据**：`TileLevels`/`GroupLevels`/`ItemLevels`/`WarpLevels`/`WarpgroupLevels` 是写死的五种作用域，Hopper 的 `wgmma` 规则写在构造器里；可用硬件只有 H200（sm_90a），以及 fjolnir 上的 RTX PRO 6000（sm_120，没有 tcgen05/TMEM）。
- **文献**：FA4（2026）在 B200 上靠 TMEM 让多个 MMA 在 softmax 期间保持在途，比 cuDNN 快 1.3×，并用多项式模拟 `exp` 缓解 SFU 瓶颈；cuTile/Tile IR 是 NVIDIA 的官方 tile 级 IR，可导出 TileIR 字节码，有独立的内存模型文档。
- **做法**：两条路线，取舍要明确。
  - **自建**：把作用域层级和每级的 MMA/拷贝原语抽象成“目标描述”类型，层级链按目标实例化；Hopper 和 Blackwell 是两个实例。控制力最强，工作量最大。
  - **接 Tile IR**：把我们的 tile 级（work-group 内部）降到 Tile IR，warp 特化、TMEM、tcgen05 交给 NVIDIA；我们保留多级调度、融合、形式化的部分。工作量小、跟进新硬件快，但失去 warpgroup 级以下的控制，形式化的边界也止于 Tile IR 的语义。
- **形式化含义**：目标描述让“这条规则依赖哪种硬件事实”显式化（`tile-check-rules.md` 里的 e 类）；接 Tile IR 则需要信任它的内存模型。
- **代价**：高。没有 Blackwell 数据中心卡之前，先做抽象、用 sm_120 验证可移植性。

### D6 覆盖面：反向、动态形状、低精度、端到端

- **原理**：价值函数里的“覆盖”项。目前每个算子靠一份手写模板。
- **证据**：只有前向；符号尺寸会让请求回退到不切分（`tile_symbolic_strict.mim`）；`mimir-frontend` 能从 PyTorch FX 导入，但还没适配 GPU 改动。
- **做法**：
  - **反向**：attention 反向是标准难题（dQ 跨块累加、对不同轴的归约），也是检验调度表达力的好例子。autodiff 插件已有，需要让 `tile_req` 在反向图里也能用。
  - **动态形状**：cuTile 的 `ParameterConstraint` 给了现成做法：在 kernel 签名上声明假设（整除、对齐、范围），编译期按假设特化，运行期按假设守卫。这正是 gaps §1 里“运行时 assert”的落地点。
  - **低精度**：FP8 及块缩放格式，它们改变的是 D1 的布局（缩放因子的布局）和 MMA 种类（`Contraction` 索引自然扩展）。
  - **端到端**：用前端跑一个完整的 transformer block，暴露跨算子的融合与常驻问题（gaps G6、文献 §8）。
- **形式化含义**：动态形状把“PE 判定”推到“编译期假设 + 运行期守卫”，需要 gaps §6 列为未知的 assert 语义（效应顺序、DCE、失败协议）先定下来。
- **代价**：分项可做，反向和动态形状的研究价值最高。

### D7 算法与调度联合改写：应对非对称扩展

- **原理**：约束 4。张量核越快，`exp`、rescale、归一化占比越大；单靠调度改不了运算量，要改算法（online softmax、条件 rescale、`exp` 的多项式模拟、`exp2` 与 log2e 折叠）。
- **证据**：attention 的编译路径和手写版本的差距主要在 softmax 段；本会话的若干加速来自改写规格（先求行最大值再比较、以 2 为底），而不是调度。
- **文献**：Mirage 在 kernel、block、thread 三级上联合搜索代数改写和调度，在广泛使用的 DNN 上最高 3.3×；FA4 的条件 rescale 与软件 `exp`。
- **做法**：把这类改写写成 Mim 规则（仓库里已有 `rules/` 的图模式匹配基础设施），带上“改写前后等价”的义务；等价由 D3 的验证器承担。
- **形式化含义**：改写的正确性落在 D3；浮点改写还需要一个明确的“允许的数值误差”规格，这是目前完全没有的。
- **代价**：中。

### D8 元语言的可靠性：PE 是“规则写在 Mim 里”的地基

- **原理**：D1、D2、D4、D7 都假设能放心地把判定逻辑写成 Mim 函数、由 PE 求值。
- **证据**：迁移 `check_levels` 时遇到了这些问题：带隐式参数的 lam 递归会把调用归约成实参元组；元组选择两边都求值；必须手写 `is_closed` filter；normalizer 分不清规则变量和用户符号量；报错丢失数值（`docs/tile-check-rules.md`）。
- **做法**：修隐式参数递归的 bug；提供惰性的条件表达式；让规则能返回带数值的诊断；给 PE 的代价（本次测得 +10–17 ms）设缓存。
- **代价**：低到中，收益是后面所有方向的开发效率。

## 3. 优先级与依赖

```
D8 (PE 可靠) ──┬──> D1 (F₂ 布局) ──┬──> D4 (调度合成)
               │                   │
               └──> D2 (异步协议) ──┼──> D5 (Blackwell / 目标描述)
                                   │
D3 (翻译验证, 第 1 级立即可做) ─────┴──> D7 (算法改写)
D6 (覆盖面) 与上面并行；动态形状依赖 assert 语义
```

建议顺序：

1. **D3 第 1 级 + D8**：成本最低、立刻提高每次改动的可信度，本会话的多次回归都会被它提前发现。
2. **D1**：形式化收益（G1、G3、G10）和性能收益（布局转换、向量化）同时到手，而且完全可以用现有的“Mim 规则 + PE”机制实现。
3. **D2**：同步是剩下最大的一块未形式化义务，也是 Blackwell 的前提。
4. **D4**：此时约束和代价的输入已齐，把形式化成果变成自动化和性能。
5. **D5、D6、D7**：按硬件可得性和论文需要选择；反向 attention 和动态形状适合作为展示整套机制的应用。

形式化贯穿其中：D1 给出布局的判定过程，D2 给出同步的按构造正确，D3 给出 lowering 的逐次验证，D4 让合法性由构造保证。四者合起来，覆盖了 gaps 文档里“未检查”的大部分条目。

## 4. 不确定的地方

- Linear Layouts 对非 2 的幂尺寸的处理细节，以及它能覆盖多少我们现有的 panel 布局，需要读论文正文后对照 `panel_layout.cpp` 逐项核对。
- Tile IR 能否作为外部编译器的目标（字节码格式是否公开稳定、能否表达我们需要的融合）没有验证，只确认了 cuTile 能导出 TileIR 字节码并有独立的内存模型文档。
- Volta 的“structured-CTA”类与我们 kernel 的重合程度，是根据摘要判断的，没有实测。
- D4 的求解规模（层级链 × tile 尺寸 × 级数）是否可解，Twill 的结果针对单个 kernel 的流水与特化，未必能直接推广到多级 tiling。

## 参考

- Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References — https://arxiv.org/html/2510.14719v2
- Twill: Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs — https://arxiv.org/html/2512.18134v1
- Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F₂ — https://arxiv.org/abs/2505.23819
- Mirage: A Multi-Level Superoptimizer for Tensor Programs — https://arxiv.org/abs/2405.05751v3
- FlashAttention-4 — https://tridao.me/blog/2026/flash4/
- Descend: A Safe GPU Systems Programming Language — https://arxiv.org/pdf/2305.03448
- Volta: Equivalence Checking of ML GPU Kernels — https://arxiv.org/abs/2511.12638v2
- Tatva — https://neurips.cc/virtual/2026/178980
- Faial (CAV'21) — https://cogumbreiro.github.io/assets/faial-cav21.pdf
- TileLang — https://arxiv.org/pdf/2504.17577
- cuTile Python（本地 `/home/chzh00001/workspaces/cutile-python`）

## 5. 进展（2026-11-13）

- **D8**：修了带隐式参数的 lam 递归时调用停在空壳 lam 上的 bug（`World::app`，`lit/pe_implicit_rec.mim`）；tile 规则去掉 `go` 包装，报错带数值。
- **D3 第 1 级**：`bench/validate/fuzz.py` 把随机形状与调度和去掉 `tile_req` 的规格比较（整数输入，逐位一致）。
  已发现并修复：未流水的 wgmma 面板缺 `fence.proxy.async`（两处静默算错）、单块 specialized 归约的非法指令（改为规则拒绝）、
  单轮树形合并的编译器内部错误；记录了两个未修的既有问题。详见 `bench/validate/README.zh-CN.md`。
- **D1 第一步**：`layout_expr` 的同一个 `interpret` 增加 F₂ 索引代数（每行一个 `I64` 掩码），`injective`、`contiguous` 由高斯消元判定；
  所有选定的面板布局在 2 的幂形状上验证单射，fill 的向量连续性在可判定时取代 64 位置抽查（`lit/tensor/layout_f2.mim`）。
  编译时间无可测开销。
- **D2 第一步**：TMA ring 的 mbarrier 协议写进 `pipeline_plan`（`full_parity`、`empty_parity`），lower_to_mem 改为从这里取 parity，只剩一份公式。
  `ring_safe fills depth lag offset` 是协议的可执行模型：在所有（已填 p，已消费 c）状态上，由 PE 穷举检查不覆盖未消费的槽、不读未填的槽、不死锁；
  正确的 parity 通过，偏一位的被抓出（`lit/tensor/ring_protocol.mim`）。模型还给出比注释更精确的结论：延迟释放的步数必须小于槽数。
  lower_to_mem 在块数为字面量时对所用配置调用 `ring_safe`。
  差分测试把 “ping-pong 只有一个消费者”从 lowering 的晚期拒绝变成规则：Group 层的块按实际长度裁剪后再数 warpgroup（`lit/error/tile_pingpong_clipped.mim`）。
- **D4 第一步**：`bench/tune/tune.py` 枚举模板的调度参数，以 elaboration（规则）剪枝、实测排序。softmax 4096² 找到 0.0314 ms（手选 0.0385），addmm 确认手选最优。
  搜索暴露了规则还缺的一类：资源可行性。超过 1024 个 work-item 的 work-group 由 outline 分轮次执行，正确但慢，大 grid 下崩溃（已记录、未修），暂按代价剪掉。
- **D7 第一步**：nvptx 后端对 `afn` 模式的 F32 超越函数改调 `__nv_fast_*`，softmax 4096² 0.0314 → 0.0259 ms，cross_entropy 1.168 → 0.984 ms（`lit/nvptx/fast_math.mim`）。
- **D5 第一步**：规则读取 `-X nvptx:sm` 指定的目标（`check.target`），warpgroup 层（wgmma）要求 sm_90a，TMA 与 cluster 要求 sm_90 以上，warp mma 要求 sm_80 以上；
  为别的目标写的调度在调用处被拒绝（`lit/error/target_wgmma.mim`、`target_mma.mim`），未指定目标时由后端探测、规则放过。
- **D6 第一步（调查，未提交代码）**：让 work-group 层接受运行时长度（M 为参数的 GEMM、softmax、逐元素算子），逐层打通后得到阻塞链：
  1. lower_to_mem 的 bound work-group 层：运行时块数、混合进制分解下标、cap 尾块，原型可行；
  2. 内层 bound 层（item、warpgroup）看到的是被截断的运行时尾块长度，需要按常量上界（外层步长）切分并屏蔽越界；TMA 填充把尾块按整块算，所以 TMA 路径越过了这一步；
  3. `gpu.outline` 需要运行时 grid（kernel 的 group 下标用 `Idx 0`、launch 截断为 i32，原型可行），
     并且要在 kernel 内重建捕获了运行时形状的函数与类型：形状依赖 `m` 的 tile 体不再是闭项，变成带自由变量的函数值；
  4. 重建时 RWPhase 对依赖类型（`«m; *»`）的 coerce 失败。
  “cannot coerce `TileLevels_…`” 查明不是 nom 比较的问题，而是类型索引 `root_loops op` 在 `op` 被改写后不再归约；已修（`DropTileRequests`）。
  第 4 条（第二轮原型，逐元素算子 `y = 2x`，M 为参数，只有 work-group 层）推进到了：
  - outline 不再把函数值和类型当作 kernel 参数，而是递归收集它们的自由变量（`m`）；
  - 类型引用了 host 值的指针（`Ptr «m, 64; F32»`）以字节指针跨过 kernel 边界，在 kernel 内映射完 `m` 之后再 bitcast 回去；
  - 剩下的失败：同一个运行时长度经由两个不同的 lam 绑定出现在依赖类型里（`_a#ff` 与 `_b#ff` 都是 `m`），原 world 中二者并存，
    RWPhase 重建时 `coerce` 严格比较便失败。这需要 core 在重建时统一这类等价的形状表达式，或在 lowering 时只让 `m` 经由一个绑定流动。
  原型在分支 `wip/dynamic-extents`（700e4539f），不在主线：outline 覆盖 `coerce` 对指针做 bitcast 之后，`World::app` 的定义域检查又因同一原因失败，
  逐处打补丁不是正确的修法。要继续，先在 core 选定方案：重建时统一等价的依赖形状，或在 CPS 转换时让运行时形状只经由一个绑定流动。
  **2026-11-14 决定：动态形状往后放。**
- **D1 第二步**：`layout_expr.matches_sw128` 用 F₂ 判定 K-major wgmma 面板的填充布局就是描述符让硬件读取的 128 字节 swizzle（CuTe `Swizzle<3, 4, 3>`），
  错一位的 swizzle 与不 swizzle 的布局都被判为不一致（`lit/tensor/layout_f2.mim`）。fragment 布局（m16n8k8、m16n8k16）的双射与 ldmatrix 一致性此前已有穷举检查。
- **D2 第二步**：`cp.async` ring 的协议也写成可执行模型 `cp_ring_safe blocks depth waits`：逐步跟踪每个缓冲装的块与填它的 group，检查计算时缓冲装着自己的块、group 已完成、重填的是另一个缓冲；
  `pipeline_plan.pending` 给出允许在途的 group 数（`depth − 2`），lower_to_mem 由此取值。多留一个 group 在途或单缓冲都被抓出（`lit/tensor/cp_ring_protocol.mim`）。
- **TileLaw（2026-11-14）**：设计见 `docs/tile-law-design.zh-CN.md`；原型 `tensor.law` 已能构造定律、以其事实为层级类型索引并检查规则、在 CPU 上执行两条切分引理。
