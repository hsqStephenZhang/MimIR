# GPU 多级 tile-and-fuse 的形式化建模（草案，2026-11-09）

本文回答一个问题：`cgo27-paper248.pdf`（下称"论文"）第 7 节把 tiling 建模为单个 op 上的 `Tileable → Tileable` 变换，
怎样把它推广到本分支实现的完整 GPU 多级 tiling——grid/group/warp/lane 绑定、融合、shared 面板、协作填充、
跨 item 归约、cp.async 多级流水和 tensor-core fragment——使每一步都有参考语义和明确的合法性条件。
它是设计草案，不描述已实现的类型；§2 说明现状，§3–§4 是目标模型，§5 是落地顺序。
初稿经 codex 审阅（`gpu-tiling-formalization-review.zh-CN.md`），本文已按其确认的问题修订。
本文以 IREE 为主要实现参照：两者都在结构化 op 上做 op→op 的多级 tiling，并把各级绑定到 GPU 执行层次。

## 1. 论文模型及其边界

论文的 tiling 由三件事组成：

1. `Tileable` 是一个依赖类型的存在包：结果类型构造子 `Out`、迭代域 `Sr`（`Ro` 个并行维、`Rr` 个归约维）、结果形状 `So`、
   每个并行维写哪个输出轴 `oax`、调度字段（`vdim`、`unroll`），以及 `retile`，它接收新的迭代域和两张仿射映射
   `iota: 新下标 → 旧下标`、`omega: (旧输出坐标, 新下标) → 新输出坐标`，重建计算。
2. 变换是普通 Mim 函数 `Schedule = Tileable → Tileable`：`strip_mine_par/red`、`interchange`、`vectorize`、`unroll_trailing`。
   类型保证秩与各字段之间的一致；`iota/omega` 是否双射、是否覆盖、是否保义是各组合子的证明义务，由显式构造和
   `refly.check` 守卫的条件（如 `K % ts == 0`）承担。`if_static` 只判断条件是否已是字面量，不是证明构造子。
3. 落地靠部分求值：nest builder 拿着 `Tileable` 的调度展开成循环。论文 §3 P4 把证明义务归结为
   "partial evaluator 正确 + rewriter 正确 + `loop` 与 `tile` 两个函数等价"。

这个模型的边界：

- **单个 op**：论文 §7 的 tensor 实例只处理一个 `map_reduce`，没有 producer/consumer，没有 region，没有 box 之间的关系。
- **只改迭代空间**：变换改变迭代域的分解（strip-mining 会改变秩）、顺序、向量维与展开窗口，但不涉及数据放在哪里、谁来算。
- **单一执行者**：结果是一个顺序循环嵌套；没有并行执行单元、共享存储、同步、异步效果或分布式的值。

GPU 多级 tiling 正好越过了这三条边界。

## 2. 现状：哪些部分已经是项，哪些仍在 C++

| 关注点 | 现在的表示 | 参考语义 | 合法性由谁保证 |
|---|---|---|---|
| 级的序列与作用域 | `TileLevels/GroupLevels/ItemLevels/WarpLevels` 互递归 nominal 变体（`tensor.mim`） | 无（只是数据） | 链类型只保证构造语法；sizes 字面量、与 root 循环数一致等在 TileFuse C++；整块的绑定级曾被跳过而丢失作用域，第 3 步起只跳过不绑定、不提升的级 |
| 单级切分 | TileFuse 产出 `tf_par`/`tf_reduce` → `tiled`/`tile_fold` | `tiled`、`tile_fold_impl` 的 Mim 定义 | box 覆盖、preimage 精确、归约块顺序：C++ 分析，无证据（`tiling-formalization-review.zh-CN.md` §1） |
| 融合 | `tile_of`/`tile_of_box` 生产者方程，`slice`/`slice_box` 视图 | 有 | provenance/包含关系在 C++，无证据 |
| promote / 面板 | `tensor.stage (pos, tiles, perm, pad, scope)` 表达逻辑的 split/perm/pad 形状 | `stage_read_impl`（及 exact 版本） | 其后的物理布局（perm/split/pad/旋转 swizzle）由 `panel_layout` 代价模型在 `lower_generate` 中另行选择，选出的布局是受检查的 `tensor.layout_expr` 项（第 4 步），但仍挂在旁表 `layouts_` 上而不在 buffer 类型里；`panels_` 只标记可按向量读的面板 |
| 执行绑定 | `tensor.bind n`（1 group，2 item，3 warp）；`gpu.bind.(group,item,warp,cooperative,staged,chained,async)` 标注 `affine.For` | 标注是恒等 | 一致性由 LowerToMem 的上下文 `uniform(value)`（只看是否依赖 item 原点）判断；barrier 只来自 §3.5.1 的三种构造 |
| 跨 item 归约（R8） | `merge_with`、`group_red_items` | step 有 Mim 定义，merge 树在 C++ 生成 | 归约律与 seed 单位元由 `reduction_law` 检查（第 2 步）；树形合并的覆盖漏洞（`n=8, elems=3, s=4` 只处理 12 个 pair 中的 8 个）已在第 1 步修复 |
| 流水 | `-X tensor:pipeline=n`（编译选项，不在 IR 里） | 无 | 环形 buffer、`commit_group`/`wait_group` 计数在 `lower_tile_fold` C++ |
| tensor core | `warp_mma (16, 8, 8)` → `tensor.mma` 标注 | 标注是恒等，参考计算来自它包住的 `map_reduce_post`；没有独立的 MMA 参考定义 | fragment 布局与 ldmatrix 地址是 `tensor.mma_m16n8k8_tf32` 的 Mim 定义，另有 `mma_ref` 参考（第 5 步）；累加器仍是 `fragments_` 旁表；TF32 默认由硬件截断，`-X tensor:tf32-rna=1` 显式舍入 |

结论：论文模型在本分支里覆盖的是"一级切分的值语义"；GPU 特有的决策（放置、布局、分布、同步、时序、分布式值）
都由 C++ 在 lowering 中**作出并同时实现**，它们既不是项，也没有可单独检查的合法性条件。

## 3. 目标模型：GPU 调度的七个语义分面

GPU 多级调度的每个决策落在下面七个**语义分面**上。它们不正交：T5 依赖 T3 的放置与 T4 的分布，T6 复用 T5 的 happens-before，
T7 同时依赖 T3、T4、T5；一个 level 构造子通常同时产生多个分面的义务（§3.8）。每个分面都满足同一个模式——

> 一个 **Mim 级的参考定义**（值语义，部分求值后是顺序代码），一个 **实现定义**（GPU 语义），
> 和一个 **合法性谓词** P，满足 `P ⇒ 实现 ~ 参考`；P 由构造保证，或以可归一化的 Bool（`refly.check`）出现在 IR 里由编译器判定。

MimIR 的依赖类型靠归一化工作：没有证明项、没有 tactic，用户无法陈述并证明定理。
因此"`P ⇒ 实现 ~ 参考`"由 MimIR 开发者对每个原语论证一次（论文 P4 的同类义务），用户只选择调度，从不提供证明；
每个具体实例上的 P 由 §4.1 的四种方式之一消解。

论文的 `Schedule` 是 T1 在 P 为简单整除条件时的特例。关系 `~` 有三种，每个分面要说明用哪一种：

- `≡value`：精确的值语义相等（T1–T4，归约在 OrderedChunk 下）；
- `≈mode`：在数值许可下的关系（`fast` 的重结合、TF32 转换、ParallelFold 的重组）；
- `⊑obs`：GPU 并发程序对可观察输出的精化（T5、T6），需要一个指定的 GPU 内存模型。

本草案**不覆盖**：多 kernel 的划分、数据常驻与 kernel 间同步（记作 T0）；跨 group 的 split-K（需要全局通信或第二个 kernel）；
warp shuffle 归约（需要 warp collective、active mask 与归约律）；persistent kernel、cluster、TMA；
以及寄存器、shared 与 block 大小这类资源可行性（记作 T8）。

### 3.1 T1 迭代空间切分（论文模型本身）

`tile(op, grid) ≡value op`：输出按 box 网格精确划分，每个 box 计算 `tile_of(op, box)`。

- 参考：`tiled`/`tile_fold_impl`。
- 合法性 `ExactCover(grid, So)`：box 两两不交、并集为 `So`；cap/peel 尾块的有效 extent 与容量分开（`ext ≤ cap`，不要求 `off + cap ≤ So`）。
- 归约块 `OrderedChunk`：块覆盖归约域，且只切最外层归约维或显式携带字典序前缀；否则块内次序改变（上次 review §2 的 `(a:2, b:4)` 反例）。
  允许重组时改用 T5 的 `ParallelFold`，关系降为 `≈mode`。

### 3.2 T2 融合（region 形成）

`tile_of(c ∘ p, box) ≡value tile_of(c, box)[p ↦ tile_of(p, pre_c(box))]`。

- 参考：`tile_of_box` 的规格化（`tensor:tile-ref=1` 已是差分入口）。
- 合法性：生产者是纯的（无可观察效果），`Covers(pre_c(box), reads_c(box))`（生产者 box 包含消费者在该 box 上的全部读），
  resident 复用满足 `Subview(parent_ext, box)` 且不跨所有者；`Covers` 只是其中的空间部分。
- 这一类与 GPU 无关，是论文模型从单 op 到 op 图的推广：变换对象从 `Tileable` 变成"带输入 box 的 region"。

### 3.3 T3 放置与布局（staging / promotion）

`stage(x, L) ≡value x`：值不变，只改排布；放置另行表示。

- 布局 L 取自一个**封闭的布局代数** `LayoutExpr`（perm、常数 split、pad、块旋转 swizzle），解释为 semi-affine 映射
  `«r; Index» → «p; Index»`，读 map 与它复合：`read' = L ∘ read`。现在的 `layout_lam` 就是用 `semiop.mul/floordiv/rem` 构造它；
  现有的 swizzle 是块旋转 `chunk' = (chunk + group) mod chunks`（`panel_layout`），不需要 XOR。pad 只改变物理形状（步长），不改变坐标映射。
- 合法性 `Injective(L, s_logical, s_physical)`：在 `LayoutExpr` 上、字面量形状下可以有限枚举或用专门判定器检查；不接受任意 Mim 函数作为布局。
- 代价模型（选哪个 L）可以留在 C++，但**选出的 L 必须是项**，挂在 stage 上，而不是旁表；这样 T5 的依赖分析和 T7 的 fragment 都能引用同一个 L。
- 放置（`Global | Shared(group) | Private(item)`）与布局分开：前者决定所有者、生命周期与地址空间，后者只决定坐标映射，
  避免 Injective、生命周期与 barrier 的条件纠缠。warp fragment 是值的分布（T7），不是一种放置。

### 3.4 T4 分布（执行绑定）

`bind(D, region) ≡value region`：一个并行 region 的各次迭代由处理器层次上的不同执行者完成。

- 分布 D 是一个映射 `proc: «q; Idx Pq» → tile 原点`（group 号 → group tile，warp 号 → warp tile，lane → 元素），并带一个覆盖模式：
  `UniqueOwner`（输出 tile，配 T1 的 `ExactCover`）、`Replicated`（广播）、`ReductionPartition`（R8 的 partial）、
  `CooperativeTransfer`（协作填充）；未激活的 lane 与谓词化也属于 D。
- 合法性 `Independent(region)`：UniqueOwner 的并行迭代写不相交的输出，只读不可变的输入或本 region 已完成的值。
- 当前的 `GroupLevels`/`ItemLevels`/`WarpLevels` 链是 D 的**语法**；目标是让每个绑定级产生一个 `Dist` 项，
  launch 的 grid/block 大小从 Dist 推出，而不是 outline 再从循环 trip count 求最大值；整块简化只能消去尺寸切分，必须保留作用域。

### 3.5 T5 协作与同步：group 是 BSP 程序

group 内的代码不再是"每个 item 一个独立循环"，而是**整体同步并行（BSP）**的超步序列：

```
group_region = superstep₁ ; barrier ; superstep₂ ; barrier ; …
superstep    = forall item ∈ items. (本地计算 ; 对 shared 的写)
```

- **一致值**（不依赖 item 号）在参考语义里只算一次；实现里由 `forall item` 分摊（协作填充），这正是 `cooperative` 循环。
- **必须建立的 happens-before 边由产生 shared 读写的变换结构化建立**：tile、promotion、跨 item 归约和 pipeline 在构造 shared
  生产/消费时同时插入同步；同步尽量挂在 stage 产生的值上，采用与 IREE `iree_gpu.value_barrier` 相同的值语义视角。
  现在 staged 填充、`lower_tile_fold` 流水循环顶部和 R8 各轮的 `fence` 已是这种做法的手写实例。首选由组合子固定
  barrier 的作用域、动态实例和相对位置，而不是在事后对整个函数推导 barrier。
- 合法性 `DRF(superstep)`：同一超步内，不同 item 对同一 shared 位置没有写-写或读-写冲突；barrier 只出现在一致控制流上，
  同一作用域、同一动态实例被所有参与者到达，并提供所需的顺序与可见性（不只是控制上的会合）。
  在这些前提下，且限定在无原子、无异步代理访问、无 warp collective 的子语言里，BSP 程序对可观察输出精化（`⊑obs`）
  保持每线程程序序与超步边界的顺序程序。不能依赖"warp 内隐式同步"：Volta 之后的独立线程调度不提供它。
- 上述 DRF 保证的范围是这些受控构造生成的静态 shared 面板、结构化控制流、已知分布和已知别名关系。若布局、别名、控制一致性
  或异步完成关系超出构造器可知范围，则退回更强的 workgroup barrier、同步 copy、取消 shared 复用/协作分布，最终可退回串行；
  不能以用户保证代替。受限 BSP 检查器只作为调试验证，检查构造器是否兑现不变量，不是正确性的来源。
- R8（跨 item 归约）是 BSP 里的一个超步序列：私有 fold → 写 partial → barrier → 树形合并若干轮。
  它额外需要归约律 `ParallelFold(step, seed, merge, mode)`，至少表达 `fold(seed, X ++ Y) ≈ merge(fold(seed, X), fold(seed, Y))`，
  以及 `ReductionPartition` 的覆盖（第 1 步修复的树形合并漏算即违反此条）。

#### 3.5.1 三种同步构造

`LowerToMem` 只在下面三个构造里产生 barrier，每个 barrier 的作用域、动态实例和相对位置都由构造固定，不做事后推导。
记 `B` 为 work-group barrier（`gpu.fence`，或 cooperative 循环首尾由 `gpu.outline` 生成的 `sync_work_items`）。

| 构造 | 超步结构 | 构造给出的义务 | 实现位置 |
|---|---|---|---|
| 填充批 | `B ; (fill₁ ∥ … ∥ fillₖ) ; B ; consume` | 各 fill 写各自分配的面板，互不相交；fill 只读 global 或已发布的值；首个 `B` 结束上一个读这些面板的超步（WAR） | `cooperative()` 的 `staged`/`chained` 标签，`lower_map_reduce`、`lower_mma` 在消费者输入内存上的 `fence` |
| R8 合并轮 | `write partial ; B ; (merge 一轮 ; B)*` | 每轮的配对互不相交且覆盖所有存活 partial；`B` 位于 item 守卫之外 | `lower_bound_fold` |
| 流水环槽 | `wait_group(n − 2) ; B ; compute(slot b) ∥ refill(slot b + n − 1) ; commit` | 读槽前其写组已 wait 且其后有 `B`；重填的槽上一次读它的超步已被同一个 `B` 结束 | `lower_tile_fold` |

单个不成批的一致填充是 `B ; fill ; B`（`cooperative` 标签）；异步填充在批末 `B` 前加 `async_wait`（`async` 标签）。

- **值语义同步。** MimIR 的内存是 SSA 值 `mem.M`，`gpu.fence: mem.M 0 → mem.M 0` 已是挂在内存值上的 value barrier，
  因此不另设 `value_barrier`。填充批的末尾 barrier 挂在**消费者自己的输入内存**上，所以它随消费者一起存活；
  游离在未被使用的内存值上的 fence 会在 `AddMem` 重新串接内存时作为死代码消失，这也是 cooperative 循环的末尾 barrier
  必须属于循环结构、而不能写成循环之后单独一个 fence 的原因。
- **标签是协议，不是正确性来源。** `staged`/`chained` 只表示"本批末尾的 `B` 由消费者给出"，这一义务由 `fence_pending_`
  在同一次输入改写内兑现；离开这次改写（`fe::Restore`）时协议结束，所以标签不会泄漏到批外。
- **一致到达。** 三个构造的 `B` 都在 work-group 一致的控制流上；`gpu.outline` 拒绝 item 守卫内的 fence、带 barrier 的
  cooperative 循环和非 warp 对齐守卫内的 warp collective（第 3 步）。
- **合并只做局部。** `gpu.remove_double_syncs` 只删去内存链上紧跟另一个 barrier、或位于 kernel 入口的 barrier；
  跨构造的 barrier 消除需要知道 fill 之间相互独立，这一知识只在构造内部可得，所以不做全局消除。

### 3.6 T6 时序：流水与异步拷贝

`pipeline(n, fold) ⊑obs fold`：把 `fold step acc (blocks)` 中每块的 stage 提前 n − 1 块执行，放进 n 个 buffer 的环。

- 参考：原 `tile_fold`。实现：现在 `lower_tile_fold` 生成的环（`ring[s]` 存当前块之后第 s 块）。
- 这是一个**重定时**：stage(b) 的计算从迭代 b 移到 b − (n − 1)，读它的 compute(b) 不变。
- 异步拷贝 `store_async` 的语义是"一个延迟完成的写"：写入组号 g；`commit` 关闭组，`wait_group k` 只保证**本线程**除最新 k 组外都已完成，
  不为其他线程提供可见性，跨线程消费还需要其后的 CTA barrier。
- 等价是**整程序**的，不是操作级的：`store_async` 不能在任意上下文中替换 `store`。可证明的是，满足 `AsyncSafe` 的完整流水程序
  对可观察输出精化同步 staging 的程序。`AsyncSafe` 包括：
  - prologue 恰好提交 depth − 1 组；每轮每个线程提交相同数量与顺序的组（含空组），所以 `wait_group (depth − 2)` 等到当前块；
  - 读环槽 s 之前，写它的组已 wait 且之后有 barrier；写环槽 s 之前，上一次读它的超步已结束（同一个 barrier）；
  - 同一组内的写目标互不相交；
  - 末尾 clamp 后重复填充的槽不再被消费；循环出口的 wait-all 在 shared 生命周期结束前；
  - depth 在 PTX 立即数与硬件支持范围内（`lower_tile_fold` 取 `depth = min(pipeline, 块数)`）。

### 3.7 T7 分布式值：warp fragment 与 tensor core

tensor core 引入了**分布在 lane 上的值**：一个 16×8 的块不在任何一个 lane 里，而是 32 个 lane 各持 4 个元素。

- 类型：`Frag (L, s, T)`，其中 `L: [lane: Idx 32, reg: Idx k] → «2; Idx s»` 是双射（fragment 布局，PTX ISA 给出 A/B/C 的 L）。
  一个 lane 的局部值是 `«k; T»`；整个 fragment 的语义是"按 L 散布到 s 上的张量"。
- `mma: Frag(L_A, (16, 8)) × Frag(L_B, (8, 8)) × Frag(L_C, (16, 8)) → Frag(L_C, (16, 8))`，
  参考语义：`scatter_{L_C}(gather_{L_A} a · gather_{L_B} b + gather_{L_C} c)`，即 16×8×8 的 map_reduce。
  它是 `.sync.aligned` 的 collective：warp 内所有未退出的 lane 必须一致执行。
- `ldmatrix: Panel(L_panel) → Frag(L_A)`：参考语义是"按 L_A 从面板里取元素"。它是一个集体置换，实现需要三个关系
  并证明与 `L_panel` 相容：`Addr_LD`（提供地址的 lane → shared 行地址，x4 时 lane 0–7、8–15、16–23、24–31 各提供一个矩阵的 8 行）、
  `Transfer_LD`（(提供者 lane, 加载的元素) → (接收者 lane, 寄存器)）和 `L_A`。
  这些关系现在是 `tensor.mma_m16n8k8_tf32` 里的 Mim 定义（`l_a`/`l_b`/`l_c`、`addr_ld`、`transfer_ld`、面板行选择 `panel_a_addr`
  及其组合 `ld_a`）；`lit/tensor/mma_reference.mim` 穷举检查三个布局是双射、`ld_a = l_a`，并在 CPU 上把 `mma_ref` 与矩阵乘逐元素比对。
  `lower_mma` 与 `store_fragments` 的地址由这些定义部分求值得到，不再有手写的行列公式。
- warp 级累加器：现在是每个 lane 的 SSA 值 `«F; «4; F32»»` 加 `fragments_` 旁表；目标是 `Frag(L_C^F, (WM, WN), F32)` 这个类型本身。
- 数值：TF32 与 `fast` 一样是**许可**（`≈mode`），不是等价：`warp_mma` 的语义是"操作数先转成 TF32 再做 map_reduce"。
  转换有两种实现：默认由 `mma` 截断 F32 的低 13 位；`-X tensor:tf32-rna=1` 时在 fragment 载入后显式调用 `gpu.tf32_rna`
  （`cvt.rna.tf32.f32`，就近、平局远离零；其位级参考 `gpu.tf32_rna_ref` 按 sm_80 实测行为给出 Inf/NaN 的结果）。
  两者都在许可之内；显式舍入在 A100 上使 4096³ GEMM 慢约 27%，所以不作默认。

### 3.8 组合

一个 `tile_req` 的 level 链按顺序给出这些分面的参数：

| 构造子 | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| `serial_group (sizes, tail, promote, …)` | 网格 | 融合 members | promote 位 → group 级 stage | Dist(group) | 开始一个 BSP 程序 | | |
| `group_warp (sizes, tail, …)` | 子网格 | | | Dist(warp) | | | |
| `warp_red (sizes, tail, promote, …)` | 有序块 | | 面板 stage(L) | | 填充超步 + barrier | `pipeline n` | |
| `warp_mma (16, 8, 8)` | 叶子 | | | Dist(lane) | | | Frag 布局与 mma |
| `group_red_items (…)` | 拆分归约 | | partials 在 shared | Dist(item, ReductionPartition) | R8 超步序列 | | |

链的类型现在保证"列"之间的作用域关系；目标是让每个构造子**展开为上述分面的复合**（像论文的 `dot_schedule` 由组合子复合而成），
合法性谓词随展开出现在 IR 里，由归一化消去或作为 `refly.check` 残留报错。

## 4. 落到 MimIR：项、公理与可信计算基

| 分面 | 变成项的部分 | 保留为公理（实现给后端） | 留在 C++ 的可信分析 |
|---|---|---|---|
| T1 | 网格、box、`OrderedChunk` 条件 | `tiled`、`tile_fold` | 动态形状、cap/peel 下的覆盖证明 |
| T2 | `Covers`/`Subview` 的 Bool 条件 | `tile_of_box` | preimage/image 计算（返回带理由的结果） |
| T3 | `LayoutExpr`、放置 | `stage` | 代价模型选择 L |
| T4 | `Dist` 项及覆盖模式、launch 形状 | `gpu.launch` | 动态形状下的覆盖证明 |
| T5 | 超步结构、stage 值及其同步、归约律 witness | `gpu.fence`（挂在内存值上的 barrier）、cooperative 循环首尾的 barrier | 三种构造（§3.5.1）按构造插入；受限 DRF/一致性检查仅作可选调试验证 |
| T6 | `pipeline n` 作为 tile_fold 的变换子 | `store_async`、`async_commit`、`async_wait_group` | 环的生成（或写成 Mim 定义） |
| T7 | `Frag` 类型、L_A/L_B/L_C、`Addr_LD`/`Transfer_LD`、ldmatrix/mma 的参考定义 | `mma_tf32`、`ldmatrix`、TF32 转换 | ISA 的 fragment 表、intrinsic ABI、对齐与收敛前提（后端可信边界） |

几点取舍：

- **代价模型不进 IR，选择结果进 IR。** 与论文"调度是项"一致；GPU 的布局和 tile 尺寸搜索空间大，选择器留在 C++ 是务实的，
  但选择的输出（L、D、n）必须是项，供检查和后续变换引用。
- **影响语义、跨 pass 的信息进入 IR，lowering 内部的缓存不必。** 旁表分三类：必须成为项的（选出的物理布局、fragment 布局、
  流水深度、所有者与作用域）；可以重算但要验证的（一致性、依赖、覆盖）；纯 lowering 缓存（`fresh_memo_`、`storage_shapes_`、
  `fill_inner_`、`multiples_` 这类）。目标是消除第一类旁表，不是消除所有旁表。
- **同步由 shared 生产/消费构造建立。** `cooperative/staged/chained/async` 四个标签目前编码 barrier 放置；目标是由 tile、stage、
  R8 和 pipeline 组合子直接产生值语义同步及后端 barrier。可选检查器可以复核其 happens-before 与一致到达，但不重新承担全局推导。
- **证明义务的形状只在纯函数部分与论文一致。** T1–T3 只需证明"实现 ≡value 参考（在 P 下）"一次，再加 rewriter 与部分求值的正确性；
  T5–T7 还需要 GPU 内存模型、并发、生命周期、数值近似与 collective 的未定义行为，不能只复用论文的三项义务。

### 4.1 义务由谁消解（没有用户证明）

| 方式 | 含义 | 例子 |
|---|---|---|
| 构造即正确 | 组合子的形式本身蕴含 P；由开发者对组合子论证一次 | 网格公式生成的分块满足 ExactCover；`LayoutExpr` 的每个构造子（perm、常数 split、pad、块旋转）都是单射，组合后仍是单射；产生 shared 读写的 stage/R8/pipeline 构造同时建立同步；pipeline 变换子生成的环满足 AsyncSafe；ISA 给出的 fragment 布局表 |
| 可判定检查 | 编译器对具体实例自动判定，条件归一化为 `tt` 才应用；同步检查可作为调试断言 | OrderedChunk（切的是否最外层归约维）；Covers/Subview（仿射 preimage/image）；对构造结果做受限 DRF 与 barrier 一致性复核；已知 combiner 的归约律（整数 `+`、`max` 等） |
| 许可 | 用户表态接受语义变化，编译器不声称等价 | `fast` 允许浮点重结合；TF32 转换；用户给的 `merge` 的结合律与交换律 |
| 保守退路 | 以上都不成立时 | 不融合、不并行、按串行 fold；动态形状下生成运行时守卫；或作为 `refly.check` 残留报错 |

设计原则：**能构造即正确的，就不留给判定；能判定的，就不留给许可。**
例如布局只允许 `LayoutExpr` 的构造子而不接受任意 Mim 函数，于是单射不需要判定器；
用户自定义的 `merge` 只能走"许可"，其结合律与交换律没有机器检查，所以 R8 的结果对它只承诺 `≈mode`。
文献中需要证明项的工作（Kuiper、ATL）只能作为开发者离线论证原语正确性的依据，不能要求用户提供同类证明。

## 5. 渐进路线

按"能发现真实错误 / 改动量"排序：

1. **修复 R8 树形合并的覆盖**（`lower_bound_fold`）：每个 item 遍历 `pair = item + q·trip, pair < pairs`，
   加 `n = 8, elems = 3` 的 device 用例。这是已确认、局部、可测试的真实 bug；它违反的是 `ReductionPartition` 的覆盖，不是输出的 ExactCover。
2. **归约契约与 root descriptor**（上次 review 的 P0）：`OrderedChunk`、`ParallelFold`；levels 按真实的 accumulator 类型 `To` 索引，而不是输出元素类型。
3. **作用域与 collective 的保持检查**：绑定级不会因整块简化而消失，barrier 一致可达。
4. **选出的物理布局成为受控项**：`LayoutExpr` 支持现有 perm/split/pad/旋转子集，配有限域 Injective 检查。
5. **T7 的 fragment 参考定义 + 差分测试**：L_A/L_B/L_C、`Addr_LD`/`Transfer_LD` 写成 Mim；显式 TF32 转换；CPU 上用参考定义执行同一请求比对。
6. **结构化 barrier 与值语义同步**：stage、R8 和 pipeline 在产生 shared 读写时同时产生同步；同步尽量挂在 stage 值上；
   可选的受限 BSP 调试检查覆盖仿射访问、结构化控制流、CTA shared 以及受控异步形态。
7. **T6 的 `pipeline` 变换子**：把 `-X tensor:pipeline` 变成链上的参数，`AsyncSafe` 由 T5 的超步结构给出。

进度（2026-11-10）：第 1–5 步已实现，第 4 步只完成了布局成为项的第一阶段，布局进入 buffer 类型尚未开始。
第 6 步经核对已由现有实现满足，整理为 §3.5.1 的三种构造；`value_barrier` 由挂在内存值上的 `gpu.fence` 代替，未新增代码。

风险：T5 不再要求一个全局分析器证明正确性；其剩余难点是为每个 shared 构造审清 GPU 内存模型前提、异步完成与 collective
一致到达，并让可选检查器发现构造器回归。工程迁移最大的是 T3（触及 buffer 类型、物理形状、向量访问与 mma 面板）；
T6 的环槽生命周期和 wait/barrier 配对仍需逐个构造论证。最先能抓到真实 bug 的是第 1 步，其次是第 5 步的差分测试。

## 6. 开放问题

- `Frag` 的布局 L 是否需要一般的"线性布局"（如 Triton 的 linear layouts、CuTe 的 layout 代数），还是 PTX 给出的几种固定布局足够。
- DRF 检查只在访问函数、别名、调用、动态下标与控制条件都受限时可判定；"结构化控制流"本身不够。
- 内存所有者与生命周期（Shared(group)、Private(item)）是否用 nominal 参数表示即可，还是需要 effect/ownership 检查；后者在没有用户证明的前提下只能是编译器内的自动检查。
- 原语"实现 ~ 参考"的一次性论证是否要机械化（例如在 Coq 中对核心演算形式化），以及怎样让 C++ 中的判定过程（preimage、DRF）本身可信。
- 动态形状下 ExactCover/Injective 依赖 `core.nat` 的归一化强度（论文 §6 的同一前提）。
- T0（多 kernel 与常驻）和 T8（资源可行性）如何进入同一框架。

## 7. 相关工作

- IREE：GPU `LoweringConfigAttr` 的 `TilingLevel`（Workgroup、Reduction、PartialReduction、Serial、Thread、Subgroup、Lane）在
  结构化 op 上驱动多级 tile-and-fuse，并由 vector distribution 把内层 tile 分发到 subgroup/lane；`MMAIntrinsicAttr` 与
  `MMASingleSubgroupLayout` 描述 MMA 及每 lane 的分布；`iree_gpu.value_barrier` 为 tensor/vector 提供值语义同步。它与 MimIR 的
  level 链起点相同，都是 op→op 的多级 tiling 并绑定执行层次，因此本文以 IREE 为主基准。
- Halide / TVM：`gpu_blocks`/`gpu_threads`、`compute_at`、`store_in(GPUShared)`、`cache_read(scope)`——调度是 DSL，合法性靠边界推断，不是 IR 中的项。
- TensorIR（TVM）：block 的 iter var 绑定与作用域验证，最接近 T4/T5 的检查器。
- Graphene（ASPLOS'23）：把 GPU 张量计算建模为按线程层次分解的张量与布局，与 T3/T7 的布局和 `Frag` 最接近。
- CuTe / Triton `LinearLayout`：布局表示、组合、求逆、XOR swizzle 与 MMA fragment 双射，可作为 T3/T7 中 L 的一般形式；
  Triton 的 block 级 tile 编程模型不是本文多级 op→op tiling 的主基准。
- Triton MMA/ldmatrix lowering 与 dot `inputPrecision`：作为 T7 的指令映射和显式 TF32 语义参照。
- Fireiron：数据移动作为一等公民，对应 T3 的 stage 与 T6 的异步拷贝。
