# GPU 多级 tile-and-fuse 的形式化：模型、实现与论证（2026-11-10）

`cgo27-paper248.pdf`（下称"论文"）第 7 节把 tiling 建模为单个 op 上的 `Tileable → Tileable` 变换。
本文说明怎样把它推广到本分支实现的完整 GPU 多级 tiling：grid/group/warp/lane 绑定、融合、shared 面板及其布局、
协作填充、跨 item 归约、cp.async 多级流水和 tensor-core fragment。
对每一部分，本文给出参考语义、实现、合法性条件，以及该条件在没有用户证明的前提下由谁消解，并标明哪些已经是 IR 中的项、哪些仍在 C++。
设计与实现步骤见 `gpu-tiling-implementation.zh-CN.md`，初稿的审阅意见见 `gpu-tiling-formalization-review.zh-CN.md`，文献调研见 `gpu-tiling-literature.zh-CN.md`。
仍由 C++ 决定或相信的前提及其消解办法见 `gpu-tiling-formalization-gaps.zh-CN.md`，那份文档的评审见 `gpu-tiling-formalization-gaps-review.zh-CN.md`。
实现参照以 IREE 为主：两者都在结构化 op 上做 op→op 的多级 tiling，并把各级绑定到 GPU 执行层次。

## 0. 概要

GPU 调度的决策分成七个语义分面 T1–T7。
每个分面有一个 Mim 级参考定义、一个 GPU 实现，以及合法性谓词 P，满足 `P ⇒ 实现 ~ 参考`。
`~` 是三种关系之一：值相等 `≡value`、数值许可下的 `≈mode`、并发程序对可观察输出的精化 `⊑obs`。

| 分面 | 关系 | IR 中的项 | 合法性的主要消解方式 |
|---|---|---|---|
| T1 迭代空间切分 | `≡value` | `tiled`、`tile_fold`、level 链 | 构造即正确（网格公式）；`OrderedChunk` 可判定检查 |
| T2 融合 | `≡value` | `tile_of`/`tile_of_box`、`slice` | C++ 的 preimage/image 分析；`tensor:tile-ref=1` 差分 |
| T3 放置与布局 | `≡value` | `tensor.stage`、`tensor.placed L`、`layout_expr` | 布局代数构造即单射，参数由 `refly.check` 检查 |
| T4 分布 | `≡value` | `tensor.bind n`、`gpu.bind.*` | 作用域保持检查、分歧检查；`Dist` 尚不是项 |
| T5 协作与同步 | `⊑obs` | `gpu.fence`（挂在内存值上） | 三种同步构造按构造插入 barrier |
| T6 流水 | `⊑obs` | 归约级的 `pipeline` 字段、`tensor.pipeline n` | 环由构造满足 AsyncSafe；不满足时退为同步 |
| T7 分布式值 | `≡value` / `≈mode` | `mma_m16n8k8_tf32` 的布局与参考、`gpu.tf32_rna` | 布局双射穷举检查；TF32 是许可 |

## 1. 论文模型及其边界

论文的 tiling 由三件事组成：

1. `Tileable` 是一个依赖类型的存在包：结果类型构造子 `Out`、迭代域 `Sr`（`Ro` 个并行维、`Rr` 个归约维）、结果形状 `So`、
   每个并行维写哪个输出轴 `oax`、调度字段（`vdim`、`unroll`），以及 `retile`，它接收新的迭代域和两张仿射映射
   `iota: 新下标 → 旧下标`、`omega: (旧输出坐标, 新下标) → 新输出坐标`，重建计算。
2. 变换是普通 Mim 函数 `Schedule = Tileable → Tileable`：`strip_mine_par/red`、`interchange`、`vectorize`、`unroll_trailing`。
   类型保证秩与各字段之间的一致；`iota/omega` 是否双射、覆盖、保义是各组合子的证明义务，由显式构造和 `refly.check` 守卫的条件承担。
3. 落地靠部分求值：nest builder 拿着 `Tileable` 的调度展开成循环。论文 §3 P4 把证明义务归结为
   "partial evaluator 正确 + rewriter 正确 + `loop` 与 `tile` 两个函数等价"。

这个模型有三条边界，GPU 多级 tiling 都越过了：

- **单个 op**：没有 producer/consumer，没有 region，没有 box 之间的关系。
- **只改迭代空间**：不涉及数据放在哪里、以什么布局存放、谁来算。
- **单一执行者**：结果是顺序循环嵌套，没有并行执行单元、共享存储、同步、异步效果或分布在 lane 上的值。

## 2. 方法

### 2.1 没有证明项

MimIR 的依赖类型靠归一化工作：没有证明项、没有 tactic，用户无法陈述并证明定理。
因此 `P ⇒ 实现 ~ 参考` 由 MimIR 开发者对每个原语论证一次，这与论文 P4 是同一类义务；用户只选择调度，从不提供证明。
每个具体实例上的 P 由下面四种方式之一消解：

| 方式 | 含义 | 本分支中的例子 |
|---|---|---|
| 构造即正确 | 组合子的形式本身蕴含 P，开发者对组合子论证一次 | 网格公式给出的分块满足 ExactCover；`layout_expr` 的构造子都是单射；三种同步构造；流水环满足 AsyncSafe；ISA 给出的 fragment 布局 |
| 可判定检查 | 编译器对具体实例判定，条件归一化为 `tt` 才应用 | `OrderedChunk`；`reduction_law` 识别的归约律与 seed 单位元；`layout_expr.make` 的参数检查；fragment 布局的有限域双射检查 |
| 许可 | 用户表态接受语义变化，编译器不声称等价 | `fast` 的浮点重结合；TF32；`merge_with` 给出的 merge |
| 保守退路 | 以上都不成立时 | 不融合、不提升、按串行 fold；cp.async 不可用时同步填充；或作为 `refly.check` 残留报错。不是所有情况都有退路：绑定级的动态 extent、非整除尾块的 peel、MMA 的不整除是直接报错 |

设计原则：**能构造即正确的，就不留给判定；能判定的，就不留给许可。**
例如布局只接受 `layout_expr` 的构造子而不接受任意 Mim 函数，于是单射不需要判定器。
需要证明项的工作（Kuiper、ATL）只能作为开发者离线论证原语正确性的依据。

### 2.2 三种关系

- `≡value`：精确的值语义相等。T1–T4 与 T7 的布局部分属于这一类，归约在 OrderedChunk 下也是。
- `≈mode`：数值许可下的关系，包括 `fast` 的重结合、TF32 转换，以及用 ParallelFold 重组归约。
- `⊑obs`：GPU 并发程序对可观察输出的精化，适用于 T5、T6。它依赖一个指定的 GPU 内存模型，在下文的受限子语言里成立。

### 2.3 什么必须是项

影响语义、跨 pass 的信息必须是 IR 中的项，lowering 内部的缓存不必。
旁表分三类：

- **必须成为项**：选出的物理布局、fragment 布局、流水深度、作用域与所有者。前三者已经是项，作用域由 `tensor.bind` 表示，所有者尚未表示。
- **可重算但要验证**：一致性、依赖、覆盖。
- **lowering 内部的表**：`fill_inner_`、把布局项从填充处带到读取处的 `layouts_` 是纯缓存。
  `storage_shapes_`（容量）、`multiples_`（整除与对齐）和 `fresh_memo_`（按 lambda 限定以避免悬空变量）可以不公开，但它们承载的事实与正确性相关，属于上一类：可重算，但要有依据。

代价模型（选哪个 L、选哪组 tile 尺寸）留在 C++，但它选出的结果必须是项，供检查和后续变换引用。

## 3. 七个语义分面

### 3.1 T1 迭代空间切分

`tile(op, grid) ≡value op`：输出按 box 网格精确划分，每个 box 计算 `tile_of(op, box)`。

- **参考**：`tiled`、`tile_fold_impl` 的 Mim 定义。
- **合法性**：
  - `ExactCover(grid, So)`：box 两两不交、并集为 `So`。cap/peel 尾块的有效外延与容量分开，`ext ≤ cap`，不要求 `off + cap ≤ So`。
  - `OrderedChunk`：归约块覆盖归约域，且只切最外层归约维；否则块内次序改变。
- **消解**：
  - ExactCover 由 TileFuse 的网格公式构造保证。
  - OrderedChunk 由 TileFuse 检查：内层归约维只有在识别出可重结合的归约律，或用户以 `merge_with` 许可时才能切，此时关系降为 `≈mode`。
- **剩余**：动态形状下的覆盖仍依赖 C++ 分析与 `core.nat` 的归一化强度。

### 3.2 T2 融合

`tile_of(c ∘ p, box) ≡value tile_of(c, box)[p ↦ tile_of(p, pre_c(box))]`。
请求、参考切片 `tile_of_ref` 和部分 builder 是 Mim 定义；一般 producer 的重写由 C++ 的 `producer_tile` 构造，这条方程描述的是它应满足的规格。

- **参考**：`tile_of_box` 规格化为"整个生产者的切片"；`tensor:tile-ref=1` 用它做差分。
- **合法性**：
  - 生产者是纯的。
  - `Covers(pre_c(box), reads_c(box))`：生产者 box 包含消费者在该 box 上的全部读。
  - resident 复用满足 `Subview(parent_ext, box)` 且不跨所有者。
- **消解**：preimage/image 在 C++ 中计算，失败时不融合。
- 这是论文模型从单 op 到 op 图的推广，与 GPU 无关：变换对象从 `Tileable` 变成"带输入 box 的 region"。

### 3.3 T3 放置与布局

值不变，只改排布。按当前接口这是两条等价，不是一条同类型的相等：
`stage` 是逻辑重索引，改变秩和形状，等价的是"经读映射读回的元素"；`placed L` 类型上是恒等，等价的是"物理布局不改变逻辑值"。

- **IR 表示**：
  - `tensor.stage (pos, tiles, perm, pad) scope x` 给出逻辑的 split/perm 形状。
  - 物理布局是 `tensor.placed L (stage …)` 中的项 `L: tensor.layout_expr.Expr`。
  - 布局由调度器 TileFuse 在 promotion 时选出，LowerToMem 只解码 L，不再自己选择。
- **布局代数**：`layout_expr.Data` 有以下字段：
  - 逻辑秩；
  - 末维常数 split；
  - `rank + 1` 维置换；
  - 各物理维的 pad；
  - 块 swizzle 参数（宽度、组数、移位、块数）；
  - swizzle 方式：旋转 `c ↦ (c + g) mod chunks`，或异或 `c ↦ c ⊕ g`，其中 `g = (row >> shift) mod groups`。
- **解释器**：只有一个解释器 `interpret`，以索引代数 `IndexOps` 为参数。`map_affine` 用 affine 索引实例化它，`map_i64` 用 I64 实例化它，所以符号读映射和具体坐标计算的是同一个函数。
- **单射的论证**（构造即正确）：
  - 末维的商与余数，加上置换，再做混合进制合并，三者组合起来是单射；pad 只改变步长，不改变坐标映射。
  - 对固定的一行，块旋转是 `[0, chunks)` 上的置换。
  - 若 `chunks` 是 2 的幂，则 `g < groups ≤ chunks` 时 `c ↦ c ⊕ g` 也是置换。
- **参数检查**：`layout_expr.make` 用 `refly.check` 检查以下条件，不满足时作为残留报错：
  - 秩不小于 2；
  - split 非零且整除末维；
  - 置换不重复；
  - swizzle 参数非零，且 `groups ≤ chunks`；
  - `chunks · width` 等于物理行宽；
  - 异或时 `chunks` 是 2 的幂。
- **代价模型**：在 C++ 中（`phase/panel_layout.cpp`），按 warp 的 shared 访问波前数计算填充与 item 读取的银行冲突，
  在置换、split、pad 和两种 swizzle 中取代价最小者；代价相同时优先异或，因为它的块号不需要加法。
  TileFuse 把 region 外延与块大小代换为其上界，得到与 LowerToMem 特化后相同的面板形状、连续维、向量宽度和 item 读取方式。
- **放置**：
  - 现在只由 `scope` 与一致性上下文区分 shared 与私有，`Global | Shared(group) | Private(item)` 还没有作为项表示。
  - warp fragment 是值的分布（T7），不是一种放置。
- **剩余**：布局还没有进入 buffer 类型，所以 `layouts_` 仍要在 region 参数间转发（`forward_layouts`）。

### 3.4 T4 分布

`bind(D, region) ≡value region`：一个并行 region 的各次迭代由处理器层次上的不同执行者完成。

- **IR 表示**：
  - 区域体标注 `tensor.bind n`，n 为 1（group）、2（item）或 3（warp）。
  - LowerToMem 把绑定级降为 `gpu.bind.(group, item, warp)` 标注的 `affine.For`，再由 `gpu.outline` 映射到 block/thread 号。
- **合法性 `Independent(region)`**：UniqueOwner 的并行迭代写互不相交的输出，只读不可变输入或本 region 已完成的值。
- **覆盖模式**：分布要带覆盖模式：
  - `UniqueOwner`：输出 tile；
  - `Replicated`：广播；
  - `ReductionPartition`：R8 的 partial；
  - `CooperativeTransfer`：协作填充。
- **消解**：
  - 绑定级的作用域由构造保持：只有不绑定、不提升的整块级才能被消去，只有一个 tile 的绑定级也保留其绑定变量，以免嵌套作用域浮出。
  - `gpu.outline` 拒绝分歧守卫内的 barrier，以及非 warp 对齐守卫内的 warp collective。
- **剩余**：
  - 分布 D 还不是 `Dist` 项，level 链只是它的语法。
  - launch 的 grid/block 大小由 outline 从循环 trip count 求出，而不是从 D 推出。

### 3.5 T5 协作与同步

work-group 内的代码是**整体同步并行（BSP）**的超步序列：

```
group_region = superstep₁ ; B ; superstep₂ ; B ; …
superstep    = forall item ∈ items. (本地计算 ; 对 shared 的写)
```

- **一致值**（不依赖 item 号）在参考语义里只算一次，实现里由 `forall item` 分摊，这就是协作填充。
- **合法性 `DRF(superstep)`**：
  - 同一超步内，不同 item 对同一 shared 位置没有写-写或读-写冲突；
  - barrier 只出现在一致控制流上，同一作用域、同一动态实例被所有参与者到达，并提供所需的顺序与可见性。
- **精化的成立范围**：在这些前提下，且限定在无原子、无异步代理访问、无 warp collective 的子语言里，BSP 程序精化保持每线程程序序与超步边界的顺序程序（`⊑obs`）。不能依赖"warp 内隐式同步"，因为 Volta 之后的独立线程调度不提供它。
- **保证的范围**：静态 shared 面板、结构化控制流、已知分布与已知别名。超出时退为更强的 barrier、同步 copy、取消协作分布，最终退为串行。

**三种同步构造。**
`LowerToMem` 只在下面三个构造里产生 barrier，每个 barrier 的作用域、动态实例和相对位置都由构造固定，不做事后推导。
`B` 指 work-group barrier（`gpu.fence`，或 cooperative 循环首尾由 `gpu.outline` 生成的 `sync_work_items`）。

| 构造 | 超步结构 | 构造给出的义务 | 实现位置 |
|---|---|---|---|
| 填充批 | `B ; (fill₁ ∥ … ∥ fillₖ) ; B ; consume` | 各 fill 写各自分配的面板，互不相交；fill 只读 global 或已发布的值；首个 `B` 结束上一个读这些面板的超步（WAR） | `cooperative()` 的 `staged`/`chained` 标签，`lower_map_reduce`、`lower_mma` 在消费者输入内存上的 `fence` |
| R8 合并轮 | `write partial ; B ; (merge 一轮 ; B)*` | 每轮的配对互不相交且覆盖所有存活 partial；`B` 位于 item 守卫之外 | `lower_bound_fold` |
| 流水环槽 | `wait_group(n − 2) ; B ; compute(slot b) ∥ refill(slot b + n − 1) ; commit` | 读槽前其写组已 wait 且其后有 `B`；重填的槽上一次读它的超步已被同一个 `B` 结束 | `lower_tile_fold` |

单个不成批的一致填充是 `B ; fill ; B`（`cooperative` 标签）；异步填充在批末 `B` 前加 `async_wait`（`async` 标签）。

- **值语义同步**：
  - MimIR 的内存是 SSA 值 `mem.M`，所以 `gpu.fence: mem.M 0 → mem.M 0` 已是挂在内存值上的 value barrier，相当于 IREE 的 `iree_gpu.value_barrier`，不需另设。
  - 填充批末尾的 barrier 挂在消费者自己的输入内存上，随消费者一起存活。
  - 游离在未用内存值上的 fence 会在 `AddMem` 重新串接内存时作为死代码消失，所以 cooperative 循环的末尾 barrier 必须属于循环结构。
- **标签是协议，不是正确性来源**：`staged`/`chained` 只表示"本批末尾的 `B` 由消费者给出"。这一义务由 `fence_pending_` 在同一次输入改写内兑现，离开这次改写时协议结束。
- **合并只做局部**：`gpu.remove_double_syncs` 只删去内存链上紧跟另一个 barrier、或位于 kernel 入口的 barrier。跨构造的消除需要知道 fill 之间相互独立，这一知识只在构造内部可得。

**R8 跨 item 归约**是一个超步序列：私有 fold → 写 partial → `B` → 树形合并若干轮。
它需要两个额外条件：

- **归约律 `ParallelFold`**：至少满足 `fold(seed, X ++ Y) ≈ merge(fold(seed, X), fold(seed, Y))`。`reduction_law` 识别以下 combiner，并要求 seed 是其单位元：
  - wrap 加、乘；
  - bit2 与、或、异或；
  - 整数极值；
  - 带 reassoc 的浮点加、乘；
  - 带 reassoc、nnan、nsz 的浮点 min/max。

  其余 combiner 只能由 `merge_with` 许可，结果对它只承诺 `≈mode`。
- **`ReductionPartition` 的覆盖**：每个 item 遍历 `pair = item + q · trip < pairs`，使每轮所有配对都被处理。早先的实现在 item 数不整除配对数时漏算，就是违反了这一条。

### 3.6 T6 时序：流水与异步拷贝

`pipeline(n, fold) ⊑obs fold`：把 `fold step acc (blocks)` 中每块的 stage 提前 n − 1 块执行，放进 n 个 buffer 的环。

- **IR 表示**：
  - 流水深度是 Group/Item/Warp 链中归约级 `Red` 的字段，用 `*_red_pipelined` 写入；`*_red` 写 0，由 TileFuse 解析为 `-X tensor:pipeline`，默认为 1。
  - 深度大于 1 时，TileFuse 把 step 写成 `tensor.pipeline n (…)`，LowerToMem 只从该标注读深度。
- **参考与实现**：参考是原 `tile_fold`，实现是 `lower_tile_fold` 生成的环，`ring[s]` 存当前块之后第 s 块。这是一个**重定时**：stage(b) 从迭代 b 移到 b − (n − 1)，读它的 compute(b) 不变。
- **异步拷贝的语义**：
  - `store_async` 是一个延迟完成的写，属于当前组；`async_commit` 关闭这一组。
  - `async_wait_group k` 只保证**本线程**除最新 k 组之外都已完成，不为其他线程提供可见性，所以跨线程消费还需要其后的 `B`。
  - NVPTX 只在目标的 compute capability 支持异步拷贝、大小为 4/8/16 字节、且值恰好来自整块 global 读时发射 `cp.async`，否则是同步写。这个开关不是 MMA/ldmatrix 的完整 feature 检查。
- **等价是整程序的**：`store_async` 不能在任意上下文中替换 `store`。可论证的是，满足 `AsyncSafe` 的完整流水程序精化同步 staging 的程序。`AsyncSafe` 由环的构造满足：
  - prologue 恰好提交 n − 1 组；每轮每个线程提交相同数量与顺序的组（含空组），所以 `wait_group (n − 2)` 等到当前块；
  - 读环槽之前，写它的组已 wait 且之后有 `B`；写环槽之前，上一次读它的超步已被同一个 `B` 结束；
  - 同一组内的写目标互不相交；末尾 clamp 后重复填充的槽不再被消费；循环出口的 wait-all 在 shared 生命周期结束前；
  - `n = min(pipeline, 块数)`。它还必须在 PTX 的 wait 立即数与硬件的异步组范围内，这一条目前**没有检查**：`ring.size() − 2` 直接作为立即数发射。
- **保守退路**：没有 stage、块数不是字面量、step 不是原地更新时，不流水。

### 3.7 T7 分布式值：warp fragment 与 tensor core

tensor core 引入了**分布在 lane 上的值**：一个 16×8 的块不在任何一个 lane 里，而是 32 个 lane 各持 4 个元素。

- **类型**：`Frag (L, s, T)`，其中 `L: [lane: Idx 32, reg: Idx k] → «2; Idx s»` 是双射。一个 lane 的局部值是 `«k; T»`，整个 fragment 的语义是"按 L 散布到 s 上的张量"。
- **`mma`**：`Frag(L_A, (16, 8)) × Frag(L_B, (8, 8)) × Frag(L_C, (16, 8)) → Frag(L_C, (16, 8))`，
  参考语义是 `scatter_{L_C}(gather_{L_A} a · gather_{L_B} b + gather_{L_C} c)`。
  它是 `.sync.aligned` 的 collective，warp 内所有未退出的 lane 必须一致执行。
- **`ldmatrix`**：一个集体置换，由三个关系构成：
  - `Addr_LD`：提供地址的 lane → shared 行地址；x4 时 lane 0–7、8–15、16–23、24–31 各提供一个矩阵的 8 行；
  - `Transfer_LD`：(提供者 lane, 加载的元素) → (接收者 lane, 寄存器)；
  - 面板行的选择。

  它们复合后必须等于 `L_A`。
- **IR 表示**：
  - `tensor.mma_m16n8k8_tf32` 是一组 Mim 定义：`l_a`、`l_b`、`l_c`、`addr_ld`、`transfer_ld`、`panel_a_addr`、它们的组合 `ld_a`，以及参考实现 `mma_ref`。
  - `lower_mma` 与 `store_fragments` 的地址由这些定义部分求值得到。
  - 面板布局由 `panel::mma` 选出：源的连续维放在最内，并 pad 以消除 ldmatrix 的银行冲突。
- **消解**：
  - `lit/tensor/mma_reference.mim` 穷举检查三个布局是双射，以及 `ld_a = l_a`；
  - 它还在 CPU 上把 `mma_ref` 与普通矩阵乘逐元素比对。`mma_ref` 用普通 F32 乘加，不模拟 TF32 转换，所以这验证的是布局和乘加参考，不是完整的 TF32 数值语义；
  - 布局表本身来自 PTX ISA，属于可信边界。
- **数值**：TF32 是许可（`≈mode`）：`warp_mma` 的语义是"操作数先转成 TF32 再做 map_reduce"。转换有两种实现：
  - 默认由 `mma` 截断 F32 的低 13 位；
  - `-X tensor:tf32-rna=1` 时，fragment 载入后显式调用 `gpu.tf32_rna`（`cvt.rna.tf32.f32`，就近舍入、平局远离零）。它的位级参考 `gpu.tf32_rna_ref` 按 sm_80 上实测的 Inf/NaN 行为定义。
- **剩余**：warp 累加器仍是每个 lane 的 SSA 值 `«F; «4; F32»»` 加旁表 `fragments_`，还不是 `Frag` 类型本身。

## 4. 组合：level 链

一个 `tile_req` 的 level 链按顺序给出各分面的参数。
链是互递归的 nominal 变体 `TileLevels / GroupLevels / ItemLevels / WarpLevels`，按累加器类型 `Acc` 索引；类型保证各级只出现在提供它的作用域里。

| 构造子 | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---|---|---|---|---|---|
| `serial_group (sizes, tail, promote, …)` | 网格 | 融合 members | promote → group 级 stage | group | 开始一个 BSP 程序 | | |
| `group_item` / `group_warp (sizes, tail, …)` | 子网格 | | | item / warp | | | |
| `group_red` / `item_red` / `warp_red (sizes, tail, promote, …)` | 有序块 | | 面板 stage 与 `placed L` | | 填充批 | 深度取默认 | |
| `*_red_pipelined (…, n, …)` | 有序块 | | 同上 | | 填充批 | `pipeline n` | |
| `group_red_items` / `group_red_merge (…)` | 拆分归约 | | partials 在 shared | ReductionPartition | R8 合并轮 | | |
| `warp_mma (16, 8, 8)` | 叶子 | | mma 面板布局 | lane | | | fragment 与 mma |

TileFuse 解释这条链：它在 C++ 中做各级的合法性判定，并把选出的布局 L 和流水深度 n 写回 IR。
各构造子还没有展开为 Mim 中各分面组合子的复合，像论文的 `dot_schedule` 那样；合法性谓词也还没有全部以 `refly.check` 出现在 IR 里。

## 5. 可信计算基

| 分面 | 是项 | 公理（实现交给后端） | 留在 C++ 的可信部分 |
|---|---|---|---|
| T1 | 网格、box、level 链 | `tiled`、`tile_fold` | 动态形状、cap/peel 下的覆盖；OrderedChunk 判定 |
| T2 | 请求、参考切片、部分 builder | `tile_of_box` | 一般 producer 的重写（`producer_tile`）；preimage/image 计算；resident 视图的包含关系 |
| T3 | `stage`、`placed L`、`layout_expr` 及其解释器 | — | 银行冲突评分只影响选择；`Fill::contiguous`（只查前 64 个填充位置）决定向量写是否合法，与正确性相关；`placed` 的布局与实际形状的对应 |
| T4 | `tensor.bind n` | `gpu.bind.*`、`gpu.outline` | launch 形状推导；作用域与分歧检查 |
| T5 | 挂在内存值上的 `gpu.fence` | `sync_work_items` | 三种同步构造的生成 |
| T6 | 流水深度 | `store_async`、`async_commit`、`async_wait_group` | 环的生成 |
| T7 | fragment 布局、ldmatrix 关系、`mma_ref`、`tf32_rna_ref` | `mma_tf32`、`ldmatrix`、`tf32_rna` | 算子匹配（`Levels::mma` 不核实 combiner 是乘加，lowering 固定生成 `mma_tf32`）；PTX 的 fragment 表、intrinsic ABI、对齐与收敛前提 |

证明义务的形状只在纯函数部分与论文一致：T1–T3 只需证明一次"实现 ≡value 参考（在 P 下）"，再加 rewriter 与部分求值的正确性。
T5–T7 还涉及 GPU 内存模型、并发、shared 生命周期、数值近似与 collective 的未定义行为。

## 6. 验证方法与证据

下列证据针对"实现符合参考"和"重构保持行为"，不是机械化证明：

- **有限域穷举**：
  - `lit/tensor/layout_expr.mim` 对 perm、split、pad、旋转与异或各给一个布局，检查所有逻辑坐标的像互不相同且在物理形状之内；
  - `lit/error/layout_expr_*.mim` 检查非法置换、非 2 的幂块数的异或都会被拒绝；
  - `lit/tensor/mma_reference.mim` 检查 fragment 布局与 ldmatrix 组合。
- **参考执行**：`mma_ref` 在 CPU 上与矩阵乘逐元素比对；`tf32_rna_ref` 的边界值取自 A100 实测，覆盖平局、溢出与 NaN。
- **GPU 执行**：`lit/nvptx/` 的 tile、mma、pipeline、swizzle 用例在 A100 上运行并与 CPU 参考比较；`bench/gpu_tiling` 的 kbench 对 64 个采样点与 FP64 参考比较。
- **重构差分**：布局选择从 LowerToMem 移到 TileFuse 时，在 33 个 GPU 配置上把生成的设备 IR 与改动前逐一比较。
  - 这些配置包括全部 `lit/nvptx/tile_*`、`mma*` 与 `bench/gpu_tiling` 基准；
  - 迁移期间两边同时选择，82 个面板的选择内容全部一致。
- **已知缺口**：
  - Release 构建关闭断言和调试日志，部分 lit 用例的 LOG 检查只在 Debug 下有效。
  - Debug 下还有一批 tensor/GPU 用例停在一个与本文无关的核心断言上：phase 期间推迟的 universe 层级约束无人求解。

## 7. 未完成与开放问题

未完成：

- **T3**：布局进入 buffer 类型；放置（所有者与生命周期）成为项。
- **T4**：`Dist` 项及其覆盖模式；launch 形状从 D 推出。
- **T7**：`Frag` 成为累加器的类型。
- **链的展开**：level 链展开为 Mim 中各分面组合子的复合，合法性谓词随展开出现在 IR 里。
- **不在本文范围内**：
  - **T0**：多 kernel 的划分、数据常驻与 kernel 间同步；
  - **T8**：寄存器、shared 与 block 大小的资源可行性；
  - 跨 group 的 split-K；warp shuffle 归约；persistent kernel、cluster、TMA。

开放问题：

- `Frag` 与面板的布局 L 是否需要一般的线性布局代数（Triton 的 linear layouts、CuTe 的 layout 代数），还是 PTX 的固定布局加当前的 `layout_expr` 足够。
- DRF 检查只在访问函数、别名、调用、动态下标与控制条件都受限时可判定，仅有"结构化控制流"不够。
- 所有者与生命周期用 nominal 参数表示是否足够，还是需要 effect/ownership 检查；后者在没有用户证明的前提下只能是编译器内的自动检查。
- 原语"实现 ~ 参考"的一次性论证是否要机械化，例如在 Coq 中对核心演算形式化；以及怎样让 C++ 中的判定过程（preimage、OrderedChunk）本身可信。
- 动态形状下 ExactCover 与 Injective 依赖 `core.nat` 的归一化强度，这与论文 §6 是同一前提。
- T0 与 T8 如何进入同一框架。

## 8. 相关工作

- **IREE**：GPU `LoweringConfigAttr` 的 `TilingLevel`（Workgroup、Reduction、PartialReduction、Serial、Thread、Subgroup、Lane）在结构化 op 上驱动多级 tile-and-fuse，并由 vector distribution 把内层 tile 分发到 subgroup/lane。
  - `MMAIntrinsicAttr` 与 `MMASingleSubgroupLayout` 描述 MMA 及每个 lane 的分布；
  - `iree_gpu.value_barrier` 为 tensor/vector 提供值语义同步；
  - 它与 MimIR 的 level 链起点相同，因此本文以它为主基准。
- **Halide / TVM**：`gpu_blocks`/`gpu_threads`、`compute_at`、`store_in(GPUShared)`、`cache_read(scope)`；调度是 DSL，合法性靠边界推断，不是 IR 中的项。
- **TensorIR（TVM）**：block 的 iter var 绑定与作用域验证，最接近 T4/T5 的检查器。
- **Graphene（ASPLOS'23）**：把 GPU 张量计算建模为按线程层次分解的张量与布局，与 T3/T7 最接近。
- **CuTe / Triton `LinearLayout`**：布局的表示、组合、求逆、XOR swizzle 与 MMA fragment 双射，可作为 L 的一般形式；Triton 的 block 级 tile 编程模型不是本文的主基准。
- **Triton 的 MMA/ldmatrix lowering 与 dot `inputPrecision`**：T7 指令映射和显式 TF32 语义的参照。
- **Fireiron**：把数据移动作为一等公民，对应 T3 的 stage 与 T6 的异步拷贝。
