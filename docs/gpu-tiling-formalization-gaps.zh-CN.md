# GPU 多级 tiling 的形式化空白与消解方案（2026-11-11，按 codex 与 agy 两轮评审修订）

本文列出分支 `feat/gpu-tiling`（`4642222e88`）上仍由 C++ 决定或相信的东西，并给出把它们搬进 MimIR 的办法。
目标有两个，按优先级：

1. **减少 C++**：能写成 Mim 函数加部分求值的，不留在 C++ 里。
2. **至少在 IR 里反映出来**：搬不动的，C++ 作出的决策和它依赖的前提要以项的形式留在 IR 里。

模型见 `gpu-tiling-formalization.zh-CN.md`。两轮评审见 `gpu-tiling-formalization-gaps-review.zh-CN.md`（codex）和 `gpu-tiling-formalization-gaps-review-agy.zh-CN.md`（agy），本文已按它们修订。
来自评审而没有被复现的说法标了"（评审）"；复现过的标"（已复现）"。两轮评审都报告了现存的实现缺陷，汇总在 §2.5。

## 0. 四种"看不见"

初稿把四件事混在一起，这里分开，后面每条空白都标出属于哪种：

| 记号 | 含义 |
|---|---|
| **无表示** | 请求阶段和 lowering 之后的 IR 里都没有对应物 |
| **晚可见** | TileFuse 输出里没有，LowerToMem 之后才以循环、内存操作、同步的形式出现 |
| **非类型** | IR 里有值，但类型不约束它（恒等标注、裸 `Nat`） |
| **未检查** | IR 里有表示，但其前提只由生成它的 C++ 保证，没有独立检查 |

## 1. 三类义务，以及断言能做什么

假设已有运行时断言 `assert (cond, value, msg)`，与 `refly.check` 同形，`cond` 不是字面量时留在 IR 里。

断言**不能**把 C++ 分析普遍降级为优化。它只在两个条件同时成立时减少对某项分析的信任：检查的是完整前提，而且检查在内存顺序上先于、并支配依赖它的操作。
把分析结果包成 `assert`，或者只检查必要不充分的条件，都不算。
另外，`refly.check` 和 `assert` 都只返回原值，不产生定义相等，消解不了依赖类型要求的等式。

所以义务分三类，各有各的承担者：

1. **局部数值条件**（范围、整除、对齐、容量）：生成有序的检查。静态可判定时消失，否则是运行时检查。
   检查必须放对位置：纯值性质（对齐、非负的减法）的断言要在数据流上**包住被消费的值**，并且在会截断的算子之前。
   例如 `mem.align` 是不带内存 token 的纯值，把断言挂在内存链上拦不住它被提前求值，要写成 `mem.align (assert (p % 16 == 0, p, …), 16)`；
   `Nat` 减法先截断成 0 再检查包含关系，检查恒真，所以要在相减之前比较两个起点。
2. **有限结构性质**（映射互逆、覆盖、布局单射、调度合法）：由独立的检查器在具体实例上验证，或写成 Mim 函数后用测试穷举。
   "两个映射写在同一个定义里"不等于互逆，没有证明项时互逆只能靠在有效域上检查。
3. **并发与代数性质**（全组一致到达、无数据竞争、槽位的无别名复用、归约律）：不能由断言承担，也不能只靠 nominal 类型：MimIR 没有线性类型和发散效应，一个 group 令牌照样可以被捕获进 item 相关的分支。
   它们由静态的控制流与生命周期检查器、受限组合子的语义论证，或明确的用户许可承担。

独立检查器、原语的实现和后端始终在可信边界之内。本文不承诺"分析算错只会 trap"。

## 2. 空白清单

### G1 box 的合法性与覆盖 —— 未检查

- 输出网格来自输出形状和 tile sizes：cap 模式是 `⌈S/t⌉` 个 tile，peel 模式是 `⌊S/t⌋` 个整块加尾区域（多输出路径 `parallel_multi` 生成 `tiled_n`，单输出路径由 `lower_tiled` 按维递归）。输入 box 的 `off/ext/cap` 由 `producer_tile` 从读映射的像算出。
- 类型已保证：body 返回 `Tensor r ext T`。
- 未检查：逐维的 `off ≤ shape` 与 `ext ≤ shape − off`、`ext ≤ cap`、各 tile 恰好覆盖输出一次、box 包含 body 会读的全部元素。
- body 内迭代域与 `ext` 的关系只在写映射是 canonical 的情形下是简单相等；一般情形要检查的是输出映射。

**空 tile 不再由每个 box 防御**（已实施）。`affine.image/cover` 原来在 extent 可能为 0 时把结果的每个分量包进 `(x, 0)#empty`，于是偏移和 extent 都成了选择项，逐级累积。
现在按 `affine.mim` 里本来就写着的约定处理："box 的 extent 为正"，只有字面量为 0 的 box 才返回空区域。非空由 region 一层保证：`lower_tiled` 对 extent 为 0 的 tile 走跳过分支，不进 body。
同时给 `affine.op.add/sub` 加了归一化（加减同一项相消、加减 `lit 0` 消去）。四级 1024³ GEMM 在 TileFuse 之后的 IR 从 113 行降到 64 行，面板的读映射就是 `(k, (i + ioff) / 8, (i + ioff) % 8)`。
验证：全量 lit 与单元测试不变，GEMM 0.737 ms、常驻 softmax 0.613 ms。
两点范围说明：绑定级本来就要求字面量 extent，所以"残缺 group 里的空 item tile"在 GPU 路径上不会出现（会直接报错）；CPU 的 cap 路径由 `tile_fuse_levels_exec.mim` 覆盖。`tile_ends_` 旁表里把"extent 等于 0"改写成假的那一半已删除；折叠钳位 `min(off + ext, 形状)` 的那一半还需要，去掉会让 12 个 GPU tile 测试失败。cap 模式的尾 tile 靠这个钳位得到有效 extent，所以它不是冗余防御；要删掉它得先在 IR 里把请求的 extent 和有效的 extent 分开（G1、G2）。

### G2 两级偏移的包含 —— 未检查

- `compose` 把"tile 的 tile"化成原 producer 在复合偏移处的 box；`resident_` 把读换成共享 tile 的视图，偏移是两个起点相减。
- `Nat` 减法在负值时截断成 0，不报错。所以缺的是：负偏移会被静默截成 0，没有包含关系检查。
- 包含关系要以父 tile 的**有效范围**为准，不是容量，并且要在合并视图、丢掉父边界之前检查。

### G3 读映射与 stage 的关系 —— 未检查

- `promote_input` 同时生成 `stage (pos, tile, perm)` 和读映射（按参数 `tile` 做 floordiv/rem）。
- 两层要分开：`stage` 是逻辑重索引，改变秩和形状；`placed L` 是物理布局，类型上是恒等。
- 未检查：在有效源坐标上，读映射与 stage 的重索引互逆。padding 区域不要求。

### G4 有序归约分块 —— 部分未检查

- 整块、尾块、post 已由 Mim 的 `tf_reduce/tf_tail/tf_chunk` 描述，不是纯 C++ 算法。
- 未检查：cap 与 peel 两种模式各自的精确分割关系。注意 peel 下不是 `count × stride ≥ K`：K = 10、stride = 4 时整块是 2 个，余下 2 由尾块处理。
- 归约律的识别（`reduction_law`）在 C++，结果不留痕迹；seed 单位元的检查只针对自动 R8 路径。

### G5 R8 —— 晚可见，前提未检查

两条路径要分开：

- **自动拆分**（`split_over_items`）：把融合进来的归约 producer 拆到 item，条件是单归约循环且 K 被 item 数整除。
- **R8 降级**（`lower_bound_fold`）：私有 fold、shared partials、合并、发布。item 数是 2 的幂时用树，否则串行合并；2 的幂不是合法性条件。

TileFuse 输出里只有 `bind 2` 和 `merge_with` 两个标注；lowering 之后算法以显式循环和 fence 出现。每轮 pair 的分配、partials 的原地复用所需的轮间顺序没有独立检查。

整除不只是自动拆分的条件：根级 `RedItems` 复用 `tf_reduce`，peel 且 K 不被块长整除时会留下一个不带 `bind 2` 的尾块（§2.5 D5）。两轮评审在这点上意见相反，以 agy 给出的代码路径为准，待复现。

### G6 放置 —— 请求阶段无表示

- LowerToMem 用 `uniform(value)`（是否依赖 item 原点）推断 group 共用的值，分配进 shared。
- lowering 之后有 `shared_scope/slot_shared` 和地址空间类型，所以不是"完全没有"。
- 缺的是高层的放置意图、所有者（哪个 group、哪个 item）和生命周期的稳定契约；`stage` 的 `scope: Nat` 恒为 0。

### G7 同步 —— 未检查

- 三种结构化构造由 LowerToMem 插入 `gpu.fence`。
- `outline` 的发散检查只识别它自己为较短 item 循环生成的守卫，不是一般控制流的 uniformity 检查。
- `fence: M → M` 不要求 group 作用域，不表达全组一致到达。

### G8 执行映射 —— 非类型

- `bind 1/2/3` 是恒等标注。item 数是所有绑定循环 trip count 的最大值，warp 循环乘 32。
- 行主序的反线性化在 LowerToMem。launch 之后 grid/block 是显式参数。
- 缺的是 region 与 grid/block 的联系。注意逻辑 item 循环的 trip count 本来就可以不同（R8 每轮在变短），不能要求它们相等；要分开的是物理 block、逻辑工作量和 active 谓词。

### G9 流水与异步 —— 晚可见，前提未检查

- `pipeline n` 是恒等标注；lowering 之后 commit/wait/fence 是项。
- 深度只取 `min(pipeline, 块数)`，没有限制到目标支持的 wait 立即数范围，`ring.size() − 2` 直接作为立即数发射（评审）。
- 还依赖（评审）：stage 的发现与各块实例的对应稳定；面板形状不依赖 step 参数；环槽的物理布局随参数传递；尾部 clamp 重填不改变可观察的值；异步源在传输期间有效且不变。
- 每轮的顺序是 `wait_group (n − 2)`、fence、重填最远的槽并 commit、**然后**才 compute。先发起异步拷贝再计算，重叠才成立；内存链是线性的，没有"并行"算子（评审）。
- `wait_group` 只保证将要读的槽写完了；将要重填的槽上一轮的读是否结束，靠的是同一个 fence。出口的 wait 之后没有 fence（评审）。

### G10 tensor core —— 三个独立的空白

1. **算子匹配**（未检查）：`Levels::mma` 检查了作用域、秩、输入数、F32、指令形状和部分整除，但没有核实 combiner 是乘加，也没有完整核实访问关系是矩阵乘。lowering 固定生成 `gpu.mma_tf32`，不看原 combiner。已复现（§2.5 D1）。输入轴的绑定也没有检查（D2），K 不超过块长时 `mma` 被丢弃（D3）。
2. **fragment 表示**（非类型）：warp 累加器是每个 lane 的 SSA 值加旁表 `fragments_`。还要求累加器是字面量二维形状、scalar seed、可原地更新（评审）。
3. **传输与 ISA 前提**（未检查，评审）：A 面板要支持四元素向量寻址；ldmatrix 的指令数、对齐、目标支持；`mma_ref` 用普通 F32 乘加，没有模拟 TF32 截断，所以现有 CPU 测试验证的是布局和普通乘加，不是完整的 TF32 数值语义。

### G11 level 链残留的 C++ 检查 —— 非类型

- sizes 是字面量。
- 链的 `n` 与 `Acc` 已由类型给出：`tile_req` 的 levels 类型是 `TileLevels (root_loops op, root_acc op)`，两个公理在 `op` 归一化为 `map_reduce_post` 时得到它的循环数和累加器类型，链的循环数或累加器类型不符是类型错误（`lit/error/tile_req_{loops,acc}.mim`）。`op` 不是 `map_reduce_post` 时两者卡住，请求无法构造。
- 输入个数 `nis` 还没有这样的公理，所以 `promote` 位掩码的范围仍是 C++ 检查。
- 每个归约级只切一个归约维（不是"root 只能有一个归约维"）。
- 向量级必须最内层；在一个并行维上标 1；另外还支持最后一个归约维的拆分因子。
- `promote` 位掩码不超过输入个数；tail 是 0 或 1。

### G12 LowerToMem 承载正确性的旁表 —— 无表示

当前的表见 `lower_to_mem.h`；初稿列的 `origin_dims_`、`item_grid_` 已不存在，相关信息在 TileFuse 的 `PanelSite` 里。
要区分两种：纯缓存可以不公开；但 `storage_shapes_`（容量）、`multiples_`（整除与对齐）、`linear_step` 的结论承载的事实与正确性相关，可重算不等于无需依据。

### G13 初稿漏掉的前提（评审）

- **纯性、共享与存储复用**：`RealizationRequests::pure` 用公理名白名单加闭包分析；`sequential_tile` 决定 scratch 的复用资格。
- **生命周期**：`fresh_memo_` 按 lambda 限定以避免悬空变量，动态 buffer 由 tile body 释放。这不是性能缓存。
- **非空事实**：`tile_ends_` 被用来把空 extent 的判断直接改成假。
- **表示能力限制**：buffer 的内维必须静态；绑定级要求字面量 extent，不能 peel 非整除的尾块。这些情况是直接报错，没有串行退路。
- **有限整数**：形式化里的自然数公式不能直接当成 `Nat/u64` 上的定理，尺寸乘积可能溢出。
- **`panel_layout.cpp` 里与正确性相关的部分**：`Fill::contiguous` 只检查前 `min(total, 64)` 个填充位置，结果决定向量写是否合法。
  布局与形状的关系已进类型：`Expr (r, s)` 以单例字段 `logical: «s»` 携带它的逻辑形状，所以 `s` 不同的两个 `Expr` 是不同的类型（只作类型参数的索引不会区分 nom 实例）；`placed` 携带容量 `cap` 与 `Expr (r, cap)`，LowerToMem 核对 `cap` 等于它算出的存储形状。容量与张量逻辑形状 `s` 的关系（`s ≤ cap`）仍未检查。
  agy 给了两个"前 64 个位置通过、第 65 个失败"的布局（奇数 pad 的 `(2, 256)`；`shift = 6` 的 swizzle），并指出它检查的是一维线性偏移连续，而 `lower_generate` 是对存储的**最后一维**除以向量宽度（评审）。改法不是找周期性论证，而是按布局结构做 O(1) 判定。
- **NVPTX 发射侧**：`cp.async` 只在目标支持、大小为 4/8/16 字节、值恰好来自整块 global 读时发射，目标地址直接 cast 成 shared；mma 固定按 4/2/4 寄存器的 ABI 发射；shared 超过 48 KiB 后改用动态 shared，这些决策在 emitter 里。

### G14 `gpu.outline` 的宿主–设备假设（评审）—— 未检查

- kernel 结束后对闭包捕获的所有宿主指针无条件拷回宿主，只读输入也拷；两个捕获指针别名同一数组时互相覆盖。
- trip count 按 `(end − begin + step − 1) / step` 算，循环变量却直接映射成 `0 … count − 1`，隐含假设绑定循环 `begin = 0`、`step = 1`。
- `bind.group` 循环除内存 token 外的循环携带值被丢弃。

### G15 NVPTX 发射侧的假设（评审）—— 未检查

- `store_async` 不满足 `cp.async` 条件时静默发射同步 `store`（见 §2.5 D4）。
- `ldmatrix` 把指针直接 cast 成 shared，不验证 16 字节行地址对齐和 `n ∈ {1, 2, 4}`。
- `slot_shared` 按整个 kernel 静态累加、逐槽按 16 字节取整，超过 48 KiB 改用动态 shared，不考虑循环内复用和目标的 shared 上限。

### G16 universe 层级约束的求解（评审）—— 未检查

- `RWPhase` 结束、交换 world 之前求解它推迟的层级约束（`b4dd6a16a`）。
- 仍未覆盖（评审）：直接继承 `Phase` 原地改 world 的 pass、`GraphRewrite::publish`、`Driver::move_to_library` 的导入、`SplitOffKernels` 的辅助 `RWPhase`。可改为在 `Phase::run` 的 `start()` 之后统一求解。
- 求解器本身（评审）：右侧不是字面量的 `LMax` 约束只被推迟、不做合一，`solve_levels` 把未绑定的洞置 0 后比较，`max(?a, 0) = 1 + ?b` 这类有唯一最小解的约束会被报成没有唯一解。

## 2.5 两轮评审报告的实现缺陷

这些不是形式化空白，是评审认为现存的 bug。它们恰好都落在"C++ 相信而没有检查"的地方，所以列在这里；修它们先于后面所有的重构。

| # | 缺陷 | 状态 |
|---|---|---|
| D1 | 非乘加的 combiner 走 `warp_red` 接 `warp_mma` 时被编译成 `gpu.mma_tf32`：顺序归约切最外层归约维时不查 `reduction_law`，`Levels::mma` 和 `lower_mma` 都不看 combiner | **已修**：`tile_req` 的 levels 带第三个索引 `root_contraction op`，`warp_mma` 把它固定为 `tt`；只有 F32 且 combiner 返回 `acc + a · b` 时归一化为 `tt`。`lit/error/tile_mma_combiner.mim` |
| D2 | mma 的输入轴没有和 (m, k)、(k, n) 绑定：`panel_axis` 只要求某根轴读任意一个被切的并行维，`lower_mma` 却硬编码两个输入的坐标 | **已修**：同一个 `root_contraction` 要求输出映射为 (m, n)、A 为 (m, k)、B 为 (k, n)。`lit/error/tile_mma_axes.mim` |
| D3 | K 不超过归约块长时（如 K = 8、块长 8）TileFuse 不生成 `tile_fold`，`fragments_` 没登记，`mma` 被静默丢弃，退回的标量循环被 warp 的 32 个 lane 重复执行 | 评审，未复现 |
| D4 | `store_async` 静默退化成同步 `store`，而 `bind.async` 循环前后没有 barrier，环槽的读写隔离被破坏；带越界分支的面板、非 4/8/16 字节的元素都会触发 | 评审，未复现 |
| D5 | 根级 `group_red_items` 配非整除的 peel：尾块 `last` 没有 `bind 2`，在 group 作用域里被所有线程并发重复累加；带非恒等 post 时同理 | 评审，未复现 |
| D6 | `Fill::contiguous` 的两个问题（见 G13） | 评审，未复现 |
| D7 | `outline.cpp` 的发散检查对 `chained`/`async` 循环豁免（条件里的 `&& !chained`），但这些循环仍按全组线程数分摊；落进较短 item 守卫时面板填不全 | 评审，未复现 |
| D8 | 流水深度没有限制到 wait 立即数范围 | 两轮评审都指出，未复现 |
| D9 | 外层 cap 模式下的 work-item 归约是 bounded 的，`lower_into` 回退到协作拷贝，却按运行时外延而不是源的存储上界折叠读下标，类型不合 | **已修**：拷贝按 `storage_shapes_` 中的源存储形状读；`lit/{gpu,nvptx}/tile_reduce_cap.mim` 在 CPU 与 A100 上执行 |
| D10 | 剥离尾块的面板：`stage_block` 把字面量余数 `k` 写进 `PanelSite.caps`，代换时把所有等于它的维度改成块长；单个 work-group 时只有第 0 维的偏移依赖 group 变量，沿其他维切片的尾块面板被提升到 group 循环之外，`gpu.outline` 报错 | **已修**：字面量的 `k` 不作代换键；一个 tile 的绑定级每一维偏移都依赖绑定变量；`lit/{gpu,nvptx}/tile_panel_tail.mim`（32×24×32）在 CPU 与 A100 上执行 |
| D11 | `layout_expr.make` 不要求 `split = 1` 时单位维在第 2 位（`panel::decode` 按位置跳过它，否则越界读），也不检查 swizzle 的 `shift < 64`（`2^shift` 溢出为 0 后除零） | **已修**：两条 `refly.check`，`lit/error/layout_expr_{unit,shift}.mim` |

## 3. 消解办法

### A. 写成 Mim 函数

要区分两件事：**纯算法的迁移**（下标、映射、计划）现在就能做；**GPU 实现的迁移**（shared、发布、复用、异步）要先有执行接口，部分求值不会自动补出这些语义。

**A0 搬样板代码（已完成）**。agy 指出初稿漏了两块不涉及图分析、只是用 C++ API 手工拼 Mim 项的代码，并估计可删 600–800 行。实施后的实际情况：

| 原 C++ | 现在的 Mim 定义 | 说明 |
|---|---|---|
| `lower_stage` 里手工 `mut_lam` 的 `getVector` | `tensor.mim` 的 `get_vector` | 向量宽度、连续维的判定留在 C++ |
| `lower_map_reduce` 里的 `memComb`、`memPost` | `btensor.mim` 的 `lift_mem` | 两处合成一个包装器 |
| `tileWriteMap`、`tilePostMap` | `btensor.mim` 的 `tile_write_map`、`tile_post_maps` | |
| `producer_tile` 里重建 `map_reduce_post` | `tensor.mim` 的 `tile_rebuild`（返回 `full` 与 `cut_reduction` 两个闭包）和 `tile_init` | preimage/image、cap、合法性判定留在 C++ |
| `promote_input` 里构造读映射、替换输入、重建 op | `tensor.mim` 的 `stage_input` | group slice、面板与布局的选择留在 C++ |

- **实测净减 66 行 C++**（C++ 减 124 加 58，Mim 加 100），是估计的十分之一。
- **没搬的三个候选**：`tileCopy`、`copy_buffer`/`copy_box`、`lower_splat`。它们依赖 lowering 的状态（协作标注的回调、scratch 分配与 transient 登记、逻辑 extent 与物理存储形状的区别），而且已经共用 C++ 的 `build_pointwise`；只搬外层要另造一套循环协议。结论是：与状态无关的纯样板比评审以为的少，剩下的要等放置和同步有了 IR 表示（B1、B2）才搬得动。
- **`stage_input` 的一个脆弱点**：它要在"第 slot 个输入换成 staged 面板"的依赖类型元组上做选择，用 `core.bitcast` 把两边统一到同一类型。slot 是字面量时这些 bitcast 归一化掉（四级 GEMM 的输出里出现 0 次）；slot 不是字面量时会留下对张量和函数的 bitcast。现在 C++ 只传字面量。
- **验证**：Debug 全量 lit 仍只有两个既存失败，260 个单元测试通过，1024³ GEMM 0.767 ms 持平；四级 GEMM 在 TileFuse 之后的 IR 与重构前逐行对比，只差绑定名。

A4 只动约 60 行，B 类的每个新公理还会在 LowerToMem、outline、nvptx 三处增加 C++；它们的价值是缩小可信边界，不是减少行数。

| 方案 | 现在能做的 | 卡点 |
|---|---|---|
| **A3** 面板的 encode/decode、shape、padding、有效域由一个受限构造子派生 | 全部。`layout_expr.IndexOps` 已经是"一个解释器同时服务 affine 和整数"的先例 | 两个字段来自同一函数不自动互逆，要对有效坐标做 round-trip 检查（第 2 类义务）。`lower_stage` 靠识别 `stage (slice …)` 才得到连续维、向量宽度和无越界分支的读；不能把 `stage` 展开成普通的 `generate`，要保留"带向量宽度的源切片搬运"这个节点（评审） |
| **A5** 网格定义由参考语义和 lowering 共用 | `tiled_impl` 已有 Mim 公式 | 不要返回完整的 `«count; Box»`，那会在编译期大规模展开；返回计数和按需的 `box_at`，让循环留着 |
| **A4** level 链的残留检查进类型 | `Vec` 去掉 `rest`；`tail` 改成变体；`promote` 改成按 `nis` 索引的 `Bool` 数组 | `Idx n` 要收紧到并行维并保留归约拆分因子；类型不能保证一个 `Nat` 可归一化成字面量；root descriptor 要由受控构造子和 root 一起封装，否则只是并列的两个值。descriptor 的设计必须先于 A4，B6 的 contraction 描述也依赖它 |
| **A1** R8 | 树归并的计划、pair 分配、非 2 的幂的退路写成 Mim，在 CPU 上测试 | shared 的生成、发布、复用没有可用契约；step 的签名要绑定 block、stride、输入和线程分配。`lower_bound_fold` 还有两条特化路径：标量累加器走寄存器，最后一步直接写进 destination；值层的纯 Tensor 写法会丢掉它们（评审）。删除 `lower_bound_fold` 是后续的验收项，不是第一步 |
| **A2** 流水 | 环的旋转、prologue、clamp、提交序列 | 不能只包装 `tile_fold`：先要让 step 显式分成 `prepare (block)` 和 `compute (acc, ready_panels)`，否则只是把隐式的 C++ 分析藏进新公理 |

A1、A2 的纯计划部分已完成：`tensor.mim` 的 `r8_plan`（`is_tree`、`rounds`、`pair_coords`）与 `pipeline_plan`（`depth`、`ring_block`、`refill_block`、`rotated_source`）由 LowerToMem 经部分求值消费，C++ 行数不变；`lit/tensor/gpu_plans.mim` 在编译期对多组字面量参数检查整份 R8 计划的覆盖、互斥、轮次衔接与结果位置，以及流水每块恰好填一次、读前就绪、clamp 与重填槽。上表的卡点（shared 的契约、prepare/compute 的分离）仍在。

A1、A2 的 Mim 展开不能在 TileFuse 阶段写死成 GPU 公理：同一个 `tile_req` 在不加载 gpu 插件时要退化成 CPU 循环（`lit/gpu/tile_reduce.mim` 的 `-p ll` 那一行）。展开要在 LowerToMem 或 gpu 专用阶段按目标注入。

### B. 类型与组合子

类型能检查同一个描述符的字段是否一致。它不能证明两个函数互逆、任意 merge 结合、任意控制流收敛。

| 方案 | 修正后的提法 |
|---|---|
| **B1** 放置 | 初稿的 `placed_in: … → T → T` 仍是恒等标注，只传达意图。要让跨 owner 的读成为类型错误，得是 `Placed (p, T)` 或带 place 的 buffer 类型，并约束所有访问操作、group/item 身份的来源和逃逸。`uniform` 检查不能删：别人也可能给依赖 item 的值贴上 Shared |
| **B2** 同步 | 改为 §3 B′ 的相位类型方案：内存 token 按 (group, 代数, 相位) 索引，shared buffer 带可见性，发散靠三层 token，剩下的由 `gpu.check_sync` 的四条规则检查 |
| **B3** 执行形状 | region 按物理 block 形状索引；逻辑工作量和 active 谓词是另外两个量；barrier 位于统一的物理 group 层 |
| **B4** `Frag` | 能避免把普通四元素数组当 fragment。删旁表还要求 fragment 数、tile 形状、布局随 fold 状态和 region 参数传播，并区分 A/B/C 角色与数值格式。TileFuse 阶段的 fold 状态是 `Tensor`（`tile_of_partial/post` 的签名要求），在那里换成 `Frag` 会让整套 `tf_reduce` 类型不合；留到 LowerToMem 再换则消不掉旁表（评审） |
| **B5** 归约律 | `[merge, identity, commutative]` 是声明不是证明。用受限的"已识别律"构造子加独立匹配；用户自己构造的明确叫 permission |
| **B6** MMA 算子匹配（新增） | 已完成：`root_contraction` 公理在 root 归一化为 `map_reduce_post` 时判定它是否为 F32 矩阵乘（访问关系与乘加 combiner，含融合后经局部 continuation 返回的 combiner）；level 链的类型以它为索引，`Mma` 变体携带单例证据 `‹tt›: «mma»`，所以只有矩阵乘的请求能以 `warp_mma` 结尾。索引必须出现在字段类型里才区分 nom 实例，只在递归引用里出现的索引会被忽略 |

### B′. 同步与相位类型（方案，2026-11-11）

**相位进内存 token 的类型。** group 级内存 token 是

```mim
nom GM (g e: Nat, p: Phase) = [witness: «(g, e, p)», raw: mem.M 0];
axm enter_write:  GM (g, e, Synced)  → GM (g, e, Writing);      // 不发 barrier
axm sync_publish: GM (g, e, Writing) → GM (g, e, Reading);      // 一道 barrier
axm sync_reclaim: GM (g, e, Reading) → GM (g, e + 1, Synced);   // 一道 barrier，代数加一
```

`g` 是 group 标识，`e` 是超步代数，`p` 是相位。单例字段让不同 `(g, e, p)` 的 token 互不相容（索引必须出现在字段类型里，见 B6）；后端由擦除 pass 退化回 `mem.M 0`。没有线性类型，但代数 `e` 让大部分"在错误的超步复用 token 或 buffer"成为类型不匹配。

**shared buffer 带可见性。** `SharedBuf (g, e, Unpublished | Published, …)`：未发布的没有读公理，已发布的没有写公理。staged/chained 的"多面板共用一道 barrier"变成：一批 `coop_fill` 都停在 `Writing`，一次 `sync_publish`，再逐个 `publish_view`。

**发散靠三层 token。** `GM`（全组一致）、`WM`（warp 内一致）、`IM`（单线程，可发散）。barrier 只接受 `GM`，`ldmatrix`/`mma` 只接受 `WM`。逐线程代码只能经 `run_items` 进入，它的回调只拿得到 `IM`；全局的 `item_id`、`lane_id` 撤掉。逐线程的值回到 group 层是 `PerItem (g, T)`，不是 `Bool`，不能当分支条件。

**流水的槽**是六状态自动机，由 `Slot (g, round, state, …)` 的类型索引表达，每轮的顺序由公理签名锁定。

**类型拦不住的**交给独立检查 `gpu.check_sync`：S1 禁止 continuation 经词法作用域捕获 token；S2 每个 token 恰好线性使用一次；S3 对分支条件做污点分析；S4 检查 `store_async` 的源和宽度。

**对接。** TileFuse 不变；LowerToMem 是唯一生成这些原语的地方；`outline` 退化成机械展开；`remove_double_syncs`、`fence_pending_`、七种 `bind` 标注和裸 `fence` 删除。不加载 gpu 插件时，这些原语改写成恒等和顺序循环。

**可信边界**（方案自列）：不同线程写入的下标互不重叠且覆盖完整、物理存储不别名、`cp.async` 源地址有效、`n_items` 与 launch 配置一致、公理的后端展开正确、硬件提交组的 FIFO 顺序。

**落地要先解决的问题**：

- **循环携带的 token。** `affine.For` 的累加器类型是固定的 `«i: n; Ts#i»`，不依赖迭代变量，而流水和 R8 的每一轮都让 `e` 加一。要么给 `For` 加依赖迭代的累加器类型，要么在循环入口用一个公理把代数换成循环体绑定的新变量（`e` 在体内是 `e₀ + i·k` 的抽象）。这决定了第一步之后的形状。
- **R8 的轮次**在 LowerToMem 里是展开的（每轮一段代码），可以直接逐轮推进 `e`；流水环是真正的循环，受上一条约束。
- **擦除的时机**：要在 `mem.add_mem` 线程化内存之前还是之后擦除，取决于 token 是否参与 AddMem 的 fresh-mem 机制。
- **`staged`/`chained` 的协议**（`fence_pending_`）要整个替换成显式的 `Writing` 批，不能两套并存。

**顺序**：第一步只引入 `GM` 与 `SharedBuf`，用它重写填充批与 R8 合并轮（不含流水环），预计覆盖现有六处手工 fence 中的五处；`WM`/`IM` 与 `run_items` 是第二步；流水槽自动机依赖循环携带 token 的解决，是第三步；`gpu.check_sync` 第四步。

### C. 局部数值条件的检查

只列第 1 类义务。每条都要"先检查，再声明并使用"，检查在内存顺序上存活并支配访问。

| 空白 | 检查 | 备注 |
|---|---|---|
| G1 | 逐维：`off ≤ shape`，再 `ext ≤ shape − off`；`ext ≤ cap` | 先比较再相减，避开加法溢出。字面量形状不代表带运行时原点的条件能静态消失 |
| G2 | 在**相减之前**：`outer.off ≤ inner.off` 且 `inner.off + inner.ext ≤ outer.off + outer.ext` | 相减之后再查是恒真的。证不出时保留对原 producer 的读，或留运行时检查 |
| G4 | cap 与 peel 各自的精确分割等式 | 静态 |
| G9 | 流水深度在目标的立即数范围内 | 静态，必检 |
| G12 | 实际访问要求的对齐量；断言包住指针值本身，再交给 `mem.align` | 只删声明会丢优化信息，所以是"检查后声明" |
| G12 | 按 cap 分配之前：`ext ≤ cap`，容量乘积不溢出 | 不能验证物理布局和生命周期 |
| G13 | `layout` 在应用处与实际 logical/storage shape 重新核对 | 已把形状绑进 `Expr (r, s)`；`cap` 与张量形状的关系待补 |

**不在这张表里的**：原地更新（`linear_step` 的完整条件不是一次下标比较）、同步的全员到达、归约律、R8 的 2 的幂。前者属第 2 类，后三者属第 3 类或根本不是合法性条件。

## 4. 顺序

0. **修缺陷**（§2.5）：先逐条复现，属实的修掉并加拒绝路径的测试。D1、D6、D7 不改结构就能修。
1. **A0**：搬样板代码。已完成，见 §3 A。
2. **修正谓词与可信边界**：能静态完成的检查（C 表），流水深度的立即数检查。
3. **root descriptor**：让 `tile_req` 接收一个暴露并行维数、归约维数、输入个数、累加器类型的描述（扩展的 `Tileable`），A4 和 B6 都以它为前提。
4. **A3、A5，然后 A4**。
5. **物理 group/block、active 工作量、group 与 warp 两种作用域**：B3 的基础部分，加 B1、B2 的同步 collective。
6. **B5，再做 A1**；展开在 LowerToMem 侧按目标注入。
7. **显式的 prepare/compute 接口和异步传输**，再做 A2。
8. **B6、B4 和 mma 的加载、存储改造**。

**验收**分层，不用"Debug 全量 lit 通过"作唯一门槛，因为基线本身有两个既存失败：记录基线，要求相关测试无回归，并分别验证拒绝路径、CPU 参考和实际 GPU 行为。

## 5. C++ 会剩多少

初稿的"可搬走 1200–1500 行"没有依据，撤回。按当前的物理行数（评审量的）：

| 区域 | 行数 | 说明 |
|---|---:|---|
| `lower_bound_fold` | 216 | A1 的主要候选，但含分配、destination 和发布衔接 |
| `lower_tile_fold` | 257 | 只有一部分是流水，还有普通 fold、初始化、原地更新和 fragment 状态 |
| `lower_mma` 到 `store_fragments` | 113 | B4 只改类型，没有提供替代它们的 Mim 实现 |
| `promote_input` | 60 | A3 主要替换映射构造，op 重建和输入替换还在 |

合计约 646 行，且不能都算净删除。新增一个公理再配一段 C++ 实现，不算把算法搬走了。
agy 估计样板代码有 600–800 行可搬；A0 实施后实测净减 66 行。两轮估计都偏高一个量级，原因相同：C++ 里的大头不是拼项，而是带着 lowering 状态的决策，状态没有 IR 表示之前搬不走。
要评估净减少量，得按函数列出迁移的部分、保留的部分、新增的 lowering 与检查器、替代的 Mim 实现，这张表还没有做。

`panel_layout.cpp` 的银行冲突评分是启发式，可以留在 C++；同文件里的连续性、对齐、形状兼容判定是正确性相关的，要分开列账。

## 6. 未知

- 运行时 `assert` 的效果顺序、DCE 行为、device 上失败的终止与报告协议，都是假设，没有验证。
- 部分求值后的代码规模、nominal 类型的诊断质量、CPU/GPU 性能，需要原型实验。现有机制支持表达的方向，不支持"展开没有问题"这类确定结论。
- `Fill::contiguous` 查 64 个位置对当前受限的布局集合是否充分，可能有周期性论证，实现和文档里都没有给出；这里指出的是前提缺失，不是已知的错误输出。
- 标"（评审）"的条目来自 codex 和 agy 对代码的静态核对。我复现过的只有 D1；另外抽查过三条事实：两个旁表名已不存在、树形合并只在 2 的幂时启用、`Levels::mma` 没有匹配 combiner。
- agy 声明它没有编译运行，D2–D7 的控制流是从源码读出来的。
- PTX `wait_group` 立即数的确切上限随目标和 LLVM 版本不同，没有用 ptxas 实测。
