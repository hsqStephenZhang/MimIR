# GPU 多级 tiling 的形式化空白与消解方案（2026-11-11，按 codex 评审修订）

本文列出分支 `feat/gpu-tiling`（`4642222e88`）上仍由 C++ 决定或相信的东西，并给出把它们搬进 MimIR 的办法。
目标有两个，按优先级：

1. **减少 C++**：能写成 Mim 函数加部分求值的，不留在 C++ 里。
2. **至少在 IR 里反映出来**：搬不动的，C++ 作出的决策和它依赖的前提要以项的形式留在 IR 里。

模型见 `gpu-tiling-formalization.zh-CN.md`。初稿的评审见 `gpu-tiling-formalization-gaps-review.zh-CN.md`，本文已按它修订；
评审指出而我没有逐条核实的说法在文中标了"（评审）"。

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
2. **有限结构性质**（映射互逆、覆盖、布局单射、调度合法）：由独立的检查器在具体实例上验证，或写成 Mim 函数后用测试穷举。
3. **并发与代数性质**（全组一致到达、无数据竞争、归约律）：不能由断言承担。由受限组合子的语义论证、静态的作用域检查，或明确的用户许可承担。

独立检查器、原语的实现和后端始终在可信边界之内。本文不承诺"分析算错只会 trap"。

## 2. 空白清单

### G1 box 的合法性与覆盖 —— 未检查

- 输出网格来自输出形状和 tile sizes；输入 box 的 `off/ext/cap` 由 `producer_tile` 从读映射的像算出。
- 类型已保证：body 返回 `Tensor r ext T`。
- 未检查：逐维的 `off ≤ shape` 与 `ext ≤ shape − off`、`ext ≤ cap`、各 tile 恰好覆盖输出一次、box 包含 body 会读的全部元素。
- body 内迭代域与 `ext` 的关系只在写映射是 canonical 的情形下是简单相等；一般情形要检查的是输出映射。

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

### G10 tensor core —— 三个独立的空白

1. **算子匹配**（未检查）：`Levels::mma` 检查了作用域、秩、输入数、F32、指令形状和部分整除，但没有核实 combiner 是乘加，也没有完整核实访问关系是矩阵乘。lowering 固定生成 `gpu.mma_tf32`，不看原 combiner。我核对过前半句：那里只有"F32 矩阵"的检查。能否从公开的调度入口触发错误结果，没有反例验证。
2. **fragment 表示**（非类型）：warp 累加器是每个 lane 的 SSA 值加旁表 `fragments_`。还要求累加器是字面量二维形状、scalar seed、可原地更新（评审）。
3. **传输与 ISA 前提**（未检查，评审）：A 面板要支持四元素向量寻址；ldmatrix 的指令数、对齐、目标支持；`mma_ref` 用普通 F32 乘加，没有模拟 TF32 截断，所以现有 CPU 测试验证的是布局和普通乘加，不是完整的 TF32 数值语义。

### G11 level 链残留的 C++ 检查 —— 非类型

- sizes 是字面量；链的 `n` 等于 root 循环数；`Acc` 等于 root 累加器类型（不匹配只是警告）。
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
- **`panel_layout.cpp` 里与正确性相关的部分**：`Fill::contiguous` 只检查前 `min(total, 64)` 个填充位置，结果决定向量写是否合法；`layout_expr.make` 检查过的逻辑形状没有存进 `Expr`，`placed` 接受的 `Expr` 与输入形状没有类型关系。
- **NVPTX 发射侧**：`cp.async` 只在目标支持、大小为 4/8/16 字节、值恰好来自整块 global 读时发射，目标地址直接 cast 成 shared；mma 固定按 4/2/4 寄存器的 ABI 发射；shared 超过 48 KiB 后改用动态 shared，这些决策在 emitter 里。

## 3. 消解办法

### A. 写成 Mim 函数

要区分两件事：**纯算法的迁移**（下标、映射、计划）现在就能做；**GPU 实现的迁移**（shared、发布、复用、异步）要先有执行接口，部分求值不会自动补出这些语义。

| 方案 | 现在能做的 | 卡点 |
|---|---|---|
| **A3** 面板的 encode/decode、shape、padding、有效域由一个受限构造子派生 | 全部。`layout_expr.IndexOps` 已经是"一个解释器同时服务 affine 和整数"的先例 | 两个字段来自同一函数不自动互逆，要对有效坐标做 round-trip 检查（第 2 类义务） |
| **A5** 网格定义由参考语义和 lowering 共用 | `tiled_impl` 已有 Mim 公式 | 不要返回完整的 `«count; Box»`，那会在编译期大规模展开；返回计数和按需的 `box_at`，让循环留着 |
| **A4** level 链的残留检查进类型 | `Vec` 去掉 `rest`；`tail` 改成变体；`promote` 改成按 `nis` 索引的 `Bool` 数组 | `Idx n` 要收紧到并行维并保留归约拆分因子；类型不能保证一个 `Nat` 可归一化成字面量；root descriptor 要由受控构造子和 root 一起封装，否则只是并列的两个值 |
| **A1** R8 | 树归并的计划、pair 分配、非 2 的幂的退路写成 Mim，在 CPU 上测试 | shared 的生成、发布、复用没有可用契约；step 的签名要绑定 block、stride、输入和线程分配。删除 `lower_bound_fold` 是后续的验收项，不是第一步 |
| **A2** 流水 | 环的旋转、prologue、clamp、提交序列 | 不能只包装 `tile_fold`：先要让 step 显式分成 `prepare (block)` 和 `compute (acc, ready_panels)`，否则只是把隐式的 C++ 分析藏进新公理 |

### B. 类型与组合子

类型能检查同一个描述符的字段是否一致。它不能证明两个函数互逆、任意 merge 结合、任意控制流收敛。

| 方案 | 修正后的提法 |
|---|---|
| **B1** 放置 | 初稿的 `placed_in: … → T → T` 仍是恒等标注，只传达意图。要让跨 owner 的读成为类型错误，得是 `Placed (p, T)` 或带 place 的 buffer 类型，并约束所有访问操作、group/item 身份的来源和逃逸。`uniform` 检查不能删：别人也可能给依赖 item 的值贴上 Shared |
| **B2** 同步 | 拆成两个设计。**同步 collective**：全组协作算出一个新的不可变结果，返回时可见；它能覆盖"新分配的协作填充"和"每轮生成新快照的 R8"。复用存储、原地更新 partials 还需要独占访问和轮间顺序。**异步传输**：至少要表达 pending/ready、提交组、等待、槽位复用，通过内存顺序和受限 region 的检查落实。仅有签名拦不住 item 相关分支里的调用，所以 B2 完成前 `outline` 的发散检查不能撤 |
| **B3** 执行形状 | region 按物理 block 形状索引；逻辑工作量和 active 谓词是另外两个量；barrier 位于统一的物理 group 层 |
| **B4** `Frag` | 能避免把普通四元素数组当 fragment。删旁表还要求 fragment 数、tile 形状、布局随 fold 状态和 region 参数传播，并区分 A/B/C 角色与数值格式 |
| **B5** 归约律 | `[merge, identity, commutative]` 是声明不是证明。用受限的"已识别律"构造子加独立匹配；用户自己构造的明确叫 permission |
| **B6** MMA 算子匹配（新增） | 受限的 contraction 描述符或独立 matcher，在选择 `mma` 之前核实 combiner 和访问关系。这先于 `Frag` |

### C. 局部数值条件的检查

只列第 1 类义务。每条都要"先检查，再声明并使用"，检查在内存顺序上存活并支配访问。

| 空白 | 检查 | 备注 |
|---|---|---|
| G1 | 逐维：`off ≤ shape`，再 `ext ≤ shape − off`；`ext ≤ cap` | 先比较再相减，避开加法溢出。字面量形状不代表带运行时原点的条件能静态消失 |
| G2 | 在合并视图之前：子 box 落在父 tile 的有效范围内 | 证不出时保留对原 producer 的读，或留运行时检查 |
| G4 | cap 与 peel 各自的精确分割等式 | 静态 |
| G9 | 流水深度在目标的立即数范围内 | 静态，必检 |
| G12 | 实际访问要求的对齐量，先于 `mem.align` | 只删声明会丢优化信息，所以是"检查后声明" |
| G12 | 按 cap 分配之前：`ext ≤ cap`，容量乘积不溢出 | 不能验证物理布局和生命周期 |
| G13 | `layout` 在应用处与实际 logical/storage shape 重新核对 | 或把形状绑进 `Expr` |

**不在这张表里的**：原地更新（`linear_step` 的完整条件不是一次下标比较）、同步的全员到达、归约律、R8 的 2 的幂。前者属第 2 类，后三者属第 3 类或根本不是合法性条件。

## 4. 顺序

1. **修正谓词与可信边界**：加入能静态完成的检查（C 表）和对应的拒绝路径测试；给 `Fill::contiguous` 的 64 个位置补上周期性论证或改成完整验证；流水深度的立即数检查。
2. **A3、A5**：共享纯映射和网格定义。
3. **A4**：改链和 root 描述接口，恢复 `merge` 的类型错误，保留必要的实例检查。
4. **物理 group/block、active 工作量、作用域契约**：B3 的基础部分，加 B1、B2 的同步 collective。
5. **B5，再做 A1**。
6. **显式的 prepare/compute 接口和异步传输**，再做 A2。
7. **B6、B4 和 mma 的加载、存储改造**。这条路线独立，可以提前。

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
要评估净减少量，得按函数列出迁移的部分、保留的部分、新增的 lowering 与检查器、替代的 Mim 实现，这张表还没有做。

`panel_layout.cpp` 的银行冲突评分是启发式，可以留在 C++；同文件里的连续性、对齐、形状兼容判定是正确性相关的，要分开列账。

## 6. 未知

- 运行时 `assert` 的效果顺序、DCE 行为、device 上失败的终止与报告协议，都是假设，没有验证。
- 部分求值后的代码规模、nominal 类型的诊断质量、CPU/GPU 性能，需要原型实验。现有机制支持表达的方向，不支持"展开没有问题"这类确定结论。
- G10 的算子匹配缺失有代码依据，但没有跑过非乘加的反例。
- `Fill::contiguous` 查 64 个位置对当前受限的布局集合是否充分，可能有周期性论证，实现和文档里都没有给出；这里指出的是前提缺失，不是已知的错误输出。
- 标"（评审）"的条目来自 codex 对代码的核对，我只抽查了三条：两个旁表名已不存在、树形合并只在 2 的幂时启用、`Levels::mma` 没有匹配 combiner。
