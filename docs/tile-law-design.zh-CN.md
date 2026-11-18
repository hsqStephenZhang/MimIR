# TileLaw：把多级 tiling 的语义放进 MimIR（设计，2026-11-14）

## 0. 问题

MLIR 的 `TilingInterface` 和 Burn/CubeCL 的 trait 族都把 tiling 建模在宿主语言里：
MLIR 是 op 上的 C++ 方法（`getIterationDomain`、`getLoopIteratorTypes`、`getTiledImplementation`、`getResultTilePosition`、
融合用的 `getIterationDomainTileFromOperandTiles`，部分归约的 `PartialReductionOpInterface`），CubeCL 是按层（global/stage/tile）的 Rust trait 族。
两者的共同局限：**“切片之后语义不变”这条定律写在宿主语言里**，IR 看不见，不能组合、不能检查；MLIR 按 op 逐个实现，CubeCL 按算法逐个实现。

MimIR 现在的状况：

| 部分 | 现在在哪里 |
|---|---|
| 迭代域、并行/归约维 | `Tileable` 的字段；`tile_req` 的类型索引经反射公理 `root_loops`/`root_acc`/`root_contraction` 从 `op` 现算 |
| 并行切片 | `tile_of`、`tile_of_box`（C++ normalizer 给出语义） |
| 归约分块 | `tile_of_init`、`tile_of_partial`、`tile_of_post`（C++ normalizer） |
| 合并 | `TileMerge`；split-k、cluster split、lane、work-item 归约在 TileFuse/lower_to_mem 里各自特判 |
| 调度 | `TileLevels` 等层级类型 + `tensor.check` 规则（PE） |
| 实现 | `tiled`、`tile_fold`、`tf_*`（Mim）+ TileFuse 1869 行、lower_to_mem 4782 行（C++） |

两个已经暴露的问题都来自“事实是从 `op` 现算的”：
- 反射公理在 `op` 被前面的 phase 改写后不再归约，`tile_req` 的层级类型随之失效（“cannot coerce `TileLevels`”，已用提前去掉请求的办法绕过）；
- 评审要求“描述符必须由受控构造子与根一起封装，否则只是并列的两个值”（`gpu-tiling-formalization-gaps` A4）。

## 1. 原则

1. **定律、调度、实现三分。** 定律回答“这个计算可以怎样切、切完怎样合”；调度回答“在哪一级、多大、放哪、何时”；实现把二者降成循环、内存与同步。
2. **定律是 IR 里的一等值**，由受控构造子与它描述的根一起产生；事实（循环数、并行维数、长度、累加器、合并律、收缩种类）是它的字段，不再从 `op` 反推。
3. **对 `map_reduce` 族，定律由构造子推导，不逐个手写。** 不在这一族里的算子显式给出定律，其义务由差分验证（D3）承担。
4. **调度组合子作用在定律上，结果仍是定律**，多级就是组合；层级类型以定律的字段为索引。
5. **义务按能力分配，不要求用户写证明**：类型（作用域、操作数种类、tile 形状）、PE 判定（数值约束、F₂ 布局、协议模型）、逐次验证（实现等价）。

## 2. 定律的组成

一个定律描述在迭代域 `D = [0, S₀) × … × [0, S_{n−1})` 上的计算，前 `ro` 维并行、其余归约：

| 字段 | 含义 | 对应 |
|---|---|---|
| `n`、`ro`、`extents` | 循环数、并行维数、各维长度 | MLIR `getIterationDomain` + `getLoopIteratorTypes` |
| `inputs` | 输入个数（与各自的读映射） | 融合与提升的依据 |
| `Acc`、`init` | 累加器类型与初值 | `tile_of_init` |
| `step` | 在一个块上折叠：块 → 累加器 → 累加器 | `tile_of_partial`、MLIR `getTiledImplementation` |
| `merge` | 合并两个部分结果的幺半群，带种类：已识别律 / 用户许可 / 有序（不可重结合） | `PartialReductionOpInterface.mergeReductions`、`TileMerge`、B5 |
| `finish` | 输出块上的尾处理（epilogue） | `tile_of_post` |
| `place` | 输出块的位置：并行维到输出轴的映射 | `oax`、`getResultTilePosition` |
| `kind` | 收缩种类（`NoMma`/`Tf32Mma`/`HalfMma`） | `root_contraction` |
| `root` | 这个定律描述的根张量 | 封装：定律与根一起构造 |

**语义**（参考解释）：`⟦law⟧ = finish ∘ (按并行维 place 拼接) ∘ (对每个输出点，沿归约维 step 折叠 init)`；它就是根 `map_reduce_post` 的含义。

**切分引理**（调度正确性的核心，纸面证明一次）：
- 并行维切成覆盖且不相交的块（box 的覆盖与互斥由仿射/F₂ 判定）：拼接各块的 `⟦law|块⟧` 等于 `⟦law⟧`；
- 归约维按原顺序分块折叠：恒成立（结合律不需要）；
- 归约维乱序或并行地分块（split-k、cluster split、lane、work-item 归约）：要求 `merge` 是幺半群且种类不是“有序”，`⟦⟧` 等于各部分的 `merge`。

这三条覆盖了现在 TileFuse 的全部层级种类；每种层级只是在“块大小、作用域、合并方式”上取不同值。

## 3. 调度组合子

层级链保留现有的类型结构（`TileLevels`/`GroupLevels`/…，作用域、tile 形状、操作数种类作为索引），变化在两点：

1. **索引来自定律的字段**：`levels: TileLevels (law#n, law#Acc, law#kind)`，不再经反射现算；`check` 规则的事实（并行维数、输入个数、长度）也取自定律。
2. **每种层级对应一个“定律 → 定律”的组合子**（`split_par`、`split_red`、`bind`、`stage`、`pipeline`、`mma`），TileFuse 的逐级递归变成在定律上解释调度：
   并行切分把定律限制到块上，归约切分把 `step` 换成块上的折叠，`bind` 只改作用域，`stage` 换掉一个输入的读映射，`mma` 换掉块上的 `step`。
   这些都能用现有的 Mim 实现原语（`tiled`、`tile_fold`、`tf_*`、`stage_input`）表达。

GPU 专属的展开（R8 树、流水环、warp 特化）仍按评审意见放在 LowerToMem 按目标注入；定律给它们的是“要实现的语义”，协议模型（`ring_safe`、`cp_ring_safe`）给的是“实现必须满足的约束”。

## 4. 义务的分配

| 义务 | 承担者 |
|---|---|
| 作用域嵌套、操作数种类、tile 形状一致 | 层级类型（索引取自定律） |
| 数值约束（整除、尾块、资源、目标能力） | `check` 规则，PE 判定，事实取自定律 |
| 块覆盖且不相交、布局单射与连续、swizzle 与描述符一致 | 仿射/F₂ 判定 |
| 同步与环的协议 | 协议模型（`ring_safe`、`cp_ring_safe`） |
| 合并的结合性 | `merge` 的种类：已识别律由识别器给出，用户许可显式标注，有序的不允许乱序切分 |
| 实现等于定律 | D3 差分验证（`bench/validate`），逐次 |

## 5. 表达力要留出的位置

- 非矩形的迭代域（因果 attention 的三角域）：`extents` 之外加一个仿射谓词，块的覆盖判定随之换成 Presburger 集；
- 带前缀的合并（scan）：`merge` 之外允许“带偏移的合并”；
- 运行时长度（D6，暂缓）：`extents` 允许非字面量，阻塞点见 `next-directions.zh-CN.md`；
- 数据依赖的切片（gather、MoE 路由）：`step` 的读映射允许经由张量取值。

## 6. 推进步骤

1. **定律记录与受控构造子**（本次原型）：`tensor.law` 模块定义 `Law` 与 `law (…)`：由与 `map_reduce` 相同的参数同时构造根与事实；
   `law_req` 以定律的字段为层级类型的索引，`check` 规则直接用定律的事实，不经反射。
2. **定律的参考语义与切分引理的可执行检查**：`step`、`merge`、`finish` 用现有 `tile_of_*` 表达，lit 在 CPU 上执行“切分后的实现等于根”。
3. **TileFuse 在定律上解释调度**：逐级的 `tf_par`/`tf_reduce` 调用由 Mim 函数生成，C++ 只剩图分析（融合成员、提升面板的选择）。
4. **合并统一为幺半群**：split-k、cluster split、lane、work-item 归约共用 `merge`。
5. 资源与协议进类型。

每一步用 D3 的差分验证与完整 lit 验收。

### 符号形状（方案，2026-11-16）

退回统计里的“符号形状”分开数：张量形状是符号的 27 次，层级大小是符号的 2 次。
`plain_parts` 要求字面量形状，是因为 work-group 暂存的面板要字面量的存储（`stage_shape`、`placed` 及其布局搜索）；`tiled` 区域本身接受运行时的长度，`TileFuse` 对符号形状用封顶的存储。
所以分两步：先对没有暂存面板的链（只有 par/red 层，`promote` 为 0）放开形状的字面量要求；再让暂存的面板取 `affine.image` 的 cap（字面量上界）作存储。

### 驻留的子调度（方案，2026-11-16）

现状：work-group 层把驻留成员的盒子整块暂存，盒子里的元素在叶子里用 `generate` 描述，即每个 work-item 区域里的 `each`（逐元素组合，不切分）或 `parts` 加 `joint`（切成 `C` 段，一次填充写部分和，另一次合并）。
这是对“内层定律在盒子上按某条 work-group 子链运行”的手工特化：`each` 相当于每个 work-item 一个元素，`parts`/`joint` 相当于 `group_red_items` 的合并树。

要做的是让盒子由内层定律自己的 `group_step` 在给定的子链上算出，分三步，前两步不改变生成的代码：

1. 驻留视图从叶子提到 work-group 层，用设备 IR 逐字比较验收。
   区域输入的槽已经接好（`5cf873d86`）：步多了 `re: «i: nis; [Bool, «Ris#i; Nat»]»` 与 `res: Res re`，没有驻留盒子的输入在 `res` 里是空元组，不占缓冲；`split_par`、`split_bound` 的区域输入是 `(xs, res)`，生成的代码与之前相同。
   最初用长度为 0 的 `splat` 作空盒子，缓冲化拒绝零长度的张量，所以换成按标志选择的类型。
   不能放进上下文 `Cx` 沿链传：盒子是张量，`tiled` 的区域体不能读外层张量（`check_region_inputs`），只能作为区域输入穿过每一层；而区域输入是类型固定的 `Phys`，盒子的形状又随块大小变化，所以要先给 `Phys` 留出盒子的位置。
2. `Law` 加一个字段：给定物理输入、盒子的起点与长度和一条 `GroupLevels` 子链，返回盒子；子链的类型以 work-group 的块大小为索引，所以字段对块大小多态，`{M: *, g: «n; Nat»} → GroupLevels (n, Acc, M, g) → …`。
   记录里放这种带隐式参数的函数字段、在使用处用 `@g` 实例化，已在一个小例子上通过类型检查并正确求值（2026-11-16）。
3. 视图的来源换成这个字段，子链先取与现状等价的两条：不切分时每个 work-item 一个元素，可切分（`splits`）时 `group_red_items`；之后 `each`/`parts`/`joint` 整段删掉，`Input` 的 `merge`、`split`、`K` 也不再需要。

第 3 步对应退回统计里“work-group 里的归约成员”（12 次）：`TileFuse` 正是用 work-item 上的合并树实现它们。
在 sm_120 上读设备 IR 可以看到差距的来源：`softmax_resident`（`TileFuse`，20.6 µs）的部分和放在 shared 里，用 11 次 `shfl.sync.bfly` 按蝴蝶树合并；`softmax_resident_law`（55.9 µs）与 `_shared`（38.9 µs）的部分和由 `joint` 的 `vec.fold` 串行折叠 64 段，4 行只有 4 个线程在合并，其余 252 个在等。
所以在合并树进入定律之前放开 `realize` 对这 12 个请求的守卫会让它们变慢；`softmax_resident_law` 追上 `TileFuse` 是第 3 步的验收标准。

## 7. 原型进展（2026-11-14）

`tensor.law`（`tensor.mim`）完成了第 1 步和第 2 步的一部分：

- `law.of_map_reduce` 以 `map_reduce` 的参数同时构造根与定律；定律是结构化记录 `Law (r, s, T)`，字段为循环数、并行维数、输入个数、长度、累加器、收缩种类、根，
  以及两个切分：构造子在根的输入仍在作用域时把它们做成闭包，`par ts` 拼接根的各块，`red (loop, ks)` 按原顺序折叠整块、再折叠剥离的尾块、最后做 epilogue。
- 第三条切分引理：`of_map_reduce_merging` 带用户许可的 `merge`，定律的 `red_par (loop, parts)` 把归约维切成 `parts` 段，各段独立从初值折叠，再逐元素 `merge`，最后做 epilogue。
  前提是初值是 `merge` 的单位元、`merge` 可结合，由许可者担保；不带 `merge` 的定律（“有序”）的 `red_par` 退回按序折叠。这是第 4 步“合并统一为幺半群”的入口：split-k、work-item、lane 归约都是不同作用域上的 `red_par`。
- `law.req` 以定律的字段为层级类型的索引，并用定律的事实（不经反射）调用 `check.verdict`，经 `check.report` 在调用处报错。
- 第 3 步的语义部分：`law.schedule (l, levels)` 是层级链的参考解释器，与 `TileFuse` 接受同一个 `TileLevels` 值。
  解释器逐级递归（`tile_step`/`group_step`/`item_step`/`warp_step`/`wg_step`），每一级只落到三条切分引理之一：并行类（`Par`、`Group`、`Item`、`Warp`、`Warpgroup`）是 `split_par`，按块拼接并把块的偏移与长度传给下一级；
  归约类（`Red`、`RedItems`、`Lanes`、`RedTma`、`RedWs`）是 `split_red`，用 `tile_fold` 按原顺序折叠，尾块的长度截到剩余部分；`Done`、`Mma`、`Wgmma` 在块上做 `map_reduce_acc`；`Vec`、`Cluster`、`Persistent` 不改变语义，直接跳过。
  于是任何一条合法的调度都有一个由定律给出的含义，`TileFuse` 的实现可以逐条与它比较。
- 第 3 步的实现部分：解释器的并行切分带上作用域（`Group` 绑定 work-group、`Item` 绑定 work-item、`Warp`/`Warpgroup` 绑定 warp 与 warpgroup，与 `TileFuse` 的 `tensor.bind` 一致），并遵守层级的尾块选择（`tail = 1` 剥离，`0` 截断）。
  这样，对“朴素”的链，定律的含义本身就是实现：`law.run (l, levels, control)` 用 Mim 的判定 `law.plain` 在 PE 时选择——链里没有面板提升、向量化、warp 级与 cluster/persistent 时用解释器，否则仍交给 `TileFuse`。
  `lit/gpu/law_run.mim` 中，work-group × 剥离尾块的归约 × work-item 的链由解释器生成，outline 成一个 kernel，并在 GPU 上执行正确（`lit/nvptx/law_run.mim`）；带 `item_vec` 的链回到 `TileFuse`。把剥离的尾块去掉的变异在 CPU 与 GPU 用例中都被测出。
  解释器的每一步带一个 `fresh` 标记：累加器仍是种子 `splat (So, init)` 时，并行切分不把它当区域输入，而由每个块自己构造 `splat (ext, init)`；归约折叠之后的累加器才经区域输入传递。
  因此 outline 的设备缓冲与 `TileFuse` 相同（3 个）。不能用 `core.pe.is_closed acc` 代替这个标记：它只会归约成 `tt`，内层不闭合的累加器会把判定留到后端。
- 第 4 步的入口：work-item 归约（`RedItems`）与 lane 归约（`Lanes`）也是朴素的。解释器的归约切分 `split_bound` 带作用域与层级的 `TileMerge`：累加器仍是种子时，折叠的内层 lambda 被标上与 `TileFuse` 相同的 `tensor.bind`（2 或 4）和显式的 `tensor.merge_with`，于是 `lower_to_mem` 让各 work-item/lane 从自己的种子折叠，再用合并树或 warp shuffle 合并部分结果；累加器不是种子时退回按序折叠。
  语义上，这正是定律的第三条切分引理在一个作用域上的实例；义务（种子是单位元、合并可结合）与 `TileFuse` 一样由识别出的组合子或显式的 `merge` 承担。
  `lit/gpu/law_run_items.mim`（行求和，4 组 × 4 项，共享内存合并）与 `lit/gpu/law_run_lanes.mim`（64×64 GEMM，每项 4 条 lane，warp shuffle）都在 GPU 上执行正确；去掉绑定的变异被测出。
- 用 `of_map_reduce_merging` 建一个累加器是二元组（在线 `(max, sum)`）的定律时，碰到两个 core 问题。
  其一已修：`World::extract1` 取元素类型时没有 zonk，类型若仍是已求解的隐式参数的 Hole，arity 被当成 1（`lit/tensor/generic_pair_merge.mim`）。
  其二已修：Hole 求解之后，同一个张量类型曾一份是嵌套的 `«ext; «2; F32»»`、另一份是融合的 `«ext, 2; F32»`，`Tensor` nom 按包裹的类型哈希，两份不相等。根源是 `ext` 是绑定变量的投影，类型 `«?r; Nat»` 里的 `?r` 已求解为 1，但 `Shape::rank` 读 arity 时没有 zonk，秩未知，`World::seq` 就不融合；现在 `Shape::rank` zonk arity。
  于是元组累加器的定律可用：`lit/gpu/law_run_merge.mim` 用 `of_map_reduce_merging` 建在线 `(max, sum)` 的定律，work-item 归约用它担保的 `merge`，在 GPU 上执行正确。这是 FlashAttention 在线 softmax 的前提。
- 注意力作为一个定律（`lit/tensor/law_attention.mim`）：循环 `(h, i, j)`，每行的累加器 `(max, sum, out: «N; F»)`，输入是 `q`、`k`、`v` 的行（元素类型 `«K; F»`/`«N; F»`，用 `as_vectors` 把最后一维看成一个元素，是同一块内存的视图，比原先用 `generate` 复制少一半分配；`buffer.vectorize` 声明的对齐改为整除行字节数的最大 2 的幂，`«3; F32»` 这样的行才是合法的 LLVM 对齐），`merge` 是在线 softmax 的合并，由 `of_map_reduce_merging` 担保。
  于是 FlashAttention 的 KV 分块就是有序切分引理，split-KV（flash-decoding）就是合并切分引理：`denote`、`serial_red` 分块、3 段 `red_par` 都等于 softmax 注意力。
  两条朴素调度经 `law.run` 在 GPU 上执行正确（`lit/gpu/law_attention.mim`、`lit/nvptx/law_attention.mim`）：每行一个 work-item 按序折叠 KV 块（2 组 × 5 项），以及每行一个 work-group、其 work-item 各折一段 KV 再用定律的 `merge` 合并（10 组 × 3 项）。把 `merge` 里第二段的缩放去掉的变异在 CPU 的 `red_par` 和 GPU 的 split-KV 中都被测出。
  这一版逐个 KV 位置做点积，不用张量核，只证明语义与调度；性能要等矩阵形式的块（行块 × KV 块的 `mma`）进入解释器。
  记下的限制：
  - 在 `math.mode.fast` 下，种子里的 `-inf` 经 `sub`/`fmax` 成为 poison（`S = 1` 时 LLVM 直接折叠），所以这里用 `math.mode.none`。
    已修其余各处：`sdpa_online_inputs` 与 `attn_*`、`law_blocks`、`attention_fold`、`warpgroup_owned` 里碰到 `-inf` 种子的运算（运行最大值的 `fmax`、修正项的 `sub`/`exp`）改用去掉 `ninf` 的模式，其余运算仍用 `fast`。
    这些程序的 `-inf` 都在循环里，LLVM 看不到常量，所以原先没有算错（单个 KV 块的情形在 CPU 与 GPU 上也正确），但这是未定义行为；去掉 `ninf` 后 `attn_warp` 的时间不变（同一时刻 0.328 对 0.328 ms，32 × 2048）。
  - 已修：组块不整除时（如 `M = 5`、组块 2），内层绑定层的 extent 是运行时值，`lower_to_mem` 原先拒绝——`TileFuse` 同样如此，并非只有解释器。现在绑定的 `tiled` 按 `nat_upper` 给出的上界启动，超出运行时 extent 的块跳过（`lit/gpu/tile_items_capped.mim`，两条路径都在 GPU 上执行正确）；注意力的行调度改回不整除的组块 `(1, 2)`。
  - 已修：带 gpu 插件时，留在主机上的代码（没有调度的注意力根、块定律的串行调度）编译失败，原因有三：
    `gpu.lower_btensor_map_reduce` 把以张量为初值的 `map_reduce_acc` 也当成一线程一输出的 kernel；它也把嵌在主机循环里、映射或形状用到外层循环变量的 `map_reduce` 搬成 kernel，kernel 不闭合；
    `mem.AddMem` 只看续延的首个参数是否是内存，于是给 kernel 里 `[I64, mem.M 1, acc]` 形状的循环头又加了一个 `mem.M 0`。
    现在前两种 `map_reduce` 留在主机上，已带内存参数的续延不再加内存。`lit/gpu/law_attention.mim`、`law_blocks.mim` 的主机与设备情形都在加载 gpu 插件时执行正确。
- 块定律（`law.of_blocks`、`law.blocks_run`）：元素级定律的叶子是逐个位置的 `map_reduce_acc`，达不到张量核的性能；块定律把实现交给用户的块：累加器是一族 tile `A ext`（注意力取 `attn_warp` 的结构体数组 `(max, sum, O)`），`init`、`step`（一个块的行 × 一个块的 key）、`finish` 由用户给出，`denote` 是每个调度都必须等于的含义。
  `step` 与 `finish` 和 `merge` 一样是被担保的义务，由差分测试检查：`lit/tensor/law_blocks.mim` 在 CPU 上比较含义、串行调度与 work-group × warp 调度和 softmax 注意力；`lit/gpu`、`lit/nvptx/law_blocks.mim` 在 GPU 上执行 warp 调度（TF32 `mma`、`cp.async` 双缓冲的 K/V 面板）。把并行块的偏移去掉的变异在三处都被测出。
  解释器分两段：并行段的每个块从 `init` 开始，到第一条归约层或叶子时折叠 `step`，最后 `finish`；warp 层绑定为 warp 拥有的区域（`tensor.bind 5`），未切分的维取根的字面 extent。
  性能：`bench/gpu_attn/attn_law.mim` 与 TileFuse 的 `attn_warp` 生成同样的程序，RTX PRO 6000 上 32 × 1024 到 128 × 2048 的时间相同（0.086–1.04 ms）。
  为此 `lower_to_mem` 的 warp 翻译与 `store_warp` 接受带前导单位维的 tile（head 作为单独的循环时 tile 是 `(1, 16, D)`），这样 head 与 warp 偏移无关，面板仍可协作装载。
- 第 3 步完成（2026-11-15）：`TileFuse` 的全部层级都由解释器实现，`law.run` 不再退回 `TileFuse`（`plain` 已删除）。逐项：
  - `Vec`：并行维为 1 的那一维成为向量循环；最后一个归约循环按其大小切成 `(K / f, f)` 并展开内层，与 `split_reduction` 相同。
  - 提升（`promote`）：归约层为每个块生成计划，被提升的输入把块的盒子（`affine.image`，由单位斜率驱动的维直接取那个循环的块大小，区域专门化后是字面量）`stage` 成面板，面板维与宽度由单位盒的像在 Mim 里探测（与 `panel_axis` 相同的规则），叶子经面板读映射读取；work-group 内 work-item 或 warp 的归约放宽为整个 work-group 的块（work-group 范围的面板），work-group 层自己的提升把输入的盒子整块驻留。
  - 面板布局：存储形状、最内连续维、向量宽度、work-item 的读法与张量核模式这些事实由 Mim 算出。
    张量核操作数的布局是确定的规则，在 Mim 里（`tensor.tc_layout.mma`/`wgmma`）：连续维放最内并填充末维，或把唯一的行维排成 128 字节段的行并按段 swizzle；七个 Hopper 链的设备 IR 与 `TileFuse` 的逐行相同。
    SIMT 面板的布局是一个按 bank 冲突建模的搜索（约 550 个候选、每个模拟数个 warp 的共享内存访问），按实测的 PE 速度（约 2 µs 一次迭代）每个面板要几分钟，所以它留作 oracle：公理 `tensor.panel_layout` 的 normalizer 调用原来的 `panel::choose`，结果是 `layout_expr` 项，由已有的 F₂ 判定检查。
  - `pipeline`（深度为 0 时取 `check.depth`，即 `-X tensor:pipeline`）、`RedTma`（`fill_tma`）、`Mma`/`Wgmma`（`tensor.mma`）、`RedWs`（`specialize`）、`Cluster`、`Persistent`：与 `TileFuse` 相同的标注，包在相同的位置。
  - 修了 `root_contraction` 的 normalizer：泛型的根（组合子还是参数、映射还不是 lam）原先被判成 `NoMma`，所以定律的收缩种类永远是 `NoMma`，带 `Mma`/`Wgmma` 层级的链对定律不能通过类型检查；现在它在这些事实已知前保持不规约。
  验证：`bench/gpu_tiling` 的 GEMM 逐个对比（RTX PRO 6000，`law_*.mim` 是 `gemm_*.mim` 的定律版本）——SIMT 的 `tiled`、`g3`、`g3_128` 与 `TileFuse` 持平，`g2`/`g2cap`/`g2s` 更快，TF32 与 16 位 `mma`（含 `pipeline=2`）持平；七个 Hopper 链（`wgmma`、TMA、cluster、cluster split、persistent、warp 特化）的设备 IR 中 `wgmma`、TMA、mbarrier 与描述符的每一行在去掉名字后完全相同，在 H200 NVL 上实测结果正确、时间持平（`bench/gpu_tiling/RESULTS.zh-CN.md`）。
  `lit/{tensor,gpu,nvptx}/law_{vec,promote,mma,wgmma_*}.mim` 用与对应 `tile_*` 测试相同的检查覆盖这些层级。
- 成员融合第 1 步（2026-11-15）：逐点生产者的融合是定律的复合 `law.fuse slot (Sp) (g, ginit) gmaps ps (消费者的参数)`。
  含义是消费者读生产者 `map_reduce (Sp, Sp) (g, ginit) id gmaps ps`，与 `is` 在 `slot` 处放的是什么无关，所以融合不改变含义，只改变实现。
  定律的构造子 `build` 不再假定穿过各级的就是根的输入：它带一个物理输入 `phys: P` 与 `unfold: P → 逻辑输入`。普通定律取 `is` 与恒等；`law.fuse` 穿过 `(消费者的输入，slot 处换成 (), 生产者的输入)`，`unfold` 在叶子与面板里用穿过的生产者输入重建生产者。
  读生产者的地方取它需要的盒子 `tile_of (生产者, 盒子)`：被提升的面板在装载时算出生产者（张量核的情形，与 `TileFuse` 的成员相同），未提升的叶子算它读的那一块（重算）。
  `lit/gpu/law_fuse.mim`：TF32 `mma` 链与 SIMT 链的定律版本和 `TileFuse` 成员版本都在 H200 上执行正确，各一个 kernel、4 个设备缓冲；1024³ 的 dequant GEMM 与 `TileFuse` 持平（SIMT `g3` 0.0821 对 0.0825 ms，TF32 0.0617 对 0.0615 ms，`bench/gpu_tiling/law_dqf_{g3,tc}.mim`）。
  一路修的 core 问题：`gpu.outline` 把闭合的区域输入映射到 kernel 参数时没有把 kernel 的变量绑进作用域，映射落进根作用域，同一模块里另一个函数的主机代码用到同一个闭合值时被改写成 kernel 参数（两个函数共享闭合输入时外部函数不闭合）；`core.pe.is_closed` 把未求解的 Hole 当作闭合，构造子在隐式参数还没推断完时就展开。
  限制：一次融合一个生产者（嵌套的生产者、多个成员留待后续）。
- 成员融合第 2 步（驻留，2026-11-15）：`law.fuse` 的生产者参数就是 `map_reduce` 的参数，归约生产者也可以融合；`law.fuse_merging` 还带生产者的 `merge`。
  work-group 层提升融合的槽时，它的盒子驻留在共享内存里，由 work-item 合作算出：叶子里整块 `stage` 的源是两次合作填充的 `generate`——先把盒子 × `C` 段归约（`C` 约为 work-item 数除以盒子大小，整除 K 时才切分）各自从种子折叠成部分结果，再用 `merge` 把每个元素的 `C` 个部分结果合起来；没有 `merge` 时每个元素由一个 work-item 整个归约。
  这样做而不是像 `TileFuse` 那样在 work-group 层插入绑定到 work-item 的折叠，是因为解释器的叶子在 work-item 区域里，而绑定的折叠必须直接嵌在 work-group 层下（`lower_to_mem` 的规定）；合作填充在 work-item 区域里也是 work-group 统一的。
  `lit/gpu/law_resident.mim`（逐元素）与 `law_resident_merge.mim`（4 行 × 4 段，部分结果 `«4, 4, 2»` 与合并后的 `«4, 2»` 都在共享内存）在 GPU 上执行正确。
  12 × 1024 × 1024 的在线 softmax（`bench/gpu_fusion/softmax_resident_law.mim`）在 H200 上 37.4 µs，`TileFuse` 的 `softmax_resident` 29.4 µs；不切分时 228 µs。
  更正：最初测得的 37.5 µs 出自有数据竞争的 kernel（合并部分结果的填充与写部分结果的填充同在一个 work-group 批次里，批次只在最后放一道屏障）；`lit/gpu/law_resident_merge.mim` 规模太小没有测出来。现在 `lower_to_mem` 让读另一次填充的合作填充离开批次，两次填充前后各有屏障，结果正确且时间不变。
  修的 core 问题：合作填充里每个元素的计算原先按 work-group 统一的作用域分配，一元素的临时 tile 落进同一个共享槽，各 work-item 竞争写它；现在元素的计算按私有作用域分配。区域输入检查把 `splat` 当作读外层张量，它其实只构造张量。
  仍差的 27% 主要是流量：生产者从全局内存再读一遍 `x`，而 `TileFuse` 的折叠读暂存的行。让生产者读消费者暂存的视图（别名）试过但撤回了：把视图交给生产者需要在部分实例化时按符号下标取异构元组，`Check`/`extract` 给出并类型或折叠错的 arity（另记：`World::extract1` 对 arity 1 的依赖 `Arr` 也按单元素折叠，返回值的类型仍是 `Arr`，这是同一类问题）。
  yields（成员同时被区域外使用时由融合的 kernel 交回）还没有做：需要多输出的定律。
  别名（2026-11-15）：`law.fuse_shared … ps share …` 声明生产者的输入 `m` 就是消费者的输入 `share#m`，含义按构造读它（不是待担保的义务）。
  `build` 多了钩子 `restage`：`values_of` 先算出各输入的暂存视图（是否整块暂存、盒子的原点与大小、暂存的盒子），成员的生产者由 `restage` 重建；`fused_shared` 的 `restage` 调 `shared_producer`，在消费者整块暂存了共享输入、且生产者为成员盒子读的范围落在暂存盒子里时，让生产者经平移到盒子原点的映射读暂存的盒子。
  范围检查用 `affine.image` 的 `cap`（与符号原点无关的范围上界），比较才可判定；`lower_to_mem` 中读暂存盒子的合作填充也离开批次。
  `bench/gpu_fusion/softmax_resident_shared.mim` 在 H200 上 27.0 µs（原 37.6 µs，`TileFuse` 29.4 µs），`lit/nvptx/law_resident_shared.mim` 20 次运行都正确。
  一路修的 core 问题：arity 1 的可变（依赖）`Arr` 上取元素不再折叠成值本身（`World::is_folded_axis`）——折叠后元素的类型还是那个 `Arr`，`npp = 1` 时 `gmaps#m` 不能当函数用。
  仍有的限制：共享与不共享在同一个 `fused` 里实现时（`(ps#m, is#j)#shared`），不共享的分支在 `npp = 1` 且隐式参数未定时仍会把 `Spp#m` 折叠错，所以分成了 `fused` 与 `fused_shared` 两个构造子。
  评审第五轮后补的：`bitcast` 不查类型，所以 `fused_shared` 检查共享输入的秩、形状与元素大小和生产者声明的一致，并拒绝 `share` 指向融合的槽（消息 23、24，`lit/error/law_shared_{shape,slot}.mim`）；对共享的 `m`，`ps#m` 被忽略。
  不切分的路径（每个 work-item 归约整行）由 `lit/nvptx/law_resident_shared_each.mim`（64 × 288，`288 % 64 ≠ 0`）覆盖，20 次运行都正确。
  `reads_fill` 原先只认“`get` 的源剥到 `stage`”，而成员的元素是 `get (tile_of (map_reduce … (stage …)))`，`stage` 只是生产者的输入，这条规则命中不到；现在可达的 `stage` 都算读了合作填充。
  追踪确认：两种路径下读暂存盒子的填充都已在批次之外（前后各有屏障），所以屏障数不变；这条规则是对日后把这类填充放进批次的调度的保护。
  合并 `fused` 与 `fused_shared` 的再次排查（2026-11-15）：现在 `ps2` 的选择本身已无问题（`share` 在消费者参数之前、`build` 的 `kinds` 之后），失败只出现在不共享（`share = ‹npp; 0›`）时调用 `shared_producer`：用户调用细化期间 `shared_producer` 的体被展开，其中 `map_reduce` 规范化出的 `map_reduce_post` 的隐式 `Ris` 是未定的 Hole、`Sis` 是 `Sq`，进而 `«Sq#m; T»` 折叠错。把 `core.pe.is_closed` 改成连类型一并深查 Hole（含已解未 zonk 的）不起作用，说明 Hole 不是经过滤器的实参进来的，根因仍未查清。
  未做：比较折叠不成字面量时用 `if_static` 退回不别名（现在会在 lowering 报错而不是算错）；填充离开批次时的断言；`is_folded_axis` 的单元测试（需要构造带未定 Hole 的依赖 `Arr`）。
- 成员融合第 3 步（归约融合，2026-11-15）：归约生产者融进同一归约循环的消费者，靠的是滚动更新，现在由定律推出，不再手写。
  `law.of_rolling` 取一个运行统计量（`stat`、种子、`rmerge`）、每个元素相对统计量的贡献（`f`、种子、`plus`）和修正（`ratio`、它在贡献上的作用 `scale`），
  推出逐元素的步 `(r', plus (scale (ratio (r, r'), a), f (xs, r')))` 与合并 `(rmerge, 两边都修正到合并后的统计量再 plus)`，再交给 `of_map_reduce_merging`。
  义务（`scale` 对 `plus` 分配、`scale (ratio (r, r'), f (xs, r)) = f (xs, r')`、比值可复合）与 `merge` 一样由使用者担保、由差分测试检查。
  注意力由此从部件得到：统计量是行的运行最大值，每个 key 贡献权重与加权的值，修正是 `exp (m - m')`；`lit/{tensor,gpu}/law_roll.mim` 的含义、KV 分块、split-KV 与两条 GPU 调度都等于 softmax 注意力，把修正换成 1 的变异被测出。
  块级同理：`law.blocks_rolling` 从块的统计量（新的行最大值）、修正（`exp (m - mx)` 作用于和与输出 tile）、块的累加（行和与 `P·V` 的 `map_reduce_acc`）推出块定律的步；累加器保持使用者的扁平元组，由 `split`/`join` 看成统计量与其余部分（嵌套的元组累加器让 warp 区域的累加器落到设备堆上，大尺寸时越界）。
  `bench/gpu_attn/attn_roll.mim` 与手写步的 `attn_law.mim` 生成同样的 kernel（TF32 `mma`、`cp.async` 双缓冲），H200 上 32 × 1024 到 128 × 2048 的时间相同（0.092–1.274 ms）。
  一并改了：用担保的 `merge` 建的定律（无序）在 work-item 归约里把它显式交给 `merge_with`（`TileMerge` 0），有序定律仍由 lowering 识别组合子；lane 归约总由 lowering 识别组合子，所以只接受识别出的组合子。
- 定律的组合子义务（2026-11-15，评审第 1 条）：评审复现了两种静默算错——有序定律切内层归约循环，以及把不可结合、种子不是单位元的组合子切到 work-item。
  现在 `TileFuse` 的组合子识别（`reduction_law`）移进插件库，经 `check.combiner` 反射成根的事实；定律多一个字段记录组合子允许的事（担保的合并、种子是单位元的可结合组合子、种子不是单位元、都不是），`law.run`/`law.req` 的 `check.law_verdict` 与 `law.red`/`law.red_par` 据此报错。
  `lit/error/law_ordered_inner.mim`、`law_items_combiner.mim`、`law_red_inner.mim` 是三个复现。
  评审修复后又指出：担保的合并若只承诺结合律与单位元，切内层归约循环（重排元素）与 work-item 的合并树（蝶形配对，`r8_plan`）都会让不可交换的合并静默算错。
  所以担保的合同改为：`merge` 是以种子为单位元的可交换幺半群，组合子把一个元素折进来等于与它从种子的折叠做 `merge`（现有的担保合并——加法、在线 softmax、滚动律——都满足）；在此合同下内层循环与 work-item 切分都放行。`law.schedule` 也走同样的检查；`check.combiner` 在组合子含外层变量时保持不规约。
- 第 4 步（流量模型，2026-11-15）：定律多了一个字段 `traffic: TileLevels → Nat`，按 Welder 的算法在 Mim 里数一个调度经过全局内存的字节：第一条把输出切成块的层级（`Par`/`Group`，穿过 `Cluster`/`Persistent`）给出块，每个块读每个输入经映射成像的盒子（`affine.image`），乘元素大小与块数，再加一次输出。
  `law.traffic (l, levels)` 在 PE 时归约成字面量；`law.best (l, candidates)` 在一组候选层级链里取流量最小的，同样在 PE 时选定，可以直接交给 `law.run`。
  `lit/tensor/law_traffic.mim`：64³ F32 GEMM，16 × 16 块 147456 字节、32 × 32 块 81920、整块 49152，`best` 取 32 × 32。
  模型只数第一层的块，不看更内层的暂存与复用，也不含计算量；融合的生产者按它在消费者里的槽数（物化的大小）。它是决定调度的默认依据，不是性能预测。
- 第 5 步（去掉 `TileFuse` 的逐级递归）的现状（2026-11-15）：`tile_req` 默认先试着按定律实现（`-X tensor:tile-law=0` 关掉）。
  根是普通 `map_reduce`（无 epilogue、标量种子、规范调度、形状与层级大小都是字面量）的请求，没有成员时由 `law.run (law.of_map_reduce (根的各部分), levels, control)` 实现，只有一个根直接读的成员时由 `law.fuse`/`law.fuse_merging`（成员带 `merge_with`）实现。
  仍交给逐级递归的：形状或层级大小是符号的；work-item/lane 归约没有显式合并（要靠 `TileFuse` 识别组合子是否可结合、种子是否单位元，`tile_reduction_contracts` 的诊断）或有序层切了内层归约循环；成员在请求外也被使用（yields）；多个或嵌套成员；work-group 链里的归约成员（`TileFuse` 把它切到 work-item 上；定律的驻留成员按元素在私有缓冲里算，每个元素一次设备 `malloc`，见下面的调查）。
  这样默认打开后全部 lit 通过：86 个用 `tile_req` 的测试文件里 31 个至少有一个请求走定律；只检查 `TileFuse` 日志的 7 个测试改为检查走了定律，`tile_group_chain` 放宽成它说明的性质（中间结果被释放，ASan 查泄漏），因为定律路径每个乘积多一个 64 × 64 的缓冲。
  一路修的：`lower_to_mem` 在打开 work-group 批次前先实现计算出的累加器（剥离的尾块之前的整块折叠，原先嵌套打开批次）；`law.req` 交给 `tile_req` 的层级带 `core.bitcast`，路由时剥掉；驻留成员的合并改用展开的 `vec.fold`（二元组累加器的 `tile_fold` 走了按 tensor 字段的乘积折叠），去掉了恒等的 epilogue。
  所以逐级递归还删不掉：上面那几类要先在定律里有对应物（组合子性质的检查、yields、多成员、按 work-item 切分的驻留归约）。
  驻留成员每个元素一次的设备 `malloc`（2026-11-15 的调查）：元素的一元素 tile 由 `lower_map_reduce` 分配，`heap2stack` 只转换不逃逸的分配，而逃逸分析把它钉成 ⊤；是哪一处 `escape` 钉住它还没有查清。
  已知的只是两条修法无效并已撤回：给元素计算一个在填充循环外绑定的私有池（这块缓冲不走 `allocate` 的池路径）；让逃逸分析把直接风格调用的参数按 SCCP 流进被调的续延（缓冲交给降低后的归约循环的续延使用，读写与释放都在别的续延里，但这样改后它仍是 ⊤）。
  要消掉它，要么把一元素的归约降低成标量循环（不经缓冲），要么让 lowering 在 work-group 统一的位置把计算的 `map_reduce` 按输出分给 work-item（合作计算，直接写进面板）。
  去掉 work-group 链里归约成员的守卫的试验（2026-11-15，已撤回）：去掉后命中守卫的是 `gpu/tile_resident`、`tile_reduce_fused`、`tile_reduce_fused_online`，走定律都算对，但 `tile_resident` 与 `tile_reduce_fused_online` 的 `promote` 只提升了 `x`，成员的盒子落在每个 work-item 的私有缓冲（`[4 x [2 x float]]`，设备 `malloc`）。
  把路由时的层级改成也提升成员的槽（`law.promote_member`，`TileFuse` 正是这样把归约成员切到 work-item 上），再在 `C` 为 1 而有合并时按整除 K 的 2 的幂切分，成员与部分结果都进了共享内存（`[4 x [4 x [2 x float]]]`），仍对。
  剩下的 `malloc` 是每块一个元素的部分结果（8 字节）：每个 work-item 循环处理 4 块，`heap2stack` 不转换循环里的分配（`in_loop`），而 LLVM 后端把 `mem.slot` 的 `alloca` 发在原地而不是入口块，在循环里转换会让栈逐次增长。
  64 × 256 的情形每个 work-item 恰一块、没有这层循环，所以没有 `malloc`。要去掉守卫，得先让后端把定长 `slot` 发到入口块、并让 `heap2stack` 证明缓冲不跨迭代存活；在此之前守卫保留。
- 第 5 项（2026-11-15）：容量是合法性而不是代价（评审第六轮）。`Law` 加了字段 `shared`：work-group 暂存的共享内存，即第一个 work-group 层常驻的盒子，加其中第一个归约层的面板（在 work-item、warp、warpgroup 之下也覆盖 work-group 的块）乘环的深度。
  `verdict_of` 在 `check.law_verdict` 之外加一条规则（消息 25）：超过目标的上限即报错，上限按 `check.target ()` 取（sm_80 163 KB，sm_86/89/120 99 KB，sm_90/100 227 KB，未指明时按 H200）；`law.run`、`law.req`、`law.schedule` 与 `law.best` 都用它，所以 `best` 不会选放不下的调度，`run` 对它报错（`lit/error/law_capacity.mim`）。
  `law.traffic` 仍是纯字节数；1024³ 的 GEMM 里整 K 提升的 256 × 256 候选流量最小（37.7 MB），但被规则排除（`lit/tensor/law_traffic.mim`）。仍未计入：`RedItems` 的部分累加器、驻留的融合成员、布局的填充。
  `Law` 的字段改由具名访问函数取（`ro_of` 等），加字段只动访问函数。
  `law.run` 的 `control` 明确为不用：定律的构造已经决定融合什么，融合与否含义相同；在融合与不融合的实现之间由 `control` 选择，要等第 4 项的 `Source`（`plain` 与 `produced`）之后才有对象可选。
  TileSight（tile-ai 的解析式性能模型）的研读建议把代价从字节改成时间（按资源的 II 下界、波与尾块）、加派发顺序的 L2 复用，留待第 4 项之后。
- 第 4 项第一步（2026-11-15）：输入不再是位置上的特例。`build` 原来的 `slot`、`phys`、`unfold`、`restage`、`pmerge`、`split`、`K` 与 `is` 换成每个输入的来源 `Src (P) = [phys, whole, producer, merge, split, K, fused]`，物理值的类型 `Ps: «nis; *»` 作为显式参数（不把类型字段放进记录：那样按 pack 下标在“普通 / 融合”之间选择时两支类型不同，会得到并类型）。
  `build` 内部的 `member i` 是 `srcs#i#6`，成员分支按 pack 下标取该来源的 `producer`、`merge`、`split`、`K`，不再在算出来的位置上取下标；普通构造子传 `plain_srcs`，`fused`/`fused_shared` 在融合槽传生产者（物理值是生产者的输入）。
  验收：9 个用例（`tile_law_exec`、`law_fuse`、`law_roll`、`law_traffic`、两个 softmax 基准、`gpu/law_fuse`、`law_resident_merge`、`law_promote`）重构前后生成的代码去掉 gid 后逐字相同。
  core 前提：`Check` 把元组逐元素 coerce 进依赖元素的 `Arr`（`lit/dep_record_pack.mim`）；带类型字段的来源记录在泛型函数里可用，已用探针验证（数值正确）。
  `fused` 与 `fused_shared` 现在只差物理值与 `producer` 两个字段，但让 `fuse` 走 `fused_shared`（`share = ‹npp; 0›`）仍报 “expected a shape”，根因在 `shared_producer` 的展开，仍未查清，两个构造子暂时保留。
  评审第六轮推测它与 `17cce8fc5`（可变 `Arr` 的 size 1 轴不折叠）有关；检验：在合并版上恢复旧的折叠行为，`law_resident` 报的仍是同一个 “expected a shape”，所以不是它。
- 第 4 项第二步（2026-11-15，部分）：`Src` 加了 `tile`（给定物理值与盒子取该输入的盒子），`values_of` 中未驻留成员的盒子由它给出，生成的代码不变（10 个用例逐字相同）。
  构造子对自己参数的判定（如 `fuse_shared` 的形状检查）放进 `Law` 的字段 `checked`，由 `verdict_of`、`denote`、`par`、`red`、`red_par` 报出，所以错误出现在使用定律的地方；`check.report` 只作用在 `*` 中的张量上。
  定律本身成为来源（2026-11-15/16）：`Law` 加类型参数 `P`（物理输入的类型）与字段 `phys`、`mean`、`merge`、`ordered`。
  `law.fuse_law slot pl …` 在融合槽放生产者定律 `pl`：消费者穿过 `pl` 的物理输入，块读成员的盒子是 `tile_of (mean_of pl, …)`；`pl` 若也融合了生产者，就得到嵌套（`lit/tensor/law_nested.mim`：行和 ← GEMM ← 反量化，调度里没有整个乘积的分配）。
  `law.fuse`/`law.fuse_merging` 现在就是 `fuse_law slot (of_map_reduce[_merging] 生产者 …)`，私有的 `fused` 删了；生成的代码与原来逐字相同（`law_fuse`、两个 softmax 基准、驻留的 GPU 用例等 11 个）。
  起初给 `Law` 加了按解释器的 `leaf` 算盒子的字段 `tile`，结果比 `tile_of` 差：`tile_of` 的规范化把嵌套的 `map_reduce` 组合成逐元素的计算，而 `leaf` 先把成员的盒子物化（驻留时每元素一行平方，`law_fuse` 里整块反量化的权重），所以删了；按内层自己的子调度算盒子留待以后。
  驻留路径同样读 `tile_of (mean_of pl, …)`，内层不会被整个算出（`lit/nvptx/law_resident_nested.mim`：平方 ← 行平方和 ← 归一化，没有一行平方的缓冲）。
  多成员（2026-11-16）：`law.of_sources … Ps ins kinds` 按输入构造定律，每个输入是 `law.source x`（原样读）或 `law.as_source pl`（另一个定律的根），记录 `Input (P) = [phys, whole, merge, split, K, fused]` 不依赖消费者，`of_sources` 在内部补出 `Src` 的 `tile`（`slice` 或 `tile_of (whole p, …)`）与 `producer`。
  `nis` 是字面量时调用者写输入的元组，由 `Check` 的逐元素 coerce 填进依赖元素的 `Arr`；`lit/tensor/law_sources.mim`：GEMM 的两个输入各是一个融合的逐点定律，`denote` 与两种调度都对，调度里两个生产者都没有整块物化。
  起初让 `source`/`as_source` 直接返回带 `Views` 的 `Src`，它们的隐式 `nis`/`Ris`/`Tis` 只出现在 `Views` 里而推不出来，所以把面向用户的记录与解释器内部的记录分开。
  `tile_req` 的定律路由一般化（2026-11-16）：`TileFuse::run_as_law` 由递归的 `law_of` 构造定律——根的输入是组成员就先给成员造定律再 `as_source`，否则 `source`，有成员时用 `of_sources`，所以多个与嵌套的成员都交给定律；所有成员都须被用到。
  定律融合每个成员，所以只有 `control` 在结构上等于 `default_control` 时才这样做；`lit/tensor/tile_fuse_exec.mim` 里 `control = never` 的请求仍由逐级递归实现（`fused=0`）。仍交给递归的：被组外使用的成员（yields）、根自己带担保合并又融合了成员、work-group 链里的归约成员、符号的形状或层级。
  work-item/lane 归约没有显式合并、或切了内层归约循环的请求，组合子能被识别且种子是其单位元时（与 `check.combiner` 同一个 `reduction_law`）也交给定律；否则仍交给递归，由它留下请求并给出合约诊断（`tile_reduction_contracts`），而不是让定律的规则报错。
  `of_sources` 现在把每个 `Input` 带的内层判定（`checked`）取第一个非零的，参数判定失败的内层定律经 `as_source` 融合后外层照样报错（`lit/error/law_sources_checked.mim`）；同一个成员被读两次时定律会算两遍，所以整个请求交给递归。
  带 epilogue 的根（2026-11-16）：`Law` 的 `T` 是 epilogue 之后的元素类型，`Acc` 是累加器类型，`merge` 作用在 `Acc` 上；`build_post`、`law.of_map_reduce_post`、`law.of_sources_post` 多带 `(post, post_maps, post_is)`，原来的构造子是它们带恒等 epilogue 的特例。
  调度的结果交给公理 `law.post_over`，它的规范化是一条交换律：逐点的 epilogue 与分块交换，所以它推进 `tiled` 区域的体（epilogue 输入作为区域输入显式穿过），停在归约折叠之外，即每个 tile 的归约结束处；恒等 epilogue 直接消去；封顶的尾块长度是运行时值，这时不推进，在整个结果上做一次（`lit/tensor/law_post.mim`：GEMM 加偏置、累加器为二元组的行和，除了输出只有每个 tile 的累加器）。
  只有根可以带 epilogue：成员的切分会把 epilogue 用在部分和上，`as_source`/`fuse_law` 的类型因此要求 `Acc = T`。
  统计 lit 里各请求退回递归的原因时，带 epilogue 是第一位（36 次），其后是符号形状（27）、yields（13）、work-group 里的归约成员（12）。
  已识别组合子的真合并（2026-11-16）：`check.merge f` 在组合子 `f` 被识别时规范化成它的二元运算本身，所以普通求和作为驻留成员时也能切到 work-item 上，像担保的合并一样（`lit/gpu/law_resident_sum.mim`）。
  `Law` 多了字段 `splits`：作为成员能否被切分，即合并担保过、或组合子被识别且根直接读输入（融合了源的根，其元素由 `tile_of` 组合，`tile_of_partial` 拆不开），且 epilogue 是恒等的；`as_source`/`fuse_law` 只看它。
  `check.merge` 起初以根张量为参数，选择条件未定时那一支被区域的输入检查拦下；改成以组合子为参数，并显式给出柯里计数 2，否则 `[T, T] → T` 也计入柯里化，只在应用到两个部分和时才规范化。
  还留在图里的 `post_over`（累加器从未成为区域）由 `TileFuse` 与 `lower_tensor` 改写成整块的 `post_tile`；`realize` 的封顶判断沿层级链带着当前块长，内外两层都整除根而内层不整除外层块时也算封顶（`tile_fuse_levels_exec` 的 `cap_inner`）。
  被组外使用的成员（yields，2026-11-16）：定律把它当作普通输入先整块算一次，根与组外的使用者都读它，而不是像 `TileFuse` 那样从各块里交回；`tile_fuse_yield_axes` 的指数仍只算一次。
  现有的 GPU 基准与 `lit/gpu` 用例都没有交回的成员，所以这一步不改变它们；`bench/gpu_fusion/softmax.mim` 在 `tile-fuse=2` 下有交回的指数，但 `TileFuse` 在那里不生成 GPU kernel，两种做法没有可比的 GPU 数字。
  编译时间基线（Release，sm_120 主机，编译到设备代码，三次取中位数）：GEMM 的 SIMT、mma、wgmma 链走定律约 6.5 s，走 `TileFuse` 约 5.8–6.0 s，多 0.6–0.7 s（约 11%）；两个 softmax 驻留版本两条路径相当（5.8–6.5 s）。
  放开之后暴露出解释器原有的一处限制：封顶的并行层里面再有 vec 层时，生成的 `map_reduce` 有运行时长度，向量化的判定定不下来；这种请求先交给递归。
  先前以为这一步卡在 core 的统一上，是误判：那次新增了 `anx` 却只把 `tensor.mim` 复制进构建目录而没重建插件，生成的公理编号错位，C++ 引用的 `law.run` 成了 `law.req`。
- `lit/tensor/tile_law_exec.mim` 在 CPU（ASan）上执行：`denote`、`req`（TileFuse 实现）、`par`、`red`、`red_par`（加法合并）、不带合并的 `red_par` 都等于参考结果，即三条切分引理在这个实例上成立；换成错误的 `merge`（取第一个参数）时 `red_par` 被测出错误；
  `law.schedule` 对 `serial_par`/`serial_red` 和 `serial_group`/`group_item`/`item_red` 两条链的结果等于参考结果，同一条链经 `law.req` 由 `TileFuse` 实现，结果也相同；
  生成的代码不带 ASan 插桩，所以输入后面放了非零哨兵，块越界读会改变结果（把尾块长度改成整块的变异由此被测出）；
  `lit/error/law_req_rule.mim`（规则用定律的事实）、`law_req_kind.mim`（收缩种类不符是类型错误）。

## 8. 类型索引与写在 Mim 里的协议（2026-11-16）

### 8.1 解释器的类型索引

`build_post` 里的解释器原先靠 PE 时的数据区分"现在在哪、手里的东西是谁的"；现在这些是类型索引，写错在插件验证时报错，生成的代码不变（`lit/gpu` 的 88 个程序输出逐字相同）。

- 执行单元：`nom At (u)`，`Step u = [lo, ext, size, xs, acc, grp, stp, pan, here: At u] → …`；`split_par u (…, scope, …, k: Step (inner_unit (u, scope)))`，`split_bound u (…, k: Step u)`。
  五个解释函数各自固定单元（0 串行、1 work-group、2 work-item、3 warp、6 warpgroup）。原先 scope 编号与续延是两个互不相关的参数。
- 累加器：`nom Acc (u) (lo) (ext) = | Seed: «(u, lo)» | Held: [«(u, lo)», Tensor Ro ext To]`，取代 `acc` 张量加 `fresh` 标记。
  种子状态下没有张量，所以不会有"标成种子却带着折过的累加器"而被并行切分静默丢掉。
- 面板计划：`nom Plan (ou) (grp) (stp) = [who: «(ou, grp, stp)», planned: Pan]`，所有者是 `grp` 处的 work-group（`ou = shared_unit u` 为 0 时是串行代码），生命周期是 `stp` 处的暂存归约块。
  只有 `split_par` 进入 work-group 和带 `promote` 的 `split_bound` 能产生另一个所有者或生命周期的计划。
- `Gx` 的执行单元与 work-group 原点两个槽位不再读（恒为 0），改从类型取。

局限：索引加在整个计划上而不是每个输入；它保证的是计划、累加器与上下文配对一致，不保证叶子读的范围落在面板的盒子里（那是仿射包含）；`stage` 的结果仍是裸 `Tensor`。
`bench/typed_tiles/` 是同一思路在解释器之外的外壳：tile 带所有者实例、生命周期、存储空间、可见性，所有者与生命周期用区域的绑定变量（`off`、`k`）索引，所以 tile 逃不出它的 work-group 或归约步。
数据 tile 的类型参照了两处：Mojo 的 `TileTensor`（`max/kernels/src/layout`）把可变性、布局、访问引擎、地址空间和生命周期的来源都放在类型参数里，`vectorize` 是改变类型的视图；cubek 的 `Tile<N, Sc>`（`cubek-matmul/src/multi_level/tile`）把执行单元放在类型里，而"整块暂存区 / 只读的共享 tile / 每单元的寄存器 tile / 张量核片段"是值层面的枚举，配错在展开时 `panic`。
据此外壳的 `Tile r unit at life space cap r2 s T` 把原先的可见性布尔值换成访问能力 `cap: gpu.Vis`：`Unpublished` 只有所有者的协作填充能写，`Published` 所有者内部的单元都能读而无人能写，`Owned` 只有所有者读写。
它直接用降低之后 `gpu.SharedBuf` 的那个 `Vis` 类型，所以调度层与同步层说同一套词；work-group 的累加器是 `Owned`，work-item 只能拿到切给自己的那一块（也是 `Owned`，但属于它自己），面板在填充函数里是 `Unpublished`、到了 work-item 手里是 `Published`。
`vectors (n, c)` 是保留全部索引的向量视图（`tensor.as_vectors`）。
新增的两个反例：work-item 直接读 work-group 独占的累加器、把已发布的面板当成自己的可写 tile（`lit/error/typed_tile_{exclusive,readonly}.mim`）。
布局也进了类型（借 Mojo 的 `LayoutType`）：`Tile … r2 s lay T` 里 `s` 是逻辑形状，`lay#d` 是存在位置 `d` 的逻辑轴，承载的张量是 `Tensor r2 (stored (s, lay)) T`。
`resident_as lay` 按这个置换暂存一个盒子；`read` 给出按存储形状的张量，`stored_at p c` 由类型算出逻辑坐标 `c` 的存储坐标，叶子的读映射经它写，不必记面板是怎么摆的；`vectors n` 沿存储的最后一维取向量，逻辑上缩短的是存在那里的那条轴，不整除时报错（`require`）。
正例的 A 面板改成列主序（k 主序）暂存，在 GPU 上执行正确；反例：列主序的面板被当作行主序交出去、向量宽度不整除（`lit/error/typed_tile_{layout,vectors}.mim`）。
限制：仿射下标没有范围类型，所以绕过 `stored_at` 直接用逻辑坐标去读一个形状碰巧相容的面板不会被拒绝；布局只是轴的置换，`placed` 的 `layout_expr`（分块、swizzle、填充）还不在类型里。
`rows` 与 `stored` 带过滤器，布局是字面量之前保持不展开：每次展开都新造一个带绑定子的 pack，两处的类型就不相等了。
张量核片段与环的槽位还不在这个类型里；解释器里 `stage` 的结果也仍是裸 `Tensor`。

### 8.2 协议骨架

`lower_to_mem` 里一部分同步协议改由 `gpu.mim` 的 Mim 函数给出，C++ 只提供闭包（填充体、一步的计算、合并函数）：

| 协议 | Mim | 用在 |
|---|---|---|
| 同步面板的超步：进入、逐个协作填充、发布、读、离开 | `gpu.superstep` | 批（`close_batch`） |
| 环：序言、每步等待/同步/补填/计算/轮转、收尾 | `gpu.ring_fold`，拷贝的跟踪方式是参数 `RingSync` | `lower_tile_fold`、`warp_fold` |
| 生产者 warp 填、消费者 warpgroup 读的环：两个角色各走环的一侧，只在槽的 mbarrier 上相遇 | `gpu.ws_fold`，同一个 `RingSync` | `lower_tile_fold` 的特化 warp（`RedWs`），含 persistent 与 ping-pong |
| `cp.async` 提交组 | `gpu.async_ring` | 同上 |
| 每槽一个 mbarrier | `gpu.mbar_ring` | 所有 TMA 环 |
| 谁共享这个环：cluster 共享面板、生产者 warp 与它的消费者、cluster、persistent work-group、ping-pong | `gpu.shared_ring`、`consumer_ring`、`cluster_ring`、`resumed_ring`/`fenced_ring`、`pingpong_ring`，各自包一层 `RingSync` | 同上 |
| 两个消费者轮流发 MMA | `gpu.turn_take`、`gpu.turn_pass` | `lower_mma` |
| work-item 归约的合并轮次，令牌与面板的类型数超步 | `gpu.merge_rounds` | `lower_bound_fold` |
| lane 的 shuffle 合并 | `gpu.lane_merge` | `lower_lane_fold`、work-item 归约的 warp 内预合并 |

验证：同步超步、`cp.async` 环、合并轮次、lane 合并的生成结果与原先结构相同或在 GPU 上执行正确，各有一个使 GPU 执行测试失败的变异；`warp_fold` 的环在 RTX 3090 上执行正确（`bench/gpu_fusion/attn_warps.mim`）；TMA 环没有可用的硬件，只比较了 sm_90a 设备 IR（行数与操作集合相同，PTX 指令集合相同）。
`warp_fold` 的环原先结束时不等最后一组拷贝，现在等。
在 H200 NVL（sm_90）上补的验证（2026-11-16）：完整 lit 1005 个测试中 997 个通过、8 个不支持，此前在 sm_75 上固定失败的 12 个都通过，TMA、cluster 共享与拆分、persistent、特化 warp、ping-pong 的环都在硬件上执行正确。
三个变异：`mbar_ring` 不等填充落地，6 个 `nvptx` 测试在硬件上算错；`shared_ring` 不释放槽，两个 cluster 测试挂死；`cluster_ring` 去掉收尾的 cluster 屏障，原有测试全部照样通过，所以三个 cluster 测试现在数设备 IR 里的 cluster 屏障（`CLUSTER-COUNT`）。
仍在 C++ 里的：批之外的协作填充、区域本身（`lower_tiled`，含 persistent 的 tile 轮次与特化区域多出的生产者 warp）、`gpu.outline`；环这一侧 C++ 只剩 mbarrier 的存储、TMA 的装载（`expect_tx` 与各次 `tma_load`）和一步的计算。

槽的状态机现在整个在 `RingSync` 里（2026-11-16）：一个槽依次是空、正在填、满、被读。
`open` 初始化 mbarrier 并让它们对所有参与者可见，`acquire (mem, n)` 等第 `n` 次填充的槽变空，`commit` 结束这次填充发出的拷贝，`await (mem, i)` 等第 `i` 块落地，`release (mem, i)` 交还第 `i` 步不再读的槽，`settle` 等完在途的拷贝。
`ring_fold` 是同一个单元先后走两侧（序言先填 `depth - 1` 块，每步等待、同步、交还、回填、计算），`ws_fold` 是生产者 warp 与消费者各走一侧；persistent 只是填充计数从 `base` 继续，ping-pong 是 `open` 里多一次交出轮次。
"空" mbarrier 的到达数（cluster 的 work-group 数，或每个消费者 4 个 warp）、它的奇偶、谁发起，与交还它的那一方写在同一个定义里，不再是 `lower_to_mem` 里分处三地的算术。
原先 `lower_tile_fold` 里特化 warp 的循环、TMA 环的序言与收尾（约 170 行）删去，剩下的手写循环只服务没有环的折叠。

验收：`lit/gpu` 的 90 个程序里只有带 TMA 环的 12 个输出有变化（初始化移到两次分配之后，以及名字），16 个 Hopper 程序的设备 IR 中内联汇编与 NVVM 指令的多重集合与改动前逐一相同（特化的三条链多一条 `and i1`，是生产者的发起条件在等空槽与装载处各算了一次）；完整 lit 通过。
在 H200 上执行的变异（新增 `lit/{gpu,nvptx}/tile_wgmma_refill.mim`：块数多于槽数的 2 槽环与晚一步交还的 4 槽环，原有的特化测试只有 2 块，生产者从不回填）：

| 变异 | 结果 |
|---|---|
| 消费者不等自己的槽落地 | 5 个特化测试算错 |
| 消费者不交还槽 | persistent 挂死，`tile_wgmma_refill` 失败；原有的 `tile_wgmma_ws`、`tile_wgmma_pingpong` 测不出 |
| persistent 的环每个 tile 重新计数 | 两个 persistent 测试挂死 |
| 生产者不等空槽就填 | `tile_wgmma_refill`、persistent 算错 |
| "空" mbarrier 每个消费者只数 1 次到达 | persistent 与 `tile_wgmma_refill` 挂死 |
| 没人把第一轮交给第一个消费者；把轮次交给自己 | ping-pong 挂死 |
| 消费者交还了错的槽（`(n + 1) % depth`） | **测不出**（两次运行都通过） |
| 共享面板的 cluster 不等空槽就回填 | **测不出**，块数多于槽数的 `tile_wgmma_cluster_refill` 也一样：回填之前本 work-group 已经过了屏障，这次等待只防 cluster 里另一个 work-group 还在读，两者步调一致时竞争不出现 |
| 共享面板的 cluster 不交还槽 | 两个 cluster 测试挂死 |

后两条是执行测不出、而槽的类型应当拦下的那一类：交还的必须是自己读过且异步读者已退休的槽，填的必须是已交还的槽。

消费者一侧的槽令牌（2026-11-16）：`gpu.Held depth (s, live)` 是一步对槽 `s` 持有的东西，只有环的同步能造——`await (mem, n)` 返回 `Held depth (n % depth, tt)`，尚未读过任何槽的一步持有的是 `Held depth (0, ff)`。
`release (mem, s, live, held)` 要的正是 `Held depth (s, live)`：交还哪个槽、是否真的交还，都从令牌的类型读出，同步里不再有 `(n + depth - lag) % depth` 和 `n != 0` 这两处算术。
晚一步交还（`lag = 1`）与 work-group 自己填的环都改成"把上一步的令牌带到下一步"，循环状态多一项 `[live: Bool, s: Idx depth, H (s, live)]`。
两个折叠骨架对令牌类型 `H` 是参数化的（`ring_fold_at`、`ws_fold_at`，公开的 `ring_fold`/`ws_fold` 以 `gpu.Held depth` 实例化它们），所以骨架写不出令牌，只能交还 `await` 给过它的槽；Mim 没有私有构造子，这是用参数性换来的。
令牌是公理而不是 `nom`：带参数的 `nom` 每次应用得到一个新的可变实例，经 `RWPhase` 重建后同一个类型成了两个互不相等的 `nom`（`Held_… ≠ Held_…`），循环状态的类型对不上；公理的应用按结构哈希，`gpu.erase_phases` 在 `lower_to_mem` 之后立刻把类型擦成 `[]`、把值擦成 `()`。
`lit/error/ring_release.mim`（拿自己的令牌交还下一个槽）与 `ring_release_idle.mim`（把没读过任何槽时持有的那个当作读过的交还）是类型错误，`lit/gpu/ring_release.mim` 是正例；上表里"交还了错的槽"的变异现在写不出来。
这次暴露的一处行为差别：persistent 的环原先靠跨 tile 连续的计数，在下一个 tile 的第一步交还上一个 tile 最后读的槽；令牌不跨折叠，带着它结束的折叠会让下一个 tile 的生产者永远等不到那个槽（两个 persistent 测试挂死）。
现在 `ws_fold` 在收尾、异步读者退休之后交还最后持有的槽，所以一次折叠等过的槽都在这次折叠里交还。
没有做的：生产者一侧（`acquire` 之后才能填、填的是等到的那个槽）没有令牌，因为填充的闭包拿到的是轮转后的面板而不是按槽号取的面板；令牌只用一次也没有检查（`check_sync` 不认识它），靠的是骨架只有两处使用。

环的状态在 Mim 里是嵌套元组（每槽一组面板），`affine.lower_for` 之后留下聚合类型的 phi，NVPTX 推断不出共享内存地址空间，叶子的读退化成通用 `ld`（RTX 3090 上慢 17–31%）；所以流水线在 `lower_for` 之后补了两次 `compile.scalarize`，之后与 C++ 版本的 PTX 读写类别相同、计时持平。

`bench/typed_tiles/gemm_buffers.mim` 是不经过 `lower_to_mem` 的可行性检查：16³ GEMM 整个写在缓冲区层，循环是函数（`steps` 加 `gpu.bind.*`），work-group 的协议是 `gpu.superstep`，填充与叶子是普通的缓冲区读写；它走流水线的后半段（`check_sync`、`erase_phases`、`gpu.outline`、nvptx），outline 成 4 组 × 4 项的一个 kernel，在 GPU 上执行正确（`driver_gemm.cpp`）。
它的填充和叶子是标量的（没有向量读写，没有寄存器块），那部分仍是 `lower_to_mem` 与 `btensor.lower_map_reduce` 的工作；要让解释器直接产出缓冲区层，叶子与填充应当是缓冲区上的 `btensor.map_reduce_post`，还没有做。

`bench/typed_tiles/gemm_buffers_mr.mim` 把上面程序的叶子换成缓冲区上的 `btensor.map_reduce`、填充换成向量视图（`buffer.vectorize`）上的协作循环：PTX 与定律版本的全局读、共享写、`fma` 数相同，只剩叶子读面板是标量。
`bench/typed_tiles/buffer_interp.mim` 是缓冲区层解释器的原型：`run_buffers` 取 `map_reduce` 的参数、目标缓冲区和同一条 `TileLevels` 链，并行类层级（`Par`、`Group`、`Item`）是按 tile 的循环并绑定到层级指明的单元，叶子是把自己那块 tile 就地写进目标缓冲区的 `btensor.map_reduce`；有序归约层级（`Red`）按序折叠它的块：第一块从种子开始，其余各块以目标缓冲区为初值就地累加（`btensor.map_reduce_post` 的累加器描述 `(形状, 读映射, 缓冲区)`），块长不整除时最后一块取余量；work-group 内带 `promote` 的归约层级把 work-group 的块读到的每个输入的盒子暂存进共享内存：叶子是一个 `gpu.superstep`，填充是按元素的协作循环，读是在面板上的 `map_reduce`；盒子用 `affine.image` 算，由单位斜率驱动的轴取那个循环的块长以保持字面量。
向量层级把叶子的调度设成它指明的向量维（`btensor.mk_sched (vdim, 0)`）。
一条层级要么提升全部输入要么都不提升（按掩码取子集又会碰到异构下标）；只有一个 tile 的绑定层级不成循环，`gpu.outline` 不接受。
每个叶子的超步要有自己的构造编号，所以解释器沿途传一个站点编号（种子块与循环体是两处静态实例）。
在缓冲区层并行切分不需要累加器和拼接：目标缓冲区一路传下去，各叶子写互不相交的块。
1024³、链 `serial_group ((64, 64, 0), …, group_item ((4, 4, 0), …, item_done ()))`，GTX 1650 SUPER：`run_buffers` 6.40 ms，`law.run` 经 `lower_to_mem` 8.61 ms，结果都正确（`bench/gpu_fusion/gemm_levels_{buffers,law}.mim`，`ptx_bench.py`）；后者为每个 work-group 的结果另分配缓冲区再拷回。
加一层 `group_red ((0, 0, 16), 0, 3, …)`（A、B 的盒子进共享内存）：`run_buffers` 1.33 ms，`law.run` 3.19 ms（`gemm_staged_{buffers,law}.mim`）。
叶子再带向量层级 `item_vec ((1, 0, 0), …)`：2.34 对 2.50 ms（`gemm_vec_*`）。
归约层放进 work-item 里（`group_item (…, item_red ((0, 0, 16), 0, 3, item_vec (…)))`，调好的配置用的链形）：`run_buffers` 2.32 ms，`law.run` 1.60 ms（`gemm_inner_*`）。
这里旧路径更快：`lower_to_mem` 给每个 work-item 一份私有累加器，跨归约块留在寄存器里；`run_buffers` 原先每个块都把结果写回全局的目标缓冲区。
给 work-item 内的归约层加上私有累加器之后（块折进 work-item 自己的缓冲区，折完写回一次；kernel 里没有 `malloc`，累加器进了寄存器）是 2.02 ms；剩下的差距在面板：`run_buffers` 的填充与读都是标量，`lower_to_mem` 是向量填充、向量读和搜索出的布局。
试过把填充改成经 `buffer.vectorize` 视图一次拷 4 个元素（解释器跟踪各循环的块原点是否 4 对齐）：填充成了 `ld.global.v4`/`st.shared.v4`，结果正确，但没有变快——这条链 2.01 ms，前两条链反而变慢（1.34 → 1.58 ms，2.32 → 2.65 ms），原因没有查清，所以没有保留。
这些链离调好的配置（0.72 ms，带剥离的尾块、更大的寄存器预算与向量面板读）还远；这里比较的只是同一条链的两条实现路径。
这次暴露了 `gpu.superstep` 的一个问题并已修：带绑定子的 pack 每次取元素都重新实例化，原先每个 `fills#i` 被取两次，同一个填充成了两份函数，令牌被用两次；现在每个填充只取一次，面板先放进同一擦除类型的槽里，全部填完再转回。
`ptx_bench.py` 靠尝试排列确定 kernel 的参数顺序，错误的排列会让 kernel 写进输入；现在每次尝试前恢复输入。

编译时间（2026-11-16）：对 12 个带调度的 `lit/gpu` 程序（`-p opt -X tensor:tile-fuse=1 … -p gpu`，各 3 次取中位数），本节之前的编译器（`3ff1594b60`）合计 85.1 s，现在 77.7 s，各程序都在 0.89–0.94 倍。
两份都是 Debug 构建、配置相同，每个程序恰好差约 0.6 s，更像固定的启动开销差异，所以结论只是类型索引与写在 Mim 里的协议没有带来可测量的变慢。

### 8.3 写这些时碰到的限制

| 现象 | 做法 |
|---|---|
| 带参数的 `nom`，其索引若不出现在载荷里，不参与类型相等（`nom P (n: Nat) = | Mk` 时 `P 2` 可传给要 `P 1` 的函数） | 像 `Tensor` 一样放单例见证 `«(…)»` |
| 异构元组按非字面量下标取元素得到并类型 | 隐式参数写成一组一次给全；只用 pack 自己的绑定子做下标；顺序线程化令牌时折叠同类型的 `W → W` 步 |
| 布尔索引的依赖类型不能分情况消去（两个 lambda 按 `f` 选择得到并类型） | "是否种子"用和类型，在 PE 时 `match` |
| 类型里的超步编号每轮加一，`e - 1 + 1` 与 `e` 在泛型下不相等 | `core.bitcast` 到声明的类型，`e` 为字面量后消去 |
| 柯里化 `lam` 上的过滤器只管最后一组参数，递归会耗尽 PE 燃料 | 递归函数只用一组参数 |
| 柯里化参数里的组绑定 `(a b: T)` 是一个 `«2; T»` 参数 | 每个参数各写一个括号 |

实现中碰到的 MimIR 限制，决定了现在的形式：

| 想要的写法 | 问题 | 现在的做法 |
|---|---|---|
| `nom Law` 带类型值字段 `I: *` 与 `ins: I` | 复查（2026-11-14）：最小形式现在可用——`[I: *, ins: I, f: I → Nat]`（宇宙是 `□`）在泛型使用者里 `(b#2) (b#1)` 能通过并正确求值 | 仍用闭包；记录里放类型字段要连同下一行一起解决才有意义 |
| `Law` 以 `I` 为参数 | 根因已查清：注解 `Box «i: nis; Tensor (Ris#i) …»` 只建一次可变 `Arr`，代换进记录的两处；值一侧是两次独立构造的 `Arr`。`Check::alpha` 的绑定表全局一致，期望侧的绑定子配了第一个 `Arr` 之后就不能再配第二个 | 同上。试过两种修法都已撤回：让第二个配对与第一个比较（双向绑定互相追逐，需余归纳，memo 不可靠）；在本次比较内临时改绑（类型检查通过，但 `outer_rows` 在 `outer-rows=0` 时编译崩溃——放宽的 `alpha` 让依赖它去重的 phase 复用了构造中的 mutable）。要修需先审查这些 phase 对 `alpha` 的用法 |
| 结构化 `Law (r, s, T)` 上按名取字段 `l#root` | 字段名不经类型函数传到 `#` | 按位置取，模块给出具名的访问函数 |
| `law.req` 把层级交给 `tile_req` | 层级经过 RWPhase 重建后成为结构相同、指针不同的 nom 实例，`bitcast` 不能归约掉 | 经 `core.bitcast` 转换，TileFuse 剥掉它；`tile_req` 的 normalizer 见到它就跳过（定律已用自己的事实检查过） |
| `check.report` 在 `law.req` 的定义体里 | normalizer 在定义时就看到未定的判定 | 判定含自由变量时不归约 |
| 叶子里用计算出的面板计划 `boxed (…)` | 与 `Law` 以 `I` 为参数同源：代换把计划里嵌套的可变 pack 复制成多份，`alpha` 不能把同一个绑定子配给两份 | 叶子是带过滤器 `@(core.pe.is_closed (slot, Ris))` 的具名 `lam`，泛型时不展开，实例化后秩是字面量才展开 |
| 生产者读消费者暂存的视图（别名） | 已做到（2026-11-15）：按 pack 自己的绑定子取下标的类型现在由 `World::generalize` 推出（不再是并类型）；`share` 放在消费者参数之前，实例化 `is` 时它已是字面量 | `law.fuse_shared` 与 `fused_shared`；同一构造子里兼顾不共享的分支仍受 arity 1 依赖 `Arr` 的折叠所限 |
| 在 `vec.fold` 的体里写 `(ins#i)#6₇`，`ins: «i: nis; Input (Ps#i) …»`、`i: Idx nis` | 报 “index does not fit within arity 1”，位置指向无关的定义；根因：`vec.fold` 的下标在异构记录元组上取出的是 `Join` 类型（XFAIL 用例 `lit/bug/fold_join_proj.mim`） | 先用 pack 解构出 `‹i: nis; let (…, c) = ins#i; c›`，再折叠 |
| 物理输入是 `«nis - 1 + npp; …»`，按下标拼接 | `npp` 先成为字面量时，异构元组按非字面量下标取出的类型是并 `T ∪ U`，`bitcast` 推断的源类型不再匹配 | 穿过二元组 `(消费者的输入, 生产者的输入)`，只按 pack 自己的绑定子取下标 |

前三条与 D6 的阻塞同源：依赖类型的实例化与比较。它们是继续推进第 3 步（TileFuse 在定律上解释调度）之前最值得在 core 里修的地方。

## 9. 义务由谁承担（2026-11-17）

调度只应影响性能而不影响含义：定律是顺序规范，三条切分引理管"怎么切"，剩下的正确性是每次读都读到规范里的那个值。
它分成何时、何处、谁三个问题；下表按错误类别列出现在由谁拦下、对应的测试，以及还没有承担者的格子。
"参考实现"一栏出自对 CUTLASS、Mojo MAX kernels、ThunderKittens、cubek、tilelang、cuda-oxide 源码的一次调研（2026-11-16），没有逐条复核，只用来说明这一格在别处通常是静态、动态还是不查。

| 问题 | 错误类别 | 现在的承担者 | 测试 | 参考实现 |
|---|---|---|---|---|
| 切分 | 层级嵌套不合法、操作数种类或收缩种类不符 | 层级类型 | `lit/tensor/tile_levels_scope.mim`、`lit/error/law_req_kind.mim` | cubek 的 scope 是幻影类型，变体与 scope 的搭配在展开时 `panic` |
| 切分 | 整除、尾块、容量、目标能力 | PE 判定（`check` 规则，事实取自定律） | `lit/error/law_req_rule.mim`、`law_capacity.mim` | 多为运行时断言 |
| 切分 | 乱序切分一个不可交换或种子非单位元的归约 | PE 判定（`check.combiner`、定律的合并合同） | `lit/error/law_ordered_inner.mim`、`law_items_combiner.mim`、`law_red_inner.mim` | 不查 |
| 谁 | 续延在错的执行单元里运行，累加器或面板计划配错所有者或生命周期 | 解释器的类型索引（`At`、`Acc`、`Plan`） | 8 个变异在插件验证时被拒（§8.1） | ThunderKittens 的 `group<4>` 静态拒绝在错的单元上调用，但不证明线程属于该组；其余是运行时角色 |
| 谁 | tile 逃出它的 work-group 或归约步，work-item 读独占的累加器，写已发布的面板 | `tensor.Tile`，解释器与外壳共用（§9.5） | `lit/error/typed_tile_{step,unit,lifetime,seed,owner,exclusive,readonly}.mim` | Mojo 的可变性在类型里，共享内存的生命周期是 `UntrackedOrigin` |
| 何时 | 同步面板：填充与读之间没有屏障，令牌或面板用了不止一次，构造进入两次 | `gpu.GM`/`gpu.SharedBuf` 的相位类型加 `gpu.check_sync` | `lit/error/superstep_fill.mim`，`check_sync` 的用例 | 不查 |
| 何时 | 环：交还了没读过的槽或别的槽，提交了没等到空的槽或别的槽 | `gpu.Held`、`gpu.Vacant` 令牌与对令牌参数化的骨架 | `lit/error/ring_release.mim`、`ring_release_idle.mim`、`ring_commit.mim` |
| 何时 | 环：同一个槽交还两次或填两次 | `gpu.check_sync` 对槽令牌计数（§9.6） | `lit/error/ring_release_twice.mim`、`ring_fill_twice.mim` | 奇偶位是裸整数，不查；Mojo 的线性句柄只保证 `release` 被调用一次 |
| 何时 | 环：到达数与参与者数、奇偶、发起者不一致 | 写在同一个 `RingSync` 定义里（按构造一致，不是检查） | 硬件上的变异（§8.2 的表） | 手工对齐 |
| 何时 | 环：深度、交还滞后与在途 MMA 数不相容 | PE 判定（`pipeline_plan.ring_safe`，枚举环的状态） | `lower_to_mem` 里的报错 | 不查 |
| 何时 | 环：不等落地就读、不等空槽就填、不交还、计数不延续 | **只有执行**：硬件上的变异测得出前三类中的大部分 | `lit/nvptx/tile_wgmma_{ws,persistent,pingpong,refill}.mim` | 不查 |
| 何时 | 环：共享面板的 cluster 不等空槽就回填 | **无**：是 work-group 之间的竞争，执行测不出（`tile_wgmma_cluster_refill` 的块数多于槽数，变异照样通过） | — | 不查 |
| 何处 | 列主序的面板被当作行主序，向量宽度不整除 | `Tile` 的布局索引（`layout_expr.Expr`） | `lit/error/typed_tile_{layout,vectors}.mim` | CuTe 与 Mojo 的布局代数在类型里，静态形状下可查 |
| 何处 | 布局不是单射、向量填充不连续、swizzle 与描述符不一致 | PE 判定（`layout_expr` 的 F₂ 解释） | `lit/tensor` 的布局用例 | Mojo 的 `TwoLevelLayout` 把 swizzle 当 F₂ 线性布局；CUTLASS 里 swizzle 不是 `Layout` |
| 何处 | `wgmma` 操作数面板的布局不是描述符要硬件读的 128 字节 swizzle | PE 判定：`layout_expr.lies_sw128`（任意秩，F₂）在造描述符处检查，K 主序与 MN 主序都查 | `lit/tensor/layout_f2.mim`；把 `tc_layout` 的 swizzle 组数改错，编译报错 | 三处各传一个枚举 |
| 何处 | TMA 把面板写到的地址不是面板布局说的地址 | PE 判定：`layout_expr.lies_tma128` 在发出 TMA 加载处检查（§9.6） | `lit/tensor/layout_f2.mim` | 同上 |
| 何处 | 读落在暂存的盒子之外，或绕过布局用逻辑坐标读面板 | **无**：仿射下标没有范围类型 | — | 不查；Descend 用"只能经 view 访问"拦下 |
| 实现 | 实现不等于定律 | 差分测试（CPU 上与参考结果比较，GPU 上执行） | `lit/tensor/tile_law_exec.mim` 等 | — |

读这张表的三点：

- 类型承担的是"配对"——谁的、何时的、哪个槽的——而不是数值；数值一律是 PE 判定。两者都不需要用户写证明。
- 环的两侧只在 mbarrier 上相遇，所以一侧的令牌不能证明另一侧的事实：消费者的 `Held` 说明它等过，不说明生产者填的是对的槽。跨角色的那部分由 `ring_safe` 的状态枚举承担，它现在只覆盖深度与滞后。
- 还没有承担者的几格是接下来最该去的地方：生产者一侧的槽；MN 主序面板与 TMA 描述符的 swizzle；访问只能经过布局。
  `lies_sw128` 的陈述是"布局等于 128 字节 swizzle 作用在它自己去掉 swizzle 的摆放上"（Hexcute 的 `M = S∘m`），不依赖逻辑轴叫什么，所以带单位轴的三维面板也能判；原有的 `matches_sw128` 只管二维，实际的操作数面板逻辑形状是 `(64, 1, 128)` 这样的三维，所以它此前没有接到造描述符的地方。
  最后一条对 `placed` 的面板其实已经成立：叶子用逻辑坐标读，坐标到地址的映射由 lowering 按布局值做，绕不过去；绕得过去的只是外壳里"按置换暂存、叶子自己写存储坐标"的那种面板（`resident_as` 加 `stored_at`）。
  所以外壳的布局索引下一步应当是 `layout_expr.Expr` 本身而不是轴的置换，`stored_at` 随之不再需要。

### 9.1 第 2、3 步动手前查清的两件事（2026-11-17）

- 外壳 `Tile` 的布局索引不能直接换成 `layout_expr.Expr`：`layout_expr.make` 要求秩至少为 2（它描述的是共享内存里的面板），而外壳的 tile 也包括秩 1 的累加器（`rowsum_block_par_steps.mim`）与寄存器里的 tile。
  所以"布局"只是面板这一类 tile 的索引；要么 `Tile` 的布局索引是"行主序或一个 `Expr`"的和类型，要么面板另有自己的类型。
  前者会碰到 §8.3 的并类型限制，后者是第四套词汇，两者都需要先定下来再改 13 份带外壳的文件。
- 解释器里 `stage` 的结果包成带所有者与生命周期的名义类型，按现在的结构拦不下新的错误：`values_of` 的结果在 `leaf_at`/`vec_at` 里当场交给 `map_reduce_post`，产生与消费相隔两行，所有者与生命周期已经由 `Plan` 的索引管住。
  真正没有被管住的是同一个输入的读映射与值是否配对：`reads_of`、`values_of`、`ris_of`、`sis_of` 各自按 `split_of pan i` 与 `q#0₆` 选一支，再用 `core.bitcast` 对齐类型，四处选错一处类型上看不出来。
  所以第 2 步更有价值的形式是把每个输入的（读映射，值，秩，形状）作为一个整体由一次选择给出，去掉这些 `bitcast`；它要改的正是 `values_of` 一带，与另一条线上未提交的改动重叠。

### 9.2 读映射与值配对（2026-11-17）

§9.1 的第二条做了：`build_post` 里原来的 `reads_of` 与 `values_of` 合成 `inputs_of`，每个输入的（读映射，值）由一个带秩与形状的函数 `read rk sk (map, value)` 成对给出——映射到 `rk` 个坐标，值是 `Tensor rk sk T`——四种情形（原样读、成员的盒子、整块暂存、面板）各调用它一次。
把面板的映射配给整块暂存的值、或把普通映射配给面板，现在是插件验证时的类型错误；此前这两处各自按计划的标志选一支，再用 `core.bitcast` 对齐到 `ris_of`/`sis_of` 给出的类型，配错在类型上看不出来。
剩下的 `bitcast` 只有两类：`read` 里对齐到 `ris_of`/`sis_of` 的那一次（计划成为字面量后两边相等而消去），和整块暂存的结果（`stage` 在 `k = 0` 时的类型是 `Tensor (r + 0) …`，泛型下不化简）。
验收：`lit/gpu` 的 93 个程序里 89 个输出逐字不变，4 个只差一个变量名，去掉名字后的结构相同；完整 lit 通过。
没有做的仍是把结果包成带所有者与生命周期的 `Tile`（理由见 §9.1），以及 `ris_of`/`sis_of` 与 `read` 给出的秩和形状是否一致——它们是同一套标志的两份展开，现在靠那一次 `bitcast` 在实例化后相等来体现。

### 9.3 暂存的值带上计划的所有者与生命周期，面板的布局进类型（2026-11-17）

- 解释器：`inputs_of` 给叶子的每个值现在是 `Kept ou grp stp r s T`，所有者与生命周期取自计划的索引；叶子（`leaf_at`、`vec_at`）拿的是 `Plan ou grp stp` 而不是裸的 `Pan`，只有持有同一个计划才能用 `kept_by` 取出张量。
  让叶子读为另一个生命周期准备的值是插件验证时的类型错误；`lit/gpu` 的输出与改动前逐字相同。
  这一层按 §9.1 的判断拦下的错误不多（值的产生与消费相邻），它的作用是让"叶子读的东西属于谁、活多久"在类型里有名字，与外壳的 `Tile`/`Panel` 用同一组索引（所有者单元、所有者的原点、生命周期）。
- 外壳：面板另立类型 `Panel r at life cap rx s lay T`，`lay` 就是 `layout_expr.Expr (rx, s)`，即编译器自己的面板带的那个布局值（`Tile` 的布局索引不能直接换成它，见 §9.1）。
  `resident_placed lay` 用 `placed` 暂存，读者经 `read_panel` 拿到逻辑形状的张量、用逻辑坐标读，坐标到地址由 lowering 按 `lay` 映射，所以不存在绕过布局的读。
  `bench/typed_tiles/gemm_panel_placed.mim`（`lit/{gpu,nvptx}/typed_panel.mim`）：A 的面板按末维填充 4 的布局暂存，设备 IR 里是 `[8 x [12 x float]]`，在 H200 上 256 个元素全对；把一种布局的面板当另一种交出去是类型错误（`lit/error/typed_tile_panel_layout.mim`）。
- 三套词汇的现状：解释器（`At`、`Acc`、`Plan`、`Kept`）与外壳（`Tile`、`Panel`、`Ctx`）用同一组索引但仍是两处定义；外壳的访问能力与 `Panel` 的布局直接用 `gpu.Vis` 与 `layout_expr.Expr`，解释器因为 `tensor` 不导入 `gpu` 而没有能力这一项。
  要让 `Kept` 也带 `gpu.Vis`，得先把 `Vis` 放到两个插件都能看到的地方。
- 之后做了这一步：`Vis` 挪到 `buffer` 插件（`buffer.Vis`，`tensor` 与 `gpu` 都导入它），`gpu.Vis` 是它的别名，外壳与同步令牌的写法不变；`Kept` 多了能力索引，`inputs_of` 给叶子的都是 `Published`，`kept_by` 只接受 `Published`。
  `lit/gpu` 的输出与改动前逐字相同，完整 lit 通过。
  所以访问能力现在是一个类型，三处共用；所有者、生命周期这组索引在解释器与外壳里仍是两处定义。

### 9.4 改动前后的计时（2026-11-17，H200 NVL）

基线是交接时的提交 `a10d52b44`，与现在的工作树用同一配置各建一份 Debug 构建。

Hopper 链的 kernel 时间（1024³ F16 GEMM，`bench/gpu_tiling/gemm_tiled_wg_*.mim`，完整 host 程序加 nsys，各 3 次，单位 ms，结果都正确）：

| 链 | 基线 | 现在 |
|---|---|---|
| `wg_ws`（特化 warp，4 槽环） | 0.0180 / 0.0179 / 0.0179 | 0.0182 / 0.0181 / 0.0182 |
| `wg_persistent` | 0.0179 / 0.0179 / 0.0179 | 0.0180 / 0.0180 / 0.0179 |
| `wg_tma` | 0.0126 / 0.0126 / 0.0126 | 0.0127 / 0.0126 / 0.0126 |
| `wg_cluster`（共享面板） | 0.0235 / 0.0236 / 0.0236 | 0.0217 / 0.0218 / 0.0217 |
| `wg_split` | 0.0138 / 0.0138 / 0.0138 | 0.0138 / 0.0138 / 0.0138 |

特化 warp 的链慢约 1%，cluster 共享面板的链快约 8%，其余持平；两处差别的原因都没有查。

编译时间（12 个带调度的 `lit/gpu` 程序，`-p opt -X tensor:tile-fuse=1 … -p gpu`，各 3 次取中位数，Debug 构建）：合计 80.5 s 对 94.5 s，每个程序都多 1.1–1.2 s，与程序内容无关。
这是加载 `tensor` 插件的固定开销：单独 `plugin tensor;` 从 5.3 s 到 6.4 s，按保存的各版本 `tensor.mim` 逐个计时，增加的 1.1 s 全部出自 §9.2 的 `inputs_of`（映射与值配对），`Kept` 与 `Vis` 没有可测的增加。
把 `read` 写成一组参数、或把同秩的三种情形先选后配，都没有降下来；它是泛型下检查四次带依赖类型的应用的代价，原来的 `bitcast` 不检查。
没有 Release 构建的数字。

### 9.5 一个 `Tile`，布局索引是 `layout_expr.Expr`（2026-11-17）

§9.1 与 §9.3 里搁下的两件事按计划的字面做了，取代 §9.3 的 `Kept` 与 `Panel`。

- 定义只有一处，在 `tensor.mim`：
  `Tile r unit at life space cap rx s c lay T`，`at`、`life` 都是 `«r; Nat»`，`cap` 是 `buffer.Vis`，`lay: layout_expr.Expr (rx, c)` 是容量 `c` 上的布局，也就是 `placed` 的参数；数据一律是逻辑形状 `Tensor rx s T`。
  `RowTile … rx s T` 是行主序（`layout_expr.rows s`）的缩写，`tile_as`、`row_tile`、`row_data` 是构造与取值，`units`、`spaces` 给出单元与内存空间的编号。
- 解释器：`inputs_of` 交给叶子的值就是这个 `Tile`，所有者 `(ou, grp)`、生命周期 `stp` 取自计划，空间由 `space_of` 给出（暂存的在共享内存，融合成员的盒子在寄存器，其余在调用方给的内存），容量与布局由 `cap_of`、`lay_of` 给出：放置过的面板是布局搜索或张量核给的那个值，其余是 `rows`。
  `Kept` 删掉了，`kept_by` 改为接受 `Published` 的 `Tile`。
- 外壳：`bench/typed_tiles/` 各文件不再自己定义 `Tile`，生命周期写成 `‹r; e›`。
  列主序不再是 `stage` 的轴置换加 `stored_at`，而是一个 `layout_expr.make` 值经 `placed` 暂存，读者用逻辑坐标；`stored_at`、`stored`、`Panel`、`resident_placed`、`read_panel`、`threads_placed` 都删掉了，`resident_as lay` 接受任何布局。
  `vectors` 只沿逻辑末维，并要求布局把每个向量存成连续的一段（`layout_expr.contiguous` 不为 1）；原来"列主序面板沿列取向量"的 `column_vectors` 在逻辑视图下写不出来，删掉了。
- 秩小于 2：`layout_expr.make` 仍要求秩至少为 2，新增的 `layout_expr.rows` 直接构造行主序的值，秩 2 起与 `make` 给出的相同，所以秩 1 的累加器也有布局索引。
  （第一版在秩 3 及以上给错了置换，见 §9.13。）
  它要到形状是字面量才展开：展开得早，每次展开各造一个包，同一个布局在两处就成了两个类型（一开始 `seed` 的结果与 `steps` 的参数因此对不上）。
- 到 `gpu.SharedBuf`：`tensor.mim` 用 `import gpu;` 取 `gpu` 的声明（像 `btensor` 对 `ll` 那样只取声明；写成 `plugin gpu;` 会把 gpu 的阶段带进 CPU 程序，ll 后端报错），于是 `Tile` 旁边的 `Lowered (g, e) tile = gpu.SharedBuf (g, e, cap, rx, layout_expr.shape lay, T)` 直接引用 `gpu.SharedBuf`，瓦片的类型决定它的面板的类型。
  `lit/gpu/tile_shared.mim` 用它检查：`Filling` 的瓦片的面板经 `publish_view` 得到的正是 `Seen` 的瓦片的面板，`view` 给出布局存储形状的缓冲（列主序的 `(8, 16)` 是 `(16, 8)`），`claim` 得到 `Mine` 的；`lit/error/tile_shared_unpublished.mim` 是对未发布的面板取 `view`。
  方向只能是 `tensor` 引用 `gpu`：`gpu.mim` 引用 `Tile` 就得导入 `tensor`，而 `tensor` 的 C++ 依赖 `gpu` 生成的头文件。
  `SharedBuf` 自己仍是 `gpu` 里的公理类型（lowering 之后还要存在，带参数的 nom 过不了 `RWPhase`）。
- 没有做到的：解释器的 `At`、`Acc`、`Plan` 与外壳的 `Ctx` 仍各自定义（它们是上下文，不是数据）；未放置的暂存在类型里写的是 `rows`，而 lowering 之后仍可能给它选一个布局，所以 `rows` 的含义是"没有指定布局"，不是"保证行主序"。

§9.5 之后的编译时间（Debug，各 3 次取中位数，单位 s；Hopper 的 kernel 时间没有重测）：

| | 基线 `a10d52b44` | 现在 |
|---|---|---|
| 只加载 `plugin tensor;` | 5.31 | 7.68 |
| `lit/gpu/typed_tiles.mim` | 5.51 | 7.80 |
| `lit/gpu/tile_wgmma_ws.mim` | 6.07 | 8.52 |

加载 `tensor` 插件的固定开销从 §9.4 的 6.4 s 又涨到 7.7 s，即统一 `Tile` 这一步再加约 1.3 s，相对基线每次编译共多约 2.4 s（+45%）。
多出来的部分没有逐项归因，最可能来自 `inputs_of` 的类型里现在带着 `space_of`、`cap_of`、`lay_of`（后者含布局搜索的调用）。

Hopper 的 kernel 时间在 §9.5 之后没有重测成：重测时容器已经换到 RTX PRO 6000 Blackwell 的主机上，`sm_90a` 的 kernel 在它上面加载失败（`CUresult 209`）。
§9.4 的表仍是 §9.5 之前的数字；§9.5 之后的两次全量 lit 是在哪块 GPU 上跑的没有记录，重测那一刻是 Blackwell。

### 9.6 生产者令牌、令牌只用一次、MN 主序与 TMA 的布局检查（2026-11-17）

本节的改动都是在 RTX PRO 6000 Blackwell 的主机上做的：`sm_90a` 的 kernel 在那里不执行（lit 里以 `cuda-sm90` 为条件），所以验证是全量 lit 加设备 IR 的比较，没有硬件上的执行。

- 生产者一侧：`acquire` 返回 `gpu.Vacant depth s`，`commit (mem, s, vacant)` 消耗它；骨架对 `V` 也是参数化的，所以它提交的只能是等到空的那个槽（`lit/error/ring_commit.mim`）。
- 令牌只用一次：原来的令牌在部分求值后不留痕迹（`release` 的最内层是恒等），相同下标的令牌又会被合并成一个节点，没法计数。
  现在 `hold`、`vacate` 带着内存线索（各自是一个值），最内层的交还与提交是公理 `retire`、`occupy`，`check_sync` 对每个槽令牌数使用次数，超过一次就报错，`erase_phases` 把四者都擦掉。
  检查的是"至多一次"：`ring_fold` 结束时最后一个槽仍被持有，不交还。
  10 个环形程序加令牌前后的设备 IR 除值的名字前缀外逐行相同。
- MN 主序：`lies_sw128` 的原子改为 `128 / bytes`，不再取摆放的末维，于是与面板朝哪边为主无关；lowering 对两种朝向都查，现有程序的判定都是"成立"。
- TMA：`lies_tma128 (layout, bytes, row, col)` 说的是硬件的写法——一行切成若干 128 字节的原子，逐个原子作为一个盒子堆叠，盒内做 128 字节 swizzle——给出的地址等于面板布局的地址；`row`、`col` 是矩阵的行、列落在面板的哪个逻辑维，由 `stage` 的参数算出。
  发出 TMA 加载的地方原来按 `a · rows · 128` 字节直接算目的地址，这个假设现在有判定核对；K 主序与 MN 主序的面板判定都是"成立"，行列对调或去掉 swizzle 是"不成立"。
- 仍没有承担者：共享面板的 cluster 不等空槽就回填（work-group 之间的竞争，要在 H200 上先做出能复现的测试）；读落在暂存的盒子之外。

### 9.7 还留在 C++ 里的协议搬进 Mim（2026-11-17）

按价值排的六项，各一个提交；C++ 留下的是"哪个盒子、哪块缓冲、哪条指令"，顺序与配对在 Mim 里。
验证同 §9.6：全量 lit 加改动前后设备 IR 的比较；其中第 5 项的测试不依赖 `sm_90a`，在当时的 GPU 上实际执行过，其余没有硬件执行。

| 协议 | 现在在 Mim 的定义 | C++ 留下的 | 改动前后的设备 IR |
|---|---|---|---|
| TMA 填充 | `gpu.tma_fill`：mbarrier 期望的字节数是各面板字节数之和，随后逐原子发出拷贝，multicast 的份额偏移也在这里 | 每个面板的基址、描述符、起点与行列数 | 逐行相同 |
| 环的同步由哪几层组成 | `gpu.tma_ring`：由 cluster、是否共享、生产者与消费者数、ping-pong、是否持久化决定套哪些 `*_ring` | 这些标志的取值 | 逐行相同 |
| wgmma 的一组 MMA | `gpu.wgmma_batch`：发布到 async 代理、`wgmma.fence`、逐条指令、`commit`、轮转的交接，累加器只从 `wait` 的结果里拿回来 | 每条指令的操作数与描述符 | 有副作用的指令顺序相同，每个 kernel 少 18 条重复的地址计算 |
| wgmma 描述符的字段 | `layout_expr.wgmma_fields`：MN/K 主序、LBO、SBO、swizzle 都从面板的布局值算出 | 取地址 | 逐行相同 |
| work-item 绑定归约的合并 | `gpu.merge_pairs`、`merge_first`、`merge_serial`，作为 `gpu.merge_rounds` 的各轮与读阶段 | 每个 work-item 的部分和怎么算、部分和的元素怎么读写、结果元素写到哪里 | 未比较；`lit/nvptx/tile_reduce_merge.mim` 等执行通过 |
| 持久化 work-group 的轮次 | `gpu.persistent_tiles`：第 `round` 轮处理 `id + round · ctas` 号瓦片，越界则跳过，轮末 `settle` | 一块瓦片的计算、轮末是否加 `fence` | 只有两个标签名不同 |

没有搬的：

- `warp_fold` 里环的深度仍写死为 2、同步写死为 `async_ring`，不带流水线的那条路仍是手搭的计数循环；它给 wgmma 的 B 面板选布局仍调 C++ 的 `panel::wgmma`，与 Mim 的 `tc_layout.wgmma` 是两份。
- `lower_tile_fold` 里不带环的折叠循环、`lower_product_fold`、`lower_lane_fold` 的循环骨架。
- 找哪些面板是暂存的、哪些 A 片段可以提到循环外，这些是对 IR 的搜索，不是协议。

合并骨架第一版把"第 `t` 份部分和的第 `e` 个元素在哪"也写在了 Mim 里，按缓冲的类型造完整下标；记录型元素的两个程序（`lit/gpu/law_roll.mim`、`law_attention.mim`）因此编译失败，而当时只跑了没覆盖它们的子集就推送了。
原因是缓冲的下标约定：长度为 1 的轴整个不出现在下标里（`Shape::fold`），这在 Mim 的类型里写不出来。
现在读写部分和由 C++ 给的 `get`、`set` 完成，骨架只有循环与合并。

写的时候碰到的两件事：`tensor` 插件的 annex 标签已到上限 256（顶层名字各占一个，一个 `mod` 里的所有 annex 共占一个），新的 annex 只能放进已有 annex 的模块；骨架返回的续延要标 `@tt`，否则它不内联，`gpu.outline` 就认不出 work-group 循环与 work-item 循环的嵌套。

### 9.8 对照调研之后的三步（2026-11-17）

依据 `docs/gpu-protocol-skeletons-survey-codex.zh-CN.md` §4.2、§6.2 的规则 A 与 §6.4，以及 `docs/gpu-multilevel-first-principles.zh-CN.md`；验证方式同 §9.6。

- **拒绝的原因分类**。
  `check.messages` 的每一条现在在 `check.kinds` 里有一个类别，报错后面跟一行说明：目标的指令要求如此、否则算出来的是别的东西、目标的资源不够、只是这个 lowering 的限制。
  `lower_to_mem.cpp` 里与调度有关的 39 处报错按同样的四类加了说明。
  调研点名的两处因此不再读成硬件约束："specialized 的归约不跨 cluster 共享面板"与"持久化环每块瓦片的块数要填满整圈"都标为 lowering 的限制。
- **归还的滞后与在途的 MMA 组数对上**。
  `gpu.wgmma_batch` 的等待返回 `InFlight pending`，即"此刻至多 `pending` 组 MMA 在途"；`gpu.ws_fold` 要求每一步交回 `InFlight lag`，`lag` 就是它晚几步归还槽。
  这两个数原来在 C++ 里各传一次（`wgmma_pending_` 与 `overlap`），现在不一致就是类型错误（`lit/error/ring_release_in_flight.mim`）。
  它说的是数目一致，不是规则 A 要的"归还由计算完成证明"：见证不带是哪一步的 MMA，`ws_fold` 也不消耗它。
  10 个环形程序改动前后的设备 IR 逐行相同。
- **tensor-core 面板的布局只剩一份**。
  原来 C++ 的 `panel::mma`、`panel::wgmma` 与 Mim 的 `tc_layout.mma`、`tc_layout.wgmma` 各算一遍，前者给 `tile_fuse` 与 `warp_fold` 用，后者给解释器用。
  Mim 那份挪进 `layout_expr`（`tc_mma`、`tc_wgmma`，该模块已有 annex 标签），C++ 的两个函数改为调用它；`tc_layout` 模块删掉了。
  10 个 wgmma 程序的设备 IR 不变。
- **没有搬的循环**：lane 归约、成对缓冲的 `product_fold`、不带环的折叠，看过之后里面只有数据搬运，没有先后或配对的约定，留在 C++。
- **调研文档进入这个分支**，并在 §4.1.1 加了它的骨架清单与现有定义的对照；PDF 没有带过来。

没有做的：

- `warp_fold` 的环深仍是 2。
- 解除"specialized 不与 cluster 共享同用"和"持久化环块数是环深倍数"两个限制：都要在 H200 上执行验证。

### 9.9 在 H200 上的复核（2026-11-17）

§9.5–§9.8 的改动此前只在 Blackwell 主机上做过全量 lit 与设备 IR 的比较。
容器回到 H200 NVL 后重跑了全量 lit：1024 个测试，1015 通过，8 个不支持，1 个预期失败，没有失败；以 `cuda-sm90` 为条件的 wgmma、TMA、cluster、持久化、ping-pong 程序都实际执行并通过。
所以统一的 `Tile`、槽令牌、`tma_fill`、`tma_ring`、`wgmma_batch`、`persistent_tiles`、布局的单一定义现在都有 Hopper 上的执行为证。
没有重测的仍是 §9.4 的 kernel 时间。

### 9.10 持久化环不再要求整圈（2026-11-17，H200）

调研 §4.2 指出"持久化环每块瓦片的块数是环深的倍数"只是这个构造器的前提。现在去掉了。

- 一块瓦片的第 `i` 个块仍用第 `i % depth` 个缓冲（缓冲每块瓦片从第一个开始轮转），变的是它算作环的第几次填充：`pipeline_plan.persistent_fill (round, count, depth) i` 数出这个槽在之前各块瓦片和本块瓦片里已经被填过几次，mbarrier 的槽号与奇偶都由它得出。
  块数是整圈时它就是原来的 `round · count + i`。
- `gpu.ws_fold` 的参数从"之前的填充数 `base`"改为"块到填充编号的映射 `nth`"；检查规则 12 与 C++ 里对应的报错删掉了。
- 先试过的另一种做法不行：让缓冲按之前的填充数转过去再开始，需要运行期在几块共享内存缓冲之间选择，而它们的地址是按定义静态算出的，kernel 在 H200 上报非法地址（CUDA 716）。
- `lit/{gpu,nvptx}/tile_wgmma_persistent_turn.mim`：3 槽的环、每块瓦片 4 个块，在 H200 上执行通过；把编号换回 `round · count + i`，它失败而 4 槽的原测试仍通过。
  原来的拒绝测试 `lit/error/tile_persistent_ring.mim` 删掉了。

### 9.11 生产者 warp 与 cluster 可以同用（2026-11-17，H200）

调研 §4.2 指出的另一处实现限制："specialized 的归约不跨 cluster 共享面板"。现在两种情形都能生成并执行。

- **cluster 里各 work-group 不共享面板**：每个 work-group 有自己的环与生产者 warp，`tma_ring` 在 `consumer_ring` 外面再套 `cluster_ring`。
- **cluster 共享面板**：每个 work-group 的生产者 warp 把自己那一份行用 multicast 拷进所有 work-group 的槽。
  一个槽要等所有 work-group 的消费者都交还才算空，所以 `consumer_ring` 多了参数 `readers`：槽的 "empty" mbarrier 期望 `4 · consumers · readers` 次到达，每个消费者 warp 交还时到达 cluster 里每个 work-group 的那个 mbarrier。
  这就是调研规则 A 说的"到达集合由接收者推出"在这一种拓扑下的样子；`readers` 仍是一个数，不是从接收者图算出来的。
- 检查规则 10 与 C++ 里对应的报错删掉了；持久化的 work-group 仍不能成 cluster。
- 测试：`lit/{gpu,nvptx}/tile_wgmma_ws_cluster.mim`（不共享）与 `tile_wgmma_ws_shared.mim`（共享 A 的面板），在 H200 上执行，结果全对；原来的拒绝测试 `lit/error/tile_ws_cluster.mim` 删掉了。
- 没有做变异：这里可能出的错是 work-group 之间的竞争（到达数少算、少到达一个 work-group），§9 的表里已经记过这类错执行测不出，少到达则是死锁；所以这一步的证据只有"执行结果正确"，没有"错了会被发现"。

### 9.12 计时（2026-11-17，H200 NVL）

方法同 §9.4：1024³ 的 F16 GEMM，完整 host 程序加 nsys，取 kernel 每次调用的平均时间，各 3 次，单位 ms，结果都正确。
基线是交接时的提交 `a10d52b44`，"现在"是 `603492ff0`。

| 链 | 基线 | 现在 |
|---|---|---|
| `wg_ws`（生产者 warp，4 槽环，128×256 的瓦片） | 0.0179 / 0.0179 / 0.0179 | 0.0181 / 0.0181 / 0.0181 |
| `wg_persistent`（4 槽环） | 0.0179 / 0.0179 / 0.0178 | 0.0180 / 0.0179 / 0.0180 |
| `wg_tma` | 0.0126 / 0.0126 / 0.0126 | 0.0126 / 0.0127 / 0.0126 |
| `wg_cluster`（共享面板） | 0.0236 / 0.0236 / 0.0235 | 0.0219 / 0.0218 / 0.0217 |
| `wg_split` | 0.0137 / 0.0138 / 0.0138 | 0.0138 / 0.0138 / 0.0138 |

与 §9.4 相同：生产者 warp 的链慢约 1%，cluster 共享面板的链快约 8%，其余持平；原因仍没有查。

这一轮新能生成的组合（基线生成不了，所以只有"现在"）：

| 调度 | 时间 | 可比的已有链 |
|---|---|---|
| 生产者 warp，128×128 的瓦片 | 0.0127 / 0.0127 / 0.0127 | `wg_tma` 0.0126 |
| 生产者 warp + cluster，不共享面板 | 0.0130 / 0.0130 / 0.0131 | 上一行 0.0127 |
| 生产者 warp + cluster，共享 A 的面板 | 0.0208 / 0.0207 / 0.0208 | `wg_cluster` 0.0218 |
| 持久化，3 槽环（每块瓦片 16 个块，不是整圈） | 0.0184 / 0.0184 / 0.0184 | `wg_persistent` 0.0180 |

共享面板的 cluster 加上生产者 warp 比不加快约 5%，但共享面板本身仍比不共享慢 60% 以上，这一点与基线相同。
3 槽环比 4 槽环慢约 2%：环浅于 4 时 MMA 不跨步在途（§8.2）。

编译时间（Debug 构建，各 3 次取中位数，单位 s；测量时机器上另有任务在跑，负载约 8）：

| | 基线 `a10d52b44` | 主工作区 `feat/gpu-tiling`（`45c064d7f`） | 现在 |
|---|---|---|---|
| 只加载 `plugin tensor;` | 5.21 | 9.50 | 10.29 |
| `lit/gpu/typed_tiles.mim` | 5.40 | — | 10.35 |
| `lit/gpu/tile_wgmma_ws.mim` | 5.99 | — | 11.67 |
| `lit/gpu/tile_wgmma_cluster.mim` | 6.16 | — | 11.67 |
| `lit/gpu/tile_reduce_merge.mim` | 5.86 | — | 11.24 |

每次编译比基线多约 5 s，几乎都是加载 `tensor` 插件的固定开销。
其中大部分随合并 `feat/gpu-tiling` 而来：那条线自己加载就要 9.5 s，这边的改动在它之上再加约 0.8 s。
没有 Release 构建的数字。

### 9.13 一次外部审阅及其处理（2026-11-17）

用另一个代理（`agy`，只读，没有运行任何程序）审了 `a10d52b44..HEAD` 的全部改动。它列了 9 条，下面是逐条核对的结果；"核对"指我用运行或读代码确认过。

| 审阅意见 | 核对 | 处理 |
|---|---|---|
| `ws_fold` 结束时先过 `settle`（cluster 的收尾屏障）再交还最后一个槽；共享面板且晚一步归还时，交还要到达别的 work-group 的 mbarrier，那个 work-group 可能已经退出 | 属实（读代码） | 顺序改为等 MMA、交还、再 `settle` |
| `layout_expr.rows` 在秩 3 及以上把单位维放错了位置，`shape (rows (2, 3, 4))` 得 `(2, 12, 1)` | 属实（运行复现），是 §9.5 引入的 | 置换改为把单位维放在第 2 位；受影响的只有类型里的布局索引与 `Lowered` 的形状，数据不受影响 |
| `tc_wgmma` 两个朝向的布局都会被构造，解释器对每块面板都调它，于是非 wgmma 的面板会打出 `make` 的检查错误日志 | 属实（运行复现），原来就有 | 改为形状合法才构造，且只构造被选中的朝向 |
| `lies_sw128` 只核对布局相对它自己去掉 swizzle 的摆放，不核对描述符假定的行宽；`16 / bytes` 没有防零 | 属实；行宽这一条是 §9.6 把判定改成与朝向无关时丢掉的 | 加上"描述符步进的行是 128 字节"（K 主序看摆放的末维，MN 主序看切出的原子），防零 |
| `lies_sw128`、`lies_tma128` 判不出（非 2 的幂、带填充）时被放行 | 属实，是有意的 | 没改：判不出不等于错；写在这里 |
| `check_sync` 对槽令牌的计数可以绕过：把令牌包进元组，或在外层取得、在循环里交还；分支没折叠时会误报 | 属实 | 没改：`ws_fold` 自己就在两个分支里传同一个元组，按元组计数会误报；`check_sync` 的文档改为如实说明 |
| `gpu.in_flight` 是公理，可以凭空构造 | 属实，§9.8 已写 | 没改 |
| A 在寄存器里的 wgmma（`wgmma_lane`）调 `wgmma_batch` 时写死在途 0、不参与轮转 | 属实，原来就有 | 没改：与晚归还同用时现在是类型错误而不是错结果 |
| `ring_safe` 没有按 `persistent_fill` 的编号建模，注释却说覆盖了跨瓦片 | 属实 | 注释改为如实；模型没扩 |
| `wgmma_batch` 的 `publish` 在"外层通用存储驻留的面板 + 内层环"下可能漏掉 `fence.proxy.async` | 没有核对，原来就有 | 没改 |
| 绑定归约按 `n` 而不是 warp 数分配部分和的缓冲 | 属实，原来就有，只是多占内存 | 没改 |

审阅指出的证据缺口里补了两处，都在 H200 上执行通过：

- `lit/{gpu,nvptx}/tile_wgmma_ws_shared_refill.mim`：共享面板的 cluster 加生产者 warp，4 槽、8 个块，槽被回填且晚一步归还。
- `lit/{gpu,nvptx}/tile_wgmma_persistent_late.mim`：持久化的 4 槽环、每块瓦片 6 个块，不是整圈且晚一步归还。

仍然只有"执行结果正确"而没有反例的：上面第一行的修复（错的顺序是与另一个 work-group 退出的竞争，执行测不出）；`tma_fill`、`tma_ring`、`wgmma_batch` 传错参数会怎样。

## 10. 删掉 Mim 里的调度解释器（2026-11-18）

按 `refactor.md` 的第 2 步，`mod law` 只留规范与小分析，所有调度只由 `TileFuse` 的 C++ 逐级递归（`Levels`）实现。

- 先量差距：`--param tile_law=0`（所有 `tile_req` 都走 `Levels`）下全量 lit 1032 个测试全部照常通过，所以 `run_as_law` 接手的请求 `Levels` 本来都接得住。
- 删掉的：`build_post` 里的解释器（`tile_step`…`wg_step`、`Plan`/`Acc`/`At`/`Pan`/`Re`/`Res`、`inputs_of`、`place_of`、`panel_of` 的那一套）、`boxed`、`schedule`、`verdict_on`、`shared_producer`、`Views`，以及 C++ 的 `run_as_law`/`law_of`/`plain_parts` 与 `-X tensor:tile-law`、lit 的 `tile_law` 参数和 `tile-fuse-path` 特性。
  `tensor.mim` 净少约 570 行，`tile_fuse.cpp` 净少约 120 行。
- `Law` 不再带物理输入的类型 `P`（`phys`/`mean`/`Src` 的 8 个字段都只为解释器服务），改带 `request`：根本身，但被融合的生产者包在 `tensor.law.member` 里。
  `as_source`/`fuse_law`/`fuse_shared` 用 `marked_of` 标记生产者；生产者担保了合并时再包一层 `merge_with`，`Levels::fuse` 本来就认它。
  `law.of_sources` 不再收 `Ps`；`law.schedule` 删了，测试改用 `law.run`。
- `law.req (l, members, levels, control)` 就是 `tile_req (request, members, levels, control)`；`law.run` 是不带额外成员的 `law.req`。
  定律自己担保合并（`of_map_reduce_merging`、`of_rolling`）时根被 `merge_with` 标记：`root_loops`/`root_acc`/`root_contraction` 透过它看根，`TileFuse::realize` 剥下它，作为没有显式合并的 work-item 归约层级的合并。
- `TileFuse::start`/`realize` 剥掉 `member` 标记（`strip_marks`，内层先），被标记的生产者加入成员组；标记本身算作组内，所以显式给出的同一个成员不会因此被当成组外使用。
- `Levels` 补了一处：work-group 层级提升（`promote`）的输入若是成员，不整块暂存它，而是像其它成员一样融合进读它的瓦片（`law_resident`、`law_resident_sum` 原来多一次 launch；§13 起取代了先前“暂存填充里就地算 box”的做法，那个做法在 1024 长的行上编不过）。
- 测试：断言“走定律”的日志行删掉；`law_promote` 的布局日志从 3 次变 2 次，因为 `law.run` 与 `law.req` 现在生成同一个请求，hash-cons 后只实现一次。
- 全量 lit（H200，`cuda-sm90` 执行）：1032 个测试，1023 通过、8 不支持、1 预期失败，与改动前相同。
- 编译时间（Debug，同一构建目录，三次取中位数，编到 GPU 代码）：只写 `plugin tensor;` 12.2 s → 1.2 s；`lit/gpu/law_promote` 18.8 → 3.7 s，`law_fuse` 17.5 → 2.7 s，`law_wgmma_ws` 13.4 → 1.6 s。
- 失去的：解释器由 `Plan ou grp stp`/`Acc u lo ext` 的类型保证面板不越出其所有者与生存期；现在由 `Levels` 的一份 C++ 实现与 `denote` 的差分执行测试共同承担。

## 11. 一个层级类型（2026-11-18）

`refactor.md` 第 1 步：五个互递归的 nom（`TileLevels`/`GroupLevels`/`ItemLevels`/`WarpLevels`/`WarpgroupLevels`）合成一个 `TileLevels (n, Acc, Mma)`，分支与 C++ 的 `LevelSpec` 一一对应：

- `Par [bind, sizes, tail, promote, rest]`：`bind` 0 在所在单元里依次跑，1 work-group，2 work-item，3 warp，6 warpgroup；
- `Red [bind, sizes, tail, promote, pipeline, fill, pingpong, merge, rest]`：`bind` 0 依次折叠，2 分给 work-item（按 `merge` 合并），4 分给 lane；`fill` 1 用 TMA、2 用生产者 warpgroup；
- `Vec`、`Tc [shape, operands]`（warp 的 mma 与 warpgroup 的 wgmma 都是它）、`Cluster`、`Persistent`。

构造函数（`serial_group`、`group_item`、`wg_red_ws`、`warp_mma`……）名字与参数都不变，只是不再带 `{g}`/`{b}`/`{t}` 这类切块索引；`ScopeTile`/`scope_tile` 删了，`within` 只剩 check 在用。

- 嵌套规则从类型挪到 `check.walk`：一个函数、一遍，状态是所在单元、块、A/B 是否暂存、cluster、持久化与计算的 warpgroup 数，原来的 `tile_ok`…`wg_ok` 与 `tile_law`…`wg_law` 十个函数都并进去了（`law` 为 4 时不查组合子，即普通 `tile_req`）。
  新增的规则（消息 26–32）：work-group/cluster/持久化层级只在串行代码里；work-item、warp、warpgroup 只在 work-group 里；warp 与 warpgroup 里只有归约和张量核层级；work-item 归约在 work-group 里、lane 归约在 work-item 里；TMA 与生产者填充的归约在 warpgroup 里；张量核层级只结束 warp 或 warpgroup；warpgroup 的切块正好是它的 wgmma 的 m × n（原来由 `wg_mma` 的类型索引保证）。
- law 的共享内存估算（`staged_of`）与 `of_blocks` 的解释器各自从五个按 nom 拆开的函数变成一个；bench 里五份相同的 buffer 解释器（`bench/gpu_fusion/gemm_*_buffers.mim`、`bench/typed_tiles/buffer_interp.mim`）同样合成一个按单元分派的 `step`。
- C++ 的 `tile_levels` 不再查表推断“哪个 nom 的第几个构造子”，直接读字段。
- 测试：`tile_levels_scope` 现在由 check 报"work-items, warps and warpgroups run only inside a work-group"，不再是类型错误；`tile_wgmma_rows`/`tile_wgmma_n` 报新规则 32；`tile_warp_mma_divides` 的 warp 层改为封顶尾块，否则先报"bound level cannot peel"，现在报 6 号规则。
  `tile_pingpong_consumers` 原来靠类型里未裁剪的 work-group 切块（192 行，3 个 warpgroup）报错，但根只有 128 行，裁剪后只有 2 个 warpgroup 计算，ping-pong 其实合法；测试的根改成 192 行，由 16 号规则报。
- 代价：违反嵌套规则的链不再是类型错误，而是 PE 求出的 check 报错，报在请求处而不是层级处；符号层级时这些规则要等到字面量才查。

## 12. 槽令牌检查补上的三处（2026-11-18）

§9.13 审阅里记下的 `check_sync` 缺口，按 `refactor.md` 第 3 步补上：

- 包进元组的令牌：类型里含 `Held`/`Vacant` 的值（元组、`hold` 的结果、带令牌的参数）只能被投影使用，否则把元组交出去两次就把槽交还了两次（`lit/error/ring_token_tuple.mim`）。
- 循环里交还外层的令牌：令牌的使用者落在它所在的 mutable 里嵌套的、可能重复执行的 mutable 中时报错（`lit/error/ring_token_loop.mim`）；直接调用、分支选择与 `cps2ds` 只执行一次，调用自身或被交给别处（`affine.For` 的体、`vec.fold` 的步）则可能重复。
  只按嵌套报错会把 `ring_fold_at` 里只跑一次的 `refilled` 之类 continuation 也拦下（第一次全量 lit 56 个失败）。
- 互斥分支：同一令牌（或装着它的元组）的多次使用若分别只在同一个选择 `(…)#c` 的不同分支里（分支是值，或是 continuation 及嵌在其中的 mutable），算作一次；原来没折叠的分支会误报（`lit/gpu/ring_token_branch.mim`），`ring_fold_at` 的 `skip`/`go` 也靠这一条。

三个测试在旧的检查下：前两个通过（漏报），第三个报 "used 2 times"（误报）。

`Tile`/`Lowered` 在 §10 之后只剩显式调度的 DSL（`lit/gpu/typed_*`、`bench/typed_tiles`）在用，自动调度器不再碰它们，所以没有再挪到单独的模块里。
`refactor.md` 第 3 步里 `ws_done` 的顺序、`layout_expr.rows`、`tc_wgmma` 的急切求值和 `lies_sw128` 的行宽已在 §9.13 修过。

## 13. 重构前后的 kernel 性能（2026-11-18，H200 NVL）

基线是 `f566f6a61`（`~/workspaces/MimIR-tilebase`，同样的 Debug 配置），新的是本分支；两边都编到 `sm_90a`。
先比较规范编号后的设备 IR：新构建与“基线加 `-X tensor:tile-law=0`”逐字相同，所以差别全部来自原先默认走定律解释器的请求，`attn_warps`、各个 `softmax_*`、`rowsum`、`buffer_interp` 等 13 个 kernel 两边相同。

- F16 GEMM（`gemm_levels.sh`，nsys，三次中位数，µs）：写 F16 时 1024（64×128，4 级）6.26 / 6.23，2048 30.46 / 30.25，4096 200.35 / 199.90，8192 1691 / 1692；写 F32 时 1024 6.74 / 6.75，2048 30.05 / 30.08，4096 210.39 / 210.45，8192 1705 / 1704。
  1024 的 64×128 调度起初只有定律路径编得出（"a wgmma operand panel needs its swizzled layout"）：只有一个瓦片的 warpgroup 区域不让里面暂存的东西外提，面板的填充于是不算工作组统一的，布局没有记下。
  现在一个 work-group 唯一的 warpgroup 按 work-group 填面板（`lower_to_mem` 的绑定层级保持 1）。
- `law_tiled*`（`krun.sh`，kbench 中位数，ms，1024 / 4096）：`law_tiled` 0.288→0.208 / 10.51→9.46，`g2`、`g2cap` 0.213→0.191 / 9.97→9.31，`g3_128` 0.157→0.157 / 4.34→3.95，其余（`g3`、`g3_128w`、`tc`、`tc8`、`h16`、`bf16`）在 1% 以内；`g2s` 1024 上 0.251→0.272（复测三次一致），4096 上 14.71→14.72。
- softmax（12×1024×1024，`softmax_driver.cpp`，nsys，µs）：`softmax_resident` 29.6 / 29.7，`softmax_resident_law` 37.5→29.6，`softmax_resident_shared` 27.0→29.6。
  后者失去了 `law.fuse_shared` 让生产者读消费者暂存行的优化，现在与 `softmax_resident` 相同。

之后修的两处（全量 lit 通过）：

- 融合进工作组的归约分给 work-item 时（`split_over_items`），先按瓦片的元素分，每个 work-item 只折叠一个元素的一段：`softmax_resident` 的 4 行 × 1024 原来每个 work-item 折 4 行各 4 个元素、每行 8 层合并，现在 1 行 16 个元素、6 层合并，与定律当时的做法相同。
  softmax（µs，基线 / 新）：`softmax_resident` 29.61 / 27.38，`softmax_resident_law` 37.62 / 27.41，`softmax_resident_shared` 27.00 / 27.40。
- 暂存的向量化填充（`vector_width`）不再要求长度是字面量，只要求可证明是向量宽度的倍数：`nat_multiple` 带一个偏移，认得选择（`min`）、加字面量与饱和差。
  夹在张量边界上的块 `min (k0 + 15, 1023) − k0 + 1` 由此可以 128 位地填。
  `law_tiled*`（ms，基线 / 新）：`g2s` 1024 0.2512 / 0.2450，4096 14.71 / 14.62；`law_tiled` 0.288 / 0.208，`g2` 0.213 / 0.178，`g2cap` 0.213 / 0.191，4096 上 10.52 / 9.46、9.97 / 9.24、9.97 / 9.31，`g3_128` 4.34 / 3.95；其余在 ±0.6% 以内。
  F16 GEMM 各规模不变（写 F16：6.22 / 6.21、30.36 / 30.32、200.77 / 200.28、1689 / 1697 µs）。
- `bench/tools/cmp_device_ir.sh` 用两棵树的构建编译同一组文件并比较重新编号的设备 IR。

## 14. 第二轮审阅（`refactor-review.md`）的高危项（2026-11-18）

- A1 `stage_fragments` 的写回循环是 `rows · runs / 128` 次，N 为 8 的奇数倍（24、40…）时向下取整，漏写行：`lit/{gpu,nvptx}/tile_wgmma_n24.mim` 在修复前 2048 个元素为 0。
  现在只有 16 字节的段能平分给 128 个 work-item 时才经 shared 暂存，否则直接写回。
- A2 `wg_red_ws` 配 `serial_cluster_split`：生产者 warpgroup 跳过写回，永远到不了合并部分和的 cluster 屏障。
  `check.walk` 拒绝 `fill == 2 && split != 0`（复用不再发出的第 10 条消息），`write_fragments` 里留一道兜底（`lit/error/tile_ws_cluster_split.mim`）。
- A3 外层通用写入的暂存加内层 TMA 环漏发 `fence.proxy.async`：`TileFuse` 要求张量核的两个操作数都由归约层级提升进环（"the tensor-core operands must be promoted into work-group panels"），这种组合现在写不出来。
  仍加了防护：工作组里有尚未发布的通用写入时，下一批 `wgmma` 先发布（`generic_pending_`，进入 work-group 时清零）；现有的 kernel 都不会触发它。
- A4 `check_sync` 的槽令牌检查改成白名单：
  令牌不进 `Pack`（`has_slot` 也认 `Arr`）；装着令牌的值要么按字面下标拆开、要么整体交出一次，拆出的令牌的使用者与整体的使用两两互斥（`ring_fold_at` 的 `skip`/`go` 正是这样）；
  只有各个分支都是 mutable 的选择算互斥（值的选择两边都会计算，`lit/error/ring_token_select.mim`；`lit/gpu/ring_token_branch.mim` 改用 continuation）；
  捕获令牌的 mutable 的每次使用都只调用它一次（直接调用、`cps2ds_dep` 且其结果只调用一次、选择里被选一次），且各次使用两两互斥（`lit/error/ring_token_calls.mim`、`ring_token_mixed.mim`）。
  这三个报错测试在旧的检查下都通过（漏报）。

## 15. 合法性规则收拢（2026-11-18）

- Cluster 与 Persistent 的规则只在 `check.walk`：一条链至多一个 cluster 或 persistent 层（任意顺序、中间隔着别的层也算，消息 33），它们下面要有 work-group 层（消息 34）。
  `serial_persistent` 构造时不再检查紧邻的下一层（`lit/error/tile_persistent_cluster.mim`、`tile_cluster_persistent.mim`、`tile_cluster_alone.mim`）。
- `law` 的 `staged_of` 累加 work-group 里每个暂存面板的归约，不再在第一个归约处停下（`lit/error/law_capacity_nested.mim`：两层各 128 KiB）。
- `check.walk` 记着 work-group 的块（`gblock`），warpgroup 数按它算，和 `TileFuse` 的 `group_scope_` 一致；之前中间夹一个 `group_par` 时按缩小后的块算。
- 消息 12 不再报告，只保留编号；消息 10 改为报告 A2 的规则。

## 16. bench 整理（2026-11-18）

- 五份相同的 buffer 解释器合成 `bench/typed_tiles/buffer_levels.mim`，`buffer_interp.mim` 与 `bench/gpu_fusion/gemm_*_buffers.mim` 只留各自的形状与层级；`step` 的分支是 thunk，选中的才展开。生成的设备 IR 与拆分前相同。
- `bench/tools/compile_all.sh` 把占位符（`HH`、`LL`、`DD`、`BM`、`BK`、`SCALE`、`PIPE`、`CL`）换成小的默认值后编译每个 GPU bench，`reject_*` 反过来要编不过；现在全部通过。
  `bench/gpu_attn/*.mim` 是 `kattn.sh` 等脚本替换占位符后的模板，不是过时的文件；`attn_fa` 要 `BK=64`。
  `bench/gpu_fusion/attn_tiled.mim` 的向量层原来同时标了 h 与 m 两个循环，改为只标 m 后可以编译，结果正确。
- `bench/gpu_*/k*.sh` 的仓库根不再写死为主 checkout，而是由脚本位置推出（或 `ROOT=`）。
