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
- 第 5 项（2026-11-15）：`law.traffic` 加了容量约束——work-group 暂存的共享内存（第一个 work-group 层常驻的盒子，加其中第一个归约层的面板乘环的深度；warp 与 warpgroup 的暂存不计）超过 H200 的 227 KB 时流量记为无穷，`law.best` 因而不会选它（`lit/tensor/law_traffic.mim`：1024³ 的 GEMM，256 × 256 整 K 提升的候选不计容量时流量最小，现在被跳过）。
  `law.run` 的 `control` 明确为不用：定律的构造已经决定融合什么，融合与否含义相同；在融合与不融合的实现之间由 `control` 选择，要等第 4 项的 `Source`（`plain` 与 `produced`）之后才有对象可选。
  TileSight（tile-ai 的解析式性能模型）的研读建议把代价从字节改成时间（按资源的 II 下界、波与尾块）、加派发顺序的 L2 复用，留待第 4 项之后。
- `lit/tensor/tile_law_exec.mim` 在 CPU（ASan）上执行：`denote`、`req`（TileFuse 实现）、`par`、`red`、`red_par`（加法合并）、不带合并的 `red_par` 都等于参考结果，即三条切分引理在这个实例上成立；换成错误的 `merge`（取第一个参数）时 `red_par` 被测出错误；
  `law.schedule` 对 `serial_par`/`serial_red` 和 `serial_group`/`group_item`/`item_red` 两条链的结果等于参考结果，同一条链经 `law.req` 由 `TileFuse` 实现，结果也相同；
  生成的代码不带 ASan 插桩，所以输入后面放了非零哨兵，块越界读会改变结果（把尾块长度改成整块的变异由此被测出）；
  `lit/error/law_req_rule.mim`（规则用定律的事实）、`law_req_kind.mim`（收缩种类不符是类型错误）。

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
| 物理输入是 `«nis - 1 + npp; …»`，按下标拼接 | `npp` 先成为字面量时，异构元组按非字面量下标取出的类型是并 `T ∪ U`，`bitcast` 推断的源类型不再匹配 | 穿过二元组 `(消费者的输入, 生产者的输入)`，只按 pack 自己的绑定子取下标 |

前三条与 D6 的阻塞同源：依赖类型的实例化与比较。它们是继续推进第 3 步（TileFuse 在定律上解释调度）之前最值得在 core 里修的地方。
