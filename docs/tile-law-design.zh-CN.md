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
- `lit/tensor/tile_law_exec.mim` 在 CPU（ASan）上执行：`denote`、`req`（TileFuse 实现）、`par`、`red`、`red_par`（加法合并）、不带合并的 `red_par` 都等于参考结果，即三条切分引理在这个实例上成立；换成错误的 `merge`（取第一个参数）时 `red_par` 被测出错误；
  `law.schedule` 对 `serial_par`/`serial_red` 和 `serial_group`/`group_item`/`item_red` 两条链的结果等于参考结果，同一条链经 `law.req` 由 `TileFuse` 实现，结果也相同；
  生成的代码不带 ASan 插桩，所以输入后面放了非零哨兵，块越界读会改变结果（把尾块长度改成整块的变异由此被测出）；
  `lit/error/law_req_rule.mim`（规则用定律的事实）、`law_req_kind.mim`（收缩种类不符是类型错误）。

实现中碰到的 MimIR 限制，决定了现在的形式：

| 想要的写法 | 问题 | 现在的做法 |
|---|---|---|
| `nom Law` 带类型值字段 `I: *` 与 `ins: I` | 依赖 sigma 的后续字段不能用前面字段的值代换 | 输入不进记录，切分在构造子里做成闭包（与 `Tileable` 的 `retile` 同理） |
| `Law` 以 `I` 为参数 | 参数含依赖数组类型时，类型级 lam 在泛型构造子里实例化不完整 | 同上 |
| 结构化 `Law (r, s, T)` 上按名取字段 `l#root` | 字段名不经类型函数传到 `#` | 按位置取，模块给出具名的访问函数 |
| `law.req` 把层级交给 `tile_req` | 层级经过 RWPhase 重建后成为结构相同、指针不同的 nom 实例，`bitcast` 不能归约掉 | 经 `core.bitcast` 转换，TileFuse 剥掉它；`tile_req` 的 normalizer 见到它就跳过（定律已用自己的事实检查过） |
| `check.report` 在 `law.req` 的定义体里 | normalizer 在定义时就看到未定的判定 | 判定含自由变量时不归约 |

前三条与 D6 的阻塞同源：依赖类型的实例化与比较。它们是继续推进第 3 步（TileFuse 在定律上解释调度）之前最值得在 core 里修的地方。
