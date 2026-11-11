# 结论

**第二份文档适合作为重构议题清单，但还不能作为实施方案或可信性论证。** G1–G12 大体抓住了问题，然而混淆了四件事：信息没有写进类型、没有独立检查、仅在某个 lowering 阶段不可见、完全没有 IR 表示。部分条目还与当前代码不符。

主要结论是：

- **A 类方向可行，但 A1、A2 尚缺足以承载其算法的执行、存储和同步接口。** 部分求值不会自动补齐这些语义。
- **B 类多数能表示更多信息，但 nominal 类型和参数索引本身不能保证所有权、收敛、生命周期或归约律。**
- **C 类适合检查局部数值条件；不能普遍把 C++ 分析降级为优化。**
- 第 5 节有明确的依赖倒置；“搬走 1200–1500 行 C++”目前没有足够依据。

以下核对基于当前 `4642222e88`、两份文档、指定实现及论文第 7 节。全程只读，未修改文件，也未运行会生成产物的测试。

# 高严重度问题

## 1. “加上断言，C++ 就退出可信计算基”不成立

**文档位置：** `docs/gpu-tiling-formalization-gaps.zh-CN.md:18–22、190、204`。

断言只有在**独立地检查实际前提，且检查先于依赖该前提的危险操作**时，才能减少对某项分析的信任。把分析结果包装成 `assert`，或者检查一个必要但不充分的条件，都不够。

最直接的例子是原地更新。`linear_step` 检查的不只是读写下标相等，还包括：

- 输出映射对循环变量的系数；
- 迭代范围不超过输出范围；
- accumulator 的读映射与写映射一致；
- 输入、combiner、post 等其余依赖中不能再次引用 accumulator。

证据见 [lower_to_mem.cpp:1950](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1950)、[lower_to_mem.cpp:1962](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1962)。C 表的“断言读写下标相等”覆盖不了这些条件。

同步的全员到达、跨迭代别名、归约结合律，也不是普通 kernel 内比较即可验证的性质。若检查本身置于错误的发散控制流中，它不能充当收敛性的证明。

**建议改法：**

将总规则改成三类义务：

1. 可直接执行的局部条件：生成有序的运行时检查；
2. 有限结构性质：由独立检查器验证调度、映射或描述符；
3. 并发与代数性质：由受限组合子的语义论证、静态检查或明确的用户许可承担。

同时保留独立检查器、原语实现和后端为可信边界。不要承诺“分析算错只会 trap”。

## 2. B2 的 `collective_map` 既不能禁止发散调用，也不足以表达流水协议

**文档位置：** 第二份 `:164–172、237`；这里正文的确定性结论与风险节直接冲突。

```mim
collective_map ... → (Idx n → T) → Shared g «n; T»
```

这个签名没有表达：

- 调用点是否全组一致；
- callback 能否捕获 item/lane 相关值；
- shared 存储何时可以覆盖、复用或释放；
- 调用之间的内存顺序。

即使删除裸 `fence`，仍然可以在 item 相关分支中调用 `collective_map`。没有额外的作用域、效果或收敛检查，“错误不可写”并不成立。

当前实现也远不止“返回时可见”：

| 同步构造 | 能否由当前提议覆盖 |
|---|---|
| 协作填充 | 新分配的不可变结果可以；复用存储还需要填充前的 WAR 边界，批量填充还要表达共同发布。 |
| R8 合并轮 | 每轮生成新快照可以；当前原地复用 partials 的实现还需要独占访问和轮间顺序。 |
| 流水环槽 | **不能保持现有异步时序。** 返回即 ready 会强迫等待；当前协议需要提交未完成写、等待较老组、全组发布及安全复用槽位。 |

代码中的差别是实质性的：`staged/chained/async` 决定首尾 barrier 的位置，见 [outline.cpp:155](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/phase/outline.cpp:155)；流水分别生成 commit、wait-group、fence、出口 wait-all，见 [lower_to_mem.cpp:2341](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2341)、[lower_to_mem.cpp:2372](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2372)、[lower_to_mem.cpp:2397](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2397)。

**建议改法：** 把同步 collective 和异步传输分开设计。后者至少需要表达 pending/ready、提交组、等待与槽位复用；通过内存/CPS 顺序和受限 region 检查落实。B2 完成前，不能撤掉 outline 的发散检查。

## 3. G10 漏掉了比 `Frag` 类型更关键的 MMA 算子语义前提

**文档位置：** 第二份 `:90–93、179–181、211`；第一份 `:234–253`。

`Levels::mma` 检查了 scope、秩、输入数、F32、指令形状、stage 轴及部分整除条件；**没有核实 combiner 确实是乘加，也没有完整核实操作数访问关系是所需矩阵乘关系**。见 [tile_fuse.cpp:1569](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1569)。

这不能由 `graph::map_reduce` 隐含保证：它仅匹配一般的 `map_reduce_post` 调用，见 [pattern.cpp:30](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/pattern.cpp:30)。

而后面的 lowering 固定生成 `gpu.mma_tf32(a,b,c)`，没有使用原 combiner，见 [lower_to_mem.cpp:2131](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2131)。TF32 的数值许可不能解释把其他算子改成乘法。

G10 还应记录以下前提：

- accumulator 必须是字面量二维形状、scalar seed、可原地更新，见 [lower_to_mem.cpp:2237](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2237)；
- A 面板要支持四元素向量寻址及正确的 shared 行地址，见 [lower_to_mem.cpp:2105](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2105)；
- ISA 支持、地址空间、对齐、warp 一致执行及 intrinsic ABI；
- `mma_ref` 使用普通 F32 乘加，没有模拟默认 TF32 转换，见 [tensor.mim:239](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:239)。现有 CPU 测试因此验证的是布局和普通乘加参考，不是完整 TF32 数值语义。

**建议改法：** 将 G10 拆为“算子匹配与数值许可”“fragment 表示”“面板传输与 ISA 前提”。先增加受限 contraction 描述或独立 matcher，再讨论删除专用 lowering。`Frag` 本身解决不了错误的算子识别。

## 4. G9 的流水前提没有列全，第一份“硬件范围由构造满足”缺少实现依据

**文档位置：** 第二份 `:85–88、129–132`；第一份 `:222–227`。

当前深度只有：

```cpp
depth = min(pipeline, literal_count)
```

没有同时限制到目标支持的 wait 立即数或异步组资源范围，见 [lower_to_mem.cpp:2293](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2293)。后端将 `ring.size()-2` 对应的值直接发射为立即数，见 [nvptx.cpp:773](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/phase/nvptx.cpp:773)。因此 `n ≤ blocks` 不能推出第一份声称的硬件合法性。

此外，流水还依赖：

- stage 的发现、probe 与各块实例之间对应稳定；
- 面板形状不依赖 step 参数，实际分配为 shared fill；
- 环槽的物理布局随参数传递；
- 尾部 clamp 重填不会改变之后可观察的值；
- 异步源在传输期间保持有效、稳定。

前三项明显由 C++ 实现，见 [lower_to_mem.cpp:2168](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2168)、[lower_to_mem.cpp:2303](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2303)。

**NVPTX 发射侧还漏了这些事实：**

| 发射路径 | 当前 C++ 决定或相信的条件 |
|---|---|
| `cp.async` | 目标支持异步拷贝、大小为 4/8/16 字节、值恰好来自整个 global load；否则同步 store。目标地址直接 cast 成 shared，地址空间与对齐没有在这里独立验证。见 [nvptx.cpp:689](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/phase/nvptx.cpp:689)。 |
| `ldmatrix` | `n` 取字面量并拼进 intrinsic 名，指针可直接 cast 为 shared；此处没有完整验证指令数量、对齐、收敛和目标支持。见 [nvptx.cpp:710](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/phase/nvptx.cpp:710)。 |
| TF32/MMA | 固定按 A/B/C 的 4/2/4 寄存器 ABI 发射；默认路径没有显式截断转换。见 [nvptx.cpp:737](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/phase/nvptx.cpp:737)。 |
| shared 分配 | 按整个 kernel 的静态 slot 集合布局、逐槽按 16 字节取整，超过 48 KiB 后改用动态 shared；这些决策在 emitter 中，见 [nvptx.cpp:605](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/phase/nvptx.cpp:605)。 |

异步拷贝的目标开关来自 compute capability，见 [nvptx.cpp:206](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/nvptx/nvptx.cpp:206)；不能笼统写成“sm_80”，也不能将该开关当作 MMA/ldmatrix 的完整 feature 检查。

**建议改法：** 增加目标能力和传输描述符，将立即数合法性列为静态必检项；把 shared 资源限制作为明确的外部前提，与第一份排除在范围外的 T8 对接。

## 5. `panel_layout.cpp` 不能整体归为“只影响选择，不影响正确性”

**文档位置：** 第二份 `:212`；第一份 `:279`。

银行冲突代价确实只是启发式，但同文件的 `Fill::contiguous` 会影响向量写是否合法。它只检查前 `min(total,64)` 个填充位置，见 [panel_layout.cpp:123](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/panel_layout.cpp:123)。

其结果进入 `choice.contiguous`，并用于决定把若干逻辑元素合成一次向量写，见 [lower_to_mem.cpp:853](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:853)、[lower_to_mem.cpp:888](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:888)。这是正确性相关判定；布局单射并不意味着任意逻辑向量仍是连续且对齐的一段物理地址。

另外，`layout_expr.make` 检查的 logical shape 没有保存在 `Expr` 中，见 [tensor.mim:106](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:106)；`placed` 接受的 `Expr` 也没有与输入形状建立类型关系，见 [tensor.mim:2499](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2499)。创建时检查通过，不等于在另一 storage shape 上使用仍然合法。

**建议改法：**

- 将成本评分与连续性、对齐、形状兼容性判定分开列账；
- 证明“检查前 64 个位置足够”的周期性前提，或改为完整的静态验证；
- 将 layout 与实际 logical/storage shape 绑定，或在应用处重新检查。

## 6. C 表中有错误谓词，也有不能达到所述效果的谓词

**文档位置：** 第二份 `:194–202`。

| C 表条目 | 评审与建议 |
|---|---|
| G1：`off+ext ≤ shape`、`ext ≤ cap` | 可执行，但必须逐维检查并处理加法溢出。更稳妥是先查 `off ≤ shape`，再查 `ext ≤ shape-off`。字面量 shape 不代表带运行时 group/item 原点的条件必然静态消失。 |
| G1：迭代域 extent `== ext` | body 返回形状已经由依赖类型绑定；内部循环域则还可能有归约维和非恒等写映射。只在受限的 canonical 情况下比较相应并行维；一般应检查输出映射关系。 |
| G2：内外 box 包含 | 方向正确；须明确使用父 tile 的有效范围还是容量，且在合并视图、丢失父边界之前检查。 |
| G3：逐元素互逆 | 有限字面量域中可检查；要限定有效源坐标，padding 区域不要求双向互逆，并明确 encode/decode 的方向。 |
| G4：`count×stride ≥ K` | **peel 下错误。** `K=10, stride=4` 时 full blocks 为 2，余下 2 由 `last` 处理。公式见 [tensor.mim:2266](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2266)。应分别验证 cap 与 peel 的精确分割关系。 |
| G5：`K%n==0`、`is_pow2 n` | 前者属于 `split_over_items` 的特定拆分条件；后者不是整个 R8 的合法性条件，当前有非 2 的幂退路。不能统一加入。 |
| G12：地址 `%16==0` | 可检查，但应检查实际访问要求的对齐量，并先于 `mem.align` 和受其影响的访问。仅删掉对齐声明也可能失去优化信息。 |
| G12：读写下标相等 | 不足以替代 `linear_step`；必须保留其完整的无额外 accumulator 依赖及独立写等条件。 |
| G12：分配后查 `ext≤cap` | 可作局部防护，但不能验证物理布局、容量乘积溢出、所有实际访问及生命周期。应在相关分配和访问前建立条件。 |

`mem.align` 的确会发射 LLVM alignment assumption，见 [ll.cpp:969](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/ll/ll.cpp:969)；它来自 buffer 向量化，见 [lower_ptr.cpp:37](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/buffer/phase/lower_ptr.cpp:37)。可行的方案是“先检查，再声明并使用”，同时保证检查在内存顺序上存活并支配访问。

还有一个机制边界：现有 `refly.check` 在 `tt` 时返回原值，`ff` 时记录错误并保留残项，并不产生相等证明或类型转换，见 [refly/normalizers.cpp:81](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/refly/normalizers.cpp:81)。假设的运行时 `assert` 同样不能自动消解依赖类型要求的定义相等。

# 中严重度问题

## 7. G1–G12 需要逐项修订，并补充生命周期、静态表示限制和算术前提

**文档位置：** 第二份 `:40–105`。

| 条目 | 准确性、代码证据与建议 |
|---|---|
| **G1** | **部分准确。** 输出网格来自输出形状与 tile sizes，不是统一由读映射的像算出，见 [tile_fuse.cpp:1184](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1184)。`tiled` body 返回 `Tensor r ext T` 已绑定结果形状，见 [tensor.mim:2185](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2185)。此外 `tiled_n` 已在 C++ 穷举检查越界、重叠和覆盖，见 [normalizers.cpp:240](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/normalizers.cpp:240)。应区分已有检查和缺少 IR 中的检查表达。 |
| **G2** | **核心准确。** resident 复用确实直接相减，见 [tile_fuse.cpp:1078](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1078)。但 `Nat` 字面量减法是截断到零，不是普通无符号下溢，见 [core/normalizers.cpp:488](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/core/normalizers.cpp:488)。应写成“负偏移可能被静默截成零，缺少包含关系检查”。 |
| **G3** | **核心准确，例子过窄。** 实现使用参数 `tile`，不是固定 `/8、%8`，见 [tile_fuse.cpp:875](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:875)。还要区分逻辑 stage 的重索引与 `placed L` 的物理映射。 |
| **G4** | **准确但遗漏重要结构。** full blocks、尾块和 post 已由 Mim 的 `tf_reduce/tf_tail/tf_chunk` 描述，不能全算成未表达的 C++ 算法。归约律的识别仍在 C++，且 seed 单位元检查针对自动 R8 路径，见 [tile_fuse.cpp:1491](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1491)。 |
| **G5** | **有明确错误。** `K%items==0` 是融合 producer 的自动拆分条件，见 [tile_fuse.cpp:1035](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1035)。R8 在 `n` 非 2 的幂时顺序合并 partials，见 [lower_to_mem.cpp:2654](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2654)。应写成两种合并算法及各自前提。 |
| **G6** | **对 TileFuse 输出基本准确。** 但“存储类别在 IR 中没有”过于绝对：后续有 `shared_scope/slot_shared` 和地址空间类型；见 [gpu.mim:285](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/gpu.mim:285)。真正缺的是高层放置意图、owner 和 lifetime 的稳定契约。 |
| **G7** | **基本准确。** 应明确 outline 只识别它自己生成的较短 item 循环守卫，不是一般控制流的 uniformity 检查，见 [outline.cpp:125](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/phase/outline.cpp:125)。 |
| **G8** | **部分准确。** 最大 trip count 还包含 warp 数乘 32，见 [outline.cpp:31](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/phase/outline.cpp:31)。行主序反线性化在 LowerToMem，见 [lower_to_mem.cpp:1585](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1585)。launch 后 grid/block 大小已是显式参数；缺的是上游契约与映射描述。 |
| **G9** | **方向正确，“全靠构造满足”过强。** 范围限制至少尚未落实；而 commit/wait/fence 在 lowering 后已是项，不能说“一条也没有”。应区分已生成的协议与未显式验证的协议前提。 |
| **G10** | **fragment 类型缺口准确但明显不完整。** 需补第 3、4 项列出的算子语义、地址与 ISA 前提。 |
| **G11** | **基本准确，有两个限定。** “单归约维”是每个 reduction level 只切一个维度，不是 root 只能有一个归约维；见 [tile_fuse.cpp:1393](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1393)。Vec 还支持最后归约维的拆分因子，见 [tile_fuse.cpp:1602](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1602)。Acc 不匹配默认警告并回退，strict 模式报错，见 [group.cpp:518](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/group.cpp:518)。 |
| **G12** | **部分过时，分类不清。** 当前树没有 `origin_dims_`、`item_grid_`；相关信息在 TileFuse 的 `PanelSite` 中，见 [tile_fuse.cpp:1441](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:1441)。当前 LowerToMem 的实际表见 [lower_to_mem.h:135](/workspaces/compile/MimIR/.worktrees/gpu-tiling/include/mim/plug/tensor/phase/lower_to_mem.h:135)。应分别列分析事实、跨表示转发和内部状态。 |

**清单还应补充：**

- **纯性、共享与存储复用前提。** `RealizationRequests::pure` 使用公理名白名单和闭包分析；`sequential_tile` 决定 scratch 复用资格，见 [lower_to_mem.cpp:164](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:164)、[lower_to_mem.cpp:286](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:286)。
- **生命周期与内存请求作用域。** `fresh_memo_` 按 lambda 限定以避免悬空变量；动态 buffer 由 tile body 释放。这不是纯粹性能缓存，见 [lower_to_mem.cpp:526](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:526)、[lower_to_mem.cpp:1444](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1444)。
- **非空与范围事实。** `tile_ends_` 被用于把空 extent 判断直接改为假，见 [lower_to_mem.cpp:637](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:637)。
- **表示能力限制。** buffer 内维必须静态；绑定级要求字面量 extent，不能 peel 非整除尾块，见 [lower_to_mem.cpp:41](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:41)、[lower_to_mem.cpp:1592](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1592)。
- **有限机器整数与尺寸乘积不溢出。** 形式化中的自然数公式不能直接当成实际 `Nat/u64` 的无界算术定理。

## 8. A1–A5 能否实现：要区分纯算法迁移与 GPU 实现迁移

**文档位置：** 第二份 `:115–148`。

| 方案 | 可行性与卡点 | 建议 |
|---|---|---|
| **A1** | 树归并、pair 分配和非 2 的幂退路可以写成 Mim，静态递归加残留循环也符合现有机制。但 `shared_generate/collective_map` 尚无可用的完整契约；示例的 `step: [Acc,Idx n]→Acc` 也未绑定当前 block/stride、输入、tile 形状和实际线程分配。 | 先迁移纯索引与合并计划，并做 CPU 测试；再实现 shared、发布、复用的执行接口。删除整个 `lower_bound_fold` 应是后续验收项。 |
| **A2** | 环旋转、prologue、clamp、提交序列可以写。不能仅在普通 `tile_fold` 上包装：当前 pass 要发现 stage、将其写入指定环槽、把消费改成槽读取，并维持 accumulator 的原地更新。 | 先让 step 显式分离 `prepare(block)` 和 `compute(acc,ready_panels)`，再迁移流水。否则只是把隐式 C++ 分析藏进新公理。 |
| **A3** | 最适合先做。现有 `layout_expr.IndexOps` 已展示“同一解释器用于 affine 和具体整数”的机制，见 [tensor.mim:66](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:66)。但两个字段来自同一函数并不自动证明互逆。 | 用受限构造子派生 encode/decode、shape、padding 和有效域；对有效坐标检查 round-trip。 |
| **A4** | terminal Vec、Tail 变体、按 `nis` 索引的 promote 数组都可写。`Idx n` 应收紧为并行维索引，并保留现有归约拆分因子。普通类型不能保证 Nat 一定可归一化成字面量。root descriptor 也不能与任意 Tensor 并列后就自动真实。 | 用受控构造器把 root 与 descriptor 一起封装；现有 `tile_req` 的 Tensor 类型只暴露结果形状和元素类型，未暴露内部循环数或 accumulator，见 [tensor.mim:2021](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2021)。 |
| **A5** | 网格公式可写，且 `tiled_impl` 已有 Mim 参考公式，见 [tensor.mim:2323](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2323)。但返回完整 `«count;Box»` 可能造成大规模编译期展开；nominal `Box shape` 也不自动保证包含与覆盖。 | 优先返回计数和按需 `box_at`，让循环残留。目标应是共享参考和 lowering 的网格定义，而非再增加一份公式。 |

论文第 7 节支持这种迁移方向，但论文中的 retile、schedule 和 nest builder 接口本来就是 Mim 值。当前 GPU 路径尚缺对应的显式接口，不能只把“C++ 构造循环”替换为“调用部分求值”就获得同样效果。

## 9. B1–B5 的类型收益被高估

**文档位置：** 第二份 `:154–185`。

| 方案 | 评审与建议 |
|---|---|
| **B1** | 示例 `placed_in: …→T→T` **仍是恒等标注**。能传达意图，不能让跨 owner 读取成为类型错误。真正的 `Placed(p,T)` 或带 place 的 buffer 可区分类型；还须约束所有访问操作、group/item 身份的来源和逃逸。不能直接删除 uniform 检查，因为用户或变换也可能给依赖 item 的值贴上 Shared。当前 uniform 的实际定义见 [lower_to_mem.cpp:503](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:503)。 |
| **B2** | 见高严重度第 2 项。nominal `Shared g` 能表达归属，不能独自表达全组调用和时间顺序。 |
| **B3** | 可将物理 block 形状索引到 region；但不能要求所有逻辑 item 循环有相同 trip count。R8 每轮的 `trip=min(n,pairs)` 本来就变短，见 [lower_to_mem.cpp:2657](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2657)。应分开物理 block、逻辑工作量和 active predicate，并让 barrier 位于统一的物理 group 层。 |
| **B4** | nominal `Frag` 可实现，能避免将普通四元素数组误当 fragment。但删掉旁表还要求 fragment 数、全局 tile 形状和布局随 fold 状态、投影、region 参数传播，并改造初始化与 scatter。也应区分 A/B/C 角色和数值格式；仅有 `Frag×Frag×Frag` 不足以约束指令组合。当前旁表存的不只是一个布局标签，见 [lower_to_mem.h:70](/workspaces/compile/MimIR/.worktrees/gpu-tiling/include/mim/plug/tensor/phase/lower_to_mem.h:70)。 |
| **B5** | `[merge,identity,commutative:Bool]` 是声明记录，不是归约律的证明。它既未绑定实际 step，也未表达结合性、seed 条件、浮点 mode 或允许的重排。没有证明项时，可用受限的已识别 law 构造器，加独立匹配；任意用户构造应明确是 permission。当前识别器的前提远多于该记录，见 [tile_fuse.cpp:245](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/tile_fuse.cpp:245)。 |

依赖类型能检查**同一描述符的字段是否一致**；不能自动证明任意两个函数互逆、任意 merge 结合，或任意控制流收敛。部分求值只对可归一化的具体实例有帮助。

## 10. 第 5 节的依赖顺序确实排错，并漏了 A5

**文档位置：** 第二份 `:220–230`。

明确的倒置是：

- A1 在 `:125` 明说依赖 B2，实施顺序却把 A1 放在 B1/B2 之前。
- B1/B2 中的 group 身份、物理线程集合和一致调用规则尚未定义，B3 不能整体留到最后。
- A2 需要可分离的 staging/compute 接口，以及异步协议；不只依赖当前 B2。
- A5 没有进入实施顺序。
- 第一步“只加断言、不改结构”对需要内存/CPS 顺序的运行时断言不成立。

**建议顺序：**

1. 修正谓词、前提和可信边界；加入可静态完成的检查及针对性失败测试。
2. A3、A5：共享纯映射和网格定义。
3. A4：改链和 root 描述接口，保留必要的实例检查。
4. 明确物理 group/block、active work 和作用域契约；实现 B1/B2 所需的基础部分。
5. B5，再做 A1。
6. 显式 staging/compute 和异步传输接口，再做 A2。
7. B4 及 MMA 算子匹配、加载和存储改造；可以作为独立路线提前推进。

验收也应分层。第一份 `:301–303` 已承认 Debug 全量测试有既存失败，第二份不能直接用“Debug 全量 lit 通过”作为每步的唯一门槛；应记录基线、要求相关测试无回归，并分别验证拒绝路径、CPU 参考和实际 GPU 行为。

## 11. 第一份的“已经是项”需要限定，部分说法与第二份互相抵触

**文档位置：** 第一份 `:69–76、109、176、275–285`；第二份 `:88、104–105、211–214`。

主要有四处：

1. **T2 的“生产者方程是项”说得过满。** `tile_of_ref` 是 Mim 定义，但主要 producer 重写在 C++ 的 `producer_tile` 中构造，见 [normalizers.cpp:336](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/normalizers.cpp:336)。应写成“请求、参考切片及部分 builder 是项；一般 producer 方程由 C++ 实现”。

2. **T3 的 `stage(x,L) ≡value x` 按当前接口并非同类型相等。** `stage` 改变秩和形状，真正保持类型的标注是 `placed`，见 [tensor.mim:2490](/workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/tensor.mim:2490)。应分别定义逻辑重索引的读取等价和物理布局不改变逻辑值的等价。

3. **第一份把 `storage_shapes_、multiples_` 列为纯 lowering 缓存，第二份又要求它们的结论成为断言。** 两者都需细分：缓存可不公开，但其中承载的容量、整除和对齐事实仍与正确性相关。可重算不意味着无需可信依据。

4. **“已有项”与“合法性已体现”被交替使用。** `bind`、pipeline 深度、fragment 布局函数确实是项；调度算法、owner、收敛义务和完整 fragment 值表示尚不是。相反，第二份说 lowering 后的流水或 R8 算法“IR 里没有”，也忽略了已生成的显式循环、内存操作和同步。

第一份 `:176` 的一般保守退路也应收紧：当前有不少 unsupported 情况直接报错，例如绑定级动态 extent、peel 尾块及 MMA 不整除，并非都有自动串行退路。

**建议改法：** 每个分面固定列“请求阶段表示、lowering 后表示、已检查前提、仍被相信的前提”。这样两份文档才能使用同一个“可见”的定义。

# 较低严重度问题

## 12. “搬走 1200–1500 行 C++”目前不可信

**文档位置：** 第二份 `:127、211、216`。

按当前物理行数：

| 区域 | 大小 | 能否按现方案整体删除 |
|---|---:|---|
| `lower_bound_fold`，2490–2705 | 216 行 | A1 的主要候选，但包括分配、destination 和发布衔接。 |
| `lower_tile_fold`，2188–2444 | 257 行 | 只有一部分是流水，还包含普通 fold、初始化、原地更新和 fragment 状态。 |
| `lower_mma` 至 `store_fragments`，2054–2166 | 113 行 | B4 只是类型改造，没有提供替代这些加载、计算和 scatter 的 Mim 实现。 |
| `promote_input`，838–897 | 60 行 | A3 主要替换映射构造，仍有 op 重建和输入替换。 |

这些区域合计约 646 行，而且不能全部归为净删除。要达到 1200–1500 行，还必须迁移更多 grid、bufferization、stage、region 专化或融合代码；文档没有给出相应替代接口。

**建议改法：** 删除当前数字，或标为未经验证的长期目标。按函数列“迁移部分、保留部分、新增 C++ lowering/检查器、替代 Mim 实现”，以净减少量评估。新增一个公理再增加 C++ 实现，不应算作算法已经搬走。

# 不确定的点

- **MMA 算子识别缺失有明确代码依据，但我未编译运行非乘加反例。** 最终是否能由当前公开调度入口触发，应补针对性用例验证。
- `Fill::contiguous` 检查前 64 个位置是否对当前受限布局集合充分，可能有周期性论证；实现和这两份文档中没有给出。这里指出的是可信前提缺失，尚不能据此断言已有错误输出。
- 按用户前提接受运行时 `assert` 已存在；其实际效果顺序、DCE 行为、device 失败终止和报告协议未在本次材料中验证。
- 部分求值后的代码规模、nominal 诊断质量及 CPU/GPU 性能，需要原型实验。现有机制支持表达方向，不能支持文档中“展开没有问题”或具体净删行数的确定结论。
- 历史 GPU 测试、33 个配置差分及实测 TF32 行为未重新运行；本评审区分了测试代码能说明的性质与文档报告的既有实验结果。