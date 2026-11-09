# Codex review：GPU 多级 tiling 形式化草案（2026-11-09）

由 codex-cli 0.167.0（只读沙箱）对 `docs/gpu-tiling-formalization.zh-CN.md` 初稿的 review 原文，提示词要求对照论文抽取文本 `cgo27-paper248.pdf`、上一次 review 与代码。未经修改；草案已按其确认的问题修订。

一句话结论：草案抓住了“规格—合法性—GPU 实现”这一正确方向，但目前还不能称为 GPU tile-and-fuse 的“完整、正交分解”；T5 的内存模型前提、T6 的整程序等价表述、T7 的 `ldmatrix` 传输关系与 TF32 表示，以及 T3 对 affine/layout 的定义都需要实质性收紧。

## 1. 对论文模型的概括是否准确

### 确认的问题

总体概括基本准确，尤其是：

- `Tileable`、`Schedule = Tileable → Tileable`、`retile(iota, omega)` 和部分求值的关系，与论文 §7 相符。论文确实给出了“部分求值器正确、rewriter 正确、`loop` 与 `tile` 等价”三项证明义务，见 [cgo27-paper248.txt:353](/home/chzh00001/workspaces/MimIR/bench/gpu_tiling/out/cgo27-paper248.txt:353)。
- affine map 是普通函数、通过函数复合连接 schedule 与 computation，也概括准确，见 [cgo27-paper248.txt:617](/home/chzh00001/workspaces/MimIR/bench/gpu_tiling/out/cgo27-paper248.txt:617)。
- 论文没有执行层级、shared memory、barrier、异步拷贝和 fragment，这个边界判断成立。

但 §1 有以下误读或遗漏：

1. **遗漏 `Out` 和 `oax`，因而弱化了 `Tileable` 的关键依赖关系。**

   论文的 `Tileable` 不只有 `Sr/So/vdim/unroll/retile`，还包括：

   - `Out`：结果类型构造子；
   - `oax`：每个 parallel 维写哪个输出轴。

   论文定义见 [cgo27-paper248.txt:804](/home/chzh00001/workspaces/MimIR/bench/gpu_tiling/out/cgo27-paper248.txt:804)，当前代码见 [tensor.mim:644](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/tensor.mim:644)。`oax` 正是 parallel strip-mining 能正确改变结果形状的依据，不应从论文模型摘要中省去。

2. **“同一迭代域上的重新索引（双射）”不准确。**

   `strip_mine_par/red` 会改变 rank 和 domain shape；当前实现把 `N` 变成 `(N/ts, ts)`，见 [tensor.mim:775](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/tensor.mim:775)。parallel strip-mining 还会产生 packed 输出形状，见 [tensor.mim:808](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/tensor.mim:808)。

   更重要的是，`retile` 的类型只保证 `iota/omega` 的输入输出 rank 对得上，并不从类型上证明它们是双射、覆盖或保持值语义。论文自己也只声称 dependent type 捕获维数不一致，见 [cgo27-paper248.txt:825](/home/chzh00001/workspaces/MimIR/bench/gpu_tiling/out/cgo27-paper248.txt:825)。

3. **论文并非只“改变迭代顺序”。**

   它还改变迭代域的因子分解、选择 vector dimension 和 unroll window；`vdim/unroll` 是 nest builder 的调度字段，不只是 permutation。更准确的说法应是“不涉及并行执行归属和显式存储层级”。

4. **`if_static` 不能与 `refly.check` 并列描述成 Bool 合法性 witness。**

   论文的例子先算 `ts0`，再以 `if_static(ts0, ts0, 1)` 在静态未知时退回 1，见 [cgo27-paper248.txt:868](/home/chzh00001/workspaces/MimIR/bench/gpu_tiling/out/cgo27-paper248.txt:868)。上次 review 已指出当前 `if_static` 的第一个参数是 `Nat`，判断是否静态，而不是证明一个 Bool 为真，见 [tiling-formalization-review.zh-CN.md:210](/home/chzh00001/workspaces/MimIR/docs/tiling-formalization-review.zh-CN.md:210)。

5. **“单个 op”应限定为论文 §7 的 tensor 实例。**

   `Tileable` 抽象本身通过 `Out` 可以封装别的计算；“只有一个 `map_reduce`”是本文 tensor plugin 的实例边界，而不是 `Tileable` 类型在逻辑上的唯一可能实现。

### 建议

把 §1.1 改成完整的：

```text
Tileable = Out + Ro/Rr + So/oax/Sr + sched + retile
```

并把“变换是双射”改成：

> 论文示例中的受支持 schedule 通过显式构造 `iota/omega` 和整除检查实现保义重索引；类型主要保证 rank/coherence，双射、覆盖和语义等价仍是各组合子的证明义务。

---

## 2. T1–T7 是否完整、正交

### 确认的问题

1. **标题自身不一致。**

   §3 标题和核心主张都写“六类”，实际列出 T1–T7 七类，见 [gpu-tiling-formalization.zh-CN.md:43](/home/chzh00001/workspaces/MimIR/docs/gpu-tiling-formalization.zh-CN.md:43)。

2. **作为分析维度有用，但不是正交变换。**

   明显的依赖包括：

   - T5 必须知道 T3 的地址空间/layout 和 T4 的读写者分布；
   - T6 必须复用 T5 的 happens-before/barrier 语义；
   - T7 同时依赖 T3 的 panel layout、T4 的 lane 分布、T5 的 warp convergence；
   - `Frag` 在 T3 的 `space = Frag(...)` 和 T7 中重复承担 placement/value representation 两种职责；
   - register tiling 同时是 T1 的分块、T3 的 private placement 和 T4 的 per-thread ownership。

   因而它们适合作为“证明义务分面”，不适合作为可任意交换次序的正交变换。

3. **T3 把 placement 和 layout 合在一起过宽。**

   `Shared(group)` 的所有权、生命周期、分配作用域，与逻辑到物理坐标的 layout 是不同性质。将二者都放进 `stage(x,L,space)`，会使 Injective、lifetime、alias、barrier 等不变量纠缠。

4. **存在未覆盖的调度决策。**

   - **split-K / multi-CTA reduction**：不是现有 T4+T5 就自动覆盖，因为它涉及跨 work-group partial、第二 kernel 或 global atomic，以及 kernel 间 happens-before。
   - **warp shuffle reduction**：既不是 shared BSP，也不是普通 `Frag`；需要 warp collective、active mask、lane permutation和归约律。
   - **epilogue fusion**：T2 可覆盖纯 producer/consumer 内联，但不能表达在 accumulator fragment 上执行 epilogue、量化/转换、再决定 store layout 的阶段选择。
   - **多 kernel 数据常驻**：需要 device allocation lifetime、stream/event、kernel DAG 和 host/device token；草案仅有单 kernel region。
   - **persistent kernel / cooperative groups / cluster/TMA**：执行者和同步作用域超出 group/warp。
   - **资源合法性**：register/shared budget、最大 block size、dynamic shared opt-in 等不是值等价，但属于“完整 GPU 调度”不可缺的 feasibility predicate。
   - **向量化与合并访存**：当前被分散在 T1/T3/T4，没有明确的 vector access/alignment 契约。

### 建议

将“完整、正交分解”降格为：

> 七个相互依赖的语义分面；每个构造子可以同时产生多个分面的义务。

并至少补两个顶层维度：

- T0：kernel partition/fusion 与跨 kernel residency/synchronization；
- T8：resource/target feasibility。

split-K 则应明确区分：

- CTA 内 split-K：T4 + T5 + `ParallelFold`；
- CTA 间 split-K：T0 + global communication/atomic + reduction law。

---

## 3. 各类的参考定义、实现和合法性谓词

### 确认的问题

### T1：切分

`ExactCover` 对并行输出 box 合理，但不能同时承担 reduction chunk 的正确性。归约需要另外证明：

- chunk 覆盖 reduction domain；
- chunk 内和 chunk 间的字典序；
- 或允许重组时提供 `ParallelFold`。

草案已经写了 `OrderedChunk`，这是正确修正；但 §4 又称“覆盖可由网格公式归一化，无可信分析”，对动态 shape、cap/peel 和复杂 preimage 仍过强。

### T2：融合

给出的等式只在以下条件下成立：

- producer 纯、无可观察 effect；
- consumer 的读集合被精确覆盖；
- 不改变 producer 的求值次数所涉及的异常、非终止或随机行为；
- alias/view 生命周期合法；
- 多 consumer residency 不跨 owner。

`Covers` 只解决其中的空间覆盖，不足以独自推出整条融合等价。

### T3：layout

这里有三处重要问题：

1. **现有 `panel_layout` 的 swizzle 不是草案所说的 XOR。**

   当前实现是 chunk rotation：

   ```text
   chunk' = (chunk + group) mod chunks
   ```

   见 [lower_to_mem.cpp:1126](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:1126) 和 [lower_to_mem.cpp:1341](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:1341)。它由 `add/floordiv/rem/mul` 表达，已经能放进 semi-affine 插件；不需要 XOR semiop。

2. **split/perm/pad 不都属于同一种“affine map”。**

   - split：常数 `floordiv/rem`，属于 semi-affine；
   - perm/merge：affine/semi-affine；
   - pad：主要改变 physical shape/stride，并不单独改变逻辑坐标；
   - 当前 rotation swizzle：带常数 mod，可表达为 semi-affine；
   - 任意 XOR swizzle：不属于当前 affine grammar。

   当前 `layout_lam` 正是用 `semiop.mul/floordiv/rem` 构造，见 [lower_to_mem.cpp:1319](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:1319)。

3. **`Injective` 不是仅靠把 L 变成项就自动可判定。**

   对固定字面量 domain 可以有限枚举；对受限 Presburger/semi-affine map 也可设计专门判定器。但对一般 Mim 函数或加入按位运算后的 L，Injective 并非普通归一化就可判定。必须把 `Layout` 限定成受控代数，而不是任意函数。

### T4：分布

`ExactCover` 只适合唯一输出 owner；它不能表达：

- reduction partial 的多个 owner；
- broadcast/replication；
- cooperative fill 中多个线程共同产生一个逻辑值；
- inactive lanes 和 predication。

应把 `Dist` 的覆盖模式区分为 `UniqueOwner / Replicated / ReductionPartition / CooperativeTransfer`。

### T5：BSP + DRF

“DRF ⇒ 等价于把 forall 任意顺序化”需要补充很多前提：

- shared 非原子访问必须由 barrier 建立 happens-before；
- barrier 必须是同一作用域、同一动态实例、所有参与线程一致到达；
- barrier 必须同时提供所需的 ordering/visibility，而不只是 control rendezvous；
- 每个 superstep 内只能有确定性局部计算和互不冲突的 shared access；
- 原子、volatile、异步代理访问、warp collective 不能直接套用普通 DRF 串行化；
- 顺序参考语义必须保留每线程 program order 和 superstep 边界，不能是任意全序。

当前 `gpu.fence` 最终变成 `barrier0`，见 [nvptx.cpp:682](/home/chzh00001/workspaces/MimIR/src/mim/plug/nvptx/phase/nvptx.cpp:682)，但 `gpu.fence` 的类型本身没有 group capability，见 [gpu.mim:281](/home/chzh00001/workspaces/MimIR/src/mim/plug/gpu/gpu.mim:281)。

此外，不应依赖“warp 内隐式同步”。Volta 以后独立线程调度使这种假设尤其危险；`ldmatrix/mma` 自身是 `.sync.aligned` collective，要求所有未退出 lane 一致执行，否则行为未定义。[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 对 `ldmatrix` 明确规定了这一点。

因此草案的 T5 定理应写成“在指定 GPU memory model、barrier semantics、collective convergence 和无异步/原子例外的子语言上成立”，而不是裸 `DRF`。

### T6：异步流水

1. **`AsyncSafe ⇒ store_async ≡ store` 不是操作级等价。**

   `store_async` 返回时写可以尚未完成，显然不能与同步 store 作上下文等价。可证明的是：

   > 对一个满足等待、barrier、槽位生命周期和无冲突条件的完整 pipeline program，其最终可观察结果等价于同步 staging program。

   只有在先把全部 async copy 抽象为同步 copy 后，`wait` 才能在那个抽象程序中删除；不能在原异步语义中直接宣称 `wait_group ≡ skip`。

2. **还缺 proxy/visibility 前提。**

   PTX 明确说明 `cp.async.wait_group` 只等待当前线程提交的 cp.async group，而且不为其他普通内存操作提供 ordering/visibility；跨线程消费仍需要后续 CTA barrier。[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

3. **当前 `wait_group(n−2)` 的主体计数基本正确，但草案论证不完整。**

   当前实现实际是：

   - `depth = min(pipeline, block_count)`：[lower_to_mem.cpp:2709](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2709)；
   - prologue 先提交 `depth−1` 组：[lower_to_mem.cpp:2781](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2781)；
   - 每轮先 `wait_group(depth−2)`，再 CTA barrier：[lower_to_mem.cpp:2821](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2821)；
   - 每轮恰好再 commit 一组；
   - 尾部把最后一块重复填入释放的槽，以消除分支：[lower_to_mem.cpp:2824](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2824)；
   - 循环出口 `async_wait`/wait-all：[lower_to_mem.cpp:2790](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2790)。

   因而正确不变量不是仅“每轮 commit 一组”，还包括：

   - prologue 已有恰好 `depth−1` 个已提交组；
   - 每线程提交相同数量/顺序的组，包括空组；
   - 每轮 barrier 也保证上一轮对被复用槽的读取结束；
   - clamped 重复填充的槽不再被消费；
   - 重复填充源是合法的最后 block，且不会覆盖仍 live 的槽；
   - exit wait-all 在 shared lifetime 结束和 kernel 返回前执行；
   - `depth` 在 PTX immediate/硬件支持范围内。

   PTX 还规定同一 async group 内多个 copy 写同一位置会产生未定义行为，故 AsyncSafe 还需组内 destination-disjoint。

### T7：fragment、ldmatrix、mma

1. **`Frag(L,s,T)` 的双射模型适合 A/B/C 的逻辑 fragment。**

   对当前 m16n8k8：

   - A：32 lanes × 4 regs = 16×8；
   - B：32×2 = 8×8；
   - C：32×4 = 16×8。

   `store_fragments` 的 C 公式与 PTX 一致：`group=lane/4`、`pair=2(lane%4)`，`e<2`/`e≥2` 分别落在相差 8 的两行，见 [lower_to_mem.cpp:2561](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2561)。官方 C fragment 公式也完全相同，[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

   B 的标量读取公式也与官方 TF32 B layout 相符，见 [lower_to_mem.cpp:2540](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2540)。

2. **草案把 ldmatrix 的“地址提供布局”误称为 `L_A`。**

   `lower_mma` 的 `lane % 16`、`4·(lane/16)` 是每个 lane 传给 `ldmatrix.x4` 的 shared row 地址，见 [lower_to_mem.cpp:2525](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:2525)。但 `ldmatrix` 是 collective permutation：

   - 各 lane 提供若干矩阵行地址；
   - 硬件再把加载结果分发到可能不同的接收 lane/register。

   官方明确说明 x4 的四组地址分别由 lane 0–7、8–15、16–23、24–31 提供，[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

   所以不能简单写成“实现地址是 `L_panel ∘ L_A` 在 lane 方向的切片”。至少需要三个关系：

   ```text
   Addr_LD : provider_lane → shared row address
   Transfer_LD : (provider_lane, loaded element) → (receiver_lane, reg)
   L_A : (receiver_lane, reg) → logical A coordinate
   ```

   然后证明三者与 `L_panel` 相容。

3. **TF32 参考语义与当前实现之间存在高风险缺口。**

   `nvptx.cpp` 目前只是把普通 F32 bitcast 为 i32，再传给 TF32 MMA intrinsic，见 [nvptx.cpp:732](/home/chzh00001/workspaces/MimIR/src/mim/plug/nvptx/phase/nvptx.cpp:732)。代码中没有显式 `cvt.rna.tf32.f32`。

   PTX 把 TF32 定义为 implementation-defined 的专门 32-bit 格式，寄存器应为 `.b32`，并提供显式 F32→TF32 conversion；不能仅从“原始 F32 位模式”推出已经具有 TF32 格式。[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

   因而草案不能先声称当前实现等价于“TF32 舍入后的 map_reduce”。需要核实 LLVM intrinsic 是否隐式完成转换；若没有，这是实际 correctness bug，而不只是证明缺失。

### 建议

- T3 使用封闭的 `LayoutExpr` 代数，不接受任意函数作为可判定 layout。
- T5 明确定义 GPU memory/event semantics、barrier scope 和 collective convergence。
- T6 把定理改成整程序 refinement/equivalence。
- T7 分开 `FragLayout`、`LdMatrixAddressLayout` 和 `LdMatrixTransfer`；TF32 conversion 成为显式参考操作和 IR 操作。

---

## 4. §2 现状表与 C++ 旁表

### 确认的问题

总体事实较准确，尤其是树形合并漏洞仍然存在：

```text
pairs = s * elems
trip  = min(n, pairs)
per   = pairs / trip
```

没有处理余数，见 [lower_to_mem.cpp:3050](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:3050)。`n=8, elems=3, s=4` 仍只处理 8/12 个 pair。

需要修正的描述：

1. **“作用域转移由类型保证”只保证 level 链语法。**

   它不保证 lowering 后 region 仍保留作用域。绑定的 whole level 仍可能被跳过，见 [tile_fuse.cpp:944](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/tile_fuse.cpp:944)。这正是上次 review 指出的 scope 丢失问题。

2. **`uniform_` 不是旁表，也不是 outline 推断。**

   它是 LowerToMem 的动态上下文布尔量；`uniform(value)` 只检查当前 bound level 和是否依赖 `item_origin_`，见 [lower_to_mem.cpp:533](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:533)。

3. **`panels_` 不是 layout 数据本身。**

   它标记可按向量消费的 work-group panel；真正保存 non-identity layout 和 logical shape 的是 `layouts_`、`logical_shapes_`，声明见 [lower_to_mem.h:170](/home/chzh00001/workspaces/MimIR/include/mim/plug/tensor/phase/lower_to_mem.h:170)。

4. **旁表清单不完整。**

   与布局/放置相关的还有：

   - `fill_inner_`、`fill_vector_`；
   - `panel_reads_`；
   - `storage_shapes_`；
   - `item_grid_`、`origin_dims_`、`multiples_`；
   - `shared_fills_`；
   - `fragments_`。

   并非它们都必须进入最终 IR；部分只是分析 cache 或 lowering-local fact。草案“全部都是本该成为类型或项的信息”说得过满。

5. **`stage` 已有的 layout 与 `panel_layout` 不是同一层布局。**

   `tensor.stage(pos, tiles, perm, pad, scope)` 表达逻辑 split/permutation/padding shape，见 [tensor.mim:2272](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/tensor.mim:2272)；bank-conflict physical layout 是后续 `lower_generate` 另行选择的，见 [lower_to_mem.cpp:870](/home/chzh00001/workspaces/MimIR/src/mim/plug/tensor/phase/lower_to_mem.cpp:870)。表中应明确区分二者。

6. **“tensor core CPU 路径即 map_reduce”应更谨慎。**

   `tensor.mma` 本身是 identity annotation，但参考计算来自其包裹的原始 `map_reduce_post`；它还不是一个独立、可直接执行的 MMA 参考定义。

### 建议

把旁表分成三类：

- 必须进入 IR：chosen physical layout、fragment kind/layout、pipeline depth、ownership/scope；
- 可重算但需要验证：uniformity、dependence、coverage；
- 纯 lowering cache：memo、临时 shape、遍历状态。

不要以“消灭所有 C++ 旁表”为目标；目标应是消灭跨 pass、影响语义却无法重建的隐式信息。

---

## 5. §5 落地顺序

### 确认的问题

当前顺序不完全符合“最先抓真实 bug”的目标：

- 第 1 步把 R8 树形漏洞称作 T4+T1 ExactCover 的顺带结果，但 R8 是 reduction pair partition，不是输出 tile 的 UniqueOwner ExactCover。若先设计通用 `Dist`，可能反而延迟一个已经明确、局部且容易测试的修复。
- “归约律”被排到第 6 步，与上次 review 将其列为 P0 的结论冲突，见 [tiling-formalization-review.zh-CN.md:50](/home/chzh00001/workspaces/MimIR/docs/tiling-formalization-review.zh-CN.md:50)。
- 上次 review 的另一个 P0——levels 应索引真实 reduction descriptor，而不是输出元素类型——没有进入路线，见 [tiling-formalization-review.zh-CN.md:84](/home/chzh00001/workspaces/MimIR/docs/tiling-formalization-review.zh-CN.md:84)。
- scope-preservation validator 也应早于完整 BSP 推导，否则会在已经丢失 group region 的 IR 上分析 barrier。

### 建议

更合理的顺序是：

1. **立即修复并验证 R8 pair coverage。**  
   用 `pair = item + q·trip; pair < pairs` 的 guarded ceiling loop；加入 `n=8, elems=3` device test。这一步最先能抓到并修复真实 bug。

2. **补 `OrderedChunk`、`ParallelFold` 和 root descriptor。**  
   这是已有错误语义的入口约束，不应放到最后。

3. **加 scope/collective preservation validator。**  
   检查 binding level 未因 whole-tile 简化消失，barrier 一致可达。

4. **让 chosen physical layout 成为受控项。**  
   先支持当前 perm/split/pad/rotation 子集和有限域 injectivity checker，不急于支持任意 XOR。

5. **T7 fragment reference + 差分测试。**  
   这一步很可能较早发现 TF32 conversion、A fragment permutation或 store layout 问题。

6. **T5 的有限 BSP 验证器。**  
   只支持 affine access、结构化控制流、CTA shared、无原子/异步的子语言。

7. **最后引入 T6 pipeline transformation。**  
   因为它依赖 T5 的 barrier、槽位 lifetime 和 happens-before。

风险判断：

- **语义风险最大：T5。** GPU memory model、collective convergence、alias/dependence 很难压成一个 `DRF` Bool。
- **工程迁移风险最大：T3。** 它触及 buffer types、physical shape、vector access、mma panel 和多张旁表。
- **最先能抓真实 bug：R8 coverage；其次是 T7 的 CPU/device differential。**

---

## 6. 超出可证明范围或与上次 review 矛盾的说法

### 确认的问题

以下表述目前过强：

- “完整 GPU 调度可以分解成七类”：遗漏跨 kernel 调度、资源可行性及其他 collective。
- “每类都有 P，且 `P ⇒ 实现 ≃ 参考`”：现在只是目标 schema，尚未给出 GPU operational semantics、观察等价和组合定理。
- “证明义务的形状与论文一致”：论文处理纯函数 schedule；GPU 版本新增 memory model、并发、lifetime、数值近似和 collective UB，不能只复用论文的三项证明义务。
- “barrier 是推导出来的，不是选出来的”：必须建立的 happens-before 边是可推导的，但 barrier 放在哪、是否合并、使用 warp/CTA/mbarrier/双缓冲仍是调度选择。
- “DRF 检查在结构化 region 上可判定”：只有在访问函数、alias、调用、动态索引和控制条件均被严格限制时成立；“结构化”本身不够。
- “Injective 在字面量形状上可判定”：若 L 是受控且可执行的总函数，可以有限枚举；若允许任意 Mim closure，则还涉及终止和 totality。
- “T7 无可信分析”：ISA fragment 表、LLVM intrinsic ABI、alignment、convergence 和 target feature 仍属于可信后端边界。
- “C++ 旁表消失”：应只要求语义相关、跨 pass 的选择结果进入 IR。
- “TF32 是许可而非等价”方向正确，但仍须给出具体近似关系；不能同时使用符号 `≃` 而不说明它是 exact equality、允许重结合还是误差关系。

与上次 review 的主要矛盾：

1. 归约律从 P0 降到路线最后；
2. root descriptor 的 `To/Tp` 接口问题被遗漏；
3. scope 只被称为“链类型已保证”，没有承接 whole-level 消去反例；
4. §4 对 normalizer/可信分析边界的描述比上次 review 更乐观；
5. `if_static` 仍被近似当作条件证明机制。

### 建议

定义至少三种不同关系，不要统一写 `≃`：

```text
≡value     精确值语义相等
≈mode      在 fast/TF32/重结合许可下的数值关系
⊑obs       GPU 并发程序对可观察输出的 refinement
```

然后为每个 T 明确采用哪一种关系，以及组合规则是否成立。

---

## 未核实的点

- 未编译或运行任何测试；全程只读，没有修改文件。
- 未在 A100 上复现 R8 漏算、流水尾部或 tensor-core 数值问题。
- 未检查最终生成的 PTX/SASS，因此尚未确认 LLVM 的 `llvm.nvvm.mma.m16n8k8...tf32` intrinsic 是否会隐式插入 TF32 conversion；源码表面上只有 raw bitcast。
- 未把 `ldmatrix.x4.b16` 的全部 bit-level transfer 手工展开成 32 lane × 4 register 的形式证明；已确认草案把 provider address map 等同于 `L_A` 是不成立的。
- 未穷举所有 dynamic-shape、cap/peel、零长度 fold 和 pipeline depth 超大时的路径。
- 论文材料是 PDF 抽取文本；图中的 fragment/layout 信息以 NVIDIA 官方 PTX ISA 补充核对，未核对论文原 PDF 的排版和脚注。