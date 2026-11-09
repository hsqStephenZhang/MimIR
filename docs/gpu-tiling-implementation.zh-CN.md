# 实现细化：参照 Triton 与 IREE（2026-11-09）

由 codex-cli 0.167.0（只读沙箱）阅读 `/home/chzh00001/workspaces/triton`（d315663）与 `/home/chzh00001/workspaces/iree`（61759ce），按 `gpu-tiling-formalization.zh-CN.md` §5 路线细化的实现方案原文，未经修改。
抽查核实：Triton `ReduceOpToLLVM.cpp` 的 `reduceWithinThreads/reduceWithinWarps`、`Membar.cpp` 的 `MembarOrFenceAnalysis`、`LinearLayout.cpp` 的求逆，IREE `GPUNestedLayoutDistributionPatterns.cpp` 的 `DistributeMultiReduction`、`VectorReductionToGPU.cpp` 的 `groupReductionFn`、`MMASingleSubgroupLayout`、`GPUCheckResourceUsage`、Triton dot 的 `inputPrecision` 均存在于所述位置。
更正：文中建议的测试路径 `test/mim/plug/tensor/…` 不存在，MimIR 的 lit 测试位于 `lit/`（如 `lit/gpu/`、`lit/nvptx/`）。

# 实现路线总表

以下结论基于当前三个工作区的源码，只读核实；没有修改文件。草案中的“无证明项、用户不能提交证明”是所有方案的硬约束：合法性只能由构造保证、编译器可判定检查、显式数值许可或保守退路消解。

| 步骤 | Triton 参考 | IREE 参考 | MimIR 主要改动点 | 工作量 |
|---|---|---|---|---|
| 1. 修复 R8 树形合并 | `ReduceOpConversion`：线程内树归约、warp shuffle、必要时 shared 多轮归约 | `DistributeMultiReduction`：线程内、subgroup butterfly、shared 跨 subgroup、最终 subgroup | 修正 `pair` 分配；补覆盖检查和 `n=8, elems=3` 测试 | 小 |
| 2. 归约契约与 root descriptor | `ReduceOp` 的 region、axis、layout 决定合法 lowering；不接受用户结合律证明 | lowering config 的 partial/reduction levels；浮点重关联由选项控制 | `RootDesc Acc`、`OrderedChunk`、`ParallelFold`；levels 改按 accumulator 类型索引 | 中 |
| 3. 保持作用域与 collective | encoding 保留 lane/warp/CTA 分布；collective lowering 有固定参与范围 | subgroup/MMA layout 和 `scf.forall` 映射显式保留范围 | 整块 tile 也保留 bound region；新增 collective 一致到达检查 | 中 |
| 4. 物理布局成为项 | `LinearLayout` 的 GF(2) 基、组合、求逆、秩检查；shared XOR swizzle | `TileSwizzle`、MMA operand swizzle、promotion layout | 新增 `LayoutExpr` 项；stage 携带 placement/layout；消除语义性 `layouts_` 旁表 | 大 |
| 5. Frag/MMA 参考语义 | `NvidiaMmaEncodingAttr`、`DotOperandEncodingAttr`、`ldmatrix`/MMA lowering、`inputPrecision` | `MMAIntrinsicAttr`、`MMASingleSubgroupLayout`、`multi_mma` | `Frag`、`L_A/L_B/L_C`、`Addr_LD/Transfer_LD`；显式 `f32_to_tf32` | 大 |
| 6. 受限 BSP/barrier 推导 | `MembarOrFenceAnalysis`：跨 CFG/region fixed-point、allocation slice 冲突分析 | reduction/prefetch 等变换按结构插 barrier；未发现同等全局分析器 | shared access summary、RAW/WAR/WAW、uniform-control 检查；标签降级为提示 | 大 |
| 7. pipeline 成为链参数 | coarse schedule、latency assignment、pipeline expander、async commit/wait、mbarrier ring | ROCDL shared prefetch、multi-buffer、SCF pipeline、显式 barrier/wait | `Pipeline` 项放进 `warp_red`；`lower_tile_fold` 消费项而非 CLI 全局参数 | 中–大 |
| T8. 资源可行性 | shared allocation/liveness/复用；运行时 `OutOfResources` 检查 shared/tmem/threads | 配置期 shared/VGPR 约束；lowering 后 `GPUCheckResourceUsagePass` | TileFuse 估算筛选，LowerToMem 精确 shared 检查，后端核对目标上限 | 中 |

建议顺序是：

1 → 2 → 3 → 4 → 5 → 6 → 7 → T8 的完整闭环。

但 T8 的“静态 shared 上限检查”应在第 4 步同时加入，避免新布局和 multi-buffer 在很晚才失败。

---

# 1. 修复 R8 树形合并覆盖

## Triton

Triton 的归约 lowering 位于：

- `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`
- `lib/Conversion/TritonGPUToLLVM/ReduceScanCommon.h`
- NVIDIA warp 原语在 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp`

核心是 `ReduceOpConversion::matchAndRewrite`：

1. `reduceWithinThreads` 先归约每个线程寄存器内的元素。
2. `reduceWithinWarps` 用 shuffle/redux 完成 warp 内归约。
3. 若归约轴仍跨 warp/block，则用 `ReduceOpHelper::getInterLayout` 构造中间布局，经 `convertLayoutValues` 写入/读出 shared，再做下一轮 warp 归约。
4. shared scratch 重用前调用 `sync`；源码断言最多两轮跨 warp 合并。
5. `treeReduce` 对不能整除 arity 的尾部保留单元素组，不丢元素。

NVIDIA 的 `TargetInfo::warpReduce` 只在完整 32-lane mask、操作可映射到 NVVM redux 时使用硬件 redux；否则通用路径使用 shuffle。这里维护的不变量是：布局上的每一个归约输入恰好进入一个归约组，跨 warp 转换前后由 `LinearLayout` 描述同一逻辑位置。

退路：硬件 redux 不适用时返回 `false`，回到 shuffle/通用 combine；布局转换需要 shared 时走 scratch，不尝试不安全的 warp 隐式同步。

## IREE

相关实现：

- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`
- `compiler/src/iree/compiler/Codegen/Common/GPU/VectorReductionToGPU.cpp`

`DistributeMultiReduction` 的源码注释明确给出四阶段：

1. 每线程本地 reduction。
2. subgroup 内 butterfly shuffle。
3. 每个 subgroup 把 partial 写入 shared。
4. 一个 subgroup 读取并完成跨 subgroup reduction。

`VectorReductionToGPU.cpp` 配置 `groupReductionFn`，并用 shared allocation 和 `gpu.barrier` 支持 group reduction；缺失 subgroup size 会直接报错。

退路主要是 pattern failure 或保留到较通用的 vector/GPU lowering；不能合法映射时不生成不完整的归约树。

## 无用户证明下的消解

- 固定 pair/shuffle/shared 分配公式：**构造即正确**。
- lane/subgroup 数、布局、支持的数据类型：**可判定检查**。
- 浮点树归约改变顺序：必须由 `fast`/reassociate 模式提供**许可**。
- 不允许重关联时：串行 fold 或保持原顺序，属于**保守退路**。

## MimIR 修复

当前漏洞位于 `LowerToMem::lower_bound_fold`：

```cpp
auto pairs = s * elems;
auto trip  = std::min(*n, pairs);
auto per   = pairs / trip;
pair = item + q * trip;
```

当 `n=8, elems=3, s=4` 时：

```text
pairs = 12
trip  = 8
per   = floor(12 / 8) = 1
```

于是只处理 pair 0…7，pair 8…11 丢失。

正确分配应直接是：

```text
trip = min(n, pairs)

for item in [0, trip):
    for pair = item; pair < pairs; pair += trip:
        partial = pair / elems
        element = pair % elems
        merge(partial, partial + s, element)
```

其覆盖证明由欧几里得分解直接得到：

```text
item = pair mod trip
q    = pair div trip
pair = item + q × trip
```

所以 `[0,pairs)` 中每个 pair 恰好被一个 item 处理。

具体 C++ 修改建议：

- 在 `src/mim/plug/tensor/phase/lower_to_mem.cpp` 的 power-of-two 分支中删除 `per = pairs / trip` 和固定次数 `counting_for(per)`。
- 为每个 item 生成起点 `t`、终点 `pairs`、步长 `trip` 的 `affine.For`。
- 或使用次数：

```text
q_count(item) = ceil((pairs - item) / trip)
              = (pairs - item + trip - 1) / trip
```

但直接以 `pair` 为 IV 更少一次乘加，也更容易审查。

### 测试

新增 lit：

- `test/mim/plug/tensor/lower_bound_fold.mim`，检查生成循环为 `begin=item, end=pairs, step=trip`。
- 覆盖：
  - `n=8, elems=3`，第一轮 `s=4`，验证 pair 8…11 存在。
  - `elems=1`、`elems<n`、`elems>n`。
  - `n=1,2,4,8,32`。
  - rank 0 scalar 与 rank 1/2 tile。

新增 device bench：

- 从 `bench/gpu_fusion/softmax_r8.mim` 或 `rowsum.mim` 派生三元素输出 tile。
- 8 items，每 item 一个有唯一编码的 partial，例如 `100*item+element`，用整数加法得到可精确比对的结果。
- 再加 F32 `fast` 用例，但整数用例才是覆盖 bug 的稳定回归测试。

风险：小。应第一个完成。

---

# 2. 归约契约与 root descriptor

## Triton

`triton::ReduceOp` 的 combine 是 region；`ReduceOpConversion` 只对该 region 机械调用 `accumulate`。Triton 没有让用户提交“结合律证明”，而是：

- lowering 按 layout 构造完整归约。
- 是否可以使用目标 redux，由 `TargetInfo::warpReduce`/`matchReduxKind` 判定。
- 浮点重关联由语言/编译选项的数值语义控制，而不是 IR 中证明项。
- `ReduceOpHelper::reducedRegLaneLayout`、`getInterLayout` 把 reduction axis 与分布 layout 对齐。

## IREE

GPU lowering config 的 tiling levels在：

- `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp`
- `LoweringConfigAttr::getStaticTilingLevelSizes`
- `getTilingLevelName`

核实到的层级包括：

```text
workgroup
partial_reduction
reduction
serial
thread
subgroup
lane
```

split/partial reduction 相关实现还包括：

- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUPackPartialReductions.cpp`
- `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ReductionConfigUtils.cpp`

例如 `GPUPackPartialReductions.cpp` 会检查 partial reduction size 能否被 thread tile size 整除；不满足即失败，而非要求用户证明。

## MimIR 方案

当前 `TileLevels (n, T)` 的 `T` 来自输出元素类型，但 `tile_fold` 的真正状态是 `Acc`，两者在复合 accumulator、在线 softmax、tuple accumulator 时不同。

建议增加：

```mim
anx nom ReductionMode =
    | Ordered
    | Fast
    | ExactKnown;

anx nom ReductionLaw =
    | Serial
    | IntegerAdd
    | IntegerMul
    | IntegerMin
    | IntegerMax
    | BitAnd
    | BitOr
    | BitXor
    | Licensed: ReductionMode;

anx let ParallelFold (Acc: *) =
    [seed: Acc, merge: [Acc, Acc] → Acc, law: ReductionLaw];

anx let RootDesc (n: Nat, Acc: *) =
    [parallel_rank reduction_rank: Nat,
     accumulator: *,
     reduction_order: «n; Nat»,
     fold: ParallelFold Acc];
```

签名草图中 `accumulator: *` 与外层 `Acc` 有重复，可在真正实现时改成索引参数；重点是 descriptor 和 level chain 必须共享真实 `Acc`。

链改为：

```mim
nom TileLevels (n: Nat, Acc: *) = ...
nom GroupLevels (n: Nat, Acc: *) = ...
```

虽然现有名义定义表面上已有 `Acc` 参数，实际 `tile_req` 仍写成 `TileLevels (n, T)`；应改成：

```mim
axm tile_req: {T Acc: *, r: Nat, s: «r; Nat», M K: *, n: Nat}
            → [op: Tensor r s T,
               root: RootDesc (n, Acc),
               members: M,
               levels: TileLevels (n, Acc),
               control: K]
            → Tensor r s T;
```

新增可归一化检查：

```mim
anx lam ordered_chunk (order: «n; Nat», sizes: «n; Nat»): Bool = ...;
anx lam known_parallel_fold {Acc: *} (fold: ParallelFold Acc): Bool = ...;
```

`tile_fuse.cpp`：

- `tile_levels` 同时解析 `RootDesc`。
- reduction level 应检查只切最外层尚未固定的 reduction axis，或显式保持字典序前缀。
- `RedItems`：
  - `Ordered` 禁止。
  - `IntegerAdd` 等已知律自动允许。
  - `Licensed Fast` 允许浮点重组。
  - 任意 `group_red_merge` 不再靠注释声称结合交换；必须带 `Licensed Fast`，否则退回串行。

差异与取舍：不照搬 MLIR 的任意 dictionary attribute；`ReductionMode`、`ParallelFold`、`RootDesc` 都是 Mim 项，可被归一化和模式匹配。

测试：

- lit：tuple accumulator 的 levels 能正确实例化。
- lit-negative：切非最外层 reduction axis 且无 `Fast` 时保留请求或诊断。
- CPU/GPU differential：整数加法并行结果与串行一致。
- F32：只有显式 `Fast` 才出现 `RedItems`。

风险：中。第 1 步后立即做。

---

# 3. 作用域与 collective 的保持

## Triton

Triton 通过 encoding 中的 `register/lane/warp/block` 输入维维护执行范围。`LinearLayoutConversions.cpp` 将 MMA、dot operand 和 blocked encoding 映成这些输入维；布局简化不能无声删除 warp/CTA 参与关系。

collective lowering 还会显式检查或固定：

- warp size；
- MMA operand parent encoding；
- Hopper WGMMA 哪个 operand 可在寄存器；
- ldmatrix/MMA 的集体参与范围。

不满足时通常 verification failure 或选择非 MMA lowering。

## IREE

`MMASingleSubgroupLayout` 和 subgroup/lane tiling 明确保留 collective 范围。相关代码：

- `IREEGPUAttrs.cpp::getSingleSubgroupLayout`
- `DistributeInnerTiledToLanes.cpp`
- `GPUNestedLayoutDistributionPatterns.cpp`

缺少 subgroup size 时，`VectorReductionToGPU.cpp` 明确报错。也就是说，IREE 不把“tile 等于整个 extent”解释为“可以删除 subgroup 语义”。

## MimIR 现状问题

`Levels::parallel` 中：

```cpp
if (whole && (depth || !ro))
    return tile(anchor, rest, depth + 1);
```

会跳过整个 level。若该 level 同时携带 `bind=group/item/warp`，尺寸切分虽为空，作用域、所有权和 collective 语义却被删除。

## MimIR 方案

把“空间切分是否恒等”和“是否进入执行范围”分开：

```mim
anx nom ProcScope =
    | Serial
    | Group
    | Item
    | Warp;

anx nom CoverMode =
    | UniqueOwner
    | Replicated
    | ReductionPartition
    | CooperativeTransfer;

anx let Dist (q r: Nat) =
    [scope: ProcScope,
     processors: «q; Nat»,
     origin: «q; Nat» → «r; Nat»,
     active: «q; Nat» → Bool,
     mode: CoverMode];
```

并让 bind 携带 Dist 项：

```mim
axm bind: {T: *, q r: Nat} → [dist: Dist (q, r)] → T → T;
```

`tile_fuse.cpp`：

- `whole` 只能省略 `tiled` 的空间分块。
- `bind != 0` 时必须仍建立一个 region/bind 节点。
- `group_scope_`、`item_scope_` 从 `Dist` 派生，不再依赖某次 tile 是否被简化。
- launch 大小优先从最外层 `Dist` 推导。

新增 collective 检查器：

- MMA/ldmatrix/barrier 必须位于对应 `Warp`/`Group` scope。
- 所有参与线程在相同结构化控制路径上到达。
- 条件若依赖 `item_id/lane_id`，collective 不能位于单边分支。
- 短 bound loop 的 surplus-item 分支必须像当前 `outline.cpp` 那样把 barrier 放在分歧前；无法做到则拒绝。

消解方式：

- 固定 scope 构造：构造即正确。
- uniform control、scope nesting：可判定检查。
- 不可判定或动态分歧：保守退路，取消 collective/MMA 或拒绝该调度。

测试：

- whole-shape `serial_group` 仍产生 group kernel。
- whole warp tile 仍保留 `gpu.bind.warp`。
- divergent condition 内 barrier/MMA 的 negative lit。
- surplus items 的 barrier 位于分支前。

风险：中。必须早于布局、MMA 和 BSP。

---

# 4. 物理布局成为受控项

## Triton LinearLayout

真实实现位于：

- `include/triton/Tools/LinearLayout.h`
- `lib/Tools/LinearLayout.cpp`
- `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`

表示方式：

- 每个 input dimension 有若干二进制 basis。
- 每个 basis 是各 output dimension 上的向量。
- `apply` 将输入的置位 bit 对应 basis 做 XOR。
- 因此本质是有限 GF(2) 线性映射，输入/输出维大小要求为 2 的幂。

组合与求逆：

- `LinearLayout::compose` 对每个内层 basis 调用外层 `apply`。
- `invertAndCompose` 拼接矩阵、做 RREF，并求最小范数解。
- `invert` 要求 layout square 且 invertible。
- `pseudoinvert` 保证 `A(B(x))=x`；允许源 layout 有广播。

单射/满射：

- `checkInvariants` 用 GF(2) 矩阵秩。
- `isSurjective()`：rank 等于输出 bit 数。
- `isInjective()`：rank 等于输入 bit 数。
- `isInvertible()`：满射且输入输出规模相等。

这是可判定的有限线性代数，不依赖用户证明。

### shared XOR swizzle

`swizzledSharedToLinearLayout` 构造：

```text
col basis: (0, col)
row basis: (row, vec * ((row/perPhase) % maxPhase) mod numCols)
```

属性文档给出的等价坐标公式是：

```text
((col / vec) xor phase) * vec + col % vec
phase = (row / perPhase) % maxPhase
```

关键点：

- 对固定 row，XOR 是自身逆，所以在有效列域上是置换。
- `LinearLayout` 构造会进行秩/满射检查。
- 但 `SwizzledSharedEncodingAttr::verify` 本身只核实 order/rank；“双射保证”来自受控构造及随后 `LinearLayout` invariant，而不是这个 attr verifier 单独完成。

## IREE

IREE 的相关结构：

- `IREEGPUAttrs.cpp` 中 MMA/single-subgroup layout。
- `GPUTileSwizzleUtils.cpp` 中 `TileSwizzle`，由 expand-shape 和 permutation 构造 MMA operand swizzle。
- `PromotionImpls.cpp`、`GPUPromoteMatmulOperands.cpp` 物化 promotion。
- AMD async DMA 路径还使用 `iree_codegen.xor_shuffle`；测试可见 `amdgpu_lower_async_dma.mlir`。

IREE 更偏向“为已知 intrinsic 构造已知 layout”，而不是提供 Triton 那样通用的 GF(2) layout 运算库。

## MimIR 最小 LayoutExpr

MimIR 当前 `LowerToMem::Layout` 已隐含：

- `perm`
- `split`
- padded physical shape
- 块旋转 swizzle
- MMA 特殊 layout

但选择结果只存于 `layouts_`、`storage_shapes_`、`fragments_` 等旁表。

建议最小代数：

```mim
anx nom LayoutExpr (r p: Nat) =
    | Identity: LayoutExpr (r, r)
    | Permute: [order: «r; Idx r»] → LayoutExpr (r, r)
    | Split: [axis tile: Nat] → LayoutExpr (r, r + 1)
    | Pad: [axis amount: Nat] → LayoutExpr (r, r)
    | Rotate: [row col width shift groups chunks: Nat] → LayoutExpr (r, r)
    | Xor: [row col vec per_phase max_phase: Nat] → LayoutExpr (r, r)
    | MmaFragment: [kind operand: Nat] → LayoutExpr (2, 2)
    | Compose: {m: Nat} → [LayoutExpr (r, m), LayoutExpr (m, p)]
             → LayoutExpr (r, p);

anx let Placement =
    | Global
    | Shared: Nat
    | Private: Nat;
```

这里 `Pad` 应解释为物理 extent/stride 的改变，逻辑坐标仍不变；不应伪装成坐标置换。

解释器与检查：

```mim
anx lam layout_shape {r p: Nat}
    (layout: LayoutExpr (r, p), logical: «r; Nat»): «p; Nat» = ...;

anx lam layout_apply {r p: Nat}
    (layout: LayoutExpr (r, p), logical: «r; Nat», x: «r; Nat»): «p; Nat» = ...;

axm layout_injective: {r p: Nat}
    → [layout: LayoutExpr (r, p), logical: «r; Nat»] → Bool;
```

`layout_injective` 不接受任意 Mim lambda，只接受封闭 `LayoutExpr`：

- Permute：检查为排列。
- Split：`tile>0`，商余数构造即双射到有效域。
- Pad：逻辑点映射仍单射。
- Rotate：固定 row 上加法 mod chunks 是置换。
- Xor：要求 `vec/per_phase/max_phase` 为非零 2 的幂，且列域容纳 phase；然后构造即双射。
- Compose：各子项单射则单射。
- MMA fragment：来自后端内置表。

对静态小形状可再有限枚举作为 debug checker，与专门判定器差分。

`stage` 改为：

```mim
axm stage: {T: *, r p: Nat}
         → [logical_shape: «r; Nat»,
            placement: Placement,
            layout: LayoutExpr (r, p),
            physical_shape: «p; Nat»,
            pad: T]
         → Tensor r logical_shape T
         → Tensor r logical_shape T;
```

值类型保持逻辑 tensor；物理 layout 由 stage 项携带。这样不会像当前 `stage` 那样把 split 后物理 rank 暴露为逻辑 tensor rank。若迁移代价过大，可先保留现有返回类型，但必须附加 layout 项。

C++：

- `panel_layout`/`mma_panel_layout` 返回 `LayoutExpr` 项和 physical shape。
- `lower_generate` 将该项写回 stage，而非只写 `layouts_`。
- `layout_lam` 改为解释 `LayoutExpr`。
- `lower_mma`、barrier access analysis 都从同一个 layout 项读取。
- 旁表只作 memo，不再是唯一语义来源。

测试：

- 每个构造子的 normalize/apply。
- perm/split/pad/rotate 与当前 `layout_lam` 差分。
- XOR 自反：`xor(xor(x))=x`。
- 非单射 layout negative lit。
- `MmaFragment` 与 `store_fragments` 坐标差分。
- bank-conflict cost model 只测试选择质量，不参与正确性判定。

风险：大，是工程迁移最大的一步。

---

# 5. T7：Frag、ldmatrix、MMA 与 TF32

## Triton

关键结构：

- `NvidiaMmaEncodingAttr`
- `DotOperandEncodingAttr`
- `LinearLayoutConversions.cpp::{NvidiaMmaEncodingAttr,DotOperandEncodingAttr}::toLinearLayout`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp`
- `.../ConvertLayoutOpToLLVM.cpp`
- `.../DotOpToLLVM/MMAv2.cpp`
- `.../DotOpToLLVM/WGMMA.cpp`

`DotOperandEncodingAttr::verify` 检查：

- operand index 只能是 0/1。
- parent encoding 必须存在且是支持的 MMA/blocked encoding。
- Ampere/Hopper 的 `kWidth` 必填。
- Hopper WGMMA 只有允许的 operand 可以在寄存器。

MMAv2 中 TF32 指令为：

```text
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
```

`tt.dot` 的 `inputPrecision` 是显式字段，定义在 `TritonOps.td`，支持 `tf32/tf32x3/ieee/...`。`F32DotTC.cpp::TF32x3` 明确生成：

```text
cvt.rna.tf32.f32
```

然后构造三次 TF32 dot 近似。

## IREE

关键实现：

- `IREEGPUAttrs.td::MMAIntrinsicAttr`
- `IREEGPUAttrs.cpp::getSingleSubgroupLayout`
- `IREEGPUOps.td` 的 `multi_mma`
- `UnrollToIntrinsics.cpp`
- `DistributeInnerTiledToLanes.cpp`

`MMASingleSubgroupLayout` 给出 operand 在一个 subgroup 中的 lane/register 分布；`getPerLaneElements` 据此构造每 lane vector 类型。IREE 以目标 intrinsic 枚举和 layout table 为可信边界。

当前 NVIDIA target intrinsic 列表中核实到 F16/BF16/F8 等 `NV_MMA_SYNC`；本次检查未找到 IREE GPU dialect 中与 Triton `inputPrecision=tf32` 完全等价的用户级精度字段，因此这一点标为：**未核实**。

## MimIR 方案

定义分布式 fragment 类型：

```mim
anx nom MmaOperand = | A | B | C;

anx nom MmaKind =
    | NvM16N8K8Tf32
    | NvM16N8K16F16
    | Reference;

anx let FragLayout (lanes regs r: Nat, shape: «r; Nat») =
    [map: [Idx lanes, Idx regs] → «d: r; Idx shape#d»];

anx let Frag {T: *} (lanes regs r: Nat, shape: «r; Nat»,
                     layout: FragLayout (lanes, regs, r, shape)): * =
    «regs; T»;
```

`Frag` 的运行时表示仍是每 lane 的寄存器 tuple；布局作为类型索引描述整个 warp 值。

固定布局项：

```mim
anx let L_A: FragLayout (32, 4, 2, (16, 8)) = ...;
anx let L_B: FragLayout (32, 2, 2, (8, 8)) = ...;
anx let L_C: FragLayout (32, 4, 2, (16, 8)) = ...;

anx let Addr_LD =
    [provider: Idx 32, matrix row: Nat] → «2; Nat»;

anx let Transfer_LD =
    [provider: Idx 32, element: Nat] → [lane: Idx 32, reg: Nat];
```

参考定义：

```mim
anx lam frag_gather ... = ...;
anx lam frag_scatter ... = ...;

anx lam mma_ref
    (a: Frag @(math.F32, 32, 4, 2, (16, 8), L_A),
     b: Frag @(math.F32, 32, 2, 2, (8, 8), L_B),
     c: Frag @(math.F32, 32, 4, 2, (16, 8), L_C))
    : Frag @(math.F32, 32, 4, 2, (16, 8), L_C) = ...;
```

GPU axioms：

```mim
axm f32_to_tf32: math.F32 → math.F32;

axm ldmatrix_frag:
    [panel: Panel, addr: Addr_LD]
    → Frag @(math.F32, 32, 4, 2, (16, 8), L_A);

axm mma_tf32:
    [Frag @(math.F32, 32, 4, 2, (16, 8), L_A),
     Frag @(math.F32, 32, 2, 2, (8, 8), L_B),
     Frag @(math.F32, 32, 4, 2, (16, 8), L_C)]
    → Frag @(math.F32, 32, 4, 2, (16, 8), L_C);
```

TF32 必须显式：

```mim
let a = frag_map f32_to_tf32 a;
let b = frag_map f32_to_tf32 b;
mma_tf32 (a, b, c)
```

`nvptx.cpp` 当前把 F32 bitcast 为 i32 后直接交给 TF32 MMA。应新增 `f32_to_tf32` lowering，发出 `cvt.rna.tf32.f32`；`mma_tf32` 只接受已转换的 fragment。若选择硬件截断模式，则另设明确许可项，不能伪装成同一语义。

C++ 迁移：

- `Fragments` 只保留缓存；真实布局来自 `Frag` 类型。
- `lower_mma` 用 `Addr_LD`/`L_A/L_B` 的项生成地址和寄存器。
- `store_fragments` 用 `L_C`，替换手写 `group/pair/row/col`。
- `gpu::buf_ldmatrix` 返回明确的 `Frag`。
- `nvptx.cpp` 校验 kind、shape、layout 与 intrinsic ABI 一致。

差分测试：

- CPU 展开 `mma_ref` 与普通 16×8×8 matmul 比对。
- 对所有 32 lanes 枚举 `L_A/L_B/L_C`，检查覆盖和无重复。
- `Addr_LD + Transfer_LD` 合成结果必须等于 `L_A`。
- GPU MMA 与 CPU reference 比对；TF32 使用允许的误差界。
- 检查 LLVM IR 中出现 `cvt.rna.tf32.f32`。
- `store_fragments` 对每个 16×8 坐标恰写一次。

消解：

- ISA layout table：构造即正确。
- shape/alignment/scope：可判定检查。
- TF32：许可。
- 不满足时：普通 FMA/map-reduce 退路。

风险：大。布局项完成后实施。

---

# 6. T5：受限 BSP 与 barrier 推导

## Triton Membar

文件：

- `include/triton/Analysis/Membar.h`
- `lib/Analysis/Membar.cpp`
- `lib/Analysis/Allocation.cpp`

核心结构：

- `AllocationSlice`：物理 allocation interval、buffer id、memdesc layout、subslice offsets/shape。
- `BlockInfo`：尚未同步的 read/write slice 集合。
- `MembarOrFenceAnalysis::resolve`：worklist fixed-point。
- “virtual block”把 basic block 在每个 region-branch 处拆开。
- `visitTerminator` 同时处理 `BranchOpInterface`、`RegionBranchOpInterface`、region terminator，因而能跨 `scf.for` 等循环传播状态。
- join 是 read/write slice 集合并集。
- 若不能证明 subslice 不相交，`AllocationSlice::intersects` 保守返回相交。
- barrier、wait 等同步点调用 `BlockInfo::sync()`。
- shared RAW/WAR/WAW 冲突触发 barrier；backend filter 可排除目标已知不需要同步的情况。
- async wait 后若下一条不是 barrier，会补 barrier。

这是“自动推导 + 保守 alias”的典型实现。

## IREE

核实到的 IREE 路径没有一个与 Triton Membar 等价的、覆盖整个函数的 shared-memory fixed-point barrier analysis。

IREE 主要由产生相关结构的 pass 插同步：

- `GPUNestedLayoutDistributionPatterns.cpp` 在跨 subgroup reduction 的 shared 阶段插入同步。
- `VectorReductionToGPU.cpp` 管理 group reduction 的 barrier。
- `ROCDLPrefetchSharedMemoryCopy.cpp` 在重排流水后删除旧 barrier，再按 read/write 状态机重建：
  - read 后遇 write：WAR barrier。
  - write 后遇 read：RAW barrier。
  - async DMA wait 放在消费 shared 前的 barrier 之前。
- 对含 global read 与 shared write 的复杂 `scf.if`，prefetch 直接判为 unpipelineable。

所以 IREE 的做法是局部“构造即正确 + 可判定拒绝”，不是通用证明器。

## 与 MimIR 当前标签的对应

当前 `LowerToMem::cooperative()`：

```text
async_fill    → async
!staging      → cooperative
fence_pending → chained
otherwise     → staged
```

`outline.cpp` 解释为：

| 标签 | 入口 barrier | 出口 barrier | async store |
|---|---:|---:|---:|
| cooperative | 是 | 是 | 否 |
| staged | 是 | 否 | 否 |
| chained | 否 | 否；等后续 fence | 否 |
| async | 否 | 否；等 wait+fence | 是 |

这些标签把“依赖事实”和“最终放置选择”混在一起。

## MimIR BSP 分析器

建议新增 C++ 数据结构：

```cpp
enum class AccessKind { Read, Write, AsyncWrite };

struct SharedRegion {
    const Def* allocation;
    LayoutExprView layout;
    AffineBox box;
};

struct Access {
    AccessKind kind;
    SharedRegion region;
    DistView distribution;
    const Def* op;
};

struct SuperstepState {
    std::vector<Access> pending_reads;
    std::vector<Access> pending_writes;
    std::vector<Access> pending_async_writes;
};
```

分析范围首版严格限制：

- `affine.For` 和已知结构化 branch。
- 静态 shared allocation。
- `LayoutExpr` 可解释的 affine/semi-affine访问。
- 无原子。
- 无任意函数间 alias。
- async 只允许 T6 生成的形态。
- barrier 条件必须 uniform。
- collective scope 已通过第 3 步检查。

判定：

- 跨 item RAW：写者分布与读者分布可能不同且区域相交，插 CTA barrier。
- WAR：环槽复用前插 barrier。
- WAW：若不能证明不同 item 写不相交，调度非法；不要仅靠 barrier 掩盖同一 superstep 的数据竞争。
- RAR：无需 barrier。
- async write → read：先 wait，再 CTA barrier。
- 相邻 barrier 合并。
- 已知同一 lane/private address 可消除同步。

新的项只需描述结构，不让用户提交证明：

```mim
anx nom SyncScope = | Warp | Group;

axm barrier: [scope: SyncScope] → mem.M 0 → mem.M 0;
axm async_wait_group: [pending: Nat] → mem.M 0 → mem.M 0;
```

`cooperative/staged/chained/async` 在迁移期可保留为 hint，但分析器必须验证 hint 产生的顺序满足推导出的边；最终删除这些语义性标签。

失败策略：

- 无法判定 alias：视为相交，多插 barrier。
- 无法证明 uniform：拒绝该 GPU schedule。
- 无法证明 WAW 无冲突：取消 cooperative distribution 或回退串行。
- 不允许用户用“我保证安全”绕过。

测试：

- RAW、WAR、WAW、RAR 四类。
- disjoint subslice 不插 barrier。
- layout 不同或动态 offset 时保守插入。
- 循环跨 iteration WAR。
- divergent barrier negative。
- async wait 后仍要求 CTA barrier。
- 现有 pipeline/RedItems/MMA bench 的 barrier 数量回归。

风险：大，语义上最难。实现范围必须保持受限。

---

# 7. T6：pipeline 作为 level-chain 参数

## Triton

相关文件：

- `Pipeliner/AssignLatencies.cpp`
- `Pipeliner/Schedule.cpp`
- `Pipeliner/ScheduleLoops.cpp`
- `Pipeliner/PipelineExpander.cpp`
- `Pipeliner/LowerLoops.cpp`
- `Pipeliner/PipeliningUtility.cpp`
- `Pipeliner/WGMMAPipeline.cpp`

机制：

1. `AssignLoadLatencies` 根据 `numStages` 和依赖层次给 load 分配 latency。
2. `CoarseSchedule` 为 op 记录 `(stage, cluster)`，并递归把依赖安排到不晚于使用者的 stage。
3. `PipelineExpander` 验证定义先于使用，生成 prologue、steady-state kernel、epilogue；短循环或动态循环需要 predicate。
4. 跨 stage value 成为 loop-carried ring state。
5. `LowerLoops.cpp` 生成 async commit/wait，按 `stageDiff` 决定 buffer 数。
6. 新架构路径用 `createBarrierAlloc`、mbarrier phase/counter 管理多个环槽。
7. `WGMMAPipeline.cpp` 根据 commit 数修正 wait count。

失败时 schedule 为空、依赖不合法、嵌套 region 不支持或短循环无法 predicate，pipeliner 会 bail，而不是生成不完整流水。

## IREE

核实的主要实现是 AMD/ROCDL：

- `ROCDLPrefetching.cpp::prefetchSharedMemory`
- `ROCDLPrefetchSharedMemoryCopy.cpp::prefetchSharedMemoryCopy`
- `Passes.cpp` 中 `prefetchNumStages >= 2` 才启用。

它会：

- 识别 global→shared copy 与 compute。
- 验证 loop trip count 足够。
- 拒绝无法安全拆分的 `scf.if`。
- multi-buffer shared alloc。
- 用 `scf::pipelineForLoop` 生成流水。
- 删除原 loop-body barrier，再按新 schedule 重建。
- async DMA 模式插 wait/asyncmark。
- 对 global→shared→register 数据流，源码限制最大有意义深度为 3。

## MimIR 当前实现

`LowerToMem::lower_tile_fold`：

- 从全局 `-X tensor:pipeline=n` 读取 `pipeline_`。
- `depth=min(pipeline_, literal_count)`。
- 找到 uniform staged panels。
- 分配 `depth` 个 ring buffer。
- prologue 填 `depth-1` 个未来 block。
- 每次 `async_commit`。
- steady state 用 `async_wait_group(ring.size()-2)`，随后 `gpu.fence`。
- 退出执行 `async_wait`。
- 依赖 `fence_pending_`、`async_fill_` 和多张旁表。

算法已接近专用 software pipeline，但调度参数不是 IR 项。

## MimIR 方案

增加：

```mim
anx nom PipelineKind =
    | Synchronous
    | AsyncCopy;

anx let Pipeline =
    [depth: Nat, kind: PipelineKind];

anx lam no_pipeline (_: []): Pipeline = (1, PipelineKind#Synchronous);
anx lam async_pipeline (depth: Nat): Pipeline =
    (depth, PipelineKind#AsyncCopy);
```

修改 level：

```mim
nom WarpLevels (n: Nat, Acc: *) =
    | Done
    | Red: [sizes: «n; Nat»,
            tail promote: Nat,
            pipeline: Pipeline,
            rest: WarpLevels (n, Acc)]
    | Mma: [shape: «n; Nat»];
```

构造器：

```mim
anx lam warp_red {n: Nat, Acc: *}
    (sizes: «n; Nat», tail promote: Nat, pipeline: Pipeline,
     rest: WarpLevels (n, Acc)): WarpLevels (n, Acc) =
    ...;
```

为迁移现有程序可临时保留旧 `warp_red`，另加 `warp_red_pipeline`，但最终以项为准。

`tile_fuse.cpp`：

- `LevelSpec` 加 `PipelineSpec`。
- 只有 reduction level 可带 pipeline。
- depth 必须 literal、`1 ≤ depth ≤ target_limit`。
- 无可流水 stage 时规范化为 depth 1。
- 动态 block count 首版退回同步；不要依赖 runtime clamp 来掩盖静态合法性。

`lower_to_mem.cpp`：

- 删除 `pipeline_` 作为语义输入；CLI 只能在构造默认 schedule 时注入 `Pipeline` 项。
- `lower_tile_fold` 从 fold body/level 项读取 depth。
- 把现有 ring 生成提取成 `lower_pipeline_fold(PipelineView, ...)`。
- 生成显式 stage plan：
  - `Fill(b+s)`
  - `Commit(b+s)`
  - `Wait(b)`
  - `Barrier(b)`
  - `Compute(b)`
  - `ReleaseSlot(b)`
- 由第 6 步 BSP 分析验证 wait/barrier，而非 `fence_pending_` 手工推断。

许可与退路：

- 同步流水重定时：构造即正确。
- async depth、trip count、panel shape、commit/wait 配对：可判定检查。
- 不满足：depth 1，同步 copy。
- pipeline 不需要数值许可；它不应改变 reduction 顺序。

测试：

- level chain 打印/归一化含 depth。
- depth 1 无 async。
- depth 2/3 的 prologue、wait count、ring index。
- block count 小于 depth 的退路。
- 空 copy group 仍 commit 的测试。
- exit 前 wait-all。
- CUDA sm<80 时 `nvptx.cpp` 同步 store 退路仍正确。
- 与原 `-X tensor:pipeline` bench 性能对照。

风险：中–大。BSP 分析之后实施。

---

# T8. 资源可行性

## Triton

shared allocation：

- `lib/Analysis/Allocation.cpp::AllocationAnalysis`
- 收集 explicit shared alloc 和 reduce/scan/convert-layout scratch。
- 解析 liveness。
- 建 interference graph。
- 以 first-fit coloring/offset 分配复用无干涉的 shared 区间。
- `Allocation::getSharedMemorySize()` 给出最终字节数。

最终检查在 `python/triton/compiler/compiler.py::CompiledKernel._init_handles`：

- `metadata.shared > max_shared_mem` → `OutOfResources`。
- tmem 超限 → `OutOfResources`。
- `num_warps*warp_size > n_max_threads` → `OutOfResources`。
- autotuner 捕获该异常并尝试其他配置。

寄存器数和 spill 来自 binary load/ptxas 元数据；源码 TODO 明确指出其检查不像 shared 那样完整前置。

消解方式：精确 allocation 是可判定检查；autotuner 选下一配置是保守退路。

## IREE

配置期：

- `GPUHeuristics.cpp` 的 `isValidSchedule` 调用 `calculateTotalSharedMemoryUsedInBytes`，要求不超过 target shared limit，并通过 `fitScheduleInSharedMemory` 缩减/选择 schedule。
- `LLVMGPUConstraintGenerator.cpp` 构造 shared-memory SMT constraints。
- 同文件对部分路径有 `max_vgprs` 约束，例如 `prod(wg_m) <= 512`、packed K 不超过 512。这是估算约束，不是最终机器寄存器分配证明。
- `ConfigUtils.cpp` 还会因 LDS 导致 occupancy 下降而拒绝 DMA 配置。

lowering 后：

- `GPUCheckResourceUsage.cpp::checkGPUAllocationSize`
- 累加所有静态 shared `memref.alloc`，处理 alignment 和溢出。
- 动态 shared allocation 直接报不支持。
- 超过 `TargetAttr.getWgp().getMaxWorkgroupMemoryBytes()` 则 pass failure。

这形成“配置期估算筛选 + lowering 后精确 shared 检查”的双层模型。

## MimIR 现状

`nvptx.cpp`：

- 枚举 kernel 中所有 `gpu.slot_shared`。
- 每个 slot 16-byte 对齐后累计。
- `Static_Shared_Limit = 48*1024`。
- 超过 48 KiB 时改用一个 dynamic shared array，并设置 `Cu_Max_Dynamic_Shared`。
- 但没有核实总量是否超过具体设备的 opt-in dynamic shared 上限。
- 没有在 TileFuse/LowerToMem 阶段对 pipeline depth、R8 partials、MMA panel 的总 shared 做统一预算。
- 未发现 MimIR 侧的寄存器压力/线程数资源检查。

## MimIR 分层方案

### 1. TileFuse：配置期估算和选择

新增目标资源项：

```mim
anx let GpuLimits =
    [max_items: Nat,
     warp_size: Nat,
     shared_bytes: Nat,
     registers_per_group: Nat,
     async_groups: Nat];

anx let ResourceEstimate =
    [shared_bytes: Nat,
     registers_per_item: Nat,
     items: Nat];
```

TileFuse 为候选 level chain 估算：

```text
shared =
    staged panels × pipeline depth
  + RedItems partials
  + fragment/layout-conversion scratch
  + alignment padding
```

寄存器估算至少计入：

```text
private accumulator
+ A/B/C fragment registers
+ vectorized temporaries
+ pipeline live values
```

用途仅是筛选候选，不作为最终正确性结论。超限时按以下顺序退路：

1. 减 pipeline depth。
2. 取消 C promotion/部分 panel promotion。
3. 缩 workgroup/warp tile。
4. 取消 RedItems/MMA。
5. 串行/标量 lowering。

### 2. LowerToMem：精确 IR 级检查

此处已经知道：

- physical layout/padding；
- 每个 shared allocation 的精确类型；
- pipeline ring 数；
- R8 partial buffer；
- scratch 生命周期。

新增 `SharedAllocationPlan`：

```cpp
struct SharedSlot {
    const Def* owner;
    uint64_t bytes;
    uint64_t alignment;
    LifetimeInterval lifetime;
};

struct KernelResources {
    uint64_t shared_bytes;
    uint64_t items;
    uint64_t estimated_registers_per_item;
};
```

像 Triton allocation analysis 一样按 lifetime 复用 slot；若首版不做复用，先求保守总和。LowerToMem 完成后：

- shared 超目标上限：回报对应 level/panel/depth。
- threads 超限：失败。
- async outstanding groups 超限：把 pipeline 降到支持深度或失败。
- 寄存器仍只能估算，不能声称精确。

LowerToMem 不宜默默改变已经成为项的 schedule；如果需要自动降级，应回到 TileFuse 重新选择，或在项中明确写入选中的最终配置。

### 3. 后端：目标真实上限

`nvptx.cpp`：

- 保留 static/dynamic shared 划分。
- 发射前检查 total shared 不超过具体 compute capability/device limit。
- launch 时核对 opt-in dynamic shared 设置结果。
- 线程数检查。
- 若可获得 ptxas register/spill 信息，记录诊断；寄存器超限最终仍由后端/driver 拒绝。

不要把 48 KiB 当硬件总上限；它只是当前源码中的 static declaration threshold。

测试：

- TileFuse 选择较小 pipeline depth。
- LowerToMem 精确统计 pad/ring/R8 partial。
- 48 KiB 以下 static、以上 dynamic。
- 超目标总 shared 的 negative test。
- thread count > target limit。
- layout padding 导致原本可行配置越限。
- pipeline depth × panel bytes 的边界测试。

消解方式：

- 静态 shared/threads/async group：可判定检查。
- 配置估算：保守筛选。
- 寄存器：估算 + 后端检查；无用户证明入口。
- 超限：更小配置或串行退路。

风险：中。

---

# 统一落地边界

建议明确划分可信边界：

| 层 | 负责内容 | 不负责内容 |
|---|---|---|
| Mim 项 | schedule、Dist、LayoutExpr、Pipeline、Frag layout、数值模式 | 不携带证明对象 |
| 归一化 | 字面量尺寸、排列、整除、已知 reduction law、固定 layout 表 | 不证明任意用户 lambda |
| TileFuse | cover/preimage、调度选择、资源估算、保守退路 | 不决定最终物理地址 |
| LowerToMem | 物理布局解释、shared access、BSP/barrier、精确 shared 资源 | 不接受“用户保证安全” |
| NVPTX | ISA ABI、ldmatrix/MMA/mbarrier、目标上限、显式 TF32 指令 | 不补救已经错误的分布或同步 |

最重要的三个取舍是：

1. `LayoutExpr`、`Dist`、`Pipeline`、`Frag` 必须成为项，但代价模型仍留在 C++。
2. 任意用户 combiner 不可能靠 MimIR 的类型系统获得结合律；只能识别已知原语、要求数值许可或退回串行。
3. barrier 推导首版必须是受限语言上的完整检查器。遇到未知 alias、动态 layout 或非一致控制流时保守插同步或拒绝，不能把现有标签当作证明。