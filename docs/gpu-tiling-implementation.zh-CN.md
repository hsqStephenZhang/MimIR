# 实现细化：参照 Triton 与 IREE（2026-11-09）

由 codex-cli 0.167.0（只读沙箱）阅读 `/home/chzh00001/workspaces/triton`（d315663）与 `/home/chzh00001/workspaces/iree`（61759ce），按 `gpu-tiling-formalization.zh-CN.md` §5 路线细化的实现方案；初版为其原文，现版本由 codex 按下文修订说明重写。
抽查核实：Triton `ReduceOpToLLVM.cpp` 的 `reduceWithinThreads/reduceWithinWarps`、`Membar.cpp` 的 `MembarOrFenceAnalysis`、`LinearLayout.cpp` 的求逆，IREE `GPUNestedLayoutDistributionPatterns.cpp` 的 `DistributeMultiReduction`、`VectorReductionToGPU.cpp` 的 `groupReductionFn`、`MMASingleSubgroupLayout`、`GPUCheckResourceUsage`、Triton dot 的 `inputPrecision` 均存在于所述位置。
更正：文中建议的测试路径 `test/mim/plug/tensor/…` 不存在，MimIR 的 lit 测试位于 `lit/`（如 `lit/gpu/`、`lit/nvptx/`）。

修订说明：下文已按维护者决定重写，以 IREE 的结构化 op 多级 tiling 为主要实现参照；仅在第 4 步引用 `LinearLayout`，在第 5 步引用 MMA/ldmatrix 与 `inputPrecision`。

# 0. 基准与约束

IREE 与 MimIR 都在结构化 op 上做 op→op 多级 tiling。IREE 的
`compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td` 定义
`TilingLevel`：Workgroup、Reduction、PartialReduction、Serial、Thread、Subgroup、Lane；
`compiler/src/iree/compiler/Codegen/Common/GPU/GPUApplyTilingLevel.cpp` 的
`GPUApplyTilingLevelPass::runOnOperation` 按 lowering config 应用一级。这与 MimIR 的
`TileLevels`/`GroupLevels`/`ItemLevels`/`WarpLevels` 起点相同，故以 IREE 为基准。

硬约束不变：MimIR 没有证明项，用户不能提交证明。义务只能由构造即正确、可判定检查、许可或保守退路消解。
下列 IREE 路径和符号均已在 `/home/chzh00001/workspaces/iree` 核实；没有标为“未核实”的引用。
除已写出完整路径者外，IREE 短路径均相对 `compiler/src/iree/compiler/Codegen/`。

# 实现路线总表

| 步骤 | IREE 主参照 | MimIR 改动 | 退路 | 工作量 |
|---|---|---|---|---|
| 1. R8 覆盖 | `DistributeMultiReduction`、`groupReductionFn` | 修正 pair 分配 | 串行 fold | 小 |
| 2. 归约契约 | `TilingLevel`、`LoweringConfigAttr`、`GPUPackPartialReductions` | root descriptor 按真实 accumulator 索引 | 禁止重排 | 中 |
| 3. 作用域 | subgroup/lane tiling、`DistributeInnerTiledToLanes` | 恒等 tile 仍保留 bind | 一般 lowering | 中 |
| 4. 布局 | `TileSwizzle`、promotion；补充 `LinearLayout` | `LayoutExpr` 进入 stage | identity/取消 promotion | 大 |
| 5. MMA | `MMAIntrinsicAttr`、`MMASingleSubgroupLayout`、`multi_mma`；补充 MMA/TF32 | `FragLayout`、显式 TF32 | 普通 map-reduce | 大 |
| 6. 同步 | `value_barrier` 及各变换局部插 barrier | stage/R8/pipeline 构造同步 | 强同步或取消优化 | 中 |
| 7. 流水 | `GPUPipelining`、multi-buffering | `Pipeline` 成为 level 参数 | depth 1 | 中–大 |
| T8. 资源 | `GPUHeuristics`、`GPUCheckResourceUsage` | 估算加精确 shared 检查 | 更小配置 | 中 |

顺序：1 → 2 → 3 → 4 → 5 → 6 → 7；第 4 步同时加入 shared 预算，随后完成 T8。

# 1. 修复 R8 树形合并覆盖

## IREE 主线

- `Common/GPU/GPUNestedLayoutDistributionPatterns.cpp::DistributeMultiReduction`：线程本地归约、subgroup butterfly、shared partial、最终 subgroup 合并。
- `Common/GPU/VectorReductionToGPU.cpp::groupReductionFn`：构造 group reduction，并在 shared 阶段插入 `gpu::BarrierOp`。

不变量是每个逻辑 partial 恰好进入一个组；缺少合法 subgroup/shape 时 pattern 失败。

## MimIR 改动

漏洞在 `src/mim/plug/tensor/phase/lower_to_mem.cpp::LowerToMem::lower_bound_fold`。
当 `n=8, elems=3, s=4` 时，固定执行 `pairs/trip` 次会漏掉 pair 8…11。改为：

```text
trip = min(n, pairs)
for item in [0, trip):
    for pair = item; pair < pairs; pair += trip:
        merge(pair / elems, pair % elems)
```

删除固定次数循环，生成 `begin=item, end=pairs, step=trip` 的 `affine.For`；保留现有各轮 fence。
由商余数唯一性保证每个 pair 恰好一次。无重关联许可时退串行。

## 测试与工作量

- `lit/gpu/tile_reduce.mim`、`tile_reduce_merge.mim`：覆盖 `n=8, elems=3`、`elems<n`、`elems>n` 和 tuple accumulator。
- `lit/nvptx/tile_reduce.mim`、`tile_reduce_merge.mim`：整数 device 精确回归。

工作量：小。

# 2. 归约契约与 root descriptor

## IREE 主线

- `IREEGPUEnums.td` 区分 Reduction、PartialReduction、Serial。
- `IREEGPUAttrs.cpp::getTilingLevelName` 与 `LoweringConfigAttr` 访问器保存各级配置。
- `Common/GPU/GPUPackPartialReductions.cpp` 读取 PartialReduction、Thread、Workgroup、Subgroup 级，并检查整除和映射条件。
- `GPUApplyTilingLevelPass::runOnOperation` 一次只应用指定级。

## MimIR 改动

现有 nominal chain 已带 `Acc`，但 `tile_req` 仍用 `TileLevels (n, T)`。最小草图：

```mim
anx nom ReductionMode =
    | Ordered
    | Fast
    | ExactKnown;

anx nom ReductionLaw =
    | Serial
    | IntegerAdd
    | IntegerMul
    | Licensed: ReductionMode;

anx let RootDesc (n: Nat, Acc: *) =
    [parallel_rank reduction_rank: Nat,
     order: «n; Nat»,
     law: ReductionLaw];

axm tile_req: {T Acc: *, r: Nat, s: «r; Nat», M K: *, n: Nat}
    → [op: Tensor r s T,
       root: RootDesc (n, Acc),
       members: M,
       levels: TileLevels (n, Acc),
       control: K]
    → Tensor r s T;
```

- `src/mim/plug/tensor/tensor.mim`：增加 descriptor 与 law。
- `tile_fuse.cpp::tile_levels`、`LevelSpec`：解析 descriptor；有序归约检查 `OrderedChunk`，RedItems 检查已知 law 或许可。
- reduction lowering 使用真实 accumulator 类型。

未知用户 merge 不视为已证明结合/交换；无法判定时退串行，浮点重排只在 Fast 下使用。

## 测试与工作量

- `lit/gpu/`：tuple accumulator、非最外层有序归约 negative、未知 merge 无许可。
- `lit/nvptx/`：整数串行/并行差分；F32 只有 Fast 才选 RedItems。

工作量：中；需迁移所有 `tile_req` 构造点。

# 3. 保持作用域与 collective

## IREE 主线

- `Dialect/GPU/Transforms/DistributeInnerTiledToLanes.cpp::DistributeInnerTiledToLanesPass::runOnOperation` 分发 inner tile 到 lane。
- `GPUNestedLayoutDistributionPatterns.cpp` 显式使用 subgroup/thread 分布。
- `IREEGPUAttrs.h/.cpp` 的 `MMASingleSubgroupLayout`、`getSingleSubgroupLayout` 固定 MMA collective 范围。
- `GPUGreedilyDistributeToThreads.cpp` 用 `SCFTileAndFuseOptions` 和 `tileConsumerAndFuseProducersUsingSCF` 同时构造 tiling 与处理器映射。

## MimIR 改动

`tile_fuse.cpp::Levels::parallel` 的 `whole` 只能省略空间分块，不能删除 group/item/warp scope。

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
```

- `tensor.mim`：bind 逐步携带受控 Dist。
- `Levels::parallel`、`LevelScope`：从 Dist 派生 scope/launch，不依赖 tile 是否简化。
- `src/mim/plug/gpu/phase/outline.cpp`：检查 barrier、MMA、ldmatrix 的 scope 与一致到达。

不满足时取消 collective/MMA 或拒绝 schedule。

## 测试与工作量

- `lit/gpu/tile_outline.mim`：whole group tile 仍 outline。
- `lit/gpu/tile_mma.mim`：whole warp tile 仍保留 warp bind。
- `lit/gpu/manual_sync_kernel.mim`：分歧 collective negative。
- `lit/nvptx/tile_mma.mim`：scope 保持到目标 lowering。

工作量：中。

# 4. 物理布局成为受控项

> 进度（2026-11-10）：第一阶段把布局写成 `tensor.layout_expr` 项；第二阶段由 TileFuse 在 promotion 时选布局，
> 写成 `tensor.placed L (stage …)`，成本模型移到 `phase/panel_layout.cpp`，LowerToMem 不再选择，只解码该项。
> TileFuse 用代换 region 外延上界的方式复现 LowerToMem 特化后看到的面板形状、连续维、向量宽度和 item 读取；
> 33 个 GPU 配置（`lit/nvptx/tile_*`、`mma*` 与 `bench/gpu_tiling`）的选择与原实现逐一相同。
> `layout_expr.Data` 改为 annex，使布局值跨 phase 保持同一类型。
> XOR swizzle：`Data` 增加 `xor` 字段，`make` 检查块数为 2 的幂（异或小于块数的组号是块的置换），`IndexOps` 增加 `xor`，
> affine 侧新增 `affine.op.xor`；成本模型在银行冲突代价相同时优先 XOR（块号不需加法），`gemm_tiled_g3*` 因此由旋转改为 XOR。
> 未做：布局进入 buffer 类型。

## IREE 主线

- `Dialect/GPU/IR/GPUTileSwizzleUtils.h/.cpp` 的 `TileSwizzle` 表示 expand-shape 与 permutation。
- `Dialect/GPU/IR/PromotionImpls.cpp`、`Common/GPU/GPUPromoteMatmulOperands.cpp` 物化 promotion。
- `IREEGPUAttrs.cpp::getSingleSubgroupLayout` 为已知 intrinsic 构造 operand layout。

选择结果应进入 IR，而不是只存在 C++ 旁表。

## 补充参照

本步骤使用 Triton `LinearLayout`：`include/triton/Tools/LinearLayout.h`、`lib/Tools/LinearLayout.cpp`、
`LinearLayoutConversions.cpp`。其 GF(2) basis、`compose`、`invert`/`invertAndCompose`、秩检查用于布局组合、求逆、
XOR swizzle 与 fragment 双射，不作为多级 tiling 基准。

## MimIR 改动

```mim
anx nom LayoutExpr (r p: Nat) =
    | Identity: LayoutExpr (r, r)
    | Permute: [order: «r; Idx r»] → LayoutExpr (r, r)
    | Split: [axis tile: Nat] → LayoutExpr (r, r + 1)
    | Pad: [axis amount: Nat] → LayoutExpr (r, r)
    | Rotate: [row col width shift groups chunks: Nat] → LayoutExpr (r, r)
    | Xor: [row col vec per_phase max_phase: Nat] → LayoutExpr (r, r);
```

- `tensor.mim`：stage 携带 placement、layout、logical/physical shape。
- `lower_to_mem.cpp::panel_layout`、`mma_panel_layout` 返回布局项。
- `LowerToMem::lower_generate` 消费该项；`layout_lam` 成为解释器。
- `lower_mma`、`store_fragments` 读取同一项；`layouts_`/`fragments_` 仅作缓存。

perm/split/rotate/XOR 必须为有效域上的单射；失败用 identity/no-swizzle，必要时取消 promotion。

## 测试与工作量

- `lit/gpu/`：各构造子的 apply/组合/求逆，XOR 双射与非单射 negative。
- `lit/gpu/tile_outline_panel.mim`：和现有地址公式差分。
- `lit/gpu/tile_mma.mim`、`lit/nvptx/tile_mma.mim`：fragment/store 坐标与目标地址。

工作量：大。

# 5. Frag/MMA 参考语义与 TF32

## IREE 主线

- `IREEGPUAttrs.td::IREEGPU_MMAIntrinsicAttr`。
- `IREEGPUAttrs.h/.cpp` 的 `MMASingleSubgroupLayout`、`getSingleSubgroupLayout`、`getPerLaneElements`。
- `IREEGPUOps.td` 的 `multi_mma`。
- `Dialect/GPU/Transforms/UnrollToIntrinsics.cpp::UnrollToIntrinsicsPass::runOnOperation`。

IREE 以 intrinsic 枚举和单 subgroup layout 表为可信边界。未匹配时退一般 vector lowering。
本地 IREE 中未找到与用户级 `inputPrecision=tf32` 完全等价的字段，此点不从 IREE 推断。

## 补充参照

本步骤使用 Triton 的 `MMAv2.cpp`、ldmatrix lowering、`TritonOps.td::inputPrecision` 和
`F32DotTC.cpp::TF32x3`。已核实 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` 与
`cvt.rna.tf32.f32` 路径。

## MimIR 改动

```mim
anx nom MmaKind =
    | NvM16N8K8Tf32
    | NvM16N8K16F16
    | Reference;

anx let FragLayout (lanes regs r: Nat, shape: «r; Nat») =
    [map: [Idx lanes, Idx regs] → «d: r; Idx shape#d»];

axm f32_to_tf32: math.F32 → math.F32;
```

- `tensor.mim`：定义 `L_A/L_B/L_C`、`Addr_LD`、`Transfer_LD`、`mma_ref`；warp_mma 携带 kind/数值模式。
- `lower_to_mem.cpp::lower_mma`、`store_fragments` 从布局项生成读写。
- `gpu.mim` 与 NVPTX lowering：MMA/ldmatrix 显式携带 kind/layout；TF32 先做显式 conversion。

fragment 映射必须双射，collective 必须一致执行。TF32 是许可；不满足 shape/alignment/scope/目标能力时退普通 map-reduce。

## 测试与工作量

- `lit/gpu/tile_mma.mim`：枚举 lane/register，检查覆盖无重复；CPU reference 对比。
- `lit/gpu/tile_mma_deep.mim`：多 fragment 累加/store。
- `lit/nvptx/tile_mma.mim`：MMA 指令、显式 TF32 conversion 与误差界。
- negative：错误 shape/layout/alignment/scope。

工作量：大；依赖第 3、4 步。

# 6. T5：结构化 barrier 与值语义同步

> 结论（2026-11-10）：核对后现有实现已满足本节目标，未改代码。三种构造见 `gpu-tiling-formalization.zh-CN.md` §3.5.1；
> `gpu.fence` 本身是挂在 SSA 内存值上的 value barrier，所以不新增 `SyncScope`/`value_barrier`。
> 循环末尾的 barrier 不能改成循环后的游离 fence：`AddMem` 重新串接内存时未被使用的 fence 会作为死代码消失。

## IREE 主线

IREE 不在这些路径上做全函数 shared-memory barrier 推导；产生 shared 读写的变换结构化插同步：

- `GPUDistributeSharedMemoryCopy.cpp`：分发 shared copy，保守处理 barrier。
- `GPUVectorAlloc.cpp::allocateTensorAndWriteVector`：插 `gpu::BarrierOp`；相关路径产生 `ValueBarrierOp`。
- `GPUReuseSharedMemoryAllocs.cpp::runOnOperation`：复用 allocation 时在 liveness 边界插 barrier。
- `GPUPipelining.cpp`：调度 async copy、wait 与 barrier；前提不足则不可调度。
- `VectorReductionToGPU.cpp::groupReductionFn`：group reduction 的 shared 阶段插 barrier。
- `IREEGPUOps.td::iree_gpu.value_barrier`：输入 tensor/vector，返回等值结果，把同步附着在 SSA 值上。

GPU 内存模型前提不变：同一动态实例中的参与线程必须一致到达，barrier 提供所需地址空间的顺序和可见性；
async wait 若只保证完成，跨线程消费仍需相应 barrier。

## MimIR 改动

现有同步点已位于 `LowerToMem::lower_generate` 的 staged 填充、`lower_tile_fold` 的流水循环顶部和
`lower_bound_fold` 的 R8 各轮。目标是收拢为 stage、R8、pipeline 三种构造，而非新增全局分析器。

```mim
anx nom SyncScope =
    | Warp
    | Group;

axm value_barrier: {T: *} → [scope: SyncScope, value: T] → T;
axm barrier: [scope: SyncScope] → mem.M 0 → mem.M 0;
```

- `tensor.mim`：stage 值可显式经过 `value_barrier`。
- `lower_generate`：面板填充完成时产生同步后的 stage 值，消费者只用该值。
- `lower_bound_fold`：R8 的 write→barrier→merge 由单一 helper 构造。
- `lower_tile_fold`：环槽 wait→barrier→compute→release 由 pipeline helper 构造。
- `outline.cpp`：降为目标 workgroup barrier；只局部合并同 scope 冗余 barrier。
- `cooperative/staged/chained/async` 迁移期保留为标签，但不作为正确性依据。

构造保证范围限于静态 shared 面板、已知 LayoutExpr/Dist、结构化控制流、R8 和受控 async。
未知时依次使用更强 barrier、同步 copy、取消复用/协作分布、退串行。可选调试检查只复核一致到达、WAW/RAW/WAR、
wait 与环槽生命周期；它不推导 barrier，也不是正确性依据。

## 测试与工作量

- `lit/gpu/tile_outline_panel.mim`：stage 值同步。
- `lit/gpu/tile_reduce.mim`：R8 各轮 barrier。
- `lit/gpu/tile_pipeline.mim`：wait 后 barrier、release 后重写。
- `lit/gpu/manual_sync_kernel.mim`：分歧 negative 与局部合并。
- `lit/nvptx/` 对应文件：目标 barrier/async 顺序。

工作量：中；难点是逐个构造审清内存模型、async 完成和 collective 一致到达。

# 7. T6：pipeline 作为 level-chain 参数

> 结论（2026-11-10）：Group/Item/Warp 链的 `Red` 增加 `pipeline: Nat` 字段，构造子 `*_red_pipelined` 写入深度，`*_red` 写 0；
> TileFuse 把 0 解析为 `-X tensor:pipeline`（默认 1），深度大于 1 时把 step 写成 `tensor.pipeline depth (…)`；
> LowerToMem 从标注读深度，不再读命令行。未实现 `PipelineKind`：同步与 `cp.async` 由目标架构决定，链上没有使用者；
> `lower_tile_fold` 的环生成尚未提取为独立 helper。

## IREE 主线

- `Common/GPU/GPUPipelining.cpp` 检查 async copy、wait、barrier、ldmatrix 后做 loop pipelining。
- `Common/GPU/GPUMultiBuffering.cpp` 对 shared allocation 多缓冲。
- `LLVMGPU/Utils/ROCDLPrefetchSharedMemoryCopy.cpp::prefetchSharedMemoryCopy` 分类 copy、应用 multi-buffering，并调用 `scf::pipelineForLoop`。
- `LLVMGPU/ROCDLPrefetching.cpp::prefetchSharedMemory` 驱动 prefetch。

前提失败时 bail，不留下部分流水。

## MimIR 改动

```mim
anx nom PipelineKind =
    | Synchronous
    | AsyncCopy;

anx let Pipeline =
    [depth: Nat, kind: PipelineKind];

anx lam no_pipeline (_: []): Pipeline =
    (1, PipelineKind#Synchronous);
```

- `tensor.mim`：WarpLevels#Red 增加 pipeline 字段。
- `tile_fuse.cpp::LevelSpec`、`tile_levels`：解析字面量 depth/kind。
- `LowerToMem`：`pipeline_` 不再是 lowering 语义输入；CLI 只在默认 schedule 构造时写入项。
- `lower_tile_fold` 提取 helper，统一生成 prologue、steady state、epilogue、ring index 和 slot 生命周期。
- 第 6 步构造生成 Fill→Commit→Wait→Barrier→Compute→ReleaseSlot。

消费前必须 wait+barrier，覆盖环槽前必须 release，退出前 wait-all；流水不改变归约顺序。
动态 trip count、无 stage、目标不支持或资源不足时退 depth 1/synchronous。

## 测试与工作量

- `lit/gpu/tile_pipeline.mim`：depth 1/2/3、短循环、空 copy group、ring index。
- `lit/gpu/tile_mma_deep.mim`：多 panel 流水。
- `lit/nvptx/tile_pipeline.mim`：commit/wait/barrier、wait-all、低架构同步退路。

工作量：中–大。

# T8. 资源可行性

## IREE 主线

- `Common/GPU/GPUHeuristics.cpp::calculateTotalSharedMemoryUsedInBytes` 估算 schedule，
  `fitScheduleInSharedMemory` 缩小配置。
- `Common/GPU/GPUCheckResourceUsage.cpp::checkGPUAllocationSize` 在 lowering 后统计静态 shared、alignment 与溢出并检查目标上限。
- `LLVMGPU/LLVMGPUConstraintGenerator.cpp` 在配置期加入 shared/VGPR 等约束。

## MimIR 改动

- `tile_fuse.cpp`：估算 panels×depth、R8 partial、MMA scratch、alignment；超限依次降 depth、取消 promotion、缩 tile、取消 R8/MMA。
- `lower_to_mem.cpp`：physical layout/padding/ring 已知后精确累计 shared slot；首版可保守不复用。
- NVPTX lowering：核对 shared、items 和目标能力；48 KiB 仅是当前静态声明阈值，不当作设备总上限。

测试放在 `lit/gpu/` 的资源统计/退路用例和 `lit/nvptx/dyn_smem_kernel.mim`。工作量：中。

# 统一落地边界

| 层 | 负责内容 | 不负责内容 |
|---|---|---|
| Mim 项 | descriptor、level、Dist、LayoutExpr、Pipeline、Frag、数值模式 | 证明对象 |
| 归一化 | 字面量尺寸、排列、整除、已知 law、固定布局表 | 任意用户 lambda |
| TileFuse | cover/preimage、逐级构造、选择、资源估算、退路 | 最终物理地址 |
| LowerToMem | 布局解释、shared 构造、结构化同步、流水、精确 shared | 用户安全保证 |
| NVPTX | ISA ABI、MMA/ldmatrix、barrier/async、目标上限、TF32 | 补救错误分布或同步 |

核心取舍：

1. IREE `TilingLevel` 是 level-chain 和逐级变换的主参照。
2. barrier 由产生 shared 读写的构造建立，优先挂在 stage 值上；可选检查不承担正确性。
3. LayoutExpr、Dist、Pipeline、FragLayout 成为项，代价模型与目标表留在 C++。
4. 任意用户 combiner 只能识别已知原语、要求许可或退串行。
