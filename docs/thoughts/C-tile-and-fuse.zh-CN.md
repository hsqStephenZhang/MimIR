# C. Tile and fuse：各系统怎样表达与决定切块、融合、多级分解、布局和软件流水（调研，2026-11-19）

范围：切块的表示、谁选块大小、生产者怎样融进消费者、合法性怎样检查、流水与多缓冲怎样生成。
证据优先用本机克隆（路径:行号）；没有本地克隆的系统只给出文献里的公认做法，标 **未核实**。
克隆版本：`llvm-project/mlir` 68fddbbe0（2026-09-27）、`iree` 61759ce（2026-11-09）、`tvm` a4a2bf5（2026-06-10）、`triton` d315663（2026-05-13）、`cutlass` b2f249ac（2026-11-06）、`ThunderKittens` 67845f5f、`tilelang` 8a68bbc5。
MimIR 一侧以 `~/workspaces/MimIR-thoughts`（f12e45a50）为准，下文 `tensor.mim`、`tile_fuse.cpp` 等都指 `src/mim/plug/tensor/` 下的文件。

---

## 1. 各系统

### 1.1 MLIR：`TilingInterface`、SCF tile-and-fuse、transform 方言

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | op 实现 C++ 接口：`getIterationDomain`、`getTiledImplementation`、`getResultTilePosition`、`generateResultTileValue`，融合用 `getIterationDomainTileFromOperandTiles` 等 | `mlir/include/mlir/Interfaces/TilingInterface.td:19,47-61` |
| 块大小 | 调用者给 `SCFTilingOptions::setTileSizes` 或回调 `SCFTileSizeComputationFunction`；transform 方言里由脚本写 `structured.tile_using_for/forall`、`multitile_sizes`、`continuous_tile_sizes` | `TileUsingInterface.h:31,63,70`；`LinalgTransformOps.td:892,2208,2253,2391` |
| 融合 | `tileConsumerAndFuseProducersUsingSCF`：贪心地沿 `tensor.extract_slice` 把生产者拉进循环；每个候选问 `fusionControlFn(slice, producer, isDestination)`，返回 `ControlFnResult{yieldProducerReplacement}` | `TileUsingInterface.h:279-312,445`；`tileAndFuseProducerOfSlice` `:337`；消费者融合 `:470` |
| 合法性 | 接口契约写在散文里；transform 脚本在解释器运行时才发现不适用（silenceable failure）；逐元素融合另有 `areElementwiseOpsFusable` | `Linalg/Transforms/Transforms.h:500` |
| 流水/多缓冲 | `scf` 的 `PipeliningOption::getScheduleFn` 由调用者给每个 op 的 stage；`memref::multiBuffer` 把 alloc 扩成 N 份；`nvgpu.pipeline_shared_memory_copies` 的文档写明“没有跨 depth 迭代的依赖、共享内存够用”都是**用户责任** | `SCF/Transforms/Transforms.h:129-171`；`MemRef/Transforms/Transforms.h:135`；`NVGPUTransformOps.td:82,97-99` |

- 归约切块有 `PartialReductionOpInterface` 与 `structured.tile_reduction_using_for/forall`（`LinalgTransformOps.td:1984,2095`），合并律由 op 的 C++ 实现给出，不检查结合性。

### 1.2 IREE：dispatch 区域 + 分级的 lowering config

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | 每个根 op 带 `lowering_config`，按级给块大小；GPU 级的枚举是 `Workgroup/Reduction/PartialReduction/Serial/Thread/Subgroup/Lane`，注释给出级之间的约束（Thread 与 Subgroup/Lane 互斥，Lane 需要同一 op 也在 Subgroup 级切过） | `Codegen/Dialect/GPU/IR/IREEGPUEnums.td:488-510` |
| 块大小 | 启发式：`deduceMMASchedule`、`setMatmulLoweringConfig`；另有调优器：`LLVMGPUConstraintGenerator.cpp`（2431 行）把 workgroup/reduction/subgroup 块、mma 种类、`prefetch_num_stages` 等“旋钮”与 SMT 约束一起发出，供 Python 调优器搜索 | `Codegen/Common/GPU/GPUHeuristics.cpp:959`；`Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp:1275`；`Codegen/LLVMGPU/LLVMGPUConstraintGenerator.cpp:62-95,684,2071` |
| 融合 | 两层：图级由 `FormDispatchRegions` 以根 op 为中心吸收生产者/消费者（`isFusableWithProducer` 等启发式）；dispatch 内部复用 MLIR 的 `tileConsumerAndFuseProducersUsingSCF` | `DispatchCreation/FormDispatchRegions.cpp:299,359,921`；`Codegen/Common/TileDispatchUsingForall.cpp:354` |
| 合法性 | 融合规则是 C++ 谓词；级之间的约束是注释与 pass 内断言 | 同上 |
| 流水 | `gpu_pipeline_options.prefetch_num_stages`；ROCDL 路径的 `ROCDLPrefetchSharedMemoryPass` | `Codegen/LLVMGPU/Passes.cpp:626-628,868-870` |

### 1.3 TVM：TensorIR 调度、Ansor、MetaSchedule

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | 在 TensorIR 上逐条应用调度原语：`split/reorder`、`ComputeAt`/`ReverseComputeAt`/`ComputeInline`、`CacheRead`、`Blockize`/`Tensorize` | `include/tvm/s_tir/schedule/schedule.h:462,567,588,600,697,712` |
| 块大小 | MetaSchedule：`MultiLevelTiling` 规则按结构串生成候选，GPU 推荐 `"SSSRRSRS"`（S=空间、R=归约，每个字母一级），进化搜索 + XGBoost 代价模型 + 实测 | `src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.h:202-206`；`schedule_rule.cc:141,244,259`；`python/tvm/s_tir/meta_schedule/cost_model/xgb_model.py:294`；`search_strategy/evolutionary_search.py` |
| 融合 | `ComputeAt` 把生产者块挪进消费者循环、按需区域重算；`ReverseComputeAt` 把消费者（epilogue）挪进来 | 同上 |
| 合法性 | 每条原语有前置检查并抛 `ScheduleError`：如块必须是 complete/reduction、不是输出块、同一 scope、依赖的块都访问到 | `src/s_tir/schedule/primitive/compute_at.cc:36,81,713,720` |
| 流水/多缓冲 | 循环上的注解 `software_pipeline_stage`/`order`/`async_stages`；`InjectSoftwarePipeline` 先验证“顺序唯一、重排不破坏读后写等依赖”，再按需算每个 buffer 的版本数做多缓冲 | `src/s_tir/transform/inject_software_pipeline.cc:1046-1050,1146-1169,356-358,475` |

- Ansor 的草图规则与 TensorIR 的 `rfactor` 等只按论文了解，**未核实**到具体文件。

### 1.4 Halide（无本地克隆，**未核实**）

- 算法与调度分离；调度原语 `split/tile/reorder/compute_at/store_at/gpu_blocks/gpu_threads`。
- 融合即 `compute_at` 的粒度选择；区域由边界推断（interval analysis）算出。
- 合法性：调度只改执行顺序与存储，语义由算法定义，故“任何合法表达的调度都保持语义”，归约顺序除外（需 `rfactor` 并要求结合律，Halide 在 rfactor 中做律的识别）。
- 块大小：用户手写，或 autoscheduler（Mullapudi 2016、Adams 2019 的学习代价模型）。
- 软件流水：传统 Halide 没有一等的 GPU 异步流水（**未核实**最新版本）。

### 1.5 Triton

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | 块程序：一个 program 处理一个块，块形状是 `tl.constexpr`；warp/线程分解由编译器选的分布式布局（`LinearLayout`，F₂ 上的线性映射）决定 | `include/triton/Tools/LinearLayout.h:20-53,315` |
| 块大小 | 用户给 `triton.Config` 候选（`BLOCK_*`、`num_warps`、`num_stages`），`Autotuner` 实测挑选，无代价模型 | `python/triton/runtime/autotuner.py:19,33`；`python/tutorials/06-fused-attention.py:136,145` |
| 融合 | 用户在一个 kernel 里写多个运算（或由 Inductor 生成）；编译器不做跨 kernel 融合 | 同上 tutorial `:48-84`（在线 softmax 手写在 kernel 里） |
| 合法性 | 布局一致性由 `convert_layout` 插入保证；流水只检查“前置条件”（不处理距离 >1 的依赖、不流水外层循环），不满足就跳过 | `lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:28-38` |
| 流水/多缓冲 | `num_stages` → `AssignLatencies` → `ScheduleLoops` → `LowerLoops`（建多份 alloc、mbarrier 数组、异步拷贝）→ `PipelineExpander`；WGMMA 另有 `WGMMAPipeline.cpp`；自动 warp 特化在 `WarpSpecialization/` | `Pipeliner/SoftwarePipeliner.cpp:37-42,198,216`；`Pipeliner/LowerLoops.cpp:146,154,384,513`；`WarpSpecialization/AutomaticWarpSpecialization.cpp` |

### 1.6 CUTLASS / CuTe

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | CuTe `Layout<Shape,Stride>` 与布局代数：`composition`、`complement`、`logical_divide`、`right/left_inverse`；块与线程划分都是布局的除法/乘法 | `include/cute/layout.hpp:99,1136,1223,1305,1364,1599` |
| 块大小 | 用户模板参数（`TileShape`、`ClusterShape`）；级数可由 `StageCountAutoCarveout` 按共享内存容量算：`(容量 − 预留) / 每级字节` | `include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:55-83` |
| 融合 | 主循环固定为 GEMM；只有 epilogue 可融合，用 Epilogue Visitor Tree | `include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp` 等 |
| 合法性 | 编译期 `static_assert` + 运行期 `can_implement`（如 TMA 128 位对齐） | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:244-262` |
| 流水/多缓冲 | 手写的 collective mainloop + `PipelineTmaAsync`：`producer_acquire/commit`、`consumer_wait/release`，`PipelineState` 记槽号与相位；协议正确性靠库作者 | `include/cutlass/pipeline/sm90_pipeline.hpp:289,422-449` |

### 1.7 ThunderKittens

- 切块表示：寄存器 tile `rt<T,rows,cols,layout>` 与共享 tile `st<...>`，形状必须是 16×16 基块的倍数，`static_assert` 检查（`include/types/register/rt.cuh:82-95`，`include/types/shared/st.cuh:59-81`）。
- 块大小、融合：用户手写。
- 流水：LCF/LCSF 模板，用户填 `producer::load`、`consumer::compute/finish`，模板管 `INPUT_PIPE_STAGES` 个槽与屏障（`prototype/lcf/lcf.cuh:13-41`）。
- 合法性：只有 tile 形状与类型的 `static_assert`；同步协议在模板里，模板本身未验证。

### 1.8 TileLang

| 维度 | 做法 | 证据 |
|---|---|---|
| 切块表示 | 用户写 `T.Kernel` 网格、`T.alloc_shared/fragment`、`T.copy`、`T.gemm`、`T.Parallel`；fragment 布局由 `layout_inference` 推出 | `src/transform/layout_inference.cc:102,116,311` |
| 块大小 | 用户；或 `carver` 按硬件约束生成/排序候选；`autotune` 实测 | `tilelang/carver/README.md`；`tilelang/autotuner/tuner.py:1344` |
| 融合 | 用户写在一个 kernel 里 | — |
| 合法性 | `pipeline_planning` 用区域相交分析排 stage | `src/transform/pipeline_planning.cc:68,419,980` |
| 流水/多缓冲 | `T.Pipelined(num_stages, order, stage, sync, group)`；`inject_pipeline`、`multi_version_buffer_rewriter`；Hopper 上 `producer_consumer_ws` 在 tile-op 级拆成生产者/消费者分支并插 mbarrier | `tilelang/language/loop.py:112-135`；`src/cuda/transform/producer_consumer_ws.cc:1-15`；`src/cuda/transform/multi_version_buffer_rewriter.cc:3` |

### 1.9 Graphene（无本地克隆，**未核实**）

- 把 GPU 数据与线程都表示成 CuTe 式的分层形状/步长张量，kernel 由“对张量的分解 + 指令规格”组成；tile 与线程分解都在同一布局代数里（CGO/ASPLOS 2023 的 Graphene 论文）。
- 块大小与融合由用户/上层生成器给定；论文展示了手写的融合 kernel（如 FMHA、LayerNorm）。

### 1.10 FlashAttention 式融合

- 不是编译器机制，而是一个算法改写：把 softmax 改成带运行最大值的在线形式，使 QKᵀ、softmax、PV 能在一个 KV 块循环里融合；两个部分结果的合并按最大值的比值重标。
- 本地证据：Triton 教程手写版 `triton/python/tutorials/06-fused-attention.py:48-84`（`m_ij`、`alpha = exp2(m_i − m_ij)`）；CUTLASS `examples/88_hopper_fmha`、`77_blackwell_fmha`。
- 编译器侧：IREE 有专门的 attention op 与在线分解（见 `docs/attention-schedule-experiment.zh-CN.md` 的 IREE 源码依据）；Triton/TileLang 都靠用户写。FA3 源码在本机（`tritonbench/submodules/flash-attention`）是空的子模块，**未核实**。

### 1.11 多面体方法（简述，无本地克隆，**未核实**）

- Pluto：用 ILP 求仿射变换，使 tiling 合法（依赖距离非负）且通信量最小；块大小通常是参数或启发式。
- PPCG：Pluto 式调度 + GPU 映射（block/thread）+ 共享内存提升；流水与多缓冲基本没有。
- Tiramisu：四层 IR（算法、调度、数据布局、通信），调度命令由用户写，合法性由依赖分析检查。
- 共同点：合法性 = 依赖分析，完整而自动；短板是只覆盖仿射程序，不表示异步拷贝、warp 特化、tensor core 等 GPU 协议。

---

## 2. MimIR 的做法

### 2.1 三层分工（以代码为准）

| 层 | 表示 | 位置 |
|---|---|---|
| 语义（定律） | `law.Law` 记录：`ro`、`extents`、`request`（根，被融合的生产者带 `tensor.law.member` 标记）、合并律等 | `tensor.mim:3379` 起；构造子 `of_map_reduce`/`of_map_reduce_merging`/`fuse`/`fuse_shared`/`of_rolling`（`:3556,3586,3682,3717,3738`） |
| 调度（层级链） | 一个名义变体 `TileLevels (n, Acc, Mma)`：`Par`、`Red`（带 `bind`、`sizes`、`tail`、`promote`、`pipeline`、`fill`、`pingpong`、`merge`）、`Vec`、`Tc`、`Cluster`、`Persistent` | `tensor.mim:2585-2592`；构造函数 `serial_group`、`wg_red_ws` 等 `:2612,2689` |
| 请求 | `tile_req (root, members, levels, control)`，恒等公理 | `tensor.mim:2904` |
| 实现 | `TileFuse`：C++ 逐级递归 `Levels::parallel/reduction/mma/wgmma/fuse`，产出 `tensor.tiled`/`tile_fold`/`bind`/`stage`/`placed`/`pipeline`/`fill_tma`/`specialize`/`mma`；`LowerToMem` 再降成缓冲、循环、GPU 原语 | `tile_fuse.cpp:1078,1211,1447,1654,1683,1796`；`lower_to_mem.cpp:1281`；`lower_to_mem_stage.cpp:11,319,902,976`；`lower_to_mem_tc.cpp:64,503` |

要点（诚实的现状）：
- 2026-11-18 起 Mim 里的调度解释器已删除，`Law` 只作规范与小分析（`denote`、`traffic`、`cost`），**调度只由 C++ 的 `Levels` 实现**（`docs/tile-law-design.zh-CN.md` §10）。
- 五个按单元区分的层级类型已并成一个 `TileLevels`，嵌套合法性从**类型**挪到了 PE 求值的 `check.walk`（同文 §11）。所以“类型保证层级作用域”已不再成立。

### 2.2 谁选块大小

| 途径 | 状态 | 证据 |
|---|---|---|
| 用户在 Mim 里写层级链 | 已实现，GPU 的主要途径 | `lit/gpu/tile_*.mim`、`law_*.mim` |
| CPU 自动：`tile_group` 按根的种类套 `default_tile_config`（BLIS 式块，`group_blocks` 用寄存器与 cache 预算） | 已实现（`-X tensor:tile-fuse=2`） | `tensor.mim:2974-3013`；`phase/tile_group.cpp:224`；`group_rules.cpp:493-495` |
| 在用户给的候选中选：`law.best`（全局内存字节）、`law.fastest`（波数 × 每工作组工作量，再按字节） | 已实现；bank 冲突、寄存器压力不在模型里 | `tensor.mim:3761-3825`；`lit/tensor/law_cost.mim`、`law_traffic.mim`；同文 §26 |
| 枚举 + 规则剪枝 + 实测（`tune.py`） | 已实现，“没有代价模型” | `bench/tune/README.zh-CN.md` |
| 从规则合成调度（D4） | 只设计 | `docs/next-directions.zh-CN.md` D4 |

### 2.3 融合怎样决定

| 层 | 机制 | 证据 |
|---|---|---|
| 图级 `tensor.fuse_tensor` | 把生产者并进消费者的读映射、epilogue 并进 `post`；条件是读映射可证明单射（每个元素至多读一次，不重算） | `phase/fuse.cpp:24-59` |
| 成员组 | 用户在 `tile_req` 里列出成员，或 `tile_group` 自动收集无归约、作用域合适的逐元素生产者 | `phase/tile_group.cpp:157-200` |
| 每个生产者 | `Levels::fuse` 沿 slice 找生产者，构造 `FuseQuery`（层深、层种类、是否被更深的归约层暂存、是否归约、是否有组外使用、是否在组内），交给 Mim 函数 `control`，返回 `(fuse, yield)`；默认 `default_control`：并行层融合并 yield 组外使用者，归约层只融合无组外使用者的 | `tile_fuse.cpp:1078-1166`；`tensor.mim:2961-2972` |
| 归约生产者进工作组 tile | `split_over_items`：按 2 的幂分段给 work-item，warp 内 shuffle 合并，少于 8 份串行合并；段数受 4 个 warp、每段 ≥ 8 元素、48 KiB 共享内存约束 | `tile_fuse.cpp:1039-1076`；同文 §19、§20 |
| 驻留共享 tile | 多个读者复用一个工作组暂存的生产者 tile（`resident_`） | `tile_fuse.cpp:1103-1123` |
| 带合并律的融合 | `law.fuse`、`fuse_shared`；`of_rolling` 把在线 softmax/attention 的“运行统计 + 重标”写成定律字段，合并律由调用者担保 | `tensor.mim:3682-3760`；`lit/gpu/law_attention.mim`、`law_resident*.mim` |

### 2.4 环与流水的降级

- 调度层：`Red` 层的 `pipeline`（在飞块数）、`fill`（1 TMA，2 生产者 warpgroup）、`pingpong`。
- `TileFuse` 产出 `tensor.pipeline depth`、`fill_tma`、`specialize` 标记（`tensor.mim:2928-2941`）。
- `LowerToMem::lower_tile_fold` 把标记了的 `tile_fold` 降成 Mim 写的环协议：普通/TMA 环用 `gpu.ring_fold`，生产者 warpgroup 用 `gpu.ws_fold`（`lower_to_mem_stage.cpp:319,660-781`）。
- 协议本身在 `gpu.mim`：`ring_fold`/`ring_fold_at`、`ws_fold`、`tma_ring` 由 `mbar_ring`、`shared_ring`、`consumer_ring`、`cluster_ring`、`resumed_ring`、`pingpong_ring` 组合出 `RingSync`（`src/mim/plug/gpu/gpu.mim:933-1222`）。
- 多缓冲 = 环的 `«depth; P»` 个槽；`turns` 选择按圈展开读字面量槽，或单份步骤代码轮转槽（同文 §28）。
- 环槽与 epilogue 暂存区按构造叠放共享内存（同文 §27）。

### 2.5 检查了什么

| 检查 | 对象 | 实现 | 证据 |
|---|---|---|---|
| 单层参数 | 尾块、cluster 大小、mma 形状、向量层在最内 | 构造函数里的 `require` | `tensor.mim:2596-2598` 等；同文 §17 |
| 层间与层对根 | 嵌套、整除、目标能力、共享内存估算，34 条消息 | `check.walk`，`tile_req` 的 normalizer 用 PE 求出 | `tensor.mim:2705-2763,2850`；`docs/tile-check-rules.md`；`lit/error/` 下 72 个 `tile_*/law_*/ring_*` 测试 |
| 只有根或 lowering 才知道的 | 根把循环 d 写到输出轴 d、TMA 面板形状与 swizzle、`cannot bufferize` | `TileFuse`/`LowerToMem` 的报错（默认警告并回退，`-X tensor:strict=1` 报错） | 同文 §17 |
| 合并律 | 已识别的规约律，或 `merge_with` 由用户担保（交换幺半群、种子为单位元、步是同态） | 识别在 C++，担保是公理 | `tensor.mim:2956-2960`（`merge_with`） |
| 布局 | 面板布局单射、swizzle 与 wgmma/TMA 描述符一致 | `layout_expr` 的 F₂ 解释器、`lies_sw128`/`lies_tma128` | `tensor.mim:57-110,224-246,305` |
| 环协议 | 槽令牌 `Held`/`Vacant` 恰用一次、读槽须持有该槽令牌、互斥分支、捕获令牌的闭包 | `gpu.check_sync` | `src/mim/plug/gpu/phase/check_sync.cpp:46`；同文 §12、§14、§24 |
| 协议模型 | 单 CTA 环的 T1（读到本代数据）、T2（覆盖前读完）、无 ABA、排空当且仅当 `d ≥ 2 ∨ count = 0` | Lean 4 小步模型 + 执行轨迹的检查器，无项目自定义公理；11 个环中 7 个归入模型 | `proofs/ring/README.md`；`lit/gpu/ring_trace_*.mim` |
| 实现等于语义 | 每个用例把 `tile_req` 换成根在 CPU 上跑作参考 | `bench/validate/fuzz.py` | `bench/validate/README.zh-CN.md` |

已知没覆盖的（同文 §24、§29）：`ws_fold` 逐步轮转时令牌槽号与读位置的对应；包进名义类型的令牌；`check.walk` 放过 warp 层下的 `wg_mma`；`law.cost` 不含 bank 冲突与寄存器。

### 2.6 已实现与只设计

| 已实现（有 lit 或 bench） | 只设计或部分 |
|---|---|
| SIMT 同步面板、cp.async 环、`mma.sync`、wgmma + TMA、warp 特化 + ping-pong、cluster/multicast、persistent、TMA 存回（默认关） | 非矩形迭代域（因果 attention 的三角域）、scan 式合并、数据依赖切片（同文 §5） |
| 工作组内的融合：逐元素生产者、归约生产者分给 work-item、驻留共享 tile、在线 softmax 式的 rolling 定律 | 由定律字段在子链上算驻留盒子（同文“驻留的子调度”）：解释器删除后未继续 |
| `check.walk` 34 条规则、F₂ 布局判定、槽令牌线性检查、7 个环的 Lean 证书 | 规则合成调度（D4）、逐 kernel 翻译验证（D3 只到差分测试） |
| `law.best`/`law.fastest` 在候选中选；`tune.py` 实测 | 符号形状：只放开了无暂存面板的链 |

---

## 3. 对照表

| 系统 | 切块表示 | 块大小由谁定 | 生产者融合 | 合法性 | 流水/多缓冲 |
|---|---|---|---|---|---|
| MLIR linalg/SCF | op 的 C++ 接口；transform 脚本 | 调用者/脚本 | 沿 slice 贪心，`fusionControlFn` 决定是否融合与 yield | 散文契约；运行脚本时失败 | `getScheduleFn` 给 stage；`multiBuffer`；依赖由用户负责 |
| IREE | 每 op 的分级 lowering config（7 种级） | 启发式；SMT 约束 + 调优器 | dispatch 区域启发式 + SCF tile-and-fuse | C++ 谓词、断言 | `prefetch_num_stages` |
| TVM TensorIR/MetaSchedule | 调度原语序列 | `SSSRRSRS` 候选 + XGBoost + 进化搜索 | `ComputeAt`/`ReverseComputeAt` | 每条原语的前置检查 | 注解 + `InjectSoftwarePipeline`（验证依赖后按需版本化） |
| Halide（未核实） | 算法/调度分离 | 用户或 autoscheduler | `compute_at` | 构造上保语义（rfactor 除外） | 无一等异步流水 |
| Triton | 块程序 + 线性布局 | 用户候选 + 实测 | 用户写在 kernel 里 | 布局转换自动插入；流水前置条件不满足就跳过 | `num_stages` → 自动 lowering，自动 WS |
| CUTLASS/CuTe | 布局代数 | 模板参数；级数按 smem 算 | 只融 epilogue（EVT） | `static_assert` + `can_implement` | 手写 `PipelineTmaAsync`，未验证 |
| ThunderKittens | `rt`/`st` tile 类型 | 用户 | 用户 | tile 形状 `static_assert` | LCF/LCSF 模板 |
| TileLang | tile op + 布局推断 | 用户；carver；autotune | 用户 | 区域相交分析排 stage | `T.Pipelined`；自动 WS |
| Graphene（未核实） | 分层形状/步长张量 | 用户/生成器 | 用户 | 布局代数的形状检查 | 用户 |
| 多面体（未核实） | 仿射调度 | ILP/参数 | 依赖驱动 | 依赖分析，完整 | 基本没有 |
| **MimIR** | IR 内的 `TileLevels` 值 + `Law` 规范 | 用户链；CPU 默认配置；`law.fastest` 在候选中选；`tune.py` | Mim 写的 `control` 逐生产者决定；归约生产者分给 work-item；rolling 定律 | `require` + PE 的 `check.walk` + C++ 晚检；F₂ 布局；槽令牌线性；7 个环有 Lean 证书；差分测试 | `Red` 层的 depth/fill/pingpong → Mim 写的 `ring_fold`/`ws_fold` 协议 |

---

## 4. 新意判断

### 4.1 站得住的（需要按下面的限定来说）

1. **流水环的协议写在 IR 里，并有静态的线性令牌检查和机械化的协议模型。**
   - CUTLASS 的 `PipelineTmaAsync` API 形状几乎相同，但没有检查；MLIR 明说依赖是用户责任；Triton/TileLang 自动生成但不验证；TVM 只检查 stage/order 与读写依赖，不建模 mbarrier 相位与异步代理。
   - 限定：令牌检查有已知缺口（轮转槽、名义类型里的令牌）；Lean 证书覆盖 11 个环中的 7 个，而且是“协议模型 + 轨迹核对”，不是对生成 PTX 的证明。
2. **调度合法性规则写成 IR 里的函数，由部分求值在 elaboration 时判定。**（汇总时更正：`law.fastest`/`law.best` 只用 `check.fits` 筛掉超出共享内存的候选，并不调用 `check.walk`，见 `tensor.mim:3805-3820`；`check.verdict` 只在 `tile_req` 处调用。）
   - 对照：CUTLASS 的 `can_implement` 与 IREE 的约束都写在 C++ 里；transform 方言在解释时才失败。
   - 限定：IREE 已经把 GPU 配置的约束发成 SMT 供调优器使用（`LLVMGPUConstraintGenerator.cpp`），“规则即约束”本身不新；新意只在“规则、调度、语义在同一个 IR 里，用同一个求值器”。两份规则（Mim 与 C++ 晚检）的一致性没有证明（`tile-check-rules.md`）。
3. **在线 softmax 式的融合写成带担保的“rolling 定律”，由同一套层级去切。**
   - FlashAttention 与 Triton 教程是手写，IREE 是专门的 attention op。
   - 限定：律是“调用者担保”，不检查；在线规范化与幺半群视角本身已有文献（Milakov & Gimelshein 2018 等，**未核实**原文细节），新意只在把它做成通用构造子 `of_rolling`。

### 4.2 弱的或已被前人覆盖的

| 说法 | 已有的 |
|---|---|
| 多级 tiling（grid/CTA/warpgroup/warp/lane）作为一串级 | IREE 的 7 种级（`IREEGPUEnums.td:488-510`）、TVM 的 `SSSRRSRS`、CuTe 的分层布局 |
| 融合由控制函数逐生产者决定，可 yield 给组外 | MLIR `fusionControlFn` + `yieldProducerReplacement`（`TileUsingInterface.h:287-312`），几乎一一对应 |
| 调度是一等值、与算法分离 | Halide、TVM、transform 方言；基础论文（cgo27）已经声称 `Tileable` 是 IR 里的类型化值 |
| “类型保证层级作用域” | 已不成立：§11 后嵌套规则在 `check.walk` 里，是 PE 判定而非类型检查 |
| 布局的 F₂ 判定、swizzle 正确性 | Triton 线性布局（`LinearLayout.h:20-53`）、CuTe 布局代数更完整 |
| 按 smem 容量定级数、按波数估代价 | CUTLASS `StageCountAutoCarveout`（`sm90_gmma_builder.inl:69-83`）；波量化是常见启发式；我们的模型不含 bank 冲突与寄存器 |
| 自动 warp 特化、TMA 环、ping-pong、persistent、cluster | CUTLASS 手写、Triton 与 TileLang 自动生成都已有 |
| 归约分给 work-item 再树形/shuffle 合并（split-k、R8） | MLIR `PartialReductionOpInterface`、IREE `PartialReduction` 级、TVM `rfactor` |
| “定律”（迭代域 + 合并律 + 放置）作为切分规范 | 本质是 `TilingInterface` + `PartialReductionOpInterface` 的字段化；而且调度解释器已删除，`Law` 现在不驱动实现 |

### 4.3 建议的表述

- 把贡献落在“**可检查的 GPU 异步协议**”（令牌 + 模型 + 证书）与“**规则、调度、语义同处一个 IR 并由 PE 统一求值**”这两点，tile-and-fuse 的机制本身（级、融合控制、布局）作为工程底座，不作新意主张。
- 在写出“合法性按构造成立”之前，先补：`ws_fold` 的槽对应、Mim 规则与 C++ 晚检的一致性（让 `lower_to_mem` 调用同一个 `check.levels`），以及其余 4 个环的模型覆盖。
- 性能对照要单列：本文不涉及与 cuBLAS/Triton/CUTLASS 的同机测量，**未核实**。

## 补记（汇总时，依据 agy 核对）

- FA3 的源码与论文已在 `docs/gpu-protocol-skeletons-survey-codex.zh-CN.md` 核实过，本文写的“未核实”只针对本机的空子模块。
- 漏掉的相关工作：Cypress（PLDI'25）、Twill（OSDI'26）、Tawa（CGO'26）、ATL（PLDI'24）、Exo 2（ASPLOS'25）、Linear Layouts（ASPLOS'26）、CuTe 布局代数（arXiv 2603.02298）。其中多数在 `docs/gpu-tiling-literature.zh-CN.md` 已有调研，影响见 [README §6](README.zh-CN.md)。
