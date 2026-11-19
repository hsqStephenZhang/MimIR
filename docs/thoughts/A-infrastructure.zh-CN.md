# A. 生产与研究编译基础设施：多级降级、GPU 代码生成与异步（2026-11-19）

问题：主流基础设施怎样组织多级降级、GPU 代码生成和异步流水？MimIR 与它们相比，哪里相同、哪里不同、能借什么？

## 0. 口径

- 只读源码与文档，没有构建或运行任何东西。
- 【本地】表示读过下列 clone 的代码，路径相对各仓库根目录，行号以下表的提交为准：

| 仓库 | 提交 | 日期 |
|---|---|---|
| `llvm-project`（`mlir/`） | `68fddbbe0` | 2026-09-27 |
| `iree` | `61759ce` | 2026-11-09 |
| `tvm` | `a4a2bf5` | 2026-06-10 |
| `triton` | `d315663` | 2026-05-13 |
| `modular` | `983d67b` | 2026-11-16 |
| `tilelang` | `8a68bbc5` | 2026-06-24 |
| `cutile-python` | `d51eab0` | 2026-07-15 |
| `cubek` | `5a75441` | 2026-11-16 |
| `cuda-oxide` | `6a9a6373` | 2026-07-18 |
| `oneDNN` | `80b7f3d` | 2026-10-02 |
| TorchInductor | `mimir-frontend/.venv` 里的 `torch 2.12.0+cu130` | — |
| MimIR（协议类型） | `MimIR-types`，`f12e45a50` | 2026-11-19 |

- 【网页】表示只读了官方页面或论文摘要；【未核实】表示既没读代码也没读原文。
- 本机没有 PDF 工具，`cgo27-paper248.pdf` 的原文没读；论文内容一律转引 `E-contributions-and-directions` §0。
- **`pmlc-docker` 不是 PlaidML。** 它是 MimIR 的 KernelBench 复现镜像（`README.md:1-20`：MimIR 与 torch-mlir 的对比，LLVM `068c6c5c0c8a`），所以 PlaidML 一节只有网页来源。
- 与已有文档的关系：
  - 协议骨架与 kernel 级证据见 `gpu-protocol-skeletons-survey-codex`；
  - 同步验证文献见 `gpu-tiling-literature` §4–§5、§9；
  - 图编译器与 DSL 的分工见 `B-graph-and-dsl`，类型系统相关工作见 `D-types-and-verification`。
  - 本文只补“基础设施怎样表示异步、检查什么”。与已有文档不一致之处在 §4.3 列出。

## 1. MimIR 这一侧的参照点

| 方面 | MimIR（`feat/protocol-types`） |
|---|---|
| IR 层级 | 只有一张 hash-consed、高阶的 `Def` 图。“层级”是插件里的 axiom（`tensor`→`affine`/`mem`→`gpu`→`nvptx`/`ll`），不是分开的 IR |
| 降级机制 | `RWPhase` 重建世界，调度与协议骨架写成 Mim 函数，由部分求值（PE）展开；默认流水在 `src/mim/plug/opt/opt.mim:35-70` |
| 异步的表示 | 协议值带索引类型：`Held (rid, depth) (slot, live)`、`Vacant`、`Ring (rid, depth) phase`、`Panels`、`InFlight`（`src/mim/plug/gpu/gpu.mim:868-1007`）；环骨架是 `SyncOf` 的实例（`gpu.mim:896-903`） |
| 检查 | 前端类型检查；PE 之后的 `gpu.check_sync`（单次使用、同槽）；`gpu.ring_trace` 提取逐角色事件，交给 Lean 证明过的检查器；之后 `gpu.erase_phases` 擦除（`opt.mim:48-50`，`proofs/ring/Ring/AsyncCode.lean:607`、`WsCode.lean:664`） |
| 覆盖面 | 11 个真实环中的 7 个（`linear-types-design` §21.2、§22.1）；变异评测 51/51（§22） |

## 2. 逐项

### 2.1 MLIR 上游（linalg、transform、vector、gpu/nvgpu/nvvm、async）

| 项 | 内容 |
|---|---|
| 性质 | 生产级基础设施，本身不是端到端编译器 |
| IR 层级 | linalg（结构化 op）→ tiling/fusion（`scf.forall`/`scf.for`）→ vector → gpu/nvgpu → nvvm → LLVM。由 dialect conversion 和 rewrite pattern 降级。调度可以写成 transform 方言里的 IR：`LinalgTransformOps.td` 定义了 59 个 op |
| 异步的表示 | **三层互不相通。**<br>• 主机侧：`async.token`/`async.execute`/`async.await`（`Async/IR/AsyncTypes.td:28`、`AsyncOps.td:35,318`）；`gpu.async.token` 只串主机上的 launch 依赖（`GPU/IR/GPUBase.td:158,196`）。<br>• 设备侧 cp.async：`nvgpu.device_async_copy` 返回 `!nvgpu.device.async.token`，`device_async_create_group` 把 token 打成组，`device_async_wait {numGroups}` 等组（`NVGPU/IR/NVGPUOps.td:194-312`，`NVGPUTypes.td:29`）。<br>• 设备侧 mbarrier/TMA/WGMMA：`!nvgpu.mbarrier.group`，`mbarrier.arrive.expect_tx`，`mbarrier.try_wait.parity`（相位是 `I1` 运行时值，`NVGPUOps.td:445-461`），`tma.async.load`，`warpgroup.mma`（`:491-631`）；NVVM 层对应 `cp.async.bulk.tensor`、`wgmma.mma_async`、`wgmma.wait.group.sync.aligned`、`fence.proxy`（`LLVMIR/NVVMOps.td:1496,4450,5175,5231`） |
| 流水 | 通用 `scf::pipelineForLoop`（`SCF/Transforms/LoopPipelining.cpp:789`）。调度由用户回调 `getScheduleFn` 给出 (op, stage) 表（`SCF/Transforms/Transforms.h:129-134`）。nvgpu 的 transform op `pipeline_shared_memory_copies`、`create_async_groups`、`rewrite_copy_as_tma`（`NVGPUTransformOps.td:37-186`） |
| 检查 | • op verifier 只查局部形状：TMA 坐标数等于描述符秩（`NVGPU/IR/NVGPUDialect.cpp:444-460`），WGMMA 的矩阵形状与类型（`:586` 起）。<br>• “每个 token 只能进一个组”只写在描述里（`NVGPUOps.td:266`），`device_async_create_group` 没有 verifier。<br>• 流水的正确性明文交给用户：“It is user responsibility to ensure that there are no dependency between `depth` iterations … by using multi-buffering”（`NVGPUTransformOps.td:97-100`）。<br>• transform 方言有句柄失效检查：消费过的句柄再用就报错，`expensive-checks` 下检查（`Transform/Interfaces/TransformInterfaces.h:117-140,637-742`）。这是**调度脚本**上的线性性，不是 payload 的同步协议 |
| 与 MimIR | • **共享**：“调度是 IR 里的值”，transform 方言与 MimIR 的 `Tileable → Tileable` 同属一类。<br>• **不同**：MLIR 的 token 只表示“哪个 copy 属于哪个组”，不带槽、相位或环的身份；mbarrier 的相位是 `i1` 值；没有任何 pass 检查 wait 的数目或相位是否正确。MimIR 把槽和相位放进类型索引，并在擦除前检查。<br>• **可借**：transform 句柄的“consumed”语义与我们 `check_sync` 的单次使用是同一个想法，可作为写论文时的类比；`getScheduleFn` 的 (op, stage) 表是描述流水的最小接口，可以作为 `pipeline_plan` 的对照 |

### 2.2 IREE

| 项 | 内容 |
|---|---|
| 性质 | 生产级端到端编译器（MLIR 之上） |
| IR 层级 | 输入 → `flow`（dispatch 区域）→ `stream`（调度与资源）→ `hal`；codegen 在每个 dispatch 内跑 linalg → vector → gpu。<br>• dispatch 形成：`DispatchCreation/FormDispatchRegions.cpp`，选项见 `DispatchCreation/Passes.td:327-345`。<br>• 每个 dispatch 用 `translation_info` 选 codegen 流水、用 `lowering_config` 给 tile 尺寸（`Codegen/Dialect/Codegen/IR/IREECodegenAttrs.td:238,306`）。<br>• LLVMGPU 的主流水是 `addGPUTileAndFusePassPipeline`（`Codegen/LLVMGPU/Passes.cpp:431`）和 `addGPUVectorDistributePassPipeline`（`:746`） |
| 异步的表示 | • 主机侧：`!stream.timepoint`（`Dialect/Stream/IR/StreamTypes.td:22-27`）。<br>• 设备侧：值语义的 `iree_gpu.value_barrier`（见 `B-graph-and-dsl` §1.4）。<br>• AMD 上用 `ROCDLPrefetchSharedMemory` 做共享内存预取（`Passes.cpp:628,870`）。<br>• **这个提交的 codegen 里没有 TMA、mbarrier 或 WGMMA**：在 `Codegen/` 下 grep `TmaAsyncLoad`、`WarpgroupMma`、`mbarrier`，非测试代码没有命中。<br>• 通用的 `GPUPipelining`/`GPUMultiBuffering` 只剩 SPIR-V 流水在用（`Codegen/SPIRV/Passes.cpp:425,481,561`） |
| 检查 | `GPUVerifyDistribution` 检查写共享资源的 op 是否处在 lane/thread 分布的上下文里（`Codegen/Common/GPU/GPUVerifyDistribution.cpp:125-127`）。这是结构检查，不涉及同步协议 |
| 证据补充 | `B-graph-and-dsl` §1.4；`attention-schedule-experiment.zh-CN.md` 的 IREE 源码对照 |
| 与 MimIR | • **共享**：编译器自己划 dispatch、自己选 tile。<br>• **不同**：IREE 当前的 NVIDIA 路径不生成 Hopper 异步协议，所以“检查自动生成的 TMA/WS 协议”在 IREE 上没有对照物；对它只能比 cp.async 级以下的东西。<br>• **可借**：`lowering_config`/`translation_info` 把“调度选择”做成附在 op 上的属性，可复现、可覆写，与我们 `tile_req` 的作用相同，适合作为评测时的接口对照 |

### 2.3 TVM（TensorIR / Relax / MetaSchedule）

| 项 | 内容 |
|---|---|
| 性质 | 生产级开源编译器，学术出身：TensorIR（Feng 等，ASPLOS'23）、MetaSchedule（Shao 等，NeurIPS'22）、Relax（Lai 等，ASPLOS'25）【网页，未读原文】 |
| IR 层级 | Relax（图层，`src/relax/transform/fuse_ops.cc`、`fuse_tir.cc`、`legalize_ops.cc`）→ 可调度的 TIR（这个提交改名 `s_tir`，`src/s_tir/schedule`）→ `tirx` → 目标代码。<br>• MetaSchedule 的搜索规则在 `src/s_tir/meta_schedule/schedule_rule/multi_level_tiling*.cc`，后处理在 `postproc/` |
| 异步的表示 | • 流水由循环注解驱动：`software_pipeline_stage`、`software_pipeline_order`、`software_pipeline_async_stages`（`src/s_tir/transform/inject_software_pipeline.cc:1146-1170`）。<br>• 异步是**作用域属性**：`async_commit_queue_scope`、`async_wait_queue_scope`、`async_wait_inflight_count`（`include/tvm/s_tir/stmt.h:37-39`）。<br>• pass 内部用 `AsyncStateGlobal`/`AsyncStateLocal` 推算 `producer_head` 和等待数（`inject_software_pipeline.cc:549-596`），再由 `inject_ptx_async_copy.cc` 变成 cp.async。<br>• TMA/mbarrier/WGMMA 只是 `tirx` 的内建函数，例如 `ptx_mbarrier_arrive_expect_tx`、`ptx_cp_async_bulk_tensor_global_to_cluster`、`ptx_wgmma_encode_matrix_descriptor`（`src/tirx/op/target_builtin/cuda.cc:127-224`）。**没有**自动生成 mbarrier 环的 pass（搜 `src/s_tir/transform` 未见）【本地，按文件名与 grep】 |
| 检查 | • schedule primitive 自带前置条件检查（TensorIR 论文的主张【网页】）。<br>• MetaSchedule 后处理 `verify_gpu_code.cc` 只查资源上限。<br>• 等待数是 pass 算出来的，没有独立检查 |
| 与 MimIR | • **共享**：调度与算法分离；把等待数作为符号表达式推导，与我们 `pipeline_plan.pending` 是同一件事。<br>• **不同**：TVM 的异步是注解加 pass 内部状态，等待数算错（相当于我们的 `pending_formula` 变异）不会被任何后续检查发现。<br>• **可借**：`async_wait_inflight_count` 是“在途组数”的显式标注，可以对应我们的 `InFlight` 见证，用作差分测试的参照 |

### 2.4 Triton（TritonGPU、流水、自动 WS、TMA、Gluon、ConSan）

| 项 | 内容 |
|---|---|
| 性质 | 生产级编译器与 tile DSL。Inductor 和 XLA 的 GPU 后端都依赖它（§2.12–§2.13） |
| IR 层级 | `tt` → `ttg`（带布局编码）→ `ttng`（NVIDIA 专用）→ LLVM/NVVM。Gluon 直接写 `ttg` 级 IR，跳过布局推断与自动流水（`python/tutorials/gluon/`） |
| 异步的表示 | • cp.async：`!ttg.async.token`，`async_copy_global_to_local`、`async_commit_group`、`async_wait {num}`（`include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:13`，`TritonGPUOps.td:47-90`）。描述里写明 `async_wait` 不做 CTA 同步，要另加 `ttg.local_barrier`（`:52-53`）。<br>• mbarrier/TMA/WGMMA：`ttng.init_barrier`、`barrier_expect`、`wait_barrier`、`arrive_barrier`、`async_tma_copy_global_to_local`、`warp_group_dot`/`warp_group_dot_wait`（`TritonNvidiaGPUOps.td:202-412`）。<br>• **aref**（NVWS 方言）：`!nvws.aref<[...]>` 包住多缓冲，`aref.put.enter/exit`、`aref.get.enter/exit` 划出读写区间，`enter` 返回缓冲和 `!ttg.async.token`，`stage`/`phase` 是可选的 `i32` 操作数（`third_party/nvidia/include/Dialect/NVWS/IR/NVWSTypes.td:34-49`，`NVWSOps.td:44-170`）。<br>• 流水：`assign_latencies` → `schedule_loops` → `pipeline`；自动 WS 在其间（`third_party/nvidia/backend/compiler.py:287-300`）。调度以 `loop.stage`/`loop.cluster` 属性表示（`TritonGPU/Transforms/PipeliningUtility.h:19-22`）。<br>• 自动 WS 的子流水：partition 调度 → `NVWSInsertAref` → `NVWSLowerAref({numStages})` → `PartitionLoops` → `LowerWarpGroup` → `ScheduleLoops`（`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95-122`）。<br>• aref 的 stage/phase 由 `AssignStagePhase.cpp`（563 行）计算，`LowerAref.cpp`（962 行）降成 mbarrier |
| 检查 | • 自动 WS 的每一步后都跑 `VerifyWarpSpecializationPartitions`（`AutomaticWarpSpecialization.cpp:27-44,96-100`）。它只调 `verifyPartitionAttrs`（`Partition.cpp:389-402`），即检查 partition 属性是否良构，不涉及同步。<br>• **ConSan**（Concurrency Sanitizer）：运行时插桩，追踪每个缓冲的“可见写/读前沿”，模拟 mbarrier 与 commit 计数，覆盖 cp.async、WGMMA、TMA store；逻辑线程分成 16 个 WS 槽加 TMA、Tensor Core、CLC 三类对端（`include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md:1-60`，`lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp`）。用 `instrumentation_mode="consan"` 打开（`compiler.py:185-186,380-381`）。它是**动态的、逐次执行的**检查，不是证明 |
| 与 MimIR | • **共享**：aref 的 enter/exit 与我们的 `await`/`release` 是同一协议形状；Triton 也是从 tile 程序**自动**生成 WS 和环。<br>• **不同**：aref 的 stage/phase 是 `i32` 运行时值，由 pass 赋值；静态检查只有 partition 属性。ConSan 在一次运行里查竞争，覆盖面取决于输入和调度；我们在擦除前对每个 kernel 做静态检查和证书检查，但模型只覆盖 7/11 个环。<br>• **可借**：<br>  1. ConSan 的“可见性前沿”模型和逻辑线程分类（WS 槽 × TMA/TC/CLC 对端）可以直接作为我们 cluster、persistent 两类尚无模型的环的**差分测试**手段，也是 Lean 模型的参考语义；<br>  2. 评测时把 Triton 自动 WS 的输出交给我们的提取器检查，可以说明方法不只适用于 MimIR 自己的 lowering |

### 2.5 Modular（Mojo / MAX kernels）

| 项 | 内容 |
|---|---|
| 性质 | 生产级。kernel 库用 Mojo 手写；Mojo 编译器建在 MLIR 上【网页】 |
| IR 层级 | 对 kernel 而言主要是 Mojo 的 `comptime` 元编程：参数在编译期求值，再展开成直线代码 |
| 异步的表示 | • `PipelineState[num_stages]` 持有 `_index`、`_phase`、`_count`，都是运行时值（`max/kernels/src/layout/tma_async.mojo:552-600`）。<br>• `ProducerConsumerPipeline` 对 producer/consumer 各维护 stage 与 phase，同步后端可插拔：默认 `NvidiaMbarBackend`，AMD 用原子计数器（`max/kernels/src/structured_kernels/pipeline.mojo:97-130`）。<br>• **线性的阶段句柄**：`ProducerStage`、`ConsumerStage` 声明为 `not Deinitable else "Must call release() ..."`，只能 `Movable`，只能通过 `release(deinit self)` 销毁（`pipeline.mojo:509-567,636-715`）。也可以用 `with` 作用域（`:610-628`）。<br>• 生产 kernel 确实在用：`blockwise_fp8_matmul_kernel.mojo:900-909` 的 `acquire_consumer()` … `input_tiles^.release()`，注释写“compiler enforces these calls”。<br>• AMD 的流水调度器在编译期从依赖图和代价模型推出调度（`max/kernels/src/pipeline/DESIGN.md:1-60`） |
| 检查 | • 编译器强制“阶段句柄必须释放、不能复制”。<br>• `verify_schedule()` 在编译期用 `constrained[]` 做 6 项结构检查，其中一项是 warp 交错下跨迭代 LDS 的安全性；文档说它抓到过真实 bug（`DESIGN.md:125-142`）。<br>• 同一文档承认：检查是事后的，不在调度时进行；有些配置组合会“silent wrong results”；只有 AMD CDNA3 一个目标（`DESIGN.md:155-190`）。<br>• 句柄里的 stage/phase 仍是运行时值；origin 参数把句柄绑到所属 pipeline 的生命周期，是否能区分两个 pipeline【未核实】 |
| 与 MimIR | • **共享**：作用域化获取与释放，加线性句柄，正是 `linear-types-design` §13.3 第 2 条想借的做法。Mojo 已经在生产 kernel 里做到了。<br>• **不同**：Mojo 的线性性管“必须 release、不能复制”；槽号与相位不在类型里；但 SM100 结构化 kernel 的 `InputProducerStage[origin, Payload, num_group_stages, …]` 在类型里带着 pipeline 的 origin、环深和面板负载（`sm100_structured/structured_kernels/tile_pipeline.mojo:809-839`，汇总时依据 agy 核对补充），环的身份因此部分在类型里。我们的 `Held (rid, depth) (s, live)` 和 `Ring … phase` 把它们放进索引，“读错槽”“同一状态 await 两次”在类型层被拒（`linear-types-design` §21.1、§21.6）。Mojo 的 kernel 是手写的；我们检查的是编译器生成的。<br>• **可借**：<br>  1. 后端可插拔的 `PipelineBackend`（mbarrier 或原子计数器）正好对应我们 `SyncOf` 的实例化，可作为非 NVIDIA 目标的设计先例；<br>  2. “事后验证调度”的 6 项检查清单，可以对照我们的 `ring_trace` 结构检查 |

### 2.6 TileLang

| 项 | 内容 |
|---|---|
| 性质 | 研究出身的开源 DSL 与编译器（Wang 等，ICLR'26），建在 TVM 之上 |
| IR 层级 | tile op（`tl.tileop.copy`、`gemm` 等）→ 布局推断 → `LowerTileOp` → TIR → CUDA C。流水在 `tilelang/cuda/pipeline.py:68-217` |
| 异步的表示 | • 自动 WS：`ProducerConsumerWarpSpecialized` 在布局推断之前、在 tile op 层把循环拆成生产者和消费者分支，插入 `mbarrier_wait_parity`/`ptx_arrive_barrier`（`src/cuda/transform/producer_consumer_ws.cc:1-21`，调用处 `pipeline.py:94-95`）。<br>• 文件头列出 v1 的限制：只支持纯 TMA；不支持带条件的循环体；每个 block 只能有一个流水循环。<br>• 流水：`PipelinePlanning` → `InjectSoftwarePipeline`（`pipeline.py:108-109`）；mbarrier 降级在 `LowerSharedBarrier`、`FuseMBarrierArriveExpectTx`（`:169-174`）；另有 `inject_fence_proxy.cc` |
| 检查 | • `VerifyParallelLoop` 用 z3 找 `T.Parallel` 里多线程写同一位置的情况，但只发 `LOG(WARNING)`（`src/transform/verify_parallel_loop.cc:73-78`）。<br>• `PreLowerSemanticCheck` 只查循环嵌套形状（`tilelang/engine/semantic_check.py:21-30`）。<br>• 同步协议没有检查 |
| 与 MimIR | • **共享**：都在 tile op 层而不是循环层做 WS 拆分。<br>• **不同**：TileLang 在 C++ pass 里直接插 parity 计数器，没有中间的协议抽象，也不检查结果。<br>• **可借**：z3 写冲突检查是 Faial 式 `AddressConflict` 的一个便宜实现（`gpu-tiling-literature` §4），可以补上我们证书检查里“各线程写入区间不冲突”这一可信假设（`linear-types-design` §21.4） |

### 2.7 cuTile（`cutile-python`）

| 项 | 内容 |
|---|---|
| 性质 | NVIDIA 的生产级 tile 语言；后端闭源 |
| IR 层级 | Python AST → HIR → IR（`src/cuda/tile/_passes/ast2hir.py`、`hir2ir.py`）→ CUDA Tile IR 字节码（`_bytecode/`）→ 外部的 `tileiras` 编译器（`_compile.py:391,522-620`，用 `subprocess` 调用） |
| 异步的表示 | • 用户层没有异步原语。`ct.load` 只带提示：`latency`（1–10）和 `allow_tma`（`_stub.py:369-392,820-839`，`_ir/ops.py:889-895`）。<br>• 内存顺序靠 Tile IR 的 token：`make_token`/`join_tokens`，`token_order.py` 按别名集合分析内存效果并串起 token（`_passes/token_order.py`，`_ir/ops.py:1714-1733`）。<br>• 流水、TMA、WS 都在闭源的 `tileiras` 里【未核实】 |
| 检查 | 这一侧只有 dtype 支持检查（`_passes/check_dtype_support.py`） |
| 与 MimIR | • **共享**：token 表达的是内存效果的顺序，与 `mem.M` 链同类。<br>• **不同**：cuTile 把异步全部藏在闭源后端；它的 token 是可复制的顺序 token，不是协议能力。<br>• 对我们的意义：cuTile 是“编译器全包异步”路线的生产代表，但无法检查也无法对照它的中间结果 |

### 2.8 cubek（CubeCL kernel 库，Rust）

| 项 | 内容 |
|---|---|
| 性质 | 开源生产级 kernel 库（Burn 生态） |
| IR 层级 | Rust 宏写成 CubeCL IR。matmul 按 batch/global/stage/tile 四层 trait 组合（`crates/cubek-matmul/src/multi_level/components/{batch,global,stage,tile}`，`global/base.rs:24,78`） |
| 异步的表示 | • 显式的 `Barrier` 对象和 parity：特化 matmul 用 full/empty 两组 barrier，`wait_parity(phase ^ 1)`/`arrive()`（`components/global/multi_stage/specialized/matmul.rs:181-279`）。<br>• 读取策略按 `async_barrier`、`async_copy`、`async_full_cyclic` 等分文件（`components/global/read/strategy/`） |
| 检查 | 只在启动时检查配置是否可行，例如 `BarrierUnavailable`、`InvalidConfig`（`definition/error.rs:11,62-63`）；不检查协议 |
| 与 MimIR | • **共享**：多级组件化。<br>• **不同**：协议手写，parity 手算。<br>• **可借**：四层 trait 的组合方式可以作为“层级是可选作用域集合”的又一个实例（`gpu-protocol-skeletons-survey-codex` §1） |

### 2.9 cuda-oxide（NVlabs，alpha）

| 项 | 内容 |
|---|---|
| 性质 | 研究性质的 rustc 后端：Rust → MIR → Pliron（Rust 写的类 MLIR 框架）→ LLVM IR → PTX（`README.md`；`crates/dialect-mir`、`dialect-nvvm`、`mir-lower`） |
| 异步的表示 | • `ManagedBarrier<State, Kind, ID>` 的 typestate 是 `Uninit → Ready → Invalidated`，`Kind` 区分 `TmaBarrier`、`MmaBarrier`、`GeneralBarrier`（`crates/cuda-device/src/barrier.rs:201-307,380,531`）。<br>• `arrive()` 返回 `BarrierToken`，但它 `#[derive(Clone, Copy)]`（`:244-246`），所以不是线性的。<br>• `try_wait_parity(parity: u32)` 的相位是普通整数（`:505`）。<br>• 另有 `DisjointSlice`，用类型保证每个线程写不同元素（`disjoint.rs:6-20`） |
| 检查 | Rust 类型系统：防住未初始化就用、失效之后再用 |
| 与 MimIR | 与 `D-types-and-verification` §2 的判断一致：它管屏障的生命周期，我们管槽、代和相位，两者正交。**可借**：我们的 `ring_idle` 不计使用次数（`linear-types-design` §21.6 的缺口）；cuda-oxide 的 `Uninit → Ready` 只能由 `init` 推进，正好是关掉这个缺口的写法 |

### 2.10 PlaidML

- 【网页】Stripe（Zerrell、Bruestle，arXiv 1903.06498，2019）是 PlaidML 的核心 IR，基于嵌套多面体模型；前端是 Tile（爱因斯坦记号式的 eDSL）。README 说后来转向 MLIR。
- pmlc 里 PXA 等方言的具体降级路径【未核实】；项目是否仍在维护【未核实】。
- 异步与 GPU 流水：没有找到资料【未核实】。
- 与 MimIR：嵌套多面体的“块套块”与我们的多级 tile 链思路相近，但年代较早，不涉及 Hopper 异步。

### 2.11 oneDNN（Intel GPU JIT）

| 项 | 内容 |
|---|---|
| 性质 | 生产级算子库 |
| IR 层级 | 仿 Halide 的语句 IR（`src/gpu/intel/jit/README.md`“Intermediate Representation”），经过 pass（`jit/pass/`：`barrier.cpp`、`slm.cpp`、`alloc.cpp` 等）后由 nGEN 直接发汇编 |
| 异步的表示 | • 卷积的流水：`inject_prefetch_pipeline`、`inject_simple_slm_buffering`、`inject_unrolling`（`src/gpu/intel/conv/jit/pipeline.hpp:30-41`）；`optimize_barrier` 删除多余的 barrier（`jit/pass/barrier.hpp:28`）。<br>• ISA 级的异步：Xe 的 SWSB/SBID 记分板由 nGEN 的 `autoSWSB` 自动分配，做法是在基本块上做依赖分析并传播（`third_party/ngen/ngen_auto_swsb.hpp:118-324,1702,2003,2655`） |
| 检查 | 没有协议检查；SBID 由分析**综合**出来，正确性依赖这个分析 |
| 与 MimIR | auto-SWSB 是“同步由编译器从依赖推出”的 ISA 级先例，相当于 `gpu-tiling-literature` §4 里 AutoSync 的工程版。我们的立场相反：同步由协议库给出，再检查实例。可以作为讨论“综合还是检查”时的对照 |

### 2.12 XLA（无本地 clone）

- 【网页，openxla.org/xla/gpu_architecture】StableHLO → HLO 级 pass（SPMD 分区、布局分配、融合）→ 每个融合**恰好生成一个 kernel**。
- matmul 和 softmax 类融合转成 Triton IR，由 Triton 生成 PTX；`backend_config` 里带 `block_m/n/k`、`num_warps`、`num_stages`。其余融合直接生成 LLVM IR，或调用 cuBLAS/cuDNN。
- 页面没有讲 TMA、WS 或异步协议；GPU 上的异步实际交给 Triton。
- 与 MimIR：XLA 是“图编译器负责融合，tile 编译器负责异步”的分层代表。我们把两层放在同一个 IR 里，这是 `B-graph-and-dsl` 已经写过的定位。

### 2.13 TorchInductor

- 【网页】Ansel 等，“PyTorch 2”，ASPLOS'24：FX 图 → Inductor 的循环级 IR → Triton（GPU）或 C++（CPU）。
- 【本地，torch 2.12】matmul 走模板：`mm_persistent_tma`、`blackwell_ws_persistent_device_tma`（`torch/_inductor/kernel/mm.py:97-129`）。后者的 jinja 模板用 `make_tensor_descriptor` 和 `tl.range(..., warp_specialize=WARP_SPECIALIZE)`（`kernel/templates/triton_blackwell_ws_persistent_device_tma_mm.py.jinja:20-39`），也就是把 TMA 和 WS 全部交给 Triton 的自动 WS（§2.4）。
- `codegen/` 下另有 `cutlass`、`cutedsl` 后端。
- 不检查同步。
- 与 MimIR：Inductor 是我们最直接的用户侧对手（输入同为 PyTorch）。它的异步正确性完全建立在 Triton 之上，因此“编译器生成的异步协议无人检查”这一动机对 Inductor 同样成立。

## 3. 对照表

“异步表示”一栏指设备侧的 cp.async、mbarrier、TMA、WGMMA。

| 系统 | 性质 | 多级机制 | 异步表示 | 槽/相位在哪 | 谁生成同步 | 静态检查 | 动态检查 | 形式化 |
|---|---|---|---|---|---|---|---|---|
| MLIR 上游 | 基础设施 | 方言 + transform 方言 | token（组）、mbarrier op | 运行时 `i1`/index | 用户或下游 | op 局部形状 | 无 | 无 |
| IREE | 生产编译器 | flow/stream/hal + codegen 流水 | 主机 timepoint；值语义 barrier；无 TMA/WGMMA | — | 编译器 | 分布结构 | 无 | 无 |
| TVM | 生产/研究 | Relax → s_tir → tirx | 作用域注解 + 内建函数 | pass 内部的符号式 | 编译器（cp.async） | 资源上限 | 无 | 无 |
| Triton | 生产编译器 | tt → ttg → ttng；NVWS aref | async token、mbarrier op、aref | aref 的 `i32` 操作数 | 编译器（自动 WS）或用户（Gluon） | partition 属性 | **ConSan** | 无 |
| Mojo/MAX | 生产库 | comptime 元编程 | `PipelineState`、线性阶段句柄 | 运行时字段 | 用户（手写） | **线性：必须 release**；AMD 调度 6 项 | debug_assert | 无 |
| TileLang | 研究/开源 | tile op → TIR | 插入 parity 计数器 | 运行时 | 编译器（自动 WS v1） | z3 写冲突（警告） | 无 | 无 |
| cuTile | 生产，闭源后端 | Python → Tile IR → tileiras | 只有提示 | 不可见 | 闭源编译器 | 不可见 | — | — |
| cubek | 生产库 | 4 层 trait | 显式 Barrier + parity | 运行时 | 用户 | 配置可行性 | 无 | 无 |
| cuda-oxide | 研究 alpha | Rust MIR → Pliron | typestate 屏障，Copy 的 token | 运行时 | 用户 | **生命周期 typestate** | 无 | 无 |
| oneDNN | 生产库 | Halide 式 IR → nGEN | ISA 记分板 | — | 编译器（auto-SWSB） | 无 | 无 | 无 |
| XLA | 生产编译器 | HLO 融合 → Triton/LLVM | 交给 Triton | — | Triton | — | — | 无 |
| Inductor | 生产编译器 | FX → 循环 IR → Triton 模板 | 交给 Triton | — | Triton | — | — | 无 |
| **MimIR** | 研究编译器 | 单一 IR + PE + RWPhase | 带索引的协议类型（擦除） | **类型索引** `(rid, depth, slot, live, phase)` | 编译器（协议库实例） | 类型 + `check_sync` | 无 | **Lean 检查器 + 逐 kernel 证书**（7/11 环） |

## 4. 新颖性主张

### 4.1 证据支持的

| 主张 | 依据 |
|---|---|
| 在所查的基础设施里，没有一个对**编译器自动生成**的 GPU 异步协议做静态检查。MimIR 在擦除前对每个 kernel 检查，并能给出 Lean 证书 | MLIR 把流水正确性写成“user responsibility”（`NVGPUTransformOps.td:97-100`）；Triton 自动 WS 的 verifier 只查 partition 属性（`Partition.cpp:389-402`）；TileLang、TVM 只插入不检查；IREE 的 NVIDIA 路径没有 Hopper 异步；XLA 和 Inductor 交给 Triton |
| 把**槽号和代（相位）放进类型索引**，在已查的生产系统里没有先例（Mojo 已把 pipeline 身份和面板负载放进类型参数） | 所有系统的 stage/phase 都是运行时值或 pass 内部状态：MLIR `I1:$phaseParity`、Triton aref 的 `Optional<I32>:$stage/$phase`、Mojo `PipelineState._index/_phase`、cuda-oxide `parity: u32`、cubek `phase ^ 1` |
| 协议既是编译器 lowering 的目标，也能手写；两条路径走同一套检查 | 已查系统里，生成路径与手写路径用两套机制：Triton 的自动 WS 对 Gluon，Inductor 的模板对 CUTLASS/CuTe DSL。MimIR 的两条路径共用 `gpu.mim` 的协议库（`linear-types-design` §15） |
| 一张 IR 覆盖图层到设备协议，调度与协议骨架都由 PE 展开 | 其余编译器都跨多个 IR，或把异步交给另一个编译器（XLA/Inductor → Triton，cuTile → tileiras）。注意：“单一 IR + PE”是基础论文的贡献，不是本工作的 |

### 4.2 证据不支持的（或必须收窄）

| 不能说 | 反证 |
|---|---|
| “首次用线性类型管理 GPU 流水阶段” | **Mojo 生产 kernel 已经在用**：`ProducerStage`/`ConsumerStage` 不可复制、必须 `release`，由编译器强制（`structured_kernels/pipeline.mojo:509-567,636-715`；`blockwise_fp8_matmul_kernel.mojo:904-909`）。我们只能说：线性性**加上依赖索引**（槽、代、相位、环 id） |
| “首次给屏障加类型状态” | cuda-oxide 的 `ManagedBarrier<Uninit/Ready/Invalidated>`（`barrier.rs:201-307`） |
| “首次用 token 表示 GPU 异步” | MLIR `nvgpu.device.async.token`、Triton `ttg.async.token`、cuTile 的内存 token、TVM 的 queue 作用域 |
| “首次检查 GPU 异步的并发错误” | Triton ConSan 已经在运行时检查 cp.async/WGMMA/TMA 的可见性（`TritonInstrument.md`）。我们的差别是**静态、逐 kernel、有证书**，但只覆盖已建模的 7 类环 |
| “调度即 IR 值是新的” | MLIR transform 方言、TVM schedule、Halide |
| “自动 WS / 自动流水是新的” | Triton 自动 WS、TileLang `ProducerConsumerWarpSpecialized`、Tawa（`D-types-and-verification` §1.1） |
| “比基础设施更通用” | MimIR 只有 NVIDIA 后端；cluster、persistent 没有模型；IREE、TVM、Mojo 覆盖 AMD 等多个目标 |
| “检查覆盖整个降级链” | 证书只覆盖 `check_sync` 时刻的残余；`mem.add_mem`、`affine.lower_for` 和后端不在范围内（`linear-types-design` §21.4） |

### 4.3 与已有文档的差异（需要修正）

- `B-graph-and-dsl` §2.8 和 §4 把 Mojo 写成“没有类型级检查（未核实）”、“`PipelineState` 是运行时值”。后一句对；但 Mojo 的阶段句柄是**编译器强制的线性类型**，§2.5 给出了源码位置。B 的 §6.1 第二条（“TK、CuTe DSL、Mojo 都靠手算槽号和相位”）仍然成立，因为线性性不涉及槽号算术。
- `gpu-protocol-skeletons-survey-codex` [L38] 引用的 IREE `GPUPipelining.cpp`：在 `61759ce` 上只有 SPIR-V 流水调用它，LLVMGPU 流水不再调用（§2.2）。
- B 和 D 都没有提到 Triton ConSan。它是我们“检查编译器生成的同步”这一主张最接近的**动态**对手，相关工作应写进去。

### 4.4 可以借的（按收益排序）

1. **Triton ConSan 的可见性模型**：给 cluster、persistent 这类还没有 Lean 模型的环做差分测试；也可以检查 Triton 自动 WS 的输出，拓宽评测。
2. **cuda-oxide 的初始化 typestate**：让 `ring_idle` 只能由 `open` 产生，关掉 §21.6 的缺口。
3. **TileLang/z3 的写冲突检查**：补上证书的可信假设“各线程写入区间不冲突”。
4. **Mojo 的可插拔同步后端**：作为 `SyncOf` 实例化到 AMD 的设计先例。
5. **MLIR transform 句柄的 consumed 语义**：写论文时作为“线性使用”在编译器基础设施里已被接受的旁证。

## 5. 来源

- 网页：[XLA GPU architecture](https://openxla.org/xla/gpu_architecture)；[Stripe, arXiv 1903.06498](https://arxiv.org/pdf/1903.06498)；[PlaidML README](https://github.com/plaidml/plaidml)。
- 论文（只引书目，未读原文）：TensorIR（ASPLOS'23）、MetaSchedule（NeurIPS'22）、Relax（ASPLOS'25）、TileLang（ICLR'26）、PyTorch 2 / Inductor（ASPLOS'24）。
- 本地代码：见各节行号。
