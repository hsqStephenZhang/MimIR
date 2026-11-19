# B. 图编译器与 kernel DSL 编译器：谁决定什么，什么被检查（2026-11-19）

本文是调研维度 B。
沿两条轴比较各系统：**图编译器**从模型出发，自动选择融合和调度；**kernel DSL** 由专家手写 kernel。
关注五件事：融合、tiling、流水、warp 特化（WS）、同步；再看 fused attention 从哪里来，以及对同步给出什么保证。

## 0. 口径

- 证据优先用本地克隆，写成 `仓库/路径:行号`；行号对应下表的提交。
- 没有本地源码、只凭记忆的内容标“未核实”。宁可少写，不写猜测。
- 已有文档讲过的内容只引用不复述：
  - Async Graphene：`linear-types-design.zh-CN.md` §13；
  - 定位：同文 §15；
  - 同步验证与流水的文献：`gpu-tiling-literature.zh-CN.md` §4、§5、§8；
  - IREE attention pipeline 的源码对照：`attention-schedule-experiment.zh-CN.md`“IREE 源码依据”和步骤 5。

| 本地克隆（`~/workspaces/`） | 提交 | 日期 |
|---|---|---|
| `triton` | `d315663` | 2026-05-13 |
| `tilelang` | `8a68bbc5` | 2026-06-24 |
| `ThunderKittens` | `67845f5f` | 2026-09-12 |
| `cutlass`（CuTe DSL） | `b2f249ac` | 2026-11-06 |
| `cutile-python` | `d51eab0` | 2026-07-16 |
| `modular`（Mojo / MAX kernels） | `983d67b` | 2026-11-16 |
| `tvm` | `a4a2bf5` | 2026-06-10 |
| `iree` | `61759ce` | 2026-11-09 |
| `oneDNN` | `80b7f3d` | 2026-10-02 |
| `tritonbench` | `ad57d52` | 2026-11-12 |
| TorchInductor | `mimir-frontend/.venv` 里的 PyTorch `2.12.0+cu130` | — |

- **没有本地源码**：XLA、PlaidML、Halide、Graphene（ASPLOS'23）。
- **`pmlc-docker` 不是 PlaidML。** 它是 MimIR 的 KernelBench 复现镜像，对照对象是 torch-mlir（`pmlc-docker/README.md:1-30`）。
- 本机没有 `pdftotext`/`pdftoppm`，读不了 `cgo27-paper248.pdf` 和 `async_graphene.pdf`。论文的具体主张**未核实**；Async Graphene 一律引用 §13 的已有摘要。

---

## 1. 图编译器

### 1.1 总表

| 系统 | 用户写什么 | 编译器决定 | 怎样得到 fused attention | 同步的保证 |
|---|---|---|---|---|
| TorchInductor | PyTorch 程序 | 融合（scheduler）、Triton 代码与 autotune | 匹配 → 库 kernel；或 FlexAttention 模板 | 交给 Triton 编译器或 CuTe DSL 模板 |
| TVM Relax + MetaSchedule | 模型（Relax） | 按算子类别融合；TIR 调度靠搜索；插入同步 | BYOC 模式 → CUTLASS/FlashInfer | `ThreadSync` 是分析式插入，没有证明 |
| IREE | 模型（MLIR） | dispatch 划分、tiling、分布、流水 | raise 成 online attention，tile 后再分解 | 有值语义的 barrier op；没有证明 |
| oneDNN Graph | 图 API | 匹配固定融合模式 | `sdp` 模式 → 手写 SDPA kernel | 不适用（库 kernel） |
| XLA | HLO（JAX/TF/PyTorch-XLA） | 融合、GSPMD 分片 | 未核实 | 未核实 |
| PlaidML | Tile 语言 | 未核实 | 未核实 | 未核实 |

### 1.2 TorchInductor

- **融合**：`Scheduler`（`torch/_inductor/scheduler.py:3070`）在 `fuse_nodes`（`:4035`）中按 `can_fuse`（`:5802`）合并节点，决定权完全在编译器。
- **attention 有两条路，都不是编译器自己推出来的：**
  1. **模式 → 库。** `fx_passes/fuse_attention.py` 有 28 个手写的 `_sfdp_pattern_*`，把展开的 softmax(QKᵀ)V 替换成 `aten._scaled_dot_product_*`。这些 op 在 `lowering.py:3428-3460` 是 `make_fallback`，最终调用 FlashAttention、mem-efficient 或 cuDNN 的库 kernel。
  2. **模板。** FlexAttention 在 `kernel/flex/flex_attention.py:109` 注册 lowering。用户写 `score_mod`/`mask_mod`，代入固定的 Triton 模板（`kernel/flex/templates/flex_attention.py.jinja`）。另有 CuTe DSL 版：`flex_flash_attention.py:77-84` 用 `flash_attn.cute` 的模板。
     - tile 和 `num_stages`/`num_warps` 来自配置表（`flex_attention.py:407-408`）。
     - 在线 softmax、流水、WS 都写死在模板里。
- **同步**：Inductor 生成的是 Triton 或 CuTe DSL 代码，同步由下层负责，见 §2。

### 1.3 TVM Relax + MetaSchedule

- **融合**按算子类别：`kElemWise`、`kInjective`、`kCommReduce`、`kOutEWiseFusable`（`include/tvm/relax/op_attr_types.h:36-48`）；`src/relax/transform/fuse_ops.cc` 据此分组，`fuse_tir.cc` 把每组合并成一个 TIR 函数。
  - `mimir-frontend/operators_summary.md` 用的就是同一套类别给 ATen 算子分类。它只是一份文档，代码里没有引用。
- **调度**：在 `src/s_tir/meta_schedule/`（cost model、database、mutator 等）中搜索；流水和异步拷贝由 `src/s_tir/transform/inject_software_pipeline.cc`、`inject_ptx_async_copy.cc` 完成。
- **attention**：`python/tvm/relax/backend/patterns.py:219` 的 `make_attention_pattern` 匹配 `relax.nn.attention*`，交给 `backend/cuda/cutlass.py` 或 `flashinfer.py`（BYOC）。也就是说，fused attention 同样来自外部库。
- **同步**：`src/s_tir/transform/thread_storage_sync.cc` 按共享内存的读写冲突插入 barrier，是分析加插入，没有证明。

### 1.4 IREE

- **dispatch 划分**：`DispatchCreation/FormDispatchRegions.cpp:1317-1325`，决定哪些 op 进同一个 kernel。
- **attention**：沿“保留复合语义 → tile → tile 内分解”的路线，源码见 `attention-schedule-experiment.zh-CN.md`。
  - `ConvertAttentionToOnlineAttentionPass`（`Dialect/LinalgExt/Transforms/Passes.td:126`）；
  - GPU 配置：`Codegen/LLVMGPU/KernelConfig.cpp:754` 对 `OnlineAttentionOp` 分别给出 QK 和 PV 的配置（`:1059-1061`）。
  - **它与 MimIR 的路线最接近**：编译器自己产生在线算法，不调用库 kernel。
- **同步**：
  - `iree_gpu.value_barrier`（`Codegen/Dialect/GPU/IR/IREEGPUOps.td:120-140`）是 `Pure` 的 op，作用于值语义的 tensor 或 vector：输入什么值就返回什么值，只表示“写入对所有 worker 可见”。
  - **这与 MimIR 的哈希共享值图对应得很好**：把 barrier 做成值上的恒等变换。但 IREE 只提供这样的表示，不检查 barrier 放得对不对。
  - 流水：`Codegen/Common/GPU/GPUPipelining.cpp`；在 `61759ce` 上只有 SPIR-V 路径调用它，LLVMGPU 路径不再调用（见 A §2.2）。

### 1.5 oneDNN Graph

- `src/graph/backend/dnnl/patterns/sdp.cpp:116-905` 注册多种 SDPA 融合模式（float、GQA、MQA、JAX 风格、反向等），匹配后交给 `src/graph/backend/dnnl/kernels/sdp_*.cpp`。Intel GPU 上最终用手写的 `src/gpu/intel/sdpa/micro.cl`。
- 融合是“模式表 + 专家 kernel”，与 Inductor 的第一条路相同。

### 1.6 XLA、PlaidML（未核实）

- XLA：HLO 融合 pass、GSPMD 分片传播；GPU 上 GEMM 融合由 Triton 发射器生成，attention 可改写为 cuDNN 的 fused MHA。以上均**未核实**，本地没有源码。
- PlaidML：Tile 语言写收缩，编译器分块。**未核实**；本地 `pmlc-docker` 不是它。

### 1.7 attention 在图编译器里的三种来源

| 来源 | 系统 | 在线 softmax 由谁写 |
|---|---|---|
| 模式 → 库 kernel | Inductor `sfdp`、oneDNN `sdp`、TVM BYOC、XLA+cuDNN（未核实） | 库作者手写 |
| 模板 + 用户函数 | FlexAttention（Triton / CuTe DSL 模板） | 模板作者手写 |
| raise 成复合 op，编译器分解 | IREE `OnlineAttentionOp`；**MimIR 的 `sdpa_online`** | 编译器（规则或 pass） |

- 前两类等于把“attention 怎么做”交给专家；编译器只负责认出它。
- 只有第三类要求编译器自己推出流水和同步，也只有这一类需要“检查编译器生成的同步”。这正是 MimIR 的类型检查有意义的地方（§15）。

---

## 2. kernel DSL

### 2.1 总表

“用户”指写 kernel 的专家。✔ 表示由用户写，✗ 表示由编译器决定，— 表示该 DSL 不涉及。

| DSL | 用户写的层级 | 融合 | tile 大小 | 流水 | WS | 同步 / barrier | 检查 |
|---|---|---|---|---|---|---|---|
| Triton | CTA 级 tile 程序 | ✔ | ✔（autotune） | ✗（给 `num_stages`） | ✗（循环上一个开关） | ✗（Membar 自动插入） | 无 |
| Gluon | CTA，显式布局、smem、mbarrier | ✔ | ✔ | ✔ | ✔ | ✔ | 无 |
| TileLang | CTA 级 tile op + `T.Pipelined` | ✔ | ✔ | ✗（给 `num_stages`） | ✗（自动 WS pass） | ✗（自动插入，也可手写） | 无 |
| ThunderKittens | CUDA C++ + tile 类型 | ✔ | ✔ | ✔ | ✔ | ✔（semaphore、相位手算） | tile 形状的 C++ 静态检查（未核实细节） |
| CuTe DSL | CuTe 布局 + `Pipeline*` 类 | ✔ | ✔ | ✔ | ✔ | ✔（经 pipeline 类） | 无（`PipelineState` 是运行时值） |
| cuTile | 块级 tile，看不到线程 | ✔ | ✔ | ✗（`latency` 提示） | ✗ | 块内不允许写同步 | 块内同步由语言排除 |
| Mojo（MAX kernels） | 显式 mbarrier、相位 | ✔ | ✔ | ✔ | ✔ | ✔ | 线性阶段句柄：必须 `release()`，由编译器强制；槽号与相位仍是运行时值 |
| Async Graphene | 时间修饰的张量类型 | ✔ | ✔ | 半自动（`delay`） | ✔ | 部分自动（aref） | future 的“已等待”；没有定理（§13） |
| Halide | 算法 + 调度 | 调度决定 | 调度决定 | 未核实 | 未核实 | 编译器插入（未核实） | 调度保持语义（构造保证，未核实细节） |

### 2.2 Triton

- **用户写**：`@triton.jit` 的 CTA 级程序，`BLOCK_M`/`BLOCK_N` 是 `constexpr`。`num_stages`、`num_warps` 由 `triton.Config` 搜索（`python/tutorials/06-fused-attention.py:136-145`）。
- **attention**：手写在线 softmax（同文件 `_attn_fwd`，`:179`），融合是用户的事。
- **编译器决定：**
  - 流水：`lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp:171`。
  - 同步：`lib/Analysis/Membar.cpp:241`（`insertBarrier`）、`:267`（`update`），按共享内存的缓冲区区间做冲突分析后插入 barrier。
  - 自动 WS：用户只在循环上写 `tl.range(..., warp_specialize=...)`（tutorial `:69`）。`AutomaticWarpSpecialization.cpp:95-112` 依次运行 partition 调度、`NVWSInsertAref`、`NVWSLowerAref({numStages})`、`PartitionLoops`。aref 的概念与 Async Graphene、Tawa 相同；Triton 的 NVWS 与 Tawa 论文之间的关系**未核实**。
- **检查**：WS 每一步之后都跑 `VerifyWarpSpecializationPartitions`（同文件 `:32-43`），但 `verifyPartitionedLoop`（`Partition.cpp:389`）只检查 partition 属性格式对不对：有没有、是否重复、是否有序（`Partition.cpp:23-122` 的 `emitOpError`）。**不检查同步语义。**
- **对 MimIR**：Triton 是“编译器自动生成同步、无检查”的典型，正是 §15 所说编译器轴上的对照对象。

### 2.3 Gluon

- 与 Triton 同一套编译栈，但把布局、内存分配、数据搬运和异步都交给用户（`python/tutorials/gluon/01-intro.py:13-30`）。
- WS 要手写：`gl.warp_specialize([...])`（`gluon/08-warp-specialization.py:278`）；每个 partition 一个函数，mbarrier 的 init、wait、arrive 和相位都手写（`:145-230`、`:245-255`）。
- **proxy fence 也要用户自己记得**：tutorial 正文明确写着，跨 partition 时 smem 写与 TMA 读之间需要 `fence_async_shared`（`:96-120`）。这只是一条文字约定。这一类错误正是我们变异评测里的 `no_proxy_fence`（`linear-types-design` §22）。

### 2.4 TileLang

- **用户写**：`T.Kernel`、`T.alloc_shared`、`T.copy`、`T.gemm`、`T.Pipelined(loop, num_stages=...)`。FA 示例见 `examples/flash_attention/example_mha_fwd_bhsd.py:40-100`，在线 softmax 由用户手写。
- **编译器决定**（`tilelang/cuda/pipeline.py`）：
  - 自动 WS：`ProducerConsumerWarpSpecialized`（`:94-95`，实现在 `src/cuda/transform/producer_consumer_ws.cc`）；
  - 流水：`PipelinePlanning` + `InjectSoftwarePipeline`（`:108-109`）；
  - 布局推断（`:113`）；
  - 同步：`InjectFenceProxy` 与 `ThreadSync("shared")`（`:229-232`；`src/transform/thread_storage_sync.cc:1897`）。
- **检查**：有 `VerifyParallelLoop`（`:80`）等合法性检查。同步本身没有检查器，靠插入算法正确。

### 2.5 ThunderKittens

- CUDA C++ 加 tile 类型（`st_bf`、`rt_fl`）。H100 attention 见 `kernels/attention/mha_h100/mha_h100.cu`：
  - semaphore 手动初始化（`:77-83`）；
  - 生产者 `warpgroup::decrease_registers<32>`（`:106`），消费者 `increase_registers<160>`（`:128`）；
  - **槽号和相位手算**：`wait(compute_done[kv_idx % stages], (kv_idx / stages) % 2)`（`:123`）、`wait(k_smem_arrived[kv_idx % stages], (kv_idx / stages) % 2)`（`:151`）。
- 我们的 `Held (rid, depth) (n % d, …)` 和代的索引要检查的，正是这两行里的算术（§5、§18）。

### 2.6 CuTe DSL / CUTLASS Python

- `cutlass/pipeline/sm90.py`：`PipelineAsync`（`:45`）、`PipelineCpAsync`（`:394`）、`PipelineTmaAsync`（`:469`），接口是 `producer_acquire`、`producer_commit`、`consumer_wait`、`consumer_release`（`:242-310`）。
- `PipelineState` 在运行时持有 `(index, phase)`（`pipeline/helpers.py` 约 `:833-855`），由用户传递和推进。
- Hopper FMHA 示例：`examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py`，手写的 warp 角色分支（`:723`）和 `setmaxregister_*`（`:730`、`:853`）。
- **保证**：typestate 只是 API 约定，不进类型；`gpu-tiling-literature` §5 已有同样结论。

### 2.7 cuTile（`cutile-python`）

- 块级 tile 程序，看不到线程：“Explicit synchronization or communication within a block is not permitted”（`docs/source/execution.rst:23-27`）。块之间用带 memory order 和 scope 的 atomic（`docs/source/memory_model.rst:9-22`）。
- FMHA 示例 `samples/AttentionFMHA.py`：`@ct.kernel(occupancy=2)`（`:26`），`ct.load(..., latency=2)`（`:89`）只是延迟提示；tile 形状和 `num_ctas` 由用户在配置表里选（`:240-246`）。
- **保证的方式是“不让写”**：块内同步完全交给编译器（Tile IR），所以用户不会写错；编译器内部是否有检查**未核实**。

### 2.8 Mojo（MAX kernels）

- `max/kernels/src/nn/attention/gpu/nvidia/sm90/mha.mojo:1647-1650`：`produced_mbar_kv[read_idx].wait(read_phase)`；`sm90/attention.mojo:624-629`：`consumed_mbar_kv[write_idx].wait(write_phase)`，相位从 `state.phase()` 取。
- 相位与槽是运行时值，这一点与 CuTe DSL 相同。但结构化 kernel 的流水阶段句柄是**线性类型**：`ProducerStage`/`ConsumerStage` 声明为 `not Deinitable`，必须调用 `release()` 推进阶段，编译器强制（`max/kernels/src/structured_kernels/pipeline.mojo:495-540`、`636`）。它不带槽号或代的类型索引，`_index` 是 `UInt32`（由汇总者核实）。
- MAX 的图编译器怎样融合，本地开源部分里找不到，**未核实**。

### 2.9 Graphene / Async Graphene

- Graphene：把线程、warp tile 和 fragment 映射作为 IR 中的一等对象（`gpu-tiling-literature` §6）。没有本地源码。
- Async Graphene：见 `linear-types-design` §13。要点：
  - future 只区分“已等待”和“未等待”；
  - aref 的 full/empty 是运行时状态；
  - 延迟释放靠手写；
  - 有构成规则，没有使用规则，也没有定理。
- 它是 kernel 语言，但流水由 `delay` 加算法 2 生成，所以在本文的轴上是**半自动**。

### 2.10 Halide（未核实细节）

- 算法与调度分离：调度由用户写，或由自动调度器搜索。调度只改变计算顺序和存储位置，不改变语义。GPU 上的 barrier 由编译器插入。以上为一般认知，本地没有源码，**未核实**。
- MimIR 的 `law.of_blocks` 加层级链（§3.3）在形态上最像 Halide 的“算法 + 调度”。

---

## 3. MimIR 自己的路径

### 3.1 `mimir-frontend` 导入什么

| 项 | 事实 | 依据 |
|---|---|---|
| 入口 | `torch.compile(model, backend="mimir")`，Dynamo 交来 FX 图 | `src/mimir_frontend/backend.py:1-10`、`:679-681` |
| 算子 | ATen 翻译成 `%torch.*` axiom（例如 `torch.linalg.mm`）和 `%tensor.*` | `operators.py:2346-2359`；`MimIR-fork/extra/torch/torch.mim:2627`（`mm`）、`:3691`（`softmax`）（本树没有 torch 插件） |
| 分解 | 默认只分解一小组 op；`scaled_dot_product_attention` 和 MHA 是**触发**分解的 op，`native_layer_norm` 保留 | `decomposition.py:30-45` |
| 优化 | `world.optimize()`（`opt` 插件默认流水线），再 `mim.JIT` | `backend.py:410-440` |
| 目标 | **CPU**：LLVM IR JIT 成共享库；float32 输出，float32/int64 输入，静态形状，只做推理 | `backend.py:55-60`；`src/` 里没有 CUDA/NVPTX 字样 |

- **attention 的路径**：PyTorch 把 SDPA 分解成 matmul、softmax、matmul，MimIR 再用规则把它 raise 回 `sdpa`（§3.2）。与 Inductor 的方向相同，即“展开图 → 认出 attention”，但认出之后不调用库，由编译器自己做在线算法。
- **结论：PyTorch 模型到 GPU kernel 的端到端路径目前没有接通。** 前端只出 CPU 代码；而 MimIR 在加载 `gpu` 插件时会关掉 group 和 attention 规则：
  - `group_rules.cpp:474`、`:498`、`:522`、`:536`；
  - `group.cpp:330`；
  - `normalizers.cpp:231`；
  - 另见 `flash-attention-plan` §0。

### 3.2 图层：规则与 group（CPU 默认开启）

这部分已有文档详述，这里只汇总：

- **规则机制**：`compile.rule_app`（单根）和 `compile.graph_rule_app`（多根）。规则写在 Mim 里，由 C++ 的 `PatternProgram` 匹配，按 World 重建提交（`graph-pattern-matching-design`“当前实施状态”、P5）。
- **具体的图重写**：
  - `tensor.combine_products`：共享左操作数的 QKV 三个乘法合并（同文 P4）；
  - `fuse_tensor`：producer、read-through、epilogue 融合；
  - group → scoped rules → `fuse_views`：CPU 默认开启（`graph-pattern-tile-fuse-experiments` Task 19A.3）。
- **attention 全部写在 Mim 里**：
  - raise 规则：`tensor.mim:1987`（`attention_rule`）、`:2015`（masked）；
  - 在线规则：`:4031`（`attention_online_rule` → `sdpa_online`，`:4027`）；
  - 调度规则：`:4057`（`attention_schedule_rule`，tile 由 `attention_row_tile` 算出）；
  - C++ 只负责驱动：`group_rules.cpp:473`（`AttentionRaise::start`）、`:521`（`AttentionSchedule::start`）。
- **已知边界**：
  - 规则匹配成功，不等于调度完整兑现。跨 region 的 placement、寄存器驻留、目标 vector 合法性都不在规则的保证之内（`attention-schedule-experiment` 步骤 5）。
  - 模型里的 GEMM 没有得到 BLIS，layout 成本也缺失（`model-scheduling-diagnosis`）。

### 3.3 GPU 层：手写的 tile 级 Mim 与编译器的 lowering

| 层 | 谁写 | 例子 |
|---|---|---|
| 算法 + 调度 | 人在 Mim 里写：块定律 `law.of_blocks`，加层级链 `serial_group → group_warp → warp_red` | `bench/gpu_attn/attn_law.mim`；`tensor.mim:3852`（`of_blocks`）、`:3935`（`blocks_run`） |
| 块步骤由编译器推出 | `law.blocks_rolling` 从运行最大值和修正项推出块步骤 | `attn_roll.mim`；`tensor.mim:3918` |
| 面板、布局、双缓冲、环 | 编译器：TileFuse、`LowerToMem` | `attn_wg.mim`（编译出来的版本是手写 `flash_wg` 的 0.96–0.97 倍，`bench/gpu_attn/RESULTS.zh-CN.md`“编译路径”） |
| FA3 结构（WS + TMA + mbarrier + ping-pong） | 人写 Python 生成器，输出 Mim | `flash_fa3.py`；相对 cuDNN 0.85–1.00（同文 FA3 节） |
| 同步原语 | `gpu.mim` 的 axiom | `gpu.mim:431-461`（mbarrier、TMA、named barrier、`setmaxnreg`） |
| 同步检查 | 编译器：`gpu.check_sync`；协议类型、Lean 证书 | `gpu/phase/check_sync.cpp`；`MimIR-types` 的 `gpu.mim:868-933`（`Held`、`Ring`、`async_ring`、`mbar_ring`）；`linear-types-design` §16–§22 |

- **GPU 上的状态**：融合、tiling 和缓冲由编译器做；“用哪种调度”由人用 Mim 的调度项给出；WS 还停在手写生成器。这处在 Halide（调度由人给）和 TileLang（编译器插同步和流水）之间。
- **同步检查的覆盖面**（§22 的变异评测）：
  - 7 个原则上覆盖的真实环都通过了 Lean 证书检查；
  - 44 个改变了生成代码的（变异，kernel）组合全部被拒：类型层与 `check_sync` 8 个，有界模拟 1 个，Lean 35 个（口径见 README §2）；
  - 11 个真实环里，cluster、persistent 和叠放的 4 个还没有模型（§21.2）。

---

## 4. 定位表：谁决定，什么被检查

| 系统 | 融合 | tiling | 流水 | WS | 同步放置 | 对同步检查什么 |
|---|---|---|---|---|---|---|
| Inductor | 编译器 | 模板 + autotune | 下层 | 下层 | 下层 | 无（交给 Triton 或库） |
| TVM | 编译器（按类别） | 搜索 | 编译器 | 未核实 | 编译器（`ThreadSync`） | 无 |
| IREE | 编译器（dispatch） | 编译器 | 编译器 | 未核实 | 编译器（`value_barrier`） | 无 |
| oneDNN Graph | 模式表 | 库 | 库 | 库 | 库 | 无 |
| Triton | 用户 | 用户 + autotune | 编译器 | 编译器（开关） | 编译器（Membar） | 只检查 partition 属性格式 |
| TileLang | 用户 | 用户 | 编译器 | 编译器 | 编译器 | 无 |
| Gluon / CuTe DSL / TK / Mojo | 用户 | 用户 | 用户 | 用户 | 用户 | Gluon、CuTe DSL、TK 无（`PipelineState` 是运行时值）；Mojo 有线性阶段句柄，带 pipeline origin 与面板负载，但槽号和相位是运行时值 |
| cuTile | 用户 | 用户 | 编译器 | 编译器 | 编译器（块内禁止手写） | 由语言排除块内同步；编译器内部未核实 |
| Async Graphene | 用户 | 用户 | 半自动 | 用户 | 部分自动 | 未等待就使用会被拒；没有定理 |
| **MimIR（CPU，PyTorch 路径）** | 规则 + group（编译器） | 规则内的策略函数 | — | — | — | — |
| **MimIR（GPU，Mim 路径）** | 编译器（TileFuse） | 人写调度项 | 编译器（环） | 人写生成器（`flash_fa3.py`） | 编译器（`LowerToMem` 的协议） | 类型层：槽令牌、环身份、两相状态；Lean：7 个环的证书 |

---

## 5. MimIR 在哪里

- **目标定位**：图编译器，即“PyTorch 进，kernel 出”，并且在同一个 IR 里把同步协议带上类型（§15）。
- **实际位置**：
  - CPU 上是图编译器，attention 走 IREE 式的 raise 加在线分解；
  - GPU 上更像一个**带调度语言的 kernel 编译器**：人写 tile 级 Mim 和调度，编译器做 lowering 和同步；
  - 两者之间没有接通。
- **与最接近者的差别**：
  - **IREE**：attention 路线相同。差别在于 MimIR 的规则和在线算法都写在 Mim 里（`tensor.mim:4027-4079`），而 IREE 写在 C++ pass 里（`TileAttention.cpp`）。IREE 的 `value_barrier` 是值语义 barrier 的先例。
  - **Triton 自动 WS 与 TileLang**：都是“编译器自动插同步”；它们不检查，我们检查。
  - **Async Graphene**：类型层的先例，见 §13.4 的差异表。

## 6. 新颖性主张

### 6.1 可以辩护的

| 主张 | 依据 | 需要注意 |
|---|---|---|
| 检查**编译器自动生成**的 GPU 同步协议，类型层加证书，在擦除前进行 | Triton 的 WS verifier 只看属性格式；TileLang、TVM、IREE 只插入不检查；变异评测 51/51（§22） | 只覆盖 11 个环中的 7 个；cluster、persistent、叠放没有模型 |
| 同一套协议既是 lowering 的目标，也可以手写 | `gpu.mim` 的 `async_ring`/`mbar_ring` 被 `LowerToMem` 使用，也被 20 个手写协议测试使用（§21.6） | 手写路径的人体工程学还没评估 |
| attention 的 raise、在线化和调度全部用 IR 内的规则表达，没有专用的 C++ 生成器 | `tensor.mim:1987`、`:4031`、`:4057`；`attention-schedule-experiment` 步骤 5 | 只验证了 CPU 路径；GPU 上这些规则是关闭的 |
| 槽号和相位的算术由类型或证书检查，而 TK、CuTe DSL、Mojo 都靠手算（Mojo 已有线性阶段句柄，但不索引槽和代） | `mha_h100.cu:123/151`；`sm90.py` 的 `PipelineState` | “比 Async Graphene 强”只在 §13.4 列出的维度上成立 |

### 6.2 不能辩护的（或者现在还不能）

| 说法 | 为什么不行 |
|---|---|
| “从 PyTorch 模型自动生成 GPU kernel” | 前端只出 CPU 代码；加载 `gpu` 时 group 和 attention 规则关闭（§3.1） |
| “自动 warp 特化” | FA3 结构来自手写的 `flash_fa3.py`；Triton 和 TileLang 已经有自动 WS |
| “性能达到或超过 FlashAttention / cuDNN” | 0.85–1.00 倍 cuDNN（手写生成器），编译路径是手写 FA2 的 0.97 倍；Async Graphene 声称达到 FA3 的 98%（§13，论文数字未核实） |
| “首个在 IR 里表达 tile 调度的系统” | Halide、TVM、TileLang 都已经这样做 |
| “首个把异步同步放进类型” | Async Graphene 已有 future 和 aref；Kuiper 和 Hopper GEMM 的 Lean 验证（`linear-types-design` §14.2）已有形式化先例 |
| “barrier 作为值”是新的 | IREE 的 `value_barrier` 已有 |
| “图编译器里的 fused attention” | Inductor、oneDNN、TVM 早已有（经库实现）；IREE 已有编译器自己生成的版本 |

### 6.3 建议的写法

我们是在一个做 PE、哈希共享的高阶 IR 里，从高层张量程序**自动生成**异步 GPU 协议，并在擦除前用类型和可执行证书检查这些协议。这里的高层张量程序，现在指手写的 tile 级 Mim，将来指 PyTorch 导入的图。

- 对比轴选编译器轴：Triton 自动 WS、TileLang、IREE。
- 评测用编译器 bug 和变异。
- 性能只用来说明“检查不付代价”：在 71/73 个 benchmark 上，加上类型后设备 IR 逐字节不变（§21.6）。
- 在 PyTorch → GPU 接通之前，“compiler for PyTorch models”只能写成目标，不能写成结果。

## 补记（汇总时，依据 agy 核对）

- 本文原来的 `tensor.mim` 行号取自删除 law 解释器（`0d85d5206`）之前的旧树，已按当前树更新为 1987、2015、3852、3918、3935、4027、4031、4057。
- 自动生成 TMA、mbarrier 和 warp 特化的还有 Cypress（PLDI'25）、Twill（OSDI'26）、Tawa（CGO'26，即 Triton 的 aref），见 [README §6](README.zh-CN.md)。“kernel DSL 里只有 Triton 和 TileLang 自动生成 WS”需按此修正。
- §3.2 引用的 CPU attention 实验已完成步骤 1–5，结论是性能为负，见 E (f3)。
