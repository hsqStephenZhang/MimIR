# 从抽象机（Abstract Machine）视角审视 `pe-cpu-tiling`：ISA 设计、横向对比与扩展边界

本文从 **CESK 抽象机 + 部分求值（PE）** 的视角系统整理 `pe-cpu-tiling`（`src/mim/plug/tensor/tensor.mim`）的架构定位、现有指令集诊断与演进方案，给出与 TorchInductor、TVM `tir.Schedule` / DLight、MLIR `TilingInterface` / Transform Dialect 的带引用与实测横向对比（含 6 类调度不变式的实证探针），并调研超出稠密仿射 `map_reduce` 模型的算子边界及“微型抽象机家族”扩展策略。

---

## 1. `run_levels` 作为一台张量调度抽象机（CESK + PE）

按 Danvy 的经典对应关系，将高阶调度组合子（HOF）去函数化（defunctionalize）为一阶代数数据类型 `SchedTree` 与解释器 `run_levels_at`，本质上就得到了一台**领域抽象机（Domain-Specific Abstract Machine）**。
这与 `cgo27-paper248.pdf` 引用的 Truffle/Graal 思想（Würthinger et al., *One VM to Rule Them All*, Onward! 2013）及第一 Futamura 投射一脉相承：**定义一台极简抽象机，再由 PE 在编译期把指令分派、环境表与抽象机开销彻底擦除，自动得到零开销的目标循环 IR。**

### 1.1 抽象机四元组映射（对应 `tensor.mim:1957-2192`）

| 抽象机部件 | 在 `run_levels_at` 中的精确对应 | 职责 |
| :--- | :--- | :--- |
| **Control（指令树 `C`）** | `SchedTree (N, Rn)` | 7 条结构化指令：`Par / Seq / Peel / Pack / Stage / Switch / Leaf` |
| **Environment（环境寄存器 `E`）** | `axes: «Rn; (lo, size, cap, dyn)»` + `views: «N; LevelsView»` | 迭代窗口寄存器（当前聚焦子盒区间、静态上界与动态标记）+ 输入视图表（原 buffer / packed 面板 / 按需 `Prod#Fill`） |
| **Store（存储 `S`）** | `(tok: mem.M 0, dst: Buf)` + 作用域内临时 `panel`/`stage` buffer | 显式线程化的内存令牌、目标累加张量及局部临时存储 |
| **Continuation（控制流 `K`）** | `Body = State → [mem.M 0, D]`（及 `affine.For` / `Peel` / `Switch` 分支 continuation） | 后续待执行的计算模板 |

### 1.2 抽象机视角的双重价值
- **系统视角**：不再维护按算子家族分裂的代码模板或庞大的 C++ 循环变换 pass，全量 CPU 调度降级收束为一个仅 250 行 Mim 的统一抽象机解释器（含 `Aff`/`Pack`/`SchedTree`/策略共 590 行）。
- **形式化视角**：抽象机的状态转移规则天然构成操作语义，无需额外发明语义层即可在 Lean 4 中按结构归纳证明 `∀ t, WF(t) → run_machine(t) ≡ map_reduce`。

---

## 2. 现有 7 指令 AM 的 ISA 诊断与正交化扩充

### 2.1 当前 7 条指令的 3 处结构性耦合与冗余
1. **`Par` / `Seq` / `Peel` 实为同一条 `Split` 指令的 3 种模式，且 `Par` 名不副实**：三者在 `tensor.mim:2056-2104` 共用同一个 `split_serial(rt_tail, peel, d, ts, k)`，`Par` 实际发射的是串行 `affine.For` 而非多线程并行，只是“对多条轴连续静态切分”的语法糖。
2. **存储分配（Store 生命周期）与数据填充（Computation）耦合**：`Pack` 与 `Stage` 强制把 `buffer.alloc`、`fill`、`body` 与 `buffer.free` 绑定在同一个词法作用域，无法表达 Halide 的 `store_at != compute_at`（例如将 `panel_b` 的 `alloc` hoist 到外层循环之外复用）。
3. **输入与输出不对称，`Leaf` 承载过多特判**：输入侧有完整的 `views` 视图抽象，但输出侧 `dst` 缺乏对称抽象——带外部输入的 epilogue（如 bias）在 `lower_to_mem.cpp:688` 只能退回 `apply_post` 额外扫一遍内存；`rfactor`（沿归约轴拆部分累加器再合并）硬编码在 `step_leaf` 中，无法复用于外层 Split-K 归约。

### 2.2 AM ISA 设计的 2 条原则（对应 `cgo27-paper248.pdf` P2）
1. **按修改的抽象机状态正交划分指令**：修改 `axes`（控制流切分/并行/分支）、修改 `views`（数据重排/物化）、修改 `Store`（存储分配/归约拆分）、执行算术（`Leaf`）。
2. **One Instruction = One Lean Lemma**：凡可由现有指令组合表达的（如多轴 `Par(sizes)`、`sched_levels`），一律作为宏/派生组合子，不进入核心 ADT；每条进入核心 ADT 的指令都应对应 Lean 4 结构归纳中一条独立的局部等价引理。

### 2.3 建议的最小正交指令集（CPU 核心 + GPU 分层扩展）
重构后的 CPU 核心 AM 可收敛为 **7–8 条完全正交的指令**：
- **迭代窗口与控制流**：
  - `Split(d, ts, tail_mode)`：统一原 `Par`/`Seq`/`Peel`（`tail_mode ∈ {StaticPeel, DynamicLoop, DynamicPeel}`）。
  - `Parallel(d, ts, body)`：真正的多核并行切分（限制 `d < Ro`，输出子盒天然互不相交，无数据竞争）。
  - `Switch(d, c, small, big)`：按轴长阈值分派子树并收紧静态上界 `cap`。
- **存储与数据流**：
  - `Alloc(id, shape_expr, body)`：解耦存储分配与填充，按静态 `cap` 在外层分配复用槽位（原生支持 buffer hoisting）。
  - `Pack(id, d, w, slot, dep, body)`：向指定槽位做 BLIS 面板重排并切换 `views[slot]`。
  - `Stage(id, slot, body)`：按 `aff_box` 向指定槽位物化上游 `Prod#Fill`。
- **归约与执行**：
  - `ReduceSplit(d, factor, merge, body)`：统一表达 SIMD `rfactor` 与多核 Split-K 并行归约，使 `Leaf` 彻底瘦身。
  - `Leaf(vdim)`：纯叶子微内核（支持将外部 epilogue 输入纳入视图实现寄存器内融合）。
- **异步加速器扩展（独立分层，不污染同步 CPU AM）**：
  - `Pipe(Stages, issue, arrive, compute)` + `AllocRing`：仅面向 GPU 异步搬运与计算重叠（`cp.async` / Hopper TMA + `mbarrier` + `wgmma`），安全属性由类似 `proofs/ring/` 的异步协议校验器独立保证。

---

## 3. 横向对比：TorchInductor、TVM `tir.Schedule` / DLight 与 MLIR

在策略选参层（`cpu_tree` / `generic_tree`），我们与 DLight 规则库、Inductor 模板启发式同属规则驱动。
本质差异在于**调度表达、降级机制与可信计算基（TCB）**：

### 3.1 系统对比总表

| 维度 | **PyTorch TorchInductor** (ASPLOS '24) | **TVM TensorIR + DLight** (ASPLOS '23 / '25) | **MLIR Transform / Linalg** (CGO '25) | **MimIR `SchedTree` AM + PE**（本项目） |
| :--- | :--- | :--- | :--- | :--- |
| **调度表达形态** | 外部 Python 类 + Jinja2 字符串模板 | 外部 Python 规则脚本 + 命令式 TIR 原语序列 | IR 内脚本编排外部 C++ pass | IR 内一阶依赖类型 ADT（`SchedTree` + `Aff`） |
| **代码生成机制** | `template.render` 渲染 C++/Triton 源码文本 | C++ `ConcreteSchedule` 原地改写 TIR AST（`SBlockNode`） | C++ `TilingInterface` / `Linalg` pass 重写 IR | IR 内 250 行 `run_levels` 抽象机经 **PE** 擦除为 Loop IR |
| **典型源码锚点** | `torch/_inductor/codegen/{cpp_template.py, cpp_template_kernel.py, cpp_gemm_template.py, cpp_micro_gemm.py}` | `python/tvm/dlight/gpu/{matmul.py, gemv.py, reduction.py, general_reduction.py}` + `src/tir/schedule/concrete_schedule.cc` | `mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp` | `src/mim/plug/tensor/tensor.mim`（全套 590 LOC，核心解释器 250 LOC） |
| **类型与合法性检查** | 无静态 IR 类型检查，延迟至下游 C++/Triton 编译器 | 运行时执行 `sch.*` 原语时由 C++ 逐条校验 | 运行 transform 解释器时动态校验 | 构造期由依赖类型检查 + 可擦除 `refly.check` 静态校验 |
| **半仿射 `floordiv`/`mod`** | 模板内手写处理 | 依赖 TIR 区间分析与 predicate 插入 | `validateTilingSemiAffineMaps` 仅放行单变量 `d op C` 且强制 `tileSize % C == 0 \|\| C % tileSize == 0`，因无符号 map 无法做 per-tile shift | 当前 `aff_box` 保守拒绝；因 `Aff` 原生支持运行时符号 `vals`，原则上可表达 `aff_eval(at + x) - blo` 的通用 per-tile shift |
| **TCB 与可验证性** | 每个手写字符串模板都是新 TCB，极难形式化验证 | 依赖数万行 C++ 命令式 AST 变换原语构成 TCB | 依赖各 dialect 的 C++ `TilingInterface` 与重写 pass | **一次 7–8 节点结构归纳证毕抽象机**，所有上层规则/搜索移出 TCB |

### 3.2 与 `tvm.s_tir.schedule.Schedule` 的深度优缺点权衡

`tvm.s_tir.schedule.Schedule` 是**面向既有循环 AST 的命令式手术刀**，而 `SchedTree` AM 是**面向纯数学规约 `map_reduce` 的声明式虚拟机 + PE 一次性合成器**。

- **我们的核心优势**：
  1. **TCB 与验证成本低一个量级**：从不修改已有循环 AST，无需在已被前序变换扰乱的 AST 上重做依赖分析、`compact_dataflow` 或 `decompose_reduction`；每条指令只刻画局部状态转移，适合 Lean 4 结构归纳。
  2. **静态类型安全与纯不可变树变换**：没有易失效的运行时句柄（`BlockRV`/`LoopRV`）；`sched_unpack`、`sched_retile`、`sched_at_leaves` 均为 5 行纯函数 fold。
  3. **原生多版本调度分支与零 predicate 快路径**：`Switch`（按维度切换整棵子树）与 `Peel`（让整块主循环恢复成 `dyn = ff` 的静态满速代码）是原生一等指令。
  4. **极低编译器代码量**：核心抽象机仅 250 行 Mim，C++ 侧无循环重写 pass。
- **我们的真实缺点（`tir.Schedule` 占优之处）**：
  1. **通用性受限于 `map_reduce` + `Aff` 模型**：`tir.Schedule` 可直接改写任意手写 TIR `PrimFunc`（不规则循环、间接访存、数据相关控制流、自定义微内核）。
  2. **DAG 多消费者共享融合较弱**：`SchedTree` 是以单一消费者为根的树，遇到两个消费者共享同一个局部物化 producer 的钻石型 DAG 时，要么重算要么退回外层独立物化，不如 `tir.Schedule` 在完整 `PrimFunc` 中通过 `compute_at` / `cache_write` 灵活移动 `SBlock`。
  3. **硬件内联替换（`sch.tensorize`）与底层精细旋钮**：TensorIR 可通过 `SBlock` + `sch.tensorize` 原位替换任意硬件指令块（AMX/WMMA/WGMMA），并具备 `storage_align`、`compute_inline` 等细粒度控制；我们的 `Leaf` 目前固定绑定 `btensor` 向量化微内核。
  4. **单步交互式调试体验**：`tir.Schedule` 每执行一条原语即可打印中间 TIR，`SchedTree` 经 PE 一次性展开后需要专门工具辅助查看中间形态。

### 3.3 不变式校验机制实证对照（MLIR vs TVM vs AM + Ghost State）

本节在真实本机环境（`mlir-opt` LLVM `23.1.5` 与本地 `llvm-project/mlir` 源码、`tvm 0.27.0.post1` 与本地 `tvm/src/s_tir` 源码、`mim -X refly:strict=1`）上对 6 类跨原语/路径敏感调度不变式构造最小负例并直接运行，对比三者在检查时机、报错行为与静默盲区上的差异。

#### 实测总对照表

| Invariant 类别 | MLIR（Linalg / Transform / GPU / NVGPU） | TVM（`s_tir.Schedule` + Lowering Passes） | MimIR（`SchedTree` AM + Ghost State） |
| :--- | :--- | :--- | :--- |
| **1. Layout–Split 一致性**<br>（Pack 面板宽 $u$ 与内层 Split $ts$ 对齐） | LLVM `23.1.5` 对 `(d0 floordiv 4, d0 mod 4)` 切 `ts=6` **静默错编（`rc=0`，内层越界+丢元素）**；上游 `main` 才在 `TilingInterfaceImpl.cpp:87-242` 补了 `validateTilingSemiAffineMaps`。<br>`linalg.unpack(6)` 融进 `tile=4` 不报错，退化为 8 个 `affine_map` + 动态 `tensor<?x6xf32>` 临时切片。 | `transform_layout` 对不可整除 shape（如 22 按 6 切）当场报 `ScheduleError`（`layout_transformation.cc:1088-1098`）。<br>但随后 `split(i, [None, 4])` **静默通过**，把 `vi // 6, vi % 6` 原样留在内层循环。 | AM ghost 域 `unit` 沿树传播，内层 `Split` 当场查 `ts % u == 0`（`tensor.mim:2179`），失败直接报 `SchedTree: a Split must cut a packed axis into whole panels`。 |
| **2. 路径敏感动态上界**<br>（`size ≤ cap` 与 `dyn` 传播，静态 scratch 安全） | 动态 `tensor<?x?xf32>` tiling 生成 `affine.min` + 动态切片，`scf.if` 分支内不向类型传播静态 `cap`，bufferization 易退化为动态分配或整块 pad。 | `split` + `cache_read` 在 schedule 期保留符号 shape `(M, K)` 与 `T.where` 掩码，依赖后置 `CompactBufferAllocation` 倒推静态界。 | AM 状态携带 `(size, cap, dyn)` 三元组，`Split`/`Switch` 沿路径收紧 `cap`，无界动态轴进 `Pack`/`Leaf` 在 PE 阶段直接拒斥（`tensor.mim:2147-2240`）。 |
| **3. Staging Typestate & 规约完备性**<br>（禁止在部分和上挂非线性 epilogue） | `fuse_into_containing_op` 拒绝把 `exp` 塞进 `matmul` K 循环；生产者输出 map 非 projected permutation 时发 `error` 但 **`rc=0` 仍继续输出 IR**（`TilingInterfaceImpl.cpp:386-388`）。 | `decompose_reduction` 位置错当场报 `LoopHeightError`（`reduction.cc:140-150`）。<br>**实测静默数值错编**：`reverse_compute_at(blk_d, ko)` 允许把归约 epilogue 塞进 split-K 循环且零报错，实际运行结果错误。 | AM 状态 `prod`（`Fill → Plain`）强制先 `Stage` 后读（`tensor.mim:2080-2237`）；`whole` 仅在归约轴闭合时为真，从构造上杜绝在部分和上触发 epilogue。 |
| **4. 片上存储峰值预算 & 空间类型** | 空间类型强查（`nvgpu.ldmatrix` 读 global 立即报错，`NVGPUDialect.cpp:330-335`）。<br>**容量 0 检查**：4 MiB `workgroup` SMEM 经 `--convert-gpu-to-nvvm` **`rc=0` 静默产出** `!llvm.array<1048576 x f32>`。 | Schedule 期 0 检查；显式调用 `VerifyGPUCode`（`verify_gpu_code.cc:66-105`）才报错，且：① 是**扁平 AST 求和**（两段互斥 32 KiB、峰值 32 KiB ≤ 48 KiB 却**误报超限**）；② 动态 shape 计 `const_size = 0` **静默漏报**；③ `launch_thread` 外 alloc 被 `Reset_()` 清零**漏报**。 | AM 词法作用域递归栈天然精确统计 `live_peak ≤ SMEM_CAP`（配合 `cap` 对动态轴也 sound）；GPU 端已用 `refly.check` 拦截线程数与对齐（`gpu.mim:506-1686`）。 |
| **5. 异步流水线协议**<br>（`cp.async` / TMA / `mbarrier` / `wgmma` 活性与 WAR） | Lowering 级无协议状态机：`device_async_wait {numGroups = 99}` 竞态与无 `arrive` 的 `mbarrier.try_wait.parity`（必死锁）经 `--convert-nvgpu-to-nvvm` 均 **`rc=0` 静默放行**；`scf::pipelineForLoop` 只查 stage 序（`LoopPipelining.cpp:154-190`）。 | `sch.annotate` 期 0 检查；直到 `InjectSoftwarePipeline` 才用 `ICHECK` 报 `ValueError`（`inject_software_pipeline.cc:1078`）检查 stage/order 单调，但不验证底层异步 barrier parity / WAR / 活性。 | AM Ghost 携带环缓冲槽位状态机（`Free → InFlight(tok) → Ready`）与 `mbarrier` phase/expected-bytes（对应 `gpu.ring_fold` 与独立协议校验器），展开期即拦截超发、漏等与死锁。 |
| **6. 并行无竞态 & 规约幺半群合法性** | `tile_using_forall` 切 `matmul` K 维**仅发 `warning: tiling is not thread safe at axis #2` 且 `rc=0` 照常生成竞态 `scf.forall`**（`TileUsingInterface.cpp:183-193`）；`scf.parallel` 写共享 `memref` 静默放行；`tile_reduction_using_for` 靠 `getNeutralElement` 白名单拒斥 `subf`。 | `parallel` 标到 `T.axis.reduce` 报 `ScheduleError`（`for_kind.cc:107-110`）；`C[0] += A[vi]` 被 `CompactDataFlow` 条件 #3（`reads ∩ writes == ∅`）拦下；**但纯 WAW 竞态 `C[0] = A[vi]` 经 `sch.parallel(i)` 静默通过**；`rfactor` 查 `ReducerRegistry` 白名单拒斥 `sub`。 | `Par` 与 `Reduce` 分离；无 `merge` 的 combiner 禁止切分非首归约轴（`tensor.mim:2059`），`Reduce` 强制显式提供 `merge` 并限定归约轴（`tensor.mim:2143-2223`）。 |

#### 典型实测输出与暴露出的静默缺陷

1. **MLIR `23.1.5` 半仿射 tiling 静默错编**（`tile_using_for tile_sizes [6]` 作用于 `affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>`）：
   - `mlir-opt --transform-interpreter` 返回 `rc=0`，产出 `tensor.extract_slice %arg0[%1, %2] [2, 2] [1, 1]` 喂给 `tensor<6xf32>` 的内层 `linalg.generic`，导致 `d0 mod 4 ∈ {2, 3}` 直接越界并漏读跨行元素。
2. **MLIR `23.1.5` 并行化归约轴只给警告不阻断**（`tile_using_forall tile_sizes [0, 0, 16]` 切 `linalg.matmul` K 维）：
   - 输出 `warning: tiling is not thread safe at axis #2` 但返回 `rc=0`，继续发射 4 个并发线程向同一 `[0, 0][64, 64]` 执行 `tensor.parallel_insert_slice` 的 `scf.forall`。
3. **MLIR `nvgpu` / `gpu` 缺乏容量与异步协议状态机**：
   - 4 MiB `workgroup` SMEM 经 `--convert-gpu-to-nvvm` 返回 `rc=0` 并生成 `!llvm.array<1048576 x f32>`；`nvgpu.device_async_wait {numGroups = 99}` 与无 `arrive` 的 `nvgpu.mbarrier.try_wait.parity` 经 `--convert-nvgpu-to-nvvm` 均以 `rc=0` 直接落成对应 `nvvm` 指令。
4. **TVM `0.27.0.post1` 归约消费者融进 split-K 导致真实静默数值错编**：
   - 对 `C[i,j] = Σ_k A[i,k]*B[k,j]` 与 `D[i] = Σ_j exp(C[i,j])`（$16 \times 16$ 全 1 输入），执行 `split(k, [None, 4])` + `reverse_compute_at(blk_d, ko)` 零报错通过。
   - 经 `tvm.compile(..., target="llvm")` 实跑：`D_orig[0] = 1.4217778e+08`（正确值 $16 e^{16}$），而 `D_sched[0] = 1.4466464e+08`（在每个部分和上重复累加，多出 $15(e^4 + e^8 + e^{12})$）。
5. **TVM `0.27.0.post1` `VerifyGPUCode` 扁平求和误报 + 动态分配漏报 + `parallel` WAW 漏报**：
   - 两个互斥 `if` 分支内各分配 32 KiB `shared`（真实峰值 32 KiB ≤ 48 KiB），`verify_gpu_code` 误判 `False`；动态 `alloc_buffer((n,), scope="shared")` 因 `const_size = 0` 漏判 `True`；`C[0] = A[vi]` 经 `sch.parallel(i)` 零报错生成并发写竞态。
6. **MimIR `SchedTree` AM 编译期当场拦截**：
   - 运行 `mim -X refly:strict=1 lit/tensor/pe_cpu_sched_check.mim` 直接在展开点报 `error: SchedTree: a Split must cut a packed axis into whole panels`，且同文件一并覆盖了 `Fixed Split` 动态轴、`Leaf`/`Pack` 缺前置 `Stage`、无界动态 `Pack`/edge、缺 `merge` 的 `Reduce` 等全部负例。

#### 架构启示

MLIR 与 TVM 的上述盲区源自同一结构性原因：命令式调度原语与独立重写 pass 之间没有一份随作用域同步演化的统一抽象机状态，因此跨原语（Layout–Split、Staging–Reduction、异步 Acquire–Release）与路径敏感（动态 `cap`、SMEM 峰值）不变式只能退回到事后局部启发式倒推，或干脆推迟到后端/运行时才暴露。
相反，AM + Ghost State 沿着 `SchedTree` 的词法与动态执行结构正向推进一份统一的抽象机器状态 `(size, cap, dyn, unit, prod, fresh/whole, mem_live, async_ring)`，使原本分散在多个编译阶段甚至无人检查的 6 类不变式收敛到同一检查点。

> 更完整的挑刺论文全景（MLIR-TV / Ratte / DESIL / Tzer / NNSmith / WhiteFox / AlignGuard / Exo / TensorRight）、PyTorch / MLIR / TVM 之间的一手互相批评矩阵，以及扩展后的 **13 类遗漏不变式（I1–I13）** 与三框架实测复现，见专项调研文档 [compiler-invariants-critique-survey.zh-CN.md](compiler-invariants-critique-survey.zh-CN.md)。

---

## 4. 超出稠密仿射 `map_reduce` 模型的算子调研（附出处）

当前 `map_reduce` + `Aff`（含 `Stage` 要求 `map_out == id`）覆盖的是**稠密矩形域 + 仿射读映射 + 无写冲突独立输出轴 + 结合归约**。
超出该模型的真实 ML / HPC 算子按打破的数学前提可分为 6 类：

1. **稀疏张量与压缩格式遍历**（SpMM、SDDMM、SpMV，打破稠密矩形域与仿射下标）
   - Kjolstad et al., *The Tensor Algebra Compiler* (TACO), **OOPSLA 2017**, DOI `10.1145/3133901`.
   - Bik et al., *Compiler Support for Sparse Tensor Computations in MLIR*, **ACM TACO 2022**, DOI `10.1145/3544559`.
   - Ye et al., *SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning*, **ASPLOS 2023**, DOI `10.1145/3582016.3582047` / arXiv `2207.04606`.
2. **变长锯齿张量（Ragged）、三角域与分页 KV Cache**（Varlen Attention、PagedAttention、三角/带状矩阵，打破笛卡尔矩形边界与直接仿射寻址）
   - Fegade et al., *The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding*, **MLSys 2022**, arXiv `2110.10221`.
   - Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), **SOSP 2023**, DOI `10.1145/3600006.3613165` / arXiv `2309.06180`.
   - Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*, **MLSys 2025**, arXiv `2501.01005`.
   - Pizzuti et al., *Position-Dependent Arrays and Their Application for High Performance Code Generation*, **FHPNC@ICFP 2019**, DOI `10.1145/3331553.3342614`（MimIR 的依赖数组类型 `«i: n; «i + 1; T»»` 原生具备表达三角形状的潜力）。
3. **循环携带递推与并行前缀扫描**（Mamba 选择性 SSM `h_t = A_t h_{t-1} + B_t x_t`、`cumsum`、RNN、波前三角求解 `TRSM`，打破输出轴独立并行）
   - Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv `2312.00752`.
   - Blelloch, *Prefix Sums and Their Applications*, CMU-CS-90-190 (1990).
   - Henriksen et al., *Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates*, **PLDI 2017**, DOI `10.1145/3062341.3062354`（将 `scan` 与 `scatter` 作为与 `map`/`reduce` 并列的一等组合子）。
4. **数据依赖间接写（Scatter）、MoE 路由与变长 Grouped GEMM**（打破静态无冲突 `map_out`）
   - Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*, **MLSys 2023**, arXiv `2211.15841`.
   - Ye et al., *SparseTIR* (ASPLOS '23，覆盖 GNN 不规则图聚合 `scatter_add`).
5. **跨两级收缩的耦合在线重缩放归约**（FlashAttention 在单个循环中耦合维护 `(m, l, O)` 并在最大值更新时用 `exp(m_old - m_new)` 回溯重缩放旧输出累加器）
   - Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, **ICLR 2024**, arXiv `2307.08691`.
6. **数据依赖输出形状与排序/筛选类算法**（`nonzero`、`masked_select`、`sort`、`topk`、`NMS`，打破迭代域由输入形状决定）
   - Lai et al., *Relax: Composable Abstractions for End-to-End Dynamic Machine Learning*, **ASPLOS 2025**, DOI `10.1145/3676641.3716249`.
   - Ansel et al., *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*, **ASPLOS 2024**, DOI `10.1145/3620665.3640366`.

### 业界三层处理现状
- **第一层（通用编译器扩展一等抽象）**：稀疏格式（TACO / MLIR SparseTensor / SparseTIR）、锯齿张量（CoRa）、前缀扫描（Futhark `scan`）。
- **第二层（领域可编程内核骨架/模板 DSL）**：FlashAttention / FlexAttention、PagedAttention / FlashInfer、MegaBlocks Grouped GEMM。
- **第三层（保持为算法库/独立原语）**：`sort`、`topk`、`nonzero` 流压缩、`NMS`。

---

## 5. 演进方向：一族共享 CESK + PE 底座的微型抽象机

不应试图用一台巨无霸抽象机强吞所有算子，而应利用 MimIR“每台小 AM 仅一两百行 Mim、零 C++ pass”的轻量特性，构建**按数学骨架分层的微型抽象机家族**：

| 目标算子 / 模式 | CESK AM + PE 策略 | 与现有架构的复用关系 |
| :--- | :--- | :--- |
| **稠密规约 + `gather`** | 直接复用现有 `SchedTree` AM（`tensor.mim:1774` 已通过 `gather_fill` 将 `gather` 接入 `Prod#Fill`） | 100% 复用现有 AM 与 `Stage` 机制 |
| **`MoE` Grouped GEMM** | 在现有 `SchedTree` AM 外层新增 1 条 `RaggedFor(offsets, body)` 或 MegaBlocks 风格的 `TileTableFor(table, body)` | **内层整棵 GEMM `SchedTree`（含动态 `M` 的 `Switch` 冷/热 expert 选 pack B、`Peel` 满块静态微内核）与 Lean 证明原封不动复用** |
| **`scan` / Mamba SSM** | 构建同构的 3–4 指令 `Scan` AM（在 Environment 中增加前缀进位寄存器 `carry`，核心指令为 `ChunkScan` + `Split`/`Parallel` + `Stage`） | 共享 `Aff`、`Prod#Fill`、PE 擦除机制与基于结合律的 Lean 结构归纳框架 |
| **`scatter` / 排序 / 流压缩** | 保持为独立写回原语或算法级库原语（如私有桶归并 / Radix Sort / CUB），不硬塞进窗口切分 AM | 保护核心 `SchedTree` AM 的“输出子盒无写冲突”不变式与极小 TCB |

---

## 6. 核心参考文献索引

1. Jason Ansel et al. *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation*. ASPLOS 2024, Article 71. DOI: `10.1145/3620665.3640366`.
2. Siyuan Feng et al. *TensorIR: An Abstraction for Automatic Tensorized Program Optimization*. ASPLOS 2023. arXiv: `2207.04296`.
3. Ruihang Lai et al. *Relax: Composable Abstractions for End-to-End Dynamic Machine Learning*. ASPLOS 2025. DOI: `10.1145/3676641.3716249` (arXiv: `2311.02103`).
4. Martin Paul Lücke et al. *The MLIR Transform Dialect: Your Compiler Is More Powerful Than You Think*. CGO 2025. DOI: `10.1145/3696443.3708922`.
5. Roland Leißa et al. *AnyDSL: A Partial Evaluation Framework for Programming High-Performance Libraries*. OOPSLA 2018. DOI: `10.1145/3276489`.
6. Roland Leißa, Marcel Ullrich, Joachim Meyer, and Sebastian Hack. *MimIR: An Extensible and Type-Safe Intermediate Representation for the DSL Age*. POPL 2025. DOI: `10.1145/3704840`.
7. Thomas Würthinger et al. *One VM to Rule Them All*. Onward! 2013. DOI: `10.1145/2509578.2509581`.
8. Jonathan Ragan-Kelley et al. *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines*. PLDI 2013. DOI: `10.1145/2491956.2462176`.
9. Bastian Hagedorn et al. *Achieving High-Performance the Functional Way: A Functional Pearl on Expressing High-Performance Optimizations as Rewrite Strategies* (Elevate). ICFP 2020. DOI: `10.1145/3408974`.
10. Fredrik Kjolstad et al. *The Tensor Algebra Compiler* (TACO). OOPSLA 2017. DOI: `10.1145/3133901`.
11. Aart J. C. Bik et al. *Compiler Support for Sparse Tensor Computations in MLIR*. ACM TACO 2022. DOI: `10.1145/3544559`.
12. Zihao Ye et al. *SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning*. ASPLOS 2023. DOI: `10.1145/3582016.3582047` (arXiv: `2207.04606`).
13. Pratik Fegade et al. *The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding*. MLSys 2022. arXiv: `2110.10221`.
14. Woosuk Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. DOI: `10.1145/3600006.3613165` (arXiv: `2309.06180`).
15. Zihao Ye et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*. MLSys 2025. arXiv: `2501.01005`.
16. Federico Pizzuti, Michel Steuwer, and Christophe Dubach. *Position-Dependent Arrays and Their Application for High Performance Code Generation*. FHPNC@ICFP 2019. DOI: `10.1145/3331553.3342614`.
17. Albert Gu and Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv: `2312.00752`.
18. Trevor Gale et al. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*. MLSys 2023. arXiv: `2211.15841`.
19. Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024. arXiv: `2307.08691`.
20. Troels Henriksen et al. *Futhark: Purely Functional GPU-Programming with Nested Parallelism and In-Place Array Updates*. PLDI 2017. DOI: `10.1145/3062341.3062354`.
