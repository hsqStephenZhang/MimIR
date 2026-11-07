# 多级 tiling 对接 GPU SIMT：设计

本文规划把 `tensor.tile_req` / `tensor.tile_fuse` 的多级 tiling 接到 GPU（SIMT）后端，作为今后 tiling 实验的主平台。
CPU 上的 driver（TileGroup + TileFuse）保持不变，GPU 只在"层级绑定到硬件"和"bufferization"两处分叉。
数字都是先写下的预测，测量结果追加在后面，不覆盖预测。

## 0. 目标与非目标

目标：

- 同一套 `TileLevel` 配置和 R1–R7 改写目录，既能降到 CPU 循环，也能降到 GPU kernel。
- tiling 的每一级可以绑定到 grid（block）、block 内线程（thread）或线程内串行，绑定只是调度注解，不改变语义。
- block 级 promote 落到 shared 内存，线程级累加器落到寄存器。
- 在一张卡上完成 GEMM、批量 GEMM、softmax/layernorm 行归约和整个 attention 的端到端实验，基线为 PyTorch CUDA（cuBLAS）。

非目标（本阶段）：

- tensor core（`mma`/`wgmma`）、异步拷贝（`cp.async`/TMA）、多卡。
- AMD 后端；地址空间按 LLVM 的约定已经与 AMDGPU 对齐，但不验证。
- 自动调优；配置由公式给出，搜索留到之后。

## 1. 现状

### 1.1 GPU 插件已有的东西

- `gpu.mim`：Global/Shared/Const/Local 四个地址空间（1/3/4/5）、`gpu.launch`（`n_groups`、`n_items`、stream、动态 shared 内存）、`sync_work_items`、`synced_scope`、stream 与拷贝原语、`buf_alloc_copy` / `buf_copy_to_host`。
- `gpu.lower_btensor_map_reduce`（`phase/lower_map_reduce.cpp`）：把单个 btensor `map_reduce_post` 降成一个 kernel，每个输出点一个线程，归约维在线程内串行，grid 拉平成一维；每个算子各自 `alloc_copy` 输入、算完拷回、释放（`:289` 起）；只支持静态形状（`:335` 遇到运行时 grid 会告警跳过）。
- `nvptx` 后端：只读 `ctaid.x` / `tid.x`（`nvptx.cpp:576`）；每个 kernel 只允许一块动态 shared 内存（`:497`）；同步是 `llvm.nvvm.barrier0`；静态 shared 变量以 `addrspace(3)` 全局变量形式发出（`:579` 起）；数学函数走 libdevice。
- `check_addr_spaces_repl` 禁止在 shared/const/local 里 `malloc`/`free`。

### 1.2 tiling 侧与 GPU 的断点

- 加载 `gpu` 插件时 `GroupDefault::start` 第一行就返回（`group_rules.cpp:494`），TileGroup/TileFuse 不会运行。
- 流水线是 `group_default → lower_to_mem → add_mem → gpu.lower_btensor_map_reduce → btensor.lower_map_reduce`（`opt.mim`）。
  如果强行打开 tiling，LowerToMem 会把 `tiled` / `tile_fold` 降成 host 串行循环，`lower_btensor_map_reduce` 再把每个 tile 体里的小 map_reduce 各自变成一次 launch，并带上一来一回的拷贝。
- LowerToMem 的 promote（`stage`）、transients、`bounded_shape` 都在地址空间 0 上 `malloc`。
- `default_tile_config` / `group_blocks` 按 CPU 推导（寄存器数、L2 大小、mr/nr/kc/nc）。

### 1.3 实验机器

| 机器 | GPU | 状态 |
|---|---|---|
| magic.root | GTX 1650 SUPER（TU116，sm_75，4 GB，无 tensor core） | 驱动 595，`/opt/cuda`、`/usr/local/cuda`，libdevice 可用，PyTorch CUDA 可用；nvcc 不在 PATH |
| ss420e | 2× RTX 3090（sm_86） | 驱动 535 的 libcuda 缺 `cuCtxCreate_v4`，nvptx 执行测试失败，需先解决 |

GTX 1650 SUPER 的峰值：FP32 约 4.4 TFLOP/s（1280 核 × 1.725 GHz × 2），显存带宽约 192 GB/s，每 SM 64 KB shared/L1、64K 个 32 位寄存器。
所有预测以这张卡为准。

## 2. 总体设计

### 2.1 层级绑定（新增 `bind` 字段）

`TileLevel` 从 `[kind, sizes, tail, promote]` 扩成 `[kind, sizes, tail, promote, bind]`：

| bind | 含义 | 对应 |
|---|---|---|
| 0 | 串行（CPU 现状；GPU 上为线程内循环） | `scf.for` |
| 1 | grid：每个 tile 一个 block | `#gpu.block`、TVM `bind(blockIdx)` |
| 2 | block 内线程：每个 tile 一个线程 | `#gpu.thread`、TVM `bind(threadIdx)` |
| 3 | warp（预留，用于 warp 级归约与 shuffle） | `#gpu.warp` |

约束：

- 只有 par 级可以绑定到 1/2；red 级只能是 0，跨线程归约见 §2.6。
- 绑定由外向内单调：一旦出现 2，外层不能再出现 1。
- CPU 后端忽略 `bind`（所有级都当 0），所以现有配置只要在末尾补 0 即可。

绑定不改变语义：`tiled_multi_impl` 是逐元素定义的，tile 以什么顺序、在哪个硬件单元上执行都得到同一结果。
所以 R1–R7 的证明义务不变，绑定和 `sched` 一样属于调度注解。

### 2.2 典型配置（GEMM，`C[M,N] = A[M,K]·B[K,N]`）

```
L0 par  (BM, BN)  bind=1           // 每个 block 一个 BM×BN 输出块
L1 red  (BK)      promote=A|B      // K 分块，A/B 面板进 shared
L2 par  (TM, TN)  bind=2           // 每个线程一个 TM×TN 微块
L3 par  (1, 1)    bind=0           // 线程内寄存器块，串行展开
```

初始值：BM=BN=64，BK=16，每块 256 线程，TM=TN=4（线程块 16×16）。
寄存器：累加器 16 个，加上 A/B 的片段，每线程约 40 个，留出 occupancy 余量。
shared：`(64+64)×16×4 B = 8 KB`，单缓冲。

### 2.3 kernel 的生成（LowerToMem 的 GPU 分支）

新增 `lower_tiled_gpu`，在 `lower_tiled` 里按最外层绑定分派：

1. 最外层 `bind=1` 的 region 变成一次 `gpu.launch`。
   grid 大小 = 各 par 维 tile 数的乘积；kernel 内用 `ctaid.x` 经 `affine.delinearize` 还原多维 tile 坐标，再算出 `off`。
   nvptx 只有 x 维，因此统一线性化，不依赖 y/z。
2. `bind=2` 的级不生成循环：用 `tid.x` 线性化后还原线程坐标；线程数为 tile 数之积，作为 `n_items`。
3. `bind=0` 的级照旧生成循环（`affine.counting_for`）。
4. region 的 ins 成为 kernel 参数（全局指针）；outputs 在 host 侧分配到 global 内存后作为参数传入。
5. kernel 体从 host 世界切出，复用 `split_off_kernels`。

cap 尾部用谓词处理（`in_range` 式守卫，block/thread 越界时直接返回）；GPU 上不用 peel，避免生成多个分叉的 kernel。

### 2.4 存储层次

| 对象 | CPU 现状 | GPU |
|---|---|---|
| block 级 promote 面板 | `stage` → heap | shared（地址空间 3），静态大小 |
| 线程级累加器 | `tile_fold` 的 acc buffer | local/寄存器（地址空间 5，`mem.mslot`） |
| region 输出 | heap | global（地址空间 1） |
| transients | heap，tile 末尾释放 | 不允许在 kernel 内分配；用 `bounded_shape` 的常量上界改成静态 shared/local |

做法：

- `stage` 已经带一个 `scope: Nat` 参数，目前都传 0，看起来是为存储层级预留的。
  第一步先核实它的现有语义；若未使用，则约定 0=默认、1=shared、2=local，由 driver 按绑定级填写。
- `bounded_shape` 已能给出 tile 存储的常量上界，GPU 分配直接用它，不需要运行时大小。
- 多块静态 shared 变量由 nvptx 以 `addrspace(3)` 全局变量发出；动态 shared 只在大小依赖运行时参数时使用。

### 2.5 协作拷贝与同步

block 级 promote 的拷贝循环由块内所有线程分摊：

- 按线性线程号做跨步，最内维按连续地址分给相邻线程，保证合并访存。
- 拷贝前后各插入一次 `sync_work_items`；用 `gpu.synced_scope` 包住，`remove_double_syncs` 去掉相邻的重复同步。
- 面板布局按 GPU 选：A 面板转置存放（`[BK][BM]`），按需加 1 个元素的 padding 避免 bank conflict。
  `stage` 的 `outer_perm` 和 `pad` 参数已经可以表达这两点。
- 双缓冲留到 G4。

### 2.6 跨线程归约（新改写 R8）

行归约（softmax 的 max/sum、layernorm 的 mean/var）在 GPU 上需要块内并行归约：

- R8：把归约维按线程切开，每个线程先串行归约自己的一段，再在 shared 里做树形合并。
- 与 R3 不同，R8 改变了合并顺序，**需要结合律和交换律**，是新的证明义务。
  浮点加法不满足结合律，所以 R8 只在 `math.mode` 允许 reassoc 时合法，与 `math.mode.fast` / `contract` 的约定一致；max/min 不受此限制。
- 先用 shared 树形归约实现，warp shuffle（bind=3）作为后续优化，需要新增 shuffle 原语。

### 2.7 数据常驻设备

- 多组模型的中间张量在 global 内存上分配和释放，只在 extern 入口拷入、出口拷出。
- 一个 extern 一个 GPU 会话：沿用 `auto_init` / `auto_deinit` 的合并逻辑。
- 现有 `lower_btensor_map_reduce` 的逐算子拷贝保留给没有被 tiling 覆盖的算子，作为回退路径。

### 2.8 GPU 配置（`default_tile_config_gpu`）

输入的硬件参数换成：warp 大小、每 SM shared 内存、每线程寄存器上限、目标每块线程数。
kind 1（收缩）用 §2.2 的四级；kind 2（行归约）用 `L0 par (1, …) bind=1` 加 R8；kind 0（逐元素）用 `L0 bind=1`、`L1 bind=2`，每线程处理 4 个连续元素以便向量化访存。

## 3. 流水线改动

1. `GroupDefault`：gpu 加载时仍运行 TileGroup + TileFuse（只在 `tile-fuse` 模式下），配置换成 GPU 版。
2. `LowerToMem`：新增 GPU 分支（§2.3、§2.4），产生 `gpu.launch`。
3. `gpu.lower_btensor_map_reduce`：跳过已经位于 kernel 内的 map_reduce，只处理 host 侧剩下的算子。
4. `AddMem`：kernel 体内的内存链用 GlobalM/SharedM/LocalM；需要核实 fresh-mem 机制在 kernel 体里是否成立。
5. 关掉 kernel 内不适用的 CPU 专属 lowering：多项式 exp（已在 gpu 加载时关闭）、BLIS 打包规则、`prepack`。

## 4. 里程碑与验收

每个里程碑先写预测，再测量；正确性一律与 CPU 结果逐元素比较。

| 里程碑 | 内容 | 验收 | 预测（GTX 1650S，1024³ SGEMM） |
|---|---|---|---|
| G0 | 打通：`bind` 字段、gpu 加载时运行 tile-fuse、lit 中检查生成的 launch 结构 | lit 结构测试；CPU 回归不变 | — |
| G1 | 两级（block × thread，线程内串行 k），无 shared | 与 PyTorch CUDA 结果一致 | 约 5–10% cuBLAS（访存受限，每个 FMA 两次 global 读） |
| G2 | block 级 promote 到 shared、协作拷贝、同步、寄存器累加（4×4） | 1024³ 达到 cuBLAS 的一定比例 | 约 40–60% cuBLAS |
| G3 | 线程块 8×8、A 转置存放与 padding、向量化 global 读 | 同上 | 约 60–80% cuBLAS |
| G4 | 数据常驻设备、R8 行归约、批量 GEMM、整个 attention | gemm-bench 模型在 GPU 上运行 | attention 进入 PyTorch CUDA 的 2× 以内 |
| G5 | 符号形状与运行时 grid、双缓冲 | 符号 M 不慢于静态 10% 以上 | — |

## 5. 风险

- **CPS 与闭包**：kernel 体内的 `tile_fold` 步进函数若产生闭包，nvptx 无法处理；需要保证 kernel 内全部是一阶循环（`affine.lower_for` 之后 `clos` 不应介入）。
- **内存链**：本轮刚修过 AddMem 的串接顺序（`0f77adab10`）；kernel 内有多条地址空间的内存，可能暴露新的分叉。
- **nvptx 的限制**：一维 id、单块动态 shared；多维全部靠线性化，dynamic shared 只能有一块。
- **测试环境**：magic 是 4 GB 的入门卡，大尺寸 attention 放不下时改用 ss420e，需要先解决其驱动问题。
- **调试**：kernel 内错误只能靠结果比较定位；需要一个"CPU 解释执行同一 tile 程序"的对照（`bind` 全部当 0 即可得到）。

## 6. 待定问题（逐个讨论）

1. `bind` 放进 `TileLevel`，还是像 `sched` 那样做成独立的调度注解？前者改动小，后者与"调度和语义分离"的方向更一致。
2. `stage` 的 `scope` 参数的现有语义，以及是否借用它表达存储层级。
3. R8 的合法性条件放在哪里：`math.mode` 检查放在 driver，还是做成规则的 witness。
4. GPU 实验的主力机器：先用 magic（马上可用），还是先修 ss420e 的驱动问题用 3090。
5. 基线：PyTorch CUDA（cuBLAS）之外，是否需要 IREE（CUDA 目标）和 TVM。
