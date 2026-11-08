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

## 7. 实现记录（2026-11-07，分支 `feat/gpu-tiling`）

测量数据见 `bench/gpu_tiling/RESULTS.zh-CN.md`，这里只记录决定和与设计的出入。

### 7.1 已完成：G0、G1

- §6 问题 1：`bind` 作为 `TileLevel` 的第 5 个字段 `[kind, sizes, tail, promote, bind]`；TileFuse 仍接受 4 元组（视为 `bind = 0`），`default_tile_config` 全部补 0。
  进入 IR 后它是调度注解：TileFuse 用 `tensor.bind to` 包住 `tiled` / `tiled_multi` 的 body lambda，`tiled` 的元组 `(s, ts, tail)` 不变，`grid_tile_plan` 剥掉注解并给出 `plan.bind`。
  约束按 §2.1 检查：只有 par 级可绑定，work-group 级不能嵌在 work-item 级内。
- LowerToMem 对绑定层不再逐维生成循环，而是生成一个线性 `affine.For`（tile 总数），用 `core.nat.div/rem` 还原各维偏移，body 用 `gpu.bind.group` / `gpu.bind.item` 注解（与 `ll.vec` 同一种"For body 上的恒等注解"）。
  绑定层要求字面量 extent，尾部只能 cap（§2.3 的谓词方案），peel 报错。
- 与 §2.3 不同，kernel 不在 LowerToMem 里生成，而是由新 phase `gpu.outline` 在 `buffer.lower_ptr` 之后外提：
  此时 `Buf` 已是 `mem.Ptr`，捕获的主机指针一一 `mem.alloc (_, 1)` + `copy_to_device`，launch 后 `copy_to_host`、释放、`auto_deinit`。
  kernel 体保持 `mem.M 0` 的 generic 链，入口用 `gpu.generic_mem` / `gpu.global_mem` 转换内存记号（都擦除为空），指针参数 `addrspacecast` 到 generic；LLVM 的 InferAddressSpaces 把 load 还原成 `ld.global`。
  这样 kernel 内不需要 `Buf` 带地址空间，也回避了 §2.4 中"kernel 内 `buffer` 层"的改造。
- `gpu.lower_btensor_map_reduce` 跳过绑定循环内的 `map_reduce_post`，它们由 `btensor.lower_map_reduce` 降成线程内的串行循环。
- `GroupDefault` 在 gpu 加载时也跑 TileGroup / TileFuse（`tensor:tile-fuse=1/2`）。

顺带修掉的后端问题：

- `affine.lower_for` 只支持累加器里一个 `mem.M`；kernel 内 GlobalM + SharedM 的循环会 `cannot coerce to mem.M 3`，现在每个 `mem.M` 分量各占一个 bb 参数。
- `ll` 的指针 `bitcast` 跨地址空间时发 `addrspacecast`。
- nvptx 把地址空间 0 的 `mem.slot` 发成模块级全局（所有线程共享），现在按 ll 的方式 `alloca`。
- 没有用到 libdevice 时 device IR 不经 `opt`：cherry-pick 了 `fix/nvptx-llvm-opt`。
- 捕获的 region 输入元组必须按分量传递，否则指针仍指向主机内存；驱动 595 + Linux 6.12 的 HMM 让 kernel 能读主机页，所以这个错误只表现为慢。

### 7.1b G2：promote 到 shared（功能完成，性能未达标）

- `tensor.lower_to_mem` 记录当前所在绑定层（`bound_level_`）：work-group 级的逐点填充（`stage` 的面板、fold 的初值拷贝等）走 `btensor::build_pointwise` 的线性循环，body 用 `gpu.bind.cooperative` 注解。
- `gpu.outline` 把 cooperative 循环改成 `begin = item_id, step = n_items` 的跨步循环，前后各插一次 `sync_work_items`（通过 `global_mem`/`generic_mem` 接到 generic 链上）；work-group 级的 `mem.alloc (_, 0)` 变成 `gpu.slot_shared`（nvptx 发成 `addrspace(3)` 全局变量再 `addrspacecast` 到 generic），对应的 `mem.free` 删除。
  §2.4 设想用 `stage` 的 `scope` 参数区分存储层级；实际上按"分配发生在哪一绑定层"判断就够了，`scope` 仍未使用。
- `mem.add_mem`（resolve 模式）为此做了两处放宽：只替换与当前链同地址空间的 `mem.M` 操作数、只在同地址空间内推进当前内存；以及 pinned 回调的传递闭包不再穿过"第一个参数是 `mem.M`"的 lam（kernel 和它的 body），否则 kernel 里的 `mem.fresh` 永远不会被解析。
- 结果正确；1024³ 从 5.7 ms 修到 2.60 ms（cuBLAS 的 26%）：去掉物化的 `splat` 初值和单独的首块，面板改成 k 最外的 `(k, ceil(n/4), 4)` 布局消除 bank conflict（ncu：5880 万 → 8 万）。
  与同结构的 CUDA C++（累加器在 shared，1.76 ms）差 1.48×，与经典寄存器累加 CUDA（1.47 ms）差 1.77×；float4、swizzle、寄存器累加器留作性能工作。

### 7.1c G3：归约级放进 work-item 级，累加器在寄存器

§2.2 的四级顺序 `L0 group → L1 red(promote) → L2 item` 把 item 循环放在 k 循环里，累加器只能落在 work-group 共享的存储；
要得到经典 CUDA 的寄存器累加结构，顺序必须是 `group → item → red(promote)`，而 promote 的面板仍要覆盖整个 group tile 并由全组协作填充。
实现要点：TileFuse 在 item 级内做 "group-wide" promote（切片沿线程轴放大到 group 的 `off/ext`，map 按 item 相对偏移平移）；
LowerToMem 把 item 体内不依赖 item origin 的值视为组内一致，填充标 `cooperative`、分配标 `gpu.shared_scope`；affine 线性形式支持两层 tile 偏移的符号加减。
结果：1.66 ms，与经典寄存器累加 CUDA（1.47 ms）差 13%，差距在每 k-block 多两次 barrier 和 i64 地址运算。
§2.2 的级序建议应改为 `group → item → red`；`L1 red` 在 group 级只适合累加器放 shared 的场合。

### 7.1d G3 续：barrier 对、计数协作循环、A 面板按源连续维填充

- barrier：`gpu.outline` 原来给每个协作循环前后各一次 barrier，两次填充之间就多出两次。
  现在 `gpu.bind` 多了 `staged`（只在前面放 barrier）和 `chained`（不放）两种协作标注，`gpu.fence: mem.M 0 → mem.M 0` 由消费者发出、外提时变成 `sync_work_items`。
  LowerToMem 的 `lower_map_reduce` 在改写输入时置 `staging_`，第一个 uniform 填充标 `staged`、后续标 `chained`，输入改写完后在消费者的内存链上放一次 `fence`；进入嵌套 body（`pending_` 重置处）时 `staging_` 清零，保证只有同一链上的填充被接续。
  `remove_double_syncs` 也看穿 `generic_mem`/`global_mem`/`slot_shared` 并把 kernel 入口 var 的投影当作入口记号。每 k-block 从 4 次 barrier 到 2 次，但只省 0.01 ms。
- 协作循环：extent 能被 work-item 数整除时，外提成 `for j < extent / n_items` 的计数循环，体内 `i = item + j·n_items`，ptxas 能完全展开并批量发出 `LDG`；之前的跨步形式（`i = item; i < extent; i += n_items`）被编成带 prologue/epilogue 的通用循环（每次填充约 90 条 SASS）。也只省 0.005 ms。
- 真正的差距在 A 面板的填充访存模式（§2.5 早就写了"最内维按连续地址分给相邻线程"，但实现按面板行主序分配）：
  `lower_stage` 用读 map 求出面板哪一维对应源的最后一维（`contiguous_dim`），`lower_generate` 让它跑最内（`btensor::linear_loop` 的 `inner` 参数），并给转置填充的 shared 面板加 padding（`padded_panel`，64 → 68）。
  kernel 从 1.647 ms 到 0.868 ms；ncu 里 L1 tag 访问从 1730 万降到 157 万，其余指标（指令数、LDS wavefront）本来就与 CUDA 版一致。
  这也说明 CUDA "经典写法"（每线程读 4 个连续 k）本身不是合并访存的最优形状，生成版现在比它快 1.7 倍、达 cuBLAS 的 78%。

### 7.1e float4 与 swizzle：向量填充、对齐证明、面板布局模型

- 向量访存的表示是一个**重解释视图**而不是访问变体：`buffer.vectorize (n, c) buf: Buf (r, s, T) → Buf (r, c, «n; T»)` 把最后一维按 `n` 个一组看成向量元素（`c` 是 `s` 的最后一维除以 `n`），之后普通的 `read`/`write` 就搬一个向量；
  `lower_ptr` 把它降成指针类型的 `bitcast` 加一次 `mem.align (p, n·sizeof T)`（所有分配都 16 字节对齐，slot 的 `alloca` 和 shared 全局变量也标 `align 16`，所以这个声明成立）。
  ll 后端把 `mem.align` 发成 `llvm.assume` 的 `align` bundle，并把声明经 `mem.lea` 按元素大小的 gcd 传到每条 load/store 上直接写 `align N`（只靠 assume 不够：NVPTX 的地址空间推断会复制指针链，对齐事实就丢了）。
  LLVM 把 `load [4 x float] align 16` 拆成 4 个标量再由 LoadStoreVectorizer 合回 `ld.global.v4` / `st.shared.v4`。
- tensor 级是同样的视图 `tensor.as_vectors (n, c)`（`vectorize` 这个名字已被调度原语占用）；`lower_get` 经 `storage_access` 处理它：块坐标乘 `n` 得到首元素坐标，走源缓冲区的布局，再在 `buffer.vectorize` 过的缓冲区上按块读。
  `stage_read_impl` 多了 `s_out`（分块后的面板形状）、`scale`（向量维的坐标乘以 `n`）和 `getter` 三个参数，标量路径传 `scale = ‹1›`、`getter = get`，生成的 IR 不变；向量路径的 getter 是 `λ (t, i). get (as_vectors n t) (i 的最后一维 / n)`。
- 对齐证明在 `lower_stage`：源最后一维的 Linear 形式里其它项的 gcd、`extent`、`step = 1`、物理行长都要是 `n` 的倍数，`start` 由 `nat_multiple` 判定——tile 区域的 origin 变量在 `lower_tile_body`/`lower_tile_fold` 登记为各自 tile 尺寸的倍数（`multiples_`），`core.nat.add/mul` 递归。
  向量维是面板里"原样"对应源连续维的那一维：若该逻辑维被面板的内层 tile 切开（如 B 面板 `(k, n/4, 4)`），就是内层 tile 维；否则是外层维。
- `lower_generate` 对向量填充按分块形状循环，每步 `compute` 得一个 `«n; T»`：目的维连续时一次 `write_vec`，转置时 `n` 次标量写（A 面板）。
- 面板布局改成一个 bank 模型（`panel_layout`，替代 `padded_panel`）：按填充的真实线程映射（`linear_loop` 的分块线性序、每线程 `n` 个元素）模拟一个 warp 的 shared 写，
  按 128 位访问分 8 线程一相位，数每相位内落到同一 bank 的不同地址数；候选是"不变"、对 `inner+1` 维 padding 1–7（保持行 16 字节对齐）、以及 16 字节块的旋转式 swizzle
  `chunk' = (chunk + (row >> shift) % groups) % chunks`（行长须为 2 的幂）。选 wavefront 最少者，平手依次取不变、padding、swizzle。
  用旋转而不是 XOR，是因为消费者的读 map 是 affine 的（`floordiv`/`rem` 可表达，`affine.lower_index` 会折成移位和掩码），swizzle 就能作为"存储映射"复合进 `btensor.map_reduce_post` 的输入 map（`swizzle_map`）；
  `lower_get` 走 `swizzle_coords`（现为 `layout_coords`）；整块拷贝的几处入口对 swizzled 面板直接报错。
- 效果（64×64，BK=16，4×4）：float4 填充 0.868 → 0.860 ms（填充已合并访存，只省指令）；A 面板转置写的 2 路冲突经 padding 无解（16 字节对齐下 68 仍 2 路），模型选中 swizzle 后 st/ld bank conflict 都归零，**0.830 ms，cuBLAS 的 82%**。
  128×128，BK=8，8×8 的配置：A 面板 padding 只能按内层 tile（8 元素）加，无法达到 132；swizzle 后写冲突归零，但每线程 8 列连续的 B 读是 2 路冲突（经典写法把 8 列拆成两段 4 列、间隔 64），寄存器 128，0.91 ms——需要 warp 级分条的 thread tile 布局（见 §7.3）。

### 7.1f 消费者感知的面板布局与向量读：对照 SGEMM_CUDA 的 K7/K10

用 `/workspaces/compile/SGEMM_CUDA` 在同一块卡上定位差距（1024³，harness 自带计时）：K6 vectorize 0.831 ms（与生成版持平），K7 resolve-bank-conflicts 0.770 ms，K10 warptiling 0.679 ms（即 PyTorch 的 cuBLAS 水平）。
生成版的 4×4 thread tile 每 k 步 2 条 LDS.128 配 16 条 FFMA，总指令 6380 万，而 K10 的 8×16 tile 是 6 配 128、3680 万；大 tile 又要求 B 面板读无 bank conflict——K7 靠重排 B 面板（线程号最内），K10 靠 warp 级分条。

- 布局模型泛化为 `Layout {split, perm, swizzle}`：面板最后一维可按 16 字节块切开、各维任意置换（物理秩不变：多出来的维合并回第二维，`merge_coords`）、再加 padding 或旋转 swizzle。
  代价函数同时模拟填充的 warp 写和 **消费者的 warp 读**：`panel_reads`（`lower_stage`）从切片 start 里的 work-group origin 分量（`origin_dims_`）找出面板各维对应的循环维，item 级的 tile 网格（`item_grid_`，由 `lower_tiled` 的线性 item 循环登记）给出线程 → tile 索引；
  被 stage 按 item 步长切开的维由线程选 tile、其内层 tile 维整块读，其余维逐步固定。一个线程落在同一对齐 16 字节段内的元素算一次 128 位访问，否则逐元素；代价 = 发射槽 + wavefront。
- 消费者的向量读显式化：`btensor.vectorize n f` 标记输入 map（访问函数上的调度标注，类型层面表达不了"按向量读但逐 lane 送进 combiner"），btensor 的 reader 在 `buffer.vectorize n` 过的输入上读坐标所在的块再取 lane（`Axm::isa<vectorize, 1>`，因为 map 本身的函数类型是再一层 curry）。
  LowerToMem 对 work-group 面板（`panels_`）且最后一维是 `n` 倍数的输入都加此标记。之前靠 ptxas 合并 LDS，一旦布局带 split 就只合并成标量读，模型假设与实际不符（128 配置 B 读仍 2 路冲突）。
- 结果（128×128，BK=16，8×8）：B 面板选 `split=4, perm=(2,0,1,3)`，即 `(h, k·n_hi, 4)`——16 字节块按"块内序号 h"分组、线程号 n_hi 相邻，和 K10 的分条读等价但线程的 tile 仍是连续 8 列；每 k 步 4 条 LDS.128，读冲突归零，MIO 节流 5%，**0.775 ms（cuBLAS 的 88%）**，与 K7 持平。64×64 配置 0.839 ms（显式向量读比 ptxas 合并多了点地址运算）。
- 8×16 thread tile（`gemm_tiled_g3_128w.mim`，128 线程）累加器 128 个元素没进寄存器（232 LDL/448 STL，55 寄存器）：LLVM 没把 8×16 的 tile 循环完全展开，SROA 失败。要么 MimIR 侧展开，要么给 `affine.For` 带 unroll 元数据，是到 K10 的最后一步。

### 7.1g 8×16 thread tile：寄存器预算、lane 的来源与展开

- 8×16 的累加器原来落到 device `malloc`（1.9 s）：不是 LLVM 没展开，而是 btensor 的 leaf 嵌套只在输出 tile ≤ 64 个单元时用 SSA 寄存器累加器（`registers` 嵌套），超过就退回带临时缓冲区的 `blocked` 嵌套，那个每 k 步分配的 128 元素缓冲区在循环内、`heap2stack` 不转。
  这个预算改成 `btensor.register_budget ()`（normalizer 读 `-X btensor:registers=<n>`，默认 64）；GPU 的 work-item 本来就把整块 tile 放寄存器，128 配置用 256。
- swizzle 只在最后一个物理维恰好是 16 字节块时才作为候选（否则先 split）：块内 lane 必须是最后一维的坐标，才可能是字面量。
- `btensor.vectorize` 标注改为携带逻辑形状、布局 map 和逻辑 map：reader 先按逻辑坐标算 lane 和块首，再用布局 map 求块首的物理位置。
  即便如此，lane 在 `affine.lower_index` 时仍不是字面量：`range_unroll` 展开后的单元坐标是字面量，但 leaf 的 `cell_at`/`cell` 回调是独立的 continuation，坐标作为参数进入；给 `cell_at` 加 `@tt` 会触发 tensor.mim 的类型错误，留待下一步。
  目前 lane 的字面量化靠 LLVM 内联 + SROA：64/8×8 配置默认即可，8×16 要 `-X nvptx:Xopt="… -unroll-threshold=20000"`。
- 用 vec 级 `(2, (1, 0, 16), …)` 在 Mim 层展开 16 步 k 循环对 8×8/64 配置略慢（0.775 → 0.82 ms），且 8×8×16 以上的展开会让 `Check::alpha` 递归爆栈（`ulimit -s unlimited` 可过）——核心的 alpha 等价检查是递归的，对大段直线代码不够健壮。
- 结果：**8×16/128 线程/BK=16 配置 0.718 ms，cuBLAS 的 94%**（226 寄存器，每 k 步 6 条 LDS.128，k 循环全展开）；K10 为 0.679 ms。8×8 配置 0.770 ms，64 配置 0.843 ms。

### 7.1h R8：绑定到 work-item 的归约级

- `tile_req` 的归约级现在可以 `bind = 2`（`(1, (0, 0, k), tail, promote, 2)`，须直接嵌在 work-group 级里）：TileFuse 把 `tile_fold` 的 step 体包进 `tensor.bind 2`，`lower_tile_fold` 读到后走 `lower_bound_fold`，gpu 插件未加载时退回顺序 fold。
- 结构：每个 work-item 把自己的 block 折进一份私有的、用 seed 初始化的 tile（走和顺序 fold 一样的 in-place step），写进 shared 里按 item 堆叠的 partials；`gpu.fence`；再由"每个输出元素一个 work-item"的第二个 item 循环串行合并该元素的全部 partials，`fence` 发布结果 tile。
- merge 从 step 的 combiner 结构提取：`λ (acc, ins). op (acc, e)` 或 `op (e, acc)`、`e` 不含 `acc` 时取 `op`（reduce 的 `add`/`fmax`、GEMM 的 `mac` 都满足），并要求 seed 是 scalar（视为 `op` 的单位元）。这就是 §2.6 的证明义务：`op` 结合、交换，`math.mode.fast` 之下成立。元组 carry（online softmax 的 (m, l)）需要显式 merge，尚未支持。
- `gpu.outline`：kernel 的 item 数取所有 item 循环的最大值；比它短的 item 循环加 `item_id < count` 守卫。第一版守卫把循环状态当自由变量引用，调度器把 `fence` 和 shared slot 下沉进了两个分支，barrier 落在发散路径上，sm_75 上直接死锁；现在把状态作为分支调用的参数传入，barrier 就在分支前统一执行。同理 combine 的汇合 continuation 必须 `filter = false`，否则被 beta 归约复制到两条路径里。
- 结果（12×1024×1024）：行求和 kernel 从 E0 的 2.61 ms 到 **0.291 ms**，与 PyTorch `x.sum(-1)` 的 0.286 ms 持平（DRAM 读 51 MB，不再放大；每请求 16 sector）。三趟 softmax（max、Σexp(x−max) 融合 map、div 融合 exp，每趟自己的 `tile_req`）1.26 ms，对 E0 的 6.5 ms 和 PyTorch 单 kernel 的 0.62 ms。
- 顺带发现：同一函数里 `tile_req` 外提的 kernel 和 E0 路径的 kernel 混用时，中间 op 的内存链类型冲突（`core.M 1` 对 `core.M 0`），所以 softmax 里的 `exp(x−max)` 被两处重算以全部走 tile_req；这是 E0 与 outline 两套 host glue 没有统一的问题。
- **online softmax**（`bench/gpu_fusion/softmax_online.mim`）：归约级可以带第六个字段 `merge`（`(1, sizes, tail, promote, 2, merge)`），TileFuse 用 `tensor.merge_with` 标在 step 上，`lower_bound_fold` 优先用它、否则才从 combiner 提取。
  carry 是 `(max, sum)` 元组——Mim 把 `[F32, F32]` 归一成 `«2; F32»`，bufferizer 原来拒绝数组元素类型；现在允许体为标量、长度字面量的扁平数组元素（内存里就是 struct），seed 取 `splat` 的闭合元组。
  两趟：pass 1 一次读 x 做 (m, l) 的 online fold + R8 merge（0.459 ms），pass 2 读 x 归一化（0.692 ms），**1.15 ms**，对三趟 1.26 ms、PyTorch 0.62 ms。pass 1 是指令密集（每元素两次 exp，2750 万条指令对行求和的 206 万），merge 的 8 个 item 串行扫 256 个 partial 时其余 warp 在 barrier 上等（stall 46%）；pass 2 已经是带宽（读 51 写 52 MB，153 GB/s）。
  所以 2-pass online 在 GPU 上的收益上限就是省掉一趟读；要到 PyTorch 的单趟得让一行常驻 block（8 行 × 1024 × 4 B 进 shared/寄存器），即归约 producer 融进消费者 tile。

### 7.1i 整行融合单 kernel：归约 producer 融进 work-group tile

- TileFuse：work-group 级（`bind = 1`）融合一个归约 producer 时，若后面紧跟 work-item 级（`bind = 2`），就把该归约 tile 再按 item 数拆成 `bind = 2` 的归约级（`split_over_items`：唯一的归约循环长度 K 被 item 数整除时，`(1, (…, K/items), 1, 0, 2[, merge])`），即 R8 fold 直接嵌在 group 体里。
  拆开的 tile 记在 `resident_`（按原 producer 索引）：后续同一 producer 的读——另一个 producer 的 step 体里的（s 读 mx），或 item 级 body 的——都改读这份共享 tile（偏移、extent 一致时直接别名，否则 `slice_box` 视图），不再按 member 重新融合。否则 item 级会把 member 再融一遍，每个 item 串行重算整行（第一版就是这样：单 kernel 但 204 ms）。
  显式 merge 从 producer 值上的 `tensor.merge_with merge (map_reduce …)` 取：`fuse()` 剥掉标注、把 merge 传给拆出的归约级。
- LowerToMem：
  融进 group tile 后归约的 extent 是 `min(x, size)`、`size` 本身是 tail 选择项，`linear_step` 原来只认 alpha 相等或字面量上界；加 `nat_bounded_by`（alpha 相等、字面量上界、或 `min` 的一侧有界）。
  region 输入里的 `slice`/`slice_box`（源已 realize 或是变量）保持为读而不是物化（`view_flags`/`view_sources`/`view_reads`）：源经 proxy 传入，体里重建视图，`storage_access` 把它组合成纯读。原来 `materialize` 走 `slice_impl` 复制一份（每个 item 一次 device `malloc`，三份 8 元素的拷贝让 kernel 慢到 290 ms）。视图判定只做一次：先物化源会让后面的输入变成"已 realize"，两次判断不一致曾把 proxy 类型弄错（`outer_rows.mim`）。单个 tensor 输入是一个分量而不是其记录类型的两个字段。
  `producer_post` 在 `nps = 0`、post 是恒等（`λ (x, ()) = x` 或 `return x`）且类型一致时直接返回 acc，省掉归约 tile 之后的恒等 map_reduce 拷贝。
- 结果（12×1024×1024，`bench/gpu_fusion`）：
  `softmax_fused.mim`（members `(mx, s)`，exp 重算）：**1 kernel、0 malloc、5 barrier，1.20 ms**；DRAM 200 MB（读 x 三遍 + 写一遍），DRAM 吞吐 90% 峰值——已是这个三趟读结构的带宽上限。
  `softmax_fused_online.mim`（member 是 `merge_with merge` 的 (m, l) online fold）：**1 kernel、0 malloc、3 barrier，0.92 ms**（读两遍 + 写一遍 ≈ 150 MB）。PyTorch 0.62 ms 只读一遍。
  剩下的差距是 x 的重读：group 的 8 行 × 1024 × 4 B = 32 KB，并发 block 的工作集（~20 SM × 5 block × 32 KB）远超 1 MB L2，所以每趟都回 DRAM。要到单趟得把 group 的输入 tile 常驻 shared（§2.7/G4：parallel 级的 promote，目前 promote 只在归约级上），或者 item 把自己的 4 列 × 8 行留在寄存器里。

### 7.1j G4：work-group 级 promote 把输入 tile 常驻 shared；R8 的树形合并

- TileLevel 的 `promote` 位现在也可以放在 `bind = 1` 的 parallel 级上：`parallel()` 在融合之前，把 root tile 对该输入的读（一个 `slice_box` box）换成 `stage @(T, r, 0) s ((), (), id, ⊥) 0 (slice x box)`——k = 0、恒等排列的 `stage` 就是"整块拷进 shared"。这份 staged tile 也登记进 `resident_`，于是后面融合进来的归约 producer（(m, l) fold 的 step 读 x）和 item 级的读都变成它的视图；`stage` 自己的填充读（`depends_on(shared, read)`）要排除在外，否则 resident 替换会把自己的输入换成自己（出现过 `stage (stage (slice x))`）。
  `lower_stage` 对 `slice` 输入不再走"已有 buffer 则别名"的分支（slice 是视图不是存储，而且其 box 类型带运行时 extent，`buf_of` 会拒绝），直接走协作填充路径；`is_view` 把 `stage` 结果也当作可视图的源。
- `Heap2Stack` 原来把循环 head 支配的所有 continuation 都算"在循环里"，所以 fold 的 per-item 私有 tile 一旦排在某个循环的 exit 之后就留成 device `malloc`（每 thread 一次，kernel 慢到 178 ms）。现在只有能重新到达递归祖先的 continuation 才算每次迭代都跑；exit 只跑一次，其分配转成 slot。
  连带两处：CPU 上更多 alloc 变成 slot 后，SSA 构造产出了带元组下标的 `insert`（二维数组 `«4, 8; F32»` 的 `(i, j)`），ll 后端的 alloca 模拟原来只接受标量下标，现在按分量逐个给 GEP 下标；`blis_workloads.mim` 的乘法计数在 LLVM -O3 下多出一次 8 lane 的 `fmul`（alloca 让 LLVM 多推测了一个向量乘法，Mim 层的 `fmul` 条数不变、结果正确），该计数改成在 `-O0` 上数，只衡量 MimIR 的 realization。
- R8 的 combine 从"每个输出元素一个 item 串行扫 n 个 partial"改成 shared 里的树：步长 s = n/2 … 1，每步 `min(n, s·elems)` 个 item 各合并 `s·elems / items` 对 `p[i] ⊕ p[i+s]`，`fence`；最后每元素一个 item 发布 `p[0]`。n 不是 2 的幂时退回串行。合并链从 n 步降到 log₂ n 步，4 行 × 256 item 的 kernel 从 barrier stall 61%、1.09 ms 到 **0.61 ms**。
  `lower_for` 会把 For 的 exit continuation 的体复制进循环结构，所以内层 pair 循环的 exit 必须是自己的 continuation 再调 yield，不能直接用外层 item 循环的 yield（否则 `boundTreeDone` 的变量泄漏成自由变量）。
- 结果：`softmax_resident.mim`（group `(1, 4, 1024)` promote x，item `(1, 4, 4)`，member 是 `merge_with` 的 online fold）：**1 kernel，0.611 ms，DRAM 100 MB（读一遍写一遍）、89% 峰值带宽**，与 PyTorch 的 0.619 ms 持平。shared 24.6 KB/block（16 KB 行 + 8 KB partial），2 block/SM。8 行 × 128 item 的变体 40 KB shared 只剩 1 block/SM，1.55 ms。
  行求和 0.286 ms 不变（已是带宽），非常驻 online 单 kernel 0.911 ms。

### 7.2 测量与预测的对照

| 里程碑 | 预测 | 实测（1024³ SGEMM，GTX 1650 SUPER） |
|---|---|---|
| G1 手写 | 5–10% cuBLAS | 36.5%（4×4 寄存器 tile 把每个 FMA 的 global 读压到 0.5 次） |
| G1 生成 | — | 36.7%，与手写持平 |
| G2 手写 | 40–60% | 59% |
| G2 生成 | — | 26%（2.60 ms）；同结构的 CUDA C++（累加器在 shared）为 1.76 ms，经典寄存器累加 CUDA 为 1.47 ms |
| G3 生成（`group → item → red`，寄存器累加） | — | 41%（1.66 ms）；经典寄存器累加 CUDA 为 1.47 ms（46%） |
| G3 生成 + §7.1d（barrier 对、计数协作循环、A 面板按连续维填充 + padding） | — | 78%（0.868 ms），超过经典 CUDA 版；瓶颈与其相同（MIO 节流 43%，FMA 管线 52%） |
| G3 生成 + §7.1e（float4 填充 + 旋转 swizzle） | — | 82%（0.830 ms）；shared bank conflict 为 0，MIO 节流 27% |
| G3 生成 128×128/8×8 + §7.1f（消费者感知布局 + 显式向量读） | — | 88%（0.770 ms），与 SGEMM_CUDA K7 持平 |
| G3 生成 128×128/8×16/128 线程 + §7.1g（寄存器预算 256，LLVM 全展开） | — | **94%（0.718 ms）**；K10（warptiling）0.679 ms 是这块卡的 cuBLAS 水平 |
| 行求和 12×1024×1024，group 8 行 × 256 item 的 R8 | — | 0.291 ms，PyTorch 0.286 ms；E0 一线程一行 2.61 ms |
| 三趟 softmax（R8 + map 融合） | — | 1.26 ms，PyTorch 单 kernel 0.62 ms；E0 四 kernel 6.5 ms |
| 单 kernel 融合 softmax（§7.1i，两个归约 member 拆到 item） | — | 1.20 ms，DRAM 90% 峰值（读三遍） |
| 单 kernel 融合 online softmax（§7.1i，`merge_with` 的 (m, l) member） | — | 0.92 ms（读两遍）；PyTorch 0.62 ms（读一遍） |
| 常驻单 kernel online softmax（§7.1j，group 级 promote + 树形 combine） | — | **0.61 ms**，DRAM 100 MB、89% 峰值，与 PyTorch 持平 |

G1 生成要靠 vec 级 `vdim=0` 触发小 blocked leaf 的 SSA 累加器（tile constructs 文档 §6.1），否则 leaf 是 16 个独立 dot product，退回到 E0 的水平。

### 7.3 下一步

1. 到 K10 还差 6%：leaf 的单元回调在 Mim 层内联、让 lane 和累加器下标成为字面量（§7.1g），摆脱对 LLVM 展开阈值的依赖；指令开销（i64 地址运算与布局的移位掩码）可试 32 位索引；`Check::alpha` 对大段直线代码的递归深度。
2. softmax 单 kernel 常驻版已与 PyTorch 持平（§7.1j）。剩下：group 级 promote 的填充还是标量（`vector_width` 的对齐证明没通过；float4 填充会再省一些指令），树形 combine 的前几级可换 warp shuffle 省 barrier；`split_over_items` 只处理单归约循环、K 被 item 数整除的情形；shared 预算没有模型（8 行配置的 40 KB 把占用率压到 1 block/SM）。
   访存侧已收成 `vectorize` 视图（`buffer.vectorize`/`tensor.as_vectors`）加一个消费者侧的 map 标注 `btensor.vectorize`；后者仍是标注而非类型，因为 combiner 要的是标量 lane。
   `staged`/`chained` 目前只由 `lower_map_reduce` 发起，其他消费者（`tiled` leaf、tile_fold 累加器初值）仍用前后各一次 barrier 的 `cooperative`。
   `panel_layout` 只对字面量面板形状生效，swizzle 还要求行长是 2 的幂；`contiguous_dim` 要求读 map 把某一维原样送到源的最后一维；向量填充只在绑定区域内启用。
2. 每次 extern 调用重建 context 与模块（端到端 140 ms 中 kernel 只占 2 ms）：§2.7 的数据常驻。
3. `default_tile_config_gpu`（§2.8）与 TileGroup 对接，使 `tensor:tile-fuse=2` 不带显式 levels 也能走 GPU。
4. 符号形状：绑定层目前要求字面量 extent。

## 8. 多级 tiling 方案总览（2026-11-08 现状）

入口是 `tile_req (root, members, levels, control)`：`root` 是一个 `map_reduce_post`（类型 `Tensor r s T`），`members` 是允许融进 tile 的 producer，`levels: TileLevels (n, T)` 是一条由外到内的 level 链，`n` 是 root 的循环数，`T` 是它的累加器类型。
TileFuse 逐级实现，每一级嵌在前一级 tile 的 body 里，所以"多级"就是 region 的嵌套；LowerToMem 和 gpu 插件再把 region 落到硬件。

### 8.1 levels 是类型化的链

链的类型名就是作用域：`TileLevels` 是串行代码，`GroupLevels` 是 work-group 内部，`ItemLevels` 是 work-item 内部。三个类型互相递归，每个构造子只提供该作用域允许的 level，并决定下一段链的作用域：

| 作用域 | 构造子（函数形式） | 含义 | 下一段 |
|---|---|---|---|
| `TileLevels` | `serial_par (sizes, tail, rest)` | 串行并行级 | `TileLevels` |
| | `serial_group (sizes, tail, promote, rest)` | 绑定到 work-group 的并行级，`promote` 位把输入整块放进 shared | `GroupLevels` |
| | `serial_red (sizes, tail, promote, rest)` | 顺序归约级，`promote` 位按块放面板 | `TileLevels` |
| | `serial_vec (sizes, rest)` | 向量级 | `TileLevels` |
| `GroupLevels` | `group_par`、`group_red`、`group_vec` | 同上，但在 group 内 | `GroupLevels` |
| | `group_item (sizes, tail, rest)` | 绑定到 work-item 的并行级 | `ItemLevels` |
| | `group_red_items (sizes, tail, rest)` | 拆到 work-item 的归约级，merge 从 combiner `op (acc, e)` 推出 | `GroupLevels` |
| | `group_red_merge (sizes, tail, merge, rest)` | 同上，显式给 `merge: [T, T] → T` | `GroupLevels` |
| `ItemLevels` | `item_par`、`item_red`、`item_vec` | item 内的级 | `ItemLevels` |
| 任一 | `serial_done ()`、`group_done ()`、`item_done ()` | 链尾 | |

`sizes: «n; Nat»`（0 表示该维不切）、`tail`（0 cap、1 peel）、`promote`（位掩码）仍是 `Nat`；`merge` 的类型由 root 的 `T` 决定。
于是以前 TileFuse 用 `fail("…")` 检查的那组约束全部成为类型错误：work-item 级必须在 work-group 级里、拆到 item 的归约级必须直接在 group 级下、work-group 级不能嵌在 work-item 级里、只有归约级和 work-group 级能 promote、只有拆到 item 的归约级能带 merge、merge 的类型要匹配累加器（`lit/tensor/tile_levels_scope.mim`、`tile_levels_merge.mim`）。
报错落在 `tile_req` 的实参上，note 里只给 nominal 实例名（`TileLevels_280567` 对 `TileLevels_280501`），要让它打印成 `TileLevels (2, F32)` 还得改 dump。
留在 C++ 里的检查只剩类型表达不了的：tile 大小是字面量、链的 `n` 等于 root 的循环数、拆到 item 的 K 被 item 数整除、树形合并要求 item 数是 2 的幂。
`default_tile_config` 也返回这条链；两种形状的配置共用一条链、在各级的 sizes 上做选择，因为 `(a, b)#cond` 对两个 nominal 实例会得到 union 类型。
TileFuse 内部仍用 `LevelSpec {kind, sizes, tail, promote, bind, merge}`，`tile_levels` 沿着 `Name`（newtype 引入）→ `Inj`（变体 case）→ payload 的最后一个分量走完整条链，作用域在遍历时跟着构造子切换。

### 8.2 每一级对应的 IR 与硬件变化

| 级 | TileFuse 产出 | GPU 上的落地 |
|---|---|---|
| `serial_group ((1, 64, 64, 0), 1, 0, …)` | 一个 `tiled` region，body 拿到 `(off, ext)` 和 inputs，members 按 tile 融进去 | `gpu.outline` 把 tile 循环变成 grid、body 变成 kernel；body 里的 buffer 都在 shared；`promote` 位把该输入的 box 用 k = 0 的 `stage` 协作填进 shared（§7.1j）；融进来的归约 member 按下一级的 item 数拆成 R8 fold，后续读改读这份共享 tile（§7.1i） |
| `group_item ((1, 4, 4, 0), 1, …)` | 再一层 `tiled`，切出每个 item 的子 tile | item 循环改写成 `item_id`，比 item 数短的循环加守卫；私有数据在寄存器或 local |
| `item_red ((0, 0, 0, 16), 1, 3, …)` | `tile_fold`：按块顺序折叠归约循环，step 原地更新累加器 | 顺序 k 循环；`promote` 位让每块的输入面板用向量填充进 shared，面板布局按消费者的读模式选 padding 或 swizzle（§7.1e、§7.1f），前后 fence |
| `group_red_merge ((0, 0, 4), 1, merge, …)` | step 体包上 `bind 2`（和 `merge_with`） | R8（§7.1h）：每个 item 折自己的块进私有 tile → shared partials → 树形合并 → 每元素一个 item 发布 |
| `item_vec ((1, 1, 0, 0), …)` | leaf 变成直线代码 | 累加器进 SSA 寄存器（`btensor:registers` 预算，§7.1g），读按 lane 向量化 |

贯穿所有级的机制：

- 融合：每级只融 `members`（`default_control`），其他 producer 成为 region 输入；输入里的 slice 保持为视图而不是拷贝。
- 内存：group 级的值是 uniform 的，分配在 shared、填充是协作的；item 级私有；`Heap2Stack` 把循环 exit 处的分配转成 slot。
- 对齐与向量化：`buffer.vectorize`/`tensor.as_vectors` 视图加 `mem.align` 声明，ll 把对齐带到每条 load/store。

目前的限制：

- tile 大小必须是字面量；归约 member 拆到 item 要求只有一个归约循环且 K 被 item 数整除；树形合并要求 item 数是 2 的幂。
- shared 预算没有模型（8 行 softmax 的 40 KB 把占用率压到 1 block/SM）。
- group 级 promote 的填充还是标量读。
- 执行映射（grid/block、barrier、host 与 device 的内存 token）还没有类型：同一函数里两个 `tile_req` kernel 首尾相接时，host glue 会在 `host_malloc2gpualloc` 报 `core.M 1` 对 `core.M 0` 的类型冲突，attention 的 QKᵀ → softmax → PV 三 kernel 版本因此还编不过；它与 promote 无关（`bench/gpu_fusion/attn_tiled.mim` 的任意两 kernel 组合都触发），而 `softmax_r8.mim` 的三个 kernel 又没事，原因待查。
