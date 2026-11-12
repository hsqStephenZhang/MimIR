# GPU 多级 tiling 实验记录

机器：rusty，GTX 1650 SUPER（TU116，sm_75，4 GB），驱动 595.71，CUDA `/opt/cuda`，PyTorch 2.11.0+cu130。
FP32 峰值约 4.4 TFLOP/s，显存带宽约 192 GB/s。
所有 GEMM 为 `C[1024,1024] = A[1024,1024]·B[1024,1024]`，F32，行主序。
设计与预测见 `docs/gpu-tiling-design.zh-CN.md`；本文只记录测量。

## 基线：PyTorch CUDA（cuBLAS）

`bench/gpu_tiling/torch_baseline.py`，CUDA event 计时，50 次取中位数。

| n | 中位数 ms | GFLOP/s |
|---|---|---|
| 256 | 0.035 | 972 |
| 512 | 0.169 | 1591 |
| 1024 | 0.679 | 3165 |
| 2048 | 7.643 | 2248 |

## E0：现有 `gpu.lower_btensor_map_reduce`（一线程一输出，k 串行）

`bench/gpu_tiling/gemm_mr.mim`（`product_2d`）经 `-p opt -p gpu -p nvptx`，`gemm_driver.cpp` 驱动。
grid = 1024 块 × 1024 线程，每线程 k 串行 1024 次，每个 FMA 两次 global 读。

| 项 | 值 |
|---|---|
| kernel（nsys 中位数，device IR 未经 `opt`） | 17.77 ms，123 GFLOP/s，cuBLAS 的 3.9% |
| kernel（cherry-pick `fix/nvptx-llvm-opt` 后，`opt -O2`） | 12.17 ms，176 GFLOP/s，cuBLAS 的 5.6% |
| H2D 拷贝（2 × 4 MB） | 2 × 0.82 ms |
| D2H 拷贝（4 MB） | 0.97 ms |
| 端到端每次调用 | 155 ms（含每次调用的 `cuCtxCreate` / `cuModuleLoadFatBinary`） |
| ptxas | 16 寄存器（未优化）/ 61 寄存器（优化后），无 spill |

设计文档对 G1 的预测是 cuBLAS 的 5–10%；E0 没有 block 级 tile，落在预测下方。
本分支的 device IR 原本不经 `opt`（只有用到 libdevice 时才跑），内层循环每次迭代重做 `udiv`/`urem`；已 cherry-pick `fix/nvptx-llvm-opt`（`26762b155c`），之后的数字都带 `opt -O2`。
每次 extern 调用重新建上下文、装载模块，占端到端时间的 95%，对应设计 §2.7。

## G1：手写 kernel，block 64×64 × thread 4×4，k 串行，无 shared

`bench/gpu_tiling/gemm_g1.mim`：grid 256 块 × 256 线程，线程块 16×16，每线程 4×4 寄存器累加器（`«16; F32»` pack），每步 k 读 4 个 A、4 个 B。
这是设计 §2.2 四级配置去掉 L1（promote）后的形状，即 `L0 par (64,64) bind=1; L2 par (4,4) bind=2; L3 par (1,1) bind=0; k 串行`。
`gemm_hand_driver.cpp` 驱动，256 个采样点与 CPU 双精度参考比较。

| 项 | 值 |
|---|---|
| kernel（nsys 中位数） | 1.857 ms，1156 GFLOP/s，cuBLAS 的 36.5% |
| ptxas | 46 寄存器，无 spill |

设计对 G1 的预测是 5–10%，但那是按每个 FMA 两次 global 读估的；4×4 寄存器 tile 把每个 FMA 的 global 读降到 0.5 次，L1/L2 命中率高，实际远高于预测。
写 kernel 时发现的后端限制：`cc#i#j` 形式索引二维数组 `«4, 4; F32»` 时 ll 后端发出非法的 `getelementptr ... [2 x i2]`，改用一维 `«16; F32»` 绕过。

## G2：手写 kernel，加 shared 分块（BK=16）、协作拷贝、同步

`bench/gpu_tiling/gemm_g2.mim`：设计 §2.2 的完整四级形状。
每个 block 两块静态 shared（`mem.slot (…, 3)`，A 转置存为 `[16][64]`，B 为 `[16][64]`，共 8 KB），256 线程每人拷 4 个 A、4 个 B（A 按行 4 线程覆盖 16 个连续 k，B 按行 16 线程覆盖 64 个连续列，global 读合并），`gpu.sync_work_items` 前后各一次。

| 项 | 值 |
|---|---|
| kernel（nsys 中位数） | 1.151 ms，1866 GFLOP/s，cuBLAS 的 59% |
| ptxas | 64 寄存器，1 barrier，8192 B smem，无 spill |

设计对 G2 的预测是 40–60%，命中区间上沿。
发现并修了一个 tiling 走 GPU 必踩的坑：`affine.lower_for` 只允许累加器里有一个 `mem.M`，kernel 内同时携带 `GlobalM` 与 `SharedM` 的循环会报 `cannot coerce to mem.M 3`。
现在 `lower_for` 为累加器里每个 `mem.M` 分量在 head/body/exit 各建一个 mem 参数（`src/mim/plug/affine/phase/lower_for.cpp`）。
这对应设计 §5 "内存链" 风险。

## G1（生成）：`tile_req` 经 TileFuse → LowerToMem → `gpu.outline`

`bench/gpu_tiling/gemm_tiled.mim`，请求为
`((0, (64, 64, 0), 0, 0, 1), (0, (4, 4, 0), 0, 0, 2), (2, (1, 0, 0), 1, 0, 0))`：
L0 par 64×64 绑定 work-group（cap 尾部），L1 par 4×4 绑定 work-item，L2 vec `vdim=0` 让 leaf 用 `«16; F32»` SSA 累加器（tile constructs 文档 §6.1 的小 blocked leaf）。
命令：`mim -p opt -p gpu -X tensor:tile-fuse=1 gemm_tiled.mim -p nvptx`，驱动同 E0。

| 配置 | kernel（nsys 中位数） | 相对 cuBLAS | 备注 |
|---|---|---|---|
| 两级，无 vec 级 | 10.24 ms | 6.6% | 16 个标量 dot product，无寄存器复用，等同 E0 |
| 加 vec `vdim=1` | 8.92 ms | 7.6% | 累加器是 `«4; F32»` 内存 slot，i{k{j}} 嵌套 |
| 加 vec `vdim=0` | 3.53 ms | 19% | 16 个寄存器累加器，但输入指针走 generic load（实际靠 HMM 读主机内存） |
| 同上，捕获修复后 | **1.852 ms** | **36.7%** | 与手写 G1（1.857 ms）持平；`ld.global` ×8，44 寄存器 |

生成 kernel 与手写 G1 的 PTX 内层循环一致：每步 k 读 4 个 A、4 个 B，16 条 FMA。
过程中修掉的问题：
- `gpu.outline` 捕获的 region 输入是一个 `«2; Ptr»` 元组，按整体传给 kernel 后指针仍指向主机内存（驱动 595 + Linux 6.12 的 HMM 让 kernel 能直接读主机页，所以结果居然正确，只是慢）；现在元组按分量拆开，每个指针单独拷贝，整体元组再由 kernel 参数重建。
- nvptx 把地址空间 0 的 `mem.slot` 发成模块级全局变量，所有线程共享；改为与 ll 一样在 kernel 入口 `alloca`（shared/const 槽仍是全局变量）。

## G2（生成）：加 promote 的归约级，面板进 shared，协作填充

`bench/gpu_tiling/gemm_tiled_g2.mim`，请求为
`((0, (64, 64, 0), 0, 0, 1), (1, (0, 0, 16), 1, 3, 0), (0, (4, 4, 0), 0, 0, 2), (2, (1, 0, 0), 1, 0, 0))`；
`gemm_tiled_g2s.mim` 把 group tile 改成 32×32（64 线程）。

| 配置 | kernel（nsys 中位数） | 相对 cuBLAS | smem | 备注 |
|---|---|---|---|---|
| 32×32，fold 非 in-place | 6.77 ms | 10% | 20 KB | `linear_step` 没剥 `tensor.bind` 注解，每个 k-block 整块拷贝累加器 |
| 32×32，修复后 | 5.71 ms | 12% | 16 KB | |
| 64×64，修复后 | 5.76 ms | 12% | 48 KB | 每 SM 只能驻留 1 个 block |

kernel 结构已经是目标形状：每个 k-block 两次协作填充（每线程 4 个元素，约 11 条指令一个）、4 次 `bar.sync`、
micro-kernel 每步 k 8 次 `ld.shared` + 16 条 FMA，累加器在寄存器里跨 16 步 k，k-block 末尾 16 次 `st.shared`。
但 shared 用量过大压低了 occupancy，与手写 G2（8 KB，1.15 ms）差 5 倍：

- fold 的累加器（64×64，16 KB）放在 shared 里；手写版放寄存器、生成版应直接写目的地 C（`lower_tile_fold` 还不接受 destination，因为 in-place 的累加器 tensor 与 buffer 形状必须一致）。
- fold 初值的 `splat` 被 `buffer.lit` 物化成一块 16 KB 的零缓冲加非协作的嵌套清零循环，第一块本应直接用标量种子。
- 第一块（`foldFirst`）单独降低，面板再分配一份（多 8 KB）。

过程中修掉的问题：`affine.lower_for` 的多内存累加器；`mem.add_mem` 在 resolve 模式下把 lam 内所有 `mem.M` 操作数（含 kernel 的 Shared/Const/Local 记号）替换成当前链、把 `gpu.launch` 可达的所有 lam 当作 pinned 回调而不解析其中的 `mem.fresh`；`tensor.bind` 注解让 `linear_step` 误判 in-place；K 整除时仍 peel 尾块。

### G2 生成：后续修复与公平参照（同日）

修复项（均在 `bound_level_ == 1` 即 work-group 内生效，CPU 路径不变）：
- fold 累加器初值：标量 splat 直接协作填充，不再经 `buffer.lit` 物化（`lower_tile_fold` 的分支顺序）；
- 不再为首块单独生成 `foldFirst`（省一份面板、一份代码）；
- 面板布局：k 放最外、线程切分轴的 panel 索引次之、4 元素最内（`(k, ceil(n/4), 4)`），相邻线程读相邻 panel，无 bank conflict；
  kernel 内的缓冲外维也取上界（`bounded_shape(…, outer)`），shared slot 必须是静态的；
- 绑定层改用 peel 尾部（整除时无尾块），避免 cap 带来的条件形状。

| 配置（64×64 block，BK=16，4×4/线程，标量访存，无 swizzle） | kernel | 相对 cuBLAS | smem | ncu bank conflict（ld） | occupancy |
|---|---|---|---|---|---|
| 生成 G2，修复前 | 5.03 ms | 13.5% | 24 KB | 5880 万 | 49% |
| **生成 G2，修复后** | **2.60 ms** | **26%** | 24 KB | 8 万 | 49% |
| CUDA C++ 经典写法，累加器在寄存器（`gemm_smem.cu`，nvcc -O3） | 1.47 ms | 46% | 8 KB | — | — |
| CUDA C++，累加器放 shared（与生成 G2 同结构） | 1.76 ms | 39% | 24 KB | — | — |
| 手写 Mim G2（A 转置存放，寄存器累加） | 1.15 ms | 59% | 8 KB | 0 | 91% |

公平比较：生成 G2 与"累加器放 shared"的 CUDA 版本结构相同（每 k-block 两次协作填充、累加器从 shared 取出再写回），差距 1.48×，
来自每 k-block 4 次 barrier（CUDA 版 2 次）、填充循环的 64 位地址计算和累加器的 shared 读写；与标准寄存器累加版差 1.77×。
目标是验证 tiling 的能力与可行性，这个量级可以接受；float4 向量化读取、swizzle、寄存器累加器属于后续性能工作。

## G3（生成）：`group → item → red(promote)`，寄存器累加器

`bench/gpu_tiling/gemm_tiled_g3.mim`，请求为
`((0, (64, 64, 0), 1, 0, 1), (0, (4, 4, 0), 1, 0, 2), (1, (0, 0, 16), 1, 3, 0), (2, (1, 0, 0), 1, 0, 0))`：
归约级放在 work-item 级**里面**，每个线程的 4×4 累加器跨整个 K 留在寄存器；promote 的面板按整个 work-group 的 tile 切（"group-wide"），
在 item 体内作为组内一致的值由所有线程协作填充到 shared。

| 配置（64×64 block，BK=16，4×4/线程，标量访存，无 swizzle） | kernel | 相对 cuBLAS | smem | 寄存器 | bank conflict | occupancy | 指令数 |
|---|---|---|---|---|---|---|---|
| **生成 G3** | **1.663 ms** | **41%** | 8 KB | 46 | 0 | 94% | 6150 万 |
| CUDA C++ 经典寄存器累加（目标） | 1.469 ms | 46% | 8 KB | — | — | 91% | 4100 万 |
| CUDA C++ 累加器放 shared | 1.758 ms | 39% | 24 KB | — | — | — | — |
| 生成 G2（`group → red → item`） | 2.594 ms | 26% | 24 KB | 39 | 8 万 | 49% | — |
| 手写 Mim G2 | 1.15 ms | 59% | 8 KB | — | 0 | 91% | — |

生成 G3 与目标差 13%：每 k-block 4 次 `bar.sync`（CUDA 2 次；两次填充之间的 barrier 多余，但填充的内存链在 `gpu.outline` 时还是
`mem.fresh` 占位、要到 `mem.resolve_fresh` 才接起来，外提时看不到相邻关系），以及 i64 地址运算（micro-kernel 每步 36 条指令对 CUDA 约 28 条）。

为此做的改动：
- TileFuse：`par → par → red` 原本因 affine 线性形式只容一个符号常量（两层 tile 偏移 `g + i` 再减 `g`、减 `i`）判 preimage 不精确而失败；
  `lin_add` 现在把两个符号常量合成 `core.nat.add`，`lin_sub` 做抵消。
- TileFuse：记录 group/item 两级的 region 变量与 tile 尺寸；item 级内的 promote 从输入 map 找出归约维，用 group 的 `off/ext`、当前 k 块
  和整段其余维从零构造切片（直接改原切片的 start/size 会带上 item 的越界守卫，变成 item 相关），map 用 item 相对 group 的偏移平移。
- LowerToMem：item 级内不含 item origin 变量的 `generate`/`tile_fold`/`splat` 视为"组内一致"，其填充标 `cooperative`、其分配内存标
  `gpu.shared_scope`；item 私有的累加器走普通分配（`mem.alloc` → heap2stack → alloca → SROA 进寄存器）。
- `gpu.outline`：只把带 `shared_scope` 的分配变 shared slot，item 体内的 cooperative 循环同样跨步 + barrier。

## G3 续：barrier、填充循环与 A 面板的合并访存（同日）

从 1.663 ms 到目标 1.469 ms 的差距逐项排查（都以 nsys 中位数计，kernel 结构见 `out/g3*.sass`）：

| 改动 | kernel | 说明 |
|---|---|---|
| 起点 | 1.663 ms | 每 k-block 4 次 `bar.sync` |
| `remove_double_syncs` 看穿 `generic_mem`/`global_mem`/`slot_shared`，识别 kernel 入口的 shared 记号 | 1.652 ms | 3 次 barrier |
| 填充标 `staged`/`chained`，消费者发 `gpu.fence` | 1.652 ms | 2 次 barrier，与 CUDA 相同；几乎不省时间 |
| 协作循环在整除时改为计数循环（`j < 4`，`i = item + 256·j`） | 1.647 ms | SASS 从 256 条降到 128 条，8 个 `LDG` 批量发出；仍不省时间 |
| LLVM 把 k 循环完全展开（实验，`-X nvptx:Xopt`） | 1.648 ms | 256 条 FFMA 直线代码，仍不省时间；同一实验对 CUDA 版关掉展开则从 1.47 变 2.04 ms |
| **A 面板填充按源的连续维迭代（k 最内）并把面板中维 16 填到 17** | **0.868 ms** | L1 tag 访问从 1730 万降到 157 万，kernel 快 1.9 倍 |

诊断依据（ncu）：指令数、shared 读写 wavefront 数与 CUDA 版完全一致（LDS.128 各 419 万条、1258 万 wavefront），
差异只剩 `l1tex__t_set_accesses`（global 读触碰的 cache line 数）：生成版 1730 万，CUDA 版 629 万。
原因是 A 面板为了让 micro-kernel 无 bank conflict 而 k 在外（`(k, n/4, 4)`），协作填充却按面板的行主序线性分配，
一个 warp 的 32 个线程读 A 的 32 行（间隔 4 KB），每个请求 32 条 cache line；barrier 又让所有 warp 等最慢的填充。

修法：`lower_stage` 用读 map 求出哪一面板维对应源的最后一维（`contiguous_dim`），`lower_generate` 让该维跑最内（`linear_loop` 的 `inner`），
并为这种转置填充的 shared 面板加 padding（`padded_panel`：在保持行 16 字节对齐的前提下选 0–7 个元素，使 16 行的写入落到最多的 bank 上；
这里 64 → 68，2 路冲突）。CUDA 经典写法本身每线程读 4 个连续 k（一个 warp 8 行×64 B，16 个 sector），生成版现在是 2 行×64 B。

| 1024³ SGEMM，64×64 block，BK=16，4×4/线程，标量访存 | kernel | 相对 cuBLAS | smem | 寄存器 | st bank conflict | L1 tag 访问 |
|---|---|---|---|---|---|---|
| **生成 G3（本节之后）** | **0.868 ms** | **78%** | 8.25 KB | 62 | 63 万 | 157 万 |
| CUDA 经典寄存器累加（`gemm_smem.cu`） | 1.469 ms | 46% | 8 KB | 62 | 157 万 | 629 万 |
| 手写 Mim G2 | 1.15 ms | 59% | 8 KB | — | 0 | — |
| 生成 G3（本节之前） | 1.663 ms | 41% | 8 KB | 46 | 0 | 1730 万 |
| 生成 G2（`group → red → item`，累加器在 shared，顺带受益） | 1.956 ms | 35% | 8.4 KB | — | — | — |

生成版现在的瓶颈与 CUDA 版相同：MIO（shared 读）节流 43%，FMA 管线 52% 忙；再往上要 float4 读、更大的 thread tile 或 ldmatrix 类手段。

## float4 与 swizzle（同日）

| 改动（64×64 block，BK=16，4×4/线程） | kernel | st bank conflict | ld bank conflict | MIO 节流 | 寄存器 |
|---|---|---|---|---|---|
| 上节终点（标量填充 + padding 68） | 0.868 ms | 60 万 | 1.4 万 | 43% | 62 |
| 向量填充（`buffer.vectorize` 视图上的 read/write）：`ld.global.v4` + B 面板 `st.shared.v4`、A 面板 4 次转置标量写 | 0.860 ms | 60 万 | 1.4 万 | 42% | 45 |
| **面板布局模型选旋转 swizzle（不再 padding，面板 16×64）** | **0.830 ms** | **0** | **0** | 27% | 43 |

- 向量填充本身省得少：填充已经合并访存，只是指令数从 128 降到 120 条；SASS 里 `LDG.E.128` 2 条、`STS.128` 1 条、`STS` 4 条。
- padding 在 16 字节对齐约束下对分块填充的 A 转置写（每条写指令覆盖 4 行×8 列）只能做到 2 路冲突；旋转 swizzle（16 字节块按 `(row >> 2) % 4` 旋转）无冲突。
  代价是消费者读地址多了移位/掩码：指令数 5310 万 → 6380 万，但 MIO 节流从 42% 降到 27%。
- 128×128 block，BK=8，8×8/线程（`gemm_tiled_g3_128.mim`）：0.91 ms。写冲突归零，但 B 读 419 万次 2 路冲突（每线程 8 列连续，一个 quarter-warp 的 LDS.128 跨 256 B），寄存器 128；
  经典 float4 kernel 把每线程的 8 列拆成相距 64 的两段 4 列，这要 item 级支持分段 tile，是下一步。
- G2（`gemm_tiled_g2s`）顺带从 1.956 降到 1.584 ms，G1 不变（1.853 ms，无面板）。

| 1024³ SGEMM | kernel | 相对 cuBLAS |
|---|---|---|
| **生成 G3 + float4 + swizzle** | **0.830 ms** | **82%** |
| 生成 G3 + float4 | 0.860 ms | 79% |
| 生成 G3（上节） | 0.868 ms | 78% |
| 手写 Mim G2 | 1.15 ms | 59% |
| CUDA 经典寄存器累加（标量访存） | 1.469 ms | 46% |

## 对照 SGEMM_CUDA，消费者感知的面板布局与显式向量读（同日）

`/workspaces/compile/SGEMM_CUDA` 编到 sm_75（`SIZE={1024,2048}`），harness 计时，1024³：

| kernel | 时间 | shared 读 wavefront | 指令数 | 寄存器 | FMA 管线 |
|---|---|---|---|---|---|
| K6 vectorize（128×128，BK=8，8×8） | 0.831 ms | 1049 万（419 万次 2 路冲突） | 3800 万 | 118 | 64% |
| K7 resolve bank conflicts（B 面板线程号最内） | 0.770 ms | 629 万 | 4250 万 | 102 | 71% |
| K9 autotuned | 0.815 ms | — | — | — | — |
| K10 warptiling（128×128，BK=16，warp 64×64，线程 8×16） | 0.679 ms | 526 万 | 3680 万 | 168 | 79% |
| K11 double buffering | 0.749 ms | — | — | — | — |
| harness 的 cuBLAS（`volta_sgemm_64x64_nn`） | 0.975 ms | 425 万 | 4070 万 | 126 | 84% |

K10 与 PyTorch 测的 cuBLAS（0.679 ms）持平，是这块卡的上限。

| 生成版 | kernel | ld bank conflict | shared 读 wavefront | 指令数 | 寄存器 | 相对 cuBLAS |
|---|---|---|---|---|---|---|
| 64×64，BK=16，4×4，swizzle（上节） | 0.830 ms | 0 | 1258 万 | 6380 万 | 43 | 82% |
| 128×128，BK=16，8×8，布局模型只看填充 | 0.91 ms | 419 万 | 1049 万 | 4770 万 | 128 | 75% |
| 同上，模型加消费者读，选 K7 式 `(k, n_lo, n_hi)` + padding 20 | 0.806 ms | 0 | 629 万 | 5180 万 | 114 | 84% |
| **同上，`btensor.vectorize` 显式向量读，模型选 `(h, k·n_hi, 4)`** | **0.775 ms** | **0** | **629 万** | 4850 万 | 110 | **88%** |
| 64×64 配置 + 显式向量读 | 0.839 ms | 0 | 1258 万 | 5960 万 | 44 | 81% |
| 128×128，8×16（128 线程） | 1.88 s | — | — | 16 亿 | 55 | 累加器落到 local memory（LLVM 未展开） |

要点：
- 布局搜索空间 = 最后一维按 16 字节切块 × 维置换 × padding × 旋转 swizzle；代价 = 填充写 + 消费者读的（发射槽 + wavefront）。
  没有发射槽项时 K7 式排布（8 次标量读）与 `(h, k·n_hi, 4)`（2 次 LDS.128）wavefront 相同，模型选不出后者。
- 消费者的向量读必须显式（`btensor.vectorize` 标注 → `buffer.vectorize` 视图上读一块 + lane）：ptxas 对 split 布局只合并出标量 LDS，模型按 128 位算就会误判（B 读 2 路冲突，0.86 ms）。
- G3 的 64 配置和 G2 不受影响（正确性 OK）。

## 8×16 thread tile：寄存器预算与展开（同日）

| 配置（128×128，BK=16） | 编译选项 | kernel | 寄存器 | 每 k 步 LDS.128 | 相对 cuBLAS |
|---|---|---|---|---|---|
| 8×16，128 线程（`gemm_tiled_g3_128w.mim`），默认 | — | 1.9 s | 55 | — | 累加器 128 元素超过 leaf 的 64 单元寄存器预算，退回 blocked 嵌套，临时缓冲区成了 device `malloc` |
| 同上 + `-X btensor:registers=256` | | 3.1 ms | 200 | 6 | 累加器进 SSA，但 lane 下标非字面量，LLVM 默认阈值下留下 36 LDL/64 STL |
| **同上 + `-X nvptx:Xopt="-passes=\"default<O2>,nvvm-reflect\" -unroll-threshold=20000"`** | | **0.718 ms** | 226 | 6（k 展开 16×） | **94%** |
| 8×8，256 线程（`gemm_tiled_g3_128.mim`） | 默认 | 0.770 ms | 111 | 4 | 88% |
| 64×64，4×4 | 默认 | 0.843 ms | 46 | 2 | 81% |
| SGEMM_CUDA K10 | nvcc | 0.679 ms | 168 | 6 | 100% |

- 展开阈值对 8×8/64 配置反而有害（0.770 → 0.827，0.843 → 0.836），所以不改 nvptx 的默认，只作为该配置的旋钮。
- vec 级 `(2, (1, 0, 16), …)` 的 Mim 层 k 展开：64 配置 0.836 ms（略慢），8×8/8×16 编译时 `Check::alpha` 递归爆栈，`ulimit -s unlimited` 后 8×8 为 0.824 ms。

## A100：cp.async 流水与 TF32 tensor core（2026-11-09）

机器：A100-PCIE-40GB（sm_80），驱动 580，CUDA 12.9，LLVM 20（`opt` 需 `-X nvptx:Xopt='-passes="default<O2>,function(nvvm-reflect)"'`）。
计时改用 `krun.sh` + `kbench.cpp`：直接加载 cubin，在同一个 context 里背靠背 launch 200 次取中位数。
生成的 host 代码每次调用都重建 context，GPU 时钟在两次 kernel 之间回落到 210 MHz，nsys 单次测量不可靠。
基线（cuBLAS 12.9 / PyTorch 2.14 cu126，CUDA event 中位数）：

| n | cuBLAS FP32 | PyTorch FP32 | cuBLAS TF32 |
|---|---:|---:|---:|
| 1024 | 0.244 ms | 0.219 ms | 0.057 ms |
| 2048 | 1.242 ms | 1.528 ms | 0.269 ms |
| 4096 | 7.256 ms | 7.907 ms | 1.584 ms |

FP32（SIMT）生成 kernel：

| 配置 | 1024 | 4096 | 相对 cuBLAS FP32（4096） |
|---|---:|---:|---:|
| G3 64×64，4×4 | 0.421 ms | 11.07 ms | 66% |
| G3 128×128，8×8 | 0.504 ms | 9.28 ms | 78% |
| 同上，`-X tensor:pipeline=2` | 0.481 ms | 9.30 ms | 78% |

`pipeline=2` 让 k 循环带两套面板，下一块的填充在 sm_80 上发为 `cp.async`（`LDGSTS`），每块一次 barrier。
SIMT kernel 每个 SM 驻留多个 block，访存延迟已被掩盖，瓶颈在 FMA 与 shared 读，所以流水基本不提速，默认关闭。

TF32 tensor core（`group_warp ((64, 32, 0), 1, warp_red ((0, 0, 16), 1, 3, warp_mma (16, 8, 8)))`，128×128 work-group，8 个 warp）：

| 配置 | 1024 | 2048 | 4096 | 相对 cuBLAS TF32（4096） |
|---|---:|---:|---:|---:|
| 生成，无流水 | 0.138 ms | — | — | — |
| **生成，`pipeline=2`** | **0.102 ms** | **0.503 ms** | **2.168 ms** | **73%** |
| 手写 Mim（`gemm_tc.py`，同结构） | 0.102 ms | 0.501 ms | 2.146 ms | 74% |

- A 面板强制为 (m, k)、k 最内、行宽 pad 到 20，fragment 由 `ldmatrix.x4` 读；B 面板 (k, n)、pad 8，标量读 b0/b1，无 bank conflict。
- 累加器是每个 lane 的 SSA 值 `«(WM/16)·(WN/8); «4; F32»»`，跨 k 块留在寄存器，最后按 PTX 的 C 布局写回。
- tensor core 上流水是必要的：没有它 1024³ 慢 35%。
- 4096³ 下约为 FP32 cuBLAS 的 3.3 倍；与 cuBLAS TF32 的差距来自两级流水（cuBLAS 用 3–4 级）、B 面板的标量读和 128×128 的固定 tile。

### 多级流水与动态 shared memory（同日）

`-X tensor:pipeline=n` 给每个面板一个 n 个 buffer 的环，提前 n − 1 块填充。
每块的填充是一组 `cp.async`（`commit_group`），每步先 `wait_group (n − 2)` 等最老的一组，再一次 barrier。
kernel 的 shared slot 超过静态上限 48 KiB 时，nvptx 后端把它们放进一个 `extern` 的动态 shared 数组，
host 在 launch 前用 `cuFuncSetAttribute` 把上限提到所需大小。

TF32 tensor core（128×128 work-group，8 个 warp，BK=16），同一时段背靠背测量，4096³ 三次中位数一致：

| 流水 | shared | 1024 | 2048 | 4096 | 相对 cuBLAS TF32（4096） |
|---|---:|---:|---:|---:|---:|
| 2 | 37 KiB | 0.090 ms | 0.477 ms | 1.94 ms | 82% |
| 3 | 56 KiB | 0.093 ms | 0.468 ms | 1.94 ms | 82% |
| **4** | **74 KiB** | 0.092 ms | **0.457 ms** | **1.87 ms（73.5 TFLOP/s）** | **85%** |

BK=8 时每级只有 10 KiB，但每块的 mma 只有一步 k8，流水再深也只有 0.11–0.13 ms（1024³），不如 BK=16。
FP32 SIMT kernel 在 3、4 级下仍然正确，速度与 2 级相当。

## Dequant GEMM（2026-11-12）

`C = A · W'`，`W' = s2f(W) · s`：A 为 F32，W 为 I8，s 为按列的 F32 scale。
驱动 `kbench_dq.cpp`，用 `krun_dq.sh` 运行；PyTorch 基线 `torch_dq.py`。A100，同一时段测量，单位 ms。

| 实现 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| dequant 写进 combiner，SIMT（`gemm_dq_g3`） | 0.445 | 1.56 | 11.6 |
| W' 作为融合成员，TF32 tensor core（`gemm_dqf_tc`），无流水 | 0.160 | 0.561 | 2.30 |
| 同上，`pipeline=2`（三次） | 0.120–0.136 | — | 2.16–2.17 |
| 同上，`pipeline=4`（三次） | 0.121–0.134 | — | 2.18 |
| 对照：纯 TF32 SGEMM（`gemm_tiled_tc`），`pipeline=2`（三次） | 0.090–0.104 | — | 1.99 |
| PyTorch FP32：先 dequant 再 mm | 0.288 | 1.085 | 8.10 |
| PyTorch TF32：先 dequant 再 mm | 0.093 | 0.258 | 1.59 |
| PyTorch TF32：对预先 dequant 的 W 做 mm | 0.064 | 0.214 | 1.42 |

- combiner 写法让每次乘加都重做一次 dequant，并且破坏了矩阵乘的形状，不能用 tensor core。
- 融合成员写法：W' 的面板在填充时从 I8 的 W 和 s 直接计算（`tensor.get_fused`），矩阵乘仍是纯的 `x·y`，可以走 MMA，只比同调度的纯 SGEMM 慢 9–30%。
- 与 PyTorch TF32 的差距（4096 下 1.37×）主要来自 tensor core 主干本身（纯 SGEMM 也慢约 1.25×）。
- 初版记录的 1024 / `pipeline=4` 为 0.075 ms，是一次异常测量；重测已推送的提交也在 0.115–0.128 ms，已更正。
- `gemm_dqf_g3`（SIMT 版的融合写法）结果正确，但很慢（约 16 ms）。原因是 group 级把 s 的列切片复制进 scratch 槽位后，item 累加器的分配落进了 `mslot` 的回调，逃逸分析判它逃逸，累加器因此留在设备堆上。
- 已知问题：W' 若由两层逐元素算子合成（`fuse_tensor` 合并后有两个只按列读的输入，如 s 和 t），放宽后 work-group 级会各复制一份列切片，其中一份留在设备堆上，`mem.free_locals` 为它插入 `free` 时用错了地址空间，编译失败。改动之前这类程序能编译。
