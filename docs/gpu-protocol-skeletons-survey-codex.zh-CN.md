# GPU 同步与搬运协议是否是“小空间”：反向源码与文献审阅

审阅日期：2026-11-17。

## 1. 结论

**判断：部分成立；“常见局部协议可复用”成立，“原语决定基本唯一的正确高效用法”不成立，“编译器不需要构造变换”也不成立。**
这里的判断来自下面逐来源记录及其归并，不把 MimIR 已有组合子的名字当作外部来源的分类依据。

| 被检验的子论断 | 判断与适用范围 | 最直接的证据 |
|---|---|---|
| 规则稠密 GEMM 的局部共享存储协议可以由少量模板复用 | 成立于本次样本；填充、发布、消费、回收及有界在途操作反复出现 | SGEMM_CUDA 的同步循环 [L01]、LeetCUDA 的多级流水 [L05]、cubek 的单阶段与双缓冲 [L12][L13]、CUTLASS 的 pipeline [L32] |
| 同一组硬件原语只有一种正确且高效的结构 | 不成立；同代 Hopper 的 TMA/WGMMA 下，`Pingpong` 和 `Cooperative` 有不同的 consumer 协作拓扑，builder 会按 tile 条件选择 | 两个真实 kernel [L30][L31]，builder 的自动选择 [L29] |
| 少量固定骨架加数值参数足以覆盖完整 kernel | 不成立于本文采用的可检验定义；事件之间的依赖、warp 分工、跨迭代交错和融合后的存活区间也必须进入表示 | Twill §3–§5.4 [P18]、Mojo 的搜索与实际发射 [L09][L10]、FlashAttention-3 §3.2 Algorithm 2 [P20] |
| 编译器主要组合和选择已知协议，并检查义务 | 是有用的工程方向，但并不排除依赖分析、循环变换和调度合成 | TensorIR §3.2、§4 [P13]，Exo §3.3–§5 [P08]，Tawa §III-C–E [P21] |
| `grid → cluster → block → warpgroup → warp → thread` 是通用且必须经过的层级链 | 不能这样外推；它可以作为 NVIDIA 上的可选作用域集合，AMD 等目标需要自己的参与者、存储域和完成机制 | AMD kernel 的 `s_waitcnt`/LDS/MMA 发射 [L10a]、HIP/CUDA/OpenCL scan 分派 [L25]、Blackwell 的 TMEM 与三角色 kernel [L35] |

**建议保留的设计命题是：把常见局部协议做成带契约的可组合库，再用依赖图、布局代数与资源约束生成和检查具体调度。**
这个命题同时解释了 CUTLASS 的 policy/builder、Tawa 的 `aref` 和 Mojo 的 pipeline 为什么有价值，也保留了 Twill 所展示的联合调度合成空间。[L28][L29][P21]（§III）[L09][P18]（§4）
硬件原语可以约束合法事件顺序，但本次证据不能证明硬件设计意图导致“高效用法唯一”，更不能证明所有稠密张量算法都落在某个封闭的五项清单内。[L29][L30][L31][P18]（§5.4）

### 1.1 什么才算“结构”，避免把论断变成恒真句

本文把结构定义为：参与者及其角色、存储对象的所有权、事件之间的先后关系、循环携带依赖、分支和终止条件。
单纯改变 `BM/BN/BK`、环深、向量宽度或参与者数量而不改变上述关系，算参数化；改变哪些角色共同处理一个 tile、哪些事件必须等待另一轮、是否需要跨 CTA 状态轮询，则算结构选择。
允许“参数”直接是一段任意 kernel、任意事件图或任意控制流程序，当然可以把所有程序装进“骨架 + 参数”，但这不再说明设计空间很小。
这个区分是本文的审阅口径，不是某篇论文已经证明的分类定理。
尤其要区分三件事：同一握手协议、同一事件调度、同一数值算法；Tawa 的 `aref` 统一第一件事，Twill 优化第二件事，FlashAttention 的在线 softmax 则还涉及第三件事。[P21]（§III-B）[P18]（§4）[P20]（§3）

### 1.2 证据范围与复查方法

本次按“先读外部 kernel/调度，再记录固定项与变量，再归并，最后读本项目背景并对照”的顺序进行。
下表的数量只统计明确限定的排布维度或已读实现，`≥` 表示样本下界，不能把不同维度的数量相乘当成合法组合数。
对语言和编译器，没有有限命名菜单时记录“开放”，不把 pass 数、模板数量或整个仓库的文件数冒充骨架数。
论文阅读范围是与本题有关的算法、代码、调度、类型规则和限制章节，不声称逐页精读全部附录。
“禁止”表示读到了检查或明确限制，“未实现”表示来源承认的实现边界，“未见”只表示本次没有找到证据。
教学 kernel 的存在不等于本次验证了正确性或峰值性能；本文没有运行构建、测试或 benchmark，性能取舍仅按原作者的代码选择或论文实验解释。
所有本地证据链接指向当前工作树的真实文件和起始行，邻近行共同构成所述代码段。
本项目和其他仓库可能有本地修改，因此以下 HEAD 仅供定位基线，不声称引用内容等于干净提交。

| 本地根目录 | 只读取得的 HEAD |
|---|---|
| `/workspaces/ml-compiler/LeetCUDA` | `c5e985fe4c11` |
| `/workspaces/compile/SGEMM_CUDA` | `5a7dcc513d95` |
| `/workspaces/compile/modular` | `da0db31d422e` |
| `/workspaces/ml-compiler/cubek` | `74921bf41688` |
| `/workspaces/ml-compiler/tilelang` | `0e687c17bb3b` |
| `/workspaces/ml-compiler/triton` | `0ff064ba9587` |
| `/workspaces/ml-compiler/tvm` | `7a48cb5da03d` |
| `/workspaces/ml-compiler/Halide` | `3498eb47a0eb` |
| `/workspaces/ml-compiler/iree` | `7a8b0039c12d` |
| `/workspaces/ml-compiler/futhark` | `796ec9903aa1` |
| `/workspaces/ml-compiler/modern-gpu-programming-for-mlsys` | `3a24144defa6` |
| 本报告所在 MimIR 工作树 | `a10d52b44503` |

CUTLASS 实际读取的是 TVM 内 vendored 的 `3rdparty/cutlass/include/cutlass/`，FlashAttention-3 实际读取的是 PyTorch 内 vendored 的 `third_party/flash-attention/hopper/`，不把二者当作独立取得了版本号的上游最新源码。[L28][L34]
Graphene 与 Async Graphene 的官方摘要已核实，但 ACM 全文请求返回 403，作者页和学术检索返回的入口也未取得正文，因此它们的 kernel、约束、排布数量一律标为“未核实”。[U01][U02]
此前调研文档仅在外部记录之后用于对照，没有把其中关于 Graphene 的进一步断言升级为本次已读原文的结论。[M06]

## 2. 逐来源记录：先保留各自的结构和名字

### 2.1 本地 kernel、pipeline 与调度器

| 来源与已读对象 | 写死的控制结构、事件顺序、同步位置 | 留作参数的内容 | 排布种类、名字及选择依据 | 明确禁止、未实现与证据位置 |
|---|---|---|---|---|
| LeetCUDA：SGEMM 阶梯 | 普通 tiled 版每轮协作写 A/B，CTA 同步，读并累加，再 CTA 同步；`sgemm_async.cu` 所读版本对 B 发 `cp.async`、对 A 经寄存器转置，先发下一块再计算当前块 | `BM/BN/BK/TM/TN/OFFSET`，运行时 M/N/K | 已读两种搬运顺序：同步单面板、异构 A/B 搬运的双缓冲；线程与数据分布也随模板改变 | 所读快速路径存在无逐元素尾部保护的向量访问，不能据 `ceildiv` 推断任意形状安全；源位置 [L04][L04a] |
| LeetCUDA：HGEMM、WMMA、MMA | 普通 WMMA 是 load、CTA barrier、fragment load、`mma_sync`、CTA barrier；stage 版先填 `K_STAGE-1` 组，再按模下标取槽，`wait_group` 与 CTA barrier 后消费；MMA 版把 shared→register 写成 `ldmatrix` 再发 MMA | fragment/tile 形状、warp tiling、A/B padding、`K_STAGE` | 在已读对象内至少两种时间排布：同步、预填与排空的多阶段；WMMA 与显式 MMA 是指令/fragment 表达选择，不能另算一个同步协议 | `hgemm/naive/hgemm_async.cu:89–121` 所读循环实际上用普通寄存器预取，文件名的 async 不是 `cp.async` 证据；尾部与阶段合法性须逐路径核对 [L05][L05a][L05b][L05c] |
| LeetCUDA：TMA/WGMMA GEMM | WG0 的一个线程等 empty、发 A/B TMA、登记字节数；consumer 等 full、发 WGMMA、`wait_group(0)` 后归还 empty；两条独立循环只经槽位握手 | `BM/BN/BK/NUM_THREADS/K_STAGE/BLOCK_SWIZZLE`；消费者数从线程数推导 | 此文件一条专门的 WS 路径；与前述所有 warp 同步推进的 stage kernel 是不同角色结构，不能仅视为 stage 数变化 | 这里每轮 WGMMA 都 wait 到 0，不等于所有高效 WGMMA kernel 都必须如此；K 循环用 `K/BK`，不是通用动态尾部方案 [L06] |
| LeetCUDA：FlashAttention `split_q`、`split_kv` | 都有 QK、行 max/sum、指数、PV 和在线状态更新；`split_kv` 把跨 warp 的 partial max/sum 写 shared 后同步合并，`split_q` 的 warp 分工避免同样的跨 warp 行归约 | head dim、MMA/warp tile、`kStage`、padding | 已读两种 intra-CTA 分工：`split_q`、`split_kv`；分的是 Q/KV 轴，不能把名字中的 `split_kv` 误读成必然另起跨 CTA 归并 kernel | `split_q` 明确限定 MMA atom、`kStage∈{1,2}`、padding 为 8 的倍数及 warp tile 关系；性能优劣未在本次实测 [L07][L08] |
| SGEMM_CUDA：3、11、12 号 kernel | 3 号为写 shared、barrier、计算、barrier；11 号按半个 CTA 分组，两个分支交错 load/compute 并在对应位置会合；12 号以 `memcpy_async` 绑定前后 barrier，等待当前、交换 buffer 和 barrier、最终排空 | block/warp/thread tile、线程数、alpha/beta、M/N/K | 已读三种排布：`sgemm_shared_mem_block`、`sgemmDoubleBuffering`、`runSgemmDoubleBuffering2`；不能把 11 和 12 号仅按“双缓冲”名字合为同一事件顺序 | 三份所读代码不是任意尾部与参数组合的通用实现；教学实现的正确/高效范围本次未运行验证 [L01][L02][L03] |
| Mojo/MAX：AMD matmul 与 `src/pipeline` | 实际 kernel 按 schedule entry 发 barrier、`s_waitcnt`、fragment/MMA；固定 prologue、稳态 K 步进、epilogue；scheduler 依据操作依赖与硬件模型选择合法顺序 | depth、prefetch、loop-carried edge、MMA grid、partition、fragment order、wait/load 分配、VGPR 预算与策略 | AMD buffer/角色维度两条落地路径：single-buffer default、double-buffer ping-pong；顺序算法另有 `IDENTITY/GREEDY/CSP` 三项；grid 层另有 `NONE/TILE1D/TILE2D/DS_SCHEDULER` 四项，均非同一互斥轴 | `DESIGN.md` 明说没有完整 II 优化、MII 与 C2/C4 验证，phase recipe 依赖既定 kernel；实际 `verify_schedule` 有 `debug_assert` 且声明不检查全部依赖，不能宣称完整静态验证 [L09][L10a][L11][L11a][L11b][L11c][L11d] |
| cubek-matmul：multi-level global matmul | simple 每轮同步回收、load、sync、MMA；double 固定 A/B 两槽并交错当前计算和下次读取；specialized 固定 full/empty 双向 barrier 和 load/compute 分支 | tile/stage/partition 形状、reader/writer family、同步 strategy、Cmma/Mma、加载两侧与 role partition | 已读三种执行结构：`SimpleMatmul`、`DoubleBufferingMatmul`、`SpecializedMatmul`；`Strategy` enum 另列出 19 个命名项，它们混合了算法、指令和加载选择，不等于 19 个协议 | double 的 LHS/RHS/acc loader 在 trait 约束上要求同一 `SyncStrategy`；specialized 要求 `AsyncPartialLoadingStrategy`；storage-tiled 输入被 multi-level launch 明确拒绝；偶数轮通过补齐处理 [L12][L13][L14][L15] |
| cubek-matmul：hypercube last-arrival kernel | 每个 cube 获取自己覆盖的 tile 和 K 区间，调用带 last-arrival merge 的 staged microkernel；空区间直接无工作 | `Partitioning`、`StageStorage`、microkernel、类型与向量宽度 | 本行已读一条 `hypercube_last_arrival_matmul_kernel` 路径，区别于“一 CTA 独占一个完整输出 tile”；不能从 wrapper 推断底层原子内存序 | wrapper 已核实，`slots.last_arrival` 的底层握手未展开审计，内存序保证标为未核实 [L16] |
| TileLang：WS GEMM、AMD GEMM、变长与稀疏 attention | WS 示例显式写 `T.ws` 和 full/empty 相位；AMD `ss` 两输入驻 shared，`rs` 的 A 驻 fragment；变长 attention 从 `cu_seqlens` 得到循环界；稀疏版按 `block_mask[k]` 条件执行搬运和两个 GEMM | block shape、threads、stages、`k_pack`、GEMM policy、dtype、causal、运行时每样本长度/mask | WS 文件中单槽/双槽是两种已读容量安排；双槽示例的 `num_stages=2` 实际写死；AMD `ss/rs` 是两种输入驻留选择，autotune 搜索；长度与 mask 是运行时控制输入，不能合成一个有限总数 | pipeline planner 禁止某些 overlapping 多写与不满足 scalar stage/order 的排布；注释明确下游若干 pass 不支持 nested pipelines；这是实现限制，不是硬件不允许嵌套 [L17][L18][L19][L20][L21][L21a][L21b][L21c] |
| Triton/Gluon：async copy、WS、fused attention、layout conversion | 已读 async-copy 示例逐轮 commit/wait 后消费；WS add 是 load→compute→store 三角色与两个独立 buffer 环；attention 的 QK/softmax/PV 顺序固定在算法中；layout conversion 按参与者跨度选择通信路径 | tile、load/store 环各自的深度、layout、warp/register 分配、`num_stages`、`warp_specialize`、dtype | WS 可开关；layout conversion 明确有等价消除、线程内寄存器重排、warp shuffle、shared/DSM 转移四类，warp 内还可按成本退回 shared；不是一张全库排布菜单 | Gluon 文档限定 Hopper+；worker signature 与 tensor 参数受约束；attention 有 block/head shape 断言和配置剪枝；barrier 不能替代 async/generic proxy fence [L22][L22a][L23][L24] |
| TVM/TensorIR：实际 pipeline 输入/展开 IR 与 meta-schedule | 输入为每条语句的 stage/order，展开为 prologue、模索引多版本 buffer、稳态、epilogue；tiling mega-rule 顺序组合 tiling、write reuse、read reuse、async pipeline | tiling `structure`、factor、cache/reuse 位置、stage/order 数组、tensor intrinsic、线程绑定 | 不是有限命名排布；本次读取了二阶段与一般 `num_stages` 输入、动态 k 及其 lowering 描述；meta-schedule 在给定规则内产生候选 | 检查 order 唯一和 producer/consumer 依赖；某个 wait 对多 async stage 的依赖明确未支持；候选 stage `{4,5}` 注释写的是已测试选择，不能说别的 stage 都非法 [L26][L27][L27a] |
| Halide：GPU tutorial 与 `FuseGPUThreadLoops` | tutorial 把 LUT 独立计算，`padded` 放在每个 curved block 内，`sharpen` inline；lowering 在串行 block 级阶段之间插 barrier，按共享/设备读写决定 fence mask，带内部同步的串行循环再加迭代间 barrier | `compute_at/compute_root/inline`、tile、线程/块绑定、存储位置；算法的访问域影响分配与重算 | 调度语言开放；同一 blur 在论文中至少列 breadth-first、full fusion、sliding window、tiled、sliding-in-tiles 五个具体点，不能概括为只选一个同步模板 | 局部 shared 分配和 GPU loop 层级有检查；这是依赖阶段驱动的同步构造，不是任意 kernel 的异步 WS 支持证明 [L36][L37][P06]（§3.1、Fig.3） |
| IREE：`GPUPipelining.cpp` | generic 路径把全局 load 放 stage 0、其余放后面；专门的 Ampere 路径在 shared fragment load 与 `mma.sync` 间交错，再安排更早的 `cp.async` | `maxDepth`、store 所在 stage、epilogue peel/predicate、pipeline strategy | 此文件已读三项互斥 scheduling strategy：`loadGlobalStage0`、`loadStoreStage0`、`nvidiaTensorCore`，按选项与匹配结果构造具体计划；不是 IREE 全部 lowering pipeline 的总数 | Ampere matcher 要求至少 2 kgroups、3 stages、1 commit、1 wait、2 barriers，否则不匹配；未 peel 的副作用指令 predication 受限，均是这条变换的适用条件 [L38][L38a] |
| Futhark：SegRed | 按 segment 尺寸分 small/large；非交换归约维持顺序；非分段/部分 large 路径先发布 partial，再用原子计数选最后到达 CTA 执行最终归约 | block/chunk、segment 数与长度、归约 lambda、交换性、累加器类型 | segmented 路径明确两类 `Small/Large`，运行时选择；另有 last-arrival 汇合机制；不能把所有归约等同于可任意打乱的 XOR 树 | 静态 counter 分配限制同一 SegRed 的算子数量；非交换路径不同于交换路径，是语义约束而非缺了一个模板 [L39][L39a] |
| Futhark：SegScan | 单遍版发布 aggregate/prefix 状态并沿前驱 look-back，根据状态决定直接使用 prefix 或继续回溯；两遍版是另一算法 | scan lambda、segment/block/chunk、backend、运行时前驱状态 | 明确两种：`SinglePass`、`TwoPass`；CUDA/HIP 且返回 primitive、无 assert 等条件满足时选前者，否则后者 | 单遍不能直接处理带 shape 的 operator、非 primitive 返回或含 assert 的 body；它不是双遍算法换一个 stage 数 [L25][L40] |
| modern-gpu-programming-for-mlsys：GEMM steps 7–9、CLC、attention | step 7 有 TMA、MMA、writeback 三角色，四条 full/empty 方向的 barrier；step 8 为两 CTA 协作；step 9 增加 consumer；CLC 用 request、计算、wait、查询、失败退出；attention 另有数据相关 rescale 分支 | `PIPE_DEPTH`、block shape、warpgroup 数、CTA mask、persistent tile mapping、rescale threshold | GEMM 已读调度描述的三种角色/CTA拓扑；CLC 是外层任务领取协议；attention 的 TMEM 往返与可选 correction 是另一操作图 | 角色分支内不能用全 CTA barrier 替代 subgroup 会合；CLC 失败后再次 request 或读取无效坐标被明确禁止；这些不是共享面板参数能自动表达的条件 [L35][L41][L42] |
| CUTLASS：dispatch、collective builder、SM90 kernels、pipeline | tags 分派不同 kernel；`Pingpong` 有 ordered sequence barrier 轮换两个 Math WG 的 mainloop/epilogue；`Cooperative` 的两个 consumer 共同算同一 tile；TMA pipeline 分 full/empty 并向 cluster 对应成员回送释放 | tile/cluster shape、stage count、dtype、alignment、MMA atom/layout、schedule、epilogue 与 tile scheduler | TMA WS 轴至少三种：`KernelTmaWarpSpecialized`、`...Pingpong`、`...Cooperative`；Auto 在 toolkit/对齐/M 维条件下选择，并可选 CpAsync 路径 | cooperative 要 256 MMA threads、M≥128；TMA alignment、静态 shape、mixed-input/transpose 等组合有断言；cluster 的 arrival count 根据行列消费者推导，不能只改 cluster size [L28][L29][L29a][L29b][L30][L31][L32][L33] |
| FlashAttention-3 Hopper 实码 | producer 管 Q/K/V 搬运，consumer 做 QK/softmax/PV；`IntraWGOverlap` 改变首轮、稳态、尾轮的交错；mask 分成不同区间；paged/append 情况可改用 CpAsync | tile、`MmaPV_is_RS`、`IntraWGOverlap`、FP8、head dim、causal/local/varlen/split/paged、transpose | 至少两个明确时间排布：有/无 intra-WG overlap；PV 驻留又有 RS/SS 两种选择；由 head dim、mask、paged 情况和资源取舍选择，不是四种全部合法的笛卡尔积 | FP8 和 transpose_V 要 RS，large head dim 又约束 SS 与 block_M；`tile_size.h` 直接解释 shared/register 容量造成的选择，完整兼容矩阵需逐配置求解 [L34][L34a][L34b][L34c] |

### 2.2 论文中的实际算法、调度与组合方式

以下 17 篇已有 PDF 都读取了表内指明的正文位置。
两篇未取得正文的论文单独保留在表末，不用摘要补写 kernel 细节。

| 来源 | 写死的结构或实际调度描述 | 参数与排布数量、名称、选择依据 | 组合原则、禁止或实现边界 | 原文证据 |
|---|---|---|---|---|
| Fireiron | Listing 1/2/3 将 GEMM 逐层 tile、load、split、epilog，叶子匹配 microkernel/instruction；shared load 可生成同步，`noSync` 可取消 | 开放 decomposition 链；tile、层级、storage layout、padding、swizzle、unroll 与叶子实现；不是只调 tile 大小 | 以 `Spec→Spec` 组合分解，可添加自定义分解与 microkernel；本文没有读到覆盖任意分解的全局同步安全证明 | [P01] §3.1–§3.4、§4.2、Listings 1–3 |
| Sequoia | Fig.4 的 `matmul::inner` 按 U/X/V 分块并调用子任务，leaf 实现计算；Fig.6/7 给 Cell 与 cluster 的不同任务实例树 | 两种任务 variant `inner/leaf`，实例层数与映射不是固定为两层；U/X/V 是静态 tunable，数组尺寸参数可随调用变化 | 子任务通过显式 array block 接口交换数据；内层任务隔离工作集与 leaf 的平台实现，不是 NVIDIA barrier 的分类 | [P02] §3.1–§3.5、§4、Figs.4/6/7 |
| Goto | 以被打包面板、缓存驻留块和 register blocking 构造内核及外层 GEMM | §5.6 明说六种 GEMM 策略，底层 `gebp/gepb/gepdot` 等路径；m_c/k_c/m_r/n_r、packing 与循环方向根据缓存/TLB、带宽和列主序取舍 | 文章论证某条路径在其假设下优先，并非证明指令只有一种用法；CPU 内核不能直接充当 GPU 同步协议证据 | [P03] §3、§4.1、§5.1–§5.6、§6 |
| BLIS | §5.1/Fig.11 把 packing、portable macro-kernel 与最内层 k_c micro-kernel 分开，再复用到 level-3 BLAS | 不给 GPU 协议有限菜单；cache/register block、packing、datatype、操作语义是变化轴；三角操作需要专门处理 | 通过更小 microkernel 和 packing 复用多个算子；矩阵操作差异并非全部可以靠同一个无条件 store 解决 | [P04] §5.1、Fig.11 |
| Multi-BSP | level-i superstep 中子组件独立执行至 barrier，随后同父级内存通信，再进入下一步 | 深度 d 与每层 `(p_i,g_i,L_i,m_i)`；没有按 GPU 指令列若干排布 | 模型规定层次同步与成本，算法仍需构造；模型中的计算 cluster 不能等同于 NVIDIA thread block cluster | [P05] §2、§4–§6 |
| Halide | 两阶段 blur 对应不同计算/存储位置与重算策略，lowering 合成循环与边界 | Fig.3 至少五点：breadth-first、full fusion、sliding window、tiled、sliding-in-tiles；tile、compute/store level、vector/parallel 等参数 | schedule 表达 locality/parallelism/recomputation 权衡，搜索调度；不是只选同步位置且不改变生产次数 | [P06] §3.1、§4.1–§4.6、§5 |
| Lift | reduce 通过 split、partial reduce、map、join 与 reorder 规则导出不同实现，再映射 OpenCL 层级 | 开放 rewrite derivation；split/stride、map 层级、地址空间与规则序列均为变量；已实现 reorder 有 identity/stride 两类 | 按规则前提与类型组合，重排归约利用交换性；规则搜索与参数搜索并存，不能把所有 reduce 都当固定 builtin | [P07] §3.2、§4.1–§4.3、§5、§6 |
| Exo | 从标量/循环过程经 divide/reorder/fission 等重写、显式 staging、`replace` 变为硬件过程调用 | 开放 rewrite 序列；loop factor、buffer、memory、instruction、configuration state；无有限 GPU 骨架总数 | 每步重写检查等价与 effect，指令替换使用推理/SMT；这是组合正确性原则，论文对象也包括非 GPU 加速器 | [P08] §2、§3.3–§3.4、§4–§5 |
| Exo 2 | `tile2D` 由 divide/lift 组合，`tilenD` 由循环生成，`seq/repeat/try_else` 组合调度器 | 46 个 primitive 不是 46 个 kernel 骨架；调度脚本可按维数/硬件/精度改变，支持高阶 schedule 函数 | 用户可定义新的复合变换并由安全 primitive 保义；调度脚本的动态控制不要误当作生成 kernel 必定动态分支 | [P09] §3.1–§3.4、§6、Appendix A |
| Descend | transpose 等代码通过 `sched`、`forall/split` 收窄执行资源，对共享存储进行借用、同步与重借用 | grid/block/thread 的维数、切分与 view 可变，命名排布数不适用 | 执行资源检查加 borrow/access conflict 检查；split block 内全 block 同步被拒绝；安全类型系统并不声称只有一种高效 kernel | [P10] §2（Synchronization）、§3、§4、§5 |
| Prism / Typed Perspectives | Fig.6/Appendix B.1 是实际 TF32 GEMM：partition shared，窄视角写，再恢复宽视角并同步，claim 给 Tensor Core 组；Fig.13 的 async scope 限定只用异步搬运 intrinsic，后续再次使用原变量前等待完成 | group/split/partition/claim/async 的组合开放；group 大小、索引函数和视角层级是变量 | partition 结束后在再次使用原变量前恢复收敛；group 不允许向更宽层级上升或非整除分组；以类型/所有权规则组合，不是内置 GEMM 模板；不能把核心 calculus 的证明范围自动扩大到全部 unsafe intrinsic | [P11] §3.1–§3.7、§4、Figs.6/7/13、Appendix B.1 |
| MLIR Linalg / composable code generation | Fig.4 把卷积 tile 成 `scf.for` 和 tensor slices，再 padding/packing/vectorization/bufferization/progressive lowering | tile、padding、pack、vectorization、lowering 顺序；没有声明有限 GPU 协议菜单 | 各 IR 层保留结构供局部变换；该文实验主要是 single-thread CPU，不能据此认定已解决 TMA/WS 组合 | [P12] §2–§3.6、§4 标题及 Fig.4 |
| TensorIR | blocks 提供读写和迭代元信息，调度变换改变 loop/block，再把匹配的块 tensorize；sketch 同时安排计算和搬运 | tile/cache/reorder/compute placement、intrinsic、blockization 与候选 sketch；开放变换而非固定总数 | 依赖分析和 block signature 支持合法性检查；§4 还搜索 tensorized program 结构，不能缩成只选 tile factor | [P13] §3.1–§3.3、§4.1–§4.3、Fig.5 |
| Hidet | Fig.5 明确写出双缓冲的预载、当前计算、下一槽写入和同步；task mapping 由 repeat/spatial 组合，把分工直接写进 tensor program | 手写 schedule 程序配 hardware-centric space；该文 GEMM 空间小于 200 候选；predicated loading 解耦输入整除性 | mapping product 有组合定义；优化后的 anchor 再做 post-scheduling fusion；小空间是该实现人为选择与硬件资源裁剪的结果 | [P14] §2.3、§4.2–§4.3、§5.1–§5.2、Figs.5/8 |
| TVM | Fig.9 的 virtual threads、buffer 与 push/pop dependency 在不同单元间安排搬运/计算；§4.3 定义可扩展 tensor intrinsic | schedule template 中的 factor/order/binding/annotation 与 intrinsic；没有硬编码所有硬件指令清单 | virtual threading 与依赖插入用于隐藏时延；注意该处 DAE 加速器案例不能直接当成 Hopper WS 实码 | [P15] §4.1、§4.3–§4.4、§5.1–§5.3、Fig.9 |
| HBL | 从迭代集合及访问映射给通信下界；§4.1 执行模型包含 read/write/compute/allocate/free | d、访问映射、快存容量 M 等是数学输入；排布数量“不适用” | 这是下界理论，约束可达到的通信量，不构造 barrier/MMA 事件协议，也不证明最优结构唯一 | [P16] §2、§3、§4.1–§4.6 |
| CuTe Layout | canonical loop/tensor folding 以及分层 shape/stride 的布局表示，composition、inverse、complement 等代数 | 布局表达式是结构化对象，不能只当几个整数；协议菜单“不适用” | 组合布局代数而非枚举 layout pair；本文不把该文当同步安全或调度最优性的证明 | [P17] §1.2–§1.3、§2、§3.1–§3.5 |
| Twill，新取得 PDF | 将单层循环变为带 delay/iteration distance 的依赖图，求 modulo schedule、II、warp assignment，并建模存活值和跨 warp spill | 变量包括每个操作的时刻、warp、II、memory/register liveness；不是只有“选一个预置骨架”；论文重现 Hopper/Blackwell attention 专家调度 | 统一 dependence/resource/memory/warp/spill/blocking 约束；§5.4 仅支持无额外控制流的单层循环，tile size 外置；最优性仅相对模型，§6.1 的具体 kernel 实现仍有人工作业 | [P18] §3–§5.4、§6.1–§6.3 |
| Triton Linear Layouts，新取得 PDF v5 | §5.4 用目标布局右逆与源布局组合，构造线程内重排、warp shuffle 和 shared swizzle；轮数取决于布局子空间 | binary matrix、dtype、lane/warp/register 分解与 vectorization；不是有限 bespoke layout 对表 | 由代数求 conversion/swizzle，shape operator 有闭包讨论；power-of-two 限制与 affine extension 在 §8 明说 | [P19] §4、§5.1–§5.4、§8、§9.2 |
| FlashAttention-3，新取得 PDF | Algorithm 1 为 producer/consumer 有界环；Algorithm 2 在 consumer 内将 QK_next、PV_current、softmax_next 跨迭代交错；pingpong 另控制不同 consumer 的 Tensor Core/SFU 交替 | `B_r/B_c`、shared 环深 s、consumer 数、register 分配、FP16/FP8；至少三种可区分时间安排：基本 WS、inter-WG pingpong、intra-WG overlap，后两种可组合，不能作为互斥总数 | §3.2 明说 compute overlap depth 不必等于 shared 环深；§4.2 有阶段消融与寄存器压力取舍，支持同原语多种认真评估的排布 | [P20] §3.1–§3.3、Algorithms 1/2、§4.2 |
| Tawa，补充取得 PDF | Fig.2/4 将跨 partition 边变为 `aref`，语义为 empty→put/full→get/borrowed→consumed/empty；loop distribution 后各角色独立执行 | buffer depth D、在途 MMA depth P、tile 和 partition；Hopper 典型两角色，Blackwell 可增加角色；不是固定只有一 producer/一 consumer | 依赖闭包、共享标量复制、沿跨 partition 边插 channel、tuple 合并同消费输入；细/粗粒度 pipeline 再组合；实验中可行区 D≥P，并有资源代价 | [P21] §III-B–§III-E、§V-E、Fig.11 |
| Stream-K，补充取得 PDF | Algorithm 5 将整个 GEMM 的 MAC-loop iterations 连续分配给固定 CTA grid，跨 tile 边界时发布 partial 并由指定 CTA 汇合 | tile 与 grid g、工作区、tile/iteration 分割；Fig.3 三种：basic Stream-K、DP+one-tile SK、two-tile SK+DP；g 由成本模型选 | §4 说明 fixed-split/data-parallel 是特例；§5.2 讨论汇合串行化与错峰，说明全局任务分解仍有结构选择 | [P22] §4–§5.2、Algorithm 5、Fig.3 |
| PPCG | Scheduling for PPCG 的 Listing 9 是真实生成的 GEMM：带边界条件的 shared copy、barrier、private 累加、barrier、store；从 schedule tree 的可并行 band 生成 kernel | tile/grid/block `--sizes`、affine schedule、fusion/band、reference grouping；开放 schedule tree，不是有限模板数 | validity/proximity/coincidence/conditional-validity 约束及 shared/register 放置规则；report §5.1 承认无通用尺寸性能模型，部分 may-write 与不固定矩形暂存未支持 | [P23] §4.1–§4.4、§5.1、Listings 6–9、§6；本次核实的是 2017 调度报告，未冒充精读 2013 全文 |
| Graphene（ASPLOS 2023） | 未核实：未取得正文 kernel 或调度代码 | 未核实：不能据摘要填写排布数量和限制 | 官方作者页只足以确认 data/thread tensor 与 hierarchical decomposition 的研究方向 | [U01]，全文入口 DOI `10.1145/3582016.3582018` 请求失败 |
| It's about Time / Async Graphene（CC 2026） | 未核实：未取得正文 kernel 或调度代码 | 未核实：不能把三项摘要贡献当三种排布 | 官方会议摘要可确认 async tensor types、concurrency primitives、tensor-unit orchestration；其具体规则与证明未核实 | [U02]，全文入口 DOI `10.1145/3771775.3786277` 请求失败 |

### 2.3 两个容易被设计文档误导的地方

Mojo `DESIGN.md:18–19,187–193` 把 MI355X 写成 CDNA3，而实际 kernel 文件开头写的是 `MI355X (CDNA4)`；本文只据实际 AMD 指令和源码确认其非 NVIDIA 路径，不沿用设计文档的代际标签。[L09][L10]
该文档还以“6 structural checks via constrained”概述检查，而实际 `verify_schedule` 列出七项、使用 `debug_assert`，并明确说依赖顺序由 scheduler 保证；源码与说明不是同一强度的证据。[L09][L11]
因此这里认可“操作级依赖图搜索 + 固定 phase 构造”的实现存在，但不认可仅凭设计说明就声称它已是一般 modulo scheduler 或已证明全部参数组合安全。[L09][L11][L11a]

## 3. 反向归并出的骨架清单

下面是完成来源记录之后的归并，采用“局部事件协议 / 角色拓扑 / 全局调度”三个不同尺度，避免把不同尺度的结构重复计数。
编号只是本文的归并结果，不声称这是最小、完备或互斥的协议基底。
`→` 表示必须保持的先后关系，`∥` 表示可重叠，`repeat` 可以有运行时次数。

| 编号与尺度 | 归并出的结构与事件序列 | 必要参数 | 重复来源与证据 |
|---|---|---|---|
| R1：局部，无跨线程交换的映射 | 已有寄存器值→按映射重排→交给下一算子；映射等价时直接消除 | 源/目标布局、element type、局部寄存器索引 | Triton 的 layout conversion 明确有线程内重排与 identity 两条路径；CuTe 提供表示此类映射的代数 [L24][P17]（§3） |
| R2：局部，协作物化与回收 | 保护旧读者→协作填充→发布同步→读/算→确认读完→允许覆盖 | panel 集合、fill/read 映射、参与组、共享存储容量、边界谓词、同步作用域 | SGEMM_CUDA、LeetCUDA WMMA/SGEMM、cubek simple、PPCG GEMM、Halide GPU stage、Prism partition 恢复 [L01][L04][L05a][L12][P23]（§5.1 Listing 9）[L37][P11]（Fig.6） |
| R3：局部，同角色迭代流水 | prologue 预填→稳态中 next-load∥current-compute→等待所需组→轮换槽位→epilogue drain；共享存储与寄存器层可以分别流水 | 环深、预取距离、stage/order、commit grouping、每槽容量、tail/drain 策略、A/B 独立搬运方式 | LeetCUDA cp.async stage、SGEMM_CUDA 12、Hidet Fig.5、TVM pipeline、IREE；Mojo 为非 NVIDIA 的类似重复结构 [L05][L03][P14]（Fig.5）[L26][L38][L10a] |
| R4：局部，异步计算的完成队列 | issue compute→commit→允许其他独立计算/标量工作→wait 到所需完成位置→读取结果/释放最后一次源使用 | 在途 MMA 深度、commit 边界、结果版本、source-last-use、accumulator 依赖、寄存器预算 | FA3 Algorithm 2、Tawa fine-grained pipeline、LeetCUDA 的退化 `wait_group(0)`；后者只是这个结构的完全等待特例 [P20]（§3.2）[P21]（§III-D）[L06] |
| R5：角色拓扑，带借用状态的 channel 网络 | producer 等 empty→发起写并登记完成→full→consumer acquire→读/异步使用→实际使用结束后 release→empty；可串成 load→compute→store | role graph、每角色 warp 数与寄存器预算、channel payload、环深、初始 credit/phase、expected bytes、consumer arrival 集合 | LeetCUDA TMA、cubek specialized、CUTLASS、Gluon 三角色 add、Tawa `aref`、Blackwell 教材三角色 GEMM [L06][L14][L32][L22][P21]（§III-B）[L35] |
| R6：角色拓扑，多 consumer 的合作或轮换 | 合作：同一输入发布→多个 consumer 分片共同完成同一 tile→全体归还；轮换：不同 tile 的 consumer 按 turn 发 compute/epilogue，阶段交错 | tile ownership、consumer assignment、turn/ordered barrier、每角色 tile stride、stage advancement、各角色 arrival 数 | CUTLASS Cooperative/Pingpong、FA3 inter-WG pingpong；SGEMM_CUDA 11 也有分组交错，但不能把其 CTA 会合等同于 Hopper 的 named-barrier 实现 [L30][L31][P20]（§3.1）[L02] |
| R7：局部及 cluster，多接收者发布与回收 | 初始化各目的地可见的 barrier→单次/分片搬运或 multicast→目的组消费→各相关组回送 release→达到所需集合后重用→退出前保证资源仍有效 | cluster shape、接收者 mask、行/列共享关系、arrival 集合、目的地址、remote lifetime、CTA group | CUTLASS 按 cluster 行列计算 consumer arrival 和释放；教材两 CTA 的 MMA/barrier handoff；这里既有数据分发，也有联合计算两种用途 [L32][L33][L35a] |
| R8：局部，分层 partial 归约 | thread 内 partial→warp shuffle/redux 合并→必要时 shared 重分布→同步→更高层合并→交给指定 writer | reduce operator、identity、关联/交换条件、lane mask、segment 边界、归约轴、结果布局、数值顺序策略 | Triton reduce lowering、LeetCUDA split_kv 的 max/sum、Futhark blockReduce、Lift reduce derivation [L43][L08][L39][P07]（§4） |
| R9：全局，partial 发布与 last-arrival/fixup | CTA 计算局部 K 区间→发布 partial→宣布可用→由最后到达者或指定 owner 读齐并最终写出 | output owner、K 分割/连续工作区间、workspace、完成计数、发布/获取语义、归并顺序 | Futhark last-arrival、Stream-K 的 owner 汇合、cubek last-arrival wrapper；三者选择最终执行者的规则不同，不能把它们说成相同树 [L39a][P22]（§4–§5）[L16] |
| R10：全局，有序 prefix 的 look-back | 本块 scan→发布 aggregate 或 prefix 状态→查询前驱→遇 prefix 则结束、遇 aggregate 则继续累积、未就绪则重试→发布本块 prefix | operator、identity、segment 起点、state machine、look-back 范围/并行宽度、发布内存语义、进度条件 | 本次审计只有 Futhark 展开了此协议；不能据此说它在整个 GPU 生态中只出现一次 [L25][L40] |
| R11：全局，持久工作者与任务领取 | 取得 tile→执行内部协议→取得下一 tile→无任务则退出；CLC 可把“领取下一项”与“计算当前项”重叠 | task mapping、tile order/L2 swizzle、worker/grid 数、领取方式、validity、任务携带状态、跨任务 phase 延续 | Mojo tile scheduler、CUTLASS persistent scheduler、教材 persistent/CLC、Stream-K 的固定 CTA 工作区间；静态分工与动态领取是不同选择 [L11b][L30][L35][L41][P22]（§4） |

R2/R3/R5 是反复出现且最适合先库化的局部协议，R6/R7 描述角色与参与范围，R9–R11 则处理局部协议不能决定的全局工作组织。[L01][L05][L22][L30][L32][L39a][L40][L41]
R3 和 R5 可以共同使用环缓冲，但前者靠同角色程序中的预填/等待顺序推进，后者靠跨角色 credit/借用归还推进，因此“环”是共享的存储纪律，不是能抹去角色拓扑的完整 kernel 分类。[L05][L06][L26][P21]（§III-B–C）
R4 不能被“数据搬完即可释放”代替，因为异步 MMA 可能仍在读 source，FA3 和 Tawa 都把 compute completion 与 channel release 关联起来。[P20]（Algorithm 2）[P21]（§III-D）
R1 和 R8 也不能合为“warp shuffle”：前者保存每个元素、改变 owner，后者应用 combine 消去多个 partial，语义不同。[L24][L43]
packing、在线 softmax 的 `(m,l,O)` 更新和条件 rescale 是这些协议携带的算法/布局负载，不应仅因也有循环就新增一种 barrier 骨架。[P04]（§5.1）[P20]（Algorithms 1/2）[L42]

### 3.1 重复与仅在少数来源展开的结构

在本次样本中，协作暂存、环状版本化、异步计算等待、full/empty 握手、持久外层调度都有多源重复；这支持“可以沉淀一套可复用局部构件”，但没有给出封闭性的证明。[L01][L05][L06][L10][L22][L30][P21]
同样明确的非重复性是：Futhark 展开了运行时 prefix 状态回溯，教材展开了 CLC 的失败终止语义，Twill 展开了联合时刻/warp/liveness 求解，三者都要求五项原始清单之外的信息。[L40][L41][P18]（§4–§5.4）
这里的“只在某来源展开”是阅读覆盖情况，不是发明优先权或生态唯一性判断。

## 4. 最后与本项目骨架对照

对照范围取自第一性原理文档 §1.8–§1.9 和 TileLaw §8.2，并以当前 `gpu.mim`、`lower_to_mem.cpp` 为实现证据。[M07][M08][M01][M03]
第一性原理文档本身已说明 accumulator 寿命可能跨越别级循环、流水执行顺序不一定时间嵌套，因此不应把本次反驳强加为“本项目原先完全没考虑时间或寿命”。[M07]

### 4.1 一致、缺少与不宜作为独立项的部分

| 本项目归纳 | 外部反向记录的对应 | 审阅判断 |
|---|---|---|
| 同步面板超步：填、同步、读 | R2，SGEMM、PPCG、Halide、Prism/cubek 反复出现 | 一致，但必须补齐“覆盖之前确认旧读者结束”的回收边；MimIR 的 `group_enter/sync_reclaim` 已明确区分这件事 [L01][L37][P11][M01] |
| 异步拷贝环：有界缓冲区 | R3 与 R5 的槽位纪律 | 一致，但不能只有 depth：还需 phase/credit、expected bytes、每轮发多少组、consumer 最后一次使用及 drain；`RingSync` 已抽出等待、提交、排空、释放接口 [L32][P21]（§III-B）[M02] |
| warp 特化生产者/消费者 | R5，并需要 R6 的多 consumer 关系 | 一致，但一生产者一消费者不是完整模式；Gluon 和 Blackwell GEMM 已有独立 store/writeback 角色，CUTLASS 又区分同 tile 合作和不同 tile 轮换 [L22][L35][L30][L31] |
| 乱序归约树与 warp 内 shuffle 合并 | R8 的部分实现 | 一致于相应代数前提；需另外容纳有序/非交换归约、scan、last-arrival 全局汇合；shuffle 还可用于非归约布局转换 [L39][L25][L39a][L24] |
| 跨 cluster 的共享面板 | R7 | 一致于 multicast/DSM 与多组回收；它同时是作用域、发布拓扑和资源存活规则，不一定是独立于 channel 的第六种时间骨架 [L32][L33][M02] |

**本项目清单缺的主要是：异步计算完成队列、布局转换、consumer 合作/轮换关系、全局结果汇合、持久任务调度，以及条件/数据相关控制的接口。**
这些缺口分别可由 FA3/Tawa、Triton Linear Layouts、CUTLASS、Futhark/Stream-K、CLC 和 sparse/varlen attention 的已读代码支撑。[P20][P21][P19][L30][L31][L39a][P22][L41][L19][L20]
不过“清单缺”不等于当前实现全缺：`lower_to_mem.cpp` 已有 `pingpong_`、persistent ring 和 overlap 检查，不能忽略代码中超出五项短语的内容。[M03]

本项目清单没有发现“毫无外部对应的多余项”。
较准确的“多”是分类轴重复：环是容量纪律，WS 是角色分割，cluster 是参与/地址作用域，shuffle 是通信机制，把四者当互斥整块 kernel 骨架会重复计数。[L22][L24][L32][P21]
建议把它们改成可以组合的契约维度，把“整个 kernel 的调度拓扑”另外记录。

### 4.1.1 审阅之后的进展（2026-11-17）

本文的对照基于交接时的 `gpu.mim` 与 `lower_to_mem.cpp`；此后的改动记在 `docs/tile-law-design.zh-CN.md` §9.5–§9.7，与本文骨架清单的对应如下。
`docs/papers/` 下的 PDF 没有随本文进入这个分支，§6.5 的表给出了来源与校验和。

| 本文的结构或缺口 | 现在的定义 | 仍然没有的 |
|---|---|---|
| R5 通道的借用状态 | 消费者的 `gpu.Held` 与生产者的 `gpu.Vacant` 令牌，骨架对两者参数化，`gpu.check_sync` 拒绝同一令牌用两次 | 令牌不带 epoch；归还仍按滞后的步数，不由计算完成证明（规则 A 的前半句） |
| R4 异步计算的完成队列 | `gpu.wgmma_batch`：累加器只从 `wait` 的结果里拿回来 | 完成与槽的归还之间没有类型上的联系 |
| R5 的 expected bytes | `gpu.tma_fill`：期望的字节数就是各次拷贝的字节数之和 | — |
| R6、R7 的组合 | `gpu.tma_ring` 由 cluster、共享、生产者、ping-pong、持久化决定套哪几层；生产者 warp 与 cluster（共享或不共享面板）可以同用，持久化环不要求整圈 | 到达数由 `readers` 这个数给出，不是从接收者图推出的；持久化与 cluster 不能同用 |
| R8 | `gpu.merge_pairs`、`merge_first`、`merge_serial`、`lane_merge` | 有序树、scan |
| R11 | `gpu.persistent_tiles` | 领取下一项与计算当前项的重叠（CLC） |
| 布局对象 | `Tile` 的布局索引是 `layout_expr.Expr`；`lies_sw128`、`lies_tma128`、`wgmma_fields` 从布局值判定或导出 | R1 的布局转换 |
| R9、R10、规则 C 的依赖图 | 无 | 全部 |

### 4.2 对现有 `blame(...)` 的独立判断

不能把所有 `blame(...)` 都解释成“硬件原理证明这些组合非法”。
例如本项目拒绝 specialized reduction 与 cluster shared panels 同时使用，但 CUTLASS 的 `MainloopSm90TmaGmmaWarpSpecialized` 带 `ClusterShape`，TMA pipeline 又确实处理多 CTA 消费者释放，因此这里至少存在本实现未覆盖的合法结构族。[M03][L28][L32][L33]
这里比较的是“WS 与 cluster 共享/多播”的能力交集，不声称 CUTLASS 已经实现 MimIR 中每一种 reduction 语义和 split 组合。
本项目 persistent ring 要求每 tile 的 block 数为环深倍数，是为让下个 tile 从 slot 0 开始的实现不变量，源码注释直接说明了原因。[M03]
它应标为“此构造器的前提”，不能提升为 persistent kernel 的硬件必然约束；其他生成器可以携带跨 tile 的 stage/phase 状态，教材的 `PipelineState.advance()` 就跨 persistent tile 循环延续。[L35]
`lane_merge` 注释要求交换性和 2 的幂宽度，但 `Owned` panel 中各 work-item 的元素访问互不干扰，源码自己写明“nothing verifies”，因此类型索引不是完整的数据竞争证明。[M01][M02]
`ring_safe` 在这里对截取的有限 fills 模型做检查，这有实际价值，但本段代码本身不足以证明所有 count、所有 callback 和所有执行交错的安全性。[M03]
静态形状、literal tile、WGMMA layout、互斥工作组 batch、lane width、阶段释放延迟应分别归到表示边界、指令契约、同步不变量或当前实现限制，再说明能否通过别的 lowering 路径满足。[M03][M04][M05]

## 5. 反例与不成立的范围

### 5.1 最强直接反例：相同 TMA/WGMMA，consumer 拓扑不同

`KernelTmaWarpSpecializedPingpong` 的 Math WG 通过 `math_wg_order_barrier.wait/arrive` 排定轮次，在一个 WG 执行 epilogue 时让另一个推进 MMA，mainloop state 按 `k_tile_count * NumMmaWarpGroups` 跨自己的下个 tile 前进。[L30]
`KernelTmaWarpSpecializedCooperative` 则明确写着两个 consumer 协作同一 tile，要求 256 个 MMA threads，builder 把 atom layout 改为 `Shape<_2,_1,_1>`。[L31][L29a]
前者的关系可概括为 `WG0:MMA(tile a)→epilogue(a) ∥ WG1:MMA(tile b)`，后者是 `WG0:MMA(part0 of a) ∥ WG1:MMA(part1 of a)→共同完成 a`。[L30][L31]
这是 ownership、工作粒度与事件关系的变化，不是单纯换 tile 的整数值。
更关键的是 Auto builder 对 M=64 选择 Pingpong、对另一分支选择 Cooperative，并对 toolkit 和 TMA 对齐进一步分派，所以它们不是为了反驳论断而人为构造的死代码。[L29]
这个证据推翻“同原语高效结构基本唯一”，但没有证明任意给定形状下两者一样快，也没有证明 Auto 的启发式全局最优。[L29]

### 5.2 相同 attention 算法和原语，存在独立的两层重叠选择

FA3 Algorithm 1 的 consumer 每次等 QK 完成、做 softmax、等 PV 完成，Algorithm 2 则保持额外中间结果，把下一轮 QK 和当前 PV/softmax 交错；§3.1 还引入跨 WG pingpong。[P20]（§3.1–§3.2）
论文明确说 consumer 计算流水的深度不必等于 shared-memory 环深，并对两阶段和更深流水进行性能/寄存器压力讨论。[P20]（§3.2、§4.2）
实际 Hopper kernel 以 `IntraWGOverlap` 和 `MmaPV_is_RS` 选择不同代码，tile chooser 的注释说明 shared 或 register 容量会改变选择。[L34][L34a]
因此“使用环”只确定了部分资源复用结构，还没有确定 QK/PV/softmax 的时序，也没有确定 P 的驻留位置。
把这种选择都叫参数可以接受，但参数已经包含离散事件拓扑和驻留策略，而不再只是块大小与 stage 数。

### 5.3 相同 shuffle/shared 原语，布局转换也不唯一

Triton 的同 warp 布局转换既可由 shuffle 完成，也会在预计昂贵时选择 shared-memory 路径；跨线程与跨 warp 的选择来自 `minimalCvtLayout` 的维度和成本判断。[L24]
Linear Layouts §5.4 又从 `B⁻¹∘A` 的解、子空间和 swizzle 构造通信计划，轮数和路由随布局变化，并不是简单查出一个固定 shuffle 模板。[P19]（§5.4）
这支持“有限通信机制 + 通用布局求解”，反驳的是“通信机制决定唯一高效用法”，并不反驳把 shuffle exchange 做成库构件。

### 5.4 是否找到了“任何骨架 + 参数都绝对表示不了”的 kernel

**没有，而且在允许任意 callback 或新骨架时，这个命题本来就无法由有限个 kernel 证明。**
本次能够核实的是更具体的边界：五项原始骨架若只展开固定事件序列与固定轮数的局部等待，就没有显式描述 look-back 的前驱状态遍历、CLC 的失败终止或 mask 控制的事件缺席。[L40][L41][L20]
Futhark 单遍 scan 读取 `statusFlags` 后，在已有 prefix、aggregate 和未就绪状态之间选择，并在 `sWhile` 中继续回溯；其等待/依赖的前驱集合会随运行时完成状态和 segment 边界变化。[L40]
TileLang block-sparse attention 在 `if block_mask[k] != 0` 内才执行搬运、QK、softmax 和 PV，这改变该次迭代是否产生这些事件。[L20]
CLC 的动态工作者只有 query 成功时才解释下一任务坐标，失败必须退出，不能把它降成无条件读取下一 panel 的环。[L41]
可以给这些行为增加 R10/R11 或一般 guarded event graph，但这恰好说明需要扩展协议语言或让“参数”成为控制程序，不能从五个名字直接推出覆盖性。
另一方面，它们仍可以由有限 DSL 构造表达，故不能据此宣称“所有可组合骨架设计都不可能”。

### 5.5 动态形状、稀疏与非 NVIDIA 的边界不同

**动态形状本身不是反例。**
TVM 的 dynamic-k pipeline 测例、TileLang 变长 attention 的 `cu_seqlens` 循环界，以及 Hidet 的 predicated loading 都展示了保留规则结构、改变次数与边界的方法。[L26][L19][P14]（§4.3）
但当短循环不足以填满 prologue、不同角色的有效迭代数不一致、尾部有副作用或 expected byte/arrival 与谓词不一致时，不能无条件套用同一实现；IREE 的 predication 限制、Gluon 的相同迭代数要求和 TMA 的字节登记显示了这些额外义务。[L38a][L22][L06]
**稀疏可分为规则 tile 内的稠密计算与外层不规则枚举。**
TileLang 的块稀疏示例保留内部 GEMM/softmax 协议，同时根据 mask 跳过整块，证明复用可行，但没有证明该同步策略对任意稀疏格式、指针链或数据依赖任务图都成立。[L20]
**非 NVIDIA 可以复用 abstract ownership/credit/last-use 等概念，不能直接复用整条硬件层级链和全部完成语义。**
Mojo AMD kernel 发射 `s_waitcnt`、LDS load 和 MMA，并由操作计数推导 wait；它不通过 NVIDIA TMA 的 expected-byte mbarrier 来确认同一种搬运。[L10a][L11c]
Futhark 的相同 scan 接口对 CUDA/HIP 可以选择单遍，对其他 backend 回退两遍，已直接说明支持能力和进度条件会改变算法选择。[L25]
本次没有独立审计 Intel/Metal GPU 的完整高性能异步 GEMM，相关硬件上能保留多少具体协议标为未核实。

### 5.6 “编译器不发明变换”需要拆开解释

若意思是编译器实现通常复用研究者已知的变换和硬件契约，这与 Exo、TVM、TensorIR、Tawa 的描述相容，但不能推出每个输入只需从几张固定调度图中选一张。[P08]（§3.3）[P15]（§4–§5）[P13]（§4）[P21]（§III）
Twill 对每个输入操作图联合求时刻和 warp 分配，Mojo 对操作依赖做 permutation 搜索，PPCG 构造 affine schedule tree，都是明确的计划生成工作。[P18]（§4）[L09a][P23]（§4、§6）
若把求解结果也叫“参数”，那么这些结果包括了待合成的调度本身，论断中的“不是发明变换”就只剩术语选择，不能用来删去编译器的依赖分析、代码重写和调度构造职责。
这些来源也没有证明编译器必须在线发明全新数值算法；本文反对的是把“复用已知理论”误推成“无需一般变换表示”。

## 6. 对“骨架 + 参数”设计的建议

本节是依据前述反例提出的设计建议，不把建议写成现有 MimIR 已具备的保证。

### 6.1 参数应分成数值、结构对象与契约三组

| 参数组 | 建议进入 IR/构造器的内容 | 为什么需要，来源如何处理 |
|---|---|---|
| 形状与资源 | logical/physical tile、tail policy、dtype/accumulator type、vector width、alignment、SMEM/TMEM/register 容量、每角色 warp 数与 occupancy 目标 | CUTLASS builder 按 alignment 与 shared carveout 求 stages；FA3 chooser 用 head dim、shared/register 约束；Mojo scheduler 有 VGPR budget [L29][L34a][L09a] |
| 空间与布局对象 | 每级 owner map、logical→physical layout、swizzle、fragment map、源/目标布局转换计划、cluster receiver mask | CuTe/Linear Layouts 提供代数，Triton 根据转换跨度选择 shuffle/shared；这些对象比枚举 `row_major/col_major` 更大 [P17]（§2–§3）[P19]（§4–§5）[L24] |
| 时间结构 | operation DAG、循环携带距离、stage/order、issue/commit/wait 分组、MMA 在途深度、prefetch 距离、每 buffer 独立环深、last-use/release 时间 | Twill 用约束求解，TVM 接受并检查 stage/order，FA3 区分 shared 环与计算 overlap 深度 [P18]（§4）[L27][P20]（§3.2） |
| 角色与全局工作 | producer/compute/reduction/store/scheduler role graph、cooperative/pingpong ownership、tile order、persistent work assignment、split-K/Stream-K 与 final writer | CUTLASS 分派已知结构，Tawa 用依赖闭包/复制/loop distribution 构造，Stream-K 用分析模型选 grid [L30][L31][P21]（§III-C）[P22]（§5） |
| 完成与一致性 | channel 的 empty/full/borrowed 状态、初始 phase/credit、arrival 集合、expected bytes、async-proxy fence、source/accumulator 完成 token、remote lifetime | Tawa `aref` 是状态规则，Gluon 明确跨角色也需 proxy fence，CUTLASS cluster release 计算完整接收者集合 [P21]（§III-B）[L22][L32][L33] |
| 数值与控制 | identity、关联/交换性见证、允许的重结合/精度策略、segment 与 mask、动态 count、空块、条件事件、退出/drain 策略 | Futhark 区分非交换路径和 scan backend；稀疏/变长 attention 需要动态控制输入；IREE 对带副作用尾部明确限制 [L39][L25][L19][L20][L38a] |

建议用已知协议预设覆盖常见 R2/R3/R5，但保留事件图与结构化循环/条件作为底层表示。
否则要么每遇 FA3 一类复合图就增加专门的整块 kernel 模板，要么最终把任意代码塞进 opaque callback，丢失跨 op 分析需要的信息。[P20]（Algorithm 2）[P21]（§III-C–D）

### 6.1.1 骨架之外的空间，各来源究竟怎么选

| 来源 | 真正继续变化的内容 | 已核实的选择方法 |
|---|---|---|
| LeetCUDA、SGEMM_CUDA | 数据到线程的分配、tile、寄存器预取、bank conflict、向量化、阶段结构 | 已读 kernel 是人工写出的不同模板和控制顺序；没有核实全仓库是否还有统一自动搜索器 [L01][L02][L03][L04][L05][L06] |
| CUTLASS、FA3 | kernel policy、tile/cluster、shared/寄存器驻留、stages、mask/paged/精度组合 | builder 静态分派与形状/对齐/toolkit 启发式，FA3 另有人工 tile 选择表与资源解释；不能把这个规则表当最优性证明 [L29][L29a][L34a] |
| Mojo | 给定操作图的顺序、load 分配、wait、VGPR 峰值、固定 phases 的配合 | `IDENTITY/GREEDY/CSP`，CSP 在模型下搜索轮内顺序；硬件配置/phase recipe 仍由代码提供 [L09a][L11a][L11c][L11d] |
| cubek | routine、输入 packed 形式、stage/partition、默认设置与回退路径 | 当前 selector 明分 `Autotune` 实测、`Heuristic` 规则和 `Specific` 指定；另有 routine/settings 的 winner 表示，不应把旧 multi-level enum 当整个项目的唯一空间 [L44][L45][L15] |
| TileLang、Triton | 手写程序内的 tile、stage、threads/warps、`ss/rs`、WS、shape-specific 配置 | 示例显式列候选并 autotune；Triton 用条件过滤和 early pruning；编译器内部仍要构造布局转换与流水 [L21a][L23][L24][L21] |
| TVM/TensorIR、Hidet | sketch/变换序列、tensor intrinsic、cache/reuse、task mapping、融合位置 | TVM 原论文用学习成本模型引导搜索，TensorIR 搜索 tensorized sketch；Hidet 采用手工定义且硬件裁剪的小候选集枚举 [P15]（§5.1–§5.3）[P13]（§4）[P14]（§4.3） |
| Halide、Lift | compute/store/recompute、循环组织、规则推导与数值参数 | 所读 Halide 论文用 genetic autotuning 与常见 pattern 先验；Lift 搜索 rewrite derivation 及参数并用 macro rule 缩短路径 [P06]（§5）[P07]（§6） |
| IREE、Futhark | target intrinsic、workgroup/subgroup tile、pipeline 匹配；segment 和 operator 对算法的影响 | IREE 用架构 seed、容量与 intrinsic 匹配推导 MMA schedule，失败后尝试 accumulator upcast；Futhark 用尺寸/语义/backend 条件分派 [L46][L38][L39][L25] |
| Exo/Exo 2、Descend、Prism | 程序变换或手写控制结构、布局、角色/参与组 | Exo 以用户调度库加保义检查组合；Descend/Prism 对显式程序的资源/借用/视角作检查，本文没有找到它们自动优化全部异步调度的证据 [P08]（§3–§5）[P09]（§3、§6）[P10]（§3–§5）[P11]（§3） |
| Twill、Tawa、Stream-K、PPCG | 事件时刻/warp/liveness；partition/channel；全局工作划分；affine schedule | 分别是 ILP/SMT 联合约束、语义依赖切分与流水策略、分析成本模型、polyhedral schedule 求解，优化对象不能混成同一个“调参器” [P18]（§4–§5）[P21]（§III）[P22]（§5）[P23]（§4、§6） |

这些来源显示“大空间”至少分布在布局映射、融合/重算、角色/任务分解和时间/资源分配四个相互影响的轴上，但本次没有统一计数其基数，也没有测得哪个轴在所有 workload 上占主导。[P06][P18][P19][P21][P22]

### 6.2 可以成体系的组合规则

**规则 A：用资源所有权与 epoch 定义回收，而不是仅用“执行到第几条 barrier”。**
建议为每个 slot 维护 `Empty→Filling→Full→Borrowed→Empty`，异步计算借用结束应由 compute completion 证明，TMA copy completion 只证明 Filling→Full。[P21]（§III-B–D）[P20]（Algorithm 2）
多 consumer 时，最后一个相关 consumer 的 release 才能产生可重用 credit；cluster 的到达集合应由 receiver graph 推导，不能只用一个总线程常数代替。[L32][L33]
建议将 `buffer_id/slot_id/epoch/phase/scope` 与 token 关联，以避免串错相位、串错资源和跨 tile 重置错误。[M01][M02][L35]

**规则 B：顺序组合要求前一算子的 last-use 与后一算子的覆盖不冲突。**
建议把某操作的读写 effect、异步 source use、accumulator use 和可复制的纯标量分别记录，只有满足对应资源顺序才能移动或融合。[P08]（§5）[P18]（§4.2–§4.3）
Tawa 对两个 partition 都需要的纯计算进行复制，而不是无条件添加一个跨 warp 值；这是一条已实现的构造原则，可以借鉴。[P21]（§III-C）
Halide 的 compute/store 位置与重算空间提示融合还会改变工作量和存储量，因此不能把融合只看作删 barrier。[P06]（§3.1、Fig.3）

**规则 C：流水组合用带迭代距离的依赖边和资源容量约束。**
对 producer 事件 u、consumer 事件 v，建议至少表达 `t(v,i+δ) ≥ t(u,i)+latency(u,v)`，再加入每个功能单元的占用、buffer 的 live range 和每角色寄存器容量；这是 Twill 式组合的核心。[P18]（§3.1、§4）
用它统一 R3 的预取、R4 的延迟等待和 R5 的 role/channel，而不是手工枚举每个 depth 与每种角色数的全部排布。
若只采用轻量实现，也应像 TVM 一样检查 stage/order 的依赖顺序，并像 Mojo 一样明确区分“轮内 makespan 优化”与“跨轮 II 优化”。[L27][L09][L09a]

**规则 D：并行角色组合除了局部安全，还要检查进度。**
建议检查所有 role 的迭代/任务流一致、必要 arrival 不被条件跳过、环的初始 credit 足够启动、退出时不会留下另一角色永久等待，以及调用的 barrier 参与组与分支收敛组相同。[L22][P10]（§2，Synchronization）[L35][L41]
单个 channel 的状态机无数据竞争，不自动推出多个 channel 形成的有向环没有死锁；本文未发现能直接套在全部列举框架上的通用自动死锁判定，因此这里提出的是检查义务而非已有完备算法。[P21]（§III-B–C）[L22]
对于跨 CTA spin/look-back，还需目标允许的前进条件，不能仅因为 CUDA/HIP 某路径存在就推广到所有 GPU backend。[L25][L40]

**规则 E：层级组合以 capability 和作用域关系约束，不要求穿过固定六级链。**
建议目标描述提供参与者分组、可寻址存储、可用完成机制和同步 scope，再实例化 grid/cluster/CTA/warp 等关系；AMD 的计数等待与 NVIDIA 的 mbarrier 可实现相似逻辑接口，但其完成语义分别 lowering。[L10][L32][L35]
对 tensor instruction，必须同时核验 operand layout、alignment、参与者收敛与 source lifetime，不能只检查某一 tile 维度是否整除。[L29a][L31][L34][M05]

**规则 F：数据相关事件必须显式有 guard，或通过统一 predication 保持协议次数。**
建议将“填零但仍提交事件”“完全跳过该事件”“结束整个角色”分为不同操作，并推导对应 expected bytes/arrival/phase；三者不能共享一个模糊的 `tail=true` 参数。[L19][L20][L22][L38a]
结构化条件可以保留在骨架语言中，但对 Twill 当前不覆盖的额外控制流，需要单独的层次调度或回退路径，不能声称其现有最优性直接适用。[P18]（§5.4）

**规则 G：归约协议以代数和浮点语义授权。**
建议区分保序 fold、只要求关联的有序树、允许交换/重排的 unordered reduction、scan，以及 source-preserving 的 layout exchange。[L39][L25][L24][L43]
具体浮点重结合策略应由语言或用户选择；Triton 代码甚至保留既有 shuffle bit 顺序来维持 bitwise comparison，这说明“同一个数学和”不足以决定可接受的实际结果顺序。[L43]

### 6.3 哪些组合已有原则，哪些仍是手工枚举

| 层面 | 已读来源提供的原则 | 仍未被该原则解决的部分 |
|---|---|---|
| 程序变换保义 | Exo/Exo 2 的可组合安全重写；Lift 的带前提 rewrite；TensorIR block 与依赖检查 | 不自动给出所有目标上最优的同步/布局组合 [P08]（§3–§5）[P09]（§3）[P07]（§4–§6）[P13]（§3–§4） |
| 空间映射组合 | Hidet mapping product、CuTe 布局代数、Triton Linear Layouts 的 conversion 生成 | 不决定完整异步事件时序与所有 kernel 的最优融合 [P14]（§5.1）[P17]（§3）[P19]（§5） |
| 角色通信组合 | Tawa 依赖闭包、跨 partition 边生成 `aref`、同消费者 tuple 合并、loop distribution | 角色选择仍由语义分析/策略驱动，不是任意任务图全部资源和进度的完备求解 [P21]（§III-C–D） |
| 时间与资源联合组合 | Twill 对 dependence、memory、warp、spill、blocking 的联合约束 | 不自动选 tile，不处理任意额外控制流，实际代码实现仍有未自动化部分 [P18]（§4、§5.4、§6.1） |
| 已落地的局部 builder | CUTLASS tag/specialization 与 static assertions；Mojo 的依赖搜索加固定 phase recipe | 仍有手工命名结构、专门分派和 recipe，不能把有限已实现菜单当成硬件合法结构全集 [L28][L29][L09][L11a] |
| affine 全局循环组织 | PPCG 从 validity/coincidence/proximity 等约束生成 schedule tree，再插 data copy/sync | 固定形状矩形暂存、非 affine 控制和现代 Tensor Core 异步协议需要额外处理 [P23]（§4–§6） |

这些证据支持把组合规则做成体系，但现有来源解决的是不同层面，没有哪一个已读来源独自证明“任意上述骨架自由叠加都正确且高效”。[P08][P18][P19][P21][L09][L29]

### 6.4 实施优先级与检查的分类

建议先固化 R2、R3、R5 的可审计构造器及其 callback effect，再把 R4 的 compute completion、R6 的 ownership/turn 和 R7 的 receiver-set 加入同一个资源协议模型。
第二步用布局代数减少 bespoke conversion，用带迭代距离的图记录 QK/PV/softmax 等复合结构，把参数搜索与事件调度分开但允许联合剪枝。[P19]（§5）[P18]（§4）[P20]（Algorithm 2）
第三步保留 R9/R10/R11 的全局控制接口，明确哪些采用预置算法、哪些从问题图生成、哪些暂不实现。[L39a][L40][L41][P22]
诊断建议分为 `硬件/指令不满足`、`语义或安全义务未满足`、`资源不可行`、`当前 lowering 未实现` 四类，不再用一个“非法组合”概念混淆本项目的 `specialized+cluster`、persistent slot-0 前提与真正的 source lifetime 错误。[M03][M04][M05][L29][L32]
上述建议的目标是让常见 kernel 只需少量声明，同时在遇到新的操作图、布局或角色拓扑时仍能构造和验证计划。

### 6.5 新增 PDF 与核实缺口

本次仅新增以下五份已验证为 PDF、可提取正文的论文文件，失败下载没有以 `.pdf` 名义落盘。
PPCG 的 2017 报告通过网页工具读取了 PDF 正文，直接下载入口返回 HTML，因此未保存伪 PDF。[P23]
Graphene 与 Async Graphene 的正文缺口见 §1.2 和 §2.2，相关 kernel/规则的任何进一步判断均为未核实。[U01][U02]

| 文件 | 下载的原文入口 | SHA-256 |
|---|---|---|
| `docs/papers/twill-2512.18134.pdf` | [arXiv PDF](https://arxiv.org/pdf/2512.18134) | `c8b936fdd03c781316c05f0b3af17a40512d108a7a4baa40846ce8f99bdd1a1d` |
| `docs/papers/triton-linear-layouts-2505.23819.pdf` | [arXiv v5 PDF](https://arxiv.org/pdf/2505.23819v5) | `c6eaa53ed2c1badfb1c6680e5eb3140bc0b3bd26c7efbf85d56c8836e7be4e4f` |
| `docs/papers/flashattention3-2407.08608.pdf` | [arXiv PDF](https://arxiv.org/pdf/2407.08608) | `3d05ca102802e6b8ebbae5181efe777b7d258cd75e2fe9cca209efd2f5e6c6cb` |
| `docs/papers/tawa-cgo26.pdf` | [作者 PDF](https://www.csl.cornell.edu/~zhiruz/pdfs/tawa-cgo2026.pdf) | `2427b803a91fa3d4534a7afa6245b384bb9bab7d92de5e317ee72236ce8eca2a` |
| `docs/papers/streamk-2301.03598.pdf` | [arXiv PDF](https://arxiv.org/pdf/2301.03598) | `f2db58086debe599590650cac1069102ddf9396ac398e5a57284d3d524832cc5` |

### 6.6 证据索引

`L` 链接为本地外部项目实码或调度文档，`P` 链接为已取得正文的论文，`U` 为仅核实官方摘要的缺口，`M` 为最后对照的本项目材料。
论文的具体章节、图和算法编号见相应表格或论证句；本地代码链接的目标含行号，正文提及的相邻范围可在同文件复查。

[L01]: /workspaces/compile/SGEMM_CUDA/src/kernels/3_kernel_shared_mem_blocking.cuh:34
[L02]: /workspaces/compile/SGEMM_CUDA/src/kernels/11_kernel_double_buffering.cuh:134
[L03]: /workspaces/compile/SGEMM_CUDA/src/kernels/12_kernel_double_buffering.cuh:102
[L04]: /workspaces/ml-compiler/LeetCUDA/kernels/sgemm/sgemm.cu:94
[L04a]: /workspaces/ml-compiler/LeetCUDA/kernels/sgemm/sgemm_async.cu:143
[L05]: /workspaces/ml-compiler/LeetCUDA/kernels/hgemm/mma/basic/hgemm_mma_stage.cu:122
[L05a]: /workspaces/ml-compiler/LeetCUDA/kernels/hgemm/wmma/hgemm_wmma.cu:120
[L05b]: /workspaces/ml-compiler/LeetCUDA/kernels/hgemm/wmma/hgemm_wmma_stage.cu:121
[L05c]: /workspaces/ml-compiler/LeetCUDA/kernels/hgemm/naive/hgemm_async.cu:89
[L06]: /workspaces/ml-compiler/LeetCUDA/kernels/hgemm/wgmma/hgemm_wgmma_fp32acc_stages_tn.cu:137
[L07]: /workspaces/ml-compiler/LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu:35
[L08]: /workspaces/ml-compiler/LeetCUDA/kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu:416
[L09]: /workspaces/compile/modular/max/kernels/src/pipeline/DESIGN.md:9
[L09a]: /workspaces/compile/modular/max/kernels/src/pipeline/schedulers.mojo:415
[L10]: /workspaces/compile/modular/max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo:13
[L10a]: /workspaces/compile/modular/max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo:661
[L11]: /workspaces/compile/modular/max/kernels/src/pipeline/program_builder.mojo:924
[L11a]: /workspaces/compile/modular/max/kernels/src/pipeline/compiler.mojo:223
[L11b]: /workspaces/compile/modular/max/kernels/src/linalg/matmul/gpu/tile_scheduler.mojo:146
[L11c]: /workspaces/compile/modular/max/kernels/src/pipeline/config.mojo:387
[L11d]: /workspaces/compile/modular/max/kernels/src/pipeline/program_builder.mojo:1034
[L12]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/multi_level/components/global/single_stage/simple/matmul.rs:175
[L13]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/multi_level/components/global/multi_stage/double_buffering/matmul.rs:58
[L14]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/multi_level/components/global/multi_stage/specialized/matmul.rs:181
[L15]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/multi_level/strategy.rs:69
[L16]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/kernels/dense/routines/hypercube/last_arrival/kernel.rs:18
[L17]: /workspaces/ml-compiler/tilelang/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py:14
[L18]: /workspaces/ml-compiler/tilelang/examples/warp_specialize/example_warp_specialize_gemm_barrierpipe_stage2.py:13
[L19]: /workspaces/ml-compiler/tilelang/examples/flash_attention/example_mha_fwd_varlen.py:64
[L20]: /workspaces/ml-compiler/tilelang/examples/blocksparse_attention/example_tilelang_block_sparse_attn.py:77
[L21]: /workspaces/ml-compiler/tilelang/src/transform/pipeline_planning.cc:653
[L21a]: /workspaces/ml-compiler/tilelang/examples/gemm_fp8/example_tilelang_gemm_amd.py:27
[L21b]: /workspaces/ml-compiler/tilelang/src/transform/pipeline_planning.cc:830
[L21c]: /workspaces/ml-compiler/tilelang/src/transform/pipeline_planning.cc:1288
[L22]: /workspaces/ml-compiler/triton/python/tutorials/gluon/08-warp-specialization.py:68
[L22a]: /workspaces/ml-compiler/triton/python/tutorials/gluon/03-async-copy.py:145
[L23]: /workspaces/ml-compiler/triton/python/tutorials/06-fused-attention.py:128
[L24]: /workspaces/ml-compiler/triton/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp:36
[L25]: /workspaces/ml-compiler/futhark/src/Futhark/CodeGen/ImpGen/GPU/SegScan.hs:51
[L26]: /workspaces/ml-compiler/tvm/tests/python/s_tir/transform/test_s_tir_transform_inject_software_pipeline.py:99
[L27]: /workspaces/ml-compiler/tvm/src/s_tir/transform/inject_software_pipeline.cc:1053
[L27a]: /workspaces/ml-compiler/tvm/src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.cc:81
[L28]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/dispatch_policy.hpp:109
[L29]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:991
[L29a]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:225
[L29b]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl:396
[L30]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp:800
[L31]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp:350
[L32]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:271
[L33]: /workspaces/ml-compiler/tvm/3rdparty/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:611
[L34]: /workspaces/ml-compiler/pytorch/third_party/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:44
[L34a]: /workspaces/ml-compiler/pytorch/third_party/flash-attention/hopper/tile_size.h:9
[L34b]: /workspaces/ml-compiler/pytorch/third_party/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1138
[L34c]: /workspaces/ml-compiler/pytorch/third_party/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:211
[L35]: /workspaces/ml-compiler/modern-gpu-programming-for-mlsys/chapter_gemm_advanced/index.md:195
[L35a]: /workspaces/ml-compiler/modern-gpu-programming-for-mlsys/chapter_gemm_advanced/index.md:58
[L36]: /workspaces/ml-compiler/Halide/tutorial/lesson_12_using_the_gpu.cpp:134
[L37]: /workspaces/ml-compiler/Halide/src/FuseGPUThreadLoops.cpp:1249
[L38]: /workspaces/ml-compiler/iree/compiler/src/iree/compiler/Codegen/Common/GPU/GPUPipelining.cpp:520
[L38a]: /workspaces/ml-compiler/iree/compiler/src/iree/compiler/Codegen/Common/GPU/GPUPipelining.cpp:634
[L39]: /workspaces/ml-compiler/futhark/src/Futhark/CodeGen/ImpGen/GPU/SegRed.hs:1
[L39a]: /workspaces/ml-compiler/futhark/src/Futhark/CodeGen/ImpGen/GPU/SegRed.hs:941
[L40]: /workspaces/ml-compiler/futhark/src/Futhark/CodeGen/ImpGen/GPU/SegScan/SinglePass.hs:628
[L41]: /workspaces/ml-compiler/modern-gpu-programming-for-mlsys/chapter_clc/index.md:47
[L42]: /workspaces/ml-compiler/modern-gpu-programming-for-mlsys/chapter_flash_attention/index.md:98
[L43]: /workspaces/ml-compiler/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:359
[L44]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/kernels/select/selector.rs:23
[L45]: /workspaces/ml-compiler/cubek/crates/cubek-matmul/src/kernels/tune/routine_settings.rs:15
[L46]: /workspaces/ml-compiler/iree/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp:356

[P01]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/fireiron-2003.06324.pdf
[P02]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/sequoia-sc06.pdf
[P03]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/goto-anatomy-toms08.pdf
[P04]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/blis-toms15.pdf
[P05]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/multibsp-valiant.pdf
[P06]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/halide-pldi13.pdf
[P07]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/lift-icfp15.pdf
[P08]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/exo-pldi22.pdf
[P09]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/exo2-2411.07211.pdf
[P10]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/descend-2305.03448.pdf
[P11]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/prism-2511.11939.pdf
[P12]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/mlir-composable-2202.03293.pdf
[P13]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/tensorir-2207.04296.pdf
[P14]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/hidet-2210.09603.pdf
[P15]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/tvm-1802.04799.pdf
[P16]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/hbl-1308.0068.pdf
[P17]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/cute-layout-2603.02298.pdf
[P18]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/twill-2512.18134.pdf
[P19]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/triton-linear-layouts-2505.23819.pdf
[P20]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/flashattention3-2407.08608.pdf
[P21]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/tawa-cgo26.pdf
[P22]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/papers/streamk-2301.03598.pdf
[P23]: https://www.cs.kuleuven.be/publicaties/rapporten/cw/CW706.pdf
[U01]: https://mgarland.org/papers/2023/graphene/
[U02]: https://conf.researchr.org/details/CC-2026/calls/10/It-s-about-Time-Temporal-Abstractions-for-Asynchronous-GPU-Tensor-Computations

[M01]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/gpu.mim:618
[M02]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/gpu/gpu.mim:704
[M03]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:3949
[M04]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:1783
[M05]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/src/mim/plug/tensor/phase/lower_to_mem.cpp:2588
[M06]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/gpu-tiling-literature.zh-CN.md:39
[M07]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/gpu-multilevel-first-principles.zh-CN.md:138
[M08]: /workspaces/compile/MimIR/.worktrees/gpu-tiling/docs/tile-law-design.zh-CN.md:278

