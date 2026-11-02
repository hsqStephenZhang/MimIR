# 模型调度诊断：LayerNorm 重算与 contraction 入口

## 范围与方法

输入来自 `models-3137f3dbfd-seq{128,512,1024}` 的实际 `kernel.mim`。
模型为单线程 F32、batch 1、embedding 768、12 heads、hidden 3072。
诊断不调整流水线默认值、分块参数或 BLIS 的适用条件。
并行构建会污染延迟，结论优先使用动态运算次数、循环边界与物理访问布局。

调度选择需要 `-VVVV`；`-VV` 只显示 phase 边界。
group 开关为 `-X tensor:group=1`，没有 `tensor:group=report` 模式。
LLVM 输出使用 `-X ll:o=<文件>`。

## LayerNorm：共享规约在 tile 特化后丢失

模型的归一化尾部还包含逐列 scale、bias 与 leading-singleton reshape。
Task21A 的 `row_group` 只接受同形二维输入或第一坐标的行投影；
列广播权重不属于该闭包，所以不能直接套用其 4096×1024 的整行 recipe 结论。
模型实际选择 centered 与 variance/denom 的 reverse group：

```text
kind=2 dim=0 tile=21 scratch=258216 exact=true factor=1 members=3
```

`reverse_source` 在 grid point 变成字面量时才构造 producer tile。
此前 `capture_producers` 遇到可捕获的整个 centered 就停止，不再记录其中的共享 mean。
tile 特化移除了这个外层 centered 节点，却仍需要 mean；
随后每个 tile 的独立 memo 又各自实现了一遍完整的 mean。
成本模型估算的是缩小后的 producer 域，不能反映这个 lowering 的实际重算。

修复让 destination body 的 capture 继续检查可捕获 producer 的依赖，
并复用已有的类型化占位替换，把这些依赖的同一 buffer 带进每个特化实例。
普通 `generate` 的 capture 截断行为保持不变。
这不是把整个 producer DAG 无条件内联进每个 tile。

在 LLVM 优化前给每条 `fadd` 按 lane 数插入外部计数调用，结果如下：

| 输入 | 修复前 group=1 | 修复后 group=1 |
|---|---:|---:|
| 17×8，cache=256 | 2890 | 578 |
| 33×8，cache=256 | 9834 | 1122 |
| 512×768，默认 cache | 11404288 | 1573888 |
| 1024×768，默认 cache | 41682944 | 3147776 |

固定宽度 768 时，旧计数为
`(ceil(rows / 21) + 4) * rows * 768 + 2 * rows`，
新计数为 `4 * rows * 768 + 2 * rows`。
512 行 LLVM 中以 512 为边界的循环也从 29 处降到 4 处。
规约重算的二次项消失，但这条 reverse recipe 仍会实现整体结果与局部结果，
并不具备 Task21A 的“一份行 scratch、只写最终输出”能力。

`reverse_capture.mim` 强制多 tile 与尾块，在 strict 下核对 reverse 请求实现；
辅助程序同时验证数值、线性运算上限与 ASan 生命周期，另有 group=0 对照。
旧工具链不能通过该运算上限。

## Attention 与 Transformer：不是同一种二次重算

seq512 的完整默认流水线没有为这两个模型选择 kind-0/1/2 请求。
其动态加法次数在两种 group 设置下分别完全相同：

| 模型 | group=1 | group=0 |
|---|---:|---:|
| attention | 1615331328 | 1615331328 |
| transformer | 4036363264 | 4036363264 |

因此不能把两个模型的 group=1 延迟增幅归因于上面的 mean-per-tile bug。
Attention 的 QKᵀ/PV 本来就有 `O(heads * rows² * head_dim)` 的工作。
Transformer 内部的归一化又连接到 contraction/residual DAG，
并没有复现独立 LayerNorm 的 reverse 请求。

但 group 流水线的额外 world rewrite 会暴露更多可识别的 pure-read view，
bufferization 随后可以直接绕过 Q/K 的重排。
group=0 的 LLVM 有 Q/K 的 head-major 连续缓冲区；
group=1 的 QKᵀ 内循环直接读 packed QKV：

```text
head-major K[h, token, channel]：相邻 token 的跨度为 64 个 float
packed QKV：相邻 token 的逻辑跨度为 2304 个 float
```

后者还需要把逻辑下标拆成 `(token/2, token%2, column/8, column%8)`。
seq512 LLVM 的整数 `udiv/urem` 指令静态数量从 68 变成 113。
在 scratch 中单独保留 `tensor.group` 或 `tensor.group_rules`，即使不执行
`fuse_views`，也能复现这一布局变化；不能简单归咎于最后一遍 `fuse_views`。
同机原生调用的量级对照约为 98 ms 与 225 ms，仅用来确认差异不是计时噪声。

这里缺少的是 stride/cache-aware 的 view 实现成本与必要的重排保留能力，
而不是漏算或重复执行 contraction。
本次不通过关闭 group 或修改默认融合策略掩盖它。
具体缓存未命中比例仍需硬件计数器，不能仅凭延迟推导。

## 模型 GEMM 为什么没有得到 BLIS

Linear 的权重按 `[out, in]` 存储，模型使用 `dot_product` 的 `(1,1)` 收缩，
并非先提供一个标准 `[K,N]` 矩阵。
`lower_tensor` 因而先选择 `dot_schedule_kvec`，把输出作 2×8 cell 的 jam。
等 group planner 运行时，它面对的已不是二维 canonical GEMM。

seq512 的具体结构如下：

| contraction | 逻辑 M×N×K | group 前的结构 | kind-1 |
|---|---|---|---|
| MLP W1 | 512×3072×768 | Ro=4，So=(256,2,384,8)，Sr=(256,384,2,8,768) | 不请求 |
| MLP W2 | 512×768×3072 | Ro=4，So=(256,2,96,8)，Sr=(256,96,2,8,3072) | 不请求 |
| attention QKV | 512×2304×768 | 同样的 packed kvec，So=(256,2,288,8) | 不请求 |
| attention 输出 projection | 512×768×768 | 同样的 packed kvec | 不请求 |
| QKᵀ | 12 个 512×512×64 | Ro=3，Sr=(12,512,512,16,4) | 不请求 |
| PV | 12 个 512×64×512 | Ro=3，Sr=(12,512,64,128,4) | 不请求 |

日志首先报告 `BLIS needs two parallel axes and two inputs`。
MLP 的两个输入数量正确，失败的是输出 parallel 秩；
第二层还读取第一层的 packed rank-4 输出。
即使把 logical GEMM 提前交给当前 recipe，B 的 `(n,k)` 访问也不符合它要求的
`(m,k),(k,n)` 与 `[M,K],[K,N]`。
QKᵀ/PV 则需要真正的 batched BLIS 入口，batch 1 的外层类型折叠不能消除 12 个 heads。

MLP 两层最终都保留 `(vdim,unroll)=(4,2)`：
2 行 × 8 列的 16 个 accumulator cell 沿 K 规约，没有 BLIS 的 KC/NC panel。
QKᵀ/PV 是输出向量维加 4-way reduction strip-mine，
不是“kind-1 请求失败后静默退回”。
seq512 MLP 的两层共约 4.832 GFLOP，126 ms 对应约 38 GFLOPS；
与既有 aligned BLIS 的约 95 GFLOPS 不是同一个 leaf。

已有 F32 BLIS recipe 的默认块为 `(MR,NR,KC,NC)=(4,16,128,256)`。
128/512/1024 行以及这些模型的 K/N 都满足相应块对齐；
M=128 不是这一入口失败的原因。
scalar bias/ReLU epilogue 已由既有 BLIS post 测试覆盖，也不是首个阻断条件。
要获得相同能力，需要 packed/transposed contraction 的识别与桥接、
带 batch 的 panel recipe，以及 view-layout 成本判断；本次不扩展这些能力。

## 复测接口

`model_bench.py --backends` 只接受可选 backend，eager 会自动作为 baseline。
因此 eager 与 MimIR 的组合应写 `--backends mimir`，不是 `eager,mimir`。
本次修复实际影响独立 LayerNorm；复测只选择该模型，并使用提交的完整 SHA。
延迟和完整命令保留在指定 scratchpad 的 `p5_work/seq512_<sha>`，
不把并发构建期间的小幅延迟差异写成调度收益。

## Task 25A：先识别 contraction，再选择面板布局

BLIS 的入口放在 `dot_schedule_kvec` 之前，而不是再识别一个已经 packed 的
rank-4 特例。`blis_candidate` 与 planner 共用 `blis_group` 的合法性检查和
`group_blocks` 的目标参数；不支持的 contraction、关闭 grouping 或 GPU
仍走原来的 dot 调度。结构请求仍由 group planner 生成、由作用域内的规则实现，
入口判断本身不安装调度。

这一顺序与本地源码中的做法一致：

- TVM `src/s_tir/meta_schedule/schedule_rule/multi_level_tiling.cc` 在原始 block
  上分块，再通过 `CacheRead` / `ComputeAt` 建立读取复用；
  `postproc/rewrite_layout.cc` 根据读取方布局插入 weight-layout rewrite block。
- IREE `DispatchCreation/AnnotateDataTilingHints.cpp` 先识别 contraction 的
  batch/M/N/K，再选择 `Mmt4DOp` / `BatchMmt4DOp`；
  `CPUPropagateDataLayout.cpp` 向上冒泡 pack、折叠 reshape。
- OpenBLAS `driver/level3/level3.c` 用布局相关的 `INCOPY` / `ITCOPY` 与
  `ONCOPY` / `OTCOPY` 读取面板，后面调用相同的 GEMM kernel。

因此转置不是另一种微内核。A 的 `(k,m)` 和 B 的 `(n,k)` 读取被表达成逻辑
operand view，面板 stage 组合这个读取映射，直接从原始存储复制到原有连续面板。
转置 A 也在 leaf 前 stage；B 的 NR 面板和已有微内核不变。
pure-read view 不再先被 slice 规则当成算术 producer 分块，
避免为裁剪后的 view 建立内层 stride 为运行时值的临时数组。
stage 的读取组合是通用的，不依赖 Linear、权重名称或模型尺寸。

`blis_transpose.mim` 覆盖 NN/NT/TN/TT、非整块的 M/N/K、bias/ReLU epilogue，
并在 grouping 开启和关闭时与标量结果比较；开启时要求四个 kind-1 请求。
已有 `hlo_train.mim` 同时覆盖共享 forward/backward contraction 的转置路径。

### Singleton 折叠后的 grouping 坐标

`2988413893` 的 transformer 错误发生在 `tensor.group` 的 `Cost::reads/visit`，
不是报错源码位置所指的 `dot_schedule`。
`«2, 1, 3; T»` 与 `«2, 3; T»` 是同一个 tensor 类型，
但 producer 的输出坐标仍有三个轴，reader 的逻辑输入坐标只有两个轴。
旧成本分析将 reader 的二维 box 原样传给 producer 的三维 `affine.preimage`，
在 seq512 transformer 中对应四维 box 与三维输出的冲突。

成本分析现在先检查 reader 的逻辑输入 shape 与 producer 的 `So` 是否相同，
不相同则拒绝该 grouping 候选并记录原因，不把类型折叠当作坐标恒等。
这不禁止两个算子各自调度，也不改变 BLIS contraction 的合法性。
跨不同逻辑 shape 的融合需要显式坐标桥接及 box 精确性证明，尚未在该成本路径实现。
`group_folded_rank.mim` 用两个 map_reduce 和一个 singleton 轴复现，
grouping 开关两种情况均与整数参考比较；原 seq512 transformer 已可生成 LLVM。

### Batched BLIS

BLIS 将最后两个 parallel 轴视为 M/N，其余 parallel 轴作为 batch 前缀，
将 reduction 轴按原顺序展平成 K。
operand view 保留原始物理 shape，通过 batch 坐标和 M/K 或 K/N 坐标组合原读映射；
epilogue 也收到完整 batch 前缀，因此每 batch 的 bias 不会被广播到其他 batch。
候选仍要求 literal shape、恒等输出坐标、纯 scalar combiner/epilogue 和默认 schedule，
且 A 不得读取 N、B 不得读取 M。
空 batch 或无法表示的 batch/K/panel 大小会被拒绝。

batch 使用 `tiled` 的运行时外层 grid，内部仍调用原有 `group_blis_impl`，
不改变 jo → K fold → B panel → io 的顺序或目标 block selector。
没有 batch 前缀时直接走二维 BLIS，恒等 operand 不另建 copy view；
这也保留 `group_fold.mim` 对 aligned kernel 无 `llvm.memcpy` 的约束。
`blis_batch.mim` 覆盖 NN/NT/TN、size-1 batch、多维 batch 前缀、
每 batch bias 及非整块的 M/N/K，grouping 开关均执行标量比较。
