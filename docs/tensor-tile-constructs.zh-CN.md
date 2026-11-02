# Tensor-level tiling constructs（Task 7 / destination lowering）

## 7. Cross-op grouping

### 7.1 Pure affine boxes

从 `feature/default-sched` 移植 `affine.image/cover/preimage` normalizers，
保留 symbolic origin、literal cap、clamp、exactness 与不 β-reduce 时的 residual axiom。
它们只依赖参数与 World，不读取 schedule、uses 或 placement。
`btensor::falling` 复用 image 的单点差分判断反向读取轴。
`lit/affine/image.mim` 检查静态结果与 opaque-map residual；
`image_exec.mim` 对运行时 origins 枚举实际 domain，验证 image/cover 包含、
exact axis 的覆盖，以及 preimage 的包含和 exact box 的反向包含。
本次不移植 default-sched 的 Sched 字段或 C++ placement。

### 7.2 Producer tile equation

`tile_of (producer, off, ext)` 的 reference 为 `tile_of_ref`，即 unit-stride slice。
normalizer 仅从 MRP operands 计算 write preimage，要求 exact 且所有 reduction dims 保留；
随后用 image 计算每个 input box，slice operands 并平移 read/write maps。
post input 的 image 与平移使用 output box `off/ext`，不使用 loop-domain origin。
tensor outs 有独立 box/map；init 在 reduction index 为零处读取，
因此空 reduction 仍读取 outs seed，而不是对空 reduction domain 求 init image。
无 input reads 的空 box 保留原 operand，避免产生无意义的零形状 slice。
只接受可还原为 `mk_sched` 的 canonical chooser，不猜测自定义 nest 的执行域。

实现不读取 graph uses 或盈利信息；原始 slice 与重建 MRP 的等式独立于预算。
不满足前提时保留 axiom，`lower_tensor` 发出原因并执行 reference slice，
不会把未证明的 box 当作精确 tile。`tile_of_exec` 对 runtime origin、permutation、
shifted write、带 post input 的 row reduction、空 reduction/tensor outs，以及 tailed tiled
分别检查 scalar/reference 结果；IR 测试检查正常重建与 inexact/reduction-cut 的拒绝。

#### Literal-cap storage

逻辑 `tile_of` 仍具有 `«ext; T»` 的 slice 类型，不能把裁剪后的有效 extent 当成物理 stride。
重建的 MRP 通过 `btensor.clip_sched` 携带有效 parallel 范围和 output/loop caps；
这是 box witness，不扩展两字段 `Sched`，也不改变 canonical chooser 的 root 判定。
实例化后 bufferization 使用 literal caps 构造结果 buffer 与循环域，
有效范围只控制叶子的 init、input read/combine、post read/write。
空 reduction 仍执行有效 cell 的 init/post，不把 reduction extent 纳入有效性判断。
未裁剪的 nest 使用常真谓词，partial evaluation 消去 wrappers。

若 symbolic extent 使 affine cap 也未成为 literal，以源域的 literal cap 作保守上界；
最终仍无 literal 上界则明确报错，不向 LLVM 传递未知 inner stride。
input/post/outs 的 box slices 保留给 recursive fusion Rule 匹配；
bufferization 在读取时合成这些纯 view 的 maps，不分配裁剪后形状的中间数组，
且不穿过 stage。消费者的 buffer bounds 和元素 index 使用物理 cap，而非逻辑 extent。
reference tiled 的逐元素读取也做相同的 bound conversion。

`tile_cap_exec` 保留原始 column-sum 复现，检查结果 `42, -84, -54, 54`，
并覆盖不整除尾块、同一 capped tensor 同时作为 input/outs/post，以及两条 tiled pipeline。
辅助程序用 ASan 构建；existing tile_of 和两种 Rule frontend 的递归融合检查不变。
该复现还触发 SEO 对 universe 的空 type 解引用；先检查 closedness 后才查询 memory type，
避免在无法拥有 free variables 的 universe 上调用 `isa_mem`。

### 7.3 Closure-aware use index 与预算 witness

每次 Rule phase 在改写前建立一个只读 program GraphView；
从全部 externals 追踪可达 mutables，把所有函数 body 纳入同一快照。
单纯捕获 producer 不记为 data read，callback body 内的实际读取仍被索引。
`only_used_by` 保持三值证明；未注册操作、opaque callback、直接返回 producer 等
无法完整描述的边界仍为 Unknown，不把缺失的 consumer 数量当成零。
`rule_app_with` 与 `graph_rule_app_with` 的 guards 使用同一类 program facts，
graph frontend 的局部 view 只负责匹配与发布。

插件声明 semantic data operands 和 region callbacks；tiled 记录 grid trip count，
tile_fold 记录 block count。occurrence 保留整个 region 路径和乘法后的动态次数；
未知 count、乘法溢出或递归 callback 不会变成有限次数证明。
`within_budget` 在 scratch World 逐轴枚举 full/tail extent 组合，精确累加
每个 slice 的 cells×trips，并给完整 producer reader 收取 whole-size 成本。
sum 不超过 budget×whole-size 才返回 true；literal rank/shape/count 不足时拒绝融合。
没有 diamond realization cache，也没有把静态 consumer 数直接当作执行次数。

`tensor.fuse_producer (T, rank)` 生成低 universe 的专用 Rule，
避免修改已有 compile phase 的类型接口；rank specialization 同时消去 singleton Pack。
Rule 保留 contraction/custom-schedule roots，并只接受已证明纯的 scalar combiner/post。
对 tiled callback 重复应用 Rule 和 cleanup，直到没有新的 input slices 可继续融合。
graph frontend 的显式 purity 声明仍是调用方的证明责任；
native tiled/fold decoder 不因拥有 region metadata 就默认承诺 callback 纯。

`tf_group_budget` 在两个前端检查 tails、nested tiled、重复 fold、未知 count、
完整 reader、直接返回和两个独立 slice 的累计预算。
`tf_group_exec` 检查两级 producer chain 与 nested tails 的实际数值，
并与未分块的完整 producer 对照；IR 另检查 chain 中没有 whole-size MRP。
unit tests 检查跨 callback 独占性、opaque callback、递归、未知次数和溢出。

### 7.4 Grouping strategy

`group_outer (P, tile_size)` 是按结果类型专用的 Rule factory：
native `group_candidate` 从同一 program snapshot 计算 grouped/reaches，
纯 `group_build` 根据 literal output shape 构造 outer-axis tiled 和 `tile_of`。
策略依次运行 contraction 的结构 Rule（例如 BLIS pack）与 group_outer，
再把 fuse_producer 和 cleanup 运行到 fixed point；不新增 pipeline 或 Sched 字段。
outer axis 是 canonical nest 的第零 parallel 轴；它本身是 vector loop 时不分组。
输入已被 tiled/fold 内的 slice 接管时，不先把该 producer 独立分组。
未知 shape、空 output、自定义 chooser 和 region-local producer 保持原状。

`tf_group_strategy` 对两个 frontends 同时检查 IR 和数值：

| default-sched 用例 | 此处的决策 |
| --- | --- |
| group_exec 的 row reduction → column reduction | 相同：每 column 融合 producer，无 whole `6×4` MRP |
| group_exec 的 contraction → column reduction | 相同：contraction 保留 whole；结构 root 由自己的 BLIS Rule 处理 |
| group_exec 的 Q → B → D，D 重复读取 B 四次 | 相同：budget 拒绝 B→D；B 自己分组 Q，无 whole `6×5×3` MRP |
| compute_at 的 nested chain / tailed placement | tf_group_exec、tile_cap_exec：recursive fusion 和 tail 的数值一致 |

没有照搬 reference 分支的 action/prepack/cache/placement bits：本分支没有这些 record
字段，stage 是显式 barrier，custom chooser 保守作为 root。
不支持 reference 的 reverse placement 和跨调用 prepack；halo 的显式 opt-in 路径见 §7.7；
也没有把其支持的任意 loop order 推广到本分支的 canonical outer-axis strategy。
reference 会先标记每个 eligible op；此处先选择 terminal group，再在后续快照处理剩余
whole producer，避免既给 producer 独立分组又在消费者中重建它。
具体 IR 容器不同（reference 为 placement fields，此处为 tiled/tile_of），上述表比较的是
动态 producer realization 和数值，不声称 LLVM 文本或 scratch allocation 个数逐字相同。

source clipping 依赖 symbolic origin 时，预算先尝试 full/tail 组合；
仅在其不能求值时枚举 literal origins，工作量上限为 65536 个叶子，超限仍为 Unknown。
此处的 importer 只替换未绑定的 free variables，不用 bottom 覆盖 dependent lambda 的 binder。

### 7.5 BLIS 的 structured epilogue

`blis.mim` 的 `pack_post` Rule 把 product+bias+ReLU marker 改写为
`packed_with_post`；普通 `pack` 通过关闭 epilogue 的 wrapper 保持原接口。
fold 中所有完整 K blocks 的 `nps=0`，不读 bias，也不运行 ReLU；
fold 后显式的最后一个 block 才选择 `nps=1` 的 post。
bias map 使用全局 `NC off + NR row + local column`，不是 panel-local column。
MR/NR/NC/KC 均可有 tail；bias 在写回之前只加一次，不改变 ordered K fold。

`blis_post_exec` 对 rule/hand、destination/reference 两条 pipeline 验证
8³、13×7×11、1×7×1、5×9×19、5×11×21；输入包含负值和多 K blocks，
因此逐 block ReLU 或重复 bias 不能通过检查。
benchmark 使用 `build/task8-bench.py --epilogue`，target 与 task11 相同，
F32、单线程、native `-O3`、1 次 warmup/数值检查 + 7 次计时取 median，含 allocation/free。
unfused 为同一 BLIS GEMM 后的独立 bias/ReLU loops；OpenBLAS 也运行同一 loops。

| M×K×N | BLIS fused post | BLIS + loops | OpenBLAS + loops |
| --- | ---: | ---: | ---: |
| 256³ | 90.718 | 92.172 | 80.154 |
| 512³ | 97.543 | 96.587 | 100.197 |
| 1024³ | 96.865 | 95.896 | 98.354 |
| 513×257×129 | 87.302 | 86.141 | 88.046 |

单位 GFLOPS，只计 GEMM FLOPs。整表来自第二次完整执行
`build/task12-bench/epilogue-confirm.csv`，不是逐项挑最好值。
首次执行的 256³/512³ 三种实现均较慢，原始数据保留在 epilogue-results.csv；
不把小形状的执行间波动解释为融合本身的收益。

### 7.6 验收负载与 MLP 的 realization 边界

`blis_workloads` 在 destination/reference 两条 pipeline 上做标量结果对照，
并在仅运行 Rule 的 IR 上检查实际 grouping 决策：

| 负载 | 决策 |
| --- | --- |
| 13×7×11 GEMM → ReLU → 13×11×5 GEMM | 两个 contraction 都保留 root；第一个的最后 K block 执行 ReLU |
| elementwise A(5×3) → GEMM(5×3×11) | producer 融入 A slices，消除 whole A 的 elementwise MRP |
| elementwise A(5×9) → GEMM(5×9×19) | 多个 NR/K regions 的预算拒绝融合，保留 whole producer |
| row sum / row max(5×7) → consumer(5×11) | consumer 按两行分块；每个块内计算对应 reduction，原 seven-element reduction 顺序不变 |

`math.extrema.*` 与已有的 core extrema 一样，是纯 scalar callback，
因此 native purity witness 接受 fmax；没有放宽未知 axiom 或 effectful callback。
row consumer 计算 `b[i,j] - reduction(a[i,:])`，覆盖 softmax 的减 row-max 步骤，
不是完整的 exp/normalize softmax。
多 K-block 的 elementwise A 仍可能被预算保守拒绝；未通过提高 budget 或忽略 capture 来强制融合。

`build/task8-bench.py --grouping` 生成六种 shape 的 GEMM+bias+ReLU 与 MLP。
MLP 第一层为 M×K 乘 K×N，第二层为 M×N 乘 N×N；bias 置零。
单线程、native `-O3`、1 次 warmup/数值检查 + 7 次计时取 median，包含 allocation/free，
GFLOPS 只计两次 GEMM 的预期 FLOPs。所有实现先与 OpenBLAS 结果逐元素比较。

| M×K×N | Rule post | BLIS + loops | OpenBLAS + loops |
| --- | ---: | ---: | ---: |
| 256³ | 84.618 | 85.625 | 76.073 |
| 512³ | 79.726 | 91.990 | 96.228 |
| 1024³ | 94.176 | 95.679 | 96.434 |
| 2048³ | 95.113 | 94.848 | 101.718 |
| 1000³ | 95.845 | 96.466 | 100.829 |
| 513×257×129 | 86.396 | 88.513 | 90.405 |

| M×K×N | Rule MLP | 独立 BLIS GEMM + ReLU + GEMM | OpenBLAS + ReLU + GEMM |
| --- | ---: | ---: | ---: |
| 256³ | 61.373 | 93.212 | 75.340 |
| 512³ | 61.984 | 95.919 | 102.217 |
| 1024³ | 64.358 | 93.577 | 95.065 |
| 2048³ | 61.425 | 91.520 | 103.652 |
| 1000³ | 62.336 | 92.630 | 102.359 |
| 513×257×129 | 53.554 | 85.677 | 85.647 |

整表为 `build/task12-bench/workload-final-confirm.csv`，此前完整运行保留在
`workload-final-first.csv`、`workload-final.csv`；未逐项选择最快值。
小形状的 post 数据有明显执行间波动，不能概括为稳定收益。

MLP 算对，但没有实现跨 contraction 的 realization 共享，也没有获得性能收益。
首个 GEMM 的结果 h 同时被第二层的 tile_fold 和显式最后 K block 捕获；
`lower_tile_fold`、`lower_tile_body` 各自重置 fresh-memory memo，两个 scope 分别 materialize h。
256³ 的优化 LLVM 中，独立 GEMM 各有 8 个 `fmul` sites，MLP 有 24 个；
这还不足以证明动态重算，因此 `build/task12-count-work.py` 对 optimized LLVM 的
每条浮点乘法按 vector lane 数插桩计数，并再次检查输出：
两次独立 GEMM 分别为 16,777,216 次 scalar multiplies，
合并 MLP 为 50,331,648 次，而预期为 33,554,432 次。
证据保留在 `build/task12-bench/multiply-counts.log`。
benchmark 使用未插桩的可执行文件；乘法计数只用于归因，不用于计时。

消除该差距需要把 h 的 realization 置于两个使用 scope 的共同 dominator，
并给出 lifetime / memory sequencing 证明；仅为 fold 加一组 invariant captures
没有消除重复计算，该尝试未保留。
本任务不实现 diamond realization cache，因此这里明确保留限制，
不把“两个 contraction 都匹配了 BLIS Rule”称为两个 GEMM 间的高效融合。

### 7.7 Task 13B：convolution halo

保留原来的 `tile_of` / `fuse_producer` exact 路径和 budget=1 策略。
halo 使用带 box witness 的 `tile_of_box (p, off, ext, cap)`；
其值仍是 `tile_of_ref (p, off, ext)`，`cap` 不是逻辑 shape。
调用者须证明逐轴 `ext ≤ cap`，并保证 producer 在所计算的 over-box 上纯且有定义。
已知 literal extent 超过 cap 时不重建 producer，lowering 明确诊断后回退 reference slice。
最终实例的 cap 必须为 literal，否则仍由现有 storage-cap 检查报错。

例如按输出行分块时，外层仍是普通 `tiled`，body 写：

```mim
λ (off ext: «4; Nat»): «ext; I64» = tensor.tile_of_box (p, off, ext, ext)
```

producer 的 write preimage 仍须 exact，reduction dims 仍须完整保留。
这里放宽的是 **input read image**，不是任意 inexact write map：
`oh * stride + kh * dilation` 的 bounding box 可包含没有实际读取的空隙，
相邻 output tiles 的 input boxes 也可重叠；producer 在整个有效 over-box 上重算。
单轴未裁剪 cap 为 `(tile_rows − 1) * stride + (kernel_rows − 1) * dilation + 1`，
再取 producer 域的上界；不把实际读取点数冒充 box 体积。

逻辑 input box 用实际 clipped loop extent 求 `affine.image`；
物理 cap 另用 loop cap 求 image 的第三字段，不能用前者的 runtime extent 推导。
两个域共享同一 read map 和 origin。盒子通过
`slice_box (shape, (off, ext, cap), p)` 保留到递归 Rule：
它在值上只是 unit-stride slice，但不丢失 literal-cap witness。
`tile_of_box` 将该 witness 沿 write preimage 传到 `btensor.clip_sched`，
buffer 和循环使用 cap；依赖 runtime offset 的有效部分只在叶子执行 init/read/combine/post/write。
消费者的 read map 仍减去实际 box origin，不能用 cap 替代逻辑坐标。
未融合的 `slice_box` 走普通 view lowering，不穿过 `stage`。

决策是显式 Rule factory `fuse_halo (T, rank, budget)`，匹配 `slice_box`，
以同一个 staged `within_budget` witness 决定是否改写成 `tile_of_box`。
它允许纯 canonical contraction producer 被重算，因而不复用 exact 策略的 contraction-root veto；
这不会自动放开 `group_outer` 或默认 pipeline。
`within_budget` 对每个 occurrence 累加实际 over-box cells，包含重叠与 dilation 空隙；
同一 producer 的 reduction 域完整不变，因此重复 output cell 也重复其整段 reduction。
whole reader、nested tiled/fold、Unknown/overflow 和 65536 次枚举上限沿用原合同。
graph frontend 仍须显式声明本负载的 MRP、pad、schedule 和 affine expression 的纯性。

**Padding 选择保留 whole `tensor.pad`，只切片其 consumer。**
padding 不通过饱和减法伪装成 affine input index，也不把域外位置当作可读取的 producer cell。
验收负载先对 NCHW input 做 whole zero-pad，再计算保持零值的 elementwise scale；
因此 scale 可以在每个 conv tile 的 halo 内重算，pad 本身只 materialize 一次。
这与标量 reference 中先 scale 再 zero-pad 相等。
若 elementwise producer 位于 pad 之前，或两层 conv 之间有 pad，本规则不跨过该 pad；
producer 留 whole，不声称已实现局部 padding 或一般带条件的 map。

| 回归 | 覆盖 |
| --- | --- |
| `tf_halo_exec` | NCHW conv → bias → ReLU，stride 1/2、padding 1、dilation 2、不整除两行 tile、一行 tile；valid conv → ReLU → conv 的两行 grouping |
| `tf_halo_budget_one` | 长度 7 的 producer，三点 window、两行 tile，over-box cells 为 `4+4+3=11`，budget 1 拒绝 |
| `tf_halo_budget` | 同一请求 budget 2 接受；一行 tile 的 `3*5=15 > 2*7` 拒绝；whole reader 的 `11+7` 和未知 fold count 拒绝 |
| `tile_of_box_exec` | runtime origin 对应有效长度 3/2/1、固定 cap=3；七元素域按三元素 tile 分块 |
| `Tensor halo` unit | symbolic valid extent 与 literal image cap=4 分离；递归 producer 的 output/loop caps 均为 4；原 exact path 仍使用普通 slice |

两个 Rule frontends 均做 IR 决策检查；五个 conv 负载各在 rule/graph destination 路径和
rule reference-tiles 路径对照独立标量实现，辅助程序启用 ASan。
同一组负载还以 budget=0 执行 whole-producer fallback；
`slice_box` 保留到 bufferization，避免 L1 先把 scale 融进丢失 cap 的 symbolic-shape slice。
bias/ReLU 在 reduction 完成后执行一次，不改变 MAC 顺序。
pad、weights 和 bias 留 whole；IR 检查 scale 和第一层 conv 没有 whole-size MRP。
本任务不运行执行时间 benchmark，也不实现共享 realization cache。

限制仍包括自定义 schedule chooser、切断 reduction 的请求、非 exact write preimage、
没有最终 literal cap 的动态 shape，以及 producer 在 over-box 上未定义或有副作用的情况。
负载直接使用与 `tensor.conv` 相同的 NCHW/read/write maps 和完整七维 reduction，
但显式选择 canonical scalar nest；没有改写 `conv_schedule` 的 packed/vector heuristics。
该实现不是对任意已调度 conv 的自动 regrouping，也不是通用 inexact preimage 求解器。

本地 Debug 验证为 CTest **241/241**，完整 lit **644 passed、9 unsupported**
（共 653；没有失败或 XFAIL），新增五个 lit 与一个 unit case。
首次构建所要求的六个 targets 用时 **62.21 s**（wall）。
下表为同一文件的五个 external 负载从 Mim 到 LLVM 的单次编译 wall time，
不包含 clang 编译/链接、执行或计时循环；共享机器上的单次数字不作为性能结论。

| 编译模式 | wall time |
| --- | ---: |
| rule、budget=0，producer 留 whole | 0.564 s |
| rule、budget=3，halo destination | 0.533 s |
| graph、budget=3，halo destination | 0.577 s |
| rule、budget=3，reference-tiles | 0.574 s |

原始编译输入、日志和结果保留在 `build/whole-rule.*`、`build/halo-{rule,graph,reference}.*`
及 `build/halo-compile-results.json`；脚本为 `build/halo-compile.py`。

## 8. Multi-result tiles（Task 13A.1）

`tiled_n @(n, g, Ts, rs) (shapes, grid) boxes body` 在一个共享 grid 上产生 n 个结果。
grid 给出每轴的 point 数量，不是元素总数；point 为 row-major 的 Nat 坐标。
`boxes point` 的第 i 个 slot 是 `[off, ext]: TileBox (rs#i)`，
body 在同一 point 返回 `«i: n; «(boxes point)#i#tt; Ts#i»»`。
不同 slot 可以有不同 rank、shape、元素类型和 owned box。
每个结果各有一个全尺寸 destination，point 顺序为 row-major，point 内 slot 顺序为求值顺序。

当前 ownership proof 接受 literal rank/shape/grid、逐 point 可 β-reduce 为 literal 的 boxes；
最多枚举 65536 个 points，超出或未知时明确诊断，不猜测 ownership。
每个 owned box 必须位于对应结果内；同一结果的非空 boxes 必须两两不交，
其体积总和必须等于结果体积。由此每个结果元素恰好写一次。
空 owned boxes 不执行相应 slot；空 grid 只能产生空结果；
rank-0 的标量结果只有一个拥有者。零 result count 允许空 tuple。
此版 destination 显式展开 grid，尚未把相同 extents 的 points 压回运行时 loop；
因此大型 grid 的编译规模和 reference specialization fuel 仍有限制。

reference `_impl` 按同一 grid 更新不可变结果 tuple：
每步调用 body，将各 owned box 的元素插入对应旧结果。
成员测试同时检查 `global >= off` 和 `global - off < ext`，不能依赖 Nat 减法下溢。
destination 路径只分配各结果自己的 destination；
按 slot 顺序用 `lower_into` 写入 owned box，再将该 slot 的原 tile 值替换为结果的只读 slice，
后续 slots 从已写结果读，不重新计算该 tile。
memory token 显式依次经过所有 slots；不同 point 的 owned 区域不重叠。
普通 fallback 仍可能需要 temporary+copy，不把它声称为 in-place producer。

`tiled_n_exec` 对 reference/destination 验证：
5×3 的 doubled producer 与每行 sum 两个不同 rank 结果、tail、whole-reader、
空 grid、rank-0/singleton，以及 2×3 grid 上 I64/I32 两个转置 ownership 结果。
`tiled_n_invalid` 在两条 pipeline 拒绝重叠、缺口、越界、空 grid 非空结果。
graph registry 把 body 登记为 region 并提供 literal trip count；
尚无 tiled_n 专用 cost witness 时，fusion budget 保留 Unknown。
没有修改 tile_of 的 normalizer。

## 9. Exact realization cache（Task 13A.2）

`lower_to_mem` 在每个实际 lowering scope 建立只读请求计划。
同一 producer 的完整表达式按 `Check::test_alpha` 分组：内部 binder 可以 α-renaming，
自由变量必须相同，element type、stage layout、box offset/extent、clip predicate 和 schedule
都是 key 的一部分。只共享完全相同的请求，不合并不同 boxes，也不推断两个算法数值等价。
这也覆盖 fold specialization 克隆出的等价 producer，而不只依赖节点指针。

计划只穿过已定义的 tiled/tiled_n/tile_fold/generate region，且跨越的 region
必须有正的 literal trip count；未知、空 region 和 opaque callback 不提供 hoist 证据。
每组在包围全部 readers 且 producer 自由变量全部可用的 dependency scope 请求 realization；
不依赖内层 tile index 的 producer 可以提到外层，避免每个 microtile 重填整个 producer。
未达到该 dependency scope 的请求由内层 lowering 再规划。
依赖 tile origin 或 fold offset 的值因此在对应 iteration 内重新 fill，不跨 iteration 缓存。

选中的 buffer 在该 scope 的 fresh-memory continuation 中完整 fill 后才绑定给所有 aliases。
reader 共享只读 buffer，不向它写回；每次 realization 使用独立 allocation，没有循环复用
同一 scratch 的覆盖风险。共享值仍交给现有 `mem.free_locals` 做 last-reader lifetime，
不能逃逸的 allocation 可继续由 heap2stack 转为局部 storage；没有新增全局缓存或永久存活的 storage。
physical cap metadata 随 enclosing scope 继承，避免 clipped tensor 的逻辑 shape 被误当作存储 stride。

`realization_exec` 对 sibling readers、嵌套 tile iteration、不同 boxes、相依 producer、
多次改变输入分别执行 destination/reference 校验；另对优化 LLVM 插入动态乘法计数，
检查 shared/nested 各 8 次，而不是把静态 allocation sites 当运行次数。
不同 boxes/producer 的拒绝共享由数值对照检查，不锁定 LLVM 可继续化简的算术次数。

### 9.1 MLP：specialization 与 eager capture（Task 13A.3）

只合并原始 region 中的相同节点还不够：fold 的 first/remaining step 会克隆闭包，
且 outer tiled body 的 inner shape 在 β-reduction 前依赖其 extent 参数。
已有 capture_producers 会先于该 reduction materialize invariant producers。
现在这些已经发生的 pure captures 也登记到 scope-local template cache；
后续 specialization 以严格 α-equivalence 查询，继承外层只读 storage，
退出 region 时恢复 cache，不把不同 iteration 的 scratch 留给下一次 iteration。
没有借此新增对未知 trip count region 的 speculative fill。

256³ 的动态 scalar-equivalent 乘法数为 33,554,432；512³ 为 268,435,456，
都恰为两个 GEMM。后者覆盖 first peel 后仍有 remaining iterations 的情形；
只验证 256³ 会遗漏旧实现仍多计算 100,663,296 次的情况。
`blis_workloads` 的两种 K extent 在优化 LLVM 插入动态计数，并与 scalar/reference path 对照。

M2 Pro，F32，Homebrew clang++ `-O3 -mcpu=native`，单线程，
1 warmup + 7 runs 取中位数，含 allocation/free。两轮计时前均检查无 mim/clang build。
复测结果（GFLOPS，`build/task13-mlp-confirm.csv`）：

| Shape | bias+ReLU / separate / OpenBLAS | MLP / separate / OpenBLAS |
| --- | --- | --- |
| 256³ | 89.033 / 87.581 / 75.644 | 90.205 / 90.601 / 78.697 |
| 512³ | 94.975 / 88.600 / 99.553 | 93.789 / 90.429 / 98.995 |
| 1024³ | 91.445 / 93.367 / 93.777 | 90.048 / 90.648 / 95.568 |
| 2048³ | 89.889 / 90.127 / 98.517 | 89.902 / 90.044 / 98.809 |
| 1000³ | 91.109 / 92.711 / 100.433 | 89.332 / 89.356 / 97.886 |
| 513×257×129 | 85.322 / 83.807 / 87.088 | 85.737 / 85.528 / 87.312 |

首轮保留在 `build/task13-mlp-final.csv`：MLP 为 86.957 / 88.410 / 94.154 /
89.376 / 89.188 / 86.505，separate 为 88.798 / 92.801 / 91.958 / 89.991 /
89.376 / 85.124。两轮的胜负有反转，不声称每个 shape 都稳定快于 separate；
消除重复工作后整体达到 separate 的吞吐级别，复测差值约 −0.7%..+3.7%。
这里仍 materialize 第一层的完整 hidden tensor，不是消除 hidden storage 的 streaming MLP。

### 9.2 单个 row region 的 realization placement（Task 14A.1）

5×7、row tile=2 的 softmax 复现中，lower_to_mem 的 exp MRP 和 stage 请求分别是
同一个节点，第二次查询命中同一 scope 的 memo；并不存在两个需要合并的不同 box key。
bufferization 后仍只有一个 exp MRP。重复发生在 btensor lowering：
scalar-init MRP 的 `readInputs` 直接捕获 producer 的 direct-style 表达式，
CPS 展开时 producer 被带到 consumer 的 cell loop 内，LLVM 优化后仍执行 63 次 exp。
原始 LLVM 的重复更严重，不能只依赖 LLVM 的循环外提判断 exactly-once。

scalar-init 和 callback-init 路径现在与已有 accumulator-descriptor 路径一样，
在 nest 入口用 `ds_splice` 绑定 memory、destination、inputs 和 post_inputs；
reader 闭包只捕获 runtime parameters，不再捕获可重执行的 producer 表达式。
这保留现有 exact/α-equivalent cache key，没有放宽任何 callback 的 purity/region 识别，
也没有把不等 box、layout 或 boundary predicate 当作相同请求。

此前扩大 request 扫描触发的类型错误被避免而非放宽类型系统：
generic tiled body 的返回类型仍是依赖 extent 的 value array，
`conv_type` 不把这种 direct-style callback 改成 buffer ABI；
在其 body 上提前安装 array→Buf 的 realization mapping 会让函数返回 Buf、
声明却仍为 array。现有 `lower_tile_body` 在 literal tail specialization 后再 bufferize，
本修复不改变这条边界，也不提前 rewrite generic callback。

`row_norm` 保留该复现，destination/reference 均与 scalar softmax/layernorm 对照，
包含不整除 row tile、singleton 轴、空 rows、常量输入和大幅度有限输入。
5×7 softmax 的 exp 计数为 35，1×11 为 11，空 rows 为 0；
singleton-column 可化简为 1 而没有 exp。分别在 LLVM 优化前、优化后插入动态计数，
防止优化器掩盖 producer 重算。count callback 只存在于测试产物。

### 9.3 Layernorm 的 shared centered tile（Task 14A.2）

同一 row tile 内先求 mean，再写 `centered = x - mean` 的 stage，
variance 与最终 normalize 都读取这一份 centered buffer。
§9.2 的 nest-entry binding 同时保证该路径的 realization 不被带进 reader 的 cell loop；
不需要第二套 key、额外 schedule 字段或新的 callback 规则。

`row_norm` 在 LLVM 优化前对 F32 `fsub` 插入动态计数；
layernorm kernel 中唯一的减法是 centered tile 的逐 cell fill。
5×7、1×11、7×1、0×7 的计数分别严格为 35、11、7、0，
包括 tail tile，并在三次改变输入后重复检查。
这验证每个 centered cell 只计算一次，而不是仅检查同一静态 stage 节点存在。
mean/variance 使用完整 row reduction，epsilon=1e−5，scalar reference 用 double 累加；
destination 与 reference-tiles 两条路径均对照有限、大幅度及常量输入。

### 9.4 Row normalization benchmark（Task 14A.3）

复现：`python3 scripts/bench-row-norm.py` 生成
`build/task14-row-bench/bench`，确认机器没有 mim/clang/cmake/ninja build 后单独运行。
M2 Pro，F32，Homebrew clang++ `-O3 -mcpu=native`，单线程，
1 warmup + 7 runs 中位数，包含 allocation/free。
每个路径在 warmup 时与 double scalar reference 对照全部输出。
unfused 是四个独立导出的 Mim kernel，由 C++ 顺序调用：
softmax 为 rowmax/exp/rowsum/div，layernorm 为 mean/center/variance/normalize；
各步之间有完整 tensor，不允许跨函数的 tensor fusion。
grouped 使用相同 combiner、schedule 和计算定义，在一个 row region 内共享 exp/centered。
row tile 为 `max(1, min(16, 32768 / (cols * 4)))`，不是按结果挑选最快 tile。

确认轮结果（ms，`build/task14-row-bench/confirm.csv`）：

| Rows×Cols | row tile | softmax grouped / unfused | speedup | layernorm grouped / unfused | speedup |
| --- | --- | --- | --- | --- | --- |
| 1024×1024 | 8 | 2.365 / 1.814 | 0.767× | 0.458 / 0.295 | 0.644× |
| 4096×1024 | 8 | 8.279 / 8.119 | 0.981× | 2.351 / 1.527 | 0.649× |
| 1024×4096 | 2 | 8.024 / 8.481 | 1.057× | 1.564 / 1.491 | 0.953× |

首轮保留在 `results.csv`：softmax speedup 为 0.745/0.904/0.941×，
layernorm 为 0.756/0.734/0.973×。两轮前及确认轮后均未发现 compiler/build 进程；
不把最后一个 softmax shape 的胜负反转解释为稳定加速。
exactly-once 已解决，但 grouping 在这些尺寸通常仍慢于 unfused。
生成汇编 `probe.s` 显示 grouped layernorm 的每个 row tile
仍有两次 32KiB malloc、center stage 的逐行 memcpy 和两次 free，
而独立 kernel 的 transform 输出直接交给后两步，不需要这个 identity-layout stage copy。
这些开销及小 tile 的固定控制开销没有被 realization 去重消除；
softmax 还保留 scalar `expf` 调用。这里没有声称 grouping 已获得性能收益，
也没有为计时删除用于检验共享的 stage 边界。

## 4. Destination tiled lowering（Task 8）

默认 `lower_tensor` 保留 tiled，`lower_to_mem` 顺序遍历 tile grid；
每维 full extent 与 tail extent 分开生成 callback，body 实例的 ext 均为 literal，
singleton axes 仍按 logical rank 传递。outer destination 只分配一次。
root mrp 的 map_out 加 composed origin，post read maps 减 origin 后使用原 map；
nested tiled 使用同一 destination 并累加 origin。其他 root 先算 tile buffer，再 copy
owned extent，warning 明确报告该 fallback，不把临时结果当作 zero-copy。

`-X tensor:reference-tiles=1` 保留原 tiled_impl/generate 路径。
tile_constructs_exec、nested_stage_exec、pgemm_exec 都分别编译执行两个路径并对照同一
scalar reference。tiled_destination 检查一次 grid iteration 的 root mrp 不在 per-element
generate 中，直接写 whole destination，没有 tileCopy。
下面的 fold/stage 使用 destination 路径；reference option 保留三者原有语义。

### 4.1 Ordered destination fold

tensor accumulator 的 tile_fold 在 loop 前分配一个 buffer 并复制 init，不能修改 caller
传入的 init。每次 step 携带 `[mem, acc_buffer]`，offset 为 iteration × ks。
root mrp 或 nested tiled 在本地证明通过后写同一 accumulator，不每步复制。
证明要求 acc 只出现在 outs，输入/combiner/post/map 无额外读；write map 为 leading
parallel coordinates、parallel extents 等于 local output extents，且 init map 与加 origin
后的 write coordinates 相同。这同时排除非单射 write 和读取其他 cell 的 alias hazard，
不能把“一个 outs use”误当作 pointwise read-before-write 证明。
unit-stride slice 的 affine semiop.mul(x,1) 仅在证明中消去，不改生产 normalizer。

未证明时整个 step 先算到临时 buffer，然后复制回 acc，并 warning 报告。
fold_destination 检查 linear step 内无 buffer.copy；tile_constructs_exec 对额外 acc input
和所有 cells 读取 acc#0 的两个非线性例子验证 fallback，两个 pipeline 数值相同。
pgemm 的完整 K-block fold 均通过证明，没有每 step 的 accumulator copy。
纯 scalar fold 保留原 ordered `_impl`；reference-tiles=1 保留全部旧 reference folds，
fresh_before/fresh_after 显式选择该路径继续测试 pinned callbacks 的 late memory threading。

### 4.2 Dependency-scoped stage realization

默认 stage 在 lower_to_mem 才展开其 reindexing 为一个 allocating fill loop。
进入 tiled 前收集 body 捕获的 stage：只有其全部 free Vars 已是 body 的 captures 时
才在该层 materialize，不能提升依赖 tile off/ext 或内层 fold 参数的 stage。
产生的 buffer 作为显式运行时 operand 传给 tile loop，per-tile rewrite 的 scoped map
复用这个 operand，不再为每个 row reader 重新生成 packing。
pgemm 的 B stage 因而在每个 column/K block 下、row tile loop 上方 realization 一次。
stage_destination 有三次 buffer.alloc：整个结果、裁剪输入和共享 stage，无每 tile allocation；
generate 的显式输入 buffer 同时保证 crop 不在 stage 每个 reader 中重算。
stage_barrier 分别检查默认 stage 边界和 reference generate 边界。

`reference-tiles=1` 同时选择旧 tiled/fold/stage `_impl`，布局/padding 与 singleton tests
在两条 pipeline 均执行验证。已有 mem.free_locals 根据最后读和控制流边释放不逃逸 allocations，
不再增加一套只看语法 scope 的 free 判断。GEMM LLVM 中 crop 在 packing 后释放，K-block stage
在 row tile loop 出口释放，fold acc 在 final block 写回后释放；output 由 caller 释放。
escaping/unknown readers 仍保守处理，没有同 extent 下跨 iterations 的 scratch buffer reuse，
persistent stage/resource identity 也未实现。

### 4.3 Destination/reference correctness checks

tile_constructs_exec、nested_stage_exec、pgemm_exec 和 outs_exec 都执行两条 pipeline。
tiled_destination 检查 grid 内的 MRP，不经每 element generate；fold_destination 区分
loop 前 init copy 与无 copy 的 linear step；stage_destination 检查共享 fill buffer operand。
pgemm_exec 同时要求 destination fallback report 为零，不可退化为 step/tile copies 后仍通过。
非 pointwise map_init 与 acc 额外作为 data input 的两个用例强制 whole-step copy 后检查数值，
因此 no-copy proof 的拒绝路径也不是未覆盖的性能假设。

### 4.4 Captured producers 与性能验证

大 shape 运行暴露了两处仅靠 root destination 不能消除的工作：
stage fill 的 slice 输入会被 generate callback 重新算；final-block seed 的 slice callback
会重新算其捕获的 tile_fold。现在 tiled/generate 均把不依赖自身参数的 tensor producers
作为显式 buffer operands 捕获，并用 scoped rewrite map 传给 reader。
依赖自身 off/ext/index 或更内层 binder 的 producer 不提升，destination root 本身不先分配
另一个临时结果。captured_fold_destination 检查 fold buffer 被传入 generate，而非在每 cell 重算。
这也适用于 reference generate 的纯输入共享，但 reference tiled 的 per-element tile body
依赖 reader index，仍保留原执行方式。

fold 必须进行的 init copy 及非线性 step copy 使用已有 btensor pointwise loop builder。
没有修改所有 buffer.copy：其 lower_ptr 的整数组 load/store 在本实验会生成大量 aggregate
extract/insert conversion。切换这两处 copy 后，12-function LLVM 从 63,967,677 bytes
降为 351,803 bytes，再消除 reader-side captures 重算后为 285,941 bytes；
native clang compile 从 203.94 秒降为 0.80 秒。
消除 reader-side producer 重算后，1024³ rule 从约 0.16 提升到 7.60 GFLOPS。

实验用与 pgemm 相同的 generic M/K/N/tile rule，将固定元素类型改为 math.F32，
captured add/mul 改为 math.arith 的 fast mode；tile sizes 为 MR=32、KR=64、NR=32，
没有添加 bias/ReLU，也没有修改原 I64 regression。四种 shape 的 rule、hand、plain
和 cblas_sgemm 均对照完整 scalar reference，使用有限的 dyadic 输入与相对误差检查。
graph frontend 的 local packed RHS helper 限制未改变，本实验用 rule_app。

Apple M2 Pro，Homebrew clang 22.1.4，`-O3 -mcpu=native`；tile loops sequential。
OpenBLAS 显式 `openblas_set_num_threads(1)`，同时设 OMP_NUM_THREADS=1、
OPENBLAS_NUM_THREADS=1。每路径一次 warmup、七次测量取中位数；计时包括 output
allocation/free，Mim 内部 scratch fill/free 也计入，OpenBLAS 输入已在内存中。
GFLOPS 以 2×M×K×N 计算，不把 copying/packing 工作计为额外 FLOPs。

| M×K×N | Rule GFLOPS | Hand GFLOPS | Default product_2d GFLOPS | OpenBLAS sgemm GFLOPS |
| --- | ---: | ---: | ---: | ---: |
| 256×256×256 | 7.560 | 7.654 | 35.056 | 80.082 |
| 512×512×512 | 7.701 | 7.676 | 33.682 | 99.075 |
| 1024×1024×1024 | 7.596 | 7.484 | 32.856 | 96.279 |
| 513×257×129 | 7.383 | 7.356 | 41.370 | 88.880 |

12-function module 的单次 Debug Mim compile 为 0.534 秒，clang compile 为 0.795 秒，
LLVM 285,941 bytes / 6,494 lines；linked benchmark 52,936 bytes（包括 harness，不包括
动态 OpenBLAS）。1024³ rule/hand 各 570 LLVM lines / 24,990 bytes；tailed 各
1,254 lines / 52,760 bytes，不是 whole-module 数值冒充单函数 size。

完整生成器/driver、源 module、LLVM、测量 CSV 和 metadata 保留在 build，不增加 benchmark
到每次 lit：`python3 build/task8-bench.py` 生成/编译；随后运行
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/task8-bench/bench`。
driver 在 `build/task8-bench.cpp`，结果在 `build/task8-bench/results.csv` 和 `metrics.json`；
before-copy / before-capture 记录保留为同目录的对应 JSON/CSV。

本 GEMM 没有 destination fallback report，full K folds 原地累加，最后 block 写 outer
destination。但这不等于完全没有 copy：每个 column tile 初始化独立 fold acc；
B slice 和 stage 仍分别 materialize/fill。该 GEMM LLVM 中 A/outs slices 已融合为 read maps，
没有 A/acc tile allocation，但尚未把 stage input 的 B crop copy 合进 packing。
结构版本的叶子仍是经典标量 MRP nest，未接入 default C++ schedule 的 vector/register
blocking，因此低于 default 和 OpenBLAS，并未宣称达到同等速度。
后续需要并行 grid、register/vector leaf、specialization 后的 slice/read fusion、scratch
跨 iterations reuse，以及跨调用 persistent packing；GPU scopes 和 effectful stage 未验证。

## 5. Task 9 — leaf schedule 与 packing 优化

### 5.1 参数化 vector leaf

mark/pack 捕获 LeafSchedule callback，hand 调用传相同 callback；由实际 tile extent
(m, cols, k) 和 vec_width 8 reflection 选择。沿用 default non-transposed dot_schedule
的 column-vector accumulator 和四项 reduction unroll：将 K 分为 (k/kt, kt)，
kt 在可整除且目标 width ≥4 时为4，否则为1；cols>1、width>1 时 vdim=1，
singleton/no-SIMD 采用 sentinel。没有直接改 reduction 顺序。
callback 而非固定 Sched literal 支持 tails 不同 k，并仍是 generic rule 的参数。
MRP 的两条 reduction 轴映射回原 k，leading parallel output/outs maps 不变。
default kvec 仅适用于 contraction-contiguous B；本 packing 的 lane 轴为 columns，
故不采用该水平 reduction 路径。

pgemm_exec 对 destination/reference 校验数值，pgemm_vector 检查 F32 优化 LLVM 的 vector multiply/add；
F32 benchmark 的 native LLVM 为 <4 x float> accumulators 与 SIMD multiply/add。
tile sizes、输入、线程、warmup/七次测量与 Task 8 相同。下表为每次独立运行的 median GFLOPS，
default/OpenBLAS 的运行噪声不作为本改动的加速。

| Item 1 shape | Rule | Hand | Default | OpenBLAS |
| --- | ---: | ---: | ---: | ---: |
| 256³ | 34.823 | 38.173 | 29.593 | 71.443 |
| 512³ | 55.219 | 55.706 | 35.404 | 103.412 |
| 1024³ | 51.902 | 51.069 | 34.619 | 101.751 |
| 513×257×129 | 51.486 | 49.942 | 41.751 | 89.987 |

数据：build/task8-bench/item1-results.csv、item1-metrics.json、item1-native.ll。
MR=32、KR=64、NR=32 尚未调优；B crop+pack 和 acc init copy 尚保留，下一项分别消除。

### 5.2 stage(slice) 一次 box fill

lower_tensor 在默认路径保留 stage 的直接 slice operand，不把它先展开为 copy MRP；
lower_to_mem 将 start/step、source shape、box extent 交给 stage_box_impl 的 reindexing。
padding 仍按 box extent 判断，只在 present branch 用 start+step×coord 读取 source，
不在 padding lanes 构造越界 load。stage_impl 复用同一算法，以 start=0、step=1 给出原语义；
reference 路径仍先 materialize slice，再 pack。fuse_tensor 的 stage barrier 未改变。
其他 readers 仍可独立 materialize 保留下来的 slice，不改变其值或全局消除共享 copy。

stage_destination 的裁剪 source 用例只有 output+stage 两个 allocations（原来三个）；
tile_constructs_exec 两路径新增非零 start、step=2、padding 和 singleton-box 校验。
GEMM 每个 full/final K block 只剩一个 packed panel fill，vector nest 的 accumulator
allocations 与 fold init 不属于 stage crop。

| Item 2 shape | Rule | Hand | Default | OpenBLAS |
| --- | ---: | ---: | ---: | ---: |
| 256³ | 34.735 | 40.059 | 28.035 | 70.579 |
| 512³ | 53.291 | 54.095 | 34.709 | 100.615 |
| 1024³ | 50.336 | 51.102 | 32.581 | 95.865 |
| 513×257×129 | 53.860 | 53.992 | 41.442 | 89.680 |

数据为 build/task8-bench/item2-results.csv / item2-metrics.json。小 shape 的噪声较明显，
不据单次重复推断 packing 已显著提高算术吞吐。Item 1 commit 后另一次相同设置运行
为 Rule 50.109/48.625/49.997/49.705，Hand 48.041/49.093/49.118/48.860，
Default 33.198/33.937/33.664/40.824，OpenBLAS 76.456/94.311/96.669/88.036。
所有运行均以相同 scalar-reference 验证结果；未选择性隐藏较慢测量。

### 5.3 Scalar first block，无 accumulator init copy

lower_to_mem 的 tile_fold 仅在 count 为正 literal、init 为 uniform splat/Pack 且原 step
通过 linear/pointwise proof 时 peel first step。独立 accumulator 仍只分配一次，但不填零
也不复制 init；first step 在 off=0 用 scalar seed 完整写出该 buffer，随后执行 n−1
次 step，offset=(i+1)×ks。最后 K block 和每步 post 的既有顺序不变。
count=0 仍 normalizes 为原 init；未知 count、非 uniform seed、其他 acc 读或非 pointwise
maps 保留原路径。mrp 的 uniform outs 在 leaf bufferization 转为 scalar init，不分配 seed buffer。

fold_seed_destination 检查 scalar seed、foldFirst 和无 foldCopy/buffer.lit；
tile_constructs_exec 在两条 pipeline 验证 n=0/1/3、init=3、非结合 combiner
acc×10+off 的结果分别为3/30/3024，避免只测加法的等价代理。
新 reference 用例发现同类型 buffer.lit 的 bottom-memory allocations 可共享：
step 的 off fill 覆盖了初始 seed buffer（实际0/24而非30/3024）。
splat_buffer 改为每个 lowering scope 请求 fresh_mem，使两个 fill 的 allocation memory
不同；这是本项 scalar seed/reference 对照直接暴露并必须修复的 alias，不修改 AddMem 算法。

| Item 3 shape | Rule | Hand | Default | OpenBLAS |
| --- | ---: | ---: | ---: | ---: |
| 256³ | 52.436 | 51.695 | 34.655 | 80.805 |
| 512³ | 53.118 | 52.934 | 33.805 | 98.745 |
| 1024³ | 51.261 | 53.569 | 34.065 | 99.525 |
| 513×257×129 | 53.514 | 53.917 | 41.632 | 88.274 |

数据：build/task8-bench/item3-results.csv / item3-metrics.json。1024³ LLVM 中只有一个
131,072-byte column accumulator allocation，无旧同大小 init buffer 和 foldCopy；
peeling 会增加一个静态 first-block stage/vector-accumulator site，但动态每 K block
仍只 packing 一次。没有把 malloc 的静态 sites 数等同于运行时 allocation 次数。
Item 2 commit 后 benchmark 的结果为 Rule 38.847/52.328/50.072/53.497，
Hand 43.436/54.079/50.380/53.750，Default 31.630/34.104/33.583/40.873，
OpenBLAS 76.645/94.545/95.165/85.608；保留于 item2-post-results.csv。

### 5.4 剩余差距

三项的主要提升来自 vector leaf（约7.5到50+ GFLOPS）；去掉 crop 和 init copies
减少了确定的内存工作，但整体计算吞吐提升未超出所有 shapes 的测量噪声。
结构 rule/hand 在大多数 shape 已超过 default，不同 implementation 的 independently
compiled loops 仍可能有噪声，不能以小差异推断 rule 有运行时 overhead。
OpenBLAS 仍约1.5–2倍快：当前 row 每次独立 vector nest，缺少多行 register blocking
共享 B loads、target-specific microkernel/FMA tuning，NR/MR/KR 固定而非 cache/autotune selector。
packed B 每 column/K block scratch 重建，尚无跨调用 packing，A 未专门 packing；
stride、tail panels 和循环顺序也未用成本模型搜索。
下一步应先比较不同 MR/KR/NR 和 multi-row blocked nest，检查 spill/assembly 与 bandwidth，
再加入 scratch reuse、A packing 和 target microkernel，不能仅凭进一步去除 copies
保证达到 OpenBLAS。parallel grid 不用于这里的单线程对比。

## 6. Register accumulator 的 LLVM 前提

从 `fix/ll-simd-emit` 移植最终修复，不保留其被替代的中间实现。
CPU 的静态、二次幂 F32 数组在 SSA 和内存访问中使用相同的 vector 类型，
load/store 使用 `align 1`，因此 loop-carried accumulator 不再需要 lane-wise repack。
gpm 已有其他数值类型的双形式 conversion；这里保留它，仅让含 F32 vector 的类型
共用 canonical form，避免回退该分支已有的 integer/F64 SIMD 支持。

含 F32 vector 的结构体使用 packed LLVM type 加显式 padding，偏移与
`core.trait.align/size` 一致；rec、字段 extract/insert 和 `mem.lea` 采用同一字段编号映射。
动态 extent 的数组仍可通过指针传递，但不支持按值 load/store，改为明确诊断，
不再静默发出 `[0 x T]` 的零字节复制。

移植后的 `fc_exec`、`attn_exec` LLVM 行数不变；变化仅为 F32 内层数组的
vector/opaque-pointer 类型拼写，循环与 scalar arithmetic 不变。
`buffer/exec_conv` 去掉生成 ID 后完全一致。
gpm 不再有 `Global` IR 节点，所以对应单元测试改为外部 vector pointer 与 memory-pair erasure。

### 6.1 小 blocked leaf 的 SSA nest

`vdim = ro - 2` 且 trailing parallel block 不超过 64 cells 时，使用扁平
`«MR * NR; To»` SSA accumulator：unrolled seed、ordered K loop、每点 value update、
K 结束后的单次 write-back；不分配 accumulator panel，不扩展 `Sched`。
64 cells 是限制 compile-time expansion 的保守边界，不是通用 target register budget；
更大的 block 保留已有 buffer-backed nest。MR/NR 仍来自 leaf 的 literal extents。
unroll 的有限 continuation wrappers 必须强制 specialize，否则 aggregate 参数会留下
未消除的 higher-order continuation。

1024³ 的 4×16 F32 / 四 K 展开热点有 64 `fmla.4s`、8 `ldp q` B loads，
A 为 4 `ldr q`，通过 lane-indexed FMA 使用，没有 accumulator stores 或栈 spill。
256³ 的编译版本把 A 合并为 8 `ldr s` 加 4 `ldr d`，load merging 尚受 shape 影响。
1024³ 的一次比较为 rule 98.75、hand 99.12、旧 rule 53.27、
plain default 33.93、OpenBLAS 100.65 GFLOPS；使用单线程、native `-O3`、1 warmup + 7 runs。
这不是完整六形状 benchmark 的结果。

并不需要显式 vector-element tensor 来得到大形状的四次 `ldr q`。普通 `reshape` 把 A 分成四元素 cells，
并让 leaf 接收 `«4; F32»` 元素的尝试，在 `LowerToMem::collect_tensor_types` 被
“tensor with array element type” 拒绝。该转换目前以整个 array type 作为 bufferization
key，尚不能区分 scalar rank-3 tensor 与 vector-element rank-2 view；
不能简单删除这个检查；该方案没有采用。尝试和错误日志保留于
`build/partB-vector-A-attempt.patch`、`build/partB-vector-A-blocker.log`。
与 LLVM 移植后的基线比较，`fc_exec` 与 `attn_exec` 去掉生成 ID 后完全一致；
`buffer/exec_conv` 的小 blocked case 则去掉 panel alloca，改为 SSA accumulator。

### 6.2 BLIS-shaped Rule 与完整 benchmark

`lit/tensor/blis.mim` 的 RHS 为 N/NC tiled → ordered K/KC fold →
`stage(slice B)` → MR/NR tiled → blocked leaf；B 布局为 `(ceil(n/NR), k, NR)`。
最后一个 K block 在 fold 后执行。线性 outs 原地更新，首块从 scalar zero seed 开始，
不复制或清零整块 accumulator；B 每个 NC/KC box 只填一次。benchmark 没有 destination-copy fallback。
保留 optional `pack_a`，默认关闭；没有 MC level，也没有 persistent packing。
Rule 和 hand 的 I64、F32 小形状与 MR/NR/KC tails 都与 scalar reference、reference-tiles 路径一致。
decimal combiner 的独立测试验证 K 的顺序。

目标描述作为 selector 的显式参数输入，不声称自动发现硬件：
本机使用 `vec_width 4`、32 vector registers、256 KiB cache budget。
`NR = 4 * lanes`，`MR = min(lanes, (registers - 12)/4)`；
`NC = min(4 * lanes * NR, 256)`，`KC = min(cache_bytes/(2 * NC * sizeof(T)), 256)`。
packed B 占 cache budget 至多一半。本机得到 **MR=4、NR=16、NC=256、KC=128**。
F32 selector 的 literal 输出由 FileCheck 验证；`Sched` 仍只有 vdim/unroll。

KC=128 时，rule/hand 的所有完整 4×16 四 K 热点均有
64 `fmla.4s`、4 A `ldr q`、8 B `ldp q`、**0 accumulator stores、0 stack accesses**，
即每 K 16 vector FMA、4 B vectors；每行四 K 只加载一次 A vector。
K/M/N tails 使用较小的 block 或原有 classic/vectorized 路径。
KC=256 的 256³ code 则未合并全部 A loads，因此没有作为最终配置。

重现命令：

```sh
python3 build/task8-bench.py --blis --skip-packa
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/task8-bench/bench
```

harness 仍在 ignored `build/`，可用 `--mr/--nr/--nc/--kc` sweep，
以及 `--lanes/--registers/--cache-bytes` 设置 target 描述。
Homebrew clang++ `-O3 -mcpu=native`，F32，一个线程，1 warmup + 7 runs，median；
包含 allocation/free。所有路径在同一 thread 上；macOS 显式设为 user-interactive QoS，
但这不是 core affinity 或锁频。没有增加额外 warmup。
下表来自同设置的第二次完整执行，不混合不同执行的最好结果：

| M×K×N | BLIS Rule | hand BLIS | 旧 Rule | plain default | OpenBLAS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256³ | 78.45 | 81.86 | 52.41 | 31.72 | 78.00 |
| 512³ | 96.71 | 97.06 | 56.17 | 35.46 | 99.92 |
| 1024³ | 98.08 | 97.68 | 51.86 | 34.87 | 97.71 |
| 2048³ | 97.47 | 93.35 | 53.27 | 28.29 | 104.14 |
| 1000³ | 97.90 | 98.09 | 53.75 | 36.45 | 103.79 |
| 513×257×129 | 87.41 | 90.79 | 54.16 | 41.67 | 89.42 |

单位 GFLOPS。512³ 及以上 square 达到 90+；**256³ 尚未达到 90**。
短形状有明显 machine-state 波动：同参数前一次执行为 Rule 51.26 / OpenBLAS 63.71；
仅 QoS 不能消除这种变化。不能用长形状的成绩代替这个未达到的目标。
剩余成本包括 packing、每 KC 的 accumulator write-back、microtile loop/地址运算，
以及尚未实现的 prefetch/software pipelining、persistent scratch reuse 与 parallel loops。
256³ 需要单独降低这些固定成本，并在稳定的核心/频率条件下验证。

当前编译了全部六形状和 unused optional-A variants：Mim 63.73 s，
LLVM 26,899,257 bytes / 398,290 lines，clang 9.05 s，harness executable 334,712 bytes。
这不是单一 microkernel 的 code size。日志、两次完整原始表、assembly 与 counts 分别在
`build/partB-blis-k128-build.log`、`build/partB-blis-k128-{results,repeat}.csv`、
`build/partB-blis-k128.s`、`build/partB-hot-loop-counts.json`。
`feature/default-sched` 的 read-only binary 仍因缺少
`libabsl_raw_hash_set.2508.0.0.dylib` 而不能载入 opt plugin，故跳过该额外 baseline。

## 0. Generic shape Rules


Rule 定义期仍检查 guard 的 Bool 类型，以及两侧已经闭合的类型不一致。
两侧类型依赖 Rule 元变量时，不能证明相等不再立即拒绝；每次 instantiation 的
replacement type check 必须成功，否则该 candidate 为 Unknown，不发布，继续寻找其他规则。
定义位置会发出 deferred type-check warning；实例化 type mismatch 默认以 warning 报告，
不需要 `-V`。`-X compile:report-unknown` 也报告其他 Unknown candidates。
这不是 cast 或 Hole 推断，也不允许非法 RHS 经配置成功后绕过最终检查。

开放 mutable Pi 的类型匹配不再要求模板与源指针相等：先绑定其中的 Rule captures，
再在隔离 World 中代入并只读 alpha-check。没有在 source 构造投影或修改类型。
`grshape` / `_g` 以同一规则匹配两种 shape 和动态 offsets；错误 shape 的优先规则被拒绝，
后续合法规则仍改写。`grwtype` / `_g` 从定义期 gap 翻为目标检查：
staged box 求值后才能确定的 RHS shape 仍受最终 type check 约束。

目前 generic 指 shape/offset 的 Nat 元变量；配置中含 `T: *` 的高 universe Rule list
仍需要显式适配配置的 universe，不等同于自动 universe polymorphism。

`grimmut` / `grpack` 在两个 frontends 检查 packed GEMM 所需的两项 canonicalization：
开放 immutable Arr/Pi 和开放 mutable 类型一样，在 captures 齐全后于隔离 World 内
instantiate-and-alpha-check。比如源 `«3, 4; Nat»` 应匹配模板 `«M+1, N; Nat»`
在 M=2、N=4 的实例，不能要求源的 literal 3 结构上还是 nat.add App。
这不是允许任意 term 用 alpha-check 匹配，入口仍仅限 type-level Def；最终 replacement
type check 保留。

Tuple pattern 可以匹配同长度的 immutable homogeneous Pack；`(M,K,N)` 对
canonical `‹3; 4›` 把三个 captures 都绑定成 4。每个 operand pattern 均必须匹配
同一 Pack body，长度及整体类型仍检查；重复元变量 `(M,M,N)` 不会接受 `(4,5,6)`。
非 homogeneous Tuple 沿原路径匹配，mutable Pack 不因此获得结构匹配权限。
这使 8³ 和 1×K×1 的 dimension metadata 与非方阵使用同一 generic rule，
没有针对特定 literal shape 的第二条规则。

## 0.1 Operand placement 的 dominance

`schedule_index.mim` 不依赖 tensor：parallel body 把 I32 counter 窄化为 `Idx 3`，
同一 index 用于初值读取、inner loop 的读取和 fold 后 continuation 的写回。
master 的 `Scheduler::smart` 只以 uses 的 structural `late` 为上界，各 use 独立上移，
写回 GEP 被放到 outer head，而 trunc 留在 outer body，LLVM verifier 报 operand 不支配 user。
修复取自 `feature/default-sched` 的 `2eeda4cc2d`：`smart` 的上界改为 uses 实际 `smart`
placement 的 lca，与那条分支保持同一实现。

## 1. Tensor accumulator

`map_reduce_post` 增加推断的 `Init` group，原 scalar init 调用不变。
`outs shape map_init value` 提供显式 read map，`outs_default shape value`
使用该 op 的 `map_out`。descriptor 的 shape 保留 logical rank，不从 singleton-collapsed
数组类型推断。scalar 与 descriptor 在 lowering 验证，非法 init 报错。
tensor data-use schema 包含 outs，fuse 的同 rank slice/read-through 可以合成 `map_init`；
暂不融合需要改变 accumulator logical rank 的 read。

经典、vectorized、blocked、redvec 的既有 seed 点统一调用
`init_cell: Cn [mem.M 0, «Rn; I32», Cn [mem.M 0, To]]`。
tensor.lower_to_mem 只 bufferize descriptor；btensor.lower_map_reduce 创建读回调。
outs buffer 作为运行时参数在进入 nest 前求值，不能只藏在逐 cell callback 的 capture 中：
否则 producer 会在每个 cell 重跑，且被共享的 allocation 会覆盖先前结果。
scalar 使用常量 callback，FC/redvec、conv、attention 的 scalar LLVM 去掉 ID 后仍完全相同。

`outs_exec` 检查 split fold、仅最后 block 的 I64→I32 post、空 reduction、
非 identity init map，以及 classic/vectorized/blocked/redvec。
现有 `tensor.lower_map_reduce` 保留 value-level counting-loop 参考路径，并在 reduction 前读初值；
没有另造 buffer loop nest。一般 `map_out` 不要求可逆，故未改成通过输出坐标反求循环坐标的
generate 实现；新 tiled/stage 的 generate 参考语义另列于下一节。

## 2. tiled / tile_fold / stage 的参考语义

`tiled @(T, r) (s, ts) body` 拼合互不重叠的 owned tiles。
body 的参数是全局 `off` 和该 tile 的实际 `ext`，返回 `«ext; T»`。
尾块按每维 remainder 剥离，tile 大于 extent 时缩到 extent；singleton collapse
不改变 logical rank。零 extent 不调用 body，空 tensor 的 LLVM ABI 为 void。
静态零 tile size 报错。当前 generate 参考实现会为**每个输出元素重算整个 tile**，
不是一次 tile realization，也没有分配/性能保证。

`tile_fold (n, ks) init step` 依次以 offsets `0, ks, …, (n-1)*ks`
调用 `step(acc, off)`，严格维持 Acc 类型及先后顺序，不重排非结合 combiner。
`n = 0` 直接返回 init；非空 fold 的 ks 必须正数。只包含完整 blocks，
最后的 reduction block 及最终 epilogue 由调用者在 fold 后显式构造。

`stage @(T, r, k) s (inner_dims_pos, inner_tiles, outer_perm, pad) scope input`
按 linalg.pack 的约定重索引：先把选定维度 ceil-div，按 outer_perm 的
**输出轴顺序**排列，再追加 inner_tiles。inner_dims_pos 为互异的原始轴号，
outer_perm 必须双射；这与 tensor.transpose 的 input→output 约定不同。
越界 lanes 返回 pad，不执行源 load；scope 目前仅保留语法位置，不映射存储地址空间。
lower_tensor 展开成 generate，fuse_tensor 不穿过 generate 合成读取；
identity stage 也不由 normalizer 消除。`stage_barrier` 检查该边界。

参考 callbacks 可以产生 tensor 分配，lower_to_mem 把其局部 fresh-memory 请求包装成
返回式 CPS 区域。第一次 `mem.add_mem` 前请求已经存在，不是 cps.conv 新增分配：
`tile_fold_impl` 的 nullary `run` 包含 `next → mem.fresh(0, fresh_mem)`，
`tiled_impl` 的 tail-dispatch `choose` 也包含 slice buffer 的同类请求。
两者通过 `cps.cps2ds_dep` 被调用；AddMem::analyze 把该 axiom 的回调以及其传递可达
continuations 加入 preserved_，rewrite_mut_Lam 在 preserving_ 模式原样复制它们。
因此 next 仍为 `Cn []`，不能在那里注入 memory，fresh 也没有进入普通 resolve 分支。
generate 的 element callback 经 btensor 的 axiom-pinned ABI 也存在同样限制。

CPS 转换及 affine loop 展开移除这些调用边界后，`mem.resolve_fresh` 仅在仍有
`mem.fresh` 时调用现有线程化逻辑。放在 affine.lower_for 之前会改变循环 callback 的
ABI 而使调用不匹配；第一次 AddMem 直接 unpin 同样破坏 cps2ds 的 dependent type groups。
前端分支 e76c4d05f9 / e9d856fb28 / 9d2bf7f6a7 优化的是普通 resolve 分支中
tt continuation 的 splice、递归判断及每次请求的 binding scope；
它们不改变 preserving_ 的入口，不能解决上述 pinned 区域。
本实现没有搬用其不适用的旧 Rewriter mapping 逻辑；若移植该优化，仍必须保持 scoped binding。
没有剩余请求的程序不再改写，既有 GPU pipeline 保持不变。
`fresh_before` 正向断言第一次 AddMem 的残留请求；`fresh_after` 断言 ABI 展开后请求消失；
`tile_constructs_exec` 验证 repeated fold/tail iterations 的数值，不把“无 fresh”当数值正确的代理。
目前这些参考构造只验证 CPU LLVM，GPU scope 与 address spaces 不在此实现内。
helper ABI 的识别只沿实际 metadata callbacks，不沿输入 producer 或 Var binder 回边。

`tile_constructs_exec` 对照 C++ 标量参考检查 divisible/tailed/singleton/empty tiled、
一轴及双轴 pack、padding、标量非结合 fold 和 tensor accumulator fold。
该测试也覆盖空 tensor 的 erased ABI；非法 tile、block size、布局有正向诊断断言。

§5.4 destination-passing 尚未实现：需要 outer destination 的 logical shape/origin、
每次 iteration 的 memory/accumulator、scratch scope/lifetime、outs 的线性使用检查，
以及不能原地写时的显式 copy/report。当前 fold 的中间 heap buffers 可能保留到调用结束
之后；不能把参考实现当作 scratch 生命周期或 persistence 契约。

### 2.1 Nested closure 的 fresh-memory memo

`nested_stage_exec` 的最小结构为 outer tiled → stage(slice(input, off, ext))
→ inner tiled → slice(stage, row, size)，不需要 tile_fold、outs、combiner 或 captured access map。
单纯 nested tiled 读取输入、单个 tiled 内的 stage、单个 tiled 内的 mrp/slice、captured combiner
和 captured access map 均不是这次 assertion 的充分条件。
当 inner row 和 outer column 有尾块，原 lower_to_mem 留下 stage allocation 的 fresh_mem Var。
失败的是闭合 external `stage_share` 经 RWPhase::rewrite_external 统一 externalize，
不是 lowering 把某个 local Lam 主动提升为 external；相同结构不经 Rule 也失败。

只给 allocating App 一个 per-Lam fresh_memo_ 不够：它的 users、argument tuples、
slice 组合仍进入 Rewriter 的全局映射，并可把另一个 closure 中的 fresh_mem Var 带回来。
每个 Lam 现在打开 Rewriter scope，新建 fresh_mem Var 登记为该 scope 的 new_var；
依赖它的映射只在该 Lam 内有效，其他不依赖该 Var 的映射仍可共享。
结束时销毁这些临时映射，不改变 ordinary closures 的 environment 或 external ABI，
也不新增第二套 loop builder。测试执行两次、每次改变输入并检查全部 tail cells。

### 2.2 AddMem 的 continuation-local memory memo

`mem.add_mem` 和 `mem.resolve_fresh` 共用 AddMem，均在每个普通 Lam 的 body
改写前打开 Rewriter scope，并登记该 Lam 的 memory 参数所属的 Var。
allocation、projection、argument tuple 等结果只要依赖这个 Var，其 memo 就留在该 scope；
离开 Lam 后删除，不把另一 continuation 的 memory token 带回来。
不依赖该 Var 的映射仍留在 root，Lam stub 在打开 scope 前登记以保留递归与闭合共享。
pinned ABI 的 preserving_ 路径不注入 memory，不改变已有 GPU callbacks 的签名。

`add_mem_shared` 的两个 continuations 都使用同一 hash-consed
`mem.alloc (I64,0) ⊥:(mem.M 0)`，`resolve_shared` 再通过 fresh 接收 callbacks 测试
第二次 pass。原实现两个文件都在 externalize 时 assertion；新实现各自在本地 memory
上 allocation，检查输出 token，且没有残留 fresh。不是针对 alloc 的特判。
这也修复同 module hand/rule packed GEMM 的 late-pass allocation/projection memo 泄漏。

### 2.3 AddMem 兼容性检查（Task 7f）

修复前后用同一 Debug build 配置及默认 `-p opt … -p ll` 生成
`fc_exec`、`attn_exec`、`buffer/exec_conv` 的 LLVM；仅把 `_数字` 生成 ID
统一成 `_ID` 后，三个文件完全一致，没有 placement、instruction 或 ABI 差异。
`exec_conv_run` 对该 conv module 做实际执行并检查 exit code 63。
完整 CTest 235/235，FULL lit 609 passed / 9 unsupported；
包含已有 GPU/NVPTX 编译路径，unsupported 项不代表 GPU 硬件执行已经验证。
原始 LLVM、去 ID diff 和完整日志保存在 `build/task7f-before`、
`build/task7f-after`、`build/task7f-item1-{ctest,lit}.log`。

## 3. Generic packed GEMM（Task 7f）

`lit/tensor/pgemm.mim` 的同一 Rule 对 M、K、N 和 MR、KR、NR 泛化，
把 marked real product_2d 改写成 tiled(N) → tile_fold(K, outs=acc) → tiled(M)。
每个 K block 在 row tiles 外绑定 `bp = stage(slice(B,…))`，
packed read map 为 `(j/NR, k, j%NR)`；A 与 accumulator 按实际 row ext 切片。
先 fold `(K-1)/KR` 个完整 K blocks，再单独执行最后一个正长度 block。
M/N 尾块使用 actual ext，stage 的 padding lanes 不进入逻辑结果。
这里固定 I64 wrapping add/mul，保留 matched ring operators；plain GEMM 不含 bias/ReLU，
不声称浮点结合性或 reduction 重排权限。

`pgemm_exec` 在**同一个 LLVM module** 包含三个 Rule-generated 与三个直接调用
`packed` 的 hand-written externals。两种路径均与 C++ scalar reference 逐 cell 比较：

| Shape M×K×N | MR×KR×NR | Rule / hand |
| --- | --- | --- |
| 8×8×8 | 4×4×4 | 全部正确，重复两次且改变输入 |
| 13×7×11 | 4×3×4 | M/K/N tails 全部正确，重复两次且改变输入 |
| 1×7×1 | 2×3×4 | singleton-collapse ABI 与 K tail 正确 |

`pgemm` 的 IR 测试检查 stage/mrp/tiled/fold 存在且所有 marked calls 消失；
graph frontend 保留三个 marked roots，并正向检查 recipe builder 的拒绝诊断。
拒绝的是 RHS `packed` local Lam call，不是缺少一个 pure-axiom 名字；没有放宽
graph purity 合同。支持它还需要纯 closure-call recipe 或安全的 helper 展开。
执行使用 rule_app，经现有 `_impl` 参考语义 lowering。
最终 CTest 235/235、FULL lit 611 passed / 9 unsupported；
同一 C++ reference 也链接 `clang -O2` object 重新执行并通过。

### 3.1 编译时间与代码体积基线

macOS arm64，同一 Debug Mim build，默认 `-p opt` → LLVM。
命令为 `build/bin/mim -I lit/tensor -p opt lit/tensor/pgemm_exec.mim -p ll`，
在独立输出目录执行，包含全部六个 externals；先 warmup 一次，再测五次。
计时含 frontend/plugin loading 和所有 Mim phases，不含随后 clang/object 编译或运行。

| 项目 | 结果 |
| --- | --- |
| 五次 wall time（秒） | 0.643、0.640、0.657、0.663、0.662 |
| 中位数（秒） | 0.657 |
| LLVM IR | 356,110 bytes / 7,942 lines |
| clang -O0 object | 42,720 bytes；Mach-O __TEXT 39,712 bytes |
| clang -O2 object | 19,104 bytes；Mach-O __TEXT 15,920 bytes |

原始命令、计时、LLVM、objects 和 numerical executable 位于
`build/task7f-gemm`，完整 metadata 为 `metrics.json`。
这些不是运行速度或 allocation 数量的优化结果，也不是 per-function code size；
对象体积不包含 C++ reference 或 linked runtime。

### 3.2 Task 7 的 reference baseline（当前进展见 §4）

Task 7 的 tiled generate 参考实现可为每个 output cell 重算整个 tile，
语法上的外层 bp binding 尚不保证 stage 运行时 exactly-once realization。
§5.4 destination passing 需要传入 outer destination 的 shape/origin、每次 iteration
的 memory/accumulator、scratch lifetime，以及 outs 的线性 alias 检查。
不能原地写时须显式 copy/report；fold 中间 buffers 的释放与 reuse 也需实现。
因此本实验验证 generic Rule、closures、tail 数值与完整 lowering 通路，不是 packed GEMM
的性能实现；GPU storage scopes 和 fused bias/ReLU epilogue 尚未在该实验验证。
