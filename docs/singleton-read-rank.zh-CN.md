# Task 22A：singleton reshape 的 read-map rank

## 复现与定位

当前分支 build/bin/mim 复现四个独立模型目录的原始命令；
仅将 binary、配对的 `-P build/lib/mim`、kernel 输入和输出路径改为当前 worktree。
ResNet 的 group=0/1、attention/transformer 的 group=1
均在 map 参数检查处失败，expected rank 4、actual rank 3。
ResNet 失败阶段为 tensor.lower_to_mem，不是 grouping。

最小结构不需要 dot、convolution、浮点运算或 grouping：
对显式 shape `(1,2,3,4)` 做 transpose，
把 `(1,3,2,4)` reshape 为 `(3,2,4)`，再按 rank 3 transpose。
size-1 轴使 reshape 的输入输出 COC array type 相同，
`normalize_reshape` 因 type equality 直接返回 rank-4 producer，
但下游 slot 仍声明 rank-3 shape/map。
L1 的 `reads_as_produced` 发现输出 shape 不同，正确拒绝直接 compose；
destination lowering 的 `lower_mrp` 再次展开 tensor_read 时却无对应的 rank/shape 桥接，
把 rank-3 slot map 的结果传给 rank-4 producer read map，触发类型错误。
dot_product_impl 的 reshape 只是模型中留下的 source blame，不是 dot 算法错误。

复现文件 `build/task22/minimal.mim`；当前未修复时 group=0 编译即失败。
四个模型的日志位于 `build/task22/*_mimir_group*.log`。
仅保留 normalize_reshape 的显式桥接可修复最小 transpose 用例及 attention/transformer，
但不能修复 ResNet：conv 的输出仍按 rank 4 描述，binary helper 从折叠类型得到 rank 3，
即使不显式调用消除 singleton 的 reshape，也会出现相同边界。
因此根因落点是 destination read-through 的 map composition，
不能仅修改 dot_product_impl 或 reshape normalizer。
短期修复应在读取 declared shape 与 producer shape 不同时插入 row-major reshape_map，
再 compose producer 的读取 map；不能把 type equality 当作 rank equality。
此处的临时处理将由 mimir/mimir#641 的 nominal tensor 表示取代；
本任务不进行 nominal-tensor 迁移。

## 短期修复

仅修改 `lower_to_mem.cpp` 的 tensor_read：
读取纯 read MRP、slice/slice_box 时同时接收 consumer 声明的 shape，
若与 producer 的显式 output shape 不同，先构造
`reshape_map(producer_shape, consumer_shape)` 将 consumer 坐标转回 producer 坐标，
再复用原有 compose_read_map。
combiner/post 输入、tensor-init 以及 linear-step accumulator 分析
全部使用同一个 helper，避免分析与真正读取发生分歧。
不改变 normalize_reshape 的既有折叠行为，不改 dot/conv 调度，
不以跳过类型检查、放宽 affine proof 或强制 materialization 绕过错误。

回归 `lit/tensor/singleton_reshape.mim` 仅含两个 transpose 和一个 reshape，
以 I64 序列对 group=0/1 都执行 ASan 数值检查，
输出应逐元素等于输入，并保持独立返回 storage 的所有权。
保留原有 normalize.mim 中 singleton reshape 的 structural identity 检查。

## 四份模型的重新验证

独立目录只附带 kernel、命令、编译错误/IR，没有输入参数或 PyTorch 输出，
因此不能声称重放了 gemm-bench 原始随机参数。
新增 `scripts/check-singleton-models.py <model_fail目录>`：
使用当前 build/bin/mim 与配对的 build/lib/mim，保留 JSON 原命令其它 flags，
每份 kernel 分别覆盖 group=0/1，另输出 LLVM；
以 Homebrew clang++ `-O3 -mcpu=native -dynamiclib` 编译后调用完整 `mim_model`。
无计时，不使用或修改 B 的 worktree。

纯 Python scalar reference 完整实现原 kernel 的两层 padded conv、residual/ReLU、
QKV 两头 attention、softmax、输出 projection，以及 transformer 两个 layernorm/MLP。
三组确定性输入/参数含正负值、零输入和非平凡 layernorm affine weights；
reference 从传入的 F32 参数值起算，使用 double 标量运算。
检查输出有限、abs error ≤ `2e-5 + 2e-4*abs(expected)`，并检查输入未被修改；
每次返回结果均释放。无需安装 Python tensor package。

| reproducer | group=0 | group=1 | 每模式 trials | 最大绝对误差 |
| --- | --- | --- | ---: | ---: |
| resnet_mimir_group0 | 编译、数值通过 | 编译、数值通过 | 3 | 0 |
| resnet_mimir_group1 | 编译、数值通过 | 编译、数值通过 | 3 | 0 |
| attention_mimir_group1 | 编译、数值通过 | 编译、数值通过 | 3 | 5.263e-8 |
| transformer_mimir_group1 | 编译、数值通过 | 编译、数值通过 | 3 | 2.558e-7 |

结果 CSV 为 `build/task22/model-numeric.csv`；
每模式命令、日志、IR 和动态库保留在 `build/task22-model-check/`。
完整 CTest 243/243，full lit 666 passed、9 unsupported。
未处理另一会话负责的 VGG16 mem.seo fixed-point 问题。
