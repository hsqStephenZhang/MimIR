# 统一 row normalization 基准

## Task 21A.1：同程序、同 harness

复现：`python3.14 scripts/bench-row-pipeline.py`。
`--build-only` 只构建，`--run-only` 复用已构建的 binary。
默认目录为 `build/task21-row-bench/`；保留生成的 Mim、LLVM、profile、counter LLVM 和 CSV。
Homebrew LLVM clang++，F32、单线程、`-O3 -mcpu=native`，
每个 workload/shape/mode 一次校验 warmup，再轮换 mode 次序执行 7 次、取中位数。
计时包含返回值 allocation/free，不含 reference；全部输出与同一个 double scalar reference 比较。
计数 binary 在相同的 `-O3 -mcpu=native` 优化后 LLVM 上插桩，
以 `-O0` 编译插桩结果，避免再次优化计数点；不使用它计时。
脚本在计时前、计数前和结束后检查 mim/clang/cmake/ninja 构建，忙时等待。

前三种模式使用 `lit/tensor/row_norm.mim` 的同一 `row_soft/row_layer`：
softmax 为 rowmax → retained exp → rowsum → div；
layernorm 为 mean → retained centered → variance → normalize，epsilon 为 1e-5。
`unfused` 指 group=0，仍保留默认 L1 fusion；`automatic` 不传 group 开关；
`explicit` 使用 Task 15A 原有 Rule、rows=1，且 group=0，不叠加自动分组。
`api` 使用 `tensor.softmax`；本分支没有 layernorm 单一 API，
因此用 tensor.reduce/unary/binary_ax 写出相同数学定义。
API 是单独一列，不把它当作前三列的相同 IR。

另保留两个只用于解释旧表的诊断模式：
`legacy_default` 是旧 default benchmark 的无 retained barrier 整表达式、group=0；
`legacy_stages` 是 Task 15A 的四个分别导出阶段，C++ 显式保存 exp/centered。
两种旧输入也在同一个 binary 内分别运行：
`task15 = (i % 23 - 11)/8`，`task20 = (i % 11 - 5)/32`。
相同输入下诊断模式仍复现约 2 倍 softmax 差异，不是输入分布或线程数导致。

### 改动编译器前的结果

下表为 task15 输入，ms；同一次执行同时校验 task20 输入。

| workload | shape | group=0 | automatic | explicit rows=1 | tensor API |
| --- | --- | ---: | ---: | ---: | ---: |
| softmax | 1024×1024 | 1.936916 | 2.015375 | 1.827042 | 3.411750 |
| softmax | 4096×1024 | 7.978125 | 8.377958 | 7.629292 | 13.935083 |
| softmax | 1024×4096 | 7.834917 | 8.008417 | 7.394083 | 13.437708 |
| layernorm | 1024×1024 | 0.348625 | 0.475667 | 0.253292 | 0.185375 |
| layernorm | 4096×1024 | 1.761625 | 2.532417 | 1.034958 | 0.958958 |
| layernorm | 1024×4096 | 1.837083 | 1.964792 | 0.935708 | 0.982167 |

4096×1024 softmax 的 `legacy_default` 为 14.454083 ms、exp=8388608；
`legacy_stages` 为 7.638917 ms、exp=4194304。
Task 20A 表中的整表达式没有保留 exp，L1 将它内联进 sum 和最终 div，计算两遍；
Task 15A 的外部阶段边界强制只物化一次 exp。
API softmax 也有同样的双重 exp，而前三列因 retained stage 只计算一次。
因此旧两表不支持“自动分组只达到手写分组持平”的结论。
layernorm 则是另一个原因：旧整表达式允许中心化融合，
外部四阶段需要完整中间张量，不能把其两条 baseline 等同。

对全部三个 shape、两种 workload，automatic 每次调用为 heap=4、stack=0，
explicit 为 heap=2、stack=0，均含输出的一个 heap allocation；
输出由 harness 在计时范围内释放，表中 free 只统计生成模块内部释放。
softmax 的 exp 都恰好为 M×N，layernorm 为 0。
automatic 的六个 reverse 请求确实实现，但只将 first reduction 与 transform 放在一起，
仍保留 whole-size 中间结果及 identity-stage copy；它不是 Task 15A 的完整 row region。
这些计数和时间是下一项优化的基线，不能以“请求已实现”替代性能等价证明。

## Task 21A.2：完整 row region 与 identity-stage alias

自动路径原先优先选择 first reduction → transform 的 kind 2；
其 `tiled_n` 将 first 与 transform 都写回 whole-size 结果，后续 reduction/final 在 region 外。
因此选择 31/7 行不是主要差距，只把 tile 调成 1 也不能得到手写版的生命周期。
现在 kind 0 增加一个通用 row-separable closure recipe，在 greedy reverse 之前尝试：
只接受 literal M×N 的输出、identity 二维 map 或首坐标 row projection、
完整的同形状 parallel/reduction domain、纯 scalar comb/post、scalar init、canonical schedule，
以及没有 split/permutation 的 identity stage。
不识别 softmax/layernorm 名称，也不构造按类型列举的 `_compile` Rule。

导致旧表 2× exp 的 L1 问题也直接修复：producer inlining 复用已有的
`sole_consumer` proof，只有证明 producer 恰好被消费一次才内联。
原来只检查单个 consumer 的读取 injective，没有检查不同 consumers 会重复计算同一 producer。
共享/Unknown producer 保留为图节点，因而 tensor.softmax 的 exp 也能进入完整 row region；
没有在 softmax helper 中手工添加 stage 来绕过 L1。

内部节点必须由同一 frozen GraphView 证明所有 data users 都在闭包内；
Unknown、外部 reader/escape、跨行 map、非 identity stage 均拒绝这个 recipe。
闭包限制 64 个节点；逐行 storage 上界仍须符合 group-cache。
它作为 kind 0 请求由既有 staged guard 和 group request Rule 实现，
构造 sequential `tiled`，rows=1，保留 reduction 的原有次序；
DAG 节点只实例化一次，既有 realization cache 和 scratch placement 负责共享及 refill。
预算证明来自完整 row partition 与内部 exclusivity，不提高默认 recomputation budget。
已有 BLIS/packed roots 仍优先，其他程序继续使用原有 outer/reverse recipe。

另将 identity-stage alias 从 scratch scope 扩展至已拥有 storage 的 tensor 临时结果：
新分配的 map/reduce、slice 或 tiled_n result 已经可以原位读取，不应再复制。
外部 borrowed buffer 不走这条 alias 分支，返回其 stage 时仍生成独立结果；
不移除 tensor-level stage，不允许 fuse_tensor 透过非 identity stage。
`stage_alias_exec` 用 ASan 检查 whole temporary、borrowed input、非 identity permutation；
`stage_destination` 保留两次 allocation 检查并确认不再生成 stage copy。

`row_norm` 的自动模式对 5×7、2×7、64×7、65×7 做 scalar-reference/ASan、
精确 exp 计数及 O(1) allocation 检查；旧 explicit 模式继续覆盖 empty/singleton。
`row_group_scope` 另检查闭包成立、额外 whole reader、permuted stage、跨行读取四种情况。
tensor.softmax 的自动 API 路径另做 strict 编译、scalar-reference/ASan 和精确 exp 计数。
本轮顺带发现旧 reverse 多结果 tuple 与小 tensor 类型重合时的 pre-existing 分类问题：
混合全部 empty/singleton 与 2×7 导出会报 partial tensor read；
禁用本轮 row recipe 仍复现，未扩大本任务去修改通用 tensor type 分类。
保留复现 `build/task21-mixed-shape-repro/probe.mim`，
生成命令为 `python3.14 lit/tensor/row_norm.py build/bin/mim lit/tensor/row_norm.mim build/task21-mixed-shape-repro automatic all-shapes`。

同一 harness 的修复后检查：automatic 与 explicit 均为每次 **1 heap + 1 stack**，
heap 是返回输出，stack 是循环外 row scratch；模块内部 free=0，
输出由 harness 在计时范围内释放。
softmax 的 exp 恰好 M×N，layernorm 为 0，两个输入分布及全部三个 shape 均相同。
自动 LLVM 从原先约 3.6 MiB 降至约 46 KiB，不再枚举整组 tiled_n 的 grid points。
