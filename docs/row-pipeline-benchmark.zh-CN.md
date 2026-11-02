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
