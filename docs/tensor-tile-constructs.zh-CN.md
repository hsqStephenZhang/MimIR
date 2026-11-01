# Tensor-level tiling constructs（Task 7）

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
