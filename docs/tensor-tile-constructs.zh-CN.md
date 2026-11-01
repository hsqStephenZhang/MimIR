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
返回式 CPS 区域。CPS 转换及 affine loop 展开后，`mem.resolve_fresh` 仅在仍有
`mem.fresh` 时调用现有线程化逻辑；没有剩余请求的程序不再改写，既有 GPU pipeline
保持不变。目前这些参考构造只验证 CPU LLVM，GPU scope 与 address spaces 不在此实现内。
helper ABI 的识别只沿实际 metadata callbacks，不沿输入 producer 或 Var binder 回边。

`tile_constructs_exec` 对照 C++ 标量参考检查 divisible/tailed/singleton/empty tiled、
一轴及双轴 pack、padding、标量非结合 fold 和 tensor accumulator fold。
该测试也覆盖空 tensor 的 erased ABI；非法 tile、block size、布局有正向诊断断言。

§5.4 destination-passing 尚未实现：需要 outer destination 的 logical shape/origin、
每次 iteration 的 memory/accumulator、scratch scope/lifetime、outs 的线性使用检查，
以及不能原地写时的显式 copy/report。当前 fold 的中间 heap buffers 可能保留到调用结束
之后；不能把参考实现当作 scratch 生命周期或 persistence 契约。
