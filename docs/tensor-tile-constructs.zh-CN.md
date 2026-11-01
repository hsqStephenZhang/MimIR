# Tensor-level tiling constructs（Task 7）

## 0. Generic shape Rules

Rule 定义期仍检查 guard 的 Bool 类型，以及两侧已经闭合的类型不一致。
两侧类型依赖 Rule 元变量时，不能证明相等不再立即拒绝；每次 instantiation 的
replacement type check 必须成功，否则该 candidate 为 Unknown，不发布，继续寻找其他规则。
这不是 cast 或 Hole 推断，也不允许非法 RHS 经配置成功后绕过最终检查。

开放 mutable Pi 的类型匹配不再要求模板与源指针相等：先绑定其中的 Rule captures，
再在隔离 World 中代入并只读 alpha-check。没有在 source 构造投影或修改类型。
`grshape` / `_g` 以同一规则匹配两种 shape 和动态 offsets；错误 shape 的优先规则被拒绝，
后续合法规则仍改写。`grwtype` / `_g` 从定义期 gap 翻为目标检查：
staged box 求值后才能确定的 RHS shape 仍受最终 type check 约束。

目前 generic 指 shape/offset 的 Nat 元变量；配置中含 `T: *` 的高 universe Rule list
仍需要显式适配配置的 universe，不等同于自动 universe polymorphism。
