# MimIR 图模式匹配实现方案

建议采用 **C++ 图匹配内核、独立的 Pattern 表示、现有 Rule 适配器、基于 World 重建的批量重写**。借鉴 TVM DPL 的表达式模式和图关系约束，以及 MLIR PDLL、PDL、PDLInterp 将规则描述、匹配执行和重写驱动分层的做法。先稳定匹配与提交语义，再扩展 Mim 语言接口。

目标包括单根 DAG 匹配、多根图匹配、带类型和分析条件的规则，以及保留共享的多节点替换。首版面向纯数据流子图；tensor fusion 是首个实际使用场景。内存、控制流和递归函数体的通用重写需要后续独立定义合法性。

文中的 MLIR 参考对象是 **PDLL**。除下面的实施状态外，后文 API、文件布局和阶段均为目标设计。

## 当前实施状态

已加入 P0/P1 内核、P2 单根 Rule 重写通路、P3 的 C++ 语义图与多根重写，以及 P4 的首批 tensor schema、现有 fusion 接入和真实三路矩阵乘合并。单根入口位于 [pattern.h](../include/mim/pattern/pattern.h)、[match.h](../include/mim/pattern/match.h)、[pattern/rule.h](../include/mim/pattern/rule.h) 和 [phase/rule_rewrite.h](../include/mim/phase/rule_rewrite.h)；P3、P4 入口见下节。

- `PatternProgram` 使用独立 arena 表达 Any、Exact、typed Literal、原始 Node、Capture、All、Either、Not、Type 和 Where，不通过 World 构造模式。
- `match` 执行带后续条件回溯的单根搜索，返回 Matched、No_Match、Unknown 或 Budget_Exceeded；失败不暴露部分 captures。
- 类型模式在结构匹配后运行，Where 在类型匹配完成后运行。Not 使用最终已绑定 captures 进行隔离探测，不向外导出绑定。
- `GraphView` 同时保存原始 Def 边和可选的算子语义视图。`OpRegistry` 明确哪些应用参数组是数据输入、需要展开几层 Tuple/Pack，以及算子的纯性。类型参数不会因为也是 Def 就自动成为语义数据输入。
- 原有 `only_used_by` 仍检查原始 operand owner；新增 `data_used_by`、`data_only_used_by` 检查语义消费者。重复槽位分别计数，多个槽位属于同一个算子时仍只有一个消费者。Open coverage、opaque 使用、符号包装或截断的解码不能证明排他使用。
- 有步数和活动递归深度限制。序列 binder 的 Var 类型可能需要构造 IR，因此当前类型查询对这类 Var 返回 Unknown。
- `RulePattern::match_lhs` 将现有 Rule 的 Var/Extract 元变量降低为 captures，延迟匹配其类型，并返回原始 metavariable 到源节点的绑定。
- `RulePattern::build_replacement` 为闭合表达式在同一 Driver 的独立 World 中导入 captures、求值 guard 并构造 RHS。仅接受归约为 true 的 guard，拒绝缺少绑定的 RHS 和类型不保持的替换；不创建 Hole 补缺，也不发布程序 roots。
- 开放表达式可传入正在重建图的 `Rewriter&`，复用其 Var/binder 映射；guard/RHS 的外部自由变量必须属于候选表达式的环境。缺少映射时返回 Unknown。
- `compile.rule_app` 接受显式 Rule 或 Rule tuple，按列表顺序选择首条满足 LHS 和 guard 的规则，完成一次自底向上的 World 重建。与 `compile.phases tt` 组合可迭代，identity rewrite 不请求额外轮次。
- Rule 模板存放在独立 World，在每次 phase 运行时重新导入当前 World。前后运行 cleanup 或在不动点迭代中重建 phase，不会保留已被删除的源图指针。模板自身不作为普通程序 root 发布，也不被该 phase 改写。

下面是当前可用 API 的示例，`op` 为一个已有的 curried 二元 axiom，`value` 为待匹配表达式：

```cpp
mim::pattern::PatternProgram patterns;
auto x = patterns.capture();
auto inner = patterns.node(mim::Node::App, {patterns.exact(op), x});
auto root = patterns.node(mim::Node::App, {inner, x});
auto outcome = mim::pattern::match(patterns, root, value);
if (outcome.result) {
    const mim::Def* captured = outcome.result->isa_capture(x);
    // Both arguments of the matched call are this exact Def.
}
```

已有 Rule 可通过下面的接口构造闭合表达式的替换，调用方检查结果后决定如何提交：

```cpp
mim::pattern::RulePattern compiled(rule);
auto target = source_world.inherit();
auto replacement = compiled.build_replacement(closed_expression, *target);
if (replacement.value) {
    // The replacement belongs to target; source roots have not been modified.
}
```

Mim 程序现在可直接指定要运行的规则，例如：

```mim
plugin compile;
axm step: Nat -> Nat;
rule collapse (x: Nat): step (step x) => step x;
extern lam result (x: Nat): Nat = step (step (step x));
extern lam _compile (): compile.Phase =
    compile.phases tt (compile.cleanup, compile.rule_app collapse, compile.cleanup);
```

该示例输出的 `result` 函数体为 `step x`。实际回归用例见 [rule_app.mim](../lit/rules/rule_app.mim)、[identity.mim](../lit/rules/identity.mim) 和 [arith.mim](../lit/rules/arith.mim)；后者将 `(x + 1) * 2` 重写为 `x * 2 + 2`。

当前验证通过 138 个 CTest（其中 38 个 Pattern/Rule 测试、30 个 P3 Graph 测试、15 个 P4 Tensor 测试）、91 个相关 lit 测试，另有 1 个按现有条件跳过。lit 范围包括 tensor、btensor、rules 和相关错误用例，包含原有 14 个 fusion 回归及 I64 三路矩阵乘的 LLVM 执行测试；4 个 rules 用例与 5 个新增 tensor 用例还各自通过 Mim 和 AST roundtrip，共 18 次。P3 以及本次改动的匹配／重建内核、tensor adapter、合并 phase 和 fusion 源码另作 AddressSanitizer 与 UndefinedBehaviorSanitizer 检查；基础 IR 库和其他插件未全部插桩，当前环境无法运行 LeakSanitizer，未作泄漏检查。验证使用未加载旧版外接 runtime/torch 插件的隔离构建；根目录现有 build 的配置仍受这些插件的旧 CMake 参数阻塞。

尚未完成的部分包括依赖类型实例化求值、开放 closure scope 的自动环境导入、通用 tensor dtype／shape 模式库，以及 P5 的 Mim 多根描述语法和性能优化。当前 `compile.rule_app` 仍是单根入口，P3 接口可由 C++ phase 调用；真实 tensor 多根使用入口是 `tensor.combine_products`。Rule adapter 不做逆归一化，也不猜测缺少绑定的 telescope；原始 Pack 边数不代表动态读取次数。

## P3 语义图与多根重写

P3 的可执行测试位于 [graph_pattern.cpp](../test/graph_pattern.cpp)，覆盖无共同 tuple 根的三路投影、跨根回溯、参数与结果投影路径、共享 recipe、CPS 参数、既有递归引用、作用域隔离及失败时不发布新图。

| 接口 | 作用 |
| --- | --- |
| [OpRegistry](../include/mim/pattern/op.h) | 按 axiom flags 注册数据参数组、包装层数及纯性，也允许只读的自定义解码器 |
| [GraphView](../include/mim/pattern/graph.h) | 为选定 roots 和可选函数 scope 建立原始边、语义输入边、输出边界和快照 |
| [GraphPattern 与 find_matches](../include/mim/pattern/graph_match.h) | 多个 anchor 共用绑定、类型约束和回溯状态，返回 anchors 与 captures |
| [RewritePlan](../include/mim/pattern/rewrite.h) | 用 import、typed literal、axiom call、tuple 和 extract 组成共享的 replacement DAG |
| [rewrite_graph 与 GraphRewrite](../include/mim/pattern/rewrite.h) | 检查计划、重建独立 World；检查通过后调用 publish 一次性提交 |

### 匹配范围与搜索状态

默认 anchor 只枚举已注册的语义算子，也可显式选择原始 Nodes。多个 anchor 必须通过共享的正向 capture 或 `used_by` / `only_used_by` 关系连通，默认不能绑定到同一个节点。`UseSite` 可限制参数组、参数包装路径和结果投影路径。

搜索先按结构缩小候选，再优先匹配候选较少的 anchor。所有根共享现有匹配器的 undo log，后续根失败时可以回溯到前一个根的 OR 分支。类型与 Where 条件在跨根绑定完成后检查。每个 anchor tuple 返回一份确定的成功绑定，不重复输出同一个 tuple。

`GraphMatches::status` 区分完整枚举 `Complete`、存在未知条件 `Unknown`、预算耗尽 `Budget_Exceeded` 和达到结果上限 `Result_Limit`。后面三种状态不代表“完整搜索后无匹配”；其中已返回的每个匹配仍通过了自身约束。步数、候选数、结果数、递归深度及包装展开都有边界。

### 替换与提交

算子纯性默认未证明。注册可以给出无条件 pure，也可以提供基于完整 curry 参数组的只读纯性检查；源节点和 recipe 的目标参数都必须通过检查，后者在调用 normalizer 前执行。一个 recipe 可以被多个输出引用，求值只执行一次。替换的类型必须保持，导入值必须来自当前图，计划之间不能有重叠写入或读写冲突。

导入的 immutable 子图也参与替换依赖检查，因此 `a → import(b)`、`b → import(a)`，以及通过更深子表达式形成的循环都会在构建目标图前被拒绝。带排他使用条件或 native Where 的匹配保守地记录更宽的读取范围，避免批量提交使分析条件失效。

有 scope 时，目前要求一个已定义、闭合的 mutable Lam。重建为它建立新 stub 和 Var 映射，并为其他函数使用独立 importer：同一个 hash-consed Def 在其他函数内保留原实现。所有原 external roots 都被保留。无 scope 时只重建显式的闭合 roots，结果可供调用方继续处理，不允许将其直接发布为整个程序。

`publish` 前检查源图的存储代次、mutable 状态和 external 集合。中间运行 cleanup、修改其他 external 或增加 external 都会使旧结果无法提交。World 的存储 token 随 IR 一起交换，token 的持有不保留旧 Def arena，也不会因内存地址复用而使旧句柄复活。含 Exact/Literal 引用的 PatternProgram 和 RulePattern 也会拒绝在 World 重建后继续使用。

原子提交指校验全部通过后交换 World；调用方仍须独占 World 修改。提交后旧 Def 指针和匹配句柄失效，后续轮次应重新建立视图及引用当前 World 的模式。

### 三路共享计算示例

下面是当前 C++ API 的用法。假设 `q/k/v: Nat → Nat`、`combined: Nat → «3; Nat»`，且插件已证明四个算子纯、combined 的三个输出分别等价于原三路计算。框架检查匹配、类型和作用域，不证明这个语义等价关系。

```cpp
#include <array>
#include <stdexcept>
#include <mim/pattern/rewrite.h>
#include <mim/world.h>

using namespace mim;
using namespace mim::pattern;

Lam* merge_three(Lam* fn, const Axm* q, const Axm* k,
                 const Axm* v, const Axm* combined) {
    OpRegistry ops;
    for (auto axm : {q, k, v, combined})
        ops.add(axm->flags(), 1, {{0}}, true);
    GraphView view({fn->body()}, {.operators = &ops, .scope = fn});

    PatternProgram p;
    auto input = p.capture();
    auto call = [&](const Axm* axm) {
        return p.node(Node::App, {p.exact(axm), input});
    };
    GraphPattern pattern(p);
    auto aq = pattern.anchor(call(q));
    auto ak = pattern.anchor(call(k));
    auto av = pattern.anchor(call(v));
    auto found = find_matches(pattern, view);
    if (found.status != SearchStatus::Complete)
        throw std::runtime_error("incomplete graph search");
    if (found.matches.empty()) return fn;

    const auto& match = found.matches.front();
    RewritePlan plan(match);
    auto shared = plan.call(combined, {plan.import(match.isa_capture(input))});
    plan.replace(match.at(aq), plan.extract(shared, 0));
    plan.replace(match.at(ak), plan.extract(shared, 1));
    plan.replace(match.at(av), plan.extract(shared, 2));
    std::array plans{&plan};
    auto next = rewrite_graph(view, plans);
    auto new_fn = next.scope();
    next.publish(view);
    return new_fn;
}
```

这个例子不要求原图存在 `(q, k, v)` tuple。三个算子可以处在函数内不同的使用路径上，替换后引用同一个新计算。CPS 返回参数也已覆盖；tuple 的 η 归约还可将三个结果投影重新合并成整个 combined 结果。这里验证的是通用图机制；下面的 P4 实现进一步覆盖真实权重拼接和 shape 条件，后端收益分析尚未加入。

## P4 tensor 接入

实际实现入口是 [tensor/pattern.h](../include/mim/plug/tensor/pattern.h)、[tensor/pattern.cpp](../src/mim/plug/tensor/pattern.cpp)、[fuse.cpp](../src/mim/plug/tensor/phase/fuse.cpp) 和 [combine_products.cpp](../src/mim/plug/tensor/phase/combine_products.cpp)，可执行单元测试在 [tensor_pattern.cpp](../test/tensor_pattern.cpp)。

### 算子视图与现有 fusion

`tensor::graph::register_operations` 注册以下语义输入，参数组编号从 0 开始：

| 算子 | 语义数据输入 | 纯性 |
| --- | --- | --- |
| `map_reduce_post` | 第 7 组内的 combiner slots `{0,i}` 与 epilogue slots `{1,j}` | 不默认证明；combiner 和 post 可以是任意代码 |
| `product_2d` | 第 2 组的左右张量 | 仅识别到标准 Nat 或 mode=0 的 wrapping integer ring 时证明 |
| `broadcast` | 第 1 组的第 2 个字段；排除输入／输出 shape | 纯 |
| `concat` | 第 3 组按 `nis` 展开的张量列表 | 纯 |
| `slice` | 第 3 组的张量 | 纯 |

解码只检查现有节点，不构造 Extract 或触发归一化。singleton 参数组按 schema 处理，真正的 tensor-valued Pack 不会被当作参数包装无限展开。无法只读拆解的 opaque aggregate、融合了多维 shape 的参数 Pack、符号 slot 数量或预算截断返回 incomplete；这些情况不能证明排他使用。

现有 producer fusion、read-through、epilogue fusion 的 `map_reduce_post`／`broadcast` 结构筛选已切换到共享的 PatternProgram。索引映射组合、injectivity、shape 一致、无 reduction 及 identity post 等合法性判断与 CPS 构造仍保留在原生分析／构造函数中。

epilogue 的单次消费门槛现在使用覆盖所有可达 binder body 的语义图：必须只有一个数据输入槽，并证明没有其他用户或输出逃逸。旧计数将 `Pack(2, producer)` 的物理 body 边只计一次，可能把一个 consumer 两次读取的 reduction 当作单次消费；现在会拒绝这种 sink，避免既保留原 reduction 又执行一份融合后的 reduction。共享 read-through 的新增使用仍由 `shared_` 跟踪，重建前后的对应仍由 `new2old_` 保留。回归见 [fuse_symbolic_shared_slot.mim](../lit/tensor/fuse_symbolic_shared_slot.mim)。

### 真实三路矩阵乘合并

`tensor.combine_products` 使用 P3 的三 anchor 匹配、Where 条件和共享 recipe，将共享左输入的三个 `product_2d` 改为：

```text
Q = X × Wq                         W = concat(Wq, Wk, Wv, axis=1)
K = X × Wk          →              merged = X × W
V = X × Wv                         Q/K/V = 三段对应列范围的 slice(merged)
```

三个权重允许有不同列宽；替换会计算各自的列偏移，保持每个结果的声明类型。三个乘法必须有相同 ring 和左输入，并且显式的行数、收缩维度必须相同，不能仅凭 size-1 轴折叠后的 tensor 类型相等推断逻辑 shape 相等。行数和收缩维度可为同一个符号值；列宽目前须为正整数字面量，合计不得溢出 `u64`。

当前支持标准 Nat ring，以及 `core.wrap.add/mul` 的 mode=0 整数 ring；不接受浮点、自定义函数、非零 overflow mode 或未知 ring。该规则改变输出分组，不改变每个输出元素的 fold；I64 执行用例包含普通值和环绕溢出输入，逐元素与三个独立矩阵乘的标量参考比较。见 [combine_products.mim](../lit/tensor/combine_products.mim)、[combine_products_exec.mim](../lit/tensor/combine_products_exec.mim) 和 [数值参考](../lit/tensor/combine_products_aux.cpp)。

该 phase 是显式入口，应在 `lower_tensor` 前运行，例如：

```mim
extern lam _compile (): compile.Phase =
    compile.phases ff (
        compile.cleanup,
        tensor.combine_products,
        tensor.lower_tensor,
        tensor.fuse_tensor,
        compile.cleanup,
    );
```

目前每次调用至多重写每个闭合 external Lam 中的一组三路匹配；需要多轮时可放入 `compile.phases tt`，每次成功重写都减少两个 product 节点。搜索状态不是 Complete 时整组跳过并记录诊断，不用部分枚举结果提交。其他函数共享的旧 Def 保留原实现。尚未加入后端收益模型，因此此 phase 不放入默认 pipeline；额外 concat 和 slice 的成本可能抵消合并收益。

这完成了 P4 的首批真实 tensor 接入与正确性验收，尚不等于完整的 TVM DPL：缺少通用 dtype／shape helper、浮点 QKV 合并、任意数量分支的合并、通用融合函数提取和后端分发。现有 `fuse_tensor` 继续使用原生 CPS recipe，未将其改成一套通用 native rewrite 回调注册系统。

## 源码依据

本地版本：MimIR `468c4c0120`，TVM `7a48cb5`，llvm-project `bab6b327a1`。工作区中的现有文档和未提交内容不等同于这些提交的已发布功能。

| 对象 | 源码入口 | 对方案的作用 |
| --- | --- | --- |
| TVM 表达式模式 | [pattern.py](/workspaces/ml-compiler/tvm/python/tvm/relax/dpl/pattern.py)、[dataflow_matcher.cc](/workspaces/ml-compiler/tvm/src/relax/ir/dataflow_matcher.cc) | 模式对象身份、组合模式、captures、失败回滚 |
| TVM 图关系 | [context.py](/workspaces/ml-compiler/tvm/python/tvm/relax/dpl/context.py)、[dataflow_block_rewriter.cc](/workspaces/ml-compiler/tvm/src/relax/ir/dataflow_block_rewriter.cc) | 多根搜索、producer consumer 约束、多 binding 替换 |
| TVM 重写 API | [rewrite.py](/workspaces/ml-compiler/tvm/python/tvm/relax/dpl/rewrite.py) | 区分局部表达式重写与图级重写 |
| MLIR 规则前端 | [PDLL.md](/workspaces/llvm/llvm-project/mlir/docs/PDLL.md)、[MLIRGen.cpp](/workspaces/llvm/llvm-project/mlir/lib/Tools/PDLL/CodeGen/MLIRGen.cpp) | 声明式规则、native constraint、native rewrite |
| MLIR 匹配编译 | [PredicateTree.h](/workspaces/llvm/llvm-project/mlir/lib/Conversion/PDLToPDLInterp/PredicateTree.h)、[ByteCode.cpp](/workspaces/llvm/llvm-project/mlir/lib/Rewrite/ByteCode.cpp) | 共享判定、匹配控制流、可替换的执行后端 |
| MLIR 驱动 | [PatternApplicator.cpp](/workspaces/llvm/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp)、[GreedyPatternRewriteDriver.cpp](/workspaces/llvm/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp) | benefit、规则选择、驱动与 matcher 分工 |
| MimIR 现有规则 | [rule.h](../include/mim/rule.h)、[rule.cpp](../src/mim/rule.cpp) | `Rule/Reform`、元变量、一致性、guard 与替换 |
| MimIR 重写与作用域 | [rewrite.h](../include/mim/rewrite.h)、[phase.h](../include/mim/phase.h)、[def.h](../include/mim/def.h) | World 重建、binder 映射、共享、递归 stub、自由变量 |
| MimIR fusion | [fuse.cpp](../src/mim/plug/tensor/phase/fuse.cpp)、[tensor.h](../include/mim/plug/tensor/tensor.h) | 实际消费者分析、访问映射分析和新旧节点对应 |

公开说明参见 [TVM DPL](https://tvm.apache.org/docs/deep_dive/relax/dpl.html)、[MLIR PDLL](https://mlir.llvm.org/docs/PDLL/)、[PDL](https://mlir.llvm.org/docs/Dialects/PDLOps/) 与 [Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)。具体行为以以上本地源码为准。

## 当前基础与实现缺口

### 现有 Rule 能匹配共享子图

`Rule::its_a_match_` 递归检查节点、类型和操作数；属于 Rule 的 `Var` 或其 `Extract` 是元变量，重复出现时要求绑定到相同 `Def*`。因此已有 matcher 能识别单个根可达的共享 DAG，不能简单归类成只能匹配树。

`Rule::replace` 用 `VarRewriter` 实例化 guard 和 RHS，仅在 guard 归约为 `lit_tt()` 时替换。这可以作为兼容入口，但目前存在以下限制：

- 绑定结果通过 `Def2Def` 隐式传递，没有稳定的 captures、失败诊断和组合模式接口。
- 匹配过程先递归类型，可能在决定 rank、shape 的元变量尚未绑定时遇到已归一化的类型。
- 失败路径没有完整的事务式绑定回滚；加入 OR、NOT、多根回溯前必须解决。
- 普通结构递归没有通用的节点对访问状态，不能直接扩展为递归 binder 匹配。
- 字面量匹配的补偿分支会实例化模式，`tuple_of_dict` 还可能填充 Hole；这与只读、可回溯的匹配契约不一致。
- `replace` 依赖找到 rule binder 的替换值；无捕获常量规则和只绑定部分 telescope 的规则需要专门处理。

实施前的基线包含 Rule 定义和前端 emission，尚未接线 `compile.rule_app` / `compile.rule_drop`。目前已新增显式传入规则的 `compile.rule_app`，不依赖扫描 external Rule roots；历史分支中的 `compile.rule_drop` 仍不属于此实现。已有 [rule fusion 实验文档](fusion-as-rules.zh-CN.md)明确区分了历史分支能力，其中的历史覆盖率不是当前实现的验收结果。

### 图使用关系需要独立索引

`Def::users()` 返回的是 mutable 局部引用关系，不是任意 immutable 算子的完整 use list。现有 `tensor::count_consumers` 从 externals 遍历，穿过 Tuple/Pack 包装归属到消费者，同一参数列表中重复使用会重复计数。它提供了实践经验，但不能直接充当带 consumer 身份、参数位置和图边界的通用索引。

MimIR 普通 let 通常直接复用 `Def*`，无需照搬 Relax 的 `var2val`。但 immutable 节点可能在多个 binder 或函数中共享，图节点身份、作用域中的出现位置和实际计算次数必须分开。

### 重写基础应继续复用

`Rewriter` 已维护分作用域的新旧映射，mutable 使用先建 stub 再填 operands 的协议。`RWPhase` 以 old/new World 分离重建，最后交换 World。新框架应复用这些机制，并为匹配快照、批量替换和失败时不发布新图提供明确协议。

## 参考设计的取舍

| 设计问题 | TVM DPL 的参考 | MLIR 的参考 | MimIR 选择 |
| --- | --- | --- | --- |
| 模式表达 | DFPattern 组合、共享模式对象 | PDLL 变量和 PDL handles | 独立 PatternId 与 CaptureId；Rule 可降低到同一表示 |
| 图关系 | PatternContext、used_by、only_used_by | 从 root 绑定其他 IR 实体 | 独立 GraphPattern，可声明多根和边约束 |
| 附加条件 | dtype、shape、check callback | native constraint | 依赖类型约束加 C++ 分析回调 |
| 替换 | rewrite_call、rewrite_bindings | PatternRewriter 和 native rewrite | 单根入口与多节点 RewritePlan，共用 World 重建 |
| 执行优化 | memo 和回滚 | PDLInterp、共享 predicate tree、bytecode | 先解释执行与根索引，测量后增加共享判定 |
| 规则调度 | 回调和 pass 策略 | benefit、PatternApplicator、不同 driver | 显式驱动策略，避免把规则顺序藏进 matcher |

PDLL 是规则描述语言，PDL 是规则表示，PDLInterp 和字节码是执行层。MimIR 第一阶段只需实现相同的职责划分，无需依赖 MLIR 或立即增加一个核心 dialect。另需注意，本地 `CPPGen.cpp` 生成的是承载 PDL 文本和 native callbacks 的 `PDLPatternModule` 包装，不能据此宣称所有 PDLL 规则都直接生成了手写式 C++ matcher。

## 总体架构

```text
现有 Rule       C++ PatternBuilder       后续 Mim pattern 接口
     \                 |                       /
      +---------- PatternProgram --------------+
                         |
                verify / root indexing
                         |
                MatchEngine + MatchContext
                         |
                   MatchResult
                         |
             constraints / rewrite builder
                         |
                     RewritePlan
                         |
             validate / select / rebuild World
                         |
                  下轮图快照或结束
```

建议分成六个可独立验证的部件：

1. `PatternProgram`：保存结构模式、captures、关系约束及规则元数据。
2. `GraphView`：定义匹配边界，并构建原始边和可选的算子语义边。
3. `MatchEngine`：只负责搜索与约束求值，输出匹配结果。
4. `ConstraintRegistry`：注册有输入签名和依赖信息的 native predicates。
5. `RewritePlan`：描述本次匹配产生的所有替换及依赖。
6. `GraphRewritePhase`：选择计划、验证、重建、更新快照并报告收敛状态。

## Pattern 表示与匹配契约

### PatternProgram

第一版使用普通 C++ arena，节点用稠密 ID 引用。Pattern 身份不能被普通表达式 hash consing 意外合并：两个独立 wildcard 可以绑定到同一个或不同的 Def，复用同一个捕获模式则必须绑定到同一 Def。

建议支持以下构造：

| 构造 | 语义 |
| --- | --- |
| `Any` | 匹配任意允许的 Def，默认不穿透边界 |
| `Capture(id, p)` | 将满足 p 的整个原节点绑定到 capture，支持 as pattern |
| `Exact(def)` | 匹配指定节点身份，跨 World 导入时显式重定位 |
| `Literal(type, bits)` | 同时检查字面量类型和值 |
| `Node(kind, flags, operands)` | 原始 Def 结构模式，检查有语义的 flags |
| `AxmCall(id, groups, mode)` | 按 annex flags 和应用层级匹配调用 |
| `Tuple`、`Pack`、`Extract` | 精确结构，或显式启用有界 aggregate view |
| `TypeOf(p)` | 对被匹配节点的类型施加模式约束 |
| `All`、`Either`、`Not` | 合取、析取、否定，具有明确回滚规则 |
| `Where(predicate, captures)` | 所需 captures 绑定后执行的条件 |
| `Graph(anchors, relations)` | 多 anchor 搜索，检查共享和使用关系 |

`AxmCall` 应复用 `Axm::isa`、`App::uncurry_args` 的知识，保留 curry groups、饱和状态和 subtype/subtag，避免把所有 App 链平铺成没有语义的参数数组。参数组内的 tuple 只按明确的 schema 展开；不能因为类型参数也是 Def 就把它当作运行时数据输入。

独立模式表示还能避免构建 pattern 时被 `World::app` 的 PE、normalizer 或单元素 tuple 归一化提前消掉。Rule adapter 只能处理 emission 后仍可见的结构，不能恢复已丢失的源码结构；必要时给出可定位的诊断。

### 身份与组合规则

- 同一 capture 重复出现要求同一节点身份；alpha equivalence 和代数等价分别通过显式 predicate 表达。
- 不同 capture 默认允许相同节点，需区分时使用 `distinct(a, b)`。多根的独立算子 anchor 默认互异，可显式声明允许别名。
- `All(p, q)` 对同一候选运行两个模式，共享绑定环境并检查一致性。
- `Either(p, q)` 按声明顺序尝试；分支失败恢复全部状态。左分支 Unknown 时仍可尝试右分支；右分支成功则整体成功，否则保留 Unknown。RHS 可用的 captures 必须在所有成功分支中定义。
- `Not(p)` 在隔离状态中探测，不向外导出捕获；预算耗尽或未知不能当作否定成功。
- 首版 pattern 本身必须无环，mutable 默认只按身份匹配。目标图遍历使用 visited set 并在 binder 边界停止，因此递归 IR 不会引起无限下降。

Match API 至少区分 `Matched`、`NoMatch`、`Unknown`、`BudgetExceeded` 和无效模式。driver 对 Unknown 不应用规则，同时保留原因；诊断和否定逻辑必须保留这种区别。

### 对外 API 草案

```text
compile(patterns, registry) -> PatternProgram 或诊断
make_view(world, roots, scope_policy, op_views) -> GraphView
match(program, pattern_id, candidate, context) -> MatchOutcome
find_matches(program, graph_pattern_id, context, budget) -> MatchStream
make_plan(rule, match_result) -> RewritePlan 或拒绝原因
rewrite(world, plans, driver_options) -> RewriteReport
```

`MatchContext` 只读引用 GraphView、分析结果和预算；MatchStream 提供最终是否完整枚举的状态。`RewriteReport` 记录成功提交数、拒绝原因、轮数和 fuel 状态。MatchResult 与图句柄受 snapshot 生命周期约束，不能跨 World 发布后继续使用。RuleSet 中的执行程序可复用，但含具体 Def 的 Exact 模式及实例化数据必须随 World 重新定位。

## GraphView 的范围与边

### 图范围

`GraphView` 包含 scope 标识、快照 epoch、计算 roots、节点集合、输入边界和输出边界。初期以一个明确的纯数据流区域为单位，遍历 immutable 数据依赖，到 Lam、Pi、递归 mutable、opaque annex 或未支持算子处停止。lambda 自由变量可以作为输入叶子，matcher 不自动进入 lambda body。

`TypeOf` 单独访问类型图。图边分类至少区分数据、callee、类型与 binder 关系，类型引用不能计入数据消费者。

所有由区域外可达的输出、函数返回值、未知使用和 opaque 使用，必须保留为边界使用。若区域不能证明已覆盖某节点的全部相关用户，`only_used_by` 返回 Unknown。共享 Def 在不同 scope 的候选位置以 `(scope, def)` 区分，涉及 scope 的替换也使用这个键。

首版可以保守地拒绝跨 scope 共享的内部节点消除。扩展局部重写时，由 `ScopedRewriter` 在指定 binder 环境内应用计划，避免把一个函数中的匹配扩散到其他函数。

### 原始边与算子语义边

同时提供两层 view：

- 原始 view：`(producer, owner, operand index, edge kind)`，用于精确结构匹配。
- 算子 view：`(producer, consumer operation, argument group, slot path)`，用于数据流关系约束。

算子 view 由插件提供的 `OpView` 解码 App 链及参数包装。tensor 可先覆盖 `map_reduce_post`、broadcast 和需要的投影；未知算子保留原始边，不猜测参数意义。

Tuple、Pack、Extract 的处理必须保留投影路径和符号 arity。Tuple 作为真正的返回值时就是输出边界，不能一律当作可穿透包装。Pack 可保留符号 multiplicity，不能直接假设 `ops().size()` 等于实际使用次数。对确切展开次数无法证明的查询返回 Unknown。

索引至少维护：节点 kind、annex ID、producer 到 use edges、consumer 到 inputs、边界使用，以及候选的稳定遍历序号。先由一次可达图扫描构建；有界视图展开另计预算，避免共享包装导致隐式指数展开。

### 使用关系的精确定义

```text
used_by(p, c, path?)
    存在一条 p 到 c 的数据 use edge，可约束参数位置。

only_used_by(p, c)
    p 至少被 c 使用一次；全部可观察数据消费者恰为 {c}；
    没有输出边界或未知外部使用。

use_count(p)
    use occurrence 的数量，重复槽位分别计数。

consumer_count(p)
    不同 consumer occurrence 的数量。
```

因此 `add(p, p)` 可满足 `only_used_by(p, add)`，但 `use_count(p) == 2`。若优化还要求只计算或读取一次，必须另加 occurrence 或访问映射约束。消费者个数不是动态执行次数，不能用来替代 injectivity 分析。

`no_external_users(matched_nodes, outputs)` 与 `only_used_by` 分开：多输出 fusion 允许显式 outputs 被外部使用，但内部值若逃逸，就必须额外保留输出或拒绝该计划。

## 匹配算法

### 单根搜索与回滚

每次尝试分配 `MatchState`：稠密 capture slots、待检查约束、binder 环境标识、已匹配节点与 use edges，以及 undo log。

```text
try_match(pattern, candidate, state):
    checkpoint = state.mark()
    检查 root kind / annex / arity / flags
    若 capture 已绑定，则检查身份一致
    匹配 operands，记录绑定和类型 obligation
    执行当前已具备全部输入的约束
    失败、未知或超限：恢复 checkpoint，返回相应状态
    成功：保留绑定并返回 Matched
```

undo log 必须覆盖 capture 写入、约束状态、range bindings 和访问状态，不能只清空部分 map。完全匹配成功后才生成不可变 `MatchResult`。

不要只用 `(PatternId, Def*)` 缓存任意匹配结果：结果可能依赖已绑定 captures、scope 和待定类型条件。第一版只缓存不含捕获依赖的结构检查；完整 memoization 待测量后再做，key 必须包含相关环境信息。

### 多根搜索

将多根匹配作为约束满足问题处理：

1. 为各 anchor 按 kind、annex、已知类型筛选候选。
2. 选择候选最少的 anchor，绑定并检查其局部结构。
3. 优先沿 inputs、users 和已知 capture identity 扩展邻接 anchor。
4. 每次绑定后执行已就绪的关系条件；失败回滚，继续枚举其他候选。
5. 所有 anchors 和必要约束成功后，输出结果；按 anchor tuple 去重。

第一版只接受通过数据边或共同 capture 连接的多根模式。没有连接约束的任意多根笛卡尔积搜索暂不开放。搜索提供候选数、回溯步数和结果数上限；超限报告原因，不能静默当作完整的“无匹配”。

例如 Q/K/V 三个独立算子根可以通过同一个输入 capture 连接，先找一个投影，再只从该输入的用户中寻找其他投影，无需程序里已有 `(q, k, v)` tuple 根。首版以测试算子验证通用能力，实际 QKV 合并还需独立的算子语义和 shape 合法性检查。

### 依赖类型与归一化

保留 MimIR 的依赖类型优势，但将绑定和校验分开：

1. 结构匹配优先绑定能够直接观察到的 rank、shape、type 和值参数。
2. 类型模式可产生形如 `expected_type[captures] == candidate.type()` 的 obligation。
3. 编译 PatternProgram 时记录每个 obligation 所依赖的 CaptureIds；相关捕获齐备后再实例化和检查。
4. 不能从归一化结果唯一恢复的参数，不做逆归一化猜测。例如 `«1; T»` 已塌缩为 T 时，应从显式 rank 参数、其他约束或插件 view 绑定 rank。
5. unresolved obligation 返回 Unknown；缺少 RHS 必需 capture 的规则在验证期拒绝。

类型条件不是一律放到最后：不依赖未绑定 capture 的廉价检查应尽早执行，以便剪枝。

匹配不得改写源 World、设置 Hole 或修改 mutable。需要实例化类型或 guard 时，在拥有显式源目标映射的临时 World 中导入闭包和 binder 环境，使用有 fuel 的求值。图分析谓词直接读取原快照中的 capture handles，不能对临时副本重新计算消费者关系。

## 约束接口与语义分析

```cpp
// 接口草案，非当前 MimIR API。
struct ConstraintSpec {
    Symbol name;
    CaptureIds inputs;
    ConstraintKind kind;  // LocalType, LocalValue, Graph, NativeAnalysis
};

CheckResult check(const ConstraintSpec&, const MatchResult&,
                  const MatchContext&);  // True / False / Unknown
```

规则验证器检查输入绑定、类型签名、NOT 捕获和 OR 分支可见性。native callback 在插件注册时声明所依赖的图分析；回调只读，不能自行重写程序。

建议首批谓词包括 `same_type`、`same_shape`、`is_literal`、`distinct`、`used_by`、`only_used_by`、`no_external_users`，以及 tensor 专用 `reads_injectively`。后者可提取和复用现有 C++ 分析，不必先提供通用 IR 反射语言。

guard 仅在能够证明 True 时接受。符号 shape、opaque map、未绑定参数均不能默认放行。涉及图使用关系的条件只在 MatchContext 中求值，不注册为与 World 版本无关的常驻 normalizer 结果。

首版只允许已知纯算子参与图级消除、复制和合并；缺少纯性信息的算子拒绝相关重写。副作用支持后续通过插件 effect 描述、mem token 与 alias 分析加入。匹配到结构本身不构成重写正确性的证明。

## 多节点替换与 World 提交

### RewritePlan

```cpp
// 类型形状示意；Recipe 在目标 World 中构造值。
struct MatchResult {
    SnapshotId snapshot;
    ScopeId scope;
    CaptureBindings captures;  // 指向旧图
    MatchedNodes nodes;
    MatchedEdges edges;
};

struct RewritePlan {
    SnapshotId snapshot;
    ReadDependencies reads;
    ScopedReplacements replacements;  // (scope, old Def) -> Recipe
    OutputBoundary outputs;
};
```

recipe 可以引用旧 capture、目标 builder 参数和其他 recipe 结果。一次 recipe 实例化产生的公共节点由所有替换共享，例如 Q/K/V 三个替换共同引用同一个合并后的矩阵乘，而不是分别构造三份。

### 提交流程

1. 在固定旧图快照上收集匹配，运行约束，生成 recipe，不提前连接新图。
2. 验证计划仍属于当前 snapshot、替换 key 唯一、scope 合法、输出边界完整。
3. 按 driver 策略选择计划。首版对共享已匹配内部节点、重叠替换和读写依赖冲突采取保守互斥。
4. 将替换表提前安装到重建驱动；某个旧节点第一次被请求时就走对应 recipe，避免先建立普通映射再补替换。
5. 在目标 World 中重写所有捕获引用和 recipe；使用既有 binder substitution、stub 和 seal 协议。禁止直接把旧 World 的 Def 塞入目标 operands。
6. 所有替换统一做类型与自由变量校验，保留 externals、annex opacity 和适用的 debug 信息。
7. 全部完成后发布目标 World；失败则丢弃目标，旧图保持可用。每次发布销毁旧快照的索引和结果。

多个替换之间允许无环 recipe 依赖，但要检查替换依赖环。首版不支持以 recipe 新建递归 SCC，也不支持改变被替换值的对外类型。需要变更函数签名的 lowering 继续使用专门的 RWPhase。

替换作用于选定 scope 内该值的全部使用。若只想改变一条边，应重建并替换对应 consumer，不能把 producer 全局映射当作局部改边。改变跨 binder 共享节点时必须使用 scope 路由，或保守拒绝。

计划的“原子”是编译器图发布语义，不表示任意 native callback 的外部副作用可以撤销；因此 callbacks 必须限制为框架管理的目标构造。

### 快照与迭代

首版不在构建目标 World 的中途继续使用旧 use 索引匹配新节点。每轮按旧快照匹配、选择一组独立计划、重建一次，然后重新扫描新图。这样能避免 fusion 中新图出现的新共享关系被旧消费者计数遗漏。

如果未来引入局部 worklist，需要先提供精确 use index 更新、关系约束失效和 scope 更新协议。单独增加 `new2old` 映射不足以维护完整图关系。

## 驱动策略与 Rule 兼容

matcher 不决定优化策略。建议最初提供两个明确入口：

- `Once`：对一个快照应用所选的独立计划，生成的新图留给后续 phase。
- `Greedy`：在每轮新快照继续匹配，直到没有有效替换或达到 fuel。

规则按显式 benefit 降序，平局按注册顺序和稳定候选序号处理。benefit 只表达选择偏好，不证明语义合法或优化必然有收益。首版无需动态成本模型。

组合 phase 时，将 GraphRewritePhase 的内部迭代与 `PhaseMan` 的外层 fixed point 分开配置，避免重复迭代。只有成功发布了具有可观察变化的图才 `invalidate()`；recipe 返回原值或规范化后等价于原图时不应触发无休止重跑。旧新 World 指针不同，不能只靠指针不等判断变化。

设置 rewrite 数量和 round 上限，报告触发上限的规则与匹配；可对忽略 gid/debug 的结构指纹记录重复图，再验证是否发生循环。所有规则共同保证终止通常不现实，因此 fuel 是必要的执行契约。

`FuseOpsByPattern` 式的有序分区、Composite 提取和 backend dispatch 作为后续独立 driver；不能默认等同于“每个节点尝试所有规则到不动点”。

### Rule adapter

将现有 `Rule::lhs` 降为结构模式，Rule binder 的 Var/Extract 降为 capture，guard 降为延迟条件，RHS 降为实例化 recipe。保留旧 `its_a_match` / `replace` 签名的适配层，迁移中用回归测试固定行为变化。

迁移必须明确旧 matcher 的字面量补偿路径：可证明且不会修改源图的封闭求值可放入兼容求值器；缺参猜测和填 Hole 路径不进入新内核。对于未支持形式给出诊断，不能悄悄声明完全兼容。

新增显式 `RuleSet` 和 compile phase 参数，规则仅在被选择的 pipeline 中生效。不要把全部规则默认 externalize 为程序 root，也不要自动为普通 `World::app` 安装带全图条件的 rewrite。构造时 normalization 与图级优化保留各自职责。

## 接口使用示例

以下均为设计伪代码，`OpA`、`OpB` 是插件注册的纯测试算子。

```cpp
auto x = p.capture("x", p.any());
auto a = p.capture("a", p.axm_call(OpA, {x}));
auto b = p.capture("b", p.axm_call(OpB, {a}));

auto rule = p.rule("combine_a_b", b)
    .where(only_used_by(a, b))
    .where(same_type(a, b))
    .benefit(10)
    .rewrite([](BuildContext& ctx, const MatchResult& m) {
        return ctx.replace(m["b"], ctx.call(CombinedAB, {ctx.import(m["x"])}));
    });
```

tensor 的 producer consumer 模式可用同一方式捕获完整 `map_reduce_post` App，再调用现有 C++ recipe 构造 fused combiner、maps 和 epilogue。结构部分迁入 pattern 后，injectivity、shape 兼容、reduction 和 pack 布局的判断仍保留为显式分析条件。

多根模式的使用形状：

```text
x = capture(any)
wq, wk, wv = capture(any), capture(any), capture(any)
q = capture(ProjectQ(x, wq))
k = capture(ProjectK(x, wk))
v = capture(ProjectV(x, wv))

graph anchors(q, k, v)
where distinct(q, k, v), compatible_projection_shapes(q, k, v)
rewrite:
    combined = CombinedProjection(import(x), import(wq), import(wk), import(wv))
    replace q with result(combined, 0)
    replace k with result(combined, 1)
    replace v with result(combined, 2)
```

三个输出可以拥有其他用户；重写必须统一更新三者在所选 scope 中的使用，并保证 combined 只构造一次。

## 文件布局与实施阶段

建议新增的核心目录为 `include/mim/pattern/` 和 `src/mim/pattern/`，避免 `rule.cpp` 同时承担语言表示、图搜索和 pass 驱动。

| 阶段 | 主要文件与改动 | 完成条件 |
| --- | --- | --- |
| P0 图范围与模式表示 | `pattern.h/.cpp`、`graph.h/.cpp`；测试 OpView | 原始边、语义边、边界使用、scope 身份有可执行测试；递归目标可安全遍历 |
| P1 单根匹配 | `match.h/.cpp`、`constraint.h/.cpp` | captures、共享 DAG、AND/OR/NOT、回滚、Unknown、类型 obligations 正确 |
| P2 Rule 与局部重写 | `rule_adapter.h/.cpp`、`phase/graph_rewrite.h/.cpp`；compile 插件显式 RuleSet 入口 | 现有 Rule 支持子集跑通，无捕获规则可用；失败不发布新图；Once/Greedy 有界 |
| P3 图关系与多根 | 扩展 `graph.cpp`、`match.cpp`、`rewrite.h/.cpp` | used_by、only_used_by、多 anchor 回溯和共享 recipe 的批量替换完整通过 |
| P4 tensor 接入 | tensor OpView 与 constraint/recipe 注册；逐条迁移 Fuse 的结构筛选 | 现有 fusion 正反例保持语义；重复读取、共享和符号形状边界覆盖齐全 |
| P5 表达与性能 | 根据 P0 至 P4 使用反馈决定 Mim pattern 语法、range 模式和共享判定编译 | 多前端共用 PatternProgram；优化有测量收益，解释器可作对照 |

核心 sources 加入 [src/mim/CMakeLists.txt](../src/mim/CMakeLists.txt)，doctest 加入 [test/CMakeLists.txt](../test/CMakeLists.txt)，集成样例放 `lit/pattern/`。每阶段可以独立提交，P2 不依赖完整 tensor fusion 迁移。

第一阶段不扩充 `Def::Node`，不实现独立 PDLL 风格 parser，不接入 e graph，不做隐式交换律/结合律搜索，不进行通用高阶 lambda 匹配。后续 Mim 接口可复用 `rule`，增加 as capture、graph relations 或 pattern library；`refly.Code` 当前的 opaque handle 不能直接视为已具备这些接口。

## 验证与验收

### 正确性测试

| 范围 | 必须覆盖的情形 |
| --- | --- |
| capture 与身份 | `f(x,x)` 成功及不一致失败；不同 capture 可同值；整节点 as capture |
| 回滚与组合 | OR 左分支绑定后失败，右分支不受污染；AND 绑定冲突；NOT 不导出绑定，Unknown 不变成成功 |
| 图边 | 共享 Tuple 参数、同一 consumer 重复槽位、Extract 路径、符号 Pack、返回值逃逸、类型边不计数 |
| 图范围 | 同一 Def 被多个 binder 使用，局部替换不污染另一 scope；opaque 用户导致排他关系未知 |
| 依赖类型 | rank 0/1/N、单元素塌缩、uniform tuple 压缩、符号 shape；改变遍历顺序不改变可证明的结果 |
| 递归与 binder | 递归目标遍历终止；默认不进入 Lam；RHS 不捕获错误 binder，不遗留旧 World 引用 |
| 多根 | 无 tuple 共同根的三投影；早期候选导致失败后可回溯到有效组合；anchors 去重和别名约束 |
| 批量重写 | 多输出共用一个新节点；替换冲突、依赖环、类型不符拒绝；失败旧图仍有效 |
| 驱动 | 无变化不 invalidate；benefit 平局可重复；往返规则触发 fuel 并给出可定位诊断 |

对小型有界 DAG 使用穷举参考 matcher 验证多根搜索，尤其检查回溯完整性。改变候选枚举顺序后匹配集合应相同；达到预算上限的结果单独标识。

### tensor 验收

以当前 `lit/tensor/fuse_*.mim` 为基线，重点包括 `fuse_skip_noninjective.mim`、`fuse_skip_epilogue_shared.mim`、`fuse_skip_epilogue_shared_read.mim`、`fuse_strip_pack.mim` 和多输入 fusion。新增 rank 0/1/3、`nis > 2`、`nps > 1`、动态形状与新图共享关系变化的用例。

结构相同比较忽略 gid/debug 命名，保留共享关系；结构有意变化时比较可运行结果或语义参考，不能只数算子个数。达到这些标准后再逐步替换现有 Fuse 路径。历史 13/14 的实验数据不计为本方案已经通过。

### 性能验收

记录建图时间、候选数、结构检查数、回溯数、constraint 时间、重建轮次和峰值内存。基准至少包含长链、菱形共享、高 fanout、重复算子、多规则以及实际 tensor workload。

原始索引构建预期为可达节点与边的线性扫描；语义包装展开有独立预算。固定单根模式通常可被根索引显著剪枝，多根和无序匹配一般具有组合搜索成本，不承诺普遍线性。先建立解释执行基线，再依据热点决定共享 predicate DAG、字节码或增量索引的必要性。

## 首个可交付版本

当前已交付 P0 至 P3 的 C++ 基础及 P4 的首批 tensor 接入：C++ 构造模式、Rule 支持子集、带 Unknown 的图使用关系、连通多根搜索、作用域内批量替换，以及可诊断的单根 Once/Greedy 驱动。现有 tensor fusion 的结构筛选与单次消费检查已接入框架，真实三路整数矩阵乘也已有 LLVM 执行和数值参考验证；完整 DPL 表达层与性能优化属于后续工作。

最先实现并评审的是 **GraphView 的 scope 与边界语义、失败回滚、依赖类型延迟校验和 World 提交协议**。这四项确定后，Mim 表层语法和匹配执行优化可以分别推进。
