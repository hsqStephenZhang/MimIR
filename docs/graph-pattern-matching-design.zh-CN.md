# MimIR 图模式匹配实现方案

建议采用 **C++ 图匹配内核、独立的 Pattern 表示、现有 Rule 适配器、基于 World 重建的批量重写**。借鉴 TVM DPL 的表达式模式和图关系约束，以及 MLIR PDLL、PDL、PDLInterp 将规则描述、匹配执行和重写驱动分层的做法。先稳定匹配与提交语义，再扩展 Mim 语言接口。

目标包括单根 DAG 匹配、多根图匹配、带类型和分析条件的规则，以及保留共享的多节点替换。首版面向纯数据流子图；tensor fusion 是首个实际使用场景。内存、控制流和递归函数体的通用重写需要后续独立定义合法性。

文中的 MLIR 参考对象是 **PDLL**。除下面的实施状态外，后文 API、文件布局和阶段均为目标设计。

## 当前实施状态

已加入 P0/P1 内核、P2 单根 Rule 重写通路、P3 的 C++ 语义图与多根重写、P4 的首批 tensor 接入，以及 P5 的多根 Rule 前端、原始 operand range 和索引搜索。单根入口位于 [pattern.h](../include/mim/pattern/pattern.h)、[match.h](../include/mim/pattern/match.h)、[pattern/rule.h](../include/mim/pattern/rule.h) 和 [phase/rule_rewrite.h](../include/mim/phase/rule_rewrite.h)；P3 至 P5 的接口和当前范围见下节。

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

P4 验证基线为 138 个 CTest、91 个相关 lit 和 18 次 Mim／AST roundtrip。P5 新增的表达、索引对照和 review 复现用例见 [pattern_p5.cpp](../test/pattern_p5.cpp)，最新验证结果记录在下方 P5 节。验证使用未加载旧版外接 runtime/torch 插件的隔离构建；根目录现有 build 的配置仍受这些插件的旧 CMake 参数阻塞。

尚未完成的部分包括依赖类型实例化求值、开放 closure scope 的自动环境导入、通用 tensor dtype／shape 模式库、独立 PDLL 风格 parser、字节码后端和跨轮次增量索引。`compile.rule_app` 仍是单根入口，`compile.graph_rule_app` 是显式多根 Rule 入口，真实 tensor 多根优化还可调用 `tensor.combine_products`。Rule adapter 不做逆归一化，也不猜测缺少绑定的 telescope；原始 Pack 边数不代表动态读取次数。

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

mutable 节点在检查 node／flags 后即视为结构不透明，不能用操作数数量证明不匹配。Indexed 的反向路径索引在每个缺失操作数层保留 mutable／Lam／Pi 边界以及其已知祖先；这些省略的边不能证明 capture 不可达，最终仍由结构匹配器决定 Unknown 或拒绝。边界扫描与候选扩展计入原有工作预算。

`GraphMatches::status` 区分完整枚举 `Complete`、存在未知条件 `Unknown`、预算耗尽 `Budget_Exceeded` 和达到结果上限 `Result_Limit`。后面三种状态不代表“完整搜索后无匹配”；其中已返回的每个匹配仍通过了自身约束。步数、候选数、结果数、递归深度及包装展开都有边界。

### 替换与提交

算子纯性默认未证明。注册可以给出无条件 pure，也可以提供基于完整 curry 参数组的只读纯性检查；源节点和 recipe 的目标参数都必须通过检查，后者在调用 normalizer 前执行。一个 recipe 可以被多个输出引用，求值只执行一次。替换的类型必须保持，导入值必须来自当前图，计划之间不能有重叠写入或读写冲突。

显式 `num_groups` 注册要求 recipe 参数组数量完全相等；它不是 Axm 的 normalization curry counter。每次继续应用参数前，已归一化的 callee 必须仍是该 recipe 已验证的原始应用前缀。即使自定义 decoder 没有显式组数，normalizer 返回另一个函数也不能借原 axiom 的 pure 声明继续调用它。函数值本身可作为结果，不代表调用它也纯。

所有未证明纯性的 App 都必须在重建结果中保留，包括返回 Pi 的计算和独立的部分应用。纯算子的完整已解码应用可以证明自身的 curry callee spine，但不能证明作为参数传入的函数生产计算。Lam／Var／binder 叶子不是这样的应用计算；开放函数和 continuation binder 的映射仍被保留。

导入的 immutable 子图也参与替换依赖检查，因此 `a → import(b)`、`b → import(a)`，以及通过更深子表达式形成的循环都会在构建目标图前被拒绝。带排他使用条件或 native Where 的匹配保守地记录更宽的读取范围，避免批量提交使分析条件失效。

有 scope 时，目前要求一个已定义、闭合的 mutable Lam。重建为它建立新 stub 和 Var 映射，并为其他函数使用独立 importer：同一个 hash-consed Def 在其他函数内保留原实现。所有原 external roots 都被保留。无 scope 时只重建显式的闭合 roots，结果可供调用方继续处理，不允许将其直接发布为整个程序。

`publish` 前检查源图的存储代次、mutable 状态和 external 集合。中间运行 cleanup、修改其他 external 或增加 external 都会使旧结果无法提交。World 的存储 token 随 IR 一起交换，token 的持有不保留旧 Def arena，也不会因内存地址复用而使旧句柄复活。含 Exact/Literal 引用的 PatternProgram 和 RulePattern 也会拒绝在 World 重建后继续使用。

原子提交指校验全部通过后交换 World；调用方仍须独占 World 修改。提交后旧 Def 指针和匹配句柄失效，后续轮次应重新建立视图及引用当前 World 的模式。

`GraphRewrite::is_changed()` 在同一目标 World 中比较 canonical 结果与不应用替换的导入结果，复用 binder 与不透明节点的映射，不对源 World 的 mutable 节点执行 α 检查。scope body 的两次构建都在共同的新 Lam stub 尚未填充时完成，避免递归自应用只在一侧展开。GraphRuleRewrite 丢弃归一化后相同的结果，继续后面的 match／Rule，不 publish 或 invalidate。该比较识别 canonical 图相同，不提供任意算术等价或通用高阶函数等价证明；显式调用 publish 的其他 driver 仍自行决定提交策略。

### TVM DPL 机制对 MimIR 局部重写的启示

本节根据本地 TVM Relax 源码整理优化方向；下面的 P0–P3 是**未实现提案**，不改变当前 `rewrite_graph` 的整 World 提交语义。

**MimIR 当前成本。** 搜索和匹配只读 `GraphView`。每次 `rewrite_graph` 候选都会调用 `World::inherit()` 创建独立目标 World，再由 `GraphRebuilder` 导入并重建选中 scope body 或 roots；有 scope 时，为保留程序还会通过 importer 导入其他 external 函数。`inherit()` 继承的是 World `State`，Def arena/hash-cons 存储并未与源 World 共享。`GraphRewrite::publish()` 交换整个 World 并使旧快照失效。因此 scope 限定的是哪些节点接受替换，不代表只分配或提交这些节点。`GraphRuleRewrite` 对每个候选调用 `rewrite_graph`；`is_changed() == false` 能跳过 publish/invalidate，却是在目标图已经构造之后才判断，不能省掉该候选的导入和构建成本。依据：[World::inherit](../include/mim/world.h)、[World storage](../include/mim/world.h)、[rewrite_graph/GraphRebuilder](../src/mim/pattern/rewrite.cpp)、[GraphRewrite::publish](../src/mim/pattern/rewrite.cpp)、[GraphRuleRewrite::start](../src/mim/phase/graph_rule_rewrite.cpp)。

**DPL 实际可借鉴什么。** Relax DPL 的 `PatternMatchingMutator` 对 `SeqExpr` 和 binding block 做局部遍历。在 `dataflow_expr_rewriter.cc` 的 block rewrite 中，整个 `DataflowBlock` 可作为候选区域；普通 `BindingBlock` 则按允许重排的 dataflow bindings 分段，在 `If` 或 impure call 等边界先处理已收集区段。仅当 bindings 实际变化时才重建 block；有新增依赖时再拓扑排序。外层仅在 block/表达式变化时返回新对象。应用到 `IRModule` 时先收集变化的函数，只有存在更新才对 module 执行 `CopyOnWrite()` 并 `Update()`。这些机制来自 [PatternMatchingMutator](</workspaces/ml-compiler/tvm/src/relax/ir/dataflow_expr_rewriter.cc>)、[block rewriter](</workspaces/ml-compiler/tvm/src/relax/ir/dataflow_block_rewriter.cc>)、[DPL Python API](</workspaces/ml-compiler/tvm/python/tvm/relax/dpl/rewrite.py>) 和 [IRModule COW/update](</workspaces/ml-compiler/tvm/src/ir/module.cc>)。

DPL 适合借鉴的原则是：以可安全重排的 block/region 限定重写范围；保持未变表达式、block、function 的原对象引用；只有模块成员变化时才构建 module 更新。这利用 TVM 不可变、引用计数对象与 `same_as` 判等做结构共享。DPL 仍要遍历被处理的函数/绑定以寻找匹配，也没有提供跨轮次增量匹配索引或事务式 World merge。MimIR 的 Def 存于各自 World 的 arena/hash-cons 表中，mutable binder、external roots、gid 和快照 token 又受 World 生命周期约束；不能把 TVM 的 `CopyOnWrite` 直接等同于 MimIR World 的局部提交。DPL 的 `BlockBuilder::Normalize` 也不能直接套到 MimIR：MimIR 的 normalizer/callback 及 axiom 纯性、binder 和 use 覆盖都必须纳入安全边界。

| 阶段 | 提案与安全不变量 | 收益、成本及验收 |
| --- | --- | --- |
| P0：测量 | 为 `rewrite_graph` 和 GraphRule 候选记录 target World 创建数、导入 Def 数、重建 Def 数、外部函数数、构建时间、publish 时间、no-op 数；加大程序中仅改一个小 scope 的基准。 | 先量化成本归属，不改变语义；固定 IR/规则/编译器配置，报告中位数、峰值内存和规模增长曲线。 |
| P1：降低重复工作 | 缓存一次 `GraphView`/规则批次中可安全共享的只读分析和 importer 映射；用 recipe 结构、已有 RHS 和类型做保守的 cheap no-op preflight。只有能证明相同才跳过构建，未知时仍走完整验证。 | 改动较小，收益取决于候选数和共享导入量。不得跨 World/snapshot 复用裸 Def；不得因为结构相似跳过 purity、type、cycle 或 scope 检查。用多候选同一函数、no-op 在前/productive 在后验证 fallback，并测导入节点数。 |
| P2：局部变更集与原子提交 | 让临时 World 产出经验证的变更集，再将新 Def 增量合入源 World，最终只更新选中 Lam body；或为 World arena、hash-cons 项、gid、externals/use 索引提供可回滚 transaction。提交前不得暴露新 body，失败必须回滚所有插入和 mutable 变化。 | 目标是将成本从全程序可达图导入降到变更子图及其必要依赖；实现复杂度高。需证明新 Def 最终均归属源 World、hash-cons 唯一性不破坏、递归 stub/Var 映射一致、类型和 normalizer 执行语义不变、externals/use 覆盖正确、快照只在成功提交后失效。基准比较单 scope 小改写在增长的未变函数数下，时间/内存应主要随变更区域而非整程序增长；另测失败注入后的 World 完全不变。 |
| P3：持久化/COW World（远期） | 若 P2 仍显示瓶颈，再考虑共享不可变 Def 存储、分层 hash-cons arena 和版本化 World 根。 | 可能减少跨版本存储复制，但需要重定义 Def 所有权/回收、mutable 节点隔离、ID 与快照规则，风险和迁移面最大；在 P0/P2 数据证明必要前不建议启动。需长期版本并存、回收和跨 World 引用压力测试。 |

建议借 DPL 的局部性和 unchanged identity reuse 原则，**不照搬它的对象模型**。先做 P0/P1；只有数据证明 World 重建是主成本且有明确 transaction/merge 设计后，再进入 P2。以上路线尚未实现，当前代码仍按本节开头描述的独立目标 World 构建和整体 publish 工作。

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

目前每次调用至多重写每个闭合 external Lam 中的一组三路匹配；需要多轮时可放入 `compile.phases tt`，每次成功重写都减少两个 product 节点。P5 将该 phase 明确为首个合法匹配选择：设置 `max_results=1`，接受这一主动结果上限；返回的匹配仍须通过全部条件。Unknown 或 Budget_Exceeded 仍跳过并记录诊断。24 个共享左输入的乘法有 2024 个可能三元组，现在不必完整枚举后才能提交一组；对应回归覆盖了原来超过 1000 结果上限就整组跳过的情况。其他函数共享的旧 Def 保留原实现。尚未加入后端收益模型，因此此 phase 不放入默认 pipeline；额外 concat 和 slice 的成本可能抵消合并收益。

这完成了 P4 的首批真实 tensor 接入与正确性验收，尚不等于完整的 TVM DPL：缺少通用 dtype／shape helper、浮点 QKV 合并、任意数量分支的合并、通用融合函数提取和后端分发。现有 `fuse_tensor` 继续使用原生 CPS recipe，未将其改成一套通用 native rewrite 回调注册系统。

## P5 表达与索引搜索

### 复用 Mim Rule 的多根前端

P5 复用现有 `rule`，以显式的 `compile.graph_rule_app (rules, pure_ops)` 区分多根重写。Rule 的 LHS tuple 表示多个独立 anchor，RHS 表示对应输出；共享 metavariable 连接这些 anchor，目标程序不必有共同 tuple。实现是 [graph_rule.h](../include/mim/pattern/graph_rule.h)、[graph_rule.cpp](../src/mim/pattern/graph_rule.cpp) 和 [graph_rule_rewrite.cpp](../src/mim/phase/graph_rule_rewrite.cpp)。

```mim
plugin compile;
axm q: Nat -> Nat;
axm k: Nat -> Nat;
axm v: Nat -> Nat;
axm combined: Nat -> «3; Nat»;
axm sum: [Nat, Nat] -> Nat;

rule merge (x: Nat): (q x, k x, v x) => combined x;
extern lam result (x: Nat): Nat = sum (q x, sum (k x, v x));
extern lam _compile (): compile.Phase =
    compile.phases tt (
        compile.cleanup,
        compile.graph_rule_app (merge, (q, k, v, combined)),
        compile.cleanup,
    );
```

RHS 的三个结果若已通过 η 归约合并为整个 `combined x`，前端会在 recipe 中投影三个输出，不在源 World 补建 Extract。C++ `GraphRulePattern` 与 Mim phase 均复用同一 `PatternProgram`、共享绑定、多根搜索和事务式 recipe。规则模板保存于独立 World，跨 cleanup 和 PhaseMan 重建后重新导入；每次调用至多提交每个闭合 external 函数中的一组匹配，规则按显式列表顺序尝试。

`pure_ops` 是调用方对 primitive 纯性的显式断言，不由函数类型推断。列表应包含被替换的 primitive 与 RHS 新调用的 primitive；callee 为任意 Lam 的 RHS 尚不支持。前端不猜测数据参数 schema，排他使用不能因列入该列表而自动成立。RHS 支持 captures、Nat／Idx 位字面量、pure axiom call、Tuple、可静态展开的 Pack 和固定 index 的 Extract。未绑定的 metavariable、未知 guard、预算耗尽或不支持的 RHS 不能发布替换；会消除未证明纯性的内部计算的计划也会被拒绝。whole-telescope capture 不会自动导出缺少的字段 capture；tuple η 归约消除字段结构后，仍可能缺少可表示的跨根连接。这属于当前 Rule 支持子集，不进行逆归一化补全。

正反例见 [graphapp.mim](../lit/rules/graphapp.mim)、[graphgrd.mim](../lit/rules/graphgrd.mim) 和 [grapure.mim](../lit/error/grapure.mim)。匹配结果的共享应检查实际 Def 身份；文本打印器可能在不同 Extract 内重复打印同一个调用，不能仅数 `combined` 字符串来判断是否复制计算。

### 原始 operand range

`PatternProgram::node_range(node, prefix, tail, min, max, flags)` 匹配固定 operand 前缀和指定数量的尾部 operands。它针对物理节点，不隐式解包 tensor-valued Pack，也不恢复被归一化消除的结构。尾部复用同一 capture 仍要求 Def 身份相同，并不捕获一个动态序列。例如：

```cpp
auto operands = p.node_range(mim::Node::Tuple, {p.exact(first)}, p.any(), 1, 8);
```

### 有界索引 join 与穷举对照

`GraphMatchOptions::mode` 默认为 `Indexed`，也可选择 `Exhaustive`。两条路径使用同一个回溯解释器；Exhaustive 关闭结构判定缓存与动态 capture 索引，用于结果对照和性能基线。

Indexed 缓存不依赖当前绑定的结构判定，并预先计算每个 anchor 必须绑定的原始 captures。AND 合并必需 captures，OR 仅保留两边共同保证的 captures；Type／NOT 中的 capture 不用于推断原始 operand 祖先。对路径固定的必需 capture，从已绑定值沿对应的反向 operand 路径寻找候选；路径不能统一时才使用保守的祖先集合。候选按原枚举顺序排列，并选择当前候选最少的未绑定 anchor。约束依赖和路径分析计入 step 预算。

同一 anchor tuple 找到合法绑定后停止该 tuple 内无意义的 OR 备选，同时保留其他 tuples 所需的跨根回溯。所有 native Where 的读取范围都包含在冲突分析中，包括 Not 内部的 Where。snapshot-wide 读取列表在结果之间共享；普通结果只收集自身读取子图，避免每个结果再次扫描整个图。

`GraphMatches` 除 `steps`、`candidates` 外，新增 `prefilter_checks` 和 `index_steps`。这些是计数器，不等同于总 CPU 时间；原生分析、归一化与建图仍有各自成本。基准程序 [pattern_bench.cpp](../test/pattern_bench.cpp) 单独构建：

```sh
cmake --build <build> --target mim-pattern-bench
<build>/bin/mim-pattern-bench 160
```

输出包含 parallel、chain、diamond、spine、fanout 五类图的匹配数、状态、候选数、steps、结构检查数、索引工作、建图时间和搜索时间。耗时不用于单元测试断言；单元测试比较解集合与候选工作量。跨轮次增量缓存、共享 predicate 字节码、完整真实 workload 性能验收和峰值内存测量仍未实现。

本机测量使用 Intel Core i5-14600KF、GCC 15.2、Debug／`-Og` 构建，每个尺寸运行 5 次，取计时中位数；源图建图时间单列，不计入下表搜索时间。可通过 [bench-patterns.py](../scripts/bench-patterns.py) 重复采样，完整结果在 [graph-pattern-benchmark.csv](graph-pattern-benchmark.csv)：

```sh
python3 scripts/bench-patterns.py <build>/bin/mim-pattern-bench --sizes 80,160,320 --repeats 5
```

| 图形与尺寸 | 穷举候选 | 索引候选 | 穷举搜索 ms | 索引搜索 ms |
| --- | ---: | ---: | ---: | ---: |
| parallel 160 | 51360 | 480 | 6.612 | 1.129 |
| chain 160 | 160 | 160 | 0.679 | 0.578 |
| diamond 160 | 76960 | 480 | 7.608 | 0.812 |
| spine 160 | 51360 | 480 | 5.159 | 1.042 |
| fanout 160 | 51360 | 480 | 5.916 | 1.036 |

这组结果支持共享 capture join 的候选剪枝收益，不代表 tensor 程序的端到端加速。chain 的候选数不变，索引初始化和缓存是否获益取决于具体模式与图；不承诺所有单根查询都更快。

### Copilot review 修复

2026-10-31 使用 Copilot CLI 1.0.103 对 `edc772d080` 的 P0 至 P4 实现进行只读 review，运行日志确认模型为管理员配置的 `gpt-6.1-sol`。review 提出 6 项 P1、2 项 P2；复现与修复测试集中在 [pattern_p5.cpp](../test/pattern_p5.cpp)。

| 问题 | 修复后的行为 |
| --- | --- |
| 结果与集合访问器暴露已释放的 Def | 单根结果、GraphView／GraphMatch 集合，以及 GraphRewrite 的结果访问均验证存储代次；已取得的 span 仍须由调用方遵守生命周期 |
| Not 内 Where 未记录分析读取 | 捕获导出分析与读取分析分离；读取分析穿透 Not，批量冲突检查可见这些依赖 |
| 包装预算只统计成功输出 | 每次包装展开访问消耗工作预算，零输出的共享包装 DAG 也会有界终止 |
| 独立 importer 绕过重建预算 | GraphImporter 与 GraphRebuilder 共享深度与 source work 预算，保留其他 external 时也受限 |
| 纯外层替换间接消除不纯计算 | 验证未证明纯性的内部计算仍在所选结果中可达，归一化丢掉这些输入也不能放行 |
| 聚合类型的非法位字面量 | literal recipe 只允许 Nat／Idx scalar；聚合常量通过 Tuple／Pack 等显式构造 |
| 同时替换误报依赖环 | 遇到已替换节点后停止扫描其旧操作数，由其新 recipe 决定后续依赖 |
| OR 成功分支污染 Unknown 状态 | 分离成功 tuple 与继续枚举的控制；按完整 anchor tuple 记录未决状态，同一 tuple 后续成功时清除，不掩盖其他 tuple 或部分赋值的未知条件 |

2026-11-01 对 `860841a059` 的后续静态 review 提出四项问题。本次按源代码语义补充下列回归；遵照执行限制未运行构建或测试，运行复现和修复验证交由父会话。

| 后续问题 | 修复与新增回归 |
| --- | --- |
| 返回函数的 App 绕过纯性可达性检查 | [graph_pattern.cpp](../test/graph_pattern.cpp) 的 `Review purity: function-valued computations` 覆盖 `sink (make 0)`、部分应用、normalizer 丢弃输入、显式 pure 正例；另有函数 binder 正例和既有 continuation 回归 |
| overapplied recipe 借用错误的纯性声明 | 同文件的 `Review purity: overapplication` 检查组数拒绝，以及 normalizer 已执行但未证明函数调用次数仍为零；`registered group count` 覆盖合法多组应用与不同的 Axm curry counter |
| 归一化 no-op 与递归 stub | [pattern_pdll.cpp](../test/pattern_pdll.cpp) 用 add-zero normalizer 检查 canonical no-op、源图不变；[graph_pattern.cpp](../test/graph_pattern.cpp) 另检查递归 stub 的 no-op。Rule fallback 与无有效替换不发布继续由既有 P5 回归覆盖 |
| Indexed 将 mutable 不透明路径当作不存在 | [pattern_p5.cpp](../test/pattern_p5.cpp) 的 `Review join: skipped mutable operands` 对 `q(x)` 与 `Lam(any(), x)` 在 Indexed／Exhaustive、正序／逆序中都要求 Unknown 且零结果；`Review opacity` 覆盖 arity 先于 opacity 的回归 |

### TVM DPL 示例回归

使用 Copilot CLI 的 `gpt-6.1-sol`，参考本地 TVM 的 [test_dataflow_pattern.py](/workspaces/ml-compiler/tvm/tests/python/relax/test_dataflow_pattern.py) 新增 [pattern_dpl.cpp](../test/pattern_dpl.cpp) 中的 18 个用例。它们将 TVM 示例的行为要求改写为 MimIR 断言，不依赖 Python 或 TVM 运行时：

| TVM 示例／行为 | MimIR 回归断言 |
| --- | --- |
| diamond、parallel、nested diamond | 精确 anchor 组合、共享 Def 身份、重复参数位置、排他使用及输出逃逸的正反例 |
| fake QKV、combine matmul twice | 拒绝输入不同的假三分支；两组独立替换分别共享各自的新计算 |
| commutative、repeated pattern | 显式 OR 的不同 operand 路径、子模式身份、失败回滚、连续匹配不残留 captures |
| variadic wildcard | 空尾部、重复尾部 capture、前缀 OR 回溯；保留 Tuple 与 Pack 的物理结构区别 |
| no-op rewrite fallback | 实际 GraphRuleRewrite 跳过 identity 与 false guard，选择首个有效规则；无有效规则时不发布 |
| MimIR 图表示与三值判定 | type-only／raw-or-type captures、嵌套投影与包装路径、Unknown 分支回滚及跨 tuple 状态汇总 |

图匹配断言同时运行 Indexed／Exhaustive 与正序／逆序候选，比较完整解集合和 captures。新用例复现了跨根 OR 的未决状态残留：先前绑定分支得到 Unknown，后续绑定分支对同一完整 tuple 成功后，搜索仍错误返回 Unknown；现已修复，并保留另一个 tuple 未决时仍返回 Unknown 的反例。

这些测试遵守现有支持边界：交换律通过显式 `either` 表达，range 不捕获动态序列，嵌套投影使用 Sigma 边界保留标量 Extract 链；尚未覆盖融合后的多维 Extract index 解码。未将自动重复重写、tensor shape／dtype 推导或任意 Python callback 作为已实现能力。

后续核对了 TVM `test_dataflow_pattern.py` 的全部 58 个顶层测试函数，并在 [pattern_dpl_extended.cpp](../test/pattern_dpl_extended.cpp) 新增 26 个用例，覆盖显式无序 tuple、声明的静态／符号 shape、shape 否定、双 CBR、concat–matmul–split、attention 图、独立匹配程序的约束与两轮 checked rewrite。逐项结果为 31 项语义适配覆盖、22 项部分覆盖、5 项当前不支持；完整清单与剩余差异见[参考用例覆盖](graph-pattern-reference-tests.zh-CN.md)。该分类不表示 Python API 兼容，也不把手动执行两轮重写计作自动收敛。

### MLIR PDLL 与 PDL 示例回归

继续使用 Copilot CLI 的 `gpt-6.1-sol`，参考本地 `mlir/test/mlir-pdll` 与 `TestPDLL.pdll` 的声明、native constraint、结果访问和 replace 行为，在 [pattern_pdll.cpp](../test/pattern_pdll.cpp) 新增 17 个用例。PDL bytecode、PDLToPDLInterp 和 CPU 多根集成测试作为执行语义对照，来源与 PDLL 分列于[参考用例覆盖](graph-pattern-reference-tests.zh-CN.md#mlir-pdll-与-pdl)。

测试覆盖 Nat／Bool 异构结果的索引、共享 producer、反向注册替换时的结果位置、原输入透传、组合与否定的只读 native constraints、部分字段类型错误和非法 index 的原子拒绝，以及实际 GraphRuleRewrite 发布。两个实际 Mim 前端输入 [pdllmix.mim](../lit/rules/pdllmix.mim) 和 [pdllpass.mim](../lit/rules/pdllpass.mim) 分别验证异构多根合并和 guard 控制的透传；不同输入、false guard 与未知 guard 保留原计算。

这批用例不实现 PDLL parser／诊断、ODS、native constraint 返回新绑定、动态 ValueRange／TypeRange、benefit 调度、任意 native rewrite callback、显式零结果 erase 或结果类型推导。不可达的已证明纯性的 DAG 节点可以在重建后消失，不等同于任意 MLIR operation 的 erase。

`860841a059` 的修复前验证记录为 215 个 CTest、96 个相关 lit 测试和 26 次 Mim／AST roundtrip 通过；另有 1 个 lit 按原有 release 条件跳过。144 个 Pattern／Rule／Graph 用例与四个 Mim 多根 Rule 前端用例通过 AddressSanitizer／UndefinedBehaviorSanitizer；插桩范围为当时的匹配、图重建与多根 Rule phase，基础 IR 库和其他插件未全部插桩，未运行 LeakSanitizer。该记录不代表上述四项后续修复已执行验证。数值回归继续包含 I64 矩阵乘的 LLVM 执行与标量参考。

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

当前已交付 P0 至 P4 的基础与首批 tensor 接入，以及 P5 的多根 Mim Rule 前端、原始 operand range、索引 join 和穷举性能对照。现有 tensor fusion 的结构筛选与单次消费检查已接入框架，真实三路整数矩阵乘也已有 LLVM 执行和数值参考验证；独立 pattern parser、通用 tensor 模式库、字节码和增量索引仍属于后续工作。

最先实现并评审的是 **GraphView 的 scope 与边界语义、失败回滚、依赖类型延迟校验和 World 提交协议**。这四项确定后，Mim 表层语法和匹配执行优化可以分别推进。
