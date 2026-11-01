# P5 tile-and-fuse 实验

## Task 3：独立 frontend 回归与 annex 引用

以下更新优先于后面的 Task 2 历史结果。
每个 `tf_*.mim` 现在只运行 rule frontend，对应 `tf_*_g.mim` 只运行 graph frontend。
可工作的路径使用目标 CHECK；尚未工作的路径使用 unchanged IR 和精确诊断的正向 GAP 检查，
不再使用 XFAIL，避免 graph 的已知缺口掩盖 rule_app 回归。
`tf_share_g` 分别检查 flat 多根成功和 nested 尚未重写，rule 文件独立保护 NEST 成功。

`pure_ops` 新接受 qualified annex-name strings，与原有低阶 Axm 值可以混合。
例如 `("tensor.slice", "tensor.unary")` 在构造 phase 时解析为源 World 中实际的 Axm，
再与已有机制一样导入模板 World，重新运行时导入当前 World。
这选择了 annex references 路线，没有引入不存在的 Level 参数多态或改变类型检查器。
没有找到 annex、或引用非 Axm 时显式报错；没有自动推断 purity 或把未知 op 当作纯。
`tf_poly` 已翻转为正常通过；真实 graph probes 的配置均通过，剩下 LHS/guard/recipe 缺口
由各自 GAP 检查独立记录。`tf_run.py` 将真实 qualified op 转为这种显式引用。

实验 4 的 observed IR：rule_app 输出把 `stage _5` 打印在 io body 的 mm 参数中，
尽管 stage 只依赖 jo/ko。
这不是 rule_app 的错误；lexical let 不是 placement 指令，模型尚无 stage lowering 合同。

### Task 3：真实 LHS 的最小修复

临时在 MatchEngine 的 Exact 失败处记录 expected/actual 后，四个真实 probe 的首个阻挡
均是模板导入克隆后的 mutable Pi type 与源算子 type 的地址不同。
slice 在 `{T,r}` 部分应用之后的 dependent Pi、unary 在 f 部分应用之后的 implicit shape Pi、
product_2d 在 Ring 部分应用之后的 shape Pi 都出现此现象；不是 Pack metadata 或参数类型推断。
把 axiom 按 flags 匹配的试验只把失败推迟到下一层 partial-App Pi，不能单独解决问题。
最终没有保留临时日志或这个 axiom-only shortcut。

Rule adapter 现在仅对**闭合 mutable type**增加只读 `Check::test_alpha` 备选，
保留 exact 快路径、source World 跟踪、匹配预算、metavariable 的 deferred type 检查和最终
replacement type check。开放 dependent type、Hole 推断和 term-level lambda alpha matching
没有增加。这个修改不需要复制 tensor schema adapter，因为失败是共用 Rule 的 type identity。

`tf_rmap` 的 flat、共享 whole reader 和 tiled body 内 pushdown 都已通过 rule_app；
相应 `_g` 的 flat 已通过，但 nested 仍以 unchanged IR 检查 region 缺口。
`tf_rrev` 两个 frontend 都已生成 together；`tf_rmm` 的 rule_app 已生成真实 product/slice nest，
graph 仍正向检查 `unsupported graph Rule recipe value`。
`tf_rfact_g` 已到 `guard is not true`，不再是 LHS 不匹配。
新增 length-16 negative case 防止 length-8 rule 误匹配；C++ 测试还检查导入 dependent Pi 的
匹配不会构造源 IR，并拒绝不同 axiom。

## 范围与复现

基线为 `cd8526dd22` 加 Task 1 的两个修复和 `031b1d7626`。
本轮只修复拒绝绑定后的回溯；没有增加 region、lambda recipe、图 guard 或调度 lowering。
实验在 Debug 构建运行，使用 `compile.rule_app` 和 `compile.graph_rule_app`。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools cmake --build build \
    --target mim mim_all_plugins mim-test mim_lit_tests mim-regex-test mim-tensor-pattern-test
ctest --test-dir build --output-on-failure
cd lit
/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
    ./lit ../build/lit -a --filter 'rules/tf_'
```

`lit/rules/tf_run.py` 为同一个输入分别追加两个 frontend 的 `_compile`，运行一次
`phases ff` 和 cleanup，并将 IR 与 debug 日志分开。
生成文件使用短 module 名，退出时清理；不是新的编译器机制。
`TF-RULES` 和 `TF-PURE` 是该测试驱动的配置，不是 Mim 语法。
可单独复现某个 frontend，例如：

```sh
python3 lit/rules/tf_run.py build/bin/mim lit/rules/tf_build.mim build/tf_probe rule
python3 lit/rules/tf_run.py build/bin/mim lit/rules/tf_build.mim build/tf_probe graph
```

每个有缺口的测试保留正向的目标 IR 检查并标记 `XFAIL: *`。
因此“XFAIL”不是语法报错就算完成：模型的 single-root 成功检查在 graph 缺口前运行；
真实多态算子的已知类型错误另以 `not` 和诊断文本检查，再检查尚未得到的目标 IR。
XFAIL 文件不会继续执行第一个失败的 RUN，下面的分项结论也依据独立的 frontend 输出。

这些是**表达能力与 IR 结构实验，不是数值、bufferization 或性能实验**。
模型的 `Nat` 是不透明 tensor handle，不是 tensor 元素值。
`slice`、`panel`、`tiled`、`fold`、`stage` 等的含义由文件中的短注释约定；
没有 normalizer/lowering 来执行这些约定。
尤其 `tiled_n` 返回两个值不等于已经实现同一个 buffer；
`stage_persistent` 被列为 pure，只代表一个可引用的资源值，不证明分配、初始化或跨调用缓存纯。

最终验证：CTest **228/228**；`rules/tf_` **4 通过、10 XFAIL**；
全部 `rules|error|tensor` **181 通过、10 XFAIL、4 unsupported**，无意外失败或 XPASS。
unsupported 是两个 error fixture、Linux-only `free_temps` 和 Release-only `vgg16`。

## 逐项结果

### 1. Local compute_at

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `slice(ew(f,x),o,e) → ew(f,slice(x,o,e))`，位于捕获 `x/f` 的 tiled body | rule_app | **通过**；`tf_local.mim` 的 LOCAL 检查看到 body 内先 `slice(x,o,2)`，再 `ew(f,…)` |
| 同一规则、同一输入 | graph_rule_app | **XFAIL**；编译成功但 `Complete (no accepted match visited)`，body 中仍是 `slice(ew(…),o,2)` |
| 实际 `tensor.slice` / `tensor.unary`，含 offset 参数 | rule_app | **XFAIL**；`tf_rmap.mim` 中仍先 unary 再 slice，没有目标 pushdown，也没有 Unknown-plan 日志 |
| 同一真实算子 | graph_rule_app | **XFAIL**；配置在进入 phase 前报 `argument is not assignable to callee's domain` |

第一项推翻了“两个 frontend 都不能进入开放 Lam body”的预期。
`RuleRewrite::rewrite_imm` 使用 enclosing Rewriter，绑定已有变量后替换
（`src/mim/phase/rule_rewrite.cpp:60–82`；
`src/mim/pattern/rule.cpp:134–162`）。
GraphView 却在 Lam/Pi/mutable 处停止原始 operand 遍历
（`src/mim/pattern/graph.cpp:46–55,65–79`），phase 只选择闭合 external Lam
（`src/mim/phase/graph_rule_rewrite.cpp:65–96`）。
这里只缺 graph frontend 的 region 查询与发布，不应重写已经可用的 single-root 路径。

### 2. Producer 同时被 whole reader 使用

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| local 规则只重算 slice，保留 `whole(ew(f,x))` | rule_app / graph_rule_app | **通过**；`tf_share.mim` 的 VALUE 检查分别保存 tile 与 whole producer |
| 同样的 value-correct 形式，tile 在嵌套 Lam 内 | rule_app | **通过**；NEST 检查看到 tiled body 的 pushdown，whole reader 仍存在 |
| 同一嵌套形式 | graph_rule_app | **未重写**；与实验 1 相同的 region 缺口 |
| `(tiles(p),whole(p)) → tiled_n(p)`，结果恰好是共同 tuple | rule_app / graph_rule_app | **通过** |
| 两个 anchor 位于不同深度的 sink 参数，没有共同的 LHS tuple | graph_rule_app | **通过**；JOIN 检查 `multi_nested` 中出现 `tiled_n`，原来的两个 reader 消失 |
| 上述无共同 tuple 形式 | rule_app | **未重写**；single-root 入口不会把两个独立节点组成匹配根 |
| 实际 unary producer 的 slice 与 whole 输出 | 两者 | `tf_rmap.mim` 的 `shared` 仍受实验 1 的真实算子匹配 / pure_ops 类型缺口阻挡 |

图规则能表达“共享一个新逻辑值”，但尚不能表达 cover witness、realization cache 或
`tiled_n` 的目的 buffer、初始化顺序与寿命。
VALUE 的 whole producer 被保留，不代表 tiles 与 whole 不重复计算。
本实验也没有声称任意 map、非纯 producer 或 reduction 都能使用这个等式。

### 3. reverse_compute_at

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `(tiled(e,body),consume(tiled(e,body))) → tiled_n(e,body)` | rule_app | **通过**；本输入的 sink 参数本来包含共同 tuple |
| 同一规则 | graph_rule_app | **通过**；`tf_rev.mim` 的 REV 检查看到新 pair，consume 消失 |
| 实际两个 `tensor.unary` 的 producer/reader，RHS 为 modeled `together` | rule_app | **XFAIL**；`tf_rrev.mim` 保留两个 unary |
| 同一真实算子 | graph_rule_app | **XFAIL**；polymorphic pure_ops 类型错误 |

`body` 是从源图捕获的函数值，不是新建 lambda；这解释了 graph recipe 为什么能接受它。
共享 capture 可以编码“C 的 operand 是 P”这一特例，不需要 guard。
它不能代替一般的 `used_by`、精确 preimage、其他输入不依赖 P、完整覆盖以及 reduction-order witness。
若 reader 跨 tile 维归约，需有 accumulator continuation 的 `tile_fold`，本实验没有模拟其合法性。

### 4. 构造时 cache_read

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `mark(mm(a,b))` 生成 jo tiled → ko fold → `bp=stage(panel(b,ko,jo,2))` → io tiled | rule_app | **通过**；`tf_build.mim` 的 BUILD 检查覆盖 panel、stage、两个 tiled 和 fold，mark 消失 |
| 同一规则，三层新 lambda | graph_rule_app | **XFAIL**；`unsupported graph Rule recipe value`，原 mark/mm 保留 |
| 实际 `product_2d`、矩阵 slice 与 modeled region/stage 的同形 RHS | rule_app | **XFAIL**；`tf_rmm.mim` 仍为 mark(product_2d)，没有 stage |
| 同一真实 RHS | graph_rule_app | **XFAIL**；pure_ops 类型错误先于 lambda recipe 验证 |

所以“single-root 不能生成带 binder 的 RHS”也不成立。
但 lexical `let bp` 并不是 placement 指令：IR printer 可以在 inner body 中内联 `stage`，
而资源应在哪个 region 实例执行仍需 lowering 保证。
模型 fold 的 0 约定为零 accumulator handle；真实实验使用正确的 2×2 A/B panel，
把 accumulator 的行 tile 加到新的 product 上，但该 fold/rows 仍无执行实现。
不能由 BUILD 通过推出 packing once per ko tile 或 equal GEMM performance。

### 5. Post-hoc cache_read

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| 匹配已有 jo/io nest，把 B tile 的 stage 放到 jo body | rule_app | **XFAIL**；`tf_post.mim` 没有 stage |
| 同一 lambda-rich LHS | graph_rule_app | **XFAIL**；独立运行得到 `Complete (no accepted match visited)` |
| 内部读取改为实际 `tensor.slice` 的同一 nest | 两者 | **XFAIL**；`tf_rpost.mim` 没有 stage，graph 配置自身可通过类型检查 |

这里不是仅仅增加 RHS lambda 就能解决。
`RulePattern::lower` 将 mutable 节点及非 Rule Var 变成 exact pattern
（`src/mim/pattern/rule.cpp:47–52`），新的 LHS lambda 不会与另一个实例做 binder 对应匹配。
需要 alpha-aware LHS binder 捕获，以及跨两个 region 的 scope/environment 映射和事务。
若提前留有外层 stage marker，可做局部单根替换；但这不等价一般的后置 box union 与 inner map rewrite。

### 6. Persistent prepack

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| 函数级 `frozen(b) → stage_persistent(b)`，两个 slice 读取此值 | rule_app / graph_rule_app | **通过**；`tf_pack.mim` |
| 同一 marker 规则，slice 换成实际 `tensor.slice` | rule_app / graph_rule_app | **通过**；`tf_rpack.mim` 的 PACK 检查绑定一个 packed 值，两次实际 slice 都读取它 |

真实 slice 不被改写，也没有被声明 pure，因此无需把其多态 axiom 放进 pure_ops。
这不是绕过 purity：slice computation 保留在重建后的图中。
实验只证明 frozen marker 能被替换且两个读取共享同一个 SSA 值。
冻结条件、权重版本、设备/布局键、缓存初始化/释放、并发调用以及跨 invocation 复用仍完全缺失。
该分支的 tensor API 没有设计文档中的 `stage_persistent` 资源 op；本文件只建模它。

### 7. Graph-fact guard

| 规则与输入 | Frontend | 结果与证据 |
| --- | --- | --- |
| `consumer(producer(x))` 的 guard 调用 modeled `only_used_by` | rule_app | **XFAIL**；`guard did not normalize to true` |
| 同一规则，分别输入 single / shared reader | graph_rule_app | **XFAIL**；两者都是 `guard is not true`，没有虚构 single-consumer 成功 |
| 实际 unary 输入的 `only_used_by` guard | rule_app | `tf_rfact.mim` 中仍为 unary，尚未到可用的 guard 实例 |
| 同一真实图 | graph_rule_app | **XFAIL**；pure_ops 类型错误先于图 guard |

guard importer 只在隔离 World 中做 term 求值
（`src/mim/pattern/graph_rule.cpp:75–91`）；这里没有 GraphView 查询入口。
C++ `GraphView::data_only_used_by` 和 GraphPattern 的关系接口已经存在；
但 phase 对 `pure_ops` 的注册没有数据 schema，并将完整性设为 false
（`src/mim/phase/graph_rule_rewrite.cpp:70–91`）。
不能把“有纯性声明”当作闭合、完整的 tensor data-use 信息。
给 only_used_by 写一个普通 normalizer 也不能正确查询随事务变化的 snapshot。

## 真实算子还揭示的两个前置缺口

1. **polymorphic pure_ops 的入口类型。**
   `compile.graph_rule_app` 的 `O` 当前是 `*`，而实际 `tensor.slice` /
   `tensor.unary` 含类型参数的函数类型需要更高 universe
   （`src/mim/plug/compile/compile.mim:136`；
   `src/mim/plug/tensor/tensor.mim:312–316,383–387`）。
   `tf_poly.mim` 把规则换成 Nat identity，只传 singleton `tensor.slice`，仍得到
   `expected [Rule Nat, ?], got [Rule Nat, {T: *, r: Nat} …]`。
   因而不只是 heterogeneous tuple 的问题，也不是 tensor matcher、RHS 或 guard 引起的错误。
   尝试 `refly.type` 推出显式 O 也在该 helper 的 `T: *` 入口遇到同类错误。
   最小修正方向是使配置对 rule/pure-op 值的 universe 多态，或用 staged annex 描述符列表；
   不是将 tensor 算子谎称为低阶函数，也不是补齐 purity 后就停止验证。

2. **真实多态 App 的 Rule LHS 匹配。**
   rmap/rmm/rrev 的 rule_app 编译成功，但没有目标重写。
   当前 Rule adapter 对每个结构节点也匹配 type，mutable/ground 节点使用 exact identity
   （`src/mim/pattern/rule.cpp:37–59`），这不是 tensor op schema 加 alpha-equivalent
   dependent-type matching。
   本轮仅确定“LHS 没有产生可替换匹配”，未给每个失败的中间 App/type 子模式加 instrumentation；
   不把“所有 array capture 均不支持”当作结论，rpack 的成功已否定这个说法。
   下一小步应隔离部分应用的 Pi、隐式类型组和 Pack metadata，复用
   `src/mim/plug/tensor/pattern.cpp` 的真实 op adapter，保留最终替换类型检查。

规则 telescope 或函数参数的 tuple η 归约是另一个实验干扰因素：
仅有 `(a,b)` 时 `mm(a,b)` 可能变为 whole-telescope capture，RHS 单独引用 a/b 会缺绑定。
模型构造规则显式多一个未使用字段，候选函数也保留 salt，以隔离 lambda 能力；
不声称这修复了 whole-telescope/projection 支持。
现有 `validate_bindings` 不能反向构造缺少的 capture
（`src/mim/pattern/rule.cpp:79–101`）。

## 最小扩展的优先级

以下“解锁”指移除一个必要阻挡，不承诺单项扩展就完成调度与 lowering。

| 优先级 | 最小扩展 | 涉及实验 | 边界 |
| --- | --- | --- | --- |
| 1 | Universe-safe 的 graph phase 配置 | 真实 1/2、3、4、7，4 个 probe 文件；`tf_poly` 隔离入口 | 只移除配置类型错误；6 已可通过，不应强迫把未改写的 slice 列入 pure_ops |
| 2 | 真实 tensor Rule LHS 的 schema/type 匹配适配 | 同上 4 个 probe 的 rule_app 未匹配问题 | 先定位中间 type/metadata 子模式；不取消最终替换类型检查 |
| 3 | Region-local GraphView + 带捕获环境的 scope publisher | 嵌套 1/2，以及后置 5 的 region 部分 | owner/environment 必须保留 offset、accumulator 和 type 依赖，不能仅把 Lam body 加入原始遍历 |
| 4 | 带新 binder/环境的 RHS recipe，仍在独立 World 构造与验证 | graph 4，以及 graph 5 的构建部分；真实 GEMM 的后续生成 | single-root 已能建 lambda；不要把它当作两个 frontend 共同缺口 |
| 5 | Alpha-aware 的 LHS binder matcher，加 region 间 box/map 事务 | 模型/真实 5 | 只给 lambda wildcard 会丢失 offsets 与内部 use 的约束 |
| 6 | Staged snapshot guards / native witnesses，真实 tensor data-use schema | 模型/真实 7；2/3 一般合法性 | Unknown、coverage 和 witness generation 要显式；不放入全局纯 normalizer |
| 7 | tiled/tile_fold/stage/outs 的实际 lowering、realization 与持久资源协议 | 所有实验的数值、共享和性能含义，尤其 2/3/4/6 | 这是 tensor/storage 语义，不是 P5 matcher 的小补丁，另列为性能实验的前提 |

**结论：** 用现有 single-root 路径可先做“整棵模型 nest 的规则生成”；
用现有 graph 路径可做“闭合函数内、lambda-free recipe 的多根值替换”。
不能以这两项通过宣称 C++ tile_fuse 已被替代：
真实算子适配、跨 region 合法性，以及 stage/accumulator 的执行合同仍需分别验证。
