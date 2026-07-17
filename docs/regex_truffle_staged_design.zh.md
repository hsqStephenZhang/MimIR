# Regex AST、Automata 与 Partial Evaluation 设计

## 1. 目标

本文讨论如何参考 Graal Truffle 的 AST 与 partial evaluation 方式，重新组织 MimIR 的 regex 实现：

```text
RegexExpr
    -> NFA graph
    -> DFA/state table
    -> matcher interpreter
    -> partial evaluation
    -> LLVM/JIT
```

目标不是立即替换现有 `LowerRegex`，而是验证一个更强的 object-language staging 模型：

- regex syntax 是静态值；
- input string 是动态值；
- AST、NFA/DFA 构造和 interpreter dispatch 在 partial evaluation 中消失；
- 最终只保留 input load、字符比较、位置更新和控制流；
- 生成的 residual program 可以继续进入 LLVM/JIT backend。

## 2. Truffle 的参考模型

Truffle 通常不依赖递归 algebraic data type 表达 AST，而是使用递归的 Node object graph：

```java
abstract class RegexNode extends Node {
    abstract MatchResult execute(Input input, int position);
}

final class ConcatNode extends RegexNode {
    @Child private RegexNode lhs;
    @Child private RegexNode rhs;

    @Override
    MatchResult execute(Input input, int position) {
        MatchResult left = lhs.execute(input, position);
        return rhs.execute(input, left.position());
    }
}
```

`@Child` 和 `@Children` 描述静态 node graph。Graal partial evaluator 可以识别具体的 node class，内联 `execute`，并消除虚调用和 AST dispatch。

在 DFA executor 中，DFA state table 通常通过 `@Children` 和 `@CompilationFinal` 保存。状态数量、successor table 和 matcher table 是 compilation constants，input buffer 和当前位置则是 runtime values。

MimIR 中对应的概念是：

| Truffle | MimIR |
|---|---|
| Node class hierarchy | 递归 union / recursive node type |
| `@Child RegexNode` | 递归 child field |
| `execute()` dispatch | union `match` interpreter |
| static Node object graph | closed MimIR value graph |
| `@CompilationFinal` | closed/static value fact |
| `CompilerAsserts.partialEvaluationConstant` | `@(%core.pe.is_closed x)` 或静态参数 |
| `@ExplodeLoop` | static array unroll / affine unroll |

## 3. 当前实现与目标实现

当前 tagged AST POC 定义了：

```text
Leaf      = Empty | Any | Lit Char | Range
Expr      = Leaf
          | Concat Leaf Leaf
          | Alt Leaf Leaf
          | Optional Leaf
```

解释器使用普通 MimIR `match`。对静态 union injection，MimIR 的 `World::match` 会直接选择对应 arm，从而消除 `Match` 和 AST dispatch。

当前 POC 只支持有限深度，主要用于验证：

```text
static tagged AST
    -> pattern-matching interpreter
    -> static Match reduction
    -> residual load/compare/branch code
```

完整目标应当支持递归表达式：

```text
RegexExpr =
    Empty
  | Any
  | Lit Char
  | Range Range
  | Concat RegexExpr RegexExpr
  | Alt RegexExpr RegexExpr
  | Optional RegexExpr
  | Star RegexExpr
  | Plus RegexExpr
  | Not RegexExpr
  | NegLookahead RegexExpr
```

## 4. 递归类型设计

从类型理论角度，可以表示为：

```text
RegexExpr = μ E. RegexExprF E
```

其中：

```text
RegexExprF E =
    Empty
  | Any
  | Lit Char
  | Range Range
  | [E, E]       // Concat
  | [E, E]       // Alt，实际需要不同 payload 类型
  | E            // Optional
```

更适合 MimIR 的第一版是 iso-recursive type：

```text
roll   : RegexExprF RegexExpr -> RegexExpr
unroll : RegexExpr -> RegexExprF RegexExpr
```

解释器通过 `unroll` 暴露一个 node，再对 node 做 pattern matching：

```mim
interpret node input =
    match unroll node with
        | Empty       => matcher.empty input
        | Any         => matcher.any input
        | Lit c       => matcher.lit c input
        | Range r     => matcher.range r input
        | Concat p    => ...
        | Alt p       => ...
        | Star e      => ...
```

`roll/unroll` 的优势是明确递归边界，便于：

- 检查递归类型的 positivity；
- 控制 normalizer 是否展开递归；
- 对闭合 AST 做递归 partial evaluation；
- 对动态 AST 保留 residual interpreter。

如果暂时不增加完整的递归类型语法，也可以增加专门的 `type rec` 声明，内部 lowering 为 `roll/unroll`。

## 5. RegexExpr 到 NFA

NFA 不应首先表示为递归 heap graph，而应表示为静态 table：

```text
State = [
    transition_begin: Nat,
    transition_count: Nat,
    accepting: Bool
]

Transition = [
    target: Nat,
    range: «2; I8»,
    epsilon: Bool
]

NFA = [
    states: «num_states; State»,
    transitions: «num_transitions; Transition»
]
```

`Concat`、`Alt`、`Optional`、`Star` 和 `Plus` 可以通过显式 epsilon transition 表示。NFA 的 graph topology 是静态的，runtime 只需要处理 input 字符和 state set。

第一版不建议引入动态 map 或动态 vector。可以使用：

- static-size arrays；
- `Nat` index；
- fixed-capacity worklist；
- bitset state set。

例如：

```text
StateSet = «num_nfa_states; Bool»
```

这样 epsilon closure 可以通过静态循环和 bitset membership 实现。

## 6. NFA 到 DFA

DFA subset construction 的核心操作是：

```text
epsilon_closure : StateSet -> StateSet
move            : StateSet × Char -> StateSet
subset_construct: NFA -> DFA
```

DFA 可以表示为：

```text
DFAState = [
    accepting: Bool,
    transitions: «num_ranges; TransitionTarget»
]

DFA = «num_dfa_states; DFAState»
```

对于静态 regex，`num_nfa_states` 和 `num_dfa_states` 都是 partial-evaluation constants。工作列表可以使用固定大小数组：

```text
worklist : «capacity; StateSet»
```

第一版可以使用线性查找已有 state set：

```text
find_state(set, states) -> Option (Idx n)
```

这不是最优的 runtime 算法，但当整个 DFA construction 在 compile time 消除时，不会进入最终 matcher。

## 7. DFA Matcher Interpreter

matcher 的 runtime 输入可以统一表示为：

```mim
lam Input (n: Nat): * = [%mem.M 0, Str n, Idx n];
lam Res (n: Nat): * = [%mem.M 0, Bool, Idx n];
let RE = {n: Nat} -> Input n -> Res n;
```

DFA interpreter 的动态状态只有：

```text
input pointer
input position
current DFA state
loaded character
```

静态状态包括：

```text
DFA state table
transition ranges
successor state ids
accepting flags
```

解释器伪代码：

```text
match_dfa(dfa, input):
    state = dfa.entry
    position = input.start

    loop:
        if dfa[state].accepting:
            return success(position)
        if input[position] == '\0':
            return failure(position)

        c = input[position]
        state = dfa[state].transition(c)
        position = position + 1
```

当 `dfa` 是 closed value 时，partial evaluation 可以消除对 state table 的间接访问，并生成类似 `dfa2matcher.cpp` 的状态机代码。

### 7.1 已实现的 closed-DFA specialization

第一阶段已经加入以下静态入口：

```mim
axm %regex.specialize_dfa: {ns k n: Nat} ->
    [%regex.DFA ns k, Input n] -> Res n;
```

其中 `ns` 是状态数，`k` 是每个状态固定容量的 transition 数。normalizer 只在 DFA table 是 closed value 时触发，并执行：

1. 解码 entry、error、accepting、fallback 和 transition ranges；
2. 先为全部 `ns` states 分配 continuation placeholder；
3. 再填充各 state body，并直接引用已有 placeholder 表示回边；
4. 将固定的 `k` 个 transition ranges 展开为字符比较和分支；
5. 保留动态 input load、position update 和 continuation jump。

先分配、后填充相当于一个按 state identity 建立的有限 specialization cache，因此 `a+` 一类带自环的 DFA 不会导致 partial evaluation 无限展开。它对应 Truffle 中：

```text
@ExplodeLoop + partialEvaluationConstant(stateNode)
```

当前实现位于 regex 插件 normalizer，而不是通用 PE phase。这已经证明 continuation graph 和 LLVM/JIT 路径可行，但还不能视为 MimIR 对任意静态 graph interpreter 都自动支持 loop explosion。

## 8. Partial Evaluation 边界

应该明确区分：

```text
static:
    RegexExpr
    NFA
    DFA
    state table

dynamic:
    input string
    memory token
    input position
```

目标 residual code：

```text
load(input, position)
compare(character, constant)
branch
increment(position)
```

应该被消除的内容：

```text
RegexExpr tags
recursive AST dispatch
NFA construction
epsilon closure
DFA subset construction
state-set equality
static transition table lookup
```

建议对 regex 参数增加闭合约束：

```mim
lam run (expr: RegexExpr)@(%core.pe.is_closed expr)
    (input: Input n): Res n =
    interpret expr input;
```

如果 regex 是动态值，则不能期待 compile-time DFA construction；这时应保留 runtime interpreter，或者显式拒绝该调用。

`affine.unroll` 最多用于展开固定的 state/transition 迭代范围。它本身不能消除：

```text
dfa.states[dynamic_state]
```

真正需要的是把动态 state dispatch residualize 为有限的 per-state continuations。输入长度循环应保持动态，避免代码大小随输入上界增长。

## 9. 与 LowerRegex 的关系

现有路径是：

```text
%regex.* graph
    -> LowerRegex
    -> regex2nfa.cpp
    -> NFA
    -> DFA
    -> dfa2matcher.cpp
    -> LLVM/JIT
```

新路径是：

```text
RegexExpr
    -> MimIR NFA construction
    -> MimIR DFA construction
    -> MimIR matcher interpreter
    -> partial evaluation
    -> LLVM/JIT
```

完整实现后，两者可以达到语义等价，但生成 graph 的时间和位置不同：

| 项目 | `LowerRegex` | MimIR staged pipeline |
|---|---|---|
| syntax representation | implicit `%regex.*` axiom graph | explicit recursive `RegexExpr` |
| NFA/DFA construction | C++ compiler pass | MimIR object-language computation |
| specialization | C++ lowering directly creates matcher | partial evaluation specializes interpreter |
| runtime AST dispatch | none after lowering | none after PE |
| runtime input | dynamic | dynamic |
| current feature coverage | full existing regex subset | initially smaller POC |
| optimization visibility | compiler implementation detail | visible and inspectable in MimIR |

当前 tagged-AST POC 仍然弱于 `LowerRegex`，因为它只有有限深度并且尚未支持 `Star/Plus/Not/NegLookahead`。要达到等价，必须同时补齐递归 AST 和完整 automata semantics。

当前 closed-DFA POC 又向前验证了一步：手工构造的静态 DFA table 已经可以生成有环 matcher 并通过 LLVM/JIT，但 `RegexExpr -> NFA -> DFA` 仍未在 MimIR object language 中实现。因此当前完成的路径是：

```text
closed DFA table
    -> specialization normalizer
    -> cyclic continuation graph
    -> LLVM/JIT
```

## 10. 分阶段实施计划

### Phase 1: DFA interpreter

保留现有 C++ `regex2nfa` 和 DFA construction，只把静态 DFA table 交给 MimIR matcher interpreter：

```text
C++ RegexExpr/NFA/DFA
    -> MimIR DFA interpreter
    -> partial evaluation
    -> LLVM/JIT
```

这是最低风险的验证，可以直接比较 MimIR residual matcher 和 `dfa2matcher.cpp` 的输出。

### Phase 2: NFA interpreter

在 MimIR 中实现：

```text
epsilon_closure
move
NFA execution
```

NFA table 仍可由 C++ 生成。此阶段验证 bitset、static loops 和 epsilon semantics。

### Phase 3: MimIR DFA construction

再将 subset construction 移入 MimIR：

```text
NFA -> StateSet worklist -> DFA
```

使用 fixed-capacity arrays 和 static `Nat` loops，避免一开始引入 runtime allocator、hash map 和动态 graph。

### Phase 4: 完整 RegexExpr

增加：

- recursive type / `type rec`；
- `Star`；
- `Plus`；
- character-class complement；
- negative lookahead；
- 与现有 LowerRegex 的 differential tests。

## 11. 结论

该 pipeline 是可行的，并且与 Truffle 的静态 AST + dynamic input + partial evaluation 模型一致。

最稳妥的实现顺序不是立即重写整个 `LowerRegex`，而是：

```text
现有 C++ DFA
    -> MimIR DFA interpreter
    -> partial evaluation
    -> LLVM/JIT
```

随后逐步迁移 NFA interpreter 和 DFA construction。这样可以分别验证：

1. 递归 AST 表达；
2. 静态 graph 的 partial evaluation；
3. matcher interpreter specialization；
4. automata construction 的语言表达能力；
5. 与 `LowerRegex` 的语义等价性。
