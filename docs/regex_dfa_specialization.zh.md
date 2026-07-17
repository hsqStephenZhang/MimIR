# MimIR Closed-DFA 特化设计与实现记录

## 1. 目标

本阶段验证 regex pipeline 的最后一段：

```text
closed DFA table
    -> matcher interpreter specialization
    -> cyclic continuation graph
    -> LLVM/JIT
```

核心 binding-time split 是：

```text
static:
    state 数量 ns
    每个 state 的 transition 容量 k
    entry/error state
    accepting/fallback/transition table

dynamic:
    memory token
    input pointer
    input position
    loaded character
```

输入长度循环必须保持动态；只将有限的 DFA topology residualize 为控制流。

## 2. Object-Language 表示

```mim
lam %regex.dfa.Transition (ns: Nat): * =
    [lo hi: Char, target: Idx ns];

lam %regex.dfa.State (ns: Nat) (k: Nat): * =
    [accepting: Bool,
     fallback: Idx ns,
     transitions: «k; %regex.dfa.Transition ns»];

lam %regex.DFA (ns: Nat) (k: Nat): * =
    [entry error: Idx ns,
     states: «ns; %regex.dfa.State ns k»];
```

`Idx ns` 在类型层保证 entry、error、fallback 和 target 都引用同一张表中的 state。`k` 是统一容量；out-degree 较小的 state 使用 `[255, 0, error]` 填充，因为这个 unsigned inclusive range 永远不匹配。

裸 projection，例如：

```mim
state#(2:(Idx 3))
```

虽然紧凑，但不容易看出 `2` 表示 transitions。因此增加了普通 lambda accessor：

```mim
%regex.dfa.state_accepting
%regex.dfa.state_fallback
%regex.dfa.state_transitions
%regex.dfa.transition_lo
%regex.dfa.transition_hi
%regex.dfa.transition_target
%regex.dfa.entry
%regex.dfa.error
%regex.dfa.states
```

这些 accessor 会被 beta reduction 完全消除，不增加 runtime cost。

## 3. Interpreter 语义

完整 matcher 的 Python 伪代码是：

```python
def interpret(dfa, input_bytes, pos):
    state_id = dfa.entry

    while state_id != dfa.error:
        current = dfa.states[state_id]
        c = input_bytes[pos]

        if c == 0:
            return current.accepting, pos

        state_id = next_state(current, c)
        pos += 1

    return False, pos
```

单个 transition 的语义：

```python
def pick(c, next_state, transition):
    lo, hi, target = transition
    return target if lo <= c <= hi else next_state
```

transition fold 的语义：

```python
def fold_transitions(transitions, c, i, next_state):
    while i < len(transitions):
        next_state = pick(c, next_state, transitions[i])
        i += 1
    return next_state
```

state successor 的语义：

```python
def next_state(state, c):
    return fold_transitions(
        transitions=state.transitions,
        c=c,
        i=0,
        next_state=state.fallback,
    )
```

如果 transition ranges 重叠，后面的 transition 覆盖前面的结果。规范化 DFA 应使用互不重叠的 ranges。

该 DFA interpreter 是 full-string matcher：只有在 accepting state 看到 NUL 才成功。它不同于当前 `%regex.interpret`/`RE` 的 prefix-match 语义。

## 4. Higher-Order Staging 接口

底层 staging axiom 是：

```mim
axm %regex.specialize_dfa: {ns k n: Nat} ->
    [%regex.DFA ns k, Input n] -> Res n;
```

面向调用者的第一 Futamura projection 是：

```mim
let %regex.dfa.compile = lm {ns k: Nat}
    (dfa: %regex.DFA ns k): RE =
        lm {n: Nat} (input: Input n): Res n =
            %regex.specialize_dfa (dfa, input);
```

使用方式：

```mim
let matcher = %regex.dfa.compile static_dfa;
let (mem, matched, pos) = matcher dynamic_input;
```

`compile` 和 accessor 都是 library-level lambda abstraction。beta reduction 后，normalizer 直接看到 closed DFA 和 dynamic input。

## 5. Truffle-Style Specialization

Truffle DFA executor 常组合：

```text
@CompilationFinal state table
@ExplodeLoop
partialEvaluationConstant(stateNode)
```

MimIR 中的等价实现为：

1. 将 closed MimIR table 解码为 host-side `DFATable`；
2. 先为全部 `ns` states 创建空 continuation；
3. 再逐个填充 state body；
4. 将 `k` 个 ranges 直接生成为 unsigned comparisons；
5. transition/fallback 直接跳转到目标 continuation；
6. 用 `%cps.cps2ds_dep` 接回 direct-style call site。

关键是“先分配、后填充”。例如：

```text
state_1 --'a'--> state_1
```

回边直接引用已经存在的 `state_1` placeholder，不会再次递归展开 interpreter。最终代码大小为 `O(ns * k)`，与输入长度无关。

## 6. 遇到的问题与解决方式

### 6.1 动态 state 索引无法被普通 PE 消除

最初的 table interpreter 会残留：

```text
current = static_states[dynamic_state]
```

MimIR scalarizer 无法将这个动态 aggregate extraction 自动变成 per-state CFG，LLVM backend 也不能直接 lower 该 nested table lookup。

解决方式：将 state identity 编码为 continuation identity，生成有限的 mutually recursive continuation graph，从 residual IR 中完全移除 table lookup。

### 6.2 `is_closed state` 不能独立解决 DFA 环

当前 reduct cache 按完整参数缓存，其中包括动态 `pos`，并且在 body rewrite 完成后才写入缓存。循环 DFA 会在缓存建立前以新的 position 再次进入同一个 state，可能无限特化。

解决方式：显式预分配每个 static state 的 placeholder，相当于按 static state identity 建立 specialization cache。

### 6.3 `affine.unroll` 不等于 state specialization

`affine.unroll` 可以展开固定的 `ns`/`k` 循环，但不能消除 `states[dynamic_state]`。

解决方式：unroll 只负责有限 transition 结构；state dispatch 必须 residualize 为 continuation graph。动态 input loop 不展开。

### 6.4 Annex 名称层级限制

尝试定义 `%regex.dfa.transition.lo` 时，autogen C++ enum 产生了包含 `.` 的非法枚举项。

解决方式：保持一层 member 命名：

```text
%regex.dfa.transition_lo
%regex.dfa.state_transitions
```

### 6.5 `k = 1` 的 array folding 与类型推断

单元素 dependent array 会折叠为 element type，导致测试中无法从裸 tuple 稳定推断 `k = 1`。

解决方式：测试使用 `k = 2` 并加入 never-match padding。长期可以考虑为 DFA constructor 增加更明确的 type annotation/constructor API。

### 6.6 Prefix Match 与 Full Match

旧 tagged-AST `%regex.interpret` 返回 prefix match；closed DFA 在 accepting state 遇到 NUL 才成功。测试最初错误地假设两者相同，也曾把 Bool 退出码 1/0 写反。

解决方式：分别记录语义并增加正反例：`pe_expr` 明确测试 prefix match，DFA 测试明确测试 full match。

## 7. Residual IR

specialization 后保留：

```text
state_N continuations
input loads
constant range comparisons
position increments
direct branches
```

应当完全消失：

```text
%regex.specialize_dfa
%regex.dfa.compile
static DFA aggregate
dynamic static-table extraction
transition fold interpreter
```

## 8. 测试覆盖

| 测试 | 对应 pattern | 覆盖点 |
|---|---|---|
| `pe_dfa.mim` | `a+b` | self-loop、accepting tail、正反例 |
| `pe_dfa_abc_plus.mim` | `(abc)+` | Python example、多 state cycle、空串 |
| `pe_dfa_email.mim` | email representative subset | ranges、标点、多个 loops、严格 full match |
| `pe_expr.mim` | `ab` AST | static Match elimination、prefix-match 语义 |

每个 DFA test 都执行：

1. FileCheck 确认 staging axiom 和 static table 消失；
2. LLVM emission；
3. Clang native compilation；
4. 多个 positive/negative runtime cases。

最终验证结果：

```text
regex lit: 54 passed, 1 pre-existing unsupported
regex CTest: 9/9 passed
```

## 9. 当前边界与下一步

已经证明：

```text
closed DFA -> cyclic matcher graph -> LLVM/JIT
```

尚未实现：

```text
RegexExpr -> NFA graph -> DFA/state table
```

下一阶段可以在 object language 中实现 NFA construction、subset construction 和可选 minimization。更通用的语言机制则是将本次 placeholder 算法抽象为 finite static-dispatch specialization，但第一版保留在 regex plugin 中更容易控制语义和代码大小。
