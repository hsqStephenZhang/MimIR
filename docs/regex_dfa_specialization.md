# Closed DFA Specialization

## Purpose

`%regex.specialize_dfa` is the first cyclic-graph case study for regex staging
in MimIR. It specializes a closed DFA table while leaving the input string,
memory token, and input position dynamic:

```text
closed DFA table + dynamic input
    -> per-state continuation graph
    -> LLVM IR
    -> native matcher
```

This sits between the tagged regex AST experiment and the existing C++
`LowerRegex` implementation. It proves that static automaton topology can
disappear without unrolling the dynamic input loop.

## Object-Language Representation

The public table types are:

```mim
lam %regex.dfa.Transition (ns: Nat): * = [Char, Char, Idx ns];

lam %regex.dfa.State (ns: Nat) (k: Nat): * =
    [accepting: Bool,
     fallback: Idx ns,
     transitions: «k; %regex.dfa.Transition ns»];

lam %regex.DFA (ns: Nat) (k: Nat): * =
    [entry error: Idx ns,
     states: «ns; %regex.dfa.State ns k»];
```

`ns` fixes the number of states. `k` is a uniform transition capacity, which
makes the table homogeneous even when states have different out-degrees.
Unused slots can use the impossible inclusive range `[255, 0]`.

A transition `[lo, hi, target]` matches `c` when both unsigned comparisons
hold:

```text
lo <= c && c <= hi
```

Well-formed DFA tables should use disjoint ranges. The reference interpreter
and specializer both give the later array element priority if ranges overlap.

## Matcher Semantics

The dynamic input and result aliases are:

```mim
lam Input (n: Nat): * = [%mem.M 0, Str n, Idx n];
lam Res   (n: Nat): * = [%mem.M 0, Bool, Idx n];
```

Starting at `entry` and `input.pos`, the matcher repeatedly performs:

```python
def match(dfa, input, pos):
    state = dfa.entry

    while True:
        if state == dfa.error:
            return False, pos

        c = input[pos]
        if c == 0:
            return dfa.states[state].accepting, pos

        current = dfa.states[state]
        state = current.fallback
        for lo, hi, target in current.transitions:
            if lo <= c <= hi:
                state = target
        pos += 1
```

This is full-string matching: reaching an accepting state is insufficient
until the next byte is NUL. A transition to `error` rejects after consuming the
current byte. The error state itself never loads input.

## Staging API

```mim
axm %regex.specialize_dfa: {ns k n: Nat} ->
    [%regex.DFA ns k, Input n] -> Res n;
```

The DFA argument must be closed, and its entry, error, flags, fallback ids,
range bounds, and targets must be projectable literals. The input remains
dynamic. If these static requirements are not met, normalization declines and
the staging axiom remains in the IR; it currently has no runtime/LLVM lowering.
`%regex.dfa.interpret` is the reference semantics for experiments with a
runtime table.

## Specialization Algorithm

The normalizer first decodes the closed MimIR value to `DFATable`. It then:

1. Creates shared `accept` and `reject` continuations.
2. Allocates one empty continuation for every DFA state.
3. Fills each state continuation after all placeholders exist.
4. Emits one inclusive range check per transition slot.
5. Connects matching and fallback edges directly to state continuations.
6. Wraps the CPS matcher with `%cps.cps2ds_dep` at the call site.

The allocation order is the important part. For a self-loop such as
`state_1 --'a'--> state_1`, body construction refers to the existing
`state_1` placeholder. It does not recursively specialize another invocation.
The generated graph is finite even for arbitrary DFA cycles.

## Relation to Truffle

The implementation corresponds to the following Truffle mechanisms:

| Truffle | MimIR closed-DFA specialization |
|---|---|
| `@CompilationFinal` state table | closed `%regex.DFA ns k` value |
| `partialEvaluationConstant(stateNode)` | identity of a state continuation |
| `@ExplodeLoop` over transitions | emitted constant range checks |
| cyclic node graph | preallocated mutually recursive continuations |

`affine.unroll` can help expand a fixed `ns` or `k` iteration, but it cannot by
itself remove `states[dynamic_state]`. The specializer removes that operation
by residualizing state identity into control-flow identity.

The input loop is intentionally not unrolled. Runtime code size is
`O(ns * k)`, independent of input length.

## Residual Program

After specialization and optimization, the residual program contains:

```text
state_N continuations
input byte loads
constant unsigned comparisons
position increments
direct continuation branches
```

It does not contain:

```text
%regex.specialize_dfa
the static DFA aggregate
dynamic extraction from the state table
transition-fold interpreter calls
```

The residual graph can pass through the existing LLVM emitter and Clang.

## Tests

The lit coverage is split by graph shape:

| Test | Pattern represented | Main property |
|---|---|---|
| `pe_dfa.mim` | `a+b` | direct self-loop and accepting tail |
| `pe_dfa_abc_plus.mim` | `(abc)+` | multi-state cycle from the Python example |
| `pe_dfa_email.mim` | representative email subset | ranges, punctuation, several loops, strict full match |

Each test checks that the staging axiom and static table names disappear,
then compiles the generated LLVM IR and runs positive and negative examples.

## Current Boundary

This implementation proves the final part of the staged pipeline:

```text
closed DFA -> specialized matcher -> LLVM/JIT
```

It does not yet construct the NFA or DFA in the MimIR object language. The
remaining stronger Futamura experiment is:

```text
closed RegexExpr
    -> object-language NFA construction
    -> object-language subset construction/minimization
    -> closed DFA specialization
```

A future general PE facility could abstract the placeholder algorithm as
finite static-dispatch specialization. Until then, keeping it in the regex
plugin makes the staging contract and code-size behavior explicit.
