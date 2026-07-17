# Regex Staging Comparison

MimIR currently has two regex construction paths. They share the same low-level
memory and LLVM infrastructure, but stage the compiler at different levels.

| Aspect | Python `RegBuilder` path | MimIR tagged-AST path |
|---|---|---|
| Frontend representation | `MimRegex` wraps a `Def` produced by calls such as `regex.lit` and `regex.conj` | `%regex.Expr` is an explicit union-based AST built by `%regex.expr.*` |
| Interpreter/compiler | Python methods construct `%regex.*` axiom applications; C++ `regex2nfa` interprets those applications | `%regex.interpret` pattern-matches the AST and directly evaluates matcher semantics |
| Static work | Python builds the graph eagerly; regex normalizers simplify it | MimIR partial evaluation removes static union injections, `match`, and interpreter recursion |
| Residual program | NFA -> DFA -> generated state machine | Loads, character comparisons, position updates, and branches |
| Backend | Existing DFA lowering, then LLVM/JIT | Direct residual matcher, then LLVM/JIT |
| Dynamic input | Passed to the generated `match_func` as a byte string | Passed as `%regex.Input n = [%mem.M 0, Str n, Idx n]` |
| Current expressiveness | Includes star, plus, negation, lookahead, and normalizer support | POC currently supports leaves plus one level of concat, alternation, and optional |

The Python path is a **host-side staged builder**. The Python API constructs a
MimIR graph, and the regex plugin later recognizes that graph as a regex syntax
tree. The tagged-AST path makes the syntax tree explicit in MimIR and expresses
the interpreter in the object language. Its first Futamura projection is
therefore visible in the IR: specializing a closed expression removes the AST
and leaves a matcher that can be lowered independently of the regex frontend.

The `Input n` alias hides the common matcher state:

```mim
lam Input (n: Nat): * = [%mem.M 0, Str n, Idx n];
let RE = {n: Nat} -> Input n -> Res n;
```

This is only a representation alias; it does not add an effect system or
change memory threading. The explicit memory token remains part of the
runtime contract and is carried through the residual matcher.

There are now also two intermediate automaton experiments. `%regex.nfa.*`
defines an object-language NFA representation with fixed-size bitset state
sets, epsilon closure, and character move. For a closed NFA, these operations
are ordinary MimIR computations and the current test shows them reducing to
constants after partial evaluation. `%regex.DFA ns k` stores accepting flags,
fallback states, and fixed-capacity range transitions. `%regex.specialize_dfa`
still decodes a closed table in C++ and preallocates one residual continuation
per state before filling any body. This finite specialization cache handles
DFA cycles and removes dynamic static-table extraction. It is the MimIR
analogue of combining Truffle loop explosion with a
partial-evaluation-constant state node.

The three paths should remain available while the design is evaluated. The
Python/DFA path is the mature general-purpose implementation; the tagged-AST
path demonstrates object-language pattern matching; the object-language NFA
path starts migrating automaton construction into MimIR; and the closed-DFA
path demonstrates cyclic control-flow specialization and LLVM/JIT. The next
missing piece is object-language subset construction from NFA state sets to a
closed `%regex.DFA` table.

Implementation details, matcher semantics, residual-IR expectations, and the
test matrix are documented in [Closed DFA Specialization](regex_dfa_specialization.md).
