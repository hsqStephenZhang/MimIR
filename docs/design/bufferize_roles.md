# Bufferization by value role, and the erasure boundary

Status: historical proposal, 2026-10-28, amended after review. Describes branch `zc/feat/erase-dynamic-extents`
before the nominal Tensor implementation. Step 1 of the old plan (section 4) was implemented there; step 2 was not.

The implementation plan is superseded by [Nominal, shape-indexed tensors](nominal_tensor.md), which adopts the
nominal alternative in §2.2 with a complete parameter witness. Its first version supports static and runtime
extents with a literal rank, including loop-based dynamic copies and static-literal array ingress. Sections 2 and 4 below retain
the earlier alternatives and plan for context, not as current recommendations. Section 3 describes the branch's existing erasure mechanism,
which the nominal implementation retains for dynamic-shape lowering. Its dynamic copy work is required by the new specification; runtime rank and runtime-sized device transfers remain deferred.

## 1. Problem

`tensor.lower_to_mem` decides which values become `buffer.Buf` handles by a set of **types**, `tensor_ty_`.
A type enters the set when it is the type of a `tensor.buf` annotation, of a tensor op's operand or result, or
(since b7478d4235) of an argument at a call of a dependent bufferized function.
Everything whose type is in the set is treated as a tensor: parameters, continuation parameters, extracts, inserts.

MimIR tensors are plain dependent arrays `«s; T»`, and types are structural and hash-consed.
So one type is shared by values with different roles:

| value | type | role |
|---|---|---|
| a shape vector `(4, 1)` | `«2; Nat»` | metadata |
| a tensor of two naturals | `«2; Nat»` | tensor |
| a schedule nest's loop vector | `«r; I32»` | element-level scalar aggregate |
| an I32 feature vector | `«r; I32»` | tensor |

Classifying by type therefore needs patches wherever a non-tensor value happens to share a tensor's type.
`lower_to_mem.cpp` already carries several:

- `add_tensor_ty` refuses fully folded shapes, since `«1; T» ≡ T` "would poison every function whose signature
  mentions the element type";
- `op_args_` exempts lams reachable from a tensor op's curry chain (combiners, index maps, schedule nests)
  so that a nest's `«r; I32»` loop vector is not mistaken for a recorded `«r; I32»` tensor;
- `covers_every_axis` rejects partial reads, but is evaluated on every `Extract` whose base type is recorded,
  including compiler-generated helper code (Codex finding 6: a `«4, 4» × «4, l»` product fails with
  "partial tensor read yields a sub-tensor" on generated affine code).

b7478d4235 widened the surface: recording the instantiated argument types at calls of dependent callees turns a
caller's `(4, 1): «2; Nat»` shape argument into a buffer (Codex finding 2, P1).
When the metadata is unused this compiles silently with a bogus `buffer.alloc`.

The root cause is not any single patch: *tensor-ness is a property of a value's role, not of its type*, and in an
IR where tensors are ordinary arrays, the type cannot carry that role.

## 2. Options

### 2.1 Classify values by provenance (earlier recommendation; superseded)

Replace `tensor_ty_: DefSet` (types) by `tensors_: DefSet` (values), computed by a forward fixed-point analysis
(`mim::Analysis`, as `mem::EscapeAnalysis` does) before the rewrite:

Seeds
- the operand of every `tensor.buf`;
- the result of every tensor op, and its tensor operands (the op's own signature says which positions are tensors:
  `is`, `post_is`, `input`, …; the ranks/shape arguments are not);
- nothing else. In particular no type is a seed.

Propagation (value to value)
- `Extract`/`Insert`: a tensor base makes the insert's result a tensor; an extract that stops short of the element
  is a partial read and is rejected *here*, only for a base that is a tensor, never for helper code;
- tuples and projections: a tensor component makes the tuple "carry" a tensor; projecting a carrying tuple at that
  position yields a tensor;
- `core.select`/dispatch between tensors: both branches are tensors; a mix is an error;
- `core.bitcast` between array types: the result is a tensor iff the operand is;
- call sites: at `f args`, argument position `i` is a tensor iff `args#i` is; this marks the callee's parameter
  `f.var(i)` (for a curried `f shape args`, positions are counted after flattening, see `uncurry`);
- continuations: when a tensor `v` is passed to a continuation `k` at position `i`, `k.var(i)` is a tensor; a
  function's `return` continuation is just such a `k`;
- element-level lams (today's `op_args_`): a lam that is an operand of a tensor op's curry chain is *elementwise*;
  its parameters are never tensors even if a tensor value flows in syntactically (they receive elements).

The fixed point is reached when no new value is marked. The lattice is a set; the analysis is monotone, so it
terminates after at most one pass per marked value.

What the review found missing (to be specified before step 2)
- One set is not enough. Three things are needed: *a value is a tensor*, *an aggregate holds tensors at given
  component paths*, and *a callable takes or returns tensors at given positions* (a summary per function).
  Homogeneous sigmas normalize to arrays, singleton tuples vanish and nested arrays fuse, so a pair of equal-shape
  tensors and a higher-rank tensor are the same type; only paths tell them apart.
- Hash-consing applies to values too: the literal `(4, 1)` can be shape metadata in one use and tensor data in
  another. A mark on the value cannot serve both. The rule is therefore *representation demand*: a value keeps its
  representation and is materialized where a tensor-demanding position (a tensor op's tensor operand, a marked
  parameter, a marked continuation argument) consumes it; a marked parameter receiving an unmarked literal is not a
  conflict (`lit/buffer/exec_get_set.mim` does exactly that today).
- The interprocedural edges have to be built explicitly: actual argument → formal component path; continuation
  parameter → the continuation values callers supply; a call through a parameter → its possible targets; a
  `select`ed callee (an `Extract`, not a `Lam`) → every target's summary; a direct-style body → its application
  results. `mim::Analysis` does not supply them (its lattice is a substitution map, and `def ↦ def` does not
  invalidate), so collect the edges and drain a plain worklist in `start()`; `mem::EscapeAnalysis` builds its
  switch/target/CPS edges by hand for the same reason.
- An unset external (a C function) has a fixed ABI; a marked argument cannot change it. Keep today's rejection
  of unconvertible externals, or state an explicit import contract.
- Element-level lams are not exempt wholesale: a helper may receive a captured tensor. Each tensor op fixes which
  of its operands are tensors and which callbacks receive elements (`generate`/`splat`: none; `map_reduce_post`:
  each component of `is` and `post_is`; `broadcast`/`pad`: `input`; `concat`: each `is`; `gather`/`scatter`:
  `input`, the index array, `updates`); ranks, bounds, maps and schedules are metadata or callbacks, and a
  callback's parameters follow its call contract.
- "Every tensor op result" needs qualifying: `shape` is metadata, `if_static` is generic dispatch, a singleton
  tensor is physically a scalar.
- Diagnostics are produced after convergence, for positions whose representation demands conflict, not for every
  marked/unmarked pair.
- The rewrite is not unchanged: `buf_of`, `splat_scalar` and axis coverage still read structural types for shape
  and element information; `conv_boundary` needs type plus role, recursively and under the dependent binder;
  `materialize` needs the demanded role; extract/insert conversion and the per-lam `fresh_memo_` need the base
  value's role. Estimated 600–1000 lines before tests.

What this subsumes
- `tensor_ty_`, `mentions_tensor`, `is_tensor_fn`, `tensor_fns_`, `shape_poly_` and the call-site
  `record_boundary` of b7478d4235 collapse into "is `f.var(i)` marked"; a bufferized function is one with a marked
  parameter, shape-polymorphic or not; a dependent dom needs no separate spelling logic because parameters, not dom
  components, are looked up;
- `op_args_` becomes the elementwise exemption above, with the same transitive closure;
- `conv_boundary` converts exactly the marked positions of a dom; `materialize` is called only for marked
  arguments; partial-read rejection fires only on marked bases (Codex finding 6).

One PR, no change to `tensor.mim` or to the frontend contract (`buf` keeps its meaning: it is a seed).

### 2.2 A nominal tensor type

MLIR's answer: `tensor<…>` and `memref<…>` are distinct type constructors, so one-shot bufferization never asks
what a value *is*. In MimIR this would be an axiom type `tensor.T (s, T)` instead of `«s; T»`.
A `let` alias does not do it: `let T s T = «s; T»` is definitionally equal and hash-conses to the same node.
Mim does have nominal newtypes (`nom`, with explicit wrap/unwrap), so no new axiom is needed; the cost is the
interface: `#` as element access, `shape` as the type's arity, and every lemma the `Tileable` interface gets from
tensors being arrays would go through wrap/unwrap. Note that MLIR's distinct `tensor` type only removes the
classification question; One-Shot Bufferize still analyzes SSA uses, aliasing and op semantics per value.
Earlier assessment: rejected because of the interface cost. The superseding specification adopts this option
with a shape-indexed, abstract nominal wrapper and explicit tensor operations; see `nominal_tensor.md`.

### 2.3 Exclude metadata by element type (rejected)

Never treating an array of `Nat` as a tensor looked like a one-line stop-gap. It is wrong: `Nat` tensors are real
data (`lit/tensor/lower_broadcast.mim`, `dot_dynamic.mim`, `reassoc.mim`); excluding them silently sends them down
the scalar-fill path. The stop-gap that was implemented instead is positional: at a call of a dependent callee an
argument's type is recorded as a boundary only where the callee's own parameter is one, parameters compared by
value (so a grouped parameter is seen as the body spells it).

## 3. The erasure boundary

Independent of the classification, b7478d4235 erases runtime extents at `buffer.lower_ptr`: a `Buf` with a
non-literal dim lowers to the flat `mem.Ptr («⊤; T», 0)`, the element address is linearized from the logical shape.
Erasure induces an order *precise ≤ erased* on pointer types, and the two variances need the two casts:

- covariant (argument): a precise `Ptr «4; T»` passed where `Ptr «⊤; T»` is expected is cast precise→erased.
  Done in `erase_arg`. This cast forgets information and is always sound.
- contravariant (continuation parameter): a callee that hands its erased result to a caller's continuation typed
  `Cn [M, Ptr «4; T»]` needs the continuation adapted: `k' (m, p) = k (m, bitcast p)` (Codex finding 1, P1).
  The erased→precise cast reintroduces information; it is sound because the *caller* chose the extent.
  This is `memref.cast memref<?xi32> to memref<4xi32>` in MLIR: an explicit, trusted narrowing at the boundary,
  never implicit.

Implemented as a leaf rule of `RWPhase::coerce` (`coerce_leaf`, overridden by `LowerPtr`): every checking
position of the rewrite (an App's argument, a Tuple's ops, a Lam's body) goes through `coerce`, which explodes
tuples, η-wraps Pis (`eta_coerce`, memoized) and asks the leaf rule for the rest. The leaf rule casts between two
pointers to arrays iff their innermost element types and address spaces agree and one side is erased.

When is narrowing sound? Not because "the caller chose the extent" in the abstract: the proof is that the position
being coerced *type-checked before erasure*. The rewrite only ever coerces at positions that were well-typed in the
old world, where the dependent signature, instantiated with the actual arguments, equated the precise type with
the callee's; erasure removed the extent from one side only. So the cast reinstates exactly what the old typing
established, nothing more, and it can never be produced for a position that did not type-check. This mirrors
`memref.cast`'s contract (matching dynamic dims, element type, layout, address space), which the LLVM lowering
likewise trusts without a runtime check. What the leaf rule does *not* yet check is layout: both sides are
row-major contiguous by construction today; a strided view would need the contract extended.

Out of scope for erasure, to be gated until done:
- GPU transfers (`gpu.alloc_copy`, NVPTX) read the allocation type off the host pointer and get `«⊤; T»`
  (Codex finding 3). Gate `gpu` lowering on literal extents; full support carries the logical size through the
  transfer and linearizes device accesses too.
- `buffer.copy`, `buffer.init`, `buffer.lit` of a runtime extent are whole-array loads/stores and bail; they need
  a fill/copy loop (`lower_map_reduce` already emits one for `lit` when the rank is literal).

## 4. Plan

1. Done on the branch: positional boundary recording by parameter value (finding 2); grouped parameters
   flattened into leaves, so an uncurried function has one binder (the nested-Sigma spelling leaked otherwise);
   shape-polymorphic detection via the inner lam's parameter types (finding 4); erasure casts as `coerce_leaf`
   with η-wrapped continuations (finding 1); GPU transfers of a runtime extent rejected (finding 3).
   Tests: `lit/buffer/exec_dynamic{,_run,_caller,_shape_param}.mim`, `lit/tensor/{dot,unary}_dynamic_run.mim`.
2. Option 2.1 as its own PR, after a second design round that specifies component-path roles, callable summaries,
   representation demand, convergence-time diagnostics and the external ABI rule (see "What the review found
   missing"); `op_args_`, `tensor_fns_`, `shape_poly_`, `leaf_`, `tensor_ty_` are deleted; lit must stay green,
   plus a schedule nest whose loop vector shares a recorded tensor type and a `«4, 4» × «4, l»` product
   (finding 6). `batch_dispatch` through the full pipeline is *not* an acceptance criterion of this step: the
   thunks are direct-style and allocate, so they need the memory-aware CPS conversion of step 3 first.
3. Follow-ups: CPS-convert tensor-returning thunks before bufferization (finding 5); copy/init/lit loops for
   runtime extents; device transfers carrying the logical size and linearized device accesses.

## 5. Open questions

- Should `buf` remain required on externals? Under 2.1 an external's parameter is marked only if a tensor op or
  `buf` consumes it. Keeping `buf` as the explicit seed matches the frontend contract and keeps "which externals
  take buffers" a decision of the frontend, not of an analysis; recommended.
- Partial reads of a *marked* base: still rejected, or lowered to a strided view? Rejection keeps the AlwaysAllocate
  policy honest; views are a separate design.
- The analysis marks per value, but a `Lam` parameter can be marked from one caller and not another. That is a
  type error at the Mim level only if the callee's parameter type cannot be both; with provenance it is an
  inconsistency to report, not to resolve.
