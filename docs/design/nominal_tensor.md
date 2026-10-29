# Nominal, shape-indexed tensors

Nominal tensor representation, public operations and conversion to the buffer layer.


## 0. Summary

At the base revision a tensor is a bare dependent array `«s; T»`. Because types are structural and hash-consed, a shape vector
`«2; Nat»`, a schedule's loop vector `«r; I32»`, a grouped parameter `(n: Nat, a: «n; I32»)` and a real tensor
can share one type, and `tensor.lower_to_mem` has to *guess* which values are tensors from a global set of types
(`tensor_ty_`, with the patches `op_args_`, `record_boundary`, `leaves`, `covers_every_axis`).
Dynamic (runtime-extent) bufferization widened that guess until it broke (a shape argument became a buffer; a
function that only forwards a tensor group was not discovered).

This document specifies replacing the guess by a type: a nominal, shape-indexed `Tensor` type, abstract outside
the `tensor` plugin, so that *a value is a tensor iff its type carries the `Tensor` key*.
Everything that identifies tensors by type set is deleted. The first version supports runtime extents directly,
with a statically known rank. Dependent binders, memory threading and the existing extent-erasure boundary remain;
dynamic copies lower to loops, while raw-array ingress accepts static literals only.

## 1. Goals and non-goals

Goals
- `lower_to_mem` classifies by type again, with no ambiguity and no call-site discovery.
- Shapes stay in the type index: `Tensor 2 (4, n) I32` depends on the runtime `n` exactly as `«4, n; I32»` did,
  so `Tileable`, `if_static`, `refly.check` divisibility and the paper's "shapes are types, extents are terms" are
  untouched.
- All three type parameters are *faithful*, including empty tensors: `Tensor 2 (1, 4) I32 ≠ Tensor 1 4 I32`,
  `Tensor 0 () I32 ≠ I32`, `Tensor 1 0 I32 ≠ Tensor 1 0 I64`.
- Bufferization supports static and runtime extents, including rank-0, singleton and zero-extent tensors;
  the full `lit` suite is green at the integration boundary in section 8.

Non-goals (first version)
- The Python frontend (`mimir-frontend`) is not migrated; section 10 lists what it will have to do.
- The `torch` plugin is not migrated.
- Runtime rank is deferred: loop generation still requires a literal rank. An extent may be a runtime term;
  a non-literal extent is not itself a reason to reject a tensor or its function boundary.
- Reuse btensor, affine, mem and ll, and preserve the `buffer.Buf` ABI and pointer-erasure boundary.
  `buffer.lower_ptr` needs the precise/erased boundary correction for literal-extent-0/1 results in §5.2;
  the experiment below shows that leaving its coercions unchanged is insufficient.
- No new analysis. If something cannot be classified by type, it is a type error or an explicit gate, not a guess.

## 2. The type

```mim
/// tensor.mim
priv nom Tensor (r: Nat) (s: «r; Nat») (T: *) = [indices: «(r, s, ⊥:T)», data: «s; T»];
pub anx let Tensor = Tensor;
```

- `nom I ndom+ = e` is sugar for `lam I ndom+ = nom I = e; I` (langref, "Nominal Newtypes"): one declaration key,
  one instance per application, instances with definitionally equal indices are the *same* `Nom` (the key and the
  normalized representation are what hash-consing compares; substitution keeps the key).
- `indices` is a singleton witness of `(r, s, ⊥:T)`. A shape-only witness is insufficient: `«0; T» ≡ []`, so
  `[«s», «s; T»]` would identify `Tensor 1 0 I32` with `Tensor 1 0 I64`. The complete witness preserves rank,
  shape and element type even when the payload folds away. It also keeps `(1, 4)` distinct from `4` and a
  rank-0 tensor distinct from its scalar element.
- `⊥:T` is a typed marker inside the singleton, never a data element or an evaluated operand. Its type records
  `T` while keeping `Tensor r s T : *`; using `«T»` directly would raise the witness's universe. The witness
  has the unique value `‹(r, s, ⊥:T)›` and carries no runtime payload. Bufferization consumes it as type
  metadata; `compile.single_erasure` removes any remaining singleton representation before codegen (§7).
- `priv` confines `inj` and `#` to `tensor.mim`; `pub anx let Tensor = Tensor` exports the *abstract* type and
  registers the constructor as an annex for C++ lowering. Outside the plugin nobody can build or open a tensor
  except through the functions of §3. This is the whole point: there is
  no "escaping unwrap" because there is no unwrap.
- Scalars are not tensors: `Tensor 0 () T` and `T` are distinct. See §6 for what that changes.

Verify first (experiment E1, §8): that the checker infers `r`, `s`, `T`, including from an empty tensor, at an
implicit-argument position (`unary f t` must find `r s Ti` from `t`), and that a `Pi` dom
`[n: Nat, a: Tensor 2 (4, n) I32]` type-checks and reduces as the array version did. E4 also checks runtime
execution through these dependent boundaries, including callers that instantiate an extent to 0 or 1.

## 3. The public surface of the plugin

All element-level access becomes an operation.

```mim
axm from_array: {r: Nat, s: «r; Nat», T: *} → [a: «s; T»] → Tensor r s T;
axm get: {r: Nat, s: «r; Nat», T: *}
       → [t: Tensor r s T, i: «j: r; Idx (s#j)»] → T, normalize_get;
axm set: {r: Nat, s: «r; Nat», T: *}
       → [t: Tensor r s T, i: «j: r; Idx (s#j)», v: T] → Tensor r s T, normalize_set;
axm scalar: {T: *} → T → Tensor 0 () T, normalize_scalar;
axm item: {T: *} → Tensor 0 () T → T, normalize_item;
pub lam shape {r: Nat, s: «r; Nat», T: *} (t: Tensor r s T): «r; Nat» = s;
```

- `from_array` and `scalar` construct the private witness/payload logically; `get` and `item`
  read that payload, and `set` returns an updated value. All five production operations are axioms, so they
  remain recognizable applications until `lower_to_mem` handles them (§5). Normalizers
  may fold compositions such as `get (from_array a, i)` without introducing an escaping `Name`/`Struc`;
  preserve every surviving boundary/data operation as an axiom application. The E1 experiment below confirms
  that both a default `lam` and an `@ff` accessor can disappear before bufferization. `shape` should inline.
- Array folding cannot recover logical singleton axes at `from_array`: write `from_array @(r, s, T) a` when
  necessary, rather than expecting `«s; T»` alone to determine the intended logical shape.
- `tensor.buf` is deleted. Its job (declare that a parameter is a tensor) is now the parameter's type.
- `axm shape` and `normalize_shape` are deleted.

## 4. Operation signatures

Rule: every *data* position `«s; T»` becomes `Tensor r s T`, where `r` and `s` are the implicit parameters already
present in the signature (every op has them); metadata (shape vectors, bounds, permutations, index maps,
schedules, combiners, element-level callbacks) is untouched. Element types `T` stay element types.

The data operations (24): `generate`, `splat`, `map_reduce_post`, `broadcast`, `broadcast_in_dim`, `concat`,
`flip`, `pad`, `repeat`, `reshape`, `slice`, `transpose`, `binary`, `map`, `select`, `unary`, `bmm`, `conv`,
`conv_grouped`, `dot_product`, `pool`, `product_2d`, `gather`, `scatter`.
Not data operations (unchanged): `fastest_axis`, `if_static`, `vec_width`, the phases.

Examples (before → after):

```mim
axm splat: {T: *, r: Nat} → [s_out: «r; Nat», value: T] → «s_out; T»;
axm splat: {T: *, r: Nat} → [s_out: «r; Nat», value: T] → Tensor r s_out T;

axm unary: {Ti To: *} → [app: Ti → To] → {r: Nat, s: «r; Nat»} → [i: «s; Ti»] → «s; To»;
axm unary: {Ti To: *} → [app: Ti → To] → {r: Nat, s: «r; Nat»} → [i: Tensor r s Ti] → Tensor r s To;

axm transpose: {T: *, r: Nat} → [permutation: «r; Idx r»] → {s: «r; Nat»} → [input: «s; T»]
             → «transpose_shape (s, permutation); T», normalize_transpose;
axm transpose: {T: *, r: Nat} → [permutation: «r; Idx r»] → {s: «r; Nat»} → [input: Tensor r s T]
             → Tensor r (transpose_shape (s, permutation)) T, normalize_transpose;

axm pad: … → [input: «s_in; T», value: T] → «pad_shape (lo, s_in, hi); T», normalize_pad;
axm pad: … → [input: Tensor r s_in T, value: T] → Tensor r (pad_shape (lo, s_in, hi)) T, normalize_pad;

axm map_reduce_post: … → [is: «i: nis; «Sis#i; Tis#i»», post_is: «j: nps; «Sps#j; Tps#j»»] → «So; To»;
axm map_reduce_post: … → [is: «i: nis; Tensor (Ris#i) (Sis#i) (Tis#i)», post_is: «j: nps; Tensor (Rps#j) (Sps#j) (Tps#j)»]
                     → Tensor Ro So To;
```

Note that `is` is still an ordinary array - of tensors. `lower_to_mem` must convert *arrays and tuples of*
tensors (today `mentions_tensor` deliberately stops at `Arr`).

`reshape` changes rank: its result is `Tensor r_out s_out T`. `reshape_shape`-style helpers that compute shapes
are unchanged.

Normalizers that today fold on *type equality* of folded arrays (`normalize_repeat` identity when
`s_in == s_out`, `normalize_reshape` identity, `normalize_pad` identity with zero padding) keep working, because
the comparison is on the shape values `s`, which are now faithful; check each one (E2).

## 5. The `_impl`s and the C++ phases

`_impl`s (18, all `priv anx lam … _impl` in `tensor.mim`): change signatures by the same rule; bodies compose
operations and need no algorithmic change, except
- `gather_pointwise_elem_impl` and `scatter_step_impl`, which read two tensor operands elementwise: use `get`;
  `scatter_step_impl` updates: use `set` (rewrap).
- `mr_tileable` (`Tileable` producer): `out_t (ro, so) = Tensor ro so To`; the `Tileable` record itself is
  unchanged (`Out` is already abstract), as are `strip_mine_par/red`, `vectorize`, `unroll_trailing`,
  `interchange`, `dot_schedule*`, `conv_schedule`, `pool_schedule`: they manipulate `So/Sr/oax/sched`, never data.

`lower_tensor` / `phase/lower.cpp`
- `lower_via_impl` re-applies curry groups and does not inspect the representation: unchanged if the curry
  layout of each op is preserved (it is, only types change).
- Its read-through of extracts (`Axm::peel<tensor::buf>` at lower.cpp:58, `fastest_axis` input) must recognize
  `get` instead of `Extract`, and `buf` is gone.

`phase/fuse.cpp`
- `injective_coord`/`reads_injectively` analyze *coordinate* vectors, not tensor data: unchanged.
- The two `isa<Arr>` gates in the read-through (fuse.cpp ≈128, ≈138) become "is `Tensor`".
- Fusion matches tensor op applications (`tensor.h:51`); element reads it may see are `get` applications.

`normalizers.cpp`
- `normalize_shape` is deleted (§3). `normalize_fastest_axis` works on pure reads and coordinate maps; its
  recognition of a tensor input changes from `Arr` to `Tensor`.

`phase/lower_map_reduce.cpp` (value-world lowering, used by the custom pipelines in lit)
- Four `w.extract` and two `w.insert` sites on tensor values become `get`/`set` or, once bufferized, read/write.
  The scalar shortcuts (rank-0 result is the element) become `scalar`/`item` (§6).

`phase/lower_to_mem.cpp` - the point of the exercise
- Classification: `is_tensor_type(t)` ≡ `t` is a `Nom` whose key is `Tensor`'s. Obtain the key once per phase
  by applying the annex to its three curry groups, for example
  `w.app(w.app(w.app(w.annex<tensor::Tensor>(), r), s), T)`, and reading `->as<Nom>()->key()` (verify in E1).
  Decode `(r, s, T)` from `indices`' singleton operand: the third marker's type is `T`. Do not reconstruct
  them from `data` or the normalized `Buf` type, which can have lost empty payloads or singleton axes.
- Delete: `tensor_ty_`, `add_tensor_ty`, `op_args_`, `record_boundary`/`record_boundary_ty`, `collect_leaves`'s
  role test (replace `mentions_tensor` with the structural predicates in §5.1), the
  `tensor.buf` case, `splat_scalar`'s type test, `covers_every_axis` (a partial read is impossible: `get` takes a
  full index by its type).
- Keep: `uncurry` and the flattening of grouped parameters (dependent binders are a different problem), the
  fresh-memory bookkeeping, `materialize` (now: a `from_array`/`scalar` *application* materializes;
  a value of type `Tensor` is always already a buffer or a parameter), `lower_call`/`explode_arg` (by type).
- Rewrite rules:
  - type `Tensor r s T` ↦ `buffer.Buf (r, s, T)` everywhere a type is rewritten (`conv_leaf`, `conv_boundary`,
    continuation doms, arrays/tuples of tensors component-wise);
  - `from_array a` accepts only static shapes and literal array values. Uniform literals lower to
    `buffer.lit`; other literals lower to `buffer.init`. There is no dynamic Pack materializer;
  - `scalar x` ↦ `buffer.init (0, (), T)`; `item t` ↦ `buffer.read (t, ())`;
  - `get (t, i)` ↦ `buffer.read`; `set (t, i, v)` ↦ alloc + copy + `buffer.write` (today's AlwaysAllocate).
    Static copies may use `buffer.copy`; runtime-sized copies must use the element loop in §5.2. Use the
    logical `(r, s)` for indexing, explicitly
    folding literal size-1 coordinates when calling a buffer helper that derives its rank from `Buf`;
  - `shape t` has normalized to `s` before this phase (it is a `lam`); nothing to do;
  - do not globally reject `Name` (`inj`) or `Struc` (`#`). Only a residual `Name` whose nominal type has the
    `Tensor` key, or a `Struc` whose operand has that type, signals an escaped tensor representation; diagnose
    that case before rebuilding it over a `Buf`. Other nominal keys use the generic rewriter. A user newtype
    or an `option.Opt` unrelated to tensors must coexist with tensor operations.

Existing btensor, affine, mem and ll operations are reused. `btensor.map_reduce_post` already takes `buffer.Buf`;
the required `buffer.lower_ptr` boundary correction is described in §5.2.

### 5.1 Function detection and conversion eligibility

Tensor classification and permission to change a function's ABI are separate questions. Use these predicates
on types after substituting the function's actual binders; none depends on whether a tensor op consumes a value:

- `contains_tensor_value(t)` recognizes a `Tensor` key, or a tensor component through `Sigma` and `Arr`.
  It stops at `Pi`: a callable carrying a tensor contract is not itself a tensor buffer.
- `has_tensor_contract(pi)` checks both the domain and codomain of a `Pi`. Its type walk recognizes `Tensor`
  and descends through `Sigma` components, `Arr` element types, nested `Pi` domains/codomains, other nominal
  representations and variant alternatives, using a visited set for recursive types. Stop at `Tensor` without
  traversing its witness; do not search arbitrary term dependencies. This includes tensor parameters, a CPS
  return continuation taking a tensor, and direct-style results, including `[] → Tensor r s T`. Finding a
  tensor inside an unsupported carrier is still a positive detection followed by a conversion diagnostic.
- `is_tensor_fn(lam)` means `has_tensor_contract(lam->type())`, regardless of external/internal status,
  whether the body is set, or whether the lambda is an inner curry group. It is only a detection predicate.

Walk all reachable lambdas, including local functions and continuation arguments, rather than discovering
functions from tensor-op uses or only from external roots' immediate parameter lists. A lambda with no tensor
contract can still contain tensor operations or capture a tensor; its body also needs conversion and effect
checks. A shape curry group ending in CPS must also be uncurried when that inner body uses tensor memory
operations, even if every parameter and result is scalar. Recognized tensor primitives and their private reference bodies follow §3's lowering contract instead
of being treated as ordinary user functions.

First-version eligibility and diagnostics:

| Function form remaining at bufferization | Handling |
|---|---|
| Defined first-order CPS function or local continuation, including one that only forwards tensors | Convert tensor-valued parameters, aggregates and supported return-continuation domains; retain fresh-memory bookkeeping and run `mem.add_mem`. |
| Fully applied shape curry groups ending in a defined CPS function | Uncurry and flatten dependent groups, preserving runtime extent terms and their binders. A shape-family lambda is not an ordinary direct-style tensor-returning thunk. |
| Direct-style function taking or returning a tensor, a tensor-returning thunk, or a residual direct-style helper requiring tensor memory effects | Require prior inlining/specialization or memory-aware CPS conversion; otherwise diagnose. A pure signature test cannot supply that conversion. |
| Unset external with a tensor contract | Diagnose until an explicit import/buffer ABI contract exists; never silently change a foreign ABI. |
| Ordinary tensor-processing function passed as a value, indirect/selected tensor callee, or unsupported higher-order callable nesting | Diagnose until callable adaptation is implemented. Supported CPS return continuations and local continuation arguments are converted under their expected continuation types. |

`Sigma` and `Arr` are the supported tensor aggregate carriers. A foreign nominal or variant containing a
tensor requires its own conversion contract; detect and diagnose that unsupported carrier, without rejecting
unrelated nominal values. Type inspection must terminate on recursive types. Captures and supported element-level
callbacks follow their memory-threading contract; there is no blanket `op_args_` exemption for a callback body.
Effect detection conservatively follows referenced local CPS functions as well as local continuations.
An inline direct-style `generate` callback may read existing tensors with `get`/`item`, but
allocating tensor operations require CPS conversion and are explicitly rejected before rewriting.

Run shape and callable eligibility checks before rewriting affected functions. Runtime extents are permitted;
static shape does not make an unsupported function form eligible. `uncurry` and dependent-binder reconstruction
remain needed where applicable; the nominal key only removes role guessing, not ABI or effect conversion.

### 5.2 Runtime extents and the erasure boundary

The first version must handle runtime extents through tensor parameters/results, `generate`, `splat`, the
map-reduce path (including `unary` and dot products), `get` and `set`. Retain existing explicit per-operation
restrictions on ranks, axes and index-map construction; document and test any remaining unsupported extent
case individually. `concat` still requires a literal axis and literal extents along that axis; the negative
regression is `lit/error/tensor_concat_dynamic_axis_extent.mim`. Gather/scatter retain their existing
static bounds proofs; unresolved obligations have explicit diagnostics and regressions in
`lit/error/tensor_{gather,scatter}_symbolic_bounds.mim`. There is no blanket literal-extent requirement or fallback to array-role guessing.

- Keep the complete logical `(r, s, T)` available to conversion and dependent calls. Rewrite every extent
  under the fresh binder, including grouped shape parameters and continuation domains. The witness is type
  metadata, not a runtime shape descriptor. Runtime extent terms must remain in scope alongside the handle
  wherever allocation, indexing or copy loops use them.
- The native boundary currently passes extents as scalar arguments beside data pointers; it has no public
  tensor descriptor or generated entry wrapper that reads shape metadata from input tensors. A returned
  pointer carries no shape metadata; callers must know or separately receive the result extents.
  `lit/tensor/mlp_dynamic.mim` exercises this explicit-extent ABI with `Linear → ReLU → Linear`, keeping
  batch, input width, hidden width and output width unresolved until execution. One compiled function passes
  56 shape combinations against an independent scalar reference, including zero and singleton extents.
- `buffer.lower_ptr` remains the erasure point: runtime-sized `Buf` handles become flat pointers, and accesses
  linearize using logical shape terms. Preserve argument coercions and continuation adapters at
  precise/erased boundaries, including a polymorphic callee instantiated at a literal extent of 0 or 1.
  Extend that adaptation to scalar precise pointers (`Ptr (T, 0)`, after size-1 axes fold) and empty precise
  pointers (`Ptr ([], 0)`, after a zero extent folds away the payload). The current array-pointer-only leaf
  rule fails both cases for a non-inlined dynamic producer returning to a precise continuation. Preserve or
  recover the old `Buf` element type and shape contract when adapting an empty pointer; it cannot be inferred
  from `[]`. Retain the proof from the pre-erasure typing position rather than permitting arbitrary pointer casts.
- Dynamic `set` allocates a fresh destination, copies every element, then writes the selected element, leaving
  the source unchanged. Emit the copy as a destination-passing `btensor.map_reduce_post` with identity maps,
  an identity element combiner and no reduction axes, or equivalent memory-threaded affine loops. Thread the
  final copy memory into the write. Do not emit runtime-sized `buffer.copy`: its whole-array load/store path
  rejects runtime extents. Empty domains execute no copy iterations. Give allocating operations distinct
  fresh-memory requests, scoped and memoized per enclosing lambda; using the same placeholder for two
  otherwise identical allocations would hash-cons them into one buffer.
- `from_array` is a static-literal ingress operation. The bufferization preflight rejects non-literal ranks/extents and
  non-literal payloads after specialization, including opaque static-array parameters. Uniform literals use `buffer.lit`; other
  literals use `buffer.init`. Dynamic tensor parameters/results use the `Tensor` contract, and dynamic tensor
  construction uses `generate` or `splat`. Dynamic Pack materialization and raw-array export are absent.
- Runtime rank and runtime-sized device transfers remain gated. Device support must carry logical sizes to
  transfer/allocation operations and linearize device accesses before that gate can be lifted.

These loops belong in tensor conversion and use the existing btensor/buffer/affine operations, keeping the
lower-layer ABI and erasure point. Runtime copy and the precise/erased boundary correction are required implementation work in this version, not follow-ups.

If an extent is already a literal zero, return the empty allocated buffer before constructing element-access
loop bodies. The shared pointwise helper and map-reduce readers also avoid unreachable accesses; an empty
reduction still writes its initial accumulator through the epilogue. Constant padding of an empty input
produces the fill value, including when an extent becomes zero at runtime; replicate padding of a statically
empty input is diagnosed. Static nonempty pad inputs use clamped reads and select; runtime input extents
use conditional reads because they may be zero. Concat skips empty input segments. A map-reduce reader with a nonempty iteration
domain and a statically empty input (including an epilogue input) is diagnosed instead of emitting a read.
Broadcast similarly rejects a statically empty input unless the output is already known to be empty.
A zero-trip loop alone is insufficient: lowering an unreachable access through the normalized empty-array
payload can fail type checking before dead-code removal. A runtime extent that becomes zero at
execution instead follows the normal loop bounds.

## 6. Scalars and singleton tensors

At the base revision, `«1; T» ≡ T`, and several paths rely on it: `lower_to_mem` loads a rank-0 result as a scalar
(≈ lines 71, 759, 778, 810 of the current file), `lower_splat`/`lower_generate` short-cut fully-folded shapes,
`lower_broadcast` has a scalar-fill path, fusion gates on it, and the frontend keeps singleton dims out of the
type. With the witness none of this folding happens: `Tensor 1 1 I32` is a one-element tensor, `Tensor 0 () T`
a rank-0 tensor, neither is `T`.

Policy for the first version: a tensor is a tensor. Rank-0 and singleton tensors bufferize like any other
(a one-element buffer). `scalar`/`item` are the explicit bridge. The shortcuts above are deleted, not ported.
If a schedule selector relied on `«1; T»` folding (audit around `tensor.mim` ≈1010, the "size-1 axes fold away"
remarks in `btensor.mim`), replace the reliance by an explicit check on the shape value.

## 7. Pipeline order

`opt.mim`: `tensor.lower_tensor` → `tensor.fuse_tensor` → `tensor.lower_to_mem` → `mem.add_mem` → `btensor.…` →
`buffer.lower_ptr` → … → clos → `compile.nom_erasure` → `compile.single_erasure` → … → `ll.emit`.
`nom_erasure` and `single_erasure` run *after* bufferization, so the `Tensor` key and the witness are visible
to `lower_to_mem` and gone before codegen. Do not move them. Custom `_compile` pipelines in lit that stop before
`lower_to_mem` print `Tensor` types in their dumps; their FileCheck lines change accordingly.

## 8. Phases and integration checks

Nominal typing, static-literal ingress and tensor-to-buffer conversion are validated independently of
runtime pointer erasure. Tests that stop after `lower_to_mem` check dynamic shape binders, grouped parameters,
shape-only curry groups and tensor-free signatures whose local continuations use tensors. Static execution
coverage runs through the default pipeline; native dynamic execution belongs to the pointer-erasure stage.

The checkpoints below record the intended validation scope.

E1 (no production migration): lit probes declaring the type of §2 and the value semantics of §3, with
- a function `{n: Nat} (a: Tensor 2 (4, n) I32)` applying `unary`-shaped and `binary`-shaped lams whose implicit
  `r s T` must be inferred from `a` (type checking only; no runtime-extent bufferization);
- `Tensor 2 (1, 4) I32` vs `Tensor 1 4 I32` and `Tensor 0 () I32` vs `I32` must be *rejected* as unequal;
- `Tensor 1 0 I32` vs `Tensor 1 0 I64` and `Tensor 2 (4, 0) I32` vs `Tensor 2 (4, 0) I64` must be rejected;
  inference must recover the element type of an empty tensor, and `Tensor r s T : *` must hold;
- `from_array` then `get` normalizes to the element; `set` then `get` likewise. Supply explicit logical indices
  at array ingress for singleton shapes; verify rank-0 `scalar`/`item` too;
- axiom forms for all five boundary/data operations surviving the default optimization prefix, `lower_tensor`
  and fusion, with normalizers allowing the value-level folds above;
- the annex/key recipe of §5 from a tiny C++ unit test (`test/`), including decoding all three indices.
If E1 fails on parameter fidelity, universe, inference or operation preservation, stop and report; the rest
depends on it. The existing full `lit` suite remains green at E1.

E2 (1–2 days): `tensor.mim` - type, §3 surface, 24 signatures, 18 `_impl`s, delete `buf`/`shape` axiom.
The C++ side will not compile tests yet; the goal is `mim --output-h`/parsing of the plugin and the `tensor`
dumps of lit files that stop before lowering. Update those FileCheck lines.

E3 (2–3 days): `lower_to_mem` per §5, then `lower.cpp`, `fuse.cpp`, `normalizers.cpp`, `lower_map_reduce.cpp`.
Include dynamic `set`'s copy loop and static-literal ingress from §5.2. Target:
`lit/tensor` and `lit/buffer` green except the files that still spell `«…; T»` for tensors.

E4 (1–2 days): migrate the lit files: tensor-typed parameters and results to `Tensor r s T`, literal inputs via
`from_array`, probes via `get`/`item` (≈ 79 files, 39 element-access sites by the last count). Add regressions
for nested dependent groups with runtime extents, forwarding-only functions, shape metadata beside a
same-structured tensor, rank-0, singleton axes, empty tensors and unrelated nominal values.
Add the callable gates of §5.1, especially a return-only thunk with no tensor parameter.

Tests that use runtime extents through `tensor.lower_to_mem` (`lit/buffer/exec_dynamic*`, `dot_dynamic_run`,
`unary_dynamic_run`) must retain their executable behavior after the API migration. In particular,
`exec_dynamic_shape_param` must pass without positional boundary logic. Add execution tests for dynamic
`set` checking both the unchanged source and updated destination, rejection of dynamic-array ingress,
runtime extents of 0 and 1, and precise/erased continuation adaptation. The latter must include a non-trivial
dynamic producer called with literal 0 and 1; an identity forwarder that inlines before erasure is insufficient.
Keep buffer-only dynamic tests for the
existing `buffer.lower_ptr` erasure path. Negative tests cover runtime rank, dynamic-array ingress,
non-literal static-array ingress and runtime-sized device transfers; they must not replace supported dynamic execution tests.

E2–E4 are intermediate steps within one integration change: plugin parsing and migrated subsets are their local
checks, not a claim that every intermediate commit builds or has a green full suite. Complete E4 with all required
build targets and the full `lit` suite green before integrating.

E5 (half a day): delete dead code and obsolete `buf` references in current docs; retain the companion note's
historical problem statement and erasure discussion with a link to this superseding specification.

Estimate: the original 6–9 working days on the compiler side is provisional and does not budget the additional
runtime copy work separately. Re-estimate after E1 with §5.2 and its execution tests included;
frontend migration, runtime rank and dynamic device transfers are excluded.

## 9. Acceptance

- Full `lit` green with static and dynamic execution cases and explicit unsupported-operation/callable diagnostics.
- `grep -n 'tensor_ty_\|op_args_\|record_boundary\|covers_every_axis\|tensor::buf' src/` is empty.
- `lower_to_mem.cpp` has no code path that decides tensor-ness from an `Arr` type.
- Classification regressions are covered: nested dependent groups have no role test in `collect_leaves`,
  forwarding-only functions and tensor-returning thunks are detected under §5.1, extent-1 tensors do not fold,
  and zero-extent tensors retain their element type. Detection and successful conversion are asserted separately.
- `Name`/`Struc` handling is keyed to `Tensor`; unrelated newtypes and option values coexist with tensor code.
- Runtime extents work through dependent inputs, results and supported continuations; dynamic `set` preserves
  value semantics, and dynamic-array ingress is explicitly rejected. No runtime-sized whole-array `copy` or
  `init` reaches `buffer.lower_ptr`. Rank remains literal, and unsupported exports/transfers have specific gates.
- Reuse btensor, affine, mem and ll without changing their interfaces. Limit buffer-layer changes to the
  required precise/erased boundary correction and its regressions.

## 10. Frontend (deferred, for the record)

The frontend must construct `Tensor r s T` through the annex, pass external data through Tensor-typed
parameters, and use `get`/`item` for element access. `from_array` is reserved for static literals. Shape queries
read the nominal indices, so singleton dimensions no longer need a separate frontend convention. The `torch`
dialect is a separate migration.

## 11. Risks

- Implicit inference from a nominal instance (E1). Mitigation: if inference needs the indices as explicit
  arguments in some ops, write them explicitly in the `_impl`s; the user-facing ops keep inferring from `s`.
- Losing an index through representation folding, especially the dtype of an empty tensor. Mitigation: the
  complete singleton witness, its universe/inference checks and negative type-equality tests in E1.
- Inlining of §3's operations before `lower_to_mem` (the recognition points). Mitigation in §3 and E1.
- The generic rewriter rebuilding tensor `Name`/`Struc` over a `Buf` (`rewrite.cpp` ≈118). Mitigation: key-specific
  handling in §5; never apply that diagnostic to all nominal types.
- Treating `is_tensor_fn` as ABI/effect eligibility. Mitigation: separate detection, conversion and explicit
  unsupported-callable diagnostics in §5.1; dynamic-shape support follows §5.2.
- Dynamic copies accidentally reaching whole-array lowering. Mitigation: the loop rules
  in §5.2 and execution tests that keep runtime extents unresolved until the compiled program runs.
- Literal-extent-0/1 result adapters losing the element type or expecting every precise pointer to contain an
  array. Mitigation: the explicit lower_ptr correction and non-inlined producer regressions in §5.2 and E4.
- Scheduling code that silently relied on singleton folding (§6). Mitigation: audit `tensor.mim` ≈1010 and
  the `btensor.mim` remarks; the lit suite covers the selectors.
- Dump churn in lit: large but mechanical; do it per directory, not per file.

## 12. Conventions for the implementer

- Follow `.github/copilot-instructions.md`: comments only for a non-obvious invariant, one line; `.clang-format`
  via `git clang-format`; Mim style per `.github/instructions/mim-style.instructions.md`.
- Build: `cmake --build build --target mim_all mim_lit_tests`; run `lit build/lit -q`. This Linux validation
  enables `MIM_BUILD_LL_RUNTIME` and uses Python multiprocessing `fork` because the sandbox disallows
  the default forkserver socket. macOS builds may require `DEVELOPER_DIR=/Library/Developer/CommandLineTools`.
- Do not touch the frontend repository.
- Apply the validation checkpoints in §8. If a required checkpoint fails, report the exact failing lit test and
  diagnostic rather than adding a type-based heuristic back; E2–E4 have an explicit intermediate migration window.
