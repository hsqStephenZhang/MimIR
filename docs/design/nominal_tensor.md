# Nominal, shape-indexed tensors

Status: implementation spec for a first version, 2026-10-28.
Base: branch `zc/feat/erase-dynamic-extents` (HEAD 047a543b57, on top of origin/master b276ad719a).
Companion: `docs/design/bufferize_roles.md` (the problem statement; this document replaces its section 2.1 plan).

## 0. Summary

Today a tensor is a bare dependent array `«s; T»`. Because types are structural and hash-consed, a shape vector
`«2; Nat»`, a schedule's loop vector `«r; I32»`, a grouped parameter `(n: Nat, a: «n; I32»)` and a real tensor
can share one type, and `tensor.lower_to_mem` has to *guess* which values are tensors from a global set of types
(`tensor_ty_`, with the patches `op_args_`, `record_boundary`, `leaves`, `covers_every_axis`).
Dynamic (runtime-extent) bufferization widened that guess until it broke (a shape argument became a buffer; a
function that only forwards a tensor group was not discovered).

This document specifies replacing the guess by a type: a nominal, shape-indexed `Tensor` type, abstract outside
the `tensor` plugin, so that *a value is a tensor iff its type carries the `Tensor` key*.
Everything that identifies tensors by type set is deleted; everything about dependent binders, memory threading
and extent erasure stays.

## 1. Goals and non-goals

Goals
- `lower_to_mem` classifies by type again, with no ambiguity and no call-site discovery.
- Shapes stay in the type index: `Tensor 2 (4, n) I32` depends on the runtime `n` exactly as `«4, n; I32»` did,
  so `Tileable`, `if_static`, `refly.check` divisibility and the paper's "shapes are types, extents are terms" are
  untouched.
- The logical rank and shape are *faithful*: `Tensor 2 (1, 4) I32 ≠ Tensor 1 4 I32`, `Tensor 0 () I32 ≠ I32`.
- `lit` stays green at the end of each phase listed in section 8.

Non-goals (first version)
- The Python frontend (`mimir-frontend`) is not migrated; section 10 lists what it will have to do.
- The `torch` plugin is not migrated.
- Nothing changes below `buffer.Buf`: btensor, `buffer.lower_ptr` (erasure), affine, mem, ll stay as they are.
- No new analysis. If something cannot be classified by type, it is a type error or an explicit gate, not a guess.

## 2. The type

```mim
/// tensor.mim
priv nom Tensor (r: Nat) (s: «r; Nat») (T: *) = [wit: «s», data: «s; T»];
pub let Tensor = Tensor;
```

- `nom I ndom+ = e` is sugar for `lam I ndom+ = nom I = e; I` (langref, "Nominal Newtypes"): one declaration key,
  one instance per application, instances with definitionally equal indices are the *same* `Nom` (the key and the
  normalized representation are what hash-consing compares; substitution keeps the key).
- `wit: «s»` is the singleton type whose sole inhabitant is `s`. It is what makes the index faithful: the bare
  `«s; T»` normalizes `(1, 4)` to `4` and `«1; T»` to `T`, so without it `Tensor 2 (1, 4) I32 ≡ Tensor 1 4 I32`
  and `Tensor 1 1 I32 ≡ Tensor 0 () I32`. `«(1, 4)»` and `«4»` are different types. The witness costs nothing at
  runtime: `compile.single_erasure` removes singletons before codegen (it runs after `lower_to_mem`, see §7).
- `priv` confines `inj` and `#` to `tensor.mim`; `pub let Tensor = Tensor` exports the *abstract* type. Outside the
  plugin nobody can build or open a tensor except through the functions of §3. This is the whole point: there is
  no "escaping unwrap" because there is no unwrap.
- Scalars are not tensors: `Tensor 0 () T` and `T` are distinct. See §6 for what that changes.

Verify first (experiment E1, §8): that the checker infers `r`, `s`, `T` from a value of type `Tensor r s T` at an
implicit-argument position (`unary f t` must find `r s Ti` from `t`), and that a `Pi` dom
`[n: Nat, a: Tensor 2 (4, n) I32]` type-checks and reduces as the array version did.

## 3. The public surface of the plugin

All element-level access becomes an operation. Names are proposals; keep them short.

```mim
/// Enter the tensor world: a value array becomes a tensor of the same shape.
pub lam from_array {r: Nat, s: «r; Nat», T: *} (a: «s; T»): Tensor r s T = (‹s›, a) inj Tensor;
/// Leave it: the value array. A runtime extent has no value array; `lower_to_mem` rejects it (§5).
pub lam to_array   {r: Nat, s: «r; Nat», T: *} (t: Tensor r s T): «s; T» = (#t)#data;
/// Element access, the former `t#i` / `t#i ← v`.
pub lam get {r: Nat, s: «r; Nat», T: *} (t: Tensor r s T, i: «j: r; Idx (s#j)»): T = (#t)#data#i;
pub lam set {r: Nat, s: «r; Nat», T: *} (t: Tensor r s T, i: «j: r; Idx (s#j)», v: T): Tensor r s T =
    (‹s›, (#t)#data#i ← v) inj Tensor;
/// The logical shape is the index; no normalizer.
pub lam shape {r: Nat, s: «r; Nat», T: *} (t: Tensor r s T): «r; Nat» = s;
/// Scalars.
pub lam scalar {T: *} (x: T): Tensor 0 () T = (‹()›, x) inj Tensor;
pub lam item   {T: *} (t: Tensor 0 () T): T = (#t)#data;
```

- `get`/`set`/`to_array`/`item` are `lam`s whose bodies contain the only `#` on tensors in the whole code base.
  `lower_tensor` must *not* inline them before `lower_to_mem` has seen them (they are the recognition points,
  §5); mark them `@ff` or make them axioms with normalizers if inlining cannot be controlled - decide in E1.
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
  by instantiating the annex (`w.app(w.annex<tensor::Tensor>(), {r, s, T})` on any arguments) and reading
  `->as<Nom>()->key()`; if that turns out not to work (E1), register the key from the plugin entry instead.
- Delete: `tensor_ty_`, `add_tensor_ty`, `op_args_`, `record_boundary`/`record_boundary_ty`, `collect_leaves`'s
  role test (`mentions_tensor` becomes "contains a `Tensor` type, through Sigma, Arr and Cn doms"), the
  `tensor.buf` case, `splat_scalar`'s type test, `covers_every_axis` (a partial read is impossible: `get` takes a
  full index by its type).
- Keep: `uncurry` and the flattening of grouped parameters (dependent binders are a different problem), the
  fresh-memory bookkeeping, `materialize` (now: a `from_array`/`scalar` *application* materializes;
  a value of type `Tensor` is always already a buffer or a parameter), `lower_call`/`explode_arg` (by type).
- Rewrite rules:
  - type `Tensor r s T` ↦ `buffer.Buf (r, s, T)` everywhere a type is rewritten (`conv_leaf`, `conv_boundary`,
    continuation doms, arrays/tuples of tensors component-wise);
  - `from_array a` ↦ `buffer.init` (or `buffer.lit` for a splat literal);
  - `scalar x` ↦ `buffer.init (0, (), T)`; `item t` ↦ `buffer.read (t, ())`;
  - `get (t, i)` ↦ `buffer.read`; `set (t, i, v)` ↦ alloc + copy + `buffer.write` (today's AlwaysAllocate);
  - `to_array t` ↦ `buffer.read` of the whole array when every extent is a literal; bail with
    "a runtime extent has no value array" otherwise;
  - `shape t` has normalized to `s` before this phase (it is a `lam`); nothing to do;
  - a `Name` (`inj`) or `Struc` (`#`) node reaching this phase is an error: the only ones exist inside the
    bodies of §3's lams, which this phase handles as applications. Override `rewrite_imm_Name`/`Struc` to bail
    with a clear message rather than letting the generic rewriter rebuild a `Struc` over a `Buf`.
- `is_tensor_fn(lam)`: some parameter type contains `Tensor` (by type, exact; no `inner` special case needed for
  detection, but `uncurry` still handles the curried `{n}` form).

`btensor`, `buffer`, `affine`, `mem`, `ll`: unchanged. `btensor.map_reduce_post` already takes `buffer.Buf`.

## 6. Scalars and singleton tensors

Today `«1; T» ≡ T`, and several paths rely on it: `lower_to_mem` loads a rank-0 result as a scalar
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

## 8. Phases (each ends with a green `lit`)

E1 (half a day, no production change): a lit file declaring the type of §2 and the lams of §3, with
- a function `{n: Nat} (a: Tensor 2 (4, n) I32)` applying `unary`-shaped and `binary`-shaped lams whose implicit
  `r s T` must be inferred from `a`;
- `Tensor 2 (1, 4) I32` vs `Tensor 1 4 I32` and `Tensor 0 () I32` vs `I32` must be *rejected* as unequal;
- `from_array` then `get` normalizes to the element (`#((‹s›, a) inj Tensor)` ↦ `(‹s›, a)`), `set` then `get`
  likewise;
- the key recipe of §5 from a tiny C++ unit test (`test/`), or the fallback.
If E1 fails on inference, stop and report; the rest depends on it.

E2 (1–2 days): `tensor.mim` only - type, §3 surface, 24 signatures, 18 `_impl`s, delete `buf`/`shape` axiom.
The C++ side will not compile tests yet; the goal is `mim --output-h`/parsing of the plugin and the `tensor`
dumps of lit files that stop before lowering. Update those FileCheck lines.

E3 (2–3 days): `lower_to_mem` per §5, then `lower.cpp`, `fuse.cpp`, `normalizers.cpp`, `lower_map_reduce.cpp`.
Target: `lit/tensor` and `lit/buffer` green except the files that still spell `«…; T»` for tensors.

E4 (1–2 days): migrate the lit files: tensor-typed parameters and results to `Tensor r s T`, literal inputs via
`from_array`, probes via `get`/`item` (≈ 79 files, 39 element-access sites by the last count). The dynamic
tests (`exec_dynamic*`, `dot_dynamic_run`, `unary_dynamic_run`) are the acceptance tests of the whole exercise:
they must pass unchanged in behavior, and `exec_dynamic_shape_param` must pass *without* any positional
boundary logic.

E5 (half a day): delete dead code and the `buf` references in docs; `docs/design/bufferize_roles.md` §2.1 is
superseded - say so at its top.

Estimate: 6–9 working days on the compiler side (Codex's independent estimate for the indexed variant was
13–20 including the frontend, which is out of scope here).

## 9. Acceptance

- `lit` green (currently 539 tests + E1's).
- `grep -n 'tensor_ty_\|op_args_\|record_boundary\|covers_every_axis\|tensor::buf' src/` is empty.
- `lower_to_mem.cpp` has no code path that decides tensor-ness from an `Arr` type.
- The three findings of the last review that were about classification are closed by construction:
  nested dependent groups (no role test in `collect_leaves`), discovery closure (`is_tensor_fn` is exact),
  extent-1 instantiation (`Tensor 1 1 I32` does not fold).
- No change under `src/mim/plug/{btensor,buffer,affine,mem,ll}` and none to `include/mim/phase.h`.

## 10. Frontend (deferred, for the record)

`mimir-frontend` builds tensor types with `world.arr` (21 sites in `operators.py`), peels `Seq` for shapes,
keeps singleton dims outside the type, reads elements with `world.extract` (1 site) and refers to a `tensor.get`
that no longer exists on this branch. After this change it builds `Tensor r s T` via the annex, wraps inputs with
`from_array`, reads probes with `get`/`item`, and stops special-casing singleton dims. The `torch` dialect is a
separate migration.

## 11. Risks

- Implicit inference from a nominal instance (E1). Mitigation: if inference needs the indices as explicit
  arguments in some ops, write them explicitly in the `_impl`s; the user-facing ops keep inferring from `s`.
- Inlining of §3's lams before `lower_to_mem` (the recognition points). Mitigation in §3.
- The generic rewriter rebuilding `Name`/`Struc` over a `Buf` (`rewrite.cpp` ≈118). Mitigation: override, §5.
- Scheduling code that silently relied on singleton folding (§6). Mitigation: audit `tensor.mim` ≈1010 and
  the `btensor.mim` remarks; the lit suite covers the selectors.
- Dump churn in lit: large but mechanical; do it per directory, not per file.

## 12. Conventions for the implementer

- Work on a branch off `zc/feat/erase-dynamic-extents`; commit per phase with the repo's `type(scope): subject`
  style, no co-author trailers.
- Follow `.github/copilot-instructions.md`: comments only for a non-obvious invariant, one line; `.clang-format`
  via `git clang-format`; Mim style per `.github/instructions/mim-style.instructions.md`.
- Build: `cmake --build build --target mim mim_all_plugins mim_lit_tests`; run lit from `lit/` with
  `/opt/homebrew/bin/lit ../build/lit -q`; `DEVELOPER_DIR=/Library/Developer/CommandLineTools` must be set.
- Do not touch the frontend repository.
- When a phase cannot be made green, stop and report the exact failing lit test and diagnostic rather than
  adding a type-based heuristic back.
