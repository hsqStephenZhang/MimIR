# TVM-style Fusion in MimIR: Current Status and Code Comparison

This document summarizes the current TVM-style tensor fusion implementation in MimIR, compares it with TVM Relax `FuseOps`, and separates graph partitioning support from actual IR/kernel fusion support.

The MimIR implementation described here consists of:

- committed graph partitioning infrastructure at commit `99173c501e` (`feat(tensor): add TVM-style fusion partition analysis`), and
- the current working-tree implementation of reduction epilogues (`%tensor.map_reduce_epilogue` and `%matrix.map_reduce_aff_epilogue`).

The TVM reference is the local checkout under `/workspaces/ml-compiler/tvm`, primarily:

- `src/relax/analysis/graph_partitioner.{h,cc}`
- `src/relax/transform/fuse_ops.cc`
- `docs/arch/fusion.rst`

## 1. Executive summary

MimIR currently implements two related layers of TVM-style fusion:

1. **TVM-like graph partitioning**
   - operator-pattern classification;
   - forward dependency graph construction;
   - post-dominator analysis;
   - union-find fusion groups;
   - path validation for diamonds and reconvergence;
   - external-use boundaries;
   - maximum fused-node limits;
   - `kOutEWiseFusable` grouping for matmul-like anchors.

2. **Actual semantic IR rewriting**
   - recursively flatten pointwise `map_reduce` producers;
   - compose affine input maps;
   - fuse a matmul/reduction anchor with a pointwise output chain;
   - represent the chain as a pure output epilogue;
   - preserve explicit epilogue tensor operands and their affine maps;
   - lower the fused form through bufferization into one reduction/output loop nest.

The tested end-to-end example is:

```text
matmul(a, b)
  -> relu
  -> add bias
  -> multiply by 2
```

It becomes one `%tensor.map_reduce_epilogue`: the contraction executes once, and ReLU/bias/multiply execute only after the reduction accumulator is complete and immediately before the output store.

The main missing TVM features are:

- tuple-aware phase 2 partitioning;
- actual tuple/multi-output IR rewriting;
- generated `Primitive=True` fused functions;
- a separate FuseTIR-style generic kernel merger;
- argument-count limits and postponed fusion;
- target/device-aware cost models and scheduling;
- general multi-anchor and compound-reduction fusion.

## 2. Pipeline comparison

### TVM

TVM separates grouping from low-level kernel merging:

```text
Relax operators
  -> LegalizeOps
  -> AnnotateTIROpPattern
  -> FuseOps
       - build graph
       - partition with post-dominators
       - create Primitive=True Relax functions
  -> FuseTIR
       - inline/merge TIR PrimFuncs
       - eliminate intermediate buffers
  -> target scheduling/codegen
```

`FuseOps` alone groups calls into a function. The final removal of intermediate TIR buffers is performed by `FuseTIR`.

### MimIR

MimIR currently performs a more direct semantic rewrite:

```text
high-level tensor operators
  -> %tensor.lower_tensor
       - rewrite unary/binary/matmul/etc. to %tensor.map_reduce
  -> %tensor.fuse_tensor
       - classify map_reduce access patterns
       - build graph and fusion groups
       - directly compose combiners and affine maps
       - emit %tensor.map_reduce or %tensor.map_reduce_epilogue
  -> %tensor.lower_to_mem
       - convert tensor operands to buffers
       - emit %matrix.map_reduce_aff[_epilogue]
  -> %matrix.lower_aff
       - allocate the output buffer
       - generate output and reduction loops
       - load explicit epilogue inputs after reduction
       - execute epilogue and store
  -> buffer/affine/LLVM lowering
```

The key architectural difference is:

| Topic | TVM | MimIR today |
|---|---|---|
| Group representation | New Relax function with `Primitive=True` | Union-find group used to drive direct IR rewriting |
| Actual composition | Later `FuseTIR` merges PrimFuncs | `Fuse` directly composes lambda semantics and affine maps |
| Intermediate elimination | TIR buffer/dataflow fusion | No intermediate tensor is emitted by the composed `map_reduce` |
| Kernel boundary | Explicit primitive function | Implicit in the surviving `map_reduce`/loop nest |
| Multi-output | Group function returns tuple | Not implemented in actual rewrite yet |

## 3. Operator pattern classification

Both systems use the same ordered pattern lattice:

```cpp
enum class OpPatternKind {
    kElemWise = 0,
    kBroadcast = 1,
    kInjective = 2,
    kCommReduce = 3,
    kOutEWiseFusable = 4,
    kTuple = 7,
    kOpaque = 8,
};
```

### TVM classification model

TVM typically obtains `op_pattern` from a legalized TIR PrimFunc:

```cpp
ffi::Optional<int64_t> opt_pattern = func->GetAttr<int64_t>("op_pattern");
if (opt_pattern.has_value()) {
  pattern = static_cast<OpPatternKind>(opt_pattern.value());
} else {
  pattern = OpPatternKind::kOpaque;
}
```

Examples in TVM's documentation:

| Pattern | Examples |
|---|---|
| `kElemWise` | add, relu, exp |
| `kBroadcast` | bias add |
| `kInjective` | reshape, transpose, concatenate |
| `kCommReduce` | sum, max, mean |
| `kOutEWiseFusable` | conv2d, matmul, dense |
| `kTuple` | tuple construction/projection flow |
| `kOpaque` | side effects and unsupported/external calls |

### MimIR classification from rich `map_reduce` semantics

MimIR classifies a lowered tensor operator by inspecting reduction rank and affine access maps:

```cpp
OpPatternKind classify_map_reduce(const App* mra) {
    auto [nis, ToRoRr, SoSr, TisRisSis, comb_init, map_out, maps]
        = mra->callee()->as<App>()->uncurry_args<7>();
    auto [To, Ro, Rr] = ToRoRr->projs<3>();

    auto Rr_lit = Lit::isa<u64>(Rr);
    if (!Rr_lit) return OpPatternKind::kOpaque;

    // Classify every input map as identity, broadcast, or injective.
    ...

    if (*Rr_lit > 0) {
        if (has_injective) return OpPatternKind::kOutEWiseFusable;
        return OpPatternKind::kCommReduce;
    }

    if (has_injective || !is_elem_wise)
        return OpPatternKind::kInjective;
    if (has_elem_wise)
        return OpPatternKind::kElemWise;
    if (has_broadcast)
        return OpPatternKind::kBroadcast;
    return OpPatternKind::kElemWise;
}
```

This gives MimIR a useful semantic advantage: classification is derived from the actual loop/access relation rather than manually attached to every high-level operator.

For example, a matrix product has:

```text
loop domain:       (m, n, k)
left access map:   (m, n, k) -> (m, k)
right access map:  (m, n, k) -> (k, n)
output map:        (m, n, k) -> (m, n)
reduction rank:    1
```

The input maps are nontrivial injective projections and `Rr > 0`, so the lowered operator becomes `kOutEWiseFusable`.

Current MimIR mapping:

```cpp
if (auto mr = Axm::isa<tensor::map_reduce>(app)) {
    record_kind(classify_map_reduce(mr), summarize_map_reduce_aff(mr));
} else if (Axm::isa<tensor::map_reduce_epilogue>(app)) {
    record_kind(OpPatternKind::kOutEWiseFusable);
} else if (Axm::isa<tensor::broadcast>(app) ||
           Axm::isa<tensor::broadcast_in_dim>(app)) {
    record_kind(OpPatternKind::kBroadcast);
} else if (Axm::isa<tensor::transpose>(app) ||
           Axm::isa<tensor::reshape>(app) ||
           Axm::isa<tensor::repeat>(app)) {
    record_kind(OpPatternKind::kInjective);
} else if (Axm::isa<tensor::map>(app) ||
           Axm::isa<tensor::unary>(app) ||
           Axm::isa<tensor::binary>(app) ||
           Axm::isa<tensor::select>(app)) {
    record_kind(OpPatternKind::kElemWise);
}
```

## 4. Graph partitioning comparison

### Shared structure

Both implementations use:

- a forward dataflow graph;
- topological/post-DFS node numbering;
- immediate post-dominators computed by LCA in a DAG;
- union-find fusion groups;
- all-path validation before committing a fusion;
- explicit external references as fusion boundaries.

This is important for diamond graphs:

```text
          matmul
         /      \
      relu     add-constant
         \      /
           add
```

The source can fuse to the reconvergent sink only if every path to the post-dominator satisfies the pattern constraints.

### TVM `CheckPath` and `CommitFuse`

TVM validates every path recursively:

```cpp
template <typename F>
bool GraphPartitioner::CheckPath_(Node* src, Node* sink, F fcond) {
  if (visited_.count(src)) return true;
  visited_.insert(src);
  Group* group = groups_[src->index]->FindRoot();
  if (!fcond(group->pattern, src == sink)) return false;
  if (src == sink) return true;
  for (auto edge = src->outputs.head; edge != nullptr; edge = edge->next) {
    if (!CheckPath_(edge->value.node, sink, fcond)) return false;
  }
  return true;
}
```

It then merges every node on the validated paths into the sink group:

```cpp
void GraphPartitioner::CommitFuse_(Node* src, Node* sink, Group* target) {
  if (src == sink || visited_.count(src)) return;
  visited_.insert(src);
  MergeFromTo(groups_[src->index], target);
  for (auto edge = src->outputs.head; edge != nullptr; edge = edge->next) {
    CommitFuse_(edge->value.node, sink, target);
  }
}
```

### MimIR equivalent

MimIR uses the same recursive all-path condition:

```cpp
bool FusionPartitionAnalysis::check_path(size_t src,
                                         size_t sink,
                                         bool allow_reduction_sink) const {
    absl::flat_hash_set<size_t> visited;
    std::function<bool(size_t)> check = [&](size_t id) {
        if (!visited.emplace(id).second) return true;
        auto pattern = groups_[find(id)].pattern;
        if (id == sink)
            return pattern <= OpPatternKind::kInjective ||
                   (allow_reduction_sink &&
                    pattern == OpPatternKind::kCommReduce);
        if (pattern > OpPatternKind::kInjective) return false;
        for (auto output : nodes_[id].outputs)
            if (!check(output)) return false;
        return true;
    };
    for (auto output : nodes_[src].outputs)
        if (!check(output)) return false;
    return true;
}
```

For `kOutEWiseFusable`, MimIR has a dedicated condition matching TVM's phase-0 rule:

```cpp
bool FusionPartitionAnalysis::check_output_path(size_t src,
                                                size_t sink) const {
    ...
    if (groups_[find(id)].pattern > OpPatternKind::kBroadcast)
        return false;
    ...
}
```

The union-find commit is likewise direct:

```cpp
void FusionPartitionAnalysis::commit_fuse(size_t src, size_t sink) {
    auto target = find(sink);
    ...
    groups_[target].size += groups_[child].size;
    groups_[target].pattern = max(groups_[target].pattern,
                                  groups_[child].pattern);
    groups_[child].parent = target;
    ...
}
```

### Phase rules: exact comparison

TVM runs three phases:

```cpp
for (int phase = 0; phase < 3; ++phase) {
  RunFuse(graph, post_dom_tree, phase);
}
```

The relevant TVM rules are:

```cpp
if (group_node->pattern == kOutEWiseFusable) {
  if (phase != 0) continue;
  if (dom_node->pattern == kElemWise) {
    auto fcond = [](OpPatternKind kind, bool is_sink) {
      return kind <= kBroadcast;
    };
    if (CheckPath(src, sink, fcond)) CommitFuse(src, sink);
  }
} else if (group_node->pattern <= kBroadcast) {
  // Producers can fuse into injective/reduction/out-elementwise groups.
  ...
} else if (group_node->pattern == kInjective ||
           group_node->pattern == kTuple) {
  if (phase != 1) continue;
  ...
} else {
  // kCommReduce never initiates fusion.
}
```

TVM phase 2 specifically fuses injective producers into intermediate tuples after the tuple has been absorbed into a later injective group.

MimIR currently runs two phases:

```cpp
for (int phase = 0; phase < 2; ++phase) {
    ...
    bool candidate = phase == 0
        ? src_pattern <= OpPatternKind::kBroadcast
        : src_pattern == OpPatternKind::kInjective;

    if (phase == 0 &&
        src_pattern == OpPatternKind::kOutEWiseFusable) {
        candidate = true;
        valid_path = check_output_path(id, sink);
    }
    ...
}
```

Therefore:

| TVM partition rule | MimIR status |
|---|---|
| Elemwise/broadcast producer fusion | Implemented |
| Injective phase | Implemented |
| Elemwise/broadcast into reduction sink | Implemented |
| `kOutEWiseFusable` output chain | Implemented |
| Opaque/external boundary | Implemented |
| Maximum fused-node count | Implemented |
| Maximum generated-function argument count | Not implemented |
| Postponed fusion for argument accounting | Not implemented |
| Tuple phase 1 special handling | Pattern exists, rewrite incomplete |
| Tuple phase 2 | Not implemented |

## 5. Actual pointwise IR fusion

Graph grouping alone does not remove intermediate tensors. MimIR's `Fuse` phase currently implements actual rewriting for pointwise `map_reduce` trees.

Given:

```text
Outer: map_reduce(..., input = Inner(...))
Inner: map_reduce(Rr = 0, map_out = identity, ...)
```

MimIR:

1. replaces the inner output slot with the inner's leaf inputs;
2. composes each inner affine access map with the outer access map;
3. builds one new combiner continuation;
4. invokes inner combiners in post-order;
5. invokes the outer combiner last;
6. emits one flattened `%tensor.map_reduce`.

The core affine-map composition is:

```cpp
// inner(outer(index))
static const Def* compose_map(World& w,
                              const Def* inner,
                              const Def* outer) {
    auto dom = outer->type()->as<Pi>()->dom();
    auto codom = inner->type()->as<Pi>()->codom();
    auto lam = w.mut_lam(dom, codom)->set("fused_map");
    lam->set(true, w.app(inner, w.app(outer, lam->var())));
    return lam;
}
```

Supported pointwise cases include:

```text
map(map(x))
binary(map(x), map(y))
nested elementwise trees
multiple pointwise inputs
injective/broadcast paths representable as map_reduce access maps
```

The actual rewriter additionally requires `can_inline(producer, consumer)`, currently meaning the producer has one direct consumer and is not externally visible. This is stricter than TVM's grouped-function multi-output handling.

## 6. Matmul/reduction output epilogue fusion

### Why matmul remains one anchor

Matmul is already represented by one semantic `map_reduce`:

```text
for m, n:
    acc = 0
    for k:
        acc += A[m, k] * B[k, n]
    C[m, n] = acc
```

Fusion must not inline ReLU into the reduction combiner:

```text
// Wrong
acc = relu(acc + A[m, k] * B[k, n])
```

Instead it must modify only the final write-back:

```text
// Correct
for m, n:
    acc = reduce_k(A[m, k] * B[k, n])
    C[m, n] = epilogue(acc, m, n)
```

### MimIR tensor-level fused representation

```mim
axm %tensor.map_reduce_epilogue: {nis: Nat}
                               -> {neis: Nat}
                               -> {Ta To: *, Ro Rr: Nat}
                               -> [So, Sr]
                               -> {Tis, Ris, Sis}
                               -> {Teis, Reis, Seis}
                               -> [f: Fn [Ta, «i: nis; Tis#i»] -> Ta,
                                   init: Ta]
                               -> [epilogue:
                                      Cn [Ta,
                                          «i: neis; Teis#i»,
                                          Cn To]]
                               -> [map_out]
                               -> [maps]
                               -> [epilogue_maps]
                               -> [is, eis]
                               -> «So; To»;
```

Meanings:

| Parameter | Meaning |
|---|---|
| `Ta` | Reduction accumulator type |
| `To` | Final output element type |
| `is` | Inputs loaded inside reduction loops |
| `maps` | Full `(output..., reduction...) -> input index` maps |
| `eis` | Tensor inputs used only by the output epilogue |
| `epilogue_maps` | `output index -> epilogue input index` maps |
| `epilogue` | Pure transformation of the final accumulator and epilogue input values |

Keeping `Ta` and `To` separate allows future mixed-precision/type-changing epilogues.

### Why epilogue tensor inputs are explicit

An earlier design could capture `bias` in the epilogue lambda:

```text
epilogue = lambda acc, index:
    relu(acc) + bias[index]
```

That hides a memory read in a supposedly pure closure and makes memory-token lowering difficult.

The current design instead stores bias as an explicit `eis` operand:

```text
eis = [bias]
epilogue_maps = [identity]

epilogue = lambda acc, [bias_value], index:
    (relu(acc) + bias_value) * 2
```

This keeps the closure pure and lets buffer lowering decide exactly where the read occurs.

### Rewrite preconditions

`Fuse::fuse_reduction_epilogue` currently requires the outer pointwise operation to have:

```cpp
Rr == 0
So == Sr
map_out == identity
```

It also requires:

- exactly one complex reduction anchor;
- the anchor to be read through an identity map;
- matching output shape and rank;
- a partition-approved, single-use producer/consumer edge;
- literal input/output/reduction rank counts for current lowering.

When another pointwise operator is encountered later, the new operator is composed after the existing epilogue:

```cpp
if (old_epilogue) {
    auto old_ret = w.mut_con(anchor_To)->set("priorEpilogueRet");
    emit_outer(old_ret, old_ret->var(0), ret);
    epilogue->app(true, old_epilogue,
                  {reduced, prior_values, coords, old_ret});
} else {
    emit_outer(epilogue, reduced, ret);
}
```

This is how:

```text
matmul -> relu -> bias add -> multiply
```

becomes one chained continuation instead of several materialized tensors.

## 7. Concrete before/after IR

### Source

```mim
let mm     = %tensor.product_2d R (a, b);
let active = %tensor.unary relu mm;
let biased = %tensor.binary add (active, bias);
let result = %tensor.unary mul2 biased;
```

### Fused tensor IR

The generated IR contains one reduction anchor:

```mim
let result = %tensor.map_reduce_epilogue
    ...
    (reduction_combiner, 0.0)
    fusedEpilogue
    output_map
    (lhs_map, rhs_map)
    (bias_map)
    ((a, b), bias);
```

The actual generated epilogue is:

```mim
fun fusedEpilogue (reduced: F32,
                   bias_value: F32): F32 =
    let active = %math.extrema.iM 0 (0.0, reduced);
    let biased = %math.arith.add 0 (active, bias_value);
    let result = %math.arith.mul 0 (2.0, biased);
    return result;
```

There is no second ordinary `%tensor.map_reduce` for ReLU, bias add, or multiplication.

### Lowered loop order

The functional lowering makes ordering explicit:

```text
forOut_0(m):
  forOut_1(n):
    writeBack(reduced):
      active     = relu(reduced)
      bias_value = get(bias, epilogue_map(m, n))
      biased     = active + bias_value
      result     = biased * 2
      set(output, output_map(m, n), result)

    forIn_0(k, acc):
      lhs = get(a, lhs_map(m, n, k))
      rhs = get(b, rhs_map(m, n, k))
      yield acc + lhs * rhs
```

The critical ordering property is:

```text
reduction loop -> writeBack -> epilogue reads/computation -> output store
```

not:

```text
reduction iteration -> epilogue -> next reduction iteration
```

## 8. Bufferization and memory semantics

MimIR introduces the corresponding buffer-world operation:

```mim
axm %matrix.map_reduce_aff_epilogue: {nis: Nat}
                                   -> {neis: Nat}
                                   -> {Ta To: *, Ro Rr: Nat}
                                   -> ...
                                   -> [f: Fn [%mem.M 0,
                                              Ta,
                                              «i: nis; Tis#i»]
                                           -> [%mem.M 0, Ta],
                                       init: Ta]
                                   -> [epilogue: Cn [Ta,
                                                      «i: neis; Teis#i»,
                                                      Cn To]]
                                   -> ...
                                   -> [%mem.M 0, is, eis]
                                   -> [%mem.M 0,
                                       %buffer.Buf (Ro, So, To)];
```

`LowerToMem`:

1. converts both `is` and `eis` to buffers;
2. wraps the reduction combiner with an explicit memory token;
3. preserves the pure epilogue continuation;
4. emits `%matrix.map_reduce_aff_epilogue`.

`LowerAff` then builds the final write-back:

```cpp
auto write_back = mem::mut_con(Ta)->set("writeBack");
auto [wb_mem, element_final] = write_back->vars<2>();

// Compute output coordinates.
auto [mem1, write_coords] = affine_map(acc_out, ..., wb_mem);

// Read every explicit epilogue tensor input at output coordinates.
for (u64 i = 0; i < neis; ++i) {
    auto [mem2, coords] = affine_map(epilogue_maps[i], ..., mem1);
    auto [mem3, value] = buffer::op_read(..., mem2, eis[i], coords);
    epilogue_elements[i] = value;
    mem1 = mem3;
}

// The pure epilogue returns to a continuation that performs the final store.
auto epilogue_ret = w.mut_con(To)->set("epilogueRet");
auto stored = buffer::op_write(...,
                               mem1,
                               output,
                               write_coords,
                               epilogue_ret->var(0));
epilogue_ret->app(true, cont, {stored_mem, output});
write_back->app(true, epilogue,
                {element_final,
                 epilogue_elements,
                 epilogue_ret});
```

This design supports CPU and GPU-host lowering because memory effects remain explicit at the buffer/memory layer. The tensor epilogue itself has no hidden read effect.

## 9. What is implemented today

### Graph analysis and grouping

- [x] Pattern lattice compatible with TVM
- [x] Elementwise classification
- [x] Broadcast classification
- [x] Injective classification
- [x] Commutative-reduction classification
- [x] `kOutEWiseFusable` classification from reduction/access semantics
- [x] Opaque boundaries
- [x] Forward dependency graph
- [x] Post-dominator tree
- [x] Diamond/reconvergence path checking
- [x] Union-find groups
- [x] External-use boundaries
- [x] Maximum fused-node count
- [ ] Maximum fused-function argument count
- [ ] TVM postponed-fusion argument accounting
- [ ] Full tuple phase 1/phase 2 partitioning

### Actual semantic fusion

- [x] Nested pointwise map fusion
- [x] Multi-input pointwise fusion
- [x] Recursive flattening of pointwise trees
- [x] Affine access-map composition
- [x] Elementwise/broadcast/injective boundaries represented by map semantics
- [x] Pointwise producers into reduction sinks
- [x] Matmul/reduction output epilogue
- [x] ReLU after matmul
- [x] Explicit bias tensor input
- [x] Additional unary elementwise operation after bias add
- [x] Composition onto an existing epilogue
- [x] Type-changing accumulator/output interface (`Ta` versus `To`)
- [x] Tensor-to-buffer lowering
- [x] Affine loop lowering
- [x] Default compilation pipeline smoke test
- [ ] Tuple/multi-output rewrite
- [ ] Multiple complex anchors in one group
- [ ] Reduction-to-reduction fusion
- [ ] Compound reduction such as complete softmax
- [ ] Generic grouped-function/kernel abstraction
- [ ] Target-aware profitability model

## 10. Tests

The graph partitioning tests cover:

- nested fusion;
- multiple inputs;
- diamonds;
- post-dominator reconvergence;
- external uses;
- broadcast boundaries;
- injective boundaries;
- reduction boundaries;
- injective producers that must remain outside a reduction anchor;
- shared reduction outputs that must not be duplicated;
- two complex anchors meeting at one pointwise consumer;
- independent fusion analysis across multiple external functions;
- both sides of the 16-node maximum fusion-group boundary;
- symbolic shapes where supported.

The reduction epilogue tests are:

1. `lit/tensor/fuse_matmul_epilogue.mim`
   - verifies exactly one `%tensor.map_reduce_epilogue`;
   - verifies ReLU, add, and multiply are inside the epilogue;
   - verifies no ordinary materialized pointwise `%tensor.map_reduce` remains.

2. `lit/tensor/lower_matmul_epilogue.mim`
   - verifies the epilogue is inside `writeBack`;
   - verifies ReLU, bias read, add, multiply, and store ordering;
   - verifies the reduction loop is distinct and feeds `writeBack`.

3. `lit/tensor/fuse_matmul_epilogue_default.mim`
   - runs the default pipeline end to end;
   - catches tensor-to-buffer or matrix/affine lowering failures.

4. `lit/tensor/fuse_matmul_epilogue_dynamic.mim`
   - uses static rank with fully symbolic `m`, `k`, and `n` extents;
   - verifies fusion still emits one complete `%tensor.map_reduce_epilogue`;
   - verifies the generated epilogue contains ReLU, bias add, and multiply.

5. `lit/tensor/fuse_matmul_epilogue_dynamic_default.mim`
   - records the current expected default-pipeline failure;
   - the dependent tensor return continuation is not yet converted to its `%buffer.Buf` ABI.

After aligning the suite with TVM's `test_transform_fuse_ops.py`, the complete
`fuse_*` regression set contains 22 tests and all pass. In particular:

- `fuse_out_ewise_broadcast_epilogue.mim` verifies a matmul absorbs a broadcast bias add;
- `fuse_shared_reduction_output.mim` verifies a shared matmul is materialized once rather than duplicated;
- `fuse_two_complex_anchors.mim` verifies one matmul may own the epilogue only after the other is materialized;
- `fuse_injective_reduction_boundary.mim` mirrors TVM's rule that an injective layout transform is not folded into a complex reduction anchor;
- `fuse_multiple_functions.mim` verifies groups do not cross external function boundaries;
- `fuse_max_nodes.mim` now checks both 16 operations (one group) and 17 operations (two groups);
- `fuse_symbolic_shape.mim` now uses the current phase API and validates an actual symbolic-extent fusion.

```text
Testing: 22 of 374 tests
Passed: 22
Failed: 0
```

### Dynamic-shape status

Here, "dynamic shape" means a statically known rank with symbolic extents, for example:

```mim
fun extern f {m k n: Nat}
             (a: «m, k; F32»,
              b: «k, n; F32»,
              bias: «m, n; F32»)
           : «m, n; F32» = ...;
```

Fusion supports this case. It requires `nis`, `Ro`, `Rr`, and operand ranks to be literals, but it does not require `m`, `k`, or `n` to be literals. The generated operation retains symbolic shapes:

```text
%tensor.map_reduce_epilogue
  output shape = (m, n)
  loop shape   = (m, n, k)
  lhs shape    = (m, k)
  rhs shape    = (k, n)
  bias shape   = (m, n)
```

Dynamic testing initially exposed a dependent-type placement bug: carrying `Idx(m)` and `Idx(n)` directly in the epilogue lambda ABI made the nested mutable depend on free symbolic shape binders. The pure epilogue does not need coordinates because all index-dependent reads are already represented by `eis + epilogue_maps`. Removing coordinates from the epilogue ABI fixed fusion-level dynamic shapes while preserving coordinates in affine lowering and the output store.

Later pipelines are not fully dynamic-shape capable yet:

- functional `%tensor.lower_epilogue` creates a dependent `mapRed` helper that is not placed correctly under the symbolic shape binder;
- the default buffer pipeline currently passes `%buffer.Buf (2, (m, n), T)` to a return continuation that still expects `«m, n; T»`;
- this is a dependent function-boundary conversion issue, not a fusion decision or affine-bound issue.

Therefore the current result is:

| Layer | Static rank + symbolic extents |
|---|---|
| Pattern classification | Supported |
| Graph partitioning | Supported |
| Pointwise/reduction epilogue fusion | Supported and tested |
| Fused tensor IR | Supported and complete |
| Functional tensor loop lowering | Not yet valid for a generic dependent external function |
| Tensor-to-buffer/default pipeline | Not yet supported at dependent return boundaries |

## 11. Tuple and multi-output comparison

### What TVM does

TVM's graph and function extraction explicitly support tuples:

```cpp
Expr body = outputs.size() == 1 ? outputs[0] : Tuple(outputs);
group_attrs.Set(attr::kPrimitive, true);
Function function(params, body, ..., DictAttrs(group_attrs));
```

At the call site, TVM remaps each original output through `TupleGetItem`:

```cpp
if (IsTupleOutput(func)) {
  for (const auto& var : pending_tuple_get[group]) {
    auto tuple_get = TupleGetItem(new_var, tuple_get_indices_[var.get()]);
    var_remap_[var->vid] = builder_->Emit(tuple_get);
  }
}
```

This allows one fused Relax group to expose several tensor outputs. FuseTIR can then produce a PrimFunc with multiple output buffers.

### What MimIR has today

MimIR has `kTuple` in the pattern enum, but:

- the partitioner has no TVM phase 2 tuple rule;
- `Fuse::rewrite_imm_App` only creates one resulting tensor operation;
- `%tensor.map_reduce_epilogue` has one output shape and one output type;
- `%matrix.map_reduce_aff_epilogue` allocates and writes one output buffer;
- `can_inline` requires one direct consumer, preventing true fan-out fusion.

Therefore tuple grouping is not yet an actual multi-output kernel transformation.

### Recommended first multi-output form

For outputs with the same shape, add an internal operation such as:

```mim
axm %tensor.map_reduce_multi_epilogue:
    {nouts: Nat}
 -> {Ta: *, Tos: «nouts; *»}
 -> [So, Sr]
 -> ...
 -> [epilogue:
        Cn [Ta,
            epilogue_inputs,
            Cn «i: nouts; Tos#i»]]
 -> «i: nouts; «So; Tos#i»»;
```

The buffer equivalent would allocate `nouts` buffers and store all returned scalar values in one write-back region:

```text
for output_index:
    reduced = reduction(...)
    values = epilogue(reduced, epilogue_inputs)
    for output in outputs:
        store(output.buffer, output_index, values[output])
```

This is tuple-of-tensors (SoA):

```text
tuple<Tensor<T0>, Tensor<T1>, ...>
```

It should not be encoded as a tensor-of-tuples (AoS):

```text
Tensor<tuple<T0, T1, ...>>
```

because separate downstream uses, layouts, and output buffers require SoA semantics.

Recommended first-version restrictions:

- one reduction/matmul anchor;
- all outputs have the same `So` iteration domain;
- each output is an elementwise/broadcast chain from that anchor;
- different output element types are allowed;
- every external use becomes an explicit output slot;
- no second complex anchor;
- no different-shape transpose/reduction output.

This would cover:

```text
matmul -> (relu(result), result + bias)
matmul -> (identity(result), clamp(result), cast(result))
elementwise -> (f(x), g(x))
```

## 12. Remaining semantic and engineering gaps

### Partitioning gaps relative to TVM

1. No tuple phase 2.
2. No generated-function argument limit.
3. No postponed fusion based on not-yet-known group arguments.
4. No group attributes or backend-specific annotations.
5. No `Primitive=True`-equivalent explicit fused function boundary.

### Rewrite gaps

1. No tuple/multi-output actual rewrite.
2. No generic fusion of arbitrary grouped operators; rewriting is currently specialized around `map_reduce` semantics.
3. A group cannot contain two complex anchors.
4. A reduction cannot fuse forward except through the dedicated output-epilogue form.
5. A separately materialized broadcast producer used as an epilogue operand is not always recursively flattened into the epilogue input map.
6. Static-rank/dynamic-extent fusion works, but dynamic `nis`, ranks, and reduction counts are rejected. The default buffer pipeline also does not yet convert dependent tensor return continuations to their buffer ABI.
7. Conv should reuse the same map-reduce anchor model, but dedicated conv-output-fusion tests are still missing.

### Backend gaps

1. No target-aware profitability/cost model.
2. No device/stream/kernel-launch boundary analysis.
3. No automatic tiling/vectorization/shared-memory scheduling as part of fusion.
4. No generic equivalent of TVM FuseTIR's symbolic buffer matching.
5. No direct CUTLASS/cuBLAS epilogue dispatch yet.

## 13. Completion estimate

The percentage depends on what is counted:

| Scope | Estimate | Reason |
|---|---:|---|
| TVM-style pattern classification | 75-85% | Main kinds work; tuple and some dynamic cases remain |
| Graph partitioning rules | 65-75% | Post-dom, union-find, OutEWise, boundaries work; tuple phase and arg accounting remain |
| Actual pointwise semantic fusion | 65-75% | Recursive map composition works for supported `map_reduce` forms |
| Matmul output epilogue | 70-80% | Single-output chain and lowering work; broadcast flattening, conv tests, multi-output remain |
| Full TVM FuseOps + FuseTIR feature set | 40-50% | No grouped primitive functions, generic TIR merger, multi-output, cost model, or scheduling |

The most important achievement is that the implementation is no longer only a grouping analysis: for supported cases it performs a real semantic rewrite and removes intermediate tensors through the complete tensor-to-buffer loop-lowering path.

The next highest-value feature is same-shape tuple/multi-output epilogues, because it closes the largest visible gap with TVM's grouped-function representation while naturally extending the existing reduction write-back design.
