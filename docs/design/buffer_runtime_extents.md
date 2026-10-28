# Buffer runtime extents

Branch: `feat/buffer-runtime-extents`, stacked on `feat/nominal-tensor` (`6cf4141817`).
Merge the nominal Tensor branch before this one.

The buffer layer supports a literal rank with runtime extents. `buffer.lower_ptr` erases a dynamically sized
`buffer.Buf (r, s, T)` to `mem.Ptr («⊤; T», 0)`. The extent values remain ordinary scalar arguments and are
used to compute allocation sizes and row-major element addresses. The returned pointer carries no shape
metadata; the caller must know or separately receive the extents. This does not introduce a tensor descriptor ABI.

Literal shapes retain precise pointer types. Calls between precise and erased representations adapt arguments
and continuation parameters using the checking position that was typed before erasure. Scalar pointees after
size-1 folding are supported. Empty pointees lose their element type, so the phase records the original Buf
contract and only permits compatible element types and equal address spaces when adapting them.

`btensor.lower_map_reduce` runs before pointer erasure and expands buffer fills into memory-threaded loops.
It bypasses literal-empty output domains, skips empty concat segments, and handles constant padding from empty
inputs. Padding uses conditional reads when a runtime input extent may be zero; statically nonempty inputs
retain read/select. Reads from statically empty inputs over a nonempty map-reduce or broadcast domain are diagnosed.

Direct whole-array `buffer.init`, `buffer.lit` and `buffer.copy` lowering still requires static extents.
A runtime fill must pass through the btensor loop lowering first. Runtime-sized device transfers remain gated.

The `coerce_leaf` hook lets pointer erasure extend the generic structural rewriter's argument and continuation
adaptation. The memory optimizer checks closed terms before inspecting their memory type, including universe
nodes reached through dependent shapes.

`lit/buffer/runtime_extents.mim` compiles one native program with an independent C caller. It covers runtime
lengths 0, 1, 2, 7 and 17, vector/matrix fills, two-dimensional read/write linearization, and nontrivial dynamic
producers returning to literal-extent-0/1 continuations. `empty_init_adapter.mim` and `empty_lit_adapter.mim`
check empty-value initialization without the btensor fill phase. `lit/error/buffer_dynamic_transfer.mim` covers
the device-transfer gate.
