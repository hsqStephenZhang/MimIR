# Buffer runtime extents

The buffer layer supports a literal rank with runtime extents. `buffer.lower_ptr` erases a dynamically sized
`buffer.Buf (r, s, T)` to `mem.Ptr («⊤; T», 0)`. The extent values remain ordinary scalar arguments and are
used to compute allocation sizes and row-major element addresses. The returned pointer carries no shape
metadata; the caller must know or separately receive the extents. This does not introduce a tensor descriptor ABI.

Literal shapes retain precise pointer types. Calls between precise and erased representations adapt arguments
and continuation parameters using the checking position that was typed before erasure. Scalar pointees after
size-1 folding are supported. Empty pointees lose their element type, so the phase records the original Buf
contract and only permits compatible element types and equal address spaces when adapting them.

Direct whole-array `buffer.init`, `buffer.lit` and `buffer.copy` lowering still requires static extents.
A runtime fill must pass through the btensor loop lowering first. Runtime-sized device transfers remain gated.

The `coerce_leaf` hook lets pointer erasure extend the generic structural rewriter's argument and continuation
adaptation.

Constant padding branches before reading an input with runtime extents, since its size may be zero at execution.
Static nonempty inputs keep clamped reads and select.
