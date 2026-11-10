#pragma once

#include <optional>
#include <vector>

#include <mim/def.h>

#include "mim/plug/tensor/tensor.h"

/// The cost model that picks the storage layout of a work-group panel, and its `tensor.layout_expr` terms.
namespace mim::plug::tensor::panel {

/// A swizzle of 16-byte chunk `c` of row `r` with `g = (r >> shift) % groups`: a rotation stores it at chunk
/// `(c + g) % chunks`, an exclusive or at `c ^ g`.
struct Swizzle {
    u64 width, groups, shift, chunks;
    bool xored = false;
};

/// A panel's storage layout: its last dim split into `(extent / split, split)`, those dims permuted (`perm[p]` is
/// the dim at physical position `p`), and the rows of the result swizzled.
struct Layout {
    std::vector<size_t> perm;
    u64 split = 1;
    std::optional<Swizzle> swizzle;
    /// The dims after the split and permutation, before merging and padding.
    std::vector<u64> permuted;
    bool identity() const;
};

/// How the work-items of the item level read a panel, per panel dim: the loop dim whose tile index selects it, or
/// whether an item reads the dim whole; the remaining dims are fixed per step.
struct Reads {
    std::vector<u64> counts, steps;
    std::vector<std::optional<size_t>> loop_dim;
    std::vector<bool> whole;
};

/// A layout of a panel with logical storage `storage`: its term (`nullptr` for the identity), its physical storage,
/// and whether a vector fill stays contiguous in it.
struct Choice {
    const Def* expr;
    const Def* physical;
    Layout layout;
    bool contiguous;
};

/// The `tensor.layout_expr.make` term of @p layout over @p logical, padded to @p physical unless that is empty.
const Def* term(const Layout& layout, const Def* logical, const std::vector<u64>& physical);

/// A tensor-core panel's layout: its fill's contiguous dim @p inner innermost, padded against bank conflicts.
Choice mma(const Def* storage, size_t inner);

/// The layout that spreads a fill of `vec` elements of `bytes` along dim @p inner, and the item level's @p reads,
/// over the 32 shared-memory banks best.
Choice choose(const Def* storage, size_t inner, u64 vec, u64 bytes, const Reads* reads);

/// The read map `inner ∘ outer`.
const Def* compose_read_map(const Def* inner, const Def* outer);

/// @p value as a pure read of a source over @p shape, seen through slices.
std::optional<PureRead> tensor_read(const Def* value, const Def* shape);

/// The dim of a rank-`rank` index that @p read sends unscaled to the last of its `physical_rank` coordinates, with the
/// gcd of the coordinate's other terms (0 if there are none).
std::optional<std::pair<size_t, u64>> contiguous_dim(const Def* read, const Def* rank, const Def* physical_rank);

/// The fields of the layout term @p expr over @p storage, for a fill as in choose().
Choice decode(const Def* expr, const Def* storage, size_t inner, u64 vec, u64 bytes);

} // namespace mim::plug::tensor::panel
