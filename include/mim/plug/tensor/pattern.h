#pragma once

#include <mim/pattern/op.h>

namespace mim::plug::tensor::graph {

/// Inspect a statically represented argument wrapper without constructing Extracts. Singleton
/// groups collapse; repeated Packs expand only when the schema supplies their slot count.
const Def* slot(const Def*, u64 count, u64 index);

/// A saturated raw axiom call, retaining all curried metadata groups.
pattern::Pattern call(pattern::PatternProgram&, flags_t, std::vector<pattern::Pattern> groups);

/// Shared, read-only structural patterns used by the existing native fusion recipes.
const App* map_reduce(const Def*);
const App* broadcast(const Def*);
const App* product(const Def*);

/// Register semantic data inputs for map_reduce_post, broadcast, product_2d, concat and slice.
/// map_reduce_post is deliberately not assumed pure: its combiner and post are arbitrary code.
/// product_2d is pure only for recognized standard Nat or wrapping integer rings.
void register_operations(pattern::OpRegistry&);

/// The supported rule preserves the exact fold, with zero/add/mul from core.
bool is_standard_ring(const Def*);

} // namespace mim::plug::tensor::graph
