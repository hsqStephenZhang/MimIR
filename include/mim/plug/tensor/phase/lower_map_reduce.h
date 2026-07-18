#pragma once

#include <functional>

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

/// Lowers the low-level tensor axioms (`map_reduce`, `pad`, `concat`, `broadcast`)
/// directly to their underlying primitives (loops, `extract`, `insert`, `pack`, …).
/// High-level axioms (`transpose`, `conv`, `broadcast_in_dim`, …) are expected to have been desugared to
/// these low-level axioms by an earlier `Lower` phase.
class LowerMapReduce : public RWPhase {
public:
    LowerMapReduce(World& world, flags_t annex, bool epilogue_only = false)
        : RWPhase(world, annex)
        , epilogue_only_(epilogue_only) {}

private:
    const Def* rewrite_imm_App(const App*) final;

    const Def* lower_broadcast(const App*);
    const Def* lower_map_reduce(const App*, bool has_epilogue = false);
    const Def* lower_pointwise_loop(const App*);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);
    const Def* lower_stablehlo_sort(const App*);

    /// Builds `ro` output loops over `So` and writes the element returned by `compute(out_iters, inputs)` at the
    /// identity output coordinates. `out_iters` are the raw i64 loop counters. Used by the non-affine pointwise
    /// lowerings (`pad`, `concat`) whose element value is chosen conditionally on the output coordinate.
    const Def* build_pointwise(const Def* inputs,
                               const Def* type,
                               const Def* So,
                               u64 ro,
                               std::function<const Def*(const DefVec&, const Def*)> compute);
    const Def* build_pointwise_loop(const Def* inputs, const Def* type, const Def* So, const Def* r, const Def* body);

    const Def* rec_broadcast(const Def* s_in, const Def* s_out, const Def* input, u64 r, u64 i);

    bool epilogue_only_;
};

class LowerEpilogue final : public LowerMapReduce {
public:
    LowerEpilogue(World& world, flags_t annex)
        : LowerMapReduce(world, annex, true) {}
};

} // namespace mim::plug::tensor::phase
