#include "mim/plug/tensor/phase/distribute.h"

#include <mim/plug/core/core.h>
#include <mim/plug/gpu/gpu.h>
#include <mim/plug/math/math.h>

namespace mim::plug::tensor::phase {

void Distribute::start() {
    auto& old    = old_world();
    auto pending = DefVec(old.externals().muts().begin(), old.externals().muts().end());
    auto seen    = DefSet();
    bool warp    = false;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (auto lanes = Axm::isa<gpu::lanes>(def)) {
            auto self = lanes->arg(2, 0);
            if (self_ && self_ != self)
                lanes->blame("tensor.distribute: the warp values name different own lanes").bail();
            self_ = self;
        }
        warp |= Axm::isa<mma_warp>(def) || Axm::isa<warp_keep>(def);
        for (auto op : def->deps())
            pending.push_back(op);
    }
    // Without `gpu.lanes` the own lane is the warp loop's, as LowerToMem names it.
    if (!self_ && warp) {
        auto lane = old.call<core::bitcast>(old.type_nat(), old.app(old.annex<gpu::lane_id>(), old.tuple()));
        self_     = old.call<core::idx>(old.lit_nat_0(), old.lit_nat(32), lane);
    }
    if (self_) RWPhase::start();
}

/// The local component of the warp value @p warp in lane `self`.
const Def* Distribute::local(const App* warp) {
    auto& old = old_world();
    if (auto lanes = Axm::isa<gpu::lanes>(warp)) return rewrite(lanes->arg(2, 1));
    if (auto keep = Axm::isa<warp_keep>(warp)) return rewrite(old.extract(keep->arg(), self_));
    auto [a, b, c] = Axm::as<mma_warp>(warp)->args<3>();
    auto& w        = new_world();
    auto own       = [&](const Def* x) { return rewrite(old.extract(x, self_)); };
    return w.app(w.annex<gpu::mma_tf32>(), {own(a), own(b), own(c)});
}

/// The @p local value as read by lane `self` from lane @p lane, which `start` compares in the old world.
const Def* Distribute::shuffle(const Def* local, const Def* lane) {
    if (lane == self_) return local;
    auto& w = new_world();
    if (auto n = Lit::isa(local->type()->arity()); n && *n != 1)
        return w.tuple(DefVec(*n, [&](size_t i) { return shuffle(local->proj(*n, i), lane); }));
    if (local->type() != w.call<math::F>(Defs{w.lit_nat(23), w.lit_nat(8)}))
        local->blame("tensor.distribute: only F32 components cross lanes, not `{}`", local->type()).bail();
    if (auto x = Axm::isa(core::bit2::xor_, lane)) {
        auto [l, r] = x->args<2>();
        auto mask   = l == self_ ? Lit::isa(r) : r == self_ ? Lit::isa(l) : std::nullopt;
        if (mask) return w.app(w.annex<gpu::shfl_xor>(), {local, w.lit_i32(u32(*mask))});
    }
    auto src = w.call(core::conv::u, w.lit_i32(), rewrite(lane));
    return w.app(w.annex<gpu::shfl_idx>(), {local, src});
}

const Def* Distribute::rewrite_imm_Extract(const Extract* extract) {
    auto tuple = extract->tuple();
    auto warp  = tuple->isa<App>();
    if (!warp || !(Axm::isa<gpu::lanes>(warp) || Axm::isa<mma_warp>(warp) || Axm::isa<warp_keep>(warp)))
        return RWPhase::rewrite_imm_Extract(extract);
    // A read of a multi-axis warp value has the fused index `(lane, rest...)`.
    auto& old  = old_world();
    auto index = extract->index();
    auto lane  = index;
    auto rest  = DefVec();
    if (!Idx::isa(index->type()) || Idx::isa_lit(index->type()) != 32) {
        auto n = Lit::as(index->type()->arity());
        lane   = index->proj(n, 0);
        for (size_t i = 1; i != n; ++i)
            rest.push_back(index->proj(n, i));
    }
    auto value = local(warp);
    if (!rest.empty()) {
        auto& w = new_world();
        value   = w.extract(value, rest.size() == 1 ? rewrite(rest[0]) : rewrite(old.tuple(rest)));
    }
    return shuffle(value, lane);
}

} // namespace mim::plug::tensor::phase
