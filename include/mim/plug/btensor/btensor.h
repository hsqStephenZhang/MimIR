#pragma once

#include <array>
#include <optional>

#include <mim/lam.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/btensor/autogen.h"

namespace mim::plug::btensor {

/// Normalize legacy `(To, Tp, Ro, Rn, TSched)` and explicit `(To, Tp, Ro, Rp, Rn, TSched)` metadata.
inline std::array<const Def*, 6> mr_meta(const Def* meta) {
    if (meta->num_projs() == 6) return meta->projs<6>();
    auto [To, Tp, Ro, Rn, TSched] = meta->projs<5>();
    return {To, Tp, Ro, Ro, Rn, TSched};
}

inline const App* map_reduce_app(const Def* d) {
    if (auto mr = Axm::isa<btensor::map_reduce_post>(d)) return mr;
    return Axm::isa<btensor::map_reduce_iter>(d);
}

/// The fields `(vdim, unroll, tail, pout, rout, ptail, par)` of a schedule value (see `btensor.mk_sched`), if it
/// reduces to literals. A lowering that builds its own loops instead of `mr_nest` must still honor `tail` and `ptail`:
/// the domain is then larger than the op's actual one (see `tail_skip`, `ptail_skip`).
inline std::optional<std::array<u64, 7>> sched_fields(const Def* sched) {
    auto& w  = sched->world();
    auto nat = w.type_nat();
    auto sig = w.sigma({nat, nat, nat, nat, nat, nat, nat});
    auto id  = w.mut_lam(sig, sig)->set("fields");
    id->set(true, id->var());
    auto res  = w.app(w.app(sched, sig), id);
    auto flds = std::array<u64, 7>();
    for (u64 i = 0; i != 7; ++i)
        if (auto l = Lit::isa<u64>(res->proj(7, i)))
            flds[i] = *l;
        else
            return {};
    return flds;
}

/// Whether the reduction point at the raw i64 loop counters `iv` (of the whole loop vector over `sr`, the leading `rp`
/// parallel) is one a schedule's `tail` excludes: the last iteration of the innermost looped reduction dim, past the
/// first `tail` points of the first unrolled dim.
/// `nullptr` if the schedule excludes none.
inline const Def* tail_skip(const Def* sched, u64 rp, Defs sr, Defs iv) {
    auto& w   = sched->world();
    auto flds = sched_fields(sched);
    u64 r     = sr.size();
    if (!flds || (*flds)[2] == 0) return nullptr;
    auto [vdim, unroll, tail, pout, rout, ptail, par] = *flds;
    auto u_lo                                         = std::max(rp, r - std::min(r, unroll));
    if (u_lo <= rp || u_lo >= r) return nullptr;
    auto last = Lit::isa<u64>(sr[u_lo - 1]);
    if (!last) return nullptr;
    auto at_last = w.call(core::icmp::e, w.tuple({iv[u_lo - 1], w.lit_i64(*last - 1)}));
    auto past    = w.call(core::icmp::uge, w.tuple({iv[u_lo], w.lit_i64(tail)}));
    return w.call(core::bit2::and_, w.lit_nat(2), w.tuple({at_last, past}));
}

/// Whether the output cell at the raw i64 loop counters `iv` (as in `tail_skip`) is one a schedule's `ptail` excludes:
/// the last iteration of the parallel dim `vdim − 1`, past the first `ptail` points of the parallel dim `vdim`.
/// `nullptr` if the schedule excludes none.
inline const Def* ptail_skip(const Def* sched, u64 rp, Defs sr, Defs iv) {
    auto& w   = sched->world();
    auto flds = sched_fields(sched);
    if (!flds || (*flds)[5] == 0) return nullptr;
    auto vdim = (*flds)[0], ptail = (*flds)[5];
    if (vdim == 0 || vdim >= rp) return nullptr;
    auto last = Lit::isa<u64>(sr[vdim - 1]);
    if (!last) return nullptr;
    auto at_last = w.call(core::icmp::e, w.tuple({iv[vdim - 1], w.lit_i64(*last - 1)}));
    auto past    = w.call(core::icmp::uge, w.tuple({iv[vdim], w.lit_i64(ptail)}));
    return w.call(core::bit2::and_, w.lit_nat(2), w.tuple({at_last, past}));
}

} // namespace mim::plug::btensor
