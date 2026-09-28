#pragma once

#include <array>
#include <optional>

#include <mim/lam.h>
#include <mim/world.h>

#include <mim/plug/core/core.h>

#include "mim/plug/btensor/autogen.h"

namespace mim::plug::btensor {

/// The fields `(vdim, unroll, tail, pout, rout)` of a schedule value (see `btensor.mk_sched`), if it reduces to literals.
/// A lowering that builds its own loops instead of `mr_nest` must still honor `tail`: the domain is then larger than
/// the op's actual one (see `tail_skip`).
inline std::optional<std::array<u64, 5>> sched_fields(const Def* sched) {
    auto& w   = sched->world();
    auto nat  = w.type_nat();
    auto sig  = w.sigma({nat, nat, nat, nat, nat});
    auto id   = w.mut_lam(sig, sig)->set("fields");
    id->set(true, id->var());
    auto res  = w.app(w.app(sched, sig), id);
    auto flds = std::array<u64, 5>();
    for (u64 i = 0; i != 5; ++i)
        if (auto l = Lit::isa<u64>(res->proj(5, i)))
            flds[i] = *l;
        else
            return {};
    return flds;
}

/// Whether the reduction point at the raw i64 loop counters `iv` (of the whole loop vector over `sr`, the leading `ro`
/// parallel) is one a schedule's `tail` excludes: the last iteration of the innermost looped reduction dim, past the
/// first `tail` points of the first unrolled dim.
/// `nullptr` if the schedule excludes none.
inline const Def* tail_skip(const Def* sched, u64 ro, Defs sr, Defs iv) {
    auto& w    = sched->world();
    auto flds  = sched_fields(sched);
    auto r     = sr.size();
    if (!flds || (*flds)[2] == 0) return nullptr;
    auto [vdim, unroll, tail, pout, rout] = *flds;
    auto u_lo = std::max(ro, r - std::min(r, unroll));
    if (u_lo <= ro || u_lo >= r) return nullptr;
    auto last = Lit::isa<u64>(sr[u_lo - 1]);
    if (!last) return nullptr;
    auto at_last = w.call(core::icmp::e, w.tuple({iv[u_lo - 1], w.lit_i64(*last - 1)}));
    auto past    = w.call(core::icmp::uge, w.tuple({iv[u_lo], w.lit_i64(tail)}));
    return w.call(core::bit2::and_, w.lit_nat(2), w.tuple({at_last, past}));
}

} // namespace mim::plug::btensor
