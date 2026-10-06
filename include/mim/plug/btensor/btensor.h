#pragma once

#include <algorithm>
#include <array>
#include <optional>

#include <mim/lam.h>
#include <mim/world.h>

#include <mim/plug/affine/affine.h>
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
    if (auto mr = Axm::isa<btensor::map_reduce_iter>(d)) return mr;
    return Axm::isa<btensor::map_reduce_masked>(d);
}

/// Common shape fields; masked operations append spatial write and fold predicates.
inline std::array<const Def*, 3> mr_shapes(const Def* shapes) {
    auto n = shapes->num_projs();
    return {shapes->proj(n, 0), shapes->proj(n, 1), shapes->proj(n, 2)};
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

/// Evaluate composed partition constraints with unbounded i64 results. Do not narrow to
/// limits before comparing: a padded coordinate must remain out of bounds. Unsigned comparison also
/// rejects negative affine coordinates. Write-back checks only spatial coordinates, not reduction ones.
inline std::pair<const Def*, const Def*> predicate_skip(const Def* shapes,
                                                        const Def* Sr,
                                                        const Def* idxs,
                                                        const Def* mem,
                                                        bool write,
                                                        std::optional<u64> peeled_rp = {}) {
    if (shapes->num_projs() != 5) return {mem, nullptr};
    auto& w        = mem->world();
    auto predicate = shapes->proj(5, write ? 3 : 4);
    auto nv        = Lit::isa<u64>(predicate->proj(3, 0));
    if (!nv) fe::throwf("masked map-reduce needs literal predicate rank");
    if (!*nv) return {mem, nullptr};
    auto r = Sr->num_projs();
    DefSet covered;
    // CPU nests already peel this exact reduction constraint. Prove its map and bound match,
    // then avoid repeating it in every SSA accumulator cell. Other backends keep their guards.
    if (!write && peeled_rp) {
        auto fields = sched_fields(shapes->proj(5, 2));
        auto lm     = predicate->proj(3, 2)->isa_mut<Lam>();
        if (fields && lm && lm->is_set() && (*fields)[2]) {
            auto lo = std::max(*peeled_rp, r - std::min(r, (*fields)[1]));
            if (*peeled_rp < lo && lo < r) {
                auto outer = Lit::isa<u64>(Sr->proj(r, lo - 1));
                auto inner = Lit::isa<u64>(Sr->proj(r, lo));
                auto tail  = (*fields)[2];
                if (outer && inner && *outer && *inner && tail <= *inner && *outer - 1 <= (u64(-1) - tail) / *inner) {
                    auto mul = w.call(affine::semiop::mul, w.tuple({lm->var()->proj(r, lo - 1), w.lit_nat(*inner)}));
                    auto coordinate = w.call(affine::op::add, w.tuple({mul, lm->var()->proj(r, lo)}));
                    for (u64 i = 0; i != *nv; ++i)
                        if (Lit::isa<u64>(predicate->proj(3, 1)->proj(*nv, i)) == (*outer - 1) * *inner + tail
                            && lm->body()->proj(*nv, i) == coordinate)
                            covered.insert(w.lit_nat(i));
                }
            }
        }
    }
    if (covered.size() == *nv) return {mem, nullptr};
    auto m = w.lit_nat(*nv), n = w.lit_nat(r);
    auto unbounded    = w.tuple(DefVec(*nv, [&](size_t) { return w.lit_nat_0(); }));
    auto map          = w.app(w.annex<affine::map>(), {m, n});
    map               = w.app(map, {Sr, unbounded});
    map               = w.app(map, predicate->proj(3, 2));
    map               = w.app(map, idxs);
    map               = w.app(map, mem->type()->as<App>()->arg());
    auto [pm, coords] = w.app(map, mem)->projs<2>();
    auto limits       = predicate->proj(3, 1);
    const Def* skip   = nullptr;
    for (u64 i = 0; i != *nv; ++i) {
        if (covered.contains(w.lit_nat(i))) continue;
        auto bound   = w.call<core::bitcast>(w.type_i64(), limits->proj(*nv, i));
        auto invalid = w.call(core::icmp::uge, w.tuple({coords->proj(*nv, i), bound}));
        skip         = skip ? w.call(core::bit2::or_, w.lit_nat(2), w.tuple({skip, invalid})) : invalid;
    }
    return {pm, skip};
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
