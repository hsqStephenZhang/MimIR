#include "mim/plug/tensor/phase/group.h"

#include <mim/driver.h>
#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/btensor/btensor.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

using F = btensor::MapRedPost;

F mr_of(const App* app) { return F::of(app->callee()->as<App>()); }

/// Whether the schedule of @p mr places inputs itself, by `compute_at`, `tile_and_fuse` or a default schedule's.
bool places(const F& mr) {
    auto& w = mr.sched->world();
    return mr.sched_field(F::At) == w.lit_tt() || mr.sched_field(F::Fuse) == w.lit_tt();
}

/// Whether a reduction of @p mr reads an input that misses a parallel dim, as a matrix product reads `A` at `(i, k)`.
bool contracts(const F& mr, const F::Ranks& rks) {
    if (rks.rr == 0) return false;
    auto& w = mr.sched->world();
    auto rn = rks.nloops();
    for (u64 k = 0; k != rks.nis; ++k) {
        auto map  = mr.maps->proj(rks.nis, k);
        auto pi   = map->type()->as<Pi>();
        auto o    = w.mut_lam(pi->dom(), pi->codom())->var();
        auto read = w.app(map, o);
        for (u64 d = 0; d != rks.rp; ++d) {
            auto at = DefVec(rn, [&](size_t e) -> const Def* {
                return e == d ? w.app(w.annex<affine::lit>(), w.lit_nat_0()) : o->proj(rn, e);
            });
            if (w.app(map, w.tuple(at)) == read) return true;
        }
    }
    return false;
}

/// A root of its own: a contraction, an op placing its own inputs, or one keeping copies from call to call.
bool root(const F& mr) {
    auto rks = mr.ranks();
    return !rks || contracts(mr, *rks) || places(mr) || mr.mask(F::Prepacks) != 0;
}

/// The dim of the outermost loop of @p mr's nest among its parallel dims, unless that loop is its vector loop.
std::optional<u64> outermost(const F& mr) {
    auto rks = mr.ranks();
    auto rn  = rks->nloops();
    for (u64 d = 0; d != rks->rp; ++d) {
        if (std::ranges::count(mr.outer(d, rn), true) != 1) continue;
        if (Lit::isa<u64>(mr.sched_field(F::VDim)) == d) return {};
        return d;
    }
    return {};
}

/// The producers of the inputs of @p app that its nest does not copy.
std::vector<std::pair<u64, const App*>> producers(const App* app) {
    auto mr    = mr_of(app);
    auto res   = std::vector<std::pair<u64, const App*>>();
    auto rks   = mr.ranks();
    auto cache = mr.mask(F::Cache);
    if (!rks) return res;
    auto is = app->arg()->proj(2, 0);
    for (u64 k = 0; k != rks->nis; ++k)
        if (auto p = Axm::isa<tensor::map_reduce_post>(is->proj(rks->nis, k)); p && !btensor::has_slot(cache, k))
            res.emplace_back(k, p);
    return res;
}

} // namespace

void Group::start() {
    // The GPU lowering runs no schedule actions, so it would leave every group to the CPU lowering.
    if (old_world().driver().is_loaded("gpu")) return RWPhase::start();

    auto ops = DefVec();
    auto wl  = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    while (!wl.empty()) {
        auto def = wl.pop();
        if (Axm::isa<tensor::map_reduce_post>(def)) ops.push_back(def);
        for (auto op : def->ops())
            if (op) wl.push(op);
        if (def->type()) wl.push(def->type());
    }

    // A producer that a schedule's tiles take in follows that schedule instead of grouping its own producers.
    // Another group may still leave it whole, so its own group stands, inside that group's tiles where they take it.
    for (auto def : ops) {
        auto app   = def->as<App>();
        auto mr    = mr_of(app);
        auto fuse  = mr.sched_field(F::Fuse) == mr.sched->world().lit_tt();
        auto slots = mr.sched_field(F::At) == mr.sched->world().lit_tt() ? mr.mask(F::AtSlots) : 0;
        for (auto [k, p] : producers(app))
            if (fuse || btensor::has_slot(slots, k)) taken_.emplace(p);
    }

    for (auto def : ops)
        if (auto d = grouped(def->as<App>())) roots_[def] = *d;
    RWPhase::start();
}

std::optional<u64> Group::grouped(const App* app) {
    if (auto d = dims_.find(app); d != dims_.end()) return d->second;
    auto mr = mr_of(app);
    auto d  = std::optional<u64>();
    if (mr.ranks() && !places(mr) && !taken_.contains(app))
        if (auto ps = producers(app);
            !ps.empty() && std::ranges::none_of(ps, [&](auto p) { return reaches(p.second); }))
            d = outermost(mr);
    return dims_[app] = d;
}

bool Group::reaches(const App* p) {
    if (auto r = reach_.find(p); r != reach_.end()) return r->second;
    if (root(mr_of(p))) return reach_[p] = true;
    return reach_[p] = !grouped(p) && std::ranges::any_of(producers(p), [&](auto q) { return reaches(q.second); });
}

const Def* Group::rewrite_imm_App(const App* app) {
    auto res = RWPhase::rewrite_imm_App(app);
    auto d   = fe::lookup(roots_, app);
    if (!d) return res;

    // tile_and_fuse at the outermost loop within a budget of 1: a producer joins only where its tiles compute it once.
    auto& w            = new_world();
    auto new_mr        = res->as<App>();
    auto mr            = mr_of(new_mr);
    auto rec           = mr.sched_record();
    auto fields        = DefVec(F::Num_Fields, [&](size_t f) { return rec->proj(F::Num_Fields, f); });
    auto at_dim        = w.mut_lam(w.pi(w.type_nat(), w.type_nat()))->set("at_dim");
    fields[F::At]      = w.lit_tt();
    fields[F::Fuse]    = w.lit_tt();
    fields[F::Budget]  = w.lit_nat_1();
    fields[F::AtDim]   = at_dim->set(true, w.lit_nat(*d));
    auto shapes        = w.tuple({mr.So, mr.Sr, w.app(w.annex<btensor::mk_sched>(), w.tuple(fields))});
    const Def* grouped = w.annex<tensor::map_reduce_post>();
    for (auto a : {mr.nis_nps, mr.meta, shapes, mr.in_tys, mr.comb_init, mr.map_out, mr.maps_all})
        grouped = w.app(grouped, a);
    log().d("group the producers of {} into the tiles of its dim {}", app, *d);
    return w.app(grouped, new_mr->arg());
}

} // namespace mim::plug::tensor::phase
