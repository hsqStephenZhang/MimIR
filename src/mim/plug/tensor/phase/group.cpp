#include "mim/plug/tensor/phase/group.h"

#include <limits>

#include <mim/pattern/graph.h>
#include <mim/pattern/witness.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {
pattern::WitnessResult grouping_candidate(const pattern::GraphView&, Defs, World&);
} // namespace mim::plug::tensor

namespace mim::plug::tensor::phase {

namespace {

std::optional<u64> product(u64 a, u64 b) {
    if (b && a > std::numeric_limits<u64>::max() / b) return {};
    return a * b;
}

std::optional<u64> cells(const Def* shape, u64 rank) {
    u64 n = 1;
    for (u64 d = 0; d != rank; ++d) {
        auto extent = Lit::isa(shape->proj(rank, d));
        auto next   = extent ? product(n, *extent) : std::nullopt;
        if (!next) return {};
        n = *next;
    }
    return n;
}

std::optional<u64> bytes(const Def* type) {
    return Lit::isa(type->world().app(type->world().annex(core::trait::size), type));
}

bool root(const Def* op) {
    auto& w = op->world();
    return w.app(w.app(w.annex<group_root>(), op->type()), op) != w.lit_ff();
}

bool canonical(const btensor::MapRedPost& mr) {
    auto& w = mr.So->world();
    if (!Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>())) return false;
    auto record = w.app(w.annex<tile_sched>(), mr.sched);
    return Check::test_alpha(mr.sched, w.app(w.annex<btensor::mk_sched>(), record));
}

const Def*
box(flags_t flags, const Def* map, const Def* n, const Def* r, const Def* off, const Def* ext, const Def* shape) {
    auto& w = map->world();
    return w.app(w.app(w.app(w.annex(flags), {n, r}), map), {off, ext, shape});
}

struct Cost {
    Cost(World& world, u64 factor)
        : w(world)
        , budget(factor) {}

    World& w;
    u64 budget;
    DefMap<u64> work;
    DefSet active;
    DefSet boundaries;
    u64 scratch = 0, visits = 0;
    bool exact = true;
    std::string reason;

    bool add(u64 value) {
        if (scratch > std::numeric_limits<u64>::max() - value) return fail("scratch arithmetic overflow");
        scratch += value;
        return true;
    }

    bool fail(std::string why) {
        reason = std::move(why);
        return false;
    }

    bool storage(const Def* type, const Def* shape, u64 rank) {
        auto size = bytes(type), count = cells(shape, rank);
        auto total = size && count ? product(*size, *count) : std::nullopt;
        return total ? add(*total) : fail("nonliteral storage size or arithmetic overflow");
    }

    bool visit(const Def* op, const Def* off, const Def* ext, u64 depth = 0) {
        if (++visits > 4096 || depth == 32 || !active.insert(op).second)
            return fail("group geometry exceeds analysis limit");
        auto mr    = btensor::MapRedPost::of(op->as<App>()->callee()->as<App>());
        auto ranks = mr.ranks();
        if (!ranks || ranks->ro > 8 || ranks->nloops() > 64 || !canonical(mr) || !pure_scalar(mr.comb)
            || !pure_scalar(mr.post))
            return fail("group member lacks literal ranks, canonical schedule, or scalar purity");
        auto n = cells(ext, ranks->ro);
        if (!n || work[op] > std::numeric_limits<u64>::max() - *n) return fail("work arithmetic overflow");
        work[op] += *n;
        if (!storage(mr.Tp, ext, ranks->ro)) return false;
        if (mr.To != mr.Tp && !storage(mr.To, ext, ranks->ro)) return false;
        auto pre = box(Annex::base<affine::preimage>(), mr.map_out, mr.Rn, mr.Ro, off, ext, mr.Sr);
        if (pre->proj(4, 3) != w.lit_tt()) return fail("write preimage is not exact");
        auto lo = pre->proj(4, 0), size = pre->proj(4, 1);
        for (u64 d = ranks->ro; d != ranks->nloops(); ++d)
            if (Lit::isa(lo->proj(ranks->nloops(), d)) != 0
                || size->proj(ranks->nloops(), d) != mr.Sr->proj(ranks->nloops(), d))
                return fail("tile cuts a reduction dimension");
        auto [inputs, posts] = op->as<App>()->args<2>();
        if (!reads(inputs, mr.nis, mr.Tis, mr.Ris, mr.Sis, mr.maps, mr.Rn, lo, size, depth)
            || !reads(posts, mr.nps, mr.Tps, mr.Rps, mr.Sps, mr.post_maps, mr.Ro, off, ext, depth))
            return false;
        if (auto init = tensor_init(mr))
            if (!storage(mr.To, init->shape, ranks->ro)) return false;
        active.erase(op);
        return true;
    }

    bool reads(const Def* values,
               const Def* count,
               const Def* types,
               const Def* ranks,
               const Def* shapes,
               const Def* maps,
               const Def* n,
               const Def* off,
               const Def* ext,
               u64 depth) {
        auto num = Lit::isa(count);
        if (!num) return fail("input count is not literal");
        for (u64 i = 0; i != *num; ++i) {
            auto rank = ranks->proj(*num, i);
            auto r    = Lit::isa(rank);
            if (!r) return fail("input rank is not literal");
            auto image
                = box(Annex::base<affine::image>(), maps->proj(*num, i), n, rank, off, ext, shapes->proj(*num, i));
            auto [start, extent, cap, axes] = image->projs<4>();
            if (!cells(cap, *r)) return fail("input box has no literal cap");
            for (u64 d = 0; d != *r; ++d)
                exact &= axes->proj(*r, d) == w.lit_tt();
            auto input = values->proj(*num, i);
            auto p     = graph::map_reduce(input);
            if (p && !boundaries.contains(p) && (budget > 1 || !root(p))) {
                if (cap != extent && !storage(types->proj(*num, i), cap, *r)) return false;
                if (!visit(p, start, extent, depth + 1)) return false;
            } else if (!storage(types->proj(*num, i), cap, *r)) {
                return false;
            }
        }
        return true;
    }
};

struct Choice {
    u64 dim = 0, tile = 0, scratch = 0;
    long double factor = 0;
    bool exact         = false;
    DefVec members;
};

class Planner {
public:
    Planner(const pattern::GraphView& source, World& w, u64 cache, u64 budget, const fe::Log& log)
        : source_(source)
        , w_(w)
        , importer_(w)
        , cache_(cache)
        , budget_(budget)
        , log_(log) {}

    const Def* import(const Def* value) {
        auto result = importer_.rewrite(value);
        if (graph::map_reduce(value)) originals_[result] = value;
        return result;
    }

    std::optional<Choice> choose(const Def* original, const Def* consumers = nullptr) {
        auto op    = import(original);
        auto mr    = btensor::MapRedPost::of(op->as<App>()->callee()->as<App>());
        auto ranks = mr.ranks();
        auto shape = ranks ? lit_projs(mr.So, ranks->ro) : std::nullopt;
        if (!ranks || !ranks->ro || ranks->ro > 8 || !shape || !canonical(mr)) {
            log_.d("tensor.group skip {}: nonliteral shape/rank or custom schedule", original);
            return {};
        }
        if (consumers) {
            auto domain = lit_projs(mr.Sr, ranks->nloops());
            if (!domain) return {};
            for (u64 d = 0; d != ranks->ro; ++d)
                (*shape)[d] = (*domain)[d];
        }
        std::optional<Choice> best;
        for (u64 dim = 0; dim != ranks->ro; ++dim) {
            auto width = (*shape)[dim];
            if (!width) continue;
            fe::Vector<u64> sizes;
            auto limit = consumers ? width / 2 : width;
            for (u64 t = 1; t <= std::min<u64>(limit, 64); ++t)
                sizes.push_back(t);
            for (u64 t = 128; t < limit && t <= std::numeric_limits<u64>::max() / 2; t *= 2)
                sizes.push_back(t);
            if (limit > 64) sizes.push_back(limit);
            for (auto tile : sizes) {
                auto result = estimate(original, op, dim, tile, consumers);
                if (!result) continue;
                auto better = !best
                           || (result->exact != best->exact     ? result->exact
                               : result->factor != best->factor ? result->factor < best->factor
                               : result->tile != best->tile     ? result->tile > best->tile
                                                                : result->dim < best->dim);
                if (better) best = result;
            }
        }
        if (best) {
            log_.d("tensor.group choose {}: kind={} dim={} tile={} scratch={} exact={} factor={} members={}", original,
                   consumers ? 2 : 0, best->dim, best->tile, best->scratch, best->exact, double(best->factor),
                   best->members.size());
            for (auto member : best->members) {
                auto source = fe::lookup(originals_, member);
                log_.d("tensor.group member {} in group {}", source ? source : member, original);
            }
        }
        return best;
    }

private:
    std::optional<Choice> estimate(const Def* original, const Def* op, u64 dim, u64 tile, const Def* consumers) {
        auto reject = [&](std::string_view reason) -> std::optional<Choice> {
            log_.d("tensor.group alternative {}: dim={} tile={} rejected: {}", original, dim, tile, reason);
            return {};
        };
        auto mr    = btensor::MapRedPost::of(op->as<App>()->callee()->as<App>());
        auto ranks = *mr.ranks();
        auto shape = *lit_projs(mr.So, ranks.ro);
        u64 trips  = shape[dim] / tile + (shape[dim] % tile != 0);
        std::optional<MultiTilePlan> reverse;
        if (consumers) {
            auto domain = *lit_projs(mr.Sr, ranks.nloops());
            trips       = domain[dim] / tile + (domain[dim] % tile != 0);
            for (u64 d = 0; d < dim; ++d) {
                auto n = product(trips, domain[d]);
                if (!n) return reject("grid arithmetic overflow");
                trips = *n;
            }
            if (!trips || trips > 256) return reject("grid exceeds 256 cost samples");
            auto geometry = reverse_tiles(op, consumers, w_.lit_nat(dim), w_.lit_nat(tile));
            if (!geometry.value) return reject(geometry.reason);
            auto tiled = geometry.value->proj(2, 0)->as<Extract>()->tuple()->as<App>();
            reverse    = multi_tile_plan(tiled);
        }
        if (!trips || trips > 256) return reject("grid exceeds 256 cost samples");
        Cost cost(w_, budget_);
        u64 peak = 0;
        for (u64 point = 0; point != trips; ++point) {
            auto offset  = DefVec(ranks.ro, w_.lit_nat_0());
            auto extent  = DefVec(mr.So->projs(ranks.ro));
            auto parts   = shape[dim] / tile + (shape[dim] % tile != 0);
            auto start   = point % parts * tile;
            offset[dim]  = w_.lit_nat(start);
            extent[dim]  = w_.lit_nat(std::min(tile, shape[dim] - start));
            cost.scratch = 0;
            if (reverse) {
                auto [off, ext] = reverse->owned[point][0]->projs<2>();
                if (!cost.visit(op, off, ext)) return reject(cost.reason);
                cost.boundaries.insert(op);
                auto cs = consumers->isa<Tuple>() ? DefVec(consumers->ops().begin(), consumers->ops().end())
                                                  : DefVec{consumers};
                for (size_t j = 0; j != cs.size(); ++j) {
                    auto [coff, cext] = reverse->owned[point][j + 1]->projs<2>();
                    if (!cost.visit(cs[j], coff, cext)) return reject(cost.reason);
                }
            } else if (!cost.visit(op, w_.tuple(offset), w_.tuple(extent))) {
                return reject(cost.reason);
            }
            peak = std::max(peak, cost.scratch);
        }
        if (peak > cache_) return reject("scratch exceeds cache budget");
        Choice result{.dim = dim, .tile = tile, .scratch = peak, .factor = 1, .exact = cost.exact, .members = {}};
        for (auto [member, work] : cost.work) {
            auto m     = btensor::MapRedPost::of(member->as<App>()->callee()->as<App>());
            auto whole = cells(m.So, Lit::as(m.Ro));
            if (!whole || !*whole) return reject("empty or nonliteral group member");
            auto selected = member == op
                         || (consumers
                             && (member == consumers
                                 || (consumers->isa<Tuple>()
                                     && std::ranges::find(consumers->ops(), member) != consumers->ops().end())));
            if (!selected) {
                auto original_member = fe::lookup(originals_, member);
                if (!original_member || source_.complete_data_uses(original_member) != pattern::CheckResult::True)
                    return reject("group member has unproven external data uses");
                DefSet readers;
                for (auto use : source_.data_uses(original_member)) {
                    if (cost.work.contains(import(use.consumer)) || !readers.insert(use.consumer).second) continue;
                    for (auto occurrence : source_.occurrences(use.consumer)) {
                        auto extra = occurrence.trips ? product(*whole, *occurrence.trips) : std::nullopt;
                        if (!extra || work > std::numeric_limits<u64>::max() - *extra)
                            return reject("outside reader has unknown trips or work overflow");
                        work += *extra;
                    }
                }
            }
            auto max = product(*whole, budget_);
            if (!max || work > *max) return reject("producer recomputation exceeds budget");
            result.factor = std::max(result.factor, static_cast<long double>(work) / *whole);
            result.members.push_back(member);
        }
        log_.d("tensor.group alternative {}: dim={} tile={} scratch={} exact={} factor={} accepted", original, dim,
               tile, peak, result.exact, double(result.factor));
        return result;
    }

    const pattern::GraphView& source_;
    World& w_;
    Rewriter importer_;
    DefMap<const Def*> originals_;
    u64 cache_, budget_;
    const fe::Log& log_;
};

} // namespace

void Group::start() {
    if (!driver().arg_nat("tensor", "group").value_or(1)) return RWPhase::start();
    auto cache     = driver().arg_nat("tensor", "group-cache").value_or(32768);
    auto registers = driver().arg_nat("tensor", "group-registers").value_or(32);
    auto budget    = driver().arg_nat("tensor", "group-budget").value_or(1);
    if (!cache || registers < 16 || !budget)
        fe::throwf("tensor.group requires positive cache/budget and at least 16 registers");
    pattern::OpRegistry ops;
    graph::register_operations(ops);
    pattern::GraphView source(old_world(), ops);
    auto scratch = old_world().inherit();
    Planner planner(source, *scratch, cache, budget, log());
    DefSet claimed;
    for (auto op : source.operations())
        for (auto occurrence : source.occurrences(op))
            if (!occurrence.regions.empty()) claimed.insert(op);
    for (auto op : source.operations())
        if (auto request = Axm::isa<group_req>(op)) {
            auto [value, kind, dim, tile, params] = request->args<5>();
            while (auto inner = Axm::isa<group_req>(value))
                value = inner->arg(5, 0);
            claimed.insert(value);
            if (Lit::isa(kind) == 2) {
                if (auto tuple = params->isa<Tuple>())
                    for (auto c : tuple->ops())
                        claimed.insert(c);
                else
                    claimed.insert(params);
            }
        }
    for (auto op : source.operations()) {
        if (!graph::map_reduce(op) || claimed.contains(op)) continue;
        auto imported = planner.import(op);
        auto mr       = btensor::MapRedPost::of(imported->as<App>()->callee()->as<App>());
        if (auto ranks = mr.ranks();
            ranks && ranks->rr && canonical(mr) && root(imported) && pure_scalar(mr.comb) && pure_scalar(mr.post)) {
            auto size  = bytes(mr.To);
            auto lanes = scratch->app(scratch->annex<vec_width>(), scratch->lit_nat(8));
            if (!size || !*size) continue;
            auto config
                = scratch->app(scratch->annex<group_blocks>(),
                               {scratch->lit_nat(*size), lanes, scratch->lit_nat(registers), scratch->lit_nat(cache)});
            auto [mr_block, nr, kc, nc] = config->projs<4>();
            if (Lit::isa(mr_block).value_or(0) == 0 || Lit::isa(kc).value_or(0) == 0) {
                log().d("tensor.group skip contraction {}: cache is too small for target BLIS blocks", op);
                continue;
            }
            requests_[op].push_back({1, 0, Lit::as(mr_block), scratch->tuple({nr, kc, nc}), {}});
            log().d("tensor.group contraction {}: MR={} NR={} KC={} NC={}", op, mr_block, nr, kc, nc);
        }
    }
    for (auto op : source.operations()) {
        if (!graph::map_reduce(op) || claimed.contains(op)) continue;
        DefVec consumers;
        for (auto use : source.data_uses(op))
            if (graph::map_reduce(use.consumer) && !claimed.contains(use.consumer)
                && std::ranges::find(consumers, use.consumer) == consumers.end()) {
                bool scope = true;
                for (auto var : use.consumer->free_vars())
                    scope &= op->has_free_var(var);
                if (scope) consumers.push_back(use.consumer);
            }
        if (consumers.empty()) continue;
        std::ranges::sort(consumers, {}, &Def::gid);
        auto cs     = scratch->tuple(DefVec(consumers.size(), [&](size_t i) { return planner.import(consumers[i]); }));
        auto choice = planner.choose(op, cs);
        if (!choice && consumers.size() > 1)
            for (auto c : consumers)
                if (auto single = planner.choose(op, planner.import(c))) {
                    if (!choice || single->tile > choice->tile) {
                        choice = single;
                        cs     = planner.import(c);
                    }
                }
        if (choice) {
            DefVec selected;
            for (auto c : consumers)
                if (cs == planner.import(c) || cs->isa<Tuple>()) selected.push_back(c);
            requests_[op].push_back({2, choice->dim, choice->tile, nullptr, selected});
            claimed.insert(op);
            for (auto c : selected)
                claimed.insert(c);
        }
    }
    for (auto op : source.operations()) {
        if (!graph::map_reduce(op) || claimed.contains(op) || requests_.contains(op)) continue;
        auto selected = grouping_candidate(source, DefVec{op, scratch->lit_nat_1()}, *scratch);
        if (selected.status != pattern::CheckResult::True || selected.value != scratch->lit_tt()) continue;
        if (auto choice = planner.choose(op))
            requests_[op].push_back({0, choice->dim, choice->tile, scratch->lit_nat(budget), {}});
    }
    RWPhase::start();
}

const Def* Group::rewrite_imm_App(const App* app) {
    auto value = RWPhase::rewrite_imm_App(app);
    auto found = requests_.find(app);
    if (found == requests_.end()) return value;
    value->set(app->dbg_key());
    auto& w = new_world();
    for (auto request : found->second) {
        Rewriter metadata(w);
        metadata.map(app, value);
        const Def* params;
        if (request.kind == 2) {
            for (auto c : request.consumers)
                for (auto var : c->free_vars())
                    metadata.map(var, rewrite(var));
            params = w.tuple(metadata.rewrite(request.consumers));
        } else {
            params = metadata.rewrite(request.params);
        }
        auto marker = w.app(w.annex<group_req>(), {value->type(), params->type()});
        value = w.app(marker, {value, w.lit_nat(request.kind), w.lit_nat(request.dim), w.lit_nat(request.tile), params})
                    ->set(app->dbg_key());
        profile_count("tensor.group.requests");
        invalidate();
    }
    return value;
}

void report_group_request(const App* app, std::string_view reason) {
    auto [op, kind, dim, tile, params] = app->args<5>();
    auto& w                            = app->world();
    if (w.driver().arg_nat("tensor", "strict").value_or(0))
        app->blame("unrealized tensor.group_req kind={} dim={} tile={} at {}: {}", kind, dim, tile, app->err_loc(),
                   reason)
            .bail();
    auto report = w.log();
    report.set(fe::Log::Level::Warn)
        .w("unrealized tensor.group_req kind={} dim={} tile={} at {}: {}", kind, dim, tile, app->err_loc(), reason);
}

const Def* ReportGroups::rewrite_imm_App(const App* app) {
    if (Axm::isa<group_req>(app)) {
        report_group_request(app, "no in-group rule realized this request");
        return rewrite(app->arg(5, 0));
    }
    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
