#include "mim/plug/tensor/phase/group_rules.h"

#include <algorithm>
#include <limits>

#include <mim/pattern/witness.h>
#include <mim/phase/graph_rule_rewrite.h>
#include <mim/phase/rule_rewrite.h>

#include "mim/plug/tensor/phase/fuse.h"
#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/phase/tile_fuse.h"
#include "mim/plug/tensor/phase/tile_group.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

pattern::WitnessResult fusion_budget(const pattern::GraphView&, Defs, World&);

namespace {

class LocalRewriter : public Rewriter {
public:
    using Rewriter::Rewriter;

    const Def* rewrite(const Def* def) override {
        if (auto value = lookup(def)) return value;
        if (def->is_closed()) {
            bool substituted = false;
            for (auto mut : def->local_muts())
                if (auto value = lookup(mut); value && value != mut) substituted = true;
            if (!substituted) return map(def, def);
        }
        return Rewriter::rewrite(def);
    }
};

} // namespace

ProducerTile blis_group(const Def* producer, const Def* tile, const Def* params) {
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "BLIS needs a map_reduce_post"};
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || ranks->ro < 2 || !ranks->rr || ranks->nis != 2)
        return {nullptr, "BLIS needs two matrix parallel axes and two inputs"};
    auto& w = producer->world();
    auto nr = lit_projs(mr.Ris, 2);
    if (!nr || !Check::test_alpha(mr.To, mr.Tp) || !Check::test_alpha(mr.init->type(), mr.To))
        return {nullptr, "BLIS needs literal input ranks and a scalar accumulator of the result type"};
    auto sizes = lit_projs(mr.Sr, ranks->nloops());
    auto shape = lit_projs(mr.So, ranks->ro);
    if (!sizes || !shape || !std::equal(shape->begin(), shape->end(), sizes->begin()))
        return {nullptr, "BLIS needs literal, untransformed parallel extents"};
    auto batch = ranks->ro - 2;
    auto m = (*shape)[batch], n = (*shape)[batch + 1], k = u64(1), batches = u64(1);
    for (u64 d = 0; d != batch; ++d) {
        if (!(*shape)[d] || batches > std::numeric_limits<u64>::max() / (*shape)[d])
            return {nullptr, "BLIS needs a nonempty batch grid with a representable size"};
        batches *= (*shape)[d];
    }
    for (u64 d = ranks->ro; d != ranks->nloops(); ++d) {
        auto extent = (*sizes)[d];
        if (!extent || k > std::numeric_limits<u64>::max() / extent)
            return {nullptr, "BLIS needs a nonempty literal reduction with a representable size"};
        k *= extent;
    }
    if (!m || !n) return {nullptr, "BLIS needs nonempty output axes"};
    if (!Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>()))
        return {nullptr, "BLIS cannot replace a custom schedule"};
    auto sched = w.app(w.annex<tile_sched>(), mr.sched);
    if (!Check::test_alpha(mr.sched, w.app(w.annex<btensor::mk_sched>(), sched)))
        return {nullptr, "BLIS cannot replace a clipped or placed schedule"};
    if (!pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {nullptr, "BLIS needs pure scalar combiner and epilogue"};
    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto fn     = w.mut_lam(domain, domain);
    auto vars   = fn->var()->projs(ranks->nloops());
    auto output = w.app(mr.map_out, fn->var());
    auto maps   = mr.maps->projs<2>();
    auto shapes = mr.Sis->projs<2>();
    if (output != w.tuple(Defs(vars.data(), ranks->ro))) return {nullptr, "BLIS needs identity output coordinates"};
    for (size_t i = 0; i != 2; ++i) {
        auto independent           = vars;
        independent[batch + 1 - i] = w.call<affine::lit>(w.lit_nat_0());
        if (!Check::test_alpha(w.app(maps[i], fn->var()), w.app(maps[i], w.tuple(independent))))
            return {nullptr, "BLIS operand reads the other matrix parallel axis"};
        if (!lit_projs(shapes[i], (*nr)[i])) return {nullptr, "BLIS needs literal physical operand shapes"};
    }
    auto blocks = lit_projs(params, 3);
    auto rows   = Lit::isa(tile);
    if (!blocks || !rows) return {nullptr, "BLIS block parameters must be literal"};
    auto lanes   = Lit::as(w.app(w.annex<vec_width>(), w.lit_nat(4)));
    auto mr_tile = *rows ? *rows : std::min(u64(5), lanes);
    auto nr_tile = (*blocks)[0] ? (*blocks)[0] : lanes * 4;
    auto nc      = (*blocks)[2] ? (*blocks)[2] : (nr_tile > 256 / (lanes * 4) ? 256 : nr_tile * lanes * 4);
    auto bytes   = Lit::isa(w.app(w.annex(core::trait::size), mr.To));
    if (!bytes || !*bytes || !nc || nc > std::numeric_limits<u64>::max() / 2 / *bytes)
        return {nullptr, "BLIS needs a literal element size and representable block storage"};
    auto kc = (*blocks)[1] ? (*blocks)[1] : std::min(u64(262144) / (nc * *bytes * 2), u64(256));
    if (!mr_tile || !nr_tile || !nc || !kc)
        return {nullptr, "BLIS block parameters must be positive after target selection"};
    if (n > std::numeric_limits<u64>::max() - (nr_tile - 1)
        || (n + nr_tile - 1) / nr_tile > std::numeric_limits<u64>::max() / nr_tile / k)
        return {nullptr, "BLIS packed panel size is not representable"};
    auto types = mr.Tis->projs<2>();
    DefVec reads(2);
    for (size_t i = 0; i != 2; ++i) {
        auto read   = w.mut_lam(w.arr(ranks->ro, w.annex<affine::Index>()), w.arr((*nr)[i], w.annex<affine::Index>()));
        auto coords = read->var()->projs(ranks->ro);
        auto red    = coords[batch + 1 - i];
        coords[batch + 1 - i] = w.call<affine::lit>(w.lit_nat_0());
        coords.resize(ranks->nloops());
        for (u64 d = ranks->nloops(); d-- != ranks->ro;) {
            auto extent = mr.Sr->proj(ranks->nloops(), d);
            coords[d]   = d == ranks->ro ? red : w.call(affine::semiop::rem, Defs{red, extent});
            if (d != ranks->ro) red = w.call(affine::semiop::floordiv, Defs{red, extent});
        }
        read->set(true, w.app(maps[i], w.tuple(coords)));
        reads[i] = read;
    }
    auto a_read   = w.app(maps[0], fn->var());
    auto direct_a = batch == 0 && ranks->rr == 1 && (*nr)[0] == 2 && a_read == w.tuple({vars[0], vars[2]});
    auto in_place = [&](size_t i) {
        return (*nr)[i] == 2 && reads[i]->as<Lam>()->body() == reads[i]->as<Lam>()->has_var();
    };
    if (batch == 0) {
        // A copied by blis_operand is already row-major (M, K) and needs no second pack.
        auto pack_a  = !direct_a && in_place(0);
        auto builder = w.app(w.annex<group_blis_impl>(), {mr.To, types[0], types[1], mr.nps, mr.Tps, mr.Rps, mr.Sps});
        builder      = w.app(builder, {w.lit_nat(m), w.lit_nat(k), w.lit_nat(n), w.lit_nat(mr_tile), w.lit_nat(nr_tile),
                                       w.lit_nat(nc), w.lit_nat(kc), w.lit_bool(pack_a)});
        builder      = w.app(builder, {mr.comb, mr.init, mr.post, mr.post_maps});
        auto [inputs, extras] = app->args<2>();
        auto data             = inputs->projs<2>();
        for (size_t i = 0; i != 2; ++i) {
            if (in_place(i)) continue;
            auto view = w.app(w.annex<blis_operand>(), {types[i], w.lit_nat((*nr)[i])});
            view      = w.app(view, shapes[i]);
            view      = w.app(view, {w.tuple({w.lit_nat(i ? k : m), w.lit_nat(i ? n : k)}), reads[i]});
            data[i]   = w.app(view, data[i]);
        }
        return {w.app(builder, {data[0], data[1], extras}), {}};
    }
    auto builder = w.app(w.annex<group_blis_batch_impl>(),
                         {mr.To, types[0], types[1], mr.nps, mr.Tps, mr.Rps, mr.Sps, w.lit_nat(batch), mr.Ris});
    builder      = w.app(builder, {mr.Sis, w.tuple(DefVec(batch, [&](size_t d) { return w.lit_nat((*shape)[d]); }))});
    // group_blis_batch_impl copies each batch's A into a row-major (M, K) operand, which needs no second pack.
    builder      = w.app(builder, {w.lit_nat(m), w.lit_nat(k), w.lit_nat(n), w.lit_nat(mr_tile), w.lit_nat(nr_tile),
                                   w.lit_nat(nc), w.lit_nat(kc), w.lit_ff()});
    builder      = w.app(builder, {mr.comb, mr.init, mr.post, mr.post_maps, w.tuple(reads)});
    return {w.app(builder, app->arg()), {}};
}

namespace {

const Def* outer_parts(const Def* producer, const btensor::MapRedPost& mr, u64 axis, u64 tile) {
    auto& w    = producer->world();
    auto rank  = Lit::as(mr.Ro);
    auto shape = *lit_projs(mr.So, rank);
    auto full = shape[axis] / tile, tail = shape[axis] % tile;
    DefVec parts, shapes;
    for (auto remainder : {false, true}) {
        auto trips = remainder ? u64(tail != 0) : full;
        if (!trips) continue;
        auto extent     = mr.So->projs(rank);
        extent[axis]    = w.lit_nat(remainder ? tail : tile);
        auto ext        = w.tuple(extent);
        auto storage    = extent;
        storage[axis]   = w.lit_nat(remainder ? tail : full * tile);
        auto part_shape = w.tuple(storage);
        auto boxes      = w.mut_lam(w.type_nat(), w.app(w.annex<TileBox>(), mr.Ro));
        auto off        = DefVec(rank, w.lit_nat_0());
        off[axis]       = w.call(core::nat::mul, Defs{boxes->var(), w.lit_nat(tile)});
        boxes->set(true, w.tuple({w.tuple(off), ext}));
        RegionInputs inputs(Defs{producer});
        auto values = inputs.values(w);
        auto body   = w.mut_lam(w.type_nat(), w.pi(values->type(), type_tensor(mr.Ro, ext, mr.Tp)));
        auto inner  = w.mut_lam(values->type(), type_tensor(mr.Ro, ext, mr.Tp));
        off[axis]   = w.call(core::nat::add, Defs{w.call(core::nat::mul, Defs{body->var(), w.lit_nat(tile)}),
                                                  w.lit_nat(remainder ? full * tile : 0)});
        auto read   = w.app(w.app(w.annex<tile_of_box>(), {mr.Tp, mr.Ro}), mr.So);
        inner->set(true, w.app(read, {inputs.rebuild(producer, inner->var()), w.tuple(off), ext, ext}));
        body->set(true, inner);
        auto tiled = w.app(w.annex<tiled_n>(), {w.lit_nat_1(), w.lit_nat_1(), mr.Tp, mr.Ro});
        tiled = w.app(w.app(w.app(w.app(w.app(tiled, {part_shape, w.lit_nat(trips)}), boxes), values->type()), values),
                      body);
        parts.push_back(tiled);
        shapes.push_back(part_shape);
    }
    if (parts.size() == 1) return parts.front();
    auto join = w.app(w.annex<concat>(), {mr.Tp, w.lit_nat(parts.size()), mr.Ro});
    return w.app(w.app(w.app(join, w.lit_idx(rank, axis)), w.tuple(shapes)), w.tuple(parts));
}

ProducerTile schedule_group(const Def* arg) {
    auto [producer, kind, dim, tile, params] = arg->projs<5>();
    auto k = Lit::isa(kind), axis = Lit::isa(dim), size = Lit::isa(tile);
    if (!k || !axis || !size) return {nullptr, "group kind, axis and tile size must be literal"};
    if (*k == 1) return blis_group(producer, tile, params);
    if (*k == 2) {
        auto plan = reverse_tiles(producer, params, dim, tile);
        return {nullptr, plan.value ? "reverse multi-root rule did not match the live consumers" : plan.reason};
    }
    if (*k != 0) return {nullptr, "unsupported group request kind"};
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "outer grouping needs a map_reduce_post"};
    auto mr     = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks  = mr.ranks();
    auto budget = Lit::isa(params);
    if (!ranks || *axis >= ranks->ro || !*size || !budget)
        return {nullptr, "outer grouping needs a parallel axis, a positive tile and a literal budget"};
    if (!pure_scalar(mr.comb) || !pure_scalar(mr.post))
        return {nullptr, "outer grouping needs pure scalar combiner and epilogue"};
    auto shape = lit_projs(mr.So, ranks->ro);
    if (!shape) return {nullptr, "outer grouping needs literal output extents"};
    for (auto extent : *shape)
        if (!extent) return {nullptr, "outer grouping needs nonempty output extents"};
    auto& w = arg->world();
    if (*axis == 0 && *size == 1 && *budget == 1) {
        DefVec members;
        if (auto row = row_group(producer, members); row.value) return row;
    }
    if (Lit::isa(w.app(w.app(w.annex<group_root>(), producer->type()), producer)) != 0)
        return {outer_parts(producer, mr, *axis, std::min((*shape)[*axis], *size)), {}};
    DefVec tiles(ranks->ro,
                 [&](size_t d) { return w.lit_nat(d == *axis ? std::min((*shape)[d], *size) : (*shape)[d]); });
    auto builder = w.app(w.annex<group_impl>(), {mr.Tp, mr.Ro, mr.So});
    return {w.app(builder, {producer, w.tuple(tiles)}), {}};
}

const Rule* reverse_group_rule(const App* req, const pattern::GraphView& source) {
    auto& w                                     = req->world();
    auto [producer, kind, dim, tile, consumers] = req->args<5>();
    auto originals                              = graph::map_reduce(consumers) ? DefVec{consumers} : consumers->projs();
    DefVec actual;
    LocalRewriter rebase(w);
    for (auto var : req->free_vars())
        rebase.map(var, var);
    rebase.map(producer, req);
    for (auto consumer : originals) {
        auto expected    = rebase.rewrite(consumer);
        const Def* found = nullptr;
        for (auto edge : source.data_uses(req))
            if (Check::test_alpha(expected, edge.consumer)) {
                if (found && found != edge.consumer) return nullptr;
                found = edge.consumer;
            }
        if (!found) return nullptr;
        actual.push_back(found);
    }
    DefVec vars{req->arg()}, pending;
    for (auto value : actual) {
        auto call = value->as<App>();
        vars.push_back(call->callee());
        pending.push_back(call->arg());
    }
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (def == req || !seen.insert(def).second) continue;
        if (def->isa<Tuple>() || def->isa_imm<Pack>()) {
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        } else
            vars.push_back(def);
    }
    DefVec types;
    for (auto var : vars) {
        if (!var->type()->is_closed()) return nullptr;
        types.push_back(var->type());
    }
    auto rule = w.mut_rule(w.reform(w.sigma(types)));
    LocalRewriter importer(w);
    for (size_t i = 0; i != vars.size(); ++i)
        importer.map(vars[i], rule->var()->proj(vars.size(), i));
    auto meta                  = req->callee()->as<App>()->arg();
    auto fields                = rule->var()->proj(vars.size(), 0);
    auto [p, unused, d, t, cs] = fields->projs<5>();
    auto marker                = w.app(w.app(w.annex<group_req>(), meta), fields);
    importer.map(req, marker);
    auto build           = w.app(w.app(w.annex<reverse_build>(), meta), {p, cs, d, t});
    const Def* condition = w.lit_tt();
    DefVec lhs{marker}, rhs{build->proj(2, 0)};
    for (size_t i = 0; i != actual.size(); ++i) {
        auto value = importer.rewrite(actual[i]);
        lhs.push_back(value);
        auto check = w.app(w.app(w.annex<group_reverse_supported>(), {p->type(), cs->type(), value->type()}),
                           {fields, value, w.lit_nat(i)});
        condition  = w.call(core::bit2::and_, w.lit_nat(2), w.tuple({condition, check}));
        rhs.push_back(graph::map_reduce(consumers) ? build->proj(2, 1) : build->proj(2, 1)->proj(actual.size(), i));
    }
    return rule->set(w.tuple(lhs), w.tuple(rhs), condition);
}

std::optional<u64> scope_budget(const pattern::GraphView& source, const Def* reader) {
    struct Visit {
        const Def* def;
        std::optional<u64> budget;
    };
    fe::Vector<Visit> pending;
    for (auto ext : source.world().externals().muts())
        if (auto fn = ext->isa<Lam>(); fn && fn->is_set()) pending.push_back({fn->body(), {}});
    DefMap<fe::Vector<std::optional<u64>>> seen;
    std::optional<u64> result;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i > 64 * (source.world().curr_gid() + 1)) return {};
        auto [def, budget] = pending[i];
        auto& visits       = seen[def];
        if (std::ranges::find(visits, budget) != visits.end()) continue;
        visits.push_back(budget);
        if (def == reader) {
            if (!budget) return {};
            result = result ? std::min(*result, *budget) : budget;
        }
        if (auto scope = Axm::isa<group_region>(def)) {
            auto [body, cap] = scope->args<2>();
            pending.push_back({body, Lit::isa(cap)});
        } else if (Axm::isa<group_req>(def)) {
            continue;
        } else if (auto fn = def->isa<Lam>()) {
            if (fn->is_set() && !fn->is_external()) pending.push_back({fn->body(), budget});
        } else if (!def->is_mutable() && !def->isa<Var>() && !def->isa<Axm>() && !def->isa<Pi>()) {
            for (auto op : def->ops())
                if (op) pending.push_back({op, budget});
        }
    }
    return result;
}

class CanonicalizeRequests : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App* app) override {
        if (auto mrp = graph::map_reduce(app)) {
            auto mr = btensor::MapRedPost::of(mrp->callee()->as<App>());
            if (Lit::isa(mr.nps) == 0) {
                auto args         = app->uncurry_args<9>();
                auto& w           = new_world();
                const Def* result = rewrite(App::uncurry_callee(app));
                for (size_t i = 0; i != args.size(); ++i) {
                    auto arg = i == 3 ? w.tuple({rewrite(mr.Tis), rewrite(mr.Ris), rewrite(mr.Sis), w.tuple(),
                                                 w.tuple(), w.tuple()})
                                      : rewrite(args[i]);
                    result   = w.app(result, arg);
                }
                return result;
            }
        }
        return RWPhase::rewrite_imm_App(app);
    }
};

class RemoveRegions : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App* app) override {
        if (Axm::isa<group_region>(app)) return rewrite(app->arg(2, 0));
        if (Axm::isa<group_req>(app)) {
            Rewriter importer(new_world());
            auto fields = importer.rewrite(app->arg());
            auto result = schedule_group(fields);
            auto reason = result.value ? "group rule did not consume the request" : result.reason;
            if (driver().arg_nat("tensor", "strict").value_or(0))
                app->blame("tensor.group_req not realized: {}", reason).bail();
            auto report = log();
            report.set(fe::Log::Level::Warn).w("tensor.group_req at {} not realized: {}", app->err_loc(), reason);
            return rewrite(app->arg(5, 0));
        }
        return RWPhase::rewrite_imm_App(app);
    }
};

} // namespace

const Def* normalize_group_schedule(const Def*, const Def*, const Def* arg) {
    auto result = schedule_group(arg);
    if (!result.value) return nullptr;
    auto& w                            = arg->world();
    auto [op, kind, dim, tile, params] = arg->projs<5>();
    auto budget                        = Lit::isa(kind) == 0 ? params : w.lit_nat_1();
    return w.app(w.app(w.annex<group_region>(), result.value->type()), {result.value, budget});
}

pattern::WitnessResult group_support_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto fields = inputs[0];
    if (Lit::isa(fields->proj(5, 1)) == 0 && Lit::isa(fields->proj(5, 2)) == 0 && Lit::isa(fields->proj(5, 3)) == 1
        && Lit::isa(fields->proj(5, 4)) == 1) {
        auto scratch = target.inherit();
        Rewriter importer(*scratch);
        DefVec members;
        if (row_group(importer.rewrite(fields->proj(5, 0)), members).value
            && !row_group_closed(source, fields->proj(5, 0)))
            return {pattern::CheckResult::Unknown, nullptr, "row region has readers outside the requested closure"};
    }
    auto scratch = target.inherit();
    Rewriter importer(*scratch);
    auto result = schedule_group(importer.rewrite(inputs[0]));
    return result.value ? pattern::WitnessResult{pattern::CheckResult::True, target.lit_tt(), {}}
                        : pattern::WitnessResult{pattern::CheckResult::Unknown, nullptr, result.reason};
}

bool group_fusion_producer(const Def* producer) {
    auto app = graph::map_reduce(producer);
    if (!app || is_pure_read(producer)) return false;
    auto& w = producer->world();
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    return Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>())
        && Check::test_alpha(mr.sched, w.app(w.annex<btensor::mk_sched>(), w.app(w.annex<tile_sched>(), mr.sched)))
        && !blis_group(producer, w.lit_nat_0(), w.pack(3, w.lit_nat_0())).value;
}

pattern::WitnessResult group_fusion_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto budget = scope_budget(source, inputs[1]);
    if (!budget) return {pattern::CheckResult::False, nullptr, "reader is not confined to requested groups"};
    auto scratch = target.inherit();
    Rewriter importer(*scratch);
    if (!group_fusion_producer(importer.rewrite(inputs[0])))
        return {pattern::CheckResult::False, nullptr, "fusion cannot cross a BLIS or placed root"};
    return fusion_budget(source, {inputs[0], inputs[1], target.lit_nat(*budget)}, target);
}

std::optional<u64> region_scope_budget(const pattern::GraphView& source, const Def* region) {
    return scope_budget(source, region);
}

pattern::WitnessResult group_reverse_witness(const pattern::GraphView& source, Defs inputs, World& target) {
    auto fields = inputs[0];
    auto kind   = graph::slot(fields, 5, 1);
    if (!kind || Lit::isa(kind) != 2) return {pattern::CheckResult::False, nullptr, "not a reverse request"};
    auto producer  = graph::slot(fields, 5, 0);
    auto consumers = graph::slot(fields, 5, 4);
    auto dim = graph::slot(fields, 5, 2), tile = graph::slot(fields, 5, 3);
    if (!producer || !consumers || !dim || !tile)
        return {pattern::CheckResult::Unknown, nullptr, "reverse group fields are not frozen"};
    auto reads = graph::map_reduce(consumers) ? DefVec{consumers} : consumers->projs();
    auto index = Lit::isa(inputs[2]);
    if (!index || *index >= reads.size())
        return {pattern::CheckResult::Unknown, nullptr, "reverse group slot is not literal"};
    for (auto consumer : reads)
        if (source.data_used_by(producer, consumer) != pattern::CheckResult::True)
            return {pattern::CheckResult::Unknown, nullptr, "reverse group needs a proven producer-consumer edge"};
    auto scratch = target.inherit();
    Rewriter importer(*scratch);
    const App* request = nullptr;
    for (auto def : source.nodes())
        if (auto req = Axm::isa<group_req>(def); req && req->arg() == fields) {
            if (request && request != req)
                return {pattern::CheckResult::Unknown, nullptr, "reverse group request is ambiguous"};
            request = req;
        }
    if (!request) return {pattern::CheckResult::Unknown, nullptr, "reverse group request is absent"};
    auto marker   = importer.rewrite(request);
    auto original = importer.rewrite(reads[*index]);
    auto actual   = importer.rewrite(inputs[1]);
    LocalRewriter rebase(*scratch);
    for (auto var : original->free_vars())
        rebase.map(var, var);
    rebase.map(importer.rewrite(producer), marker);
    if (!Check::test_alpha(rebase.rewrite(original), actual))
        return {pattern::CheckResult::Unknown, nullptr, "reverse group consumer is not the requested rebasing"};
    auto plan = reverse_tiles(importer.rewrite(producer), importer.rewrite(consumers), importer.rewrite(dim),
                              importer.rewrite(tile));
    return plan.value ? pattern::WitnessResult{pattern::CheckResult::True, target.lit_tt(), {}}
                      : pattern::WitnessResult{pattern::CheckResult::Unknown, nullptr, plan.reason};
}

namespace phase {

void AttentionRaise::start() {
    if (driver().is_loaded("gpu") || !driver().arg_nat("tensor", "group").value_or(1)
        || !driver().arg_nat("tensor", "attention").value_or(1))
        return;
    auto& w          = world();
    auto rule        = w.app(w.annex<attention_rule>(), w.lit_nat(16))->as<Rule>();
    auto masked_rule = w.app(w.annex<attention_masked_rule>(), w.lit_nat(16))->as<Rule>();
    std::vector<const Axm*> pure;
    for (auto name : {"tensor.dot_product", "tensor.unary", "tensor.binary", "tensor.map_reduce_post", "tensor.sdpa",
                      "tensor.sdpa_masked", "math.arith.add", "math.arith.mul", "math.arith.sub", "math.arith.div",
                      "math.extrema.iM", "math.exp.lbb"})
        pure.push_back(w.annex(driver().sym(name))->as<Axm>());
    Phases stages;
    stages.push_back(std::make_unique<GraphRuleRewrite>(w, std::vector{masked_rule, rule}, std::move(pure), true));
    PhaseMan pipeline(w, "tensor.raise_attention.rules");
    pipeline.apply(true, std::move(stages));
    pipeline.run();
}

void GroupDefault::start() {
    if (driver().is_loaded("gpu")) return;
    if (auto mode = driver().arg_nat("tensor", "tile-fuse").value_or(0)) {
        if (mode == 2) TileGroup(world(), Annex::base<tile_group>()).run();
        return TileFuse(world(), Annex::base<tile_fuse>()).run();
    }
    if (!driver().arg_nat("tensor", "group").value_or(1)) return;
    Phases stages;
    stages.push_back(std::make_unique<Group>(world(), Annex::base<group>()));
    stages.push_back(std::make_unique<GroupRules>(world(), Annex::base<group_rules>()));
    stages.push_back(std::make_unique<Fuse>(world(), Annex::base<fuse_views>()));
    PhaseMan pipeline(world(), "tensor.group_default.pipeline");
    pipeline.apply(false, std::move(stages));
    pipeline.run();

    auto yield_budget = driver().arg_nat("tensor", "group-budget").value_or(1);
    if (driver().arg_nat("tensor", "fuse-and-yield").value_or(1) && yield_budget) {
        Phases stages;
        auto phase = std::make_unique<FuseAndYield>(world(), Annex::base<fuse_and_yield>());
        phase->apply(world().app(world().annex<fuse_and_yield>(), world().lit_nat(yield_budget))->as<App>());
        stages.push_back(std::move(phase));
        PhaseMan fusion(world(), "tensor.fuse-and-yield");
        fusion.apply(true, std::move(stages));
        fusion.run();
    }
    if (driver().arg_nat("tensor", "prepack").value_or(1)) PrepackWeights(world(), "tensor.prepack_weights").run();
}

void AttentionSchedule::start() {
    if (driver().is_loaded("gpu") || !driver().arg_nat("tensor", "group").value_or(1)
        || !driver().arg_nat("tensor", "attention").value_or(1)
        || !driver().arg_nat("tensor", "attention-online").value_or(0))
        return;
    auto& w     = world();
    auto params = w.tuple({w.lit_nat(16), w.app(w.annex<vec_width>(), w.lit_nat(8)),
                           w.lit_nat(driver().arg_nat("tensor", "group-registers").value_or(32))});
    auto rules  = std::vector{w.app(w.annex<attention_schedule_rule>(), params)->as<Rule>(),
                              w.app(w.annex<attention_masked_schedule_rule>(), params)->as<Rule>()};
    RuleRewrite rewrite(w, std::move(rules));
    rewrite.run();
}

void GroupRules::start() {
    if (driver().is_loaded("gpu")) return;
    {
        pattern::GraphView source(world(), driver().graph_operations());
        for (auto def : source.operations())
            if (auto request = Axm::isa<group_req>(def))
                if (Axm::isa<group_req>(request->arg(5, 0)))
                    request->blame("tensor.group phase bug: stacked structural requests on one op").bail();
    }
    CanonicalizeRequests canonicalize(world(), "tensor.group_rules.canonicalize");
    canonicalize.run();
    pattern::GraphView source(world(), driver().graph_operations());
    DefVec rules, reverse;
    DefSet types, elements;
    for (auto def : source.nodes()) {
        if (auto req = Axm::isa<group_req>(def)) {
            auto meta = req->callee()->as<App>()->arg();
            if (Lit::isa(req->arg(5, 1)) == 2) {
                if (auto rule = reverse_group_rule(req, source)) reverse.push_back(rule);
                types.insert(meta);
                continue;
            }
            if (!meta->is_closed()) continue;
            if (types.insert(meta).second) rules.push_back(world().app(world().annex<group_request_rule>(), meta));
        }
        if (auto app = graph::map_reduce(def)) {
            auto mr   = btensor::MapRedPost::of(app->callee()->as<App>());
            auto meta = world().tuple({mr.Tp, mr.Ro});
            if (elements.insert(meta).second) {
                rules.push_back(world().app(world().annex<group_slice_rule>(), meta));
                rules.push_back(world().app(world().annex<group_box_rule>(), meta));
            }
        }
    }
    if (types.empty()) return;
    std::vector<const Rule*> compiled;
    for (auto rule : rules)
        compiled.push_back(rule->as<Rule>());
    Phases stages;
    if (!reverse.empty()) {
        std::vector<const Axm*> pure;
        for (auto name : {"tensor.group_req", "tensor.reverse_build", "tensor.tiled_n", "tensor.tile_of",
                          "tensor.map_reduce_post", "tensor.slice", "affine.op.add", "affine.op.sub", "affine.lit"})
            pure.push_back(world().annex(driver().sym(name))->as<Axm>());
        std::vector<const Rule*> graph_rules;
        for (auto rule : reverse)
            graph_rules.push_back(rule->as<Rule>());
        stages.push_back(std::make_unique<GraphRuleRewrite>(world(), std::move(graph_rules), std::move(pure), true));
    }
    stages.push_back(std::make_unique<RuleRewrite>(world(), std::move(compiled)));
    auto fusion = std::make_unique<FuseRegionInputs>(world(), Annex::base<fuse_region_inputs>());
    fusion->apply(
        world()
            .app(world().annex<fuse_region_inputs>(), {world().lit_nat(2), world().lit_nat_1(), world().lit_tt()})
            ->as<App>());
    stages.push_back(std::move(fusion));
    PhaseMan pipeline(world(), "tensor.group_rules.fixed_point");
    pipeline.apply(true, std::move(stages));
    pipeline.run();
    RemoveRegions cleanup(world(), "tensor.group_rules.outcomes");
    cleanup.run();
}

} // namespace phase
} // namespace mim::plug::tensor
