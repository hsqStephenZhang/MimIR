#include "mim/plug/tensor/phase/tile_group.h"

#include <mim/pattern/graph.h>

#include <mim/plug/core/core.h>

#include "mim/plug/tensor/phase/panel_layout.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

/// Whether @p map reads loop @p loop on some axis, also through a view's division or remainder.
bool reads_loop(const Def* map, u64 loops, u64 loop) {
    auto& w    = map->world();
    auto pi    = map->type()->as<Pi>();
    auto var   = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read  = w.app(map, var);
    auto index = var->proj(loops, loop);
    DefSet seen{read};
    for (DefVec pending{read}; !pending.empty();) {
        auto def = pending.back();
        pending.pop_back();
        if (def == index) return true;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return false;
}

/// Whether @p mr writes loop d to output axis d, which tiling a root's parallel loops requires.
bool canonical_write(const btensor::MapRedPost& mr, u64 ro, u64 rn) {
    auto& w    = mr.map_out->world();
    auto pi    = mr.map_out->type()->as<Pi>();
    auto var   = w.mut_lam(pi->dom(), pi->codom())->var();
    auto write = w.app(mr.map_out, var);
    for (u64 d = 0; d != ro; ++d) {
        auto form = Linear::of(write->proj(ro, d));
        for (u64 l = 0; l != rn; ++l) {
            auto atom = form.atoms.find(var->proj(rn, l));
            if ((atom == form.atoms.end() ? 0 : atom->second) != (l == d ? 1 : 0)) return false;
        }
    }
    return true;
}

/// 1 for a contraction `(…, m, n, k)` whose two inputs read `m, k` and `k, n`, 2 for another reduction, 0 otherwise.
u64 kind_of(const btensor::MapRedPost& mr) {
    auto ranks = mr.ranks();
    if (!ranks || !ranks->rr) return 0;
    if (ranks->rr != 1 || ranks->ro < 2 || ranks->nis != 2) return 2;
    auto n = ranks->nloops(), m = ranks->ro - 2, k = n - 1;
    auto a = mr.maps->proj(2, 0), b = mr.maps->proj(2, 1);
    bool contraction = reads_loop(a, n, m) && reads_loop(a, n, k) && !reads_loop(a, n, m + 1) && reads_loop(b, n, k)
                    && reads_loop(b, n, m + 1) && !reads_loop(b, n, m);
    return contraction ? 1 : 2;
}

/// Whether axis d of @p map reads loop `loops[d]` and no other loop; offsets of enclosing tiles may shift it.
bool projects(const Def* map, u64 rn, fe::Span<const u64> loops) {
    auto& w   = map->world();
    auto pi   = map->type()->as<Pi>();
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(map, var);
    if (Lit::isa(read->arity()) != loops.size()) return false;
    for (size_t d = 0; d != loops.size(); ++d) {
        auto form = Linear::of(read->proj(loops.size(), d));
        for (u64 l = 0; l != rn; ++l) {
            auto atom = form.atoms.find(var->proj(rn, l));
            if ((atom == form.atoms.end() ? 0 : atom->second) != (l == loops[d] ? 1 : 0)) return false;
        }
    }
    return true;
}

/// @p value with each contraction operand that a view reads materialized on the loops it depends on: the batch loops,
/// then its two contraction loops, so that packing a panel sees an operand whose axes are the loops themselves.
const Def* contraction_views(const Def* value) {
    auto app = graph::map_reduce(value);
    if (app != value) return value;
    auto& w    = app->world();
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = *mr.ranks();
    auto rn = ranks.nloops(), m = ranks.ro - 2, k = rn - 1;
    auto [is, post_is] = app->args<2>();
    DefVec inputs(is->projs(2)), maps(mr.maps->projs(2)), rank(mr.Ris->projs(2)), shapes(mr.Sis->projs(2));
    bool changed = false;
    for (u64 slot = 0; slot != 2; ++slot) {
        auto loops = std::vector<u64>();
        for (u64 b = 0; b != m; ++b)
            loops.push_back(b);
        loops.push_back(slot == 0 ? m : k);
        loops.push_back(slot == 0 ? k : m + 1);
        auto swapped = loops;
        std::swap(swapped[m], swapped[m + 1]);
        if (!Lit::isa(rank[slot]) || projects(maps[slot], rn, loops) || projects(maps[slot], rn, swapped)) continue;
        auto g     = loops.size();
        auto pi    = maps[slot]->type()->as<Pi>();
        auto index = w.annex<affine::Index>();
        auto read  = w.mut_lam(w.pi(w.arr(g, index), pi->codom()));
        read->set(true, w.app(maps[slot], w.tuple(DefVec(rn, [&](size_t l) -> const Def* {
            if (auto at = std::ranges::find(loops, l); at != loops.end()) return read->var(g, at - loops.begin());
            return w.call<affine::lit>(w.lit_nat_0());
        }))));
        auto shape = w.tuple(DefVec(g, [&](size_t i) { return mr.Sr->proj(rn, loops[i]); }));
        // Buffers take runtime extents only on their leading axis.
        bool bufferizable = true;
        for (size_t i = 1; i != g; ++i)
            bufferizable &= Lit::isa(shape->proj(g, i)).has_value();
        if (!bufferizable) continue;
        auto type = *TensorType::isa(inputs[slot]->type());
        auto view
            = w.app(w.app(w.app(w.app(w.app(w.annex<tf_view>(), {type.T, type.r, type.s}), w.lit_nat(g)), shape), read),
                    inputs[slot]);
        auto plain = w.mut_lam(pi->dom(), w.arr(g, index));
        plain->set(true, w.tuple(DefVec(g, [&](size_t i) { return plain->var()->proj(rn, loops[i]); })));
        inputs[slot] = view;
        maps[slot]   = plain;
        rank[slot]   = w.lit_nat(g);
        shapes[slot] = shape;
        changed      = true;
    }
    if (!changed) return value;
    auto in_tys = w.tuple({mr.Tis, w.tuple(rank), w.tuple(shapes), mr.Tps, mr.Rps, mr.Sps});
    auto result = w.app(
        w.app(w.app(w.app(w.app(w.app(w.app(w.annex<map_reduce_post>(), mr.nis_nps), mr.meta), mr.shapes), in_tys),
                    mr.init->type()),
              w.tuple({mr.comb, mr.init, mr.post})),
        mr.map_out);
    return w.app(w.app(result, w.tuple({w.tuple(maps), mr.post_maps})), w.tuple({w.tuple(inputs), post_is}));
}

} // namespace

void TileGroup::start() {
    pattern::OpRegistry ops;
    graph::register_operations(ops);
    pattern::GraphView source(old_world(), ops);
    DefSet claimed;
    for (auto op : source.operations()) {
        if (auto request = Axm::isa<tile_req>(op)) {
            claimed.insert(request->arg(4, 0));
            continue;
        }
        if (auto request = Axm::isa<group_req>(op)) {
            claimed.insert(request->arg(5, 0));
            continue;
        }
        auto occurrences = source.occurrences(op);
        if (occurrences.empty()) claimed.insert(op);
        for (auto occurrence : occurrences)
            if (!occurrence.trips || !occurrence.regions.empty()) claimed.insert(op);
    }
    auto operations = DefVec(source.operations().begin(), source.operations().end());
    std::ranges::sort(operations, {}, &Def::gid);
    for (auto kind : {u64(1), u64(2), u64(0)}) {
        // Consumers come first, so an elementwise chain joins the group of its last op.
        auto order = operations;
        if (kind == 0) std::ranges::reverse(order);
        for (auto op : order) {
            auto app = graph::map_reduce(op);
            // Pure reads are views that other phases place into their consumers' buffers.
            if (!app || claimed.contains(op) || is_pure_read(op)) continue;
            auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
            if (!mr.ranks() || kind_of(mr) != kind || !canonical_write(mr, mr.ranks()->ro, mr.ranks()->nloops()))
                continue;
            // A leaf with an explicit vector/unroll schedule already owns its register blocking.
            // In particular, reference tiling can expose these leaves outside a region.
            auto& w = old_world();
            auto plain = w.app(w.annex<btensor::mk_sched>(),
                               w.tuple({w.lit_nat(mr.ranks()->nloops()), w.lit_nat_0()}));
            if (!Check::test_alpha(mr.sched, plain) || !pure_scalar(mr.comb) || !pure_scalar(mr.post)) continue;
            // A partial reduction seeded by an explicit fold already owns its K decomposition.
            // Repartitioning its tail can change the storage caps of the fold's carried panels.
            if (auto initial = tensor_init(mr)) {
                auto seed = initial->outs, shape = initial->shape;
                while (auto read = panel::tensor_read(seed, shape)) {
                    seed  = read->src;
                    shape = read->S;
                }
                if (Axm::isa<tile_fold>(seed)) continue;
            }
            claimed.insert(op);
            DefVec members;
            // A batched contraction's producers also feed other roots' reductions, so fusing them would compute them
            // twice.
            bool batched = kind == 1 && mr.ranks()->ro > 2;
            for (DefVec todo{op}; !batched && !todo.empty();) {
                auto consumer = todo.back();
                todo.pop_back();
                auto reader = graph::map_reduce(consumer);
                if (!reader) continue;
                auto rm     = btensor::MapRedPost::of(reader->callee()->as<App>());
                auto inputs = DefVec();
                if (auto ranks = rm.ranks()) {
                    auto [is, post_is] = reader->args<2>();
                    for (auto [values, n] : {
                             std::pair{     is, ranks->nis},
                             std::pair{post_is, ranks->nps}
                    })
                        for (u64 i = 0; i != n; ++i)
                            inputs.push_back(values->proj(n, i));
                }
                for (auto input : inputs) {
                    auto producer = graph::map_reduce(input);
                    if (!producer || claimed.contains(input)) continue;
                    auto pr = btensor::MapRedPost::of(producer->callee()->as<App>());
                    if (!pr.ranks() || pr.ranks()->rr || !pure_scalar(pr.comb) || !pure_scalar(pr.post)) continue;
                    bool scope = true;
                    for (auto var : input->free_vars())
                        scope &= op->has_free_var(var);
                    if (!scope) continue;
                    claimed.insert(input);
                    members.push_back(input);
                    todo.push_back(input);
                }
            }
            groups_.emplace(op, Group{kind, std::move(members)});
            log().d("tensor.tile_group root {} kind {} members {}", op, kind, groups_[op].members.size());
        }
    }
    RWPhase::start();
}

const Def* TileGroup::rewrite_imm_App(const App* app) {
    auto value = RWPhase::rewrite_imm_App(app);
    auto group = groups_.find(app);
    if (group == groups_.end()) return value;
    if (group->second.kind == 1) value = contraction_views(value);
    auto& w     = new_world();
    auto mr     = btensor::MapRedPost::of(graph::map_reduce(value)->callee()->as<App>());
    auto ranks  = *mr.ranks();
    auto bytes  = w.app(w.annex(core::trait::size), mr.To);
    auto lanes  = w.app(w.annex<vec_width>(), w.lit_nat(4));
    auto facts  = w.tuple({w.lit_nat(ranks.ro), w.lit_nat(ranks.rr), w.lit_nat(group->second.kind), bytes, lanes,
                           w.lit_nat(driver().arg_nat("tensor", "group-registers").value_or(32)),
                           w.lit_nat(driver().arg_nat("tensor", "group-cache").value_or(262144))});
    auto levels = w.app(w.app(w.annex<default_tile_config>(), mr.To), facts);
    if (group->second.kind == 1) {
        auto args = DefVec{w.lit_nat(ranks.ro), bytes, lanes,
                           w.lit_nat(driver().arg_nat("tensor", "group-registers").value_or(32)),
                           w.lit_nat(driver().arg_nat("tensor", "group-cache").value_or(262144))};
        for (auto name : {"cpu-mr", "cpu-nr", "cpu-mc", "cpu-nc", "cpu-kc", "cpu-order"})
            args.push_back(w.lit_nat(driver().arg_nat("tensor", name).value_or(0)));
        if (Lit::isa(mr.Sr->proj(ranks.nloops(), ranks.nloops() - 1))) {
            levels = w.app(w.app(w.annex<cpu_default_levels>(), mr.To), w.tuple(args));
        } else {
            // A runtime K carried through an explicit fold still needs the reduction's panel-cap forwarding.
            // Keep that existing level chain until independent pack scopes forward those caps as well.
            auto blocks = w.app(w.annex<group_blocks>(), w.tuple({bytes, lanes, args[3], args[4]}));
            levels = w.app(w.app(w.annex<contraction_levels>(), mr.To),
                           w.tuple({w.lit_nat(ranks.ro), blocks->proj(4, 0), blocks->proj(4, 1), blocks->proj(4, 3),
                                    blocks->proj(4, 2), w.lit_nat(3)}));
            log().d("tensor.tile_group {} keeps reduction-scoped packing for runtime K", value);
        }
    }
    auto members
        = w.tuple(DefVec(group->second.members.size(), [&](size_t i) { return rewrite(group->second.members[i]); }));
    auto control = w.annex<default_control>();
    auto tensor  = *TensorType::isa(value->type());
    auto request = w.app(w.annex<tile_req>(), {tensor.T, tensor.r, tensor.s, members->type(), control->type(),
                                               w.lit_nat(ranks.nloops()), mr.To});
    return w.app(request, {value, members, levels, control});
}

} // namespace mim::plug::tensor::phase
