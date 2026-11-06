#include "mim/plug/tensor/phase/tile_group.h"

#include <mim/pattern/graph.h>

#include <mim/plug/core/core.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

/// Whether @p map reads loop @p loop on some axis.
bool reads_loop(const Def* map, u64 loops, u64 loop) {
    auto& w    = map->world();
    auto pi    = map->type()->as<Pi>();
    auto var   = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read  = w.app(map, var);
    auto arity = Lit::isa(read->arity());
    if (!arity) return false;
    for (u64 d = 0; d != *arity; ++d)
        if (Linear::of(read->proj(*arity, d)).atoms.contains(var->proj(loops, loop))) return true;
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

/// 1 for a contraction `(m, n, k)` whose two inputs read `m, k` and `k, n`, 2 for another reduction, 0 otherwise.
u64 kind_of(const btensor::MapRedPost& mr) {
    auto ranks = mr.ranks();
    if (!ranks || !ranks->rr) return 0;
    // Batched contractions are not yet tiled at the micro level: their nested write maps defeat exact preimages.
    if (ranks->rr != 1 || ranks->ro != 2 || ranks->nis != 2) return 2;
    auto n = ranks->nloops(), m = ranks->ro - 2, k = n - 1;
    auto a = mr.maps->proj(2, 0), b = mr.maps->proj(2, 1);
    bool contraction = reads_loop(a, n, m) && reads_loop(a, n, k) && !reads_loop(a, n, m + 1) && reads_loop(b, n, k)
                    && reads_loop(b, n, m + 1) && !reads_loop(b, n, m);
    return contraction ? 1 : 2;
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
            claimed.insert(op);
            DefVec members;
            for (DefVec todo{op}; !todo.empty();) {
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
                    if (!pr.ranks() || pr.ranks()->rr) continue;
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
    auto& w     = new_world();
    auto mr     = btensor::MapRedPost::of(graph::map_reduce(value)->callee()->as<App>());
    auto ranks  = *mr.ranks();
    auto bytes  = w.app(w.annex(core::trait::size), mr.To);
    auto lanes  = w.app(w.annex<vec_width>(), w.lit_nat(4));
    auto facts  = w.tuple({w.lit_nat(ranks.ro), w.lit_nat(ranks.rr), w.lit_nat(group->second.kind), bytes, lanes,
                           w.lit_nat(driver().arg_nat("tensor", "group-registers").value_or(32)),
                           w.lit_nat(driver().arg_nat("tensor", "group-cache").value_or(262144))});
    auto levels = w.app(w.annex<default_tile_config>(), facts);
    auto members
        = w.tuple(DefVec(group->second.members.size(), [&](size_t i) { return rewrite(group->second.members[i]); }));
    auto control = w.annex<default_control>();
    auto request = w.app(w.annex<tile_req>(), {value->type(), members->type(), levels->type(), control->type()});
    return w.app(request, {value, members, levels, control});
}

} // namespace mim::plug::tensor::phase
