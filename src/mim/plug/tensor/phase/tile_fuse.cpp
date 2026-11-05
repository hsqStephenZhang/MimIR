#include "mim/plug/tensor/phase/tile_fuse.h"

#include <algorithm>
#include <limits>

#include <mim/pattern/graph.h>

#include <mim/plug/core/core.h>

#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

constexpr size_t max_fusions = 256;

bool depends(const Def* def, const Var* var) {
    for (auto free : def->free_vars())
        if (free == var) return true;
    return false;
}

DefVec members_of(const Def* members) {
    if (auto tuple = members->isa<Tuple>()) return DefVec(tuple->ops().begin(), tuple->ops().end());
    if (members == members->world().tuple()) return {};
    return {members};
}

/// A unit-step `slice`, or a `slice_box` with its storage `cap`: the producer and the box it reads.
struct Read {
    const App* app;
    const Def *producer, *off, *ext, *cap;
};

std::optional<Read> unit_slice(const Def* def) {
    if (auto app = Axm::isa<tensor::slice_box>(def)) {
        auto [off, ext, cap] = app->callee()->as<App>()->arg()->projs<3>();
        return Read{app, app->arg(), off, ext, cap};
    }
    auto app = Axm::isa<tensor::slice>(def);
    if (!app) return {};
    auto [meta, shape, box] = app->callee()->as<App>()->uncurry_args<3>();
    auto rank               = Lit::isa(meta->proj(2, 1));
    if (!rank) return {};
    auto [start, step, ext] = box->projs<3>();
    for (u64 d = 0; d != *rank; ++d)
        if (Lit::isa(step->proj(*rank, d)) != 1) return {};
    return Read{app, app->arg(), start, ext, nullptr};
}

/// The unit-step slices @p value reads, without entering lambdas, nested regions or the sliced producers.
std::vector<Read> reads(const Def* value) {
    std::vector<Read> result;
    DefVec pending{value};
    DefSet seen{value};
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (auto read = unit_slice(def)) {
            result.push_back(*read);
            continue;
        }
        if (def->isa<Lam>() || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>() || Axm::isa<tensor::tiled>(def)
            || Axm::isa<tensor::tiled_multi>(def) || Axm::isa<tensor::tile_fold>(def) || Axm::isa<tensor::tiled_n>(def))
            continue;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return result;
}

/// The maximal tensor-valued subterms of @p value that do not depend on @p var; they become the region's inputs.
DefVec leaves(const Def* value, const Var* var) {
    DefVec result, pending{value};
    DefSet seen{value};
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (TensorType::isa(def->type()) && !depends(def, var)) {
            result.push_back(def);
            continue;
        }
        if (def->isa<Lam>() || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return result;
}

/// The levels of a `tile_req`; a single level is not wrapped, as a one-element tuple is its element.
DefVec tile_levels(const Def* levels) {
    if (auto sigma = levels->type()->isa<Sigma>(); sigma && sigma->num_ops() == 4 && sigma->op(1)->isa<Arr>())
        return {levels};
    if (auto count = Lit::isa(levels->arity())) return levels->projs(*count);
    return {};
}

/// Whether @p producer reaches anything but the operations in @p inside, through tuples and projections.
/// The request's own reference to its members is inside, unlike in `GraphView::complete_data_uses`.
bool escapes_group(const pattern::GraphView& view, const Def* producer, const DefSet& inside) {
    DefVec pending{producer};
    DefSet seen{producer};
    for (size_t i = 0; i != pending.size(); ++i)
        for (const auto& edge : view.uses(pending[i])) {
            if (edge.kind == pattern::EdgeKind::Type) continue;
            auto owner = edge.owner;
            if (view.isa_operation(owner)) {
                if (!inside.contains(owner)) return true;
            } else if (owner->isa<Tuple>() || owner->isa<Pack>() || (owner->isa<Extract>() && edge.index == 0)) {
                if (seen.insert(owner).second) pending.push_back(owner);
            } else {
                return true;
            }
        }
    return false;
}

/// Whether `map_out` writes loop `d` to output axis `d` for every parallel loop.
bool canonical_write(const btensor::MapRedPost& mr, u64 ro, u64 rn) {
    auto& w   = mr.map_out->world();
    auto pi   = mr.map_out->type()->as<Pi>();
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(mr.map_out, var);
    for (u64 d = 0; d != ro; ++d)
        if (read->proj(ro, d) != var->proj(rn, d)) return false;
    return true;
}

} // namespace

void TileFuse::start() {
    auto& w = old_world();
    pattern::OpRegistry ops;
    graph::register_operations(ops);
    std::vector<const App*> requests;
    DefSet outside;
    {
        pattern::GraphView source(w, ops);
        for (auto def : source.nodes())
            if (auto request = Axm::isa<tile_req>(def)) requests.push_back(request);
        for (auto request : requests) {
            auto [op, members, levels, control] = request->args<4>();
            auto group                          = members_of(members);
            DefSet inside(group.begin(), group.end());
            inside.insert(op);
            inside.insert(request);
            for (auto p : group) {
                bool escapes = escapes_group(source, p, inside);
                if (escapes) outside.insert(p);
            }
        }
    }
    for (auto request : requests)
        if (auto region = realize(request, outside)) replacements_[request] = region;
    RWPhase::start();
}

const Def* TileFuse::rewrite_imm_App(const App* app) {
    if (auto replacement = fe::lookup(replacements_, app)) return rewrite(replacement);
    return RWPhase::rewrite_imm_App(app);
}

const Def* TileFuse::realize(const App* request, const DefSet& outside) {
    auto& w                             = old_world();
    auto [op, members, levels, control] = request->args<4>();
    auto unrealized                     = [&](std::string_view reason) -> const Def* {
        report_tile_request(request, reason);
        return op;
    };

    auto root = graph::map_reduce(op);
    if (!root) return unrealized("the root is not a map_reduce_post");
    auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
    auto rks = mr.ranks();
    if (!rks) return unrealized("the root's ranks are not literal");
    auto ro = rks->ro, rn = rks->nloops();
    if (!canonical_write(mr, ro, rn)) return unrealized("the root does not write loop d to output axis d");

    auto level_list = tile_levels(levels);
    if (level_list.size() != 1) return unrealized("only a single tiling level is supported");
    auto [kind, sizes, tail, promote] = level_list.front()->projs<4>();
    if (Lit::isa(kind) != 0 || Lit::isa(promote) != 0) return unrealized("only a parallel level is supported");
    auto tiles = DefVec(ro);
    bool whole = true;
    for (u64 d = 0; d != rn; ++d) {
        auto size = Lit::isa(sizes->proj(rn, d));
        if (!size) return unrealized("tile sizes must be literals");
        if (d >= ro) {
            if (*size) return unrealized("a parallel level cannot tile a reduction loop");
            continue;
        }
        tiles[d] = *size ? w.lit_nat(*size) : mr.So->proj(ro, d);
        whole &= tiles[d] == mr.So->proj(ro, d);
    }
    if (whole) return op;

    auto group = members_of(members);
    DefSet in_group(group.begin(), group.end());

    auto box   = w.sigma({w.arr(mr.Ro, w.type_nat()), w.arr(mr.Ro, w.type_nat())});
    auto probe = w.mut_lam(box, w.type_nat());
    auto var   = probe->var()->as<Var>();
    auto off = probe->var(2, 0), ext = probe->var(2, 1);
    // Boxed tiles carry literal storage caps derived from the tile sizes into every fused producer.
    auto tile_of_app = [&](const Def* producer, const Def* at, const Def* size, const Def* cap) {
        auto tensor = *TensorType::isa(producer->type());
        auto meta   = w.app(w.app(w.annex<tensor::tile_of_box>(), {tensor.T, tensor.r}), tensor.s);
        if (cap) return w.app(meta, {producer, at, size, cap});
        return w.app(w.app(w.app(w.annex<tensor::tile_of>(), {tensor.T, tensor.r}), tensor.s), {producer, at, size});
    };
    auto value = tile_of_app(op, off, ext, w.tuple(tiles));

    size_t fused = 0;
    DefSet visited;
    struct Yield {
        const Def *producer, *tile;
    };
    std::vector<Yield> yields;

    for (size_t step = 0; step != max_fusions; ++step) {
        std::optional<Read> next;
        for (auto read : reads(value))
            if (visited.insert(read.app).second) {
                next = read;
                break;
            }
        if (!next) break;
        auto p        = next->producer;
        auto producer = graph::map_reduce(p);
        auto reduces  = producer
                     && btensor::MapRedPost::of(producer->callee()->as<App>())
                                .ranks()
                                .value_or(btensor::MapRedPost::Ranks{0, 0, 0, 0})
                                .rr
                            != 0;
        auto query    = w.tuple({w.lit_nat_0(), w.lit_nat_0(), w.lit_nat_0(), w.lit_bool(reduces),
                                 w.lit_bool(outside.contains(p)), w.lit_bool(in_group.contains(p)),
                                 w.lit_nat(std::numeric_limits<nat_t>::max())});
        auto decision = w.app(control, query);
        auto fuse = Lit::isa(decision->proj(2, 0)), yield = Lit::isa(decision->proj(2, 1));
        if (!fuse || !yield) {
            if (driver().arg_nat("tensor", "strict").value_or(0))
                request->blame("tensor.tile_req control does not decide on {}: {}", p, decision).bail();
            log().d("tensor.tile_fuse {} skip {}: control is undecided", op, p);
            continue;
        }
        if (!*fuse) continue;
        if (auto tile = producer_tile(p, next->off, next->ext, next->cap); !tile.value) {
            log().d("tensor.tile_fuse {} skip {}: {}", op, p, tile.reason);
            continue;
        }
        Rewriter substitute(w);
        for (auto free : value->free_vars())
            substitute.map(free, free);
        substitute.map(next->app, tile_of_app(p, next->off, next->ext, next->cap));
        value = substitute.rewrite(value);
        ++fused;
        log().d("tensor.tile_fuse {} fuse {}", op, p);
        if (!*yield || !outside.contains(p)) continue;
        // The region hands back a member on its own grid; that tile is not shared with the fused read, whose box
        // equals the region's only under `off + ext ≤ s`, which the clamped read box does not show syntactically.
        if (auto tensor = TensorType::isa(p->type()); tensor && tensor->s == mr.So)
            yields.push_back({p, tile_of_app(p, off, ext, w.tuple(tiles))});
        else
            log().d("tensor.tile_fuse {} skip yield of {}: its shape is not the region's", op, p);
    }

    auto inputs = leaves(value, var);
    std::erase_if(yields, [&](const Yield& y) {
        bool read = std::ranges::find(inputs, y.producer) != inputs.end();
        if (read) log().d("tensor.tile_fuse {} skip yield of {}: the region still reads it whole", op, y.producer);
        return read;
    });
    if (!yields.empty()) {
        DefVec results{value};
        for (const auto& y : yields)
            results.push_back(y.tile);
        value = w.tuple(results);
    }
    auto ins = w.tuple(inputs);
    auto pi  = w.mut_pi(w.type());
    pi->set_dom(box);
    pi->set_codom(w.pi(ins->type(), VarRewriter(var, pi->var()).rewrite(value->type())));
    auto body  = w.mut_lam(pi);
    auto inner = w.mut_lam(pi->reduce(body->var())->as<Pi>());
    VarRewriter close(var, body->var());
    for (size_t i = 0, n = inputs.size(); i != n; ++i)
        close.map(inputs[i], inner->var()->proj(n, i));
    inner->set(true, close.rewrite(value));
    body->set(true, inner);

    log().d("tensor.tile_fuse {} L0 par tiles={} inputs={} fused={} yielded={}", op, w.tuple(tiles), inputs.size(),
            fused, yields.size());
    if (yields.empty()) {
        auto region = w.app(w.annex<tensor::tiled>(), {mr.Tp, mr.Ro});
        region      = w.app(region, {mr.So, w.tuple(tiles)});
        region      = w.app(region, ins->type());
        region      = w.app(region, ins);
        return w.app(region, body);
    }
    auto n     = yields.size() + 1;
    auto types = DefVec{mr.Tp}, shapes = DefVec{mr.So};
    for (const auto& y : yields) {
        types.push_back(TensorType::isa(y.producer->type())->T);
        shapes.push_back(mr.So);
    }
    auto axes   = w.tuple(DefVec(ro, [&](size_t d) { return w.lit_idx(ro, d); }));
    auto region = w.app(w.annex<tensor::tiled_multi>(),
                        {w.lit_nat(n), mr.Ro, w.tuple(types), w.tuple(DefVec(n, [&](size_t) { return mr.Ro; }))});
    region      = w.app(region, {w.tuple(shapes), mr.So, w.tuple(tiles)});
    region      = w.app(region, w.tuple(DefVec(n, [&](size_t) { return axes; })));
    region      = w.app(region, ins->type());
    region      = w.app(region, ins);
    region      = w.app(region, body);
    for (size_t i = 0; i != yields.size(); ++i)
        replacements_[yields[i].producer] = region->proj(n, i + 1);
    return region->proj(n, 0);
}

} // namespace mim::plug::tensor::phase
