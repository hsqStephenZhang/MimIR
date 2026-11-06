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

bool opaque(const Def* def) {
    return def->isa<Lam>() || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>() || Axm::isa<tensor::tiled>(def)
        || Axm::isa<tensor::tiled_multi>(def) || Axm::isa<tensor::tile_fold>(def) || Axm::isa<tensor::tiled_n>(def);
}

/// The subterms of @p value, without entering lambdas or nested regions.
DefVec nodes(const Def* value) {
    DefVec result{value};
    DefSet seen{value};
    for (size_t i = 0; i != result.size(); ++i)
        if (!opaque(result[i]))
            for (auto op : result[i]->ops())
                if (op && seen.insert(op).second) result.push_back(op);
    return result;
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
        if (opaque(def)) continue;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return result;
}

/// The maximal tensor-valued subterms of @p value that do not depend on @p var; they become the region's inputs.
DefVec leaves(const Def* value, const Var* var, DefVec* composites = nullptr) {
    DefVec result, pending{value};
    DefSet seen{value};
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!depends(def, var)) {
            if (TensorType::isa(def->type())) {
                result.push_back(def);
                continue;
            }
            if (composites && !def->isa<Lam>() && !def->isa<Var>() && !def->isa<Axm>() && !def->isa<Pi>())
                composites->push_back(def);
        }
        if (def->isa<Lam>() || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return result;
}

/// Substitutes region inputs inside subterms without the region's variable, which VarRewriter leaves untouched.
class Inputs : public Rewriter {
public:
    using Rewriter::Rewriter;
    const Def* rewrite_mut(Def* mut) final { return map(mut, mut); }
};

/// @p value as the body of a region: @p var becomes @p binder and each of @p inputs a projection of @p ins.
const Def* close(const Def* value, const Var* var, const Def* binder, const DefVec& inputs, const Def* ins) {
    DefVec composites;
    leaves(value, var, &composites);
    VarRewriter rewriter(var, binder);
    Inputs plain(value->world());
    for (size_t i = 0, n = inputs.size(); i != n; ++i) {
        rewriter.map(inputs[i], ins->proj(n, i));
        plain.map(inputs[i], ins->proj(n, i));
    }
    for (auto composite : composites)
        rewriter.map(composite, plain.rewrite(composite));
    return rewriter.rewrite(value);
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

/// Whether output axis `d` of `map_out` reads loop `d` and no other loop, for every parallel loop.
/// Tiles of tiles write through shifted identities, so this checks dependence, not syntax.
bool canonical_write(const btensor::MapRedPost& mr, u64 ro, u64 rn) {
    auto& w   = mr.map_out->world();
    auto pi   = mr.map_out->type()->as<Pi>();
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(mr.map_out, var);
    for (u64 d = 0; d != ro; ++d) {
        auto axis = read->proj(ro, d);
        if (axis == var->proj(rn, d)) continue;
        DefVec pending{axis};
        DefSet seen{axis};
        bool own = false;
        for (size_t i = 0; i != pending.size(); ++i) {
            auto def = pending[i];
            if (def == var) return false;
            if (auto extract = def->isa<Extract>(); extract && extract->tuple() == var) {
                if (Lit::isa(extract->index()) != d) return false;
                own = true;
                continue;
            }
            for (auto op : def->ops())
                if (op && seen.insert(op).second) pending.push_back(op);
        }
        if (!own) return false;
    }
    return true;
}

/// The storage extents of @p tile: a clipped `map_reduce_post`'s storage cap, else its shape.
const Def* storage(const Def* tile) {
    auto app = graph::map_reduce(tile);
    if (!app) return TensorType::isa(tile->type())->s;
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    if (auto clipped = Axm::isa<btensor::clip_sched>(mr.sched)) return clipped->arg(4, 1);
    return mr.So;
}

/// Whether every storage extent of @p tile has a constant upper bound.
bool bounded_storage(const Def* tile) {
    auto extents = storage(tile);
    auto rank    = Lit::isa(extents->arity());
    if (!rank) return false;
    for (u64 d = 0; d != *rank; ++d)
        if (!nat_upper(extents->proj(*rank, d))) return false;
    return true;
}

/// @p app, a `map_reduce_post`, with schedule @p sched; a storage clip around the old schedule stays.
const Def* with_sched(const App* app, const Def* sched) {
    auto& w = app->world();
    DefVec args;
    const Def* head = app;
    for (auto curried = app; curried; curried = head->isa<App>()) {
        args.push_back(curried->arg());
        head = curried->callee();
    }
    std::ranges::reverse(args);
    auto [So, Sr, old] = args[2]->projs<3>();
    if (auto clipped = Axm::isa<btensor::clip_sched>(old)) {
        auto [valid, output_cap, loop_cap, _] = clipped->args<4>();
        sched                                 = w.app(clipped->callee(), {valid, output_cap, loop_cap, sched});
    }
    args[2] = w.tuple({So, Sr, sched});
    for (auto arg : args)
        head = w.app(head, arg);
    return head;
}

const Def* nat_min(const Def* a, const Def* b) {
    auto& w = a->world();
    return w.extract(w.tuple({b, a}), w.call(core::ncmp::l, Defs{a, b}));
}

/// Realizes the levels of one `tile_req`, each inside the region body of the level before.
class Levels {
public:
    Levels(const fe::Log& log, const App* request, const DefSet& members, const DefSet& outside, bool strict)
        : log_(log)
        , request_(request)
        , op_(request->arg(4, 0))
        , control_(request->arg(4, 3))
        , members_(members)
        , outside_(outside)
        , strict_(strict) {}

    /// The value of @p anchor tiled by @p levels; `nullptr` when a level cannot be realized, see reason().
    const Def* tile(const Def* anchor, fe::Span<const Def*> levels, u64 depth);
    std::string_view reason() const { return reason_; }
    /// Members handed back by the outermost region, with the region results that replace them.
    const auto& yielded() const { return yielded_; }

private:
    struct Origin {
        const Def *producer, *off;
    };
    struct Fused {
        const Def* value;
        size_t count = 0;
        DefVec yields;
    };

    const Def* fail(std::string reason) {
        reason_ = std::move(reason);
        return nullptr;
    }
    World& world() const { return op_->world(); }
    /// Whether @p own is the only open scope, so the request's members and outside uses apply.
    bool outermost(const Var* own) const {
        return std::ranges::all_of(scopes_, [&](const Var* v) { return v == own; });
    }
    /// The request's members and the tiles that enclosing levels fused.
    bool in_group(const Def* producer) const { return members_.contains(producer) || origins_.contains(producer); }
    const Def* tile_of(const Def* producer, const Def* off, const Def* ext, const Def* cap) const;
    template<class Id>
    const Def* reduction_tile(const btensor::MapRedPost& mr, const Def* anchor, DefVec args) const;
    Fused fuse(const Def* value, const Var* own, u64 depth, u64 kind);
    const Def* parallel(const Def* anchor,
                        const btensor::MapRedPost& mr,
                        u64 ro,
                        u64 rn,
                        fe::Span<const u64> sizes,
                        bool peel,
                        fe::Span<const Def*> rest,
                        u64 depth);
    const Def* reduction(const Def* anchor,
                         const btensor::MapRedPost& mr,
                         u64 ro,
                         u64 rn,
                         fe::Span<const u64> sizes,
                         bool peel,
                         fe::Span<const Def*> rest,
                         u64 depth);
    const Def* vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const Def*> rest, u64 depth);

    const fe::Log& log_;
    const App* request_;
    const Def *op_, *control_;
    const DefSet& members_;
    const DefSet& outside_;
    bool strict_;
    fe::Vector<const Var*> scopes_;
    DefMap<Origin> origins_;
    std::string reason_;
    std::vector<std::pair<const Def*, const Def*>> yielded_;
};

// Boxed tiles carry literal storage caps derived from the tile sizes into every fused producer.
const Def* Levels::tile_of(const Def* producer, const Def* off, const Def* ext, const Def* cap) const {
    auto& w     = world();
    auto tensor = *TensorType::isa(producer->type());
    if (cap) {
        auto meta = w.app(w.app(w.annex<tensor::tile_of_box>(), {tensor.T, tensor.r}), tensor.s);
        return w.app(meta, {producer, off, ext, cap});
    }
    return w.app(w.app(w.app(w.annex<tensor::tile_of>(), {tensor.T, tensor.r}), tensor.s), {producer, off, ext});
}

template<class Id>
const Def* Levels::reduction_tile(const btensor::MapRedPost& mr, const Def* anchor, DefVec args) const {
    auto& w = world();
    args.insert(args.begin(), anchor);
    auto shape = TensorType::isa(anchor->type())->s;
    auto tile  = w.app(w.app(w.app(w.annex<Id>(), {mr.To, mr.Tp, mr.Ro}), shape), w.tuple(args));
    return Axm::isa<Id>(tile) ? nullptr : tile;
}

Levels::Fused Levels::fuse(const Def* value, const Var* own, u64 depth, u64 kind) {
    auto& w = world();
    Fused result{value, 0, {}};
    DefSet visited;
    std::vector<std::pair<const Def*, Origin>> tiles;
    for (size_t step = 0; step != max_fusions; ++step) {
        std::optional<Read> next;
        for (auto read : reads(result.value))
            if (visited.insert(read.app).second) {
                next = read;
                break;
            }
        if (!next) break;
        auto p           = next->producer;
        auto producer    = graph::map_reduce(p);
        auto reduces     = producer
                        && btensor::MapRedPost::of(producer->callee()->as<App>())
                                   .ranks()
                                   .value_or(btensor::MapRedPost::Ranks{0, 0, 0, 0})
                                   .rr
                               != 0;
        auto outside_use = outermost(own) && outside_.contains(p);
        auto query
            = w.tuple({w.lit_nat(depth), w.lit_nat(kind), w.lit_nat_0(), w.lit_bool(reduces), w.lit_bool(outside_use),
                       w.lit_bool(in_group(p)), w.lit_nat(std::numeric_limits<nat_t>::max())});
        auto decision = w.app(control_, query);
        auto fuse = Lit::isa(decision->proj(2, 0)), yield = Lit::isa(decision->proj(2, 1));
        if (!fuse || !yield) {
            if (strict_) request_->blame("tensor.tile_req control does not decide on {}: {}", p, decision).bail();
            log_.d("tensor.tile_fuse {} skip {}: control is undecided", op_, p);
            continue;
        }
        if (!*fuse) continue;
        // A tile of a fused tile is a tile of the original producer at the composed offset.
        // The read lies within the fused tile, so the tile's storage bounds the new one.
        auto source = p, off = next->off, cap = next->cap;
        if (auto origin = origins_.find(p); origin != origins_.end()) {
            auto r     = *Lit::isa(TensorType::isa(p->type())->r);
            source     = origin->second.producer;
            off        = w.tuple(DefVec(r, [&](size_t d) {
                return w.call(core::nat::add, Defs{origin->second.off->proj(r, d), next->off->proj(r, d)});
            }));
            auto bound = storage(p);
            cap        = cap ? w.tuple(DefVec(r, [&](size_t d) { return nat_min(cap->proj(r, d), bound->proj(r, d)); }))
                             : bound;
        }
        if (auto tile = producer_tile(source, off, next->ext, cap); !tile.value) {
            log_.d("tensor.tile_fuse {} skip {}: {}", op_, p, tile.reason);
            continue;
        }
        auto tile = tile_of(source, off, next->ext, cap);
        if (!bounded_storage(tile)) {
            log_.d("tensor.tile_fuse {} skip {}: its tile storage {} has no constant bound", op_, p, storage(tile));
            continue;
        }
        Rewriter substitute(w);
        for (auto free : result.value->free_vars())
            substitute.map(free, free);
        substitute.map(next->app, tile);
        result.value = substitute.rewrite(result.value);
        for (auto& made : tiles)
            made.first = substitute.rewrite(made.first);
        tiles.emplace_back(tile, Origin{source, off});
        ++result.count;
        log_.d("tensor.tile_fuse {} fuse {}", op_, p);
        if (*yield && outside_use) result.yields.push_back(p);
    }
    for (auto [tile, origin] : tiles)
        origins_.emplace(tile, origin);
    return result;
}

const Def* Levels::tile(const Def* anchor, fe::Span<const Def*> levels, u64 depth) {
    if (levels.empty()) return anchor;
    auto root = graph::map_reduce(anchor);
    if (!root) return fail("the root tile is not a map_reduce_post");
    auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
    auto rks = mr.ranks();
    if (!rks) return fail("the root's ranks are not literal");
    auto ro = rks->ro, rn = rks->nloops();
    auto [kind, sizes, tail, promote] = levels.front()->projs<4>();
    auto k = Lit::isa(kind), t = Lit::isa(tail);
    if (!k || !t || *t > 1) return fail("level kinds and tails must be literals");
    if (Lit::isa(promote) != 0) return fail("promotion is not supported");
    auto literal = fe::Vector<u64>(rn);
    for (u64 d = 0; d != rn; ++d)
        if (auto size = Lit::isa(sizes->proj(rn, d)))
            literal[d] = *size;
        else
            return fail("tile sizes must be literals");
    auto rest = levels.subspan(1);
    switch (*k) {
        case 0: return parallel(anchor, mr, ro, rn, literal, *t == 1, rest, depth);
        case 1: return reduction(anchor, mr, ro, rn, literal, *t == 1, rest, depth);
        case 2: return vector(root, rn, literal, rest, depth);
        default: return fail("level kinds are 0 (parallel), 1 (reduction) or 2 (vector)");
    }
}

const Def* Levels::parallel(const Def* anchor,
                            const btensor::MapRedPost& mr,
                            u64 ro,
                            u64 rn,
                            fe::Span<const u64> sizes,
                            bool peel,
                            fe::Span<const Def*> rest,
                            u64 depth) {
    auto& w = world();
    if (!canonical_write(mr, ro, rn)) return fail("the root does not write loop d to output axis d");
    auto tiles = DefVec(ro);
    bool whole = true;
    for (u64 d = 0; d != rn; ++d) {
        if (d >= ro) {
            if (sizes[d]) return fail("a parallel level cannot tile a reduction loop");
            continue;
        }
        tiles[d] = sizes[d] ? w.lit_nat(sizes[d]) : mr.So->proj(ro, d);
        whole &= tiles[d] == mr.So->proj(ro, d);
    }
    if (whole) return tile(anchor, rest, depth + 1);

    auto box   = w.sigma({w.arr(mr.Ro, w.type_nat()), w.arr(mr.Ro, w.type_nat())});
    auto probe = w.mut_lam(box, w.type_nat());
    auto var   = probe->var()->as<Var>();
    auto off = probe->var(2, 0), ext = probe->var(2, 1);
    scopes_.push_back(var);
    auto fused = fuse(tile_of(anchor, off, ext, w.tuple(tiles)), var, depth, 0);
    auto value = tile(fused.value, rest, depth + 1);
    scopes_.pop_back();
    if (!value) return nullptr;

    struct Yield {
        const Def *producer, *tile;
    };
    std::vector<Yield> yields;
    for (auto p : fused.yields) {
        // The region hands back a member on its own grid; that tile is not shared with the fused read, whose box
        // equals the region's only under `off + ext ≤ s`, which the clamped read box does not show syntactically.
        if (auto tensor = TensorType::isa(p->type()); tensor && tensor->s == mr.So)
            yields.push_back({p, tile_of(p, off, ext, w.tuple(tiles))});
        else
            log_.d("tensor.tile_fuse {} skip yield of {}: its shape is not the region's", op_, p);
    }
    auto unfused = leaves(value, var);
    std::erase_if(yields, [&](const Yield& y) {
        bool read = std::ranges::find(unfused, y.producer) != unfused.end();
        if (read) log_.d("tensor.tile_fuse {} skip yield of {}: the region still reads it whole", op_, y.producer);
        return read;
    });
    if (!yields.empty()) {
        DefVec results{value};
        for (const auto& y : yields)
            results.push_back(y.tile);
        value = w.tuple(results);
    }
    auto inputs = leaves(value, var);
    auto ins    = w.tuple(inputs);
    auto pi     = w.mut_pi(w.type());
    pi->set_dom(box);
    pi->set_codom(w.pi(ins->type(), VarRewriter(var, pi->var()).rewrite(value->type())));
    auto body  = w.mut_lam(pi);
    auto inner = w.mut_lam(pi->reduce(body->var())->as<Pi>());
    inner->set(true, close(value, var, body->var(), inputs, inner->var()));
    body->set(true, inner);

    log_.d("tensor.tile_fuse {} L{} par tiles={} tail={} inputs={} fused={} yielded={}", op_, depth, w.tuple(tiles),
           peel ? "peel" : "cap", inputs.size(), fused.count, yields.size());
    if (yields.empty()) {
        auto region = w.app(w.annex<tensor::tiled>(), {mr.Tp, mr.Ro});
        region      = w.app(region, {mr.So, w.tuple(tiles), w.lit_nat(peel)});
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
    region      = w.app(region, {w.tuple(shapes), mr.So, w.tuple(tiles), w.lit_nat(peel)});
    region      = w.app(region, w.tuple(DefVec(n, [&](size_t) { return axes; })));
    region      = w.app(region, ins->type());
    region      = w.app(region, ins);
    region      = w.app(region, body);
    for (size_t i = 0; i != yields.size(); ++i)
        yielded_.emplace_back(yields[i].producer, region->proj(n, i + 1));
    return region->proj(n, 0);
}

const Def* Levels::reduction(const Def* anchor,
                             const btensor::MapRedPost& mr,
                             u64 ro,
                             u64 rn,
                             fe::Span<const u64> sizes,
                             bool peel,
                             fe::Span<const Def*> rest,
                             u64 depth) {
    auto& w   = world();
    auto loop = rn;
    for (u64 d = 0; d != rn; ++d) {
        if (!sizes[d]) continue;
        if (d < ro) return fail("a reduction level cannot tile a parallel loop");
        if (loop != rn) return fail("a reduction level tiles a single reduction loop");
        loop = d;
    }
    if (loop == rn) return tile(anchor, rest, depth + 1);
    auto K        = mr.Sr->proj(rn, loop);
    const Def* ks = w.lit_nat(sizes[loop]);
    if (auto k = Lit::isa(K); k && *k <= sizes[loop]) return tile(anchor, rest, depth + 1);

    auto off  = w.pack(mr.Ro, w.lit_nat_0());
    auto ext  = mr.So;
    auto init = reduction_tile<tile_of_init>(mr, anchor, {off, ext});
    if (!init) return fail("the root's accumulator has no tile");
    auto acc_type = init->type();
    auto dom      = w.sigma({acc_type, w.type_nat()});
    auto probe    = w.mut_lam(dom, w.type_nat());
    auto var      = probe->var()->as<Var>();
    auto acc = probe->var(2, 0), ko = probe->var(2, 1);
    auto chunk   = peel ? ks : nat_min(ks, w.call(core::nat::sub, Defs{K, ko}));
    auto partial = reduction_tile<tile_of_partial>(mr, anchor, {off, ext, w.lit_nat(loop), ko, chunk, acc});
    if (!partial) return fail("the root has no reduction chunk");
    scopes_.push_back(var);
    auto fused = fuse(partial, var, depth, 1);
    auto value = tile(fused.value, rest, depth + 1);
    scopes_.pop_back();
    if (!value) return nullptr;

    auto inputs = leaves(value, var);
    auto ins    = w.tuple(inputs);
    auto step   = w.mut_lam(dom, w.pi(ins->type(), acc_type));
    auto inner  = w.mut_lam(ins->type(), acc_type);
    inner->set(true, close(value, var, step->var(), inputs, inner->var()));
    step->set(true, inner);

    auto blocks = peel ? w.call(core::nat::div, Defs{K, ks})
                       : w.call(core::nat::div, Defs{w.call(core::nat::add, Defs{K, w.lit_nat(sizes[loop] - 1)}), ks});
    auto folded = w.app(w.app(w.annex<tensor::tile_fold>(), acc_type), {blocks, ks});
    folded      = w.app(w.app(w.app(w.app(folded, init), ins->type()), ins), step);
    auto tail_fused = size_t(0);
    if (peel) {
        auto hole = w.mut_lam(acc_type, w.type_nat())->var()->as<Var>();
        auto at   = w.call(core::nat::mul, Defs{blocks, ks});
        auto last = reduction_tile<tile_of_partial>(
            mr, anchor, {off, ext, w.lit_nat(loop), at, w.call(core::nat::rem, Defs{K, ks}), hole});
        if (!last) return fail("the root has no reduction tail");
        auto tail  = fuse(last, nullptr, depth, 1);
        tail_fused = tail.count;
        auto done  = tile(tail.value, rest, depth + 1);
        if (!done) return nullptr;
        folded = VarRewriter(hole, folded).rewrite(done);
    }
    auto result = reduction_tile<tile_of_post>(mr, anchor, {off, ext, folded});
    if (!result) return fail("the root's epilogue has no tile");
    log_.d("tensor.tile_fuse {} L{} red loop={} ks={} tail={} inputs={} fused={} tail_fused={}", op_, depth, loop, ks,
           peel ? "peel" : "cap", inputs.size(), fused.count, tail_fused);
    return result;
}

const Def* Levels::vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const Def*> rest, u64 depth) {
    auto& w = world();
    if (!rest.empty()) return fail("a vector level must be the innermost");
    auto vdim = rn;
    for (u64 d = 0; d != rn; ++d) {
        if (!sizes[d]) continue;
        if (vdim != rn || sizes[d] != 1) return fail("a vector level marks a single loop with size 1");
        vdim = d;
    }
    if (vdim == rn) return root;
    log_.d("tensor.tile_fuse {} L{} vec vdim={}", op_, depth, vdim);
    return with_sched(root, w.app(w.annex<btensor::mk_sched>(), w.tuple({w.lit_nat(vdim), w.lit_nat_0()})));
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
    auto [op, members, levels, control] = request->args<4>();
    auto level_list                     = tile_levels(levels);
    if (level_list.empty()) {
        report_tile_request(request, "the levels are not a literal list");
        return op;
    }
    auto group = members_of(members);
    DefSet in_group(group.begin(), group.end());
    Levels realizer(log(), request, in_group, outside, driver().arg_nat("tensor", "strict").value_or(0));
    auto value = realizer.tile(op, level_list, 0);
    if (!value) {
        report_tile_request(request, realizer.reason());
        return op;
    }
    for (auto [producer, result] : realizer.yielded())
        replacements_[producer] = result;
    return value;
}

} // namespace mim::plug::tensor::phase
