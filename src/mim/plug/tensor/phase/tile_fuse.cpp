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

/// Substitutes within one world and rebuilds only the mutables that reach a substituted Def.
/// Rebuilding any other mutable, such as a record type, would fork the identity of every term built on it.
class Substitute : public Rewriter {
public:
    using Rewriter::Rewriter;
    const Def* substitute(const Def* old_def, const Def* new_def) {
        targets_.insert(old_def);
        return map(old_def, new_def);
    }
    const Def* rewrite_mut(Def* mut) final { return reaches(mut) ? Rewriter::rewrite_mut(mut) : map(mut, mut); }

private:
    bool reaches(const Def* root) {
        DefSet seen{root};
        for (DefVec pending{root}; !pending.empty();) {
            auto def = pending.back();
            pending.pop_back();
            if (targets_.contains(def)) return true;
            for (auto dep : def->deps())
                if (dep && seen.insert(dep).second) pending.push_back(dep);
        }
        return false;
    }

    DefSet targets_;
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

/// The axes of an input of rank @p rank that read the first @p ro loops of @p map, in increasing order.
std::optional<std::vector<u64>> grid_axes(const Def* map, u64 ro, u64 rank) {
    auto& w = map->world();
    auto pi = map->type()->as<Pi>();
    auto rn = Lit::isa(pi->dom()->arity());
    if (!rn) return {};
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(map, var);
    auto axes = std::vector<u64>();
    for (u64 d = 0, a = 0; d != ro; ++d, ++a) {
        Linear expected;
        expected.atoms[var->proj(*rn, d)] = 1;
        while (a != rank && !(Linear::of(read->proj(rank, a)) == expected))
            ++a;
        if (a == rank) return {};
        axes.push_back(a);
    }
    return axes;
}

/// The members that every reader in the group reads pointwise on the root's grid, with the axes the grid maps to.
/// A root tile reads such a member only inside its grid box, whole along its other axes, so one tile serves all.
DefMap<std::vector<u64>> aligned_members(const Def* op, const DefSet& members) {
    auto root = graph::map_reduce(op);
    if (!root) return {};
    auto shape   = btensor::MapRedPost::of(root->callee()->as<App>()).So;
    auto aligned = DefMap<std::vector<u64>>();
    for (auto member : members)
        aligned.emplace(member, std::vector<u64>());
    for (bool changed = true; changed;) {
        changed    = false;
        auto found = DefMap<std::vector<u64>>();
        DefVec readers{op};
        readers.insert(readers.end(), members.begin(), members.end());
        for (auto reader : readers) {
            auto app = graph::map_reduce(reader);
            if (!app) continue;
            auto mr            = btensor::MapRedPost::of(app->callee()->as<App>());
            auto ranks         = mr.ranks();
            auto trusted       = ranks && (reader == op || aligned.contains(reader)) && mr.So == shape
                              && canonical_write(mr, ranks->ro, ranks->nloops());
            auto [is, post_is] = app->arg()->projs<2>();
            auto check         = [&](const Def* inputs, const Def* maps, u64 n) {
                for (u64 i = 0; i != n; ++i) {
                    auto input = inputs->proj(n, i);
                    if (!members.contains(input)) continue;
                    auto tensor = TensorType::isa(input->type());
                    auto rank   = tensor ? Lit::isa(tensor->r) : std::nullopt;
                    auto axes   = trusted && rank ? grid_axes(maps->proj(n, i), ranks->ro, *rank) : std::nullopt;
                    bool ok     = axes.has_value();
                    for (u64 d = 0; ok && d != axes->size(); ++d)
                        ok = tensor->s->proj(*rank, (*axes)[d]) == shape->proj(ranks->ro, d);
                    if (ok) {
                        auto [prev, fresh] = found.emplace(input, *axes);
                        ok                 = fresh || prev->second == *axes;
                    }
                    if (!ok && aligned.erase(input)) changed = true;
                }
            };
            if (!ranks) {
                changed = !aligned.empty();
                aligned.clear();
                break;
            }
            check(is, mr.maps, ranks->nis);
            check(post_is, mr.post_maps, ranks->nps);
        }
        for (auto& [member, axes] : aligned)
            if (auto i = found.find(member); i != found.end()) axes = i->second;
    }
    // A member no aligned reader reads stays aligned only if it has the root's shape.
    auto ro = Lit::isa(shape->arity()).value_or(0);
    std::erase_if(aligned, [&](auto& entry) {
        auto tensor = TensorType::isa(entry.first->type());
        if (entry.second.size() == ro) return false;
        if (!tensor || tensor->s != shape) return true;
        for (u64 d = 0; d != ro; ++d)
            entry.second.push_back(d);
        return false;
    });
    return aligned;
}

/// The storage extents of @p tile: a clipped `map_reduce_post`'s storage cap, else its shape.
const Def* storage(const Def* tile) {
    auto app = graph::map_reduce(tile);
    if (!app) return TensorType::isa(tile->type())->s;
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    if (auto clipped = Axm::isa<btensor::clip_sched>(mr.sched)) return clipped->arg(4, 1);
    return mr.So;
}

/// Whether every inner storage extent of @p tile has a constant upper bound.
bool bounded_storage(const Def* tile) {
    auto extents = storage(tile);
    auto rank    = Lit::isa(extents->arity());
    if (!rank) return false;
    // Buffers take a runtime outermost extent; only the inner ones need a bound.
    for (u64 d = 1; d < *rank; ++d)
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

/// @p app with its last reduction loop of literal extent `K` split into `(K / factor, factor)` and the inner one
/// unrolled, so a micro tile reads `factor` consecutive elements of each row per step (the template's `kt`).
const Def* split_reduction(const App* app, u64 factor, u64 vdim) {
    auto& w = app->world();
    DefVec args;
    const Def* head = app;
    for (auto curried = app; curried; curried = head->isa<App>()) {
        args.push_back(curried->arg());
        head = curried->callee();
    }
    std::ranges::reverse(args);
    auto mr   = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ro   = Lit::as(mr.Ro);
    auto rn   = Lit::as(mr.Rn);
    auto loop = rn - 1;
    auto K    = Lit::isa(mr.Sr->proj(rn, loop));
    if (loop < ro || !K || *K % factor) return nullptr;
    auto index = w.annex<affine::Index>();
    auto split = w.mut_lam(w.pi(w.arr(rn + 1, index), w.arr(rn, index)));
    split->set(true, w.tuple(DefVec(rn, [&](size_t d) -> const Def* {
        if (d != loop) return split->var(rn + 1, d);
        auto outer = w.call(affine::semiop::mul, Defs{split->var(rn + 1, d), w.lit_nat(factor)});
        return w.call(affine::op::add, Defs{outer, split->var(rn + 1, d + 1)});
    })));
    auto compose = [&](const Def* map) {
        auto pi  = map->type()->as<Pi>();
        auto lam = w.mut_lam(w.pi(w.arr(rn + 1, index), pi->codom()));
        lam->set(true, w.app(map, w.app(split, lam->var())));
        return lam;
    };
    auto widen = [&](const Def* extents) {
        return w.tuple(DefVec(rn + 1, [&](size_t d) -> const Def* {
            if (d < loop) return extents->proj(rn, d);
            if (d == loop) return w.lit_nat(*K / factor);
            return w.lit_nat(factor);
        }));
    };
    auto sched = w.app(w.annex<btensor::mk_sched>(), w.tuple({w.lit_nat(vdim), w.lit_nat_1()}));
    if (auto clipped = Axm::isa<btensor::clip_sched>(mr.sched)) {
        auto [valid, output_cap, loop_cap, _] = clipped->args<4>();
        auto clip                             = w.app(w.annex<btensor::clip_sched>(), {mr.Ro, w.lit_nat(rn + 1)});
        sched                                 = w.app(clip, {valid, output_cap, widen(loop_cap), sched});
    }
    auto init = mr.init;
    if (auto initial = tensor_init(mr); initial && mr.init->num_projs() == 3)
        init = w.tuple({initial->shape, compose(initial->map), initial->outs});
    auto nis  = Lit::as(mr.nis);
    auto maps = w.tuple(DefVec(nis, [&](size_t i) -> const Def* { return compose(mr.maps->proj(nis, i)); }));
    auto [To, Tp, Ro, Rn, TSched] = mr.meta->projs<5>();
    args[1]                       = w.tuple({To, Tp, Ro, w.lit_nat(rn + 1), TSched});
    args[2]                       = w.tuple({mr.So, widen(mr.Sr), sched});
    args[4]                       = init->type();
    args[5]                       = w.tuple({mr.comb, init, mr.post});
    args[6]                       = compose(mr.map_out);
    args[7]                       = w.tuple({maps, mr.post_maps});
    for (auto arg : args)
        head = w.app(head, arg);
    return head;
}

/// @p builder applied to each argument group in turn.
const Def* build(const Def* builder, std::initializer_list<const Def*> groups) {
    for (auto group : groups)
        builder = builder->world().app(builder, group);
    return builder;
}

/// Whether the immutable term @p def is computed from @p target.
bool depends_on(const Def* def, const Def* target) {
    DefSet seen{def};
    for (DefVec pending{def}; !pending.empty();) {
        auto d = pending.back();
        pending.pop_back();
        if (d == target) return true;
        if (d->isa_mut()) continue;
        for (auto op : d->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return false;
}

const Def* slice_box_of(const Def* tensor, const Def* off, const Def* ext, const Def* cap) {
    auto& w   = tensor->world();
    auto type = *TensorType::isa(tensor->type());
    auto view = w.app(w.app(w.annex<slice_box>(), {type.T, type.r}), type.s);
    return w.app(w.app(view, {off, ext, cap}), tensor);
}

const Def* nat_min(const Def* a, const Def* b) {
    auto& w = a->world();
    return w.extract(w.tuple({b, a}), w.call(core::ncmp::l, Defs{a, b}));
}

/// The axis of a rank-2 input read by @p map that one of the loops in @p tiles projects onto, with that loop's tile.
std::optional<std::pair<u64, u64>> panel_axis(const Def* map, u64 rank, fe::Span<const u64> tiles) {
    auto& w   = map->world();
    auto pi   = map->type()->as<Pi>();
    auto rn   = Lit::isa(pi->dom()->arity());
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(map, var);
    if (!rn || rank != 2) return {};
    std::optional<std::pair<u64, u64>> result;
    // Offsets of enclosing tiles are constants; only the loops decide the axis.
    for (u64 d = 0; d != rank; ++d) {
        auto form = Linear::of(read->proj(rank, d));
        for (u64 l = 0; l != tiles.size(); ++l) {
            bool own = true;
            for (u64 k = 0; k != *rn; ++k) {
                auto atom = form.atoms.find(var->proj(*rn, k));
                own &= (atom == form.atoms.end() ? 0 : atom->second) == (k == l ? 1 : 0);
            }
            if (!tiles[l] || !own) continue;
            if (result) return {};
            result = std::pair{d, tiles[l]};
        }
    }
    return result;
}

/// @p app with its input @p slot staged into panels of @p tile along @p axis, the panel index outermost (BLIS B).
/// The read map follows the layout: `x ↦ (x#axis / tile, x#other, x#axis % tile)`.
const App* promote_input(const App* app, u64 slot, u64 axis, u64 tile) {
    auto& w            = app->world();
    auto mr            = btensor::MapRedPost::of(app->callee()->as<App>());
    auto nis           = Lit::as(mr.nis);
    auto [is, post_is] = app->args<2>();
    auto input         = is->proj(nis, slot);
    auto type          = *TensorType::isa(input->type());
    auto other         = 1 - axis;
    auto perm          = w.tuple({w.lit_idx(2, axis), w.lit_idx(2, other)});
    auto stage_app = build(w.annex<stage>(),
                           {w.tuple({type.T, w.lit_nat(2), w.lit_nat_1()}), type.s,
                            w.tuple({w.lit_nat(axis), w.lit_nat(tile), perm, w.bot(type.T)}), w.lit_nat_0(), input});
    auto old_map   = mr.maps->proj(nis, slot);
    auto map       = w.mut_lam(w.pi(old_map->type()->as<Pi>()->dom(), w.arr(3, w.annex<affine::Index>())));
    auto coords    = w.app(old_map, map->var());
    auto at        = coords->proj(2, axis);
    map->set(true, w.tuple({w.call(affine::semiop::floordiv, Defs{at, w.lit_nat(tile)}), coords->proj(2, other),
                            w.call(affine::semiop::rem, Defs{at, w.lit_nat(tile)})}));
    auto replace = [&](const Def* tuple, u64 n, const Def* value) {
        return w.tuple(DefVec(n, [&](size_t i) { return i == slot ? value : tuple->proj(n, i); }));
    };
    auto staged = TensorType::isa(stage_app->type());
    auto in_tys = w.tuple(
        {mr.Tis, replace(mr.Ris, nis, w.lit_nat(3)), replace(mr.Sis, nis, staged->s), mr.Tps, mr.Rps, mr.Sps});
    auto result = build(w.annex<map_reduce_post>(),
                        {mr.nis_nps, mr.meta, mr.shapes, in_tys, mr.init->type(), w.tuple({mr.comb, mr.init, mr.post}),
                         mr.map_out, w.tuple({replace(mr.maps, nis, map), mr.post_maps}),
                         w.tuple({replace(is, nis, stage_app), post_is})});
    return result->as<App>();
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
        , aligned_(aligned_members(op_, members))
        , outside_(outside)
        , strict_(strict) {}

    /// The value of @p anchor tiled by @p levels; `nullptr` when a level cannot be realized, see reason().
    const Def* tile(const Def* anchor, fe::Span<const Def*> levels, u64 depth);
    std::string_view reason() const { return reason_; }
    /// Members handed back by the outermost region, with the region results that replace them.
    const auto& yielded() const { return yielded_; }

private:
    struct Origin {
        const Def *producer, *off, *ext, *cap;
    };
    struct Fused {
        const Def* value;
        size_t count = 0;
        DefVec yields;
        std::vector<std::pair<const Def*, Origin>> made;
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
    bool aligned(const Def* producer) const { return axes_of(producer) != nullptr; }
    const std::vector<u64>* axes_of(const Def* producer) const {
        auto origin = origins_.find(producer);
        auto i      = aligned_.find(origin == origins_.end() ? producer : origin->second.producer);
        return i == aligned_.end() ? nullptr : &i->second;
    }
    Origin compose(const Def* producer, const Def* off, const Def* ext, const Def* cap) const;
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
                         u64 promote,
                         fe::Span<const Def*> rest,
                         u64 depth);
    const Def* vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const Def*> rest, u64 depth);

    const fe::Log& log_;
    const App* request_;
    const Def *op_, *control_;
    const DefSet& members_;
    DefMap<std::vector<u64>> aligned_;
    const DefSet& outside_;
    bool strict_;
    fe::Vector<const Var*> scopes_;
    DefMap<Origin> origins_;
    /// Tiles that enclosing levels hand back, by their original producer, and the inner results that replace them.
    DefMap<const Def*> escaping_;
    DefMap<const Def*> handed_;
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

// A tile of a fused tile is a tile of the original producer at the composed offset.
// The read lies within the fused tile, so the tile's storage bounds the new one.
Levels::Origin Levels::compose(const Def* producer, const Def* off, const Def* ext, const Def* cap) const {
    auto origin = origins_.find(producer);
    if (origin == origins_.end()) return {producer, off, ext, cap};
    auto& w    = world();
    auto r     = *Lit::isa(TensorType::isa(producer->type())->r);
    auto bound = storage(producer);
    auto start = w.tuple(DefVec(
        r, [&](size_t d) { return w.call(core::nat::add, Defs{origin->second.off->proj(r, d), off->proj(r, d)}); }));
    cap = cap ? w.tuple(DefVec(r, [&](size_t d) { return nat_min(cap->proj(r, d), bound->proj(r, d)); })) : bound;
    return {origin->second.producer, start, ext, cap};
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
    Fused result{value, 0, {}, {}};
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
        auto origin      = compose(p, next->off, next->ext, next->cap);
        auto escaping    = escaping_.find(origin.producer);
        auto outside_use = (outermost(own) && outside_.contains(p)) || escaping != escaping_.end();
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
        auto source = origin.producer, off = origin.off, cap = origin.cap;
        if (auto tile = producer_tile(source, off, next->ext, cap); !tile.value) {
            log_.d("tensor.tile_fuse {} skip {}: {}", op_, p, tile.reason);
            continue;
        }
        auto tile = tile_of(source, off, next->ext, cap);
        if (!bounded_storage(tile)) {
            log_.d("tensor.tile_fuse {} skip {}: its tile storage {} has no constant bound", op_, p, storage(tile));
            continue;
        }
        Substitute substitute(w);
        for (auto free : result.value->free_vars())
            substitute.map(free, free);
        substitute.substitute(next->app, tile);
        result.value = substitute.rewrite(result.value);
        for (auto& made : tiles)
            made.first = substitute.rewrite(made.first);
        tiles.emplace_back(tile, origin);
        ++result.count;
        log_.d("tensor.tile_fuse {} fuse {}", op_, p);
        if (*yield && outside_use) result.yields.push_back(escaping != escaping_.end() ? escaping->second : p);
    }
    for (auto [tile, origin] : tiles)
        origins_.emplace(tile, origin);
    result.made = std::move(tiles);
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
    auto promoted = Lit::isa(promote);
    if (!promoted || (*promoted && *k != 1)) return fail("only reduction levels promote an input");
    auto literal = fe::Vector<u64>(rn);
    for (u64 d = 0; d != rn; ++d)
        if (auto size = Lit::isa(sizes->proj(rn, d)))
            literal[d] = *size;
        else
            return fail("tile sizes must be literals");
    auto rest = levels.subspan(1);
    switch (*k) {
        case 0: return parallel(anchor, mr, ro, rn, literal, *t == 1, rest, depth);
        case 1: return reduction(anchor, mr, ro, rn, literal, *t == 1, *promoted, rest, depth);
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
    // The outermost level stays a region even over the whole shape, so tile bodies own every intermediate buffer.
    // A scalar root has no grid to tile.
    if (whole && (depth || !ro)) return tile(anchor, rest, depth + 1);

    auto box   = w.sigma({w.arr(mr.Ro, w.type_nat()), w.arr(mr.Ro, w.type_nat())});
    auto probe = w.mut_lam(box, w.type_nat());
    auto var   = probe->var()->as<Var>();
    auto off = probe->var(2, 0), ext = probe->var(2, 1);
    auto root_tile = tile_of(anchor, off, ext, w.tuple(tiles));
    if (Axm::isa<tile_of_box>(root_tile)) {
        log_.d("tensor.tile_fuse {} L{} skip: the root has no tile here: {}", op_, depth,
               producer_tile(anchor, off, ext, w.tuple(tiles)).reason);
        return tile(anchor, rest, depth + 1);
    }
    scopes_.push_back(var);
    auto fused = fuse(root_tile, var, depth, 0);

    struct Yield {
        const Def *producer, *source, *tile;
        u64 rank;
        std::vector<u64> axes;
    };
    std::vector<Yield> yields;
    auto enclosing = escaping_;
    for (auto p : fused.yields) {
        auto tensor = TensorType::isa(p->type());
        auto rank   = tensor ? Lit::isa(tensor->r) : std::nullopt;
        auto axes   = std::vector<u64>();
        if (auto aligned = axes_of(p))
            axes = *aligned;
        else if (tensor && tensor->s == mr.So)
            for (u64 d = 0; d != ro; ++d)
                axes.push_back(d);
        bool fits = rank && axes.size() == ro;
        for (u64 d = 0; fits && d != ro; ++d)
            fits = tensor->s->proj(*rank, axes[d]) == mr.So->proj(ro, d);
        if (!fits) {
            log_.d("tensor.tile_fuse {} skip yield of {}: the region's grid does not cover it", op_, p);
            continue;
        }
        // The member's box follows the grid along its grid axes and is whole along the others.
        auto shape = tensor->s->projs(*rank);
        DefVec start(*rank, w.lit_nat_0()), extent(shape.begin(), shape.end()), cap(shape.begin(), shape.end());
        for (u64 d = 0; d != ro; ++d) {
            start[axes[d]]  = off->proj(ro, d);
            extent[axes[d]] = ext->proj(ro, d);
            cap[axes[d]]    = tiles[d];
        }
        auto origin = compose(p, w.tuple(start), w.tuple(extent), w.tuple(cap));
        auto handed = tile_of(origin.producer, origin.off, origin.ext, origin.cap);
        // The fused reads of an aligned producer lie inside the region's box, so they become views of the handed tile,
        // which inner levels then tile and hand back in turn.
        if (aligned(p)) {
            Substitute views(w);
            for (auto free : fused.value->free_vars())
                views.map(free, free);
            for (auto [made, from] : fused.made)
                if (from.producer == origin.producer) {
                    auto r     = Lit::as(tensor->r);
                    auto start = w.tuple(DefVec(r, [&](size_t d) {
                        return w.call(core::nat::sub, Defs{from.off->proj(r, d), origin.off->proj(r, d)});
                    }));
                    views.substitute(made, slice_box_of(handed, start, from.ext, from.cap ? from.cap : from.ext));
                }
            fused.value = views.rewrite(fused.value);
            for (auto& [made, from] : fused.made)
                if (from.producer != origin.producer) origins_.emplace(made = views.rewrite(made), from);
            origins_[handed]           = origin;
            escaping_[origin.producer] = handed;
        }
        yields.push_back({p, origin.producer, handed, *rank, std::move(axes)});
    }
    auto value = tile(fused.value, rest, depth + 1);
    scopes_.pop_back();
    escaping_ = std::move(enclosing);
    if (!value) return nullptr;
    for (auto& y : yields)
        if (auto inner = handed_.find(y.tile); inner != handed_.end()) y.tile = inner->second;
    auto unfused = leaves(value, var);
    std::erase_if(yields, [&](const Yield& y) {
        bool read = std::ranges::find(unfused, y.producer) != unfused.end();
        if (read) log_.d("tensor.tile_fuse {} skip yield of {}: the region still reads it whole", op_, y.producer);
        // The region's result would replace the producer that one of its own inputs is computed from.
        bool cycle
            = !read && std::ranges::any_of(unfused, [&](const Def* input) { return depends_on(input, y.source); });
        if (cycle) log_.d("tensor.tile_fuse {} skip yield of {}: a region input depends on it", op_, y.producer);
        return read || cycle;
    });
    // Handed tiles come first, so the region writes them before the root reads them through views.
    if (!yields.empty()) {
        DefVec results;
        for (const auto& y : yields)
            results.push_back(y.tile);
        results.push_back(value);
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
    auto grid = w.tuple({mr.So, w.tuple(tiles), w.lit_nat(peel)});
    if (yields.empty()) return build(w.annex<tf_par>(), {w.tuple({mr.Tp, mr.Ro}), ins->type(), grid, ins, body});
    auto n     = yields.size() + 1;
    auto types = DefVec(), ranks = DefVec(), shapes = DefVec(), maps = DefVec();
    for (const auto& y : yields) {
        auto tensor = TensorType::isa(y.producer->type());
        types.push_back(tensor->T);
        ranks.push_back(w.lit_nat(y.rank));
        shapes.push_back(tensor->s);
        maps.push_back(w.tuple(DefVec(ro, [&](size_t d) { return w.lit_idx(y.rank, y.axes[d]); })));
    }
    types.push_back(mr.Tp);
    ranks.push_back(mr.Ro);
    shapes.push_back(mr.So);
    maps.push_back(w.tuple(DefVec(ro, [&](size_t d) { return w.lit_idx(ro, d); })));
    auto meta   = w.tuple({w.lit_nat(n), mr.Ro, w.tuple(types), w.tuple(ranks)});
    auto layout = w.tuple({w.tuple(shapes), mr.So, w.tuple(tiles), w.lit_nat(peel)});
    auto region = build(w.annex<tf_par_n>(), {meta, ins->type(), layout, w.tuple(maps), ins, body});
    for (size_t i = 0; i != yields.size(); ++i)
        if (scopes_.empty() && members_.contains(yields[i].producer))
            yielded_.emplace_back(yields[i].producer, region->proj(n, i));
        else
            handed_[yields[i].producer] = region->proj(n, i);
    return region->proj(n, n - 1);
}

const Def* Levels::reduction(const Def* anchor,
                             const btensor::MapRedPost& mr,
                             u64 ro,
                             u64 rn,
                             fe::Span<const u64> sizes,
                             bool peel,
                             u64 promote,
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
    auto tail  = w.lit_nat(peel);
    auto chunk = build(w.annex<tf_chunk>(), {w.tuple({K, ks, tail, ko})});
    // Each promoted input (bit i of promote) is staged per block into panels as wide as the next parallel level's tile
    // of its axis.
    std::vector<std::tuple<u64, u64, u64>> panels;
    if (promote) {
        auto tiles = fe::Vector<u64>(ro, 0);
        for (auto level : rest)
            if (auto [kind, inner, tail, _] = level->projs<4>(); Lit::isa(kind) == 0) {
                for (u64 d = 0; d != ro; ++d)
                    tiles[d] = Lit::isa(inner->proj(rn, d)).value_or(0);
                break;
            }
        auto nis = Lit::as(mr.nis);
        if (promote >> nis) return fail("a promoted input does not exist");
        for (u64 slot = 0; slot != nis; ++slot) {
            if (!(promote >> slot & 1)) continue;
            auto panel = panel_axis(mr.maps->proj(nis, slot), Lit::as(mr.Ris->proj(nis, slot)), tiles);
            if (!panel) {
                log_.d("tensor.tile_fuse {} L{} skip promoting input {}: no axis tiled by the next parallel level", op_,
                       depth, slot);
                continue;
            }
            panels.emplace_back(slot, panel->first, panel->second);
            log_.d("tensor.tile_fuse {} L{} promote input {} axis {} panel {}", op_, depth, slot, panel->first,
                   panel->second);
        }
    }
    auto stage_block = [&](const Def* block) -> const Def* {
        for (auto [slot, axis, tile] : panels) {
            auto app = graph::map_reduce(block);
            if (!app) return nullptr;
            block = promote_input(app, slot, axis, tile);
        }
        return block;
    };
    auto partial = reduction_tile<tile_of_partial>(mr, anchor, {off, ext, w.lit_nat(loop), ko, chunk, acc});
    if (!partial) return fail("the root has no reduction chunk");
    scopes_.push_back(var);
    // Members are fused first, so a promoted input stages its fused tile.
    auto fused  = fuse(partial, var, depth, 1);
    fused.value = stage_block(fused.value);
    if (!fused.value) return fail("the promoted reduction chunk is not a map_reduce_post");
    auto value = tile(fused.value, rest, depth + 1);
    scopes_.pop_back();
    if (!value) return nullptr;

    auto inputs = leaves(value, var);
    auto ins    = w.tuple(inputs);
    auto step   = w.mut_lam(dom, w.pi(ins->type(), acc_type));
    auto inner  = w.mut_lam(ins->type(), acc_type);
    inner->set(true, close(value, var, step->var(), inputs, inner->var()));
    step->set(true, inner);

    auto last       = w.mut_lam(acc_type, acc_type);
    auto tail_fused = size_t(0);
    if (peel) {
        auto [at, rem] = build(w.annex<tf_tail>(), {w.tuple({K, ks})})->projs<2>();
        auto partial   = reduction_tile<tile_of_partial>(mr, anchor, {off, ext, w.lit_nat(loop), at, rem, last->var()});
        if (!partial) return fail("the root has no reduction tail");
        auto block  = fuse(partial, nullptr, depth, 1);
        block.value = stage_block(block.value);
        if (!block.value) return fail("the promoted reduction tail is not a map_reduce_post");
        tail_fused = block.count;
        auto done  = tile(block.value, rest, depth + 1);
        if (!done) return nullptr;
        last->set(true, done);
    } else {
        last->set(true, last->var());
    }
    auto post     = w.mut_lam(acc_type, anchor->type());
    auto epilogue = reduction_tile<tile_of_post>(mr, anchor, {off, ext, post->var()});
    if (!epilogue) return fail("the root's epilogue has no tile");
    post->set(true, epilogue);
    auto result = build(w.annex<tf_reduce>(), {w.tuple({acc_type, anchor->type(), ins->type()}), w.tuple({K, ks, tail}),
                                               init, ins, step, last, post});
    log_.d("tensor.tile_fuse {} L{} red loop={} ks={} tail={} inputs={} fused={} tail_fused={}", op_, depth, loop, ks,
           peel ? "peel" : "cap", inputs.size(), fused.count, tail_fused);
    return result;
}

const Def* Levels::vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const Def*> rest, u64 depth) {
    auto& w = world();
    if (!rest.empty()) return fail("a vector level must be the innermost");
    auto ro   = Lit::as(btensor::MapRedPost::of(root->callee()->as<App>()).Ro);
    auto vdim = rn, factor = u64(0);
    for (u64 d = 0; d != rn; ++d) {
        if (!sizes[d]) continue;
        if (d >= ro) {
            if (d != rn - 1 || factor) return fail("a vector level splits only the last reduction loop");
            factor = sizes[d];
            continue;
        }
        if (vdim != rn || sizes[d] != 1) return fail("a vector level marks a single parallel loop with size 1");
        vdim = d;
    }
    if (vdim == rn) return root;
    if (factor > 1)
        if (auto split = split_reduction(root, factor, vdim)) {
            log_.d("tensor.tile_fuse {} L{} vec vdim={} split={}", op_, depth, vdim, factor);
            return split;
        }
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
    if (auto root = graph::map_reduce(op))
        if (auto ranks = btensor::MapRedPost::of(root->callee()->as<App>()).ranks();
            ranks && Lit::isa(level_list.front()->proj(4, 0)) != 0) {
            auto& w = op->world();
            level_list.insert(level_list.begin(), w.tuple({w.lit_nat_0(), w.pack(ranks->nloops(), w.lit_nat_0()),
                                                           w.lit_nat_1(), w.lit_nat_0()}));
        }
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
