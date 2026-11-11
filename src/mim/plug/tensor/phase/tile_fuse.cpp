#include "mim/plug/tensor/phase/tile_fuse.h"

#include <algorithm>
#include <limits>

#include <fe/restore.h>

#include <mim/nom.h>
#include <mim/union.h>

#include <mim/pattern/graph.h>

#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>

#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/phase/panel_layout.h"
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
            // A closed splat stays inline, so its consumers seed from the scalar instead of a materialized input.
            if (TensorType::isa(def->type()) && !(def->is_closed() && Axm::isa<tensor::splat>(def))) {
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

/// One level of a `tile_req`: `kind` 0 parallel, 1 reduction, 2 vector, 3 tensor-core block; `bind` 0 serial,
/// 1 work-group, 2 work-item, 3 warp.
/// `pipeline` is a reduction's depth inside a work-group, 0 until TileFuse::realize resolves the default.
struct LevelSpec {
    u64 kind, tail, promote, bind;
    const Def* sizes;
    const Def* merge;
    u64 pipeline = 1;
};

/// The levels of a `tile_req` chain, outermost first; `nullopt` unless the chain is a literal value.
std::optional<std::vector<LevelSpec>> tile_levels(const Def* levels) {
    enum { Serial, Group, Item, Warp } scope = Serial;
    std::vector<LevelSpec> specs;
    for (auto cur = levels;;) {
        // The chain's noms are newtypes over variants: a `Name` wraps the case's `Inj`.
        if (auto name = cur->isa<Name>()) cur = name->op();
        auto inj = cur->isa<Inj>();
        if (!inj) return std::nullopt;
        auto i = inj->index();
        if (i == 0) return specs;
        // `WarpLevels#Mma` holds just its shape and ends the chain.
        if (scope == Warp && i == 2) {
            specs.push_back(LevelSpec{3, 0, 0, 0, inj->value(), nullptr});
            return specs;
        }
        auto n     = inj->value()->num_projs();
        auto field = [&](size_t j) { return inj->value()->proj(n, j); };
        auto lit   = [&](size_t j) { return Lit::isa(field(j)); };
        LevelSpec spec{0, 0, 0, 0, field(0), nullptr};
        // (kind, bind) per constructor of TileLevels / GroupLevels / ItemLevels / WarpLevels; -1 marks a case without
        // one.
        static constexpr int table[4][7][2] = {
            {{0, 0}, {0, 0},   {0, 1},   {1, 0},   {2, 0}, {-1, -1}, {-1, -1}},
            {{0, 0}, {0, 0},   {0, 2},   {1, 0},   {1, 2},   {2, 0},   {0, 3}},
            {{0, 0}, {0, 0},   {1, 0},   {2, 0}, {-1, -1}, {-1, -1}, {-1, -1}},
            {{0, 0}, {1, 0}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}}
        };
        if (i > 6 || table[scope][i][0] < 0) return std::nullopt;
        spec.kind = table[scope][i][0], spec.bind = table[scope][i][1];
        if (spec.kind != 2) {
            auto tail = lit(1);
            if (!tail) return std::nullopt;
            spec.tail = *tail;
        }
        if ((spec.kind == 1 && spec.bind == 0) || (spec.kind == 0 && spec.bind == 1)) {
            auto promote = lit(2);
            if (!promote) return std::nullopt;
            spec.promote = *promote;
        }
        if (spec.kind == 1 && spec.bind == 0 && scope != Serial) {
            auto pipeline = lit(3);
            if (!pipeline) return std::nullopt;
            spec.pipeline = *pipeline;
        }
        if (spec.kind == 1 && spec.bind == 2) {
            auto option = field(2);
            if (auto name = option->isa<Name>()) option = name->op();
            auto merge = option->isa<Inj>();
            if (!merge) return std::nullopt;
            if (merge->index() == 1) spec.merge = merge->value();
        }
        specs.push_back(spec);
        if (spec.bind == 1) scope = Group;
        if (spec.kind == 0 && spec.bind == 2) scope = Item;
        if (spec.kind == 0 && spec.bind == 3) scope = Warp;
        cur = field(n - 1);
    }
}

/// The value @p lam returns, also when it hands its mapped element to a local continuation that returns the result.
const Def* returned(Lam* lam) {
    if (auto value = Lam::isa_ret_arg(lam)) return value;
    auto call = lam->body()->isa<App>();
    auto next = call ? call->callee()->isa_mut<Lam>() : nullptr;
    if (!next || !next->is_set()) return nullptr;
    auto app = next->reduce(call->arg()).back()->isa<App>();
    return app && app->callee() == lam->ret_var() ? app->arg() : nullptr;
}

const Def* combiner_app(const Def* comb) {
    auto lam = comb->isa_mut<Lam>();
    if (!lam || !lam->is_set()) return nullptr;
    auto value = returned(lam);
    auto app   = value ? value->isa<App>() : nullptr;
    if (!app || app->arg()->num_projs() != 2) return nullptr;
    auto acc   = lam->var(2, 0)->proj(2, 0);
    auto other = app->arg(2, 0) == acc ? app->arg(2, 1) : app->arg(2, 1) == acc ? app->arg(2, 0) : nullptr;
    if (!other) return nullptr;
    DefSet seen;
    for (DefVec pending{other}; !pending.empty();) {
        auto def = pending.back();
        pending.pop_back();
        if (def == acc) return nullptr;
        if (!seen.insert(def).second || def->isa<Var>() || def->is_closed()) continue;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return app;
}

struct ReductionLaw {
    const Def* identity;
};

std::optional<ReductionLaw> reduction_law(const btensor::MapRedPost& mr) {
    auto op = combiner_app(mr.comb);
    if (!op) return {};
    auto& w   = mr.comb->world();
    auto idx  = Idx::isa(mr.To);
    auto size = idx ? Lit::isa(idx) : std::nullopt;
    if (auto wrap = Axm::isa<core::wrap>(op)) {
        if (wrap.id() == core::wrap::add) return ReductionLaw{w.lit(mr.To, 0)};
        if (wrap.id() == core::wrap::mul && size) return ReductionLaw{w.lit_idx_mod(*size, 1)};
    }
    if (auto bit = Axm::isa<core::bit2>(op); bit && size) {
        if (bit.id() == core::bit2::and_) return ReductionLaw{w.lit(mr.To, *size ? *size - 1 : u64(-1))};
        if (bit.id() == core::bit2::or_ || bit.id() == core::bit2::xor_) return ReductionLaw{w.lit(mr.To, 0)};
    }
    if (auto extrema = Axm::isa<core::extrema>(op); extrema && size) {
        switch (extrema.id()) {
            case core::extrema::sm: return ReductionLaw{w.lit(mr.To, *size ? *size - 1 : u64(-1))};
            case core::extrema::sM: return ReductionLaw{w.lit(mr.To, 0)};
            case core::extrema::Sm: return ReductionLaw{w.lit(mr.To, (*size - 1) / 2)};
            case core::extrema::SM: return ReductionLaw{w.lit(mr.To, *size ? *size - *size / 2 : u64(1) << 63)};
        }
    }
    auto float_identity = [&](bool minimum) -> std::optional<ReductionLaw> {
        auto width = math::isa_f(mr.To);
        if (!width) return {};
        auto value = minimum ? std::numeric_limits<f64>::infinity() : -std::numeric_limits<f64>::infinity();
        return ReductionLaw{math::lit_f(w, *width, value)};
    };
    auto reassociable = [](const App* app) {
        auto mode = Lit::isa(app->uncurry_args<3>()[0]);
        return mode && fe::has_flag(static_cast<math::Mode>(*mode), math::Mode::reassoc);
    };
    if (auto arith = Axm::isa<math::arith>(op); arith && reassociable(arith)) {
        if (arith.id() == math::arith::add) return ReductionLaw{math::lit_f(w, *math::isa_f(mr.To), 0.0)};
        if (arith.id() == math::arith::mul) return ReductionLaw{math::lit_f(w, *math::isa_f(mr.To), 1.0)};
    }
    if (auto extrema = Axm::isa<math::extrema>(op); extrema && reassociable(extrema)) {
        auto mode = static_cast<math::Mode>(*Lit::isa(extrema->uncurry_args<3>()[0]));
        if (!fe::has_flag(mode, math::Mode::nnan) || !fe::has_flag(mode, math::Mode::nsz)) return {};
        return float_identity(extrema.id() == math::extrema::im || extrema.id() == math::extrema::Im);
    }
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

const Def* slice_of(const Def* tensor, const Def* off, const Def* ext) {
    auto& w   = tensor->world();
    auto type = *TensorType::isa(tensor->type());
    auto r    = Lit::as(type.r);
    auto view = w.app(w.app(w.annex<slice>(), {type.T, type.r}), type.s);
    return w.app(w.app(view, {off, w.pack(r, w.lit_nat_1()), ext}), tensor);
}

/// The whole of @p tensor staged as one panel with its own layout.
const Def* stage_whole(const Def* tensor) {
    auto& w     = tensor->world();
    auto type   = *TensorType::isa(tensor->type());
    auto r      = Lit::as(type.r);
    auto perm   = w.tuple(DefVec(r, [&](size_t d) { return w.lit_idx(r, d); }));
    auto layout = w.tuple({w.tuple(), w.tuple(), perm, w.bot(type.T)});
    auto staged = w.app(w.app(w.annex<stage>(), {type.T, type.r, w.lit_nat_0()}), type.s);
    return w.app(w.app(w.app(staged, layout), w.lit_nat_0()), tensor);
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

/// The axis among the last two of an input read by @p map that one of the loops in @p tiles projects onto, with that
/// loop's tile; leading axes are batch axes.
struct Panel {
    u64 axis, loop, tile;
};

std::optional<Panel> panel_axis(const Def* map, u64 rank, fe::Span<const u64> tiles) {
    auto& w   = map->world();
    auto pi   = map->type()->as<Pi>();
    auto rn   = Lit::isa(pi->dom()->arity());
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(map, var);
    if (!rn || rank < 2) return {};
    std::optional<Panel> result;
    // Offsets of enclosing tiles are constants; only the loops decide the axis.
    for (u64 d = rank - 2; d != rank; ++d) {
        auto form = Linear::of(read->proj(rank, d));
        for (u64 l = 0; l != tiles.size(); ++l) {
            bool own = true;
            for (u64 k = 0; k != *rn; ++k) {
                auto atom = form.atoms.find(var->proj(*rn, k));
                own &= (atom == form.atoms.end() ? 0 : atom->second) == (k == l ? 1 : 0);
            }
            if (!tiles[l] || !own) continue;
            if (result) return {};
            result = Panel{d, l, tiles[l]};
        }
    }
    return result;
}

/// @p app with its input @p slot staged into panels of @p tile along @p axis, one of its last two axes.
/// The read map follows the layout: `x ↦ (batch axes, x#axis / tile, x#other, x#axis % tile)`.
/// Stages input @p slot per block into panels of @p tile along @p axis.
/// Panels of a work-group's shared memory keep @p axis's panel index innermost among the outer axes, so consecutive
/// work-items read consecutive panels without bank conflicts.
/// A bound parallel level's region var, literal tile sizes per output loop, and the extents it tiles.
struct LevelScope {
    const Var* var;
    fe::Vector<u64> tiles;
    const Def* extents;
};

/// What a reduction level knows of a work-group panel's fill when it stages it.
struct PanelSite {
    /// Upper bounds of the region's extents, and whether they are exact once the regions are specialized.
    DefMap<std::pair<u64, bool>> caps;
    /// Nat terms known to be multiples of the value, 0 for any.
    DefMap<u64> multiples;
    /// The work-group's offsets, by loop.
    DefMap<size_t> origins;
    /// The tile grid of the item level that reads the panel.
    std::optional<std::pair<std::vector<u64>, std::vector<u64>>> items;
    bool mma = false;
};

bool site_multiple(const PanelSite& site, const Def* def, u64 v) {
    if (v == 1) return true;
    if (auto lit = Lit::isa(def)) return *lit % v == 0;
    if (auto nat = Axm::isa<core::nat>(def)) {
        auto [a, b] = nat->args<2>();
        if (nat.id() == core::nat::add) return site_multiple(site, a, v) && site_multiple(site, b, v);
        if (nat.id() == core::nat::mul) return site_multiple(site, a, v) || site_multiple(site, b, v);
        return false;
    }
    if (auto multiple = fe::lookup(site.multiples, def)) return *multiple % v == 0;
    return false;
}

std::optional<size_t> site_origin(const PanelSite& site, const Def* def) {
    if (auto d = fe::lookup(site.origins, def)) return *d;
    if (auto nat = Axm::isa<core::nat>(def); nat && (nat.id() == core::nat::add || nat.id() == core::nat::mul)) {
        auto [a, b] = nat->args<2>();
        if (auto d = site_origin(site, a)) return d;
        return site_origin(site, b);
    }
    return {};
}

/// The capacity and the storage layout the panel cost model picks for the work-group panel @p staged, or `nullptr`
/// for its plain one.
/// It sees the fill as `lower_to_mem` will: the panel dim along the source's contiguous one runs fastest and moves a
/// 16-byte vector where the run is provably aligned.
const Def* panel_term(const App* staged, const PanelSite& site) {
    auto& w                           = staged->world();
    auto [meta, shape, layout, scope] = staged->callee()->as<App>()->uncurry_args<4>();
    auto [type, rank, split]          = meta->projs<3>();
    auto [pos, tiles, perm, pad]      = layout->projs<4>();
    auto r = Lit::isa(rank), k = Lit::isa(split);
    auto bytes = Lit::isa(core::op(core::trait::size, type));
    if (!r || !k || !bytes) return nullptr;
    // Specializing the regions turns their extents into these caps; only exact ones become literals.
    Substitute capped(w), exact(w);
    for (auto [extent, bound] : site.caps) {
        capped.substitute(extent, w.lit_nat(bound.first));
        if (bound.second) exact.substitute(extent, w.lit_nat(bound.first));
    }
    auto cap = [&](const Def* extent) -> std::optional<std::pair<u64, bool>> {
        auto literal = bool(Lit::isa(exact.rewrite(extent)));
        auto bounded = capped.rewrite(extent);
        if (auto lit = Lit::isa(bounded)) return std::pair{*lit, literal};
        if (auto upper = nat_upper(bounded)) return std::pair{*upper, literal};
        return {};
    };
    auto outer = std::vector<std::pair<u64, bool>>(*r);
    for (size_t d = 0; d != *r; ++d) {
        auto c = cap(shape->proj(*r, d));
        if (!c) return nullptr;
        outer[d] = *c;
        for (size_t j = 0; j != *k; ++j)
            if (Lit::isa(pos->proj(*k, j)) == d) {
                auto tile      = Lit::as(tiles->proj(*k, j));
                outer[d].first = (outer[d].first + tile - 1) / tile;
            }
    }
    auto dims = std::vector<std::pair<u64, bool>>();
    for (size_t p = 0; p != *r; ++p)
        dims.push_back(outer[Lit::as(perm->proj(*r, p))]);
    for (size_t j = 0; j != *k; ++j)
        dims.emplace_back(Lit::as(tiles->proj(*k, j)), true);

    auto inner = std::optional<size_t>();
    auto along = [&](size_t c) {
        for (size_t d = 0; d != *r; ++d)
            if (Lit::isa(perm->proj(*r, d)) == c) inner = d;
        for (size_t j = 0; j != *k; ++j)
            if (Lit::isa(pos->proj(*k, j)) == c) inner = *r + j;
    };
    u64 vec    = 1;
    auto slice = Axm::isa<tensor::slice>(Axm::peel<tensor::buf>(staged->arg()));
    auto reads = std::optional<panel::Reads>();
    if (slice) {
        auto [slice_meta, source_shape, box] = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, step, extent]           = box->projs<3>();
        auto source = slice->arg(), physical_shape = source_shape, physical_rank = rank;
        const Def* read = w.app(w.annex<affine::id>(), rank);
        while (auto view = panel::tensor_read(source, physical_shape)) {
            if (!TensorType::isa(view->src->type())) break;
            read           = panel::compose_read_map(view->map, read);
            source         = view->src;
            physical_shape = view->S;
            physical_rank  = view->R;
        }
        auto contiguous = panel::contiguous_dim(read, rank, physical_rank);
        if (contiguous) along(contiguous->first);
        if (inner && 16 % *bytes == 0 && 16 / *bytes >= 2) {
            auto c = contiguous->first, ri = Lit::as(physical_rank);
            auto v   = 16 / *bytes;
            auto e   = cap(extent->proj(*r, c));
            auto row = Lit::isa(physical_shape->proj(ri, ri - 1));
            if (e && e->second && e->first % v == 0 && Lit::isa(step->proj(*r, c)) == 1 && row && *row % v == 0
                && contiguous->second % v == 0 && site_multiple(site, start->proj(*r, c), v) && dims[*inner].second
                && dims[*inner].first % v == 0)
                vec = v;
        }
        if (site.items) {
            auto& [counts, steps] = *site.items;
            reads                 = panel::Reads{counts, steps};
            reads->loop_dim.resize(*r + *k);
            reads->whole.resize(*r + *k);
            for (size_t p = 0; reads && p != *r; ++p) {
                auto d = Lit::as(perm->proj(*r, p));
                auto j = site_origin(site, start->proj(*r, d));
                if (!j || reads->counts[*j] == 1) continue;
                u64 tile = 1;
                for (size_t i = 0; i != *k; ++i)
                    if (Lit::isa(pos->proj(*k, i)) == d) {
                        tile                 = Lit::isa(tiles->proj(*k, i)).value_or(0);
                        reads->whole[*r + i] = true;
                    }
                if (tile != reads->steps[*j])
                    reads.reset();
                else
                    reads->loop_dim[p] = *j;
            }
        }
    } else if (*r) {
        along(*r - 1);
    }
    if (!inner) return nullptr;
    auto storage = w.tuple(DefVec(dims.size(), [&](size_t d) { return w.lit_nat(dims[d].first); }));
    auto choice  = site.mma ? panel::mma(storage, *inner)
                            : panel::choose(storage, *inner, vec, *bytes, reads ? &*reads : nullptr);
    if (choice.layout.identity() && choice.physical == storage) return nullptr;
    return w.tuple({storage, choice.expr});
}

/// The input dim that reads loop @p loop with unit slope, if any.
std::optional<u64> dim_of_loop(const Def* map, u64 rank, u64 loop) {
    auto& w   = map->world();
    auto pi   = map->type()->as<Pi>();
    auto rn   = Lit::as(pi->dom()->arity());
    auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
    auto read = w.app(map, var);
    for (u64 d = 0; d != rank; ++d) {
        auto form = Linear::of(read->proj(rank, d));
        bool own  = true;
        for (u64 k = 0; k != rn; ++k) {
            auto atom = form.atoms.find(var->proj(rn, k));
            own &= (atom == form.atoms.end() ? 0 : atom->second) == (k == loop ? 1 : 0);
        }
        if (own) return d;
    }
    return {};
}

/// A reduction level's block along output loop `loop`: `[k0, k0 + k)`.
struct Block {
    u64 loop;
    const Def *k0, *k;
};

/// The input's slice over the whole enclosing work-group's tile along @p axis (output loop @p loop) and the
/// reduction block, with the work-item's offset inside it; nothing if the input is no plain slice.
std::optional<std::pair<const Def*, const Def*>> group_slice(const Def* input,
                                                             const Def* map,
                                                             u64 axis,
                                                             u64 loop,
                                                             u64 rank,
                                                             const LevelScope& group,
                                                             const LevelScope& item,
                                                             const Block& block) {
    auto slice = Axm::isa<tensor::slice>(input);
    if (!slice) return {};
    auto kdim = dim_of_loop(map, rank, block.loop);
    if (!kdim) return {};
    auto& w                 = input->world();
    auto [meta, shape, box] = slice->callee()->as<App>()->uncurry_args<3>();
    auto ro                 = group.tiles.size();
    auto off = group.var->proj(2, 0)->proj(ro, loop), ext = group.var->proj(2, 1)->proj(ro, loop);
    auto start = w.tuple(DefVec(rank, [&](size_t d) -> const Def* {
        return d == axis ? off : d == *kdim ? block.k0 : w.lit_nat_0();
    }));
    auto size  = w.tuple(DefVec(rank, [&](size_t d) -> const Def* {
        return d == axis ? ext : d == *kdim ? block.k : shape->proj(rank, d);
    }));
    auto wide  = w.app(w.app(w.app(w.annex<tensor::slice>(), meta), shape), {start, w.pack(rank, w.lit_nat_1()), size});
    return std::pair{w.app(wide, slice->arg()), item.var->proj(2, 0)->proj(ro, loop)};
}

const App* promote_input(const App* app,
                         u64 slot,
                         u64 axis,
                         u64 loop,
                         u64 tile,
                         bool shared,
                         const LevelScope* group,
                         const LevelScope* item,
                         const Block* block,
                         const PanelSite* site) {
    auto& w          = app->world();
    auto mr          = btensor::MapRedPost::of(app->callee()->as<App>());
    auto nis         = Lit::as(mr.nis);
    auto is          = app->arg(2, 0);
    auto input       = is->proj(nis, slot);
    auto rank        = Lit::as(TensorType::isa(input->type())->r);
    const Def* inner = nullptr;
    if (group)
        if (auto widened = group_slice(input, mr.maps->proj(nis, slot), axis, loop, rank, *group, *item, *block))
            std::tie(input, inner) = *widened;
    auto type   = *TensorType::isa(input->type());
    auto other  = 2 * rank - 3 - axis;
    auto source = [&](size_t i) {
        if (i + 2 < rank) return i;
        auto last = i + 2 == rank;
        return (last != shared) ? axis : other;
    };
    auto perm      = w.tuple(DefVec(rank, [&](size_t i) { return w.lit_idx(rank, source(i)); }));
    auto stage_app = build(w.annex<stage>(),
                           {w.tuple({type.T, type.r, w.lit_nat_1()}), type.s,
                            w.tuple({w.lit_nat(axis), w.lit_nat(tile), perm, w.bot(type.T)}), w.lit_nat_0(), input});
    if (site)
        if (auto chosen = panel_term(stage_app->as<App>(), *site))
            stage_app = w.app(w.app(w.app(w.app(w.annex<placed>(), {type.T, w.lit_nat(rank + 1)}),
                                          TensorType::isa(stage_app->type())->s),
                                    chosen),
                              stage_app);
    const Def* builder = w.app(w.annex<stage_input>(), w.lit_nat(slot));
    for (auto arg : app->uncurry_args<9>())
        builder = w.app(builder, arg);
    builder = w.app(builder, {w.lit_idx(rank, axis), w.lit_nat(tile), perm, w.lit_bool(bool(inner)),
                              inner ? inner : w.lit_nat_0()});
    builder = w.app(builder, TensorType::isa(stage_app->type())->s);
    return w.app(builder, stage_app)->as<App>();
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
    const Def* tile(const Def* anchor, fe::Span<const LevelSpec> levels, u64 depth);
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
    const Def* split_over_items(const Def* tile, u64 items, const Def* merge, u64 depth);
    const Def* parallel(const Def* anchor,
                        const btensor::MapRedPost& mr,
                        u64 ro,
                        u64 rn,
                        fe::Span<const u64> sizes,
                        bool peel,
                        u64 promote,
                        u64 bind,
                        fe::Span<const LevelSpec> rest,
                        u64 depth);
    const Def* reduction(const Def* anchor,
                         const btensor::MapRedPost& mr,
                         u64 ro,
                         u64 rn,
                         fe::Span<const u64> sizes,
                         bool peel,
                         u64 promote,
                         u64 bind,
                         const Def* merge,
                         u64 pipeline,
                         fe::Span<const LevelSpec> rest,
                         u64 depth);
    const Def* vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const LevelSpec> rest, u64 depth);
    const Def* mma(const App* root, u64 rn, fe::Span<const u64> shape, u64 depth);

    const fe::Log& log_;
    const App* request_;
    const Def *op_, *control_;
    const DefSet& members_;
    DefMap<std::vector<u64>> aligned_;
    const DefSet& outside_;
    bool strict_;
    fe::Vector<const Var*> scopes_;
    std::optional<LevelScope> group_scope_, item_scope_;
    /// The work-item count of the item level that follows the work-group level being fused.
    std::optional<u64> next_items_;
    /// Reductions split over the work-items, by producer; later reads view them instead of recomputing them.
    DefMap<std::pair<const Def*, Origin>> resident_;
    DefMap<Origin> origins_;
    /// Tiles that enclosing levels hand back, by their original producer, and the inner results that replace them.
    DefMap<const Def*> escaping_;
    DefMap<const Def*> handed_;
    std::string reason_;
    std::vector<std::pair<const Def*, const Def*>> yielded_;
    u64 bound_ = 0;
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

// The reduction tile @p tile as a `tile_fold` whose blocks the @p items work-items fold (R8), if its single
// reduction loop divides into them.
const Def* Levels::split_over_items(const Def* tile, u64 items, const Def* merge, u64 depth) {
    auto& w   = world();
    auto root = graph::map_reduce(tile);
    if (!root) return nullptr;
    auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
    auto rks = mr.ranks();
    if (!rks || rks->rr != 1) return nullptr;
    auto rn = rks->nloops();
    auto K  = Lit::isa(mr.Sr->proj(rn, rn - 1));
    if (!K || *K % items) return nullptr;
    auto sizes = DefVec(rn, [&](size_t d) { return w.lit_nat(d + 1 == rn ? *K / items : 0); });
    auto level = LevelSpec{1, 1, 0, 2, w.tuple(sizes), merge};
    auto split = this->tile(tile, fe::Span<const LevelSpec>(&level, 1), depth + 1);
    if (!split) log_.d("tensor.tile_fuse {} keeps a fused reduction whole: {}", op_, reason_);
    return split;
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
        const Def* merge = nullptr;
        if (auto marked = Axm::isa<tensor::merge_with>(p)) {
            merge = marked->callee()->as<App>()->arg();
            p     = marked->arg();
        }
        auto producer = graph::map_reduce(p);
        auto reduces  = producer
                     && btensor::MapRedPost::of(producer->callee()->as<App>())
                                .ranks()
                                .value_or(btensor::MapRedPost::Ranks{0, 0, 0, 0})
                                .rr
                            != 0;
        auto origin   = compose(p, next->off, next->ext, next->cap);
        if (auto resident = resident_.find(origin.producer); resident != resident_.end()) {
            auto [shared, from] = resident->second;
            // The shared tile's own fill reads the producer.
            if (shared == p || depends_on(shared, next->app)) continue;
            auto whole = Check::test_alpha(origin.off, from.off) && Check::test_alpha(next->ext, from.ext);
            log_.d("tensor.tile_fuse {} read {} from the work-items' shared tile", op_, p);
            auto r     = Lit::as(TensorType::isa(p->type())->r);
            auto start = w.tuple(DefVec(r, [&](size_t d) {
                return w.call(core::nat::sub, Defs{origin.off->proj(r, d), from.off->proj(r, d)});
            }));
            auto view  = whole ? shared : slice_box_of(shared, start, next->ext, next->cap ? next->cap : next->ext);
            Substitute reuse(w);
            for (auto free : result.value->free_vars())
                reuse.map(free, free);
            reuse.substitute(next->app, view);
            result.value = reuse.rewrite(result.value);
            for (auto& made : tiles)
                made.first = reuse.rewrite(made.first);
            continue;
        }
        auto escaping    = escaping_.find(origin.producer);
        auto outside_use = (outermost(own) && outside_.contains(next->producer)) || escaping != escaping_.end();
        auto query
            = w.tuple({w.lit_nat(depth), w.lit_nat(kind), w.lit_nat_0(), w.lit_bool(reduces), w.lit_bool(outside_use),
                       w.lit_bool(in_group(next->producer)), w.lit_nat(std::numeric_limits<nat_t>::max())});
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
        if (reduces && next_items_)
            if (auto split = split_over_items(tile, *next_items_, merge, depth))
                resident_.emplace(origin.producer, std::pair{tile = split, origin});
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

const Def* Levels::tile(const Def* anchor, fe::Span<const LevelSpec> levels, u64 depth) {
    if (levels.empty()) return anchor;
    auto root = graph::map_reduce(anchor);
    if (!root) return fail("the root tile is not a map_reduce_post");
    auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
    auto rks = mr.ranks();
    if (!rks) return fail("the root's ranks are not literal");
    auto ro = rks->ro, rn = rks->nloops();
    auto& level = levels.front();
    if (level.tail > 1) return fail("a tail is 0 (cap) or 1 (peel)");
    if (Lit::isa(level.sizes->arity()) != rn)
        return fail("the levels tile a different number of loops than the root has");
    auto literal = fe::Vector<u64>(rn);
    for (u64 d = 0; d != rn; ++d)
        if (auto size = Lit::isa(level.sizes->proj(rn, d)))
            literal[d] = *size;
        else
            return fail("tile sizes must be literals");
    auto rest = levels.subspan(1);
    switch (level.kind) {
        case 0: return parallel(anchor, mr, ro, rn, literal, level.tail == 1, level.promote, level.bind, rest, depth);
        case 1:
            return reduction(anchor, mr, ro, rn, literal, level.tail == 1, level.promote, level.bind, level.merge,
                             level.pipeline, rest, depth);
        case 2: return vector(root, rn, literal, rest, depth);
        default: return mma(root, rn, literal, depth);
    }
}

const Def* Levels::parallel(const Def* anchor,
                            const btensor::MapRedPost& mr,
                            u64 ro,
                            u64 rn,
                            fe::Span<const u64> sizes,
                            bool peel,
                            u64 promote,
                            u64 bind,
                            fe::Span<const LevelSpec> rest,
                            u64 depth) {
    auto& w         = world();
    auto scope_bind = fe::Restore(bound_, bind ? bind : bound_);
    if (!canonical_write(mr, ro, rn)) return fail("the root does not write loop d to output axis d");
    auto tiles = DefVec(ro);
    bool whole = true;
    for (u64 d = 0; d != rn; ++d) {
        if (d >= ro) {
            if (sizes[d]) return fail("a parallel level cannot tile a reduction loop");
            continue;
        }
        auto extent = mr.So->proj(ro, d);
        // An untiled runtime extent takes its constant storage bound, so the inner write preimage is exact.
        auto bound = storage(anchor)->proj(ro, d);
        tiles[d]   = sizes[d] ? w.lit_nat(sizes[d]) : Lit::isa(extent) || !Lit::isa(bound) ? extent : bound;
        whole &= !sizes[d] || tiles[d] == extent;
    }
    // The outermost level stays a region even over the whole shape, so tile bodies own every intermediate buffer.
    // A scalar root has no grid to tile.
    if (whole && !bind && !promote && (depth || !ro)) return tile(anchor, rest, depth + 1);

    auto box   = w.sigma({w.arr(mr.Ro, w.type_nat()), w.arr(mr.Ro, w.type_nat())});
    auto probe = w.mut_lam(box, w.type_nat());
    auto var   = probe->var()->as<Var>();
    auto off = probe->var(2, 0), ext = probe->var(2, 1);
    // A runtime extent whose storage fits one tile has that tile at its origin.
    auto bounds    = storage(anchor);
    auto origin    = DefVec(ro, [&](size_t d) -> const Def* {
        auto tile = Lit::isa(tiles[d]), bound = Lit::isa(bounds->proj(ro, d));
        bool one = !Lit::isa(mr.So->proj(ro, d)) && tile && bound && *tile >= *bound;
        return one ? w.lit_nat_0() : off->proj(ro, d);
    });
    off            = w.tuple(origin);
    auto root_tile = tile_of(anchor, off, ext, w.tuple(tiles));
    if (Axm::isa<tile_of_box>(root_tile)) {
        log_.d("tensor.tile_fuse {} L{} skip: the root has no tile here: {}", op_, depth,
               producer_tile(anchor, off, ext, w.tuple(tiles)).reason);
        return tile(anchor, rest, depth + 1);
    }
    scopes_.push_back(var);
    auto literal = fe::Vector<u64>(ro, [&](size_t d) { return sizes[d]; });
    auto group_scope = fe::Restore(group_scope_, bind == 1 ? std::optional{LevelScope{var, literal, mr.So}} : group_scope_);
    auto item_scope  = fe::Restore(item_scope_, bind >= 2 ? std::optional{LevelScope{var, literal, mr.So}} : item_scope_);
    // A reduction fused into a work-group's tile is split over the work-items of the item level that follows.
    auto items = std::optional<u64>();
    if (bind == 1)
        for (auto& level : rest) {
            if (level.kind != 0 || level.bind != 2) continue;
            u64 count = 1;
            for (u64 d = 0; d != ro; ++d) {
                auto item = Lit::isa(level.sizes->proj(rn, d)).value_or(0), tile = Lit::isa(tiles[d]).value_or(0);
                if (item && tile) count *= (tile + item - 1) / item;
            }
            items = count;
            break;
        }
    auto next_items = fe::Restore(next_items_, items);
    // A promoted input's box is staged into the work-group's shared memory once; every fused read views it.
    if (promote) {
        auto nis = Lit::as(mr.nis);
        if (promote >> nis) return fail("a promoted input does not exist");
        auto is = graph::map_reduce(anchor)->arg(2, 0);
        Substitute staging(w);
        for (auto free : root_tile->free_vars())
            staging.map(free, free);
        for (auto read : reads(root_tile))
            for (u64 slot = 0; slot != nis; ++slot) {
                if (!(promote >> slot & 1) || read.producer != is->proj(nis, slot)) continue;
                auto staged = stage_whole(slice_of(read.producer, read.off, read.ext));
                staging.substitute(read.app, staged);
                resident_.emplace(read.producer, std::pair{
                                                     staged, Origin{read.producer, read.off, read.ext, read.cap}
                });
                log_.d("tensor.tile_fuse {} L{} stage input {} box {} × {} in the work-group", op_, depth, slot,
                       read.off, read.ext);
            }
        root_tile = staging.rewrite(root_tile);
    }
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
    const Def* region = body;
    if (bind) region = w.app(w.app(w.app(w.annex<tensor::bind>(), body->type()), w.lit_nat(bind)), body);

    log_.d("tensor.tile_fuse {} L{} par tiles={} tail={} inputs={} fused={} yielded={} bind={}", op_, depth,
           w.tuple(tiles), peel ? "peel" : "cap", inputs.size(), fused.count, yields.size(), bind);
    auto grid = w.tuple({mr.So, w.tuple(tiles), w.lit_nat(peel)});
    if (yields.empty()) return build(w.annex<tf_par>(), {w.tuple({mr.Tp, mr.Ro}), ins->type(), grid, ins, region});
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
    auto multi  = build(w.annex<tf_par_n>(), {meta, ins->type(), layout, w.tuple(maps), ins, region});
    for (size_t i = 0; i != yields.size(); ++i)
        if (scopes_.empty() && members_.contains(yields[i].producer))
            yielded_.emplace_back(yields[i].producer, multi->proj(n, i));
        else
            handed_[yields[i].producer] = multi->proj(n, i);
    return multi->proj(n, n - 1);
}

const Def* Levels::reduction(const Def* anchor,
                             const btensor::MapRedPost& mr,
                             u64 ro,
                             u64 rn,
                             fe::Span<const u64> sizes,
                             bool peel,
                             u64 promote,
                             u64 bind,
                             const Def* merge,
                             u64 pipeline,
                             fe::Span<const LevelSpec> rest,
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
    // Each promoted input (bit i of promote) is staged per block into panels as wide as the next parallel level's tile
    // of its axis.
    struct Promoted {
        u64 slot, axis, loop, tile;
    };
    std::vector<Promoted> panels;
    // Inside a work-item level the panels cover the whole work-group, packed by the work-item's tile.
    // A warp's tensor cores read unpacked panels.
    auto group_wide = bound_ >= 2 && group_scope_ && item_scope_;
    if (promote) {
        auto tiles = fe::Vector<u64>(ro, 0);
        if (group_wide) tiles = bound_ == 3 ? group_scope_->tiles : item_scope_->tiles;
        for (auto& level : rest) {
            if (group_wide) break;
            if (level.kind == 0) {
                for (u64 d = 0; d != ro; ++d)
                    tiles[d] = Lit::isa(level.sizes->proj(rn, d)).value_or(0);
                break;
            }
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
            panels.push_back({slot, panel->axis, panel->loop, panel->tile});
            log_.d("tensor.tile_fuse {} L{} promote input {} axis {} panel {}{}", op_, depth, slot, panel->axis,
                   panel->tile, group_wide ? " group-wide" : "");
        }
    }
    // A work-group's panels are filled by all its work-items, so their layout is chosen here.
    auto uniform = bound_ == 1 || group_wide;
    auto site    = std::optional<PanelSite>();
    if (uniform && group_scope_) {
        site.emplace();
        site->mma   = std::ranges::any_of(rest, [](const LevelSpec& level) { return level.kind == 3; });
        auto& group = *group_scope_;
        auto caps   = fe::Vector<std::pair<u64, bool>>(ro);
        for (u64 d = 0; d != ro; ++d) {
            auto full = Lit::isa(group.extents->proj(ro, d));
            auto tile = group.tiles[d];
            auto off = group.var->proj(2, 0)->proj(ro, d), ext = group.var->proj(2, 1)->proj(ro, d);
            caps[d] = tile ? std::pair{full ? std::min(*full, tile) : tile, full && *full % tile == 0}
                           : std::pair{full.value_or(0), bool(full)};
            if (caps[d].first) site->caps[ext] = caps[d];
            site->multiples[off] = tile;
            site->origins[off]   = d;
        }
        if (bound_ == 2 && group_wide && item_scope_) {
            auto counts = std::vector<u64>(ro, 1), steps = std::vector<u64>(ro, 0);
            for (u64 d = 0; d != ro; ++d) {
                auto item = item_scope_->tiles[d];
                if (!item) continue;
                steps[d] = std::min(caps[d].first, item);
                if (caps[d].first != steps[d]) counts[d] = (caps[d].first + steps[d] - 1) / steps[d];
            }
            site->items = std::pair{counts, steps};
        }
    }
    auto stage_block = [&](const Def* value, const Def* k0, const Def* k) -> const Def* {
        auto block = Block{loop, k0, k};
        if (site) {
            auto literal_k = Lit::isa(K);
            // A literal block, such as a peeled tail's, is its own bound, and substituting it would rewrite every equal
            // extent.
            if (!Lit::isa(k)) site->caps[k] = {sizes[loop], literal_k && *literal_k % sizes[loop] == 0};
            site->multiples[k0] = sizes[loop];
        }
        for (auto [slot, axis, panel_loop, tile] : panels) {
            auto app = graph::map_reduce(value);
            if (!app) return nullptr;
            value = promote_input(app, slot, axis, panel_loop, tile, bound_ != 0, group_wide ? &*group_scope_ : nullptr,
                                  group_wide ? &*item_scope_ : nullptr, &block, site ? &*site : nullptr);
        }
        return value;
    };
    // A reduction that fits one block is not split, but its promoted inputs are still staged, once.
    if (auto k = Lit::isa(K); k && *k <= sizes[loop]) {
        if (panels.empty()) return tile(anchor, rest, depth + 1);
        auto block  = fuse(anchor, nullptr, depth, 1);
        block.value = stage_block(block.value, w.lit_nat_0(), K);
        if (!block.value) return fail("the promoted reduction is not a map_reduce_post");
        return tile(block.value, rest, depth + 1);
    }

    auto law = reduction_law(mr);
    // An inner reduction loop's chunks fold its elements out of order, which only a reassociable law permits.
    if (loop != ro && !law && !merge)
        return fail("an ordered reduction level can only tile the outermost reduction loop");
    if (bind == 2 && !merge) {
        if (!law) return fail("a work-item reduction requires a recognized reassociable combiner");
        if (!Check::test_alpha(mr.init, law->identity))
            return fail("a work-item reduction's seed is not the combiner's identity");
    }

    auto off  = w.pack(mr.Ro, w.lit_nat_0());
    auto ext  = mr.So;
    auto init = reduction_tile<tile_of_init>(mr, anchor, {off, ext});
    if (!init) return fail("the root's accumulator has no tile");
    auto acc_type = init->type();
    auto dom      = w.sigma({acc_type, w.type_nat()});
    auto probe    = w.mut_lam(dom, w.type_nat());
    auto var      = probe->var()->as<Var>();
    auto acc = probe->var(2, 0), ko = probe->var(2, 1);
    auto tail    = w.lit_nat(peel);
    auto chunk   = build(w.annex<tf_chunk>(), {w.tuple({K, ks, tail, ko})});
    auto partial = reduction_tile<tile_of_partial>(mr, anchor, {off, ext, w.lit_nat(loop), ko, chunk, acc});
    if (!partial) {
        auto cut = ReductionCut{loop, ko, chunk, acc};
        return fail(
            std::format("the root has no reduction chunk: {}", producer_tile(anchor, off, ext, nullptr, &cut).reason));
    }
    scopes_.push_back(var);
    // Members are fused first, so a promoted input stages its fused tile.
    auto fused  = fuse(partial, var, depth, 1);
    fused.value = stage_block(fused.value, ko, chunk);
    if (!fused.value) return fail("the promoted reduction chunk is not a map_reduce_post");
    auto value = tile(fused.value, rest, depth + 1);
    scopes_.pop_back();
    if (!value) return nullptr;

    auto inputs = leaves(value, var);
    auto ins    = w.tuple(inputs);
    auto step   = w.mut_lam(dom, w.pi(ins->type(), acc_type));
    auto inner  = w.mut_lam(ins->type(), acc_type);
    inner->set(true, close(value, var, step->var(), inputs, inner->var()));
    // A bound reduction's blocks run on the work-items, each folding its own seed; `lower_to_mem` merges them.
    const Def* marked = inner;
    if (pipeline > 1)
        marked = w.app(w.app(w.app(w.annex<tensor::pipeline>(), marked->type()), w.lit_nat(pipeline)), marked);
    if (merge) marked = w.app(w.app(w.app(w.annex<tensor::merge_with>(), {inner->type(), mr.To}), merge), marked);
    if (bind) marked = w.app(w.app(w.app(w.annex<tensor::bind>(), marked->type()), w.lit_nat(bind)), marked);
    step->set(true, marked);

    auto last       = w.mut_lam(acc_type, acc_type);
    auto tail_fused = size_t(0);
    auto literal_k  = Lit::isa(K);
    // A literal K that the blocks divide has no tail to peel.
    if (peel && !(literal_k && *literal_k % sizes[loop] == 0)) {
        auto [at, rem] = build(w.annex<tf_tail>(), {w.tuple({K, ks})})->projs<2>();
        auto partial   = reduction_tile<tile_of_partial>(mr, anchor, {off, ext, w.lit_nat(loop), at, rem, last->var()});
        if (!partial) return fail("the root has no reduction tail");
        auto block  = fuse(partial, nullptr, depth, 1);
        block.value = stage_block(block.value, at, rem);
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
    log_.d("tensor.tile_fuse {} L{} red loop={} ks={} tail={} inputs={} fused={} tail_fused={} bind={}", op_, depth,
           loop, ks, peel ? "peel" : "cap", inputs.size(), fused.count, tail_fused, bind);
    return result;
}

const Def* Levels::mma(const App* root, u64 rn, fe::Span<const u64> shape, u64 depth) {
    auto& w = world();
    if (bound_ != 3) return fail("a tensor-core level must be inside a warp");
    auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
    auto F32 = w.annex<math::F32>();
    if (rn != 3 || Lit::as(mr.Ro) != 2 || Lit::as(mr.nis) != 2)
        return fail("a tensor-core level computes a matrix product");
    if (shape[0] != 16 || shape[1] != 8 || shape[2] != 8) return fail("a TF32 tensor-core tile is (16, 8, 8)");
    if (mr.To != F32 || mr.Tis->proj(2, 0) != F32 || mr.Tis->proj(2, 1) != F32)
        return fail("a TF32 tensor-core level multiplies F32 matrices");
    // A's rows and B's columns are its panels' tiled axes, so their contiguous dims are k and n.
    for (u64 slot = 0; slot != 2; ++slot) {
        auto staged = Axm::isa<tensor::stage>(Axm::peel<tensor::placed>(root->arg(2, 0)->proj(2, slot)));
        if (!staged) return fail("the tensor-core operands must be promoted into work-group panels");
        auto layout = staged->callee()->as<App>()->uncurry_args<4>()[2];
        if (Lit::isa(layout->proj(4, 0)) != slot) return fail("the tensor-core operands must be row-major");
    }
    // The warp's extents become literal once lower_to_mem specializes the tile bodies, which checks them again.
    for (u64 d = 0; d != rn; ++d)
        if (auto extent = Lit::isa(mr.Sr->proj(rn, d)); extent && *extent % shape[d])
            return fail("the tensor-core tile does not divide the warp's block");
    log_.d("tensor.tile_fuse {} L{} mma shape={}", op_, depth, fe::Join(shape));
    auto annotation = w.app(w.app(w.annex<tensor::mma>(), root->type()),
                            {w.lit_nat(rn), w.tuple(DefVec(rn, [&](size_t d) { return w.lit_nat(shape[d]); }))});
    return w.app(annotation, root);
}

const Def*
Levels::vector(const App* root, u64 rn, fe::Span<const u64> sizes, fe::Span<const LevelSpec> rest, u64 depth) {
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
    if (auto root = graph::map_reduce(op)) {
        auto mr  = btensor::MapRedPost::of(root->callee()->as<App>());
        auto Acc = Hole::find(request->callee()->as<App>()->arg(7, 6));
        if (Hole::isa_unset(Acc)) Acc = mr.To;
        if (!Check::test_alpha(Acc, mr.To)) {
            report_tile_request(request,
                                std::format("the levels' accumulator type {} does not match the root's accumulator "
                                            "type {}",
                                            Acc, mr.To));
            return op;
        }
    }
    auto parsed = tile_levels(levels);
    if (!parsed || parsed->empty()) {
        report_tile_request(request, "the levels are not a literal chain");
        return op;
    }
    auto level_list = std::move(*parsed);
    for (auto& level : level_list)
        if (level.pipeline == 0) level.pipeline = driver().arg_nat("tensor", "pipeline").value_or(1);
    auto group = members_of(members);
    DefSet in_group(group.begin(), group.end());
    Levels realizer(log(), request, in_group, outside, driver().arg_nat("tensor", "strict").value_or(0));
    if (auto root = graph::map_reduce(op))
        if (auto ranks = btensor::MapRedPost::of(root->callee()->as<App>()).ranks();
            ranks && level_list.front().kind != 0) {
            auto& w = op->world();
            level_list.insert(level_list.begin(),
                              LevelSpec{0, 1, 0, 0, w.pack(ranks->nloops(), w.lit_nat_0()), nullptr});
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
