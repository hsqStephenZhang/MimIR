#pragma once

#include <algorithm>
#include <bit>
#include <numeric>

#include <fe/worklist.h>

#include <mim/axm.h>
#include <mim/check.h>
#include <mim/def.h>
#include <mim/driver.h>
#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/gpu/gpu.h"
#include "mim/plug/tensor/phase/constraints.h"
#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/phase/limits.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase::lower_to_mem_detail {

// Why a schedule is rejected, so that a limit of this lowering does not read as one of the target.
inline constexpr char target[]    = "the target's instructions require it";
inline constexpr char unsafe[]    = "the schedule would otherwise compute something else";
inline constexpr char unlowered[] = "it is a limit of this lowering, not of the target";

/// Does `t` (recursively through immutable sigmas) contain a Pi?
inline bool contains_pi(const Def* t) {
    if (t->isa<Pi>()) return true;
    if (auto sig = t->isa_imm<Sigma>())
        for (auto op : sig->ops())
            if (contains_pi(op)) return true;
    return false;
}

inline void check_buffer_shape(const Def* rank, const Def* shape) {
    auto r = Lit::isa(rank);
    if (!r) rank->blame("cannot bufferize: non-literal tensor rank").bail();
    for (size_t d = 1; d < *r; ++d)
        if (!Lit::isa(shape->proj(*r, d)))
            shape->blame("cannot bufferize: runtime inner extent is not supported").bail();
}

/// Peels nested `Pack`s off `d`; returns the innermost body iff `d` is a constant splat — every axis a
/// `Pack` (so `d` is uniform, not e.g. a `Tuple` of distinct rows), bottoming out in a closed scalar
/// (index-independent, non-array). So `‹784, 1024; 1e-3›` yields the scalar `1e-3`, while an index-dependent
/// pack `‹i; f i›` or a genuine literal `((1, 2), (3, 4))` yields `nullptr`.
inline const Def* splat_scalar(const Def* d) {
    d = Axm::peel<tensor::buf>(d);
    if (auto splat = Axm::isa<tensor::splat>(d)) return splat->arg(2, 1);
    if (!d->isa<Pack>()) return nullptr;
    while (auto pack = d->isa<Pack>())
        d = pack->body();
    return (d->is_closed() && !d->type()->isa<Arr>()) ? d : nullptr;
}

/// Is `app` a tensor op whose lowering consumes a LowerToMem::fresh_mem?
inline bool wants_fresh_mem(const App* app) {
    return Axm::isa<tensor::tile_of>(app) || Axm::isa<tensor::tile_of_box>(app) || Axm::isa<tensor::slice>(app)
        || Axm::isa<tensor::slice_box>(app) || Axm::isa<tensor::stage>(app) || Axm::isa<tensor::placed>(app)
        || Axm::isa<tensor::tile_fold>(app) || Axm::isa<tensor::tiled>(app) || Axm::isa<tensor::tiled_multi>(app)
        || Axm::isa<tensor::tiled_n>(app) || Axm::isa<tensor::generate>(app) || Axm::isa<tensor::broadcast>(app)
        || Axm::isa<tensor::map_reduce_post>(app) || Axm::isa<tensor::pad>(app) || Axm::isa<tensor::concat>(app)
        || Axm::isa<tensor::gather>(app) || Axm::isa<tensor::scatter>(app) || Axm::isa<tensor::set>(app)
        || Axm::isa<tensor::from_array>(app) || Axm::isa<tensor::scalar>(app) || Axm::isa<tensor::splat>(app)
        || Axm::isa<tensor::prepack>(app);
}

/// Is `app` one of the tensor ops this phase bufferizes?
inline bool is_tensor_op(const App* app) { return Axm::isa<tensor::splat>(app) || wants_fresh_mem(app); }

class RealizationRequests {
public:
    explicit RealizationRequests(const Def* root)
        : root_(root) {
        scan(root, root);
    }

    fe::Vector<DefVec> shared() {
        fe::Vector<DefVec> groups;
        for (auto value : order_) {
            if (!bool(TensorType::isa(value->type()))) continue;
            bool available = true;
            for (auto var : value->free_vars())
                available &= root_->has_free_var(var);
            if (!available || !pure(value)) continue;
            auto group = std::ranges::find_if(
                groups, [&](const DefVec& group) { return Check::test_alpha(group.front(), value); });
            if (group == groups.end())
                groups.push_back({value});
            else
                group->push_back(value);
        }
        std::erase_if(groups, [&](const DefVec& group) {
            DefSet sites;
            for (auto value : group)
                for (auto site : sites_[value])
                    sites.insert(site);
            return sites.size() < 2 && group.size() < 2;
        });
        return groups;
    }

    bool shareable(const Def* value) { return bool(TensorType::isa(value->type())) && pure(value); }
    bool reusable(const Def* value) { return pure(value); }

private:
    bool positive(const Def* shape) {
        auto rank = Lit::isa(shape->arity());
        if (!rank) return false;
        for (size_t d = 0; d != *rank; ++d)
            if (auto extent = Lit::isa(shape->proj(*rank, d)); !extent || !*extent) return false;
        return true;
    }

    void region(const Def* fn) {
        auto body = fn->isa<Lam>();
        while (body && body->is_set()) {
            if (auto inner = body->body()->isa<Lam>())
                body = inner;
            else {
                scan(body->body(), fn);
                break;
            }
        }
    }

    void scan(const Def* def, const Def* site) {
        if (!seen_[site].insert(def).second || def->is_mutable() || def->isa<Var>() || def->isa<Axm>()) return;
        auto app = def->isa<App>();
        if (app && wants_fresh_mem(app)) sites_[app].insert(site);
        if (app && Axm::isa<tensor::tiled>(app)) {
            auto [meta, shapes, it, ins, body] = app->uncurry_args<5>();
            if (positive(shapes->proj(3, 0))) region(body);
        } else if (app && Axm::isa<tensor::tiled_multi>(app)) {
            auto plan = grid_tile_plan(app);
            if (positive(plan.shape)) region(plan.body);
        } else if (app && Axm::isa<tensor::tiled_n>(app)) {
            auto [meta, shapes_grid, boxes, it, ins, body] = app->uncurry_args<6>();
            if (positive(shapes_grid->proj(2, 1))) region(body);
        } else if (app && Axm::isa<tensor::tile_fold>(app)) {
            auto [meta, blocks, init, it, ins, body] = app->uncurry_args<6>();
            auto count                               = Lit::isa(blocks->proj(2, 0));
            if (count && *count) region(app->arg());
        } else if (app && Axm::isa<tensor::generate>(app)) {
            if (positive(app->callee()->as<App>()->arg())) region(app->arg());
        }
        // A placed stage is realized with its layout, so only the annotation is a request.
        auto placed = app ? Axm::isa<tensor::placed>(app) : nullptr;
        for (auto op : def->ops())
            if (op && placed && op == placed->arg())
                for (auto inner : op->ops())
                    scan(inner, site);
            else if (op)
                scan(op, site);
        if (app && wants_fresh_mem(app) && ordered_.insert(app).second) order_.push_back(app);
    }

    bool pure(const Def* value) {
        if (auto known = pure_.find(value); known != pure_.end()) return known->second;
        DefVec pending{value};
        DefSet seen;
        for (size_t i = 0; i != pending.size(); ++i) {
            auto def = pending[i];
            if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>()) continue;
            if (auto fn = def->isa<Lam>(); fn && (!fn->is_set() || fn->is_external())) return pure_[value] = false;
            if (auto app = def->isa<App>()) {
                if (auto axm = App::uncurry_callee(app)->isa<Axm>()) {
                    auto name = axm->sym().str();
                    if (!(name.starts_with("tensor.") || name.starts_with("core.") || name.starts_with("math.")
                          || name.starts_with("affine.") || name.starts_with("vec.") || name.starts_with("tuple.")
                          || name.starts_with("cps.cps2ds") || name == "btensor.clip_sched")
                        || name == "core.abs")
                        return pure_[value] = false;
                } else {
                    auto callee = App::uncurry_callee(app);
                    while (auto ex = callee->isa<Extract>())
                        callee = ex->tuple();
                    if (auto var = callee->isa<Var>())
                        if (auto binder = var->binder()->isa<Lam>(); !binder || binder->is_external())
                            return pure_[value] = false;
                }
            }
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        return pure_[value] = true;
    }

    const Def* root_;
    DefMap<DefSet> seen_, sites_;
    DefSet ordered_;
    DefVec order_;
    DefMap<bool> pure_;
};

/// The Lam a - possibly projected - parameter belongs to; `nullptr` for anything else.
inline Lam* param_of(const Def* d) {
    while (auto ex = d->isa<Extract>())
        d = ex->tuple();
    if (auto var = d->isa<Var>()) return var->binder()->isa_mut<Lam>();
    return nullptr;
}

inline const Def* scalar_result(const Def* source, const Def* mem, const Def* buf) {
    if (TensorType::isa(source->type())) return buf;
    return buffer::op_read(mem, buf, buf->world().tuple())->proj(2, 1);
}

using Annotate = std::function<const Def*(const Def*)>;

inline const Def*
copy_buffer(const Def* mem, const Def* dst, const Def* src, const Def* shape, Annotate annotate = {}) {
    auto& w     = mem->world();
    auto result = btensor::build_pointwise(
        w.sigma({mem->type(), dst->type()}), mem, dst, src, shape, Lit::as(shape->arity()), "foldCopy",
        [&](Defs iters, const Def* input, const Def* current) {
            auto [next, value] = buffer::op_read(current, input, btensor::folded_idx(shape, iters))->projs<2>();
            return std::pair{next, value};
        },
        nullptr, annotate);
    return result->proj(2, 0);
}

// Runtime inner extents bounded by a capped tile allocate their bound, since buffers need literal inner extents.
// A kernel's slots are static, so there the outer extent takes its bound as well.
inline const Def* bounded_shape(const Def* shape, bool outer = false) {
    auto& w = shape->world();
    auto r  = Lit::as(shape->arity());
    return w.tuple(DefVec(r, [&](size_t d) -> const Def* {
        auto extent = shape->proj(r, d);
        if ((d == 0 && !outer) || Lit::isa(extent)) return extent;
        if (auto upper = nat_upper(extent)) return w.lit_nat(*upper);
        return extent;
    }));
}

// Work-items walking a work-group's tile down a column would all hit one shared-memory bank, so its rows are padded.
// Sixteen bytes of padding keep the rows aligned for vector accesses.
inline const Def* padded_rows(const Def* storage) {
    auto& w = storage->world();
    auto r  = Lit::as(storage->arity());
    if (r < 2) return storage;
    auto last = Lit::isa(storage->proj(r, r - 1));
    if (!last || *last % 32) return storage;
    return w.tuple(DefVec(r, [&](size_t d) { return d + 1 == r ? w.lit_nat(*last + 4) : storage->proj(r, d); }));
}

/// @p storage with the row length of @p buffer, which may be padded.
inline const Def* padded_like(const Def* storage, const Def* buffer) {
    auto& w = storage->world();
    auto r  = Lit::as(storage->arity());
    if (auto buf = Axm::isa<buffer::Buf>(buffer->type()); buf && r >= 2 && Lit::isa(buf->arg(3, 0)) == r)
        if (auto row = buf->arg(3, 1)->proj(r, r - 1); Lit::isa(row) && row != storage->proj(r, r - 1))
            return w.tuple(DefVec(r, [&](size_t d) { return d + 1 == r ? row : storage->proj(r, d); }));
    return storage;
}

/// Copies the @p extent box of @p src, or splats it if it is no buffer, between buffers of different storage shapes.
inline const Def* copy_box(const Def* mem,
                           const Def* dst,
                           const Def* dst_shape,
                           const Def* src,
                           const Def* src_shape,
                           const Def* extent,
                           Annotate annotate = {}) {
    auto& w = mem->world();
    auto r  = Lit::as(extent->arity());
    return btensor::build_pointwise(
               w.sigma({mem->type(), dst->type()}), mem, dst, src, extent, r, "foldCopy",
               [&](Defs iters, const Def* input, const Def* current) {
                   if (!Axm::isa<buffer::Buf>(input->type())) return std::pair{current, input};
                   auto [next, value]
                       = buffer::op_read(current, input, btensor::folded_idx(src_shape, iters))->projs<2>();
                   return std::pair{next, value};
               },
               dst_shape, annotate)
        ->proj(2, 0);
}

inline bool full_output(const btensor::MapRedPost& mr) {
    auto ranks = mr.ranks();
    if (!ranks || Axm::isa<btensor::clip_sched>(mr.sched)) return false;
    auto& w        = mr.map_out->world();
    auto parameter = w.mut_lam(mr.map_out->type()->as<Pi>());
    auto mapped    = w.app(mr.map_out, parameter->var());
    DefSet axes;
    for (size_t a = 0; a != ranks->ro; ++a) {
        bool found = false;
        for (size_t d = 0; d != ranks->nloops(); ++d) {
            auto index = parameter->var()->proj(ranks->nloops(), d);
            if (mapped->proj(ranks->ro, a) == index && mr.So->proj(ranks->ro, a) == mr.Sr->proj(ranks->nloops(), d)) {
                if (!axes.insert(index).second) return false;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

inline bool sequential_tile(const Def* root) {
    if (!RealizationRequests(root).reusable(root)) return false;
    DefVec pending{root};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->is_mutable()) continue;
        if (auto app = def->isa<App>(); app && is_tensor_op(app)) {
            if (!(Axm::isa<tensor::map_reduce_post>(app) || Axm::isa<tensor::stage>(app)
                  || Axm::isa<tensor::placed>(app) || Axm::isa<tensor::slice>(app) || Axm::isa<tensor::slice_box>(app)
                  || Axm::isa<tensor::splat>(app) || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::generate>(app)
                  || (Axm::isa<tensor::tile_fold>(app) && app->type()->isa_imm<Sigma>())))
                return false;
        }
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return true;
}

inline void validate_array_literal(const App* app) {
    auto [r, s, T] = TensorType::decode(app->type()->as<Nom>());
    auto rank      = Lit::isa<u64>(r);
    if (!rank) app->blame("from_array requires a static shape").bail();
    for (auto extent : s->projs(*rank))
        if (!Lit::isa<u64>(extent)) app->blame("from_array requires a static shape").bail();
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(app->arg());
    while (!wl.empty()) {
        auto d = wl.pop();
        if (d->isa<Lit>()) continue;
        if (d->isa<Tuple>())
            wl.push(d->ops());
        else if (auto pack = d->isa<Pack>(); pack && Lit::isa<u64>(pack->arity()))
            wl.push(pack->body());
        else
            app->blame("from_array requires a static array literal").bail();
    }
}

inline const Def* lane_index(World& w) {
    auto lane = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::lane_id>(), w.tuple()));
    return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(32), lane);
}

/// The first row a warp holds of a warpgroup's accumulator, 0 for a warp's own.
inline const Def* group_row(World& w, bool group) {
    if (!group) return w.lit_nat_0();
    auto rank = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::warp_rank>(), w.tuple()));
    return w.call(core::nat::mul, Defs{rank, w.lit_nat(16)});
}

/// The warp loop's lane as an index; `tensor.distribute` takes the same term for the own lane.
inline const Def* warp_lane(World& w) {
    auto lane = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::lane_id>(), w.tuple()));
    return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(32), lane);
}

/// The loop each coordinate of @p map reads, unscaled and up to an offset free of the loops.
inline std::optional<std::vector<u64>> map_dims(const Def* map, u64 rn) {
    auto& w = map->world();
    auto pi = map->type()->isa<Pi>();
    if (!pi) return {};
    auto var = w.mut_lam(pi)->var();
    auto out = w.app(map, var);
    auto ri  = Lit::isa(out->type()->arity());
    if (!ri) return {};
    auto dims = std::vector<u64>();
    for (u64 d = 0; d != *ri; ++d) {
        auto form = Linear::of(out->proj(*ri, d));
        auto loop = rn;
        for (u64 l = 0; l != rn; ++l)
            if (auto atom = form.atoms.find(var->proj(rn, l)); atom != form.atoms.end()) {
                if (atom->second != 1 || loop != rn) return {};
                loop = l;
            }
        if (loop == rn) return {};
        dims.push_back(loop);
    }
    return dims;
}

/// The `(row, col)` that a `tensor.mma_m16n8k8_tf32` coordinate map assigns to `arg`, as Nats.
inline std::array<const Def*, 2> fragment_coords(World& w, Enum auto map, const Def* arg) {
    auto [row, col]
        = w.app(w.annex(map), arg)->template projs<2>([&](const Def* x) { return w.call<core::idx2nat>(x); });
    return {row, col};
}

} // namespace mim::plug::tensor::phase::lower_to_mem_detail
