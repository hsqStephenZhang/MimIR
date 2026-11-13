#include <algorithm>
#include <array>
#include <limits>

#include <fe/format.h>

#include <mim/def.h>
#include <mim/driver.h>
#include <mim/plugin.h>
#include <mim/tuple.h>
#include <mim/world.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

bool tensor_input_type(const Def* type) {
    if (TensorType::isa(type)) return true;
    if (auto sigma = type->isa<Sigma>()) return std::ranges::any_of(sigma->ops(), tensor_input_type);
    return false;
}

const Def* read_through(const Def* base, const Def* index) {
    auto& w = base->world();
    const Def *input, *s_in, *s_out, *rank;
    bool wraps;
    if (auto rep = Axm::isa<tensor::repeat>(base)) {
        auto [Tr, in, out] = rep->callee()->as<App>()->uncurry_args<3>();
        input = rep->arg(), s_in = in, s_out = out, rank = Tr->proj(2, 1), wraps = true;
    } else if (auto bc = Axm::isa<tensor::broadcast>(base)) {
        auto [in, out, i] = bc->args<3>();
        auto [_, r]       = bc->callee()->as<App>()->args<2>();
        input = i, s_in = in, s_out = out, rank = r, wraps = false;
    } else {
        return nullptr;
    }
    auto r = Lit::isa<u64>(rank);
    if (!r) return nullptr;

    auto mapped = DefVec(*r);
    for (u64 d = 0; d != *r; ++d) {
        auto in  = s_in->proj(*r, d);
        auto out = s_out->proj(*r, d);
        auto idx = index->proj(*r, d);
        if (in == out)
            mapped[d] = idx;
        else if (Lit::isa(in) == 1)
            mapped[d] = w.lit_idx(1, 0);
        else if (auto e = Lit::isa(in), x = Lit::isa(idx); wraps && e && *e && x)
            mapped[d] = w.lit_idx(*e, *x % *e);
        else
            return nullptr;
    }
    return op_get(input, w.tuple(mapped));
}

} // namespace

namespace {

/// `min (a, b)` as built by `select (a < b, a, b)`.
std::optional<std::pair<const Def*, const Def*>> isa_min(const Def* x) {
    if (auto ex = x->isa<Extract>())
        if (auto tuple = ex->tuple()->isa<Tuple>(); tuple && tuple->num_ops() == 2)
            if (auto lt = Axm::isa(core::ncmp::l, ex->index())) {
                auto [a, b] = lt->args<2>();
                if (tuple->op(1) == a && tuple->op(0) == b) return std::pair{a, b};
            }
    return {};
}

std::optional<u64> min_upper(std::optional<u64> a, std::optional<u64> b) {
    if (a && b) return std::min(*a, *b);
    return a ? a : b;
}

/// An upper bound of the truncated `a − b`; tile bounds take the form `(min (lo + ext − 1, s − 1) + 1) − lo`.
std::optional<u64> sub_upper(const Def* a, const Def* b) {
    if (a == b) return 0;
    auto upper = nat_upper(a);
    if (auto c = Lit::isa(b); upper && c) upper = *upper > *c ? *upper - *c : 0;
    if (auto add = Axm::isa(core::nat::add, a)) {
        auto [x, y] = add->args<2>();
        if (y == b) return min_upper(upper, nat_upper(x));
        if (x == b) return min_upper(upper, nat_upper(y));
        if (auto c = Lit::isa(x))
            if (auto rest = sub_upper(y, b)) upper = min_upper(upper, *rest + *c);
        if (auto c = Lit::isa(y))
            if (auto rest = sub_upper(x, b)) upper = min_upper(upper, *rest + *c);
    }
    if (auto sub = Axm::isa(core::nat::sub, a)) upper = min_upper(upper, sub_upper(sub->arg(2, 0), b));
    if (auto m = isa_min(a)) upper = min_upper(upper, min_upper(sub_upper(m->first, b), sub_upper(m->second, b)));
    return upper;
}

} // namespace

bool nat_bounded_by(const Def* x, const Def* bound) {
    if (Check::test_alpha(x, bound)) return true;
    if (auto upper = nat_upper(x); upper && Lit::isa(bound) && *upper <= *Lit::isa(bound)) return true;
    if (auto m = isa_min(x)) return nat_bounded_by(m->first, bound) || nat_bounded_by(m->second, bound);
    return false;
}

std::optional<u64> nat_upper(const Def* x) {
    if (auto l = Lit::isa(x)) return l;
    if (auto op = Axm::isa<core::nat>(x)) {
        auto [a, b] = op->args<2>();
        auto ua = nat_upper(a), ub = nat_upper(b);
        auto lb = Lit::isa(b);
        u64 r;
        switch (op.id()) {
            case core::nat::add:
                if (ua && ub && !__builtin_add_overflow(*ua, *ub, &r)) return r;
                return {};
            case core::nat::mul:
                if (ua && ub && !__builtin_mul_overflow(*ua, *ub, &r)) return r;
                return {};
            case core::nat::sub: return sub_upper(a, b);
            case core::nat::div:
                if (ua && lb && *lb) return *ua / *lb;
                return {};
            case core::nat::rem:
                if (lb && *lb) return min_upper(ua, *lb - 1);
                return {};
        }
    }
    if (auto m = isa_min(x)) return min_upper(nat_upper(m->first), nat_upper(m->second));
    if (auto ex = x->isa<Extract>())
        if (auto tuple = ex->tuple()->isa<Tuple>()) {
            std::optional<u64> result = 0;
            for (auto op : tuple->ops()) {
                auto u = nat_upper(op);
                if (!u) return {};
                result = std::max(*result, *u);
            }
            return result;
        }
    return {};
}

std::pair<const Def*, u64> region_body(const Def* body) {
    if (auto bound = Axm::isa<tensor::bind>(body)) {
        auto inner = bound->arg();
        if (auto cluster = Axm::isa<tensor::cluster>(inner)) inner = cluster->arg();
        if (auto special = Axm::isa<tensor::specialize>(inner)) inner = special->arg();
        return {inner, Lit::expect(bound->decurry()->arg(), "a literal tensor.bind level")};
    }
    return {body, 0};
}

void check_region_inputs(const Def* function, const Def* region) {
    function   = region_body(function).first;
    auto outer = function->isa<Lam>();
    if (!outer || !outer->is_set()) return;
    auto inner = outer->body()->isa<Lam>();
    if (!inner || !inner->is_set()) return;
    DefSet bound;
    if (auto var = outer->has_var()) bound.insert(var);
    if (auto var = inner->has_var()) bound.insert(var);
    auto check = [&](const Def* operation, const Def* input) {
        if (tensor_input_type(input->type()) && !input->is_closed()
            && !std::ranges::any_of(input->free_vars(), [&](auto var) { return bound.contains(var); }))
            operation
                ->blame("tensor region operation {} reads outer tensor input {}; pass it explicitly",
                        App::uncurry_callee(operation)->sym(), input)
                .bail();
    };
    DefVec results{inner->body()};
    for (size_t i = 0; i != results.size(); ++i) {
        auto result = results[i];
        if (auto tuple = result->isa<Tuple>()) {
            for (auto op : tuple->ops())
                results.push_back(op);
        } else {
            auto source = result;
            while (auto extract = source->isa<Extract>())
                source = extract->tuple();
            if (source->isa<Var>()) check(region, result);
        }
    }
    auto& ops = function->driver().graph_operations();
    DefVec pending{inner->body()};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (auto view = ops.decode(def, 65536)) {
            for (auto input : view->inputs) {
                if (!tensor_input_type(input.value->type()) || input.value->is_closed()) continue;
                check(def, input.value);
                pending.push_back(input.value);
            }
            continue;
        }
        if (auto lam = def->isa<Lam>()) {
            if (!lam->is_set()) continue;
            if (auto var = lam->has_var()) bound.insert(var);
        } else if (def->is_mutable() || def->isa<Var>() || def->isa<Axm>()) {
            continue;
        }
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
}

const Def* normalize_sdpa_online(const Def*, const Def* callee, const Def*) {
    auto params = callee->as<App>()->callee()->as<App>()->arg();
    if (Lit::isa(params->proj(4, 2)) == 0 || Lit::isa(params->proj(4, 3)) == 0)
        callee->blame("tensor.sdpa_online requires positive tile sizes").bail();
    if (Lit::isa(callee->as<App>()->arg()->proj(5, 2)) == 0)
        callee->blame("tensor.sdpa_online requires a nonempty key sequence").bail();
    if (Lit::isa(callee->as<App>()->arg()->proj(5, 4)) == 0)
        callee->blame("tensor.sdpa_online requires a nonempty output feature dimension").bail();
    return nullptr;
}

const Def* normalize_blis_candidate(const Def*, const Def*, const Def* producer) {
    auto& w = producer->world();
    if (w.driver().is_loaded("gpu") || !w.driver().arg_nat("tensor", "group").value_or(1)) return w.lit_ff();
    auto app = graph::map_reduce(producer);
    if (!app) return nullptr;
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || !lit_projs(mr.So, ranks->ro) || !lit_projs(mr.Sr, ranks->nloops())) return nullptr;
    auto scratch = w.inherit();
    Rewriter importer(*scratch);
    auto imported = importer.rewrite(producer);
    auto size     = scratch->app(scratch->annex(core::trait::size), importer.rewrite(mr.To));
    if (!Lit::isa(size)) return mr.To->is_closed() ? w.lit_ff() : nullptr;
    auto config = scratch->app(scratch->annex<group_blocks>(),
                               {size, scratch->app(scratch->annex<vec_width>(), scratch->lit_nat(4)),
                                scratch->lit_nat(w.driver().arg_nat("tensor", "group-registers").value_or(32)),
                                scratch->lit_nat(w.driver().arg_nat("tensor", "group-cache").value_or(262144))});
    auto [rows, cols, kc, nc] = config->projs<4>();
    if (!Lit::isa(rows).value_or(0) || !Lit::isa(kc).value_or(0)) return w.lit_ff();
    return w.lit_bool(bool(blis_group(imported, rows, scratch->tuple({cols, kc, nc})).value));
}

MultiTilePlan multi_tile_plan(const App* app) {
    auto [meta, shapes_grid, boxes, input_type, inputs, body] = app->uncurry_args<6>();
    auto [n, g, types, ranks]                                 = meta->projs<4>();
    auto [shapes, grid]                                       = shapes_grid->projs<2>();
    auto count                                                = Lit::expect(n, "a literal tensor.tiled_n result count");
    auto dims                                                 = Lit::expect(g, "a literal tensor.tiled_n grid rank");
    auto plan = MultiTilePlan{types, ranks, shapes, grid, boxes, inputs, body, {}, {}};
    u64 trips = 1;
    for (u64 d = 0; d != dims; ++d) {
        auto extent = Lit::expect(grid->proj(dims, d), "a literal tensor.tiled_n grid extent");
        if (extent && trips > 65536 / extent)
            app->blame("tensor.tiled_n ownership proof exceeds 65536 grid points").bail();
        trips *= extent;
    }
    auto& w = app->world();
    for (u64 t = 0; t != trips; ++t) {
        DefVec point(dims);
        auto index = t;
        for (u64 d = dims; d-- != 0;) {
            auto extent = Lit::as(grid->proj(dims, d));
            point[d]    = w.lit_nat(index % extent);
            index /= extent;
        }
        plan.points.push_back(w.tuple(point));
        auto owned = w.app(boxes, plan.points.back());
        plan.owned.push_back(DefVec(count, [&](size_t i) { return owned->proj(count, i); }));
    }
    for (u64 i = 0; i != count; ++i) {
        auto rank  = Lit::expect(ranks->proj(count, i), "a literal tensor.tiled_n result rank");
        auto shape = shapes->proj(count, i);
        u64 volume = 1, covered = 0;
        for (u64 d = 0; d != rank; ++d) {
            auto size = Lit::expect(shape->proj(rank, d), "a literal tensor.tiled_n result extent");
            if (size && volume > std::numeric_limits<u64>::max() / size)
                app->blame("tensor.tiled_n result volume overflows").bail();
            volume *= size;
        }
        for (size_t t = 0; t != plan.points.size(); ++t) {
            auto [off, ext] = plan.owned[t][i]->projs<2>();
            u64 cells       = 1;
            for (u64 d = 0; d != rank; ++d) {
                auto start = Lit::expect(off->proj(rank, d), "a literal tensor.tiled_n owned origin");
                auto size  = Lit::expect(ext->proj(rank, d), "a literal tensor.tiled_n owned extent");
                auto limit = Lit::as(shape->proj(rank, d));
                if (start > limit || size > limit - start)
                    app->blame("tensor.tiled_n owned box is outside its result").bail();
                if (size && cells > std::numeric_limits<u64>::max() / size)
                    app->blame("tensor.tiled_n owned volume overflows").bail();
                cells *= size;
            }
            if (!cells) continue;
            if (covered > volume - cells) app->blame("tensor.tiled_n owned boxes overlap").bail();
            covered += cells;
            for (size_t previous = 0; previous != t; ++previous) {
                auto [other_off, other_ext] = plan.owned[previous][i]->projs<2>();
                bool overlap                = true;
                for (u64 d = 0; d != rank; ++d) {
                    auto a = Lit::as(off->proj(rank, d)), b = Lit::as(other_off->proj(rank, d));
                    auto ae = Lit::as(ext->proj(rank, d)), be = Lit::as(other_ext->proj(rank, d));
                    overlap &= ae && be && a < b + be && b < a + ae;
                }
                if (overlap) app->blame("tensor.tiled_n owned boxes overlap").bail();
            }
        }
        if (covered != volume) app->blame("tensor.tiled_n owned boxes do not cover their result").bail();
    }
    return plan;
}

const Def* normalize_group_root(const Def*, const Def*, const Def* producer) {
    auto& w  = producer->world();
    auto app = graph::map_reduce(producer);
    if (!app) return nullptr;
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || !Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>())) return w.lit_tt();
    auto schedule = mr.sched;
    if (auto clipped = Axm::isa<btensor::clip_sched>(schedule)) schedule = clipped->arg(4, 3);
    auto record = w.app(w.annex<tile_sched>(), schedule);
    if (!Check::test_alpha(schedule, w.app(w.annex<btensor::mk_sched>(), record))) return w.lit_tt();
    if (!ranks->rr) return w.lit_ff();
    for (u64 k = 0; k != ranks->nis; ++k) {
        auto map  = mr.maps->proj(ranks->nis, k);
        auto pi   = map->type()->as<Pi>();
        auto var  = w.mut_lam(pi->dom(), pi->codom())->var();
        auto read = w.app(map, var);
        for (u64 d = 0; d != ranks->ro; ++d) {
            auto at = w.tuple(DefVec(ranks->nloops(), [&](size_t e) {
                return e == d ? w.call<affine::lit>(w.lit_nat_0()) : var->proj(ranks->nloops(), e);
            }));
            if (w.app(map, at) == read) return w.lit_tt();
        }
    }
    return w.lit_ff();
}

ProducerTile
producer_tile(const Def* producer, const Def* off, const Def* ext, const Def* box_cap, const ReductionCut* cut) {
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "producer is not a map_reduce_post"};
    auto mr  = btensor::MapRedPost::of(app->callee()->as<App>());
    auto rks = mr.ranks();
    if (!rks) return {nullptr, "producer ranks are not literal"};
    auto& w = producer->world();
    auto ro = rks->ro;
    auto rn = rks->nloops();
    if (!Check::test_alpha(mr.TSched, w.annex<btensor::SchedT>()))
        return {nullptr, "producer has a custom schedule type"};
    auto schedule = mr.sched, source_loop_cap = mr.Sr, source_output_cap = mr.So;
    if (auto clipped = Axm::isa<btensor::clip_sched>(schedule)) {
        source_output_cap = clipped->arg(4, 1);
        source_loop_cap   = clipped->arg(4, 2);
        schedule          = clipped->arg(4, 3);
    }
    auto record = w.app(w.annex<tile_sched>(), schedule);
    if (!Check::test_alpha(schedule, w.app(w.annex<btensor::mk_sched>(), record)))
        return {nullptr, "producer schedule is not a canonical nest"};
    auto slice
        = [&](const Def* type, const Def* rank, const Def* shape, const Def* value, const Def* lo, const Def* size) {
              auto call = w.app(w.annex<tensor::slice>(), {type, rank});
              call      = w.app(call, shape);
              call      = w.app(call, {lo, w.pack(rank, w.lit_nat_1()), size});
              return w.app(call, value);
          };
    auto image = [&](const Def* map, const Def* n, const Def* m, const Def* lo, const Def* size, const Def* shape) {
        auto call = w.app(w.app(w.annex<affine::image>(), {n, m}), map);
        return w.app(call, {lo, size, shape});
    };
    auto pre = w.app(w.app(w.annex<affine::preimage>(), {mr.Rn, mr.Ro}), mr.map_out);
    pre      = w.app(pre, {off, ext, mr.Sr});
    if (pre->proj(4, 3) != w.lit_tt()) return {nullptr, "write preimage is not proven exact"};
    auto lo = pre->proj(4, 0), size = pre->proj(4, 1), cap = pre->proj(4, 2);
    for (u64 d = ro; d != rn; ++d)
        if (Lit::isa(lo->proj(rn, d)) != 0 || size->proj(rn, d) != mr.Sr->proj(rn, d))
            return {nullptr, "write preimage cuts a reduction dimension"};
    if (cut) {
        if (cut->loop < ro || cut->loop >= rn) return {nullptr, "reduction cut names no reduction loop"};
        if (box_cap) return {nullptr, "a reduction cut takes no storage cap"};
        auto at = [&](const Def* box, const Def* value) {
            return w.tuple(DefVec(rn, [&](size_t d) { return d == cut->loop ? value : box->proj(rn, d); }));
        };
        lo   = at(lo, cut->k0);
        size = at(size, cut->k);
        cap  = at(cap, cut->k);
    }

    auto output_cap = ext;
    if (box_cap) {
        for (u64 d = 0; d != ro; ++d)
            if (auto requested = Lit::isa(ext->proj(ro, d)), bound = Lit::isa(box_cap->proj(ro, d));
                requested && bound && *requested > *bound)
                return {nullptr, "tile extent exceeds its storage cap"};
        auto bound = w.app(w.app(w.annex<affine::preimage>(), {mr.Rn, mr.Ro}), mr.map_out);
        bound      = w.app(bound, {off, box_cap, source_loop_cap});
        if (bound->proj(4, 3) != w.lit_tt()) return {nullptr, "capped write preimage is not proven exact"};
        cap        = bound->proj(4, 2);
        output_cap = box_cap;
    } else if (!lit_projs(ext, ro) || !lit_projs(size, rn)) {
        cap        = w.tuple(DefVec(rn, [&](size_t d) -> const Def* {
            auto bound = cap->proj(rn, d);
            if (d >= ro) return size->proj(rn, d);
            if (Lit::isa(bound)) return bound;
            if (auto upper = nat_upper(bound)) return w.lit_nat(*upper);
            return source_loop_cap->proj(rn, d);
        }));
        output_cap = w.tuple(DefVec(ro, [&](size_t d) -> const Def* {
            if (auto requested = ext->proj(ro, d); Lit::isa(requested)) return requested;
            if (auto upper = nat_upper(ext->proj(ro, d))) return w.lit_nat(*upper);
            return source_output_cap->proj(ro, d);
        }));
    } else {
        cap        = size;
        output_cap = ext;
    }

    auto [values, post_values] = app->args<2>();
    auto shapes = DefVec(), starts = DefVec(), inputs = DefVec();
    auto post_shapes = DefVec(), post_starts = DefVec(), post_inputs = DefVec();
    auto boxes
        = [&](u64 count, const Def* ranks, const Def* source_shapes, const Def* source_maps, const Def* source_types,
              const Def* source_values, const Def* n, const Def* origin, const Def* domain, const Def* domain_cap,
              DefVec& result_shapes, DefVec& result_starts, DefVec& result_values) -> bool {
        for (u64 i = 0; i != count; ++i) {
            auto rank = ranks->proj(count, i);
            if (!Lit::isa(rank)) return false;
            auto shape = source_shapes->proj(count, i), map = source_maps->proj(count, i);
            auto box = image(map, n, rank, origin, domain, shape);
            if (Axm::isa<affine::image>(box)) return false;
            auto start = box->proj(4, 0), extent = box->proj(4, 1);
            auto empty = false;
            auto r     = *Lit::isa(rank);
            for (u64 d = 0; d != r; ++d)
                empty |= Lit::isa(extent->proj(r, d)) == 0;
            if (empty) {
                start  = w.pack(rank, w.lit_nat_0());
                extent = shape;
            }
            result_shapes.push_back(extent);
            result_starts.push_back(start);
            auto value = source_values->proj(count, i);
            if (!empty && box_cap) {
                auto storage = image(map, n, rank, origin, domain_cap, shape);
                if (Axm::isa<affine::image>(storage)) return false;
                auto view = w.app(w.app(w.annex<slice_box>(), {source_types->proj(count, i), rank}), shape);
                result_values.push_back(w.app(w.app(view, {start, extent, storage->proj(4, 2)}), value));
            } else {
                result_values.push_back(empty ? value
                                              : slice(source_types->proj(count, i), rank, shape, value, start, extent));
            }
        }
        return true;
    };
    if (!boxes(rks->nis, mr.Ris, mr.Sis, mr.maps, mr.Tis, values, mr.Rn, lo, size, cap, shapes, starts, inputs))
        return {nullptr, "input map does not beta-reduce or has symbolic rank"};
    if (!cut
        && !boxes(rks->nps, mr.Rps, mr.Sps, mr.post_maps, mr.Tps, post_values, mr.Ro, off, ext, output_cap, post_shapes,
                  post_starts, post_inputs))
        return {nullptr, "post map does not beta-reduce or has symbolic rank"};
    auto init = mr.init;
    if (!cut) {
        if (auto outs = tensor_init(mr)) {
            auto cells = w.tuple(DefVec(rn, [&](size_t d) { return d < ro ? size->proj(rn, d) : w.lit_nat_1(); }));
            auto box   = image(outs->map, mr.Rn, mr.Ro, lo, cells, outs->shape);
            if (Axm::isa<affine::image>(box)) return {nullptr, "outs map does not beta-reduce"};
            auto start = box->proj(4, 0), extent = box->proj(4, 1);
            auto builder = w.app(w.app(w.annex<tile_init>(), {mr.To, mr.Ro, mr.Rn}), outs->shape);
            init         = w.app(w.app(builder, {outs->map, outs->outs}), {lo, start, extent});
        }
    }
    auto sched = schedule;
    if (cap != size || output_cap != ext) {
        auto parallel = w.tuple(DefVec(ro, [&](size_t d) { return size->proj(rn, d); }));
        sched = w.app(w.app(w.annex<btensor::clip_sched>(), {mr.Ro, mr.Rn}), {parallel, output_cap, cap, sched});
    }
    const Def* builder = w.annex<tile_rebuild>();
    for (auto arg : app->callee()->as<App>()->uncurry_args<8>())
        builder = w.app(builder, arg);
    if (cut) {
        auto result = w.app(builder->proj(2, 1), {ext, size, sched});
        result      = w.app(result, w.tuple(shapes));
        result      = w.app(result, {lo, off, w.tuple(starts)});
        return {w.app(result, {cut->acc, w.tuple(inputs)}), {}};
    }
    auto result = w.app(w.app(builder->proj(2, 0), init->type()), init);
    result      = w.app(result, {ext, size, sched});
    result      = w.app(result, {w.tuple(shapes), w.tuple(post_shapes)});
    result      = w.app(result, {lo, off, w.tuple(starts), w.tuple(post_starts)});
    return {w.app(result, {w.tuple(inputs), w.tuple(post_inputs)}), {}};
}

namespace {

/// Under `tensor:tile-ref=1`, @p callee applied to @p arg as its specification @p Ref, a slice of the whole producer.
template<class Ref>
const Def* tile_spec(const Def* callee, const Def* arg) {
    auto& w = arg->world();
    if (!w.driver().arg_nat("tensor", "tile-ref").value_or(0)) return nullptr;
    auto [meta, shape] = callee->as<App>()->uncurry_args<2>();
    return w.app(w.app(w.app(w.annex<Ref>(), meta), shape), arg);
}

} // namespace

const Def* normalize_tile_of(const Def*, const Def* callee, const Def* arg) {
    if (auto spec = tile_spec<tile_of_ref>(callee, arg)) return spec;
    auto [producer, off, ext] = arg->projs<3>();
    return producer_tile(producer, off, ext).value;
}

const Def* normalize_tile_of_box(const Def*, const Def* callee, const Def* arg) {
    if (auto spec = tile_spec<tile_of_box_ref>(callee, arg)) return spec;
    auto [producer, off, ext, cap] = arg->projs<4>();
    return producer_tile(producer, off, ext, cap).value;
}

namespace {

/// Whether the access maps @p a and @p b send every index to the same coordinates.
bool same_map(const Def* a, const Def* b) {
    if (a == b) return true;
    auto& w = a->world();
    auto pi = a->type()->isa<Pi>();
    if (!pi || !Check::test_alpha(pi, b->type())) return false;
    auto var = w.mut_lam(pi)->var();
    auto x = w.app(a, var), y = w.app(b, var);
    auto r = Lit::isa(x->arity());
    if (!r) return false;
    for (u64 d = 0; d != *r; ++d)
        if (!(Linear::of(x->proj(*r, d)).expanded() == Linear::of(y->proj(*r, d)).expanded())) return false;
    return true;
}

} // namespace

ProducerTile producer_init(const Def* producer, const Def* off, const Def* ext) {
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "producer is not a map_reduce_post"};
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    auto& w = producer->world();
    if (auto outs = tensor_init(mr)) {
        if (!same_map(outs->map, mr.map_out)) return {nullptr, "accumulator is not stored in the output layout"};
        auto call = w.app(w.app(w.annex<tile_of_ref>(), {mr.To, mr.Ro}), outs->shape);
        return {w.app(call, {outs->outs, off, ext}), {}};
    }
    return {w.app(w.app(w.annex<tensor::splat>(), {mr.To, mr.Ro}), {ext, mr.init}), {}};
}

bool identity_post(const Def* post) {
    auto lam = post->isa_mut<Lam>();
    if (!lam || !lam->is_set()) return false;
    if (auto y = Lam::isa_ret_arg(lam)) return y == lam->var(lam->num_vars(), 0)->proj(2, 0);
    return lam->body() == lam->var()->proj(2, 0);
}

ProducerTile producer_post(const Def* producer, const Def* off, const Def* ext, const Def* acc) {
    auto app = graph::map_reduce(producer);
    if (!app) return {nullptr, "producer is not a map_reduce_post"};
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    if (Lit::isa(mr.nps) == 0 && identity_post(mr.post) && acc->type() == type_tensor(mr.Ro, ext, mr.Tp))
        return {acc, {}};
    auto& w      = producer->world();
    auto posts   = app->arg(2, 1);
    auto builder = w.app(w.annex<post_tile>(), {mr.To, mr.Tp, mr.Ro, mr.nps, mr.Tps, mr.Rps, mr.Sps});
    builder      = w.app(builder, {off, ext, mr.post, mr.post_maps, posts});
    return {w.app(builder, acc), {}};
}

const Def* normalize_tile_legal(const Def*, const Def*, const Def* arg) {
    auto [producer, off, ext] = arg->projs<3>();
    if (!graph::map_reduce(producer)) return nullptr;
    return arg->world().lit_bool(producer_tile(producer, off, ext).value != nullptr);
}

namespace {

/// Whether @p producer is a `map_reduce_post` whose accumulator and element types are `T` and `U` of @p callee.
bool reduction_types(const Def* callee, const Def* producer) {
    auto app = graph::map_reduce(producer);
    if (!app) return false;
    auto mr        = btensor::MapRedPost::of(app->callee()->as<App>());
    auto [T, U, _] = callee->as<App>()->uncurry_args<2>()[0]->projs<3>();
    if (!Check::test_alpha(T, mr.To) || !Check::test_alpha(U, mr.Tp))
        callee->blame("reduction tile types {} and {} differ from the producer's {} and {}", T, U, mr.To, mr.Tp).bail();
    return true;
}

} // namespace

const Def* normalize_mma_warp(const Def*, const Def*, const Def* arg) {
    auto& w = arg->world();
    return w.driver().is_loaded("gpu") ? nullptr : w.app(w.annex(mma_m16n8k8_tf32::mma_ref), arg);
}

const Def* normalize_warp_keep(const Def*, const Def*, const Def* arg) {
    return arg->world().driver().is_loaded("gpu") ? nullptr : arg;
}

const Def* normalize_tile_of_init(const Def*, const Def* callee, const Def* arg) {
    auto [producer, off, ext] = arg->projs<3>();
    if (!reduction_types(callee, producer)) return nullptr;
    return producer_init(producer, off, ext).value;
}

const Def* normalize_tile_of_partial(const Def*, const Def* callee, const Def* arg) {
    auto [producer, off, ext, loop, k0, k, acc] = arg->projs<7>();
    auto index                                  = Lit::isa(loop);
    if (!index || !reduction_types(callee, producer)) return nullptr;
    auto cut = ReductionCut{*index, k0, k, acc};
    return producer_tile(producer, off, ext, nullptr, &cut).value;
}

const Def* normalize_tile_of_post(const Def*, const Def* callee, const Def* arg) {
    auto [producer, off, ext, acc] = arg->projs<4>();
    if (!reduction_types(callee, producer)) return nullptr;
    return producer_post(producer, off, ext, acc).value;
}

namespace {

bool nests_region(const Def* body) {
    DefVec pending{body};
    DefSet seen{body};
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (Axm::isa<tensor::tiled>(def) || Axm::isa<tensor::tile_fold>(def) || Axm::isa<tensor::tiled_n>(def)
            || Axm::isa<tensor::tiled_multi>(def))
            return true;
        if (def->isa<Axm>()) continue;
        for (auto op : def->ops())
            if (op && seen.insert(op).second) pending.push_back(op);
    }
    return false;
}

} // namespace

const Def* normalize_tiled(const Def*, const Def* c, const Def* annotated) {
    auto [meta, shapes, input_type, inputs] = c->as<App>()->uncurry_args<4>();
    auto [body, bind]                       = region_body(annotated);
    check_region_inputs(body, c);
    auto [s, ts, tail] = shapes->projs<3>();
    if (auto rank = Lit::isa(meta->proj(2, 1))) {
        for (size_t d = 0; d != *rank; ++d)
            if (Lit::isa(ts->proj(*rank, d)) == 0) c->blame("tensor.tiled requires positive tile sizes").bail();
        for (size_t d = 0; d != *rank; ++d)
            if (Lit::isa(s->proj(*rank, d)) == 0) {
                auto& w = c->world();
                return w.app(w.app(w.annex<tensor::splat>(), meta), {s, w.bot(meta->proj(2, 0))});
            }
        // An open body may be a copy that an enclosing reduction is still filling; reducing it now leaves stale vars.
        // A nested region keeps its destination only through this region's boundary.
        if (!bind && s == ts && body->is_closed() && !nests_region(body))
            return c->world().app(c->world().app(body, {c->world().pack(*rank, c->world().lit_nat(0)), s}), inputs);
    }
    return nullptr;
}

GridTilePlan grid_tile_plan(const App* app) {
    if (Axm::isa<tiled>(app)) {
        auto [meta, shapes, it, ins, annotated] = app->uncurry_args<5>();
        auto [type, rank]                       = meta->projs<2>();
        auto [shape, tiles, tail]               = shapes->projs<3>();
        auto [body, bind]                       = region_body(annotated);
        auto& w                                 = app->world();
        auto r                                  = Lit::expect(rank, "a literal tensor.tiled grid rank");
        auto axes                               = w.tuple(DefVec(r, [&](size_t d) { return w.lit_idx(r, d); }));
        return {type, rank, shape, shape, tiles, tail, axes, ins, body, 1, r, false, bind};
    }
    auto [meta, shapes_grid, axes, it, ins, annotated] = app->uncurry_args<6>();
    auto [body, bind]                                  = region_body(annotated);
    auto [n, g, types, ranks]                          = meta->projs<4>();
    auto [shapes, shape, tiles, tail]                  = shapes_grid->projs<4>();
    auto count                                         = Lit::expect(n, "a literal tensor.tiled_multi result count");
    auto rank                                          = Lit::expect(g, "a literal tensor.tiled_multi grid rank");
    for (u64 d = 0; d != rank; ++d)
        if (Lit::isa(tiles->proj(rank, d)) == 0) app->blame("tensor.tiled_multi requires positive tile sizes").bail();
    for (u64 i = 0; i != count; ++i) {
        auto r            = Lit::expect(ranks->proj(count, i), "a literal tensor.tiled_multi result rank");
        auto result_shape = shapes->proj(count, i);
        auto mapping      = axes->proj(count, i);
        fe::Vector<bool> seen(r, false);
        for (u64 d = 0; d != rank; ++d) {
            auto a = Lit::expect(mapping->proj(rank, d), "a literal tensor.tiled_multi axis");
            if (a >= r || seen[a])
                app->blame("tensor.tiled_multi requires each grid dimension on a distinct result axis").bail();
            seen[a]     = true;
            auto extent = result_shape->proj(r, a);
            auto grid   = shape->proj(rank, d);
            if (!Check::test_alpha(extent, grid))
                app->blame("tensor.tiled_multi result extent must equal its mapped grid extent").bail();
        }
    }
    return {types, ranks, shapes, shape, tiles, tail, axes, ins, body, count, rank, true, bind};
}

std::pair<const Def*, const Def*> GridTilePlan::box(u64 i, const Def* off, const Def* ext) const {
    auto& w      = off->world();
    auto r       = Lit::as(ranks->proj(count, i));
    auto origin  = DefVec(r, w.lit_nat_0());
    auto extent  = shapes->proj(count, i)->projs(r);
    auto mapping = axes->proj(count, i);
    for (u64 d = 0; d != rank; ++d) {
        auto axis    = Lit::as(mapping->proj(rank, d));
        origin[axis] = off->proj(rank, d);
        extent[axis] = ext->proj(rank, d);
    }
    return {w.tuple(origin), w.tuple(extent)};
}

const Def* normalize_tiled_multi(const Def* type, const Def* c, const Def* body) {
    check_region_inputs(region_body(body).first, c);
    // A generic definition such as a builder is checked once it is applied to literal counts, ranks and axes.
    auto [meta, shapes, axes, input_type, inputs] = c->as<App>()->uncurry_args<5>();
    auto [n, g, types, ranks]                     = meta->projs<4>();
    auto count                                    = Lit::isa(n);
    auto grid                                     = Lit::isa(g);
    if (!count || !grid) return nullptr;
    for (u64 i = 0; i != *count; ++i) {
        if (!Lit::isa(ranks->proj(*count, i))) return nullptr;
        for (u64 d = 0; d != *grid; ++d)
            if (!Lit::isa(axes->proj(*count, i)->proj(*grid, d))) return nullptr;
    }
    auto& w  = c->world();
    auto app = w.raw_app(type, c, body);
    grid_tile_plan(app->as<App>());
    return nullptr;
}

const Def* normalize_tile_fold(const Def*, const Def* c, const Def* body) {
    auto [meta, blocks, init, input_type, inputs] = c->as<App>()->uncurry_args<5>();
    check_region_inputs(body, c);
    auto [n, ks] = blocks->projs<2>();
    if (Lit::isa(n) == 0) return init;
    if (Lit::isa(ks) == 0) c->blame("tensor.tile_fold requires a positive block size").bail();
    return nullptr;
}

const Def* normalize_tiled_n(const Def*, const Def* c, const Def* body) {
    check_region_inputs(body, c);
    return nullptr;
}

const Def* normalize_group_impl(const Def*, const Def* callee, const Def* arg) {
    auto& w                  = arg->world();
    auto [type, rank, shape] = callee->as<App>()->args<3>();
    auto [producer, tiles]   = arg->projs<2>();
    if (!graph::map_reduce(producer) || !Lit::isa(rank)) return nullptr;
    RegionInputs inputs(Defs{producer});
    auto values     = inputs.values(w);
    auto tiled      = w.app(w.app(w.annex<tensor::tiled>(), {type, rank}), {shape, tiles, w.lit_nat_1()});
    tiled           = w.app(w.app(tiled, values->type()), values);
    auto body       = w.mut_lam(tiled->type()->as<Pi>()->dom()->as<Pi>());
    auto inner      = w.mut_lam(body->type()->reduce(body->var())->as<Pi>());
    auto [off, ext] = body->vars<2>();
    auto r          = Lit::as(rank);
    DefVec origin(r, [&](size_t d) {
        auto s = Lit::isa(shape->proj(r, d)), t = Lit::isa(tiles->proj(r, d));
        return s && t && *s <= *t ? w.lit_nat_0() : off->proj(r, d);
    });
    auto read = w.app(w.app(w.annex<tile_of>(), {type, rank}), shape);
    inner->set(true, w.app(read, {inputs.rebuild(producer, inner->var()), w.tuple(origin), ext}));
    body->set(true, inner);
    return w.app(tiled, body);
}

const Def* returned(Lam* lam) {
    if (auto value = Lam::isa_ret_arg(lam)) return value;
    auto call = lam->body()->isa<App>();
    auto next = call ? call->callee()->isa_mut<Lam>() : nullptr;
    if (!next || !next->is_set()) return nullptr;
    auto app = next->reduce(call->arg()).back()->isa<App>();
    return app && app->callee() == lam->ret_var() ? app->arg() : nullptr;
}

namespace {

// Whether @p map sends the loop index `(i0, i1, i2)` exactly to `(i_a, i_b)`.
// Whether `map` reads loops `a` and `b` of three, each as is up to an offset free of the loops.
bool reads_loops(const Def* map, u64 a, u64 b) {
    auto& w = map->world();
    auto pi = map->type()->isa<Pi>();
    if (!pi) return false;
    auto var = w.mut_lam(pi)->var();
    auto out = w.app(map, var);
    if (Lit::isa(out->arity()) != 2) return false;
    auto reads = [&](const Def* index, u64 loop) {
        auto form = Linear::of(index);
        for (u64 l = 0; l != 3; ++l) {
            auto atom = form.atoms.find(var->proj(3, l));
            if ((atom == form.atoms.end() ? 0 : atom->second) != (l == loop ? 1 : 0)) return false;
        }
        return true;
    };
    return reads(out->proj(2, 0), a) && reads(out->proj(2, 1), b);
}

// Whether `x` and `y` are `a` and `b` in either order.
bool either(const Def* x, const Def* y, const Def* a, const Def* b) { return (x == a && y == b) || (x == b && y == a); }

// `C (m, n) += A (m, k) · B (k, n)` with B also read transposed and an F32 accumulator, whose combiner returns
// `acc + a · b` over F32 inputs or `acc + up a · up b` over F16 or BF16 ones; the inputs' type, or nullptr.
const Def* contraction_input(const App* root) {
    auto& w    = root->world();
    auto mr    = btensor::MapRedPost::of(root->callee()->as<App>());
    auto ranks = mr.ranks();
    auto F32   = w.annex<math::F32>();
    if (!ranks || ranks->ro != 2 || ranks->rr != 1 || Lit::isa(mr.nis) != 2) return nullptr;
    auto T      = mr.Tis->proj(2, 0);
    auto narrow = math::isa_bf16(T) || math::isa_f(T) == 16;
    if (mr.To != F32 || mr.Tis->proj(2, 1) != T || (T != F32 && !narrow)) return nullptr;
    if (!reads_loops(mr.map_out, 0, 1) || !reads_loops(mr.maps->proj(2, 0), 0, 2)
        || !(reads_loops(mr.maps->proj(2, 1), 2, 1) || reads_loops(mr.maps->proj(2, 1), 1, 2)))
        return nullptr;
    auto lam   = mr.comb->isa_mut<Lam>();
    auto value = lam && lam->is_set() ? returned(lam) : nullptr;
    auto add   = value ? Axm::isa(math::arith::add, value) : nullptr;
    if (!add) return nullptr;
    auto [acc, ins] = lam->var(2, 0)->projs<2>();
    auto [l, r]     = add->args<2>();
    auto product    = l == acc ? r : r == acc ? l : nullptr;
    auto mul        = product ? Axm::isa(math::arith::mul, product) : nullptr;
    if (!mul) return nullptr;
    auto [x, y] = mul->args<2>();
    if (narrow) {
        auto up_x = Axm::isa(math::conv::f2f, x), up_y = Axm::isa(math::conv::f2f, y);
        if (!up_x || !up_y) return nullptr;
        x = up_x->arg(), y = up_y->arg();
    }
    return either(x, y, ins->proj(2, 0), ins->proj(2, 1)) ? T : nullptr;
}

bool contraction(const App* root) { return contraction_input(root) != nullptr; }

} // namespace

bool is_contraction(const App* root) { return contraction_input(root) == root->world().annex<math::F32>(); }

const Def* contraction_type(const App* root) { return contraction_input(root); }

const Def* normalize_root_contraction(const Def*, const Def*, const Def* op) {
    if (auto root = Axm::isa<map_reduce_post>(Axm::peel<buf>(op))) return op->world().lit_bool(contraction(root));
    return nullptr;
}

const Def* normalize_root_loops(const Def*, const Def*, const Def* op) {
    if (auto root = Axm::isa<map_reduce_post>(Axm::peel<buf>(op)))
        if (auto ranks = btensor::MapRedPost::of(root->callee()->as<App>()).ranks())
            return op->world().lit_nat(ranks->nloops());
    return nullptr;
}

const Def* normalize_root_acc(const Def*, const Def*, const Def* op) {
    if (auto root = Axm::isa<map_reduce_post>(Axm::peel<buf>(op)))
        return btensor::MapRedPost::of(root->callee()->as<App>()).To;
    return nullptr;
}

const Def* normalize_stage(const Def*, const Def* c, const Def*) {
    auto [meta, shape, layout, scope] = c->as<App>()->uncurry_args<4>();
    auto [type, r, k]                 = meta->projs<3>();
    auto rank = Lit::isa(r), count = Lit::isa(k);
    if (!rank || !count) return nullptr;
    auto [pos, tiles, perm, pad] = layout->projs<4>();
    fe::Vector<bool> seen(*rank, false);
    for (size_t j = 0; j != *count; ++j) {
        auto axis = Lit::isa(pos->proj(*count, j));
        if (Lit::isa(tiles->proj(*count, j)) == 0) c->blame("tensor.stage requires positive inner tile sizes").bail();
        if (axis) {
            if (*axis >= *rank || seen[*axis])
                c->blame("tensor.stage inner dimensions must be distinct and in range").bail();
            seen[*axis] = true;
        }
    }
    std::ranges::fill(seen, false);
    for (size_t d = 0; d != *rank; ++d)
        if (auto axis = Lit::isa(perm->proj(*rank, d))) {
            if (*axis >= *rank || seen[*axis]) c->blame("tensor.stage outer permutation must be a bijection").bail();
            seen[*axis] = true;
        }
    return nullptr;
}

// @p tensor read at @p index through any slices, as its base tensor and the index there.
std::pair<const Def*, const Def*> through_slices(const Def* tensor, const Def* index) {
    auto& w = tensor->world();
    for (;;) {
        auto source = Axm::peel<tensor::buf>(tensor);
        if (auto slice = Axm::isa<tensor::slice>(source)) {
            auto [meta, s_in, box]     = slice->callee()->as<App>()->uncurry_args<3>();
            auto [start, step, extent] = box->projs<3>();
            auto rank                  = Lit::isa<u64>(meta->proj(2, 1));
            if (!rank) break;
            index  = w.tuple(DefVec(*rank, [&](size_t d) {
                auto offset
                    = w.call(core::nat::mul, Defs{step->proj(*rank, d), w.call<core::idx2nat>(index->proj(*rank, d))});
                auto x = w.call(core::nat::add, Defs{start->proj(*rank, d), offset});
                return w.call<core::idx>(w.lit_nat_0(), s_in->proj(*rank, d), x);
            }));
            tensor = slice->arg();
        } else if (auto reshape = Axm::isa<tensor::reshape>(source)) {
            // A reshape that only adds or drops unit dims reads its input at the same coordinates.
            auto [meta, s_in, s_out] = reshape->callee()->as<App>()->uncurry_args<3>();
            auto r_in = Lit::isa<u64>(meta->proj(3, 1)), r_out = Lit::isa<u64>(meta->proj(3, 2));
            if (!r_in || !r_out) break;
            auto unit = [](const Def* d) { return Lit::isa(d) == 1; };
            auto ins = DefVec(s_in->projs(*r_in)), outs = DefVec(s_out->projs(*r_out));
            auto coord = DefVec(*r_in);
            size_t o   = 0;
            bool plain = true;
            for (size_t i = 0; i != *r_in && plain; ++i) {
                while (o != *r_out && unit(outs[o]) && !unit(ins[i]))
                    ++o;
                if (unit(ins[i])) {
                    coord[i] = w.lit_idx(1, 0);
                    if (o != *r_out && unit(outs[o])) ++o;
                } else if (o != *r_out && outs[o] == ins[i]) {
                    coord[i] = index->proj(*r_out, o++);
                } else {
                    plain = false;
                }
            }
            for (; o != *r_out && plain; ++o)
                plain = unit(outs[o]);
            if (!plain) break;
            index  = w.tuple(coord);
            tensor = reshape->arg();
        } else {
            break;
        }
    }
    return {tensor, index};
}

// A pointwise map_reduce_post read at one coordinate is evaluated there: each input is read where its map sends it.
const Def* read_pointwise(const Def* tensor, const Def* index) {
    auto app = isa_pointwise(Axm::peel<tensor::buf>(tensor));
    if (!app) return nullptr;
    auto mr     = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks  = mr.ranks();
    auto& w     = app->world();
    auto rn     = ranks->ro;
    auto loops  = w.tuple(DefVec(rn, [&](size_t d) {
        return w.call<core::idx>(w.lit_nat_0(), mr.Sr->proj(rn, d), w.call<core::idx2nat>(index->proj(rn, d)));
    }));
    auto inputs = app->arg()->proj(2, 0);
    auto none   = w.bot(w.call<mem::M>(0));
    auto values = DefVec(ranks->nis, [&](size_t i) {
        auto coords = btensor::op_affine_map(mr.maps->proj(ranks->nis, i), mr.Ris->proj(ranks->nis, i), mr.Rn, mr.Sr,
                                             mr.Sis->proj(ranks->nis, i), loops, none);
        auto [source, at] = through_slices(inputs->proj(ranks->nis, i), coords->proj(2, 1));
        return op_get(source, at);
    });
    auto call   = [&](const Def* f, const Def* x) { return w.app(cps::op_cps2ds_dep(f), x); };
    return call(mr.post, w.tuple({call(mr.comb, w.tuple({mr.init, w.tuple(values)})), w.tuple()}));
}

// A fused read evaluates the staged pointwise producer, but not the producers of its inputs, which the region does not
// own, and reads through slices; it stays while its tensor is unknown and otherwise becomes a plain `get`.
const Def* normalize_get_fused(const Def*, const Def*, const Def* arg) {
    auto [tensor, index] = through_slices(arg->proj(2, 0), arg->proj(2, 1));
    if (auto value = read_pointwise(tensor, index)) return value;
    auto unknown = Axm::peel<tensor::buf>(tensor);
    while (auto extract = unknown->isa<Extract>())
        unknown = extract->tuple();
    if (unknown->isa<Var>() || unknown->isa_mut())
        return tensor == arg->proj(2, 0) ? nullptr : op_get_fused(tensor, index);
    return op_get(tensor, index);
}

const Def* normalize_get(const Def*, const Def* callee, const Def* arg) {
    auto [tensor, index] = arg->projs<2>();
    if (auto array = Axm::isa<tensor::from_array>(tensor)) return arg->world().extract(array->arg(), index);
    if (auto scalar = Axm::isa<tensor::scalar>(tensor)) return scalar->arg();
    if (auto set = Axm::isa<tensor::set>(tensor)) {
        auto [base, written, value] = set->args<3>();
        if (written == index) return value;
        if (auto rank = Lit::isa<u64>(callee->as<App>()->arg(3, 0))) {
            auto a = lit_projs(written, *rank);
            auto b = lit_projs(index, *rank);
            if (a && b && *a != *b) return op_get(base, index);
        }
    }
    return read_through(tensor, index);
}

const Def* normalize_set(const Def*, const Def* callee, const Def* arg) {
    auto [tensor, index, value] = arg->projs<3>();
    if (auto array = Axm::isa<tensor::from_array>(tensor)) {
        auto data = array->arg();
        auto rank = Lit::isa<u64>(callee->as<App>()->arg(3, 0));
        if (data->is_closed() && rank && lit_projs(index, *rank) && value->isa<Lit>()) {
            auto& w = arg->world();
            auto op = w.app(w.annex<tensor::from_array>(), callee->as<App>()->arg());
            return w.app(op, w.insert(data, index, value));
        }
    }
    return nullptr;
}

const Def* normalize_scalar(const Def* type, const Def*, const Def* arg) {
    if (auto item = Axm::isa<tensor::item>(arg))
        if (item->arg()->type() == type) return item->arg();
    return nullptr;
}

const Def* normalize_item(const Def*, const Def*, const Def* arg) {
    if (auto scalar = Axm::isa<tensor::scalar>(arg)) return scalar->arg();
    if (auto array = Axm::isa<tensor::from_array>(arg)) return array->arg();
    if (auto set = Axm::isa<tensor::set>(arg)) return set->arg()->proj(3, 2);
    return nullptr;
}

const Def* normalize_broadcast(const Def* type, const Def* c, const Def* arg) {
    auto& w = c->world();

    auto [s_in, s_out, input] = arg->projs<3>();
    auto callee               = c->as<App>();
    auto [T, r]               = callee->args<2>();
    w.log().d("broadcast: input = {}: {}, T = {}, r = {}, s_in = {}, s_out = {}", input, input->type(), T, r, s_in,
              s_out);

    if (s_in == s_out || input->type() == type) return input;

    return nullptr;
}

const Def* normalize_broadcast_in_dim(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_repeat(const Def* type, const Def* c, const Def* arg) {
    // Identity repeat: if the input and output shapes agree, the repeat is a no-op.
    auto [Tr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_reshape(const Def* type, const Def* c, const Def* arg) {
    // Identity reshape: if the input and output shapes agree, the reshape is a no-op.
    auto [Trr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_transpose(const Def*, const Def* c, const Def* arg) {
    auto callee = c->as<App>()->decurry();
    auto perm   = lit_perm(callee->arg());
    if (!perm) return nullptr;
    auto r = perm->size();
    if (std::ranges::equal(*perm, std::views::iota(u64(0), u64(r)))) return arg;

    auto app = Axm::isa<tensor::transpose>(arg);
    if (!app) return nullptr;
    auto inner = transpose_perm(app);
    if (!inner || inner->size() != r) return nullptr;

    // Axis `j` of the inner input ends up at `perm#(inner#j)`; an identity composite collapses in the rebuilt app.
    auto& w   = c->world();
    auto comp = DefVec(r, [&](size_t j) { return w.lit_idx(r, (*perm)[(*inner)[j]]); });
    return w.app(w.app(w.app(callee->callee(), w.tuple(comp)), app->decurry()->arg()), app->arg());
}

namespace {

struct UnitView {
    const Def *source, *shape, *start;
};

/// @p def as a unit-step `slice` or a `slice_box` of rank @p r, so a view of it is a view of its source.
std::optional<UnitView> unit_view(const Def* def, u64 r) {
    if (auto view = Axm::isa<slice_box>(def)) {
        auto [_, shape, box] = view->callee()->as<App>()->uncurry_args<3>();
        return UnitView{view->arg(), shape, box->proj(3, 0)};
    }
    if (auto view = Axm::isa<slice>(def)) {
        auto [_, shape, box] = view->callee()->as<App>()->uncurry_args<3>();
        auto step            = box->proj(3, 1);
        for (u64 d = 0; d != r; ++d)
            if (Lit::isa(step->proj(r, d)) != 1) return {};
        return UnitView{view->arg(), shape, box->proj(3, 0)};
    }
    return {};
}

const Def* add_starts(const Def* a, const Def* b, u64 r) {
    auto& w = a->world();
    return w.tuple(DefVec(r, [&](size_t d) { return w.call(core::nat::add, Defs{a->proj(r, d), b->proj(r, d)}); }));
}

} // namespace

const Def* normalize_slice_box(const Def*, const Def* c, const Def* arg) {
    auto [meta, shape, box] = c->as<App>()->uncurry_args<3>();
    auto r                  = Lit::isa(meta->proj(2, 1));
    if (!r) return nullptr;
    if (auto inner = unit_view(arg, *r)) {
        auto& w              = arg->world();
        auto [off, ext, cap] = box->projs<3>();
        auto view            = w.app(w.app(w.annex<slice_box>(), meta), inner->shape);
        return w.app(w.app(view, {add_starts(inner->start, off, *r), ext, cap}), inner->source);
    }
    return nullptr;
}

const Def* normalize_slice(const Def*, const Def* c, const Def* arg) {
    auto [Tr, s_in, params]   = c->as<App>()->uncurry_args<3>();
    auto [start, step, s_out] = params->projs<3>();
    if (auto r = Lit::isa(Tr->proj(2, 1)))
        if (auto inner = unit_view(arg, *r)) {
            auto& w   = arg->world();
            auto view = w.app(w.app(w.annex<slice>(), Tr), inner->shape);
            return w.app(w.app(view, {add_starts(inner->start, start, *r), step, s_out}), inner->source);
        }
    // Identity slice: every axis starts at 0 with step 1 and keeps its full extent (s_out == s_in) -> the input itself.
    if (s_out != s_in) return nullptr;
    auto r = Lit::isa<u64>(Tr->proj(2, 1));
    if (!r) return nullptr;
    for (u64 d = 0; d != *r; ++d) {
        auto st = Lit::isa<u64>(start->proj(*r, d));
        auto sp = Lit::isa<u64>(step->proj(*r, d));
        if (!st || *st != 0 || !sp || *sp != 1) return nullptr;
    }
    return arg;
}

const Def* normalize_flip(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_pad(const Def*, const Def* c, const Def* arg) {
    // Identity pad: every axis has lo == hi == 0 (so s_out == s_in) -> the input itself (the fill value is irrelevant).
    auto [Tr, s_in, params] = c->as<App>()->uncurry_args<3>();
    auto [mode, lo, hi]     = params->projs<3>();
    auto r                  = Lit::isa<u64>(Tr->proj(2, 1));
    if (!r) return nullptr;
    for (u64 d = 0; d != *r; ++d) {
        auto l = Lit::isa<u64>(lo->proj(*r, d));
        auto h = Lit::isa<u64>(hi->proj(*r, d));
        if (!l || *l != 0 || !h || *h != 0) return nullptr;
    }
    return arg->proj(2, 0); // input (arg = (input, value))
}

const Def* normalize_concat(const Def*, const Def*, const Def*) { return nullptr; }

const Def* normalize_if_static(const Def*, const Def*, const Def* arg) {
    // `tensor.if_static (k, s, d)` picks `s` once `k` has folded to a literal; a still-symbolic `k`
    // keeps the App stuck, and the tensor lowerings residualize it to `d` (by lowering time,
    // undecided means runtime).
    auto [k, s, d] = arg->projs<3>();
    if (Lit::isa(k)) return s;
    return nullptr;
}

const Def* normalize_fastest_axis(const Def*, const Def*, const Def* arg) {
    // `tensor.fastest_axis (r, t)` reflects which axis of `t` is the fastest-varying (unit-stride)
    // axis of the tensor actually read once `fuse_tensor`'s read-through has absorbed a pure
    // re-indexed read behind `t`: without one, `t`'s own last axis; behind one, found by evaluating
    // the read's access map on distinct `affine.lit` markers — normalization folds the map's
    // extracts over the marker tuple, and a last component that does not fold back to a marker
    // (reshape arithmetic) stays unknown. Unknown answers the sentinel `r`.
    auto& w     = arg->world();
    auto [r, t] = arg->projs<2>();
    auto r_l    = Lit::isa<u64>(r);
    if (!r_l || *r_l == 0) return r;
    auto pr = is_pure_read(t);
    if (!pr) return w.lit_nat(*r_l - 1);
    // One level only: a source that is itself absorbed would need the composed analysis.
    if (is_pure_read(pr->src) || Axm::isa<tensor::broadcast>(pr->src)) return r;
    auto r_src = Lit::isa<u64>(pr->map->type()->as<Pi>()->codom()->arity());
    if (!r_src || *r_src == 0) return r;
    // `r` is not type-coupled to `t`; a mismatched rank must answer unknown, not break the app below.
    auto r_dom = Lit::isa<u64>(pr->map->type()->as<Pi>()->dom()->arity());
    if (!r_dom || *r_dom != *r_l) return r;
    // Markers start at 1: 0 is what a broadcast map's `o#d · 0` folds to, so it must not be one.
    auto markers = DefVec(*r_l, [&](size_t i) { return w.call<affine::lit>(w.lit_nat(i + 1)); });
    auto last    = w.app(pr->map, w.tuple(markers))->proj(*r_src, *r_src - 1);
    auto c       = Axm::isa<affine::lit>(last);
    if (!c) return r;
    auto v = Lit::isa<u64>(c->arg());
    if (!v || *v < 1 || *v > *r_l) return r;
    return w.lit_nat(*v - 1);
}

/// The vector widths `-X tensor:vec-width` accepts.
constexpr std::array Vec_widths = {1_u64, 4_u64, 8_u64, 16_u64};

const Def* normalize_vec_width(const Def*, const Def*, const Def* arg) {
    // Read off the Driver so one graph can be re-emitted for another target by another Driver.
    auto& w = arg->world();
    auto n  = w.driver().arg_nat("tensor", "vec-width");
    if (!n) return arg;
    if (std::ranges::find(Vec_widths, *n) == Vec_widths.end()) {
        w.log().w("ignoring `-X tensor:vec-width={}`: not one of {}", *n, fe::Join(Vec_widths, ", "));
        return arg;
    }
    return w.lit_nat(*n);
}

MIM_tensor_NORMALIZER_IMPL

    const App*
    isa_pointwise(const Def* def) {
    auto app = Axm::isa<tensor::map_reduce_post>(def);
    if (!app) return nullptr;
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks = mr.ranks();
    if (!ranks || ranks->rr != 0 || ranks->nps != 0 || mr.init_type != mr.To) return nullptr;
    auto& w   = app->world();
    auto loop = w.mut_lam(mr.map_out->type()->as<Pi>());
    auto out  = w.app(mr.map_out, loop->var());
    if (auto lam = mr.map_out->isa_mut<Lam>(); lam && lam->is_set()) out = lam->reduce_body(loop->var());
    return out == loop->var() ? app : nullptr;
}

} // namespace mim::plug::tensor
