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

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor {

namespace {

bool tensor_input_type(const Def* type) {
    if (type->isa<Arr>()) return true;
    if (auto sigma = type->isa<Sigma>()) return std::ranges::any_of(sigma->ops(), tensor_input_type);
    return false;
}

} // namespace

void check_region_inputs(const Def* function, const Def* region) {
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

ProducerTile producer_tile(const Def* producer, const Def* off, const Def* ext, const Def* box_cap) {
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
    auto shift = [&](const Def* map, const Def* n, const Def* m, const Def* lo, const Def* origin) {
        auto call = w.app(w.annex<tile_shift>(), {n, m});
        return w.app(w.app(call, map), {lo, origin});
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
        cap        = w.tuple(DefVec(rn, [&](size_t d) {
            auto bound = cap->proj(rn, d);
            return Lit::isa(bound) ? bound : source_loop_cap->proj(rn, d);
        }));
        output_cap = w.tuple(DefVec(ro, [&](size_t d) {
            if (auto requested = ext->proj(ro, d); Lit::isa(requested)) return requested;
            return source_output_cap->proj(ro, d);
        }));
    } else {
        cap        = size;
        output_cap = ext;
    }

    auto [values, post_values] = app->args<2>();
    auto shapes = DefVec(), maps = DefVec(), inputs = DefVec();
    auto post_shapes = DefVec(), post_maps = DefVec(), post_inputs = DefVec();
    auto boxes
        = [&](u64 count, const Def* ranks, const Def* source_shapes, const Def* source_maps, const Def* source_types,
              const Def* source_values, const Def* n, const Def* origin, const Def* domain, const Def* domain_cap,
              DefVec& result_shapes, DefVec& result_maps, DefVec& result_values) -> bool {
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
            result_maps.push_back(shift(map, n, rank, origin, start));
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
    if (!boxes(rks->nis, mr.Ris, mr.Sis, mr.maps, mr.Tis, values, mr.Rn, lo, size, cap, shapes, maps, inputs))
        return {nullptr, "input map does not beta-reduce or has symbolic rank"};
    if (!boxes(rks->nps, mr.Rps, mr.Sps, mr.post_maps, mr.Tps, post_values, mr.Ro, off, ext, output_cap, post_shapes,
               post_maps, post_inputs))
        return {nullptr, "post map does not beta-reduce or has symbolic rank"};
    auto init = mr.init;
    if (auto outs = tensor_init(mr)) {
        auto cells = w.tuple(DefVec(rn, [&](size_t d) { return d < ro ? size->proj(rn, d) : w.lit_nat_1(); }));
        auto box   = image(outs->map, mr.Rn, mr.Ro, lo, cells, outs->shape);
        if (Axm::isa<affine::image>(box)) return {nullptr, "outs map does not beta-reduce"};
        auto start = box->proj(4, 0), extent = box->proj(4, 1);
        auto value = slice(mr.To, mr.Ro, outs->shape, outs->outs, start, extent);
        init       = w.tuple({extent, shift(outs->map, mr.Rn, mr.Ro, lo, start), value});
    }
    auto ins   = w.tuple({mr.Tis, mr.Ris, w.tuple(shapes), mr.Tps, mr.Rps, w.tuple(post_shapes)});
    auto write = shift(mr.map_out, mr.Rn, mr.Ro, lo, off);
    auto sched = schedule;
    if (cap != size || output_cap != ext) {
        auto parallel = w.tuple(DefVec(ro, [&](size_t d) { return size->proj(rn, d); }));
        sched = w.app(w.app(w.annex<btensor::clip_sched>(), {mr.Ro, mr.Rn}), {parallel, output_cap, cap, sched});
    }
    const Def* result = w.annex<map_reduce_post>();
    for (auto arg :
         {mr.nis_nps, mr.meta, w.tuple({ext, size, sched}), ins, init->type(), w.tuple({mr.comb, init, mr.post}), write,
          w.tuple({w.tuple(maps), w.tuple(post_maps)}), w.tuple({w.tuple(inputs), w.tuple(post_inputs)})})
        result = w.app(result, arg);
    return {result, {}};
}

const Def* normalize_tile_of(const Def*, const Def*, const Def* arg) {
    auto [producer, off, ext] = arg->projs<3>();
    return producer_tile(producer, off, ext).value;
}

const Def* normalize_tile_of_box(const Def*, const Def*, const Def* arg) {
    auto [producer, off, ext, cap] = arg->projs<4>();
    return producer_tile(producer, off, ext, cap).value;
}

const Def* normalize_tiled(const Def*, const Def* c, const Def* body) {
    auto [meta, shapes, input_type, inputs] = c->as<App>()->uncurry_args<4>();
    check_region_inputs(body, c);
    auto [s, ts] = shapes->projs<2>();
    if (auto rank = Lit::isa(meta->proj(2, 1))) {
        for (size_t d = 0; d != *rank; ++d)
            if (Lit::isa(ts->proj(*rank, d)) == 0) c->blame("tensor.tiled requires positive tile sizes").bail();
        for (size_t d = 0; d != *rank; ++d)
            if (Lit::isa(s->proj(*rank, d)) == 0) {
                auto& w = c->world();
                return w.app(w.app(w.annex<tensor::splat>(), meta), {s, w.bot(meta->proj(2, 0))});
            }
        if (s == ts)
            return c->world().app(c->world().app(body, {c->world().pack(*rank, c->world().lit_nat(0)), s}), inputs);
    }
    return nullptr;
}

GridTilePlan grid_tile_plan(const App* app) {
    if (Axm::isa<tiled>(app)) {
        auto [meta, shapes, it, ins, body] = app->uncurry_args<5>();
        auto [type, rank]                  = meta->projs<2>();
        auto [shape, tiles]                = shapes->projs<2>();
        auto& w                            = app->world();
        auto r                             = Lit::expect(rank, "a literal tensor.tiled grid rank");
        auto axes                          = w.tuple(DefVec(r, [&](size_t d) { return w.lit_idx(r, d); }));
        return {type, rank, shape, shape, tiles, axes, ins, body, 1, r, false};
    }
    auto [meta, shapes_grid, axes, it, ins, body] = app->uncurry_args<6>();
    auto [n, g, types, ranks]                     = meta->projs<4>();
    auto [shapes, shape, tiles]                   = shapes_grid->projs<3>();
    auto count                                    = Lit::expect(n, "a literal tensor.tiled_multi result count");
    auto rank                                     = Lit::expect(g, "a literal tensor.tiled_multi grid rank");
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
    return {types, ranks, shapes, shape, tiles, axes, ins, body, count, rank, true};
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
    check_region_inputs(body, c);
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
    auto tiled      = w.app(w.app(w.annex<tensor::tiled>(), {type, rank}), {shape, tiles});
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
    // Size-1 axes collapse out of the type, so equal types suffice.
    auto [Tr, s_in, s_out] = c->as<App>()->uncurry_args<3>();
    if (s_in == s_out || arg->type() == type) return arg;
    return nullptr;
}

const Def* normalize_reshape(const Def* type, const Def* c, const Def* arg) {
    // Identity reshape: if the input and output shapes agree, the reshape is a no-op.
    // Size-1 axes collapse out of the type, so equal types suffice.
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

const Def* normalize_slice(const Def*, const Def* c, const Def* arg) {
    // Identity slice: every axis starts at 0 with step 1 and keeps its full extent (s_out == s_in) -> the input itself.
    auto [Tr, s_in, params]   = c->as<App>()->uncurry_args<3>();
    auto [start, step, s_out] = params->projs<3>();
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

const Def* normalize_shape(const Def*, const Def* c, const Def* arg) {
    // `tensor.shape r arr` is the first `r` axes of `arr`'s (fused) array type.
    auto& w = c->world();
    auto r  = Lit::isa<u64>(c->as<App>()->arg()); // the explicit rank `r`
    if (!r) return nullptr;
    if (*r == 0) return w.tuple();

    auto arr = arg->type()->isa<Arr>();
    if (!arr) return nullptr;
    auto shape = arr->shape();
    auto rank  = shape.rank();
    if (!rank || *rank < *r) return nullptr; // `arr` is not (statically) at least rank `r`
    return *shape.take(*r);
}

MIM_tensor_NORMALIZER_IMPL

} // namespace mim::plug::tensor
