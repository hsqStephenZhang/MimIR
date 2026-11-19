#include "mim/plug/tensor/phase/lower_to_mem.h"

#include <set>

#include "lower_to_mem_util.h"

namespace mim::plug::tensor::phase {

using namespace lower_to_mem_detail;

const Def* LowerToMem::lower_stage(const App* app) {
    auto& old                         = old_world();
    auto [meta, shape, layout, scope] = app->callee()->as<App>()->uncurry_args<4>();
    auto [type, rank, split]          = meta->projs<3>();
    auto input                        = app->arg();
    auto owned                        = input->isa<App>() && wants_fresh_mem(input->as<App>());
    if (auto extract = input->isa<Extract>())
        owned |= bool(Axm::isa<tensor::tiled_n>(extract->tuple()) || Axm::isa<tensor::tiled_multi>(extract->tuple()));
    // A bound region fills a slice cooperatively; a box with runtime extents has no buffer type to alias anyway.
    auto view = Axm::isa<tensor::slice>(input) || Axm::isa<tensor::slice_box>(input);
    if (view && storage_level_ == 0) {
        auto [vr, vs, vT] = TensorType::decode(app->type()->as<Nom>());
        view = !std::ranges::all_of(vs->projs(Lit::as(vr)), [](const Def* e) { return bool(Lit::isa(e)); });
    }
    if ((scratch_ || owned) && !view && Lit::isa(split) == 0) {
        auto r        = Lit::isa(rank);
        auto identity = r.has_value();
        if (r)
            for (size_t d = 0; d != *r; ++d)
                identity &= Lit::isa(layout->proj(4, 2)->proj(*r, d)) == d;
        if (identity) {
            auto source = rewrite(app->arg());
            if (Axm::isa<buffer::Buf>(source->type()) && Check::alpha(source->type(), buf_of(app->type()))) {
                if (auto cap = fe::lookup(storage_shapes_, app->arg())) storage_shapes_[app] = cap;
                return source;
            }
        }
    }
    const Def* impl;
    // The panel dim that steps unscaled along the source's contiguous one: the fill runs it fastest, so a
    // work-group reads contiguously, and moves a vector per step where the run is provably aligned.
    auto inner                   = std::optional<size_t>();
    u64 vec                      = 1;
    auto [pos, tiles, perm, pad] = layout->projs<4>();
    auto panel                   = [&](std::optional<std::pair<size_t, u64>> contiguous) {
        auto r = Lit::isa(rank), k = Lit::isa(split);
        if (!r || !k || !contiguous) return;
        for (size_t d = 0; d != *r; ++d)
            if (Lit::isa(perm->proj(*r, d)) == contiguous->first) inner = d;
        for (size_t j = 0; j != *k; ++j)
            if (Lit::isa(pos->proj(*k, j)) == contiguous->first) inner = *r + j;
    };
    if (auto slice = Axm::isa<tensor::slice>(Axm::peel<tensor::buf>(app->arg()))) {
        auto [slice_meta, source_shape, box] = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, step, extent]           = box->projs<3>();
        auto source = slice->arg(), physical_shape = source_shape, physical_rank = rank;
        const Def* read = old.app(old.annex<affine::id>(), rank);
        while (auto view = panel::tensor_read(source, physical_shape)) {
            if (!TensorType::isa(view->src->type())) break;
            read           = panel::compose_read_map(view->map, read);
            source         = view->src;
            physical_shape = view->S;
            physical_rank  = view->R;
        }
        auto contiguous = panel::contiguous_dim(read, rank, physical_rank);
        panel(contiguous);
        auto r = Lit::as(rank), ri = Lit::as(physical_rank);
        auto [pr, s_out, pT] = TensorType::decode(app->type()->as<Nom>());
        auto rk              = s_out->num_projs();
        if (inner && storage_level_ != 0) {
            auto c   = contiguous->first;
            auto row = physical_shape->proj(ri, ri - 1);
            // A padded source's rows are as far apart as its storage's.
            if (auto storage = fe::lookup(storage_shapes_, source)) row = storage->proj(ri, ri - 1);
            vec = vector_width(type, extent->proj(r, c), start->proj(r, c), step->proj(r, c), row, contiguous->second);
            if (!nat_multiple(s_out->proj(rk, *inner), vec)) vec = 1;
        }
        auto element = vec == 1 ? type : old.arr(vec, type);
        auto getter  = old.app(old.annex<tensor::get>(), {physical_rank, physical_shape, type});
        if (vec != 1)
            getter = old.app(old.app(old.annex<tensor::get_vector>(), {physical_rank, physical_shape, type}),
                             old.lit_nat(vec));
        auto scale = old.tuple(DefVec(rk, [&](size_t d) { return old.lit_nat(d == inner ? vec : 1); }));
        if (vec != 1) {
            auto dims    = DefVec(s_out->projs(rk));
            dims[*inner] = old.call(core::nat::div, Defs{dims[*inner], old.lit_nat(vec)});
            s_out        = old.tuple(dims);
            pad          = old.pack(vec, pad);
        }
        // Tiles that divide their extents need no bounds check, so a fill moves each element unconditionally.
        auto k     = Lit::isa(split);
        auto exact = k.has_value();
        for (size_t j = 0; exact && j != *k; ++j) {
            auto d    = Lit::isa(pos->proj(*k, j));
            auto tile = Lit::isa(tiles->proj(*k, j));
            auto size = d ? Lit::isa(extent->proj(r, *d)) : std::nullopt;
            exact     = tile && size && *tile != 0 && *size % *tile == 0;
        }
        impl = exact ? old.annex<tensor::stage_read_exact_impl>() : old.annex<tensor::stage_read_impl>();
        impl = old.app(impl, {type, element, rank, split, physical_rank});
        impl = old.app(impl, {source_shape, extent, physical_shape});
        impl = old.app(impl, read);
        impl = old.app(impl, {pos, tiles, perm, pad});
        impl = old.app(impl, {start, step});
        impl = old.app(impl, {s_out, scale, getter});
        impl = old.app(impl, scope);
        impl = old.app(impl, source);
    } else {
        if (auto r = Lit::isa(rank); r && *r) panel(std::pair{*r - 1, u64(0)});
        impl      = old.annex<tensor::stage_impl>();
        auto args = DefVec();
        for (auto cur = app; cur; cur = cur->callee()->isa<App>())
            args.push_back(cur->arg());
        for (auto arg : args | std::views::reverse)
            impl = old.app(impl, arg);
    }
    auto generate = Axm::isa<tensor::generate>(impl);
    if (!generate) app->blame("tensor.stage reference reindexing did not produce a generate").bail();
    if (inner) fill_inner_[generate] = *inner;
    if (vec != 1) fill_vector_[generate] = vec;
    if (auto chosen = fe::lookup(placed_layouts_, app)) fill_layouts_[generate] = chosen;

    auto staged = lower_generate(generate);
    if (auto storage = fe::lookup(storage_shapes_, generate)) storage_shapes_[app] = storage;
    return staged;
}

bool LowerToMem::linear_step(const Def* root, const Def* accumulator, const Def* origin) {
    auto& w = old_world();
    for (size_t i = 0; i != 128; ++i) {
        auto app = root->isa<App>();
        if (!app) return false;
        auto lam = app->callee()->isa<Lam>();
        if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
        root = lam->reduce(app->arg()).back();
    }
    auto app = root->isa<App>();
    if (!app) return false;
    if (Axm::isa<tensor::mma>(app)) return linear_step(app->arg(), accumulator, origin);
    if (Axm::isa<tensor::tiled>(app)) {
        auto [meta, shapes, it, ins, body] = app->uncurry_args<5>();
        auto rank                          = Lit::as(meta->proj(2, 1));
        auto lam                           = region_body(app->arg()).first->isa<Lam>();
        if (!lam || !lam->is_set()) return false;
        auto param = w.mut_lam(w.pi(w.arr(rank, w.type_nat()), w.type_nat()));
        auto off   = param->var();
        auto next  = w.tuple(DefVec(
            rank, [&](size_t d) { return w.call(core::nat::add, Defs{origin->proj(rank, d), off->proj(rank, d)}); }));
        return linear_step(w.app(lam->reduce(w.tuple({off, shapes->proj(3, 1)})).back(), ins), accumulator, next);
    }
    if (!Axm::isa<tensor::map_reduce_post>(app)) return false;
    auto mr      = btensor::MapRedPost::of(app->callee()->as<App>());
    auto initial = tensor_init(mr);
    if (!initial) return false;
    while (auto read = panel::tensor_read(initial->outs, initial->shape)) {
        if (read->R != mr.Ro || !TensorType::isa(read->src->type())) break;
        initial = TensorInit{read->S, panel::compose_read_map(read->map, initial->map), read->src};
    }
    if (Axm::peel<tensor::buf>(initial->outs) != accumulator) return false;
    auto rank       = Lit::as(mr.Ro);
    auto loops      = Lit::as(mr.Rn);
    auto probe      = w.mut_lam(mr.map_out->type()->as<Pi>());
    auto output_map = mr.map_out->isa<Lam>();
    auto init_map   = initial->map->isa<Lam>();
    if (!output_map || !output_map->is_set() || !init_map || !init_map->is_set()) return false;
    auto output = output_map->reduce(probe->var()).back();
    auto read   = init_map->reduce(probe->var()).back();
    // Each step reads its accumulator element exactly where it writes it, and no two iterations share one.
    // A clipped step may iterate less than its storage; the elements it skips keep their accumulated values.
    for (size_t d = 0; d != rank; ++d) {
        auto written = Linear::of(output->proj(rank, d));
        for (size_t l = 0; l != loops; ++l) {
            auto atom = written.atoms.find(probe->var(loops, l));
            if ((atom == written.atoms.end() ? 0 : atom->second) != (l == d ? 1 : 0)) return false;
        }
        if (!nat_bounded_by(mr.Sr->proj(loops, d), mr.So->proj(rank, d))) return false;
        written.add(Linear::of(w.call<affine::lit>(origin->proj(rank, d))), 1);
        if (!(Linear::of(read->proj(rank, d)) == written)) return false;
    }
    auto pending = DefVec{app->arg(), mr.comb, mr.post, mr.map_out, mr.maps_all, initial->map};
    auto seen    = DefSet();
    for (size_t i = 0; i != pending.size(); ++i) {
        auto term = pending[i];
        if (term == accumulator) return false;
        if (!seen.insert(term).second || term->isa<Var>() || term->isa<Axm>()) continue;
        for (auto op : term->ops())
            if (op) pending.push_back(op);
    }
    return true;
}

namespace {
/// Whether @p def reads @p target, optionally only before reaching the results of other requests.
bool mentions(const Def* def, const Def* target, bool through_requests = true) {
    auto pending = DefVec{def};
    auto seen    = DefSet();
    for (size_t i = 0; i != pending.size(); ++i) {
        auto term = pending[i];
        if (term == target) return true;
        if (!seen.insert(term).second || term->isa<Var>() || term->isa<Axm>()) continue;
        if (!through_requests && Axm::isa<tensor::tiled>(term)) continue;
        for (auto op : term->ops())
            if (op) pending.push_back(op);
    }
    return false;
}
} // namespace

/// Whether @p root reads @p accumulator only where it writes its own element, so it may overwrite it in place.
bool LowerToMem::pointwise_step(const Def* root, const Def* accumulator, const Def* origin) {
    auto& w = old_world();
    for (size_t i = 0; i != 128; ++i) {
        auto app = root->isa<App>();
        if (!app) return false;
        auto lam = app->callee()->isa<Lam>();
        if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
        root = lam->reduce(app->arg()).back();
    }
    auto app = root->isa<App>();
    if (!app) return false;
    // A step that reads the accumulator only through requests finished before it never reads what it overwrites.
    if (std::ranges::none_of(app->ops(), [&](const Def* op) { return mentions(op, accumulator, false); })) return true;
    if (Axm::isa<tensor::tiled>(app)) {
        auto [meta, shapes, it, ins, body] = app->uncurry_args<5>();
        auto rank                          = Lit::as(meta->proj(2, 1));
        auto lam                           = region_body(app->arg()).first->isa<Lam>();
        if (!lam || !lam->is_set()) return false;
        auto param = w.mut_lam(w.pi(w.arr(rank, w.type_nat()), w.type_nat()));
        auto off   = param->var();
        auto next  = w.tuple(DefVec(
            rank, [&](size_t d) { return w.call(core::nat::add, Defs{origin->proj(rank, d), off->proj(rank, d)}); }));
        return pointwise_step(w.app(lam->reduce(w.tuple({off, shapes->proj(3, 1)})).back(), ins), accumulator, next);
    }
    if (!Axm::isa<tensor::map_reduce_post>(app)) return false;
    auto mr    = btensor::MapRedPost::of(app->callee()->as<App>());
    auto rank  = Lit::isa(mr.Ro);
    auto loops = Lit::isa(mr.Rn);
    auto nis   = Lit::isa(mr.nis);
    if (!rank || !nis || loops != rank) return false;
    auto output_map = mr.map_out->isa<Lam>();
    if (!output_map || !output_map->is_set()) return false;
    auto probe                 = w.mut_lam(mr.map_out->type()->as<Pi>());
    auto output                = output_map->reduce(probe->var()).back();
    auto [inputs, post_inputs] = app->arg()->projs<2>();
    auto others                = DefVec{mr.comb, mr.post, mr.init, post_inputs};
    for (size_t k = 0; k != *nis; ++k) {
        auto input  = inputs->proj(*nis, k);
        auto source = TensorInit{mr.Sis->proj(*nis, k), mr.maps->proj(*nis, k), input};
        while (auto read = panel::tensor_read(source.outs, source.shape)) {
            if (!TensorType::isa(read->src->type())) break;
            source = TensorInit{read->S, panel::compose_read_map(read->map, source.map), read->src};
        }
        if (Axm::peel<tensor::buf>(source.outs) != accumulator) {
            others.push_back(input);
            others.push_back(mr.maps->proj(*nis, k));
            continue;
        }
        auto map = source.map->isa<Lam>();
        if (!map || !map->is_set()) return false;
        auto read = map->reduce(probe->var()).back();
        for (size_t d = 0; d != *rank; ++d) {
            auto written = Linear::of(output->proj(*rank, d));
            for (size_t l = 0; l != *loops; ++l) {
                auto atom = written.atoms.find(probe->var(*loops, l));
                if ((atom == written.atoms.end() ? 0 : atom->second) != (l == d ? 1 : 0)) return false;
            }
            written.add(Linear::of(w.call<affine::lit>(origin->proj(*rank, d))), 1);
            if (!(Linear::of(read->proj(*rank, d)) == written)) return false;
        }
    }
    // Another request's result is complete before this one runs, so reading it reads no accumulator element.
    return std::ranges::none_of(others, [&](const Def* term) { return mentions(term, accumulator, false); });
}

// @p def with each region input it projects from replaced by the region's argument, as long as there is one.
bool LowerToMem::fills_with_tma(const Def* region) const {
    auto todo = DefVec{region};
    auto seen = DefSet{region};
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (Axm::isa<tensor::fill_tma>(def)) return true;
        if (def->isa<Axm>()) continue;
        for (auto op : def->deps())
            if (op && seen.insert(op).second) todo.push_back(op);
    }
    return false;
}

const Def* LowerToMem::outside(const Def* def) const {
    if (auto outer = fe::lookup(outer_of_, def)) return outside(outer);
    if (auto extract = def->isa<Extract>()) {
        auto tuple = outside(extract->tuple());
        if (tuple != extract->tuple()) return outside(def->world().extract(tuple, extract->index()));
    }
    return def;
}

// The work-group panels a fold step stages per block: the uniform `tensor.stage`s that depend on the step's @p var.
// A panel whose type depends on @p var changes shape between blocks, so its two buffers could not alternate.
DefVec LowerToMem::staged_panels(const Def* root, const Def* var) const {
    DefVec result;
    DefSet seen;
    auto todo = DefVec{root};
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (!seen.emplace(def).second || def->isa_mut() || !def->has_free_var(var->as<Var>())) continue;
        if (Axm::isa<tensor::placed>(def) || Axm::isa<tensor::stage>(def)) {
            if (uniform(def) && !def->type()->has_free_var(var->as<Var>())) result.push_back(def);
            continue;
        }
        for (auto op : def->ops() | std::views::reverse)
            if (op) todo.push_back(op);
    }
    return result;
}

const Def* LowerToMem::lower_tile_fold(const App* app, const Destination* destination) {
    auto scope = fe::Restore(uniform_, uniform(app));
    if (app->type()->isa_imm<Sigma>()) return destination ? nullptr : lower_product_fold(app);
    auto& w                                             = new_world();
    auto& old                                           = old_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto step                                           = app->arg()->isa<Lam>();
    if (!step || !step->is_set()) app->blame("tensor.tile_fold step must be a defined lambda").bail();
    auto parameter             = old.mut_lam(old.pi(old.sigma({init->type(), old.type_nat()}), old.type_nat()));
    auto [accumulator, offset] = parameter->var()->projs<2>();
    if (auto block = Lit::isa(stride)) multiples_[offset] = *block;
    auto input                                  = region_input(inputs);
    auto [stepped, level, merge, pipeline, tma] = peel_step(step->reduce(parameter->var()).back());
    auto root                                   = old.app(stepped, input);
    auto result                                 = root;
    for (size_t i = 0; i != 128; ++i) {
        auto call = result->isa<App>();
        if (!call) break;
        auto lam = call->callee()->isa<Lam>();
        if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
        result = lam->reduce(call->arg()).back();
    }
    auto tensor  = TensorType::isa(init->type());
    auto shape   = tensor ? tensor->s : old.tuple();
    auto element = tensor ? tensor->T : init->type();
    if (auto call = result->isa<App>()) {
        if (Axm::isa<tensor::tiled>(call)) shape = call->uncurry_args<5>()[1]->proj(3, 0);
        if (Axm::isa<tensor::map_reduce_post>(call)) shape = btensor::MapRedPost::of(call->callee()->as<App>()).So;
    }
    auto rank    = Lit::as(shape->arity());
    auto zero    = old.tuple(DefVec(rank, [&](size_t) { return old.lit_nat(0); }));
    auto inplace = linear_step(root, accumulator, zero);
    if (!inplace)
        log().w("tensor destination fallback: tile_fold step copies accumulator (linear pointwise use unproved)");

    auto literal_count = Lit::isa(count);
    // A separate first block would duplicate a work-group's panels; its work-items fill the seed instead.
    auto first_scalar = inplace && literal_count && *literal_count != 0 && splat_scalar(init) && storage_level_ == 0;
    // A cluster that splits the reduction gives each of its work-groups a contiguous `1 / size` of the blocks.
    const Def* base = nullptr;
    if (cluster_.split > 1 && tma && driver().is_loaded("gpu")) {
        if (!literal_count || *literal_count % cluster_.split || first_scalar)
            app->blame("a reduction split over a cluster of {} needs a literal multiple of {} blocks", cluster_.split,
                       cluster_.split)
                .n(unlowered)
                .bail();
        *literal_count /= cluster_.split;
        auto rank = w.call<core::idx2nat>(w.app(w.annex<gpu::cluster_rank>(), w.tuple()));
        base      = w.call(core::nat::mul, Defs{rank, w.lit_nat(*literal_count)});
    }
    auto shifted = [&](const Def* block) { return base ? w.call(core::nat::add, Defs{block, base}) : block; };
    auto extent  = rewrite(shape);
    auto storage = bounded_shape(extent, storage_level_ != 0);
    auto bounded = storage != extent;
    // Without the gpu plugin a bound reduction runs as the sequential fold, like bound parallel levels run as loops.
    auto bound_fold = level == 2 && driver().is_loaded("gpu");
    auto lane_fold  = level == 4 && driver().is_loaded("gpu");
    // Only a work-item bound reduction over a whole tile merges straight into a destination.
    if (destination && (!bound_fold || bounded)) return nullptr;
    if (bound_fold)
        return lower_bound_fold(app, parameter, root, result, merge, rank, shape, element, extent, storage, bounded,
                                inplace, destination);
    if (lane_fold)
        return lower_lane_fold(app, parameter, root, result, merge, rank, element, extent, storage, bounded, inplace);
    // A warp's tensor-core block keeps the accumulator as fragments in every lane's registers.
    auto mma       = driver().is_loaded("gpu") && storage_level_ != 0 ? Axm::isa<tensor::mma>(result) : nullptr;
    auto fragments = std::optional<Fragments>();
    if (mma) {
        auto extents = Lit::isa(extent->proj(2, 0)) && Lit::isa(extent->proj(2, 1));
        if (!inplace || rank != 2 || !extents)
            app->blame("a tensor-core fold needs a literal 2D accumulator updated in place").bail();
        auto shape = mma->decurry()->arg(2, 1);
        if (Lit::as(shape->proj(3, 0)) == 64)
            fragments = Fragments{16, Lit::as(extent->proj(2, 1)), 16, 8, true};
        else
            fragments = Fragments{Lit::as(extent->proj(2, 0)), Lit::as(extent->proj(2, 1)), Lit::as(shape->proj(3, 0)),
                                  Lit::as(shape->proj(3, 1))};
    }
    auto panels_scope = fe::Restore(mma_panels_, mma_panels_ || fragments);
    const Def *mem, *buffer;
    if (fragments) {
        mem = fresh_mem();
        if (auto scalar = splat_scalar(init))
            buffer = w.pack(fragments->count(), w.pack(4, rewrite(scalar)));
        else
            std::tie(mem, buffer) = load_fragments(init, *fragments, mem);
    } else {
        std::tie(mem, buffer)
            = buffer::op_alloc(w.lit_nat(rank), storage, rewrite(element), shared_mem(fresh_mem()))->projs<2>();
        transient(w.lit_nat(rank), storage, rewrite(element), buffer);
    }
    if (fragments) {
    } else if (first_scalar) {
        auto first_root
            = old.app(std::get<0>(peel_step(step->reduce(old.tuple({init, old.lit_nat(0)})).back())), input);

        auto [first, first_call]
            = btensor::ds_splice(w.tuple({mem, buffer}), w.sigma({mem->type(), buffer->type()}), "foldFirst");
        auto [first_mem, first_buffer] = first->var(2, 0)->projs<2>();
        auto pending                   = fe::Restore(pending_, {});
        auto nested_batch              = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
        auto memo                      = fe::Restore(fresh_memo_, {});
        auto cache                     = fe::Restore(realizations_, realizations_);
        auto nested                    = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
        push();
        bind_new(first->has_var());
        prepare_realizations(first_root);
        first->app(true, first->ret_var(),
                   lower_into(first_root, {first_mem, first_buffer, storage, rewrite(zero), extent}));
        first->set(true, wrap_fresh_mem(first->body()));
        pop();
        std::tie(mem, buffer) = first_call->projs<2>();
    } else if (auto scalar = splat_scalar(init)) {
        mem = copy_box(mem, buffer, storage, rewrite(scalar), nullptr, extent, cooperative());
    } else if (!bounded) {
        auto initial = make_buffer(init, w.lit_nat(rank), rewrite(shape));
        plain(initial);
        mem = copy_buffer(mem, buffer, initial, rewrite(shape), cooperative());
    } else {
        auto initial = make_buffer(init, w.lit_nat(rank), extent);
        auto source  = fe::lookup(storage_shapes_, init);
        plain(initial);
        mem = copy_box(mem, buffer, storage, initial, source ? source : extent, extent, cooperative());
    }
    // A work-group fold with at least two blocks keeps the panels of `pipeline - 1` blocks ahead in flight while it
    // computes on the current ones; each block's fills form one group of asynchronous copies.
    auto depth     = Lit::as(w.app(w.annex(tensor::pipeline_plan::depth),
                                   {w.lit_nat(pipeline), w.lit_nat(literal_count ? *literal_count : 0)}));
    auto stages    = depth > 1 && storage_level_ != 0 && driver().is_loaded("gpu") && inplace && !first_scalar
                       ? staged_panels(root, parameter->var())
                       : DefVec();
    auto blocks_of = [&](const Def* var) {
        return staged_panels(old.app(std::get<0>(peel_step(step->reduce(var).back())), input), var);
    };
    // A ring of `depth` buffers per panel; `ring[s]` holds the block `s` ahead of the current one.
    std::vector<DefVec> ring;
    std::vector<size_t> pipelined;
    u64 panel_group = 0;
    if (!stages.empty()) {
        auto probe = old.mut_lam(parameter->type()->as<Pi>());
        if (auto block = Lit::isa(stride)) multiples_[probe->var(2, 1)] = *block;
        auto probes = blocks_of(probe->var());
        if (probes.size() != stages.size()) app->blame("pipelined fold found {} panels", probes.size()).bail();
        auto memo    = fe::Restore(fresh_memo_, {});
        auto probing = fe::Restore(probe_fill_, true);
        push();
        bind_old(probe->has_var());
        map(probe->var(2, 1), w.lit_nat_0());
        DefVec kept, first;
        for (size_t i = 0; i != probes.size(); ++i) {
            auto panel = rewrite(probes[i]);
            if (!shared_fills_.contains(panel)) continue;
            if (auto shape = fe::lookup(storage_shapes_, probes[i])) storage_shapes_[stages[i]] = shape;
            kept.push_back(stages[i]);
            pipelined.push_back(i);
            first.push_back(panel);
        }
        pop();
        stages = kept;
        if (!first.empty()) ring.push_back(first);
        panel_group = Panel_Groups + panel_groups_++;
        for (u64 s = 1; s < depth && !first.empty(); ++s) {
            DefVec buffers;
            for (auto panel : first) {
                auto [r, shape, T] = Axm::as<buffer::Buf>(panel->type())->args<3>();
                const Def* other;
                // The ring's later slots die with the fold, so the epilogue's staging may reuse them.
                auto scope = tile_round_
                               ? w.app(w.annex<gpu::shared_scope>(), mem)
                               : w.app(w.app(w.annex<gpu::panel_scope>(), w.lit_nat(panel_group)), mem);
                std::tie(mem, other) = buffer::op_alloc(r, shape, T, scope)->projs<2>();
                forward_layouts(panel, other);
                buffers.push_back(other);
            }
            ring.push_back(buffers);
        }
    }
    auto panels  = ring.empty() ? size_t(0) : ring[0].size();
    auto proxied = fe::Restore(proxied_, proxied_ || panels != 0);
    // The input of the step's product each pipelined panel stages, for its multicast bit.
    auto slots = std::vector<u64>(panels, u64(64));
    auto bare  = [](const Def* def) {
        while (Axm::isa<tensor::buf>(def) || Axm::isa<tensor::placed>(def))
            def = def->as<App>()->arg();
        return def;
    };
    if (auto product = Axm::isa<tensor::map_reduce_post>(mma ? mma->arg() : result))
        for (size_t j = 0; j != panels; ++j) {
            auto ins = product->arg()->proj(2, 0);
            for (u64 i = 0, n = ins->num_projs(); i != n; ++i)
                if (bare(ins->proj(n, i)) == bare(stages[j])) slots[j] = i;
        }
    // With TMA, the leader copies each block's panels in boxes of 128-byte rows, completing the mbarrier of its slot.
    // A cluster sharing panels also gives each slot an "empty" mbarrier that every work-group of the cluster arrives on
    // once it has read the slot, so none refills a slot another still reads.
    const Def* bars    = nullptr;
    const Def* empties = nullptr;
    auto sharing       = false;
    for (size_t j = 0; j != panels; ++j)
        sharing |= cluster_.size && slots[j] < 64 && (cluster_.multicast >> slots[j] & 1);
    auto leader = [&] { return w.app(w.annex<gpu::leader>(), w.tuple()); };
    // A specialized fold's copies are issued by the producer warp's first work-item.
    auto specialized = false;
    auto issuer      = [&]() -> const Def* {
        if (!specialized) return leader();
        auto item  = w.app(w.annex<gpu::item_index>(), w.tuple());
        auto first = w.call(core::icmp::e,
                            Defs{w.call(core::bit2::and_, w.lit_nat_0(), Defs{item, w.lit_i64(127)}), w.lit_i64(0)});
        return w.call(core::bit2::and_, w.lit_nat(2), Defs{roles_.producer, first});
    };
    auto i32 = [&](const Def* nat) { return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(u64(1) << 32), nat); };
    // In a persistent work-group, the ring's fills count on across its tiles.
    const Def* fill_of = nullptr;
    auto global        = [&](const Def* fill) { return fill_of ? w.app(fill_of, fill) : fill; };
    auto at_of         = [&](const Def* buffers, const Def* slot) {
        auto [br, bs, bT] = Axm::as<buffer::Buf>(buffers->type())->args<3>();
        auto index        = w.call<core::idx>(w.lit_nat_0(), w.lit_nat(ring.size()), slot);
        return w.app(w.app(w.app(w.annex<gpu::buf_addr>(), bT), {br, bs}), {buffers, w.tuple({index})});
    };
    auto bar_at = [&](const Def* slot) { return at_of(bars, slot); };
    // Fill `fill` of the ring goes to slot `fill % depth`, which `fill / depth` fills used before.
    auto fill_tma = [&](const Def* at, const Def* block, const DefVec& targets, const Def* fill) {
        fill        = global(fill);
        auto depth  = w.lit_nat(ring.size());
        auto slot   = w.call(core::nat::rem, Defs{fill, depth});
        auto future = old.mut_lam(parameter->type()->as<Pi>());
        if (auto size = Lit::isa(stride)) multiples_[future->var(2, 1)] = *size;
        auto roots = blocks_of(future->var());
        push();
        bind_old(future->has_var());
        map(future->var(2, 1), w.call(core::nat::mul, Defs{block, rewrite(stride)}));
        // `gpu.tma_fill` is the protocol of a fill; which box of which matrix goes where comes from here.
        auto copies = DefVec();
        for (size_t j = 0; j != panels; ++j) {
            auto stage = Axm::isa<tensor::stage>(Axm::peel<tensor::placed>(roots[pipelined[j]]));
            auto slice = stage ? Axm::isa<tensor::slice>(Axm::peel<tensor::buf>(stage->arg())) : nullptr;
            if (!slice) roots[pipelined[j]]->blame("a TMA-filled panel stages a slice").n(unlowered).bail();
            auto [slice_meta, source_shape, box] = slice->callee()->as<App>()->uncurry_args<3>();
            auto source                          = rewrite(slice->arg());
            auto [sr, ss, sT]                    = Axm::as<buffer::Buf>(source->type())->args<3>();
            if (Lit::isa(sr) != 2) slice->blame("a TMA-filled panel stages a slice of a matrix").n(unlowered).bail();
            auto start = rewrite(box->proj(3, 0));
            auto rows = Lit::isa(box->proj(3, 2)->proj(2, 0)), cols = Lit::isa(box->proj(3, 2)->proj(2, 1));
            auto size = Lit::as(core::op(core::trait::size, sT));
            if (!rows || !cols || *cols * size % 128)
                slice->blame("a TMA-filled panel holds rows of 128 bytes or atoms of them").n(target).bail();
            auto atom = 128 / size;
            // The copy lands where the hardware puts it, so the panel's layout must name those addresses.
            if (auto logical = logical_of(targets[j])) {
                auto [pos, tiles, perm, pad] = stage->decurry()->decurry()->args<4>();
                auto inner                   = Lit::as(pos);
                u64 outer                    = 0;
                while (Lit::as(perm->proj(2, outer)) == inner)
                    ++outer;
                auto row = inner == 0 ? u64(2) : outer, col = inner == 0 ? outer : u64(2);
                auto query   = w.tuple({layout_of(targets[j]), w.lit_nat(size), w.lit_nat(row), w.lit_nat(col)});
                auto verdict = panel::at_shape(w.annex(tensor::layout_expr::lies_tma128), query, logical);
                log().d("TMA panel {} against the swizzled copy: {}", targets[j]->type(), verdict);
                if (Lit::isa(verdict) == 1)
                    slice->blame("a TMA-filled panel does not lie where the swizzled copy puts it").n(target).bail();
            }
            // A multicast panel is a cluster's: each work-group copies its share of the rows into all of them.
            auto shares
                = cluster_.size && slots[j] < 64 && (cluster_.multicast >> slots[j] & 1) ? cluster_.size : u64(1);
            if (*rows % (8 * shares))
                slice->blame("a multicast panel's {} rows do not split into {} shares", *rows, shares).n(target).bail();
            // The descriptor is the global matrix's, built where the matrix is known outside the kernel.
            auto map          = w.app(w.app(w.app(w.annex<gpu::buf_tensor_map>(), sT), ss),
                                      {outside(source), w.lit_nat(*rows / shares), w.lit_nat(atom)});
            auto [tr, ts, tT] = Axm::as<buffer::Buf>(targets[j]->type())->args<3>();
            auto zero         = w.tuple(
                DefVec(Lit::as(tr), [&](size_t d) { return w.lit_idx(Lit::as(ts->proj(Lit::as(tr), d)), 0); }));
            auto base = w.app(w.app(w.app(w.annex<gpu::buf_addr>(), tT), {tr, ts}), {targets[j], zero});
            copies.push_back(w.tuple({base, map, i32(start->proj(2, 0)), i32(start->proj(2, 1)), w.lit_nat(*rows),
                                      w.lit_nat(*cols), w.lit_nat(size), w.lit_nat(shares)}));
        }
        auto fill_of = w.app(w.annex<gpu::tma_fill>(), w.lit_nat(panels));
        at           = w.app(w.app(fill_of, {w.tuple(copies), bar_at(slot), issuer()}), at);
        pop();
        return at;
    };
    // Starts the copies of block @p block's panels into @p targets after @p at.
    auto fill_block = [&](const Def* at, const Def* block, const DefVec& targets) {
        auto future = old.mut_lam(parameter->type()->as<Pi>());
        if (auto size = Lit::isa(stride)) multiples_[future->var(2, 1)] = *size;
        auto roots = blocks_of(future->var());
        auto async = fe::Restore(async_fill_, true);
        push();
        bind_old(future->has_var());
        map(future->var(2, 1), w.call(core::nat::mul, Defs{block, rewrite(stride)}));
        for (size_t j = 0; j != panels; ++j) {
            auto stage   = roots[pipelined[j]];
            fill_target_ = {at, targets[j]};
            filled_mem_  = nullptr;
            rewrite(stage);
            if (!filled_mem_) stage->blame("a pipelined panel was not filled into its ring buffer").bail();
            at = filled_mem_;
        }
        pop();
        return at;
    };
    tma         = tma && panels != 0;
    specialized = tma && roles_.producer && fragments && fragments->group;
    // A persistent work-group sets its ring up in its first tile, and its fills count on across its tiles.
    auto first_round = tile_round_ ? w.call(core::ncmp::e, Defs{tile_round_, w.lit_nat_0()}) : w.lit_tt();
    if (tma && tile_round_) {
        if (!specialized)
            app->blame("a TMA reduction in a persistent work-group must be specialized").n(unlowered).bail();
        if (!literal_count)
            app->blame("a persistent ring needs a literal number of blocks per tile").n(unlowered).bail();
        fill_of      = w.app(w.annex(tensor::pipeline_plan::persistent_fill),
                             {tile_round_, w.lit_nat(*literal_count), w.lit_nat(ring.size())});
        round_fence_ = false;
    }
    if (tma) {
        std::tie(mem, bars) = buffer::op_alloc(w.lit_nat(1), w.tuple({w.lit_nat(ring.size())}), w.type_i64(),
                                               w.app(w.annex<gpu::shared_scope>(), mem))
                                  ->projs<2>();
        if (sharing || specialized)
            std::tie(mem, empties) = buffer::op_alloc(w.lit_nat(1), w.tuple({w.lit_nat(ring.size())}), w.type_i64(),
                                                      w.app(w.annex<gpu::shared_scope>(), mem))
                                         ->projs<2>();
    }
    if (specialized && !inplace)
        app->blame("a specialized reduction updates its accumulator in place").n(unlowered).bail();
    // A ring of four or more slots keeps a step's MMAs in flight and releases the previous step's slot once those
    // have read it; a shallower one would leave the producer a single slot ahead.
    // A ring its readers fill by TMA does the same from two slots on: a step's MMAs read only its own slot.
    auto overlap = u64(specialized ? ring.size() >= 4 : tma && fragments && fragments->group && ring.size() >= 2);
    if (specialized && literal_count) {
        // The protocol model covers the fills of two tiles numbered in a row, as a persistent ring of whole turns does.
        auto fills = w.lit_nat(std::min<u64>(2 * *literal_count, 16));
        auto safe  = w.app(w.app(w.app(w.app(w.annex(tensor::pipeline_plan::ring_safe), fills), w.lit_nat(ring.size())),
                                 w.lit_nat(overlap)),
                           w.lit_nat_1());
        if (Lit::isa(safe) != 1)
            app->blame("a ring of {} slots releasing {} steps late breaks its protocol", ring.size(), overlap)
                .n(unsafe)
                .bail();
    }
    // A ring is `gpu.ring_fold`'s protocol, its copies tracked by `cp.async` groups or by the slots' mbarriers, or
    // `gpu.ws_fold`'s when a producer warp fills it; what a block's fill and step compute comes from here.
    if (panels != 0) {
        auto M       = w.call<mem::M>(0);
        auto slot_ty = w.sigma(DefVec(panels, [&](size_t j) { return ring[0][j]->type(); }));
        auto acc_ty  = buffer->type();
        auto closure = [&](Lam* fun, const std::function<const Def*()>& body) {
            auto pending      = fe::Restore(pending_, {});
            auto nested_batch = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
            auto memo         = fe::Restore(fresh_memo_, {});
            auto cache        = fe::Restore(realizations_, realizations_);
            auto nested       = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
            push();
            bind_new(fun->has_var());
            fun->app(true, fun->ret_var(), body());
            fun->set(true, wrap_fresh_mem(fun->body()));
            pop();
        };
        // The producer warp's fill `n` is block `n`'s; a ring its readers fill counts the fills on its own.
        auto fill_dom = specialized ? DefVec{M, w.type_nat(), slot_ty} : DefVec{M, w.type_nat(), w.type_nat(), slot_ty};
        auto fill     = w.mut_fun(w.sigma(fill_dom), M)->set("ringFill");
        closure(fill, [&] {
            auto args = fill->var(2, 0)->projs(fill_dom.size());
            auto at = args[0], block = args[1], nth = args[fill_dom.size() - 2], slot = args.back();
            auto targets = DefVec(panels, [&](size_t j) { return slot->proj(panels, j); });
            for (size_t j = 0; j != panels; ++j)
                forward_layouts(ring[0][j], targets[j]);
            return tma ? fill_tma(at, shifted(block), targets, nth) : fill_block(at, block, targets);
        });
        // A specialized step also hands back what its MMAs left in flight, which is what its late release relies on.
        auto left      = w.app(w.annex<gpu::InFlight>(), w.lit_nat(overlap));
        auto step_cod  = specialized ? w.sigma({M, acc_ty, left}) : w.sigma({M, acc_ty});
        auto fold_step = w.mut_fun(w.sigma({M, acc_ty, w.type_nat(), slot_ty}), step_cod)->set("ringStep");
        auto step_body = [&]() -> const Def* {
            auto [at, acc, iteration, slot] = fold_step->var(2, 0)->projs<4>();
            bind_old(parameter->has_var());
            map(accumulator, acc);
            if (fragments) fragments_[acc] = *fragments;
            if (bounded) storage_shapes_[accumulator] = storage;
            map(offset, w.call(core::nat::mul, Defs{shifted(iteration), rewrite(stride)}));
            for (size_t j = 0; j != panels; ++j) {
                map(stages[j], slot->proj(panels, j));
                forward_layouts(ring[0][j], slot->proj(panels, j));
            }
            prepare_realizations(root);
            auto in_flight = fe::Restore(wgmma_pending_, tma ? overlap : wgmma_pending_);
            if (inplace) return lower_into(root, {at, acc, storage, rewrite(zero), extent});
            if (!bounded) {
                auto value = make_buffer(root, w.lit_nat(rank), rewrite(shape));
                plain(value);
                return w.tuple({copy_buffer(at, acc, value, rewrite(shape), cooperative()), acc});
            }
            auto value  = make_buffer(root, w.lit_nat(rank), extent);
            auto source = fe::lookup(storage_shapes_, root);
            plain(value);
            return w.tuple({copy_box(at, acc, storage, value, source ? source : extent, extent, cooperative()), acc});
        };
        closure(fold_step, [&]() -> const Def* {
            auto issued = fe::Restore<const Def*>(in_flight_, nullptr);
            auto next   = step_body();
            if (!specialized) return next;
            if (!in_flight_) app->blame("a specialized reduction's step issues no MMAs").bail();
            return w.tuple({next->proj(2, 0), next->proj(2, 1), in_flight_});
        });
        auto depth  = w.lit_nat(ring.size());
        auto slots  = DefVec(ring.size(), [&](size_t s) { return w.tuple(ring[s]); });
        auto blocks = base ? w.lit_nat(*literal_count) : rewrite(count);
        auto parity = [&](const Def* of, const char* name) {
            auto lam = w.mut_lam(w.pi(w.type_nat(), w.type_nat()))->set(name);
            lam->set(true, w.app(w.app(of, lam->var()), depth));
            return lam;
        };
        // Each ring's tokens carry its own number, so the tokens of two rings never stand for each other.
        auto rid = w.lit_nat(rings_++);
        if (driver().arg_nat("tensor", "ring-coverage").value_or(0)) {
            // Each covered shape names the Lean model in `proofs/ring` whose protocol theorems apply to it.
            auto consumers      = specialized ? roles_.consumers : group_threads_ / 128;
            auto depth          = ring.size();
            auto [model, scope] = tile_round_                     ? std::pair("none", "persistent")
                                : cluster_.size                   ? std::pair("none", "cluster")
                                : depth < 2                       ? std::pair("none", "depth")
                                : !tma                            ? std::pair("async", "check-layout")
                                : specialized                     ? consumers == 2 && !roles_.pingpong && overlap == 1
                                                                      ? std::pair("warp_specialized", "check-layout")
                                                                      : std::pair("none", "ws-config")
                                : !fragments || !fragments->group ? std::pair("none", "no-wgmma-batch")
                                : consumers == 1                  ? std::pair("tma", "check-layout")
                                : consumers == 2 && overlap == 1 && *literal_count + 1 >= depth
                                    ? std::pair("cooperative", "check-layout")
                                    : std::pair("none", "consumers");
            std::println(stderr,
                         "ring-coverage rid={} fold=tile depth={} lag={} consumers={} groups={} model={} scope={}",
                         Lit::as(rid), depth, overlap, consumers, ring_groups(ring), model, scope);
        }
        const Def* sync;
        if (tma) {
            auto vacant = parity(w.annex(tensor::pipeline_plan::empty_parity), "ringVacant");
            auto full   = parity(w.annex(tensor::pipeline_plan::full_parity), "ringParity");
            sync = w.app(w.app(w.annex<gpu::tma_ring>(), {rid, depth}),
                         {bars, empties ? empties : bars, full, vacant, w.lit_nat(cluster_.size), w.lit_bool(sharing),
                          w.lit_nat(specialized ? roles_.consumers : 0),
                          specialized && roles_.pingpong ? roles_.consumer : w.lit_nat_0(),
                          specialized ? roles_.producer : w.lit_ff(), w.lit_bool(specialized && roles_.pingpong),
                          w.lit_bool(tile_round_ != nullptr), first_round});
        } else {
            sync = w.app(
                w.app(w.annex<gpu::async_ring>(), {rid, depth}),
                {w.app(w.annex(tensor::pipeline_plan::pending), depth), w.lit_bool(fragments && fragments->group)});
        }
        const Def* fold;
        if (specialized) {
            auto nth = w.mut_lam(w.pi(w.type_nat(), w.type_nat()))->set("ringFill");
            nth->set(true, global(nth->var()));
            fold = w.app(w.annex<gpu::ws_fold>(), {acc_ty, slot_ty, rid, depth});
            fold = w.app(fold,
                         {sync, w.call<core::bitcast>(w.type_i64(), blocks), roles_.producer, nth, w.lit_nat(overlap)});
        } else {
            auto refill = w.mut_lam(w.pi(w.type_nat(), w.type_nat()))->set("ringRefill");
            refill->set(true, w.app(w.annex(tensor::pipeline_plan::refill_block),
                                    {refill->var(), depth, w.lit_nat(*literal_count)}));
            fold = w.app(w.annex<gpu::ring_fold>(), {acc_ty, slot_ty, rid, depth});
            fold = w.app(fold, {sync, w.call<core::bitcast>(w.type_i64(), blocks), refill, w.lit_nat(overlap),
                                w.lit_bool(tma), w.lit_tt()});
        }
        // A ring that shares its later slots starts behind a barrier, so no work-item still reads the last epilogue's
        // staging; a TMA ring's set-up has one already.
        if (!tile_round_ && !tma && ring.size() > 1) mem = w.app(w.annex<gpu::fence>(), mem);
        fold = w.app(w.app(fold, {mem, buffer, w.tuple(slots)}), {fill, fold_step});
        if (bounded) storage_shapes_[app] = storage;
        auto [out_mem, out] = fold->projs<2>();
        if (fragments) fragments_[out] = *fragments;
        return scalar_result(app, out_mem, out);
    }
    auto [run, call] = btensor::ds_splice(w.tuple({mem, buffer}), w.sigma({mem->type(), buffer->type()}), "tileFold");
    auto remaining   = first_scalar ? w.lit_nat(*literal_count - 1) : base ? w.lit_nat(*literal_count) : rewrite(count);
    auto [loop, loop_call]       = affine::counting_for(w.call<core::bitcast>(w.type_i64(), remaining), run->var(2, 0),
                                                        run->ret_var(), w.sym("tileFoldStep"));
    auto [index, carried, yield] = loop->vars<3>();
    auto [loop_mem, loop_buffer] = carried->projs<2>();
    auto pending                 = fe::Restore(pending_, {});
    auto nested_batch            = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
    auto memo                    = fe::Restore(fresh_memo_, {});
    auto cache                   = fe::Restore(realizations_, realizations_);
    auto nested                  = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
    push();
    bind_old(parameter->has_var());
    map(accumulator, loop_buffer);
    if (fragments) fragments_[loop_buffer] = *fragments;
    if (bounded) storage_shapes_[accumulator] = storage;
    auto iteration = w.call<core::bitcast>(w.type_nat(), index);
    if (first_scalar) iteration = w.call(core::nat::add, Defs{iteration, w.lit_nat(1)});
    map(offset, w.call(core::nat::mul, Defs{shifted(iteration), rewrite(stride)}));
    prepare_realizations(root);
    const Def* update;
    if (inplace) {
        update = lower_into(root, {loop_mem, loop_buffer, storage, rewrite(zero), extent});
    } else if (!bounded) {
        auto value = make_buffer(root, w.lit_nat(rank), rewrite(shape));
        plain(value);
        update = w.tuple({copy_buffer(loop_mem, loop_buffer, value, rewrite(shape), cooperative()), loop_buffer});
    } else {
        auto value  = make_buffer(root, w.lit_nat(rank), extent);
        auto source = fe::lookup(storage_shapes_, root);
        plain(value);
        update
            = w.tuple({copy_box(loop_mem, loop_buffer, storage, value, source ? source : extent, extent, cooperative()),
                       loop_buffer});
    }
    loop->app(true, yield, update);
    loop->set(true, wrap_fresh_mem(loop->body()));
    pop();
    run->set(true, loop_call);
    if (bounded) storage_shapes_[app] = storage;
    auto [out_mem, out] = call->projs<2>();
    if (fragments) fragments_[out] = *fragments;
    return scalar_result(app, out_mem, out);
}

// A fold step's body without its `tensor.bind` level, `tensor.merge_with` merge and `tensor.pipeline` depth, and those
// three.
std::tuple<const Def*, u64, const Def*, u64, bool> LowerToMem::peel_step(const Def* body) const {
    auto [stepped, level] = region_body(body);
    const Def* merge      = nullptr;
    if (auto marked = Axm::isa<tensor::merge_with>(stepped)) {
        merge   = marked->callee()->as<App>()->arg();
        stepped = marked->arg();
    }
    u64 depth = 1;
    if (auto marked = Axm::isa<tensor::pipeline>(stepped)) {
        depth   = Lit::as(marked->decurry()->arg());
        stepped = marked->arg();
    }
    auto tma = Axm::isa<tensor::fill_tma>(stepped);
    if (tma) stepped = tma->arg();
    return {stepped, level, merge, depth, bool(tma)};
}

// The binary `op` of a combiner `λ (acc, ins). op (acc, e)` (or `op (e, acc)`) whose `e` is free of `acc`: with the
// seed its identity, `op` also merges partial folds (the R8 obligation: `op` associative and commutative, as
// `math.mode.fast` allows).
const Def* LowerToMem::merge_of(const Def* comb) const {
    auto lam = comb->isa_mut<Lam>();
    if (!lam || !lam->is_set()) return nullptr;
    auto value = Lam::isa_ret_arg(lam);
    auto app   = value ? value->isa<App>() : nullptr;
    if (!app || app->arg()->num_projs() != 2) return nullptr;
    auto acc   = lam->var(2, 0)->proj(2, 0);
    auto other = app->arg(2, 0) == acc ? app->arg(2, 1) : app->arg(2, 1) == acc ? app->arg(2, 0) : nullptr;
    if (!other) return nullptr;
    auto seen  = DefSet();
    auto stack = DefVec{other};
    while (!stack.empty()) {
        auto def = stack.back();
        stack.pop_back();
        if (def == acc) return nullptr;
        if (!seen.insert(def).second || def->isa<Var>() || def->is_closed()) continue;
        for (auto op : def->ops())
            if (op) stack.push_back(op);
    }
    return app->callee();
}

// The lanes a work-item region's reduction runs on: the block count of a lane-bound fold directly in @p body.
u64 LowerToMem::lane_count(const Def* body) const {
    auto pending = DefVec{body};
    auto seen    = DefSet();
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>()) continue;
        if (i != 0 && Axm::isa<tensor::tiled>(def)) continue;
        if (auto fold = Axm::isa<tensor::tile_fold>(def))
            if (auto step = fold->arg()->isa<Lam>(); step && step->is_set() && region_body(step->body()).second == 4)
                if (auto n = Lit::isa(fold->uncurry_args<6>()[1]->proj(2, 0))) return *n;
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return 1;
}

// The blocks of a lane-bound reduction run one per lane of the work-item, each folding its own seed.
// The partials then meet by warp shuffles in a butterfly, which leaves every lane with the merged tile.
const Def* LowerToMem::lower_lane_fold(const App* app,
                                       Lam* parameter,
                                       const Def* root,
                                       const Def* result,
                                       const Def* merge,
                                       u64 rank,
                                       const Def* element,
                                       const Def* extent,
                                       const Def* storage,
                                       bool bounded,
                                       bool inplace) {
    auto& w                                             = new_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto [accumulator, offset]                          = parameter->var()->projs<2>();
    auto n                                              = Lit::isa(count);
    auto seed                                           = splat_scalar(init);
    auto call                                           = result->isa<App>();
    auto mr_app                                         = call ? Axm::isa<tensor::map_reduce_post>(call) : nullptr;
    if (!lane_ || !n || *n != lanes_)
        app->blame("tensor.tile_fold: a lane reduction must run directly in a work-item with as many lanes").bail();
    if (!inplace) app->blame("tensor.tile_fold: a lane reduction needs an in-place step").n(unlowered).bail();
    if (!seed)
        app->blame("tensor.tile_fold: a lane reduction needs a scalar seed, its combine's identity").n(unsafe).bail();
    if (!mr_app) app->blame("tensor.tile_fold: a lane reduction's step must be a map_reduce_post").n(unlowered).bail();
    if (!merge) merge = merge_of(btensor::MapRedPost::of(mr_app->callee()->as<App>()).comb);
    if (!merge) app->blame("tensor.tile_fold: a lane reduction's combiner does not merge partials").n(unsafe).bail();
    auto dims = std::vector<u64>(rank);
    u64 elems = 1;
    for (size_t d = 0; d != rank; ++d) {
        auto lit = Lit::isa(storage->proj(rank, d));
        if (!lit) app->blame("tensor.tile_fold: a lane reduction needs a literal tile").n(unlowered).bail();
        dims[d] = *lit;
        elems *= *lit;
    }
    auto literal_storage = w.tuple(DefVec(rank, [&](size_t d) { return w.lit_nat(dims[d]); }));
    auto zero            = w.tuple(DefVec(rank, [&](size_t) { return w.lit_nat(0); }));
    auto el              = rewrite(element);
    auto [mem, priv]     = buffer::op_alloc(w.lit_nat(rank), literal_storage, el, fresh_mem())->projs<2>();
    mem                  = copy_box(mem, priv, literal_storage, rewrite(seed), nullptr, literal_storage, {});
    const Def* update;
    {
        auto pending = fe::Restore(pending_, {});
        auto cache   = fe::Restore(realizations_, realizations_);
        push();
        bind_old(parameter->has_var());
        map(accumulator, priv);
        if (bounded) storage_shapes_[accumulator] = literal_storage;
        map(offset, w.call(core::nat::mul, Defs{lane_, rewrite(stride)}));
        prepare_realizations(root);
        update = lower_into(root, {mem, priv, literal_storage, zero, extent});
        pop();
    }
    auto op            = rewrite(merge);
    auto [merged, out] = update->projs<2>();
    for (u64 e = 0; e != elems; ++e) {
        auto coords = DefVec(rank);
        for (size_t d = rank, rest = e; d-- > 0; rest /= dims[d])
            coords[d] = w.lit_i64(rest % dims[d]);
        auto index         = btensor::folded_idx(literal_storage, coords);
        auto [read, value] = buffer::op_read(merged, out, index)->projs<2>();
        value              = w.app(w.app(w.app(w.annex<gpu::lane_merge>(), w.lit_nat_1()), {w.lit_nat(*n), op}), value);
        merged             = buffer::op_write(read, out, index, value)->proj(2, 0);
    }
    // The result buffer must depend on the shuffles' memory.
    auto [result_run, result_call]
        = btensor::ds_splice(w.tuple({merged, out}), w.sigma({merged->type(), out->type()}), "laneFoldResult");
    result_run->app(true, result_run->ret_var(), result_run->var(2, 0));
    if (bounded) storage_shapes_[app] = literal_storage;
    return scalar_result(app, result_call->proj(2, 0), result_call->proj(2, 1));
}

// R8: the blocks of a work-item bound reduction run one per work-item, each folding its own seed, and the
// partial tiles meet in shared memory, where the step's combiner folds them into the result.
const Def* LowerToMem::lower_bound_fold(const App* app,
                                        Lam* parameter,
                                        const Def* root,
                                        const Def* result,
                                        const Def* merge,
                                        u64 rank,
                                        const Def* shape,
                                        const Def* element,
                                        const Def* extent,
                                        const Def* storage,
                                        bool bounded,
                                        bool inplace,
                                        const Destination* destination) {
    auto& w                                             = new_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto [accumulator, offset]                          = parameter->var()->projs<2>();
    auto n                                              = Lit::isa(count);
    auto seed                                           = splat_scalar(init);
    // A flat array element's seed is a closed tuple of that type.
    if (!seed)
        if (auto splat = Axm::isa<tensor::splat>(Axm::peel<tensor::buf>(init)); splat && splat->arg(2, 1)->is_closed())
            seed = splat->arg(2, 1);
    auto call   = result->isa<App>();
    auto mr_app = call ? Axm::isa<tensor::map_reduce_post>(call) : nullptr;
    if (!group_wide_)
        app->blame("tensor.tile_fold: a work-item bound reduction must nest directly inside a work-group level")
            .n(unlowered)
            .bail();
    if (!inplace)
        app->blame("tensor.tile_fold: a work-item bound reduction needs an in-place step").n(unlowered).bail();
    // Beyond a work-group's work-items, the blocks run in whole turns of them.
    auto most = gpu_limits.group_items;
    if (!n || !*n || (*n > most && *n % most))
        app->blame("tensor.tile_fold: a work-item bound reduction needs a literal block count of at most 1024 or a "
                   "multiple of it")
            .bail();
    if (!seed)
        app->blame("tensor.tile_fold: a work-item bound reduction needs a scalar seed, its combine's identity")
            .n(unsafe)
            .bail();
    if (!mr_app)
        app->blame("tensor.tile_fold: a work-item bound reduction's step must be a map_reduce_post")
            .n(unlowered)
            .bail();
    auto mr = btensor::MapRedPost::of(mr_app->callee()->as<App>());
    if (!merge) merge = merge_of(mr.comb);
    if (!merge)
        app->blame(
               "tensor.tile_fold: a work-item bound reduction's combiner must be `op (acc, e)` with `e` free of `acc`, "
               "`op` then merging the work-items' partials; got `{}`",
               mr.comb)
            .n(unsafe)
            .bail();
    auto dims = std::vector<u64>(rank);
    for (size_t d = 0; d != rank; ++d) {
        auto lit = Lit::isa(storage->proj(rank, d));
        if (!lit) app->blame("tensor.tile_fold: a work-item bound reduction needs a literal tile").n(unlowered).bail();
        dims[d] = *lit;
    }
    auto literal_storage = w.tuple(DefVec(rank, [&](size_t d) { return w.lit_nat(dims[d]); }));
    auto zero            = w.tuple(DefVec(rank, [&](size_t) { return w.lit_nat(0); }));
    auto el              = rewrite(element);
    auto r               = w.lit_nat(rank);
    auto rows            = rank ? dims[0] : 1;
    auto prank           = rank ? rank : 1;
    // The partials of all work-items, stacked along the tile's first dim.
    auto pdims       = std::vector<u64>(rank ? dims : std::vector<u64>{1});
    pdims[0]         = rows * *n;
    auto pshape      = w.tuple(DefVec(prank, [&](size_t d) { return w.lit_nat(pdims[d]); }));
    const Def* start = destination ? destination->mem : fresh_mem();
    const Def* rbuf  = destination ? destination->buffer : nullptr;
    if (!destination) {
        std::tie(start, rbuf) = buffer::op_alloc(r, literal_storage, el, shared_mem(start))->projs<2>();
        transient(r, literal_storage, el, rbuf);
    }
    // `gpu.merge_rounds` is the protocol: one superstep writes the partials and each merge round claims them for one
    // more.
    auto g = w.lit_nat(batches_++);
    if (entered_++) app->blame("a work-item bound reduction cannot run inside a work-group batch").n(unlowered).bail();
    auto state           = w.sigma({w.call<mem::M>(0), buffer::type_buf(w.lit_nat(prank), pshape, el)});
    auto fill            = w.mut_lam(w.pi(state, state))->set("boundFoldFill");
    auto [fm, fb]        = fill->var()->projs<2>();
    auto [run, run_call] = btensor::ds_splice(w.tuple({fm, fb}), w.sigma({fm->type(), fb->type()}), "boundFold");
    fill->set(true, run_call);
    auto [run_mem, run_partials] = run->var(2, 0)->projs<2>();
    auto acc_ty                  = run_mem->type();
    auto items                   = w.mut_con({w.type_i64(), acc_ty, w.cn(acc_ty)})->set("boundFoldItems");
    auto done                    = w.mut_con(acc_ty)->set("boundFoldDone");
    done->app(true, run->ret_var(), w.tuple({done->var(), run_partials}));
    auto wrapped = w.app(w.app(w.annex<gpu::bind>(gpu::bind::item), items->type()), items);
    run->set(true, w.call<affine::For>(wrapped, done, Defs{w.lit_i64(0), w.lit_i64(*n), w.lit_i64(1), run_mem}));
    auto [index, item_mem, yield] = items->vars<3>();
    auto store                    = w.mut_con(acc_ty)->set("boundFoldStore");
    const Def* update;
    {
        // Each work-item folds its block into a private seeded tile, as the sequential fold does per block.
        auto level2        = fe::Restore(storage_level_, u64(2));
        auto per_item      = fe::Restore(group_wide_, false);
        auto origin        = fe::Restore(item_origin_, parameter->var()->as<Var>());
        auto private_scope = fe::Restore(uniform_, false);
        auto pending       = fe::Restore(pending_, {});
        auto nested_batch  = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
        auto memo          = fe::Restore(fresh_memo_, {});
        auto cache         = fe::Restore(realizations_, realizations_);
        auto nested        = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
        push();
        bind_old(parameter->has_var());
        auto [pm, priv] = buffer::op_alloc(r, literal_storage, el, item_mem)->projs<2>();
        pm              = copy_box(pm, priv, literal_storage, rewrite(seed), nullptr, literal_storage, {});
        map(accumulator, priv);
        if (bounded) storage_shapes_[accumulator] = literal_storage;
        map(offset, w.call(core::nat::mul, Defs{w.call<core::bitcast>(w.type_nat(), index), rewrite(stride)}));
        prepare_realizations(root);
        update = lower_into(root, {pm, priv, literal_storage, zero, extent});
        items->app(true, store, update->proj(2, 0));
        items->set(true, wrap_fresh_mem(items->body()));
        pop();
    }
    auto op = rewrite(merge);
    // A warp's work-items first merge their partials with shuffles, so the shared tree merges one per warp.
    auto f32      = w.annex<math::F32>();
    auto lanewise = [&](const Def* type) {
        if (type == f32) return true;
        auto n = Lit::isa(type->arity());
        if (!n || *n < 2) return false;
        for (u64 i = 0; i != *n; ++i)
            if (type->proj(*n, i) != f32) return false;
        return true;
    };
    auto warps = Lit::as(w.app(w.annex(tensor::r8_plan::is_tree), w.lit_nat(*n))) && *n % 32 == 0 && lanewise(el);
    auto parts = warps ? *n / 32 : *n;
    {
        // The private tile goes to the work-item's rows of the partials.
        Lam* cur        = store;
        const Def* exit = yield;
        const Def* acc  = store->var();
        auto iters      = affine::counting_nest(cur, exit, acc, literal_storage->projs(rank), "boundFoldStore");
        auto [rm, value]
            = buffer::op_read(acc, update->proj(2, 1), btensor::folded_idx(literal_storage, iters))->projs<2>();
        auto slot = index;
        if (warps) {
            auto k = el == f32 ? w.lit_nat_1() : el->arity();
            value  = w.app(w.app(w.app(w.annex<gpu::lane_merge>(), k), {w.lit_nat(32), op}), value);
            slot   = w.call(core::shr::l, Defs{index, w.lit_i64(5)});
        }
        auto at = DefVec(iters.begin(), iters.end());
        if (rank) {
            auto base = w.call(core::wrap::mul, core::Mode::none, Defs{slot, w.lit_i64(rows)});
            at[0]     = w.call(core::wrap::add, core::Mode::none, Defs{base, at[0]});
        } else {
            at.push_back(slot);
        }
        auto wm = buffer::op_write(rm, run_partials, btensor::folded_idx(pshape, at), value)->proj(2, 0);
        cur->app(true, exit, wm);
    }
    // The merged tile lands in the destination at its origin, or in a work-group buffer of its own.
    auto target = [&](const DefVec& coords) {
        if (!destination) return btensor::folded_idx(literal_storage, coords);
        auto at = DefVec(rank, [&](size_t d) {
            auto origin = w.call<core::bitcast>(w.type_i64(), destination->origin->proj(rank, d));
            return w.call(core::wrap::add, core::Mode::none, Defs{origin, coords[d]});
        });
        return btensor::folded_idx(destination->shape, at);
    };
    u64 elems = 1;
    for (auto dim : dims)
        elems *= dim;
    auto coords_of = [&](const Def* j) {
        auto coords     = DefVec(rank);
        const Def* rest = w.call<core::bitcast>(w.type_nat(), j);
        for (size_t d = rank; d-- > 0;) {
            coords[d] = w.call<core::bitcast>(w.type_i64(), w.call(core::nat::rem, Defs{rest, w.lit_nat(dims[d])}));
            rest      = w.call(core::nat::div, Defs{rest, w.lit_nat(dims[d])});
        }
        return coords;
    };
    // `gpu.merge_pairs`, `gpu.merge_first` and `gpu.merge_serial` are the merge; where an element lies comes from here.
    auto partials   = buffer::type_buf(w.lit_nat(prank), pshape, el);
    auto meta       = w.tuple({partials, el});
    auto partial_at = [&](const Def* t, const Def* e) {
        auto at = coords_of(e);
        auto ti = w.call<core::bitcast>(w.type_i64(), t);
        if (rank)
            at[0] = w.call(core::wrap::add, core::Mode::none,
                           Defs{w.call(core::wrap::mul, core::Mode::none, Defs{ti, w.lit_i64(rows)}), at[0]});
        else
            at.push_back(ti);
        return btensor::folded_idx(pshape, at);
    };
    auto get = w.mut_lam(w.pi(w.sigma({acc_ty, partials, w.type_nat(), w.type_nat()}), w.sigma({acc_ty, el})))
                   ->set("boundPartialGet");
    {
        auto [gm, parts, t, e] = get->vars<4>();
        get->set(true, buffer::op_read(gm, parts, partial_at(t, e)));
    }
    auto set
        = w.mut_lam(w.pi(w.sigma({acc_ty, partials, w.type_nat(), w.type_nat(), el}), acc_ty))->set("boundPartialSet");
    {
        auto [sm, parts, t, e, value] = set->vars<5>();
        set->set(true, buffer::op_write(sm, parts, partial_at(t, e), value)->proj(2, 0));
    }
    auto put = w.mut_lam(w.pi(w.sigma({acc_ty, w.type_nat(), el}), acc_ty))->set("boundCombinePut");
    {
        auto [pmem, e, value] = put->vars<3>();
        put->set(true, buffer::op_write(pmem, rbuf, target(coords_of(e)), value)->proj(2, 0));
    }
    const Def* combine;
    DefVec merges;
    // A power-of-two work-item count merges its partials as a tree, halving the live partials behind each barrier; a
    // few are merged by one work-item behind a single barrier.
    if (parts >= gpu_limits.tree_partials && Lit::as(w.app(w.annex(tensor::r8_plan::is_tree), w.lit_nat(parts)))) {
        auto rounds = w.app(w.annex(tensor::r8_plan::rounds), {w.lit_nat(parts), w.lit_nat(elems)});
        // A single round's tuple is the round itself, so the count comes from the plan, not the value's arity.
        for (auto round : rounds->projs(parts / 2)) {
            auto [active, s, pairs, trip] = round->projs<4>([](auto value) { return Lit::as(value); });
            if (!active) continue;
            merges.push_back(w.app(w.app(w.annex<gpu::merge_pairs>(), meta),
                                   {w.lit_nat(elems), w.lit_nat(s), w.lit_nat(pairs), w.lit_nat(trip), op, get, set}));
        }
        combine = w.app(w.app(w.annex<gpu::merge_first>(), meta), {w.lit_nat(elems), get, put});
    } else {
        combine = w.app(w.app(w.annex<gpu::merge_serial>(), meta),
                        {w.lit_nat(elems), w.lit_nat(parts), rewrite(seed), op, get, put});
    }
    auto protocol
        = w.app(w.annex<gpu::merge_rounds>(), {g, w.lit_nat(merges.size()), w.lit_nat(prank), pshape, el, w.sigma()});
    auto out_mem = w.app(protocol, {start, fill, w.tuple(merges), combine})->proj(2, 0);
    --entered_;
    // The result buffer must depend on the merge's memory.
    auto [result_run, result_call]
        = btensor::ds_splice(w.tuple({out_mem, rbuf}), w.sigma({out_mem->type(), rbuf->type()}), "boundFoldResult");
    result_run->app(true, result_run->ret_var(), result_run->var(2, 0));
    if (destination) return result_call;
    if (bounded) storage_shapes_[app] = literal_storage;
    return scalar_result(app, result_call->proj(2, 0), result_call->proj(2, 1));
}

const Def* LowerToMem::lower_product_fold(const App* app) {
    auto& w                                             = new_world();
    auto& old                                           = old_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto step                                           = app->arg()->isa<Lam>();
    if (!step || !step->is_set()) app->blame("tensor.tile_fold step must be a defined lambda").bail();
    auto parameter             = old.mut_lam(old.pi(old.sigma({type, old.type_nat()}), old.type_nat()));
    auto [accumulator, offset] = parameter->var()->projs<2>();
    auto root                  = old.app(step->reduce(parameter->var()).back(), region_input(inputs));

    auto fields = [&](auto&& self, const Def* t, const Def* value, auto&& leaf) -> const Def* {
        if (auto sigma = t->isa_imm<Sigma>()) {
            for (size_t i = 0; i != 128; ++i) {
                auto call = value->isa<App>();
                if (!call) break;
                auto lam = call->callee()->isa<Lam>();
                if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
                value = lam->reduce(call->arg()).back();
            }
            return w.tuple(DefVec(sigma->num_ops(), [&](size_t i) {
                return self(self, sigma->op(i), value->proj(sigma->num_ops(), i), leaf);
            }));
        }
        return leaf(t, value);
    };
    auto shape_of = [](const Def* t) {
        auto tensor = *TensorType::isa(t);
        return std::pair{tensor.s, tensor.T};
    };
    const Def* mem = fresh_mem();
    DefVec primary, spare, storages;
    fe::Vector<bool> pooled;
    // A work-group's fold keeps its state in shared memory at its bounded storage, written by all its work-items.
    auto group = storage_level_ == 1 && driver().is_loaded("gpu");
    // A field that only its own update reads, element by element, is updated in place.
    auto in_place = fe::Vector<bool>();
    if (auto sigma = type->isa_imm<Sigma>(); sigma && group) {
        auto n     = sigma->num_ops();
        auto value = root;
        for (size_t i = 0; i != 128; ++i) {
            auto call = value->isa<App>();
            if (!call) break;
            auto lam = call->callee()->isa<Lam>();
            if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
            value = lam->reduce(call->arg()).back();
        }
        for (size_t i = 0; i != n; ++i) {
            auto field  = accumulator->proj(n, i);
            auto rank   = TensorType::isa(sigma->op(i)) ? Lit::isa(TensorType::isa(sigma->op(i))->r) : std::nullopt;
            auto inside = rank && *rank != 0;
            for (size_t j = 0; inside && j != n; ++j)
                if (j != i) inside = !mentions(value->proj(n, j), field);
            auto zero = old.tuple(DefVec(rank.value_or(0), [&](size_t) { return old.lit_nat(0); }));
            in_place.push_back(inside && pointwise_step(value->proj(n, i), field, zero));
        }
    }
    auto shared     = [&w](const Def* m) { return w.app(w.annex<gpu::shared_scope>(), m); };
    auto initialize = [&](const Def* t, const Def* value) -> const Def* {
        if (!TensorType::isa(t)) {
            auto scalar = rewrite(value);
            primary.push_back(scalar);
            spare.push_back(scalar);
            storages.push_back(nullptr);
            pooled.push_back(false);
            return scalar;
        }
        auto [shape, element] = shape_of(t);
        auto rank             = w.lit_nat(Lit::as(shape->arity()));
        auto owned            = scratch_ && lit_projs(rewrite(shape), Lit::as(rank)) && rewrite(element)->is_closed();
        const Def *current, *next;
        auto extent  = rewrite(shape);
        auto storage = group ? bounded_shape(extent, true) : nullptr;
        auto alone   = primary.size() < in_place.size() && in_place[primary.size()];
        if (group) {
            std::tie(mem, current) = buffer::op_alloc(rank, storage, rewrite(element), shared(mem))->projs<2>();
            next                   = current;
            if (!alone)
                std::tie(mem, next) = buffer::op_alloc(rank, storage, rewrite(element), shared(mem))->projs<2>();
        } else if (owned) {
            current = allocate(rank, extent, rewrite(element)).second;
            next    = allocate(rank, extent, rewrite(element)).second;
        } else {
            std::tie(mem, current) = buffer::op_alloc(rank, extent, rewrite(element), mem)->projs<2>();
            std::tie(mem, next)    = buffer::op_alloc(rank, extent, rewrite(element), mem)->projs<2>();
        }
        pooled.push_back(owned || group);
        if (group) group_buffers_.insert({current, next});
        primary.push_back(current);
        spare.push_back(next);
        storages.push_back(storage);
        if (!group) {
            plain(value);
            mem = copy_buffer(mem, current, make_buffer(value, rank, extent), extent, cooperative());
        } else if (auto scalar = splat_scalar(value)) {
            mem = copy_box(mem, current, storage, rewrite(scalar), nullptr, extent, cooperative());
        } else {
            plain(value);
            auto source = fe::lookup(storage_shapes_, value);
            mem = copy_box(mem, current, storage, make_buffer(value, rank, extent), source ? source : extent, extent,
                           cooperative());
        }
        return current;
    };
    auto current           = fields(fields, type, init, initialize);
    size_t cursor          = 0;
    auto take_spare        = [&](const Def*, const Def*) { return spare[cursor++]; };
    auto alternate         = fields(fields, type, init, take_spare);
    auto state             = w.tuple({mem, current, alternate});
    auto [run, call]       = btensor::ds_splice(state, state->type(), "productFold");
    auto [loop, loop_call] = affine::counting_for(w.call<core::bitcast>(w.type_i64(), rewrite(count)), run->var(2, 0),
                                                  run->ret_var(), w.sym("productFoldStep"));
    auto [index, carried, yield]         = loop->vars<3>();
    auto [loop_mem, source, destination] = carried->projs<3>();
    if (group)
        for (size_t i = 0, n = storages.size(); i != n; ++i)
            if (storages[i]) group_buffers_.insert({source->proj(n, i), destination->proj(n, i)});
    {
        auto pending      = fe::Restore(pending_, {});
        auto step         = fe::Restore(step_root_, group ? root : nullptr);
        auto nested_batch = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
        auto memo         = fe::Restore(fresh_memo_, {});
        auto cache        = fe::Restore(realizations_, realizations_);
        auto nested       = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
        push();
        bind_old(parameter->has_var());
        map(accumulator, source);
        if (auto sigma = type->isa_imm<Sigma>(); sigma && sigma->num_ops() == storages.size())
            for (size_t i = 0; i != storages.size(); ++i)
                if (storages[i]) storage_shapes_[accumulator->proj(storages.size(), i)] = storages[i];
        map(offset, w.call(core::nat::mul, Defs{w.call<core::bitcast>(w.type_nat(), index), rewrite(stride)}));
        prepare_realizations(root);
        DefVec targets;
        auto collect = [&](const Def*, const Def* value) {
            targets.push_back(value);
            return value;
        };
        fields(fields, type, destination, collect);
        cursor                 = 0;
        const Def* updated_mem = loop_mem;
        auto update            = [&](const Def* t, const Def* value) -> const Def* {
            auto target = targets[cursor++];
            if (!TensorType::isa(t)) return rewrite(value);
            auto [shape, _] = shape_of(t);
            if (auto operation = value->isa<App>()) {
                if (Axm::isa<tensor::map_reduce_post>(operation))
                    shape = btensor::MapRedPost::of(operation->callee()->as<App>()).So;
                else if (Axm::isa<tensor::tiled>(operation))
                    shape = operation->uncurry_args<5>()[1]->proj(3, 0);
            }
            auto rank    = Lit::as(shape->arity());
            auto zero    = w.tuple(DefVec(rank, [&](size_t) { return w.lit_nat(0); }));
            auto storage = storages[cursor - 1] ? storages[cursor - 1] : rewrite(shape);
            auto written = lower_into(value, {updated_mem, target, storage, zero, rewrite(shape)});
            updated_mem  = written->proj(2, 0);
            return target;
        };
        auto result = fields(fields, type, root, update);
        loop->app(true, yield, w.tuple({updated_mem, result, source}));
        loop->set(true, wrap_fresh_mem(loop->body()));
        pop();
    }
    run->set(true, loop_call);
    auto output_type                 = w.sigma({mem->type(), current->type()});
    auto [finish, finished]          = btensor::ds_splice(call, output_type, "productFoldResult");
    auto [final_mem, result, unused] = finish->var(2, 0)->projs<3>();
    const Def* released              = final_mem;
    cursor                           = 0;
    auto stable                      = [&](const Def* t, const Def* value) {
        auto i = cursor++;
        if (pooled[i] || !TensorType::isa(t)) return value;
        auto [rank, shape, element] = Axm::isa<buffer::Buf>(primary[i]->type())->args<3>();
        const Def* target;
        std::tie(released, target) = buffer::op_alloc(rank, shape, element, released)->projs<2>();
        plain(value);
        released = copy_buffer(released, target, value, shape, cooperative());
        return target;
    };
    auto output  = fields(fields, type, result, stable);
    cursor       = 0;
    auto release = [&](const Def* t, const Def* value) {
        auto i = cursor++;
        if (!pooled[i] && bool(TensorType::isa(t))) {
            auto [rank, shape, element] = Axm::isa<buffer::Buf>(primary[i]->type())->args<3>();
            released                    = buffer::op_free(rank, shape, element, released, primary[i]);
            released                    = buffer::op_free(rank, shape, element, released, spare[i]);
        }
        return value;
    };
    fields(fields, type, unused, release);
    finish->app(true, finish->ret_var(), w.tuple({released, output}));
    if (auto sigma = type->isa_imm<Sigma>(); sigma && sigma->num_ops() == storages.size())
        for (size_t i = 0; i != storages.size(); ++i)
            if (storages[i]) storage_shapes_[app->proj(storages.size(), i)] = storages[i];
    return finished->proj(2, 1);
}

std::string LowerToMem::ring_groups(const std::vector<DefVec>& ring) {
    auto groups = std::set<std::string>();
    for (auto& slot : ring)
        for (auto buffer : slot) {
            auto alloc = buffer->isa<Extract>() ? Axm::isa<buffer::alloc>(buffer->as<Extract>()->tuple()) : nullptr;
            auto scope = alloc ? Axm::isa<gpu::panel_scope>(alloc->arg()) : nullptr;
            groups.insert(!alloc ? "unknown" : scope ? std::to_string(Lit::as(scope->decurry()->arg())) : "shared");
        }
    if (groups.size() > 1) groups.erase("shared");
    auto names = std::string();
    for (auto& group : groups)
        names += (names.empty() ? "" : ",") + group;
    return names.empty() ? "shared" : names;
}

const Def* LowerToMem::panel_mem(const Def* mem) {
    auto& w = new_world();
    if (tile_round_) return w.app(w.annex<gpu::shared_scope>(), mem);
    return w.app(w.app(w.annex<gpu::panel_scope>(), w.lit_nat(Panel_Groups + panel_groups_++)), mem);
}

} // namespace mim::plug::tensor::phase
