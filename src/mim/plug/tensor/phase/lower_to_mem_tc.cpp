#include "mim/plug/tensor/phase/lower_to_mem.h"

#include "lower_to_mem_util.h"

namespace mim::plug::tensor::phase {

using namespace lower_to_mem_detail;

// `gpu.wgmma_batch` is the protocol of a group of MMAs; which instructions it issues over @p x comes from here.
// Panels a work-group filled with generic stores reach `wgmma` through the async proxy only after every filler fenced
// them.
const Def*
LowerToMem::wgmma_batch(const Def* mem, const Def* x, Defs issue, const Def* pending, bool turn, Defs operands) {
    auto& w = new_world();
    // A ring publishes only its own slots, not a panel the work-group filled with generic stores.
    auto publish
        = !proxied_ || std::ranges::any_of(operands, [&](const Def* b) { return generic_panels_.contains(b); });
    auto batch = w.app(w.annex<gpu::wgmma_batch>(), {x->type(), w.lit_nat(issue.size())});
    batch      = w.app(batch, {w.lit_bool(publish), pending, w.lit_bool(turn), turn ? roles_.consumer : w.lit_nat_0()});
    auto [out, y, left] = w.app(w.app(batch, {mem, x}), w.tuple(issue))->projs<3>();
    in_flight_          = left;
    return w.tuple({out, y});
}

// What `layout_expr.wgmma_fields` says of a `wgmma` operand panel whose K dim is @p kdim: whether it is MN-major and
// its descriptor's fields.
std::array<u64, 4> LowerToMem::wgmma_fields(const Def* buffer, std::optional<u64> kdim) {
    auto& w      = new_world();
    auto layout  = layout_of(buffer);
    auto logical = logical_of(buffer);
    if (!layout || !logical) buffer->blame("a wgmma operand panel needs its swizzled layout").bail();
    auto T      = Axm::as<buffer::Buf>(buffer->type())->arg(3, 2);
    auto query  = w.tuple({layout, core::op(core::trait::size, T), w.lit_nat(kdim.value_or(logical->num_projs()))});
    auto fields = panel::at_shape(w.annex(tensor::layout_expr::wgmma_fields), query, logical);
    return {Lit::as(fields->proj(4, 0)), Lit::as(fields->proj(4, 1)), Lit::as(fields->proj(4, 2)),
            Lit::as(fields->proj(4, 3))};
}

std::pair<bool, u64> LowerToMem::wgmma_major(const Def* buffer, std::optional<u64> kdim) {
    auto fields = wgmma_fields(buffer, kdim);
    return {fields[0] != 0, fields[1]};
}

// The descriptor of a `wgmma` operand starting at the panel's element @p index.
const Def* LowerToMem::wgmma_desc(const Def* buffer, const Def* index, std::optional<u64> kdim) {
    auto& w                      = new_world();
    auto [mn, lbo, sbo, swizzle] = wgmma_fields(buffer, kdim);
    auto [br, bs, bT]            = Axm::as<buffer::Buf>(buffer->type())->args<3>();
    // The descriptor names the 128-byte swizzle, so the panel's layout must be that one.
    auto query   = w.tuple({layout_of(buffer), core::op(core::trait::size, bT)});
    auto verdict = panel::at_shape(w.annex(tensor::layout_expr::lies_sw128), query, logical_of(buffer));
    log().d("wgmma panel {} against the 128-byte swizzle: {}", buffer->type(), verdict);
    if (Lit::isa(verdict) == 1)
        buffer->blame("a wgmma operand panel does not lie as the 128-byte swizzle its descriptor names")
            .n(target)
            .bail();
    auto addr = w.app(w.app(w.app(w.annex<gpu::buf_addr>(), bT), {br, bs}), {buffer, index});
    return w.app(w.annex<gpu::wgmma_desc>(), {addr, w.lit_i64(lbo), w.lit_i64(sbo), w.lit_i64(swizzle)});
}

// One warp's tensor-core block: A's fragments by ldmatrix from its row-major panel, B's by a transposed ldmatrix or
// scalar loads, and the accumulator, an SSA value of fragments, updated per instruction; F32 inputs use m16n8k8, 16-bit
// ones m16n8k16.
const Def* LowerToMem::lower_mma(const App* app, const Destination& destination, const Fragments& fragments) {
    auto& w     = new_world();
    auto root   = app->arg()->as<App>();
    auto old_mr = btensor::MapRedPost::of(root->callee()->as<App>());
    auto panels = fe::Restore(mma_panels_, true);
    auto batch  = fe::Restore<std::optional<Batch>>(batch_, open_batch(destination.mem));
    auto is     = rewrite(root->arg())->proj(2, 0);
    auto T      = rewrite(old_mr.Tis->proj(2, 0));
    auto narrow = T != w.annex<math::F32>();
    auto kt     = narrow ? u64(16) : u64(8);
    return close_batch(destination.mem, destination.buffer->type(), [&](const Def* mem) -> const Def* {
        auto Sr = rewrite(old_mr.Sr);
        for (u64 d = 0; d != 3; ++d) {
            auto extent = Lit::isa(Sr->proj(3, d));
            auto tile   = std::array{fragments.m, fragments.n, kt}[d];
            if (!extent || *extent % tile)
                app->blame("the tensor-core tile does not divide the warp's block {}", Sr).n(target).bail();
        }
        auto depth = Lit::as(Sr->proj(3, 2));
        // Input `i` read at loops `(m, n, k)`: its placed physical coordinates and storage.
        auto old_is = root->arg()->proj(2, 0);
        auto coords = [&](size_t i, const Def* m, const Def* n, const Def* k) {
            auto buffer  = is->proj(2, i);
            auto r       = rewrite(old_mr.Ris->proj(2, i));
            auto storage = fe::lookup(storage_shapes_, old_is->proj(2, i));
            if (!storage) storage = rewrite(old_mr.Sis->proj(2, i));
            auto map = rewrite(old_mr.maps->proj(2, i));
            if (auto layout = layout_of(buffer)) {
                auto placed = w.mut_lam(map->type()->as<Pi>())->set("placedMap");
                placed->set(true, w.app(layout_lam(layout, logical_of(buffer)), w.app(map, placed->var())));
                map = placed;
            }
            auto loops              = w.tuple(DefVec(3, [&](size_t d) {
                auto at = std::array{m, n, k}[d];
                return w.call<core::idx>(w.lit_nat_0(), Sr->proj(3, d), at);
            }));
            auto [mapped, physical] = btensor::op_affine_map(map, r, w.lit_nat(3), Sr, storage, loops, mem)->projs<2>();
            mem                     = mapped;
            return std::tuple{buffer, storage, physical, Lit::as(r)};
        };
        auto nat  = [&](u64 v) { return w.lit_nat(v); };
        auto add  = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
        auto lane = lane_index(w);
        // Without the explicit rounding, `mma` reads the F32 operands truncated to TF32.
        auto rounded = [&](const Def* fragment, u64 size) {
            if (!tf32_rna_) return fragment;
            return w.tuple(DefVec(size, [&](size_t i) {
                return w.app(w.annex<gpu::tf32_rna>(), w.extract(fragment, w.lit_idx(size, i)));
            }));
        };
        auto ms = fragments.rows / fragments.m, ns = fragments.cols / fragments.n;
        auto acc = DefVec(destination.buffer->projs(fragments.count()));
        if (fragments.group) {
            auto operand = [&](size_t i, u64 ks) {
                auto [buffer, storage, physical, r] = coords(i, nat(0), nat(0), nat(ks * kt));
                return std::pair{buffer, *Shape(physical).fold(storage)};
            };
            auto [a_buffer, a0] = operand(0, 0);
            auto [b_buffer, b0] = operand(1, 0);
            auto kdims          = std::array<std::optional<u64>, 2>();
            for (size_t i = 0; i != 2; ++i)
                kdims[i] = dim_of_loop(old_mr.maps->proj(2, i), Lit::as(old_mr.Ris->proj(2, i)), 2);
            if (wgmma_major(a_buffer, kdims[0]).first)
                app->blame("a wgmma operand A must be K-major").n(unlowered).bail();
            auto [b_mn, b_lbo] = wgmma_major(b_buffer, kdims[1]);
            auto desc          = [&](size_t i, u64 ks) {
                auto [buffer, index] = operand(i, ks);
                return wgmma_desc(buffer, index, kdims[i]);
            };
            auto nt          = fragments.count();
            auto instr       = math::isa_bf16(T) ? w.annex<gpu::wgmma_bf16>() : w.annex<gpu::wgmma_f16>();
            const Def* tiles = w.tuple(acc);
            auto state       = w.sigma({mem->type(), tiles->type()});
            auto issue       = DefVec(depth / kt, [&](size_t ks) -> const Def* {
                auto lam = w.mut_lam(w.pi(state, state))->set("wgmmaIssue");
                lam->set(true, w.app(instr, {nat(nt), lam->var(2, 0), desc(0, ks), desc(1, ks), w.lit_bool(b_mn),
                                             lam->var(2, 1)}));
                return lam;
            });
            // Ping-pong: a consumer issues its MMAs on its turn and then hands the turn to the other one.
            std::tie(mem, tiles) = wgmma_batch(mem, tiles, issue, nat(wgmma_pending_),
                                               roles_.pingpong && roles_.consumer, {a_buffer, b_buffer})
                                       ->projs<2>();
            return w.tuple({mem, w.tuple(DefVec(nt, [&](size_t j) { return tiles->proj(nt, j); }))});
        }
        auto read = [&](const Def* lane_reg, Enum auto map, u64 ni, u64 ks) {
            auto [row, col]                     = fragment_coords(w, map, lane_reg);
            auto [buffer, storage, physical, r] = coords(1, nat(0), add(nat(ni * 8), col), add(nat(ks * kt), row));
            const Def* value;
            std::tie(mem, value) = buffer::op_read(mem, buffer, *Shape(physical).fold(storage))->template projs<2>();
            return value;
        };
        for (u64 ks = 0; ks != depth / kt; ++ks) {
            auto a = DefVec(ms);
            for (u64 mi = 0; mi != ms; ++mi) {
                auto width                          = narrow ? u64(8) : u64(4);
                auto [row, col]                     = narrow ? fragment_coords(w, mma_m16n8k16::panel_a_addr, lane)
                                                             : fragment_coords(w, mma_m16n8k8_tf32::panel_a_addr, lane);
                auto [buffer, storage, physical, r] = coords(0, add(nat(mi * 16), row), nat(0), add(nat(ks * kt), col));
                auto runs                           = DefVec(storage->projs(r));
                runs.back()                         = w.call(core::nat::div, Defs{runs.back(), nat(width)});
                auto run                            = DefVec(physical->projs(r));
                auto last                           = w.call<core::idx2nat>(run.back());
                run.back()
                    = w.call<core::idx>(w.lit_nat_0(), runs.back(), w.call(core::nat::div, Defs{last, nat(width)}));
                auto vectors      = buffer::op_vectorize(nat(width), buffer);
                auto index        = *Shape(w.tuple(run)).fold(w.tuple(runs));
                auto [vr, vs, vT] = Axm::as<buffer::Buf>(vectors->type())->args<3>();
                auto loaded = narrow
                                ? w.app(w.app(w.app(w.annex<gpu::buf_ldmatrix_b16>(), T), {vr, vs}),
                                        {nat(4), w.lit_ff(), mem, vectors, index})
                                : w.app(w.app(w.annex<gpu::buf_ldmatrix>(), {vr, vs}), {nat(4), mem, vectors, index});
                std::tie(mem, a[mi]) = loaded->projs<2>();
                if (!narrow) a[mi] = rounded(a[mi], 4);
            }
            auto b = DefVec(ns);
            // A 16-bit B panel whose n runs of 8 are contiguous and aligned gives two tiles' fragments per transposed
            // `ldmatrix`, its lanes addressing k rows as A's address m rows.
            // A 16-bit B panel gives two n tiles' fragments per ldmatrix where the caps of its map's images show n and
            // k each driving one dim and its layout keeps the innermost of them in runs of 8: transposed if that is
            // n's.
            enum class Runs { none, n, k };
            auto runs = [&] {
                auto r = Lit::isa(rewrite(old_mr.Ris->proj(2, 1)));
                if (!narrow || ns % 2 || !r) return Runs::none;
                auto buffer  = is->proj(2, 1);
                auto shape   = rewrite(old_mr.Sis->proj(2, 1));
                auto storage = fe::lookup(storage_shapes_, old_is->proj(2, 1));
                auto image
                    = w.app(w.app(w.annex<affine::image>(), {nat(3), nat(*r)}), rewrite(old_mr.maps->proj(2, 1)));
                auto driven = [&](u64 loop) -> std::optional<u64> {
                    auto ext  = w.tuple(DefVec(3, [&](size_t d) { return nat(d == loop ? 8 : 1); }));
                    auto caps = w.app(image, {w.pack(3, nat(0)), ext, shape})->proj(4, 2);
                    auto dim  = std::optional<u64>();
                    for (u64 d = 0; d != *r; ++d) {
                        auto cap = Lit::isa(caps->proj(*r, d));
                        if (!cap || (*cap != 1 && dim)) return {};
                        if (*cap != 1) dim = d;
                    }
                    return dim;
                };
                auto rows   = storage ? storage : shape;
                auto rank   = rows->num_projs();
                auto layout = layout_of(buffer);
                // The physical dim a logical dim's runs of 8 land in, unchanged.
                auto lands = [&](u64 dim) -> std::optional<u64> {
                    auto at = std::optional<u64>();
                    for (u64 j : {1, 7, 8}) {
                        auto point = DefVec(*r, [&](size_t d) { return w.lit_i64(d == dim ? j : 0); });
                        auto phys  = layout ? layout_coords(layout, logical_of(buffer), point) : point;
                        for (u64 d = 0; d != phys.size(); ++d) {
                            auto x = Lit::isa(phys[d]);
                            if (!x || (*x != 0 && (*x != j || (at && *at != d)))) return {};
                            if (*x != 0) at = d;
                        }
                    }
                    return at;
                };
                auto n = driven(1), k = driven(2);
                auto last = Lit::isa(rows->proj(rank, rank - 1));
                if (!n || !k || *n == *k || !last || *last % 8) return Runs::none;
                if (lands(*n) == rank - 1) return Runs::n;
                if (lands(*k) == rank - 1) return Runs::k;
                return Runs::none;
            };
            if (auto along = runs(); along != Runs::none) {
                for (u64 ni = 0; ni != ns; ni += 2) {
                    auto trans        = along == Runs::n;
                    auto [row, col]   = trans ? fragment_coords(w, mma_m16n8k16::panel_a_addr, lane)
                                              : fragment_coords(w, mma_m16n8k16::panel_b_addr, lane);
                    auto [n_at, k_at] = trans ? std::pair{col, row} : std::pair{row, col};
                    auto [buffer, storage, physical, r]
                        = coords(1, nat(0), add(nat(ni * 8), n_at), add(nat(ks * kt), k_at));
                    auto sizes   = DefVec(storage->projs(r));
                    sizes.back() = w.call(core::nat::div, Defs{sizes.back(), nat(8)});
                    auto at      = DefVec(physical->projs(r));
                    auto last    = w.call<core::idx2nat>(at.back());
                    at.back()
                        = w.call<core::idx>(w.lit_nat_0(), sizes.back(), w.call(core::nat::div, Defs{last, nat(8)}));
                    auto vectors      = buffer::op_vectorize(nat(8), buffer);
                    auto index        = *Shape(w.tuple(at)).fold(w.tuple(sizes));
                    auto [vr, vs, vT] = Axm::as<buffer::Buf>(vectors->type())->args<3>();
                    const Def* regs;
                    std::tie(mem, regs) = w.app(w.app(w.app(w.annex<gpu::buf_ldmatrix_b16>(), T), {vr, vs}),
                                                {nat(4), w.lit_bool(trans), mem, vectors, index})
                                              ->projs<2>();
                    b[ni]               = w.tuple({regs->proj(4, 0), regs->proj(4, 1)});
                    b[ni + 1]           = w.tuple({regs->proj(4, 2), regs->proj(4, 3)});
                }
            } else
                for (u64 ni = 0; ni != ns; ++ni) {
                    if (narrow) {
                        auto regs = DefVec(2, [&](size_t reg) {
                            return w.tuple(DefVec(2, [&](size_t h) {
                                return read(w.tuple({lane, w.lit_idx(2, reg), w.lit_idx(2, h)}), mma_m16n8k16::l_b, ni,
                                            ks);
                            }));
                        });
                        b[ni]     = w.tuple(regs);
                    } else {
                        auto half = DefVec(2, [&](size_t h) {
                            return read(w.tuple({lane, w.lit_idx(2, h)}), mma_m16n8k8_tf32::l_b, ni, ks);
                        });
                        b[ni]     = rounded(w.tuple(half), 2);
                    }
                }
            auto instr = !narrow           ? w.annex<gpu::mma_tf32>()
                       : math::isa_bf16(T) ? w.annex<gpu::mma_bf16>()
                                           : w.annex<gpu::mma_f16>();
            for (u64 mi = 0; mi != ms; ++mi)
                for (u64 ni = 0; ni != ns; ++ni) {
                    auto& c = acc[mi * ns + ni];
                    c       = w.app(instr, w.tuple({a[mi], b[ni], c}));
                }
        }
        return w.tuple({mem, w.tuple(acc)});
    });
}

// Writes each lane's accumulator fragments to their rows and columns of @p destination.
// A specialized region's producer warp holds no accumulator, so only the consumers write theirs.
// A pointwise map over a fold's accumulator fragments, such as a bias add, runs on each element as it is written.
const Def* LowerToMem::store_epilogue(const App* app, const Destination& destination) {
    auto& old = old_world();
    auto mr   = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ro = Lit::isa(mr.Ro), rn = Lit::isa(mr.Rn), nis = Lit::isa(mr.nis), nps = Lit::isa(mr.nps);
    if (ro != 2 || rn != 2 || !nis || !nps) return nullptr;
    auto [inputs, posts] = app->arg()->projs<2>();
    std::optional<u64> held;
    const Def* source = nullptr;
    for (u64 k = 0; k != *nis && !held; ++k) {
        auto value = Axm::peel<tensor::buf>(inputs->proj(*nis, k));
        auto fold  = Axm::isa<tensor::tile_fold>(value->isa<Extract>() ? value->as<Extract>()->tuple() : value);
        if (!fold || map_dims(mr.maps->proj(*nis, k), 2) != std::vector<u64>{0, 1}) continue;
        source = rewrite(value);
        if (fe::lookup(fragments_, source)) held = k;
    }
    if (!held) return nullptr;
    auto fragments = *fe::lookup(fragments_, source);
    // The other inputs are read element by element on the memory chain of the writes; the epilogue itself is pure.
    struct Input {
        const Def *buffer, *map, *rank, *shape;
    };
    auto& w      = new_world();
    auto others  = std::vector<Input>();
    auto types   = DefVec{TensorType::isa(inputs->proj(*nis, *held)->type())->T};
    auto collect = [&](const Def* input, const Def* map, const Def* rank, const Def* shape) {
        types.push_back(TensorType::isa(input->type())->T);
        // A view is read through its source, which keeps a bias tile from being copied out per work-item.
        while (auto read = panel::tensor_read(input, shape)) {
            if (!TensorType::isa(read->src->type())) break;
            input = read->src;
            rank  = read->R;
            shape = read->S;
            map   = panel::compose_read_map(read->map, map);
        }
        others.push_back({rewrite(input), rewrite(map), rewrite(rank), rewrite(shape)});
    };
    for (u64 k = 0; k != *nis; ++k)
        if (k != *held)
            collect(inputs->proj(*nis, k), mr.maps->proj(*nis, k), mr.Ris->proj(*nis, k), mr.Sis->proj(*nis, k));
    for (u64 k = 0; k != *nps; ++k)
        collect(posts->proj(*nps, k), mr.post_maps->proj(*nps, k), mr.Rps->proj(*nps, k), mr.Sps->proj(*nps, k));
    auto g      = old.mut_lam(old.pi(old.sigma(types), TensorType::isa(app->type())->T))->set("fragmentEpilogue");
    auto values = DefVec(g->var()->projs(types.size()));
    auto next   = size_t(1);
    auto ins    = old.tuple(DefVec(*nis, [&](size_t k) { return k == *held ? values[0] : values[next++]; }));
    auto ps     = old.tuple(DefVec(*nps, [&](size_t) { return values[next++]; }));
    auto call   = [&](const Def* f, const Def* arg) { return old.app(cps::op_cps2ds_dep(f), arg); };
    g->set(true, call(mr.post, old.tuple({call(mr.comb, old.tuple({mr.init, ins})), ps})));
    auto lowered = rewrite(g);
    auto so      = rewrite(mr.So);
    return store_fragments(
        source, fragments, destination, [&, so, lowered](const Def* mem, const Def* r, const Def* c, const Def* v) {
            auto loops = w.tuple({w.call<core::idx>(w.lit_nat_0(), so->proj(2, 0), r),
                                  w.call<core::idx>(w.lit_nat_0(), so->proj(2, 1), c)});
            auto args  = DefVec{v};
            for (auto& input : others) {
                auto at = btensor::op_affine_map(input.map, input.rank, w.lit_nat(2), so, input.shape, loops, mem);
                auto [br, bs, bT] = Axm::as<buffer::Buf>(input.buffer->type())->args<3>();
                std::tie(mem, at) = at->projs<2>();
                const Def* element;
                std::tie(mem, element) = buffer::op_read(mem, input.buffer, *Shape(at).fold(bs))->projs<2>();
                args.push_back(element);
            }
            return std::pair{mem, w.app(lowered, w.tuple(args))};
        });
}

const Def* LowerToMem::store_fragments(const Def* source,
                                       const Fragments& fragments,
                                       const Destination& destination,
                                       const Epilogue& epilogue) {
    if (!roles_.producer || !fragments.group) return write_fragments(source, fragments, destination, epilogue);
    // The accumulator is an argument of the branch, so the fold that computes it runs in both roles.
    auto& w          = new_world();
    auto mem_type    = destination.mem->type();
    auto [run, call] = btensor::ds_splice(w.tuple({destination.mem, source}), mem_type, "storeConsumers");
    auto [mem, acc]  = run->var(2, 0)->projs<2>();
    auto write       = w.mut_con(mem_type)->set("storeWrite");
    auto skip        = w.mut_con(mem_type)->set("storeSkip");
    run->app(true, w.extract(w.tuple({write, skip}), roles_.producer), mem);
    skip->app(true, run->ret_var(), skip->var());
    auto inner = destination;
    inner.mem  = write->var();
    write->app(true, run->ret_var(), write_fragments(acc, fragments, inner, epilogue)->proj(2, 0));
    return w.tuple({call, destination.buffer});
}

const Def* LowerToMem::write_fragments(const Def* source,
                                       const Fragments& fragments,
                                       const Destination& destination,
                                       const Epilogue& epilogue) {
    auto& w   = new_world();
    auto rank = Lit::as(destination.shape->arity());
    auto nat  = [&](u64 v) { return w.lit_nat(v); };
    auto add  = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
    auto lane = lane_index(w);
    auto ns   = fragments.cols / fragments.n;
    auto mem  = destination.mem;
    auto base = group_row(w, fragments.group);
    // The work-groups of a cluster that split the reduction hold partial accumulators in the same layout: each
    // work-item publishes its own in shared memory and adds those of its counterparts through distributed shared
    // memory.
    auto partials = DefVec();
    if (cluster_.split > 1 && fragments.group && group_threads_) {
        // The producer warpgroup skips the store, so it would never reach the cluster barriers below.
        if (roles_.producer)
            destination.buffer->blame("a producer-filled reduction cannot split its blocks over a cluster").bail();
        auto values = fragments.count() * 4;
        auto F32    = w.annex<math::F32>();
        const Def* scratch;
        std::tie(mem, scratch) = buffer::op_alloc(nat(1), w.tuple({nat(values * group_threads_)}), F32,
                                                  w.app(w.annex<gpu::shared_scope>(), mem))
                                     ->projs<2>();
        auto tid               = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::item_index>(), w.tuple()));
        auto at                = [&](u64 v) {
            auto index = add(w.call(core::nat::mul, Defs{nat(v), nat(group_threads_)}), tid);
            return w.tuple({w.call<core::idx>(w.lit_nat_0(), nat(values * group_threads_), index)});
        };
        for (u64 v = 0; v != values; ++v)
            mem = buffer::op_write(mem, scratch, at(v),
                                   w.extract(source->proj(fragments.count(), v / 4), w.lit_idx(4, v % 4)))
                      ->proj(2, 0);
        mem               = w.app(w.annex<gpu::sync_cluster>(), mem);
        auto [sr, ss, sT] = Axm::as<buffer::Buf>(scratch->type())->args<3>();
        for (u64 v = 0; v != values; ++v) {
            auto addr        = w.app(w.app(w.app(w.annex<gpu::buf_addr>(), sT), {sr, ss}), {scratch, at(v)});
            const Def* total = nullptr;
            for (u64 r = 0; r != cluster_.split; ++r) {
                const Def* part;
                std::tie(mem, part) = w.app(w.annex<gpu::load_peer>(), {mem, addr, w.lit_i32(r)})->projs<2>();
                total               = total ? w.call(math::arith::add, math::Mode::fast, Defs{total, part}) : part;
            }
            partials.push_back(total);
        }
        mem = w.app(w.annex<gpu::sync_cluster>(), mem);
    }
    // A warpgroup's 16-bit result goes through shared memory: a lane holds only two adjacent elements of a row, whose
    // 4-byte stores leave the 32-byte sectors of global memory half empty.
    auto [dr, ds, dT] = Axm::as<buffer::Buf>(destination.buffer->type())->args<3>();
    auto bytes        = Lit::isa(core::op(core::trait::size, dT));
    auto last         = Lit::isa(destination.shape->proj(rank, rank - 1));
    if (fragments.group && partials.empty() && !roles_.producer && !padded_ && rank == 2 && group_threads_ >= 128
        && bytes && *bytes == 2 && last && fragments.cols % 8 == 0 && *last % 8 == 0)
        return stage_fragments(source, fragments, destination, epilogue, mem);
    auto paired = Lit::isa(destination.shape->proj(rank, rank - 1));
    if (rank != 2 || (paired && *paired % 2)) paired = std::nullopt;
    for (u64 f = 0; f != fragments.count(); ++f) {
        auto value = partials.empty() ? source->proj(fragments.count(), f)
                                      : w.tuple(DefVec(4, [&](size_t e) { return partials[f * 4 + e]; }));
        // A lane's elements `e` and `e + 1` are adjacent columns of one row, so an even row length writes them as a
        // pair.
        for (u64 e = 0; e != 4; e += 2) {
            auto [local_row, local_col] = fragment_coords(w, mma_m16n8k8_tf32::l_c, w.tuple({lane, w.lit_idx(4, e)}));
            auto row                    = add(base, add(nat(f / ns * fragments.m), local_row));
            auto col                    = add(nat(f % ns * fragments.n), local_col);
            auto pair                   = DefVec(2);
            for (u64 h = 0; h != 2; ++h) {
                pair[h] = w.extract(value, w.lit_idx(4, e + h));
                if (epilogue) std::tie(mem, pair[h]) = epilogue(mem, row, h ? add(col, nat(1)) : col, pair[h]);
            }
            auto coords
                = DefVec(rank, [&](size_t d) { return add(destination.origin->proj(rank, d), d == 0 ? row : col); });
            // A padded tile writes only inside its destination.
            const Def* inside = nullptr;
            for (size_t d = 0; padded_ && d != rank; ++d) {
                auto last = d + 1 == rank ? add(coords[d], nat(1)) : coords[d];
                auto in   = w.call(core::ncmp::l, Defs{last, destination.shape->proj(rank, d)});
                inside    = inside ? w.call(core::bit2::and_, w.lit_nat(2), Defs{inside, in}) : in;
            }
            Lam* guard        = nullptr;
            Lam* write        = nullptr;
            const Def* joined = nullptr;
            if (inside) {
                // The pair is an argument of the branch, so what computes it stays outside.
                std::tie(guard, joined)
                    = btensor::ds_splice(w.tuple({mem, w.tuple(pair)}), mem->type(), "fragmentGuard");
                auto [m, values] = guard->var(2, 0)->projs<2>();
                write            = w.mut_con({mem->type(), values->type()})->set("fragmentWrite");
                auto pass        = w.mut_con({mem->type(), values->type()})->set("fragmentSkip");
                guard->app(true, w.extract(w.tuple({pass, write}), inside), w.tuple({m, values}));
                pass->app(true, guard->ret_var(), pass->var(2, 0));
                mem = write->var(2, 0);
                for (u64 h = 0; h != 2; ++h)
                    pair[h] = write->var(2, 1)->proj(2, h);
            }
            if (paired) {
                auto runs     = DefVec(destination.shape->projs(rank));
                runs.back()   = nat(*paired / 2);
                coords.back() = w.call(core::nat::div, Defs{coords.back(), nat(2)});
                auto index    = w.tuple(
                    DefVec(rank, [&](size_t d) { return w.call<core::idx>(w.lit_nat_0(), runs[d], coords[d]); }));
                mem = buffer::op_write(mem, buffer::op_vectorize(nat(2), destination.buffer),
                                       *Shape(index).fold(w.tuple(runs)), w.tuple(pair))
                          ->proj(2, 0);
            } else {
                for (u64 h = 0; h != 2; ++h) {
                    auto index = w.tuple(DefVec(rank, [&](size_t d) {
                        auto coord = d + 1 == rank && h ? add(coords[d], nat(1)) : coords[d];
                        return w.call<core::idx>(w.lit_nat_0(), destination.shape->proj(rank, d), coord);
                    }));
                    mem = buffer::op_write(mem, destination.buffer, *Shape(index).fold(destination.shape), pair[h])
                              ->proj(2, 0);
                }
            }
            if (guard) {
                write->app(true, guard->ret_var(), mem);
                mem = joined;
            }
        }
    }
    return w.tuple({mem, destination.buffer});
}

// Writes each warpgroup's fragments into its own rows of a work-group buffer in shared memory, and then each
// warpgroup's work-items copy those rows out in 16-byte runs.
// The barriers before and after keep the buffer clear of the panels every warpgroup still reads and of the rows
// still being written.
const Def* LowerToMem::stage_fragments(const Def* source,
                                       const Fragments& fragments,
                                       const Destination& destination,
                                       const Epilogue& epilogue,
                                       const Def* mem) {
    auto& w    = new_world();
    auto nat   = [&](u64 v) { return w.lit_nat(v); };
    auto add   = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
    auto mul   = [&](const Def* a, u64 b) { return w.call(core::nat::mul, Defs{a, nat(b)}); };
    auto idx   = [&](u64 size, const Def* x) { return w.call<core::idx>(w.lit_nat_0(), nat(size), x); };
    auto fence = [&](const Def* m) { return w.app(w.annex<gpu::fence>(), m); };
    auto lane  = lane_index(w);
    auto ns    = fragments.cols / fragments.n;
    auto T     = Axm::as<buffer::Buf>(destination.buffer->type())->arg(3, 2);
    auto rows  = 4 * fragments.rows;
    auto cols  = fragments.cols;
    // A row padded by 16 bytes puts a warp's eight rows on different banks.
    auto pitch = cols + 8;
    auto teams = group_threads_ / 128;
    auto tid   = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::item_index>(), w.tuple()));
    auto team  = w.call(core::nat::rem, Defs{w.call(core::nat::div, Defs{tid, nat(128)}), nat(teams)});
    auto first = add(mul(team, rows), group_row(w, true));
    // TMA stores a team's tile in boxes of 128-byte atoms, which lie swizzled one after another as a TMA load puts
    // them.
    auto atoms = cols / 64;
    // A tensor map encodes only floating-point 16-bit elements here.
    auto stored = tma_store_ && cols % 64 == 0 && rows % 8 == 0 && (T == w.annex<math::F16>() || math::isa_bf16(T));
    auto extent = stored ? w.tuple({nat(teams * atoms * rows), nat(64)}) : w.tuple({nat(teams * rows), nat(pitch)});
    const Def* staged;
    std::tie(mem, staged) = buffer::op_alloc(nat(2), extent, T, panel_mem(fence(mem)))->projs<2>();
    auto nat_xor          = [&](const Def* a, const Def* b) {
        auto i64 = [&](const Def* x) { return w.call<core::bitcast>(w.type_i64(), x); };
        return w.call<core::bitcast>(w.type_nat(), w.call(core::bit2::xor_, w.lit_nat_0(), Defs{i64(a), i64(b)}));
    };
    for (u64 f = 0; f != fragments.count(); ++f) {
        auto value = source->proj(fragments.count(), f);
        for (u64 e = 0; e != 4; ++e) {
            auto [local_row, local_col] = fragment_coords(w, mma_m16n8k8_tf32::l_c, w.tuple({lane, w.lit_idx(4, e)}));
            auto row                    = add(nat(f / ns * fragments.m), local_row);
            auto col                    = add(nat(f % ns * fragments.n), local_col);
            auto element                = w.extract(value, w.lit_idx(4, e));
            if (epilogue) std::tie(mem, element) = epilogue(mem, add(group_row(w, true), row), col, element);
            auto at = w.tuple({idx(teams * rows, add(first, row)), idx(pitch, col)});
            if (stored) {
                auto own_row = add(group_row(w, true), row);
                auto atom    = w.call(core::nat::div, Defs{col, nat(64)});
                auto within  = w.call(core::nat::rem, Defs{col, nat(64)});
                auto chunk   = nat_xor(w.call(core::nat::div, Defs{within, nat(8)}),
                                       w.call(core::nat::rem, Defs{own_row, nat(8)}));
                auto line    = add(add(mul(team, atoms * rows), mul(atom, rows)), own_row);
                at           = w.tuple({idx(teams * atoms * rows, line),
                                        idx(64, add(mul(chunk, 8), w.call(core::nat::rem, Defs{within, nat(8)})))});
            }
            mem = buffer::op_write(mem, staged, at, element)->proj(2, 0);
        }
    }
    if (stored) {
        mem               = fence(w.app(w.annex<gpu::fence_proxy_async>(), mem));
        auto own          = w.call(core::nat::rem, Defs{tid, nat(128)});
        auto issue        = w.call(core::ncmp::e, Defs{own, nat(0)});
        auto i32          = [&](const Def* x) { return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(u64(1) << 32), x); };
        auto [sr, ss, sT] = Axm::as<buffer::Buf>(staged->type())->args<3>();
        auto ds           = Axm::as<buffer::Buf>(destination.buffer->type())->arg(3, 1);
        auto map          = w.app(w.app(w.app(w.annex<gpu::buf_tensor_map>(), T), ds),
                                  {outside(destination.buffer), nat(rows), nat(64)});
        for (u64 a = 0; a != atoms; ++a) {
            auto at
                = w.tuple({idx(teams * atoms * rows, add(mul(team, atoms * rows), nat(a * rows))), idx(64, nat(0))});
            auto src = w.app(w.app(w.app(w.annex<gpu::buf_addr>(), sT), {sr, ss}), {staged, at});
            auto col = i32(add(destination.origin->proj(2, 1), nat(a * 64)));
            mem = w.app(w.annex<gpu::tma_store_if>(), {mem, src, map, col, i32(destination.origin->proj(2, 0)), issue});
        }
        mem = w.app(w.annex<gpu::tma_store_wait_if>(), {mem, issue});
        return w.tuple({mem, destination.buffer});
    }
    mem          = fence(mem);
    auto runs    = cols / 8;
    auto own     = w.call(core::nat::rem, Defs{tid, nat(128)});
    auto from    = buffer::op_vectorize(nat(8), staged);
    auto to      = buffer::op_vectorize(nat(8), destination.buffer);
    auto [h, wd] = destination.shape->projs<2>();
    auto wide    = Lit::as(wd) / 8;
    // The last round's work-items past the runs copy the last run again, as the runs need not split evenly.
    auto total = rows * runs;
    for (u64 k = 0; k != (total + 127) / 128; ++k) {
        auto v = add(nat(k * 128), own);
        if ((k + 1) * 128 > total)
            v = w.extract(w.tuple({nat(total - 1), v}), w.call(core::ncmp::l, Defs{v, nat(total)}));
        auto r      = w.call(core::nat::div, Defs{v, nat(runs)});
        auto c      = w.call(core::nat::rem, Defs{v, nat(runs)});
        auto in     = w.tuple({idx(teams * rows, add(mul(team, rows), r)), idx(pitch / 8, c)});
        auto [m, x] = buffer::op_read(mem, from, in)->projs<2>();
        auto row    = add(destination.origin->proj(2, 0), r);
        auto col    = add(w.call(core::nat::div, Defs{destination.origin->proj(2, 1), nat(8)}), c);
        auto out    = w.tuple({w.call<core::idx>(w.lit_nat_0(), h, row), idx(wide, col)});
        mem         = buffer::op_write(m, to, out, x)->proj(2, 0);
    }
    return w.tuple({mem, destination.buffer});
}

// Reads each lane's accumulator fragments from their rows and columns of the tensor @p init, through a unit-step slice.
std::pair<const Def*, const Def*>
LowerToMem::load_fragments(const Def* init, const Fragments& fragments, const Def* mem) {
    auto& w          = new_world();
    auto nat         = [&](u64 v) { return w.lit_nat(v); };
    auto add         = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
    auto view        = Axm::peel<tensor::buf>(init);
    auto shape       = fe::lookup(storage_shapes_, view);
    auto start       = w.tuple({nat(0), nat(0)});
    const App* slice = Axm::isa<tensor::slice>(view);
    if (slice) {
        auto [meta, s_in, box] = slice->callee()->as<App>()->uncurry_args<3>();
        auto step              = box->proj(3, 1);
        if (Lit::isa(step->proj(2, 0)) != 1 || Lit::isa(step->proj(2, 1)) != 1) slice = nullptr;
    }
    const Def* source;
    if (slice) {
        auto [meta, s_in, box] = slice->callee()->as<App>()->uncurry_args<3>();
        source                 = rewrite(slice->arg());
        shape                  = fe::lookup(storage_shapes_, slice->arg());
        if (!shape) shape = rewrite(s_in);
        start = rewrite(box->proj(3, 0));
    } else {
        source = rewrite(view);
        if (!shape) shape = rewrite(TensorType::isa(view->type())->s);
    }
    auto lane   = lane_index(w);
    auto ns     = fragments.cols / fragments.n;
    auto values = DefVec();
    for (u64 f = 0; f != fragments.count(); ++f) {
        auto elements = DefVec();
        for (u64 e = 0; e != 4; ++e) {
            auto [local_row, local_col] = fragment_coords(w, mma_m16n8k8_tf32::l_c, w.tuple({lane, w.lit_idx(4, e)}));
            auto at    = std::array{add(group_row(w, fragments.group), add(nat(f / ns * fragments.m), local_row)),
                                    add(nat(f % ns * fragments.n), local_col)};
            auto index = w.tuple(DefVec(2, [&](size_t d) {
                return w.call<core::idx>(w.lit_nat_0(), shape->proj(2, d), add(start->proj(2, d), at[d]));
            }));
            const Def* element;
            std::tie(mem, element) = buffer::op_read(mem, source, *Shape(index).fold(shape))->projs<2>();
            elements.push_back(element);
        }
        values.push_back(w.tuple(elements));
    }
    return {mem, w.tuple(values)};
}

// The element of @p input that the loop coordinates @p coords (old-world Nats) reach through its @p map.
// A tensor computed in the warp-owned region is a warp value read at the same position `(lane, j, reg)`, or for a row
// vector `(lane, reg / 2)`; any other input is read from memory.
const Def* LowerToMem::warp_element(const btensor::MapRedPost& mr,
                                    u64 k,
                                    Defs coords,
                                    const Def* lane,
                                    const Def* j,
                                    const Def* reg,
                                    bool post) {
    auto& old = old_world();
    // An epilogue input is read at the output's coordinates.
    auto rn    = Lit::as(post ? mr.Ro : mr.Rn);
    auto nis   = Lit::as(post ? mr.nps : mr.nis);
    auto input = (post ? mr_posts_ : mr_inputs_)->proj(nis, k);
    auto map   = (post ? mr.post_maps : mr.maps)->proj(nis, k);
    auto shape = post ? mr.So : mr.Sr;
    auto Ri    = (post ? mr.Rps : mr.Ris)->proj(nis, k);
    auto Si    = (post ? mr.Sps : mr.Sis)->proj(nis, k);
    auto dims  = map_dims(map, rn);
    if (!dims) input->blame("a warp-owned region reads `{}` through a map that is no projection", input).bail();
    auto value = Axm::peel<tensor::buf>(input);
    // The region's own tensors, its fold's state and results among them, are warp values.
    auto resident = fe::lookup(warp_terms_, value) || Axm::isa<tensor::map_reduce_post>(value)
                 || Axm::isa<tensor::splat>(value)
                 || (value->isa<Extract>() && Axm::isa<tensor::tile_fold>(value->as<Extract>()->tuple()));
    if (resident && j) {
        auto term = warp_term(value);
        auto lead = warp_lead_;
        if (dims->size() == 2 && (*dims)[0] == lead && (*dims)[1] == lead + 1)
            return old.extract(old.extract(old.extract(term, lane), j), reg);
        if (dims->size() == 1 && (*dims)[0] == lead)
            return old.extract(
                old.extract(term, lane),
                old.call<core::idx>(old.lit_nat_0(), old.lit_nat(2),
                                    old.call(core::nat::div, Defs{old.call<core::idx2nat>(reg), old.lit_nat(2)})));
        input->blame("a warp-owned region reads its own `{}` across positions", input).bail();
    }
    if (resident && !j) {
        auto term = warp_term(value);
        if (dims->size() == 1 && (*dims)[0] == 0) return old.extract(old.extract(term, lane), reg);
        input->blame("a warp-owned region reads its own `{}` across positions", input).bail();
    }
    auto loops = old.tuple(
        DefVec(rn, [&](size_t d) { return old.call<core::idx>(old.lit_nat_0(), shape->proj(rn, d), coords[d]); }));
    auto none = old.bot(old.call<mem::M>(0));
    auto at   = btensor::op_affine_map(map, Ri, old.lit_nat(rn), shape, Si, loops, none);
    return op_get_fused(input, at->proj(2, 1));
}

// A tensor of the warp-owned region as an old-world warp value: a `tensor.warp.C` for a `16 × 8nt` tile, an `R` for a
// row vector.
const Def* LowerToMem::warp_term(const Def* def) {
    for (size_t i = 0; i != 128; ++i) {
        auto app = def->isa<App>();
        if (!app) break;
        auto lam = app->callee()->isa<Lam>();
        if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
        def = lam->reduce(app->arg()).back();
    }
    if (auto term = fe::lookup(warp_terms_, def)) return term;
    auto& old = old_world();
    if (auto extract = def->isa<Extract>())
        if (auto fold = Axm::isa<tensor::tile_fold>(extract->tuple()))
            return old.extract(warp_fold(fold), extract->index());
    if (auto splat = Axm::isa<tensor::splat>(Axm::peel<tensor::buf>(def))) {
        auto [shape, value] = splat->args<2>();
        auto T              = value->type();
        if (Lit::isa(shape->arity()) == 1) {
            auto g = old.mut_lam(old.pi(old.sigma({old.type_idx(32), old.type_idx(2)}), T));
            g->set(true, value);
            return warp_terms_[def] = old.app(old.app(old.annex(tensor::warp::r_gen), T), g);
        }
        auto nt = Lit::as(shape->proj(2, 1)) / 8;
        auto g  = old.mut_lam(old.pi(old.sigma({old.type_idx(32), old.type_idx(nt), old.type_idx(4)}), T));
        g->set(true, value);
        return warp_terms_[def] = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g);
    }
    auto app = Axm::isa<tensor::map_reduce_post>(def);
    if (!app) def->blame("a warp-owned region keeps only map_reduce results in registers, not `{}`", def).bail();
    // Fusion gives each consumer its own copy of a producer's fused maps; equal copies share one translation.
    for (auto& [known, term] : warp_terms_)
        if (Axm::isa<tensor::map_reduce_post>(known) && Check::alpha<Check::T>(known, def))
            return warp_terms_[def] = term;
    auto mr = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ro = Lit::isa(mr.Ro), rn = Lit::isa(mr.Rn), nis = Lit::isa(mr.nis), nps = Lit::isa(mr.nps);
    if (!ro || !rn || !nis || !nps)
        def->blame("a warp-owned region needs literal ranks and no epilogue inputs: {}, {}, {}, {}", mr.Ro, mr.Rn,
                   mr.nis, mr.nps)
            .bail();
    auto dim = [&](const Def* shape, u64 n, u64 d) {
        auto extent = Lit::isa(shape->proj(n, d));
        if (!extent) def->blame("a warp-owned region needs literal extents, not `{}`", shape).bail();
        return *extent;
    };
    // A pointwise tile's leading unit dimensions, as a tile of one head has, leave the warp's rows as they are.
    u64 lead = 0;
    while (*rn == *ro && *ro - lead > 1 && dim(mr.So, *ro, lead) == 1)
        ++lead;
    auto own     = *ro - lead;
    auto leading = fe::Restore(warp_lead_, lead);
    if (dim(mr.So, *ro, lead) != (group_rows_ ? 64u : 16u))
        def->blame("a warp-owned region holds 16 rows, a warpgroup-owned 64, not `{}`", mr.So).n(target).bail();
    auto nt = own == 2 ? dim(mr.So, *ro, lead + 1) / 8 : 0;
    if (own == 2 && dim(mr.So, *ro, lead + 1) % 8)
        def->blame("a warp-owned tile's columns are a multiple of 8").n(target).bail();
    auto inputs = app->arg()->proj(2, 0);
    auto scope  = fe::Restore(mr_inputs_, inputs);
    auto posts  = fe::Restore(mr_posts_, app->arg()->proj(2, 1));
    auto input  = [&](u64 k) { return inputs->proj(*nis, k); };
    auto map    = [&](u64 k) { return mr.maps->proj(*nis, k); };
    auto I      = [&](u64 n) { return old.type_idx(n); };
    auto nat    = [&](const Def* x) { return old.call<core::idx2nat>(x); };
    auto tensor = TensorType::isa(def->type());
    auto T      = tensor->T;
    auto call   = [&](const Def* f, const Def* x) { return old.app(cps::op_cps2ds_dep(f), x); };
    auto post   = [&](const Def* x, Defs coords, const Def* lane, const Def* j, const Def* reg) {
        auto ps = old.tuple(DefVec(*nps, [&](size_t k) { return warp_element(mr, k, coords, lane, j, reg, true); }));
        return call(mr.post, old.tuple({x, ps}));
    };
    const Def* term = nullptr;
    if (*rn == *ro) {
        // Pointwise: each position combines its inputs at its own coordinates.
        auto pos    = own == 2 ? old.sigma({I(32), I(nt), I(4)}) : old.sigma({I(32), I(2)});
        auto g      = old.mut_lam(old.pi(pos, T));
        auto lane   = g->var(own + 1, 0);
        auto j      = own == 2 ? g->var(3, 1) : nullptr;
        auto reg    = g->var(own + 1, own);
        auto coords = DefVec(lead, [&](size_t) { return old.lit_nat_0(); });
        if (own == 2) {
            auto [r, c] = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {lane, j, reg})->projs<2>();
            coords.push_back(owned_row(nat(r)));
            coords.push_back(nat(c));
        } else {
            coords.push_back(owned_row(nat(old.app(old.annex(tensor::warp::r_pos), {lane, reg}))));
        }
        auto ins = old.tuple(DefVec(*nis, [&](size_t k) { return warp_element(mr, k, coords, lane, j, reg); }));
        g->set(true, post(call(mr.comb, old.tuple({mr.init, ins})), coords, lane, j, reg));
        term = own == 2 ? old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g)
                        : old.app(old.app(old.annex(tensor::warp::r_gen), T), g);
    } else if (*ro == 1 && *rn == 2) {
        // A reduction over the columns: the warp's row reduction of the combined elements.
        auto op = merge_of(mr.comb);
        if (!op) def->blame("a warp-owned row reduction needs a combiner that merges partials").n(unsafe).bail();
        auto cols = dim(mr.Sr, 2, 1);
        if (cols % 8) def->blame("a warp-owned row reduction runs over a multiple of 8 columns").bail();
        nt                  = cols / 8;
        auto g              = old.mut_lam(old.pi(old.sigma({I(32), I(nt), I(4)}), mr.To));
        auto [lane, j, reg] = g->vars<3>();
        auto [r, c] = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {lane, j, reg})->projs<2>();
        auto coords = DefVec{owned_row(nat(r)), nat(c)};
        auto ins    = old.tuple(DefVec(*nis, [&](size_t k) { return warp_element(mr, k, coords, lane, j, reg); }));
        g->set(true, call(mr.comb, old.tuple({mr.init, ins})));
        auto elems = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), mr.To), g);
        auto rows  = old.app(
            old.app(old.app(old.app(old.annex(tensor::warp::row_fold), old.lit_nat(nt)), mr.To), {op, mr.init}), elems);
        auto p        = old.mut_lam(old.pi(old.sigma({I(32), I(2)}), T));
        auto [pl, ph] = p->vars<2>();
        auto row      = owned_row(nat(old.app(old.annex(tensor::warp::r_pos), {pl, ph})));
        p->set(true, post(old.extract(old.extract(rows, pl), ph), DefVec{row}, pl, nullptr, ph));
        term = old.app(old.app(old.annex(tensor::warp::r_gen), T), p);
    } else if (*ro == 2 && *rn == 3 && *nis == 2 && group_rows_) {
        term              = wgmma_lane(app);
        auto g            = old.mut_lam(old.pi(old.sigma({I(32), I(nt), I(4)}), T));
        auto [pl, pj, pr] = g->vars<3>();
        auto [row, col]   = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {pl, pj, pr})->projs<2>();
        auto acc          = old.extract(old.extract(old.extract(term, pl), pj), pr);
        g->set(true, post(acc, DefVec{owned_row(nat(row)), nat(col)}, pl, pj, pr));
        term = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g);
    } else if (*ro == 2 && *rn == 3 && *nis == 2) {
        if (!is_contraction(app))
            def->blame("a warp-owned product must combine `acc + a · b`, not `{}`", mr.comb).bail();
        // A contraction `c + a·b` over k-chunks of 8 on the tensor cores.
        auto da = map_dims(map(0), 3), db = map_dims(map(1), 3);
        if (!da || !db || da->size() != 2 || db->size() != 2)
            def->blame("a warp-owned product reads projections").bail();
        auto kk = dim(mr.Sr, 3, 2);
        if (kk % 8) def->blame("a warp-owned product reduces a multiple of 8").n(target).bail();
        auto nk      = kk / 8;
        auto F       = T;
        auto a_value = Axm::peel<tensor::buf>(input(0));
        auto from_c  = Axm::isa<tensor::map_reduce_post>(a_value) && *da == std::vector<u64>{0, 2};
        const Def* a;
        if (auto hoisted = fe::lookup(hoisted_a_, def)) {
            a = hoisted;
        } else if (from_c) {
            a = old.app(old.app(old.annex(tensor::warp::a_from_c), old.lit_nat(nk)), warp_term(a_value));
        } else {
            auto g               = old.mut_lam(old.pi(old.sigma({I(nk), I(32), I(4)}), F));
            auto [kc, lane, reg] = g->vars<3>();
            auto [r, k]
                = old.app(old.app(old.annex(tensor::warp::a_pos), old.lit_nat(nk)), {kc, lane, reg})->projs<2>();
            auto coords = DefVec{nat(r), old.lit_nat_0(), nat(k)};
            g->set(true, warp_element(mr, 0, coords, lane, nullptr, reg));
            a = old.app(old.app(old.app(old.annex(tensor::warp::a_gen), old.lit_nat(nk)), F), g);
        }
        auto gb                  = old.mut_lam(old.pi(old.sigma({I(nk), I(nt), I(32), I(2)}), F));
        auto [kc, jb, lane, reg] = gb->vars<4>();
        auto pos                 = old.app(old.app(old.annex(tensor::warp::b_pos), {old.lit_nat(nk), old.lit_nat(nt)}),
                                           old.lit_bool(from_c));
        auto [k, n]              = old.app(pos, {kc, jb, lane, reg})->projs<2>();
        gb->set(true, warp_element(mr, 1, DefVec{old.lit_nat_0(), nat(n), nat(k)}, lane, nullptr, reg));
        auto b = old.app(old.app(old.app(old.annex(tensor::warp::b_gen), {old.lit_nat(nk), old.lit_nat(nt)}), F), gb);
        const Def* c;
        if (auto outs = tensor_init(mr)) {
            c = warp_term(outs->outs);
        } else {
            auto g = old.mut_lam(old.pi(old.sigma({I(32), I(nt), I(4)}), F));
            g->set(true, mr.init);
            c = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), F), g);
        }
        term = old.app(old.app(old.annex(tensor::warp::c_mma), {old.lit_nat(nk), old.lit_nat(nt)}), {a, b, c});
        // An epilogue fused into the product applies at each position of the accumulated tile.
        auto g            = old.mut_lam(old.pi(old.sigma({I(32), I(nt), I(4)}), T));
        auto [pl, pj, pr] = g->vars<3>();
        auto [row, col]   = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {pl, pj, pr})->projs<2>();
        auto acc          = old.extract(old.extract(old.extract(term, pl), pj), pr);
        g->set(true, post(acc, DefVec{owned_row(nat(row)), nat(col)}, pl, pj, pr));
        term = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g);
    } else {
        def->blame("a warp-owned region has no translation for `{}`", def).bail();
    }
    return warp_terms_[def] = term;
}

const Def* LowerToMem::owned_row(const Def* row) {
    auto& old = old_world();
    return group_rows_ ? old.call(core::nat::add, Defs{group_rows_, row}) : row;
}

// The A fragments of a warpgroup-owned product's `nk` chunks of 16 keys, each lane's from its warp's rows.
const Def* LowerToMem::wgmma_a(const btensor::MapRedPost& mr, u64 nk) {
    auto& old = old_world();
    auto H    = mr.Tis->proj(2, 0);
    auto nat  = [&](const Def* x) { return old.call<core::idx2nat>(x); };
    auto g = old.mut_lam(old.pi(old.sigma({old.type_idx(nk), old.type_idx(32), old.type_idx(4), old.type_idx(2)}), H));
    auto [kc, lane, reg, h] = g->vars<4>();
    auto [r, k]             = old.app(old.annex(tensor::mma_m16n8k16::l_a), {lane, reg, h})->projs<2>();
    auto key = old.call(core::nat::add, Defs{old.call(core::nat::mul, Defs{nat(kc), old.lit_nat(16)}), nat(k)});
    g->set(true, warp_element(mr, 0, DefVec{owned_row(nat(r)), old.lit_nat_0(), key}, lane, nullptr, reg));
    return old.app(old.app(old.app(old.annex(tensor::warp::a16_gen), old.lit_nat(nk)), H), g);
}

// A warpgroup-owned product `c + up a · up b` over chunks of 16 keys: each lane passes its A fragments and accumulator
// to `wgmma_lane`, which reads B from its panel; the lanes' results are kept, so a row reduction shuffles them.
const Def* LowerToMem::wgmma_lane(const App* app) {
    auto& old = old_world();
    auto mr   = btensor::MapRedPost::of(app->callee()->as<App>());
    auto H    = contraction_type(app);
    if (!H || H == old.annex<math::F32>())
        app->blame("a warpgroup-owned product must combine 16-bit inputs as `acc + up a · up b`, not `{}`", mr.comb)
            .bail();
    auto lit = [&](const Def* shape, u64 n, u64 d) {
        auto extent = Lit::isa(shape->proj(n, d));
        if (!extent) app->blame("a warpgroup-owned product needs literal extents, not `{}`", shape).bail();
        return *extent;
    };
    auto nt = lit(mr.Sr, 3, 1) / 8, kk = lit(mr.Sr, 3, 2);
    if (kk % 16) app->blame("a warpgroup-owned product reduces a multiple of 16").n(target).bail();
    auto nk     = kk / 16;
    auto inputs = app->arg()->proj(2, 0);
    auto scope  = fe::Restore(mr_inputs_, inputs);
    auto nat    = [&](const Def* x) { return old.call<core::idx2nat>(x); };
    auto self   = warp_self_;
    auto da     = map_dims(mr.maps->proj(2, 0), 3);
    auto a_val  = Axm::peel<tensor::buf>(inputs->proj(2, 0));
    const Def* a;
    if (auto hoisted = fe::lookup(hoisted_a_, app)) {
        a = hoisted;
    } else if (Axm::isa<tensor::map_reduce_post>(a_val) && da && *da == std::vector<u64>{0, 2}) {
        // Accumulator tiles `2 kc` and `2 kc + 1` are A's chunk `kc` (`mma_m16n8k16.c_pair_is_a`).
        auto p  = warp_term(a_val);
        auto np = 2 * nk;
        auto g
            = old.mut_lam(old.pi(old.sigma({old.type_idx(nk), old.type_idx(32), old.type_idx(4), old.type_idx(2)}), H));
        auto [kc, lane, reg, h] = g->vars<4>();
        auto tile = old.call(core::nat::add, Defs{old.call(core::nat::mul, Defs{nat(kc), old.lit_nat(2)}),
                                                  old.call(core::nat::div, Defs{nat(reg), old.lit_nat(2)})});
        auto elem
            = old.call(core::nat::add,
                       Defs{old.call(core::nat::mul,
                                     Defs{old.call(core::nat::rem, Defs{nat(reg), old.lit_nat(2)}), old.lit_nat(2)}),
                            nat(h)});
        g->set(true, old.extract(
                         old.extract(old.extract(p, lane), old.call<core::idx>(old.lit_nat_0(), old.lit_nat(np), tile)),
                         old.call<core::idx>(old.lit_nat_0(), old.lit_nat(4), elem)));
        a = old.app(old.app(old.app(old.annex(tensor::warp::a16_gen), old.lit_nat(nk)), H), g);
    } else {
        a = wgmma_a(mr, nk);
    }
    auto b = Axm::peel<tensor::buf>(inputs->proj(2, 1));
    if (!Axm::isa<tensor::stage>(Axm::peel<tensor::placed>(b)))
        app->blame("a warpgroup-owned product reads B from a work-group panel").bail();
    auto Ri = mr.Ris->proj(2, 1), Si = mr.Sis->proj(2, 1);
    auto none = old.bot(old.call<mem::M>(0));
    auto at   = old.tuple(DefVec(nk, [&](size_t kc) {
        auto loops = old.tuple(DefVec(3, [&](size_t d) {
            return old.call<core::idx>(old.lit_nat_0(), mr.Sr->proj(3, d), old.lit_nat(d == 2 ? 16 * kc : 0));
        }));
        return btensor::op_affine_map(mr.maps->proj(2, 1), Ri, old.lit_nat(3), mr.Sr, Si, loops, none)->proj(2, 1);
    }));
    const Def* c;
    if (auto outs = tensor_init(mr)) {
        c = warp_term(outs->outs);
    } else {
        auto g = old.mut_lam(old.pi(old.sigma({old.type_idx(32), old.type_idx(nt), old.type_idx(4)}), mr.To));
        g->set(true, mr.init);
        c = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), mr.To), g);
    }
    auto own_a
        = old.tuple(DefVec(nk, [&](size_t kc) { return old.extract(old.extract(a, old.lit_idx(nk, kc)), self); }));
    auto callee
        = old.app(old.app(old.app(old.annex<tensor::wgmma_lane>(), H), {old.lit_nat(nk), old.lit_nat(nt), Ri}), Si);
    auto own = old.app(callee, {own_a, b, at, old.extract(c, self)});
    // Two products may read one panel along different dims, so each use records its own K dim.
    if (auto db = map_dims(mr.maps->proj(2, 1), 3))
        for (size_t d = 0; d != db->size(); ++d)
            if ((*db)[d] == 2) wgmma_kdims_[own] = d;
    auto lanes = old.pack(32, own);
    return old.app(old.app(old.annex<tensor::warp_keep>(), own->type()), lanes);
}

// A fold in a warp-owned region keeps its state as each lane's components, carried by the loop; the step sees them as
// the warp values `gpu.lanes (self, x)`. Its results are such warp values too.
const Def* LowerToMem::warp_fold(const App* app) {
    if (auto terms = fe::lookup(warp_terms_, app)) return terms;
    auto& w                                             = new_world();
    auto& old                                           = old_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto step                                           = app->arg()->isa<Lam>();
    auto sigma                                          = type->isa_imm<Sigma>();
    if (!step || !step->is_set() || !sigma) app->blame("a warp-owned fold needs a product state and a step").bail();
    auto n                     = sigma->num_ops();
    auto parameter             = old.mut_lam(old.pi(old.sigma({type, old.type_nat()}), old.type_nat()));
    auto [accumulator, offset] = parameter->var()->projs<2>();
    auto input                 = region_input(inputs);
    auto root                  = old.app(step->reduce(parameter->var()).back(), input);
    auto lane                  = rewrite(warp_self_);
    auto lanes = [&](const Def* x) { return old.app(old.app(old.annex<gpu::lanes>(), x->type()), {warp_self_, x}); };
    auto inits = DefVec(n, [&](size_t i) { return warp_term(init->proj(n, i)); });
    auto local = old.sigma(DefVec(n, [&](size_t i) { return old.extract(inits[i], warp_self_)->type(); }));
    auto first = w.tuple(DefVec(n, [&](size_t i) { return w.extract(rewrite(inits[i]), lane); }));
    // A product's A operand read from memory that the step does not change is loaded once, before the loop.
    auto hoisted = fe::Restore(hoisted_a_, hoisted_a_);
    // The loop carries the loaded values unchanged, so their loads precede it.
    auto carried_values = DefVec();
    auto proxies        = std::vector<Lam*>();
    {
        auto pending = DefVec{root};
        auto seen    = DefSet();
        for (size_t i = 0; i != pending.size(); ++i) {
            auto def = pending[i];
            if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>()) continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
            auto product = Axm::isa<tensor::map_reduce_post>(def);
            auto H       = product ? contraction_type(product) : nullptr;
            if (!H || (H != old.annex<math::F32>()) != bool(group_rows_)) continue;
            auto mr = btensor::MapRedPost::of(product->callee()->as<App>());
            // A warpgroup's products read B through descriptors, from a panel in the layout their maps ask for.
            if (auto b = Axm::peel<tensor::placed>(Axm::peel<tensor::buf>(product->arg()->proj(2, 0)->proj(2, 1)));
                group_rows_ && Axm::isa<tensor::stage>(b) && !placed_layouts_.contains(b)) {
                auto [rank, shape, T] = TensorType::decode(b->type()->as<Nom>());
                auto r                = Lit::as(rank);
                auto db               = map_dims(mr.maps->proj(2, 1), 3);
                auto bytes            = Lit::as(core::op(core::trait::size, T));
                auto choice = panel::wgmma(rewrite(shape), r - 1, bytes, db && db->size() == r && db->back() == 2);
                if (choice.expr) placed_layouts_[b] = w.tuple({rewrite(shape), choice.expr});
            }
            auto input  = product->arg()->proj(2, 0)->proj(2, 0);
            auto source = Axm::peel<tensor::buf>(input);
            if (Axm::isa<tensor::map_reduce_post>(source) || source->has_free_var(parameter->has_var())) continue;
            auto nk = Lit::isa(mr.Sr->proj(3, 2));
            if (group_rows_) {
                if (!nk || *nk % 16) continue;
                *nk /= 16;
                auto scope = fe::Restore(mr_inputs_, product->arg()->proj(2, 0));
                auto a     = rewrite(wgmma_a(mr, *nk));
                auto loaded
                    = w.tuple(DefVec(*nk, [&](size_t c) { return w.extract(w.extract(a, w.lit_idx(*nk, c)), lane); }));
                auto proxy = old.mut_lam(
                    old.pi(old.arr({old.lit_nat(*nk), old.lit_nat(4), old.lit_nat(2)}, H), old.type_nat()));
                bind_old(proxy->has_var());
                carried_values.push_back(loaded);
                proxies.push_back(proxy);
                auto at = old.mut_lam(
                    old.pi(old.sigma({old.type_idx(*nk), old.type_idx(32), old.type_idx(4), old.type_idx(2)}), H));
                auto [ac, al, ar, ah] = at->vars<4>();
                at->set(true, old.extract(old.extract(old.extract(old.extract(lanes(proxy->var()), al), ac), ar), ah));
                hoisted_a_[product]
                    = old.app(old.app(old.app(old.annex(tensor::warp::a16_gen), old.lit_nat(*nk)), H), at);
                continue;
            }
            if (!nk || *nk % 8) continue;
            *nk /= 8;
            auto F     = mr.To;
            auto scope = fe::Restore(mr_inputs_, product->arg()->proj(2, 0));
            auto g     = old.mut_lam(old.pi(old.sigma({old.type_idx(*nk), old.type_idx(32), old.type_idx(4)}), F));
            auto [kc, l, reg] = g->vars<3>();
            auto [r, k] = old.app(old.app(old.annex(tensor::warp::a_pos), old.lit_nat(*nk)), {kc, l, reg})->projs<2>();
            auto nat    = [&](const Def* x) { return old.call<core::idx2nat>(x); };
            g->set(true, warp_element(mr, 0, DefVec{nat(r), old.lit_nat_0(), nat(k)}, l, nullptr, reg));
            auto a      = rewrite(old.app(old.app(old.app(old.annex(tensor::warp::a_gen), old.lit_nat(*nk)), F), g));
            auto loaded = w.tuple(DefVec(*nk, [&](size_t c) {
                auto chunk = w.extract(w.extract(a, w.lit_idx(*nk, c)), lane);
                return w.tuple(DefVec(4, [&](size_t e) { return w.extract(chunk, w.lit_idx(4, e)); }));
            }));
            auto proxy  = old.mut_lam(old.pi(old.arr({old.lit_nat(*nk), old.lit_nat(4)}, F), old.type_nat()));
            bind_old(proxy->has_var());
            carried_values.push_back(loaded);
            proxies.push_back(proxy);
            auto at = old.mut_lam(old.pi(old.sigma({old.type_idx(*nk), old.type_idx(32), old.type_idx(4)}), F));
            auto [ac, al, ar] = at->vars<3>();
            at->set(true, old.extract(old.extract(old.extract(lanes(proxy->var()), al), ac), ar));
            hoisted_a_[product] = old.app(old.app(old.app(old.annex(tensor::warp::a_gen), old.lit_nat(*nk)), F), at);
        }
    }
    // The panels a step stages are filled into one buffer while the step computes on the other, one block ahead.
    auto literal_count = Lit::isa(count);
    auto stages        = literal_count && *literal_count >= 2 ? staged_panels(root, parameter->var()) : DefVec();
    auto blocks_of     = [&](const Def* var) { return staged_panels(old.app(step->reduce(var).back(), input), var); };
    const Def* mem     = fresh_mem();
    auto ring          = std::vector<DefVec>();
    auto pipelined     = std::vector<size_t>();
    if (!stages.empty()) {
        auto probe = old.mut_lam(parameter->type()->as<Pi>());
        if (auto block = Lit::isa(stride)) multiples_[probe->var(2, 1)] = *block;
        auto probes = blocks_of(probe->var());
        if (probes.size() != stages.size()) app->blame("pipelined warp fold found {} panels", probes.size()).bail();
        // Each block's instance of a panel takes the layout chosen for the step's.
        for (size_t i = 0; i != stages.size(); ++i)
            if (auto layout = fe::lookup(placed_layouts_, stages[i])) placed_layouts_[probes[i]] = layout;
        auto memo    = fe::Restore(fresh_memo_, {});
        auto probing = fe::Restore(probe_fill_, true);
        push();
        bind_old(probe->has_var());
        map(probe->var(2, 1), w.lit_nat_0());
        DefVec kept, firsts;
        for (size_t i = 0; i != probes.size(); ++i) {
            auto panel = rewrite(probes[i]);
            if (!shared_fills_.contains(panel)) continue;
            if (auto shape = fe::lookup(storage_shapes_, probes[i])) storage_shapes_[stages[i]] = shape;
            kept.push_back(stages[i]);
            pipelined.push_back(i);
            firsts.push_back(panel);
        }
        pop();
        stages = kept;
        if (!firsts.empty()) {
            ring.push_back(firsts);
            DefVec buffers;
            for (auto panel : firsts) {
                auto [r, shape, T] = Axm::as<buffer::Buf>(panel->type())->args<3>();
                const Def* other;
                std::tie(mem, other)
                    = buffer::op_alloc(r, shape, T, w.app(w.annex<gpu::shared_scope>(), mem))->projs<2>();
                forward_layouts(panel, other);
                buffers.push_back(other);
            }
            ring.push_back(buffers);
        }
    }
    auto panels     = ring.empty() ? size_t(0) : ring[0].size();
    auto fill_block = [&](const Def* at, const Def* block, const DefVec& targets) {
        auto future = old.mut_lam(parameter->type()->as<Pi>());
        if (auto size = Lit::isa(stride)) multiples_[future->var(2, 1)] = *size;
        auto roots = blocks_of(future->var());
        auto async = fe::Restore(async_fill_, true);
        push();
        bind_old(future->has_var());
        map(future->var(2, 1), w.call(core::nat::mul, Defs{block, rewrite(stride)}));
        for (size_t j = 0; j != panels; ++j) {
            if (auto layout = fe::lookup(placed_layouts_, stages[j])) placed_layouts_[roots[pipelined[j]]] = layout;
            fill_target_ = {at, targets[j]};
            filled_mem_  = nullptr;
            rewrite(roots[pipelined[j]]);
            if (!filled_mem_)
                roots[pipelined[j]]->blame("a pipelined panel was not filled into its ring buffer").bail();
            at = filled_mem_;
        }
        pop();
        return at;
    };
    auto proxied = fe::Restore(proxied_, proxied_ || (panels != 0 && group_rows_));
    // One step of the fold at block `iteration`: the warp's terms from the carried ones and, if staged, a slot's
    // panels.
    auto fold_step = [&](const Def* iteration, const Def* locals, const Def* kept, const Def* slot) {
        map(offset, w.call(core::nat::mul, Defs{iteration, rewrite(stride)}));
        auto proxy = old.mut_lam(old.pi(local, old.type_nat()));
        bind_old(proxy->has_var());
        map(proxy->var(), locals);
        for (size_t i = 0; i != proxies.size(); ++i)
            map(proxies[i]->var(), kept->proj(proxies.size(), i));
        for (size_t j = 0; j != panels; ++j) {
            map(stages[j], slot->proj(panels, j));
            forward_layouts(ring[0][j], slot->proj(panels, j));
        }
        for (size_t i = 0; i != n; ++i)
            warp_terms_[accumulator->proj(n, i)] = lanes(proxy->var(n, i));
        for (size_t i = 0; i != 128; ++i) {
            auto call = root->isa<App>();
            if (!call) break;
            auto lam = call->callee()->isa<Lam>();
            if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
            root = lam->reduce(call->arg()).back();
        }
        return w.tuple(DefVec(n, [&](size_t i) { return w.extract(rewrite(warp_term(root->proj(n, i))), lane); }));
    };
    const Def* folded;
    if (panels != 0) {
        // `gpu.ring_fold` is the protocol of the two-slot `cp.async` ring; a block's fill and step come from here.
        auto M       = w.call<mem::M>(0);
        auto slot_ty = w.sigma(DefVec(panels, [&](size_t j) { return ring[0][j]->type(); }));
        auto kept    = w.tuple(carried_values);
        auto acc_ty  = w.sigma({first->type(), kept->type()});
        auto closure = [&](Lam* fun, const std::function<const Def*()>& body) {
            auto pending = fe::Restore(pending_, {});
            auto memo    = fe::Restore(fresh_memo_, {});
            auto cache   = fe::Restore(realizations_, realizations_);
            auto terms   = fe::Restore(warp_terms_, warp_terms_);
            push();
            bind_new(fun->has_var());
            fun->app(true, fun->ret_var(), body());
            fun->set(true, wrap_fresh_mem(fun->body()));
            pop();
        };
        auto fill = w.mut_fun(w.sigma({M, w.type_nat(), w.type_nat(), slot_ty}), M)->set("ringFill");
        closure(fill, [&] {
            auto [at, block, nth, slot] = fill->var(2, 0)->projs<4>();
            auto targets                = DefVec(panels, [&](size_t j) { return slot->proj(panels, j); });
            for (size_t j = 0; j != panels; ++j)
                forward_layouts(ring[0][j], targets[j]);
            return fill_block(at, block, targets);
        });
        auto ring_step = w.mut_fun(w.sigma({M, acc_ty, w.type_nat(), slot_ty}), w.sigma({M, acc_ty}))->set("ringStep");
        closure(ring_step, [&]() -> const Def* {
            auto [at, acc, iteration, slot] = ring_step->var(2, 0)->projs<4>();
            bind_old(parameter->has_var());
            auto next = fold_step(iteration, acc->proj(2, 0), acc->proj(2, 1), slot);
            return w.tuple({at, w.tuple({next, acc->proj(2, 1)})});
        });
        auto depth  = w.lit_nat(2);
        auto refill = w.mut_lam(w.pi(w.type_nat(), w.type_nat()))->set("ringRefill");
        refill->set(true, w.app(w.annex(tensor::pipeline_plan::refill_block),
                                {refill->var(), depth, w.lit_nat(*literal_count)}));
        auto rid  = w.lit_nat(rings_++);
        auto fold = w.app(w.annex<gpu::ring_fold>(), {acc_ty, slot_ty, rid, depth});
        auto sync = w.app(w.app(w.annex<gpu::async_ring>(), {rid, depth}),
                          {w.lit_nat_0(), w.lit_bool(group_rows_ != nullptr)});
        fold      = w.app(fold, {sync, w.call<core::bitcast>(w.type_i64(), rewrite(count)), refill, w.lit_nat_0(),
                                 w.lit_ff(), w.lit_bool(!group_rows_)});
        fold      = w.app(w.app(fold, {mem, w.tuple({first, kept}), w.tuple({w.tuple(ring[0]), w.tuple(ring[1])})}),
                          {fill, ring_step});
        folded    = fold->proj(2, 1)->proj(2, 0);
    } else {
        auto state                    = w.tuple({mem, first, w.tuple(carried_values)});
        auto [run, call]              = btensor::ds_splice(state, state->type(), "warpFold");
        auto [loop, loop_call]        = affine::counting_for(w.call<core::bitcast>(w.type_i64(), rewrite(count)),
                                                             run->var(2, 0), run->ret_var(), w.sym("warpFoldStep"));
        auto [index, carried, yield]  = loop->vars<3>();
        auto [loop_mem, locals, kept] = carried->projs<3>();
        {
            auto pending = fe::Restore(pending_, {});
            auto memo    = fe::Restore(fresh_memo_, {});
            auto cache   = fe::Restore(realizations_, realizations_);
            auto terms   = fe::Restore(warp_terms_, warp_terms_);
            push();
            bind_old(parameter->has_var());
            auto next = fold_step(w.call<core::bitcast>(w.type_nat(), index), locals, kept, nullptr);
            loop->app(true, yield, w.tuple({loop_mem, next, kept}));
            loop->set(true, wrap_fresh_mem(loop->body()));
            pop();
        }
        run->set(true, loop_call);
        folded = call->proj(3, 1);
    }
    auto result = old.mut_lam(old.pi(local, old.type_nat()));
    bind_old(result->has_var());
    map(result->var(), folded);
    return warp_terms_[app] = old.tuple(DefVec(n, [&](size_t i) { return lanes(result->var(n, i)); }));
}

// Each lane writes its components of the warp-owned region's root at their coordinates of @p destination.
const Def* LowerToMem::store_warp(const Def* root, const Destination& destination) {
    auto& w    = new_world();
    auto& old  = old_world();
    auto value = rewrite(warp_term(root));
    auto lane  = rewrite(warp_self_);
    auto rank  = Lit::as(destination.shape->arity());
    auto mem   = destination.mem;
    auto at    = [&](const DefVec& coords) {
        return w.tuple(DefVec(rank, [&](size_t d) {
            auto coord = w.call(core::nat::add, Defs{destination.origin->proj(rank, d), coords[d]});
            return w.call<core::idx>(w.lit_nat_0(), destination.shape->proj(rank, d), coord);
        }));
    };
    auto nat = [&](const Def* x) { return w.call<core::idx2nat>(x); };
    // A destination's leading unit dimensions, as a tile of one head has, take coordinate 0.
    auto lead = rank > 2 ? rank - 2 : u64(0);
    auto with = [&](DefVec coords) {
        coords.insert(coords.begin(), lead, w.lit_nat_0());
        return at(coords);
    };
    if (rank >= 2) {
        auto cols = Lit::as(destination.extent->proj(rank, rank - 1));
        auto nt   = cols / 8;
        for (u64 j = 0; j != nt; ++j)
            for (u64 reg = 0; reg != 4; ++reg) {
                auto jj = w.lit_idx(nt, j), rr = w.lit_idx(4, reg);
                auto [r, c] = w.app(w.app(w.annex(tensor::warp::c_pos), w.lit_nat(nt)), {lane, jj, rr})->projs<2>();
                auto elem   = w.extract(w.extract(w.extract(value, lane), jj), rr);
                auto row    = group_rows_ ? w.call(core::nat::add, Defs{rewrite(group_rows_), nat(r)}) : nat(r);
                mem = buffer::op_write(mem, destination.buffer, *Shape(with({row, nat(c)})).fold(destination.shape),
                                       elem)
                          ->proj(2, 0);
            }
    } else {
        for (u64 h = 0; h != 2; ++h) {
            auto hh  = w.lit_idx(2, h);
            auto row = nat(w.app(w.annex(tensor::warp::r_pos), {lane, hh}));
            if (group_rows_) row = w.call(core::nat::add, Defs{rewrite(group_rows_), row});
            auto elem = w.extract(w.extract(value, lane), hh);
            mem       = buffer::op_write(mem, destination.buffer, *Shape(at({row})).fold(destination.shape), elem)
                            ->proj(2, 0);
        }
    }
    (void)old;
    return w.tuple({mem, destination.buffer});
}

} // namespace mim::plug::tensor::phase
