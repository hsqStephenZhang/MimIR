#include "mim/plug/btensor/phase/lower_map_reduce.h"

#include <algorithm>
#include <ranges>

#include <mim/axm.h>
#include <mim/def.h>
#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/btensor/btensor.h"

namespace mim::plug::btensor::phase {

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (Axm::isa<btensor::map_reduce_post>(app)) return lower_map_reduce_post(app);
    if (Axm::isa<btensor::broadcast>(app)) return lower_broadcast(app);
    if (Axm::isa<btensor::pad>(app)) return lower_pad(app);
    if (Axm::isa<btensor::concat>(app)) return lower_concat(app);
    if (Axm::isa<buffer::lit>(app)) return lower_buffer_lit(app);
    if (Axm::isa<btensor::gather>(app)) return lower_gather(app);
    if (Axm::isa<btensor::scatter>(app)) return lower_scatter(app);
    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerMapReduce::lower_buffer_lit(const App* app) {
    // `buffer.lit (r, s, T) (mem, val)` fills every elem with `val`. Emit a fill loop (via the same
    // pointwise scaffold as pad/concat) so the store never materializes as one giant literal array. A
    // non-literal rank has no static loop nest, so leave it for `buffer.lower_ptr`'s monolithic fallback.
    auto [r, s, T] = app->callee()->as<App>()->args<3>();
    auto s_out     = rewrite(s);
    auto rn        = Lit::isa<u64>(rewrite(r));
    if (!rn) return RWPhase::rewrite_imm_App(app);

    auto [mem, val]      = rewrite(app->arg())->projs<2>();
    auto result_ty       = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s, T)]
    auto [obr, obs, obT] = Axm::isa<buffer::Buf>(result_ty->proj(2, 1))->args<3>();
    auto [a_mem, buf]    = buffer::op_alloc(obr, obs, obT, mem)->projs<2>();
    // `compute` ignores the loop counters and writes the (loop-invariant) scalar `ins` everywhere.
    return build_pointwise(
        result_ty, a_mem, buf, val, s_out, *rn, "constant_fill",
        [](const DefVec&, const Def* ins, const Def* m) -> std::pair<const Def*, const Def*> { return {m, ins}; });
}

namespace {

/// The value of type @p T whose bytes are all zero, built from its scalars; `nullptr` where a scalar has no size.
const Def* zero_of(const Def* T) {
    auto& w = T->world();
    if (auto arr = T->isa<Arr>()) {
        auto elem = zero_of(arr->body());
        return elem ? w.pack(arr->arity(), elem) : nullptr;
    }
    if (auto sigma = T->isa_imm<Sigma>()) {
        auto ops = DefVec(sigma->num_ops());
        for (size_t i = 0; i != ops.size(); ++i)
            if (!(ops[i] = zero_of(sigma->op(i)))) return nullptr;
        return w.tuple(sigma, ops);
    }
    auto bytes = Lit::isa<u64>(w.call(core::trait::size, T));
    if (!bytes || *bytes == 0 || *bytes > 8) return nullptr;
    return w.call<core::bitcast>(T, w.lit_idx(Idx::bitwidth2size(*bytes * 8), 0));
}

} // namespace

/// Reads each input the schedule caches from a copy of its box in a buffer allocated on @p mem, put into @p caches.
/// The copy runs as the schedule's action inside the loop of the input's `at_dim` or ahead of the nest.
/// @returns the memory after the allocations and the copies ahead of the nest.
const Def* LowerMapReduce::cache_inputs(MapRedPost& mr, const Def* mem, DefVec& inputs, DefVec& caches) {
    using F    = MapRedPost;
    auto& w    = new_world();
    auto rks   = *mr.ranks();
    auto rn    = rks.nloops();
    auto rec   = mr.sched_record();
    auto field = [&](F::Field f) { return rec->proj(F::Num_Fields, f); };
    auto bit   = [](u64 mask, u64 i) { return i < 64 && (mask >> i) & 1; };
    auto cache = Lit::isa<u64>(field(F::Cache)).value_or(0);
    auto slots = Lit::isa<u64>(field(F::AtSlots)).value_or(0);
    auto packs = Lit::isa<u64>(field(F::Prepacks)).value_or(0);
    if (cache == 0) return mem;

    auto fields = DefVec(F::Num_Fields, [&](size_t f) { return field(F::Field(f)); });
    auto Ris    = DefVec(rks.nis, [&](size_t i) { return mr.Ris->proj(rks.nis, i); });
    auto Sis    = DefVec(rks.nis, [&](size_t i) { return mr.Sis->proj(rks.nis, i); });
    auto maps   = DefVec(rks.nis, [&](size_t i) { return mr.maps->proj(rks.nis, i); });
    auto mem_ty = w.call<mem::M>(0);
    auto i64    = w.type_i64();
    auto lit    = [&](const Def* n) { return w.app(w.annex<affine::lit>(), n); };
    // The new mutables become external only once rewritten.
    auto param = [&](const Def* d) {
        if (auto e = d->isa<Extract>()) d = e->tuple();
        if (auto var = d->isa<Var>())
            return std::ranges::any_of(old_world().externals().muts(),
                                       [&](Def* ext) { return lookup(ext) == var->binder(); });
        return false;
    };
    auto size = [](const Def* shape, u64 n) {
        auto s = u64(1);
        for (auto e : shape->projs(n))
            s *= Lit::isa<u64>(e).value_or(0);
        return s;
    };

    auto copied = std::vector<bool>(rks.nis);
    for (u64 i = 0; i != rks.nis; ++i) {
        if (!bit(cache, i)) continue;
        auto at = bit(slots, i);
        auto give_up
            = [&](const char* why) { log().w("cache_read of input {} of {} not realized: {}", i, mr.map_out, why); };
        auto dim = Lit::isa<u64>(w.app(field(F::AtDim), w.lit_nat(i)));
        if (at && (!dim || *dim >= rks.rp + Lit::isa<u64>(field(F::Hoist)).value_or(0))) {
            give_up("not a parallel or a hoisted reduction dim");
            continue;
        }
        auto fixed = at ? mr.outer(*dim, rn) : std::vector<bool>(rn);
        auto R     = mr.Ris->proj(rks.nis, i);
        auto T     = mr.Tis->proj(rks.nis, i);
        auto r     = Lit::isa<u64>(R);
        if (!r) {
            give_up("the input has no literal rank");
            continue;
        }

        // The copy of the box the loops inside the attach point read, given the loop-vector prefix.
        auto fill          = w.mut_fun(w.sigma({mem_ty, w.pi(w.type_nat(), w.type_i32())}), mem_ty)->set("cache_fill");
        auto [fdata, fret] = fill->vars<2>();
        auto [fmem, get]   = fdata->projs<2>();
        // Sized regardless of the input's end, so that a padded vector loop finds whole runs in the copy.
        auto wide  = w.tuple(DefVec(*r, [&](size_t) { return w.lit_nat(u64(1) << 32); }));
        auto image = [&](auto ext) {
            auto lo  = DefVec(rn, [&](size_t d) -> const Def* {
                return fixed[d] ? w.call<core::bitcast>(w.type_nat(), w.app(get, w.lit_nat(d))) : w.lit_nat_0();
            });
            auto box = w.tuple({w.tuple(lo), w.tuple(DefVec(rn, ext)), wide});
            return w.app(w.app(w.app(w.annex<affine::image>(), {mr.Rn, R}), maps[i]), box);
        };
        auto box   = image([&](size_t d) { return fixed[d] ? w.lit_nat_1() : mr.Sr->proj(rn, d); });
        auto point = image([&](size_t) { return w.lit_nat_1(); });
        // An inexact image of the corner's single point does not pin down where the read starts.
        if (!std::ranges::all_of(point->proj(4, 3)->projs(*r), [&](const Def* e) { return e == w.lit_tt(); })) {
            give_up("the read of the corner is not known exactly");
            continue;
        }
        if (box->proj(4, 0) != point->proj(4, 0)) {
            give_up("the box does not start at the read of the corner");
            continue;
        }
        auto lo   = box->proj(4, 0);
        auto caps = DefVec(*r, [&](size_t a) { return box->proj(4, 2)->proj(*r, a); });
        if (!std::ranges::all_of(caps, [](const Def* c) { return Lit::isa(c).has_value(); })) {
            give_up("the box has no literal size");
            continue;
        }

        // A layout splits axis `ax` of the box by `by`: the run of a cell's split comes last, its block first.
        auto by     = Lit::isa<u64>(w.app(field(F::LayoutBy), w.lit_nat(i))).value_or(0);
        auto layout = by > 1;
        auto ax    = Lit::isa<u64>(w.app(field(F::LayoutAx), w.lit_nat(i))).value_or(0);
        // The dims inside the box that move each axis of the read.
        auto map_pi = maps[i]->type()->as<Pi>();
        auto mo     = w.mut_lam(map_pi->dom(), map_pi->codom())->var();
        auto moves  = [&](u64 d, u64 a) {
            auto still = w.tuple(DefVec(rn, [&](size_t e) { return e == d ? lit(w.lit_nat_0()) : mo->proj(rn, e); }));
            return w.app(maps[i], mo)->proj(*r, a) != w.app(maps[i], still)->proj(*r, a);
        };
        // The vector loop is the vector dim, or the dim after it where that is the block dim jammed into it; both run
        // innermost, inside the other loops of the box.
        auto vd    = Lit::isa<u64>(field(F::VDim));
        auto block = vd && *vd + 2 == rks.rp ? vd : std::nullopt;
        if (block) ++*vd;
        auto rank = [&](u64 d) { return d == vd || d == block ? rn : d; };
        // An axis sorts by the innermost loop that moves it, so that the copy follows the order the box is read in
        // (cf. TVM's layout rewrite): an axis no loop moves first.
        auto key = std::vector<u64>(*r);
        for (u64 a = 0; a != *r; ++a)
            for (u64 d = 0; d != rn; ++d)
                if (!fixed[d] && moves(d, a)) key[a] = std::max(key[a], rank(d) + 1);
        // Without a layout, a vector run that moves another axis than the last splits that axis by the run, so that
        // the vector loop reads the copy at unit stride (cf. OpenBLAS packing a transposed B).
        if (by <= 1 && vd && *vd < rks.rp && !fixed[*vd])
            if (auto run = Lit::isa<u64>(mr.Sr->proj(rn, *vd)); run && *run > 1) {
                auto moved = std::vector<u64>();
                for (u64 a = 0; a != *r; ++a)
                    if (moves(*vd, a)) moved.push_back(a);
                auto last = std::ranges::max_element(key) - key.begin();
                if (moved.size() == 1 && (u64(last) != moved[0] || std::ranges::count(key, key[last]) > 1))
                    ax = moved[0], by = *run;
            }
        auto zero = zero_of(T);
        // The split for the vector run is only worth it where its zero padding can be written.
        if (by > 1 && !layout && !zero) by = 1;
        auto split = by > 1 && ax < *r;
        auto rr    = split ? *r + 1 : *r;
        if (split && !zero) {
            give_up("the layout pads with zeros, but the input's type has none");
            continue;
        }
        auto cap_ax = split ? *Lit::isa<u64>(caps[ax]) : u64(0);
        auto order  = std::vector<u64>();
        for (u64 a = 0; a != *r; ++a)
            if (!split || a != ax) order.push_back(a);
        std::ranges::stable_sort(order, {}, [&](u64 a) { return key[a]; });
        auto others = DefVec(order.size(), [&](size_t k) { return caps[order[k]]; });
        auto shape = !split ? w.tuple(others) : [&] {
            auto dims = DefVec{w.lit_nat((cap_ax + by - 1) / by)};
            dims.insert(dims.end(), others.begin(), others.end());
            dims.push_back(w.lit_nat(by));
            return w.tuple(dims);
        }();

        // A frozen input keeps the copies of every box, stacked by the dims outside the copy that move the box.
        auto stack   = std::vector<u64>();
        auto persist = bit(packs, i);
        for (u64 d = 0; persist && d != rn; ++d)
            if (fixed[d] && std::ranges::any_of(std::views::iota(u64(0), *r), [&](u64 a) { return moves(d, a); }))
                stack.push_back(d);
        auto outer = DefVec(stack.size(), [&](size_t k) { return mr.Sr->proj(rn, stack[k]); });
        auto count = size(w.tuple(outer), outer.size());
        if (persist) {
            auto all = count * size(shape, rr);
            auto why = !param(inputs[i])                      ? "it is no input buffer of the function"
                     : all == 0 || all > 2 * size(Sis[i], *r) ? "the copies would exceed twice its size"
                                                              : nullptr;
            if (why) {
                log().w("prepack of input {} of {} not realized: {}", i, mr.map_out, why);
                persist = false;
                stack.clear();
                outer.clear();
            }
        }
        // The stack is one dim, so that LLVM still traces a read of it back to the allocation.
        auto ra    = (stack.empty() ? 0 : 1) + rr;
        auto full  = stack.empty() ? DefVec() : DefVec{w.lit_nat(count)};
        auto inner = shape->projs(rr);
        full.insert(full.end(), inner.begin(), inner.end());
        auto s_all = w.tuple(full);
        auto buf   = static_cast<const Def*>(nullptr);
        auto fresh = static_cast<const Def*>(nullptr);
        if (persist)
            std::tie(mem, buf, fresh) = buffer::op_persist(w.lit_nat(ra), s_all, T, mem, inputs[i])->projs<3>();
        else
            std::tie(mem, buf) = buffer::op_alloc(w.lit_nat(rr), shape, T, mem)->projs<2>();
        auto buf_ty      = w.sigma({mem_ty, buf->type()});
        auto add    = [&](const Def* x, const Def* y) { return w.call(core::wrap::add, core::Mode::none, Defs{x, y}); };
        auto in_box      = [&](const DefVec& iters) {
            auto xs = DefVec(*r);
            for (size_t k = 0; k != order.size(); ++k)
                xs[order[k]] = iters[k + (split ? 1 : 0)];
            if (split)
                xs[ax] = add(w.call(core::wrap::mul, core::Mode::none, Defs{iters[0], w.lit_i64(by)}), iters[*r]);
            return xs;
        };
        auto prefix = DefVec();
        if (!stack.empty()) {
            const Def* lin = w.lit_i64(0);
            for (size_t k = 0; k != stack.size(); ++k) {
                auto c
                    = w.call<core::bitcast>(i64, w.call<core::bitcast>(w.type_nat(), w.app(get, w.lit_nat(stack[k]))));
                lin = add(w.call(core::wrap::mul, core::Mode::none, Defs{lin, w.call<core::bitcast>(i64, outer[k])}),
                          c);
            }
            prefix.push_back(lin);
        }
        // A box that leaves the input copies the input's last cell there, which only a padded vector loop reads.
        // A layout pads with zeros instead.
        auto copy = build_pointwise(
            buf_ty, fmem, buf, inputs[i], shape, rr, "cache_copy",
            [&](const DefVec& iters, const Def* in, const Def* m) {
                auto xs = in_box(iters);
                auto at = DefVec(*r, [&](size_t a) { return add(w.call<core::bitcast>(i64, lo->proj(*r, a)), xs[a]); });
                auto ends   = DefVec(*r, [&](size_t a) { return w.call<core::bitcast>(i64, Sis[i]->proj(*r, a)); });
                auto coords = DefVec(*r, [&](size_t a) {
                    return clamp(at[a], w.call(core::wrap::sub, core::Mode::none, Defs{ends[a], w.lit_i64(1)}));
                });
                auto [rm, value] = buffer::op_read(m, in, folded_idx(Sis[i], coords))->projs<2>();
                if (!split) return std::pair<const Def*, const Def*>{rm, value};
                const Def* inside = w.call(core::icmp::ul, Defs{xs[ax], w.lit_i64(cap_ax)});
                for (u64 a = 0; a != *r; ++a)
                    inside = w.call(core::bit2::and_, w.lit_nat(2),
                                    w.tuple({inside, w.call(core::icmp::ul, Defs{at[a], ends[a]})}));
                return std::pair<const Def*, const Def*>{rm, w.select(inside, value, zero)};
            },
            prefix, s_all);
        fill->app(true, fret, copy->proj(2, 0));

        if (persist) {
            // Where the copies are new, make every one of them ahead of the nest, each given its dims of the stack.
            auto pre        = w.mut_fun(w.sigma({mem_ty, w.type_bool()}), mem_ty)->set("prepack");
            auto [pa, pret] = pre->vars<2>();
            auto [pm, pf]   = pa->projs<2>();
            Lam* cur        = pre;
            const Def* exit = pret;
            const Def* acc  = pm;
            auto dims       = DefVec{w.select(pf, w.lit_nat_1(), w.lit_nat_0())};
            dims.insert(dims.end(), outer.begin(), outer.end());
            auto iters       = affine::counting_nest(cur, exit, acc, dims, "prepack");
            auto stacked     = w.mut_lam(w.pi(w.type_nat(), w.type_i32()))->set("prepack_get");
            const Def* coord = w.lit_i32(0);
            for (size_t k = 0; k != stack.size(); ++k) {
                auto here = w.call(core::ncmp::e, Defs{stacked->var(), w.lit_nat(stack[k])});
                auto it   = w.call(core::conv::u, w.lit_nat(u64(1) << 32), iters[k + 1]);
                coord     = w.extract(w.tuple({coord, it}), here);
            }
            stacked->set(true, coord);
            auto next = w.mut_con(mem_ty)->set("prepack_next");
            next->app(true, exit, next->var());
            cur->app(true, fill, {w.tuple({acc, stacked}), next});
            mem = w.app(cps::op_cps2ds_dep(pre), w.tuple({mem, fresh}));
        } else if (at) {
            // The copy runs after whatever the action of its dim already does.
            auto old    = fields[F::Act];
            auto act_pi = old->type()->as<Pi>();
            auto act_of = w.mut_lam(act_pi->dom(), act_pi->codom())->set("cache_acts");
            auto then   = w.app(w.annex<btensor::act_seq>(), {w.app(old, w.lit_nat(*dim)), fill});
            auto here   = w.call(core::ncmp::e, Defs{act_of->var(), w.lit_nat(*dim)});
            act_of->set(true, w.extract(w.tuple({w.app(old, act_of->var()), then}), here));
            fields[F::Act]  = act_of;
            fields[F::Acts] = w.lit_nat(Lit::isa<u64>(fields[F::Acts]).value_or(0) | u64(1) << *dim);
        } else {
            auto none = w.mut_lam(w.pi(w.type_nat(), w.type_i32()))->set("cache_get");
            none->set(true, w.lit_i32(0));
            mem = w.app(cps::op_cps2ds_dep(fill), w.tuple({mem, none}));
        }

        // The op reads the copy at its coordinates less those of the read of the corner, split as the layout splits.
        auto pi     = maps[i]->type()->as<Pi>();
        auto index  = w.annex<affine::Index>();
        auto read   = w.mut_lam(pi->dom(), w.arr(ra, index))->set("cache_read");
        auto o      = read->var();
        auto whole  = w.app(maps[i], o);
        auto corner = w.app(
            maps[i], w.tuple(DefVec(rn, [&](size_t d) { return fixed[d] ? o->proj(rn, d) : lit(w.lit_nat_0()); })));
        auto rel = DefVec(
            *r, [&](size_t a) { return w.call(affine::op::sub, Defs{whole->proj(*r, a), corner->proj(*r, a)}); });
        auto xs = DefVec();
        if (!stack.empty())
            xs.push_back(w.call<affine::linearize>(
                Defs{w.tuple(DefVec(stack.size(), [&](size_t k) { return o->proj(rn, stack[k]); })), w.tuple(outer)}));
        if (split) xs.push_back(w.call(affine::semiop::floordiv, Defs{rel[ax], w.lit_nat(by)}));
        for (auto a : order)
            xs.push_back(rel[a]);
        if (split) xs.push_back(w.call(affine::semiop::rem, Defs{rel[ax], w.lit_nat(by)}));
        rel = xs;
        read->set(true, w.tuple(rel));
        log().d("read input {} of {} from a {}copy {}", i, mr.map_out, persist ? "persistent " : "", s_all);
        Ris[i]    = w.lit_nat(ra);
        Sis[i]    = s_all;
        maps[i]   = read;
        inputs[i] = buf;
        if (!persist) caches.push_back(buf);
        copied[i] = true;
    }

    // A vector loop over the last parallel dim may compute a cut run whole where every input it moves is a copy of
    // whole runs: the copies clamp their reads, and only the write-back of the run stops at the lim.
    auto vdim = Lit::isa<u64>(field(F::VDim));
    auto pad  = rks.rp != 0 && vdim && (*vdim + 1 == rks.rp || *vdim + 2 == rks.rp);
    for (u64 i = 0; pad && i != rks.nis; ++i) {
        if (copied[i]) continue;
        auto pi = maps[i]->type()->as<Pi>();
        auto o  = w.mut_lam(pi->dom(), pi->codom())->var();
        auto fixed
            = w.tuple(DefVec(rn, [&](size_t d) { return d + 1 == rks.rp ? lit(w.lit_nat_0()) : o->proj(rn, d); }));
        pad = w.app(maps[i], o) == w.app(maps[i], fixed);
    }
    fields[F::Pad] = w.lit_bool(pad);

    fields[F::Cache] = w.lit_nat_0();
    mr.sched         = w.app(w.annex<btensor::mk_sched>(), w.tuple(fields));
    mr.shapes        = w.tuple({mr.So, mr.Sr, mr.sched});
    mr.Ris           = w.tuple(Ris);
    mr.Sis           = w.tuple(Sis);
    mr.maps          = w.tuple(maps);
    return mem;
}

const Def* LowerMapReduce::lower_map_reduce_post(const App* app) {
    auto& w    = new_world();
    auto mr    = MapRedPost::of(rewrite(app->callee())->as<App>());
    auto ranks = mr.ranks();
    if (!ranks) {
        log().w("rank counts (nis/nps/Ro/Rn) of {} are not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }

    auto [mem, dst, old_inputs, post_inputs] = rewrite(app->arg())->projs<4>();
    auto inputs = DefVec(ranks->nis, [&](size_t i) { return old_inputs->proj(ranks->nis, i); });
    auto caches = DefVec();
    mem         = cache_inputs(mr, mem, inputs, caches);

    auto impl = rewrite(old_world().annex<btensor::map_reduce_post_impl>());
    impl      = w.app(impl, mr.nis_nps);
    impl      = w.app(impl, {mr.To, mr.Tp, mr.Ro, mr.Rp, w.lit_nat(ranks->rr)});
    impl      = w.app(impl, mr.shapes);
    impl      = w.app(impl, {mr.Tis, mr.Tps});
    impl      = w.app(impl, {mr.comb, mr.init, mr.post});
    impl      = w.app(impl, mr.map_out);

    auto [read_inputs_ty, read_post_ty, write_output_ty] = impl->type()->as<Pi>()->dom()->projs<3>();

    // A `buffer.Buf` has its literal size-1 axes folded away, so the indices fold alike.
    auto reader = [&](const Def* type, const char* name, u64 num, const Def* rank, const Def* bounds, const Def* rs,
                      const Def* ss, const Def* maps, const Def* buffers) {
        auto fun          = w.mut_lam(type->as<Pi>())->set(name);
        auto [args, ret]  = fun->vars<2>();
        auto [m, indices] = args->projs<2>();
        auto values       = DefVec(num);
        for (u64 i = 0; i != num; ++i) {
            auto si = ss->proj(num, i);
            auto [mapped, coords]
                = op_affine_map(maps->proj(num, i), rs->proj(num, i), rank, bounds, si, indices, m)->projs<2>();
            auto [read, value] = buffer::op_read(mapped, buffers->proj(num, i), *Shape(coords).fold(si))->projs<2>();
            m                  = read;
            values[i]          = value;
        }
        fun->app(true, ret, {m, w.tuple(values)});
        return fun;
    };
    auto read_inputs
        = reader(read_inputs_ty, "readInputs", ranks->nis, mr.Rn, mr.Sr, mr.Ris, mr.Sis, mr.maps, w.tuple(inputs));
    auto read_post
        = reader(read_post_ty, "readPost", ranks->nps, mr.Ro, mr.So, mr.Rps, mr.Sps, mr.post_maps, post_inputs);

    auto write_output = w.mut_lam(write_output_ty->as<Pi>())->set("writeOutput");
    {
        auto [args, ret]                = write_output->vars<2>();
        auto [m, output, coords, value] = args->projs<4>();
        auto stored                     = buffer::op_write(m, output, *Shape(coords).fold(mr.So), value);
        write_output->app(true, ret, stored->proj(2, 0));
    }

    impl     = w.app(impl, {read_inputs, read_post, write_output});
    auto res = w.app(impl, {mem, dst});
    if (caches.empty()) return res;
    auto [m, out] = res->projs<2>();
    for (auto buf : caches) {
        auto [r, s, T] = Axm::isa<buffer::Buf>(buf->type())->args<3>();
        m              = buffer::op_free(r, s, T, m, buf);
    }
    return w.tuple({m, out});
}

const Def* LowerMapReduce::lower_broadcast(const App* app) {
    auto& w                   = new_world();
    auto callee               = app->callee()->as<App>(); // (broadcast {impl}) (s_in, s_out)
    auto [s_in, s_out]        = rewrite(callee->arg())->projs<2>();
    auto [op_mem, dst, input] = rewrite(app->arg())->projs<3>();
    auto result_ty            = rewrite(app->type()); // [mem.M 0, buffer.Buf (ro, so, T)]

    auto r_l = Lit::isa(s_out->arity());
    if (!r_l) {
        log().w("rank of {} is not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }

    // Non-size-1 input dims mirror the matching output index; size-1 dims are dropped from each buffer index.
    auto compute = [&](const DefVec& iters, const Def* in_buf, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto idxs             = w.tuple(to_idxs(w, s_out->projs(*r_l), iters));
        auto [rd_mem, rd_val] = buffer::op_read(mem, in_buf, *Shape(idxs).fold(s_in))->projs<2>();
        return {rd_mem, rd_val};
    };

    return build_pointwise(result_ty, op_mem, dst, input, s_out, *r_l, "broadcast", compute);
}

const Def* LowerMapReduce::lower_pad(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    // callee: pad {T, r} [s_in] [mode, lo, hi] [s_out]. The shapes are the logical ones; buffer reads and
    // writes fold size-1 axes (the `Buf` handles are normalized), while the loops cover all logical dims.
    auto [Tr, s_in, params, s_out] = c->uncurry_args<4>();
    auto [mode, lo, hi]            = params->projs<3>();
    auto [op_mem, dst, input, val] = rewrite(app->arg())->projs<4>();
    auto result_ty                 = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s_out, T)]

    auto r_l    = Lit::isa<u64>(Tr->proj(2, 1));
    auto mode_l = Lit::isa<u64>(mode);
    if (!r_l || !mode_l) {
        log().w("rank/mode of {} is not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }

    auto compute = [&](const DefVec& iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto [in_buf, fill]  = ins->projs<2>();
        auto [coords, valid] = pad_coords(iters, lo, s_in, *mode_l != 0);
        auto [rd_mem, elem]  = buffer::op_read(mem, in_buf, *Shape(w, coords).fold(s_in))->projs<2>();
        return {rd_mem, valid ? w.select(valid, elem, fill) : elem}; // constant: fill out-of-region cells with `val`
    };

    return build_pointwise(result_ty, op_mem, dst, w.tuple({input, val}), s_out, *r_l, "pad", compute);
}

const Def* LowerMapReduce::lower_concat(const App* app) {
    auto c = rewrite(app->callee())->as<App>();

    // callee: concat {T, nis, r} [ax] {Sis} [s_out]. The shapes are the logical ones; buffer reads and
    // writes fold size-1 axes (the `Buf` handles are normalized), while the loops cover all logical dims.
    auto [TnisR, ax, Sis, s_out] = c->uncurry_args<4>();
    auto [T, nis, r]             = TnisR->projs<3>();
    auto [op_mem, dst, op_is]    = rewrite(app->arg())->projs<3>();
    auto result_ty               = rewrite(app->type()); // [mem.M 0, buffer.Buf (r, s_out, T)]

    auto nis_l = Lit::isa<u64>(nis);
    auto r_l   = Lit::isa<u64>(r);
    auto ax_l  = Lit::isa<u64>(ax);
    if (!nis_l || !r_l || !ax_l) {
        log().w("nis/rank/axis of {} are not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }
    auto nisn = *nis_l, rn = *r_l, axn = *ax_l;

    auto exts = concat_extents(Sis, nisn, rn, axn);
    if (!exts) {
        log().w("an extent of {} along the concat axis is not known at lowering time", app);
        return RWPhase::rewrite_imm_App(app);
    }

    auto compute = [&](const DefVec& iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        const Def* cur = mem;
        auto read      = [&](u64 i, const Def* ax_idx) {
            auto coords = iters;
            coords[axn] = ax_idx;
            auto [rd_mem, rd_val]
                = buffer::op_read(cur, ins->proj(nisn, i), folded_idx(Sis->proj(nisn, i), coords))->projs<2>();
            cur = rd_mem;
            return rd_val;
        };
        auto result = concat_select(iters[axn], *exts, read);
        return {cur, result};
    };

    return build_pointwise(result_ty, op_mem, dst, op_is, s_out, rn, "concat", compute);
}

const Def* LowerMapReduce::lower_gather(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]         = c->uncurry_args<3>();
    auto [T, r]                    = Tr->projs<2>();
    auto [s_src, s_idx]            = shapes->projs<2>();
    auto [op_mem, dst, input, idx] = rewrite(app->arg())->projs<4>();
    auto result_ty                 = rewrite(app->type());

    auto r_l   = Lit::isa<u64>(r);
    auto dim_l = Lit::isa<u64>(dim);
    if (!r_l || !dim_l) {
        log().w("{} doesn't have lowering-time known rank/axis", app);
        return RWPhase::rewrite_imm_App(app);
    }

    auto compute = [&](const DefVec& iters, const Def* ins, const Def* mem) -> std::pair<const Def*, const Def*> {
        auto [in_buf, idx_buf]   = ins->projs<2>();
        auto [idx_mem, selected] = buffer::op_read(mem, idx_buf, folded_idx(s_idx, iters))->projs<2>();
        auto src                 = iters;
        src[*dim_l]              = w.call<core::bitcast>(w.type_i64(), selected);
        auto [read_mem, val]     = buffer::op_read(idx_mem, in_buf, folded_idx(s_src, src))->projs<2>();
        return {read_mem, val};
    };
    return build_pointwise(result_ty, op_mem, dst, w.tuple({input, idx}), s_idx, *r_l, "gather", compute);
}

const Def* LowerMapReduce::lower_scatter(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]                     = c->uncurry_args<3>();
    auto [T, r]                                = Tr->projs<2>();
    auto [s_src, s_idx, s_updates]             = shapes->projs<3>();
    auto [op_mem, op_dst, input, idx, updates] = rewrite(app->arg())->projs<5>();
    auto result_ty                             = rewrite(app->type());

    auto r_l   = Lit::isa<u64>(r);
    auto dim_l = Lit::isa<u64>(dim);
    if (!r_l || !dim_l) {
        log().w("{} doesn't have lowering-time known rank/axis", app);
        return RWPhase::rewrite_imm_App(app);
    }

    auto [fun, call] = ds_splice(w.tuple({op_mem, op_dst, input, idx, updates}), result_ty, "scatter");
    auto [fun_mem, out_buf, in_buf, idx_buf, update_buf] = fun->var(2, 0)->projs<5>();
    const Def* cont                                      = fun->var(2, 1);

    const Def* acc            = w.tuple({buffer::op_copy(fun_mem, out_buf, in_buf), out_buf});
    auto current              = fun;
    auto iters                = affine::counting_nest(current, cont, acc, s_idx->projs(*r_l), "scatter");
    auto [loop_mem, loop_buf] = acc->projs<2>();

    auto [idx_mem, selected]  = buffer::op_read(loop_mem, idx_buf, folded_idx(s_idx, iters))->projs<2>();
    auto [update_mem, update] = buffer::op_read(idx_mem, update_buf, folded_idx(s_updates, iters))->projs<2>();
    auto dst                  = iters;
    dst[*dim_l]               = w.call<core::bitcast>(w.type_i64(), selected);
    auto write_mem            = buffer::op_write(update_mem, loop_buf, folded_idx(s_src, dst), update)->proj(2, 0);
    current->app(true, cont, w.tuple({write_mem, loop_buf}));
    return call;
}

} // namespace mim::plug::btensor::phase
