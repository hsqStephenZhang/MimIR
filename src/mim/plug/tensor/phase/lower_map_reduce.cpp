#include "mim/plug/tensor/phase/lower_map_reduce.h"

#include "mim/def.h"
#include "mim/lam.h"

#include "mim/util/types.h"

#include "mim/plug/affine/affine.h"
#include "mim/plug/core/core.h"
#include "mim/plug/cps/cps.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

const Def* LowerMapReduce::rec_broadcast(const Def* s_in, const Def* s_out, const Def* input, u64 r, u64 i) {
    auto& w = new_world();
    // Base case: all dimensions have been processed; `input` is the final scalar.
    if (i == r) return input;

    auto s_in_ri = s_in->proj(r, i), s_out_ri = s_out->proj(r, i);
    DLOG("rec_broadcast");
    DLOG("    r = {}", r);
    DLOG("    i = {}", i);
    DLOG("    s_in_ri = {} : {}", s_in_ri, s_in_ri->type());
    DLOG("    s_out_ri = {} : {}", s_out_ri, s_out_ri->type());
    DLOG("    input = {} : {}", input, input->type());

    if (s_in_ri == s_out_ri) {
        if (auto s_in_lit = Lit::isa<u64>(s_in_ri)) {
            DefVec inputs(*s_in_lit, [&](size_t j) { return rec_broadcast(s_in, s_out, input->proj(j), r, i + 1); });
            return w.tuple(inputs);
        } else {
            // TODO: we could probably support non-literal sizes as well, but we would need to generate loops to copy
            // the data instead of just packing it.
            WLOG("dimension {} of the input and output are equal but not literal: {} : {}", i, s_in_ri,
                 s_in_ri->type());
            return nullptr;
        }
    }

    if (auto s_in_lit = Lit::isa<u64>(s_in_ri); s_in_lit && *s_in_lit == 1) {
        DLOG("dimension {} of the input is 1, can be broadcasted to dimension {} of the output", i, s_out_ri);
        return w.pack(s_out_ri, rec_broadcast(s_in, s_out, input, r, i + 1));
    }

    WLOG("cannot broadcast dimension {} of size {} to size {}", i, s_in_ri, s_out_ri);
    return nullptr;
}

const Def* LowerMapReduce::lower_broadcast(const App* app) {
    auto& w  = new_world();
    auto c   = rewrite(app->callee());
    auto arg = rewrite(app->arg());

    auto [s_in, s_out, input] = arg->projs<3>();
    auto callee               = c->as<App>();
    auto [T, r]               = callee->args<2>();
    DLOG("lower_broadcast");
    DLOG("    s_out = {} : {}", s_out, s_out->type());
    DLOG("    input = {} : {}", input, input->type());
    DLOG("    T = {} : {}", T, T->type());
    DLOG("    r = {} : {}", r, r->type());
    DLOG("    s_in = {} : {}", s_in, s_in->type());

    auto r_nat = Lit::isa<u64>(r);
    if (!r_nat) {
        WLOG("{} doesn't have a lowering-time known rank: {}", app, r);
        return nullptr;
    }
    // r_nat will never be 0, as we would have normalized this case away already
    if (s_in == s_out) return input;

    if (*r_nat == 1) {
        if (auto s_in_lit = Lit::isa<u64>(s_in)) {
            assert(*s_in_lit == 1 && "input dimensions must be 1 or equal to the output dimension");
            return w.pack(s_out, input);
        }
    }

    auto result = rec_broadcast(s_in, s_out, input, *r_nat, 0);
    DLOG("result of rec_broadcast = {} : {}", result, result->type());
    return result;
}

static std::pair<Lam*, const Def*> counting_for(const Def* bound, const Def* acc, const Def* exit, Sym name) {
    auto& w       = bound->world();
    auto acc_ty   = acc->type();
    auto body     = w.mut_con({/* iter */ w.type_i64(), /* acc */ acc_ty, /* return */ w.cn(acc_ty)})->set(name);
    auto for_loop = w.call<affine::For>(body, exit, Defs{w.lit_i64(0), bound, w.lit_i64(1), acc});
    return {body, for_loop};
}

const Def* LowerMapReduce::lower_map_reduce(const App* app) {
    // meta arguments:
    // * nis = in-count (nat)
    // * To = out-type (*), Ro = #output loops = result rank, Rr = #reduction loops
    // * So = result shape (Ro*nat)
    // * Sr = the full loop bounds (Ro+Rr)*nat: the leading Ro are the output-loop bounds, the trailing Rr the
    // reductions
    // * Tis/Ris/Sis = input types/ranks/shapes
    // arguments:
    // * f = combination function (CPS), init = accumulator init
    // * acc_out = affine map from the (Ro+Rr) loop vector to the Ro write coordinates in the result «So» (the reduction
    //             part is not in scope at write-back, so acc_out must depend only on the leading Ro output indices)
    // * accs = per-input affine map from the (Ro+Rr) loop vector to the input's read coordinates
    // * is = input tensors
    auto& w     = new_world();
    auto c      = rewrite(app->callee())->as<App>();
    auto inputs = rewrite(app->arg());
    auto type   = rewrite(app->type());

    auto [nis, meta, shapes, TisRisSis, comb_init, acc_out, accs] = c->uncurry_args<7>();
    auto [To, Ro, Rr]                                             = meta->projs<3>();
    auto [So, Sr]                                                 = shapes->projs<2>();
    auto [Tis, Ris, Sis]                                          = TisRisSis->projs<3>();
    auto [comb, init]                                             = comb_init->projs<2>();

    auto nis_l = Lit::isa<u64>(nis);
    auto ro_l = Lit::isa<u64>(Ro), rr_l = Lit::isa<u64>(Rr);
    if (!nis_l || !ro_l || !rr_l) {
        WLOG("{} doesn't have lowering-time known rank counts (nis/Ro/Rr)", app);
        return nullptr;
    }
    auto nis_nat = *nis_l;
    auto ro = *ro_l, rr = *rr_l;
    auto nloops = ro + rr;           // length of the full loop vector (= length of Sr)
    auto n      = w.lit_nat(nloops); // passed as the affine maps' domain length

    // Keep the existing legality check: downstream affine lowering expects literal input ranks.
    for (u64 i = 0; i < nis_nat; ++i) {
        auto l = Lit::isa<u64>(Ris->proj(nis_nat, i));
        if (!l) {
            WLOG("input {} of {} has a non-literal rank", i, app);
            return nullptr;
        }
    }

    // Builds `%affine.map` and drops its dummy mem result via a Mim helper. The emitted `%affine.map` is lowered to
    // %core arithmetic by the subsequent `%affine.lower_index`.
    auto affine_map = [&](const Def* f, const Def* m, const Def* n, const Def* sin, const Def* sout, const Def* idxs) {
        auto apply = w.app(w.annex<tensor::affine_apply_impl>(), Defs{m, n});
        apply      = w.app(apply, Defs{sin, sout});
        apply      = w.app(apply, f);
        return w.app(apply, idxs);
    };

    try {
        auto fun    = w.mut_fun(inputs->type(), type)->set("mapRed");
        auto ds_fun = cps::op_cps2ds_dep(fun)->set("dsFun");
        auto call   = w.app(ds_fun, inputs)->set("call");

        auto new_inputs = fun->var(0)->set("is");

        // Outer (parallel) loops over the leading Ro bounds of `Sr`, collecting the output iteration indices.
        auto cont        = fun->var(1);
        auto init_mat    = w.bot(cont->type()->as<Pi>()->dom());
        auto acc         = init_mat;
        auto current_mut = fun;
        DefVec out_iters;
        out_iters.reserve(ro);
        for (u64 i = 0; i < ro; ++i) {
            auto dim                    = Sr->proj(nloops, i);
            auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
            auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("forOut_" + std::to_string(i)));
            auto [iter, new_acc, yield] = body->vars<3>();
            cont                        = yield;
            out_iters.push_back(w.call(core::conv::u, dim, iter));
            acc = new_acc;
            current_mut->set(true, for_call);
            current_mut = body;
        }
        auto wb_matrix = acc;

        // Write-back: narrow the accumulated element into the result at the affine write coordinates `acc_out`.
        // acc_out takes the full (Ro+Rr) loop vector, but the reduction loops have already been folded away here, so we
        // pass 0 for those slots; acc_out must depend only on the leading Ro output indices.
        auto write_back    = w.mut_con(To)->set("writeBack");
        auto element_final = write_back->var(0);
        DefVec wb_iters    = out_iters;
        for (u64 j = 0; j < rr; ++j)
            wb_iters.push_back(w.call(core::conv::u, Sr->proj(nloops, ro + j), w.lit(w.type_i64(), 0)));
        auto write_coords = affine_map(acc_out, Ro, n, Sr, So, w.tuple(wb_iters)); // «Ro; Idx (So#k)»
        auto store        = w.app(w.annex<tensor::indexed_store_impl>(), Defs{To, Ro});
        store             = w.app(store, So);
        write_back->app(true, cont, w.app(store, Defs{wb_matrix, write_coords, element_final}));

        // Inner (reduction) loops over the trailing Rr bounds of `Sr`, collecting the reduction iteration indices.
        acc  = init;
        cont = write_back;
        DefVec red_iters;
        red_iters.reserve(rr);
        for (u64 j = 0; j < rr; ++j) {
            auto dim                    = Sr->proj(nloops, ro + j);
            auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
            auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("forIn_" + std::to_string(j)));
            auto [iter, new_acc, yield] = body->vars<3>();
            cont                        = yield;
            red_iters.push_back(w.call(core::conv::u, dim, iter));
            acc = new_acc;
            current_mut->set(true, for_call);
            current_mut = body;
        }
        auto element_acc = acc;

        // The full loop iteration vector `(o…, r…)`; its moduli are exactly `Sr`.
        DefVec iters_v = out_iters;
        iters_v.insert(iters_v.end(), red_iters.begin(), red_iters.end());
        auto iters = w.tuple(iters_v);

        // Read one element from each input at its affine read coordinates.
        DefVec input_elements(nis_nat);
        for (u64 i = 0; i < nis_nat; ++i) {
            auto input_matrix = new_inputs->proj(nis_nat, i);
            auto Tis_i        = Tis->proj(nis_nat, i);
            auto Ris_i        = Ris->proj(nis_nat, i);
            auto sis_i        = Sis->proj(nis_nat, i);
            auto coords       = affine_map(accs->proj(nis_nat, i), Ris_i, n, Sr, sis_i, iters);
            auto load         = w.app(w.annex<tensor::indexed_load_impl>(), Defs{Tis_i, Ris_i});
            load              = w.app(load, sis_i);
            input_elements[i] = w.app(load, Defs{input_matrix, coords});
        }

        comb->set("comb");
        current_mut->app(true, comb, {w.tuple({element_acc, w.tuple(input_elements)}), cont});
        return call;
    } catch (const std::exception& e) { error("error during lowering map_reduce: {}", e.what()); }
}

const Def* LowerMapReduce::build_pointwise(const Def* inputs,
                                           const Def* type,
                                           const Def* So,
                                           u64 ro,
                                           std::function<const Def*(const DefVec&, const Def*)> compute) {
    auto& w = new_world();

    auto fun    = w.mut_fun(inputs->type(), type)->set("pointwise");
    auto ds_fun = cps::op_cps2ds_dep(fun)->set("dsFun");
    auto call   = w.app(ds_fun, inputs)->set("call");

    auto new_inputs = fun->var(0)->set("is");

    // Output loops over `So`, collecting the raw i64 iteration indices for `compute`.
    auto cont        = fun->var(1);
    auto acc         = w.bot(cont->type()->as<Pi>()->dom());
    auto current_mut = fun;
    DefVec out_iters; // raw i64 loop counters
    out_iters.reserve(ro);
    for (u64 i = 0; i < ro; ++i) {
        auto dim                    = So->proj(ro, i);
        auto bound                  = w.call<core::bitcast>(w.type_i64(), dim);
        auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("forOut_" + std::to_string(i)));
        auto [iter, new_acc, yield] = body->vars<3>();
        cont                        = yield;
        out_iters.push_back(iter);
        acc = new_acc;
        current_mut->set(true, for_call);
        current_mut = body;
    }
    auto wb_matrix = acc;

    auto element = compute(out_iters, new_inputs);
    auto store   = w.app(w.annex<tensor::pointwise_store_impl>(), Defs{element->type(), w.lit_nat(ro)});
    store        = w.app(store, So);
    store        = w.app(store, w.tuple(out_iters));
    current_mut->app(true, cont, w.app(store, Defs{wb_matrix, element}));
    return call;
}

const Def* LowerMapReduce::lower_pointwise_loop(const App* app) {
    auto c    = rewrite(app->callee())->as<App>();
    auto args = rewrite(app->arg());
    auto type = rewrite(app->type());

    auto [TinTr, s_out, body] = c->uncurry_args<3>();
    auto r                    = TinTr->proj(3, 2);

    return build_pointwise_loop(args, type, s_out, r, body);
}

const Def* LowerMapReduce::build_pointwise_loop(const Def* inputs,
                                                const Def* type,
                                                const Def* So,
                                                const Def* r,
                                                const Def* body) {
    auto& w = new_world();
    auto r_l = Lit::isa<u64>(r);
    if (!r_l) {
        WLOG("pointwise loop doesn't have a lowering-time known rank");
        return nullptr;
    }
    auto rn = *r_l;

    auto compute = [&](const DefVec& out_iters, const Def* new_inputs) -> const Def* {
        return w.app(w.app(body, w.tuple(out_iters)), new_inputs);
    };

    return build_pointwise(inputs, type, So, rn, compute);
}

const Def* LowerMapReduce::lower_pad(const App* app) {
    auto& w   = new_world();
    auto c    = rewrite(app->callee())->as<App>();
    auto args = rewrite(app->arg()); // (input, value)
    auto type = rewrite(app->type());

    // callee: pad {T, r} [s_in] [mode, lo, hi]
    auto [Tr, s_in, params] = c->uncurry_args<3>();
    auto [T, r]             = Tr->projs<2>();
    auto [mode, lo, hi]     = params->projs<3>();

    auto r_l = Lit::isa<u64>(r);
    if (!r_l) {
        WLOG("{} doesn't have a lowering-time known rank", app);
        return nullptr;
    }
    auto rn = *r_l;

    // Deduce the output shape: s_out#d = lo#d + s_in#d + hi#d.
    DefVec so(rn);
    auto inner_type = type;
    for (u64 d = 0; d < rn; ++d) {
        assert(inner_type->node() == Node::Arr || inner_type->node() == Node::Pack);
        auto inner_type_seq = (const Seq*)inner_type;
        so[d]               = inner_type_seq->arity();
        inner_type          = inner_type_seq->body();
    }
    auto s_out = w.tuple(so);

    auto compute = [&](const DefVec& out_iters, const Def* new_inputs) -> const Def* {
        auto helper = w.app(w.annex<tensor::pad_pointwise_elem_impl>(), Defs{T, r});
        helper      = w.app(helper, s_in);
        helper      = w.app(helper, Defs{mode, lo, hi});
        helper      = w.app(helper, w.tuple(out_iters));
        return w.app(helper, new_inputs);
    };

    return build_pointwise(args, type, s_out, rn, compute);
}

const Def* LowerMapReduce::lower_concat(const App* app) {
    auto& w   = new_world();
    auto c    = rewrite(app->callee())->as<App>();
    auto args = rewrite(app->arg()); // the `is` input tuple
    auto type = rewrite(app->type());

    // callee: concat {T, nis, r} [ax] {Sis}
    auto [TnisR, ax, Sis] = c->uncurry_args<3>();
    auto [T, nis, r]      = TnisR->projs<3>();

    auto nis_l = Lit::isa<u64>(nis);
    auto r_l   = Lit::isa<u64>(r);
    auto ax_l  = Lit::isa<u64>(ax);
    if (!nis_l || !r_l || !ax_l) {
        WLOG("{} doesn't have lowering-time known nis/r/ax", app);
        return nullptr;
    }
    auto nisn = *nis_l, rn = *r_l, axn = *ax_l;

    // Prefix offsets along `ax`: off#i = Σ_{j<i} Sis#j#ax (literal extents required).
    DefVec off(nisn);
    u64 acc_off = 0;
    for (u64 i = 0; i < nisn; ++i) {
        off[i]  = w.lit_i64(acc_off);
        auto ei = Lit::isa<u64>(Sis->proj(nisn, i)->proj(rn, axn));
        if (!ei) {
            WLOG("{} input {} has a non-literal extent along the concat axis", app, i);
            return nullptr;
        }
        acc_off += *ei;
    }

    // Deduce the output shape: the summed extent along `ax`, the shared extents elsewhere.
    DefVec so(rn);
    for (u64 d = 0; d < rn; ++d)
        so[d] = (d == axn) ? w.lit_nat(acc_off) : Sis->proj(nisn, 0)->proj(rn, d);
    auto s_out = w.tuple(so);

    auto compute = [&](const DefVec& out_iters, const Def* new_inputs) -> const Def* {
        auto helper = w.app(w.annex<tensor::concat_pointwise_elem_impl>(), Defs{T, nis, r});
        helper      = w.app(helper, ax);
        helper      = w.app(helper, Sis);
        helper      = w.app(helper, w.tuple(off));
        helper      = w.app(helper, w.tuple(out_iters));
        return w.app(helper, new_inputs);
    };

    return build_pointwise(args, type, s_out, rn, compute);
}

const Def* LowerMapReduce::lower_gather(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();
    auto args = rewrite(app->arg());
    auto type = rewrite(app->type());

    auto [Tr, shapes, dim_arg] = c->uncurry_args<3>();
    auto [T, r] = Tr->projs<2>();
    auto [s_src, s_idx] = shapes->projs<2>();
    auto dim = dim_arg;

    auto r_l = Lit::isa<u64>(r);
    if (!r_l) {
        WLOG("{} doesn't have a lowering-time known rank", app);
        return nullptr;
    }

    auto body = w.app(w.annex<tensor::gather_pointwise_elem_impl>(), Defs{T, r});
    body      = w.app(body, Defs{s_src, s_idx});
    body      = w.app(body, dim);

    return build_pointwise_loop(args, type, s_idx, r, body);
}

const Def* LowerMapReduce::lower_scatter(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();
    auto args = rewrite(app->arg());
    auto type = rewrite(app->type());

    auto [Tr, shapes, dim_arg] = c->uncurry_args<3>();
    auto [T, r] = Tr->projs<2>();
    auto [s_src, s_idx] = shapes->projs<2>();
    auto dim = dim_arg;

    auto r_l = Lit::isa<u64>(r);
    if (!r_l) {
        WLOG("{} doesn't have a lowering-time known rank", app);
        return nullptr;
    }
    auto rn = *r_l;

    auto fun    = w.mut_fun(args->type(), type)->set("scatter");
    auto ds_fun = cps::op_cps2ds_dep(fun)->set("dsFun");
    auto call   = w.app(ds_fun, args)->set("call");

    auto new_inputs = fun->var(0)->set("is");
    auto [input, index, updates] = new_inputs->projs<3>();

    auto cont        = fun->var(1);
    auto acc         = input;
    auto current_mut = fun;

    DefVec out_iters;
    out_iters.reserve(rn);
    for (u64 i = 0; i < rn; ++i) {
        auto dim_size               = s_idx->proj(rn, i);
        auto bound                  = w.call<core::bitcast>(w.type_i64(), dim_size);
        auto [body, for_call]       = counting_for(bound, acc, cont, w.sym("forOut_" + std::to_string(i)));
        auto [iter, new_acc, yield] = body->vars<3>();
        cont                        = yield;
        out_iters.push_back(iter);
        acc = new_acc;
        current_mut->set(true, for_call);
        current_mut = body;
    }

    auto step = w.app(w.annex<tensor::scatter_step_impl>(), Defs{T, r});
    step      = w.app(step, Defs{s_src, s_idx});
    step      = w.app(step, dim);
    step      = w.app(step, w.tuple(out_iters));
    auto next_acc = w.app(step, Defs{acc, index, updates});
    current_mut->app(true, cont, next_acc);

    return call;
}

const Def* LowerMapReduce::rewrite_imm_App(const App* app) {
    if (auto bc = Axm::isa<tensor::broadcast>(app)) {
        if (auto res = lower_broadcast(bc)) return res;
    } else if (auto mr = Axm::isa<tensor::map_reduce>(app)) {
        if (auto res = lower_map_reduce(mr)) return res;
    } else if (auto pw = Axm::isa<tensor::pointwise_loop>(app)) {
        if (auto res = lower_pointwise_loop(pw)) return res;
    } else if (auto pad = Axm::isa<tensor::pad>(app)) {
        if (auto res = lower_pad(pad)) return res;
    } else if (auto cat = Axm::isa<tensor::concat>(app)) {
        if (auto res = lower_concat(cat)) return res;
    } else if (auto gather = Axm::isa<tensor::gather_impl_axm>(app)) {
        if (auto res = lower_gather(gather)) return res;
    } else if (auto scatter = Axm::isa<tensor::scatter_impl_axm>(app)) {
        if (auto res = lower_scatter(scatter)) return res;
    }
    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::tensor::phase
