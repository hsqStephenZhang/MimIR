#include "mim/plug/tensor/phase/lower_to_mem.h"

#include <bit>

#include <fe/worklist.h>

#include <mim/axm.h>
#include <mim/def.h>
#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/btensor/btensor.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/tensor/phase/constraints.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

/// Does `t` (recursively through immutable sigmas) contain a Pi?
bool contains_pi(const Def* t) {
    if (t->isa<Pi>()) return true;
    if (auto sig = t->isa_imm<Sigma>())
        for (auto op : sig->ops())
            if (contains_pi(op)) return true;
    return false;
}

/// Peels nested `Pack`s off `d`; returns the innermost body iff `d` is a constant splat — every axis a
/// `Pack` (so `d` is uniform, not e.g. a `Tuple` of distinct rows), bottoming out in a closed scalar
/// (index-independent, non-array). So `‹784, 1024; 1e-3›` yields the scalar `1e-3`, while an index-dependent
/// pack `‹i; f i›` or a genuine literal `((1, 2), (3, 4))` yields `nullptr`.
const Def* splat_scalar(const Def* d) {
    if (!d->isa<Pack>()) return nullptr;
    while (auto pack = d->isa<Pack>())
        d = pack->body();
    return (d->is_closed() && !d->type()->isa<Arr>()) ? d : nullptr;
}

/// Is `app` a tensor op whose lowering consumes a LowerToMem::fresh_mem?
bool wants_fresh_mem(const App* app) {
    return Axm::isa<tensor::generate>(app) || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::map_reduce_post>(app)
        || Axm::isa<tensor::pad>(app) || Axm::isa<tensor::concat>(app) || Axm::isa<tensor::gather>(app)
        || Axm::isa<tensor::scatter>(app);
}

/// Is `app` one of the tensor ops this phase bufferizes?
bool is_tensor_op(const App* app) { return Axm::isa<tensor::splat>(app) || wants_fresh_mem(app); }

/// The Lam a - possibly projected - parameter belongs to; `nullptr` for anything else.
Lam* param_of(const Def* d) {
    while (auto ex = d->isa<Extract>())
        d = ex->tuple();
    if (auto var = d->isa<Var>()) return var->binder()->isa_mut<Lam>();
    return nullptr;
}

/// Does @p d - a value or an aggregate of them - contain a `buffer.Buf`?
bool holds_buf(const Def* d) {
    if (Axm::isa<buffer::Buf>(d->type())) return true;
    if (auto tuple = d->isa<Tuple>()) return std::ranges::any_of(tuple->ops(), holds_buf);
    return false;
}

/// @p buf, or its sole element if @p app's result has been folded to a scalar.
const Def* scalar_result(const App* app, const Def* mem, const Def* buf) {
    if (app->type()->isa<Arr>()) return buf;
    return buffer::op_read(mem, buf, buf->world().tuple())->proj(2, 1);
}

} // namespace

void LowerToMem::collect_tensor_types() {
    // The default pipeline lowers tensors exclusively through buffers — there is no value-semantics
    // fallback. A program shape the conversion cannot handle is a hard error, not silent residue.
    auto gate = [](const char* why, const Def* culprit) { fe::throwf("cannot bufferize: {} (`{}`)", why, culprit); };
    auto elem = [&gate](const Def* T) {
        if (T->isa<Arr>()) gate("tensor with array element type", T);
    };
    // Fully folded shapes (`«1; T»` ≡ `T`) denote plain scalars — recording them would poison every
    // function whose signature mentions the element type, so only genuine array types count as tensors.
    auto add_tensor_ty = [this](const Def* t) {
        if (t->isa<Arr>()) tensor_ty_.emplace(t);
    };

    fe::BFSWorklist<DefSet> wl;
    auto push = [&wl](const Def* d) {
        if (d) wl.push(d);
    };

    for (auto mut : old_world().externals().muts())
        push(mut);

    while (!wl.empty()) {
        auto def = wl.pop();

        if (auto app = def->isa<App>()) {
            auto [axm, curry, trip] = Axm::get(app);
            bool tensor_op          = axm && curry == 0 && axm->plugin() == tensor::Plugin_Id;
            ops_seen_ |= tensor_op;
            if (Axm::isa<tensor::buf>(app)) {
                // What an array value *is* cannot be read off its type - an `(x y: I32)` group *is* a
                // `«2; I32»` - so a tensor no other tensor op mentions says so itself.
                add_tensor_ty(app->type());
            } else if (Axm::isa<tensor::splat>(app)) {
                auto [T, r] = app->callee()->as<App>()->args<2>();
                elem(T);
                add_tensor_ty(app->type());
            } else if (Axm::isa<tensor::generate>(app)) {
                auto [meta, s_out] = app->callee()->as<App>()->uncurry_args<2>();
                auto [T, r]        = meta->projs<2>();
                elem(T);
                if (!Lit::isa<u64>(r)) gate("non-literal rank of `tensor.generate`", app);
                add_tensor_ty(app->type());
            } else if (Axm::isa<tensor::map_reduce_post>(app)) {
                // result and each of the `nis` inputs / `nps` epilogue inputs are tensors.
                add_tensor_ty(app->type());
                auto mr            = btensor::MapRedPost::of(app->callee()->as<App>());
                auto [is, post_is] = app->arg()->projs<2>();
                elem(mr.To), elem(mr.Tp);
                if (auto nis_l = Lit::isa<u64>(mr.nis))
                    for (u64 i = 0; i < *nis_l; ++i) {
                        elem(mr.Tis->proj(*nis_l, i));
                        add_tensor_ty(is->proj(*nis_l, i)->type());
                    }
                if (auto nps_l = Lit::isa<u64>(mr.nps))
                    for (u64 j = 0; j < *nps_l; ++j) {
                        elem(mr.Tps->proj(*nps_l, j));
                        add_tensor_ty(post_is->proj(*nps_l, j)->type());
                    }
            } else if (Axm::isa<tensor::broadcast>(app)) {
                // result «s_out; T» and input «s_in; T» (the 3rd argument) are tensors.
                auto [T, r] = app->callee()->as<App>()->args<2>();
                elem(T);
                add_tensor_ty(app->type());
                add_tensor_ty(app->arg()->proj(3, 2)->type());
            } else if (Axm::isa<tensor::pad>(app)) {
                // callee: pad {T, r} [s_in] [mode, lo, hi]; result «s_out; T» and `input` are tensors.
                auto [Tr, s_in, params] = app->callee()->as<App>()->uncurry_args<3>();
                auto [T, r]             = Tr->projs<2>();
                auto [mode, lo, hi]     = params->projs<3>();
                elem(T);
                if (!Lit::isa<u64>(r) || !Lit::isa<u64>(mode)) gate("non-literal rank/mode of `tensor.pad`", app);
                add_tensor_ty(app->type());
                add_tensor_ty(app->arg()->proj(2, 0)->type());
            } else if (Axm::isa<tensor::concat>(app)) {
                // callee: concat {T, nis, r} [ax] {Sis}; result «s_out; T» and each input are tensors.
                auto [TnisR, ax, Sis] = app->callee()->as<App>()->uncurry_args<3>();
                auto [T, nis, r]      = TnisR->projs<3>();
                elem(T);
                auto nis_l = Lit::isa<u64>(nis);
                auto r_l   = Lit::isa<u64>(r);
                auto ax_l  = Lit::isa<u64>(ax);
                if (!nis_l || !r_l || !ax_l) {
                    gate("non-literal nis/rank/axis of `tensor.concat`", app);
                } else {
                    add_tensor_ty(app->type());
                    for (u64 i = 0; i < *nis_l; ++i) {
                        // The loop generation needs literal extents along `ax` for the prefix offsets.
                        if (!Lit::isa<u64>(Sis->proj(*nis_l, i)->proj(*r_l, *ax_l)))
                            gate("non-literal extent along the concat axis", app);
                        add_tensor_ty(app->arg()->proj(*nis_l, i)->type());
                    }
                }
            } else if (Axm::isa<tensor::if_static>(app)) {
                // Value-level binding-time dispatch; this phase's rewrite residualizes it to its
                // dynamic branch - not a tensor op.
            } else if (Axm::isa<tensor::gather>(app)) {
                auto [Tr, shapes, dim] = app->callee()->as<App>()->uncurry_args<3>();
                auto [T, r]            = Tr->projs<2>();
                elem(T);
                if (!Lit::isa<u64>(r) || !Lit::isa<u64>(dim)) gate("non-literal rank/axis of `tensor.gather`", app);
                add_tensor_ty(app->type());
                add_tensor_ty(app->arg()->proj(2, 0)->type());
                add_tensor_ty(app->arg()->proj(2, 1)->type());
            } else if (Axm::isa<tensor::scatter>(app)) {
                auto [Tr, shapes, dim] = app->callee()->as<App>()->uncurry_args<3>();
                auto [T, r]            = Tr->projs<2>();
                elem(T);
                if (!Lit::isa<u64>(r) || !Lit::isa<u64>(dim)) gate("non-literal rank/axis of `tensor.scatter`", app);
                add_tensor_ty(app->type());
                add_tensor_ty(app->arg()->proj(3, 0)->type());
                add_tensor_ty(app->arg()->proj(3, 1)->type());
                add_tensor_ty(app->arg()->proj(3, 2)->type());
            } else if (tensor_op) {
                // Any other tensor op (a symbolic `shape`, …) has no buffer-world lowering.
                gate("unbufferizable tensor op", app);
            }
            // Lams passed inside a tensor op's curry chain (combiners, affine index maps, schedule
            // nests) are element-level — TRANSITIVELY, including their local helper lams: e.g. the
            // loop-vector prefixes «r; I32» inside a schedule nest must not be mistaken for value
            // tensors of a recorded «r; I32» tensor type.
            if (is_tensor_op(app)) {
                auto args = fe::BFSWorklist<DefSet>();
                for (const App* a = app; a; a = a->callee()->isa<App>())
                    args.push(a->arg());
                while (!args.empty()) {
                    auto d = args.pop();
                    if (auto k = d->isa_mut<Lam>()) op_args_.emplace(k);
                    for (auto op : d->ops())
                        if (op) args.push(op);
                }
            }
        }

        for (auto op : def->ops())
            push(op);
        push(def->type());
    }

    for (auto mut : old_world().externals().muts())
        if (auto lam = mut->isa_mut<Lam>(); lam && is_tensor_fn(lam)) tensor_fns_.emplace(lam);
    // No tensor boundaries AND no tensor ops: nothing for the sweeps below to check.
    // (Ops without boundaries still bufferize: their value-world operands are materialized.)
    if (tensor_fns_.empty() && !ops_seen_) return;

    // Higher-order bufferized functions: a continuation parameter whose domain itself contains a Pi would
    // need boundary conversion inside nested continuation types, which `conv_boundary` does not perform.
    for (auto old_fn : tensor_fns_) {
        auto dom = old_fn->type()->dom();
        auto n   = dom->num_projs();
        for (size_t i = 0; i != n; ++i)
            if (auto pi = Pi::isa_cn(dom->proj(n, i)); pi && contains_pi(pi->dom()))
                return gate("higher-order bufferized function", old_fn);
    }

    // A partial access - an index that stops short of the element - denotes a sub-tensor, which has no buffer
    // of its own; `tensor.get`/`set` used to reject this by construction.
    auto covers_every_axis = [this](const Def* agg, const Def* index) {
        auto arr = agg->type()->isa<Arr>();
        if (!arr || !tensor_ty_.contains(arr)) return true;
        auto r = arr->shape().rank();
        auto k = Shape(index).rank();
        return !r || !k || *k >= *r;
    };

    // Second sweep: shapes the conversion cannot adapt — the value-semantics path lowers everything instead.
    fe::BFSWorklist<DefSet> wl2;
    for (auto mut : old_world().externals().muts())
        wl2.push(mut);
    while (!wl2.empty()) {
        auto def = wl2.pop();

        if (auto ex = def->isa<Extract>(); ex && !covers_every_axis(ex->tuple(), ex->index()))
            ex->blame("cannot bufferize: partial tensor read yields a sub-tensor").bail();
        if (auto in = def->isa<Insert>(); in && !covers_every_axis(in->tuple(), in->index()))
            in->blame("cannot bufferize: partial tensor write targets a sub-tensor").bail();

        for (auto op : def->ops()) {
            if (!op) continue;
            if (auto fn = op->isa_mut<Lam>()) {
                // A bufferized function referenced as a value (not as the callee of a call, and not the
                // binder back-reference of its own variable) cannot be adapted.
                if (tensor_fns_.contains(fn))
                    if (!(def->isa<App>() && def->as<App>()->callee() == op) && !def->isa<Var>())
                        return gate("bufferized function used as a value", fn);
                if (!tensor_fns_.contains(fn) && !op_args_.contains(fn) && mentions_tensor(fn->type()->dom())) {
                    // A tensor-typed function the conversion does not rewrite: an unset external keeps its
                    // value ABI, a direct-style local cannot be rebuilt as a continuation — either way a
                    // bufferized caller would pass a buffer against a value-array signature.
                    if (fn->is_external() || !Pi::isa_cn(fn->type()))
                        return gate("unconvertible tensor-typed function", fn);
                }
            }
            wl2.push(op);
        }
        if (def->type()) wl2.push(def->type());
    }
}

void LowerToMem::start() {
    collect_tensor_types(); // hard-errors on shapes the conversion cannot handle
    // Nothing tensor-related in the program: skip the whole-world rebuild entirely.
    if (tensor_fns_.empty() && !ops_seen_) return;
    consumers_ = count_consumers(old_world(), [](const Def* d) { return Axm::isa<tensor::map_reduce_post>(d); });
    auto wl    = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    while (!wl.empty()) {
        auto def = wl.pop();
        if (auto c = Axm::isa<tensor::map_reduce_post>(def))
            if (auto rks = btensor::MapRedPost::of(c->callee()->as<App>()).ranks())
                for (u64 i = 0; i != rks->nis; ++i)
                    if (auto in = c->arg()->proj(2, 0)->proj(rks->nis, i); Axm::isa<tensor::map_reduce_post>(in))
                        if (auto& rs = readers_[in]; std::ranges::find(rs, c) == rs.end()) rs.push_back(c);
        for (auto op : def->ops())
            if (op) wl.push(op);
        if (def->type()) wl.push(def->type());
    }
    RWPhase::start();
    // Every fresh-memory continuation must have found an enclosing lam to be chained into.
    assert(pending_.empty());
}

const Def* LowerToMem::buf_of(const Def* arr_ty) {
    auto& w  = new_world();
    auto arr = arr_ty->isa<Arr>();
    if (!arr) return buffer::type_buf(w.lit_nat_0(), w.tuple(), rewrite(arr_ty));
    auto shape = rewrite(*arr->shape());
    return buffer::type_buf(shape->arity(), shape, rewrite(arr->body()));
}

const Def* LowerToMem::bot_mem() {
    auto& w = new_world();
    return w.bot(w.call<mem::M>(0));
}

const Def* LowerToMem::fresh_mem() {
    auto k = mem::mut_con(new_world())->set("fresh_mem");
    pending_.push_back(k);
    return k->var();
}

const Def* LowerToMem::wrap_fresh_mem(const Def* body) {
    auto& w = new_world();
    // The last-minted continuation carries the original body; the lam ends up requesting the first memory.
    for (auto k : pending_ | std::views::reverse) {
        k->set(true, body); // filter `tt`: k vanishes as soon as AddMem substitutes the real memory
        body = w.app(w.annex<mem::fresh>(), w.tuple({w.lit_nat_0(), k}));
    }
    return body;
}

const Def* LowerToMem::rewrite(const Def* old_def) {
    // An op lowering that consumes a fresh memory references its receiving continuation's var (see
    // fresh_mem). The global memo would share such a lowering with every other function that mentions the
    // same (closed) old op - where that var would dangle - so these ops are memoized per enclosing lam.
    auto app = old_def->isa<App>();
    auto ins = old_def->isa<Insert>(); // allocates the copy it writes into
    if ((app && wants_fresh_mem(app)) || (ins && tensor_ty_.contains(ins->tuple()->type()))) {
        if (auto new_def = fe::lookup(fresh_memo_, old_def)) return new_def;
        auto new_def = app ? rewrite_imm_App(app) : rewrite_imm_Insert(ins);
        fresh_memo_.emplace(old_def, new_def);
        return new_def;
    }
    return RWPhase::rewrite(old_def);
}

bool LowerToMem::mentions_tensor(const Def* t) const {
    if (tensor_ty_.contains(t)) return true;
    if (auto sig = t->isa<Sigma>()) {
        for (auto op : sig->ops())
            if (mentions_tensor(op)) return true;
    } else if (auto pi = t->isa<Pi>()) {
        return mentions_tensor(pi->dom()); // descend into continuation domains, but never into `Arr` elements
    }
    return false;
}

bool LowerToMem::is_tensor_fn(Lam* lam) const {
    return lam->is_external() && lam->is_set() && mentions_tensor(lam->type()->dom());
}

const Def* LowerToMem::conv_boundary(const Def* t) {
    if (tensor_ty_.contains(t)) return buf_of(t);
    if (auto sig = t->isa_imm<Sigma>(); sig && mentions_tensor(sig)) {
        auto n = sig->num_ops();
        DefVec ops(n);
        for (size_t i = 0; i != n; ++i)
            ops[i] = conv_boundary(sig->op(i));
        return new_world().sigma(ops);
    }
    return rewrite(t);
}

const Def* LowerToMem::rewrite_mut_Lam(Lam* lam) {
    // Scope the fresh-memory bookkeeping: ops lowered while this body is rewritten mint their receiving
    // continuations into pending_, which are chained in front of the finished body. Nested lams anchor
    // their own requests (and their own per-lam op memo).
    auto _       = fe::Restore(pending_, {});
    auto __      = fe::Restore(fresh_memo_, {});
    auto new_lam = new_world().mut_lam(conv_type(lam));
    map(lam, new_lam);
    if (!lam->is_set()) return new_lam;

    auto ___    = enter(lam);
    auto filter = rewrite(lam->filter());
    auto body   = wrap_fresh_mem(rewrite(lam->body()));
    new_lam->set(filter, body);
    return seal(lam, new_lam);
}

const Pi* LowerToMem::conv_type(Lam* lam) {
    auto& w = new_world();
    // A bufferized function: convert tensor-typed parameters to `buffer.Buf`, including inside grouped
    // sigma parameters and continuation domains. No memory is introduced here — AddMem does that.
    if (is_tensor_fn(lam)) {
        auto dom = lam->type()->dom();
        auto n   = dom->num_projs();
        DefVec doms(n, [&](size_t i) -> const Def* {
            auto d = dom->proj(n, i);
            if (auto pi = Pi::isa_cn(d)) return w.cn(conv_boundary(pi->dom()));
            return conv_boundary(d);
        });
        return w.cn(w.sigma(doms));
    }

    // A local continuation carrying tensor types (a return continuation of a bufferized call, a join point,
    // an error continuation): convert its domain the same way. This is context-independent, so the order in
    // which references reach it does not matter.
    if (!lam->is_external() && lam->is_set() && !op_args_.contains(lam))
        if (auto pi = Pi::isa_cn(lam->type()); pi && mentions_tensor(pi->dom())) return w.cn(conv_boundary(pi->dom()));

    return rewrite(lam->type())->as<Pi>();
}

const Def* LowerToMem::rewrite_imm_App(const App* app) {
    if (Axm::isa<tensor::buf>(app)) return rewrite(app->arg()); // consumed: the buffer *is* the tensor now
    // A `tensor.if_static` still stuck at lowering time guards a runtime value: residualize to
    // its dynamic branch.
    if (Axm::isa<tensor::if_static>(app)) return rewrite(app->arg(3, 2));
    if (Axm::isa<tensor::splat>(app)) return lower_splat(app);
    if (Axm::isa<tensor::generate>(app)) return lower_generate(app);
    if (Axm::isa<tensor::broadcast>(app)) return lower_broadcast(app);
    if (Axm::isa<tensor::map_reduce_post>(app)) return lower_map_reduce(app);
    if (Axm::isa<tensor::pad>(app)) return lower_pad(app);
    if (Axm::isa<tensor::concat>(app)) return lower_concat(app);
    if (Axm::isa<tensor::gather>(app)) return lower_gather(app);
    if (Axm::isa<tensor::scatter>(app)) return lower_scatter(app);

    // Call of a bufferized function: adapt the call site.
    if (auto callee = app->callee()->isa_mut<Lam>(); callee && tensor_fns_.contains(callee))
        return lower_call(app, callee);

    // Call of a converted continuation (a local lam or a parameter var whose domain mentions a tensor):
    // materialize value-world tensor arguments into buffers. Element-level lams (op_args_) — and their
    // continuation PARAMETERS (e.g. a schedule nest's `cell`, whose «r; I32» loop vector may look like
    // a recorded tensor type) — keep value ABI.
    if (auto pi = Pi::isa_cn(app->callee()->type()); pi && mentions_tensor(pi->dom())) {
        auto elementwise = [&](const Def* callee) {
            auto lam = callee->isa_mut<Lam>();
            if (!lam) lam = param_of(callee);
            return lam && op_args_.contains(lam);
        };
        if (elementwise(app->callee())) return RWPhase::rewrite_imm_App(app);
        auto& w = new_world();
        return w.app(rewrite(app->callee()), materialize(pi->dom(), app->arg()));
    }

    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerToMem::lower_call(const App* app, Lam* old_callee) {
    auto& w         = new_world();
    auto new_callee = rewrite(old_callee);
    auto dom        = old_callee->type()->dom();
    auto n          = dom->num_projs();

    DefVec args(n);
    for (size_t i = 0; i != n; ++i) {
        auto d = dom->proj(n, i);
        auto a = app->arg()->proj(n, i);
        // Continuations pass through: their domains are converted by `rewrite_mut_Lam` to exactly the
        // domain the callee's new signature expects.
        args[i] = Pi::isa_cn(d) ? rewrite(a) : materialize(d, a);
    }
    return w.app(new_callee, w.tuple(args));
}

const Def* LowerToMem::splat_buffer(const Def* arr_ty, const Def* scalar) {
    // `buffer.lit` sets every element to `scalar`; `btensor.lower_map_reduce` fills it with a loop rather
    // than storing a monolithic literal array (which the LLVM backend cannot digest for large shapes).
    auto [bro, bso, boT] = Axm::isa<buffer::Buf>(buf_of(arr_ty))->args<3>();
    auto [m, out]        = buffer::op_lit(bro, bso, boT, bot_mem(), scalar)->projs<2>();
    return out;
}

const Def* LowerToMem::materialize(const Def* old_ty, const Def* old_arg) {
    auto& w = new_world();
    if (tensor_ty_.contains(old_ty)) {
        // A constant splat `‹s; c›` (e.g. a learning-rate or bias literal): a `buffer.init` would store the
        // whole array as one giant LLVM constant. Emit `buffer.lit` (lowered to a fill loop) instead.
        if (auto c = splat_scalar(old_arg)) return splat_buffer(old_ty, rewrite(c));
        auto v = rewrite(old_arg);
        if (Axm::isa<buffer::Buf>(v->type())) return v; // already a buffer
        if (holds_buf(v)) {
            const Def* culprit = old_arg;
            if (old_arg->num_ops() != 0)
                if (auto lam = param_of(old_arg->op(0))) culprit = lam;
            culprit->blame("cannot bufferize: tensor parameter was split into parts; annotate it with `tensor.buf`")
                .bail();
        }
        auto [br, bs, bT] = Axm::isa<buffer::Buf>(buf_of(old_ty))->args<3>();
        auto [m, buf]     = buffer::op_init(br, bs, bT, bot_mem(), v)->projs<2>();
        return buf;
    }
    if (auto sig = old_ty->isa_imm<Sigma>(); sig && mentions_tensor(sig)) {
        auto n = sig->num_ops();
        DefVec ops(n);
        for (size_t i = 0; i != n; ++i)
            ops[i] = materialize(sig->op(i), old_arg->proj(n, i));
        return w.tuple(ops);
    }
    return rewrite(old_arg);
}

const Def* LowerToMem::to_buffer(const Def* val, const Def* old) {
    if (Axm::isa<buffer::Buf>(val->type())) return val;
    return materialize(old->type(), old);
}

const Def* LowerToMem::buffer_list(const Def* list, const Def* old_list, const Def* n) {
    auto n_l = Lit::isa<u64>(n);
    if (!n_l) return list;

    DefVec ins(*n_l, [&](size_t i) { return to_buffer(list->proj(*n_l, i), old_list->proj(*n_l, i)); });
    for (auto in : ins)
        if (!Axm::isa<buffer::Buf>(in->type())) return nullptr;
    return new_world().tuple(ins);
}

std::pair<const Def*, const Def*> LowerToMem::peel_tensor(const Def* d) {
    // One Extract carries every axis, so there is no chain to walk back up and no sub-tensor to mistake the
    // base for: an index that stops short of the element yields an array type, which the caller rejects.
    auto ex = d->isa<Extract>();
    if (!ex || !tensor_ty_.contains(ex->tuple()->type())) return {nullptr, nullptr};
    return {ex->tuple(), rewrite(ex->index())};
}

const Def* LowerToMem::rewrite_imm_Extract(const Extract* extract) {
    // A partial read yields a sub-tensor, which has no buffer of its own; `collect_tensor_types` rejects it.
    if (extract->type()->isa<Arr>()) return RWPhase::rewrite_imm_Extract(extract);

    auto [old_arr, index] = peel_tensor(extract);
    if (!old_arr) return RWPhase::rewrite_imm_Extract(extract);

    // Only read through a tensor that *is* a buffer already - materializing one here would bufferize a plain
    // value tuple whose type merely coincides with some tensor's.
    auto arr = rewrite(old_arr);
    if (!Axm::isa<buffer::Buf>(arr->type())) return RWPhase::rewrite_imm_Extract(extract);

    return buffer::op_read(bot_mem(), arr, index)->proj(2, 1); // the loaded value
}

const Def* LowerToMem::rewrite_imm_Insert(const Insert* insert) {
    if (!tensor_ty_.contains(insert->tuple()->type())) return RWPhase::rewrite_imm_Insert(insert);

    auto arr = rewrite(insert->tuple()); // cf. rewrite_imm_Extract: no materialization here
    auto buf = Axm::isa<buffer::Buf>(arr->type());
    if (!buf) return RWPhase::rewrite_imm_Insert(insert);
    auto [br, bs, bT] = buf->args<3>(); // actual (folded) buffer metadata
    auto x            = rewrite(insert->value());
    auto idx          = rewrite(insert->index()); // one write, every axis

    // AlwaysAllocate policy: allocate a fresh buffer, copy the source in, then write the element.
    // This local chain is properly threaded; AddMem splices its placeholder root into the global chain.
    auto [m1, q] = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    auto m2      = buffer::op_copy(m1, q, arr);
    return buffer::op_write(m2, q, idx, x)->proj(2, 1);
}

const Def* LowerToMem::lower_splat(const App* app) {
    auto value = app->arg()->proj(2, 1);

    // MimIR folds rank-zero tensors and tensors whose literal dimensions are all
    // one into their scalar element type. Such results have no tensor boundary to
    // bufferize, so lower the splat directly to its scalar value.
    if (!app->type()->isa<Arr>()) return rewrite(value);
    return splat_buffer(app->type(), rewrite(value));
}

const Def* LowerToMem::lower_generate(const App* app) {
    auto& w            = new_world();
    auto [meta, s_out] = rewrite(app->callee())->as<App>()->uncurry_args<2>();
    auto [T, r]        = meta->projs<2>();
    auto r_l           = Lit::isa<u64>(r);
    if (!r_l) return RWPhase::rewrite_imm_App(app);
    auto rn   = *r_l;
    auto body = rewrite(app->arg());

    // A logical rank-zero tensor, or one consisting only of literal singleton
    // axes, is represented by its scalar element in MimIR.
    if (!app->type()->isa<Arr>()) {
        DefVec zeros(rn, [&](size_t) { return w.lit_i64(0); });
        return w.call(body, w.tuple(zeros));
    }

    auto out_ty       = buf_of(app->type());
    auto [br, bs, bT] = Axm::isa<buffer::Buf>(out_ty)->args<3>();
    auto result_ty    = w.sigma({w.call<mem::M>(0), out_ty});
    auto compute      = [&](Defs iters, const Def*, const Def* m) -> std::pair<const Def*, const Def*> {
        return {m, w.call(body, iters)};
    };
    auto [am, dst] = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    return btensor::build_pointwise(result_ty, am, dst, w.tuple(), s_out, rn, "generate", compute)->proj(2, 1);
}

const Def* LowerToMem::lower_broadcast(const App* app) {
    // Thin bufferization: map the SSA `tensor.broadcast` onto the buffer-world `btensor.broadcast`.
    // The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w                   = new_world();
    auto c                    = rewrite(app->callee())->as<App>();
    auto arg                  = rewrite(app->arg());
    auto [s_in, s_out, input] = arg->projs<3>();
    auto [T, r]               = c->args<2>();

    // No-op broadcast (already normalized away in most cases).
    if (s_in == s_out) return input;

    input       = to_buffer(input, app->arg()->proj(3, 2));
    auto in_buf = Axm::isa<buffer::Buf>(input->type());

    // Rank-0 source: an all-size-1 input shape folds to a plain scalar that is never recorded as a tensor
    // type, so `materialize` leaves it as a value. Broadcasting a scalar fills every element with it.
    if (!in_buf) return splat_buffer(app->type(), input);

    // Actual (size-1-folded) input/output buffer shapes — `btensor.broadcast` is parameterised by them.
    auto [bri, bsi, biT] = in_buf->args<3>();
    auto [bro, bso, boT] = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();

    // NB: no `w.call` here — `{ro, so}` (the *output* buffer shape) occur nowhere but in the result type, so
    // there is no argument for inference to read them off.
    auto op        = w.annex<btensor::broadcast>();
    op             = w.app(op, w.tuple({T, bri, bsi, bro, bso, r}));
    op             = w.app(op, w.tuple({s_in, s_out}));
    auto [am, dst] = buffer::op_alloc(bro, bso, boT, fresh_mem())->projs<2>();
    auto [m, out]  = w.app(op, w.tuple({am, dst, input}))->projs<2>();
    return out;
}

const Def*
LowerToMem::btensor_op(const btensor::MapRedPost& mr, const Def* shapes, const Def* in_tys, const Def* maps_all) {
    auto& w = new_world();

    // Wrap the pure tensor combiner `Fn [To, «nis; Tis»] → To` into the mem-threaded combiner
    // `Fn [mem.M 0, To, «nis; Tis»] → [mem.M 0, To]` that `btensor.map_reduce_post` expects.
    auto mem_ty           = w.call<mem::M>(0);
    auto inner            = mr.comb->type()->as<Pi>()->dom()->proj(2, 0); // [To, «nis; Tis»]
    auto [cTo, ins_ty]    = inner->projs<2>();
    auto memcomb          = w.mut_fun(w.sigma({mem_ty, cTo, ins_ty}), w.sigma({mem_ty, cTo}))->set("memComb");
    auto [cdata, cret]    = memcomb->vars<2>();
    auto [cm, cacc, cins] = cdata->projs<3>();
    auto after            = w.mut_con(cTo)->set("afterComb");
    after->app(true, cret, w.tuple({cm, after->var()}));
    memcomb->set(true, w.app(mr.comb, w.tuple({w.tuple({cacc, cins}), after})));

    // Likewise wrap the pure epilogue `Fn [To, «nps; Tps»] → Tp` into
    // `Fn [mem.M 0, To, «nps; Tps»] → [mem.M 0, Tp]`.
    auto pTp               = mr.Tp;
    auto pin               = mr.post->type()->as<Pi>()->dom()->proj(2, 0); // [To, «nps; Tps»]
    auto [pTo, pexts_ty]   = pin->projs<2>();
    auto mempost           = w.mut_fun(w.sigma({mem_ty, pTo, pexts_ty}), w.sigma({mem_ty, pTp}))->set("memPost");
    auto [pdata, pret]     = mempost->vars<2>();
    auto [pm, pacc, pexts] = pdata->projs<3>();
    auto pafter            = w.mut_con(pTp)->set("afterPost");
    pafter->app(true, pret, w.tuple({pm, pafter->var()}));
    mempost->set(true, w.app(mr.post, w.tuple({w.tuple({pacc, pexts}), pafter})));

    // NB: no `w.call` here — the meta groups are *forwarded* from the tensor op on purpose. Leaving them to
    // inference would re-derive `{Tis, Ris, Sis}` from the `buffer.Buf` operands, whose literal size-1 axes are
    // already folded away, so they would no longer agree with the logical shapes the loop generation iterates.
    auto op = w.annex<btensor::map_reduce_post>();
    op      = w.app(op, mr.nis_nps);
    op      = w.app(op, mr.meta);
    op      = w.app(op, shapes);
    op      = w.app(op, in_tys);
    op      = w.app(op, w.tuple({memcomb, mr.init, mempost}));
    op      = w.app(op, mr.map_out);
    return w.app(op, maps_all);
}

const Def* LowerToMem::lower_map_reduce(const App* app) {
    // Thin bufferization: map the SSA `tensor.map_reduce_post` onto the buffer-world `btensor.map_reduce_post`,
    // reusing the (rewritten) meta. The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    if (auto res = lower_compute_at(app)) return res;

    auto& w            = new_world();
    auto c             = rewrite(app->callee())->as<App>();
    auto args          = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
    auto [is, post_is] = args->projs<2>();

    auto mr = btensor::MapRedPost::of(c);

    // The generic rewrite rebuilds the argument tuple with its stale value-array type even when its
    // elements were converted to buffers, which would not be assignable to the op's
    // `«nis; buffer.Buf …»` domain - hence the re-tupling in `buffer_list`.
    auto [old_is, old_post_is] = app->arg()->projs<2>();
    is                         = buffer_list(is, old_is, mr.nis);
    post_is                    = buffer_list(post_is, old_post_is, mr.nps);
    if (!is || !post_is) return RWPhase::rewrite_imm_App(app); // leave it alone

    auto op        = btensor_op(mr, mr.shapes, mr.in_tys, mr.maps_all);
    auto [am, dst] = buffer::op_alloc(mr.Ro, mr.So, mr.Tp, fresh_mem())->projs<2>();
    auto [m, out]  = w.app(op, w.tuple({am, dst, is, post_is}))->projs<2>();
    return scalar_result(app, m, out);
}

namespace {

/// The body of @p f at a scratch binder's var: `{var, body}`.
std::pair<const Def*, const Def*> probe(const Def* f) {
    auto& w  = f->world();
    auto pi  = f->type()->as<Pi>();
    auto var = w.mut_lam(pi->dom(), pi->codom())->var();
    return {var, w.app(f, var)};
}

/// Whether an old def in @p to occurs in @p from, where the defs of @p skip and of mutables end the search.
bool reaches(const Def* from, const DefSet& to, const Def* skip = nullptr) {
    auto seen = DefSet();
    auto todo = DefVec{from};
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (to.contains(def)) return true;
        if (def == skip || def->isa_mut() || !seen.emplace(def).second) continue;
        for (auto op : def->ops())
            if (op) todo.push_back(op);
    }
    return false;
}

} // namespace

const Def* LowerToMem::lower_compute_at(const App* app) {
    using F  = btensor::MapRedPost;
    auto& w  = new_world();
    auto mr  = F::of(rewrite(app->callee())->as<App>());
    auto rks = mr.ranks();
    if (!rks || mr.sched_field(F::At) != w.lit_tt()) return nullptr;
    if (Lit::isa<u64>(mr.sched_field(F::Rev)).value_or(0) != 0)
        if (auto res = lower_reverse(app, mr)) return res;

    auto [old_is, old_post_is] = app->arg()->projs<2>();
    auto post_is               = buffer_list(rewrite(old_post_is), old_post_is, mr.nps);
    if (!post_is) return nullptr;

    // Lowering an input whole may lower another op with a compute_at.
    auto outer = std::exchange(alloc_mem_, fresh_mem());
    auto res   = lower_staged(
        {mr, mr, std::vector<bool>(rks->nloops()), std::vector<bool>(rks->nloops()), old_is, post_is, app, false, 0, 1},
        nullptr, nullptr);
    alloc_mem_ = outer;
    if (!res) return nullptr;
    auto [m, out] = res->projs<2>();
    return scalar_result(app, m, out);
}

const DefSet& LowerToMem::local_defs() {
    if (auto lam = curr_mut(); lam != local_of_) {
        local_of_ = lam;
        local_.clear();
        auto todo = lam && lam->is_set() ? DefVec(lam->ops().begin(), lam->ops().end()) : DefVec();
        while (!todo.empty()) {
            auto def = todo.back();
            todo.pop_back();
            if (!def || def->isa_mut() || !local_.emplace(def).second) continue;
            for (auto op : def->ops())
                todo.push_back(op);
        }
    }
    return local_;
}

const Def* LowerToMem::lower_reverse(const App* app, const btensor::MapRedPost& mr) {
    using F     = btensor::MapRedPost;
    auto& w     = new_world();
    auto rev_of = [](const F& op) { return Lit::isa<u64>(op.sched_field(F::Rev)).value_or(0); };
    auto dim_of = [&](const F& op, u64 j) { return Lit::isa<u64>(w.app(op.sched_field(F::AtDim), w.lit_nat(j))); };
    auto rks    = *mr.ranks();
    auto j      = u64(std::countr_zero(rev_of(mr)));
    auto dim    = dim_of(mr, j);

    auto give_up = [&](const char* why) -> const Def* {
        log().w("reverse_compute_at of {} inside dim {} of the producer of its input {} not realized: {}", app,
                dim ? *dim : 0, j, why);
        return nullptr;
    };
    if (!dim || j >= rks.nis) return give_up("no such input");

    auto old_p = Axm::isa<tensor::map_reduce_post>(app->arg()->proj(2, 0)->proj(rks.nis, j));
    if (!old_p) return give_up("the input is not a map_reduce_post");
    if (fresh_memo_.contains(old_p)) return give_up("another op has computed the producer already");
    auto mp  = F::of(rewrite(old_p->callee())->as<App>());
    auto prk = mp.ranks();
    if (!prk || *dim >= prk->rp) return give_up("not a parallel dim of the producer");
    auto prn = prk->nloops();

    // The action after the nest inside the producer's loop of `dim`; it gets the loop-vector prefix.
    auto mem_ty      = w.call<mem::M>(0);
    auto post        = w.mut_fun(w.sigma({mem_ty, w.pi(w.type_nat(), w.type_i32())}), mem_ty)->set("tile_after");
    auto [pdata, pk] = post->vars<2>();
    auto [pm, get]   = pdata->projs<2>();

    // The box of the producer one iteration writes.
    auto origin = DefVec(prn, [&](size_t d) -> const Def* {
        return d <= *dim ? w.call<core::bitcast>(w.type_nat(), w.app(get, w.lit_nat(d))) : w.lit_nat_0();
    });
    auto ext    = DefVec(prn, [&](size_t d) { return d <= *dim ? w.lit_nat_1() : mp.Sr->proj(prn, d); });
    auto box    = w.app(w.app(w.app(w.annex<affine::image>(), {mp.Rn, mp.Ro}), mp.map_out),
                        w.tuple({w.tuple(origin), w.tuple(ext), mp.So}));
    // A bound that is not exact may hold cells a later iteration writes.
    for (u64 a = 0; a != prk->ro; ++a)
        if (box->proj(4, 3)->proj(prk->ro, a) != w.lit_tt())
            return give_up("the box one iteration of the producer writes is not known exactly");
    // A cell another iteration of the loops up to `dim` writes as well is not final after this one.
    auto writers = w.app(w.app(w.app(w.annex<affine::preimage>(), {mp.Rn, mp.Ro}), mp.map_out),
                         w.tuple({box->proj(4, 0), box->proj(4, 2), mp.Sr}));
    if (writers->proj(4, 3) != w.lit_tt()) return give_up("the iterations that write the box are not known exactly");
    for (u64 d = 0; d <= *dim; ++d)
        if (writers->proj(4, 0)->proj(prn, d) != origin[d] || writers->proj(4, 2)->proj(prn, d) != w.lit_nat_1())
            return give_up("other iterations of the producer write the box too");
    u64 runs    = 1;
    for (u64 d = 0; d <= *dim; ++d)
        runs *= Lit::isa<u64>(mp.Sr->proj(prn, d)).value_or(1);

    // A part writes into the op's whole result and may stop before its literal size.
    struct Part {
        const App* app;
        F mr, mc;
        u64 j;
        std::vector<bool> shifted, clipped, post_p;
        const Def* buf = nullptr;
    };
    const char* why = nullptr;
    auto plan       = [&](const App* c, const F& op, u64 jc) -> std::optional<Part> {
        auto fail = [&](const char* r) {
            why = r;
            return std::optional<Part>();
        };
        auto rk = *op.ranks();
        auto rn = rk.nloops();
        if (mp.So != op.Sis->proj(rk.nis, jc)) return fail("the input's shape is not the producer's");
        for (u64 k = 0; k != rk.nis; ++k)
            if (k != jc && c->arg()->proj(2, 0)->proj(rk.nis, k) == old_p)
                return fail("the op reads the producer through another input too");
        // An epilogue input that is the producer, read at the cells the input reads, reads the box as well.
        auto pi_out  = op.map_out->type()->as<Pi>();
        auto o_out   = w.mut_lam(pi_out->dom(), pi_out->codom())->var();
        auto post_p  = std::vector<bool>(rk.nps);
        auto old_pis = c->arg()->proj(2, 1);
        for (u64 k = 0; k != rk.nps; ++k)
            if (old_pis->proj(rk.nps, k) == old_p) {
                auto at = w.app(op.post_maps->proj(rk.nps, k), w.app(op.map_out, o_out));
                if (at != w.app(op.maps->proj(rk.nis, jc), o_out))
                    return fail("the op's epilogue reads the producer elsewhere than its input");
                post_p[k] = true;
            }
        // The box at its static size bounds the part by a static size too.
        auto part = [&](const Def* ext) {
            return w.app(
                w.app(w.app(w.annex<affine::preimage>(), {op.Rn, op.Ris->proj(rk.nis, jc)}), op.maps->proj(rk.nis, jc)),
                w.tuple({box->proj(4, 0), ext, op.Sr}));
        };
        auto [lo, len, _, exact] = part(box->proj(4, 1))->projs<4>();
        auto cap                 = part(box->proj(4, 2))->proj(4, 2);
        if (exact != w.lit_tt()) return fail("the part of this op that reads one box is not exact");
        auto caps = DefVec(rn, [&](size_t d) { return cap->proj(rn, d); });
        for (u64 d = 0; d != rn; ++d) {
            if (!Lit::isa(caps[d])) return fail("the part has no literal size");
            if (d >= rk.rp && (caps[d] != op.Sr->proj(rn, d) || lo->proj(rn, d) != w.lit_nat_0()))
                return fail("the part does not span a reduction dim");
        }

        auto lit   = [&](const Def* n) { return w.app(w.annex<affine::lit>(), n); };
        auto shift = [&](const Def* f, const char* name) -> const Def* {
            auto pi  = f->type()->as<Pi>();
            auto lam = w.mut_lam(pi->dom(), pi->codom())->set(name);
            auto o   = lam->var();
            auto at  = w.tuple(DefVec(rn, [&](size_t d) {
                return d < rk.rp ? w.call(affine::op::add, Defs{o->proj(rn, d), lit(lo->proj(rn, d))}) : o->proj(rn, d);
            }));
            lam->set(true, w.app(f, at));
            return lam;
        };
        auto mc     = op;
        mc.Sr       = w.tuple(DefVec(rn, [&](size_t d) { return d < rk.rp ? caps[d] : op.Sr->proj(rn, d); }));
        mc.maps     = w.tuple(DefVec(rk.nis, [&](size_t i) { return shift(op.maps->proj(rk.nis, i), "tile_map"); }));
        mc.map_out  = shift(op.map_out, "tile_out");
        auto starts = DefVec(rn), ends = DefVec(rn);
        for (u64 d = 0; d != rk.rp; ++d) {
            starts[d] = lo->proj(rn, d);
            if (len->proj(rn, d) != caps[d]) ends[d] = w.call(core::nat::add, Defs{starts[d], len->proj(rn, d)});
        }
        mc.sched     = clip_lim(op, starts, ends);
        mc.shapes    = w.tuple({op.So, mc.Sr, mc.sched});
        mc.maps_all  = w.tuple({mc.maps, op.post_maps});
        auto shifted = std::vector<bool>(rn), clipped = std::vector<bool>(rn);
        for (u64 d = 0; d != rk.rp; ++d)
            shifted[d] = true, clipped[d] = ends[d] != nullptr;
        return Part{c, op, mc, jc, shifted, clipped, post_p};
    };

    auto first = plan(app, mr, j);
    if (!first) return give_up(why);
    auto parts = std::vector<Part>{*first};

    // Only the ops of this body share its memo and memory.
    auto mates = std::vector<std::pair<const App*, u64>>();
    for (auto c : readers_[old_p]) {
        if (c == app || !local_defs().contains(c)) continue;
        auto op = F::of(rewrite(c->callee())->as<App>());
        auto rk = op.ranks();
        if (!rk || rev_of(op) == 0) continue;
        auto jc = u64(std::countr_zero(rev_of(op)));
        if (jc < rk->nis && c->arg()->proj(2, 0)->proj(rk->nis, jc) == old_p && dim_of(op, jc) == dim)
            mates.emplace_back(c, jc);
    }
    auto inputs_reach = [&](const App* c, const DefSet& to) {
        auto [is, post_is] = c->arg()->projs<2>();
        return reaches(is, to, old_p) || reaches(post_is, to);
    };
    // An op that reads another one computed inside the producer reads the producer whole once that one has run there.
    for (auto [c, _] : mates)
        if (inputs_reach(app, DefSet{c})) {
            rewrite(c);
            return give_up("an input depends on another op computed inside the producer");
        }

    for (auto [c, jc] : mates) {
        auto op   = F::of(rewrite(c->callee())->as<App>());
        auto mine = DefSet();
        for (const auto& pt : parts)
            mine.emplace(pt.app);
        if (inputs_reach(c, mine)
            || std::ranges::any_of(parts, [&](const Part& pt) { return inputs_reach(pt.app, DefSet{c}); })) {
            log().d("reverse: {} reads {} whole, as its inputs and an op computed inside it depend on each other", c,
                    old_p);
            continue;
        }
        if (auto pt = plan(c, op, jc))
            parts.push_back(*pt);
        else
            log().w("reverse_compute_at of {} inside dim {} of the producer of its input {} not realized: {}", c, *dim,
                    jc, why);
    }

    auto old_p_post = old_p->arg()->proj(2, 1);
    auto p_post_is  = buffer_list(rewrite(old_p_post), old_p_post, mp.nps);
    if (!p_post_is) return give_up("an epilogue input is not a buffer");

    auto outer = std::exchange(alloc_mem_, fresh_mem());
    // A give-up from here on leaves nothing lowered on the abandoned memory behind in the memo.
    auto memo   = fresh_memo_;
    auto revert = [&](const char* r) {
        alloc_mem_  = outer;
        fresh_memo_ = std::move(memo);
        return give_up(r);
    };
    for (auto& pt : parts)
        std::tie(alloc_mem_, pt.buf) = buffer::op_alloc(pt.mr.Ro, pt.mr.So, pt.mr.Tp, alloc_mem_)->projs<2>();
    auto [bm, p_full] = buffer::op_alloc(mp.Ro, mp.So, mp.Tp, alloc_mem_)->projs<2>();
    alloc_mem_        = bm;

    // On a fresh memory, a buffer read only inside the producer's loop could collapse with an equal one here.
    auto whole = [&](const Def* old) -> const Def* {
        if (auto old_k = Axm::isa<tensor::map_reduce_post>(old)) {
            if (auto memo = fe::lookup(fresh_memo_, old_k)) return to_buffer(memo, old_k);
            auto mk     = F::of(rewrite(old_k->callee())->as<App>());
            auto kr     = mk.ranks();
            auto old_kp = old_k->arg()->proj(2, 1);
            if (auto kp_post = kr && rev_of(mk) == 0 ? buffer_list(rewrite(old_kp), old_kp, mk.nps) : nullptr) {
                auto [km, kb] = buffer::op_alloc(mk.Ro, mk.So, mk.Tp, alloc_mem_)->projs<2>();
                alloc_mem_    = km;
                auto none     = std::vector<bool>(kr->nloops());
                if (auto res = lower_staged({mk, mk, none, none, old_k->arg()->proj(2, 0), kp_post, old_k, false, 0, 1},
                                            nullptr, kb)) {
                    auto [rm, rb]      = res->projs<2>();
                    alloc_mem_         = rm;
                    fresh_memo_[old_k] = scalar_result(old_k, rm, rb);
                    return rb;
                }
            }
        }
        return to_buffer(rewrite(old), old);
    };
    auto bit     = [](u64 mask, u64 i) { return i < 64 && (mask >> i) & 1; };
    auto given   = std::vector<DefVec>();
    auto post_is = DefVec();
    for (const auto& pt : parts) {
        auto rk   = *pt.mr.ranks();
        auto rec  = pt.mr.sched_record();
        auto skip = rev_of(pt.mr) | Lit::isa<u64>(rec->proj(F::Num_Fields, F::Cache)).value_or(0);
        if (rec->proj(F::Num_Fields, F::At) == w.lit_tt())
            skip |= Lit::isa<u64>(rec->proj(F::Num_Fields, F::AtSlots)).value_or(0);
        auto& g = given.emplace_back(rk.nis);
        g[pt.j] = p_full;
        for (u64 k = 0; k != rk.nis; ++k)
            if (auto old = pt.app->arg()->proj(2, 0)->proj(rk.nis, k);
                !bit(skip, k) && Axm::isa<tensor::map_reduce_post>(old))
                g[k] = whole(old);
        auto old_post = pt.app->arg()->proj(2, 1);
        auto posts    = DefVec(rk.nps, [&](size_t k) { return pt.post_p[k] ? p_full : whole(old_post->proj(rk.nps, k)); });
        if (!std::ranges::all_of(posts, [](const Def* b) { return bool(Axm::isa<buffer::Buf>(b->type())); }))
            return revert("an epilogue input is not a buffer");
        post_is.emplace_back(w.tuple(posts));
    }

    // Each part computes the boxes of the other producers it runs in ahead of itself.
    const Def* chain = pm;
    for (size_t i = 0; i != parts.size(); ++i) {
        const auto& pt = parts[i];
        auto res = lower_staged({pt.mc, pt.mr, pt.shifted, pt.clipped, pt.app->arg()->proj(2, 0), post_is[i], pt.app,
                                 false, 0, runs, 0, nullptr, given[i], rev_of(pt.mr) & ~(u64(1) << pt.j)},
                                chain, pt.buf);
        if (!res) return revert("an input is not a buffer");
        chain = res->proj(2, 0);
    }
    post->app(true, pk, chain);
    // Lowering the parts' inputs may have lowered the producer, with this op inside it or not.
    if (fresh_memo_.contains(old_p)) {
        alloc_mem_ = outer;
        if (auto done = fresh_memo_.find(app); done != fresh_memo_.end()) return done->second;
        return give_up("lowering the inputs computed the producer already");
    }

    auto res   = lower_staged({mp, mp, std::vector<bool>(prn), std::vector<bool>(prn), old_p->arg()->proj(2, 0),
                               p_post_is, old_p, false, 0, 1, *dim, post},
                              nullptr, p_full);
    if (!res) return revert("the producer's inputs are not buffers");
    alloc_mem_ = outer;

    // The producer's other readers read its result; the ops computed inside it are whole once it is done.
    auto [m, p_out]    = res->projs<2>();
    fresh_memo_[old_p] = scalar_result(old_p, m, p_out);
    const Def* result  = nullptr;
    for (const auto& pt : parts) {
        auto after   = w.app(w.app(w.annex<buffer::after>(), pt.buf->type()->as<App>()->arg()), w.tuple({m, pt.buf}));
        auto [cm, c] = after->projs<2>();
        m            = cm;
        log().d("reverse: computing {} inside dim {} of the producer of its input {}", pt.app, *dim, pt.j);
        if (pt.app == app)
            result = scalar_result(app, cm, c);
        else
            fresh_memo_[pt.app] = scalar_result(pt.app, cm, c);
    }
    return result;
}

const Def* LowerToMem::clip_lim(const btensor::MapRedPost& op, const DefVec& starts, const DefVec& ends) {
    using F     = btensor::MapRedPost;
    auto& w     = new_world();
    auto rec    = op.sched_record();
    auto own    = rec->proj(F::Num_Fields, F::Lim);
    auto n      = starts.size();
    auto i32    = w.type_i32();
    auto get_ty = w.pi(w.type_nat(), i32);
    auto i32_of = [&](const Def* x) { return w.call<core::bitcast>(i32, x); };
    auto select = [&](const Def* c, const Def* t, const Def* f) { return w.extract(w.tuple({f, t}), c); };
    auto is     = [&](const Def* x, u64 d) { return w.call(core::ncmp::e, Defs{x, w.lit_nat(d)}); };

    auto lim       = w.mut_lam(w.pi(w.sigma({get_ty, w.type_nat()}), w.sigma({w.type_bool(), i32})))->set("tile_lim");
    auto [get, jj] = lim->vars<2>();
    // The op's own `lim` reads the loops of the whole op, where a shifted dim starts at the tile.
    auto whole = w.mut_lam(get_ty)->set("tile_get");
    auto e     = whole->var();
    auto at    = w.app(get, e);
    for (u64 d = 0; d != n; ++d)
        if (starts[d])
            at = select(is(e, d), w.call(core::wrap::add, core::Mode::none, Defs{w.app(get, e), i32_of(starts[d])}),
                        at);
    whole->set(true, at);

    const Def* body = w.app(own, w.tuple({whole, jj}));
    for (u64 d = 0; d != n; ++d)
        if (ends[d]) {
            auto mine = w.app(own, w.tuple({whole, w.lit_nat(d)}));
            auto end  = i32_of(ends[d]);
            auto cut  = w.call(core::icmp::ul, Defs{mine->proj(2, 1), end});
            end       = select(mine->proj(2, 0), select(cut, mine->proj(2, 1), end), end);
            auto stop = w.call(core::wrap::sub, core::Mode::none, Defs{end, i32_of(starts[d])});
            body      = select(is(jj, d), w.tuple({w.lit_tt(), stop}), body);
        }
    lim->set(true, body);

    auto fields    = DefVec(F::Num_Fields, [&](size_t f) { return rec->proj(F::Num_Fields, f); });
    fields[F::Lim] = lim;
    return w.app(w.annex<btensor::mk_sched>(), w.tuple(fields));
}

const Def* LowerToMem::lower_staged(const Staged& s, const Def* mem, const Def* dst) {
    using F  = btensor::MapRedPost;
    auto& w  = new_world();
    auto& mr = s.mr;
    auto rks = mr.ranks();
    if (!rks) return nullptr;
    auto rn = rks->nloops();

    // An op's own intent computes each input inside the loop of its dim, an enclosing tile_and_fuse ahead of the op.
    auto rec    = mr.sched_record();
    auto field  = [&](F::Field f) { return rec->proj(F::Num_Fields, f); };
    auto own    = field(F::At) == w.lit_tt();
    auto fuse   = own ? field(F::Fuse) == w.lit_tt() : s.fuse;
    auto budget = own ? Lit::isa<u64>(field(F::Budget)).value_or(0) : s.budget;
    auto slots  = Lit::isa<u64>(field(F::AtSlots)).value_or(0);
    auto dim_of = [&](u64 j) { return own ? Lit::isa<u64>(w.app(field(F::AtDim), w.lit_nat(j))) : std::nullopt; };

    auto cur     = std::optional<u64>();
    auto give_up = [&](u64 j, const char* why) {
        if (fuse)
            log().d("tile_and_fuse: input {} of {} is computed whole: {}", j, s.who, why);
        else
            log().w("compute_at of input {} inside dim {} of {} not realized: {}", j, cur.value_or(rn), s.who, why);
        return false;
    };

    // The action computing the tiles inside the loop of a dim; it gets the loop-vector prefix.
    // `fixed` names the dims of that loop and of the loops around it, which a hoisted nest reorders.
    struct Act {
        Lam* run;
        const Def *get, *mem;
        std::vector<bool> fixed;
        bool used = false;
    };
    auto mem_ty = w.call<mem::M>(0);
    auto acts   = std::vector<std::optional<Act>>(rn);
    auto act_at = [&](u64 d) -> Act& {
        if (!acts[d]) {
            auto run         = w.mut_fun(w.sigma({mem_ty, w.pi(w.type_nat(), w.type_i32())}), mem_ty)->set("tile_run");
            auto [rdata, rk] = run->vars<2>();
            auto [rmem, get] = rdata->projs<2>();
            acts[d]          = Act{run, get, rmem, mr.outer(d, rn)};
        }
        return *acts[d];
    };
    auto pre      = mem;
    auto no_fixed = std::vector<bool>(rn);
    auto hoist    = Lit::isa<u64>(field(F::Hoist)).value_or(0);

    auto lit = [&](const Def* n) { return w.app(w.annex<affine::lit>(), n); };
    auto sub = [&](const Def* a, const Def* b) { return w.call(affine::op::sub, Defs{a, b}); };
    auto add = [&](const Def* a, const Def* b) { return w.call(affine::op::add, Defs{a, b}); };
    auto nat = [&](const Def* d) { return Lit::isa<u64>(d); };

    auto ins         = DefVec(rks->nis);
    auto write_backs = std::vector<std::pair<const App*, const Def*>>();
    auto Sis         = DefVec(rks->nis, [&](size_t j) { return mr.Sis->proj(rks->nis, j); });
    auto maps        = DefVec(rks->nis, [&](size_t j) { return mr.maps->proj(rks->nis, j); });

    auto ahead = [&](u64 j) { return j < 64 && (s.ahead >> j) & 1; };
    auto stage = [&](u64 j) -> bool {
        auto dim = ahead(j) ? std::nullopt : dim_of(j);
        cur      = dim;
        if (own && !dim && !ahead(j)) return give_up(j, "its loop is not known");
        if (dim && *dim >= rks->rp + hoist) return give_up(j, "not a parallel or a hoisted reduction dim");
        if (dim && mr.inside_red(*dim)) return give_up(j, "the vector loop runs inside the reduction loops");
        auto act          = dim ? &act_at(*dim) : nullptr;
        const auto& fixed = act ? act->fixed : no_fixed;
        auto get          = act ? act->get : nullptr;

        // The producer: a map_reduce_post that writes through a projection.
        auto old_p = Axm::isa<tensor::map_reduce_post>(s.old_is->proj(rks->nis, j));
        if (!old_p) return give_up(j, "the input is not a map_reduce_post, e.g. fuse_tensor inlined it");
        for (u64 k = 0; k != rks->nis; ++k)
            if (k != j && s.old_is->proj(rks->nis, k) == old_p)
                return give_up(j, "the op reads the input through another slot too");
        auto mp  = F::of(rewrite(old_p->callee())->as<App>());
        auto prk = mp.ranks();
        if (!prk) return give_up(j, "the input's ranks are not known");
        auto old_pp = old_p->arg()->proj(2, 1);
        auto p_post = prk->nps == 0 ? w.tuple() : buffer_list(rewrite(old_pp), old_pp, mp.nps);
        if (!p_post) return give_up(j, "an epilogue input of the input is not a buffer");
        auto prn = prk->nloops();

        auto [ovar, obody] = probe(mp.map_out);
        auto axis_of       = std::vector<std::optional<u64>>(prn);
        for (u64 a = 0; a != prk->ro; ++a) {
            auto e  = obody->proj(prk->ro, a);
            auto dd = std::optional<u64>();
            if (e == ovar) dd = 0;
            if (auto ex = e->isa<Extract>(); ex && ex->tuple() == ovar) dd = Lit::isa<u64>(ex->index());
            if (!dd || *dd >= prn || axis_of[*dd] || mp.Sr->proj(prn, *dd) != mp.So->proj(prk->ro, a))
                return give_up(j, "the input's write map is not a projection");
            axis_of[*dd] = a;
        }

        auto map = maps[j];
        auto pro = Lit::isa<u64>(mr.Ris->proj(rks->nis, j));
        if (!pro || *pro != prk->ro) return give_up(j, "the read rank does not match the input");
        // Unit axes fold «1, 4» and «4, 1» to one type, so only the shapes tell whether the read and the write agree.
        if (mp.So != Sis[j]) return give_up(j, "the input's shape is not the producer's");

        // The box one run of the tile reads: the fixed dims at a run-time origin, the rest whole from 0.
        auto image = [&](const F& op, const Def* f, auto lo, auto ext) {
            auto n   = op.ranks()->nloops();
            auto box = w.tuple({w.tuple(DefVec(n, lo)), w.tuple(DefVec(n, ext)), mp.So});
            return w.app(w.app(w.app(w.annex<affine::image>(), {op.Rn, op.Ris->proj(rks->nis, j)}), f), box);
        };
        auto origin = [&](size_t d) -> const Def* {
            return fixed[d] ? w.call<core::bitcast>(w.type_nat(), w.app(get, w.lit_nat(d))) : w.lit_nat_0();
        };
        auto zero = [&](size_t) -> const Def* { return w.lit_nat_0(); };
        auto one  = [&](size_t) -> const Def* { return w.lit_nat_1(); };
        auto box  = image(mr, map, origin, [&](size_t d) { return fixed[d] ? w.lit_nat_1() : mr.Sr->proj(rn, d); });

        auto caps = DefVec(prk->ro, [&](size_t a) { return box->proj(4, 2)->proj(prk->ro, a); });
        if (!std::ranges::all_of(caps, [&](const Def* cap) { return nat(cap).has_value(); }))
            return give_up(j, "the box has no literal size");
        auto exact = [&](const Def* img, u64 r) {
            for (u64 a = 0; a != r; ++a)
                if (img->proj(4, 3)->proj(r, a) != w.lit_tt()) return false;
            return true;
        };
        // Inside the loop of `dim`, the read can only refer to the box by its read of the iteration's corner, which an
        // inexact image of that single point does not pin down.
        if (dim) {
            auto corner = image(mr, map, origin, one);
            if (!exact(corner, *pro)) return give_up(j, "the read of the corner is not known exactly");
            if (box->proj(4, 0) != corner->proj(4, 0))
                return give_up(j, "the box does not start at the read of the corner");
        }
        // A box starts at most at the read of its corner, which the whole op reaches with the fixed dims spanning their
        // range and each shifted one the starts of the tiles, which fit the whole op.
        auto& g    = s.global;
        auto gn    = g.ranks()->nloops();
        auto reach = image(g, g.maps->proj(rks->nis, j), zero, [&](size_t d) -> const Def* {
            if (fixed[d]) return g.Sr->proj(gn, d);
            if (!s.shifted[d]) return w.lit_nat_1();
            if (s.clipped[d]) return g.Sr->proj(gn, d);
            return w.call(core::nat::add,
                          Defs{w.call(core::nat::sub, Defs{g.Sr->proj(gn, d), mr.Sr->proj(rn, d)}), w.lit_nat_1()});
        });
        // A box that may leave the input is computed only up to the input's end.
        u64 tile = 1, whole = 1;
        auto clip = std::vector<bool>(prk->ro);
        for (u64 a = 0; a != prk->ro; ++a) {
            auto cap = *nat(caps[a]);
            auto so  = nat(mp.So->proj(prk->ro, a));
            if (!so || cap > *so) return give_up(j, "the box is larger than the input");
            auto lo = nat(reach->proj(4, 0)->proj(prk->ro, a)), ext = nat(reach->proj(4, 1)->proj(prk->ro, a));
            clip[a] = !lo || !ext || *lo + *ext - 1 + cap > *so;
            tile *= cap, whole *= *so;
        }

        auto runs = s.runs;
        for (u64 d = 0; d != rn; ++d)
            if (fixed[d]) {
                auto n = nat(mr.Sr->proj(rn, d));
                if (!n) return give_up(j, "the loops around the tile have no literal extent");
                runs *= *n;
            }
        if (fuse && tile * runs > budget * whole) return give_up(j, "the tiles would recompute it beyond the budget");

        // Another reader of the producer gets the whole buffer the tiles fill, so the op must read every cell.
        auto fills = [&] {
            auto all = image(g, g.maps->proj(rks->nis, j), zero, [&](size_t d) { return g.Sr->proj(gn, d); });
            if (!exact(all, prk->ro)) return false;
            for (u64 a = 0; a != prk->ro; ++a)
                if (all->proj(4, 0)->proj(prk->ro, a) != w.lit_nat_0()
                    || all->proj(4, 1)->proj(prk->ro, a) != mp.So->proj(prk->ro, a))
                    return false;
            // The axes cover the box jointly only where no loop moves two of them, as the diagonal `(i, i)` does.
            auto gm   = g.maps->proj(rks->nis, j);
            auto gpi  = gm->type()->as<Pi>();
            auto o    = w.mut_lam(gpi->dom(), gpi->codom())->var();
            auto read = w.app(gm, o);
            for (u64 d = 0; d != gn; ++d) {
                auto still = w.app(
                    gm, w.tuple(DefVec(gn, [&](size_t e) { return e == d ? lit(w.lit_nat_0()) : o->proj(gn, e); })));
                auto moved = 0;
                for (u64 a = 0; a != prk->ro; ++a)
                    moved += read->proj(prk->ro, a) != still->proj(prk->ro, a);
                if (moved > 1) return false;
            }
            return true;
        };
        auto uses   = consumers_.find(old_p);
        auto shared = !mem && dim && uses != consumers_.end() && uses->second > 1 && fills();
        if (shared && fresh_memo_.contains(old_p)) return give_up(j, "another op computes the input whole");

        auto So_t         = w.tuple(caps);
        auto [am, tile_b] = buffer::op_alloc(mp.Ro, shared ? mp.So : So_t, mp.Tp, alloc_mem_)->projs<2>();
        alloc_mem_        = am;

        // The producer on the tile: its parallel dims are shifted to the box's start on the axis they write and
        // span the tile; its reduction dims stay whole.
        auto base  = box->proj(4, 0);
        auto shift = [&](const Def* f, const char* name) -> const Def* {
            auto pi  = f->type()->as<Pi>();
            auto lam = w.mut_lam(pi->dom(), pi->codom())->set(name);
            auto o   = lam->var();
            auto at  = w.tuple(DefVec(prn, [&](size_t dd) {
                return axis_of[dd] ? add(o->proj(prn, dd), lit(base->proj(prk->ro, *axis_of[dd]))) : o->proj(prn, dd);
            }));
            lam->set(true, w.app(f, at));
            return lam;
        };
        auto pmaps = DefVec(prk->nis, [&](size_t i) { return shift(mp.maps->proj(prk->nis, i), "tile_map"); });
        auto Sr_t
            = w.tuple(DefVec(prn, [&](size_t dd) { return axis_of[dd] ? caps[*axis_of[dd]] : mp.Sr->proj(prn, dd); }));

        auto mt = mp;
        mt.So   = shared ? mp.So : So_t;
        mt.Sr   = Sr_t;
        if (shared) mt.map_out = shift(mp.map_out, "tile_out");
        mt.maps = w.tuple(pmaps);
        // A tile at its own origin reads its epilogue inputs at the cells of the whole result it writes.
        if (!shared && prk->nps != 0)
            mt.post_maps = w.tuple(DefVec(prk->nps, [&](size_t k) -> const Def* {
                auto f   = mp.post_maps->proj(prk->nps, k);
                auto pi  = f->type()->as<Pi>();
                auto lam = w.mut_lam(pi->dom(), pi->codom())->set("tile_post_map");
                auto oo  = lam->var();
                lam->set(true, w.app(f, w.tuple(DefVec(prk->ro, [&](size_t a) {
                    return add(oo->proj(prk->ro, a), lit(base->proj(prk->ro, a)));
                }))));
                return lam;
            }));
        mt.maps_all = w.tuple({mt.maps, mt.post_maps});
        if (std::ranges::any_of(clip, [](bool c) { return c; })) {
            auto starts = DefVec(prn), ends = DefVec(prn);
            for (u64 dd = 0; dd != prn; ++dd)
                if (auto a = axis_of[dd]) {
                    starts[dd] = base->proj(prk->ro, *a);
                    if (clip[*a]) ends[dd] = mp.So->proj(prk->ro, *a);
                }
            mt.sched = clip_lim(mp, starts, ends);
        }
        mt.shapes    = w.tuple({mt.So, Sr_t, mt.sched});
        auto shifted = std::vector<bool>(prn), clipped = std::vector<bool>(prn);
        for (u64 dd = 0; dd != prn; ++dd)
            shifted[dd] = axis_of[dd].has_value(), clipped[dd] = shifted[dd] && clip[*axis_of[dd]];
        auto& chain = act ? act->mem : pre;
        auto res = lower_staged({mt, mp, shifted, clipped, old_p->arg()->proj(2, 0), p_post, old_p, fuse, budget, runs},
                                chain, tile_b);
        if (!res) return give_up(j, "the input's inputs are not buffers");
        chain = res->proj(2, 0);

        // Inside the loop of `dim` the box starts at the read of the corner, which only depends on the fixed dims.
        auto read_pi = map->type()->as<Pi>();
        auto read    = w.mut_lam(read_pi->dom(), read_pi->codom())->set("tile_read");
        auto o       = read->var();
        auto at      = w.app(map, o);
        auto corner
            = dim ? w.app(map,
                          w.tuple(DefVec(rn, [&](size_t d) { return fixed[d] ? o->proj(rn, d) : lit(w.lit_nat_0()); })))
                  : nullptr;
        read->set(true, w.tuple(DefVec(*pro, [&](size_t a) {
            return sub(at->proj(*pro, a), dim ? corner->proj(*pro, a) : lit(base->proj(*pro, a)));
        })));

        log().d("compute input {} of {} one tile {} at a time{}", j, s.who, So_t, dim ? " inside its loop" : " ahead");
        if (act) act->used = true;
        ins[j] = tile_b;
        if (shared) {
            write_backs.emplace_back(old_p, tile_b);
            return true;
        }
        Sis[j]  = So_t;
        maps[j] = read;
        return true;
    };

    // A cached input is copied by `btensor.lower_map_reduce` instead.
    auto cache = Lit::isa<u64>(field(F::Cache)).value_or(0);
    for (u64 j = 0; j != rks->nis; ++j) {
        if (j < s.given.size() && s.given[j]) {
            ins[j] = s.given[j];
            continue;
        }
        if (!(j < 64 && (cache >> j) & 1) && (fuse || ahead(j) || (own && j < 64 && (slots >> j) & 1)) && stage(j))
            continue;
        auto old = s.old_is->proj(rks->nis, j);
        ins[j]   = to_buffer(rewrite(old), old);
        if (!Axm::isa<buffer::Buf>(ins[j]->type())) return nullptr;
    }

    // The schedule, its intent now done: the action of each dim runs its tiles inside the loop of that dim.
    auto fields     = DefVec(F::Num_Fields, [&](size_t f) { return field(F::Field(f)); });
    fields[F::At]   = w.lit_ff();
    fields[F::Fuse] = w.lit_ff();
    fields[F::Rev]  = w.lit_nat_0();
    if (std::ranges::any_of(acts, [](const auto& a) { return a && a->used; })) {
        auto mask       = Lit::isa<u64>(field(F::Acts)).value_or(0);
        auto act_pi     = field(F::Act)->type()->as<Pi>();
        auto act_of     = w.mut_lam(act_pi->dom(), act_pi->codom())->set("tile_acts");
        const Def* body = w.app(field(F::Act), act_of->var());
        for (u64 d = 0; d != rn; ++d)
            if (acts[d] && acts[d]->used) {
                acts[d]->run->app(true, acts[d]->run->var(2, 1), acts[d]->mem);
                mask |= u64(1) << d;
                auto here = w.call(core::ncmp::e, Defs{act_of->var(), w.lit_nat(d)});
                body      = w.extract(w.tuple({body, acts[d]->run}), here);
            }
        fields[F::Acts] = w.lit_nat(mask);
        fields[F::Act]  = act_of->set(true, body);
    }
    if (s.post) {
        fields[F::Posts]   = w.lit_tt();
        fields[F::PostDim] = w.lit_nat(s.post_dim);
        fields[F::Post]    = s.post;
    }
    auto sched  = w.app(w.annex<btensor::mk_sched>(), w.tuple(fields));
    auto in_tys = w.tuple({mr.Tis, mr.Ris, w.tuple(Sis), mr.Tps, mr.Rps, mr.Sps});
    auto op     = btensor_op(mr, w.tuple({mr.So, mr.Sr, sched}), in_tys, w.tuple({w.tuple(maps), mr.post_maps}));
    if (!mem) {
        if (!dst) std::tie(alloc_mem_, dst) = buffer::op_alloc(mr.Ro, mr.So, mr.Tp, alloc_mem_)->projs<2>();
        pre = alloc_mem_;
    }
    auto res = w.app(op, w.tuple({pre, dst, w.tuple(ins), s.post_is}));
    // The other readers of a shared producer read its whole buffer once this op has filled it.
    for (auto [old_p, buf] : write_backs) {
        auto after
            = w.app(w.app(w.annex<buffer::after>(), buf->type()->as<App>()->arg()), w.tuple({res->proj(2, 0), buf}));
        auto [m, b]        = after->projs<2>();
        fresh_memo_[old_p] = scalar_result(old_p, m, b);
    }
    return res;
}

const Def* LowerToMem::lower_pad(const App* app) {
    // Thin bufferization: map the SSA `tensor.pad` onto the buffer-world `btensor.pad`.
    // The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w             = new_world();
    auto c              = rewrite(app->callee())->as<App>();
    auto [input, value] = rewrite(app->arg())->projs<2>();

    auto [Tr, s_in, params] = c->uncurry_args<3>();
    auto [T, r]             = Tr->projs<2>();
    auto [mode, lo, hi]     = params->projs<3>();

    input = to_buffer(input, app->arg()->proj(2, 0));
    if (!Axm::isa<buffer::Buf>(input->type()))
        return RWPhase::rewrite_imm_App(app); // not a recorded tensor type: leave it alone

    auto r_l = Lit::isa<u64>(r);
    if (!r_l) return RWPhase::rewrite_imm_App(app);

    // The LOGICAL output shape `s_out#d = lo#d + s_in#d + hi#d` — the loop generation iterates it, so it
    // must keep size-1 axes (the result type's `Buf` folds them away and cannot be used here).
    auto add = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
    DefVec so(*r_l, [&](size_t d) { return add(add(lo->proj(*r_l, d), s_in->proj(*r_l, d)), hi->proj(*r_l, d)); });
    auto s_out = w.tuple(so);

    // `{T, r}` is inferred: `s_in` pins `r`, and `input`'s `buffer.Buf (r, s_in, T)` pins `T`.
    auto [am, dst] = buffer::op_alloc(r, s_out, T, fresh_mem())->projs<2>();
    auto [m, out]  = w.call<btensor::pad>(s_in, Defs{mode, lo, hi}, s_out, Defs{am, dst, input, value})->projs<2>();
    return out;
}

const Def* LowerToMem::lower_concat(const App* app) {
    // Thin bufferization: map the SSA `tensor.concat` onto the buffer-world `btensor.concat`.
    // The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w  = new_world();
    auto c   = rewrite(app->callee())->as<App>();
    auto arg = rewrite(app->arg());

    auto [TnisR, ax, Sis] = c->uncurry_args<3>();
    auto [T, nis, r]      = TnisR->projs<3>();

    auto nis_l = Lit::isa<u64>(nis);
    auto r_l   = Lit::isa<u64>(r);
    auto ax_l  = Lit::isa<u64>(ax);
    if (!nis_l || !r_l || !ax_l) return RWPhase::rewrite_imm_App(app);

    auto inputs = buffer_list(arg, app->arg(), nis);
    if (!inputs) return RWPhase::rewrite_imm_App(app); // not a recorded tensor type: leave it alone

    // The LOGICAL output shape: the summed extent along `ax` (literal, gated in `collect_tensor_types`),
    // the shared extents elsewhere — the loop generation iterates it, so it must keep size-1 axes (the
    // result type's `Buf` folds them away and cannot be used here).
    u64 sum_ax = 0;
    for (u64 i = 0; i < *nis_l; ++i) {
        auto e = Lit::isa<u64>(Sis->proj(*nis_l, i)->proj(*r_l, *ax_l));
        if (!e) return RWPhase::rewrite_imm_App(app);
        sum_ax += *e;
    }
    DefVec so(*r_l, [&](size_t d) { return d == *ax_l ? w.lit_nat(sum_ax) : Sis->proj(*nis_l, 0)->proj(*r_l, d); });
    auto s_out = w.tuple(so);

    // NB: `{T, nis, r}` and `{Sis}` cannot be left to inference (hence no `w.call` here). `Sis` holds the
    // *logical* per-input shapes, but the operands are `buffer.Buf` handles with their literal size-1 axes
    // already folded away — unifying `buffer.Buf (r, Sis#i, T)` against a folded handle fails, because the
    // left-hand side only folds once `Sis#i` is known. Passing the logical shapes makes both sides fold alike.
    auto op        = w.annex<btensor::concat>();
    op             = w.app(op, w.tuple({T, nis, r}));
    op             = w.app(op, ax);
    op             = w.app(op, Sis);
    op             = w.app(op, s_out);
    auto [am, dst] = buffer::op_alloc(r, s_out, T, fresh_mem())->projs<2>();
    auto [m, out]  = w.app(op, w.tuple({am, dst, inputs}))->projs<2>();
    return out;
}

const Def* LowerToMem::make_buffer(const Def* old_arg, const Def* r, const Def* shape) {
    auto value = materialize(old_arg->type(), old_arg);
    if (Axm::isa<buffer::Buf>(value->type())) return value;
    return buffer::op_lit(r, shape, value->type(), bot_mem(), value)->proj(2, 1);
}

const Def* LowerToMem::lower_gather(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim] = c->uncurry_args<3>();
    auto [T, r]            = Tr->projs<2>();
    auto [s_src, s_idx]    = shapes->projs<2>();
    if (!check_gather_shape_constraints(app, r, dim, s_src, s_idx)) return RWPhase::rewrite_imm_App(app);
    auto [old_input, old_index] = app->args<2>();
    auto input                  = make_buffer(old_input, r, s_src);
    auto index                  = make_buffer(old_index, r, s_idx);

    auto op                 = w.app(w.annex<btensor::gather>(), {T, r});
    auto [am, dst]          = buffer::op_alloc(r, s_idx, T, fresh_mem())->projs<2>();
    auto [out_mem, out_buf] = w.call(op, Defs{s_src, s_idx}, dim, Defs{am, dst, input, index})->projs<2>();
    return scalar_result(app, out_mem, out_buf);
}

const Def* LowerToMem::lower_scatter(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]         = c->uncurry_args<3>();
    auto [T, r]                    = Tr->projs<2>();
    auto [s_src, s_idx, s_updates] = shapes->projs<3>();
    if (!check_scatter_shape_constraints(app, r, dim, s_src, s_idx, s_updates)) return RWPhase::rewrite_imm_App(app);
    auto [old_input, old_index, old_updates] = app->args<3>();
    auto input                               = make_buffer(old_input, r, s_src);
    auto index                               = make_buffer(old_index, r, s_idx);
    auto updates                             = make_buffer(old_updates, r, s_updates);

    auto op        = w.app(w.annex<btensor::scatter>(), {T, r});
    auto [am, dst] = buffer::op_alloc(r, s_src, T, fresh_mem())->projs<2>();
    auto [out_mem, out_buf]
        = w.call(op, Defs{s_src, s_idx, s_updates}, dim, Defs{am, dst, input, index, updates})->projs<2>();
    return scalar_result(app, out_mem, out_buf);
}

} // namespace mim::plug::tensor::phase
