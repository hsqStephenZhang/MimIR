#include "mim/plug/tensor/phase/lower_to_mem.h"

#include <fe/worklist.h>

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

const Def* splat_scalar(const Def* d) {
    if (!d->isa<Pack>()) return nullptr;
    while (auto pack = d->isa<Pack>())
        d = pack->body();
    return d->is_closed() && !d->type()->isa<Arr>() ? d : nullptr;
}

bool wants_fresh_mem(const App* app) {
    return Axm::isa<tensor::generate>(app) || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::map_reduce_post>(app)
        || Axm::isa<tensor::pad>(app) || Axm::isa<tensor::concat>(app) || Axm::isa<tensor::gather>(app)
        || Axm::isa<tensor::scatter>(app) || Axm::isa<tensor::set>(app) || Axm::isa<tensor::from_array>(app)
        || Axm::isa<tensor::scalar>(app) || Axm::isa<tensor::splat>(app);
}

bool has_zero_extent(const Def* shape) {
    for (auto extent : shape->projs())
        if (Lit::isa<u64>(extent) == 0) return true;
    return false;
}

bool contains_pi(const Def* type) {
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(type);
    while (!wl.empty()) {
        auto t = wl.pop();
        if (t->isa<Pi>()) return true;
        if (t->isa<Sigma>())
            wl.push(t->ops());
        else if (auto array = t->isa<Arr>())
            wl.push(array->body());
    }
    return false;
}

} // namespace

bool LowerToMem::is_tensor_type(const Def* t) const {
    if (auto nom = t->isa<Nom>()) return nom->key() == tensor_key_;
    return false;
}

bool LowerToMem::mentions_tensor(const Def* type) const {
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(type);
    while (!wl.empty()) {
        auto t = wl.pop();
        if (is_tensor_type(t)) return true;
        if (t->isa<Sigma>() || t->isa<Variant>())
            wl.push(t->ops());
        else if (auto arr = t->isa<Arr>())
            wl.push(arr->body());
        else if (auto pi = t->isa<Pi>())
            wl.push(pi->ops());
        else if (auto nom = t->isa<Nom>())
            wl.push(nom->op());
    }
    return false;
}

bool LowerToMem::is_tensor_fn(Lam* lam) const { return mentions_tensor(lam->type()); }

bool LowerToMem::has_memory_effects(const Def* body, bool allocations_only) const {
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(body);
    while (!wl.empty()) {
        auto d = wl.pop();
        if (d->isa<Var>()) continue;
        if (auto lam = d->isa_mut<Lam>(); lam && (!lam->is_set() || lam->is_external() || !Pi::isa_cn(lam->type())))
            continue;
        if (auto app = d->isa<App>()) {
            auto [axm, curry, trip] = Axm::get(app);
            if (axm && !curry && axm->plugin() == tensor::Plugin_Id
                && (wants_fresh_mem(app)
                    || (!allocations_only && (Axm::isa<tensor::get>(app) || Axm::isa<tensor::item>(app)))))
                return true;
        }
        for (auto op : d->ops())
            if (op) wl.push(op);
    }
    return false;
}

void LowerToMem::validate() {
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    DefVec nodes;
    while (!wl.empty()) {
        auto d = wl.pop();
        nodes.emplace_back(d);
        if (d->isa<Axm>()) continue;
        if (is_tensor_type(d)) {
            tensors_seen_      = true;
            auto [r, shape, T] = TensorType::decode(d->as<Nom>());
            if (!Lit::isa<u64>(r)) d->blame("cannot bufferize: non-literal tensor rank").bail();
            if (T->isa<Arr>() || mentions_tensor(T))
                d->blame("cannot bufferize: tensor with aggregate tensor element type").bail();
        } else if ((d->isa<Nom>() || d->isa<Variant>()) && mentions_tensor(d)) {
            d->blame("cannot bufferize: unsupported tensor aggregate carrier").bail();
        }
        if (auto app = d->isa<App>())
            if (Axm::isa<tensor::generate>(app))
                if (auto body = app->arg()->isa_mut<Lam>(); body && body->filter() == old_world().lit_tt())
                    inline_callbacks_.emplace(body);
        if (auto lam = d->isa_mut<Lam>(); lam && is_tensor_fn(lam)) {
            if (!lam->is_set()) lam->blame("cannot bufferize: tensor external requires an explicit buffer ABI").bail();
            if (Pi::isa_cn(lam->type()))
                tensor_fns_.emplace(lam);
            else if (auto inner = lam->body()->isa_mut<Lam>(); inner && Pi::isa_cn(inner->type()))
                shape_poly_.emplace(lam);
            else
                lam->blame("cannot bufferize: tensor function requires memory-aware CPS conversion").bail();
            for (auto param : lam->vars()) {
                auto type = param->type();
                if (!mentions_tensor(type)) continue;
                if (auto pi = type->isa<Pi>()) {
                    if (!Pi::isa_cn(pi) || contains_pi(pi->dom()))
                        param->blame("cannot bufferize: unsupported higher-order tensor contract").bail();
                } else if (contains_pi(type)) {
                    param->blame("cannot bufferize: unsupported higher-order tensor contract").bail();
                }
            }
        }
        for (auto op : d->ops())
            if (op) wl.push(op);
        if (auto app = d->isa<App>()) {
            auto [axm, curry, trip] = Axm::get(app);
            // Partial axiom applications still carry uninstantiated tensor ranks in their schemas.
            if (axm && curry) continue;
        }
        if (d->type()) wl.push(d->type());
    }
    for (auto d : nodes)
        if (auto lam = d->isa_mut<Lam>(); lam && lam->is_set() && !Pi::isa_cn(lam->type()))
            if (auto inner = lam->body()->isa_mut<Lam>();
                inner && inner->is_set() && Pi::isa_cn(inner->type()) && has_memory_effects(inner->body()))
                shape_poly_.emplace(lam);
    for (auto d : nodes) {
        if (auto lam = d->isa_mut<Lam>();
            lam && inline_callbacks_.contains(lam) && has_memory_effects(lam->body(), true))
            lam->blame("cannot bufferize: allocating generate callback requires CPS conversion").bail();
        if (auto lam = d->isa_mut<Lam>(); lam && lam->is_set() && !Pi::isa_cn(lam->type()) && !shape_poly_.contains(lam)
                                          && !inline_callbacks_.contains(lam) && has_memory_effects(lam->body()))
            lam->blame("cannot bufferize: direct-style tensor memory effects require CPS conversion").bail();
        if (auto app = d->isa<App>())
            if (auto shape = app->callee()->isa<App>())
                if (auto f = shape->callee()->isa_mut<Lam>(); f && shape_poly_.contains(f)) continue;
        for (auto op : d->ops()) {
            if (!op) continue;
            if (auto lam = d->isa_mut<Lam>(); lam && shape_poly_.contains(lam) && lam->body() == op) continue;
            if (auto f = op->isa_mut<Lam>(); f && (shape_poly_.contains(f) || tensor_fns_.contains(f)))
                if ((f->ret_pi() || shape_poly_.contains(f)) && !d->isa<Var>()
                    && !(d->isa<App>() && d->as<App>()->callee() == op))
                    f->blame("cannot bufferize: tensor-processing function used as a value").bail();
            if (auto shape = op->isa<App>())
                if (auto f = shape->callee()->isa_mut<Lam>(); f && shape_poly_.contains(f))
                    if (!(d->isa<App>() && d->as<App>()->callee() == op))
                        op->blame("cannot bufferize: partially applied shape-polymorphic function").bail();
        }
    }
}

void LowerToMem::start() {
    tensor_key_ = tensor_key(old_world());
    validate();
    if (!tensors_seen_) return;
    RWPhase::start();
    assert(pending_.empty());
}

const Def* LowerToMem::buf_of(const Def* type) {
    auto [r, s, T] = TensorType::decode(type->as<Nom>());
    return buffer::type_buf(rewrite(r), rewrite(s), rewrite(T));
}

const Def* LowerToMem::rewrite_imm_Nom(const Nom* nom) {
    return is_tensor_type(nom) ? buf_of(nom) : RWPhase::rewrite_imm_Nom(nom);
}

const Def* LowerToMem::rewrite_imm_Name(const Name* name) {
    if (is_tensor_type(name->type())) name->blame("cannot bufferize: escaped Tensor representation").bail();
    return RWPhase::rewrite_imm_Name(name);
}

const Def* LowerToMem::rewrite_imm_Struc(const Struc* struc) {
    if (is_tensor_type(struc->op()->type())) struc->blame("cannot bufferize: escaped Tensor representation").bail();
    return RWPhase::rewrite_imm_Struc(struc);
}

const Def* LowerToMem::bot_mem() { return new_world().bot(new_world().call<mem::M>(0)); }

const Def* LowerToMem::fresh_mem() {
    auto k = mem::mut_con(new_world())->set("fresh_mem");
    pending_.push_back(k);
    return k->var();
}

const Def* LowerToMem::wrap_fresh_mem(const Def* body) {
    auto& w = new_world();
    for (auto k : pending_ | std::views::reverse) {
        k->set(true, body);
        body = w.app(w.annex<mem::fresh>(), {w.lit_nat_0(), k});
    }
    return body;
}

const Def* LowerToMem::rewrite(const Def* def) {
    if (auto app = def->isa<App>(); app && wants_fresh_mem(app)) {
        if (auto value = fe::lookup(fresh_memo_, def)) return value;
        auto result = rewrite_imm_App(app);
        fresh_memo_.emplace(def, result);
        return result;
    }
    return RWPhase::rewrite(def);
}

void LowerToMem::collect_leaves(const Def* pv, DefVec& out) {
    if (auto sig = pv->type()->isa<Sigma>(); sig && mentions_tensor(sig)) {
        for (auto part : pv->projs(sig->num_ops()))
            collect_leaves(part, out);
        return;
    }
    leaf_.emplace(pv);
    out.emplace_back(pv);
}

const DefVec& LowerToMem::leaves(Lam* lam) {
    auto [it, fresh] = leaves_.try_emplace(lam);
    if (fresh)
        for (auto param : lam->vars())
            collect_leaves(param, it->second);
    return it->second;
}

const Def* LowerToMem::conv_leaf(const Def* pv) { return rewrite(pv->type()); }

const Def* LowerToMem::rebuild_param(const Def* pv) {
    if (leaf_.contains(pv)) return rewrite(pv);
    auto n = pv->type()->as<Sigma>()->num_ops();
    return new_world().tuple(DefVec(n, [&](size_t i) { return rebuild_param(pv->proj(n, i)); }));
}

void LowerToMem::explode_arg(const Def* pv, const Def* a, DefVec& out) {
    if (leaf_.contains(pv)) {
        out.emplace_back(rewrite(a));
        return;
    }
    auto n = pv->type()->as<Sigma>()->num_ops();
    for (size_t i = 0; i != n; ++i)
        explode_arg(pv->proj(n, i), a->proj(n, i), out);
}

const Def* LowerToMem::rewrite_mut_Lam(Lam* lam) {
    auto pending = fe::Restore(pending_, {});
    auto memo    = fe::Restore(fresh_memo_, {});
    if (shape_poly_.contains(lam) || (tensor_fns_.contains(lam) && lam->type()->dom()->isa_binder<Sigma>().second))
        return uncurry(lam);
    auto result = new_world().mut_lam(rewrite(lam->type())->as<Pi>());
    map(lam, result);
    if (!lam->is_set()) return result;
    auto scope  = enter(lam);
    auto filter = rewrite(lam->filter());
    result->set(filter, wrap_fresh_mem(rewrite(lam->body())));
    return seal(lam, result);
}

const Def* LowerToMem::uncurry(Lam* lam) {
    auto& w     = new_world();
    auto poly   = shape_poly_.contains(lam);
    auto pi     = lam->type()->as<Pi>();
    auto pi_mut = poly ? lam->type()->isa_mut<Pi>() : nullptr; // dependent iff some boundary names a shape parameter
    auto inner  = poly ? lam->body()->as_mut<Lam>() : lam;
    auto k      = poly ? pi->dom()->num_projs() : 0;
    auto m      = inner->num_vars();
    auto& ls    = leaves(inner);
    auto n      = k + ls.size();

    // Dependent boundaries must refer to the new domain's binders.
    auto sigma = w.mut_sigma(n);
    push();
    if (pi_mut) bind_old(pi_mut->has_var());
    bind_old(lam->has_var());
    bind_old(inner->has_var());
    for (size_t i = 0; i != k; ++i) {
        sigma->set(i, rewrite(pi->dom()->proj(k, i)));
        if (pi_mut) map(pi_mut->var(k, i), sigma->var(n, i));
        map(lam->var(k, i), sigma->var(n, i));
    }
    if (k > 1) {
        auto vars = w.tuple(DefVec(k, [&](size_t i) { return sigma->var(n, i); }));
        if (pi_mut) map(pi_mut->var(), vars);
        map(lam->var(), vars);
    }
    for (size_t l = 0; l != ls.size(); ++l) {
        sigma->set(k + l, conv_leaf(ls[l]));
        map(ls[l], sigma->var(n, k + l));
    }
    pop();

    auto new_lam = w.mut_con(sigma)->set(lam->dbg());
    map(lam, new_lam);
    if (poly) map(inner, new_lam);
    for (size_t i = 0; i != k; ++i)
        map(lam->var(k, i), new_lam->var(n, i));
    if (k > 1) map(lam->var(), w.tuple(DefVec(k, [&](size_t i) { return new_lam->var(n, i); })));
    for (size_t l = 0; l != ls.size(); ++l)
        map(ls[l], new_lam->var(n, k + l)->set(ls[l]->dbg()));
    for (size_t j = 0; j != m; ++j)
        map(inner->var(m, j), rebuild_param(inner->var(m, j)));
    if (m > 1) map(inner->var(), w.tuple(DefVec(m, [&](size_t j) { return rewrite(inner->var(m, j)); })));

    auto ___ = enter(new_lam);
    new_lam->set(rewrite(inner->filter()), wrap_fresh_mem(rewrite(inner->body())));
    return seal(lam, new_lam);
}

const Def* LowerToMem::rewrite_imm_App(const App* app) {
    auto& w = new_world();
    if (Axm::isa<tensor::if_static>(app)) return rewrite(app->arg(3, 2));
    if (Axm::isa<tensor::from_array>(app)) return lower_from_array(app);
    if (Axm::isa<tensor::get>(app)) return lower_get(app);
    if (Axm::isa<tensor::set>(app)) return lower_set(app);
    if (Axm::isa<tensor::scalar>(app)) {
        auto T = rewrite(app->arg()->type());
        return buffer::op_init(w.lit_nat_0(), w.tuple(), T, fresh_mem(), rewrite(app->arg()))->proj(2, 1);
    }
    if (Axm::isa<tensor::item>(app)) return buffer::op_read(bot_mem(), rewrite(app->arg()), w.tuple())->proj(2, 1);
    if (Axm::isa<tensor::splat>(app)) return lower_splat(app);
    if (Axm::isa<tensor::generate>(app)) return lower_generate(app);
    if (Axm::isa<tensor::broadcast>(app)) return lower_broadcast(app);
    if (Axm::isa<tensor::map_reduce_post>(app)) return lower_map_reduce(app);
    if (Axm::isa<tensor::pad>(app)) return lower_pad(app);
    if (Axm::isa<tensor::concat>(app)) return lower_concat(app);
    if (Axm::isa<tensor::gather>(app)) return lower_gather(app);
    if (Axm::isa<tensor::scatter>(app)) return lower_scatter(app);
    if (auto sh = app->callee()->isa<App>())
        if (auto f = sh->callee()->isa_mut<Lam>(); f && shape_poly_.contains(f)) {
            auto callee = rewrite(f);
            auto inner  = f->body()->as_mut<Lam>();
            DefVec args;
            for (auto part : sh->arg()->projs())
                args.emplace_back(rewrite(part));
            auto m = inner->num_vars();
            for (size_t j = 0; j != m; ++j)
                explode_arg(inner->var(m, j), app->arg()->proj(m, j), args);
            return w.app(callee, w.tuple(args));
        }
    if (auto f = app->callee()->isa_mut<Lam>(); f && tensor_fns_.contains(f)) return lower_call(app, f);
    if (auto cast = Axm::isa<core::bitcast>(app); cast && is_tensor_type(app->type()))
        return w.call<core::bitcast>(buf_of(app->type()), rewrite(cast->arg()));
    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerToMem::lower_call(const App* app, Lam* f) {
    if (f->type()->dom()->isa_binder<Sigma>().second) {
        auto callee = rewrite(f);
        DefVec args;
        auto n = f->num_vars();
        for (size_t i = 0; i != n; ++i)
            explode_arg(f->var(n, i), app->arg()->proj(n, i), args);
        return new_world().app(callee, new_world().tuple(args));
    }
    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerToMem::buffer_list(const Def* list, const Def* n) {
    auto count = Lit::isa<u64>(n);
    if (!count) list->blame("cannot bufferize: non-literal tensor operand count").bail();
    return new_world().tuple(list->projs(*count));
}

const Def* LowerToMem::splat_buffer(const Def* type, const Def* value) {
    auto [r, s, T] = Axm::isa<buffer::Buf>(buf_of(type))->args<3>();
    return buffer::op_lit(r, s, T, fresh_mem(), value)->proj(2, 1);
}

const Def* LowerToMem::lower_from_array(const App* app) {
    auto [r, s, T] = TensorType::decode(app->type()->as<Nom>());
    if (auto value = splat_scalar(app->arg()); value && value->type() == T)
        return splat_buffer(app->type(), rewrite(value));
    auto [br, bs, bT] = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();
    return buffer::op_init(br, bs, bT, fresh_mem(), rewrite(app->arg()))->proj(2, 1);
}

const Def* LowerToMem::lower_get(const App* app) {
    auto [old, index] = app->args<2>();
    auto [r, s, T]    = TensorType::decode(old->type()->as<Nom>());
    auto shape        = rewrite(s);
    auto idx          = Shape(rewrite(index)).fold(shape);
    return buffer::op_read(bot_mem(), rewrite(old), *idx)->proj(2, 1);
}

const Def* LowerToMem::lower_set(const App* app) {
    auto& w                  = new_world();
    auto [old, index, value] = app->args<3>();
    auto [r, s, T]           = TensorType::decode(app->type()->as<Nom>());
    auto shape               = rewrite(s);
    auto source              = rewrite(old);
    auto [br, bs, bT]        = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();
    auto [m, dst]            = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    if (has_zero_extent(shape)) return dst;
    auto result = w.sigma({m->type(), dst->type()});
    auto copy
        = btensor::build_pointwise(result, m, dst, source, shape, *Lit::isa<u64>(r), "tensor_copy",
                                   [shape](Defs iters, const Def* src, const Def* mem) {
                                       auto [m, elem]
                                           = buffer::op_read(mem, src, btensor::folded_idx(shape, iters))->projs<2>();
                                       return std::pair(m, elem);
                                   });
    auto [after, out] = copy->projs<2>();
    auto idx          = *Shape(rewrite(index)).fold(shape);
    return buffer::op_write(after, out, idx, rewrite(value))->proj(2, 1);
}

const Def* LowerToMem::lower_splat(const App* app) {
    return splat_buffer(app->type(), rewrite(app->arg()->proj(2, 1)));
}

const Def* LowerToMem::lower_generate(const App* app) {
    auto& w           = new_world();
    auto [r, s, T]    = TensorType::decode(app->type()->as<Nom>());
    auto shape        = rewrite(s);
    auto body         = rewrite(app->arg());
    auto [br, bs, bT] = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();
    auto [m, dst]     = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    if (has_zero_extent(shape)) return dst;
    auto result = w.sigma({m->type(), dst->type()});
    return btensor::build_pointwise(result, m, dst, w.tuple(), shape, *Lit::isa<u64>(r), "generate",
                                    [body](Defs iters, const Def*, const Def* mem) {
                                        return std::pair(mem, mem->world().call(body, iters));
                                    })
        ->proj(2, 1);
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

    auto in_buf = Axm::isa<buffer::Buf>(input->type());

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

const Def* LowerToMem::lower_map_reduce(const App* app) {
    // Thin bufferization: map the SSA `tensor.map_reduce_post` onto the buffer-world `btensor.map_reduce_post`,
    // reusing the (rewritten) meta. The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w            = new_world();
    auto c             = rewrite(app->callee())->as<App>();
    auto args          = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
    auto [is, post_is] = args->projs<2>();

    auto mr = btensor::MapRedPost::of(c);

    is      = buffer_list(is, mr.nis);
    post_is = buffer_list(post_is, mr.nps);

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
    auto op        = w.annex<btensor::map_reduce_post>();
    op             = w.app(op, mr.nis_nps);
    op             = w.app(op, mr.meta);
    op             = w.app(op, mr.shapes);
    op             = w.app(op, mr.in_tys);
    op             = w.app(op, w.tuple({memcomb, mr.init, mempost}));
    op             = w.app(op, mr.map_out);
    op             = w.app(op, mr.maps_all);
    auto [am, dst] = buffer::op_alloc(mr.Ro, mr.So, mr.Tp, fresh_mem())->projs<2>();
    auto [m, out]  = w.app(op, w.tuple({am, dst, is, post_is}))->projs<2>();
    return out;
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

    auto r_l = Lit::isa<u64>(r);
    if (!r_l) app->blame("cannot bufferize: pad requires a literal rank").bail();

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
    if (!nis_l || !r_l || !ax_l)
        app->blame("cannot bufferize: concat requires literal operand count, rank and axis").bail();

    auto inputs = buffer_list(arg, nis);

    // The logical output shape retains size-1 axes for loop generation.
    u64 sum_ax = 0;
    for (u64 i = 0; i < *nis_l; ++i) {
        auto e = Lit::isa<u64>(Sis->proj(*nis_l, i)->proj(*r_l, *ax_l));
        if (!e) app->blame("cannot bufferize: concat requires literal extents along its axis").bail();
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

const Def* LowerToMem::lower_gather(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim] = c->uncurry_args<3>();
    auto [T, r]            = Tr->projs<2>();
    auto [s_src, s_idx]    = shapes->projs<2>();
    if (!check_gather_shape_constraints(app, r, dim, s_src, s_idx))
        app->blame("cannot bufferize: gather requires a literal axis and proven non-axis shape bounds").bail();
    auto [old_input, old_index] = app->args<2>();
    auto input                  = rewrite(old_input);
    auto index                  = rewrite(old_index);

    auto op                 = w.app(w.annex<btensor::gather>(), {T, r});
    auto [am, dst]          = buffer::op_alloc(r, s_idx, T, fresh_mem())->projs<2>();
    auto [out_mem, out_buf] = w.call(op, Defs{s_src, s_idx}, dim, Defs{am, dst, input, index})->projs<2>();
    return out_buf;
}

const Def* LowerToMem::lower_scatter(const App* app) {
    auto& w = new_world();
    auto c  = rewrite(app->callee())->as<App>();

    auto [Tr, shapes, dim]         = c->uncurry_args<3>();
    auto [T, r]                    = Tr->projs<2>();
    auto [s_src, s_idx, s_updates] = shapes->projs<3>();
    if (!check_scatter_shape_constraints(app, r, dim, s_src, s_idx, s_updates))
        app->blame("cannot bufferize: scatter requires a literal axis and proven index shape bounds").bail();
    auto [old_input, old_index, old_updates] = app->args<3>();
    auto input                               = rewrite(old_input);
    auto index                               = rewrite(old_index);
    auto updates                             = rewrite(old_updates);

    auto op        = w.app(w.annex<btensor::scatter>(), {T, r});
    auto [am, dst] = buffer::op_alloc(r, s_src, T, fresh_mem())->projs<2>();
    auto [out_mem, out_buf]
        = w.call(op, Defs{s_src, s_idx, s_updates}, dim, Defs{am, dst, input, index, updates})->projs<2>();
    return out_buf;
}

} // namespace mim::plug::tensor::phase
