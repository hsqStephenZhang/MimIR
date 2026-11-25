#include "mim/plug/tensor/phase/lower_to_mem.h"

#include <charconv>

#include <format>
#include <ranges>

#include <fe/worklist.h>

#include <mim/axm.h>
#include <mim/check.h>
#include <mim/def.h>
#include <mim/driver.h>
#include <mim/lam.h>
#include <mim/nom.h>
#include <mim/plugin.h>
#include <mim/variant.h>

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
        || Axm::isa<tensor::scatter>(app) || Axm::isa<tensor::set>(app) || Axm::isa<tensor::from_array>(app)
        || Axm::isa<tensor::scalar>(app) || Axm::isa<tensor::splat>(app);
}

void validate_array_literal(const App* app) {
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

void LowerToMem::validate() {
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    while (!wl.empty()) {
        auto d = wl.pop();
        if (d->isa<Axm>()) continue;
        if (is_tensor_type(d)) {
            tensors_seen_      = true;
            auto [r, shape, T] = TensorType::decode(d->as<Nom>());
            if (!Lit::isa<u64>(r)) d->blame("cannot bufferize: non-literal tensor rank").bail();
            auto rank = *Lit::isa<u64>(r);
            for (u64 i = 0; i != rank; ++i) {
                auto extent = shape->proj(rank, i);
                if (Lit::isa<u64>(extent) == 0)
                    d->blame("cannot bufferize: zero-extent tensors are not supported").bail();
            }
            if (T->isa<Arr>() || mentions_tensor(T))
                d->blame("cannot bufferize: tensor with aggregate tensor element type").bail();
        } else if ((d->isa<Nom>() || d->isa<Variant>()) && mentions_tensor(d)) {
            d->blame("cannot bufferize: unsupported tensor aggregate carrier").bail();
        }
        if (auto app = Axm::isa<tensor::from_array>(d)) validate_array_literal(app);
        if (auto lam = d->isa_mut<Lam>()) {
            if (is_tensor_fn(lam)) {
                if (!lam->is_set())
                    lam->blame("cannot bufferize: tensor external requires an explicit buffer ABI").bail();
                if (!Pi::isa_cn(lam->type())) {
                    if (auto inner = lam->body()->isa_mut<Lam>(); inner && Pi::isa_cn(inner->type()))
                        shape_poly_.emplace(lam);
                    else
                        lam->blame("cannot bufferize: tensor function requires memory-aware CPS conversion").bail();
                }
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
        }
        for (auto op : d->ops()) {
            if (!op) continue;
            if (auto lam = d->isa_mut<Lam>(); lam && shape_poly_.contains(lam) && lam->body() == op) {
                wl.push(op);
                continue;
            }
            if (auto sh = op->isa<App>())
                if (auto f = sh->callee()->isa_mut<Lam>(); f && shape_poly_.contains(f))
                    if (!(d->isa<App>() && d->as<App>()->callee() == op))
                        sh->blame("cannot bufferize: partially applied shape-polymorphic function").bail();
            if (auto f = op->isa_mut<Lam>(); f && is_tensor_fn(f) && f->ret_pi() && !d->isa<Var>()
                                             && !(d->isa<App>() && d->as<App>()->callee() == op)
                                             && !Axm::isa<cps::cps2ds_dep>(d))
                f->blame("cannot bufferize: tensor-processing function used as a value").bail();
            wl.push(op);
        }
        if (auto app = d->isa<App>()) {
            auto [axm, curry, trip] = Axm::get(app);
            // Partial axiom applications still carry uninstantiated tensor ranks in their schemas.
            if (axm && curry) continue;
        }
        if (d->type()) wl.push(d->type());
    }
}

void LowerToMem::start() {
    tensor_key_ = tensor_key(old_world());
    validate();
    if (!tensors_seen_) return;
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    while (!wl.empty()) {
        auto d = wl.pop();
        for (auto op : d->ops())
            if (op) ++uses_[op], wl.push(op);
    }
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
    if (app && wants_fresh_mem(app)) {
        if (auto new_def = fe::lookup(fresh_memo_, old_def)) return new_def;
        auto new_def = rewrite_imm_App(app);
        fresh_memo_.emplace(old_def, new_def);
        return new_def;
    }
    return RWPhase::rewrite(old_def);
}

const Def* LowerToMem::rewrite_mut_Lam(Lam* lam) {
    // Scope the fresh-memory bookkeeping: ops lowered while this body is rewritten mint their receiving
    // continuations into pending_, which are chained in front of the finished body. Nested lams anchor
    // their own requests (and their own per-lam op memo).
    auto _  = fe::Restore(pending_, {});
    auto __ = fe::Restore(fresh_memo_, {});
    if (shape_poly_.contains(lam)) return uncurry(lam);
    auto new_lam = new_world().mut_lam(rewrite(lam->type())->as<Pi>());
    map(lam, new_lam);
    if (!lam->is_set()) return new_lam;

    auto ___    = enter(lam);
    auto filter = rewrite(lam->filter());
    auto body   = wrap_fresh_mem(rewrite(lam->body()));
    new_lam->set(filter, body);
    return seal(lam, new_lam);
}

const Def* LowerToMem::uncurry(Lam* lam) {
    auto& w     = new_world();
    auto pi     = lam->type()->as<Pi>();
    auto pi_mut = pi->isa_mut<Pi>();
    auto inner  = lam->body()->as_mut<Lam>();
    auto k = pi->dom()->num_projs(), m = inner->num_vars(), n = k + m;
    auto sigma = w.mut_sigma(n);
    push();
    if (pi_mut) bind_old(pi_mut->has_var());
    bind_old(lam->has_var());
    bind_old(inner->has_var());
    for (size_t i = 0; i != k; ++i) {
        sigma->set(i, rewrite(lam->var(k, i)->type()));
        if (pi_mut) map(pi_mut->var(k, i), sigma->var(n, i));
        map(lam->var(k, i), sigma->var(n, i));
    }
    if (k > 1) {
        auto vars = w.tuple(DefVec(k, [&](size_t i) { return sigma->var(n, i); }));
        if (pi_mut) map(pi_mut->var(), vars);
        map(lam->var(), vars);
    }
    for (size_t i = 0; i != m; ++i) {
        sigma->set(k + i, rewrite(inner->var(m, i)->type()));
        map(inner->var(m, i), sigma->var(n, k + i));
    }
    pop();

    auto new_lam = w.mut_con(sigma)->set(lam->dbg());
    map(lam, new_lam);
    for (size_t i = 0; i != k; ++i)
        map(lam->var(k, i), new_lam->var(n, i));
    if (k > 1) map(lam->var(), w.tuple(DefVec(k, [&](size_t i) { return new_lam->var(n, i); })));
    for (size_t i = 0; i != m; ++i)
        map(inner->var(m, i), new_lam->var(n, k + i));
    if (m > 1) map(inner->var(), w.tuple(DefVec(m, [&](size_t i) { return new_lam->var(n, k + i); })));
    auto adapter = w.mut_con(rewrite(inner->type())->as<Pi>()->dom())->set(inner->dbg());
    map(inner, adapter);
    DefVec args;
    for (size_t i = 0; i != k; ++i)
        args.emplace_back(new_lam->var(n, i));
    for (auto arg : adapter->vars())
        args.emplace_back(arg);
    adapter->app(true, new_lam, w.tuple(args));
    auto ___ = enter(lam);
    new_lam->set(rewrite(inner->filter()), wrap_fresh_mem(rewrite(inner->body())));
    return seal(lam, new_lam);
}

const Def* LowerToMem::rewrite_imm_Extract(const Extract* extract) {
    // A residualized choice among tensors lowers only the chosen one.
    if (auto tuple = extract->tuple()->isa<Tuple>(); tuple && mentions_tensor(tuple->type()))
        if (auto i = Lit::isa(rewrite(extract->index()))) return rewrite(tuple->op(*i));
    return RWPhase::rewrite_imm_Extract(extract);
}

const Def* LowerToMem::rewrite_imm_App(const App* app) {
    auto& w = new_world();
    if (Axm::isa<tensor::if_static>(app)) return rewrite(app->arg(3, 2));
    if (Axm::isa<tensor::pe_cpu_candidate>(app)) return w.lit_ff();
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
            DefVec args;
            for (auto shape : sh->arg()->projs())
                args.emplace_back(rewrite(shape));
            for (auto arg : app->arg()->projs(f->body()->as_mut<Lam>()->num_vars()))
                args.emplace_back(rewrite(arg));
            return w.app(callee, w.tuple(args));
        }
    return RWPhase::rewrite_imm_App(app);
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
    auto [old, index, value] = app->args<3>();
    auto [r, s, T]           = TensorType::decode(app->type()->as<Nom>());
    auto shape               = rewrite(s);
    auto source              = rewrite(old);
    auto [br, bs, bT]        = Axm::isa<buffer::Buf>(source->type())->args<3>();
    auto [m, dst]            = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    auto copied              = btensor::copy_buffer(m, dst, source, shape, *Lit::isa<u64>(r));
    auto idx                 = *Shape(rewrite(index)).fold(shape);
    return buffer::op_write(copied, dst, idx, rewrite(value))->proj(2, 1);
}

const Def* LowerToMem::lower_splat(const App* app) {
    return splat_buffer(app->type(), rewrite(app->arg()->proj(2, 1)));
}

const Def* LowerToMem::lower_generate(const App* app) {
    auto& w            = new_world();
    auto [meta, s_out] = rewrite(app->callee())->as<App>()->uncurry_args<2>();
    auto [T, r]        = meta->projs<2>();
    auto r_l           = Lit::isa<u64>(r);
    auto rn            = *r_l;
    auto body          = rewrite(app->arg());

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

namespace {

/// A `tensor.SchedTree` as one indented node per line, e.g. `Par (8, 0, 7)` above `Stage 0` above `Leaf 1`.
void show_tree(std::string& out, const Def* tree, size_t depth) {
    static constexpr std::array names = {"Leaf", "Par", "Pack", "Seq", "Peel", "Stage", "Switch"};
    auto indent                       = std::string(2 * depth, ' ');
    auto name                         = tree->isa<Name>();
    auto inj                          = name ? name->op()->isa<Inj>() : nullptr;
    if (!inj || inj->index() >= names.size()) {
        out += std::format("{}{}\n", indent, tree);
    } else if (inj->index() == 0) {
        out += std::format("{}Leaf {}\n", indent, inj->value());
    } else {
        auto arg  = inj->value();
        auto n    = arg->num_projs();
        auto kids = inj->index() == names.size() - 1 ? 2_u64 : 1_u64;
        auto args = DefVec(n - kids, [&](size_t i) { return arg->proj(n, i); });
        out += std::format("{}{} {}\n", indent, names[inj->index()], fe::Join(args, ", "));
        for (auto i = n - kids; i != n; ++i)
            show_tree(out, arg->proj(n, i), depth + 1);
    }
}

} // namespace

std::pair<const Def*, u64> LowerToMem::cpu_levels(u64 rn) {
    auto& w   = new_world();
    auto text = arg_value(w.driver().args("tensor"), "cpu-levels");
    if (!text) return {};
    auto levels = DefVec();
    for (auto level : std::views::split(*text, '/')) {
        auto sizes = DefVec();
        for (auto size : std::views::split(level, ',')) {
            auto n = u64(0);
            auto v = std::string_view(size.begin(), size.end());
            if (auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), n);
                ec != std::errc() || ptr != v.data() + v.size()) {
                log().w("ignoring `-X tensor:cpu-levels={}`: `{}` is not a number", *text, v);
                return {};
            }
            sizes.emplace_back(w.lit_nat(n));
        }
        if (sizes.size() != rn) {
            log().w("ignoring `-X tensor:cpu-levels={}` for a contraction over {} loops", *text, rn);
            return {};
        }
        levels.emplace_back(w.tuple(sizes));
    }
    return {w.tuple(levels), levels.size()};
}

std::pair<const Def*, bool> LowerToMem::fuse_producer(const Def* input, const Def* map, const Def* Rn, const Def* r) {
    auto& w   = new_world();
    auto papp = Axm::isa<tensor::map_reduce_post>(input);
    // A map-reduce read twice is computed in both consumers, unless it is a contraction.
    if (!papp || uses_[input] > 2 || !w.driver().arg_nat("tensor", "pe-fuse").value_or(1)) return {};
    if (w.app(w.app(w.annex<tensor::aff_boxable>(), {Rn, r}), map) != w.lit_tt()) return {};
    auto old_mr = btensor::MapRedPost::of(papp->callee()->as<App>());
    auto ranks  = old_mr.ranks();
    if (!ranks || ranks->nps != 0) return {};
    if (!Check::alpha(old_mr.post, old_world().app(old_world().annex<tensor::id>(), old_mr.To))) return {};
    auto c  = rewrite(papp->callee())->as<App>();
    auto mr = btensor::MapRedPost::of(c);
    u64 ro = ranks->ro, rn = ranks->nloops(), nis = ranks->nis;
    // The consumer bounds the box of the output it asks for, but not the reduction.
    for (u64 d = ro; d != rn; ++d)
        if (!Lit::isa(mr.Sr->proj(rn, d))) return {};
    // `fill_of` computes a box of the output from loops that start at its corner, so the output loops come first.
    auto domain = w.arr(mr.Rn, w.annex<affine::Index>());
    auto probe  = w.mut_lam(domain, domain);
    auto vars   = probe->var()->projs(rn);
    if (w.app(mr.map_out, probe->var()) != w.tuple(Defs(vars.data(), ro))) return {};

    // A contraction is tiled by `cpu_tree` and may fuse its own producer; any other map-reduce is one leaf over plain
    // inputs.
    const Def *maps, *xs, *sis, *prods, *tree_of;
    auto plan        = PeCpu{};
    auto gen         = PeGeneric{};
    auto contraction = pe_cpu_plan(c, &plan) == w.lit_tt();
    if (contraction && uses_[input] != 1) return {};
    if (contraction) {
        const Def* fuse;
        std::tie(xs, sis, prods, fuse) = pe_slots(papp, mr.Rn, mr.Ris, mr.Sis, mr.Tis, plan.maps, plan.a, true);
        maps                           = plan.maps;
        tree_of                        = w.app(w.app(w.annex<tensor::cpu_tree_at>(), {mr.Rn, mr.Ro}),
                                               {w.lit_nat(plan.bytes), plan.a, plan.dep, plan.unit_b, fuse});
    } else if (pe_generic_plan(c, &gen)) {
        const Def* fuse;
        std::tie(xs, sis, prods, fuse) = generic_slots(papp, mr.Rn, mr.Ris, mr.Sis, mr.Tis, gen.maps);
        maps                           = gen.maps;
        tree_of = w.app(w.app(w.annex<tensor::generic_tree>(), {mr.nis, mr.Rn, mr.Ro}),
                        {gen.red_unit, gen.out_unit, gen.wdim, gen.rdim, w.lit_nat_0(), gen.bytes, fuse});
    } else {
        return {};
    }
    auto [cTo, ins_ty] = mr.comb->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto memcomb       = w.app(w.app(w.annex<btensor::lift_mem>(), {cTo, ins_ty, cTo}), mr.comb);
    const Def* fill    = w.annex<tensor::fill_of>();
    fill               = w.app(fill, {mr.nis, mr.To, mr.Ro, mr.Rn, mr.Tis, mr.Ris, sis});
    fill               = w.app(fill, mr.Sr);
    fill               = w.app(fill, {memcomb, mr.init});
    fill               = w.app(fill, {maps, prods});
    fill               = w.app(fill, tree_of);
    log().i("fuse the map-reduce over {} into its consumer", mr.Sr);
    return {w.app(fill, xs), contraction};
}

std::array<const Def*, 4> LowerToMem::pe_slots(const App* app,
                                               const Def* Rn,
                                               const Def* Ris,
                                               const Def* Sis,
                                               const Def* Tis,
                                               const Def* maps,
                                               const Def* a,
                                               bool may_fuse) {
    auto& w     = new_world();
    auto old_is = app->arg()->proj(2, 0);
    auto ai     = Lit::as(a);
    auto fills  = std::array<const Def*, 2>{};
    auto kinds  = std::array<bool, 2>{};
    // One staged slot per contraction: the policy bands the loop it reads, which the other slot's producer would not
    // read.
    for (auto s : {ai, 1 - ai})
        if (may_fuse) {
            std::tie(fills[s], kinds[s]) = fuse_producer(old_is->proj(2, s), maps->proj(2, s), Rn, Ris->proj(2, s));
            if (fills[s]) break;
        }
    auto xs = DefVec(2), sis = DefVec(2), prods = DefVec(2), fuse = DefVec(2);
    for (size_t i = 0; i != 2; ++i) {
        auto r = Ris->proj(2, i), T = Tis->proj(2, i);
        fuse[i] = w.lit_nat(fills[i] ? 1 + kinds[i] : 0);
        if (fills[i]) {
            // The slot's buffer is never read; it only fills the call's signature.
            auto shape = w.tuple(DefVec(*Lit::isa(r), [&](size_t) { return w.lit_nat(2); }));
            xs[i]      = buffer::op_alloc(r, shape, T, fresh_mem())->proj(2, 1);
            sis[i]     = shape;
            prods[i]   = fills[i];
        } else {
            xs[i]    = rewrite(old_is->proj(2, i));
            sis[i]   = Sis->proj(2, i);
            prods[i] = w.app(w.annex<tensor::prod_plain>(), {r, T});
        }
    }
    return {w.tuple(xs), w.tuple(sis), w.tuple(prods), w.tuple(fuse)};
}

std::array<const Def*, 4> LowerToMem::generic_slots(const App* app,
                                                    const Def* Rn,
                                                    const Def* Ris,
                                                    const Def* Sis,
                                                    const Def* Tis,
                                                    const Def* maps) {
    auto& w     = new_world();
    auto old_is = app->arg()->proj(2, 0);
    auto nis    = btensor::MapRedPost::of(app->callee()->as<App>()).ranks()->nis;
    auto xs = DefVec(nis), sis = DefVec(nis), prods = DefVec(nis), fuse = DefVec(nis);
    for (size_t i = 0; i != nis; ++i) {
        auto r = Ris->proj(nis, i), T = Tis->proj(nis, i);
        auto fill = fuse_producer(old_is->proj(nis, i), maps->proj(nis, i), Rn, r).first;
        fuse[i]   = w.lit_nat(fill ? 1 : 0);
        if (fill) {
            // The slot's buffer is never read; it only fills the call's signature.
            auto shape = w.tuple(DefVec(*Lit::isa(r), [&](size_t) { return w.lit_nat(2); }));
            xs[i]      = buffer::op_alloc(r, shape, T, fresh_mem())->proj(2, 1);
            sis[i]     = shape;
            prods[i]   = fill;
        } else {
            xs[i]    = rewrite(old_is->proj(nis, i));
            sis[i]   = Sis->proj(nis, i);
            prods[i] = w.app(w.annex<tensor::prod_plain>(), {r, T});
        }
    }
    return {w.tuple(xs), w.tuple(sis), w.tuple(prods), w.tuple(fuse)};
}

const Def* LowerToMem::lower_pe_cpu(const App* app) {
    auto& w     = new_world();
    auto old_mr = btensor::MapRedPost::of(app->callee()->as<App>());
    auto ranks  = old_mr.ranks();
    if (!ranks) return nullptr;
    auto c    = rewrite(app->callee())->as<App>();
    auto mr   = btensor::MapRedPost::of(c);
    auto plan = PeCpu{};
    if (pe_cpu_plan(c, &plan) != w.lit_tt()) return nullptr;
    log().i("tile the contraction over {} with tensor.run_levels", mr.Sr);
    // A run-time extent has no static bound until a `Seq` gives it one; the sentinel only loses every `min`.
    u64 rn                 = ranks->nloops();
    auto extent            = [&](size_t d) { return Lit::isa(mr.Sr->proj(rn, d)); };
    auto sizes             = w.tuple(DefVec(rn, [&](size_t d) { return w.lit_nat(extent(d).value_or(0)); }));
    auto caps              = w.tuple(DefVec(rn, [&](size_t d) { return w.lit_nat(extent(d).value_or(u64(1) << 40)); }));
    auto dyn               = w.tuple(DefVec(rn, [&](size_t d) { return w.lit_bool(!extent(d)); }));
    auto [levels, nlevels] = cpu_levels(rn);
    auto [xs, sis, prods, fuse] = pe_slots(app, mr.Rn, mr.Ris, mr.Sis, mr.Tis, plan.maps, plan.a, !levels);
    auto tree                   = w.app(w.app(w.annex<tensor::cpu_tree>(), {mr.Rn, mr.Ro}),
                                        {w.lit_nat(plan.bytes), plan.a, plan.dep, plan.unit_b, sizes, dyn, fuse});
    if (levels) {
        auto n = w.lit_nat(nlevels);
        tree   = w.app(w.app(w.annex<tensor::sched_levels>(), {w.lit_nat(2), mr.Rn, n}),
                       {levels, w.lit_nat(ranks->ro - 2)});
        log().i("tile the contraction over {} at the levels {}", mr.Sr, levels);
    }

    if (log().level() >= fe::Log::Level::Verbose) {
        auto text = std::string();
        show_tree(text, tree, 0);
        log().v("schedule tree:\n{}", text);
    }
    auto [cTo, ins_ty] = mr.comb->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto memcomb       = w.app(w.app(w.annex<btensor::lift_mem>(), {cTo, ins_ty, cTo}), mr.comb);

    const Def* run = w.annex<tensor::run_levels_with>();
    run            = w.app(run, {mr.nis, mr.To, mr.Ro, mr.Rn, mr.Tis, mr.Ris, sis});
    run            = w.app(run, {mr.So, mr.Sr});
    run            = w.app(run, {memcomb, mr.init});
    run            = w.app(run, mr.map_out);
    run            = w.app(run, {plan.maps, prods});
    run            = w.app(run, {tree, caps, dyn});
    auto [m, out]  = w.app(run, {fresh_mem(), xs})->projs<2>();

    return apply_post(app, m, out);
}

const Def* LowerToMem::apply_post(const App* app, const Def* m, const Def* out) {
    auto& w     = new_world();
    auto old_mr = btensor::MapRedPost::of(app->callee()->as<App>());
    if (old_mr.ranks()->nps == 0
        && Check::alpha(old_mr.post, old_world().app(old_world().annex<tensor::id>(), old_mr.To)))
        return out;
    auto mr      = btensor::MapRedPost::of(rewrite(app->callee())->as<App>());
    auto post_is = rewrite(app->arg()->proj(2, 1));

    // Apply fused epilogue as a pointwise pass (in-place when To == Tp).
    auto [pTo, pexts_ty] = mr.post->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto mempost         = w.app(w.app(w.annex<btensor::lift_mem>(), {pTo, pexts_ty, mr.Tp}), mr.post);
    auto take_fn         = w.app(w.annex<tensor::take>(), mr.To);
    auto memtake         = w.app(w.app(w.annex<btensor::lift_mem>(), {mr.To, mr.To, mr.To}), take_fn);
    auto each            = w.app(w.annex<affine::id>(), mr.Ro);
    auto sched_e         = w.app(w.annex<btensor::mk_sched>(), w.tuple({mr.Ro, w.lit_nat_0()}));
    bool same_ty         = Check::test_alpha(mr.To, mr.Tp);
    const Def* dst_p     = out;
    if (!same_ty) std::tie(m, dst_p) = buffer::op_alloc(mr.Ro, mr.So, mr.Tp, m)->projs<2>();

    const Def* pass = w.annex<btensor::map_reduce_post>();
    pass            = w.app(pass, {w.lit_nat_1(), mr.nps});
    pass            = w.app(pass, {mr.To, mr.Tp, mr.Ro, mr.Ro, sched_e->type()});
    pass            = w.app(pass, {mr.So, mr.So, sched_e});
    pass            = w.app(pass, {mr.To, mr.Ro, mr.So, mr.Tps, mr.Rps, mr.Sps});
    pass            = w.app(pass, mr.To);
    pass            = w.app(pass, {memtake, mr.init, mempost});
    pass            = w.app(pass, each);
    pass            = w.app(pass, {each, mr.post_maps});
    return w.app(pass, {m, dst_p, out, post_is})->proj(2, 1);
}

const Def* LowerToMem::lower_pe_generic(const App* app) {
    auto& w = new_world();
    if (!w.driver().arg_nat("tensor", "pe-generic").value_or(1)) return nullptr;
    auto ranks = btensor::MapRedPost::of(app->callee()->as<App>()).ranks();
    if (!ranks) return nullptr;
    auto c    = rewrite(app->callee())->as<App>();
    auto mr   = btensor::MapRedPost::of(c);
    auto plan = PeGeneric{};
    if (!pe_generic_plan(c, &plan)) return nullptr;
    u64 rn = ranks->nloops();
    // A lone leaf would bound a run-time extent's scratch tile by nothing.
    for (u64 d = 0; d != rn; ++d)
        if (!Lit::isa(mr.Sr->proj(rn, d))) return nullptr;

    auto nis                    = ranks->nis;
    auto [levels, nlevels]      = cpu_levels(rn);
    auto N                      = w.lit_nat(nis);
    auto [xs, sis, prods, fuse] = generic_slots(app, mr.Rn, mr.Ris, mr.Sis, mr.Tis, plan.maps);
    bool inner                  = ranks->nps == 0 && Check::test_alpha(mr.To, mr.Tp);
    auto lanes                  = inner ? plan.lanes : w.lit_nat_0();
    auto dyn                    = w.tuple(DefVec(rn, [&](size_t) { return w.lit_ff(); }));
    auto tree = w.app(w.app(w.app(w.annex<tensor::generic_tree>(), {N, mr.Rn, mr.Ro}),
                            {plan.red_unit, plan.out_unit, plan.wdim, plan.rdim, lanes, plan.bytes, fuse}),
                      {mr.Sr, dyn});
    if (levels)
        tree = w.app(w.app(w.app(w.annex<tensor::sched_with_levels>(), {N, mr.Rn, w.lit_nat(nlevels)}), levels), tree);
    log().i("tile the map-reduce over {} with tensor.run_levels{}", mr.Sr, levels ? " at the given levels" : "");
    if (log().level() >= fe::Log::Level::Verbose) {
        auto text = std::string();
        show_tree(text, tree, 0);
        log().v("schedule tree:\n{}", text);
    }

    auto [cTo, ins_ty] = mr.comb->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto memcomb       = w.app(w.app(w.annex<btensor::lift_mem>(), {cTo, ins_ty, cTo}), mr.comb);

    // The leaves, which may also merge partial results, run an epilogue without inputs that keeps the type.
    auto merge     = plan.merge ? w.app(w.annex<tensor::levels_merge>(), mr.To) : nullptr;
    merge          = merge ? w.app(merge, plan.merge)
                           : w.app(w.app(w.annex<btensor::lift_mem>(), {mr.To, mr.To, mr.To}),
                                   w.app(w.annex<tensor::take>(), mr.To));
    const Def* run = inner ? w.annex<tensor::run_levels_post>() : w.annex<tensor::run_levels_with>();
    run            = w.app(run, {mr.nis, mr.To, mr.Ro, mr.Rn, mr.Tis, mr.Ris, sis});
    run            = w.app(run, {mr.So, mr.Sr});
    if (inner) {
        auto [pTo, pexts_ty] = mr.post->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
        run = w.app(run,
                    {memcomb, mr.init, w.app(w.app(w.annex<btensor::lift_mem>(), {pTo, pexts_ty, mr.Tp}), mr.post)});
        run = w.app(run, {merge, lanes});
    } else {
        run = w.app(run, {memcomb, mr.init});
    }
    run           = w.app(run, mr.map_out);
    run           = w.app(run, {plan.maps, prods});
    run           = w.app(run, {tree, mr.Sr, dyn});
    auto [m, out] = w.app(run, {fresh_mem(), xs})->projs<2>();
    return inner ? out : apply_post(app, m, out);
}

const Def* LowerToMem::lower_map_reduce(const App* app) {
    if (auto pe = lower_pe_cpu(app)) return pe;
    if (auto pe = lower_pe_generic(app)) return pe;
    // Thin bufferization: map the SSA `tensor.map_reduce_post` onto the buffer-world `btensor.map_reduce_post`,
    // reusing the (rewritten) meta. The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w            = new_world();
    auto c             = rewrite(app->callee())->as<App>();
    auto args          = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
    auto [is, post_is] = args->projs<2>();

    auto mr = btensor::MapRedPost::of(c);

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
    op             = w.app(op, mr.To);
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
    auto [m, out]  = w.app(op, w.tuple({am, dst, arg}))->projs<2>();
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
