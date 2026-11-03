#include "mim/plug/tensor/phase/lower_to_mem.h"

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
#include <mim/plug/mem/mem.h>

#include "mim/plug/tensor/phase/constraints.h"
#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

class UnitScale : public Rewriter {
public:
    using Rewriter::Rewriter;

    const Def* rewrite(const Def* def) override {
        if (def->isa<Var>() || def->isa<Axm>() || def->is_mutable()) return def;
        return Rewriter::rewrite(def);
    }

    const Def* rewrite_imm_App(const App* app) override {
        if (Axm::isa(affine::semiop::mul, app))
            if (Lit::isa(app->arg(2, 1)) == 1) return rewrite(app->arg(2, 0));
        if (Axm::isa(affine::op::add, app)) {
            auto [left, right] = app->args<2>();
            if (auto lit = Axm::isa<affine::lit>(left); lit && Lit::isa(lit->arg()) == 0) return rewrite(right);
            if (auto lit = Axm::isa<affine::lit>(right); lit && Lit::isa(lit->arg()) == 0) return rewrite(left);
        }
        return Rewriter::rewrite_imm_App(app);
    }
};

/// Does `t` (recursively through immutable sigmas) contain a Pi?
bool contains_pi(const Def* t) {
    if (t->isa<Pi>()) return true;
    if (auto sig = t->isa_imm<Sigma>())
        for (auto op : sig->ops())
            if (contains_pi(op)) return true;
    return false;
}

void check_buffer_shape(const Def* rank, const Def* shape) {
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
const Def* splat_scalar(const Def* d) {
    d = Axm::peel<tensor::buf>(d);
    if (auto splat = Axm::isa<tensor::splat>(d)) return splat->arg(2, 1);
    if (!d->isa<Pack>()) return nullptr;
    while (auto pack = d->isa<Pack>())
        d = pack->body();
    return (d->is_closed() && !d->type()->isa<Arr>()) ? d : nullptr;
}

const Def* compose_read_map(const Def* inner, const Def* outer) {
    auto& w  = outer->world();
    auto pi  = outer->type()->as<Pi>();
    auto map = w.mut_lam(pi->dom(), inner->type()->as<Pi>()->codom());
    return map->set(true, w.app(inner, w.app(outer, map->var())));
}

std::optional<PureRead> tensor_read(const Def* value, const Def* shape) {
    value      = Axm::peel<tensor::buf>(value);
    auto& w    = value->world();
    auto adapt = [&](std::optional<PureRead> read, const Def* produced) {
        if (read && produced != shape) {
            auto rank   = shape->arity();
            auto bridge = w.app(w.annex<tensor::reshape_map>(), {produced->arity(), rank});
            bridge      = w.app(bridge, {produced, shape});
            read->map   = compose_read_map(read->map, bridge);
        }
        return read;
    };
    if (auto slice = Axm::isa<tensor::slice>(value)) {
        auto [meta, shape, box]    = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, step, extent] = box->projs<3>();
        auto [type, rank]          = meta->projs<2>();
        auto map                   = w.app(w.app(w.annex<tensor::slice_map>(), rank), {start, step});
        return adapt(PureRead{slice->arg(), map, type, rank, shape}, extent);
    }
    if (auto slice = Axm::isa<tensor::slice_box>(value)) {
        auto [meta, shape, box]   = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, extent, cap] = box->projs<3>();
        auto [type, rank]         = meta->projs<2>();
        auto map = w.app(w.app(w.annex<tensor::slice_map>(), rank), {start, w.pack(rank, w.lit_nat_1())});
        return adapt(PureRead{slice->arg(), map, type, rank, shape}, cap);
    }
    auto read = is_pure_read(value);
    return read ? adapt(read, btensor::MapRedPost::of(value->as<App>()->callee()->as<App>()).So) : read;
}

/// Is `app` a tensor op whose lowering consumes a LowerToMem::fresh_mem?
bool wants_fresh_mem(const App* app) {
    return Axm::isa<tensor::tile_of>(app) || Axm::isa<tensor::tile_of_box>(app) || Axm::isa<tensor::slice>(app)
        || Axm::isa<tensor::slice_box>(app) || Axm::isa<tensor::stage>(app) || Axm::isa<tensor::tile_fold>(app)
        || Axm::isa<tensor::tiled>(app) || Axm::isa<tensor::tiled_multi>(app) || Axm::isa<tensor::tiled_n>(app)
        || Axm::isa<tensor::generate>(app) || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::map_reduce_post>(app)
        || Axm::isa<tensor::pad>(app) || Axm::isa<tensor::concat>(app) || Axm::isa<tensor::gather>(app)
        || Axm::isa<tensor::scatter>(app) || Axm::isa<tensor::set>(app) || Axm::isa<tensor::from_array>(app)
        || Axm::isa<tensor::scalar>(app) || Axm::isa<tensor::splat>(app) || Axm::isa<tensor::prepack>(app);
}

/// Is `app` one of the tensor ops this phase bufferizes?
bool is_tensor_op(const App* app) { return Axm::isa<tensor::splat>(app) || wants_fresh_mem(app); }

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
            if (positive(shapes->proj(2, 0))) region(body);
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
        for (auto op : def->ops())
            if (op) scan(op, site);
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
Lam* param_of(const Def* d) {
    while (auto ex = d->isa<Extract>())
        d = ex->tuple();
    if (auto var = d->isa<Var>()) return var->binder()->isa_mut<Lam>();
    return nullptr;
}

const Def* scalar_result(const Def* source, const Def* mem, const Def* buf) {
    if (TensorType::isa(source->type())) return buf;
    return buffer::op_read(mem, buf, buf->world().tuple())->proj(2, 1);
}

const Def* copy_buffer(const Def* mem, const Def* dst, const Def* src, const Def* shape) {
    auto& w     = mem->world();
    auto result = btensor::build_pointwise(
        w.sigma({mem->type(), dst->type()}), mem, dst, src, shape, Lit::as(shape->arity()), "foldCopy",
        [&](Defs iters, const Def* input, const Def* current) {
            auto [next, value] = buffer::op_read(current, input, btensor::folded_idx(shape, iters))->projs<2>();
            return std::pair{next, value};
        });
    return result->proj(2, 0);
}

bool full_output(const btensor::MapRedPost& mr) {
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

bool sequential_tile(const Def* root) {
    if (!RealizationRequests(root).reusable(root)) return false;
    DefVec pending{root};
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->is_mutable()) continue;
        if (auto app = def->isa<App>(); app && is_tensor_op(app)) {
            if (!(Axm::isa<tensor::map_reduce_post>(app) || Axm::isa<tensor::stage>(app) || Axm::isa<tensor::slice>(app)
                  || Axm::isa<tensor::slice_box>(app) || Axm::isa<tensor::splat>(app)
                  || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::generate>(app)
                  || (Axm::isa<tensor::tile_fold>(app) && app->type()->isa_imm<Sigma>())))
                return false;
        }
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    return true;
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
    auto boundaries = fe::BFSWorklist<DefSet>();
    for (auto external : old_world().externals().muts())
        boundaries.push(external->type());
    while (!boundaries.empty()) {
        auto type = boundaries.pop();
        if (auto tensor = TensorType::isa(type)) {
            check_buffer_shape(tensor->r, tensor->s);
            continue;
        }
        if (type->isa<Pi>() || type->isa<Sigma>() || type->isa<Arr>()) boundaries.push(type->ops());
    }
    auto wl = fe::BFSWorklist<DefSet>();
    wl.push(old_world().externals().muts());
    while (!wl.empty()) {
        auto d = wl.pop();
        if (d->isa<Axm>()) continue;
        if (auto app = d->isa<App>()) {
            if (Axm::isa<tensor::tiled>(app) || Axm::isa<tensor::tiled_multi>(app)) {
                wl.push(grid_tile_plan(app).inputs);
                wl.push(app->type());
                continue;
            }
            if (Axm::isa<tensor::tiled_n>(app)) {
                wl.push(multi_tile_plan(app).inputs);
                wl.push(app->type());
                continue;
            }
            if (Axm::isa<tensor::tile_fold>(app)) {
                auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
                wl.push(init);
                wl.push(inputs);
                wl.push(app->type());
                continue;
            }
        }
        if (is_tensor_type(d)) {
            tensors_seen_      = true;
            auto [r, shape, T] = TensorType::decode(d->as<Nom>());
            if (!Lit::isa<u64>(r)) d->blame("cannot bufferize: non-literal tensor rank").bail();
            for (auto extent : shape->projs(Lit::as(r)))
                if (Lit::isa(extent) == 0) d->blame("cannot bufferize: zero-extent tensors are not supported").bail();
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
                if (!Pi::isa_cn(lam->type()) && !lam->body()->isa<Lam>())
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
        }
        for (auto op : d->ops()) {
            if (!op) continue;
            if (auto f = op->isa_mut<Lam>(); f && is_tensor_fn(f) && f->ret_pi() && !d->isa<Var>()
                                             && !(d->isa<App>() && d->as<App>()->callee() == op)
                                             && !Axm::isa<cps::cps2ds>(d) && !Axm::isa<cps::cps2ds_dep>(d))
                f->blame("cannot bufferize: tensor-processing function used as a value").bail();
            wl.push(op);
        }
        if (auto app = d->isa<App>()) {
            auto [axm, curry, trip] = Axm::get(app);
            // Partial axiom applications still carry uninstantiated tensor ranks in their schemas.
            if (axm && curry) continue;
        }
        if (auto app = Axm::isa<tensor::map_reduce_post>(d))
            if (Axm::isa<btensor::clip_sched>(btensor::MapRedPost::of(app->callee()->as<App>()).sched)) continue;
        if (d->type()) wl.push(d->type());
    }
}

void LowerToMem::start() {
    tensor_key_ = tensor_key(old_world());
    validate();
    RWPhase::start();
    assert(pending_.empty());
}

const Def* LowerToMem::buf_of(const Def* type) {
    auto [r, s, T] = TensorType::decode(type->as<Nom>());
    check_buffer_shape(r, s);
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
    auto var = k->var();
    bind_new(k->has_var());
    return var;
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
        if (auto realized = lookup(old_def)) return realized;
        if (is_tensor_type(old_def->type()))
            for (auto [request, realized] : realizations_)
                if (Check::test_alpha(request, old_def)) {
                    if (auto shape = fe::lookup(storage_shapes_, request)) storage_shapes_[old_def] = shape;
                    return realized;
                }
    }
    if (app && wants_fresh_mem(app)) {
        if (auto new_def = fe::lookup(fresh_memo_, old_def)) return new_def;
        if (app) prepare_realizations(app);
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
    auto _       = fe::Restore(pending_, {});
    auto __      = fe::Restore(fresh_memo_, {});
    auto storage = fe::Restore(storage_shapes_, storage_shapes_);
    auto cache   = fe::Restore(realizations_, realizations_);
    auto scratch = fe::Restore<Scratch*>(scratch_, nullptr);
    push();
    auto new_lam = new_world().mut_lam(rewrite(lam->type())->as<Pi>());
    map(lam, new_lam);
    if (!lam->is_set()) {
        pop();
        return new_lam;
    }

    auto ___    = enter(lam);
    auto filter = rewrite(lam->filter());
    prepare_realizations(lam->body());
    auto body = rewrite(lam->body());
    if (!pending_.empty() && !Pi::isa_cn(new_lam->type())) {
        auto& w  = new_world();
        auto run = w.mut_fun(w.tuple()->type(), body->type())->set("tensorValue");
        run->app(true, run->ret_var(), body);
        run->set(true, wrap_fresh_mem(run->body()));
        body = w.app(cps::op_cps2ds_dep(run), w.tuple());
    } else {
        body = wrap_fresh_mem(body);
    }
    new_lam->set(filter, body);
    pop();
    return seal(lam, new_lam);
}

std::pair<const Def*, const Def*> LowerToMem::allocate(const Def* rank, const Def* shape, const Def* type, bool fills) {
    check_buffer_shape(rank, shape);
    auto r       = Lit::isa(rank);
    auto literal = r.has_value();
    if (r)
        for (size_t d = 0; d != *r; ++d)
            literal &= Lit::isa(shape->proj(*r, d)).has_value();
    if (!scratch_ || !fills || !literal || !type->is_closed())
        return buffer::op_alloc(rank, shape, type, fresh_mem())->projs<2>();
    auto allocation    = buffer::op_alloc(rank, shape, type, scratch_->mem);
    auto [run, call]   = btensor::ds_splice(allocation, scratch_->result_type, "tileScratch");
    auto [mem, buffer] = run->var(2, 0)->projs<2>();
    bind_new(run->has_var());
    scratch_->mem = mem;
    scratch_->bindings.emplace_back(run, call);
    return {fresh_mem(), buffer};
}

void LowerToMem::prepare_realizations(const Def* root, const Lam* region, Defs destinations) {
    for (auto group : RealizationRequests(root).shared()) {
        auto value = group.front();
        if (std::ranges::any_of(destinations, [&](const Def* output) { return Check::test_alpha(output, value); }))
            continue;
        if (region) {
            bool available = true;
            for (auto var : value->free_vars())
                available &= region->has_free_var(var);
            if (!available) continue;
        }
        auto realized = lookup(value);
        if (!realized) realized = rewrite(value);
        realizations_[value] = realized;
        for (auto request : group) {
            map(request, realized);
            if (auto shape = fe::lookup(storage_shapes_, value)) storage_shapes_[request] = shape;
        }
        profile_count("tensor.realizations.shared");
    }
}

const Def* LowerToMem::rewrite_imm_App(const App* app) {
    auto& w = new_world();
    if (Axm::isa<tensor::frozen>(app)) return rewrite(app->arg());
    if (Axm::isa<tensor::prepack>(app)) return lower_prepack(app);
    if (Axm::isa<tensor::group_req>(app)) {
        report_group_request(app, "request reached bufferization without realization");
        return rewrite(app->arg(5, 0));
    }
    if (Axm::isa<cps::cps2ds_dep>(app)) return cps::op_cps2ds_dep(rewrite(app->arg()));
    if (Axm::isa<tensor::buf>(app)) {
        auto value = rewrite(app->arg());
        if (auto shape = fe::lookup(storage_shapes_, app->arg())) storage_shapes_[app] = shape;
        return value;
    }
    // A `tensor.if_static` still stuck at lowering time guards a runtime value: residualize to
    // its dynamic branch.
    if (Axm::isa<tensor::if_static>(app)) return rewrite(app->arg(3, 2));
    if (Axm::isa<tensor::from_array>(app)) return lower_from_array(app);
    if (Axm::isa<tensor::get>(app)) return lower_get(app);
    if (Axm::isa<tensor::set>(app)) return lower_set(app);
    if (Axm::isa<tensor::scalar>(app)) {
        auto T = rewrite(app->arg()->type());
        return buffer::op_init(w.lit_nat_0(), w.tuple(), T, fresh_mem(), rewrite(app->arg()))->proj(2, 1);
    }
    if (Axm::isa<tensor::item>(app)) return buffer::op_read(fresh_mem(), rewrite(app->arg()), w.tuple())->proj(2, 1);
    if (Axm::isa<tensor::splat>(app)) return lower_splat(app);
    if (Axm::isa<tensor::reshape>(app)) {
        auto input = rewrite(app->arg());
        if (input->type() == buf_of(app->type())) return input;
        const Def* value = old_world().annex<tensor::reshape_impl>();
        for (auto arg : app->uncurry_args<4>())
            value = old_world().app(value, arg);
        return rewrite(value);
    }
    if (Axm::isa<tensor::tiled>(app)) return lower_tiled(app);
    if (Axm::isa<tensor::tiled_multi>(app)) return lower_tiled(app);
    if (Axm::isa<tensor::tiled_n>(app)) return lower_tiled_n(app);
    if (Axm::isa<tensor::tile_fold>(app)) return lower_tile_fold(app);
    if (Axm::isa<tensor::stage>(app)) return lower_stage(app);
    if (Axm::isa<tensor::slice>(app) || Axm::isa<tensor::slice_box>(app)) {
        auto [meta, shape, box] = app->callee()->as<App>()->uncurry_args<3>();
        auto& old               = old_world();
        auto impl
            = Axm::isa<tensor::slice_box>(app) ? old.annex<tensor::slice_box_impl>() : old.annex<tensor::slice_impl>();
        auto result = old.app(old.app(old.app(old.app(impl, meta), shape), box), app->arg());

        return rewrite(result);
    }
    if (Axm::isa<tensor::generate>(app)) return lower_generate(app);
    if (Axm::isa<tensor::broadcast>(app)) return lower_broadcast(app);
    if (Axm::isa<tensor::map_reduce_post>(app)) return lower_map_reduce(app);
    if (Axm::isa<tensor::pad>(app)) return lower_pad(app);
    if (Axm::isa<tensor::concat>(app)) return lower_concat(app);
    if (Axm::isa<tensor::gather>(app)) return lower_gather(app);
    if (Axm::isa<tensor::scatter>(app)) return lower_scatter(app);
    return RWPhase::rewrite_imm_App(app);
}

const Def* LowerToMem::splat_buffer(const Def* arr_ty, const Def* scalar) {
    // `buffer.lit` sets every element to `scalar`; `btensor.lower_map_reduce` fills it with a loop rather
    // than storing a monolithic literal array (which the LLVM backend cannot digest for large shapes).
    auto [bro, bso, boT] = Axm::isa<buffer::Buf>(buf_of(arr_ty))->args<3>();
    if (scratch_) {
        auto [mem, buffer] = allocate(bro, bso, boT);
        auto& w            = new_world();
        return btensor::build_pointwise(
                   w.sigma({mem->type(), buffer->type()}), mem, buffer, scalar, bso, Lit::as(bro), "scratchSplat",
                   [](Defs, const Def* value, const Def* current) { return std::pair{current, value}; })
            ->proj(2, 1);
    }
    auto [m, out] = buffer::op_lit(bro, bso, boT, fresh_mem(), scalar)->projs<2>();
    return out;
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
    auto source       = rewrite(old);
    auto shape        = fe::lookup(storage_shapes_, old);
    if (!shape) shape = rewrite(s);
    auto& w          = new_world();
    auto rank        = Lit::as(r);
    auto coordinates = rewrite(index);
    auto adjusted    = w.tuple(DefVec(rank, [&](size_t d) {
        return w.call<core::idx>(w.lit_nat_0(), shape->proj(rank, d),
                                 w.call<core::idx2nat>(coordinates->proj(rank, d)));
    }));
    return buffer::op_read(fresh_mem(), source, *Shape(adjusted).fold(shape))->proj(2, 1);
}

const Def* LowerToMem::lower_set(const App* app) {
    auto [old, index, value] = app->args<3>();
    auto [r, s, T]           = TensorType::decode(app->type()->as<Nom>());
    auto source              = rewrite(old);
    auto [br, bs, bT]        = Axm::isa<buffer::Buf>(source->type())->args<3>();
    auto [m, dst]            = buffer::op_alloc(br, bs, bT, fresh_mem())->projs<2>();
    auto copied              = buffer::op_copy(m, dst, source);
    auto idx                 = *Shape(rewrite(index)).fold(rewrite(s));
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

    auto out_ty              = buf_of(app->type());
    auto [br, bs, bT]        = Axm::isa<buffer::Buf>(out_ty)->args<3>();
    auto result_ty           = w.sigma({w.call<mem::M>(0), out_ty});
    auto [sources, captures] = capture_producers(app->arg());
    auto compute             = [&](Defs iters, const Def* input, const Def* m) -> std::pair<const Def*, const Def*> {
        push();
        if (!sources.empty()) bind_new(param_of(input)->has_var());
        for (size_t i = 0; i != sources.size(); ++i)
            map(sources[i], input->proj(sources.size(), i));
        auto value = w.call(rewrite(app->arg()), iters);
        pop();
        return {m, value};
    };
    auto [am, dst] = allocate(br, bs, bT);
    return btensor::build_pointwise(result_ty, am, dst, w.tuple(captures), s_out, rn, "generate", compute)->proj(2, 1);
}

std::pair<DefVec, DefVec> LowerToMem::capture_producers(const Def* function) {
    if (auto body = function->isa<Lam>(); body && body->is_set()) prepare_realizations(body->body(), body);
    auto sources = DefVec(), captures = DefVec();
    auto body = function->isa<Lam>();
    if (!body || !body->is_set()) return {sources, captures};
    RealizationRequests requests(body->body());
    auto queue = fe::BFSWorklist<DefSet>();
    queue.push(body->body());
    while (!queue.empty()) {
        auto def = queue.pop();
        if (def->isa<Var>() || def->isa<Axm>()) continue;
        if (auto app = def->isa<App>(); app && wants_fresh_mem(app)) {
            bool available = true;
            for (auto var : def->free_vars())
                available &= body->has_free_var(var);
            if (available) {
                auto value = rewrite(def);
                if (requests.shareable(def)) {
                    realizations_[def] = value;
                    map(def, value);
                }
                sources.push_back(def);
                captures.push_back(value);
                continue;
            }
        }
        for (auto op : def->ops())
            if (op) queue.push(op);
    }
    return {sources, captures};
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
    auto [am, dst] = allocate(bro, bso, boT);
    auto [m, out]  = w.app(op, w.tuple({am, dst, input}))->projs<2>();
    return out;
}

const Def* LowerToMem::lower_map_reduce(const App* app, const Destination* destination) {
    // Thin bufferization: map the SSA `tensor.map_reduce_post` onto the buffer-world `btensor.map_reduce_post`,
    // reusing the (rewritten) meta. The loop generation lives in the btensor plugin (`btensor.lower_map_reduce`).
    auto& w      = new_world();
    auto old_mr  = btensor::MapRedPost::of(app->callee()->as<App>());
    auto& old    = old_world();
    auto changed = false;
    auto group   = [&](const Def* values, const Def* ranks, const Def* shapes, const Def* maps, u64 n) {
        auto sources = DefVec(), source_ranks = DefVec(), source_shapes = DefVec(), reads = DefVec();
        for (u64 i = 0; i != n; ++i) {
            auto value = values->proj(n, i), rank = ranks->proj(n, i), shape = shapes->proj(n, i),
                 map = maps->proj(n, i);
            while (auto read = tensor_read(value, shape)) {
                if (!TensorType::isa(read->src->type())) break;
                value   = read->src;
                rank    = read->R;
                shape   = read->S;
                map     = compose_read_map(read->map, map);
                changed = true;
            }
            sources.push_back(value);
            source_ranks.push_back(rank);
            source_shapes.push_back(shape);
            reads.push_back(map);
        }
        return std::tuple{old.tuple(sources), old.tuple(source_ranks), old.tuple(source_shapes), old.tuple(reads)};
    };
    auto [values, posts]               = app->args<2>();
    auto [inputs, ranks, bounds, maps] = group(values, old_mr.Ris, old_mr.Sis, old_mr.maps, Lit::as(old_mr.nis));
    auto [post_values, post_ranks, post_bounds, post_maps]
        = group(posts, old_mr.Rps, old_mr.Sps, old_mr.post_maps, Lit::as(old_mr.nps));
    auto init = old_mr.init;
    if (auto initial = tensor_init(old_mr)) {
        auto value = initial->outs, shape = initial->shape, map = initial->map;
        while (auto read = tensor_read(value, shape)) {
            if (read->R != old_mr.Ro || !TensorType::isa(read->src->type())) break;
            value   = read->src;
            shape   = read->S;
            map     = compose_read_map(read->map, map);
            changed = true;
        }
        init = old.tuple({shape, map, value});
    }
    if (changed) {
        const Def* result = old.annex<tensor::map_reduce_post>();
        for (auto arg : {old_mr.nis_nps, old_mr.meta, old_mr.shapes,
                         old.tuple({old_mr.Tis, ranks, bounds, old_mr.Tps, post_ranks, post_bounds}), init->type(),
                         old.tuple({old_mr.comb, init, old_mr.post}), old_mr.map_out, old.tuple({maps, post_maps}),
                         old.tuple({inputs, post_values})})
            result = old.app(result, arg);

        auto value = lower_map_reduce(result->as<App>(), destination);
        if (auto shape = fe::lookup(storage_shapes_, result)) storage_shapes_[app] = shape;
        return value;
    }
    auto initial       = tensor_init(old_mr);
    auto args          = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
    auto [is, post_is] = args->projs<2>();

    auto mr = btensor::MapRedPost::decode(
        rewrite(old_mr.nis_nps), rewrite(old_mr.meta), rewrite(old_mr.shapes), rewrite(old_mr.in_tys),
        w.tuple({rewrite(old_mr.comb), w.bot(rewrite(old_mr.To)), rewrite(old_mr.post)}), rewrite(old_mr.map_out),
        rewrite(old_mr.maps_all));
    if (auto clipped = Axm::isa<btensor::clip_sched>(mr.sched)) {
        auto [valid, output_cap, loop_cap, schedule] = clipped->args<4>();
        if (!lit_projs(output_cap, Lit::as(mr.Ro)) || !lit_projs(loop_cap, Lit::as(mr.Rn)))
            app->blame("tensor tile storage and loop caps must be literal").bail();
        mr.So                = output_cap;
        mr.Sr                = loop_cap;
        mr.shapes            = w.tuple({mr.So, mr.Sr, mr.sched});
        storage_shapes_[app] = output_cap;
    }
    const Def* seed = nullptr;
    if (initial) {
        if (auto scalar = splat_scalar(initial->outs)) {
            seed = rewrite(scalar);
        } else {
            auto outs  = to_buffer(rewrite(initial->outs), initial->outs);
            auto shape = fe::lookup(storage_shapes_, initial->outs);
            seed       = w.tuple({shape ? shape : rewrite(initial->shape), rewrite(initial->map), outs});
        }
    } else {
        seed = rewrite(old_mr.init);
    }
    if (destination) {
        auto ro     = Lit::as(mr.Ro);
        auto map    = w.mut_lam(mr.map_out->type()->as<Pi>())->set("tileWriteMap");
        auto coords = w.app(mr.map_out, map->var());
        map->set(true, w.tuple(DefVec(ro, [&](size_t d) {
            return w.call(affine::op::add,
                          Defs{coords->proj(ro, d), w.call<affine::lit>(destination->origin->proj(ro, d))});
        })));
        mr.map_out   = map;
        auto maps    = DefVec(Lit::as(mr.nps), [&](size_t i) -> const Def* {
            auto old   = mr.post_maps->proj(Lit::as(mr.nps), i);
            auto local = w.mut_lam(old->type()->as<Pi>())->set("tilePostMap");
            auto index = w.tuple(DefVec(ro, [&](size_t d) {
                return w.call(affine::op::sub,
                              Defs{local->var(ro, d), w.call<affine::lit>(destination->origin->proj(ro, d))});
            }));
            local->set(true, w.app(old, index));
            return local;
        });
        mr.post_maps = w.tuple(maps);
        mr.maps_all  = w.tuple({mr.maps, mr.post_maps});
        mr.So        = destination->shape;
        mr.shapes    = w.tuple({mr.So, mr.Sr, mr.sched});
    }

    // The generic rewrite rebuilds the argument tuple with its stale value-array type even when its
    // elements were converted to buffers, which would not be assignable to the op's
    // `«nis; buffer.Buf …»` domain - hence the re-tupling in `buffer_list`.
    auto [old_is, old_post_is] = app->arg()->projs<2>();
    auto shapes                = [&](const Def* values, const Def* logical, u64 n) {
        return w.tuple(DefVec(n, [&](size_t i) {
            auto source = values->proj(n, i);
            if (auto shape = fe::lookup(storage_shapes_, source)) return shape;
            return logical->proj(n, i);
        }));
    };
    auto nis = Lit::as(mr.nis), nps = Lit::as(mr.nps);
    mr.Sis    = shapes(old_is, mr.Sis, nis);
    mr.Sps    = shapes(old_post_is, mr.Sps, nps);
    mr.in_tys = w.tuple({mr.Tis, mr.Ris, mr.Sis, mr.Tps, mr.Rps, mr.Sps});

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
    op      = w.app(op, mr.shapes);
    op      = w.app(op, mr.in_tys);
    op      = w.app(op, seed->type());
    op      = w.app(op, w.tuple({memcomb, seed, mempost}));
    op      = w.app(op, mr.map_out);
    op      = w.app(op, mr.maps_all);
    const Def* am;
    const Def* dst;
    if (destination) {
        am  = destination->mem;
        dst = destination->buffer;
    } else {
        std::tie(am, dst) = allocate(mr.Ro, mr.So, mr.Tp, full_output(old_mr));
    }
    auto [m, out] = w.app(op, w.tuple({am, dst, is, post_is}))->projs<2>();
    if (destination) return w.tuple({m, out});
    return scalar_result(app, m, out);
}

const Def* LowerToMem::materialize(const Def*, const Def* value) { return rewrite(value); }
const Def* LowerToMem::to_buffer(const Def* value, const Def*) { return value; }
const Def* LowerToMem::make_buffer(const Def* value, const Def*, const Def*) { return rewrite(value); }

const Def* LowerToMem::lower_into(const Def* root, const Destination& destination) {
    for (size_t i = 0; i != 128; ++i) {
        auto app = root->isa<App>();
        if (!app) break;
        auto lam = app->callee()->isa<Lam>();
        if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
        root = lam->reduce(app->arg()).back();
    }

    prepare_realizations(root);
    if (auto app = root->isa<App>()) {
        if (Axm::isa<tensor::scalar>(app))
            return buffer::op_write(destination.mem, destination.buffer, new_world().tuple(), rewrite(app->arg()));
        if (Axm::isa<tensor::map_reduce_post>(app)) return lower_map_reduce(app, &destination);
        if (Axm::isa<tensor::tiled>(app)) return lower_tiled(app, &destination);
    }
    log().w("tensor destination fallback: tile root {} uses a temporary buffer and copy", root);
    auto& w            = new_world();
    auto value         = rewrite(root);
    auto rank          = Lit::as(destination.shape->arity());
    auto source        = to_buffer(value, root);
    auto shape         = destination.extent;
    auto [run, call]   = btensor::ds_splice(w.tuple({destination.mem, destination.buffer, source}),
                                            w.sigma({destination.mem->type(), destination.buffer->type()}), "tileCopy");
    auto [m, dst, src] = run->var(2, 0)->projs<3>();
    const Def* acc     = w.tuple({m, dst});
    const Def* exit    = run->ret_var();
    auto cur           = run;
    auto iters         = affine::counting_nest(cur, exit, acc, shape->projs(rank), "tileCopy");
    auto [loop_mem, loop_dst] = acc->projs<2>();
    const Def* element        = src;
    if (Axm::isa<buffer::Buf>(src->type()))
        std::tie(loop_mem, element) = buffer::op_read(loop_mem, src, btensor::folded_idx(shape, iters))->projs<2>();
    DefVec coords(rank);
    for (size_t d = 0; d != rank; ++d) {
        auto offset = destination.origin->proj(rank, d);
        auto index  = w.call<core::bitcast>(w.type_nat(), iters[d]);
        coords[d]   = w.call(core::nat::add, Defs{offset, index});
    }
    auto index = w.tuple(DefVec(
        rank, [&](size_t d) { return w.call<core::idx>(w.lit_nat(0), destination.shape->proj(rank, d), coords[d]); }));
    auto mem   = buffer::op_write(loop_mem, loop_dst, *Shape(index).fold(destination.shape), element)->proj(2, 0);
    cur->app(true, exit, w.tuple({mem, loop_dst}));
    return call;
}

const Def* LowerToMem::lower_tile_body(const GridTilePlan& plan,
                                       const Def* off,
                                       const Def* ext,
                                       const Def* inputs,
                                       Defs state,
                                       Scratch* pool,
                                       const Destination* destination) {
    auto& old           = old_world();
    auto& w             = new_world();
    auto rank           = Lit::as(ext->arity());
    auto parameter      = old.mut_lam(old.pi(old.arr(rank, old.type_nat()), old.type_nat()));
    auto origin         = parameter->var();
    auto specialized    = old.tuple(DefVec(rank, [&](size_t d) {
        auto component = off->proj(rank, d);
        return Lit::isa(component) == 0 ? old.lit_nat_0() : origin->proj(rank, d);
    }));
    auto literal_extent = Rewriter(old).rewrite(ext);
    const Def* root;
    if (auto lam = plan.body->isa<Lam>(); lam && lam->is_set())
        root = old.app(lam->reduce(old.tuple({specialized, literal_extent})).back(), inputs);
    else
        plan.body->blame("tensor.tiled body must be a defined lambda").bail();
    auto pending = fe::Restore(pending_, {});
    auto memo    = fe::Restore(fresh_memo_, {});
    auto storage = fe::Restore(storage_shapes_, storage_shapes_);
    auto cache   = fe::Restore(realizations_, realizations_);
    auto scratch = fe::Restore(scratch_, sequential_tile(root) ? pool : nullptr);
    push();
    bind_old(parameter->has_var());
    map(origin, off);
    const Def* mem = state[0];
    DefVec outputs;
    auto shared = RealizationRequests(root).shared();
    DefVec destinations;
    for (u64 i = 0; i != plan.count; ++i) {
        auto value = plan.multiple ? root->proj(plan.count, i) : root;
        if (Axm::isa<tensor::map_reduce_post>(value) || Axm::isa<tensor::tiled>(value)) destinations.push_back(value);
    }
    prepare_realizations(root, nullptr, destinations);
    Rewriter substitutions(old);
    for (auto var : root->free_vars())
        substitutions.map(var, var);
    for (u64 i = 0; i != plan.count; ++i) {
        auto value            = plan.multiple ? root->proj(plan.count, i) : root;
        auto [origin, extent] = plan.box(i, specialized, literal_extent);
        auto shape            = plan.shapes->proj(plan.count, i);
        auto r                = plan.ranks->proj(plan.count, i);
        auto target_origin    = rewrite(origin);
        if (destination)
            target_origin = w.tuple(DefVec(rank, [&](size_t d) {
                return w.call(core::nat::add, Defs{destination->origin->proj(rank, d), off->proj(rank, d)});
            }));
        auto result = lower_into(
            substitutions.rewrite(value),
            {mem, state[i + 1], destination ? destination->shape : rewrite(shape), target_origin, rewrite(extent)});
        mem = result->proj(2, 0);
        outputs.push_back(result->proj(2, 1));
        if (plan.multiple) {
            auto parameter = old.mut_lam(
                type_tensor(plan.ranks->proj(plan.count, i), shape, plan.types->proj(plan.count, i)), old.type_nat());
            bind_old(parameter->has_var());
            map(parameter->var(), outputs.back());
            auto read = old.app(old.app(old.annex<tensor::slice>(), {plan.types->proj(plan.count, i), r}), shape);
            auto view = old.app(old.app(read, {origin, old.pack(r, old.lit_nat_1()), extent}), parameter->var());
            substitutions.map(value, view);
            for (auto& group : shared)
                if (Check::test_alpha(value, group.front()))
                    for (auto alias : group)
                        substitutions.map(alias, view);
        }
    }
    DefVec result_fields{mem};
    result_fields.insert(result_fields.end(), outputs.begin(), outputs.end());
    const Def* result = w.tuple(result_fields);
    if (!pending_.empty()) {
        auto [run, call] = btensor::ds_splice(w.tuple(), result->type(), "tileBody");
        run->app(true, run->ret_var(), result);
        run->set(true, wrap_fresh_mem(run->body()));
        result = call;
    }
    pop();
    return result;
}

const Def* LowerToMem::lower_tiled(const App* app, const Destination* destination) {
    auto& w            = new_world();
    auto scratch_scope = fe::Restore<Scratch*>(scratch_, nullptr);
    auto plan          = grid_tile_plan(app);
    auto shape = plan.shape, tiles = plan.tiles, inputs = plan.inputs;
    auto input_type = inputs->type();
    auto rank       = std::optional<u64>(plan.rank);
    auto sizes = fe::Vector<u64>(*rank), steps = fe::Vector<u64>(*rank);
    for (size_t d = 0; d != *rank; ++d) {
        auto size = Lit::isa(shape->proj(*rank, d)), tile = Lit::isa(tiles->proj(*rank, d));
        if (!size || !tile)
            app->blame("tensor.tiled destination lowering requires literal shapes and tile sizes").bail();
        sizes[d] = *size;
        steps[d] = std::min(*size, *tile);
    }
    const Def* mem = destination ? destination->mem : fresh_mem();
    DefVec buffers(plan.count);
    if (destination) {
        mem        = destination->mem;
        buffers[0] = destination->buffer;
    } else {
        for (u64 i = 0; i != plan.count; ++i)
            std::tie(mem, buffers[i])
                = buffer::op_alloc(rewrite(plan.ranks->proj(plan.count, i)), rewrite(plan.shapes->proj(plan.count, i)),
                                   rewrite(plan.types->proj(plan.count, i)), mem)
                      ->projs<2>();
    }
    auto captured  = materialize(input_type, inputs);
    auto parameter = old_world().mut_lam(input_type, old_world().type_nat());
    auto proxy     = parameter->var();
    DefVec initial{mem};
    initial.insert(initial.end(), buffers.begin(), buffers.end());
    auto state       = w.tuple(initial);
    auto [run, call] = btensor::ds_splice(w.tuple({state, captured}), state->type(), "tiled");
    auto runtime     = run->var(2, 0);
    Scratch scratch{mem, call->type(), {}};
    push();
    bind_new(run->has_var());
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, runtime->proj(2, 1));
    DefVec offsets(*rank), extents(*rank);
    std::function<const Def*(size_t, const Def*, const Def*)> emit
        = [&](size_t d, const Def* acc, const Def* exit) -> const Def* {
        if (d == *rank) {
            auto off    = w.tuple(offsets);
            auto extent = w.tuple(extents);
            auto result = lower_tile_body(plan, off, extent, proxy, acc->projs(plan.count + 1), &scratch, destination);
            return w.app(exit, result);
        }
        if (!sizes[d]) return w.app(exit, acc);
        auto count      = sizes[d] / steps[d];
        auto tail       = sizes[d] % steps[d];
        const Def* next = exit;
        if (tail) {
            auto after = w.mut_con(acc->type())->set("tileTail");
            offsets[d] = w.lit_nat(count * steps[d]);
            extents[d] = w.lit_nat(tail);
            after->set(true, emit(d + 1, after->var(), exit));
            next = after;
        }
        auto [loop, loop_call]     = affine::counting_for(w.lit_i64(count), acc, next, w.sym("tileGrid"));
        auto [index, state, yield] = loop->vars<3>();
        offsets[d] = w.call(core::nat::mul, Defs{w.call<core::bitcast>(w.type_nat(), index), w.lit_nat(steps[d])});
        extents[d] = w.lit_nat(steps[d]);
        loop->set(true, emit(d + 1, state, yield));
        return loop_call;
    };
    auto done = w.mut_con(call->type())->set("tileScratchDone");
    run->set(true, emit(0, runtime->proj(2, 0), done));
    auto done_mem = done->var()->proj(plan.count + 1, 0);
    for (auto [binding, invoke] : scratch.bindings | std::views::reverse) {
        auto buffer              = binding->var(2, 0)->proj(2, 1);
        auto [rank, shape, type] = Axm::isa<buffer::Buf>(buffer->type())->args<3>();
        done_mem                 = buffer::op_free(rank, shape, type, done_mem, buffer);
    }
    auto finished = done->var()->projs(plan.count + 1);
    finished[0]   = done_mem;
    done->app(true, run->ret_var(), w.tuple(finished));
    initial[0] = scratch.mem;
    call       = w.app(cps::op_cps2ds_dep(run), w.tuple({w.tuple(initial), captured}));
    for (auto [binding, invoke] : scratch.bindings | std::views::reverse) {
        binding->app(true, binding->ret_var(), call);
        call = invoke;
    }
    pop();
    if (destination) return call;
    auto out_mem = call->proj(plan.count + 1, 0);
    if (!plan.multiple) return scalar_result(app, out_mem, call->proj(2, 1));
    return w.tuple(DefVec(plan.count, [&](size_t i) -> const Def* {
        auto out   = call->proj(plan.count + 1, i + 1);
        auto value = app->proj(plan.count, i);
        return scalar_result(value, out_mem, out);
    }));
}

const Def* LowerToMem::lower_tiled_n(const App* app) {
    auto plan  = multi_tile_plan(app);
    auto& w    = new_world();
    auto& old  = old_world();
    auto count = Lit::as(plan.ranks->arity());
    DefVec buffers(count);
    auto mem = fresh_mem();
    for (size_t i = 0; i != count; ++i)
        std::tie(mem, buffers[i])
            = buffer::op_alloc(rewrite(plan.ranks->proj(count, i)), rewrite(plan.shapes->proj(count, i)),
                               rewrite(plan.types->proj(count, i)), mem)
                  ->projs<2>();
    auto captured                   = materialize(plan.inputs->type(), plan.inputs);
    auto [run, call]                = btensor::ds_splice(w.tuple({mem, w.tuple(buffers), captured}),
                                                         w.sigma({mem->type(), w.tuple(buffers)->type()}), "tiledN");
    auto [current, outputs, inputs] = run->var(2, 0)->projs<3>();
    auto pending                    = fe::Restore(pending_, {});
    push();
    bind_new(run->has_var());
    auto parameter = old.mut_lam(plan.inputs->type(), old.type_nat());
    auto proxy     = parameter->var();
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, inputs);
    for (size_t t = 0; t != plan.points.size(); ++t) {
        auto memo    = fe::Restore(fresh_memo_, {});
        auto storage = fe::Restore(storage_shapes_, storage_shapes_);
        auto cache   = fe::Restore(realizations_, realizations_);
        auto roots   = old.app(old.app(plan.body, plan.points[t]), proxy);
        Rewriter substitutions(old);
        for (auto var : roots->free_vars())
            substitutions.map(var, var);
        for (size_t i = 0; i != count; ++i) {
            auto rank       = Lit::as(plan.ranks->proj(count, i));
            auto shape      = plan.shapes->proj(count, i);
            auto [off, ext] = plan.owned[t][i]->projs<2>();
            bool empty      = false;
            for (size_t d = 0; d != rank; ++d)
                empty |= Lit::as(ext->proj(rank, d)) == 0;
            if (empty) continue;
            auto root  = roots->proj(count, i);
            auto value = substitutions.rewrite(root);
            auto result
                = lower_into(value, {current, outputs->proj(count, i), rewrite(shape), rewrite(off), rewrite(ext)});
            current = result->proj(2, 0);

            auto proxy = old.mut_lam(
                old.pi(type_tensor(plan.ranks->proj(count, i), shape, plan.types->proj(count, i)), old.type_nat()));
            proxy->var();
            bind_old(proxy->has_var());
            map(proxy->var(), outputs->proj(count, i));
            auto slice = old.app(old.annex<tensor::slice>(), {plan.types->proj(count, i), plan.ranks->proj(count, i)});
            slice      = old.app(slice, shape);
            slice      = old.app(slice, {off, old.pack(rank, old.lit_nat(1)), ext});
            substitutions.map(root, old.app(slice, proxy->var()));
        }
    }
    run->app(true, run->ret_var(), w.tuple({current, outputs}));
    run->set(true, wrap_fresh_mem(run->body()));
    pop();
    auto [done, results] = call->projs<2>();
    return w.tuple(DefVec(count, [&](size_t i) -> const Def* {
        auto result = results->proj(count, i);
        if (TensorType::isa(app->type()->proj(count, i))) return result;
        auto [r, shape, type] = Axm::isa<buffer::Buf>(result->type())->args<3>();
        auto rank             = Lit::as(r);
        auto zero = w.tuple(DefVec(rank, [&](size_t d) { return w.lit_idx(Lit::as(shape->proj(rank, d)), 0); }));
        return buffer::op_read(done, result, zero)->proj(2, 1);
    }));
}

const Def* LowerToMem::lower_prepack(const App* app) {
    auto& old                          = old_world();
    auto& w                            = new_world();
    auto [meta, shape, layout, marked] = app->uncurry_args<4>();
    auto frozen                        = Axm::isa<tensor::frozen>(marked);
    if (!frozen) app->blame("tensor.prepack requires a frozen whole function input").bail();
    auto source = Axm::peel<tensor::buf>(frozen->arg());
    auto base   = source;
    while (auto ex = base->isa<Extract>())
        base = ex->tuple();
    auto variable = base->isa<Var>();
    if (!variable || !variable->binder()->isa<Lam>() || !variable->binder()->as<Lam>()->is_external())
        app->blame("tensor.prepack requires a frozen whole function input").bail();
    auto [r, s, T] = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();
    if (!lit_projs(s, Lit::as(r))) app->blame("tensor.prepack requires a literal packed shape").bail();
    auto input   = rewrite(source);
    auto persist = w.app(w.app(w.app(w.annex<buffer::persist>(), {r, s, T}), input->type()), {fresh_mem(), input});
    auto [memory, buffer, fresh] = persist->projs<3>();
    auto impl                    = old.app(old.app(old.app(old.annex<tensor::stage_impl>(), meta), shape), layout);
    auto generated               = old.app(old.app(impl, old.lit_nat_0()), marked);
    auto generate                = Axm::isa<tensor::generate>(generated);
    if (!generate) app->blame("tensor.prepack reference must produce generate").bail();
    auto [gm, gs]      = generate->callee()->as<App>()->uncurry_args<2>();
    auto rank          = Lit::as(gm->proj(2, 1));
    auto state         = w.tuple({memory, buffer, input});
    auto [run, call]   = btensor::ds_splice(state, state->type(), "prepack");
    auto [m, dst, src] = run->var(2, 0)->projs<3>();
    const Def* acc     = w.tuple({m, dst, src});
    const Def* exit    = run->ret_var();
    auto cur           = run;
    auto trips         = w.select(fresh, w.lit_nat_1(), w.lit_nat_0());
    affine::counting_nest(cur, exit, acc, Defs{trips}, "prepackFresh");
    auto [loop_mem, loop_dst, loop_src] = acc->projs<3>();
    auto filled = btensor::build_pointwise(w.sigma({loop_mem->type(), loop_dst->type()}), loop_mem, loop_dst, loop_src,
                                           rewrite(gs), rank, "prepackFill",
                                           [&](Defs indices, const Def* value, const Def* mem) {
                                               push();
                                               map(source, value);
                                               auto element = w.call(rewrite(generate->arg()), indices);
                                               pop();
                                               return std::pair{mem, element};
                                           });
    auto [next, result] = filled->projs<2>();
    cur->app(true, exit, w.tuple({next, result, loop_src}));
    return call->proj(3, 1);
}

const Def* LowerToMem::lower_stage(const App* app) {
    auto& old                         = old_world();
    auto [meta, shape, layout, scope] = app->callee()->as<App>()->uncurry_args<4>();
    auto [type, rank, split]          = meta->projs<3>();
    auto input                        = app->arg();
    auto owned                        = input->isa<App>() && wants_fresh_mem(input->as<App>());
    if (auto extract = input->isa<Extract>())
        owned |= bool(Axm::isa<tensor::tiled_n>(extract->tuple()) || Axm::isa<tensor::tiled_multi>(extract->tuple()));
    if ((scratch_ || owned) && Lit::isa(split) == 0) {
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
    if (auto slice = Axm::isa<tensor::slice>(Axm::peel<tensor::buf>(app->arg()))) {
        auto [slice_meta, source_shape, box] = slice->callee()->as<App>()->uncurry_args<3>();
        auto [start, step, extent]           = box->projs<3>();
        auto source = slice->arg(), physical_shape = source_shape, physical_rank = rank;
        const Def* read = old.app(old.annex<affine::id>(), rank);
        while (auto view = tensor_read(source, physical_shape)) {
            if (!TensorType::isa(view->src->type())) break;
            read           = compose_read_map(view->map, read);
            source         = view->src;
            physical_shape = view->S;
            physical_rank  = view->R;
        }
        impl = old.app(old.annex<tensor::stage_read_impl>(), {type, rank, split, physical_rank});
        impl = old.app(impl, {source_shape, extent, physical_shape});
        impl = old.app(impl, read);
        impl = old.app(impl, layout);
        impl = old.app(impl, {start, step});
        impl = old.app(impl, scope);
        impl = old.app(impl, source);
    } else {
        impl      = old.annex<tensor::stage_impl>();
        auto args = DefVec();
        for (auto cur = app; cur; cur = cur->callee()->isa<App>())
            args.push_back(cur->arg());
        for (auto arg : args | std::views::reverse)
            impl = old.app(impl, arg);
    }
    auto generate = Axm::isa<tensor::generate>(impl);
    if (!generate) app->blame("tensor.stage reference reindexing did not produce a generate").bail();

    return lower_generate(generate);
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
    if (Axm::isa<tensor::tiled>(app)) {
        auto [meta, shapes, it, ins, body] = app->uncurry_args<5>();
        auto rank                          = Lit::as(meta->proj(2, 1));
        auto lam                           = app->arg()->isa<Lam>();
        if (!lam || !lam->is_set()) return false;
        auto param = w.mut_lam(w.pi(w.arr(rank, w.type_nat()), w.type_nat()));
        auto off   = param->var();
        auto next  = w.tuple(DefVec(
            rank, [&](size_t d) { return w.call(core::nat::add, Defs{origin->proj(rank, d), off->proj(rank, d)}); }));
        return linear_step(w.app(lam->reduce(w.tuple({off, shapes->proj(2, 1)})).back(), ins), accumulator, next);
    }
    if (!Axm::isa<tensor::map_reduce_post>(app)) return false;
    auto mr      = btensor::MapRedPost::of(app->callee()->as<App>());
    auto initial = tensor_init(mr);
    if (!initial) return false;
    while (auto read = tensor_read(initial->outs, initial->shape)) {
        if (read->R != mr.Ro || !TensorType::isa(read->src->type())) break;
        initial = TensorInit{read->S, compose_read_map(read->map, initial->map), read->src};
    }
    if (Axm::peel<tensor::buf>(initial->outs) != accumulator) return false;
    auto rank       = Lit::as(mr.Ro);
    auto map        = w.mut_lam(mr.map_out->type()->as<Pi>());
    auto output_map = mr.map_out->isa<Lam>();
    if (!output_map || !output_map->is_set()) return false;
    auto output   = output_map->reduce(map->var()).back();
    auto simplify = UnitScale(w);
    for (size_t d = 0; d != rank; ++d)
        if (!Check::test_alpha(simplify.rewrite(output->proj(rank, d)), map->var(Lit::as(mr.Rn), d))
            || !Check::test_alpha(mr.Sr->proj(Lit::as(mr.Rn), d), mr.So->proj(rank, d)))
            return false;
    map->set(true, w.tuple(DefVec(rank, [&](size_t d) {
        return w.call(affine::op::add, Defs{output->proj(rank, d), w.call<affine::lit>(origin->proj(rank, d))});
    })));
    auto init_map = initial->map->isa<Lam>();
    if (!init_map || !init_map->is_set()) return false;
    if (!Check::test_alpha(simplify.rewrite(init_map->reduce(map->var()).back()), simplify.rewrite(map->body())))
        return false;
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

const Def* LowerToMem::region_input(const Def* inputs) {
    auto value     = materialize(inputs->type(), inputs);
    auto& old      = old_world();
    auto parameter = old.mut_lam(inputs->type(), old.type_nat());
    auto proxy     = parameter->var();
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, value);
    return proxy;
}

const Def* LowerToMem::lower_tile_fold(const App* app) {
    if (app->type()->isa_imm<Sigma>()) return lower_product_fold(app);
    auto& w                                             = new_world();
    auto& old                                           = old_world();
    auto [type, blocks, init, input_type, inputs, body] = app->uncurry_args<6>();
    auto [count, stride]                                = blocks->projs<2>();
    auto step                                           = app->arg()->isa<Lam>();
    if (!step || !step->is_set()) app->blame("tensor.tile_fold step must be a defined lambda").bail();
    auto parameter             = old.mut_lam(old.pi(old.sigma({init->type(), old.type_nat()}), old.type_nat()));
    auto [accumulator, offset] = parameter->var()->projs<2>();
    auto input                 = region_input(inputs);
    auto root                  = old.app(step->reduce(parameter->var()).back(), input);
    auto result                = root;
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
        if (Axm::isa<tensor::tiled>(call)) shape = call->uncurry_args<5>()[1]->proj(2, 0);
        if (Axm::isa<tensor::map_reduce_post>(call)) shape = btensor::MapRedPost::of(call->callee()->as<App>()).So;
    }
    auto rank    = Lit::as(shape->arity());
    auto zero    = old.tuple(DefVec(rank, [&](size_t) { return old.lit_nat(0); }));
    auto inplace = linear_step(root, accumulator, zero);
    if (!inplace)
        log().w("tensor destination fallback: tile_fold step copies accumulator (linear pointwise use unproved)");

    auto literal_count = Lit::isa(count);
    auto first_scalar  = inplace && literal_count && *literal_count != 0 && splat_scalar(init);
    auto [mem, buffer] = buffer::op_alloc(w.lit_nat(rank), rewrite(shape), rewrite(element), fresh_mem())->projs<2>();
    if (first_scalar) {
        auto first_root = old.app(step->reduce(old.tuple({init, old.lit_nat(0)})).back(), input);

        auto [first, first_call]
            = btensor::ds_splice(w.tuple({mem, buffer}), w.sigma({mem->type(), buffer->type()}), "foldFirst");
        auto [first_mem, first_buffer] = first->var(2, 0)->projs<2>();
        auto pending                   = fe::Restore(pending_, {});
        auto memo                      = fe::Restore(fresh_memo_, {});
        auto cache                     = fe::Restore(realizations_, realizations_);
        push();
        bind_new(first->has_var());
        prepare_realizations(first_root);
        first->app(true, first->ret_var(),
                   lower_into(first_root, {first_mem, first_buffer, rewrite(shape), rewrite(zero), rewrite(shape)}));
        first->set(true, wrap_fresh_mem(first->body()));
        pop();
        std::tie(mem, buffer) = first_call->projs<2>();
    } else {
        auto initial = make_buffer(init, w.lit_nat(rank), rewrite(shape));
        mem          = copy_buffer(mem, buffer, initial, rewrite(shape));
    }
    auto [run, call] = btensor::ds_splice(w.tuple({mem, buffer}), w.sigma({mem->type(), buffer->type()}), "tileFold");
    auto state       = run->var(2, 0);
    auto remaining   = first_scalar ? w.lit_nat(*literal_count - 1) : rewrite(count);
    auto [loop, loop_call] = affine::counting_for(w.call<core::bitcast>(w.type_i64(), remaining), state, run->ret_var(),
                                                  w.sym("tileFoldStep"));
    auto [index, carried, yield] = loop->vars<3>();
    auto [loop_mem, loop_buffer] = carried->projs<2>();
    auto pending                 = fe::Restore(pending_, {});
    auto memo                    = fe::Restore(fresh_memo_, {});
    auto cache                   = fe::Restore(realizations_, realizations_);
    push();
    bind_old(parameter->has_var());
    map(accumulator, loop_buffer);
    auto iteration = w.call<core::bitcast>(w.type_nat(), index);
    if (first_scalar) iteration = w.call(core::nat::add, Defs{iteration, w.lit_nat(1)});
    map(offset, w.call(core::nat::mul, Defs{iteration, rewrite(stride)}));
    prepare_realizations(root);
    const Def* update;
    if (inplace) {
        update = lower_into(root, {loop_mem, loop_buffer, rewrite(shape), rewrite(zero), rewrite(shape)});
    } else {
        auto value = make_buffer(root, w.lit_nat(rank), rewrite(shape));
        update     = w.tuple({copy_buffer(loop_mem, loop_buffer, value, rewrite(shape)), loop_buffer});
    }
    loop->app(true, yield, update);
    loop->set(true, wrap_fresh_mem(loop->body()));
    pop();
    run->set(true, loop_call);
    auto [out_mem, out] = call->projs<2>();
    return scalar_result(app, out_mem, out);
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
    DefVec primary, spare;
    fe::Vector<bool> pooled;
    auto initialize = [&](const Def* t, const Def* value) -> const Def* {
        if (!TensorType::isa(t)) {
            auto scalar = rewrite(value);
            primary.push_back(scalar);
            spare.push_back(scalar);
            pooled.push_back(false);
            return scalar;
        }
        auto [shape, element] = shape_of(t);
        auto rank             = w.lit_nat(Lit::as(shape->arity()));
        auto owned            = scratch_ && lit_projs(rewrite(shape), Lit::as(rank)) && rewrite(element)->is_closed();
        const Def *current, *next;
        if (owned) {
            current = allocate(rank, rewrite(shape), rewrite(element)).second;
            next    = allocate(rank, rewrite(shape), rewrite(element)).second;
        } else {
            std::tie(mem, current) = buffer::op_alloc(rank, rewrite(shape), rewrite(element), mem)->projs<2>();
            std::tie(mem, next)    = buffer::op_alloc(rank, rewrite(shape), rewrite(element), mem)->projs<2>();
        }
        pooled.push_back(owned);
        primary.push_back(current);
        spare.push_back(next);
        mem = copy_buffer(mem, current, make_buffer(value, rank, rewrite(shape)), rewrite(shape));
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
    {
        auto pending = fe::Restore(pending_, {});
        auto memo    = fe::Restore(fresh_memo_, {});
        auto cache   = fe::Restore(realizations_, realizations_);
        push();
        bind_old(parameter->has_var());
        map(accumulator, source);
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
                    shape = operation->uncurry_args<5>()[1]->proj(2, 0);
            }
            auto rank    = Lit::as(shape->arity());
            auto zero    = w.tuple(DefVec(rank, [&](size_t) { return w.lit_nat(0); }));
            auto written = lower_into(value, {updated_mem, target, rewrite(shape), zero, rewrite(shape)});
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
        released                   = copy_buffer(released, target, value, shape);
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
    return finished->proj(2, 1);
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
