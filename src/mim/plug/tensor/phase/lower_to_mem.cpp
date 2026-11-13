#include "mim/plug/tensor/phase/lower_to_mem.h"

#include <algorithm>
#include <bit>
#include <numeric>

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
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>

#include "mim/plug/gpu/gpu.h"
#include "mim/plug/tensor/phase/constraints.h"
#include "mim/plug/tensor/phase/group.h"
#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

const Def* group_phase(World& w, nat_t phase) { return w.inj(w.annex<gpu::GroupPhase>(), phase, w.tuple()); }

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

/// Is `app` a tensor op whose lowering consumes a LowerToMem::fresh_mem?
bool wants_fresh_mem(const App* app) {
    return Axm::isa<tensor::tile_of>(app) || Axm::isa<tensor::tile_of_box>(app) || Axm::isa<tensor::slice>(app)
        || Axm::isa<tensor::slice_box>(app) || Axm::isa<tensor::stage>(app) || Axm::isa<tensor::placed>(app)
        || Axm::isa<tensor::tile_fold>(app) || Axm::isa<tensor::tiled>(app) || Axm::isa<tensor::tiled_multi>(app)
        || Axm::isa<tensor::tiled_n>(app) || Axm::isa<tensor::generate>(app) || Axm::isa<tensor::broadcast>(app)
        || Axm::isa<tensor::map_reduce_post>(app) || Axm::isa<tensor::pad>(app) || Axm::isa<tensor::concat>(app)
        || Axm::isa<tensor::gather>(app) || Axm::isa<tensor::scatter>(app) || Axm::isa<tensor::set>(app)
        || Axm::isa<tensor::from_array>(app) || Axm::isa<tensor::scalar>(app) || Axm::isa<tensor::splat>(app)
        || Axm::isa<tensor::prepack>(app);
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
            if (positive(shapes->proj(3, 0))) region(body);
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
        // A placed stage is realized with its layout, so only the annotation is a request.
        auto placed = app ? Axm::isa<tensor::placed>(app) : nullptr;
        for (auto op : def->ops())
            if (op && placed && op == placed->arg())
                for (auto inner : op->ops())
                    scan(inner, site);
            else if (op)
                scan(op, site);
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

using Annotate = std::function<const Def*(const Def*)>;

const Def* copy_buffer(const Def* mem, const Def* dst, const Def* src, const Def* shape, Annotate annotate = {}) {
    auto& w     = mem->world();
    auto result = btensor::build_pointwise(
        w.sigma({mem->type(), dst->type()}), mem, dst, src, shape, Lit::as(shape->arity()), "foldCopy",
        [&](Defs iters, const Def* input, const Def* current) {
            auto [next, value] = buffer::op_read(current, input, btensor::folded_idx(shape, iters))->projs<2>();
            return std::pair{next, value};
        },
        nullptr, annotate);
    return result->proj(2, 0);
}

// Runtime inner extents bounded by a capped tile allocate their bound, since buffers need literal inner extents.
// A kernel's slots are static, so there the outer extent takes its bound as well.
const Def* bounded_shape(const Def* shape, bool outer = false) {
    auto& w = shape->world();
    auto r  = Lit::as(shape->arity());
    return w.tuple(DefVec(r, [&](size_t d) -> const Def* {
        auto extent = shape->proj(r, d);
        if ((d == 0 && !outer) || Lit::isa(extent)) return extent;
        if (auto upper = nat_upper(extent)) return w.lit_nat(*upper);
        return extent;
    }));
}

// Work-items walking a work-group's tile down a column would all hit one shared-memory bank, so its rows are padded.
// Sixteen bytes of padding keep the rows aligned for vector accesses.
const Def* padded_rows(const Def* storage) {
    auto& w = storage->world();
    auto r  = Lit::as(storage->arity());
    if (r < 2) return storage;
    auto last = Lit::isa(storage->proj(r, r - 1));
    if (!last || *last % 32) return storage;
    return w.tuple(DefVec(r, [&](size_t d) { return d + 1 == r ? w.lit_nat(*last + 4) : storage->proj(r, d); }));
}

/// @p storage with the row length of @p buffer, which may be padded.
const Def* padded_like(const Def* storage, const Def* buffer) {
    auto& w = storage->world();
    auto r  = Lit::as(storage->arity());
    if (auto buf = Axm::isa<buffer::Buf>(buffer->type()); buf && r >= 2 && Lit::isa(buf->arg(3, 0)) == r)
        if (auto row = buf->arg(3, 1)->proj(r, r - 1); Lit::isa(row) && row != storage->proj(r, r - 1))
            return w.tuple(DefVec(r, [&](size_t d) { return d + 1 == r ? row : storage->proj(r, d); }));
    return storage;
}

/// Copies the @p extent box of @p src, or splats it if it is no buffer, between buffers of different storage shapes.
const Def* copy_box(const Def* mem,
                    const Def* dst,
                    const Def* dst_shape,
                    const Def* src,
                    const Def* src_shape,
                    const Def* extent,
                    Annotate annotate = {}) {
    auto& w = mem->world();
    auto r  = Lit::as(extent->arity());
    return btensor::build_pointwise(
               w.sigma({mem->type(), dst->type()}), mem, dst, src, extent, r, "foldCopy",
               [&](Defs iters, const Def* input, const Def* current) {
                   if (!Axm::isa<buffer::Buf>(input->type())) return std::pair{current, input};
                   auto [next, value]
                       = buffer::op_read(current, input, btensor::folded_idx(src_shape, iters))->projs<2>();
                   return std::pair{next, value};
               },
               dst_shape, annotate)
        ->proj(2, 0);
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
            if (!(Axm::isa<tensor::map_reduce_post>(app) || Axm::isa<tensor::stage>(app)
                  || Axm::isa<tensor::placed>(app) || Axm::isa<tensor::slice>(app) || Axm::isa<tensor::slice_box>(app)
                  || Axm::isa<tensor::splat>(app) || Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::generate>(app)
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
            // A flat array element, e.g. an online softmax's `(max, sum)` pair, is a plain struct in memory.
            auto arr = T->isa<Arr>();
            if ((arr && (arr->body()->isa<Arr>() || !Lit::isa(arr->arity()))) || mentions_tensor(T))
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
    tf32_rna_   = driver().arg_nat("tensor", "tf32-rna").value_or(0) != 0;
    validate();
    RWPhase::start();
    assert(pending_.empty());
}

const Def* LowerToMem::buf_of(const Def* type) {
    auto [r, s, T] = TensorType::decode(type->as<Nom>());
    check_buffer_shape(r, s);
    return buffer::type_buf(rewrite(r), rewrite(s), rewrite(T));
}

// Every tile ends within its region, so clamping its end (or last index) to the region's yields the end itself.
const Def* LowerToMem::rewrite_imm_Extract(const Extract* extract) {
    auto result = RWPhase::rewrite_imm_Extract(extract);
    if (auto clamp = result->isa<Extract>())
        if (auto pair = clamp->tuple()->isa<Tuple>(); pair && pair->num_ops() == 2)
            for (auto [end, bound] : tile_ends_)
                if (pair->op(0) == bound && pair->op(1) == end
                    && clamp->index() == new_world().call(core::ncmp::l, Defs{end, bound}))
                    return end;
    return result;
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

bool LowerToMem::uniform(const Def* value) const {
    if (bound_level_ == 1) return true;
    return bound_level_ == 2 && item_origin_ && !value->has_free_var(item_origin_);
}

// A work-group's uniform fills run once, spread over its work-items by `gpu.outline`.
std::function<const Def*(const Def*)> LowerToMem::cooperative() {
    if (!uniform_) return {};
    auto tag = async_fill_ ? gpu::bind::async : gpu::bind::cooperative;
    return
        [&w = new_world(), tag](const Def* body) { return w.app(w.app(w.annex<gpu::bind>(tag), body->type()), body); };
}

// A consumer inside a work-group fills the panels it reads as one batch, starting from its own memory or, without
// one, from a fresh memory at the first fill.
std::optional<LowerToMem::Batch> LowerToMem::open_batch(const Def* start) {
    if (bound_level_ == 0 || !driver().is_loaded("gpu")) return {};
    return Batch{start, batches_++};
}

// Publishes the batch's panels behind one barrier; @p consume computes `[mem, result]` while they are read.
const Def*
LowerToMem::close_batch(const Def* mem, const Def* result, const std::function<const Def*(const Def*)>& consume) {
    if (!batch_ || !batch_->token) return consume(mem);
    auto& w        = new_world();
    auto g         = w.lit_nat(batch_->group);
    auto e         = w.lit_nat_0();
    auto published = w.app(w.app(w.annex<gpu::sync_publish>(), {g, e}), batch_->token);
    batch_->published->set(published);
    auto M      = w.call<mem::M>(0);
    auto reader = w.mut_lam(w.pi(M, w.sigma({M, result})))->set("batchRead");
    reader->set(true, consume(reader->var()));
    --entered_;
    auto read = w.app(w.app(w.app(w.annex<gpu::read_phase>(), {g, e}), result), w.tuple({published, reader}));
    return w.tuple({w.app(w.app(w.annex<gpu::leave_reading>(), {g, e}), read->proj(2, 0)), read->proj(2, 1)});
}

// A work-group's uniform buffers live in shared memory.
const Def* LowerToMem::shared_mem(const Def* mem) {
    if (!uniform_) return mem;
    return new_world().app(new_world().annex<gpu::shared_scope>(), mem);
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
    auto _            = fe::Restore(pending_, {});
    auto nested_batch = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
    auto __           = fe::Restore(fresh_memo_, {});
    auto storage      = fe::Restore(storage_shapes_, storage_shapes_);
    auto cache        = fe::Restore(realizations_, realizations_);
    auto scratch      = fe::Restore<Scratch*>(scratch_, nullptr);
    auto nested       = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
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

// Escape analysis loses runtime-shaped buffers at control joins, so a tile body frees its own.
void LowerToMem::transient(const Def* rank, const Def* shape, const Def* type, const Def* buffer) {
    auto r = Lit::isa(rank);
    if (!transients_ || (r && lit_projs(shape, *r))) return;
    transients_->push_back({rank, shape, type, buffer});
}

std::pair<const Def*, const Def*>
LowerToMem::allocate(const Def* rank, const Def* shape, const Def* type, bool fills, const Def* mem) {
    check_buffer_shape(rank, shape);
    auto r       = Lit::isa(rank);
    auto literal = r.has_value();
    if (r)
        for (size_t d = 0; d != *r; ++d)
            literal &= Lit::isa(shape->proj(*r, d)).has_value();
    // A work-group's shared buffer becomes a static slot, which a scratch binding could not free.
    if (!scratch_ || !fills || !literal || !type->is_closed() || uniform_) {
        auto [after, buffer] = buffer::op_alloc(rank, shape, type, shared_mem(mem ? mem : fresh_mem()))->projs<2>();
        if (!literal) transient(rank, shape, type, buffer);
        return {after, buffer};
    }
    auto allocation        = buffer::op_alloc(rank, shape, type, shared_mem(scratch_->mem));
    auto [run, call]       = btensor::ds_splice(allocation, scratch_->result_type, "tileScratch");
    auto [scratch, buffer] = run->var(2, 0)->projs<2>();
    bind_new(run->has_var());
    scratch_->mem = scratch;
    scratch_->bindings.emplace_back(run, call);
    return {mem ? mem : fresh_mem(), buffer};
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
    if (Axm::isa<tensor::mma>(app) || Axm::isa<tensor::pipeline>(app)) return rewrite(app->arg());
    if (Axm::isa<tensor::prepack>(app)) return lower_prepack(app);
    if (Axm::isa<tensor::group_req>(app)) {
        report_group_request(app, "request reached bufferization without realization");
        return rewrite(app->arg(5, 0));
    }
    if (Axm::isa<tensor::tile_req>(app)) {
        report_tile_request(app, "request reached bufferization without realization");
        return rewrite(app->arg(4, 0));
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
    if (Axm::isa<tensor::get>(app) || Axm::isa<tensor::get_fused>(app)) return lower_get(app);
    if (Axm::isa<tensor::as_vectors>(app)) return lower_as_vectors(app);
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
    if (auto placed = Axm::isa<tensor::placed>(app)) {
        if (Axm::isa<tensor::stage>(placed->arg())) placed_layouts_[placed->arg()] = rewrite(placed->decurry()->arg());
        auto value = rewrite(placed->arg());
        if (auto shape = fe::lookup(storage_shapes_, placed->arg())) storage_shapes_[app] = shape;
        return value;
    }
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
                   [](Defs, const Def* value, const Def* current) { return std::pair{current, value}; }, nullptr,
                   cooperative())
            ->proj(2, 1);
    }
    auto [m, out] = buffer::op_lit(bro, bso, boT, fresh_mem(), scalar)->projs<2>();
    transient(bro, bso, boT, out);
    return out;
}

const Def* LowerToMem::lower_from_array(const App* app) {
    auto [r, s, T] = TensorType::decode(app->type()->as<Nom>());
    auto scope     = fe::Restore(uniform_, uniform(app));
    if (auto value = splat_scalar(app->arg()); value && value->type() == T)
        return splat_buffer(app->type(), rewrite(value));
    auto [br, bs, bT] = Axm::isa<buffer::Buf>(buf_of(app->type()))->args<3>();
    return buffer::op_init(br, bs, bT, fresh_mem(), rewrite(app->arg()))->proj(2, 1);
}

// The logical @p shape with each extent the buffer stores, all but literal unit dims, replaced by its storage bound.
const Def* buffer_bounds(const Def* buffer, const Def* shape, u64 rank) {
    auto buf = Axm::isa<buffer::Buf>(buffer->type());
    if (!buf) return shape;
    auto stored = Lit::isa(buf->arg(3, 0));
    auto dims   = DefVec(shape->projs(rank));
    auto kept   = u64(std::ranges::count_if(dims, [](const Def* d) { return Lit::isa(d) != 1; }));
    if (!stored || (*stored != rank && kept != *stored)) return shape;
    auto bounds = buf->arg(3, 1);
    for (size_t d = 0, b = 0; d != rank; ++d)
        if (*stored == rank || Lit::isa(dims[d]) != 1) dims[d] = bounds->proj(*stored, b++);
    return shape->world().tuple(dims);
}

// The buffer holding @p old and @p index retyped to its storage shape.
// A `tensor.as_vectors` view is accessed through the buffer of the tensor it views: the run's first element goes
// through that buffer's layout and the physical run is read from the vectorized buffer.
std::pair<const Def*, const Def*> LowerToMem::storage_access(const Def* old, const Def* index) {
    auto& w      = new_world();
    const Def* n = nullptr;
    if (auto vectorized = Axm::isa<tensor::as_vectors>(old)) {
        n   = rewrite(vectorized->callee()->as<App>()->arg(2, 0));
        old = vectorized->arg();
    }
    auto [r, s, T] = TensorType::decode(old->type()->as<Nom>());
    auto source    = rewrite(old);
    auto shape     = fe::lookup(storage_shapes_, old);
    if (!shape) shape = buffer_bounds(source, rewrite(s), Lit::as(r));
    auto rank        = Lit::as(r);
    auto coordinates = rewrite(index);
    auto nats        = DefVec(rank, [&](size_t d) { return w.call<core::idx2nat>(coordinates->proj(rank, d)); });
    if (n) nats.back() = w.call(core::nat::mul, Defs{nats.back(), n});
    if (auto layout = fe::lookup(layouts_, source)) {
        auto i64s     = DefVec(rank, [&](size_t d) { return w.call<core::bitcast>(w.type_i64(), nats[d]); });
        auto physical = layout_coords(layout, logical_shapes_[source], i64s);
        nats = DefVec(physical.size(), [&](size_t p) { return w.call<core::bitcast>(w.type_nat(), physical[p]); });
    }
    if (n) {
        source      = buffer::op_vectorize(n, source);
        auto runs   = DefVec(shape->projs(nats.size()));
        runs.back() = w.call(core::nat::div, Defs{runs.back(), n});
        shape       = w.tuple(runs);
        nats.back() = w.call(core::nat::div, Defs{nats.back(), n});
    }
    auto m = nats.size();
    auto adjusted
        = w.tuple(DefVec(m, [&](size_t p) { return w.call<core::idx>(w.lit_nat_0(), shape->proj(m, p), nats[p]); }));
    return {source, *Shape(adjusted).fold(shape)};
}

const Def* LowerToMem::lower_get(const App* app) {
    auto [old, index]  = app->args<2>();
    auto [source, idx] = storage_access(old, index);
    return buffer::op_read(fresh_mem(), source, idx)->proj(2, 1);
}

const Def* LowerToMem::lower_as_vectors(const App* app) {
    auto& w     = new_world();
    auto source = rewrite(app->arg());
    plain(source);
    auto n = rewrite(app->callee()->as<App>()->arg(2, 0));
    if (auto storage = fe::lookup(storage_shapes_, app->arg())) {
        auto runs            = DefVec(storage->projs(storage->num_projs()));
        runs.back()          = w.call(core::nat::div, Defs{runs.back(), n});
        storage_shapes_[app] = w.tuple(runs);
    }
    return buffer::op_vectorize(n, source);
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
    auto value     = rewrite(app->arg()->proj(2, 1));
    auto [r, s, T] = TensorType::decode(app->type()->as<Nom>());
    auto shape     = rewrite(s);
    auto storage   = bounded_shape(shape);
    if (storage == shape) return splat_buffer(app->type(), value);
    storage_shapes_[app] = storage;
    auto& w              = new_world();
    auto rank = rewrite(r), element = rewrite(T);
    if (!scratch_) {
        auto buffer = buffer::op_lit(rank, storage, element, fresh_mem(), value)->proj(2, 1);
        transient(rank, storage, element, buffer);
        return buffer;
    }
    auto [mem, buffer] = allocate(rank, storage, element);
    return btensor::build_pointwise(
               w.sigma({mem->type(), buffer->type()}), mem, buffer, value, storage, Lit::as(rank), "scratchSplat",
               [](Defs, const Def* value, const Def* current) { return std::pair{current, value}; }, nullptr,
               cooperative())
        ->proj(2, 1);
}

const Def* LowerToMem::lower_generate(const App* app) {
    auto& w            = new_world();
    auto scope         = fe::Restore(uniform_, uniform(app));
    auto [meta, s_out] = rewrite(app->callee())->as<App>()->uncurry_args<2>();
    auto [T, r]        = meta->projs<2>();
    auto r_l           = Lit::isa<u64>(r);
    auto rn            = *r_l;

    // A vector fill loops over chunks of `vec` elements along dim `inner`; its buffer holds the elements.
    auto inner   = fe::lookup(fill_inner_, app);
    auto found   = fe::lookup(fill_vector_, app);
    auto vec     = found ? *found : 1;
    auto logical = s_out;
    if (vec != 1) {
        auto dims    = DefVec(s_out->projs(rn));
        dims[*inner] = w.call(core::nat::mul, Defs{dims[*inner], w.lit_nat(vec)});
        logical      = w.tuple(dims);
        T            = T->as<Arr>()->body();
    }
    auto storage      = bounded_shape(logical, bound_level_ != 0);
    auto layout       = static_cast<const Def*>(nullptr);
    auto layout_view  = std::optional<Layout>();
    auto layout_shape = storage;
    auto vector_store = inner && *inner + 1 == rn;
    // TileFuse chose a work-group panel's storage layout; its fill and reads follow it.
    auto size = Lit::isa(core::op(core::trait::size, T));
    if (auto chosen = fe::lookup(fill_layouts_, app); chosen && size && inner && uniform_) {
        auto [cap, expr] = chosen->projs<2>();
        if (cap != storage) app->blame("a placed panel's capacity {} differs from its storage {}", cap, storage).bail();
        auto choice  = panel::decode(expr, storage, *inner, vec, *size);
        storage      = choice.physical;
        vector_store = mma_panels_ || choice.contiguous;
        layout = choice.expr, layout_view = choice.layout;
    }
    // The warps of a warp-owned region read a shared panel across its rows, so its rows are padded by 16 bytes.
    if (auto n = Lit::isa(r); warp_self_ && uniform_ && !layout && size && n && *n >= 2 && 16 % *size == 0)
        if (auto last = Lit::isa(storage->proj(*n, *n - 1)); last && *last * *size % 128 == 0) {
            auto dims   = DefVec(storage->projs(*n));
            dims.back() = w.lit_nat(*last + 16 / *size);
            storage     = w.tuple(dims);
        }
    auto bounded = storage != logical;
    check_buffer_shape(r, storage);
    auto out_ty       = buffer::type_buf(r, storage, T);
    auto [br, bs, bT] = Axm::isa<buffer::Buf>(out_ty)->args<3>();
    auto result_ty    = w.sigma({w.call<mem::M>(0), out_ty});
    if (bounded) storage_shapes_[app] = storage;
    auto [sources, captures] = capture_producers(app->arg());
    auto compute             = [&](Defs iters, const Def* input, const Def* m) -> std::pair<const Def*, const Def*> {
        auto nested = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
        push();
        if (!sources.empty()) bind_new(param_of(input)->has_var());
        for (size_t i = 0; i != sources.size(); ++i)
            map(sources[i], input->proj(sources.size(), i));
        auto value = w.call(rewrite(app->arg()), iters);
        pop();
        return {m, value};
    };
    auto physical = [this, layout_shape, layout](Defs iters) {
        if (!layout) return DefVec(iters.begin(), iters.end());
        return layout_coords(layout, layout_shape, iters);
    };
    auto index = [&physical, storage](Defs iters) { return btensor::folded_idx(storage, physical(iters)); };
    auto write = std::function<const Def*(const Def*, const Def*, Defs, const Def*)>();
    if (vec != 1 || layout)
        write = [&w, &index, &physical, storage, vec, inner = inner ? *inner : 0,
                 vector_store](const Def* m, const Def* buf, Defs iters, const Def* value) {
            auto first = DefVec(iters.begin(), iters.end());
            if (vec == 1) return buffer::op_write(m, buf, index(first), value)->proj(2, 0);
            first[inner] = w.call(core::wrap::mul, core::Mode::none, Defs{iters[inner], w.lit_i64(vec)});
            if (vector_store) {
                auto vectors = buffer::op_vectorize(w.lit_nat(vec), buf);
                auto runs    = DefVec(storage->projs(storage->num_projs()));
                runs.back()  = w.call(core::nat::div, Defs{runs.back(), w.lit_nat(vec)});
                auto run     = physical(first);
                run.back()   = w.call(core::shr::l, Defs{run.back(), w.lit_i64(std::countr_zero(vec))});
                return buffer::op_write(m, vectors, btensor::folded_idx(w.tuple(runs), run), value)->proj(2, 0);
            }
            for (u64 v = 0; v != vec; ++v) {
                auto at   = first;
                at[inner] = w.call(core::wrap::add, core::Mode::none, Defs{first[inner], w.lit_i64(v)});
                m         = buffer::op_write(m, buf, index(at), value->proj(vec, v))->proj(2, 0);
            }
            return m;
        };
    auto target = std::exchange(fill_target_, std::nullopt);
    if (target && std::get<1>(*target)->type() != out_ty) app->blame("a pipelined panel changed its type").bail();
    const Def* filled = nullptr;
    if (batch_ && uniform_ && !target && !probe_fill_ && !async_fill_) {
        // The panel joins its consumer's batch: filled cooperatively without a barrier, read once the batch publishes.
        auto g = w.lit_nat(batch_->group);
        auto e = w.lit_nat_0();
        if (!batch_->token) {
            if (entered_++) app->blame("a work-group batch cannot open while another is open").bail();
            batch_->token
                = w.app(w.app(w.annex<gpu::enter_write>(), {g, e}),
                        w.app(w.app(w.annex<gpu::group_enter>(), g), batch_->start ? batch_->start : fresh_mem()));
            batch_->published
                = w.mut_hole(w.app(w.annex<gpu::GM>(), {g, e, group_phase(w, gpu::GroupPhase_ctor::Reading)}));
        }
        auto panel    = w.app(w.app(w.app(w.annex<gpu::shared_panel>(), {g, e}), {br, bs, bT}), batch_->token);
        auto fill     = w.mut_lam(w.pi(result_ty, result_ty))->set("panelFill");
        auto [fm, fb] = fill->var()->projs<2>();
        auto chained  = [&w](const Def* body) {
            return w.app(w.app(w.annex<gpu::bind>(gpu::bind::chained), body->type()), body);
        };
        fill->set(true,
                  btensor::build_pointwise(result_ty, fm, fb, w.tuple(captures), s_out, rn, "generate", compute,
                                           bounded ? storage : nullptr, chained, inner ? *inner : size_t(-1), write));
        auto filling      = w.app(w.app(w.annex<gpu::coop_fill>(), {g, e}), {br, bs, bT});
        auto [token, buf] = w.app(filling, w.tuple({panel->proj(2, 0), panel->proj(2, 1), fill}))->projs<2>();
        batch_->token     = token;
        auto publish      = w.app(w.app(w.annex<gpu::publish_view>(), {g, e}), {br, bs, bT});
        auto view         = w.app(w.app(w.annex<gpu::view>(), {g, e}), {br, bs, bT});
        filled            = w.app(view, w.app(publish, w.tuple({batch_->published, buf})));
    }
    if (!filled) {
        auto [am, dst] = target ? *target : allocate(br, bs, bT);
        filled         = dst;
        if (!probe_fill_) {
            auto fill = btensor::build_pointwise(result_ty, am, dst, w.tuple(captures), s_out, rn, "generate", compute,
                                                 bounded ? storage : nullptr, cooperative(),
                                                 inner ? *inner : size_t(-1), write);
            filled    = fill->proj(2, 1);
            if (target) filled_mem_ = fill->proj(2, 0);
        }
    }
    if (uniform_) shared_fills_.insert(filled);
    if (layout) {
        layouts_[filled]        = layout;
        layout_views_[filled]   = *layout_view;
        logical_shapes_[filled] = layout_shape;
    }
    if (inner && uniform_) panels_.insert(filled);
    return filled;
}

bool LowerToMem::nat_multiple(const Def* def, u64 v) const {
    if (v == 1) return true;
    if (auto lit = Lit::isa(def)) return *lit % v == 0;
    if (auto nat = Axm::isa<core::nat>(def)) {
        auto [a, b] = nat->args<2>();
        if (nat.id() == core::nat::add) return nat_multiple(a, v) && nat_multiple(b, v);
        if (nat.id() == core::nat::mul) return nat_multiple(a, v) || nat_multiple(b, v);
        return false;
    }
    if (auto multiple = fe::lookup(multiples_, def)) return *multiple % v == 0;
    return false;
}

// The 16-byte vector width a fill of `type` can move along a source dim with `extent`, `start`, `step`,
// row stride `row`, and whose other address terms are multiples of `align`, or 1.
u64 LowerToMem::vector_width(const Def* type,
                             const Def* extent,
                             const Def* start,
                             const Def* step,
                             const Def* row,
                             u64 align) const {
    auto size = Lit::isa(core::op(core::trait::size, type));
    if (!size || *size == 0 || 16 % *size != 0 || 16 / *size < 2) return 1;
    auto v   = 16 / *size;
    auto ext = Lit::isa(extent), st = Lit::isa(step), rw = Lit::isa(row);
    if (!ext || *ext % v || !st || *st != 1 || !rw || *rw % v || align % v || !nat_multiple(start, v)) return 1;
    return v;
}

namespace {
const Def* lane_index(World& w) {
    auto lane = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::lane_id>(), w.tuple()));
    return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(32), lane);
}

/// The warp loop's lane as an index; `tensor.distribute` takes the same term for the own lane.
const Def* warp_lane(World& w) {
    auto lane = w.call<core::bitcast>(w.type_nat(), w.app(w.annex<gpu::lane_id>(), w.tuple()));
    return w.call<core::idx>(w.lit_nat_0(), w.lit_nat(32), lane);
}

/// The loop each coordinate of @p map reads, unscaled and up to an offset free of the loops.
std::optional<std::vector<u64>> map_dims(const Def* map, u64 rn) {
    auto& w = map->world();
    auto pi = map->type()->isa<Pi>();
    if (!pi) return {};
    auto var = w.mut_lam(pi)->var();
    auto out = w.app(map, var);
    auto ri  = Lit::isa(out->type()->arity());
    if (!ri) return {};
    auto dims = std::vector<u64>();
    for (u64 d = 0; d != *ri; ++d) {
        auto form = Linear::of(out->proj(*ri, d));
        auto loop = rn;
        for (u64 l = 0; l != rn; ++l)
            if (auto atom = form.atoms.find(var->proj(rn, l)); atom != form.atoms.end()) {
                if (atom->second != 1 || loop != rn) return {};
                loop = l;
            }
        if (loop == rn) return {};
        dims.push_back(loop);
    }
    return dims;
}

/// The `(row, col)` that a `tensor.mma_m16n8k8_tf32` coordinate map assigns to `arg`, as Nats.
std::array<const Def*, 2> fragment_coords(World& w, Enum auto map, const Def* arg) {
    auto [row, col]
        = w.app(w.annex(map), arg)->template projs<2>([&](const Def* x) { return w.call<core::idx2nat>(x); });
    return {row, col};
}

} // namespace

DefVec LowerToMem::layout_coords(const Def* layout, const Def* logical, Defs coords) {
    auto& w  = new_world();
    auto n   = coords.size();
    auto map = panel::at_shape(w.annex(tensor::layout_expr::map_i64), layout, logical);
    return w.app(map, w.tuple(coords))->projs(n);
}

const Def* LowerToMem::layout_lam(const Def* layout, const Def* logical) {
    auto& w = new_world();
    return panel::at_shape(w.annex(tensor::layout_expr::map_affine), layout, logical);
}

void LowerToMem::plain(const Def* buffer) const {
    if (fe::lookup(layouts_, buffer)) buffer->blame("a panel in a storage layout can only be read element-wise").bail();
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
            while (auto read = panel::tensor_read(value, shape)) {
                if (!TensorType::isa(read->src->type())) break;
                value   = read->src;
                rank    = read->R;
                shape   = read->S;
                map     = panel::compose_read_map(read->map, map);
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
        while (auto read = panel::tensor_read(value, shape)) {
            if (read->R != old_mr.Ro || !TensorType::isa(read->src->type())) break;
            value   = read->src;
            shape   = read->S;
            map     = panel::compose_read_map(read->map, map);
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
    auto initial = tensor_init(old_mr);
    auto batch   = fe::Restore<std::optional<Batch>>(batch_, open_batch(destination ? destination->mem : nullptr));
    auto args    = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
    auto [is, post_is] = args->projs<2>();

    auto mr = btensor::MapRedPost::decode(
        rewrite(old_mr.nis_nps), rewrite(old_mr.meta), rewrite(old_mr.shapes), rewrite(old_mr.in_tys),
        w.tuple({rewrite(old_mr.comb), w.bot(rewrite(old_mr.To)), rewrite(old_mr.post)}), rewrite(old_mr.map_out),
        rewrite(old_mr.maps_all));
    if (auto clipped = Axm::isa<btensor::clip_sched>(mr.sched)) {
        auto [valid, output_cap, loop_cap, schedule] = clipped->args<4>();
        auto ro = Lit::as(mr.Ro), rn = Lit::as(mr.Rn);
        // A cap fixed while the tile extent was still a variable may bound the specialized extent instead.
        // The guard covers parallel loops only, so reduction loops run their exact extent.
        auto bounded = [&](const Def* cap, const Def* extent, u64 n) {
            return w.tuple(DefVec(n, [&](size_t d) -> const Def* {
                if (d >= ro || Lit::isa(extent->proj(n, d))) return extent->proj(n, d);
                if (Lit::isa(cap->proj(n, d))) return cap->proj(n, d);
                if (auto upper = nat_upper(cap->proj(n, d))) return w.lit_nat(*upper);
                if (auto upper = nat_upper(extent->proj(n, d))) return w.lit_nat(*upper);
                return cap->proj(n, d);
            }));
        };
        output_cap        = bounded(output_cap, mr.So, ro);
        loop_cap          = bounded(loop_cap, mr.Sr, rn);
        auto parallel_cap = w.tuple(DefVec(ro, [&](size_t d) { return loop_cap->proj(rn, d); }));
        // Buffers take a runtime outermost extent, so only inner caps must be literal.
        auto inner_literal = [&](const Def* caps) {
            for (u64 d = 1; d < ro; ++d)
                if (!Lit::isa(caps->proj(ro, d))) return false;
            return true;
        };
        if (!inner_literal(output_cap) || !inner_literal(parallel_cap))
            app->blame("tensor tile storage and loop caps must be literal: {}, {} for {}", output_cap, parallel_cap,
                       mr.So)
                .bail();
        // Caps that met specialized extents leave a guard that always holds.
        if (output_cap == mr.So && loop_cap == mr.Sr) {
            mr.sched  = schedule;
            mr.shapes = w.tuple({mr.So, mr.Sr, mr.sched});
        } else {
            mr.So                = output_cap;
            mr.Sr                = loop_cap;
            mr.shapes            = w.tuple({mr.So, mr.Sr, mr.sched});
            storage_shapes_[app] = output_cap;
        }
    }
    if (auto storage = bounded_shape(mr.So); !destination && storage != mr.So) {
        mr.So                = storage;
        mr.shapes            = w.tuple({mr.So, mr.Sr, mr.sched});
        storage_shapes_[app] = storage;
    }
    const Def* seed = nullptr;
    if (initial) {
        if (auto scalar = splat_scalar(initial->outs)) {
            seed = rewrite(scalar);
        } else {
            auto outs = to_buffer(rewrite(initial->outs), initial->outs);
            plain(outs);
            auto shape = fe::lookup(storage_shapes_, initial->outs);
            seed       = w.tuple({shape ? shape : rewrite(initial->shape), rewrite(initial->map), outs});
        }
    } else {
        seed = rewrite(old_mr.init);
    }
    if (destination) {
        mr.map_out = w.app(w.app(w.annex<btensor::tile_write_map>(), {mr.Rn, mr.Ro}), {destination->origin, mr.map_out})
                         ->set<true>("tileWriteMap");
        mr.post_maps = w.app(w.app(w.annex<btensor::tile_post_maps>(), {mr.Ro, mr.nps, mr.Rps}),
                             {destination->origin, mr.post_maps});
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
    // A work-group panel is read a vector at a time where its last extent allows; a laid-out one's map is
    // followed by its placement.
    auto input_maps = DefVec(mr.maps->projs(nis));
    for (u64 i = 0; i != nis; ++i) {
        auto buffer = is->proj(nis, i), storage = mr.Sis->proj(nis, i);
        auto layout  = fe::lookup(layouts_, buffer);
        auto view    = fe::lookup(layout_views_, buffer);
        auto r       = storage->num_projs();
        auto logical = layout ? fe::lookup(logical_shapes_, buffer) : storage;
        auto size    = Lit::isa(core::op(core::trait::size, mr.Tis->proj(nis, i)));
        auto last = Lit::isa(logical->proj(r, r - 1)), physical_last = Lit::isa(storage->proj(r, r - 1));
        auto n = size && 16 % *size == 0 ? 16 / *size : 1;
        // A vector runs along the logical last dim, so the layout must keep that dim innermost.
        auto keeps_last
            = !layout || (view->perm.back() + 1 == view->perm.size() && (view->split == 1 || view->split % n == 0));
        auto vectors   = panels_.contains(buffer) && n > 1 && last && *last % n == 0 && physical_last
                      && *physical_last % n == 0 && keeps_last;
        auto placement = layout ? layout_lam(layout, logical) : w.app(w.annex<affine::id>(), w.lit_nat(r));
        if (!vectors) {
            if (!layout) continue;
            auto lam = w.mut_lam(input_maps[i]->type()->as<Pi>())->set("placedMap");
            lam->set(true, w.app(placement, w.app(input_maps[i], lam->var())));
            input_maps[i] = lam;
            continue;
        }
        auto pi       = input_maps[i]->type()->as<Pi>();
        input_maps[i] = w.app(w.app(w.annex<btensor::vectorize>(), {pi->dom()->arity(), w.lit_nat(r)}),
                              {w.lit_nat(n), logical, placement, input_maps[i]});
    }
    for (u64 i = 0; i != nps; ++i)
        plain(post_is->proj(nps, i));
    mr.maps     = w.tuple(input_maps);
    mr.maps_all = w.tuple({mr.maps, mr.post_maps});

    auto [cTo, ins_ty]   = mr.comb->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto memcomb         = w.app(w.app(w.annex<btensor::lift_mem>(), {cTo, ins_ty, cTo}), mr.comb);
    auto [pTo, pexts_ty] = mr.post->type()->as<Pi>()->dom()->proj(2, 0)->projs<2>();
    auto mempost         = w.app(w.app(w.annex<btensor::lift_mem>(), {pTo, pexts_ty, mr.Tp}), mr.post);

    // NB: no `w.call` here — the meta groups are *forwarded* from the tensor op on purpose. Leaving them to
    // inference would re-derive `{Tis, Ris, Sis}` from the `buffer.Buf` operands, whose literal size-1 axes are
    // already folded away, so they would no longer agree with the logical shapes the loop generation iterates.
    auto op      = w.annex<btensor::map_reduce_post>();
    op           = w.app(op, mr.nis_nps);
    op           = w.app(op, mr.meta);
    op           = w.app(op, mr.shapes);
    op           = w.app(op, mr.in_tys);
    op           = w.app(op, seed->type());
    op           = w.app(op, w.tuple({memcomb, seed, mempost}));
    op           = w.app(op, mr.map_out);
    op           = w.app(op, mr.maps_all);
    auto consume = [&](const Def* dst) {
        return [&, dst](const Def* am) { return w.app(op, w.tuple({am, dst, is, post_is})); };
    };
    if (destination) return close_batch(destination->mem, destination->buffer->type(), consume(destination->buffer));
    auto own = [&](const Def* start) {
        auto [am, dst] = allocate(mr.Ro, mr.So, mr.Tp, full_output(old_mr), start);
        return consume(dst)(am);
    };
    auto [m, out] = close_batch(nullptr, buffer::type_buf(mr.Ro, mr.So, mr.Tp), own)->projs<2>();
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

    if (warp_self_) return store_warp(root, destination);
    prepare_realizations(root);
    if (auto app = root->isa<App>()) {
        if (Axm::isa<tensor::scalar>(app))
            return buffer::op_write(destination.mem, destination.buffer, new_world().tuple(), rewrite(app->arg()));
        if (Axm::isa<tensor::map_reduce_post>(app)) return lower_map_reduce(app, &destination);
        if (Axm::isa<tensor::tiled>(app)) return lower_tiled(app, &destination);
        if (Axm::isa<tensor::tile_fold>(app))
            if (auto merged = lower_tile_fold(app, &destination)) return merged;
        if (Axm::isa<tensor::mma>(app)) {
            if (auto fragments = fe::lookup(fragments_, destination.buffer))
                return lower_mma(app, destination, *fragments);
            return lower_into(app->arg(), destination);
        }
    }
    log().w("tensor destination fallback: tile root {} uses a temporary buffer and copy", root);
    auto& w     = new_world();
    auto value  = rewrite(root);
    auto rank   = Lit::as(destination.shape->arity());
    auto source = to_buffer(value, root);
    if (auto fragments = fe::lookup(fragments_, source)) return store_fragments(source, *fragments, destination);
    auto shape         = destination.extent;
    auto [run, call]   = btensor::ds_splice(w.tuple({destination.mem, destination.buffer, source}),
                                            w.sigma({destination.mem->type(), destination.buffer->type()}), "tileCopy");
    auto [m, dst, src] = run->var(2, 0)->projs<3>();
    const Def* acc     = w.tuple({m, dst});
    const Def* exit    = run->ret_var();
    auto cur           = run;
    auto scope         = fe::Restore(uniform_, uniform(root));
    auto annotate      = cooperative();
    auto iters         = annotate ? btensor::linear_loop(cur, exit, acc, shape->projs(rank), "tileCopy", annotate)
                                  : affine::counting_nest(cur, exit, acc, shape->projs(rank), "tileCopy");
    auto [loop_mem, loop_dst] = acc->projs<2>();
    const Def* element        = src;
    // A capped source is stored at its cap, which the copied extent may undercut.
    auto source_shape = fe::lookup(storage_shapes_, root);
    if (Axm::isa<buffer::Buf>(src->type()))
        std::tie(loop_mem, element)
            = buffer::op_read(loop_mem, src, btensor::folded_idx(source_shape ? source_shape : shape, iters))
                  ->projs<2>();
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
                                       Defs extents,
                                       const Def* inputs,
                                       Defs state,
                                       Scratch* pool,
                                       const Destination* destination) {
    auto& old      = old_world();
    auto& w        = new_world();
    auto rank      = extents.size();
    auto parameter = old.mut_lam(old.pi(old.arr(rank, old.type_nat()), old.type_nat()));
    auto origin    = parameter->var();
    for (size_t d = 0; d != rank; ++d)
        if (auto tile = Lit::isa(plan.tiles->proj(rank, d)); tile && *tile) multiples_[origin->proj(rank, d)] = *tile;
    auto specialized = old.tuple(DefVec(rank, [&](size_t d) {
        auto component = off->proj(rank, d);
        return Lit::isa(component) == 0 ? old.lit_nat_0() : origin->proj(rank, d);
    }));
    auto ext         = old.tuple(DefVec(rank, [&](size_t d) -> const Def* {
        if (extents[d]) return extents[d];
        auto tile = plan.tiles->proj(rank, d);
        auto rest = old.call(core::nat::sub, Defs{plan.shape->proj(rank, d), specialized->proj(rank, d)});
        return old.extract(old.tuple({tile, rest}), old.call(core::ncmp::l, Defs{rest, tile}));
    }));
    const Def* root;
    if (auto lam = plan.body->isa<Lam>(); lam && lam->is_set())
        root = old.app(lam->reduce(old.tuple({specialized, ext})).back(), inputs);
    else
        plan.body->blame("tensor.tiled body must be a defined lambda").bail();
    auto pending      = fe::Restore(pending_, {});
    auto nested_batch = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
    auto memo         = fe::Restore(fresh_memo_, {});
    auto storage      = fe::Restore(storage_shapes_, storage_shapes_);
    auto cache        = fe::Restore(realizations_, realizations_);
    auto scratch      = fe::Restore(scratch_, sequential_tile(root) ? pool : nullptr);
    auto item         = fe::Restore(item_origin_, plan.bind >= 2 ? origin->as<Var>() : item_origin_);
    auto transients   = std::vector<std::array<const Def*, 4>>();
    auto owner        = fe::Restore(transients_, &transients);
    push();
    bind_old(parameter->has_var());
    map(origin, off);
    auto ends = fe::Restore(tile_ends_, tile_ends_);
    for (size_t d = 0; d != rank; ++d) {
        auto end   = old.call(core::nat::add, Defs{specialized->proj(rank, d), ext->proj(rank, d)});
        auto last  = old.call(core::nat::sub, Defs{ext->proj(rank, d), old.lit_nat_1()});
        auto high  = old.call(core::nat::add, Defs{specialized->proj(rank, d), last});
        auto bound = plan.shape->proj(rank, d);
        tile_ends_.emplace_back(rewrite(end), rewrite(bound));
        tile_ends_.emplace_back(rewrite(high), rewrite(old.call(core::nat::sub, Defs{bound, old.lit_nat_1()})));
    }
    const Def* mem = state[0];
    DefVec outputs;
    auto shared = RealizationRequests(root).shared();
    DefVec destinations;
    for (u64 i = 0; i != plan.count; ++i) {
        auto value = plan.multiple ? root->proj(plan.count, i) : root;
        if (Axm::isa<tensor::map_reduce_post>(value) || Axm::isa<tensor::tiled>(value)) destinations.push_back(value);
    }
    // A warp-owned region's tensors are warp values, never realized into buffers.
    if (!warp_self_) prepare_realizations(root, nullptr, destinations);
    Rewriter substitutions(old);
    for (auto var : root->free_vars())
        substitutions.map(var, var);
    for (u64 i = 0; i != plan.count; ++i) {
        auto value            = plan.multiple ? root->proj(plan.count, i) : root;
        auto [origin, extent] = plan.box(i, specialized, ext);
        auto shape            = plan.shapes->proj(plan.count, i);
        auto r                = plan.ranks->proj(plan.count, i);
        auto target_origin    = rewrite(origin);
        if (destination)
            target_origin = w.tuple(DefVec(rank, [&](size_t d) {
                return w.call(core::nat::add, Defs{destination->origin->proj(rank, d), off->proj(rank, d)});
            }));
        auto storage = destination ? destination->shape : bounded_shape(rewrite(shape));
        if (!destination) storage = padded_like(storage, state[i + 1]);
        auto result
            = lower_into(substitutions.rewrite(value), {mem, state[i + 1], storage, target_origin, rewrite(extent)});
        mem = result->proj(2, 0);
        outputs.push_back(result->proj(2, 1));
        if (plan.multiple) {
            auto parameter = old.mut_lam(
                type_tensor(plan.ranks->proj(plan.count, i), shape, plan.types->proj(plan.count, i)), old.type_nat());
            bind_old(parameter->has_var());
            map(parameter->var(), outputs.back());
            if (auto storage = bounded_shape(rewrite(shape)); storage != rewrite(shape))
                storage_shapes_[parameter->var()] = storage;
            auto read = old.app(old.app(old.annex<tensor::slice>(), {plan.types->proj(plan.count, i), r}), shape);
            auto view = old.app(old.app(read, {origin, old.pack(r, old.lit_nat_1()), extent}), parameter->var());
            substitutions.map(value, view);
            for (auto& group : shared)
                if (Check::test_alpha(value, group.front()))
                    for (auto alias : group)
                        substitutions.map(alias, view);
        }
    }
    for (auto [rank, shape, type, buffer] : transients | std::views::reverse)
        mem = buffer::op_free(rank, shape, type, mem, buffer);
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
    auto rank  = std::optional<u64>(plan.rank);
    auto& old  = old_world();
    auto sizes = fe::Vector<std::optional<u64>>(*rank);
    auto steps = fe::Vector<u64>(*rank);
    // A tile as large as its runtime extent leaves that dimension untiled.
    auto whole = fe::Vector<bool>(*rank);
    for (size_t d = 0; d != *rank; ++d) {
        auto tile = Lit::isa(tiles->proj(*rank, d));
        whole[d]  = !tile && Check::test_alpha(tiles->proj(*rank, d), shape->proj(*rank, d));
        if (whole[d]) continue;
        if (!tile) app->blame("tensor.tiled lowering requires literal tile sizes").bail();
        sizes[d] = Lit::isa(shape->proj(*rank, d));
        steps[d] = sizes[d] ? std::min(*sizes[d], *tile) : *tile;
    }
    const Def* mem = destination ? destination->mem : fresh_mem();
    auto in_group  = bound_level_ == 1 && driver().is_loaded("gpu");
    DefVec buffers(plan.count);
    if (destination) {
        mem        = destination->mem;
        buffers[0] = destination->buffer;
    } else {
        for (u64 i = 0; i != plan.count; ++i) {
            auto rank = rewrite(plan.ranks->proj(plan.count, i)), type = rewrite(plan.types->proj(plan.count, i));
            auto shape = bounded_shape(rewrite(plan.shapes->proj(plan.count, i)), in_group);
            if (auto reused = in_group && plan.count == 1 ? reusable_input(app, inputs) : nullptr) {
                buffers[i] = reused;
            } else if (in_group) {
                shape = padded_rows(shape);
                // A work-group's result is shared by its work-items.
                auto scope                = w.app(w.annex<gpu::shared_scope>(), mem);
                std::tie(mem, buffers[i]) = buffer::op_alloc(rank, shape, type, scope)->projs<2>();
                group_buffers_.insert(buffers[i]);
            } else {
                std::tie(mem, buffers[i]) = buffer::op_alloc(rank, shape, type, mem)->projs<2>();
                transient(rank, shape, type, buffers[i]);
            }
        }
    }
    auto views     = view_flags(inputs);
    auto sources   = view_sources(inputs, views);
    auto captured  = materialize(sources->type(), sources);
    auto parameter = old_world().mut_lam(sources->type(), old_world().type_nat());
    auto proxy     = parameter->var();
    auto reads     = view_reads(inputs, proxy, views);
    DefVec initial{mem};
    initial.insert(initial.end(), buffers.begin(), buffers.end());
    auto state       = w.tuple(initial);
    auto [run, call] = btensor::ds_splice(w.tuple({state, captured}), state->type(), "tiled");
    auto runtime     = run->var(2, 0);
    Scratch scratch{mem, call->type(), {}};
    auto nested = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
    push();
    bind_new(run->has_var());
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, runtime->proj(2, 1));
    forward_storage(sources, proxy);
    forward_layouts(captured, runtime->proj(2, 1));
    // Offsets live in the new world; extents stay in the old one, where the body is specialized to them.
    // A capped dimension leaves its extent null for the body to bound by its origin.
    DefVec offsets(*rank), extents(*rank);
    auto cap = Lit::isa(plan.tail) == 0;
    std::function<const Def*(size_t, const Def*, const Def*)> emit
        = [&](size_t d, const Def* acc, const Def* exit) -> const Def* {
        if (d == *rank) {
            auto off    = w.tuple(offsets);
            auto result = lower_tile_body(plan, off, extents, reads, acc->projs(plan.count + 1), &scratch, destination);
            return w.app(exit, result);
        }
        if (sizes[d] == 0) return w.app(exit, acc);
        // An empty dimension has no tile, so every tile body may assume non-empty extents.
        if (whole[d]) {
            offsets[d] = w.lit_nat_0();
            extents[d] = shape->proj(*rank, d);
            auto body  = w.mut_con(acc->type())->set("tileWhole");
            auto skip  = w.mut_con(acc->type())->set("tileWholeSkip");
            body->set(true, emit(d + 1, body->var(), exit));
            skip->app(true, exit, skip->var());
            auto empty = w.call(core::ncmp::e, Defs{rewrite(extents[d]), w.lit_nat_0()});
            return w.app(w.extract(w.tuple({body, skip}), empty), acc);
        }
        // A single tile needs no loop, and its literal origin lets the body fold its clamps.
        if (sizes[d] && *sizes[d] == steps[d]) {
            offsets[d] = w.lit_nat_0();
            extents[d] = old.lit_nat(steps[d]);
            return emit(d + 1, acc, exit);
        }
        const Def* count;
        const Def* next   = exit;
        const Def* extent = old.lit_nat(steps[d]);
        if (auto size = sizes[d]) {
            count     = w.lit_i64(*size / steps[d]);
            auto tail = *size % steps[d];
            if (tail && cap) {
                count  = w.lit_i64(*size / steps[d] + 1);
                extent = nullptr;
            } else if (tail) {
                auto after = w.mut_con(acc->type())->set("tileTail");
                offsets[d] = w.lit_nat(*size - tail);
                extents[d] = old.lit_nat(tail);
                after->set(true, emit(d + 1, after->var(), exit));
                next = after;
            }
        } else if (cap) {
            auto step    = w.lit_nat(steps[d]);
            auto runtime = rewrite(shape->proj(*rank, d));
            auto padded  = w.call(core::nat::add, Defs{runtime, w.lit_nat(steps[d] - 1)});
            count        = w.call<core::bitcast>(w.type_i64(), w.call(core::nat::div, Defs{padded, step}));
            extent       = nullptr;
        } else {
            auto extent  = shape->proj(*rank, d);
            auto step    = w.lit_nat(steps[d]);
            auto runtime = rewrite(extent);
            auto rem     = w.call(core::nat::rem, Defs{runtime, step});
            count        = w.call<core::bitcast>(w.type_i64(), w.call(core::nat::div, Defs{runtime, step}));
            auto after   = w.mut_con(acc->type())->set("tileTail");
            auto tail    = w.mut_con(acc->type())->set("tileTailBody");
            auto skip    = w.mut_con(acc->type())->set("tileTailSkip");
            offsets[d]   = w.call(core::nat::sub, Defs{runtime, rem});
            extents[d]   = old.call(core::nat::rem, Defs{extent, old.lit_nat(steps[d])});
            tail->set(true, emit(d + 1, tail->var(), exit));
            skip->app(true, exit, skip->var());
            auto any = w.call(core::ncmp::ne, Defs{rem, w.lit_nat_0()});
            after->app(true, w.extract(w.tuple({skip, tail}), any), after->var());
            next = after;
        }
        auto [loop, loop_call]     = affine::counting_for(count, acc, next, w.sym("tileGrid"));
        auto [index, state, yield] = loop->vars<3>();
        offsets[d] = w.call(core::nat::mul, Defs{w.call<core::bitcast>(w.type_nat(), index), w.lit_nat(steps[d])});
        extents[d] = extent;
        loop->set(true, emit(d + 1, state, yield));
        return loop_call;
    };
    // A bound level runs one linear loop over all its tiles, which `gpu.outline` maps to a one-dimensional id.
    auto emit_bound = [&](const Def* acc, const Def* exit) -> const Def* {
        auto lanes  = plan.bind == 2 ? lane_count(region_body(app->arg()).first) : u64(1);
        u64 total   = 1;
        auto counts = fe::Vector<u64>(*rank, 1);
        for (size_t d = 0; d != *rank; ++d) {
            if (sizes[d] == 0) return w.app(exit, acc);
            if (whole[d] || (sizes[d] && *sizes[d] == steps[d])) continue;
            if (!sizes[d]) app->blame("a bound tensor.tiled level requires literal extents").bail();
            if (*sizes[d] % steps[d] && !cap) app->blame("a bound tensor.tiled level cannot peel its tail").bail();
            counts[d] = (*sizes[d] + steps[d] - 1) / steps[d];
            total *= counts[d];
        }
        auto body    = w.mut_con({w.type_i64(), acc->type(), w.cn(acc->type())})
                           ->set(plan.bind == 1   ? "tileGroups"
                                 : plan.bind == 2 ? "tileItems"
                                                  : "tileWarps");
        auto tag     = plan.bind == 1 ? gpu::bind::group : plan.bind == 2 ? gpu::bind::item : gpu::bind::warp;
        auto wrapped = w.app(w.app(w.annex<gpu::bind>(tag), body->type()), body);
        // A work-item whose reduction runs on its lanes takes that many adjacent ids, which must fill whole warps.
        if (lanes > 1 && total * lanes % 32)
            app->blame("the {} work-items' lanes of a lane reduction do not fill whole warps", total).bail();
        auto loop = w.call<affine::For>(wrapped, exit, Defs{w.lit_i64(0), w.lit_i64(total * lanes), w.lit_i64(1), acc});
        auto [index, state, yield] = body->vars<3>();
        const Def* linear          = w.call<core::bitcast>(w.type_nat(), index);
        const Def* lane            = lane_;
        if (lanes > 1) {
            lane   = w.call(core::nat::rem, Defs{linear, w.lit_nat(lanes)});
            linear = w.call(core::nat::div, Defs{linear, w.lit_nat(lanes)});
        }
        auto lane_scope  = fe::Restore(lane_, lane);
        auto lanes_scope = fe::Restore(lanes_, lanes > 1 ? lanes : lanes_);
        auto stride      = total;
        for (size_t d = 0; d != *rank; ++d) {
            if (counts[d] == 1) {
                // A one-tile bound region depends on its binder in every dim, so nothing it stages floats out.
                // The binder is 0 there, and scaling it by the step keeps the offset a known multiple of the step.
                offsets[d] = total != 1 ? w.lit_nat_0()
                           : d == 0     ? linear
                                        : w.call(core::nat::mul, Defs{linear, w.lit_nat(steps[d])});
                extents[d] = whole[d] ? shape->proj(*rank, d) : old.lit_nat(steps[d]);
                continue;
            }
            stride /= counts[d];
            auto quotient = w.call(core::nat::div, Defs{linear, w.lit_nat(stride)});
            auto tile     = w.call(core::nat::rem, Defs{quotient, w.lit_nat(counts[d])});
            offsets[d]    = w.call(core::nat::mul, Defs{tile, w.lit_nat(steps[d])});
            extents[d]    = *sizes[d] % steps[d] ? nullptr : old.lit_nat(steps[d]);
        }
        // A warp's lanes share its tile like a work-item's own; only the tensor-core lowering tells them apart.
        auto level = fe::Restore(bound_level_, std::min<u64>(plan.bind, 2));
        auto owned = fe::Restore(warp_self_, plan.bind == 5 ? warp_lane(old_world()) : warp_self_);
        body->set(true, emit(*rank, state, yield));
        return loop;
    };
    auto done = w.mut_con(call->type())->set("tileScratchDone");
    // A work-group level inside a work-group, as a `group_tile_req` has, is its one tile and runs in place.
    auto bound = plan.bind && driver().is_loaded("gpu") && !(plan.bind == 1 && bound_level_ != 0);
    // Work-items that fill a work-group's buffer publish it to the others behind a barrier.
    const Def* exit = done;
    auto shared_out = std::ranges::any_of(buffers, [&](const Def* b) { return group_buffers_.contains(b); });
    if (bound && plan.bind >= 2 && bound_level_ == 1 && (shared_out || composed_)) {
        auto published = w.mut_con(done->type()->as<Pi>()->dom())->set("tilePublished");
        auto state     = DefVec(published->var()->projs(plan.count + 1));
        state[0]       = w.app(w.annex<gpu::fence>(), state[0]);
        published->app(true, done, w.tuple(state));
        exit = published;
    }
    auto composed = fe::Restore(composed_, composed_ || (plan.bind == 1 && !bound));
    run->set(true, bound ? emit_bound(runtime->proj(2, 0), exit) : emit(0, runtime->proj(2, 0), exit));
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
    for (u64 i = 0; i != plan.count; ++i) {
        auto shape   = rewrite(plan.shapes->proj(plan.count, i));
        auto storage = bounded_shape(shape, in_group);
        if (in_group) storage = padded_like(padded_rows(storage), buffers[i]);
        if (storage != shape) storage_shapes_[plan.multiple ? app->proj(plan.count, i) : app] = storage;
    }
    auto out_mem = call->proj(plan.count + 1, 0);
    if (in_group)
        for (u64 i = 0; i != plan.count; ++i)
            group_buffers_.insert(call->proj(plan.count + 1, i + 1));
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
    auto nested_batch               = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
    auto nested                     = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
    push();
    bind_new(run->has_var());
    auto parameter = old.mut_lam(plan.inputs->type(), old.type_nat());
    auto proxy     = parameter->var();
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, inputs);
    forward_storage(plan.inputs, proxy);
    forward_layouts(captured, inputs);
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
    auto filled                         = btensor::build_pointwise(
        w.sigma({loop_mem->type(), loop_dst->type()}), loop_mem, loop_dst, loop_src, rewrite(gs), rank, "prepackFill",
        [&](Defs indices, const Def* value, const Def* mem) {
            auto nested = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
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
    // A bound region fills a slice cooperatively; a box with runtime extents has no buffer type to alias anyway.
    auto view = Axm::isa<tensor::slice>(input) || Axm::isa<tensor::slice_box>(input);
    if (view && bound_level_ == 0) {
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
        if (inner && bound_level_ != 0) {
            auto c   = contiguous->first;
            auto row = physical_shape->proj(ri, ri - 1);
            // A padded source's rows are as far apart as its storage's.
            if (auto storage = fe::lookup(storage_shapes_, source)) row = storage->proj(ri, ri - 1);
            vec = vector_width(type, extent->proj(r, c), start->proj(r, c), step->proj(r, c), row, contiguous->second);
            if (auto run = Lit::isa(s_out->proj(rk, *inner)); !run || *run % vec) vec = 1;
        }
        auto element = vec == 1 ? type : old.arr(vec, type);
        auto getter  = old.app(old.annex<tensor::get>(), {physical_rank, physical_shape, type});
        if (vec != 1)
            getter = old.app(old.app(old.annex<tensor::get_vector>(), {physical_rank, physical_shape, type}),
                             old.lit_nat(vec));
        auto scale = old.tuple(DefVec(rk, [&](size_t d) { return old.lit_nat(d == inner ? vec : 1); }));
        if (vec != 1) {
            auto dims    = DefVec(s_out->projs(rk));
            dims[*inner] = old.lit_nat(*Lit::isa(dims[*inner]) / vec);
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

/// The work-group buffer among @p inputs that @p request may overwrite: it reads the input pointwise, and every other
/// reader in the fold step runs before it.
const Def* LowerToMem::reusable_input(const App* request, const Def* inputs) {
    if (!step_root_) return nullptr;
    auto& old = old_world();
    auto n    = input_count(inputs);
    for (size_t k = 0; k != n; ++k) {
        auto input = Axm::peel<tensor::buf>(n == 1 ? inputs : inputs->proj(n, k));
        // Another request's result, or a field of the fold's state.
        auto field = input;
        while (auto extract = field->isa<Extract>())
            field = extract->tuple();
        auto owned = Axm::isa<tensor::tiled>(input) || (input != field && field->isa<Var>());
        if (!owned || !Check::test_alpha(input->type(), request->type())) continue;
        auto buffer = rewrite(input);
        if (!group_buffers_.contains(buffer)) continue;
        auto rank = Lit::isa(TensorType::isa(input->type())->r);
        if (!rank || !pointwise_step(request, input, old.tuple(DefVec(*rank, [&](size_t) { return old.lit_nat(0); }))))
            continue;
        auto before  = DefSet();
        auto pending = DefVec(request->ops().begin(), request->ops().end());
        for (size_t i = 0; i != pending.size(); ++i)
            if (before.insert(pending[i]).second && !pending[i]->isa<Var>())
                for (auto op : pending[i]->ops())
                    if (op) pending.push_back(op);
        auto later = true;
        auto seen  = DefSet();
        pending    = DefVec{step_root_};
        for (size_t i = 0; later && i != pending.size(); ++i) {
            auto def = pending[i];
            if (def == input) later = false;
            if (def == request || !seen.insert(def).second || def->isa<Var>() || def->isa<Axm>()) continue;
            if (Axm::isa<tensor::tiled>(def) && before.contains(def)) continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        if (later) return buffer;
    }
    return nullptr;
}

// Clipped tiles keep their cap as storage, so region parameters must index them by it.
void LowerToMem::forward_storage(const Def* inputs, const Def* proxy) {
    auto n = Lit::as(inputs->type()->arity());
    for (size_t i = 0; i != n; ++i)
        if (auto shape = fe::lookup(storage_shapes_, inputs->proj(n, i))) storage_shapes_[proxy->proj(n, i)] = shape;
}

// A panel passed into a region keeps the layout its fill wrote, so the region's reads place their coordinates.
void LowerToMem::forward_layouts(const Def* buffers, const Def* params) {
    auto forward = [this](const Def* buffer, const Def* param) {
        if (auto found = fe::lookup(layouts_, buffer)) {
            auto view = fe::lookup(layout_views_, buffer);
            if (!view) buffer->blame("a panel layout is missing its cost-model view").bail();
            auto copied_view       = *view;
            layouts_[param]        = found;
            layout_views_[param]   = std::move(copied_view);
            logical_shapes_[param] = logical_shapes_[buffer];
        }
        if (panels_.contains(buffer)) panels_.insert(param);
    };
    if (Axm::isa<buffer::Buf>(buffers->type())) return forward(buffers, params);
    if (auto n = Lit::isa(buffers->type()->arity()))
        for (size_t i = 0; i != *n; ++i)
            forward(buffers->proj(*n, i), params->proj(*n, i));
}

// A slice of a realized tensor among region inputs stays a read of its source, which the body composes.
// A reshape that only adds or drops unit dims, which a buffer of the same storage serves unchanged.
bool unit_reshape(const Def* def) {
    auto reshape = Axm::isa<tensor::reshape>(def);
    if (!reshape) return false;
    auto [meta, s_in, s_out] = reshape->callee()->as<App>()->uncurry_args<3>();
    auto r_in = Lit::isa<u64>(meta->proj(3, 1)), r_out = Lit::isa<u64>(meta->proj(3, 2));
    if (!r_in || !r_out) return false;
    auto kept = [](const Def* s, u64 r) {
        auto dims = DefVec();
        for (auto d : s->projs(r))
            if (Lit::isa(d) != 1) dims.push_back(d);
        return dims;
    };
    return kept(s_in, *r_in) == kept(s_out, *r_out);
}

// The slice under any unit reshapes of @p input.
const Def* view_slice(const Def* input) {
    while (unit_reshape(input))
        input = input->as<App>()->arg();
    return input;
}

bool LowerToMem::is_view(const Def* input) {
    input = view_slice(input);
    if (!Axm::isa<tensor::slice>(input) && !Axm::isa<tensor::slice_box>(input)) return false;
    auto source = Axm::peel<tensor::buf>(input->as<App>()->arg());
    if (lookup(source)) return true;
    while (auto extract = source->isa<Extract>())
        source = extract->tuple();
    return source->isa<Var>() || Axm::isa<tensor::stage>(source) || Axm::isa<tensor::placed>(source);
}

// A single tensor input is one component, not the two fields of its record type.
size_t LowerToMem::input_count(const Def* inputs) {
    return TensorType::isa(inputs->type()) ? 1 : Lit::as(inputs->type()->arity());
}

// Decided once: materializing the sources realizes them, which would turn later inputs into views.
std::vector<bool> LowerToMem::view_flags(const Def* inputs) {
    auto n     = input_count(inputs);
    auto flags = std::vector<bool>(n);
    for (size_t i = 0; i != n; ++i)
        flags[i] = is_view(n == 1 ? inputs : inputs->proj(n, i));
    return flags;
}

const Def* LowerToMem::view_sources(const Def* inputs, const std::vector<bool>& views) {
    auto n      = input_count(inputs);
    auto source = [](const Def* input) { return view_slice(input)->as<App>()->arg(); };
    if (n == 1) return views[0] ? source(inputs) : inputs;
    return old_world().tuple(DefVec(n, [&](size_t i) {
        auto input = inputs->proj(n, i);
        return views[i] ? source(input) : input;
    }));
}

// Re-applies a view's slice and unit reshapes to the region's parameter @p proxy.
const Def* LowerToMem::view_reads(const Def* inputs, const Def* proxy, const std::vector<bool>& views) {
    auto& old  = old_world();
    auto apply = [&](const Def* input, const Def* p) {
        auto callees = DefVec();
        for (auto cur = input; unit_reshape(cur); cur = cur->as<App>()->arg())
            callees.push_back(cur->as<App>()->callee());
        auto read = old.app(view_slice(input)->as<App>()->callee(), p);
        for (auto callee : callees | std::views::reverse)
            read = old.app(callee, read);
        return read;
    };
    auto n = input_count(inputs);
    if (n == 1) return views[0] ? apply(inputs, proxy) : proxy;
    return old.tuple(DefVec(
        n, [&](size_t i) { return views[i] ? apply(inputs->proj(n, i), proxy->proj(n, i)) : proxy->proj(n, i); }));
}

const Def* LowerToMem::region_input(const Def* inputs) {
    auto views     = view_flags(inputs);
    auto sources   = view_sources(inputs, views);
    auto value     = materialize(sources->type(), sources);
    auto& old      = old_world();
    auto parameter = old.mut_lam(sources->type(), old.type_nat());
    auto proxy     = parameter->var();
    if (auto var = parameter->has_var()) bind_old(var);
    map(proxy, value);
    forward_storage(sources, proxy);
    return view_reads(inputs, proxy, views);
}

// One warp's tensor-core block: A's fragments by ldmatrix from its row-major panel, B's by scalar loads, and the
// accumulator, an SSA value of fragments, updated per instruction.
const Def* LowerToMem::lower_mma(const App* app, const Destination& destination, const Fragments& fragments) {
    auto& w     = new_world();
    auto root   = app->arg()->as<App>();
    auto old_mr = btensor::MapRedPost::of(root->callee()->as<App>());
    auto panels = fe::Restore(mma_panels_, true);
    auto batch  = fe::Restore<std::optional<Batch>>(batch_, open_batch(destination.mem));
    auto is     = rewrite(root->arg())->proj(2, 0);
    return close_batch(destination.mem, destination.buffer->type(), [&](const Def* mem) -> const Def* {
        auto Sr = rewrite(old_mr.Sr);
        for (u64 d = 0; d != 3; ++d) {
            auto extent = Lit::isa(Sr->proj(3, d));
            auto tile   = std::array{fragments.m, fragments.n, u64(8)}[d];
            if (!extent || *extent % tile)
                app->blame("the tensor-core tile does not divide the warp's block {}", Sr).bail();
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
            if (auto layout = fe::lookup(layouts_, buffer)) {
                auto placed = w.mut_lam(map->type()->as<Pi>())->set("placedMap");
                placed->set(true, w.app(layout_lam(layout, logical_shapes_[buffer]), w.app(map, placed->var())));
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
        for (u64 ks = 0; ks != depth / 8; ++ks) {
            auto a = DefVec(ms);
            for (u64 mi = 0; mi != ms; ++mi) {
                auto [row, col]                     = fragment_coords(w, mma_m16n8k8_tf32::panel_a_addr, lane);
                auto [buffer, storage, physical, r] = coords(0, add(nat(mi * 16), row), nat(0), add(nat(ks * 8), col));
                auto runs                           = DefVec(storage->projs(r));
                runs.back()                         = w.call(core::nat::div, Defs{runs.back(), nat(4)});
                auto run                            = DefVec(physical->projs(r));
                auto last                           = w.call<core::idx2nat>(run.back());
                run.back() = w.call<core::idx>(w.lit_nat_0(), runs.back(), w.call(core::nat::div, Defs{last, nat(4)}));
                auto vectors      = buffer::op_vectorize(nat(4), buffer);
                auto index        = *Shape(w.tuple(run)).fold(w.tuple(runs));
                auto [vr, vs, vT] = Axm::as<buffer::Buf>(vectors->type())->args<3>();
                auto loaded       = w.app(w.app(w.annex<gpu::buf_ldmatrix>(), {vr, vs}), {nat(4), mem, vectors, index});
                std::tie(mem, a[mi]) = loaded->projs<2>();
                a[mi]                = rounded(a[mi], 4);
            }
            auto b = DefVec(ns);
            for (u64 ni = 0; ni != ns; ++ni) {
                auto half = DefVec(2);
                for (u64 h = 0; h != 2; ++h) {
                    auto [row, col] = fragment_coords(w, mma_m16n8k8_tf32::l_b, w.tuple({lane, w.lit_idx(2, h)}));
                    auto [buffer, storage, physical, r]
                        = coords(1, nat(0), add(nat(ni * 8), col), add(nat(ks * 8), row));
                    std::tie(mem, half[h]) = buffer::op_read(mem, buffer, *Shape(physical).fold(storage))->projs<2>();
                }
                b[ni] = rounded(w.tuple(half), 2);
            }
            for (u64 mi = 0; mi != ms; ++mi)
                for (u64 ni = 0; ni != ns; ++ni) {
                    auto& c = acc[mi * ns + ni];
                    c       = w.app(w.annex<gpu::mma_tf32>(), w.tuple({a[mi], b[ni], c}));
                }
        }
        return w.tuple({mem, w.tuple(acc)});
    });
}

// Writes each lane's accumulator fragments to their rows and columns of @p destination.
const Def* LowerToMem::store_fragments(const Def* source, const Fragments& fragments, const Destination& destination) {
    auto& w   = new_world();
    auto rank = Lit::as(destination.shape->arity());
    auto nat  = [&](u64 v) { return w.lit_nat(v); };
    auto add  = [&](const Def* a, const Def* b) { return w.call(core::nat::add, Defs{a, b}); };
    auto lane = lane_index(w);
    auto ns   = fragments.cols / fragments.n;
    auto mem  = destination.mem;
    for (u64 f = 0; f != fragments.count(); ++f) {
        auto value = source->proj(fragments.count(), f);
        for (u64 e = 0; e != 4; ++e) {
            auto [local_row, local_col] = fragment_coords(w, mma_m16n8k8_tf32::l_c, w.tuple({lane, w.lit_idx(4, e)}));
            auto row                    = add(nat(f / ns * fragments.m), local_row);
            auto col                    = add(nat(f % ns * fragments.n), local_col);
            auto at                     = std::array{row, col};
            auto index                  = w.tuple(DefVec(rank, [&](size_t d) {
                auto coord = add(destination.origin->proj(rank, d), at[d]);
                return w.call<core::idx>(w.lit_nat_0(), destination.shape->proj(rank, d), coord);
            }));
            mem = buffer::op_write(mem, destination.buffer, *Shape(index).fold(destination.shape),
                                   w.extract(value, w.lit_idx(4, e)))
                      ->proj(2, 0);
        }
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
            auto at = std::array{add(nat(f / ns * fragments.m), local_row), add(nat(f % ns * fragments.n), local_col)};
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
        if (dims->size() == 2 && (*dims)[0] == 0 && (*dims)[1] == 1)
            return old.extract(old.extract(old.extract(term, lane), j), reg);
        if (dims->size() == 1 && (*dims)[0] == 0)
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
    if (dim(mr.So, *ro, 0) != 16) def->blame("a warp-owned region holds 16 rows, not `{}`", mr.So).bail();
    auto nt = *ro == 2 ? dim(mr.So, 2, 1) / 8 : 0;
    if (*ro == 2 && dim(mr.So, 2, 1) % 8) def->blame("a warp-owned tile's columns are a multiple of 8").bail();
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
        auto pos    = *ro == 2 ? old.sigma({I(32), I(nt), I(4)}) : old.sigma({I(32), I(2)});
        auto g      = old.mut_lam(old.pi(pos, T));
        auto lane   = g->var(*ro + 1, 0);
        auto j      = *ro == 2 ? g->var(3, 1) : nullptr;
        auto reg    = g->var(*ro + 1, *ro);
        auto coords = DefVec();
        if (*ro == 2) {
            auto [r, c] = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {lane, j, reg})->projs<2>();
            coords      = {nat(r), nat(c)};
        } else {
            coords = {nat(old.app(old.annex(tensor::warp::r_pos), {lane, reg}))};
        }
        auto ins = old.tuple(DefVec(*nis, [&](size_t k) { return warp_element(mr, k, coords, lane, j, reg); }));
        g->set(true, post(call(mr.comb, old.tuple({mr.init, ins})), coords, lane, j, reg));
        term = *ro == 2 ? old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g)
                        : old.app(old.app(old.annex(tensor::warp::r_gen), T), g);
    } else if (*ro == 1 && *rn == 2) {
        // A reduction over the columns: the warp's row reduction of the combined elements.
        auto op = merge_of(mr.comb);
        if (!op) def->blame("a warp-owned row reduction needs a combiner that merges partials").bail();
        auto cols = dim(mr.Sr, 2, 1);
        if (cols % 8) def->blame("a warp-owned row reduction runs over a multiple of 8 columns").bail();
        nt                  = cols / 8;
        auto g              = old.mut_lam(old.pi(old.sigma({I(32), I(nt), I(4)}), mr.To));
        auto [lane, j, reg] = g->vars<3>();
        auto [r, c] = old.app(old.app(old.annex(tensor::warp::c_pos), old.lit_nat(nt)), {lane, j, reg})->projs<2>();
        auto coords = DefVec{nat(r), nat(c)};
        auto ins    = old.tuple(DefVec(*nis, [&](size_t k) { return warp_element(mr, k, coords, lane, j, reg); }));
        g->set(true, call(mr.comb, old.tuple({mr.init, ins})));
        auto elems = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), mr.To), g);
        auto rows  = old.app(
            old.app(old.app(old.app(old.annex(tensor::warp::row_fold), old.lit_nat(nt)), mr.To), {op, mr.init}), elems);
        auto p        = old.mut_lam(old.pi(old.sigma({I(32), I(2)}), T));
        auto [pl, ph] = p->vars<2>();
        auto row      = nat(old.app(old.annex(tensor::warp::r_pos), {pl, ph}));
        p->set(true, post(old.extract(old.extract(rows, pl), ph), DefVec{row}, pl, nullptr, ph));
        term = old.app(old.app(old.annex(tensor::warp::r_gen), T), p);
    } else if (*ro == 2 && *rn == 3 && *nis == 2) {
        if (!is_contraction(app))
            def->blame("a warp-owned product must combine `acc + a · b`, not `{}`", mr.comb).bail();
        // A contraction `c + a·b` over k-chunks of 8 on the tensor cores.
        auto da = map_dims(map(0), 3), db = map_dims(map(1), 3);
        if (!da || !db || da->size() != 2 || db->size() != 2)
            def->blame("a warp-owned product reads projections").bail();
        auto kk = dim(mr.Sr, 3, 2);
        if (kk % 8) def->blame("a warp-owned product reduces a multiple of 8").bail();
        auto nk      = kk / 8;
        auto F       = T;
        auto a_value = Axm::peel<tensor::buf>(input(0));
        auto from_c  = Axm::isa<tensor::map_reduce_post>(a_value) && *da == std::vector<u64>{0, 2};
        const Def* a;
        if (from_c) {
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
        g->set(true, post(acc, DefVec{nat(row), nat(col)}, pl, pj, pr));
        term = old.app(old.app(old.app(old.annex(tensor::warp::c_gen), old.lit_nat(nt)), T), g);
    } else {
        def->blame("a warp-owned region has no translation for `{}`", def).bail();
    }
    return warp_terms_[def] = term;
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
    auto root                  = old.app(step->reduce(parameter->var()).back(), region_input(inputs));
    auto lane                  = rewrite(warp_self_);
    auto lanes = [&](const Def* x) { return old.app(old.app(old.annex<gpu::lanes>(), x->type()), {warp_self_, x}); };
    auto inits = DefVec(n, [&](size_t i) { return warp_term(init->proj(n, i)); });
    auto local = old.sigma(DefVec(n, [&](size_t i) { return old.extract(inits[i], warp_self_)->type(); }));
    auto first = w.tuple(DefVec(n, [&](size_t i) { return w.extract(rewrite(inits[i]), lane); }));
    auto state = w.tuple({fresh_mem(), first});
    auto [run, call]       = btensor::ds_splice(state, state->type(), "warpFold");
    auto [loop, loop_call] = affine::counting_for(w.call<core::bitcast>(w.type_i64(), rewrite(count)), run->var(2, 0),
                                                  run->ret_var(), w.sym("warpFoldStep"));
    auto [index, carried, yield] = loop->vars<3>();
    auto [loop_mem, locals]      = carried->projs<2>();
    {
        auto pending = fe::Restore(pending_, {});
        auto memo    = fe::Restore(fresh_memo_, {});
        auto cache   = fe::Restore(realizations_, realizations_);
        auto terms   = fe::Restore(warp_terms_, warp_terms_);
        push();
        bind_old(parameter->has_var());
        map(offset, w.call(core::nat::mul, Defs{w.call<core::bitcast>(w.type_nat(), index), rewrite(stride)}));
        auto proxy = old.mut_lam(old.pi(local, old.type_nat()));
        bind_old(proxy->has_var());
        map(proxy->var(), locals);
        for (size_t i = 0; i != n; ++i)
            warp_terms_[accumulator->proj(n, i)] = lanes(proxy->var(n, i));
        for (size_t i = 0; i != 128; ++i) {
            auto call = root->isa<App>();
            if (!call) break;
            auto lam = call->callee()->isa<Lam>();
            if (!lam || !lam->is_set() || Pi::isa_cn(lam->type())) break;
            root = lam->reduce(call->arg()).back();
        }
        auto next = w.tuple(DefVec(n, [&](size_t i) { return w.extract(rewrite(warp_term(root->proj(n, i))), lane); }));
        loop->app(true, yield, w.tuple({loop_mem, next}));
        loop->set(true, wrap_fresh_mem(loop->body()));
        pop();
    }
    run->set(true, loop_call);
    auto result = old.mut_lam(old.pi(local, old.type_nat()));
    bind_old(result->has_var());
    map(result->var(), call->proj(2, 1));
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
    if (rank == 2) {
        auto cols = Lit::as(destination.extent->proj(2, 1));
        auto nt   = cols / 8;
        for (u64 j = 0; j != nt; ++j)
            for (u64 reg = 0; reg != 4; ++reg) {
                auto jj = w.lit_idx(nt, j), rr = w.lit_idx(4, reg);
                auto [r, c] = w.app(w.app(w.annex(tensor::warp::c_pos), w.lit_nat(nt)), {lane, jj, rr})->projs<2>();
                auto elem   = w.extract(w.extract(w.extract(value, lane), jj), rr);
                mem = buffer::op_write(mem, destination.buffer, *Shape(at({nat(r), nat(c)})).fold(destination.shape),
                                       elem)
                          ->proj(2, 0);
            }
    } else {
        for (u64 h = 0; h != 2; ++h) {
            auto hh   = w.lit_idx(2, h);
            auto row  = w.app(w.annex(tensor::warp::r_pos), {lane, hh});
            auto elem = w.extract(w.extract(value, lane), hh);
            mem       = buffer::op_write(mem, destination.buffer, *Shape(at({nat(row)})).fold(destination.shape), elem)
                            ->proj(2, 0);
        }
    }
    (void)old;
    return w.tuple({mem, destination.buffer});
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
    auto input                             = region_input(inputs);
    auto [stepped, level, merge, pipeline] = peel_step(step->reduce(parameter->var()).back());
    auto root                              = old.app(stepped, input);
    auto result                            = root;
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
    auto first_scalar = inplace && literal_count && *literal_count != 0 && splat_scalar(init) && bound_level_ == 0;
    auto extent       = rewrite(shape);
    auto storage      = bounded_shape(extent, bound_level_ != 0);
    auto bounded      = storage != extent;
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
    auto mma       = driver().is_loaded("gpu") && bound_level_ != 0 ? Axm::isa<tensor::mma>(result) : nullptr;
    auto fragments = std::optional<Fragments>();
    if (mma) {
        auto extents = Lit::isa(extent->proj(2, 0)) && Lit::isa(extent->proj(2, 1));
        if (!inplace || rank != 2 || !extents)
            app->blame("a tensor-core fold needs a literal 2D accumulator updated in place").bail();
        auto shape = mma->decurry()->arg(2, 1);
        fragments  = Fragments{Lit::as(extent->proj(2, 0)), Lit::as(extent->proj(2, 1)), Lit::as(shape->proj(3, 0)),
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
    auto stages    = depth > 1 && bound_level_ != 0 && driver().is_loaded("gpu") && inplace && !first_scalar
                       ? staged_panels(root, parameter->var())
                       : DefVec();
    auto blocks_of = [&](const Def* var) {
        return staged_panels(old.app(std::get<0>(peel_step(step->reduce(var).back())), input), var);
    };
    // A ring of `depth` buffers per panel; `ring[s]` holds the block `s` ahead of the current one.
    std::vector<DefVec> ring;
    std::vector<size_t> pipelined;
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
        for (u64 s = 1; s < depth && !first.empty(); ++s) {
            DefVec buffers;
            for (auto panel : first) {
                auto [r, shape, T] = Axm::as<buffer::Buf>(panel->type())->args<3>();
                const Def* other;
                auto scope           = w.app(w.annex<gpu::shared_scope>(), mem);
                std::tie(mem, other) = buffer::op_alloc(r, shape, T, scope)->projs<2>();
                forward_layouts(panel, other);
                buffers.push_back(other);
            }
            ring.push_back(buffers);
        }
    }
    auto panels = ring.empty() ? size_t(0) : ring[0].size();
    // Fills block @p block's panels into @p targets after @p at and commits them as one group.
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
        return w.app(w.annex<gpu::async_commit>(), at);
    };
    for (u64 s = 0; s + 1 < ring.size(); ++s)
        mem = fill_block(mem, w.lit_nat(s), ring[s]);
    auto state = DefVec{mem, buffer};
    for (auto& buffers : ring)
        state.insert(state.end(), buffers.begin(), buffers.end());
    auto width       = state.size();
    auto [run, call] = btensor::ds_splice(w.tuple(state), w.sigma({mem->type(), buffer->type()}), "tileFold");
    auto start       = run->var(2, 0);
    auto exit        = run->ret_var();
    if (panels != 0) {
        auto done = w.mut_con(start->type())->set("tileFoldDone");
        done->app(true, exit, w.tuple({w.app(w.annex<gpu::async_wait>(), done->var(width, 0)), done->var(width, 1)}));
        exit = done;
    }
    auto remaining = first_scalar ? w.lit_nat(*literal_count - 1) : rewrite(count);
    auto [loop, loop_call]
        = affine::counting_for(w.call<core::bitcast>(w.type_i64(), remaining), start, exit, w.sym("tileFoldStep"));
    auto [index, carried, yield] = loop->vars<3>();
    auto loop_state              = carried->projs(width);
    auto loop_mem = loop_state[0], loop_buffer = loop_state[1];
    auto at_stage     = [&](u64 s, size_t j) { return loop_state[2 + s * panels + j]; };
    auto pending      = fe::Restore(pending_, {});
    auto nested_batch = fe::Restore<std::optional<Batch>>(batch_, std::nullopt);
    auto memo         = fe::Restore(fresh_memo_, {});
    auto cache        = fe::Restore(realizations_, realizations_);
    auto nested       = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
    push();
    bind_old(parameter->has_var());
    map(accumulator, loop_buffer);
    if (fragments) fragments_[loop_buffer] = *fragments;
    if (bounded) storage_shapes_[accumulator] = storage;
    auto iteration = w.call<core::bitcast>(w.type_nat(), index);
    if (first_scalar) iteration = w.call(core::nat::add, Defs{iteration, w.lit_nat(1)});
    map(offset, w.call(core::nat::mul, Defs{iteration, rewrite(stride)}));
    if (panels != 0) {
        for (size_t j = 0; j != panels; ++j) {
            map(stages[j], at_stage(0, j));
            for (u64 s = 0; s != ring.size(); ++s)
                forward_layouts(ring[0][j], at_stage(s, j));
        }
        // The current block's group is the oldest: the younger `depth - 2` may still be pending.
        auto waited = w.app(w.app(w.annex<gpu::async_wait_group>(), w.lit_nat(ring.size() - 2)), loop_mem);
        loop_mem    = w.app(w.annex<gpu::fence>(), waited);
        // The last blocks fill their own panels once more, which keeps the loop body free of a branch.
        auto next = w.app(w.annex(tensor::pipeline_plan::refill_block),
                          {iteration, w.lit_nat(ring.size()), w.lit_nat(*literal_count)});
        auto free = DefVec(panels, [&](size_t j) { return at_stage(ring.size() - 1, j); });
        loop_mem  = fill_block(loop_mem, next, free);
    }
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
    if (panels != 0) {
        auto next = DefVec{update->proj(2, 0), update->proj(2, 1)};
        for (u64 s = 0; s != ring.size(); ++s)
            for (size_t j = 0; j != panels; ++j)
                next.push_back(at_stage(Lit::as(w.app(w.annex(tensor::pipeline_plan::rotated_source),
                                                      {w.lit_nat(s), w.lit_nat(ring.size())})),
                                        j));
        update = w.tuple(next);
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
std::tuple<const Def*, u64, const Def*, u64> LowerToMem::peel_step(const Def* body) const {
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
    return {stepped, level, merge, depth};
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
    if (!inplace) app->blame("tensor.tile_fold: a lane reduction needs an in-place step").bail();
    if (!seed) app->blame("tensor.tile_fold: a lane reduction needs a scalar seed, its combine's identity").bail();
    if (!mr_app) app->blame("tensor.tile_fold: a lane reduction's step must be a map_reduce_post").bail();
    if (!merge) merge = merge_of(btensor::MapRedPost::of(mr_app->callee()->as<App>()).comb);
    if (!merge) app->blame("tensor.tile_fold: a lane reduction's combiner does not merge partials").bail();
    auto dims = std::vector<u64>(rank);
    u64 elems = 1;
    for (size_t d = 0; d != rank; ++d) {
        auto lit = Lit::isa(storage->proj(rank, d));
        if (!lit) app->blame("tensor.tile_fold: a lane reduction needs a literal tile").bail();
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
        for (auto mask = *n / 2; mask != 0; mask /= 2) {
            auto other = w.app(w.annex<gpu::shfl_xor>(), w.tuple({value, w.lit_i32(u32(mask))}));
            value      = w.app(op, w.tuple({value, other}));
        }
        merged = buffer::op_write(read, out, index, value)->proj(2, 0);
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
    if (bound_level_ != 1)
        app->blame("tensor.tile_fold: a work-item bound reduction must nest directly inside a work-group level").bail();
    if (!inplace) app->blame("tensor.tile_fold: a work-item bound reduction needs an in-place step").bail();
    if (!n || !*n || *n > 1024)
        app->blame("tensor.tile_fold: a work-item bound reduction needs a literal block count of at most 1024").bail();
    if (!seed)
        app->blame("tensor.tile_fold: a work-item bound reduction needs a scalar seed, its combine's identity").bail();
    if (!mr_app) app->blame("tensor.tile_fold: a work-item bound reduction's step must be a map_reduce_post").bail();
    auto mr = btensor::MapRedPost::of(mr_app->callee()->as<App>());
    if (!merge) merge = merge_of(mr.comb);
    if (!merge)
        app->blame(
               "tensor.tile_fold: a work-item bound reduction's combiner must be `op (acc, e)` with `e` free of `acc`, "
               "`op` then merging the work-items' partials; got `{}`",
               mr.comb)
            .bail();
    auto dims = std::vector<u64>(rank);
    for (size_t d = 0; d != rank; ++d) {
        auto lit = Lit::isa(storage->proj(rank, d));
        if (!lit) app->blame("tensor.tile_fold: a work-item bound reduction needs a literal tile").bail();
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
    // One superstep writes the partials and each merge round claims them for one more.
    auto g     = w.lit_nat(batches_++);
    u64 epoch  = 0;
    auto panel = [&](auto axiom) { return w.app(w.app(axiom, {g, w.lit_nat(epoch)}), {w.lit_nat(prank), pshape, el}); };
    if (entered_++) app->blame("a work-item bound reduction cannot run inside a work-group batch").bail();
    const Def* token     = w.app(w.app(w.annex<gpu::enter_write>(), {g, w.lit_nat(epoch)}),
                                 w.app(w.app(w.annex<gpu::group_enter>(), g), start));
    auto fresh           = w.app(panel(w.annex<gpu::shared_panel>()), token);
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
        auto level2        = fe::Restore(bound_level_, u64(2));
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
    {
        // The private tile goes to the work-item's rows of the partials.
        Lam* cur        = store;
        const Def* exit = yield;
        const Def* acc  = store->var();
        auto iters      = affine::counting_nest(cur, exit, acc, literal_storage->projs(rank), "boundFoldStore");
        auto [rm, value]
            = buffer::op_read(acc, update->proj(2, 1), btensor::folded_idx(literal_storage, iters))->projs<2>();
        auto at = DefVec(iters.begin(), iters.end());
        if (rank) {
            auto base = w.call(core::wrap::mul, core::Mode::none, Defs{index, w.lit_i64(rows)});
            at[0]     = w.call(core::wrap::add, core::Mode::none, Defs{base, at[0]});
        } else {
            at.push_back(index);
        }
        auto wm = buffer::op_write(rm, run_partials, btensor::folded_idx(pshape, at), value)->proj(2, 0);
        cur->app(true, exit, wm);
    }
    auto [filled, partials]
        = w.app(panel(w.annex<gpu::coop_fill>()), w.tuple({fresh->proj(2, 0), fresh->proj(2, 1), fill}))->projs<2>();
    token                = w.app(w.app(w.annex<gpu::sync_publish>(), {g, w.lit_nat(epoch)}), filled);
    const Def* published = w.app(panel(w.annex<gpu::publish_view>()), w.tuple({token, partials}));
    auto op              = rewrite(merge);
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
    auto partial_at = [&](const Def* t, const DefVec& coords) {
        auto at = DefVec(coords.begin(), coords.end());
        if (rank)
            at[0] = w.call(core::wrap::add, core::Mode::none,
                           Defs{w.call(core::wrap::mul, core::Mode::none, Defs{t, w.lit_i64(rows)}), at[0]});
        else
            at.push_back(t);
        return btensor::folded_idx(pshape, at);
    };
    auto item_loop = [&](Lam* from, bool first, u64 trip, const Def* mem, const Def* exit, const char* name) {
        auto body    = w.mut_con({w.type_i64(), acc_ty, w.cn(acc_ty)})->set(name);
        auto wrapped = w.app(w.app(w.annex<gpu::bind>(gpu::bind::item), body->type()), body);
        from->set(first, w.call<affine::For>(wrapped, exit, Defs{w.lit_i64(0), w.lit_i64(trip), w.lit_i64(1), mem}));
        return body;
    };
    auto combine                     = w.mut_lam(w.pi(acc_ty, w.sigma({acc_ty, w.sigma()})))->set("boundCombine");
    auto [run_combine, combine_call] = btensor::ds_splice(combine->var(), acc_ty, "boundCombineRun");
    combine->set(true, w.tuple({combine_call, w.tuple()}));
    auto view_of = [&](const Def* buf) { return w.app(panel(w.annex<gpu::view>()), buf); };
    // A power-of-two work-item count merges its partials as a tree, halving the live partials behind each barrier.
    if (Lit::as(w.app(w.annex(tensor::r8_plan::is_tree), w.lit_nat(*n)))) {
        auto rounds = w.app(w.annex(tensor::r8_plan::rounds), {w.lit_nat(*n), w.lit_nat(elems)});
        for (auto round : rounds->projs()) {
            auto [active, s, pairs, trip] = round->projs<4>([](auto value) { return Lit::as(value); });
            if (!active) continue;
            auto merge_round             = w.mut_lam(w.pi(fill->var()->type(), acc_ty))->set("boundTreeRound");
            auto [round_mem, parts]      = merge_round->var()->projs<2>();
            auto [run_round, round_call] = btensor::ds_splice(w.tuple({round_mem, parts}), acc_ty, "boundTreeRun");
            merge_round->set(true, round_call);
            auto step = item_loop(run_round, true, trip, round_mem, run_round->ret_var(), "boundTreeItems");
            auto [t, tmem, tyield] = step->vars<3>();
            // The pair loop's exit is its own continuation; `lower_for` copies an exit's body into the loop.
            auto after = w.mut_con(acc_ty)->set("boundTreeAfter");
            after->app(true, tyield, after->var());
            auto qbody = w.mut_con({w.type_i64(), acc_ty, w.cn(acc_ty)})->set("boundTreePairs");
            step->set(true, w.call<affine::For>(qbody, after, Defs{t, w.lit_i64(pairs), w.lit_i64(trip), tmem}));
            auto [pair, qmem, qyield] = qbody->vars<3>();
            auto plan          = w.app(w.annex(tensor::r8_plan::pair_coords),
                                       {w.call<core::bitcast>(w.type_nat(), pair), w.lit_nat(elems), w.lit_nat(s)});
            auto [i, e, other] = plan->projs<3>([&](auto value) { return w.call<core::bitcast>(w.type_i64(), value); });
            auto coords        = coords_of(e);
            auto [m1, a]       = buffer::op_read(qmem, parts, partial_at(i, coords))->projs<2>();
            auto [m2, b]       = buffer::op_read(m1, parts, partial_at(other, coords))->projs<2>();
            auto merged        = w.app(op, w.tuple({a, b}));
            qbody->app(true, qyield, buffer::op_write(m2, parts, partial_at(i, coords), merged)->proj(2, 0));
            token        = w.app(w.app(w.annex<gpu::enter_update>(), {g, w.lit_nat(epoch)}), token);
            auto owned   = w.app(panel(w.annex<gpu::claim>()), w.tuple({token, published}));
            auto updated = w.app(panel(w.annex<gpu::owned_update>()), w.tuple({token, owned, merge_round}));
            auto carried = w.app(panel(w.annex<gpu::sync_carry>()), updated);
            ++epoch;
            token     = w.app(w.app(w.annex<gpu::enter_read>(), {g, w.lit_nat(epoch)}), carried->proj(2, 0));
            published = carried->proj(2, 1);
        }
        // One work-item per tile element publishes the first partial.
        auto citems = item_loop(run_combine, true, elems, combine->var(), run_combine->ret_var(), "boundCombineItems");
        auto [j, jmem, jyield] = citems->vars<3>();
        auto coords            = coords_of(j);
        auto [pmem, pval] = buffer::op_read(jmem, view_of(published), partial_at(w.lit_i64(0), coords))->projs<2>();
        citems->app(true, jyield, buffer::op_write(pmem, rbuf, target(coords), pval)->proj(2, 0));
    } else {
        // One work-item per tile element merges that element's partials in order.
        auto citems = item_loop(run_combine, true, elems, combine->var(), run_combine->ret_var(), "boundCombineItems");
        auto [j, jmem, jyield] = citems->vars<3>();
        auto coords            = coords_of(j);
        auto write             = w.mut_con({acc_ty, el})->set("boundCombineWrite");
        auto [wmem, wval]      = write->vars<2>();
        write->app(true, jyield, buffer::op_write(wmem, rbuf, target(coords), wval)->proj(2, 0));
        auto [tbody, tcall]
            = affine::counting_for(w.lit_i64(*n), w.tuple({jmem, rewrite(seed)}), write, w.sym("boundCombineStep"));
        citems->set(true, tcall);
        auto [t, tacc, tyield] = tbody->vars<3>();
        auto [tmem, tval]      = tacc->projs<2>();
        auto [pmem, pval]      = buffer::op_read(tmem, view_of(published), partial_at(t, coords))->projs<2>();
        tbody->app(true, tyield, w.tuple({pmem, w.app(op, w.tuple({tval, pval}))}));
    }
    auto at = w.lit_nat(epoch);
    auto read
        = w.app(w.app(w.app(w.annex<gpu::read_phase>(), {g, at}), w.sigma()), w.tuple({token, combine}))->proj(2, 0);
    auto reclaimed = w.app(w.app(w.annex<gpu::sync_reclaim>(), {g, at}), read);
    auto out_mem   = w.app(w.app(w.annex<gpu::group_leave>(), {g, w.lit_nat(epoch + 1)}), reclaimed);
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
    auto group = bound_level_ == 1 && driver().is_loaded("gpu");
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
