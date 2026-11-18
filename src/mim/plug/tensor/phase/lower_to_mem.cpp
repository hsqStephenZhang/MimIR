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

#include "lower_to_mem_util.h"

namespace mim::plug::tensor::phase {

using namespace lower_to_mem_detail;

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
    tma_store_  = driver().arg_nat("tensor", "tma-store").value_or(0) != 0;
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
    if (group_wide_) return true;
    return storage_level_ == 2 && item_origin_ && !value->has_free_var(item_origin_);
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
    if (storage_level_ == 0 || !driver().is_loaded("gpu")) return {};
    return Batch{start, batches_++};
}

// Publishes the batch's panels behind one barrier; @p consume computes `[mem, result]` while they are read.
const Def*
LowerToMem::close_batch(const Def* mem, const Def* result, const std::function<const Def*(const Def*)>& consume) {
    if (!batch_ || batch_->panels.empty()) return consume(mem);
    auto& w = new_world();
    auto M  = w.call<mem::M>(0);
    auto n  = batch_->panels.size();
    DefVec rs, ss, Ts, fills, views;
    for (const auto& panel : batch_->panels) {
        rs.push_back(panel.r), ss.push_back(panel.s), Ts.push_back(panel.T), fills.push_back(panel.fill);
        views.push_back(buffer::type_buf(panel.r, panel.s, panel.T));
    }
    auto reader  = w.mut_lam(w.pi(w.sigma({M, w.sigma(views)}), w.sigma({M, result})))->set("batchRead");
    auto [m, vs] = reader->vars<2>();
    reader->set(true, consume(m));
    // The consumer keys its panels' layouts by these holes, so they resolve only once it is built.
    for (size_t i = 0; i != n; ++i)
        batch_->panels[i].view->set(vs->proj(n, i));
    --entered_;
    // `gpu.superstep` is the protocol: enter, fill each panel, publish, read, leave.
    auto step = w.app(w.annex<gpu::superstep>(),
                      {w.lit_nat(batch_->group), w.lit_nat(n), w.tuple(rs), w.tuple(ss), w.tuple(Ts), result});
    return w.app(step, {batch_->start, w.tuple(fills), reader});
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
    if (Axm::isa<tensor::mma>(app) || Axm::isa<tensor::pipeline>(app) || Axm::isa<tensor::fill_tma>(app)
        || Axm::isa(tensor::law::member, app))
        return rewrite(app->arg());
    if (Axm::isa<tensor::prepack>(app)) return lower_prepack(app);
    // By bufferization every region exists, so an epilogue still waiting for one finishes its accumulator whole.
    if (Axm::isa(tensor::law::post_over, app)) return rewrite(post_over_whole(app->callee()->as<App>(), app->arg()));
    if (Axm::isa<tensor::group_req>(app)) {
        report_group_request(app, "request reached bufferization without realization");
        return rewrite(app->arg(5, 0));
    }
    if (Axm::isa<tensor::tile_req>(app)) {
        report_tile_request(app, "request reached bufferization without realization");
        auto op = app->arg(4, 0);
        if (auto marked = Axm::isa<tensor::merge_with>(op)) op = marked->arg();
        return rewrite(op);
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
    if (auto lane = Axm::isa<tensor::wgmma_lane>(app)) {
        // The fragments and accumulators are the lane's; the operand panel is read through descriptors.
        auto [a, b, at, c] = lane->args<4>();
        auto H             = rewrite(lane->decurry()->decurry()->decurry()->arg());
        auto nk            = Lit::as(a->type()->arity());
        auto nt            = Lit::as(c->type()->arity());
        auto buffer        = rewrite(b);
        auto logical       = logical_of(buffer);
        auto layout        = layout_of(buffer);
        if (!layout || !logical) app->blame("a wgmma operand panel needs its swizzled layout").bail();
        auto [br, bs, bT] = Axm::as<buffer::Buf>(buffer->type())->args<3>();
        auto kdim         = std::optional<u64>();
        if (auto i = wgmma_kdims_.find(Axm::peel<tensor::placed>(Axm::peel<tensor::buf>(b))); i != wgmma_kdims_.end())
            kdim = i->second;
        auto mn    = wgmma_major(buffer, kdim).first;
        auto place = layout_lam(layout, logical);
        auto none  = w.bot(w.call<mem::M>(0));
        auto instr = math::isa_bf16(H) ? w.annex<gpu::wgmma_rs_bf16>() : w.annex<gpu::wgmma_rs_f16>();
        auto v_a   = rewrite(a);
        auto kept  = w.tuple({rewrite(c), v_a});
        auto state = w.sigma({w.call<mem::M>(0), kept->type()});
        auto issue = DefVec(nk, [&](size_t kc) -> const Def* {
            auto logical_at = rewrite(at->proj(nk, kc));
            auto physical   = btensor::op_affine_map(place, br, br, logical, bs, logical_at, none)->proj(2, 1);
            auto desc       = wgmma_desc(buffer, *Shape(physical).fold(bs), kdim);
            auto lam        = w.mut_lam(w.pi(state, state))->set("wgmmaIssue");
            auto [mem, x]   = lam->vars<2>();
            auto [m, acc]   = w.app(instr, {w.lit_nat(nt), mem, w.extract(x->proj(2, 1), w.lit_idx(nk, kc)), desc,
                                            w.lit_bool(mn), x->proj(2, 0)})
                                  ->projs<2>();
            lam->set(true, w.tuple({m, w.tuple({acc, x->proj(2, 1)})}));
            return lam;
        });
        return wgmma_batch(fresh_mem(), kept, issue, w.lit_nat_0(), false, {buffer})->proj(2, 1)->proj(2, 0);
    }
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
    if (auto layout = layout_of(source)) {
        auto i64s     = DefVec(rank, [&](size_t d) { return w.call<core::bitcast>(w.type_i64(), nats[d]); });
        auto physical = layout_coords(layout, logical_of(source), i64s);
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
    auto storage      = bounded_shape(logical, storage_level_ != 0);
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
    // A fill that reads another fill cannot share its batch, which publishes both behind one barrier; outside a batch
    // each cooperative fill is fenced on both sides.
    auto reads_fill = [&] {
        auto queue = fe::BFSWorklist<DefSet>();
        queue.push(app->arg());
        while (!queue.empty()) {
            auto def = queue.pop();
            // A staged box is a fill even where it only feeds a producer computed here.
            if (Axm::isa<tensor::stage>(def)) return true;
            if (Axm::isa<tensor::get>(def) || Axm::isa<tensor::get_fused>(def)) {
                // A read through a view reads the fill behind it.
                auto source = def->as<App>()->arg(2, 0);
                for (bool more = true; more;) {
                    source = Axm::peel<tensor::buf>(source);
                    more   = false;
                    if (Axm::isa<tensor::slice>(source) || Axm::isa<tensor::slice_box>(source)
                        || Axm::isa<tensor::placed>(source) || Axm::isa<tensor::as_vectors>(source))
                        source = source->as<App>()->arg(), more = true;
                    else if (Axm::isa<tensor::tile_of>(source))
                        source = source->as<App>()->arg(3, 0), more = true;
                }
                if (Axm::isa<tensor::generate>(source)) return true;
            }
            if (def->isa<Var>() || def->isa<Axm>()) continue;
            for (auto op : def->deps())
                if (op) queue.push(op);
        }
        return false;
    };
    // Only a dependent fill leaves the batch; restoring it otherwise would drop the token this fill's panels advance.
    auto outside = std::optional<fe::Restore<std::optional<Batch>>>();
    if (batch_ && uniform_ && reads_fill()) outside.emplace(batch_, std::nullopt);
    auto [sources, captures] = capture_producers(app->arg());
    auto compute             = [&](Defs iters, const Def* input, const Def* m) -> std::pair<const Def*, const Def*> {
        auto nested = fe::Restore<std::vector<std::array<const Def*, 4>>*>(transients_, nullptr);
        // One work-item computes each element of a cooperative fill, so what it allocates is its own.
        auto private_scope = fe::Restore(uniform_, false);
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
        if (batch_->panels.empty()) {
            if (entered_++) app->blame("a work-group batch cannot open while another is open").n(unlowered).bail();
            if (!batch_->start) batch_->start = fresh_mem();
        }
        auto fill     = w.mut_lam(w.pi(result_ty, result_ty))->set("panelFill");
        auto [fm, fb] = fill->var()->projs<2>();
        auto chained  = [&w](const Def* body) {
            return w.app(w.app(w.annex<gpu::bind>(gpu::bind::chained), body->type()), body);
        };
        fill->set(true,
                  btensor::build_pointwise(result_ty, fm, fb, w.tuple(captures), s_out, rn, "generate", compute,
                                           bounded ? storage : nullptr, chained, inner ? *inner : size_t(-1), write));
        auto view = w.mut_hole(out_ty);
        batch_->panels.push_back({br, bs, bT, fill, view});
        filled = view;
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
    if (uniform_ && !async_fill_ && !probe_fill_) generic_panels_.insert(filled);
    if (layout) placements_[filled] = Placement{layout, *layout_view, layout_shape};
    if (inner && uniform_) panels_.insert(filled);
    return filled;
}

bool LowerToMem::nat_multiple(const Def* def, u64 v, u64 off) const {
    if (v == 1) return true;
    off %= v;
    if (auto lit = Lit::isa(def)) return (*lit + off) % v == 0;
    // A selection, such as a `min`, is one of its cases.
    if (auto select = def->isa<Extract>(); select && select->tuple()->isa<Tuple>())
        return std::ranges::all_of(select->tuple()->ops(), [&](const Def* x) { return nat_multiple(x, v, off); });
    if (auto nat = Axm::isa<core::nat>(def)) {
        auto [a, b] = nat->args<2>();
        if (nat.id() == core::nat::add) {
            if (auto c = Lit::isa(a)) return nat_multiple(b, v, off + *c);
            if (auto c = Lit::isa(b)) return nat_multiple(a, v, off + *c);
            return nat_multiple(a, v, off) && nat_multiple(b, v);
        }
        // A saturating difference of two multiples is one.
        if (nat.id() == core::nat::sub) return off == 0 && nat_multiple(a, v) && nat_multiple(b, v);
        if (nat.id() == core::nat::mul) return off == 0 && (nat_multiple(a, v) || nat_multiple(b, v));
        return false;
    }
    if (off != 0) return false;
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
    auto v  = 16 / *size;
    auto st = Lit::isa(step), rw = Lit::isa(row);
    if (!nat_multiple(extent, v) || !st || *st != 1 || !rw || *rw % v || align % v || !nat_multiple(start, v)) return 1;
    return v;
}

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
    if (layout_of(buffer)) buffer->blame("a panel in a storage layout can only be read element-wise").bail();
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
    // A computed accumulator, such as the full blocks' fold before a peeled tail, fills panels of its own first.
    if (initial && !splat_scalar(initial->outs)) rewrite(initial->outs);
    auto batch = fe::Restore<std::optional<Batch>>(batch_, open_batch(destination ? destination->mem : nullptr));
    auto args  = rewrite(app->arg()); // (is, post_is): the (bufferized) input buffers
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
    // An initial tensor a call computed, such as a fold's accumulator, leaves the memory after that call.
    const Def* init_mem = nullptr;
    if (initial) {
        if (auto scalar = splat_scalar(initial->outs)) {
            seed = rewrite(scalar);
        } else {
            auto outs = to_buffer(rewrite(initial->outs), initial->outs);
            plain(outs);
            if (auto made = outs->isa<Extract>(); made && Lit::isa(made->index()) == 1)
                if (auto call = made->tuple(); Lit::isa(call->type()->arity()) == 2)
                    if (Axm::isa<mem::M>(call->type()->proj(2, 0))) init_mem = call->proj(2, 0);
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
        auto layout  = layout_of(buffer);
        auto view    = view_of(buffer);
        auto r       = storage->num_projs();
        auto logical = layout ? logical_of(buffer) : storage;
        auto size    = Lit::isa(core::op(core::trait::size, mr.Tis->proj(nis, i)));
        auto last = Lit::isa(logical->proj(r, r - 1)), physical_last = Lit::isa(storage->proj(r, r - 1));
        auto n = size && 16 % *size == 0 ? 16 / *size : 1;
        // A vector runs along the logical last dim, so the layout must keep that dim innermost.
        auto keeps_last = !layout
                       || (view->perm.back() + 1 == view->perm.size()
                           && (view->split == 1 || (!view->axis && view->split % n == 0)));
        auto vectors    = panels_.contains(buffer) && n > 1 && last && *last % n == 0 && physical_last
                       && *physical_last % n == 0 && keeps_last;
        auto placement  = layout ? layout_lam(layout, logical) : w.app(w.annex<affine::id>(), w.lit_nat(r));
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
    auto [m, out] = close_batch(init_mem, buffer::type_buf(mr.Ro, mr.So, mr.Tp), own)->projs<2>();
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
        if (Axm::isa<tensor::map_reduce_post>(app)) {
            if (auto stored = store_epilogue(app, destination)) return stored;
            return lower_map_reduce(app, &destination);
        }
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
    auto in_group  = storage_level_ == 1 && driver().is_loaded("gpu");
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
    // A kernel captures the outermost region's own variable, so a TMA descriptor names what the kernel reads.
    auto outermost = storage_level_ == 0;
    if (!outermost) outer_of_[runtime->proj(2, 1)] = captured;
    forward_storage(sources, proxy);
    forward_layouts(captured, runtime->proj(2, 1));
    // Offsets live in the new world; extents stay in the old one, where the body is specialized to them.
    // A capped dimension leaves its extent null for the body to bound by its origin.
    DefVec offsets(*rank), extents(*rank);
    auto cap = Lit::isa(plan.tail) == 0;
    std::function<const Def*(size_t, const Def*, const Def*)> emit
        = [&](size_t d, const Def* acc, const Def* exit) -> const Def* {
        if (d == *rank) {
            // The loops carry the region's results in the buffers it started with.
            for (u64 i = 0; i != plan.count; ++i)
                outer_of_[acc->proj(plan.count + 1, i + 1)]
                    = outermost ? runtime->proj(2, 0)->proj(plan.count + 1, i + 1) : buffers[i];
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
        // A runtime extent, as an enclosing capped tile has, launches tiles up to its bound; those past it are skipped.
        auto runtime = fe::Vector<bool>(*rank, false);
        for (size_t d = 0; d != *rank; ++d) {
            if (sizes[d] == 0) return w.app(exit, acc);
            if (whole[d] || (sizes[d] && *sizes[d] == steps[d])) continue;
            auto size = sizes[d];
            if (!size) {
                size = nat_upper(shape->proj(*rank, d));
                if (!size)
                    app->blame("a bound tensor.tiled level requires literal or bounded extents").n(unlowered).bail();
                if (!cap)
                    app->blame("a bound tensor.tiled level over a runtime extent caps its tail").n(unlowered).bail();
                runtime[d] = true;
            }
            if (*size % steps[d] && !cap)
                app->blame("a bound tensor.tiled level cannot peel its tail").n(unlowered).bail();
            counts[d] = (*size + steps[d] - 1) / steps[d];
            total *= counts[d];
        }
        // A specialized warpgroup region adds a producer warp, which takes the first tile's coordinates.
        auto special = std::optional<u64>();
        if (auto bound = Axm::isa<tensor::bind>(app->arg()); bound && plan.bind == 6)
            if (auto marked = Axm::isa<tensor::specialize>(bound->arg())) special = Lit::as(marked->decurry()->arg());
        auto body    = w.mut_con({w.type_i64(), acc->type(), w.cn(acc->type())})
                           ->set(plan.bind == 1                     ? "tileGroups"
                                 : plan.bind == 2                   ? "tileItems"
                                 : plan.bind == 6 || plan.bind == 7 ? "tileWarpgroups"
                                                                    : "tileWarps");
        auto tag     = plan.bind == 1                   ? gpu::bind::group
                     : plan.bind == 2                   ? gpu::bind::item
                     : plan.bind == 6 && special        ? gpu::bind::specialized
                     : plan.bind == 6 || plan.bind == 7 ? gpu::bind::warpgroup
                                                        : gpu::bind::warp;
        auto wrapped = w.app(w.app(w.annex<gpu::bind>(tag), body->type()), body);
        // A work-item whose reduction runs on its lanes takes that many adjacent ids, which must fill whole warps.
        if (lanes > 1 && total * lanes % 32)
            app->blame("the {} work-items' lanes of a lane reduction do not fill whole warps", total).n(target).bail();
        // A cluster that splits each tile's reduction runs `size` work-groups, consecutive ids, per tile.
        auto cluster = Cluster{};
        if (auto bound = Axm::isa<tensor::bind>(app->arg()); bound && plan.bind == 1)
            if (auto marked = Axm::isa<tensor::cluster>(bound->arg())) {
                auto [size, multicast, split] = marked->decurry()->args<3>([](const Def* d) { return Lit::as(d); });
                if (!split && total % size)
                    app->blame("{} work-groups do not form clusters of {}", total, size).n(target).bail();
                cluster = Cluster{size, multicast, split ? size : 1};
            }
        u64 ctas = 0;
        if (auto bound = Axm::isa<tensor::bind>(app->arg()); bound && plan.bind == 1)
            if (auto marked = Axm::isa<tensor::persistent>(bound->arg())) {
                ctas = std::min(Lit::as(marked->decurry()->arg()), total);
                if (Axm::isa<tensor::cluster>(marked->arg()))
                    app->blame("a persistent work-group level does not run in clusters").n(unlowered).bail();
            }
        auto extra    = special ? u64(1) : u64(0);
        auto launched = ctas ? ctas : total * lanes * cluster.split + extra;
        auto loop     = w.call<affine::For>(wrapped, exit, Defs{w.lit_i64(0), w.lit_i64(launched), w.lit_i64(1), acc});
        auto [index, state, yield] = body->vars<3>();
        const Def* linear          = w.call<core::bitcast>(w.type_nat(), index);
        if (cluster.split > 1) linear = w.call(core::nat::div, Defs{linear, w.lit_nat(cluster.split)});
        // A persistent work-group runs the tiles `ctas` apart from its own in turn.
        Lam* tile            = body;
        const Def* tile_acc  = state;
        const Def* tile_exit = yield;
        const Def* round     = nullptr;
        const Def* id        = linear;
        // `gpu.persistent_tiles` is the protocol of the rounds; what a tile computes comes from here.
        if (ctas) {
            tile = w.mut_con({w.type_nat(), w.type_nat(), state->type(), w.cn(state->type())})->set("tileRound");
            std::tie(linear, round, tile_acc, tile_exit) = tile->vars<4>();
        }
        // The producer warp makes a warp count that is no multiple of four, so one sub-partition's register file holds
        // more warps than the others.
        if (special) {
            auto per_partition = (4 * total + 1 + 3) / 4;
            auto acc_state     = DefVec(tile_acc->projs(plan.count + 1));
            auto budget        = std::min<u64>(255, 16384 / (32 * per_partition) / 8 * 8);
            if (auto cap = driver().arg_nat("tensor", "maxnreg")) budget = std::min<u64>(budget, *cap);
            acc_state[0] = w.app(w.app(w.annex<gpu::reg_budget>(), w.lit_nat(budget)), acc_state[0]);
            tile_acc     = w.tuple(acc_state);
        }
        auto rounding = fe::Restore(tile_round_, ctas ? round : tile_round_);
        // Only the persistent level resets the flag, so a specialized reduction below may clear it for this level.
        auto fenced = round_fence_;
        if (ctas) round_fence_ = true;
        const Def* producer = nullptr;
        if (special) {
            if (*special && total != 2)
                app->blame("warpgroup ping-pong alternates two consumers, not {}", total).n(unlowered).bail();
            producer = w.call(core::ncmp::e, Defs{linear, w.lit_nat(total)});
            linear   = w.call(core::nat::rem, Defs{linear, w.lit_nat(total)});
        }
        auto roles      = fe::Restore(roles_, special ? Roles{producer, linear, total, *special != 0} : roles_);
        const Def* lane = lane_;
        if (lanes > 1) {
            lane   = w.call(core::nat::rem, Defs{linear, w.lit_nat(lanes)});
            linear = w.call(core::nat::div, Defs{linear, w.lit_nat(lanes)});
        }
        auto lane_scope  = fe::Restore(lane_, lane);
        auto lanes_scope = fe::Restore(lanes_, lanes > 1 ? lanes : lanes_);
        auto stride      = total;
        // A work-group whose operands TMA fills computes its last, partial tiles in full, reading zeros past the
        // edge, and stores only the rows and columns inside.
        auto padded = false;
        for (size_t d = 0; d != *rank; ++d)
            padded |= plan.bind == 1 && sizes[d] && *sizes[d] % steps[d] && fills_with_tma(app->arg());
        auto padding = fe::Restore(padded_, padded_ || padded);
        for (size_t d = 0; d != *rank; ++d) {
            if (counts[d] == 1) {
                // A one-tile bound region depends on its binder in every dim, so nothing it stages floats out.
                // The binder is 0 there, and scaling it by the step keeps the offset a known multiple of the step.
                offsets[d] = total != 1 ? w.lit_nat_0()
                           : d == 0     ? linear
                                        : w.call(core::nat::mul, Defs{linear, w.lit_nat(steps[d])});
                extents[d] = whole[d] ? shape->proj(*rank, d) : runtime[d] ? nullptr : old.lit_nat(steps[d]);
                continue;
            }
            stride /= counts[d];
            auto quotient = w.call(core::nat::div, Defs{linear, w.lit_nat(stride)});
            auto tile     = w.call(core::nat::rem, Defs{quotient, w.lit_nat(counts[d])});
            offsets[d]    = w.call(core::nat::mul, Defs{tile, w.lit_nat(steps[d])});
            extents[d]    = runtime[d] || (*sizes[d] % steps[d] && !padded) ? nullptr : old.lit_nat(steps[d]);
        }
        // A warp's lanes share its tile like a work-item's own; only the tensor-core lowering tells them apart.
        // A work-group's only warpgroup is the work-group, so it fills the work-group's panels as one; a producer
        // warpgroup beside it would never reach the work-group's barriers.
        auto whole     = plan.bind == 1 || (plan.bind == 6 && total == 1 && !special && group_wide_);
        auto level     = fe::Restore(storage_level_, whole ? 1 : std::min<u64>(plan.bind, 2));
        auto wide      = fe::Restore(group_wide_, whole);
        auto clustered = fe::Restore(cluster_, plan.bind == 1 ? cluster : cluster_);
        auto threads   = fe::Restore(group_threads_, plan.bind == 6 ? 128 * total : group_threads_);
        auto owned = fe::Restore(warp_self_, plan.bind == 5 || plan.bind == 7 ? warp_lane(old_world()) : warp_self_);
        auto& old  = old_world();
        auto first
            = old.call(core::nat::mul,
                       Defs{old.call<core::bitcast>(old.type_nat(), old.app(old.annex<gpu::warp_rank>(), old.tuple())),
                            old.lit_nat(16)});
        auto groups       = fe::Restore(group_rows_, plan.bind == 7 ? first : plan.bind == 5 ? nullptr : group_rows_);
        const Def* inside = w.lit_tt();
        for (size_t d = 0; d != *rank; ++d)
            if (runtime[d])
                inside = w.call(
                    core::bit2::and_, w.lit_nat(2),
                    w.tuple({inside, w.call(core::ncmp::l, Defs{offsets[d], rewrite(shape->proj(*rank, d))})}));
        if (inside == w.lit_tt()) {
            tile->set(true, emit(*rank, tile_acc, tile_exit));
        } else {
            auto run  = w.mut_con(tile_acc->type())->set("tileInside");
            auto skip = w.mut_con(tile_acc->type())->set("tileOutside");
            run->set(true, emit(*rank, run->var(), tile_exit));
            skip->app(true, tile_exit, skip->var());
            tile->app(true, w.extract(w.tuple({skip, run}), inside), tile_acc);
        }
        if (ctas) {
            auto settle = w.mut_lam(w.pi(state->type(), state->type()))->set("tileRoundSettle");
            auto next   = DefVec(settle->var()->projs(plan.count + 1));
            if (round_fence_) next[0] = w.app(w.annex<gpu::fence>(), next[0]);
            round_fence_ = fenced;
            settle->set(true, w.tuple(next));
            auto types
                = w.tuple(DefVec(plan.count + 1, [&](size_t i) { return state->proj(plan.count + 1, i)->type(); }));
            auto rounds = w.app(w.annex<gpu::persistent_tiles>(), {w.lit_nat(plan.count + 1), types});
            rounds      = w.app(w.app(rounds, {w.lit_nat(total), w.lit_nat(ctas), settle}), tile);
            body->app(true, rounds, {id, state, yield});
        }
        return loop;
    };
    auto done = w.mut_con(call->type())->set("tileScratchDone");
    // A work-group level inside a work-group, as a `group_tile_req` has, is its one tile and runs in place.
    auto bound = plan.bind && driver().is_loaded("gpu") && !(plan.bind == 1 && storage_level_ != 0);
    // Work-items that fill a work-group's buffer publish it to the others behind a barrier.
    const Def* exit = done;
    auto shared_out = std::ranges::any_of(buffers, [&](const Def* b) { return group_buffers_.contains(b); });
    if (bound && plan.bind >= 2 && group_wide_ && (shared_out || composed_)) {
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
        if (auto placed = fe::lookup(placements_, buffer)) {
            auto copied        = *placed;
            placements_[param] = std::move(copied);
        }
        if (panels_.contains(buffer)) panels_.insert(param);
        if (generic_panels_.contains(buffer)) generic_panels_.insert(param);
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
