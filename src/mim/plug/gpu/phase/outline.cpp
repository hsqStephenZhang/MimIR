#include "mim/plug/gpu/phase/outline.h"

#include <fe/restore.h>

#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>
#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

std::optional<u64> trip_count(const Def* args) {
    auto [begin, end, step, _] = args->projs<4>();
    auto b = Lit::isa(begin), e = Lit::isa(end), s = Lit::isa(step);
    if (!b || !e || !s || *b != 0 || *s != 1) return {};
    return *e;
}

const App* bound_loop(const Def* def, bind id) {
    if (auto loop = Axm::isa<affine::For>(def))
        if (auto bound = Axm::isa<gpu::bind>(loop->uncurry_args<3>()[0]); bound && bound.id() == id)
            return def->as<App>();
    return nullptr;
}

/// The work-item loop nested in a work-group loop's body, if any.
// The largest work-item loop count under @p def; a shorter work-item loop leaves the surplus items idle.
std::optional<u64> item_count(const Def* def, DefSet& seen) {
    if (auto [_, ins] = seen.emplace(def); !ins) return {};
    auto count = std::optional<u64>();
    auto warp  = bound_loop(def, bind::warp);
    if (auto loop = warp ? warp : bound_loop(def, bind::item)) {
        count = trip_count(loop->uncurry_args<3>()[2]);
        if (!count) loop->blame("gpu.outline requires a literal work-item count").bail();
        if (warp) *count *= 32;
    }
    for (auto op : def->ops()) {
        if (!op || op->isa<Var>()) continue;
        if (auto inner = item_count(op, seen)) count = std::max(count.value_or(0), *inner);
    }
    return count;
}

size_t mem_index(const Def* acc) {
    auto n = acc->num_projs();
    for (size_t i = 0; i != n; ++i)
        if (Axm::isa<mem::M>(acc->proj(n, i)->type())) return i;
    acc->blame("gpu.outline expects the loop state to carry a memory").bail();
}

} // namespace

/// The maximal subterms of a loop body that are not closed but free of every var the body binds.
void Outline::captures(const Def* def, const Vars& outer, DefSet& seen, DefVec& pieces) {
    if (auto [_, ins] = seen.emplace(def); !ins) return;
    if (def->is_closed()) return;
    bool captured = true;
    for (auto var : def->free_vars())
        captured &= outer.contains(var);
    if (!captured) {
        for (auto op : def->ops())
            if (op) captures(op, outer, seen, pieces);
        return;
    }
    // A captured tuple is passed by its components, so each pointer among them is copied to the device on its own.
    if (auto n = def->num_projs(); n > 1 && !Axm::isa<mem::Ptr>(def->type())) {
        for (size_t i = 0; i != n; ++i)
            if (auto proj = def->proj(n, i); !Axm::isa<mem::M>(proj->type())) captures(proj, outer, seen, pieces);
        splits_.push_back(def);
        return;
    }
    pieces.push_back(def);
}

// An `async` loop only fills work-group panels, so its stores go to shared memory.
const Def* Outline::store_async(const App* app) {
    auto& w            = new_world();
    auto store         = Axm::isa<mem::store>(app);
    auto [m, dst, val] = store->args<3>();
    auto T             = rewrite(store->decurry()->arg(2, 0));
    return w.app(w.app(w.annex<gpu::store_async>(), T), {rewrite(m), rewrite(dst), rewrite(val)});
}

/// Barrier on the generic chain: every work-item's writes before @p mem are visible after the result.
const Def* Outline::sync(const Def* mem) {
    auto& w     = new_world();
    auto synced = w.app(w.annex<gpu::sync_work_items>(), Defs{w.app(w.annex<gpu::global_mem>(), mem), k_shared_});
    return w.app(w.annex<gpu::generic_mem>(), synced->proj(2, 0));
}

const Def* Outline::rewrite_imm_App(const App* app) {
    auto& w = new_world();
    if (bound_loop(app, bind::group)) {
        if (item_id_) app->blame("gpu.outline: a work-group loop cannot nest inside a kernel").bail();
        return outline(app);
    }
    if (Axm::isa<gpu::lane_id>(app)) {
        if (!lane_) app->blame("gpu.outline: a lane id must nest inside a warp loop").bail();
        return lane_;
    }
    auto warp = bound_loop(app, bind::warp);
    if (auto loop = warp ? warp : bound_loop(app, bind::item)) {
        if (!item_id_) app->blame("gpu.outline: a work-item loop must nest inside a work-group loop").bail();
        auto [annotated, exit, args] = loop->uncurry_args<3>();
        auto body                    = Axm::isa<gpu::bind>(annotated)->arg()->isa_mut<Lam>();
        if (!body) app->blame("gpu.outline requires a lambda loop body").bail();
        auto init   = args->proj(4, 3);
        auto inside = fe::Restore(in_item_, true);
        auto width  = warp ? 32 : 1;
        auto index  = warp ? w.call(core::shr::l, Defs{item_id_, w.lit_i64(5)}) : item_id_;
        auto lane  = fe::Restore(lane_, warp ? w.call(core::bit2::and_, core::Mode::none, Defs{item_id_, w.lit_i64(31)})
                                             : lane_);
        auto count = trip_count(args);
        if (count) *count *= width;
        if (!count || *count >= n_items_) {
            push();
            map(body->var(), {index, rewrite(init), rewrite(exit)});
            auto value = rewrite(body->body());
            pop();
            return value;
        }
        // Surplus work-items skip a shorter loop and pass its state through. The state is the branch's argument, so
        // the barriers and shared slots on its memory chain run before the work-items diverge.
        auto new_init = rewrite(init);
        auto skip     = w.mut_con(new_init->type())->set("itemSkip");
        skip->app(true, rewrite(exit), skip->var());
        auto run          = w.mut_con(new_init->type())->set("itemRun");
        auto divergent    = fe::Restore(item_divergent_, true);
        auto warp_aligned = fe::Restore(warp_aligned_, warp_aligned_ && *count % 32 == 0);
        push();
        map(body->var(), {index, run->var(), rewrite(exit)});
        run->set(false, rewrite(body->body()));
        pop();
        auto in_range = w.call(core::icmp::ul, Defs{item_id_, w.lit_i64(*count)});
        return w.app(w.extract(w.tuple({skip, run}), in_range), new_init);
    }
    if (in_async_ && Axm::isa<mem::store>(app)) return store_async(app);
    if (Axm::isa<gpu::sync_work_items>(app) && item_divergent_)
        app->blame("gpu.outline: a work-item barrier cannot execute under a divergent item guard").bail();
    if (auto fence = Axm::isa<gpu::fence>(app)) {
        if (!item_id_) app->blame("gpu.outline: a fence must nest inside a work-group loop").bail();
        if (item_divergent_)
            app->blame("gpu.outline: a work-item barrier cannot execute under a divergent item guard").bail();
        return sync(rewrite(fence->arg()));
    }
    if ((Axm::isa<gpu::ldmatrix>(app) || Axm::isa<gpu::mma_tf32>(app) || Axm::isa<gpu::buf_ldmatrix>(app)
         || Axm::isa<gpu::shfl_xor>(app) || Axm::isa<gpu::shfl_idx>(app))
        && item_divergent_ && !warp_aligned_)
        app->blame("gpu.outline: a warp collective cannot execute under a non-warp-aligned item guard").bail();
    auto async   = bound_loop(app, bind::async);
    auto chained = async ? async : bound_loop(app, bind::chained);
    if (auto loop = chained ? chained : bound_loop(app, bind::cooperative)) {
        if (!item_id_) app->blame("gpu.outline: a cooperative loop must nest inside a work-group loop").bail();
        if (item_divergent_ && !chained)
            app->blame("gpu.outline: a work-item barrier cannot execute under a divergent item guard").bail();
        auto [annotated, exit, args]  = loop->uncurry_args<3>();
        auto body                     = Axm::isa<gpu::bind>(annotated)->arg();
        auto [begin, end, step, init] = args->projs<4>();
        auto new_init                 = rewrite(init);
        auto n                        = new_init->num_projs();
        auto mem_i                    = mem_index(init);
        auto with_mem                 = [&](const Def* acc, const Def* mem) {
            return w.tuple(DefVec(n, [&](size_t i) { return i == mem_i ? mem : acc->proj(n, i); }));
        };
        auto done = rewrite(exit);
        if (!chained) {
            auto after = w.mut_con(new_init->type())->set("cooperativeDone");
            after->app(true, done, with_mem(after->var(), sync(after->var()->proj(n, mem_i))));
            done = after;
        }
        auto entry    = chained ? new_init : with_mem(new_init, sync(new_init->proj(n, mem_i)));
        auto copying  = fe::Restore(in_async_, async != nullptr);
        auto new_body = rewrite(body);
        // A literal per-item trip count lets the backend unroll the fill and batch its loads.
        if (auto count = trip_count(args); count && *count % n_items_ == 0) {
            auto step            = w.mut_con(new_body->type()->as<Pi>()->dom())->set("cooperativeStep");
            auto [j, acc, yield] = step->vars<3>();
            auto offset          = w.call(core::wrap::mul, core::Mode::none, Defs{j, w.lit_i64(n_items_)});
            auto i               = w.call(core::wrap::add, core::Mode::none, Defs{item_id_, offset});
            step->app(true, new_body, {i, acc, yield});
            auto counted = w.tuple({w.lit_i64(0), w.lit_i64(*count / n_items_), w.lit_i64(1), entry});
            return w.call<affine::For>(step, done, counted);
        }
        auto strided = w.tuple({item_id_, rewrite(end), w.lit_i64(n_items_), entry});
        return w.call<affine::For>(new_body, done, strided);
    }
    // A work-group's shared allocations become static slots; their frees vanish with the kernel.
    if (auto alloc = Axm::isa<mem::alloc>(app); alloc && item_id_) {
        auto [T, a]     = alloc->decurry()->args<2>();
        const Def* slot = nullptr;
        if (auto scope = Axm::isa<gpu::shared_scope>(alloc->arg()))
            slot = w.app(w.app(w.annex<gpu::slot_shared>(), rewrite(T)), rewrite(scope->arg()));
        // The slot keeps a panel's construct for the backend to overlay the panels of different constructs.
        if (auto scope = Axm::isa<gpu::panel_scope>(alloc->arg()))
            slot = w.app(
                w.app(w.annex<gpu::slot_shared>(), rewrite(T)),
                w.app(w.app(w.annex<gpu::panel_scope>(), rewrite(scope->decurry()->arg())), rewrite(scope->arg())));
        if (slot) {
            shared_.emplace(slot->proj(2, 1));
            return slot;
        }
    }
    if (auto scope = Axm::isa<gpu::shared_scope>(app)) return rewrite(scope->arg());
    if (auto scope = Axm::isa<gpu::panel_scope>(app)) return rewrite(scope->arg());
    if (auto free = Axm::isa<mem::free>(app); free && item_id_) {
        auto [mem, ptr] = free->args<2>();
        if (shared_.contains(rewrite(ptr))) return rewrite(mem);
    }
    return Super::rewrite_imm_App(app);
}

const Def* Outline::outline(const App* app) {
    auto& w                      = new_world();
    auto [annotated, exit, args] = app->uncurry_args<3>();
    auto body                    = Axm::isa<gpu::bind>(annotated)->arg()->isa_mut<Lam>();
    if (!body) app->blame("gpu.outline requires a lambda loop body").bail();
    auto n_groups = trip_count(args);
    if (!n_groups) app->blame("gpu.outline requires a literal work-group count").bail();
    DefSet seen;
    u64 n_items = item_count(body->body(), seen).value_or(1);
    if (n_items > 1024) app->blame("gpu.outline: {} work-items exceed a work-group", n_items).bail();

    auto init  = args->proj(4, 3);
    auto n_acc = init->num_projs();
    auto mem_i = mem_index(init);
    DefVec pieces;
    seen.clear();
    captures(body->body(), body->free_vars(), seen, pieces);
    for (size_t i = 0; i != n_acc; ++i)
        if (i != mem_i && !init->proj(n_acc, i)->is_closed()) pieces.push_back(init->proj(n_acc, i));
    DefSet unique;
    std::erase_if(pieces, [&](const Def* piece) { return !unique.emplace(piece).second; });

    auto mem_ty    = w.call<mem::M>(0);
    auto global_ty = w.annex<gpu::GlobalM>();
    auto shared_ty = w.annex<gpu::SharedM>();
    auto const_ty  = w.annex<gpu::ConstM>();
    auto local_ty  = w.annex<gpu::LocalM>();

    auto new_init     = rewrite(init);
    auto new_exit     = rewrite(exit);
    auto [m0, m1, m4] = w.app(w.annex<gpu::auto_init>(), new_init->proj(n_acc, mem_i))->projs<3>();
    DefVec kernel_args, host_ptrs, dev_ptrs, dev_tys;
    for (auto piece : pieces) {
        auto value = rewrite(piece);
        if (auto ptr = Axm::isa<mem::Ptr>(value->type())) {
            auto [T, a] = ptr->args<2>();
            if (Lit::isa(a) != 0)
                piece->blame("gpu.outline: captured pointer `{}` is not in host memory", piece).bail();
            auto [m1_alloc, dev] = w.app(w.app(w.annex<mem::alloc>(), {T, w.lit_nat(1)}), m1)->projs<2>();
            auto copy
                = w.app(w.app(w.annex<gpu::copy_to_device>(gpu::copy_to_device::block), T), {m0, m1_alloc, value, dev});
            std::tie(m0, m1) = copy->projs<2>();
            host_ptrs.push_back(value);
            dev_ptrs.push_back(dev);
            dev_tys.push_back(T);
            kernel_args.push_back(dev);
        } else if (mem::has_mem_ty(value->type()) || value->type()->isa<Pi>()) {
            piece->blame("gpu.outline cannot pass `{}` of type `{}` to a kernel", piece, value->type()).bail();
        } else {
            kernel_args.push_back(value);
        }
    }
    auto n_args  = kernel_args.size();
    auto arg_tys = DefVec(n_args, [&](size_t i) { return kernel_args[i]->type(); });

    log().d("gpu.outline {}: {} groups × {} items, {} device buffers, {} arguments", body, *n_groups, n_items,
            dev_ptrs.size(), n_args);
    auto kernel = w.mut_con(Defs{global_ty, shared_ty, const_ty, local_ty, w.type_idx(*n_groups), w.type_idx(n_items),
                                 w.sigma(Defs{}), w.sigma(arg_tys), w.cn({global_ty, shared_ty, const_ty, local_ty})})
                      ->set("tileKernel");
    auto [k_global, k_shared, k_const, k_local, group_id, item_id, k_smem, k_args, k_ret] = kernel->vars<9>();
    auto group_i64 = *n_groups == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), group_id);
    auto item_i64  = n_items == 1 ? w.lit_i64(0) : w.call(core::conv::u, w.lit_nat_0(), item_id);

    push();
    for (size_t i = 0, p = 0; i != pieces.size(); ++i) {
        auto arg = k_args->proj(n_args, i);
        if (p != dev_ptrs.size() && kernel_args[i] == dev_ptrs[p]) {
            arg = w.call<core::bitcast>(rewrite(pieces[i]->type()), arg);
            ++p;
        }
        map(pieces[i], arg);
    }
    // Components come before the tuples they split off, so a whole tuple rebuilds from kernel arguments.
    for (auto split : splits_) {
        auto n = split->num_projs();
        map(split, w.tuple(DefVec(n, [&](size_t i) {
            auto proj = split->proj(n, i);
            return Axm::isa<mem::M>(proj->type()) ? w.bot(rewrite(proj->type())) : rewrite(proj);
        })));
    }
    splits_.clear();
    // The body runs on a generic chain of its own, so `mem.resolve_fresh` threads it like any host function.
    auto k_body = w.mut_con(mem_ty)->set("tileKernelBody");
    auto k_init
        = w.tuple(DefVec(n_acc, [&](size_t i) { return i == mem_i ? k_body->var() : rewrite(init->proj(n_acc, i)); }));
    auto k_exit = w.mut_con(k_init->type())->set("tileKernelExit");
    k_exit->app(true, k_ret,
                Defs{w.app(w.annex<gpu::global_mem>(), k_exit->var()->proj(n_acc, mem_i)), k_shared, k_const, k_local});
    auto item_scope = fe::Restore(item_id_, item_i64);
    k_shared_       = k_shared;
    n_items_        = n_items;
    map(body->var(), {group_i64, k_init, k_exit});
    // The body stays a call of its own, so `mem.resolve_fresh` threads it on its `mem.M 0` parameter.
    k_body->set(false, rewrite(body->body()));
    kernel->app(true, k_body, w.app(w.annex<gpu::generic_mem>(), k_global));
    pop();

    auto launch = w.app(w.annex<gpu::launch>(), Defs{w.lit_nat(n_args), w.tuple(arg_tys)});
    launch      = w.app(
        launch, Defs{w.lit_nat(*n_groups), w.lit_nat(n_items), w.annex<gpu::default_stream>(), w.lit_ff(), w.tuple()});
    launch = w.app(launch, kernel);
    launch = w.app(launch, w.tuple(kernel_args));

    auto after        = w.mut_con(Defs{mem_ty, global_ty, const_ty})->set("afterTileKernel");
    auto [a0, a1, a4] = after->vars<3>();
    for (size_t i = 0; i != dev_ptrs.size(); ++i) {
        auto copy        = w.app(w.app(w.annex<gpu::copy_to_host>(gpu::copy_to_host::block), dev_tys[i]),
                                 {a0, a1, dev_ptrs[i], host_ptrs[i]});
        std::tie(a0, a1) = copy->projs<2>();
    }
    for (size_t i = 0; i != dev_ptrs.size(); ++i)
        a1 = w.app(w.app(w.annex<mem::free>(), {dev_tys[i], w.lit_nat(1)}), {a1, dev_ptrs[i]});
    auto final_mem = w.app(w.annex<gpu::auto_deinit>(), Defs{a0, a1, a4});
    auto exit_acc = w.tuple(DefVec(n_acc, [&](size_t i) { return i == mem_i ? final_mem : new_init->proj(n_acc, i); }));
    after->app(true, new_exit, exit_acc);
    return w.app(launch, Defs{w.tuple({m0, m1, m4}), after});
}

} // namespace mim::plug::gpu::phase
