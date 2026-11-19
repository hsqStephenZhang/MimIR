#include "mim/plug/gpu/phase/ring_trace.h"

#include <mim/driver.h>
#include <mim/rewrite.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

struct Unfollowed {};

/// Runs the residual program of one thread with literal loop bounds and records its ring events.
/// It treats `ringFill` and `ringStep` as single events and runs only the first iteration of a `gpu.bind` loop.
class Tracer : public Rewriter {
public:
    Tracer(World& world)
        : Rewriter(world) {}

    const Def* run(const Def* callee, const Def* arg, const Def* ret) {
        for (;;) {
            if (++steps_ > Steps) fail(callee, "the step limit");
            arg    = rewrite(arg);
            callee = rewrite(callee);
            if (callee == ret) return arg;
            if (auto lam = callee->isa_mut<Lam>(); lam && lam->is_set()) {
                auto body = lam->reduce_body(arg)->isa<App>();
                if (!body) fail(lam, "a body that is no call");
                callee = body->callee();
                arg    = body->arg();
                continue;
            }
            if (callee == world().annex<mem::fresh>()) {
                auto [_, k] = arg->projs<2>();
                auto fresh  = world().mut_con(Pi::isa_cn(k->type())->dom());
                callee      = k;
                arg         = fresh->var();
                continue;
            }
            if (auto loop = Axm::isa<affine::For>(world().app(callee, arg))) {
                auto [body, exit, bounds]     = loop->uncurry_args<3>();
                auto [begin, end, step, init] = bounds->projs<4>();
                auto [b, e, s]                = std::tuple(Lit::isa(begin), Lit::isa(end), Lit::isa(step));
                if (!b || !e || !s) fail(loop, "a loop without literal bounds");
                auto fn    = body;
                auto bound = Axm::isa<gpu::bind>(fn) != nullptr;
                for (auto annotated = fn->isa<App>(); annotated && annotated->arg()->isa_mut<Lam>();
                     annotated      = fn->isa<App>())
                    fn = annotated->arg();
                auto state = rewrite(init);
                for (auto i = *b; i < *e; i += *s) {
                    auto yield = world().mut_lam(fn->type()->as<Pi>()->dom(3, 2)->as<Pi>());
                    state      = run(fn, world().tuple({world().lit(begin->type(), i), state, yield}), yield);
                    if (bound) break;
                }
                callee = exit;
                arg    = state;
                continue;
            }
            fail(callee, "a call it cannot follow");
        }
    }

    const Def* rewrite_mut(Def* mut) final { return mut; }

    const Def* rewrite_imm_App(const App* app) final {
        auto& w = world();
        if (auto ds = Axm::isa<cps::cps2ds_dep>(app->callee())) {
            auto fn  = rewrite(ds->arg());
            auto arg = rewrite(app->arg());
            if (auto lam = fn->isa_mut<Lam>(); lam && lam->sym() == w.sym("ringFill")) {
                auto [m, block, n, slot] = arg->projs<4>();
                auto [src, ticket]       = std::tuple(Lit::isa(block), Lit::isa(n));
                if (!src || !ticket) fail(app, "a fill of a block it cannot compute");
                emit(std::format("fill {} {{}} {}", *ticket, *src), slot);
                return map(m, m);
            }
            if (auto lam = fn->isa_mut<Lam>(); lam && lam->sym() == w.sym("ringStep")) {
                auto [m, acc, block, panel] = arg->projs<4>();
                if (!Lit::isa(block)) fail(app, "a step of a block it cannot compute");
                // `read_slot` erases to the panel at its position in the ring; the rolled ring reads its panel
                // directly.
                if (auto read = panel->isa<Extract>())
                    if (auto slot = Axm::isa<gpu::read_slot>(read->tuple())) {
                        auto [ring, at, _, _] = slot->args<4>();
                        panel                 = ring->as<App>()->arg()->proj(depth_, Lit::as(at));
                    }
                // A step's WGMMA batches read its panel, each waited for in the step; without any, it reads with loads.
                // Their effects sit on fresh memories, so `wgmma_batch` itself fixes the order within a batch.
                auto first = events_.size();
                auto ret   = w.mut_con(app->type());
                auto res   = run(fn, w.tuple({arg, ret}), ret);
                auto count = [&](std::string_view op) {
                    return std::ranges::count_if(events_.begin() + first, events_.end(),
                                                 [&](auto& event) { return event.first.starts_with(op); });
                };
                auto batches = count("wgmma_fence ");
                if (count("wgmma_commit ") != batches || count("wgmma_wait0 ") != batches
                    || events_.size() - first != size_t(3 * batches))
                    fail(app, "a step whose WGMMA batches are not each fenced, committed and waited for");
                events_.resize(first);
                if (batches == 0) {
                    emit(std::format("read {} {{}} load", Lit::as(block)), panel);
                } else {
                    auto c = batches_++;
                    emit(std::format("wgmma_fence {}", c));
                    emit(std::format("read {} {{}} wgmma {}", Lit::as(block), c), panel);
                    emit(std::format("wgmma_commit {}", c));
                    emit(std::format("wgmma_wait0 {}", c));
                }
                return map(res, res);
            }
            auto ret = w.mut_con(app->type());
            auto res = run(fn, w.tuple({arg, ret}), ret);
            return map(res, res);
        }
        auto res = Rewriter::rewrite_imm_App(app);
        // A rewritten effect reaches the next body through Lam::reduce and must not happen again there.
        map(res, res);
        if (auto vacate = Axm::isa<gpu::vacate>(res)) {
            depth_  = Lit::as(vacate->decurry()->arg(2, 1));
            active_ = true;
        } else if (Axm::isa<gpu::async_commit>(res)) {
            emit("commit");
        } else if (auto wait = Axm::isa<gpu::async_wait_group>(res)) {
            emit(std::format("wait {}", Lit::as(wait->decurry()->arg())));
        } else if (Axm::isa<gpu::fence>(res)) {
            emit("bar");
        } else if (Axm::isa<gpu::fence_proxy_async>(res)) {
            emit("proxy_fence");
        } else if (Axm::isa<gpu::wgmma_fence>(res)) {
            emit("wgmma_fence ");
        } else if (Axm::isa<gpu::wgmma_commit>(res)) {
            emit("wgmma_commit ");
        } else if (auto wait = Axm::isa<gpu::wgmma_wait>(res)) {
            if (Lit::isa(wait->arg(3, 1)) != 0) fail(res, "a WGMMA wait that leaves groups in flight");
            emit("wgmma_wait0 ");
        } else if (Axm::isa<gpu::async_wait>(res)) {
            emit("settle");
            active_ = false;
        }
        return res;
    }

    /// The events, with each panel named by the order in which fills first write it.
    /// The protocol theorems do not depend on how the slots are numbered, so any consistent naming serves.
    std::vector<std::string> events() {
        auto names = DefMap<u64>();
        for (auto& [event, panel] : events_) {
            if (!panel) continue;
            if (event.starts_with("fill ") && !names.contains(panel)) names.emplace(panel, names.size());
            auto name = names.find(panel);
            if (name == names.end()) fail(panel, "a read of a panel no fill wrote");
            event = std::vformat(event, std::make_format_args(name->second));
        }
        auto res = std::vector<std::string>();
        for (auto& [event, _] : events_)
            res.emplace_back(event);
        return res;
    }

    u64 depth() const { return depth_; }

private:
    static constexpr u64 Steps = 1'000'000;

    void emit(std::string event, const Def* panel = nullptr) {
        if (active_) events_.emplace_back(std::move(event), panel);
    }

    [[noreturn]] void fail(const Def* at, std::string_view what) {
        std::println(stderr, "ring-trace: {} at {}", what, at);
        throw Unfollowed();
    }

    std::vector<std::pair<std::string, const Def*>> events_;
    u64 depth_   = 0;
    u64 batches_ = 0;
    u64 steps_   = 0;
    bool active_ = false;
};

} // namespace

void RingTrace::start() {
    if (!world().driver().arg_nat("gpu", "ring-trace").value_or(0)) return;
    for (auto mut : world().externals().muts()) {
        auto lam = mut->isa_mut<Lam>();
        if (!lam || !lam->is_set() || !Pi::isa_returning(lam->type())) continue;
        auto tracer = Tracer(world());
        auto body   = lam->body()->isa<App>();
        if (!body) continue;
        auto events = std::vector<std::string>();
        try {
            tracer.run(body->callee(), body->arg(), lam->ret_var());
            events = tracer.events();
        } catch (const Unfollowed&) { continue; }
        if (events.empty()) continue;
        std::println(stderr, "ring-trace {} depth={}", lam->sym(), tracer.depth());
        for (auto& event : events)
            std::println(stderr, "ring-trace {}", event);
    }
}

} // namespace mim::plug::gpu::phase
