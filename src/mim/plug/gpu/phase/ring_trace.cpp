#include "mim/plug/gpu/phase/ring_trace.h"

#include <set>

#include <mim/driver.h>
#include <mim/rewrite.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/core/core.h>
#include <mim/plug/cps/cps.h>
#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

struct Unfollowed {};

/// Runs the residual program of work-item `thread` of the first work-group with literal loop bounds and records its
/// ring events; a `gpu.bind` loop runs the iteration that work-item runs.
/// A `ringFill` and a `ringStep` each become one event, with the mbarrier, TMA and WGMMA events inside them.
class Tracer : public Rewriter {
public:
    Tracer(World& world, u64 thread)
        : Rewriter(world)
        , thread_(thread) {}

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
                auto bound = Axm::isa<gpu::bind>(fn);
                for (auto annotated = fn->isa<App>(); annotated && annotated->arg()->isa_mut<Lam>();
                     annotated      = fn->isa<App>())
                    fn = annotated->arg();
                auto state = rewrite(init);
                if (slots_.empty()) ring(state);
                auto group = fe::Restore(in_warpgroup_, in_warpgroup_);
                auto first = *b;
                if (bound) {
                    auto [nth, groups] = iteration(bound.id());
                    in_warpgroup_ |= groups;
                    first = *b + nth * *s;
                }
                for (auto i = first; i < *e; i += *s) {
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
                // `ws_fold`'s fill is not told its fill number, which is its block outside a persistent work-group.
                auto three               = arg->num_projs() == 3;
                auto [m, block, n, slot] = three ? std::array{arg->proj(3, 0), arg->proj(3, 1), arg->proj(3, 1),
                                                              arg->proj(3, 2)}
                                                 : arg->projs<4>();
                auto [src, ticket]       = std::tuple(Lit::isa(block), Lit::isa(n));
                if (!src || !ticket) fail(app, "a fill of a block it cannot compute");
                emit(std::format("fill {} {{}} {}", *ticket, *src), slot);
                if (!bars_.empty()) {
                    // A TMA fill's copies arrive on an mbarrier, so its own events belong to the protocol.
                    auto ret = w.mut_con(app->type());
                    auto res = run(fn, w.tuple({arg, ret}), ret);
                    return map(res, res);
                }
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
                if (count("wgmma_commit ") != batches || count("wgmma_wait ") != batches
                    || events_.size() - first != size_t(3 * batches))
                    fail(app, "a step whose WGMMA batches are not each fenced, committed and waited for");
                auto waits = std::set<std::string>();
                for (auto& [event, _] : events_ | std::views::drop(first))
                    if (event.starts_with("wgmma_wait ")) waits.insert(event);
                if (waits.size() > 1) fail(app, "a step whose WGMMA batches wait for different counts");
                auto pending = waits.empty() ? std::string() : waits.begin()->substr(11);
                events_.resize(first);
                if (batches == 0) {
                    emit(std::format("read {} {{}} load", Lit::as(block)), panel);
                } else {
                    auto c = batches_++;
                    emit(std::format("wgmma_fence {}", c));
                    emit(std::format("read {} {{}} wgmma {}", Lit::as(block), c), panel);
                    emit(std::format("wgmma_commit {}", c));
                    emit(pending == "0" ? std::format("wgmma_wait0 {}", c) : std::format("wgmma_wait {} {}", pending, c));
                }
                return map(res, res);
            }
            auto ret = w.mut_con(app->type());
            auto res = run(fn, w.tuple({arg, ret}), ret);
            return map(res, res);
        }
        if (Axm::isa<gpu::leader>(app)) return w.lit_bool(thread_ == 0);
        if (Axm::isa<gpu::item_index>(app)) return w.lit_i64(thread_);
        if (Axm::isa<gpu::lane_id>(app)) return w.lit_i64(thread_ % 32);
        if (Axm::isa<gpu::warp_rank>(app)) return w.lit_i64(thread_ / 32 % 4);
        auto res = Rewriter::rewrite_imm_App(app);
        // A rewritten effect reaches the next body through Lam::reduce and must not happen again there.
        map(res, res);
        if (auto init = Axm::isa<gpu::mbar_init_if>(res)) {
            active_                 = true;
            auto [_, bar, count, on] = init->args<4>();
            // Every work-item names the mbarriers in the order they are set up, whoever sets them up.
            auto name = barrier(bar);
            if (issued(res, on)) emit(std::format("init {} {}", name, Lit::as(count)));
        } else if (auto expect = Axm::isa<gpu::mbar_expect_tx_if>(res)) {
            auto [_, bar, bytes, on] = expect->args<4>();
            if (issued(res, on)) emit(std::format("expect {} {}", barrier(bar), Lit::as(bytes)));
        } else if (auto load = Axm::isa<gpu::tma_load_if>(res)) {
            auto [_, dst, map, col, row, bar, on] = load->args<7>();
            if (issued(res, on)) emit(std::format("tma {} {}", barrier(bar), box_bytes(map)));
        } else if (auto wait = Axm::isa<gpu::mbar_wait_at>(res)) {
            auto [_, bar, parity] = wait->args<3>();
            emit(std::format("mbar_wait {} {}", barrier(bar), Lit::as(parity)));
        } else if (auto wait = Axm::isa<gpu::mbar_wait_if>(res)) {
            auto [_, bar, parity, on] = wait->args<4>();
            if (issued(res, on)) emit(std::format("mbar_wait {} {}", barrier(bar), Lit::as(parity)));
        } else if (auto arrive = Axm::isa<gpu::mbar_arrive_if>(res)) {
            auto [_, bar, on] = arrive->args<3>();
            if (issued(res, on)) emit(std::format("arrive {}", barrier(bar)));
        } else if (Axm::isa<gpu::hold>(res)) {
            active_ = true;
        }
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
            auto pending = Lit::isa(wait->arg(3, 1));
            if (!pending) fail(res, "a WGMMA wait for a count it cannot compute");
            emit(std::format("wgmma_wait {}", *pending));
        } else if (Axm::isa<gpu::async_wait>(res)) {
            emit("settle");
            active_ = false;
        }
        return res;
    }

    /// The events, with each panel named by its position in the ring the fold's loop starts with.
    std::vector<std::string> events() {
        for (auto& [event, panel] : events_) {
            if (!panel) continue;
            auto name = slots_.find(panel);
            if (name == slots_.end()) fail(panel, "a panel outside the ring");
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

    /// A ring fold's loop carries `[mem, acc, ring, last]`, where only `last` holds a `gpu.Ring`.
    void ring(const Def* state) {
        if (state->num_projs() != 4 || !mentions_ring(state->proj(4, 3)->type())) return;
        auto slots = state->proj(4, 2);
        depth_     = Lit::as(slots->arity());
        for (u64 i = 0; i != depth_; ++i)
            slots_.emplace(slots->proj(depth_, i), i);
    }

    static bool mentions_ring(const Def* type) {
        if (Axm::isa<gpu::Ring>(type)) return true;
        if (auto sigma = type->isa<Sigma>()) return std::ranges::any_of(sigma->ops(), mentions_ring);
        return false;
    }

    /// The work-item's iteration of a loop bound to @p kind, and whether that iteration is a warpgroup.
    std::pair<u64, bool> iteration(gpu::bind kind) const {
        switch (kind) {
            case gpu::bind::group: return {0, false};
            case gpu::bind::warpgroup:
            case gpu::bind::specialized: return {thread_ / 128, true};
            case gpu::bind::warp: return {in_warpgroup_ ? thread_ / 32 % 4 : thread_ / 32, false};
            default: return {thread_, false};
        }
    }

    /// The bytes of one box of the TMA descriptor @p map.
    u64 box_bytes(const Def* map) {
        const Def* type = nullptr;
        std::optional<u64> rows, cols;
        if (auto buf = Axm::isa<gpu::buf_tensor_map>(map)) {
            type = buf->decurry()->decurry()->arg();
            rows = Lit::isa(buf->arg(3, 1));
            cols = Lit::isa(buf->arg(3, 2));
        } else if (auto dev = Axm::isa<gpu::device_tensor_map>(map)) {
            type = dev->decurry()->decurry()->decurry()->arg();
            rows = Lit::isa(dev->arg(5, 3));
            cols = Lit::isa(dev->arg(5, 4));
        }
        auto size = type ? Lit::isa(core::op(core::trait::size, type)) : std::nullopt;
        if (!rows || !cols || !size) fail(map, "a TMA descriptor whose box it cannot size");
        return *rows * *cols * *size;
    }

    bool issued(const Def* at, const Def* on) {
        auto lit = Lit::isa<bool>(on);
        if (!lit) fail(at, "a predicate it cannot compute");
        return *lit;
    }

    /// An mbarrier as the index of its buffer, in the order of first use, and its slot.
    std::string barrier(const Def* bar) {
        auto addr = Axm::isa<gpu::buf_addr>(bar);
        auto slot = addr ? Lit::isa(addr->arg(2, 1)) : std::nullopt;
        if (!slot) fail(bar, "an mbarrier it cannot name");
        auto buf = addr->arg(2, 0);
        auto k   = bars_.emplace(buf, bars_.size()).first->second;
        return std::format("{}.{}", k, *slot);
    }

    void emit(std::string event, const Def* panel = nullptr) {
        if (active_) events_.emplace_back(std::move(event), panel);
    }

    [[noreturn]] void fail(const Def* at, std::string_view what) {
        std::println(stderr, "ring-trace: {} at {}", what, at);
        throw Unfollowed();
    }

    std::vector<std::pair<std::string, const Def*>> events_;
    DefMap<u64> bars_;
    DefMap<u64> slots_;
    u64 thread_;
    bool in_warpgroup_ = false;
    u64 depth_   = 0;
    u64 batches_ = 0;
    u64 steps_   = 0;
    bool active_ = false;
};

} // namespace

void RingTrace::start() {
    // `-X gpu:ring-trace=n` traces work-items 0, 128, ..., one per warpgroup and the producer warp after them.
    auto threads = world().driver().arg_nat("gpu", "ring-trace").value_or(0);
    for (auto mut : world().externals().muts()) {
        auto lam  = mut->isa_mut<Lam>();
        auto body = lam && lam->is_set() && Pi::isa_returning(lam->type()) ? lam->body()->isa<App>() : nullptr;
        if (!body) continue;
        for (u64 t = 0; t != threads; ++t) {
            auto tracer = Tracer(world(), 128 * t);
            auto events = std::vector<std::string>();
            try {
                tracer.run(body->callee(), body->arg(), lam->ret_var());
                events = tracer.events();
            } catch (const Unfollowed&) { break; }
            if (events.empty()) break;
            std::println(stderr, "ring-trace {} depth={} thread={}", lam->sym(), tracer.depth(), 128 * t);
            for (auto& event : events)
                std::println(stderr, "ring-trace {}", event);
        }
    }
}

} // namespace mim::plug::gpu::phase
