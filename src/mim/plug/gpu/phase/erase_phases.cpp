#include "mim/plug/gpu/phase/erase_phases.h"

#include <mim/plug/buffer/buffer.h>
#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

const Def* ErasePhases::rewrite_imm_App(const App* app) {
    auto& w    = new_world();
    auto fence = [&](const Def* mem) { return w.app(w.annex<gpu::fence>(), mem); };
    if (Axm::isa<gpu::GM>(app)) return w.call<mem::M>(0);
    if (auto buf = Axm::isa<gpu::SharedBuf>(app)) {
        auto [g, e, v, r, s, T] = buf->args<6>();
        return buffer::type_buf(rewrite(r), rewrite(s), rewrite(T));
    }
    if (Axm::isa<gpu::group_enter>(app) || Axm::isa<gpu::sync_publish>(app) || Axm::isa<gpu::sync_reclaim>(app))
        return fence(rewrite(app->arg()));
    if (Axm::isa<gpu::enter_write>(app) || Axm::isa<gpu::group_leave>(app) || Axm::isa<gpu::view>(app))
        return rewrite(app->arg());
    if (auto panel = Axm::isa<gpu::shared_panel>(app)) {
        auto [r, s, T] = panel->decurry()->args<3>();
        auto mem       = w.app(w.annex<gpu::shared_scope>(), rewrite(app->arg()));
        return buffer::op_alloc(rewrite(r), rewrite(s), rewrite(T), mem);
    }
    if (Axm::isa<gpu::coop_fill>(app)) {
        auto [mem, buffer, fill] = rewrite(app->arg())->projs<3>();
        return w.app(fill, w.tuple({mem, buffer}));
    }
    if (Axm::isa<gpu::publish_view>(app)) return rewrite(app->arg())->proj(2, 1);
    if (Axm::isa<gpu::read_phase>(app)) {
        auto [mem, body] = rewrite(app->arg())->projs<2>();
        return w.app(body, mem);
    }
    return RWPhase::rewrite_imm_App(app);
}

} // namespace mim::plug::gpu::phase
