#include "mim/plug/gpu/phase/remove_double_syncs.h"

#include <mim/axm.h>

#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

// The memory this one continues, seen through the casts and static slots that order nothing at run time.
const Def* prev_mem(const Def* def) {
    while (true)
        if (auto generic = Axm::isa<gpu::generic_mem>(def))
            def = generic->arg();
        else if (auto global = Axm::isa<gpu::global_mem>(def))
            def = global->arg();
        else if (auto extract = def->isa<Extract>(); extract && Axm::isa<gpu::slot_shared>(extract->tuple()))
            def = extract->tuple()->as<App>()->arg();
        else
            return def;
}

bool is_kernel_entry(const Def* def) {
    if (auto extract = def->isa<Extract>()) def = extract->tuple();
    auto var = def->isa<Var>();
    return var && var->binder()->is_external();
}

} // namespace

const Def* RemoveDoubleSyncs::rewrite_imm_App(const App* app) {
    if (auto sync_work_items = Axm::isa<gpu::sync_work_items>(app)) {
        auto [m1, m3]   = sync_work_items->args<2>();
        auto prev1      = prev_mem(m1);
        auto prev3      = prev_mem(m3);
        auto m1_extract = prev1->isa<Extract>();
        auto m3_extract = prev3->isa<Extract>();
        bool after_sync = m1_extract && Axm::isa<gpu::sync_work_items>(m1_extract->tuple());
        // An outlined kernel orders its shared effects on the generic chain and passes the entry shared memory
        // unchanged.
        bool shared_in_step = m3_extract && m1_extract && m3_extract->tuple() == m1_extract->tuple();
        bool shared_entry   = is_kernel_entry(prev3);
        if ((after_sync && (shared_in_step || shared_entry)) || (is_kernel_entry(prev1) && shared_entry))
            return rewrite(sync_work_items->arg());
    }
    return Super::rewrite_imm_App(app);
}

} // namespace mim::plug::gpu::phase
