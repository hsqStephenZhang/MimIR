#include "mim/plug/omp/omp.h"

#include <mim/config.h>
#include <mim/phase.h>
#include <mim/tuple.h>

#include <mim/plug/mem/escape_analysis.h>

#include "mim/plug/omp/phase/outline.h"

using namespace mim;
using namespace mim::plug;

static void reg_phases(Flags2Phases& phases) { Phase::hook<omp::outline, omp::phase::Outline>(phases); }

namespace {

bool holds_handle(const Def* type) {
    if (type->isa_mut() || type->isa<Pi>() || Axm::isa<mem::Ptr>(type)) return true;
    if (auto sigma = type->isa<Sigma>()) return std::ranges::any_of(sigma->ops(), holds_handle);
    if (auto arr = type->isa<Arr>()) return holds_handle(arr->body());
    return false;
}

bool local_control(const Def* callee, Lam* kernel) {
    if (callee == kernel->ret_var()) return true;
    if (auto lam = callee->isa_mut<Lam>()) return lam->is_rewritable();
    if (auto sw = Switch(callee)) {
        for (size_t i = 0; i != sw.num_targets(); ++i)
            if (!local_control(sw.target(i), kernel)) return false;
        return true;
    }
    return false;
}

// A deliberately conservative proof over the outlined kernel, rather than a promise about all OpenMP calls.
// Pointer loads/addressing and scalar stores borrow; pointer stores, pointer-to-integer conversions,
// unknown calls with handles, and frees other than direct local allocations fail the proof.
bool borrowing_kernel(Lam* kernel) {
    auto seen = DefSet();
    auto todo = DefVec{kernel};
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (!def->is_term() || def->isa<Var>() || !seen.emplace(def).second) continue;
        if (auto lam = def->isa_mut<Lam>()) {
            if (!lam->is_set()) return false;
            todo.emplace_back(lam->body());
            continue;
        }
        if (def->isa_mut()) return false;
        if (auto app = def->isa<App>(); app && app->curry() == 0) {
            if (auto store = Axm::isa<mem::store>(app)) {
                if (holds_handle(store->arg(3, 2)->type())) return false;
            } else if (Axm::isa<mem::free>(app)) {
                if (!mem::EscapeAnalysis::freed_alloc(app)) return false;
            } else if (Axm::isa<mem::load>(app) || Axm::isa<mem::lea>(app) || Axm::isa<mem::assume>(app)
                       || Axm::isa<mem::align>(app) || Axm::isa<mem::alloc>(app) || Axm::isa<mem::malloc>(app)
                       || Axm::isa<mem::slot>(app) || Axm::isa<mem::mslot>(app)) {
                // Known memory primitives; any continuation is checked by the traversal below.
            } else if (app->axm()) {
                if (holds_handle(app->arg()->type()) || holds_handle(app->type())) return false;
            } else {
                // A captured callback may retain its own captures even with only scalar arguments.
                if (!local_control(app->callee(), kernel)) return false;
                if (app->callee() == kernel->ret_var() && !mem::isa_mem(app->arg())) return false;
            }
        }
        for (auto op : def->ops())
            todo.emplace_back(op);
    }
    return true;
}

} // namespace

static bool mim_mem_borrows(const App* app) {
    auto launch = Axm::isa<omp::launch>(app);
    if (!launch) return false;
    auto [name, args] = launch->uncurry_args<2>();
    auto& w           = app->world();
    auto symbol       = w.sym(tuple2str(name));
    auto external     = w.externals()[symbol];
    auto kernel       = external ? external->isa_mut<Lam>() : nullptr;
    // The runtime is synchronous, but a kernel can still store or give away a captured pointer.
    return kernel && borrowing_kernel(kernel);
}

static const PluginSym syms[] = {MIM_PLUGIN_SYM(mim_mem_borrows)};

static constexpr PluginArg known_args[] = {
    {   "strict=<n>",    "Reject an OpenMP loop that would fall back to sequential execution (default 0)."},
    {"threshold=<n>", "Minimum GEMM multiply-add count for automatic CPU thread tiling (default 1048576)."},
    {  "threads=<n>",  "Compile-time task-grid hint; defaults to OMP_NUM_THREADS or the host concurrency."},
};

MIM_PLUGIN_ENTRY(omp) {
    plugin.register_normalizers = omp::register_normalizers;
    plugin.register_phases      = reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
    plugin.syms                 = syms;
    plugin.num_syms             = std::size(syms);
}
