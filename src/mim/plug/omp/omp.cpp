#include "mim/plug/omp/omp.h"

#include <mim/config.h>
#include <mim/phase.h>

#include "mim/plug/omp/phase/outline.h"

using namespace mim;
using namespace mim::plug;

static void reg_phases(Flags2Phases& phases) { Phase::hook<omp::outline, omp::phase::Outline>(phases); }

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
}
