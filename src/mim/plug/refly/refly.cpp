#include "mim/plug/refly/refly.h"

#include <mim/config.h>
#include <mim/phase.h>

using namespace mim;
using namespace mim::plug;

static constexpr PluginArg known_args[] = {
    {"strict=<n>", "Fail the compilation where a `refly.check` fails (`1`) instead of logging an error and leaving the "
                   "check stuck (default `0`)."},
};

static void reg_phases(Flags2Phases& phases) {
    MIM_REPL(phases, refly::remove_dbg_repl, {
        if (auto dbg_perm = Axm::isa(refly::dbg::perm, def)) {
            auto [lvl, x] = dbg_perm->args<2>();
            log().d("dbg perm: {}", x);
            return x;
        }

        return {};
    });
}

MIM_PLUGIN_ENTRY(refly) {
    plugin.register_normalizers = refly::register_normalizers;
    plugin.register_phases      = reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
}
