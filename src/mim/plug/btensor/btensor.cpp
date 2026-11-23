#include "mim/plug/btensor/btensor.h"

#include <mim/phase.h>
#include <mim/plugin.h>

#include "mim/plug/btensor/phase/lower_map_reduce.h"

using namespace mim;
using namespace mim::plug;

static void reg_phases(Flags2Phases& phases) {
    Phase::hook<btensor::lower_map_reduce, btensor::phase::LowerMapReduce>(phases);
}

static constexpr PluginArg known_args[] = {
    {"registers=<n>", "Maximum 32-bit cell equivalents (×4 bytes) kept in SSA registers in a blocked leaf (default `64`, i.e. 256 bytes)."},
};

MIM_PLUGIN_ENTRY(btensor) {
    plugin.register_normalizers = btensor::register_normalizers;
    plugin.register_phases      = reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
}
