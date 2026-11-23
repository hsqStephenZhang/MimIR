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
    {"registers=<n>", "Maximum 32-bit cell equivalents (×4 bytes) kept in SSA registers in a blocked leaf (default: three quarters of the vector register file)."},
    {"vec-bytes=<n>", "Bytes per vector register of the target (default: the host's, `64` with AVX-512, `32` with AVX2, else `16`)."},
    {"vregs=<n>", "Vector registers of the target (default: the host's, `32` with AVX-512 or on AArch64, else `16`)."},
};

MIM_PLUGIN_ENTRY(btensor) {
    plugin.register_normalizers = btensor::register_normalizers;
    plugin.register_phases      = reg_phases;
    plugin.args                 = known_args;
    plugin.num_args             = std::size(known_args);
}
