#include "mim/plug/xform/xform.h"

#include <mim/config.h>
#include <mim/phase.h>
#include <mim/plugin.h>

#include "mim/plug/xform/phase/schedule_for.h"

using namespace mim;
using namespace mim::plug;

void reg_phases(Flags2Phases& phases) {
    Phase::hook<xform::schedule_for, xform::phase::ScheduleFor>(phases);
}

extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"xform", MIM_VERSION, nullptr, reg_phases}; }
