#include "mim/plug/dl/dl.h"

#include <mim/config.h>
#include <mim/plugin.h>

using namespace mim;

extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"dl", MIM_VERSION, nullptr, nullptr};
}
