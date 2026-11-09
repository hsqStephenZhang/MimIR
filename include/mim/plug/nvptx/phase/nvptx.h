#pragma once

#include <map>
#include <optional>
#include <string>

#include <mim/plug/ll/ll.h>

namespace mim {

class World;

namespace plug::nvptx {

/// Prefix for this backend's fe::throwf messages; see MIM_LL_BE.
#define MIM_NVPTX_BE "nvptx backend: "

namespace ll = mim::plug::ll;

struct DeviceEmitFlags {
    bool uses_libdevice;
    /// The bytes of dynamic shared memory each kernel needs, by name: its slots exceed the static 48 KiB.
    std::map<std::string, u64> dynamic_shared;
};

void emit_host(World&,
               std::ostream&,
               std::optional<std::string>,
               std::map<std::string, u64> dynamic_shared = {},
               ll::Emitter::Rt rt                        = ll::Emitter::Rt::embed);
DeviceEmitFlags emit_device(World&, std::ostream&, bool async_copy);

} // namespace plug::nvptx

} // namespace mim
