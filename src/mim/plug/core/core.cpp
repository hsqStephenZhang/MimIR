#include "mim/plug/core/core.h"

#include <mim/config.h>
#include <mim/phase.h>

#include <mim/pattern/op.h>

using namespace mim;
using namespace mim::plug;

MIM_PLUGIN_ENTRY(core) {
    plugin.register_normalizers      = core::register_normalizers;
    plugin.register_graph_operations = [](pattern::OpRegistry& ops) {
        auto declare = [&ops]<class Id>() {
            for (size_t i = 0; i != Annex::Num<Id>; ++i) {
                auto flags = Annex::base<Id>() + i;
                if (is_commutative(static_cast<Id>(flags))) ops.declare_commutative(flags);
            }
        };
        declare.operator()<core::nat>();
        declare.operator()<core::ncmp>();
        declare.operator()<core::wrap>();
        declare.operator()<core::icmp>();
        declare.operator()<core::bit2>();
    };
}
