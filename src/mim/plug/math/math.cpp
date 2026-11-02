#include "mim/plug/math/math.h"

#include <mim/config.h>
#include <mim/phase.h>

#include <mim/pattern/op.h>

using namespace mim;

MIM_PLUGIN_ENTRY(math) {
    plugin.register_normalizers      = plug::math::register_normalizers;
    plugin.register_graph_operations = [](pattern::OpRegistry& ops) {
        auto declare = [&ops]<class Id>() {
            for (size_t i = 0; i != Annex::Num<Id>; ++i) {
                auto flags = Annex::base<Id>() + i;
                if (is_commutative(static_cast<Id>(flags))) ops.declare_commutative(flags);
            }
        };
        declare.operator()<plug::math::arith>();
        declare.operator()<plug::math::extrema>();
        declare.operator()<plug::math::cmp>();
    };
}
