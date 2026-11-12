#include <charconv>
#include <cstdlib>

#include <algorithm>
#include <string_view>
#include <thread>

#include <mim/driver.h>
#include <mim/world.h>

#include "mim/plug/omp/omp.h"

namespace mim::plug::omp {

const Def* normalize_thread_hint(const Def*, const Def*, const Def* arg) {
    auto& w = arg->world();
    if (auto n = w.driver().arg_nat("omp", "threads")) return w.lit_nat(std::max<u64>(1, *n));
    if (auto env = std::getenv("OMP_NUM_THREADS")) {
        auto value  = u64(0);
        auto end    = env + std::string_view(env).size();
        auto parsed = std::from_chars(env, end, value);
        if (parsed.ec == std::errc() && parsed.ptr == end && value) return w.lit_nat(value);
    }
    return w.lit_nat(std::max(1u, std::thread::hardware_concurrency()));
}

MIM_omp_NORMALIZER_IMPL

} // namespace mim::plug::omp
