#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <malloc.h>
#include <sys/resource.h>

extern "C" float* mim_attention(float*, float*, float*);

static size_t live_bytes() {
    auto info = mallinfo2();
    return info.uordblks + info.hblkhd;
}

int main(int argc, char** argv) {
    assert(argc == 7);
    size_t h = std::stoul(argv[1]), m = std::stoul(argv[2]), s = std::stoul(argv[3]);
    size_t d = std::stoul(argv[4]), n = std::stoul(argv[5]), calls = std::stoul(argv[6]);
    std::vector<float> q(h*m*d, 0.01f), k(h*s*d, 0.01f), v(h*s*n, 0.01f);
    auto run = [&] {
        auto out = mim_attention(q.data(), k.data(), v.data());
        for (size_t i = 0; i != h*m*n; ++i)
            assert(std::isfinite(out[i]) && std::abs(out[i] - 0.01f) < 1e-5f);
        std::free(out);
    };
    for (size_t i = 0; i != 10; ++i) run();
    size_t before = live_bytes();
    for (size_t i = 0; i != calls; ++i) run();
    size_t after = live_bytes();
    rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::printf("{\"calls\":%zu,\"warmup_calls\":10,\"live_growth_bytes\":%lld,\"maxrss_kib\":%ld}\n",
                calls, static_cast<long long>(after) - static_cast<long long>(before), usage.ru_maxrss);
}
