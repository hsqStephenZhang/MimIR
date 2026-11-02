#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>

#define DECLARE(M, N)                         \
    extern "C" float* soft_##M##_##N(float*); \
    extern "C" float* layer_##M##_##N(float*);
DECLARE(5, 7)
DECLARE(1, 11)
DECLARE(7, 1)
DECLARE(0, 7)
DECLARE(2, 7)
DECLARE(64, 7)
DECLARE(65, 7)

static uint64_t exponentials = 0;
extern "C" void record_exp(uint64_t lanes) { exponentials += lanes; }
static uint64_t centered = 0;
extern "C" void record_center(uint64_t lanes) { centered += lanes; }
static uint64_t allocations       = 0;
static uint64_t stack_allocations = 0;
extern "C" void record_stack() { ++stack_allocations; }
static std::vector<void*> live;
extern "C" void* record_malloc(uint64_t size) {
    auto pointer = std::malloc(size);
    assert(pointer);
    ++allocations;
    live.push_back(pointer);
    return pointer;
}
extern "C" void record_free(void* pointer) {
    auto found = std::find(live.begin(), live.end(), pointer);
    assert(found != live.end());
    live.erase(found);
    std::free(pointer);
}

int main() {
    struct Case {
        int m, n;
        float* (*soft)(float*);
        float* (*layer)(float*);
    };
    Case cases[] = {
        { 5,  7,  soft_5_7,  layer_5_7},
        { 1, 11, soft_1_11, layer_1_11},
        { 7,  1,  soft_7_1,  layer_7_1},
        { 0,  7,  soft_0_7,  layer_0_7},
        { 2,  7,  soft_2_7,  layer_2_7},
        {64,  7, soft_64_7, layer_64_7},
        {65,  7, soft_65_7, layer_65_7}
    };
    for (auto c : cases)
        for (int repeat = 0; repeat != 3; ++repeat) {
#ifdef ALLOCATION_COUNT
            if (c.n == 1) continue;
#endif
            std::vector<float> x(std::max(1, c.m * c.n));
            for (int i = 0; i != c.m * c.n; ++i)
                x[i] = repeat == 2 ? 3.0f : float((i * 3 + repeat) % 17 - 8) * (repeat ? 100.0f : 0.25f);
            exponentials      = 0;
            allocations       = 0;
            stack_allocations = 0;
            auto soft         = c.soft(x.data());
#ifdef ALLOCATION_COUNT
            std::printf("soft %dx%d heap=%llu stack=%llu live=%zu\n", c.m, c.n,
                        static_cast<unsigned long long>(allocations),
                        static_cast<unsigned long long>(stack_allocations), live.size());
            std::fflush(stdout);
            assert(allocations + stack_allocations <= 9);
            assert(live.size() == uint64_t(c.m != 0));
#endif
#ifdef REALIZATION_COUNT
            if (c.n != 1) assert(exponentials == uint64_t(c.m * c.n));
#endif
            centered          = 0;
            allocations       = 0;
            stack_allocations = 0;
            auto layer        = c.layer(x.data());
#ifdef ALLOCATION_COUNT
            std::printf("layer %dx%d heap=%llu stack=%llu live=%zu\n", c.m, c.n,
                        static_cast<unsigned long long>(allocations),
                        static_cast<unsigned long long>(stack_allocations), live.size());
            std::fflush(stdout);
            assert(allocations + stack_allocations <= 9);
            assert(live.size() == 2 * uint64_t(c.m != 0));
#endif
#ifdef CENTER_COUNT
            assert(centered == uint64_t(c.m * c.n));
#endif
            for (int i = 0; i != c.m; ++i) {
                double mean = 0, variance = 0, sum = 0;
                auto maximum = *std::max_element(x.begin() + i * c.n, x.begin() + (i + 1) * c.n);
                for (int j = 0; j != c.n; ++j) {
                    mean += x[i * c.n + j] / double(c.n);
                    sum += std::exp(double(x[i * c.n + j]) - maximum);
                }
                for (int j = 0; j != c.n; ++j) {
                    double centered = x[i * c.n + j] - mean;
                    variance += centered * centered / c.n;
                }
                for (int j = 0; j != c.n; ++j) {
                    auto expected_soft  = std::exp(double(x[i * c.n + j]) - maximum) / sum;
                    auto expected_layer = (x[i * c.n + j] - mean) / std::sqrt(variance + 0.00001);
                    if (!(std::abs(soft[i * c.n + j] - expected_soft) < 0.0001)) {
                        std::fprintf(stderr, "soft %dx%d repeat=%d (%d,%d): %g != %g\n", c.m, c.n, repeat, i, j,
                                     soft[i * c.n + j], expected_soft);
                        std::abort();
                    }
                    assert(std::abs(layer[i * c.n + j] - expected_layer) < 0.0005);
                }
            }
            if (c.m) {
#ifdef ALLOCATION_COUNT
                record_free(soft);
                record_free(layer);
#else
                std::free(soft);
                std::free(layer);
#endif
            }
            assert(live.empty());
        }
}
