#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>

#include <pthread.h>

struct Case {
    int m, n, rows;
    bool layer;
    float* (*grouped)(float*);
    float* (*first)(float*);
    float* (*transform)(float*, float*);
    float* (*reduce)(float*);
    float* (*finish)(float*, float*);
};
#include "cases.h"

static uint64_t heap_allocations = 0, stack_allocations = 0, frees = 0;
extern "C" void* record_malloc(uint64_t size) {
    ++heap_allocations;
    auto result = std::malloc(size);
    if (!result) std::abort();
    return result;
}
extern "C" void record_free(void* pointer) {
    ++frees;
    std::free(pointer);
}
extern "C" void record_stack() { ++stack_allocations; }

static float* unfused(const Case& c, float* x) {
    auto first = c.first(x);
    auto tile  = c.transform(x, first);
    std::free(first);
    auto reduced = c.reduce(tile);
    auto result  = c.finish(tile, reduced);
    std::free(reduced);
    std::free(tile);
    return result;
}

static void check(const Case& c, const std::vector<float>& x, const float* actual) {
    for (int i = 0; i != c.m; ++i) {
        double mean = 0, variance = 0, sum = 0;
        auto maximum = *std::max_element(x.begin() + i * c.n, x.begin() + (i + 1) * c.n);
        for (int j = 0; j != c.n; ++j) {
            mean += x[i * c.n + j] / double(c.n);
            sum += std::exp(double(x[i * c.n + j]) - maximum);
        }
        for (int j = 0; j != c.n; ++j) {
            auto centered = x[i * c.n + j] - mean;
            variance += centered * centered / c.n;
        }
        for (int j = 0; j != c.n; ++j) {
            auto expected = c.layer ? (x[i * c.n + j] - mean) / std::sqrt(variance + 0.00001)
                                    : std::exp(double(x[i * c.n + j]) - maximum) / sum;
            if (!std::isfinite(actual[i * c.n + j]) || std::abs(actual[i * c.n + j] - expected) > 0.0001) {
                std::fprintf(stderr, "%s %dx%d mismatch (%d,%d): %g != %g\n", c.layer ? "layer" : "soft", c.m, c.n, i,
                             j, actual[i * c.n + j], expected);
                std::exit(1);
            }
        }
    }
}

int main() {
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) {
        std::fputs("failed to set benchmark QoS\n", stderr);
        return 1;
    }
    std::puts("M,N,row_tile,workload,grouped_ms,unfused_ms,speedup");
    for (const auto& c : cases) {
        std::vector<float> x(c.m * c.n);
        for (size_t i = 0; i != x.size(); ++i)
            x[i] = float(int(i % 23) - 11) / 8;
        double medians[2];
        auto execute = [&](int kind) { return kind == 0 ? c.grouped(x.data()) : unfused(c, x.data()); };
        for (int kind = 0; kind != 2; ++kind) {
            heap_allocations = stack_allocations = frees = 0;
            auto result                                  = execute(kind);
            check(c, x, result);
#ifdef ALLOCATION_COUNT
            if (kind == 0) {
                std::fprintf(stderr, "%s %dx%d heap=%llu stack=%llu frees=%llu\n", c.layer ? "layer" : "soft", c.m, c.n,
                             static_cast<unsigned long long>(heap_allocations),
                             static_cast<unsigned long long>(stack_allocations),
                             static_cast<unsigned long long>(frees));
                auto limit = c.rows == 1 ? 2 : 4;
                if (heap_allocations + stack_allocations > limit || heap_allocations != frees + 1) std::abort();
            }
#endif
            std::free(result);
        }
        std::vector<double> times[2];
        for (int run = 0; run != 7; ++run)
            for (int order = 0; order != 2; ++order) {
                auto kind   = (run + order) % 2;
                auto start  = std::chrono::steady_clock::now();
                auto result = execute(kind);
                std::free(result);
                times[kind].push_back(
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            }
        for (int kind = 0; kind != 2; ++kind) {
            std::sort(times[kind].begin(), times[kind].end());
            medians[kind] = times[kind][3];
        }
        std::printf("%d,%d,%d,%s,%.6f,%.6f,%.3f\n", c.m, c.n, c.rows, c.layer ? "layernorm" : "softmax", medians[0],
                    medians[1], medians[1] / medians[0]);
        std::fflush(stdout);
    }
}
