#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

#include <pthread.h>

struct Case {
    std::string_view mode, kind;
    int m, n;
    float* (*run)(float*);
    float* (*first)(float*)             = nullptr;
    float* (*transform)(float*, float*) = nullptr;
    float* (*reduce)(float*)            = nullptr;
    float* (*finish)(float*, float*)    = nullptr;
};
#include "cases.h"

static uint64_t heaps, stacks, releases, exponentials;
extern "C" void* record_malloc(uint64_t size) {
    ++heaps;
    auto result = std::malloc(size);
    if (!result) std::abort();
    return result;
}
extern "C" void record_free(void* value) {
    ++releases;
    std::free(value);
}
extern "C" void record_stack() { ++stacks; }
extern "C" void record_exp(uint64_t count) { exponentials += count; }

static float* execute(const Case& c, float* x) {
    if (c.run) return c.run(x);
    auto first = c.first(x);
    auto tile  = c.transform(x, first);
    std::free(first);
    auto reduced = c.reduce(tile);
    auto result  = c.finish(tile, reduced);
    std::free(tile);
    std::free(reduced);
    return result;
}

static std::vector<float> reference(const Case& c, const std::vector<float>& x) {
    std::vector<float> out(x.size());
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
        for (int j = 0; j != c.n; ++j)
            out[i * c.n + j] = c.kind == "soft" ? std::exp(double(x[i * c.n + j]) - maximum) / sum
                                                : (x[i * c.n + j] - mean) / std::sqrt(variance + 0.00001);
    }
    return out;
}

int main() {
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) return 1;
#ifdef COUNTS
    std::puts("input,workload,M,N,mode,exp,heap,stack,free");
#else
    std::puts("input,workload,M,N,mode,median_ms");
#endif
    for (int distribution = 0; distribution != 2; ++distribution)
        for (auto shape : {
                 std::array{1024, 1024},
                 std::array{4096, 1024},
                 std::array{1024, 4096}
        })
            for (auto kind : {"soft", "layer"}) {
                std::vector<const Case*> selected;
                for (const auto& c : cases)
                    if (c.m == shape[0] && c.n == shape[1] && c.kind == kind) selected.push_back(&c);
                std::vector<float> x(shape[0] * shape[1]);
                for (size_t i = 0; i != x.size(); ++i)
                    x[i] = distribution == 0 ? float(int(i % 23) - 11) / 8 : float(int(i % 11) - 5) / 32;
                auto expected = reference(*selected[0], x);
                for (auto c : selected) {
                    heaps = stacks = releases = exponentials = 0;
                    auto result                              = execute(*c, x.data());
                    for (size_t i = 0; i != x.size(); ++i)
                        if (!std::isfinite(result[i]) || std::abs(result[i] - expected[i]) > 0.0001f) {
                            std::fprintf(stderr, "%.*s %s %dx%d mismatch %zu: %g != %g\n", int(c->mode.size()),
                                         c->mode.data(), kind, c->m, c->n, i, result[i], expected[i]);
                            return 1;
                        }
#ifdef COUNTS
                    if ((c->mode == "automatic" || c->mode == "explicit" || c->mode == "api")
                        && (exponentials != (c->kind == "soft" ? x.size() : 0) || heaps + stacks > 2
                            || heaps != releases + 1)) {
                        std::fputs("unexpected exp multiplicity or scratch allocation count\n", stderr);
                        return 1;
                    }
                    std::printf("%s,%s,%d,%d,%.*s,%llu,%llu,%llu,%llu\n", distribution ? "task20" : "task15", kind,
                                c->m, c->n, int(c->mode.size()), c->mode.data(),
                                static_cast<unsigned long long>(exponentials), static_cast<unsigned long long>(heaps),
                                static_cast<unsigned long long>(stacks), static_cast<unsigned long long>(releases));
#endif
                    std::free(result);
                }
#ifndef COUNTS
                std::vector<std::vector<double>> times(selected.size());
                for (int run = 0; run != 7; ++run)
                    for (size_t order = 0; order != selected.size(); ++order) {
                        auto index  = (run + order) % selected.size();
                        auto start  = std::chrono::steady_clock::now();
                        auto result = execute(*selected[index], x.data());
                        std::free(result);
                        times[index].push_back(
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                                .count());
                    }
                for (size_t i = 0; i != selected.size(); ++i) {
                    std::sort(times[i].begin(), times[i].end());
                    std::printf("%s,%s,%d,%d,%.*s,%.6f\n", distribution ? "task20" : "task15", kind, shape[0], shape[1],
                                int(selected[i]->mode.size()), selected[i]->mode.data(), times[i][3]);
                }
#endif
                std::fflush(stdout);
            }
}
