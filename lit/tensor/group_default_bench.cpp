#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <string_view>
#include <vector>

#include <cblas.h>
#include <pthread.h>

struct Case {
    std::string_view kind;
    int m, n;
    float* (*grouped)(float*, float*, float*);
    float* (*unfused)(float*, float*, float*);
    float* (*explicit_blis)(float*, float*, float*);
};
#include "cases.h"

static void check(const float* actual, const float* expected, size_t count) {
    for (size_t i = 0; i != count; ++i)
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 0.0001f + 0.001f * std::abs(expected[i])) {
            std::fprintf(stderr, "mismatch at %zu: %.9g != %.9g\n", i, actual[i], expected[i]);
            std::exit(1);
        }
}

static std::vector<float> input(size_t size, int period) {
    std::vector<float> result(size);
    for (size_t i = 0; i != size; ++i)
        result[i] = float(int(i % period) - period / 2) / 32;
    return result;
}

static float* allocate(size_t count) {
    auto result = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (!result) std::abort();
    return result;
}

static void product(const float* a, const float* b, float* out, int m, int k, int n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, out, n);
}

static std::vector<float> rows(const Case& c, const std::vector<float>& x) {
    std::vector<float> result(x.size());
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
            result[i * c.n + j] = c.kind == "soft" ? std::exp(double(x[i * c.n + j]) - maximum) / sum
                                                   : (x[i * c.n + j] - mean) / std::sqrt(variance + 0.00001);
    }
    return result;
}

static std::vector<float>
convolution(const std::vector<float>& a, const std::vector<float>& weights, int channels, int width) {
    auto output_width = width - 2;
    std::vector<float> out(2 * channels * output_width * output_width);
    for (int batch = 0; batch != 2; ++batch)
        for (int o = 0; o != channels; ++o)
            for (int y = 0; y != output_width; ++y)
                for (int x = 0; x != output_width; ++x)
                    for (int i = 0; i != channels; ++i)
                        for (int ky = 0; ky != 3; ++ky)
                            for (int kx = 0; kx != 3; ++kx)
                                out[((batch * channels + o) * output_width + y) * output_width + x]
                                    += a[((batch * channels + i) * width + y + ky) * width + x + kx]
                                     * weights[((o * channels + i) * 3 + ky) * 3 + kx];
    return out;
}

int main() {
    if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0) return 1;
    openblas_set_num_threads(1);
    std::puts("workload,M,N,group1_ms,group0_ms,openblas_loop_ms,explicit_blis_ms");
    for (const auto& c : cases) {
        bool matrix = c.kind == "plain" || c.kind == "gemm" || c.kind == "mlp";
        bool conv   = c.kind == "conv";
        auto a      = input(conv ? 2 * c.n * c.m * c.m : c.m * c.n, 11);
        auto b      = input(conv ? c.n * c.n * 9 : c.n * c.n, 13);
        auto w      = input(conv ? c.n * c.n * 9 : c.kind == "gemm" ? c.n : c.n * c.n, 7);
        auto blas   = [&]() {
            auto hidden = allocate(c.m * c.n);
            product(a.data(), b.data(), hidden, c.m, c.n, c.n);
            if (c.kind != "plain")
                for (int i = 0; i != c.m * c.n; ++i)
                    hidden[i] = std::max(0.0f, hidden[i] + (c.kind == "gemm" ? w[i % c.n] : 0));
            if (c.kind != "mlp") return hidden;
            auto out = allocate(c.m * c.n);
            product(hidden, w.data(), out, c.m, c.n, c.n);
            std::free(hidden);
            return out;
        };
        std::vector<float> expected;
        if (matrix) {
            auto reference = blas();
            expected.assign(reference, reference + c.m * c.n);
            std::free(reference);
        } else if (conv) {
            expected = convolution(convolution(a, b, c.n, c.m), w, c.n, c.m - 2);
        } else {
            expected = rows(c, a);
        }
        auto execute = [&](int kind) {
            return kind == 2 ? blas()
                             : (kind == 0   ? c.grouped
                                : kind == 1 ? c.unfused
                                            : c.explicit_blis)(a.data(), b.data(), w.data());
        };
        auto count = c.explicit_blis ? 4 : matrix ? 3 : 2;
        std::vector<double> times[4];
        for (int kind = 0; kind != count; ++kind) {
            auto result = execute(kind);
            check(result, expected.data(), expected.size());
            std::free(result);
        }
        for (int run = 0; run != 7; ++run)
            for (int order = 0; order != count; ++order) {
                auto kind   = (run + order) % count;
                auto start  = std::chrono::steady_clock::now();
                auto result = execute(kind);
                std::free(result);
                times[kind].push_back(
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            }
        double medians[4] = {0, 0, 0, 0};
        for (int kind = 0; kind != count; ++kind) {
            std::sort(times[kind].begin(), times[kind].end());
            medians[kind] = times[kind][3];
        }
        std::printf("%.*s,%d,%d,%.6f,%.6f,", int(c.kind.size()), c.kind.data(), c.m, c.n, medians[0], medians[1]);
        if (matrix)
            std::printf("%.6f", medians[2]);
        else
            std::fputs("NA", stdout);
        std::putchar(',');
        if (c.explicit_blis)
            std::printf("%.6f", medians[3]);
        else
            std::fputs("NA", stdout);
        std::putchar('\n');
        std::fflush(stdout);
    }
}
