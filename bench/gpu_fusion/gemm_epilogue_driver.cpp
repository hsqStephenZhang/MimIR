// Times `gemm_bias_relu` (1024³) and checks a sample of entries against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* gemm_bias_relu(float* a, float* b, float* bias);

int main(int argc, char** argv) {
    constexpr size_t N = 1024;
    int iters          = argc > 1 ? std::atoi(argv[1]) : 5;
    std::vector<float> a(N * N), b(N * N), bias(N * N);
    for (size_t i = 0; i < N * N; ++i) {
        a[i]    = float(i % 97) / 97.0f - 0.5f;
        b[i]    = float(i % 89) / 89.0f - 0.5f;
        bias[i] = float(i % 7) - 3.0f;
    }
    float* c = gemm_bias_relu(a.data(), b.data(), bias.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::free(c);
        c = gemm_bias_relu(a.data(), b.data(), bias.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("end-to-end: %.3f ms/call\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    int failures = 0;
    for (size_t i = 0; i < N; i += 131)
        for (size_t j = 0; j < N; j += 127) {
            double ref = bias[i * N + j];
            for (size_t k = 0; k < N; ++k) ref += double(a[i * N + k]) * b[k * N + j];
            ref = std::fmax(ref, 0.0);
            if (std::fabs(ref - c[i * N + j]) > 1e-3 * (1 + std::fabs(ref))) {
                if (failures < 5) std::printf("c[%zu][%zu] = %g, expected %g\n", i, j, c[i * N + j], ref);
                ++failures;
            }
        }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    std::free(c);
    return failures ? 1 : 0;
}
