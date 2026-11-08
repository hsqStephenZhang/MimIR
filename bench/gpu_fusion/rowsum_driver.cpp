// Times `rowsum` on (12, 1024, 1024) and checks sampled rows against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* rowsum(float* x);

int main(int argc, char** argv) {
    constexpr size_t H = 12, L = 1024;
    int iters = argc > 1 ? std::atoi(argv[1]) : 5;
    std::vector<float> x(H * L * L);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(i % 113) / 37.0f - 1.5f;
    float* y = rowsum(x.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) y = rowsum(x.data());
    auto t1 = std::chrono::steady_clock::now();
    std::printf("rowsum end-to-end: %.3f ms/call\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    int failures = 0;
    for (size_t row = 0; row < H * L; row += 97) {
        double ref = 0;
        for (size_t j = 0; j < L; ++j) ref += x[row * L + j];
        if (std::fabs(ref - y[row]) > 1e-3 * (1 + std::fabs(ref))) {
            if (failures < 40) std::printf("y[%zu] = %g, expected %g\n", row, y[row], ref);
            ++failures;
        }
    }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
