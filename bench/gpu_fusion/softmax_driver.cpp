// Times `softmax_rows` on (12, 1024, 1024) and checks sampled rows against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* softmax_rows(float* x);

int main(int argc, char** argv) {
    constexpr size_t H = 12, L = 1024;
    int iters = argc > 1 ? std::atoi(argv[1]) : 5;
    std::vector<float> x(H * L * L);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(i % 113) / 37.0f - 1.5f;
    float* y = softmax_rows(x.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::free(y);
        y = softmax_rows(x.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("softmax end-to-end: %.3f ms/call\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    int failures = 0;
    for (size_t row : {size_t(0), size_t(7 * L + 3), size_t(11 * L + 1023)}) {
        float mx = -1e30f;
        for (size_t j = 0; j < L; ++j) mx = std::fmax(mx, x[row * L + j]);
        double sum = 0;
        for (size_t j = 0; j < L; ++j) sum += std::exp(double(x[row * L + j] - mx));
        for (size_t j = 0; j < L; j += 97) {
            double ref = std::exp(double(x[row * L + j] - mx)) / sum;
            if (std::fabs(ref - y[row * L + j]) > 1e-5 * (1 + std::fabs(ref))) {
                if (failures < 5) std::printf("y[%zu][%zu] = %g, expected %g\n", row, j, y[row * L + j], ref);
                ++failures;
            }
        }
    }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    std::free(y);
    return failures ? 1 : 0;
}
