// Checks conv_pe against conv_old and prints both in GFLOP/s (best of several runs).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* conv_pe(float*, float*);
float* conv_old(float*, float*);
}

int main(int, char** argv) {
    constexpr int OH = (H + 2 * P - K) / S + 1, OW = (W + 2 * P - K) / S + 1;
    std::vector<float> x(size_t(CI) * H * W), w(size_t(CO) * (CI / G) * K * K);
    for (size_t i = 0; i != x.size(); ++i) x[i] = float(int(i * 7 % 11) - 5) / 8;
    for (size_t i = 0; i != w.size(); ++i) w[i] = float(int(i * 5 % 13) - 6) / 32;
    float *a = conv_pe(x.data(), w.data()), *b = conv_old(x.data(), w.data());
    int bad = 0;
    for (size_t i = 0; i != size_t(CO) * OH * OW; ++i)
        if (std::fabs(a[i] - b[i]) > 1e-3f * (1 + std::fabs(b[i]))) ++bad;
    std::free(a), std::free(b);
    double flop = 2.0 * CO * OH * OW * (CI / G) * K * K;
    auto time = [&](float* (*f)(float*, float*)) {
        double best = 1e30;
        int reps    = std::max(3, int(2e9 / flop));
        for (int r = 0; r != reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            std::free(f(x.data(), w.data()));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return flop / best / 1e9;
    };
    double pe = time(conv_pe), old = time(conv_old);
    std::printf("%-8s wrong=%d  pe %7.1f  conv_schedule %7.1f GFLOP/s  pe/old %.2f\n", argv[1], bad, pe, old, pe / old);
    return bad != 0;
}
