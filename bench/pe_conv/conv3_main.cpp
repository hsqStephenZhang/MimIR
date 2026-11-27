// Times one convolution three ways (im2col through run_levels, direct through run_levels, conv_schedule) after checking them against each other.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* conv_im2col(float*, float*);
float* conv_direct(float*, float*);
float* conv_sched(float*, float*);
}

int main(int, char** argv) {
    constexpr int OH = (H + 2 * P - K) / S + 1, OW = (W + 2 * P - K) / S + 1;
    std::vector<float> x(size_t(CI) * H * W), w(size_t(CO) * CI * K * K);
    for (size_t i = 0; i != x.size(); ++i) x[i] = float(int(i * 7 % 11) - 5) / 8;
    for (size_t i = 0; i != w.size(); ++i) w[i] = float(int(i * 5 % 13) - 6) / 32;
    float *a = conv_im2col(x.data(), w.data()), *b = conv_direct(x.data(), w.data()), *c = conv_sched(x.data(), w.data());
    int bad = 0;
    for (size_t i = 0; i != size_t(CO) * OH * OW; ++i)
        if (std::fabs(a[i] - c[i]) > 1e-3f * (1 + std::fabs(c[i])) || std::fabs(b[i] - c[i]) > 1e-3f * (1 + std::fabs(c[i]))) ++bad;
    std::free(a), std::free(b), std::free(c);
    double flop = 2.0 * CO * OH * OW * CI * K * K;
    auto time = [&](float* (*f)(float*, float*)) {
        double best = 1e30;
        int reps    = std::max(5, int(3e9 / flop));
        for (int r = 0; r != reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            std::free(f(x.data(), w.data()));
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        return flop / best / 1e9;
    };
    double i2c = time(conv_im2col), dir = time(conv_direct), sch = time(conv_sched);
    std::printf("%-8s wrong=%d  im2col %7.1f  direct %7.1f  conv_schedule %7.1f GFLOP/s  direct/sched %.2f  im2col/sched %.2f\n", argv[1], bad, i2c,
                dir, sch, dir / sch, i2c / sch);
    return bad != 0;
}
