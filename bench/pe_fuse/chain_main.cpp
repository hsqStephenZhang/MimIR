// Checks the fused against the layered chain and times both: chain_main K M D [reps]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* fused(float*, ...);
float* layered(float*, ...);
}

using Fn = float* (*)(float*, ...);

static float* call(Fn f, float* a, std::vector<std::vector<float>>& w) {
    switch (w.size()) {
        case 1: return f(a, w[0].data());
        case 2: return f(a, w[0].data(), w[1].data());
        case 3: return f(a, w[0].data(), w[1].data(), w[2].data());
        case 4: return f(a, w[0].data(), w[1].data(), w[2].data(), w[3].data());
        case 5: return f(a, w[0].data(), w[1].data(), w[2].data(), w[3].data(), w[4].data());
    }
    std::abort();
}

int main(int argc, char** argv) {
    int K = std::atoi(argv[1]), M = std::atoi(argv[2]), D = std::atoi(argv[3]);
    int reps = argc > 4 ? std::atoi(argv[4]) : 20;
    std::vector<float> a(M * D);
    for (int i = 0; i != M * D; ++i) a[i] = float(int((i * 7) % 11) - 5) / 8.0f;
    std::vector<std::vector<float>> w(K, std::vector<float>(D * D));
    for (int l = 0; l != K; ++l)
        for (int i = 0; i != D * D; ++i) w[l][i] = float(int((i * (5 + 2 * l)) % 13) - 6) / 32.0f;

    float* x = call((Fn)fused, a.data(), w);
    float* y = call((Fn)layered, a.data(), w);
    int bad = 0;
    for (int i = 0; i != M * D; ++i)
        if (std::fabs(x[i] - y[i]) > 1e-3f * (1.0f + std::fabs(y[i]))) ++bad;
    std::free(x), std::free(y);

    auto time = [&](Fn f) {
        double best = 1e30;
        for (int r = 0; r != reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            std::free(call(f, a.data(), w));
            best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    double tf = time((Fn)fused), tl = time((Fn)layered);
    std::printf("K=%d M=%d D=%d wrong=%d fused_us=%.1f layered_us=%.1f ratio=%.2f\n", K, M, D, bad, tf, tl, tf / tl);
    return bad != 0;
}
