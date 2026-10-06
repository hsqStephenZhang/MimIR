// Driver for dot_par_small.mim: checks mm's result against a scalar reference under real parallelism.
// Exit code: 0 if correct, 1 otherwise.

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
float* mm(float* a, float* b);
}

int main() {
    const size_t M = 32, K = 256, N = 512;
    std::vector<float> a(M * K), b(K * N);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;

    float* out = mm(a.data(), b.data());

    int bad = 0;
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k)
                s += static_cast<double>(a[m * K + k]) * b[k * N + n];
            if (std::fabs(out[m * N + n] - s) > 1e-3) {
                if (bad < 5) std::printf("mm[%zu][%zu] = %f, want %f\n", m, n, out[m * N + n], s);
                ++bad;
            }
        }
    return bad ? 1 : 0;
}
