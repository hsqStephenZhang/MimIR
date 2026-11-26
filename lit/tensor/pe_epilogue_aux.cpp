#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <initializer_list>
#include <vector>

extern "C" float* relu_mm(float*, float*);
extern "C" float* affine_mm(uint64_t, uint64_t, uint64_t, float*, float*);
extern "C" float* bias_mm(float*, float*, float*);
extern "C" float* bias_dyn(uint64_t, uint64_t, uint64_t, float*, float*, float*);

int main() {
    const int M = 37, K = 53, N = 29;
    std::vector<float> a(M * K), b(K * N);
    for (int i = 0; i != M * K; ++i)
        a[i] = float(int(i * 7 % 13) - 6) / 4;
    for (int i = 0; i != K * N; ++i)
        b[i] = float(int(i * 5 % 11) - 5) / 4;
    auto c  = relu_mm(a.data(), b.data());
    int bad = 0;
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j) {
            double s = 0;
            for (int k = 0; k != K; ++k)
                s += double(a[i * K + k]) * b[k * N + j];
            if (std::fabs(c[i * N + j] - std::fmax(s - 1.0, 0.0)) > 1e-3) ++bad;
        }
    if (bad) std::printf("relu_mm: %d wrong\n", bad);
    std::free(c);
    for (int k : {0, 1, 9, 40})
        for (int m : {1, 7, 19})
            for (int n : {3, 33}) {
                auto d = affine_mm(m, k, n, a.data(), b.data());
                for (int i = 0; i != m; ++i)
                    for (int j = 0; j != n; ++j) {
                        double s = 0;
                        for (int l = 0; l != k; ++l)
                            s += double(a[i * k + l]) * b[l * n + j];
                        if (std::fabs(d[i * n + j] - (0.5 * s + 1.0)) > 1e-3) ++bad;
                    }
                std::free(d);
            }
    std::vector<float> bias(N);
    for (int j = 0; j != N; ++j)
        bias[j] = float(j % 7) - 3;
    auto e = bias_mm(a.data(), b.data(), bias.data());
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j) {
            double s = bias[j];
            for (int k = 0; k != K; ++k)
                s += double(a[i * K + k]) * b[k * N + j];
            if (std::fabs(e[i * N + j] - std::fmax(s, 0.0)) > 1e-3) ++bad;
        }
    std::free(e);
    for (int k : {0, 9, 40})
        for (int m : {1, 19})
            for (int n : {3, 29}) {
                auto d = bias_dyn(m, k, n, a.data(), b.data(), bias.data());
                for (int i = 0; i != m; ++i)
                    for (int j = 0; j != n; ++j) {
                        double s = bias[j];
                        for (int l = 0; l != k; ++l)
                            s += double(a[i * k + l]) * b[l * n + j];
                        if (std::fabs(d[i * n + j] - s) > 1e-3) ++bad;
                    }
                std::free(d);
            }
    if (bad) std::printf("%d wrong\n", bad);
    return bad != 0;
}
