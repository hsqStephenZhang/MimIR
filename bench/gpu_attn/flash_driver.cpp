// Runs `flash` (H heads of L × 64, set with -DH -DL) and checks sampled outputs against a CPU reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef H
#define H 32
#endif
#ifndef L
#define L 1024
#endif
constexpr size_t D = 64;

extern "C" int flash(float* q, float* k, float* v, float* o);

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 1;
    std::vector<float> q(H * L * D), k(H * L * D), v(H * L * D), o(H * L * D);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = float(i % 97) / 97.0f - 0.5f;
        k[i] = float(i % 89) / 89.0f - 0.5f;
        v[i] = float(i % 83) / 83.0f - 0.5f;
    }
    for (int it = 0; it < iters; ++it) flash(q.data(), k.data(), v.data(), o.data());
    int failures = 0;
    std::vector<double> s(L);
    for (size_t h : {size_t(0), size_t(H / 2), size_t(H - 1)})
        for (size_t i : {size_t(0), size_t(17), size_t(L / 2 - 1), size_t(L - 1)}) {
            double mx = -1e30;
            for (size_t j = 0; j < L; ++j) {
                double acc = 0;
                for (size_t d = 0; d < D; ++d) acc += double(q[(h * L + i) * D + d]) * k[(h * L + j) * D + d];
                mx = std::fmax(mx, s[j] = acc / std::sqrt(double(D)));
            }
            double sum = 0;
            for (size_t j = 0; j < L; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (size_t d = 0; d < D; ++d) {
                double ref = 0;
                for (size_t j = 0; j < L; ++j) ref += s[j] / sum * v[(h * L + j) * D + d];
                auto got = o[(h * L + i) * D + d];
                if (std::fabs(ref - got) > 2e-3 * (1 + std::fabs(ref))) {
                    if (failures < 5) std::printf("o[%zu][%zu][%zu] = %g, expected %g\n", h, i, d, got, ref);
                    ++failures;
                }
            }
        }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
