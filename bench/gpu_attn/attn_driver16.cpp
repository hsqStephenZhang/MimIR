// Runs an F16 `attn` with F32 output (H heads of L × D, set with -DH -DL -DD) and checks sampled outputs against a CPU
// reference over the rounded inputs.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef H
#define H 12
#endif
#ifndef L
#define L 512
#endif
#ifndef D
#define D 64
#endif

extern "C" float* attn(_Float16* q, _Float16* k, _Float16* v);

int main(int argc, char** argv) {

    int iters = argc > 1 ? std::atoi(argv[1]) : 5;
    std::vector<_Float16> q(H * L * D), k(H * L * D), v(H * L * D);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = _Float16(float(i % 97) / 97.0f - 0.5f);
        k[i] = _Float16(float(i % 89) / 89.0f - 0.5f);
        v[i] = _Float16(float(i % 83) / 83.0f - 0.5f);
    }
    float* o = attn(q.data(), k.data(), v.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) o = attn(q.data(), k.data(), v.data());
    auto t1 = std::chrono::steady_clock::now();
    std::printf("attn end-to-end: %.3f ms/call\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    int failures = 0;
    std::vector<double> s(L);
    for (size_t h : {size_t(0), size_t(H - 1)})
        for (size_t i : {size_t(0), size_t(L / 2 - 1), size_t(L - 1)}) {
            double mx = -1e30;
            for (size_t j = 0; j < L; ++j) {
                double acc = 0;
                for (size_t d = 0; d < D; ++d) acc += double(float(q[(h * L + i) * D + d])) * float(k[(h * L + j) * D + d]);
                s[j] = acc / std::sqrt(double(D));
                mx   = std::fmax(mx, s[j]);
            }
            double sum = 0;
            for (size_t j = 0; j < L; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (size_t d = 0; d < D; d += 13) {
                double ref = 0;
                for (size_t j = 0; j < L; ++j) ref += s[j] / sum * float(v[(h * L + j) * D + d]);
                auto got = o[(h * L + i) * D + d];
                if (std::fabs(ref - got) > 1e-2 * (1 + std::fabs(ref))) {
                    if (failures < 5) std::printf("o[%zu][%zu][%zu] = %g, expected %g\n", h, i, d, got, ref);
                    ++failures;
                }
            }
        }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
