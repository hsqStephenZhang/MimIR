// Times `attn` (H = 12, L = 512, D = 64) and checks sampled outputs against a CPU reference.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" float* attn(float* q, float* k, float* v);

int main(int argc, char** argv) {
    constexpr size_t H = 12, L = 512, D = 64;
    int iters = argc > 1 ? std::atoi(argv[1]) : 5;
    std::vector<float> q(H * L * D), k(H * L * D), v(H * L * D);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = float(i % 97) / 97.0f - 0.5f;
        k[i] = float(i % 89) / 89.0f - 0.5f;
        v[i] = float(i % 83) / 83.0f - 0.5f;
    }
    float* o = attn(q.data(), k.data(), v.data());
    auto t0  = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::free(o);
        o = attn(q.data(), k.data(), v.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("attn end-to-end: %.3f ms/call\n", std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    int failures = 0;
    std::vector<double> s(L);
    for (size_t h : {size_t(0), size_t(11)})
        for (size_t i : {size_t(0), size_t(255), size_t(511)}) {
            double mx = -1e30;
            for (size_t j = 0; j < L; ++j) {
                double acc = 0;
                for (size_t d = 0; d < D; ++d) acc += double(q[(h * L + i) * D + d]) * k[(h * L + j) * D + d];
                s[j] = acc * 0.125;
                mx   = std::fmax(mx, s[j]);
            }
            double sum = 0;
            for (size_t j = 0; j < L; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (size_t d = 0; d < D; d += 13) {
                double ref = 0;
                for (size_t j = 0; j < L; ++j) ref += s[j] / sum * v[(h * L + j) * D + d];
                auto got = o[(h * L + i) * D + d];
                if (std::fabs(ref - got) > 1e-3 * (1 + std::fabs(ref))) {
                    if (failures < 5) std::printf("o[%zu][%zu][%zu] = %g, expected %g\n", h, i, d, got, ref);
                    ++failures;
                }
            }
        }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    std::free(o);
    return failures ? 1 : 0;
}
