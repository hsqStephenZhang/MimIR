// Runs a 16-bit `flash` (H heads of L × 64, set with -DH -DL, element type -DBF16 or F16) and checks sampled outputs against a CPU reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Bf16 {
    unsigned short bits;
    Bf16() = default;
    explicit Bf16(float f) {
        unsigned u;
        std::memcpy(&u, &f, 4);
        bits = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    explicit operator float() const {
        unsigned u = unsigned(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
};
#ifdef BF16
using E = Bf16;
constexpr double Tol = 3e-2;
#else
using E = _Float16;
constexpr double Tol = 1e-2;
#endif

#ifndef H
#define H 32
#endif
#ifndef L
#define L 1024
#endif
constexpr size_t D = 64;

extern "C" int flash(E* q, E* k, E* v, E* o);

int main(int argc, char** argv) {
    int iters = argc > 1 ? std::atoi(argv[1]) : 1;
    std::vector<E> q(H * L * D), k(H * L * D), v(H * L * D), o(H * L * D);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = E(float(i % 97) / 97.0f - 0.5f);
        k[i] = E(float(i % 89) / 89.0f - 0.5f);
        v[i] = E(float(i % 83) / 83.0f - 0.5f);
    }
    for (int it = 0; it < iters; ++it) flash(q.data(), k.data(), v.data(), o.data());
    int failures = 0;
    std::vector<double> s(L);
    for (size_t h : {size_t(0), size_t(H / 2), size_t(H - 1)})
        for (size_t i : {size_t(0), size_t(17), size_t(L / 2 - 1), size_t(L - 1)}) {
            double mx = -1e30;
            for (size_t j = 0; j < L; ++j) {
                double acc = 0;
                for (size_t d = 0; d < D; ++d) acc += double(float(q[(h * L + i) * D + d])) * float(k[(h * L + j) * D + d]);
                mx = std::fmax(mx, s[j] = acc / std::sqrt(double(D)));
            }
            double sum = 0;
            for (size_t j = 0; j < L; ++j) sum += (s[j] = std::exp(s[j] - mx));
            for (size_t d = 0; d < D; ++d) {
                double ref = 0;
                for (size_t j = 0; j < L; ++j) ref += s[j] / sum * float(v[(h * L + j) * D + d]);
                double got = float(o[(h * L + i) * D + d]);
                if (std::fabs(ref - got) > Tol * (1 + std::fabs(ref))) {
                    if (failures < 5) std::printf("o[%zu][%zu][%zu] = %g, expected %g\n", h, i, d, got, ref);
                    ++failures;
                }
            }
        }
    std::printf("%s\n", failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
