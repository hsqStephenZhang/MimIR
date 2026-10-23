// Driver for prepack.mim: calls each GEMM with a weight, the same weight again, another one and the first again.
// Exit code: 0 if all outputs match a scalar reference, 1 otherwise.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* pp(float* a, float* b);
float* pp1(float* a, float* b);
float* pp_t(float* a, float* b);
float* pp_c(float* a, float* b);
}

// `scale` multiplies the weight `pp_c` reads, as it adds it to itself.
static int check(const char* name, float* (*f)(float*, float*), size_t M, size_t K, size_t N, bool trans, float scale) {
    std::vector<float> a(M * K), w1(K * N), w2(K * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < w1.size(); ++i) {
        w1[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;
        w2[i] = static_cast<float>((5 * i) % 17) / 17.0f - 0.3f;
    }

    int bad = 0;
    for (auto w : {&w1, &w1, &w2, &w1}) {
        float* out = f(a.data(), w->data());
        for (size_t m = 0; m < M; ++m)
            for (size_t n = 0; n < N; ++n) {
                double s = 0.0;
                for (size_t k = 0; k < K; ++k)
                    s += double(a[m * K + k]) * double((*w)[trans ? n * K + k : k * N + n]);
                auto ref = float(s * scale), got = out[m * N + n];
                if (std::fabs(ref - got) > 1e-3f + 1e-4f * std::fabs(ref) && ++bad <= 5)
                    std::printf("%s[%zu,%zu]: got %f want %f\n", name, m, n, got, ref);
            }
        std::free(out);
    }
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int bad = check("pp", pp, 37, 513, 100, false, 1.0f) + check("pp1", pp1, 6, 128, 32, false, 1.0f)
            + check("pp_t", pp_t, 7, 300, 96, true, 1.0f) + check("pp_c", pp_c, 6, 128, 32, false, 2.0f);
    return bad != 0;
}
