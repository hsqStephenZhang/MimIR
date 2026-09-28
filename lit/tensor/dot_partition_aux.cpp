// Driver for dot_partition_exec.mim: calls the LLVM-compiled products and checks every output against a scalar reference.
// Exit code: 0 if all outputs match within tolerance, 1 otherwise.

#include <cmath>
#include <cstdio>

#include <vector>

extern "C" {
// `«s; F32»` tensors lower to plain pointers at the C ABI (see hlo_aux.cpp).
float* mm49(float* a, float* b);
float* mm65(float* a, float* b);
float* mm147(float* a, float* b);
float* mm245(float* a, float* b);
}

static int check(const char* name, float* (*f)(float*, float*), size_t M, size_t K, size_t N) {
    std::vector<float> a(M * K), b(K * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;

    float* out = f(a.data(), b.data());

    int bad = 0;
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double s = 0.0;
            for (size_t k = 0; k < K; ++k)
                s += double(a[m * K + k]) * double(b[k * N + n]);
            float ref = float(s);
            float got = out[m * N + n];
            if (std::fabs(ref - got) > 1e-4f + 1e-4f * std::fabs(ref))
                if (++bad <= 5) std::printf("%s[%zu,%zu]: got %f want %f\n", name, m, n, got, ref);
        }
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int bad = 0;
    bad += check("mm49", mm49, 6, 49, 24);
    bad += check("mm65", mm65, 6, 65, 24);
    bad += check("mm147", mm147, 6, 147, 24);
    bad += check("mm245", mm245, 6, 245, 24);
    return bad ? 1 : 0;
}
