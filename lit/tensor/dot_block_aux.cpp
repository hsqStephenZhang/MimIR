// Driver for dot_block_exec.mim: calls the LLVM-compiled products and checks every output against a scalar reference.
// Exit code: 0 if all outputs match within tolerance, 1 otherwise.

#include <cmath>
#include <cstdio>

#include <vector>

extern "C" {
// `«s; F32»` tensors lower to plain pointers at the C ABI (see hlo_aux.cpp).
float* nblk(float* a, float* b);
float* kblk(float* a, float* b);
float* nblk_fma(float* a, float* b);
float* conv(float* w, float* c);
float* pack(float* a, float* b);
float* pack_conv(float* w, float* c);
float* part(float* a, float* b);
}

// out[m, b, n] = sum_k a[m, k] * c[b, k, n] for B batches of the right operand (B = 1 for a plain product).
static int check(const char* name, float* (*f)(float*, float*), size_t M, size_t K, size_t B, size_t N) {
    std::vector<float> a(M * K), c(B * K * N);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float>(i % 13) / 13.0f - 0.4f;
    for (size_t i = 0; i < c.size(); ++i)
        c[i] = static_cast<float>((7 * i) % 11) / 11.0f - 0.5f;

    float* out = f(a.data(), c.data());

    int bad = 0;
    for (size_t m = 0; m < M; ++m)
        for (size_t b = 0; b < B; ++b)
            for (size_t n = 0; n < N; ++n) {
                double s = 0.0;
                for (size_t k = 0; k < K; ++k)
                    s += double(a[m * K + k]) * double(c[(b * K + k) * N + n]);
                float ref = float(s);
                float got = out[(m * B + b) * N + n];
                if (std::fabs(ref - got) > 1e-3f + 1e-4f * std::fabs(ref))
                    if (++bad <= 5) std::printf("%s[%zu,%zu,%zu]: got %f want %f\n", name, m, b, n, got, ref);
            }
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int bad = 0;
    bad += check("nblk", nblk, 8, 100, 1, 512);
    bad += check("kblk", kblk, 8, 2304, 1, 512);
    bad += check("nblk_fma", nblk_fma, 8, 100, 1, 512);
    bad += check("conv", conv, 6, 96, 2, 640);
    bad += check("pack", pack, 64, 130, 1, 512);
    bad += check("pack_conv", pack_conv, 64, 147, 2, 640);
    bad += check("part", part, 20, 100, 1, 512);
    return bad ? 1 : 0;
}
