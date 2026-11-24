#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* gemm_dyn(uint64_t, uint64_t, uint64_t, float*, float*);
float* gemm_dyn_nt(uint64_t, uint64_t, uint64_t, float*, float*);
float* gemm_dyn_bias(uint64_t, uint64_t, uint64_t, float*, float*, float*);
float* gemm_dyn_m(uint64_t, float*, float*);
}

static std::vector<float> values(size_t n, size_t seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i) v[i] = float(int((i * 7 + seed) % 19) - 9) / 16.0f;
    return v;
}

static int check(const char* name, float* out, size_t m, size_t k, size_t n, const std::vector<float>& a,
                 const std::vector<float>& b, bool trans_b, const float* bias) {
    int bad = 0;
    for (size_t i = 0; i != m; ++i)
        for (size_t j = 0; j != n; ++j) {
            double s = bias ? bias[j] : 0.0;
            for (size_t p = 0; p != k; ++p) s += double(a[i * k + p]) * (trans_b ? b[j * k + p] : b[p * n + j]);
            if (std::fabs(out[i * n + j] - s) > 1e-3 * (1.0 + std::fabs(s))) ++bad;
        }
    if (bad) std::printf("FAIL %s m=%zu k=%zu n=%zu bad=%d\n", name, m, k, n, bad);
    std::free(out);
    return bad;
}

int main() {
    const size_t ms[] = {0, 1, 3, 4, 8, 13, 24, 37, 130};
    const size_t ks[] = {0, 1, 5, 7, 33};
    const size_t ns[] = {1, 7, 8, 9, 31, 33, 70};
    int bad = 0;
    for (auto m : ms)
        for (auto k : ks)
            for (auto n : ns) {
                auto a = values(m * k, 1), b = values(k * n, 2), bias = values(n, 3);
                bad += check("plain", gemm_dyn(m, k, n, a.data(), b.data()), m, k, n, a, b, false, nullptr);
                bad += check("nt", gemm_dyn_nt(m, k, n, a.data(), b.data()), m, k, n, a, b, true, nullptr);
                bad += check("bias", gemm_dyn_bias(m, k, n, a.data(), b.data(), bias.data()), m, k, n, a, b, false, bias.data());
            }
    for (auto m : ms) {
        auto a = values(m * 33, 4), b = values(33 * 70, 5);
        bad += check("m", gemm_dyn_m(m, a.data(), b.data()), m, 33, 70, a, b, false, nullptr);
    }
    if (!bad) std::printf("dynamic GEMM: all shapes passed\n");
    return bad != 0;
}
