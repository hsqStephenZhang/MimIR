#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* gemm_div(float*, float*);
float* gemm_tail(float*, float*);
float* gemm_m1(float*, float*);
float* gemm_n1(float*, float*);
float* gemm_bmm(float*, float*);
float* gemm_nt(float*, float*);
float* gemm_bias(float*, float*, float*);
float* gemm_chain(float*, float*, float*);
}

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i) v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static int check_gemm(const char* name, float* out, const float* a, const float* b, int B, int M, int K, int N,
                      bool trans_b = false, const float* bias = nullptr) {
    int bad = 0;
    for (int bi = 0; bi != B; ++bi)
        for (int i = 0; i != M; ++i)
            for (int j = 0; j != N; ++j) {
                double acc = bias ? double(bias[j]) : 0.0;
                for (int k = 0; k != K; ++k) {
                    double av = a[(bi * M + i) * K + k];
                    double bv = trans_b ? b[(bi * N + j) * K + k] : b[(bi * K + k) * N + j];
                    acc += av * bv;
                }
                double got = out[(bi * M + i) * N + j];
                if (std::fabs(got - acc) > 1e-3 * (1.0 + std::fabs(acc))) ++bad;
            }
    if (bad) std::printf("FAIL %s: bad=%d\n", name, bad);
    std::free(out);
    return bad;
}

int main() {
    int bad = 0;
    {
        auto a = fill(32 * 32, 7, 13), b = fill(32 * 32, 5, 17);
        bad += check_gemm("div", gemm_div(a.data(), b.data()), a.data(), b.data(), 1, 32, 32, 32);
    }
    {
        auto a = fill(19 * 29, 7, 13), b = fill(29 * 37, 5, 17);
        bad += check_gemm("tail", gemm_tail(a.data(), b.data()), a.data(), b.data(), 1, 19, 29, 37);
    }
    {
        auto a = fill(1 * 23, 3, 11), b = fill(23 * 35, 5, 13);
        bad += check_gemm("m1", gemm_m1(a.data(), b.data()), a.data(), b.data(), 1, 1, 23, 35);
    }
    {
        auto a = fill(13 * 23, 3, 11), b = fill(23 * 1, 5, 13);
        bad += check_gemm("n1", gemm_n1(a.data(), b.data()), a.data(), b.data(), 1, 13, 23, 1);
    }
    {
        auto a = fill(3 * 13 * 19, 7, 13), b = fill(3 * 19 * 21, 5, 17);
        bad += check_gemm("bmm", gemm_bmm(a.data(), b.data()), a.data(), b.data(), 3, 13, 19, 21);
    }
    {
        auto a = fill(15 * 25, 7, 13), b = fill(21 * 25, 5, 17);
        bad += check_gemm("nt", gemm_nt(a.data(), b.data()), a.data(), b.data(), 1, 15, 25, 21, true);
    }
    {
        auto a = fill(17 * 23, 7, 13), b = fill(23 * 19, 5, 17), bias = fill(19, 3, 11);
        bad += check_gemm("bias", gemm_bias(a.data(), b.data(), bias.data()), a.data(), b.data(), 1, 17, 23, 19, false,
                          bias.data());
    }
    {
        auto a = fill(18 * 20, 7, 13), b = fill(20 * 22, 5, 17), c = fill(22 * 17, 3, 11);
        std::vector<float> ab(18 * 22, 0.0f);
        for (int i = 0; i != 18; ++i)
            for (int j = 0; j != 22; ++j) {
                double s = 0;
                for (int k = 0; k != 20; ++k) s += double(a[i * 20 + k]) * b[k * 22 + j];
                ab[i * 22 + j] = float(s);
            }
        bad += check_gemm("chain", gemm_chain(a.data(), b.data(), c.data()), ab.data(), c.data(), 1, 18, 22, 17);
    }
    return bad != 0;
}
