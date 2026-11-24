#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* gemm_gemm(float*, float*, float*);
float* matmul_sequence(float*, float*, float*, float*);
float* gemm_generic(float*, float*, float*);
float* gemm_transpose(float*, float*);
}

using Mat = std::vector<double>;

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static Mat dbl(const std::vector<float>& v) { return Mat(v.begin(), v.end()); }

static Mat mm(const Mat& a, const Mat& b, int M, int K, int N) {
    Mat c(M * N);
    for (int i = 0; i != M; ++i)
        for (int k = 0; k != K; ++k)
            for (int j = 0; j != N; ++j)
                c[i * N + j] += a[i * K + k] * b[k * N + j];
    return c;
}

static int check(const char* name, float* got, const Mat& want) {
    int bad = 0;
    for (size_t i = 0; i != want.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-3 * (1.0 + std::fabs(want[i]))) ++bad;
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    std::free(got);
    return bad;
}

int main() {
    int bad = 0;
    auto a = fill(25 * 12, 7, 11), b0 = fill(12 * 24, 5, 13), b1 = fill(24 * 28, 3, 7);
    bad += check("gemm_gemm", gemm_gemm(a.data(), b0.data(), b1.data()),
                 mm(mm(dbl(a), dbl(b0), 25, 12, 24), dbl(b1), 25, 24, 28));

    auto s2 = fill(24 * 16, 3, 7), s3 = fill(16 * 20, 5, 9);
    bad += check("matmul_sequence", matmul_sequence(a.data(), b0.data(), s2.data(), s3.data()),
                 mm(mm(mm(dbl(a), dbl(b0), 25, 12, 24), dbl(s2), 25, 24, 16), dbl(s3), 25, 16, 20));

    auto g = fill(12 * 28, 5, 13), bias = fill(28, 3, 5);
    Mat want = mm(dbl(a), dbl(g), 25, 12, 28);
    for (int i = 0; i != 25; ++i)
        for (int j = 0; j != 28; ++j)
            want[i * 28 + j] += bias[j];
    bad += check("gemm_generic", gemm_generic(a.data(), g.data(), bias.data()), want);

    Mat ag = mm(dbl(a), dbl(g), 25, 12, 28), tr(28 * 25);
    for (int i = 0; i != 25; ++i)
        for (int j = 0; j != 28; ++j)
            tr[j * 25 + i] = ag[i * 28 + j];
    bad += check("gemm_transpose", gemm_transpose(a.data(), g.data()), tr);
    return bad != 0;
}
