#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* chain2(float*, float*, float*);
float* chain3(float*, float*, float*, float*);
float* relu_mm(float*, float*);
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
    auto a = fill(40 * 24, 7, 11), b = fill(24 * 36, 5, 13), d = fill(36 * 20, 3, 7);
    int bad
        = check("chain2", chain2(a.data(), b.data(), d.data()), mm(mm(dbl(a), dbl(b), 40, 24, 36), dbl(d), 40, 36, 20));
    auto a3 = fill(40 * 12, 7, 11), b1 = fill(12 * 24, 5, 13), b2 = fill(24 * 16, 3, 7), b3 = fill(16 * 20, 5, 9);
    bad += check("chain3", chain3(a3.data(), b1.data(), b2.data(), b3.data()),
                 mm(mm(mm(dbl(a3), dbl(b1), 40, 12, 24), dbl(b2), 40, 24, 16), dbl(b3), 40, 16, 20));
    auto ra = fill(40 * 24, 7, 11), rb = fill(24 * 20, 5, 13);
    Mat rd(ra.size());
    for (size_t i = 0; i != ra.size(); ++i)
        rd[i] = ra[i] < 0 ? 0.0 : double(ra[i]);
    bad += check("relu_mm", relu_mm(ra.data(), rb.data()), mm(rd, dbl(rb), 40, 24, 20));
    return bad != 0;
}
