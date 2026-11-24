#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* relu_mm(uint64_t, uint64_t, uint64_t, float*, float*);
float* chain(uint64_t, float*, float*, float*);
}

using Mat = std::vector<double>;

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static Mat dbl(const std::vector<float>& v) { return Mat(v.begin(), v.end()); }

static Mat mm(const Mat& a, const Mat& b, size_t M, size_t K, size_t N) {
    Mat c(M * N);
    for (size_t i = 0; i != M; ++i)
        for (size_t k = 0; k != K; ++k)
            for (size_t j = 0; j != N; ++j)
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
    for (size_t m : {0, 1, 7, 8, 13, 40})
        for (size_t k : {1, 5, 24})
            for (size_t n : {1, 9, 16, 37}) {
                auto a = fill(m * k, 7, 11), b = fill(k * n, 5, 13);
                Mat r(a.size());
                for (size_t i = 0; i != a.size(); ++i) r[i] = a[i] < 0 ? 0.0 : double(a[i]);
                bad += check("relu_mm", relu_mm(m, k, n, a.data(), b.data()), mm(r, dbl(b), m, k, n));
            }
    for (size_t m : {0, 1, 7, 8, 13, 40}) {
        auto a = fill(m * 24, 7, 11), b = fill(24 * 36, 5, 13), d = fill(36 * 20, 3, 7);
        bad += check("chain", chain(m, a.data(), b.data(), d.data()), mm(mm(dbl(a), dbl(b), m, 24, 36), dbl(d), m, 36, 20));
    }
    if (!bad) std::printf("dynamic fusion: all shapes passed\n");
    return bad != 0;
}
