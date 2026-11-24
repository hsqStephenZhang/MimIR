#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <functional>
#include <vector>

extern "C" {
float* gemm_floor(float*, float*);
float* gemm_ceil(float*, float*);
float* gemm_mod(float*, float*);
}

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static int check(const char* name,
                 float* got,
                 const std::vector<float>& a,
                 const std::vector<float>& b,
                 const std::function<int(int)>& row) {
    int bad = 0;
    for (int i = 0; i != 17; ++i)
        for (int j = 0; j != 19; ++j) {
            double want = 0;
            for (int k = 0; k != 23; ++k)
                want += double(a[i * 23 + k]) * b[row(k) * 19 + j];
            if (std::fabs(got[i * 19 + j] - want) > 1e-3 * (1.0 + std::fabs(want))) ++bad;
        }
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    std::free(got);
    return bad;
}

int main() {
    auto a = fill(17 * 23, 7, 11), b = fill(12 * 19, 5, 13), b5 = fill(5 * 19, 3, 7);
    int bad = check("floor", gemm_floor(a.data(), b.data()), a, b, [](int k) { return k / 2; });
    bad += check("ceil", gemm_ceil(a.data(), b.data()), a, b, [](int k) { return (k + 1) / 2; });
    bad += check("mod", gemm_mod(a.data(), b5.data()), a, b5, [](int k) { return (k + 3) % 5; });
    return bad != 0;
}
