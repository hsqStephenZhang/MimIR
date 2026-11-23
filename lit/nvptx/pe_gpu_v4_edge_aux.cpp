#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
float* gemm_n100(float*, float*);
float* gemm_odd(float*, float*);
}

static int check(const char* name, float* (*f)(float*, float*), int m, int k, int n) {
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i != m * k; ++i) a[i] = float(i % 7) - 3;
    for (int i = 0; i != k * n; ++i) b[i] = float(i % 5) - 2;
    float* c = f(a.data(), b.data());
    int bad  = 0;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            double s = 0;
            for (int p = 0; p != k; ++p) s += a[i * k + p] * b[p * n + j];
            if (std::fabs(c[i * n + j] - s) > 1e-3) ++bad;
        }
    if (bad) std::printf("FAIL %s bad=%d\n", name, bad);
    std::free(c);
    return bad;
}

int main() {
    int bad = check("n100", gemm_n100, 64, 64, 100) + check("odd", gemm_odd, 70, 36, 68);
    if (!bad) std::printf("all v4 edge shapes passed\n");
    return bad != 0;
}
