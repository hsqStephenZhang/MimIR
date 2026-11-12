#include <cassert>
#include <cstdlib>

#include <vector>
extern "C" float* full(float*, float*);
extern "C" float* tails(float*, float*);
static void check(int m, int k, int n, float* (*kernel)(float*, float*)) {
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i < m * k; ++i)
        a[i] = float(i % 7 - 3) / 2;
    for (int i = 0; i < k * n; ++i)
        b[i] = float(i % 5 - 2) / 2;
    for (int repeat = 0; repeat < 40; ++repeat) {
        auto c = kernel(a.data(), b.data());
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float expected = 0;
                for (int p = 0; p < k; ++p)
                    expected += a[i * k + p] * b[p * n + j];
                assert(c[i * n + j] == expected);
            }
        std::free(c);
    }
}
int main() {
    check(24, 12, 32, full);
    check(25, 13, 33, tails);
}
