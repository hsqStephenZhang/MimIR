#include <cassert>
#include <cstdlib>

#include <vector>

extern "C" float* folded(float*, float*);
extern "C" float* aligned(float*, float*);

void check(float* (*kernel)(float*, float*), int m, int k, int n) {
    std::vector<float> a(m * k), b(k * n);
    for (size_t i = 0; i != a.size(); ++i)
        a[i] = float(int(i % 11) - 5) / 8;
    for (size_t i = 0; i != b.size(); ++i)
        b[i] = float(int(i % 13) - 6) / 8;
    auto out = kernel(a.data(), b.data());
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float expected = 0;
            for (int l = 0; l != k; ++l)
                expected += a[i * k + l] * b[l * n + j];
            assert(out[i * n + j] == expected);
        }

    std::free(out);
}

int main() {
    check(folded, 13, 257, 35);
    check(aligned, 8, 256, 32);
}
