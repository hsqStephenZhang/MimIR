#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef COUNT_MALLOC
static size_t a_panels, b_panels;
extern "C" void* counted_malloc(size_t size) {
    a_panels += size == 8 * 4 * sizeof(float);
    b_panels += size == 4 * 16 * sizeof(float);
    return std::malloc(size);
}
#endif

extern "C" float* full(float*, float*);
extern "C" float* tails(float*, float*);
extern "C" float* row_major(float*, float*);

static void check(int m, int k, int n, float* (*kernel)(float*, float*)) {
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i != m * k; ++i) a[i] = float(i % 11 - 5) / 8;
    for (int i = 0; i != k * n; ++i) b[i] = float(i % 13 - 6) / 8;
    auto c = kernel(a.data(), b.data());
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            double ref = 0;
            for (int p = 0; p != k; ++p) ref += double(a[i * k + p]) * b[p * n + j];
            assert(std::abs(c[i * n + j] - ref) < 1e-5);
        }
    std::free(c);
}

int main() {
    check(24, 12, 32, full);
#ifdef COUNT_MALLOC
    // Two NC blocks times three KC blocks. B must not be packed again for each of the three MC blocks;
    // A must be packed once per MC, rather than once per MR microtile.
    assert(b_panels == 6);
    assert(a_panels == 18);
    a_panels = b_panels = 0;
#endif
    check(24, 12, 32, row_major);
#ifdef COUNT_MALLOC
    assert(a_panels == 9);
    assert(b_panels == 18);
#endif
    check(22, 11, 29, tails);
}
