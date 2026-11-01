#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <vector>

extern "C" float* block_full(float*, float*);
extern "C" float* block_tail(float*, float*);
extern "C" float* block_empty(float*);
extern "C" int64_t* block_order(int64_t*);

static void check(int m, int k, int n, float* (*compute)(float*, float*)) {
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i < m * k; ++i)
        a[i] = float(i % 7 - 3) / 2;
    for (int i = 0; i < k * n; ++i)
        b[i] = float(i % 5 - 2) / 2;
    auto out = compute(a.data(), b.data());
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float expected = 1;
            for (int p = 0; p < k; ++p)
                expected += a[i * k + p] * b[p * n + j];
            assert(out[i * n + j] == expected);
        }
    std::free(out);
}

int main() {
    check(4, 24, 16, block_full);
    check(3, 7, 5, block_tail);
    float input[64] = {};
    auto out        = block_empty(input);
    for (int i = 0; i < 64; ++i)
        assert(out[i] == 1);
    std::free(out);
    int64_t digits[5] = {1, 2, 3, 4, 5};
    auto ordered      = block_order(digits);
    for (int i = 0; i < 6; ++i)
        assert(ordered[i] == 12345);
    std::free(ordered);
}
