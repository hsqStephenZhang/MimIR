#include <cassert>
#include <cstdlib>

#include <vector>

extern "C" float* rule_full(float*, float*);
extern "C" float* hand_full(float*, float*);
extern "C" float* rule_tail(float*, float*);
extern "C" float* hand_tail(float*, float*);
extern "C" float rule_skinny(float*, float*);
extern "C" float hand_skinny(float*, float*);
extern "C" float* rule_register_tail(float*, float*);
extern "C" float* hand_register_tail(float*, float*);

static void check(int m, int k, int n, float* (*compute)(float*, float*)) {
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i < m * k; ++i)
        a[i] = float(i % 11 - 5) / 8;
    for (int i = 0; i < k * n; ++i)
        b[i] = float(i % 13 - 6) / 8;
    auto out = compute(a.data(), b.data());
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float expected = 0;
            for (int p = 0; p < k; ++p)
                expected += a[i * k + p] * b[p * n + j];
            assert(out[i * n + j] == expected);
        }
    std::free(out);
}

int main() {
    check(8, 8, 8, rule_full);
    check(8, 8, 8, hand_full);
    check(13, 7, 11, rule_tail);
    check(13, 7, 11, hand_tail);
    check(5, 9, 19, rule_register_tail);
    check(5, 9, 19, hand_register_tail);
    float a[7] = {1, 2, 3, 4, 5, 6, 7};
    assert(rule_skinny(a, a) == 140 && hand_skinny(a, a) == 140);
}
