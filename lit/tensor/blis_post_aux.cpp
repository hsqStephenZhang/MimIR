#include <cassert>
#include <cstdlib>

#include <algorithm>
#include <vector>

extern "C" float* rule_full(float*, float*, float*);
extern "C" float* hand_full(float*, float*, float*);
extern "C" float* rule_tail(float*, float*, float*);
extern "C" float* hand_tail(float*, float*, float*);
extern "C" float* rule_register_tail(float*, float*, float*);
extern "C" float* hand_register_tail(float*, float*, float*);
extern "C" float* rule_nc_tail(float*, float*, float*);
extern "C" float* hand_nc_tail(float*, float*, float*);
extern "C" float* rule_skinny(float*, float*, float*);
extern "C" float* hand_skinny(float*, float*, float*);

static void check(int m, int k, int n, float* (*compute)(float*, float*, float*)) {
    std::vector<float> a(m * k), b(k * n), bias(n);
    for (int i = 0; i != m * k; ++i)
        a[i] = float(i % 11 - 5) / 8;
    for (int i = 0; i != k * n; ++i)
        b[i] = float(i % 13 - 6) / 8;
    for (int j = 0; j != n; ++j)
        bias[j] = float(j % 7 - 3) / 4;
    auto out = compute(a.data(), b.data(), bias.data());
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            float expected = 0;
            for (int p = 0; p != k; ++p)
                expected += a[i * k + p] * b[p * n + j];
            assert(out[i * n + j] == std::max(0.0f, expected + bias[j]));
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
    check(5, 11, 21, rule_nc_tail);
    check(5, 11, 21, hand_nc_tail);
    float a[7] = {1, -2, 3, -4, 5, -6, 7};
    for (float bias : {-200.0f, 3.0f}) {
        auto rule     = rule_skinny(a, a, &bias);
        auto hand     = hand_skinny(a, a, &bias);
        auto expected = bias < 0 ? 0.0f : 143.0f;
        assert(rule[0] == expected && hand[0] == expected);
        std::free(rule);
        std::free(hand);
    }
}
