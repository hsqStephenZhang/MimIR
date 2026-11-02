#include <cassert>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <limits>
#include <vector>

extern "C" float* mlp(float*, float*, float*);
extern "C" float* ew_a(float*, float*);
extern "C" float* ew_a_fused(float*, float*);
extern "C" float* row_sum(float*, float*);
extern "C" float* row_max(float*, float*);

static std::vector<float> input(int n, int period) {
    std::vector<float> result(n);
    for (int i = 0; i != n; ++i)
        result[i] = float(i % period - period / 2) / 8;
    return result;
}

int main() {
    auto a = input(13 * 7, 11), b = input(7 * 11, 13), w = input(11 * 5, 7);
    float hidden[13 * 11];
    for (int i = 0; i != 13; ++i)
        for (int j = 0; j != 11; ++j) {
            float value = 0;
            for (int k = 0; k != 7; ++k)
                value += a[i * 7 + k] * b[k * 11 + j];
            hidden[i * 11 + j] = std::max(0.0f, value);
        }
    auto out = mlp(a.data(), b.data(), w.data());
    for (int i = 0; i != 13; ++i)
        for (int j = 0; j != 5; ++j) {
            float expected = 0;
            for (int k = 0; k != 11; ++k)
                expected += hidden[i * 11 + k] * w[k * 5 + j];
            assert(out[i * 5 + j] == expected);
        }
    std::free(out);
    a   = input(5 * 9, 11);
    b   = input(9 * 19, 13);
    out = ew_a(a.data(), b.data());
    for (int i = 0; i != 5; ++i)
        for (int j = 0; j != 19; ++j) {
            float expected = 0;
            for (int k = 0; k != 9; ++k)
                expected += (a[i * 9 + k] + 1) * b[k * 19 + j];
            assert(out[i * 19 + j] == expected);
        }
    std::free(out);
    a   = input(5 * 3, 11);
    b   = input(3 * 11, 13);
    out = ew_a_fused(a.data(), b.data());
    for (int i = 0; i != 5; ++i)
        for (int j = 0; j != 11; ++j) {
            float expected = 0;
            for (int k = 0; k != 3; ++k)
                expected += (a[i * 3 + k] + 1) * b[k * 11 + j];
            assert(out[i * 11 + j] == expected);
        }
    std::free(out);
    a = input(5 * 7, 11);
    b = input(5 * 11, 13);
    for (bool max : {false, true}) {
        out = max ? row_max(a.data(), b.data()) : row_sum(a.data(), b.data());
        for (int i = 0; i != 5; ++i) {
            float value = max ? -std::numeric_limits<float>::infinity() : 0;
            for (int k = 0; k != 7; ++k)
                value = max ? std::max(value, a[i * 7 + k]) : value + a[i * 7 + k];
            for (int j = 0; j != 11; ++j)
                assert(out[i * 11 + j] == b[i * 11 + j] - value);
        }
        std::free(out);
    }
}
