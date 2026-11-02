#include <cassert>
#include <cstdlib>

extern "C" float* gemm(float*, float*);
extern "C" float* outer(float*, float*);
extern "C" float* reverse(float*, float*);

int main() {
    float a[13 * 7], b[7 * 11];
    for (int i = 0; i != 13 * 7; ++i)
        a[i] = float(i % 11 - 5) / 8;
    for (int i = 0; i != 7 * 11; ++i)
        b[i] = float(i % 13 - 6) / 8;
    auto product = gemm(a, b);
    for (int i = 0; i != 13; ++i)
        for (int j = 0; j != 11; ++j) {
            float expected = 0;
            for (int k = 0; k != 7; ++k)
                expected += a[i * 7 + k] * b[k * 11 + j];
            assert(product[i * 11 + j] == expected);
        }
    std::free(product);
    auto elementwise = outer(a, b);
    for (int i = 0; i != 5 * 7; ++i)
        assert(elementwise[i] == b[i] - (a[i] + 1));
    std::free(elementwise);
    auto reduced = reverse(a, b);
    for (int i = 0; i != 5; ++i) {
        float expected = 0;
        for (int j = 0; j != 7; ++j) {
            float value = 0;
            for (int k = 0; k != 3; ++k)
                value += a[i * 3 + k] * b[k * 7 + j];
            expected += value;
        }
        assert(reduced[i] == expected);
    }
    std::free(reduced);
}
