#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* peel_mm(uint64_t, float*, float*);
float* peel_tr(uint64_t, float*, float*);
float* epi_mm(uint64_t, float*, float*);
}

static std::vector<float> fill(size_t n, int p) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int(i * p % 7) - 3);
    return v;
}

int main() {
    int bad = 0;
    for (uint64_t r : {1, 4, 5, 8, 11}) {
        auto a = fill(r * 29, 3), b = fill(29 * 37, 5);
        float* x = peel_mm(r, a.data(), b.data());
        float* y = peel_tr(r, a.data(), b.data());
        for (uint64_t i = 0; i != r; ++i)
            for (uint64_t j = 0; j != 37; ++j) {
                float want = 0;
                for (uint64_t k = 0; k != 29; ++k)
                    want += a[i * 29 + k] * b[k * 37 + j];
                bad += x[i * 37 + j] != want;
                bad += y[j * r + i] != want;
            }
        std::free(x), std::free(y);
    }
    for (uint64_t k : {0, 1, 5}) {
        auto a = fill(5 * k, 3), b = fill(k * 7, 5);
        float* y = epi_mm(k, a.data(), b.data());
        for (uint64_t i = 0; i != 5; ++i)
            for (uint64_t j = 0; j != 7; ++j) {
                float want = 1;
                for (uint64_t l = 0; l != k; ++l)
                    want += a[i * k + l] * b[l * 7 + j];
                bad += y[i * 7 + j] != want;
            }
        std::free(y);
    }
    if (bad) std::printf("%d wrong\n", bad);
    return bad != 0;
}
