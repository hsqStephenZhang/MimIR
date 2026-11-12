#include <cassert>
#include <cmath>
#include <cstdlib>

#include <array>

extern "C" float* batched(float*, float*, float*);
extern "C" float* batch_nt(float*, float*, float*);
extern "C" float* batch_tn(float*, float*, float*);
extern "C" float* single(float*, float*, float*);
extern "C" float* prefix(float*, float*, float*);
extern "C" float* single_nt(float*, float*, float*);
extern "C" float* single_tn(float*, float*, float*);
extern "C" float* prefix_unit(float*, float*, float*);
extern "C" float* row_unit(float*, float*, float*);
extern "C" float* col_unit(float*, float*, float*);

int main() {
    std::array<float, 6 * 13 * 7> a, at;
    std::array<float, 6 * 7 * 11> b, bt;
    std::array<float, 6 * 11> c;
    for (int h = 0; h != 6; ++h) {
        for (int i = 0; i != 13; ++i)
            for (int k = 0; k != 7; ++k)
                at[h * 91 + k * 13 + i] = a[h * 91 + i * 7 + k] = float((h * 11 + i * 3 + k * 7) % 17 - 8) / 8;
        for (int k = 0; k != 7; ++k)
            for (int j = 0; j != 11; ++j)
                bt[h * 77 + j * 7 + k] = b[h * 77 + k * 11 + j] = float((h * 7 + k * 5 + j * 3) % 19 - 9) / 8;
        for (int j = 0; j != 11; ++j)
            c[h * 11 + j] = float(h + j - 5) / 4;
    }
    auto check = [&](auto kernel, float* lhs, float* rhs, int count, int m, int n, bool ta = false, bool tb = false) {
        auto out = kernel(lhs, rhs, c.data());
        for (int h = 0; h != count; ++h)
            for (int i = 0; i != m; ++i)
                for (int j = 0; j != n; ++j) {
                    double expected = c[h * n + j];
                    for (int k = 0; k != 7; ++k)
                        expected += lhs[h * m * 7 + (ta ? k * m + i : i * 7 + k)]
                                  * rhs[h * 7 * n + (tb ? j * 7 + k : k * n + j)];
                    assert(std::abs(out[h * m * n + i * n + j] - expected) < 0.00001);
                }
        std::free(out);
    };
    check(batched, a.data(), b.data(), 2, 13, 11);
    check(batch_nt, a.data(), bt.data(), 2, 13, 11, false, true);
    check(batch_tn, at.data(), b.data(), 2, 13, 11, true);
    check(single, a.data(), b.data(), 1, 13, 11);
    check(single_nt, a.data(), bt.data(), 1, 13, 11, false, true);
    check(single_tn, at.data(), b.data(), 1, 13, 11, true);
    check(prefix, a.data(), bt.data(), 6, 13, 11, false, true);
    check(prefix_unit, a.data(), bt.data(), 2, 13, 11, false, true);
    check(row_unit, a.data(), b.data(), 3, 1, 11);
    check(col_unit, a.data(), b.data(), 3, 13, 1);
}
