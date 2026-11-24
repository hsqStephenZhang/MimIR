#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* row_sum(float*);
float* col_sum(float*);
float* add2(float*, float*);
float* plane_sum(float*);
}

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static int check(const char* name, float* got, const std::vector<double>& want) {
    int bad = 0;
    for (size_t i = 0; i != want.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-3 * (1.0 + std::fabs(want[i]))) ++bad;
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    std::free(got);
    return bad;
}

int main() {
    int bad = 0;
    auto x = fill(37 * 53, 7, 11), y = fill(37 * 53, 5, 13);
    std::vector<double> rs(37), cs(37), sum(37 * 53), ps(5);
    for (int i = 0; i != 37; ++i)
        for (int j = 0; j != 53; ++j)
            rs[i] += x[i * 53 + j], sum[i * 53 + j] = double(x[i * 53 + j]) + y[i * 53 + j];
    for (int j = 0; j != 53; ++j)
        for (int i = 0; i != 37; ++i)
            cs[i] += x[j * 37 + i];
    auto z = fill(5 * 7 * 11, 3, 7);
    for (int i = 0; i != 5; ++i)
        for (int k = 0; k != 77; ++k)
            ps[i] += z[i * 77 + k];
    bad += check("row_sum", row_sum(x.data()), rs);
    bad += check("col_sum", col_sum(x.data()), cs);
    bad += check("add2", add2(x.data(), y.data()), sum);
    bad += check("plane_sum", plane_sum(z.data()), ps);
    return bad != 0;
}
