#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* flip_mm(float*, float*);
float* flip_add(float*, float*);
float* bcast_add(float*, float*);
float* reshape_add(float*, float*);
float* reshape_mm(unsigned long, float*, float*);
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
    {
        auto a = fill(21 * 17, 7, 11), b = fill(17 * 13, 5, 13);
        std::vector<double> c(21 * 13);
        for (int i = 0; i != 21; ++i)
            for (int k = 0; k != 17; ++k)
                for (int j = 0; j != 13; ++j)
                    c[i * 13 + j] += double(a[(20 - i) * 17 + (16 - k)]) * b[k * 13 + j];
        bad += check("flip_mm", flip_mm(a.data(), b.data()), c);
    }
    {
        auto x = fill(19 * 23, 7, 11), y = fill(19 * 23, 3, 7);
        std::vector<double> z(19 * 23);
        for (int i = 0; i != 19; ++i)
            for (int j = 0; j != 23; ++j)
                z[i * 23 + j] = double(x[(18 - i) * 23 + (22 - j)]) + y[i * 23 + j];
        bad += check("flip_add", flip_add(x.data(), y.data()), z);
    }
    {
        auto x = fill(31, 7, 11), y = fill(5 * 31, 3, 7);
        std::vector<double> z(5 * 31);
        for (int i = 0; i != 5; ++i)
            for (int j = 0; j != 31; ++j)
                z[i * 31 + j] = double(x[j]) + y[i * 31 + j];
        bad += check("bcast_add", bcast_add(x.data(), y.data()), z);
    }
    {
        auto x = fill(24, 7, 11), y = fill(24, 3, 7);
        std::vector<double> z(24);
        for (int i = 0; i != 24; ++i)
            z[i] = double(x[i]) + y[i];
        bad += check("reshape_add", reshape_add(x.data(), y.data()), z);
    }
    for (int m : {1, 3, 7, 16, 37}) {
        auto a = fill(m * 8, 7, 11), b = fill(8 * 5, 5, 13);
        std::vector<double> c(m * 5);
        for (int i = 0; i != m; ++i)
            for (int k = 0; k != 8; ++k)
                for (int j = 0; j != 5; ++j)
                    c[i * 5 + j] += double(a[i * 8 + k]) * b[k * 5 + j];
        bad += check("reshape_mm", reshape_mm(m, a.data(), b.data()), c);
    }
    return bad != 0;
}
