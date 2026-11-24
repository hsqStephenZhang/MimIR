#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>

extern "C" {
float* chain(float*, float*, float*);
float* relu_mm(float*, float*);
float* chain_relu(float*, float*, float*);
}

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static std::vector<double> mm(const std::vector<double>& a, const std::vector<double>& b, int M, int K, int N) {
    std::vector<double> c(M * N);
    for (int i = 0; i != M; ++i)
        for (int k = 0; k != K; ++k)
            for (int j = 0; j != N; ++j)
                c[i * N + j] += a[i * K + k] * b[k * N + j];
    return c;
}

static int check(const char* name, const float* got, const std::vector<double>& want) {
    int bad = 0;
    for (size_t i = 0; i != want.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-3 * (1.0 + std::fabs(want[i]))) ++bad;
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    return bad;
}

int main() {
    int bad = 0;
    {
        auto a = fill(26 * 20, 7, 11), b = fill(20 * 28, 5, 13), d = fill(28 * 12, 3, 7);
        std::vector<double> ad(a.begin(), a.end()), bd(b.begin(), b.end()), dd(d.begin(), d.end());
        float* e = chain(a.data(), b.data(), d.data());
        bad += check("chain", e, mm(mm(ad, bd, 26, 20, 28), dd, 26, 28, 12));
        std::free(e);
    }
    {
        auto a = fill(19 * 10, 7, 11), b = fill(10 * 9, 5, 13);
        std::vector<double> ad(a.size()), bd(b.begin(), b.end());
        std::transform(a.begin(), a.end(), ad.begin(), [](float x) { return std::max(double(x), 0.0); });
        float* c = relu_mm(a.data(), b.data());
        bad += check("relu_mm", c, mm(ad, bd, 19, 10, 9));
        std::free(c);
    }
    {
        auto a = fill(26 * 20, 7, 11), b = fill(20 * 28, 5, 13), d = fill(28 * 12, 3, 7);
        std::vector<double> ad(a.size()), bd(b.begin(), b.end()), dd(d.begin(), d.end());
        std::transform(a.begin(), a.end(), ad.begin(), [](float x) { return std::max(double(x), 0.0); });
        float* e = chain_relu(a.data(), b.data(), d.data());
        bad += check("chain_relu", e, mm(mm(ad, bd, 26, 20, 28), dd, 26, 28, 12));
        std::free(e);
    }
    return bad != 0;
}
