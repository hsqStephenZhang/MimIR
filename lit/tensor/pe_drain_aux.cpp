#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <vector>

struct Two {
    float *a, *b;
};

extern "C" {
Two mm_two(float*, float*, float*);
Two mm_stats(float*, float*);
Two center(float*);
Two mm_two_red(float*, float*);
Two dyn_stats(unsigned long, unsigned long, float*, float*);
}

using Mat = std::vector<double>;

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static Mat dbl(const std::vector<float>& v) { return Mat(v.begin(), v.end()); }

static Mat mm(const Mat& a, const Mat& b, int M, int K, int N) {
    Mat c(M * N);
    for (int i = 0; i != M; ++i)
        for (int k = 0; k != K; ++k)
            for (int j = 0; j != N; ++j)
                c[i * N + j] += a[i * K + k] * b[k * N + j];
    return c;
}

static Mat relu(Mat a) {
    for (auto& x : a)
        x = std::max(x, 0.0);
    return a;
}

static Mat row_sums(const Mat& a, int M, int N) {
    Mat r(M);
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j)
            r[i] += a[i * N + j];
    return r;
}

static int check(const char* name, float* got, const Mat& want) {
    int bad = 0;
    for (size_t i = 0; i != want.size(); ++i)
        if (std::fabs(got[i] - want[i]) > 1e-3 * (1.0 + std::fabs(want[i]))) ++bad;
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    std::free(got);
    return bad;
}

int main() {
    int bad = 0;
    auto a = fill(40 * 24, 7, 11), b = fill(24 * 36, 5, 13), bias = fill(40 * 36, 3, 7);
    auto p = mm(dbl(a), dbl(b), 40, 24, 36);
    {
        auto [r, s] = mm_two(a.data(), b.data(), bias.data());
        Mat sum(p);
        for (size_t i = 0; i != sum.size(); ++i)
            sum[i] += bias[i];
        bad += check("mm_two relu", r, relu(p));
        bad += check("mm_two bias", s, sum);
    }
    {
        auto [rows, cols] = mm_stats(a.data(), b.data());
        Mat maxima(36, -1000.0);
        for (int i = 0; i != 40; ++i)
            for (int j = 0; j != 36; ++j)
                maxima[j] = std::max(maxima[j], p[i * 36 + j]);
        bad += check("mm_stats rows", rows, row_sums(p, 40, 36));
        bad += check("mm_stats cols", cols, maxima);
    }
    {
        auto a3        = fill(40 * 24 * 5, 3, 17);
        auto [cent, r] = center(a3.data());
        Mat x(40 * 24);
        for (size_t i = 0; i != a3.size(); ++i)
            x[i / 5] += a3[i];
        auto s = row_sums(x, 40, 24);
        Mat c(x);
        for (int i = 0; i != 40; ++i)
            for (int j = 0; j != 24; ++j)
                c[i * 24 + j] -= s[i];
        bad += check("center cent", cent, c);
        for (auto& v : x)
            v *= 2;
        bad += check("center twice", r, x);
    }
    {
        auto [q, red] = mm_two_red(a.data(), b.data());
        auto rs       = row_sums(p, 40, 36);
        for (auto& x : rs)
            x *= 7;
        bad += check("mm_two_red product", q, p);
        bad += check("mm_two_red", red, rs);
    }
    for (auto [M, N] : {std::pair{1, 1}, {7, 5}, {8, 16}, {37, 45}, {64, 33}}) {
        auto da = fill(M * 24, 7, 11), db = fill(24 * N, 5, 13);
        auto dp        = mm(dbl(da), dbl(db), M, 24, N);
        auto [rows, r] = dyn_stats(M, N, da.data(), db.data());
        bad += check("dyn_stats rows", rows, row_sums(dp, M, N));
        bad += check("dyn_stats relu", r, relu(dp));
    }
    return bad != 0;
}
