#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* perm_mm(float*, float*, float*);
float* relu_chain(float*, float*, float*);
float* pad_mm(float*, float*);
float* pad_relu(float*);
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

static Mat transpose(const Mat& a, int M, int N) {
    Mat t(M * N);
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j)
            t[j * M + i] = a[i * N + j];
    return t;
}

static Mat relu(Mat a) {
    for (auto& x : a)
        x = x < 0 ? 0.0 : x;
    return a;
}

static Mat pad(const Mat& a, int M, int N, int top, int left, int bottom, int right, double value) {
    int W = left + N + right;
    Mat p((top + M + bottom) * W, value);
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j)
            p[(top + i) * W + left + j] = a[i * N + j];
    return p;
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
    auto a = fill(40 * 24, 7, 11), b = fill(24 * 36, 5, 13), d = fill(40 * 20, 3, 7), e = fill(36 * 20, 5, 9);
    auto ab = mm(dbl(a), dbl(b), 40, 24, 36);
    int bad = check("perm_mm", perm_mm(a.data(), b.data(), d.data()), mm(transpose(ab, 40, 36), dbl(d), 36, 40, 20));
    bad += check("relu_chain", relu_chain(a.data(), b.data(), e.data()), mm(relu(ab), dbl(e), 40, 36, 20));
    auto p = fill(37 * 20, 7, 11), q = fill(24 * 19, 5, 13);
    bad += check("pad_mm", pad_mm(p.data(), q.data()), mm(pad(dbl(p), 37, 20, 2, 1, 1, 3, 0.5), dbl(q), 40, 24, 19));
    bad += check("pad_relu", pad_relu(p.data()), relu(pad(dbl(p), 37, 20, 2, 1, 1, 3, -0.5)));
    return bad != 0;
}
