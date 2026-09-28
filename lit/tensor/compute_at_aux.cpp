// Driver for compute_at_exec.mim: calls the LLVM-compiled functions and checks every output against a scalar reference.
// Exit code: 0 if all outputs match within tolerance, 1 otherwise.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <vector>

extern "C" {
// `«s; F32»` tensors lower to plain pointers at the C ABI (see hlo_aux.cpp).
float* att(float* q, float* kt, float* v);
float* mlp(float* a, float* b);
float* pool(float* x, float unused);
float* twice(float* a, float* b);
}

static std::vector<float> fill(size_t n, int mul, int mod, float off) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = static_cast<float>((mul * i) % mod) / mod - off;
    return v;
}

static int compare(const char* name, const float* got, const std::vector<double>& ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::fabs(got[i] - ref[i]) > 1e-4 + 1e-4 * std::fabs(ref[i]))
            if (++bad <= 5) std::printf("%s[%zu]: got %f want %f\n", name, i, got[i], ref[i]);
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

static double relu(double x) { return std::max(x, 0.0); }

// c = f(a) · b for an (m x k) a and a (k x n) b.
static std::vector<double> product(const std::vector<double>& a, const std::vector<float>& b, int m, int k, int n) {
    std::vector<double> c(m * n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                c[i * n + j] += a[i * k + l] * b[l * n + j];
    return c;
}

int main() {
    int bad = 0;
    {
        auto q = fill(48 * 16, 7, 13, 0.5f), kt = fill(16 * 40, 5, 11, 0.5f), v = fill(40 * 24, 3, 17, 0.5f);
        std::vector<double> qd(q.begin(), q.end());
        auto s = product(qd, kt, 48, 16, 40);
        for (auto& e : s)
            e = std::exp(e);
        bad += compare("att", att(q.data(), kt.data(), v.data()), product(s, v, 48, 40, 24));
    }
    auto a = fill(20 * 36, 7, 13, 0.5f), b = fill(36 * 28, 5, 11, 0.5f);
    std::vector<double> ra(a.size());
    std::transform(a.begin(), a.end(), ra.begin(), [](float x) { return relu(x); });
    auto mlp_ref = product(ra, b, 20, 36, 28);
    bad += compare("mlp", mlp(a.data(), b.data()), mlp_ref);
    auto twice_ref = mlp_ref;
    for (auto& e : twice_ref)
        e *= 2;
    bad += compare("twice", twice(a.data(), b.data()), twice_ref);
    {
        auto x = fill(2 * 3 * 8 * 10, 7, 13, 0.5f);
        std::vector<double> ref(2 * 3 * 4 * 5);
        for (int n = 0; n < 2; ++n)
            for (int c = 0; c < 3; ++c)
                for (int oh = 0; oh < 4; ++oh)
                    for (int ow = 0; ow < 5; ++ow) {
                        double m = 0.0;
                        for (int i = 0; i < 2; ++i)
                            for (int j = 0; j < 2; ++j)
                                m = std::max(m, relu(x[((n * 3 + c) * 8 + 2 * oh + i) * 10 + 2 * ow + j]));
                        ref[((n * 3 + c) * 4 + oh) * 5 + ow] = m;
                    }
        bad += compare("pool", pool(x.data(), 0.0f), ref);
    }
    return bad ? 1 : 0;
}
