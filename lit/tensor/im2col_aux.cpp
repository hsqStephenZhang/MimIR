// Driver for im2col_exec.mim: calls the LLVM-compiled `im2col` / `conv_im2col` / `conv` and checks every output
// against a scalar reference.
// Exit code: 0 if all outputs match within tolerance, 1 otherwise.

#include <cmath>
#include <cstdio>

#include <vector>

extern "C" {
// `«s; F32»` tensors lower to plain pointers at the C ABI (see hlo_aux.cpp).
float* cols(float* x, float value);
float* conv_same(float* x, float* w);
float* conv_same_ref(float* x, float* w);
float* conv_sd(float* x, float* w);
}

struct Geo {
    int n, c, h, w, kh, kw, sh, sw, dh, dw, ph, pw;
    int oh() const { return (h + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1; }
    int ow() const { return (w + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1; }
    float at(const std::vector<float>& x, int b, int ci, int ih, int iw, float value = 0.0f) const {
        ih -= ph, iw -= pw;
        return ih < 0 || ih >= h || iw < 0 || iw >= w ? value : x[((b * c + ci) * h + ih) * w + iw];
    }
};

static std::vector<float> fill(size_t size, int mul, int mod) {
    std::vector<float> v(size);
    for (size_t i = 0; i < size; ++i)
        v[i] = static_cast<float>((mul * i) % mod) / mod - 0.5f;
    return v;
}

static int compare(const char* name, const float* got, const std::vector<float>& ref) {
    int bad = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::fabs(ref[i] - got[i]) > 1e-4f + 1e-4f * std::fabs(ref[i]))
            if (++bad <= 5) std::printf("%s[%zu]: got %f want %f\n", name, i, got[i], ref[i]);
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

static int check_cols() {
    Geo g{2, 3, 6, 5, 3, 2, 2, 1, 1, 2, 1, 1};
    auto x = fill(g.n * g.c * g.h * g.w, 7, 13);
    auto K = g.c * g.kh * g.kw, P = g.oh() * g.ow();
    auto value = -2.0f;
    std::vector<float> ref(g.n * K * P);
    for (int b = 0; b < g.n; ++b)
        for (int ci = 0; ci < g.c; ++ci)
            for (int i = 0; i < g.kh; ++i)
                for (int j = 0; j < g.kw; ++j)
                    for (int oh = 0; oh < g.oh(); ++oh)
                        for (int ow = 0; ow < g.ow(); ++ow) {
                            int row = (ci * g.kh + i) * g.kw + j, col = oh * g.ow() + ow;
                            ref[(b * K + row) * P + col] = g.at(x, b, ci, oh * g.sh + i * g.dh, ow * g.sw + j * g.dw, value);
                        }
    return compare("cols", cols(x.data(), value), ref);
}

static int check_conv(const char* name, float* (*f)(float*, float*), Geo g, int cout) {
    auto x = fill(g.n * g.c * g.h * g.w, 7, 13);
    auto w = fill(cout * g.c * g.kh * g.kw, 5, 11);
    std::vector<float> ref(g.n * cout * g.oh() * g.ow());
    for (int b = 0; b < g.n; ++b)
        for (int co = 0; co < cout; ++co)
            for (int oh = 0; oh < g.oh(); ++oh)
                for (int ow = 0; ow < g.ow(); ++ow) {
                    double s = 0.0;
                    for (int ci = 0; ci < g.c; ++ci)
                        for (int i = 0; i < g.kh; ++i)
                            for (int j = 0; j < g.kw; ++j)
                                s += double(g.at(x, b, ci, oh * g.sh + i * g.dh, ow * g.sw + j * g.dw))
                                   * double(w[((co * g.c + ci) * g.kh + i) * g.kw + j]);
                    ref[((b * cout + co) * g.oh() + oh) * g.ow() + ow] = float(s);
                }
    return compare(name, f(x.data(), w.data()), ref);
}

int main() {
    int bad = check_cols();
    bad += check_conv("conv_same", conv_same, {2, 3, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1}, 4);
    bad += check_conv("conv_same_ref", conv_same_ref, {2, 3, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1}, 4);
    bad += check_conv("conv_sd", conv_sd, {2, 2, 9, 9, 3, 3, 2, 2, 2, 2, 1, 1}, 3);
    return bad ? 1 : 0;
}
