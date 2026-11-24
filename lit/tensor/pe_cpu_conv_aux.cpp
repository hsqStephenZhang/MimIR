#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* conv_pad(float*, float*);
float* conv_stem(float*, float*);
float* conv_1x1(float*, float*);
float* conv_dil(float*, float*);
float* conv_s2(float*, float*);
float* conv_g2(float*, float*);
float* conv_g4(float*, float*);
}

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 8.0f;
    return v;
}

static int check(const char* name,
                 float* (*f)(float*, float*),
                 int n,
                 int ci,
                 int h,
                 int w,
                 int co,
                 int kh,
                 int kw,
                 int s,
                 int d,
                 int p,
                 int g = 1) {
    int oh = (h + 2 * p - d * (kh - 1) - 1) / s + 1, ow = (w + 2 * p - d * (kw - 1) - 1) / s + 1;
    auto x = fill(size_t(n) * ci * h * w, 7, 11), k = fill(size_t(co) * (ci / g) * kh * kw, 5, 13);
    float* got = f(x.data(), k.data());
    int bad    = 0;
    for (int b = 0; b != n; ++b)
        for (int o = 0; o != co; ++o)
            for (int y = 0; y != oh; ++y)
                for (int z = 0; z != ow; ++z) {
                    double want = 0;
                    for (int c = 0; c != ci / g; ++c)
                        for (int i = 0; i != kh; ++i)
                            for (int j = 0; j != kw; ++j) {
                                int iy = y * s + i * d - p, iz = z * s + j * d - p;
                                if (iy < 0 || iy >= h || iz < 0 || iz >= w) continue;
                                int ic = o / (co / g) * (ci / g) + c;
                                want += double(x[((b * ci + ic) * h + iy) * w + iz])
                                      * k[((o * (ci / g) + c) * kh + i) * kw + j];
                            }
                    double g = got[((b * co + o) * oh + y) * ow + z];
                    if (std::fabs(g - want) > 1e-3 * (1.0 + std::fabs(want))) ++bad;
                }
    if (bad) std::printf("%s: %d wrong\n", name, bad);
    std::free(got);
    return bad;
}

int main() {
    int bad = check("pad", conv_pad, 2, 5, 13, 17, 11, 3, 3, 1, 1, 1);
    bad += check("stem", conv_stem, 1, 3, 15, 16, 8, 7, 7, 2, 1, 3);
    bad += check("1x1", conv_1x1, 2, 16, 9, 10, 12, 1, 1, 1, 1, 0);
    bad += check("dil", conv_dil, 1, 4, 12, 14, 6, 3, 3, 1, 2, 2);
    bad += check("s2", conv_s2, 2, 6, 11, 12, 9, 3, 3, 2, 1, 1);
    bad += check("g2", conv_g2, 2, 8, 9, 10, 12, 3, 3, 1, 1, 1, 2);
    bad += check("g4", conv_g4, 1, 16, 11, 12, 8, 3, 3, 2, 1, 1, 4);
    return bad != 0;
}
