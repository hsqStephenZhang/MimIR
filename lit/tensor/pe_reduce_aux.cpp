#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* softmax(float*);
float* row_mean(float*);
float* layer_norm(float*);
float* max_pool(float*);
float* avg_pool(float*);
float* transpose(float*);
float* row_sum(float*);
float* row_max(float*);
float* short_mean(float*);
float* plane_max(float*);
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
    int bad     = 0;
    const int R = 37, C = 53;
    auto x = fill(R * C, 7, 11);
    std::vector<double> sm(R * C), mean(R), ln(R * C);
    for (int i = 0; i != R; ++i) {
        double mx = -INFINITY, s = 0, m = 0, v = 0;
        for (int j = 0; j != C; ++j)
            mx = std::fmax(mx, x[i * C + j]), m += x[i * C + j];
        for (int j = 0; j != C; ++j)
            s += std::exp(x[i * C + j] - mx);
        for (int j = 0; j != C; ++j)
            sm[i * C + j] = std::exp(x[i * C + j] - mx) / s;
        m /= C, mean[i] = m;
        for (int j = 0; j != C; ++j)
            v += (x[i * C + j] - m) * (x[i * C + j] - m);
        v /= C;
        for (int j = 0; j != C; ++j)
            ln[i * C + j] = (x[i * C + j] - m) / std::sqrt(v + 1e-5);
    }
    const int N = 2, Ch = 3, H = 16, W = 18;
    auto p = fill(N * Ch * H * W, 5, 13);
    std::vector<double> mp(N * Ch * 8 * 9), ap(N * Ch * 14 * 16);
    for (int nc = 0; nc != N * Ch; ++nc) {
        for (int oh = 0; oh != 8; ++oh)
            for (int ow = 0; ow != 9; ++ow) {
                double m = -INFINITY;
                for (int kh = 0; kh != 2; ++kh)
                    for (int kw = 0; kw != 2; ++kw)
                        m = std::fmax(m, p[(nc * H + 2 * oh + kh) * W + 2 * ow + kw]);
                mp[(nc * 8 + oh) * 9 + ow] = m;
            }
        for (int oh = 0; oh != 14; ++oh)
            for (int ow = 0; ow != 16; ++ow) {
                double s = 0;
                for (int kh = 0; kh != 3; ++kh)
                    for (int kw = 0; kw != 3; ++kw)
                        s += p[(nc * H + oh + kh) * W + ow + kw];
                ap[(nc * 14 + oh) * 16 + ow] = s / 9;
            }
    }
    bad += check("softmax", softmax(x.data()), sm);
    bad += check("row_mean", row_mean(x.data()), mean);
    bad += check("layer_norm", layer_norm(x.data()), ln);
    bad += check("max_pool", max_pool(p.data()), mp);
    bad += check("avg_pool", avg_pool(p.data()), ap);
    std::vector<double> tr(R * C);
    for (int i = 0; i != R; ++i)
        for (int j = 0; j != C; ++j)
            tr[j * R + i] = x[i * C + j];
    bad += check("transpose", transpose(x.data()), tr);
    auto y = fill(R * 5, 3, 7), z = fill(5 * 7 * 48, 5, 13);
    std::vector<double> rs(R), rm(R, -INFINITY), sm5(R), pm(5, -INFINITY);
    for (int i = 0; i != R; ++i) {
        for (int j = 0; j != C; ++j)
            rs[i] += x[i * C + j], rm[i] = std::fmax(rm[i], x[i * C + j]);
        for (int j = 0; j != 5; ++j)
            sm5[i] += y[i * 5 + j] / 5.0;
    }
    for (int i = 0; i != 5; ++i)
        for (int k = 0; k != 7 * 48; ++k)
            pm[i] = std::fmax(pm[i], z[i * 336 + k]);
    bad += check("row_sum", row_sum(x.data()), rs);
    bad += check("row_max", row_max(x.data()), rm);
    bad += check("short_mean", short_mean(y.data()), sm5);
    bad += check("plane_max", plane_max(z.data()), pm);
    return bad != 0;
}
