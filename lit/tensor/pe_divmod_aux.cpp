#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* up2(float*);
float* wrap5(float*);
float* ceil3(float*);
float* conv_relu(float*, float*);
}

static float relu(float v) { return std::fmax(v - 0.25f, 0.0f); }

static std::vector<float> fill(size_t n, int p, int q) {
    std::vector<float> v(n);
    for (size_t i = 0; i != n; ++i)
        v[i] = float(int((i * p) % q) - q / 2) / 4.0f;
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
    auto a  = fill(11 * 13, 7, 11);
    std::vector<double> u(22 * 26);
    for (int i = 0; i != 22; ++i)
        for (int j = 0; j != 26; ++j)
            u[i * 26 + j] = relu(a[(i / 2) * 13 + j / 2]);
    bad += check("up2", up2(a.data()), u);

    auto b = fill(7 * 5, 5, 13);
    std::vector<double> m(23 * 7);
    for (int j = 0; j != 23; ++j)
        for (int i = 0; i != 7; ++i)
            m[j * 7 + i] = relu(b[i * 5 + j % 5]);
    bad += check("wrap5", wrap5(b.data()), m);

    auto c = fill(7 * 9, 3, 7);
    std::vector<double> e(25 * 7);
    for (int j = 0; j != 25; ++j)
        for (int i = 0; i != 7; ++i)
            e[j * 7 + i] = relu(c[i * 9 + (j + 2) / 3]);
    bad += check("ceil3", ceil3(c.data()), e);

    auto x = fill(2 * 5 * 11 * 13, 7, 17), w = fill(7 * 5 * 3 * 3, 5, 11);
    std::vector<double> y(2 * 7 * 9 * 11);
    for (int n = 0; n != 2; ++n)
        for (int o = 0; o != 7; ++o)
            for (int h = 0; h != 9; ++h)
                for (int v = 0; v != 11; ++v) {
                    double s = 0;
                    for (int ci = 0; ci != 5; ++ci)
                        for (int kh = 0; kh != 3; ++kh)
                            for (int kw = 0; kw != 3; ++kw)
                                s += double(relu(x[((n * 5 + ci) * 11 + h + kh) * 13 + v + kw]))
                                   * w[((o * 5 + ci) * 3 + kh) * 3 + kw];
                    y[((n * 7 + o) * 9 + h) * 11 + v] = s;
                }
    bad += check("conv_relu", conv_relu(x.data(), w.data()), y);
    return bad != 0;
}
