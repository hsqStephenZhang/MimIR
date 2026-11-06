#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" float* mim_model(float*, float*, float*);
int main() {
    const int s = 8, e = 16, h = 2, d = 8;
    float x[s * e], w[3 * e * e], b[3 * e];
    for (int i = 0; i != s * e; ++i) x[i] = (i * 7 % 13) / 13.f - .5f;
    for (int i = 0; i != 3 * e * e; ++i) w[i] = (i * 5 % 11) / 11.f - .5f;
    for (int i = 0; i != 3 * e; ++i) b[i] = (i * 3 % 7) / 7.f - .5f;
    double qkv[s][3 * e];
    for (int i = 0; i != s; ++i)
        for (int o = 0; o != 3 * e; ++o) {
            qkv[i][o] = b[o];
            for (int k = 0; k != e; ++k) qkv[i][o] += double(x[i * e + k]) * w[o * e + k];
        }
    int bad = 0;
    for (int rep = 0; rep != 2; ++rep) {
        float* r = mim_model(x, w, b);
        for (int hd = 0; hd != h; ++hd)
            for (int i = 0; i != s; ++i) {
                double p[s], top = -1e30, sum = 0;
                for (int j = 0; j != s; ++j) {
                    p[j] = 0;
                    for (int c = 0; c != d; ++c) p[j] += qkv[i][hd * d + c] * qkv[j][e + hd * d + c];
                    top = std::fmax(top, p[j]);
                }
                for (int j = 0; j != s; ++j) sum += p[j] = std::exp(p[j] - top);
                for (int c = 0; c != d; ++c) {
                    double ctx = 0;
                    for (int j = 0; j != s; ++j) ctx += p[j] / sum * qkv[j][2 * e + hd * d + c];
                    bad += std::fabs(ctx - r[i * e + hd * d + c]) > 1e-4;
                }
            }
        free(r);
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
