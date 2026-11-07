#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" float* mim_model(float*, float*, float*);
int main() {
    const int s = 8, e = 16, h = 2, d = 8;
    float qkv[s * 3 * e], w[e * e], b[e];
    for (int i = 0; i != s * 3 * e; ++i) qkv[i] = (i * 7 % 13) / 13.f - .5f;
    for (int i = 0; i != e * e; ++i) w[i] = (i * 5 % 11) / 11.f - .5f;
    for (int i = 0; i != e; ++i) b[i] = (i * 3 % 7) / 7.f - .5f;
    double ctx[s][e];
    for (int hd = 0; hd != h; ++hd)
        for (int i = 0; i != s; ++i) {
            double p[s], top = -1e30, sum = 0;
            for (int j = 0; j != s; ++j) {
                p[j] = 0;
                for (int c = 0; c != d; ++c) p[j] += double(qkv[i * 3 * e + hd * d + c]) * qkv[j * 3 * e + e + hd * d + c];
                top = std::fmax(top, p[j]);
            }
            for (int j = 0; j != s; ++j) sum += p[j] = std::exp(p[j] - top);
            for (int c = 0; c != d; ++c) {
                ctx[i][hd * d + c] = 0;
                for (int j = 0; j != s; ++j) ctx[i][hd * d + c] += p[j] / sum * qkv[j * 3 * e + 2 * e + hd * d + c];
            }
        }
    int bad = 0;
    for (int rep = 0; rep != 2; ++rep) {
        float* r = mim_model(qkv, w, b);
        for (int i = 0; i != s; ++i)
            for (int o = 0; o != e; ++o) {
                double y = b[o];
                for (int c = 0; c != e; ++c) y += ctx[i][c] * w[o * e + c];
                bad += std::fabs(y - r[i * e + o]) > 1e-4;
            }
        free(r);
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
