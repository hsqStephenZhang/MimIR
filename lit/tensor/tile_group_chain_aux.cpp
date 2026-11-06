#include <cmath>
#include <cstdio>
#include <cstdlib>
extern "C" float* kernel(float*, float*, float*);
int main() {
    int n = 64, bad = 0;
    float *a = new float[n * n], *b = new float[n * n], *c = new float[n * n];
    for (int i = 0; i != n * n; ++i) a[i] = (i * 7 % 13) / 13.f - .5f, b[i] = (i * 5 % 11) / 11.f - .5f, c[i] = (i * 3 % 17) / 17.f - .5f;
    for (int rep = 0; rep != 3; ++rep) {
        float* r = kernel(a, b, c);
        for (int i = 0; i != n; ++i)
            for (int j = 0; j != n; ++j) {
                double acc = 0;
                for (int k = 0; k != n; ++k) {
                    double ab = 0;
                    for (int l = 0; l != n; ++l) ab += double(a[i * n + l]) * b[l * n + k];
                    acc += ab * c[k * n + j];
                }
                bad += std::fabs(acc - r[i * n + j]) > 1e-3 * (1 + std::fabs(acc));
            }
        free(r);
    }
    delete[] a, delete[] b, delete[] c;
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
