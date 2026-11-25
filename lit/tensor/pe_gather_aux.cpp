#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>
extern "C" {
float* embed_proj(float*, int8_t*, float*);
float* embed_relu(float*, int8_t*);
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
    const int V = 50, D = 24, N = 37, O = 19;
    std::vector<float> t(V * D), w(D * O);
    for (int i = 0; i != V * D; ++i)
        t[i] = float(int(i * 7 % 13) - 6) / 4;
    for (int i = 0; i != D * O; ++i)
        w[i] = float(int(i * 5 % 11) - 5) / 4;
    std::vector<int8_t> idx(N * D);
    for (int n = 0; n != N; ++n)
        for (int d = 0; d != D; ++d)
            idx[n * D + d] = int8_t((n * 17 + 3) % V);
    std::vector<double> y(N * O), r(N * D);
    for (int n = 0; n != N; ++n) {
        for (int d = 0; d != D; ++d)
            r[n * D + d] = std::fmax(t[idx[n * D + d] * D + d], 0.0);
        for (int o = 0; o != O; ++o)
            for (int d = 0; d != D; ++d)
                y[n * O + o] += double(t[idx[n * D + d] * D + d]) * w[d * O + o];
    }
    int bad = check("embed_proj", embed_proj(t.data(), idx.data(), w.data()), y);
    bad += check("embed_relu", embed_relu(t.data(), idx.data()), r);
    return bad != 0;
}
