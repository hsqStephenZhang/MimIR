#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" float* relu_mm(float*, float*);

int main() {
    const int M = 37, K = 53, N = 29;
    std::vector<float> a(M * K), b(K * N);
    for (int i = 0; i != M * K; ++i)
        a[i] = float(int(i * 7 % 13) - 6) / 4;
    for (int i = 0; i != K * N; ++i)
        b[i] = float(int(i * 5 % 11) - 5) / 4;
    auto c  = relu_mm(a.data(), b.data());
    int bad = 0;
    for (int i = 0; i != M; ++i)
        for (int j = 0; j != N; ++j) {
            double s = 0;
            for (int k = 0; k != K; ++k)
                s += double(a[i * K + k]) * b[k * N + j];
            if (std::fabs(c[i * N + j] - std::fmax(s - 1.0, 0.0)) > 1e-3) ++bad;
        }
    if (bad) std::printf("relu_mm: %d wrong\n", bad);
    std::free(c);
    return bad != 0;
}
