#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" {
float* alloc_mm(float*, float*);
float* plain_mm(float*, float*);
}

int main() {
    constexpr int M = 19, K = 29, N = 37;
    std::vector<float> a(M * K), b(K * N);
    for (int i = 0; i != M * K; ++i)
        a[i] = float(int((i * 7) % 11) - 5) / 8.0f;
    for (int i = 0; i != K * N; ++i)
        b[i] = float(int((i * 5) % 13) - 6) / 8.0f;
    float* got  = alloc_mm(a.data(), b.data());
    float* want = plain_mm(a.data(), b.data());
    int bad     = 0;
    for (int i = 0; i != M * N; ++i)
        if (std::fabs(got[i] - want[i]) > 1e-4f * (1 + std::fabs(want[i]))) ++bad;
    std::free(got), std::free(want);
    if (bad) std::printf("alloc_mm: %d wrong\n", bad);
    return bad != 0;
}
