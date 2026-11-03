#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef COUNT_MALLOC
static long allocations = 0;
extern "C" void* counted_malloc(size_t n) {
    ++allocations;
    return std::malloc(n);
}
#endif

extern "C" float* short_k(float*, float*);
extern "C" float* long_k(float*, float*);

template<int K>
static bool check(float* (*kernel)(float*, float*)) {
    constexpr int B = 2, M = 8, N = 256;
    std::vector<float> a(B * M * K), b(B * N * K);
    for (size_t i = 0; i != a.size(); ++i) a[i] = float(int(i % 13) - 6) / 8;
    for (size_t i = 0; i != b.size(); ++i) b[i] = float(int(i % 7) - 3) / 4;
    auto c  = kernel(a.data(), b.data());
    bool ok = true;
    for (int x = 0; x != B; ++x)
        for (int i = 0; i != M; ++i)
            for (int j = 0; j != N; ++j) {
                double ref = 0;
                for (int k = 0; k != K; ++k) ref += double(a[(x * M + i) * K + k]) * b[(x * N + j) * K + k];
                ok &= std::abs(c[(x * M + i) * N + j] - ref) <= 1e-3;
            }
    std::free(c);
    return ok;
}

int main() {
    if (!check<64>(short_k) || !check<300>(long_k)) return 1;
#ifdef COUNT_MALLOC
    // Copying the accumulator per micro tile would take thousands of allocations.
    if (allocations > 64) return 2;
#endif
}
