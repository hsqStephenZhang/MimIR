#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" int64_t* square(int64_t*, int64_t*);
extern "C" int64_t* tailed(int64_t*, int64_t*);
extern "C" int64_t skinny(int64_t*, int64_t*);
extern "C" int64_t* hand_square(int64_t*, int64_t*);
extern "C" int64_t* hand_tailed(int64_t*, int64_t*);
extern "C" int64_t hand_skinny(int64_t*, int64_t*);
extern "C" int64_t* register_tails(int64_t*, int64_t*);

static bool check(int m, int k, int n, int64_t* (*compute)(int64_t*, int64_t*)) {
    std::vector<int64_t> a(m * k), b(k * n);
    for (int i = 0; i != m * k; ++i)
        a[i] = i % 11 - 5;
    for (int i = 0; i != k * n; ++i)
        b[i] = i % 13 - 6;
    auto out = compute(a.data(), b.data());
    bool ok  = true;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            int64_t expected = 0;
            for (int p = 0; p != k; ++p)
                expected += a[i * k + p] * b[p * n + j];
            if (out[i * n + j] != expected) {
                std::fprintf(stderr, "%dx%dx%d (%d,%d): expected %lld, got %lld\n", m, k, n, i, j,
                             static_cast<long long>(expected), static_cast<long long>(out[i * n + j]));
                ok = false;
            }
        }
    std::free(out);
    return ok;
}

int main() {
    int64_t a[7] = {1, 2, 3, 4, 5, 6, 7};
    auto ok = check(8, 8, 8, square) && check(13, 7, 11, tailed) && skinny(a, a) == 140 && check(8, 8, 8, hand_square)
           && check(13, 7, 11, hand_tailed) && hand_skinny(a, a) == 140 && check(5, 9, 19, register_tails);
    return ok ? 0 : 1;
}
