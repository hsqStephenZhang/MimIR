#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <vector>

extern "C" int64_t* square(int64_t*, int64_t*);
extern "C" int64_t* tailed(int64_t*, int64_t*);
extern "C" int64_t* skinny(int64_t*, int64_t*);
extern "C" int64_t* hand_square(int64_t*, int64_t*);
extern "C" int64_t* hand_tailed(int64_t*, int64_t*);
extern "C" int64_t* hand_skinny(int64_t*, int64_t*);

static bool check(const char* name, int m, int k, int n, int64_t* (*compute)(int64_t*, int64_t*), int seed) {
    std::vector<int64_t> a(m * k), b(k * n);
    for (int i = 0; i != m * k; ++i)
        a[i] = (i * 3 + 1 + seed) % 11 - 5;
    for (int i = 0; i != k * n; ++i)
        b[i] = (i * 7 + 2 + seed) % 13 - 6;
    auto x  = compute(a.data(), b.data());
    bool ok = true;
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j) {
            int64_t expected = 0;
            for (int p = 0; p != k; ++p)
                expected += a[i * k + p] * b[p * n + j];
            if (x[i * n + j] != expected) {
                std::fprintf(stderr, "%s %dx%dx%d (%d,%d): expected %lld, got %lld\n", name, m, k, n, i, j,
                             static_cast<long long>(expected), static_cast<long long>(x[i * n + j]));
                ok = false;
            }
        }
    std::free(x);
    return ok;
}

int main() {
    bool ok = true;
    for (int seed = 0; seed != 2; ++seed) {
        ok &= check("rule", 8, 8, 8, square, seed);
        ok &= check("hand", 8, 8, 8, hand_square, seed);
        ok &= check("rule", 13, 7, 11, tailed, seed);
        ok &= check("hand", 13, 7, 11, hand_tailed, seed);
    }
    int64_t a[] = {-3, 2, 5, -4, 1, 6, -2}, b[] = {7, -2, 3, 1, -5, 4, 2};
    int64_t expected = 0;
    for (int i = 0; i != 7; ++i)
        expected += a[i] * b[i];
    auto rule = skinny(a, b);
    auto hand = hand_skinny(a, b);
    if (rule[0] != expected || hand[0] != expected) {
        std::fprintf(stderr, "1x7x1: expected %lld, rule %lld, hand %lld\n", static_cast<long long>(expected),
                     static_cast<long long>(rule[0]), static_cast<long long>(hand[0]));
        ok = false;
    }
    std::free(rule);
    std::free(hand);
    if (ok) std::puts("packed GEMM rule/hand: 8x8x8, 13x7x11, 1x7x1 passed");
    return ok ? 0 : 1;
}
