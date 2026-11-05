#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* whole(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" int64_t* split(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" int64_t* split7(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* split_into(int64_t, int64_t*, int64_t*, int64_t*);

static int failures = 0;

// The (2, 3) tile at (1, 2) of the (4, 5) product of a (k, 4) and b (k, 5), a stored transposed.
static int64_t dot(int64_t k, const int64_t* a, const int64_t* b, int64_t i, int64_t j, int64_t seed) {
    int64_t acc = seed;
    for (int64_t p = 0; p != k; ++p)
        acc += a[p * 4 + 1 + i] * b[p * 5 + 2 + j];
    return acc;
}

static void check(const char* what, int64_t k, const int64_t* got, int64_t i, int64_t j, int64_t expected) {
    if (got[i * 3 + j] == expected) return;
    std::printf("%s: k=%lld (%lld, %lld): %lld != %lld\n", what, (long long)k, (long long)i, (long long)j,
                (long long)got[i * 3 + j], (long long)expected);
    ++failures;
}

int main() {
    for (int64_t k = 0; k != 11; ++k) {
        auto a = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (k * 4 + 1)));
        auto b = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (k * 5 + 1)));
        int64_t bias[5], c[20];
        for (int64_t i = 0; i != k * 4; ++i)
            a[i] = 3 * i - 7;
        for (int64_t i = 0; i != k * 5; ++i)
            b[i] = 11 - 2 * i;
        for (int64_t i = 0; i != 5; ++i)
            bias[i] = 100 * i - 3;
        for (int64_t i = 0; i != 20; ++i)
            c[i] = 5 * i + 1;

        auto w = whole(k, a, b, bias), s = split(k, a, b, bias), t = split_into(k, a, b, c);
        auto s7 = k == 7 ? split7(a, b, bias) : nullptr;
        for (int64_t i = 0; i != 2; ++i)
            for (int64_t j = 0; j != 3; ++j) {
                auto gemm = 2 * dot(k, a, b, i, j, 1) + bias[2 + j];
                check("whole", k, w, i, j, gemm);
                check("split", k, s, i, j, gemm);
                if (s7) check("split7", k, s7, i, j, gemm);
                check("split_into", k, t, i, j, dot(k, a, b, i, j, c[(1 + i) * 5 + 2 + j]));
            }
        std::free(w);
        std::free(s);
        std::free(t);
        std::free(s7);
        std::free(a);
        std::free(b);
    }
    return failures != 0;
}
