#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* nested_k(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" int64_t* nested_m(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" int64_t* nested7(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* capped(int64_t, int64_t*, int64_t*, int64_t*);
extern "C" int64_t* reduce_first(int64_t, int64_t*, int64_t*, int64_t*);

using Sym = int64_t* (*)(int64_t, int64_t*, int64_t*, int64_t*);

static int failures = 0;

static int64_t* fill(int64_t cells, int64_t scale, int64_t shift) {
    auto x = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * (cells + 1)));
    for (int64_t i = 0; i != cells; ++i)
        x[i] = scale * i % 17 + shift;
    return x;
}

/// Checks `relu(a) · b · 2 + bias`, where `a` is `m×k`, or `k×m` when @p transposed.
static void check(const char* what,
                  int64_t m,
                  int64_t k,
                  bool transposed,
                  int64_t n,
                  int64_t* (*run)(int64_t*, int64_t*, int64_t*),
                  Sym sym) {
    auto a = fill(m * k, 5, -8), b = fill(k * 6, 3, -4), bias = fill(6, 7, -2);
    auto c = run ? run(a, b, bias) : sym(n, a, b, bias);
    for (int64_t i = 0; i != m; ++i)
        for (int64_t j = 0; j != 6; ++j) {
            int64_t acc = 0;
            for (int64_t l = 0; l != k; ++l) {
                auto x = transposed ? a[l * m + i] : a[i * k + l];
                acc += (x < 0 ? 0 : x) * b[l * 6 + j];
            }
            if (c[i * 6 + j] != acc * 2 + bias[j]) {
                std::printf("%s: m=%lld k=%lld (%lld, %lld): %lld != %lld\n", what, (long long)m, (long long)k,
                            (long long)i, (long long)j, (long long)c[i * 6 + j], (long long)(acc * 2 + bias[j]));
                ++failures;
            }
        }
    std::free(a);
    std::free(b);
    std::free(bias);
    std::free(c);
}

int main() {
    check("nested7", 5, 7, false, 0, nested7, nullptr);
    for (int64_t n = 0; n != 10; ++n) {
        check("nested_k", 5, n, true, n, nullptr, nested_k);
        check("nested_m", n, 7, false, n, nullptr, nested_m);
        check("capped", 5, n, true, n, nullptr, capped);
        check("reduce_first", 5, n, true, n, nullptr, reduce_first);
    }
    return failures != 0;
}
