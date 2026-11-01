#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* whole(int64_t*, int64_t*);
extern "C" int64_t* split(int64_t*, int64_t*);
extern "C" int32_t* final_post(int64_t*, int64_t*);
extern "C" int64_t* vectorized(int64_t*, int64_t*);
extern "C" int64_t* redvec(int64_t*, int64_t*);
extern "C" int64_t* empty(int64_t*, int64_t*);
extern "C" int64_t* reversed(int64_t*, int64_t*);
extern "C" int64_t* blocked(int64_t*, int64_t*);

int main() {
    int64_t x[3] = {7, -4, 11}, values[15];
    for (int i = 0; i != 15; ++i)
        values[i] = (i * 7) % 13 - 6;
    auto a       = whole(x, values);
    auto b       = split(x, values);
    auto c       = final_post(x, values);
    auto v       = vectorized(x, values);
    auto r       = redvec(x, values);
    auto z       = empty(x, values);
    auto rev     = reversed(x, values);
    bool correct = true;
    for (int i = 0; i != 3; ++i) {
        auto expected = x[i];
        for (int k = 0; k != 5; ++k)
            expected += values[i * 5 + k];
        if (a[i] != expected || b[i] != expected || c[i] != expected * 3 || v[i] != expected || r[i] != expected
            || z[i] != x[i] || rev[i] != expected - x[i] + x[2 - i]) {
            std::fprintf(stderr, "cell %d: expected %lld, whole %lld, split %lld, post %d\n", i,
                         static_cast<long long>(expected), static_cast<long long>(a[i]), static_cast<long long>(b[i]),
                         c[i]);
            correct = false;
        }
    }
    std::free(a);
    std::free(b);
    std::free(c);
    std::free(v);
    std::free(r);
    std::free(z);
    std::free(rev);
    int64_t panel[6] = {1, 2, 3, 4, 5, 6}, data[30];
    for (int i = 0; i != 30; ++i)
        data[i] = i % 7 - 3;
    auto block = blocked(panel, data);
    for (int i = 0; i != 6; ++i) {
        auto expected = panel[i];
        for (int k = 0; k != 5; ++k)
            expected += data[i * 5 + k];
        if (block[i] != expected) {
            std::fprintf(stderr, "blocked cell %d: expected %lld, got %lld\n", i, static_cast<long long>(expected),
                         static_cast<long long>(block[i]));
            correct = false;
        }
    }
    std::free(block);
    return correct ? 0 : 1;
}
