#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* tile(int64_t*, int64_t*);
extern "C" int64_t* reference(int64_t*, int64_t*);
extern "C" int64_t* runtime_tile(int64_t*, int64_t*, uint64_t);
extern "C" int64_t* permuted(int64_t*);
extern "C" int64_t* seeded_empty(int64_t*, int64_t*);
extern "C" int64_t* partition(int64_t*, int64_t*);
extern "C" int64_t* shifted(int64_t*);

int main() {
    int64_t a[30], b[6];
    for (int i = 0; i != 30; ++i)
        a[i] = i - 11;
    for (int i = 0; i != 6; ++i)
        b[i] = 100 + i;
    auto actual = tile(a, b), expected = reference(a, b);
    for (int i = 0; i != 3; ++i)
        assert(actual[i] == expected[i]);
    std::free(actual);
    actual = shifted(b);
    for (int i = 0; i != 3; ++i)
        assert(actual[i] == b[i + 1]);
    std::free(actual);
    actual = partition(a, b);
    for (int i = 0; i != 6; ++i) {
        int64_t sum = 1 + b[i];
        for (int k = 0; k != 5; ++k)
            sum += a[i * 5 + k];
        assert(actual[i] == sum);
    }
    std::free(actual);
    std::free(expected);
    actual = permuted(a);
    for (int j = 0; j != 2; ++j)
        for (int i = 0; i != 3; ++i)
            assert(actual[j * 3 + i] == a[(i + 2) * 5 + j + 1]);
    std::free(actual);
    actual = seeded_empty(a, b);
    for (int i = 0; i != 3; ++i)
        assert(actual[i] == b[i + 2]);
    std::free(actual);
    for (uint64_t off = 0; off <= 3; ++off) {
        actual = runtime_tile(a, b, off);
        for (int i = 0; i != 3; ++i) {
            int64_t sum = 1 + b[off + i];
            for (int k = 0; k != 5; ++k)
                sum += a[(off + i) * 5 + k];
            assert(actual[i] == sum);
        }
        std::free(actual);
    }
}
