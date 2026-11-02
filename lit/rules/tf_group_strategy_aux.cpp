#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* rows(int64_t*);
extern "C" int64_t* contraction(int64_t*, int64_t*);
extern "C" int64_t* repeated(int64_t*);

int main() {
    int64_t input[180];
    for (int i = 0; i != 180; ++i)
        input[i] = i % 13 - 6;
    auto result = rows(input);
    for (int j = 0; j != 4; ++j) {
        int64_t expected = 0;
        for (int i = 0; i != 6; ++i)
            for (int k = 0; k != 3; ++k)
                expected += 2 * input[(i * 4 + j) * 3 + k];
        assert(result[j] == expected);
    }
    std::free(result);
    int64_t a[15], b[12];
    for (int i = 0; i != 15; ++i)
        a[i] = i - 4;
    for (int i = 0; i != 12; ++i)
        b[i] = i % 5 - 2;
    result = contraction(a, b);
    for (int j = 0; j != 4; ++j) {
        int64_t expected = 0;
        for (int i = 0; i != 5; ++i)
            for (int k = 0; k != 3; ++k)
                expected += a[i * 3 + k] * b[k * 4 + j];
        assert(result[j] == expected);
    }
    std::free(result);
    result = repeated(input);
    for (int c = 0; c != 4; ++c)
        for (int j = 0; j != 5; ++j) {
            int64_t expected = 0;
            for (int i = 0; i != 6; ++i)
                for (int k = 0; k != 3; ++k)
                    for (int l = 0; l != 2; ++l)
                        expected += input[((i * 5 + j) * 3 + k) * 2 + l];
            assert(result[c * 5 + j] == expected);
        }
    std::free(result);
}
