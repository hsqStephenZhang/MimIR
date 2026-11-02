#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* rows(int64_t*);
extern "C" int64_t* tail(int64_t*);
extern "C" int64_t* seeded(int64_t*);

int main() {
    int64_t input[90];
    for (int i = 0; i != 90; ++i)
        input[i] = i % 13 - 6;
    for (int cols = 4; cols != 6; ++cols) {
        auto result = cols == 4 ? rows(input) : tail(input);
        for (int j = 0; j != cols; ++j) {
            int64_t expected = 0;
            for (int i = 0; i != 6; ++i)
                for (int k = 0; k != 3; ++k)
                    expected += 2 * input[(i * cols + j) * 3 + k];
            assert(result[j] == expected);
        }
        std::free(result);
    }
    auto result = seeded(input);
    for (int j = 0; j != 5; ++j) {
        int64_t expected = 0;
        for (int i = 0; i != 6; ++i)
            for (int k = 0; k != 3; ++k)
                expected += 3 * input[(i * 5 + j) * 3 + k];
        assert(result[j] == expected);
    }
    std::free(result);
}
