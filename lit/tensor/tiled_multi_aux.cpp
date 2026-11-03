#include <cassert>
#include <cstdint>
#include <cstdlib>

struct Pair {
    int64_t *full, *rows;
};
struct Mixed {
    int64_t* full;
    int32_t* transposed;
};
extern "C" Pair paired(int64_t*);
extern "C" Mixed mixed(int64_t*);

int main() {
    int64_t input[15];
    for (int i = 0; i != 15; ++i)
        input[i] = i - 7;
    auto pair   = paired(input);
    auto result = mixed(input);
    for (int i = 0; i != 5; ++i) {
        int64_t sum = 0;
        for (int j = 0; j != 3; ++j) {
            assert(pair.full[i * 3 + j] == 2 * input[i * 3 + j]);
            sum += 2 * input[i * 3 + j];
            assert(result.full[i * 3 + j] == input[i * 3 + j]);
            assert(result.transposed[j * 5 + i] == input[i * 3 + j]);
        }
        assert(pair.rows[i] == sum);
    }
    std::free(pair.full);
    std::free(pair.rows);
    std::free(result.full);
    std::free(result.transposed);
}
