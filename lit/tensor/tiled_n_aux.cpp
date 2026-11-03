#include <cassert>
#include <cstdint>
#include <cstdlib>

struct Pair {
    int64_t* full;
    int64_t* rows;
};
extern "C" Pair paired(int64_t*);
extern "C" int64_t* whole_reader(int64_t*);
extern "C" Pair singleton(int64_t);
struct Mixed {
    int64_t* full;
    int32_t* transposed;
};
extern "C" Mixed mixed(int64_t*);

int main() {
    int64_t input[15];
    for (int i = 0; i != 15; ++i)
        input[i] = i - 7;
    auto result = paired(input);
    auto rows   = whole_reader(input);
    for (int i = 0; i != 5; ++i) {
        int64_t sum = 0;
        for (int j = 0; j != 3; ++j) {
            assert(result.full[3 * i + j] == 2 * input[3 * i + j]);
            sum += 2 * input[3 * i + j];
        }
        assert(result.rows[i] == sum);
        assert(rows[i] == sum);
    }
    std::free(result.full);
    std::free(result.rows);
    std::free(rows);
    auto scalar = singleton(-9);
    assert(scalar.full[0] == -9 && scalar.rows[0] == -18);
    std::free(scalar.full);
    std::free(scalar.rows);
    auto hetero = mixed(input);
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 3; ++j) {
            assert(hetero.full[i * 3 + j] == input[i * 3 + j]);
            assert(hetero.transposed[j * 2 + i] == input[i * 3 + j] + 1);
        }
    std::free(hetero.full);
    std::free(hetero.transposed);
}
