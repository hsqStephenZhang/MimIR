#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* identity_stage(int64_t*);
extern "C" int64_t* permuted_stage(int64_t*);
extern "C" int64_t* whole_stage(int64_t*);
extern "C" int64_t* borrowed_stage(int64_t*);

int main() {
    int64_t input[35];
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 35; ++i)
            input[i] = (i - 17) * (repeat + 1);
        auto identity = identity_stage(input);
        auto permuted = permuted_stage(input);
        auto whole    = whole_stage(input);
        auto borrowed = borrowed_stage(input);
        assert(borrowed != input);
        for (int i = 0; i != 35; ++i) {
            assert(identity[i] == input[i] + 2);
            assert(permuted[i] == input[i] + 3);
            assert(whole[i] == input[i] + 2);
            assert(borrowed[i] == input[i]);
            assert(input[i] == (i - 17) * (repeat + 1));
        }
        std::free(identity);
        std::free(permuted);
        std::free(whole);
        std::free(borrowed);
    }
}
