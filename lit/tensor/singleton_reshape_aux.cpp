#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* singleton_reshape(int64_t*);
extern "C" int64_t* singleton_view(int64_t*);

int main() {
    int64_t input[24];
    for (int repeat = 0; repeat != 3; ++repeat) {
        for (int i = 0; i != 24; ++i)
            input[i] = (i - 12) * (repeat + 1);
        assert(singleton_view(input) == input);
        auto output = singleton_reshape(input);
        assert(output != input);
        for (int i = 0; i != 24; ++i)
            assert(output[i] == input[i]);
        std::free(output);
    }
}
