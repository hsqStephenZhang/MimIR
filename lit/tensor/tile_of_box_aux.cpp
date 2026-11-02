#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t edge(int64_t*, uint64_t);
extern "C" int64_t* tiles(int64_t*);

int main() {
    int64_t input[]{-4, 3, 0, -7, 8, 11, -2};
    for (uint64_t off = 4; off != 7; ++off)
        assert(edge(input, off) == 3 * (input[off] + input[6]));
    auto output = tiles(input);
    for (int i = 0; i != 7; ++i)
        assert(output[i] == 3 * input[i]);
    std::free(output);
}
