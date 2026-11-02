#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* folded(int64_t*);

int main() {
    int64_t input[]{-3, 2, 5, 11, -7, 0};
    auto output = folded(input);
    for (int i = 0; i != 6; ++i)
        assert(output[i] == input[i]);
    std::free(output);
}
