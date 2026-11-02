#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* kernel(int64_t*);

int main() {
    int64_t input[] = {-5, 0, 3, 11};
    auto output     = kernel(input);
    for (int i = 0; i != 4; ++i)
        assert(output[i] == 256 * input[i]);
    std::free(output);
}
