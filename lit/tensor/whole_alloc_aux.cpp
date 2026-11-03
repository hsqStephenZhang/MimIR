#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" float* whole_alloc(float*);
static int64_t multiplies;
extern "C" void record_multiply(int64_t lanes) { multiplies += lanes; }

int main() {
    float input[10];
    for (int i = 0; i != 10; ++i)
        input[i] = i - 4.0f;
    auto output = whole_alloc(input);
    for (int i = 0; i != 10; ++i)
        assert(output[i] == 2 * input[i] + 10);
    assert(multiplies == 10);
    std::free(output);
}
