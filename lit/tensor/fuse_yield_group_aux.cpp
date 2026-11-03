#include <cassert>
#include <cstdint>
#include <cstdlib>

struct Pair {
    float *consumer, *outside;
};
extern "C" Pair formed(float*);
static int64_t multiplies;
extern "C" void record_multiply(int64_t lanes) { multiplies += lanes; }

int main() {
    float input[10];
    for (int i = 0; i != 10; ++i)
        input[i] = i - 4.0f;
    auto result = formed(input);
    assert(multiplies == 10);
    for (int i = 0; i != 10; ++i) {
        assert(result.consumer[i] == 2 * input[i] + 1);
        assert(result.outside[i] == 2 * input[i]);
    }
    std::free(result.consumer);
    std::free(result.outside);
}
