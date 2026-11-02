#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* chain(int64_t*);
extern "C" int64_t* reference(int64_t*);
extern "C" int64_t* nested(int64_t*);

int main() {
    int64_t input[10];
    for (int i = 0; i != 10; ++i)
        input[i] = i - 7;
    auto actual    = chain(input);
    auto expected  = reference(input);
    auto partition = nested(input);
    for (int i = 0; i != 10; ++i) {
        assert(actual[i] == 2 * input[i]);
        assert(actual[i] == expected[i]);
        assert(partition[i] == input[i]);
    }
    std::free(actual);
    std::free(expected);
    std::free(partition);
}
