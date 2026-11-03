#include <cassert>
#include <cstdint>

extern "C" bool cached(int64_t*);

int main() {
    int64_t a = 1, b = 2;
    assert(cached(&a));
    assert(!cached(&a));
    assert(cached(&b));
    assert(cached(&a));
}
