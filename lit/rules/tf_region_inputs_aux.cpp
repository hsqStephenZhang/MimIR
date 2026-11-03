#include <cassert>
#include <cstdint>
#include <cstdlib>

extern "C" int64_t* fold_inputs(int64_t*);
extern "C" int64_t* multi_inputs(int64_t*, int64_t*);
extern "C" int64_t* retained_init(int64_t*);
extern "C" int64_t* shared_tiles(int64_t*);

int main() {
    int64_t a[8] = {-4, 3, 1, -2, 7, -5, 6, 8};
    int64_t b[4] = {5, -3, 2, 9};
    auto folded  = fold_inputs(a);
    for (size_t i = 0; i != 4; ++i)
        assert(folded[i] == 2 * (a[i] + a[i + 4]));
    std::free(folded);
    auto multi = multi_inputs(a, b);
    for (size_t i = 0; i != 8; ++i)
        assert(multi[i] == 2 * a[i] + b[i % 4]);
    std::free(multi);
    auto retained = retained_init(a);
    for (size_t i = 0; i != 8; ++i)
        assert(retained[i] == 6 * a[i]);
    std::free(retained);
    auto shared = shared_tiles(a);
    for (size_t i = 0; i != 8; ++i)
        assert(shared[i] == 2 * a[i]);
    std::free(shared);
}
