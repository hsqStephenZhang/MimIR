#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <unordered_map>

namespace {
std::unordered_map<void*, uint64_t> allocations;
uint64_t live;
} // namespace

extern "C" void* counted_malloc(uint64_t bytes) {
    auto ptr = std::malloc(bytes);
    if (!ptr) std::abort();
    allocations.emplace(ptr, bytes);
    live += bytes;
    return ptr;
}

extern "C" void counted_free(void* ptr) {
    if (!ptr) return;
    auto i = allocations.find(ptr);
    if (i == allocations.end()) std::abort();
    live -= i->second;
    allocations.erase(i);
    std::free(ptr);
}

struct Result {
    float *a, *b;
};
extern "C" Result fold(uint64_t, float*, float*);
extern "C" Result consume(uint64_t, float*, float*);
extern "C" Result even(uint64_t, float*, float*);
extern "C" Result odd(uint64_t, float*, float*);

int check(Result (*fn)(uint64_t, float*, float*), uint64_t n, uint64_t steps, float scale) {
    float a[] = {1, 2, 3, 4}, b[] = {-2, 3, 1};
    auto out = fn(n, a, b);
    for (uint64_t i = 0; i != steps; ++i) {
        float sum_a = 0, sum_b = 0;
        for (float x : a)
            sum_a += x;
        for (float x : b)
            sum_b += x;
        for (auto& x : a)
            x += sum_b * float(i + 1);
        for (auto& x : b)
            x += sum_a * float(i + 1);
    }
    for (int i = 0; i != 4; ++i)
        if (std::abs(scale * a[i] - out.a[i]) > 2e-4f + 2e-4f * std::abs(scale * a[i])) return 1;
    for (int i = 0; i != 3; ++i)
        if (std::abs(scale * b[i] - out.b[i]) > 2e-4f + 2e-4f * std::abs(scale * b[i])) return 1;
    counted_free(out.a);
    counted_free(out.b);
    if (live) {
        std::fprintf(stderr, "fold n=%llu leaked %llu bytes\n", (unsigned long long)n, (unsigned long long)live);
        return 1;
    }
    return 0;
}

int main() {
    for (uint64_t n : {0, 1, 2, 5})
        if (check(fold, n, n, 1) || check(consume, n, n, 2) || check(even, n, 2, 2) || check(odd, n, 5, 2)) return 1;
}
