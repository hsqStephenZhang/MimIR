#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
static uint64_t calls, bytes, alloc_ns, free_ns, frees;
static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
extern "C" void* profile_malloc(size_t n) {
    uint64_t t = now();
    void* p    = std::malloc(n);
    alloc_ns += now() - t;
    ++calls;
    bytes += n;
    return p;
}
extern "C" void profile_free(void* p) {
    uint64_t t = now();
    std::free(p);
    free_ns += now() - t;
    ++frees;
}
extern "C" void profile_reset() { calls = bytes = alloc_ns = free_ns = frees = 0; }
extern "C" void profile_dump(int n) {
    std::printf("allocation_profile calls=%.2f frees=%.2f bytes=%.0f alloc_us=%.3f free_us=%.3f\n", double(calls) / n,
                double(frees) / n, double(bytes) / n, double(alloc_ns) / n / 1000, double(free_ns) / n / 1000);
}
