#include <chrono>
#include <cstdlib>
#include <cstring>

extern "C" float* mim_attention(float*, float*, float*);

extern "C" int run_attention(float* q, float* k, float* v, float* out, size_t elements) {
    auto result = mim_attention(q, k, v);
    if (!result) return -1;
    std::memcpy(out, result, elements * sizeof(float));
    std::free(result);
    return 0;
}

extern "C" double measure_attention(float* q, float* k, float* v, int count) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        auto result = mim_attention(q, k, v);
        if (!result) return -1;
        std::free(result);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / count;
}
