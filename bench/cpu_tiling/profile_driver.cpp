#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>

#include <dlfcn.h>
int main(int argc, char** argv) {
    if (argc != 7) return 2;
    bool blas = std::strcmp(argv[1], "blas") == 0;
    void* lib = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        std::puts(dlerror());
        return 3;
    }
    int m = std::atoi(argv[3]), n = std::atoi(argv[4]), k = std::atoi(argv[5]), count = std::atoi(argv[6]);
    std::vector<float> a(size_t(m) * k), b(size_t(k) * n), c(size_t(m) * n);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = float(int(i % 13) - 6) / 8;
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = float(int(i % 17) - 8) / 8;
    auto mim  = (float* (*)(float*, float*))dlsym(lib, "mim_kernel");
    auto gemm = (void (*)(int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*,
                          int))dlsym(lib, "cblas_sgemm");
    if (blas) ((void (*)(int))dlsym(lib, "openblas_set_num_threads"))(1);
    auto run = [&]() {
        if (blas)
            gemm(101, 111, 111, m, n, k, 1, a.data(), k, b.data(), n, 0, c.data(), n);
        else {
            float* r = mim(a.data(), b.data());
            if (!r) std::abort();
            c[0]     = r[0];
            c.back() = r[size_t(m) * n - 1];
            std::free(r);
        }
    };
    for (int i = 0; i < 10; ++i)
        run();
    auto reset = (void (*)())dlsym(lib, "profile_reset");
    if (reset) reset();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i)
        run();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s %dx%dx%d count=%d ms=%.6f endpoints=%.6f %.6f\n", blas ? "blas" : "mim", m, k, n, count,
                1000 * sec / count, c[0], c.back());
    auto dump = (void (*)(int))dlsym(lib, "profile_dump");
    if (dump) dump(count);
    return !std::isfinite(c[0]);
}
