#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <malloc.h>
#include <sched.h>
#include <sys/resource.h>

static size_t live_bytes() {
    auto info = mallinfo2();
    return info.uordblks + info.hblkhd;
}

// Persistent, warmed kernels serve multiple timing requests. The caller frees only the returned tensor.
int main(int argc, char** argv) {
    try {
        if (argc != 12 && argc != 13)
            throw std::runtime_error("library inputs H M S D N causal threads cpus kind [stream]");
        bool stream = argc == 13;
        int64_t dims[6];
        for (int i = 0; i < 6; ++i)
            dims[i] = std::stoll(argv[3 + i]);
        int threads = std::stoi(argv[9]), kind = std::stoi(argv[11]);
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        std::stringstream list(argv[10]);
        std::string cpu;
        int primary = CPU_SETSIZE;
        while (std::getline(list, cpu, ',')) {
            int id = std::stoi(cpu);
            CPU_SET(id, &cpus);
            primary = std::min(primary, id);
        }
        if (sched_setaffinity(0, sizeof(cpus), &cpus)) throw std::runtime_error("affinity");
        auto library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto symbol = [&](const char* name) {
            auto address = dlsym(library, name);
            if (!address) throw std::runtime_error(dlerror());
            return address;
        };
        size_t qsize = dims[0] * dims[1] * dims[3], ksize = dims[0] * dims[2] * dims[3];
        size_t vsize = dims[0] * dims[2] * dims[4];
        std::vector<float> q(qsize), k(ksize), v(vsize), out(dims[0] * dims[1] * dims[4]);
        std::ifstream input(argv[2], std::ios::binary);
        for (auto data : {&q, &k, &v})
            input.read(reinterpret_cast<char*>(data->data()), data->size() * sizeof(float));
        if (!input) throw std::runtime_error("input file");
        void* handle       = nullptr;
        using NativeTimer  = double (*)(float*, float*, float*, int);
        using TorchTimer   = double (*)(void*, int);
        NativeTimer native = nullptr;
        TorchTimer torch   = nullptr;
        if (kind < 2) {
            reinterpret_cast<void (*)(int)>(symbol("attention_threads"))(threads);
            using Create = void* (*)(int, const int64_t*, float*, float*, float*, float*);
            handle = reinterpret_cast<Create>(symbol("create_attention"))(kind, dims, q.data(), k.data(), v.data(),
                                                                          out.data());
            if (!handle) throw std::runtime_error("Torch kernel construction failed");
            torch = reinterpret_cast<TorchTimer>(symbol("measure_attention"));
        } else {
            native = reinterpret_cast<NativeTimer>(symbol("measure_attention"));
        }
        CPU_ZERO(&cpus);
        CPU_SET(primary, &cpus);
        if (sched_setaffinity(0, sizeof(cpus), &cpus)) throw std::runtime_error("primary affinity");
        auto measure = [&](int count) {
            auto seconds = native ? native(q.data(), k.data(), v.data(), count) : torch(handle, count);
            if (!(seconds > 0) || !std::isfinite(seconds)) throw std::runtime_error("native timer failed");
            return seconds;
        };
        auto warm_start  = std::chrono::steady_clock::now();
        int warmup_calls = 3;
        for (int i = 0; i < warmup_calls; ++i)
            measure(1);
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_start).count() < 0.2) {
            measure(8);
            warmup_calls += 8;
        }
        auto warmup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_start).count();
        std::cout.precision(16);
        auto sample = [&]() {
            auto estimate = measure(1);
            int count     = std::clamp<int>(std::ceil(0.12 / estimate), 1, 10000000);
            auto before   = live_bytes();
            auto seconds  = measure(count);
            while (seconds * count < 0.1 && count < 10000000) {
                count = std::clamp<int>(std::ceil(0.12 / seconds), count + 1, 10000000);
                before = live_bytes();
                seconds = measure(count);
            }
            auto after    = live_bytes();
            rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            std::cout << "{\"seconds\":" << seconds << ",\"iterations\":" << count
                      << ",\"total_ms\":" << seconds * count * 1000 << ",\"maxrss_kib\":" << usage.ru_maxrss
                      << ",\"live_growth_bytes\":" << static_cast<int64_t>(after) - static_cast<int64_t>(before)
                      << ",\"warmup_calls\":" << warmup_calls << ",\"warmup_seconds\":" << warmup_seconds << "}\n"
                      << std::flush;
        };
        if (stream) {
            std::cout << "{\"ready\":true}\n" << std::flush;
            std::string request;
            while (std::getline(std::cin, request))
                sample();
        } else {
            sample();
        }
        if (handle) reinterpret_cast<void (*)(void*)>(symbol("destroy_attention"))(handle);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
