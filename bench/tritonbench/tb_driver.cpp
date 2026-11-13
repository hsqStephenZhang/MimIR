// Runs a compiled `FN(in_0, …, in_{n-1})` on the raw inputs in files and writes its result: `tb_driver iters out bytes in…`.
// The inputs and the result are host buffers, as a MimIR host function takes and returns them.
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void* FN(...);

static std::vector<char> slurp(const char* path) {
    auto f = std::fopen(path, "rb");
    if (!f) std::exit((std::perror(path), 2));
    std::fseek(f, 0, SEEK_END);
    std::vector<char> data(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) std::exit(2);
    std::fclose(f);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 5) return std::fprintf(stderr, "usage: %s iters out bytes in...\n", argv[0]), 2;
    int iters  = std::atoi(argv[1]);
    auto bytes = std::strtoull(argv[3], nullptr, 10);
    std::vector<std::vector<char>> ins;
    for (int i = 4; i < argc; ++i) ins.push_back(slurp(argv[i]));
    void* p[4] = {};
    for (size_t i = 0; i < ins.size() && i < 4; ++i) p[i] = ins[i].data();
    void* out = nullptr;
    for (int it = 0; it < iters; ++it) {
        if (out) std::free(out);
        switch (ins.size()) {
            case 1: out = FN(p[0]); break;
            case 2: out = FN(p[0], p[1]); break;
            case 3: out = FN(p[0], p[1], p[2]); break;
            default: out = FN(p[0], p[1], p[2], p[3]); break;
        }
    }
    auto f = std::fopen(argv[2], "wb");
    std::fwrite(out, 1, bytes, f);
    std::fclose(f);
}
