#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int64_t* stage_share(int64_t*);

int main() {
    int64_t input[35];
    for (int call = 0; call != 2; ++call) {
        for (int i = 0; i != 35; ++i)
            input[i] = i * 7 - 11 + call * 23;
        auto result = stage_share(input);
        bool ok     = true;
        for (int i = 0; i != 35; ++i)
            ok &= result[i] == input[i];
        std::free(result);
        if (!ok) {
            std::fprintf(stderr, "nested stage closure reference mismatch\n");
            return 1;
        }
    }
    return 0;
}
