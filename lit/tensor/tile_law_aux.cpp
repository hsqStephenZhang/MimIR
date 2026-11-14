#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" int64_t* law_denote(int64_t*, int64_t*);
extern "C" int64_t* law_req(int64_t*, int64_t*);
extern "C" int64_t* law_par(int64_t*, int64_t*);
extern "C" int64_t* law_red(int64_t*, int64_t*);

int main() {
    int64_t a[15], b[12], want[20];
    for (int i = 0; i != 15; ++i)
        a[i] = 3 * i - 7;
    for (int i = 0; i != 12; ++i)
        b[i] = -2 * i + 5;
    for (int m = 0; m != 5; ++m)
        for (int n = 0; n != 4; ++n) {
            want[m * 4 + n] = 0;
            for (int k = 0; k != 3; ++k)
                want[m * 4 + n] += a[m * 3 + k] * b[k * 4 + n];
        }
    int failures = 0;
    struct Case {
        int64_t* (*f)(int64_t*, int64_t*);
        const char* what;
    };
    for (auto [f, what] : {
             Case{law_denote, "denote"},
             Case{   law_req,    "req"},
             Case{   law_par,    "par"},
             Case{   law_red,    "red"}
    }) {
        auto got = f(a, b);
        for (int i = 0; i != 20; ++i)
            if (got[i] != want[i]) {
                std::printf("%s: element %d is %lld, not %lld\n", what, i, (long long)got[i], (long long)want[i]);
                ++failures;
                break;
            }
        std::free(got);
    }
    return failures != 0;
}
