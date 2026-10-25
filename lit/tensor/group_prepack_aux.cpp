#include <cstdint>
#include <initializer_list>

extern "C" {
int32_t* grouped(int32_t*);
int32_t* at(int32_t*);
int32_t* rev(int32_t*, int32_t*);
}

// x[i, j, k] = 12 i + 3 j + k, so C[j] = Σ_i Σ_k x[i, j, k] = 558 + 54 j.
// y[i, j] = 4 i + j, so rev's C[i, j] = Σ_k x[i, j, k] + y[i, j] = 40 i + 10 j + 3.
int main() {
    int32_t x[6 * 4 * 3];
    for (int i = 0; i != 6 * 4 * 3; ++i)
        x[i] = i;
    int bad = 0;
    for (auto f : {grouped, at})
        for (int call = 0; call != 2; ++call) {
            auto c = f(x);
            for (int j = 0; j != 4; ++j)
                bad += c[j] != 558 + 54 * j;
        }
    int32_t y[6 * 4];
    for (int i = 0; i != 6 * 4; ++i)
        y[i] = i;
    for (int call = 0; call != 2; ++call) {
        auto c = rev(x, y);
        for (int i = 0; i != 6; ++i)
            for (int j = 0; j != 4; ++j)
                bad += c[i * 4 + j] != 40 * i + 10 * j + 3;
    }
    return bad;
}
