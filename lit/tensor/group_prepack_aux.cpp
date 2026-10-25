#include <cstdint>
#include <initializer_list>

extern "C" {
int32_t* grouped(int32_t*);
int32_t* at(int32_t*);
}

// x[i, j, k] = 12 i + 3 j + k, so C[j] = Σ_i Σ_k x[i, j, k] = 558 + 54 j.
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
    return bad;
}
