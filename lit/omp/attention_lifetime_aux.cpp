#include <cassert>
#include <cmath>
#include <cstdlib>

#include <vector>

extern "C" float* attention(float*, float*, float*);
extern "C" float* causal(float*, float*, float*);

int main() {
    std::vector<float> q(4 * 32 * 16, 0.01f), k(q), v(q);
    for (auto fn : {attention, causal}) {
        for (int call = 0; call != 200; ++call) {
            auto output = fn(q.data(), k.data(), v.data());
            for (size_t i = 0; i != q.size(); ++i)
                assert(std::isfinite(output[i]) && std::abs(output[i] - 0.01f) < 1e-5f);
            std::free(output);
        }
    }
}
