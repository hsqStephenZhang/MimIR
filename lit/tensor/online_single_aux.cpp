#include <cmath>
#include <cstdlib>

extern "C" float* singleton(float*, float, float*);
extern "C" float scalar(float, float, float);

int main() {
    float q[] = {1, 2, -3}, v[] = {2, -5};
    auto out = singleton(q, 4, v);
    for (int i = 0; i != 6; ++i)
        if (std::abs(out[i] - v[i % 2]) > 2e-4f) return 1;
    std::free(out);
    return std::abs(scalar(2, 3, 7) - 7) > 2e-4f;
}
