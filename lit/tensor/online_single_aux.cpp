#include <cmath>
#include <cstdlib>

extern "C" float* singleton(float*, float*, float*);
extern "C" float* scalar(float*, float*, float*);

int main() {
    float q[] = {1, 2, -3}, v[] = {2, -5}, k = 4;
    auto out = singleton(q, &k, v);
    for (int i = 0; i != 6; ++i)
        if (std::abs(out[i] - v[i % 2]) > 2e-4f) return 1;
    std::free(out);
    float a = 2, b = 3, c = 7;
    auto value  = scalar(&a, &b, &c);
    auto failed = std::abs(*value - 7) > 2e-4f;
    std::free(value);
    return failed;
}
