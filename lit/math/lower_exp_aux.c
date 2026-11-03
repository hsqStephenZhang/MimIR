#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

float exp_f32(float);
double exp_f64(double);

static int64_t ord(float f) {
    int32_t i;
    memcpy(&i, &f, sizeof i);
    return i < 0 ? (int64_t)INT32_MIN - i : i;
}

int main(void) {
    for (float x = -110.f; x < 95.f; x += 0.0123f) {
        float a = exp_f32(x), b = expf(x);
        if (b >= 0x1p-126f && llabs(ord(a) - ord(b)) > 1) return 1;
        if (b < 0x1p-126f && fabsf(a - b) > 0x1p-148f) return 2;
    }
    float special[] = {INFINITY, -INFINITY, 0.f, 88.7229f, 89.f, -104.f, 1e30f, -1e30f};
    for (unsigned i = 0; i != sizeof special / sizeof *special; ++i)
        if (exp_f32(special[i]) != expf(special[i])) return 3;
    if (!isnan(exp_f32(NAN))) return 4;
    if (exp_f64(1.0) != exp(1.0)) return 5;
    return 0;
}
