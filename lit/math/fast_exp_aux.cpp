#include <cmath>
#include <cstdio>

extern "C" float fexp(float);
extern "C" float sexp(float);
extern "C" float aexp(float);

int main() {
    int bad = 0;
    for (int i = 0; i <= 200000; ++i) {
        float x     = -100.0f + 190.0f * float(i) / 200000;
        double want = std::exp(double(x));
        float got   = fexp(x);
        if (want < 1.2e-38) {
            if (x < -87.4f && got != 0.0f) ++bad;
        } else if (want < 3.4028e38 && std::fabs(got - want) > 5e-7 * want) {
            ++bad;
        }
        if (std::fabs(sexp(x) - std::exp(x)) > 1e-6 * std::exp(x)) ++bad;
    }
    if (fexp(-INFINITY) != 0.0f || fexp(0.0f) != 1.0f) ++bad;
    if (!std::isnan(aexp(NAN)) || aexp(INFINITY) != INFINITY || aexp(89.0f) != INFINITY || aexp(-INFINITY) != 0.0f) ++bad;
    if (std::fabs(aexp(1.0f) - std::exp(1.0)) > 5e-7 * std::exp(1.0)) ++bad;
    if (bad) std::printf("%d wrong\n", bad);
    return bad != 0;
}
