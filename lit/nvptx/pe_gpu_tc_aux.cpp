#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000u;
    int exp       = int((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return sign;
    if (exp >= 31) return uint16_t(sign | 0x7c00u);
    return uint16_t(sign | (uint16_t(exp) << 10) | uint16_t(mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t x    = 0;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
            mant &= 0x3ffu;
            x = sign | (uint32_t(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &x, 4);
    return out;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    return uint16_t((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t b) {
    uint32_t x = uint32_t(b) << 16;
    float out;
    std::memcpy(&out, &x, 4);
    return out;
}

extern "C" {
void* gemm_f16_nn(const uint16_t* a, const uint16_t* b);
void* gemm_bf16_nn(const uint16_t* a, const uint16_t* b);
void* gemm_f16_nt(const uint16_t* a, const uint16_t* b);
void* gemm_f16_odd(const uint16_t* a, const uint16_t* b);
void* gemm_f16_batched(const uint16_t* a, const uint16_t* b);
void* gemm_f16_bias(const uint16_t* a, const uint16_t* b, const float* bias);
}

static void check_gemm(const char* name, int B, int M, int N, int K, bool trans_b, bool bf16,
                       const uint16_t* a, const uint16_t* b, const float* bias, const float* got) {
    auto dec = [&](uint16_t v) { return bf16 ? bf16_to_f32(v) : f16_to_f32(v); };
    double max_err = 0.0;
    for (int bi = 0; bi < B; ++bi) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double ref = bias ? double(bias[j]) : 0.0;
                for (int k = 0; k < K; ++k) {
                    float av = dec(a[(bi * M + i) * K + k]);
                    float bv = trans_b ? dec(b[(bi * N + j) * K + k]) : dec(b[(bi * K + k) * N + j]);
                    ref += double(av) * double(bv);
                }
                double err = std::fabs(double(got[(bi * M + i) * N + j]) - ref);
                if (err > max_err) max_err = err;
            }
        }
    }
    if (max_err > 1e-2) {
        std::fprintf(stderr, "FAIL %s: max_err=%.6g\n", name, max_err);
        std::exit(1);
    }
    std::printf("PASS %-18s max_err=%.2e\n", name, max_err);
}

int main() {
    auto fill_f16 = [](size_t n, int seed) {
        std::vector<uint16_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = f32_to_f16(float(int((i * 7 + seed) % 13) - 6) * 0.125f);
        return v;
    };
    auto fill_bf16 = [](size_t n, int seed) {
        std::vector<uint16_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = f32_to_bf16(float(int((i * 5 + seed) % 11) - 5) * 0.125f);
        return v;
    };

    {
        auto a = fill_f16(128 * 64, 1), b = fill_f16(64 * 128, 3);
        auto* out = static_cast<float*>(gemm_f16_nn(a.data(), b.data()));
        check_gemm("gemm_f16_nn", 1, 128, 128, 64, false, false, a.data(), b.data(), nullptr, out);
        std::free(out);
    }
    {
        auto a = fill_bf16(128 * 96, 2), b = fill_bf16(96 * 128, 5);
        auto* out = static_cast<float*>(gemm_bf16_nn(a.data(), b.data()));
        check_gemm("gemm_bf16_nn", 1, 128, 128, 96, false, true, a.data(), b.data(), nullptr, out);
        std::free(out);
    }
    {
        auto a = fill_f16(64 * 64, 4), b = fill_f16(64 * 64, 7);
        auto* out = static_cast<float*>(gemm_f16_nt(a.data(), b.data()));
        check_gemm("gemm_f16_nt", 1, 64, 64, 64, true, false, a.data(), b.data(), nullptr, out);
        std::free(out);
    }
    {
        auto a = fill_f16(53 * 40, 6), b = fill_f16(40 * 72, 9);
        auto* out = static_cast<float*>(gemm_f16_odd(a.data(), b.data()));
        check_gemm("gemm_f16_odd", 1, 53, 72, 40, false, false, a.data(), b.data(), nullptr, out);
        std::free(out);
    }
    {
        auto a = fill_f16(3 * 32 * 32, 8), b = fill_f16(3 * 32 * 32, 11);
        auto* out = static_cast<float*>(gemm_f16_batched(a.data(), b.data()));
        check_gemm("gemm_f16_batched", 3, 32, 32, 32, false, false, a.data(), b.data(), nullptr, out);
        std::free(out);
    }
    {
        auto a = fill_f16(64 * 32, 10), b = fill_f16(32 * 64, 13);
        std::vector<float> bias(64);
        for (int j = 0; j < 64; ++j) bias[j] = float(j - 32) * 0.25f;
        auto* out = static_cast<float*>(gemm_f16_bias(a.data(), b.data(), bias.data()));
        check_gemm("gemm_f16_bias", 1, 64, 64, 32, false, false, a.data(), b.data(), bias.data(), out);
        std::free(out);
    }
    return 0;
}
