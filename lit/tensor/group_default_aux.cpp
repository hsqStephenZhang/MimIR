#include <cassert>
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <vector>

extern "C" float* gemm(float*, float*, float*);
extern "C" float* mlp(float*, float*, float*);
extern "C" float* soft(float*);
extern "C" float* layer(float*);
extern "C" float* conv_chain(float*, float*, float*);

static std::vector<float> input(int size) {
    std::vector<float> result(size);
    for (int i = 0; i != size; ++i)
        result[i] = float(i % 17 - 8) / 8;
    return result;
}

static std::vector<float> product(const std::vector<float>& a, const std::vector<float>& b, int m, int k, int n) {
    std::vector<float> result(m * n);
    for (int i = 0; i != m; ++i)
        for (int j = 0; j != n; ++j)
            for (int r = 0; r != k; ++r)
                result[i * n + j] += a[i * k + r] * b[r * n + j];
    return result;
}

static void check(float* actual, const std::vector<float>& expected) {
    for (size_t i = 0; i != expected.size(); ++i)
        assert(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 0.0005);
    std::free(actual);
}

static std::vector<float> convolution(const std::vector<float>& a,
                                      const std::vector<float>& weights,
                                      int inputs,
                                      int outputs,
                                      int height,
                                      int width) {
    std::vector<float> result(outputs * (height - 2) * (width - 2));
    for (int o = 0; o != outputs; ++o)
        for (int y = 0; y != height - 2; ++y)
            for (int x = 0; x != width - 2; ++x)
                for (int c = 0; c != inputs; ++c)
                    for (int ky = 0; ky != 3; ++ky)
                        for (int kx = 0; kx != 3; ++kx)
                            result[(o * (height - 2) + y) * (width - 2) + x]
                                += a[(c * height + y + ky) * width + x + kx]
                                 * weights[((o * inputs + c) * 3 + ky) * 3 + kx];
    return result;
}

int main() {
    auto a = input(13 * 7), b = input(7 * 11), w = input(11 * 5), bias = input(11);
    auto hidden = product(a, b, 13, 7, 11);
    auto biased = hidden;
    for (int i = 0; i != 13 * 11; ++i) {
        biased[i] = std::max(0.0f, biased[i] + bias[i % 11]);
        hidden[i] = std::max(0.0f, hidden[i]);
    }
    check(gemm(a.data(), b.data(), bias.data()), biased);
    check(mlp(a.data(), b.data(), w.data()), product(hidden, w, 13, 11, 5));
    a = input(5 * 7);
    std::vector<float> expected_soft(35), expected_layer(35);
    for (int i = 0; i != 5; ++i) {
        double mean = 0, variance = 0, sum = 0;
        auto maximum = *std::max_element(a.begin() + i * 7, a.begin() + (i + 1) * 7);
        for (int j = 0; j != 7; ++j) {
            mean += a[i * 7 + j] / 7.0;
            sum += std::exp(double(a[i * 7 + j]) - maximum);
        }
        for (int j = 0; j != 7; ++j)
            variance += std::pow(a[i * 7 + j] - mean, 2) / 7.0;
        for (int j = 0; j != 7; ++j) {
            expected_soft[i * 7 + j]  = std::exp(double(a[i * 7 + j]) - maximum) / sum;
            expected_layer[i * 7 + j] = (a[i * 7 + j] - mean) / std::sqrt(variance + 0.00001);
        }
    }
    check(soft(a.data()), expected_soft);
    check(layer(a.data()), expected_layer);
    a = input(2 * 7 * 8);
    b = input(3 * 2 * 3 * 3);
    w = input(2 * 3 * 3 * 3);
    check(conv_chain(a.data(), b.data(), w.data()), convolution(convolution(a, b, 2, 3, 7, 8), w, 3, 2, 5, 6));
}
