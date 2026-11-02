#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <vector>

extern "C" int64_t* stride_one(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* stride_two(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* one_row(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* dilated(int64_t*, int64_t*, int64_t*);
extern "C" int64_t* chain(int64_t*, int64_t*, int64_t*);

using Values = std::vector<int64_t>;

Values convolution(const Values& input,
                   const Values& weight,
                   const Values& bias,
                   int cin,
                   int cout,
                   int h,
                   int w,
                   int stride,
                   int dilation,
                   int padding,
                   int scale) {
    auto oh = (h + 2 * padding - 2 * dilation - 1) / stride + 1;
    auto ow = (w + 2 * padding - 2 * dilation - 1) / stride + 1;
    Values output(2 * cout * oh * ow);
    for (int n = 0; n != 2; ++n)
        for (int co = 0; co != cout; ++co)
            for (int i = 0; i != oh; ++i)
                for (int j = 0; j != ow; ++j) {
                    int64_t acc = 0;
                    for (int ci = 0; ci != cin; ++ci)
                        for (int kh = 0; kh != 3; ++kh)
                            for (int kw = 0; kw != 3; ++kw) {
                                auto y = i * stride + kh * dilation - padding;
                                auto x = j * stride + kw * dilation - padding;
                                if (y >= 0 && y < h && x >= 0 && x < w)
                                    acc += scale * input[((n * cin + ci) * h + y) * w + x]
                                         * weight[((co * cin + ci) * 3 + kh) * 3 + kw];
                            }
                    acc += bias[co];
                    output[((n * cout + co) * oh + i) * ow + j] = acc > 0 ? acc : 0;
                }
    return output;
}

void check(int64_t* actual, const Values& expected) {
    for (size_t i = 0; i != expected.size(); ++i)
        assert(actual[i] == expected[i]);
    std::free(actual);
}

int main() {
    Values input(2 * 2 * 5 * 6), weight(3 * 2 * 3 * 3), bias{-13, 7, -2};
    for (size_t i = 0; i != input.size(); ++i)
        input[i] = int64_t(i % 17) - 8;
    for (size_t i = 0; i != weight.size(); ++i)
        weight[i] = int64_t(i % 7) - 3;
    auto expected = convolution(input, weight, bias, 2, 3, 5, 6, 1, 1, 1, 2);
    check(stride_one(input.data(), weight.data(), bias.data()), expected);
    check(one_row(input.data(), weight.data(), bias.data()), expected);
    check(stride_two(input.data(), weight.data(), bias.data()),
          convolution(input, weight, bias, 2, 3, 5, 6, 2, 1, 1, 2));
    check(dilated(input.data(), weight.data(), bias.data()), convolution(input, weight, bias, 2, 3, 5, 6, 1, 2, 1, 2));
    input.resize(2 * 2 * 9 * 8);
    for (size_t i = 0; i != input.size(); ++i)
        input[i] = int64_t(i % 13) - 6;
    Values second(2 * 3 * 3 * 3);
    for (size_t i = 0; i != second.size(); ++i)
        second[i] = int64_t(i % 5) - 2;
    auto intermediate = convolution(input, weight, Values(3), 2, 3, 9, 8, 1, 1, 0, 1);
    check(chain(input.data(), weight.data(), second.data()),
          convolution(intermediate, second, Values(2), 3, 2, 7, 6, 1, 1, 0, 1));
}
