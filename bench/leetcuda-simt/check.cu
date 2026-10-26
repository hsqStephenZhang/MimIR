#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA(call)                                                                                                 \
    do {                                                                                                           \
        auto error = (call);                                                                                       \
        if (error != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error)); \
    } while (false)
#define DRIVER(call)                                                                                            \
    do {                                                                                                        \
        auto error = (call);                                                                                    \
        if (error != CUDA_SUCCESS) {                                                                            \
            const char* message = nullptr;                                                                      \
            cuGetErrorString(error, &message);                                                                  \
            throw std::runtime_error(std::string(#call) + ": " + (message ? message : "unknown driver error")); \
        }                                                                                                       \
    } while (false)

static double activation(const std::string& op, double x, double y) {
    if (op == "add") return x + y;
    if (op == "mul") return x * y;
    if (op == "relu") return std::max(x, 0.0);
    if (op == "sigmoid") return 1.0 / (1.0 + std::exp(-x));
    if (op == "swish") return x / (1.0 + std::exp(-x));
    if (op == "elu") return x > 0 ? x : std::expm1(x);
    if (op == "hardshrink") return std::abs(x) > 0.5 ? x : 0;
    if (op == "hardswish") return x * std::clamp(x + 3, 0.0, 6.0) / 6;
    if (op == "gelu_tanh")
        return 0.5 * x * (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) * (x + 0.044715 * x * x * x)));
    if (op == "gelu_erf") return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
    throw std::runtime_error("No CPU reference for " + op);
}

static std::vector<double> reference(const std::string& op,
                                     const std::vector<double>& a,
                                     const std::vector<double>& b,
                                     int no,
                                     int w,
                                     int h,
                                     int k,
                                     bool f64) {
    std::vector<double> out(no);
    if (op == "math_pow" || op == "math_erf" || op == "math_erfc") {
        for (int i = 0; i < no; ++i)
            if (f64) {
                long double x = a[i], y = b[i];
                out[i] = double(op == "math_pow" ? std::pow(x, y) : op == "math_erf" ? std::erf(x) : std::erfc(x));
            } else
                out[i] = float(op == "math_pow" ? std::pow(a[i], b[i]) : std::erf(a[i]));
        return out;
    }
    if (op == "nms_presorted") {
        for (int i = 0; i < no; ++i) {
            bool keep = true;
            for (int j = 0; j < i && keep; ++j) {
                if (!out[j]) continue;
                double width
                    = std::max(0.0, double(std::min(a[i * 4 + 2], a[j * 4 + 2]) - std::max(a[i * 4], a[j * 4])));
                double height = std::max(
                    0.0, double(std::min(a[i * 4 + 3], a[j * 4 + 3]) - std::max(a[i * 4 + 1], a[j * 4 + 1])));
                double ai           = double(a[i * 4 + 2] - a[i * 4]) * (a[i * 4 + 3] - a[i * 4 + 1]);
                double aj           = double(a[j * 4 + 2] - a[j * 4]) * (a[j * 4 + 3] - a[j * 4 + 1]);
                double intersection = width * height;
                if (intersection > 0.5 * (ai + aj - intersection)) keep = false;
            }
            out[i] = keep;
        }
        return out;
    }
    if (op == "attention") {
        for (int row = 0; row < h; ++row) {
            std::vector<double> logits(h);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < h; ++key) {
                for (int j = 0; j < w; ++j)
                    logits[key] += double(a[row * w + j]) * b[key * w + j];
                logits[key] /= std::sqrt(double(w));
                if (k && key > row) logits[key] = -std::numeric_limits<double>::infinity();
                maximum = std::max(maximum, logits[key]);
            }
            double den = 0;
            for (auto& value : logits)
                den += value = std::exp(value - maximum);
            for (int col = 0; col < w; ++col)
                for (int key = 0; key < h; ++key)
                    out[row * w + col] += logits[key] * b[h * w + key * w + col] / den;
        }
        return out;
    }
    if (op == "transpose") {
        for (int i = 0; i < no; ++i)
            out[i] = a[(i % h) * w + i / h];
        return out;
    }
    if (op == "embedding") {
        for (int i = 0; i < no; ++i)
            out[i] = a[int(b[i / w]) * w + i % w];
        return out;
    }
    if (op == "rope") {
        for (int i = 0; i < no; ++i) {
            int row = i / w, pair = (i % w) / 2, base = row * w + pair * 2;
            double angle = row * std::pow(10000.0, -2.0 * pair / w);
            out[i]       = i % 2 == 0 ? a[base] * std::cos(angle) - a[base + 1] * std::sin(angle)
                                      : a[base] * std::sin(angle) + a[base + 1] * std::cos(angle);
        }
        return out;
    }
    if (op == "merge_attention") {
        for (int i = 0; i < no; ++i) {
            double m = std::max(b[i], b[i + no]);
            double x = std::exp(b[i] - m), y = std::exp(b[i + no] - m);
            out[i] = (a[i] * x + a[i + no] * y) / (x + y);
        }
        return out;
    }
    if (op == "reduce_sum" || op == "reduce_max" || op == "dot" || op == "sgemv" || op == "sgemm"
        || op == "histogram") {
        for (int i = 0; i < no; ++i) {
            double value = op == "reduce_max" ? -std::numeric_limits<double>::infinity() : 0;
            int count    = op == "sgemm" ? k : w;
            for (int j = 0; j < count; ++j)
                if (op == "histogram")
                    value += int(a[j]) == i;
                else if (op == "dot")
                    value += double(a[j]) * b[j];
                else if (op == "sgemv")
                    value += double(a[i * w + j]) * b[j];
                else if (op == "sgemm")
                    value += double(a[(i / w) * k + j]) * b[j * w + i % w];
                else if (op == "reduce_max")
                    value = std::max(value, double(a[i * w + j]));
                else
                    value += a[i * w + j];
            out[i] = value;
        }
        return out;
    }
    if (op == "softmax" || op == "softmax_safe" || op == "log_softmax" || op == "layer_norm" || op == "rms_norm") {
        for (int row = 0; row < h; ++row) {
            double mean = 0, maximum = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < w; ++j) {
                mean += a[row * w + j] / double(w);
                maximum = std::max(maximum, double(a[row * w + j]));
            }
            if (op == "rms_norm") mean = 0;
            double den = 0;
            for (int j = 0; j < w; ++j) {
                double x = a[row * w + j];
                den += op.find("softmax") != std::string::npos ? std::exp(x - maximum) : (x - mean) * (x - mean) / w;
            }
            for (int j = 0; j < w; ++j) {
                double x         = a[row * w + j];
                out[row * w + j] = op == "log_softmax"     ? x - maximum - std::log(den)
                                 : op.find("softmax") == 0 ? std::exp(x - maximum) / den
                                                           : (x - mean) / std::sqrt(den + 1e-5);
            }
        }
        return out;
    }
    for (int i = 0; i < no; ++i)
        out[i] = activation(op, a[i], b[i]);
    return out;
}

static size_t element_size(const std::string& type) {
    if (type == "F16") return 2;
    if (type == "F32" || type == "I32") return 4;
    if (type == "F64") return 8;
    throw std::runtime_error("Unsupported storage type: " + type);
}

static std::vector<unsigned char> pack(std::vector<double>& values, const std::string& type) {
    std::vector<unsigned char> bytes(values.size() * element_size(type));
    for (size_t i = 0; i < values.size(); ++i) {
        auto ptr = bytes.data() + i * element_size(type);
        if (type == "F16") {
            __half value = __float2half_rn(values[i]);
            std::memcpy(ptr, &value, sizeof(value));
            values[i] = __half2float(value);
        } else if (type == "I32") {
            auto value = int32_t(values[i]);
            std::memcpy(ptr, &value, sizeof(value));
            values[i] = float(value);
        } else if (type == "F64")
            std::memcpy(ptr, &values[i], sizeof(double));
        else {
            auto value = float(values[i]);
            std::memcpy(ptr, &value, sizeof(value));
            values[i] = value;
        }
    }
    return bytes;
}

static std::vector<double> unpack(const std::vector<unsigned char>& bytes, const std::string& type) {
    std::vector<double> values(bytes.size() / element_size(type));
    for (size_t i = 0; i < values.size(); ++i) {
        auto ptr = bytes.data() + i * element_size(type);
        if (type == "F16") {
            __half value;
            std::memcpy(&value, ptr, sizeof(value));
            values[i] = __half2float(value);
        } else if (type == "I32") {
            int32_t value;
            std::memcpy(&value, ptr, sizeof(value));
            values[i] = float(value);
        } else if (type == "F64")
            std::memcpy(&values[i], ptr, sizeof(double));
        else {
            float value;
            std::memcpy(&value, ptr, sizeof(value));
            values[i] = value;
        }
    }
    return values;
}

int main(int argc, char** argv) {
    try {
        if (argc != 14 && argc != 15)
            throw std::runtime_error(
                "Usage: check cubin kernel operation na nb no width height depth groups Ta Tb Tc [--check-only]");
        bool check_only = argc == 15;
        if (check_only && std::string(argv[14]) != "--check-only") throw std::runtime_error("Unknown option");
        std::string op = argv[3];
        std::string ta = argv[11], tb = argv[12], tc = argv[13];
        bool math = op == "math_pow" || op == "math_erf" || op == "math_erfc";
        if (tc == "F64" && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
            throw std::runtime_error("F64 math validation requires a wider long double CPU reference");
        int na = std::stoi(argv[4]), nb = std::stoi(argv[5]), no = std::stoi(argv[6]);
        int w = std::stoi(argv[7]), h = std::stoi(argv[8]), k = std::stoi(argv[9]), groups = std::stoi(argv[10]);
        CUDA(cudaSetDevice(0));
        CUDA(cudaFree(nullptr));
        CUmodule module;
        CUfunction kernel;
        DRIVER(cuModuleLoad(&module, argv[1]));
        DRIVER(cuModuleGetFunction(&kernel, module, argv[2]));
        int registers, shared_bytes;
        DRIVER(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel));
        DRIVER(cuFuncGetAttribute(&shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));
        void *da, *db, *dc;
        CUDA(cudaMalloc(&da, na * element_size(ta)));
        CUDA(cudaMalloc(&db, nb * element_size(tb)));
        CUDA(cudaMalloc(&dc, no * element_size(tc)));
        std::array<CUdeviceptr, 3> pointers{reinterpret_cast<CUdeviceptr>(da), reinterpret_cast<CUdeviceptr>(db),
                                            reinterpret_cast<CUdeviceptr>(dc)};
        void* parameters[] = {pointers.data()};
        auto launch = [&] { DRIVER(cuLaunchKernel(kernel, groups, 1, 1, 256, 1, 1, 0, nullptr, parameters, nullptr)); };
        std::vector<double> a(na), b(nb);
        std::vector<unsigned char> output(no * element_size(tc));
        double worst = 0, worst_ulp = 0;
        for (int fixture = 0; fixture < 3; ++fixture) {
            const float edges[] = {-80, -4, -3, -0.5f, 0, 0.5f, 3, 4, 80};
            for (int i = 0; i < na; ++i)
                a[i] = fixture == 0 ? float((i * 37 + 11) % 101 - 50) / 23 : fixture == 1 ? edges[i % 9] : 0.25f;
            for (int i = 0; i < nb; ++i)
                b[i] = fixture == 0 ? float((i * 17 + 7) % 97 - 48) / 31 : fixture == 1 ? edges[(i * 7) % 9] : -0.125f;
            if (op == "embedding")
                for (int i = 0; i < nb; ++i)
                    b[i] = float((i * 7 + fixture) % h);
            if (op == "histogram")
                for (int i = 0; i < na; ++i)
                    a[i] = float((i * 11 + fixture) % h);
            if (op == "nms_presorted")
                for (int i = 0; i < no; ++i) {
                    float x      = fixture == 2 ? 0 : fixture == 1 ? float(i * 4) : float((i / 3) * 4);
                    float y      = fixture == 2 ? 0 : float((i / 9) * 3);
                    a[i * 4]     = x;
                    a[i * 4 + 1] = y;
                    a[i * 4 + 2] = x + 2;
                    a[i * 4 + 3] = y + 2;
                    b[i]         = float(no - i);
                }
            if (op == "nms_presorted" && fixture == 0) {
                const float chain[] = {0, 0, 2, 2, 0.5f, 0, 2.5f, 2, 1, 0, 3, 2, 100, 0, 103, 2, 101, 0, 104, 2};
                std::copy(std::begin(chain), std::end(chain), a.begin());
            }
            if (fixture == 1 && (op == "softmax_safe" || op == "log_softmax"))
                for (auto& value : a)
                    value += 1000;
            if (fixture == 1 && math) {
                const double inf                         = std::numeric_limits<double>::infinity();
                const double nan                         = std::numeric_limits<double>::quiet_NaN();
                const double tiny                        = tc == "F64" ? std::numeric_limits<double>::denorm_min()
                                                                       : std::numeric_limits<float>::denorm_min();
                std::vector<std::array<double, 2>> edges = {
                    {     0,             0},
                    { -0.0f,             3},
                    { -0.0f,             2},
                    { -0.0f,            -3},
                    { -0.0f,            -2},
                    {  tiny,             1},
                    {  tiny,          0.5f},
                    {  tiny,            -1},
                    {1e-30f,             2},
                    { 1e38f,             2},
                    {    -2,             3},
                    {    -2,             2},
                    {    -2,            -3},
                    {    -2,          0.5f},
                    {   nan,             0},
                    {     1,           nan},
                    {   nan,             2},
                    {   inf,             2},
                    {  -inf,             3},
                    {  -inf,            -3},
                    {    -1,           inf},
                    {  0.5f,           inf},
                    {     2,          -inf},
                    {     2,          0.5f},
                    {     2,        17.25f},
                    {0.125f,        -3.25f},
                    { 3.99f,          0.3f},
                    { 0.25f,          2.3f},
                    { -tiny,             1},
                    {    -1, 2147483648.0f},
                };
                if (tc == "F64") {
                    for (double x : {0.84375, 1.25, 1.0 / 0.35, 6.0, 28.0, std::numeric_limits<double>::min(),
                                     std::numeric_limits<double>::max()})
                        for (double value : {std::nextafter(x, 0.0), x, std::nextafter(x, inf)}) {
                            edges.push_back({value, 0.5});
                            edges.push_back({-value, 3});
                        }
                    for (auto value : {
                             std::array<double, 2>{1e-300, 2},
                             {1e300, 2},
                             {2, 1023},
                             {2, -1074},
                             {-1, 9007199254740991.0},
                             {-1, 9007199254740992.0},
                             {std::nextafter(1.0, 2.0), 4503599627370496.0},
                             {0.0, -2},
                             {0x1p512, -2},
                             {0x1p536, -2},
                             {0x1.8p512, -2},
                             {0x1p-512, -2},
                             {0x1p-537, 2},
                             {-0.0, -0.5}
                    })
                        edges.push_back(value);
                }
                for (int i = 0; i < na; ++i) {
                    a[i] = edges[i % std::size(edges)][0];
                    b[i] = edges[i % std::size(edges)][1];
                }
            }
            if (tc == "F64" && fixture != 1) {
                for (int i = 0; i < na; ++i) {
                    if (fixture == 0) {
                        a[i] = double((i * 37 + 11) % 8191 - 4095) / 137;
                        b[i] = double((i * 17 + 7) % 8179 - 4089) / 1031;
                    } else {
                        a[i] = op == "math_pow" ? std::ldexp(1.0 + double(i % 127) / 128, (i * 37) % 2098 - 1074)
                                                : double(i - na / 2) / 683;
                        b[i] = double((i * 13) % 67 - 33) / 8;
                    }
                }
            }
            auto input_a = pack(a, ta), input_b = pack(b, tb);
            auto expected = reference(op, a, b, no, w, h, k, tc == "F64");
            CUDA(cudaMemcpy(da, input_a.data(), input_a.size(), cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(db, input_b.data(), input_b.size(), cudaMemcpyHostToDevice));
            CUDA(cudaMemset(dc, 0xff, output.size()));
            launch();
            CUDA(cudaMemcpy(output.data(), dc, output.size(), cudaMemcpyDeviceToHost));
            auto got = unpack(output, tc);
            for (int i = 0; i < no; ++i) {
                if (std::isnan(expected[i]) && std::isnan(got[i])) continue;
                if (std::isinf(expected[i]) && got[i] == expected[i]) continue;
                if (!std::isfinite(expected[i]) || !std::isfinite(got[i]))
                    throw std::runtime_error(op + " special-value mismatch at " + std::to_string(i));
                if (math && expected[i] == 0 && got[i] == 0 && std::signbit(expected[i]) != std::signbit(got[i]))
                    throw std::runtime_error(op + " signed-zero mismatch at " + std::to_string(i));
                double error     = std::abs(double(got[i]) - expected[i]);
                double tolerance = tc == "I32" ? 0 : 3e-4 + 5e-4 * std::abs(expected[i]);
                if (math) {
                    double value = expected[i];
                    double next  = tc == "F64" ? std::nextafter(value, value == 0 ? 1.0 : 0.0)
                                               : std::nextafter(float(value), value == 0 ? 1.0f : 0.0f);
                    tolerance    = 2 * std::abs(next - value);
                    worst_ulp    = std::max(worst_ulp, error / std::abs(next - value));
                }
                if (!(error <= tolerance)) {
                    char values[256];
                    std::snprintf(values, sizeof(values), " got=%a expected=%a error=%g tolerance=%g", got[i],
                                  expected[i], error, tolerance);
                    throw std::runtime_error(op + " fixture=" + std::to_string(fixture) + " index=" + std::to_string(i)
                                             + values);
                }
                worst = std::max(worst, error);
            }
        }
        for (int i = 0; i < (check_only ? 0 : 20); ++i)
            launch();
        cudaEvent_t start, stop;
        CUDA(cudaEventCreate(&start));
        CUDA(cudaEventCreate(&stop));
        std::array<float, 5> samples{};
        for (auto& sample : samples) {
            if (check_only) break;
            CUDA(cudaEventRecord(start));
            for (int i = 0; i < 100; ++i)
                launch();
            CUDA(cudaEventRecord(stop));
            CUDA(cudaEventSynchronize(stop));
            CUDA(cudaEventElapsedTime(&sample, start, stop));
            sample /= 100;
        }
        std::sort(samples.begin(), samples.end());
        std::printf("{\"fixtures\":3,\"values\":%d,\"max_abs_error\":%.10g,\"median_ms\":%.9g,\"registers\":%d,"
                    "\"shared_bytes\":%d",
                    no * 3, worst, samples[2], registers, shared_bytes);
        if (math)
            std::printf(",\"max_ulp_error\":%.10g,\"reference_precision_bits\":%d", worst_ulp,
                        tc == "F64" ? std::numeric_limits<long double>::digits : std::numeric_limits<double>::digits);
        std::puts("}");
        CUDA(cudaEventDestroy(start));
        CUDA(cudaEventDestroy(stop));
        CUDA(cudaFree(da));
        CUDA(cudaFree(db));
        CUDA(cudaFree(dc));
        DRIVER(cuModuleUnload(module));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
