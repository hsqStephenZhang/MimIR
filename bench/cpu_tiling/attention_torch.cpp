#include <chrono>
#include <cmath>

#include <string>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/core/InferenceMode.h>

namespace {
struct Kernel {
    int kind;
    bool causal;
    double scale;
    at::Tensor q, k, v, q4, k4, v4, mask, output, result;

    Kernel(int op, const int64_t* dims, float* qp, float* kp, float* vp, float* out)
        : kind(op)
        , causal(dims[5])
        , scale(static_cast<float>(1.0 / std::sqrt(dims[3]))) {
        auto h = dims[0], m = dims[1], s = dims[2], d = dims[3], n = dims[4];
        q      = at::from_blob(qp, {h, m, d}, at::kFloat);
        k      = at::from_blob(kp, {h, s, d}, at::kFloat);
        v      = at::from_blob(vp, {h, s, n}, at::kFloat);
        q4     = q.unsqueeze(0);
        k4     = k.unsqueeze(0);
        v4     = v.unsqueeze(0);
        output = at::from_blob(out, {h, m, n}, at::kFloat);
        if (causal) mask = at::ones({m, s}, at::kBool).triu(1);
    }

    void run() {
        if (kind == 0) {
            result = at::scaled_dot_product_attention(q4, k4, v4, {}, 0.0, causal, scale).squeeze(0);
        } else {
            auto scores = at::bmm(q, k.transpose(1, 2)) * scale;
            if (causal) scores = scores.masked_fill(mask, -INFINITY);
            result = at::bmm(at::softmax(scores, -1), v);
        }
    }
};
thread_local std::string error;
} // namespace

extern "C" const char* attention_error() { return error.c_str(); }

extern "C" void attention_threads(int threads) {
    at::set_num_threads(threads);
    at::set_num_interop_threads(1);
}

extern "C" void* create_attention(int kind, const int64_t* dims, float* q, float* k, float* v, float* out) {
    try {
        return new Kernel(kind, dims, q, k, v, out);
    } catch (const std::exception& e) {
        error = e.what();
        return nullptr;
    }
}

extern "C" void destroy_attention(void* ptr) { delete static_cast<Kernel*>(ptr); }

extern "C" double measure_attention(void* ptr, int count) {
    try {
        auto kernel = static_cast<Kernel*>(ptr);
        c10::InferenceMode guard;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i)
            kernel->run();
        auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / count;
        kernel->output.copy_(kernel->result);
        return seconds;
    } catch (const std::exception& e) {
        error = e.what();
        return -1;
    }
}
