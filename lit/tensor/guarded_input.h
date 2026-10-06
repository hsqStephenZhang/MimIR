#pragma once

#include <cstdlib>

#include <algorithm>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

// End each input immediately before a protected page, so speculative or unguarded tail reads fault.
template<class T>
class GuardedInput {
public:
    explicit GuardedInput(const std::vector<T>& input) {
#if defined(__unix__) || defined(__APPLE__)
        size_t page  = size_t(sysconf(_SC_PAGESIZE));
        size_t bytes = input.size() * sizeof(T);
        size_        = (bytes + page - 1) / page * page + page;
        base_        = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (base_ == MAP_FAILED || mprotect(static_cast<char*>(base_) + size_ - page, page, PROT_NONE)) std::abort();
        data_ = reinterpret_cast<T*>(static_cast<char*>(base_) + size_ - page - bytes);
        std::copy(input.begin(), input.end(), data_);
#else
        copy_ = input;
        data_ = copy_.data();
#endif
    }
    ~GuardedInput() {
#if defined(__unix__) || defined(__APPLE__)
        munmap(base_, size_);
#endif
    }
    GuardedInput(const GuardedInput&)            = delete;
    GuardedInput& operator=(const GuardedInput&) = delete;
    T* data() const { return data_; }

private:
    T* data_;
#if defined(__unix__) || defined(__APPLE__)
    void* base_;
    size_t size_;
#else
    std::vector<T> copy_;
#endif
};
