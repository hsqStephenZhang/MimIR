#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class GroupRules : public Phase {
public:
    using Phase::Phase;

private:
    void start() override;
};

class GroupDefault : public Phase {
public:
    using Phase::Phase;

private:
    void start() override;
};

} // namespace mim::plug::tensor::phase
