#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class AttentionRaise : public Phase {
public:
    using Phase::Phase;

private:
    void start() override;
};

class AttentionSchedule : public Phase {
public:
    using Phase::Phase;

private:
    void start() override;
};

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

class FuseRegionInputs : public Phase {
public:
    using Phase::Phase;
    void apply(const App*) override;
    std::unique_ptr<Phase> recreate() override;

private:
    void start() override;
    u64 mode_ = 0, budget_ = 1;
    bool graph_ = true;
};

class FuseAndYield : public Phase {
public:
    using Phase::Phase;
    void apply(const App*) override;
    std::unique_ptr<Phase> recreate() override;

private:
    void start() override;
    u64 budget_ = 1;
};

} // namespace mim::plug::tensor::phase
