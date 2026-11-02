#pragma once

#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class Group : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    struct Request {
        u64 kind, dim, tile;
        const Def* params;
        DefVec consumers;
    };

    void start() override;
    const Def* rewrite_imm_App(const App*) override;
    DefMap<fe::Vector<Request>> requests_;
};

class ReportGroups : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App*) override;
};

void report_group_request(const App*, std::string_view reason);

} // namespace mim::plug::tensor::phase
