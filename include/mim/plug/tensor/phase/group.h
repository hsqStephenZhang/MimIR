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
    DefMap<Request> requests_;
};

class ReportGroups : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App*) override;
};

/// Without tile-and-fuse, drops every `tile_req` up front, as `ReportGroups` would later.
/// Rebuilding one after earlier phases rewrote its op would retype its levels by a `root_loops op` no longer known.
class DropTileRequests : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    const Def* rewrite_imm_App(const App*) override;
};

void report_group_request(const App*, std::string_view reason);
void report_tile_request(const App*, std::string_view reason);

} // namespace mim::plug::tensor::phase
