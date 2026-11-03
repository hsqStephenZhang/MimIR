#pragma once

#include "mim/rewrite.h"
#include "mim/rule.h"

#include "mim/pattern/graph.h"
#include "mim/pattern/match.h"

namespace mim::pattern {

class ParameterImporter : public Rewriter {
public:
    using Rewriter::Rewriter;

private:
    const Def* rewrite_imm_Var(const Var*) override;
};

struct DataOp {
    flags_t axm;
    size_t num_groups;
    std::vector<InputGroup> inputs;
};

std::vector<DataOp> parse_data_ops(const Def*);
OpRegistry data_operations(std::span<const DataOp>);
DefVec function_bodies(World&);

class GuardContext {
public:
    explicit GuardContext(const GraphView&);
    const GraphView& graph;
    std::optional<flags_t> used_by, only_used_by;
};

struct GuardResult {
    MatchStatus status;
    std::string reason;
    Def2Def values                      = {};
    std::shared_ptr<World> values_world = {};
};

GuardResult evaluate_guard(const Rule*, const Def2Def&, Defs lhs, Defs matched, const GuardContext* = nullptr);

} // namespace mim::pattern
