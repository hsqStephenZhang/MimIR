#pragma once

#include <stdexcept>

#include "mim/rewrite.h"

namespace mim::pattern {

class RebuildLimit : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class DAGRewriter : public Rewriter {
public:
    DAGRewriter(World& world, size_t limit)
        : Rewriter(world)
        , remaining_(limit) {}

    const Def* rewrite(const Def* def) override {
        if (auto value = lookup(def)) return value;
        struct Frame {
            const Def* def;
            bool ready;
        };
        std::vector<Frame> pending{
            {def, false}
        };
        while (!pending.empty()) {
            auto [current, ready] = pending.back();
            pending.pop_back();
            if (lookup(current)) continue;
            if (ready || !descend(current) || current->is_mutable() || current->isa<Var>()) {
                if (!remaining_) throw RebuildLimit("graph rebuild work budget exceeded");
                --remaining_;
                rewrite_ready(current);
                continue;
            }
            pending.push_back({current, true});
            if (auto type = current->type()) pending.push_back({type, false});
            // LIFO preserves the Rewriter's argument-before-callee application order.
            for (auto op : current->ops())
                if (op) pending.push_back({op, false});
        }
        return lookup(def);
    }

protected:
    virtual bool descend(const Def*) const { return true; }
    virtual const Def* rewrite_ready(const Def* def) { return Rewriter::rewrite(def); }

private:
    size_t remaining_;
};

} // namespace mim::pattern
