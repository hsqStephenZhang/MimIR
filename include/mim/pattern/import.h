#pragma once

#include <stdexcept>

#include "mim/axm.h"
#include "mim/check.h"
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
            // LIFO preserves the Rewriter's argument-before-callee application order.
            for (auto dependency : dependencies(current))
                pending.push_back({dependency, false});
        }
        return lookup(def);
    }

protected:
    const Def* share_axiom(const Axm* value) {
        if (value->plugin() == Annex::Global_Plugin || !value->is_closed()) return value;
        auto& copies = axioms_[value->flags()];
        for (auto existing : copies) {
            if (!remaining_) throw RebuildLimit("graph axiom import work budget exceeded");
            --remaining_;
            if (existing->normalizer() == value->normalizer() && existing->curry() == value->curry()
                && existing->trip() == value->trip() && Check::test_alpha(existing->type(), value->type()))
                return existing;
        }
        copies.push_back(value);
        return value;
    }
    virtual bool descend(const Def*) const { return true; }
    virtual const Def* rewrite_ready(const Def* def) { return Rewriter::rewrite(def); }
    virtual DefVec dependencies(const Def* def) const {
        DefVec result;
        if (auto type = def->type()) result.push_back(type);
        for (auto op : def->ops())
            if (op) result.push_back(op);
        return result;
    }

private:
    size_t remaining_;
    ankerl::unordered_dense::map<flags_t, fe::Vector<const Axm*>> axioms_;
};

} // namespace mim::pattern
