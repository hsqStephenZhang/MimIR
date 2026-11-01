#include "mim/pattern/match.h"

#include <map>
#include <set>
#include <stdexcept>

#include <fe/restore.h>

#include "mim/world.h"

#include "mim/pattern/graph_match.h"

namespace mim::pattern {

const Def* MatchResult::isa_capture(Pattern p) const {
    if (snapshot_ && !snapshot_->is_live()) throw std::invalid_argument("match result references a rebuilt World");
    if (p.program_ != program_ || p.index_ >= captures_.size())
        throw std::invalid_argument("capture belongs to another program");
    return captures_[p.index_];
}

namespace {

struct Budget {
    MatchOptions options;
    size_t steps        = 0;
    size_t active_depth = 0;
};

MatchStatus alternatives(MatchStatus a, MatchStatus b) {
    if (a == MatchStatus::Matched || b == MatchStatus::Matched) return MatchStatus::Matched;
    if (a == MatchStatus::Budget_Exceeded || b == MatchStatus::Budget_Exceeded) return MatchStatus::Budget_Exceeded;
    if (a == MatchStatus::Unknown || b == MatchStatus::Unknown) return MatchStatus::Unknown;
    return MatchStatus::No_Match;
}

} // namespace

/// Continuations keep choices alive until downstream operands and constraints succeed.
class MatchEngine {
public:
    MatchEngine(const PatternProgram& program, Budget& budget)
        : program_(program)
        , budget_(budget)
        , captures_(program.entries_.size(), nullptr) {}

    MatchStatus run(Pattern p, const Def* def) {
        if (!program_.is_current()) throw std::invalid_argument("pattern references a rebuilt World");
        program_.check(p);
        if (def) {
            auto snapshot         = std::make_shared<GraphSnapshot>();
            snapshot->world_      = &def->world();
            snapshot->generation_ = def->world().data_.storage_token;
            snapshot_             = std::move(snapshot);
        }
        return visit(p, def, 0, [&] { return complete_types(0, 0); });
    }

    MatchResult result() const { return MatchResult(&program_, captures_, snapshot_); }

    GraphMatches search(const GraphPattern&, const GraphView&, GraphMatchOptions);
    static GraphMatches find(const GraphPattern& pattern, const GraphView& graph, GraphMatchOptions options) {
        Budget budget{options.match};
        MatchEngine engine(pattern.program_, budget);
        return engine.search(pattern, graph, options);
    }

private:
    using Continue = std::function<MatchStatus()>;
    using Kind     = PatternProgram::Kind;

    struct Obligation {
        Pattern pattern;
        const Def* candidate;
    };

    bool is_exhausted(size_t depth) {
        if (budget_.steps >= budget_.options.max_steps || depth >= budget_.options.max_depth
            || budget_.active_depth >= budget_.options.max_depth)
            return true;
        ++budget_.steps;
        return false;
    }

    MatchStatus visit(Pattern p, const Def* def, size_t depth, const Continue& next) {
        if (is_exhausted(depth)) return MatchStatus::Budget_Exceeded;
        auto active = fe::Restore(budget_.active_depth, budget_.active_depth + 1);
        if (!def) return MatchStatus::No_Match;
        auto mark   = undo_.size();
        auto types  = types_.size();
        auto checks = checks_.size();
        auto status = visit_entry(p, def, depth, next);
        if (status != MatchStatus::Matched) {
            while (undo_.size() != mark) {
                captures_[undo_.back()] = nullptr;
                undo_.pop_back();
            }
            types_.erase(types_.begin() + types, types_.end());
            checks_.erase(checks_.begin() + checks, checks_.end());
        }
        return status;
    }

    MatchStatus visit_entry(Pattern p, const Def* def, size_t depth, const Continue& next) {
        const auto& entry = program_.entries_[p.index_];
        switch (entry.kind) {
            case Kind::Any: return next();
            case Kind::Exact: return def == entry.def ? next() : MatchStatus::No_Match;
            case Kind::Literal:
                return def->node() == Node::Lit && def->type() == entry.def && def->flags() == *entry.flags
                         ? next()
                         : MatchStatus::No_Match;
            case Kind::Node:
                if (def->node() != entry.node) return MatchStatus::No_Match;
                if (entry.flags && def->flags() != *entry.flags) return MatchStatus::No_Match;
                if (def->is_mutable()) return MatchStatus::Unknown;
                if (def->num_ops() != entry.operands.size()) return MatchStatus::No_Match;
                return operands(entry.operands, def, 0, depth + 1, next);
            case Kind::Binder: {
                auto lam = def->isa<Lam>();
                if (!lam || !lam->is_set()) return MatchStatus::No_Match;
                const Def* variable = lam->has_var();
                if (!variable) variable = lam;
                return visit(entry.operands[0], variable, depth + 1, [&] {
                    return visit(entry.operands[1], lam->filter(), depth + 1,
                                 [&] { return visit(entry.operands[2], lam->body(), depth + 1, next); });
                });
            }
            case Kind::Range: {
                if (def->node() != entry.node || (entry.flags && def->flags() != *entry.flags))
                    return MatchStatus::No_Match;
                if (def->is_mutable()) return MatchStatus::Unknown;
                auto prefix = entry.operands.size() - 1;
                if (def->num_ops() < prefix || def->num_ops() - prefix < entry.min
                    || def->num_ops() - prefix > entry.max)
                    return MatchStatus::No_Match;
                return range_operands(entry, def, 0, depth + 1, next);
            }
            case Kind::Capture:
                if (auto bound = captures_[p.index_]; bound && bound != def) return MatchStatus::No_Match;
                if (!captures_[p.index_]) {
                    captures_[p.index_] = def;
                    undo_.push_back(p.index_);
                }
                return visit(entry.operands[0], def, depth + 1, next);
            case Kind::All:
                return visit(entry.operands[0], def, depth + 1,
                             [&] { return visit(entry.operands[1], def, depth + 1, next); });
            case Kind::Either: {
                auto left = visit(entry.operands[0], def, depth + 1, next);
                if (left == MatchStatus::Matched) return left;
                return alternatives(left, visit(entry.operands[1], def, depth + 1, next));
            }
            case Kind::Type:
                if (auto var = def->isa<Var>();
                    var && (var->binder()->node() == Node::Arr || var->binder()->node() == Node::Pack))
                    return MatchStatus::Unknown;
                types_.push_back({entry.operands[0], def->type()});
                return next();
            case Kind::Not: checks_.push_back({p, def}); return next();
            case Kind::Where: checks_.push_back({p, def}); return visit(entry.operands[0], def, depth + 1, next);
        }
        throw std::logic_error("invalid pattern kind");
    }

    MatchStatus
    operands(const std::vector<Pattern>& patterns, const Def* def, size_t index, size_t depth, const Continue& next) {
        if (index == patterns.size()) return next();
        return visit(patterns[index], def->op(index), depth,
                     [&] { return operands(patterns, def, index + 1, depth + 1, next); });
    }

    MatchStatus range_operands(const PatternProgram::Entry& entry,
                               const Def* def,
                               size_t index,
                               size_t depth,
                               const Continue& next) {
        if (index == def->num_ops()) return next();
        auto p = entry.operands[std::min(index, entry.operands.size() - 1)];
        return visit(p, def->op(index), depth, [&] { return range_operands(entry, def, index + 1, depth + 1, next); });
    }

    MatchStatus complete_types(size_t index, size_t depth) {
        if (index == types_.size()) {
            auto status = complete_checks(depth);
            return status == MatchStatus::Matched && on_match_ ? on_match_() : status;
        }
        // visit may append more obligations and reallocate types_.
        auto obligation = types_[index];
        return visit(obligation.pattern, obligation.candidate, depth + 1,
                     [&] { return complete_types(index + 1, depth + 1); });
    }

    MatchStatus complete_checks(size_t depth) {
        bool unknown = false;
        for (auto obligation : checks_) {
            if (is_exhausted(depth)) return MatchStatus::Budget_Exceeded;
            const auto& entry = program_.entries_[obligation.pattern.index_];
            if (entry.kind == Kind::Not) {
                MatchEngine probe(program_, budget_);
                probe.captures_ = captures_;
                auto status     = probe.visit(entry.operands[0], obligation.candidate, depth + 1,
                                              [&] { return probe.complete_types(0, depth + 1); });
                if (status == MatchStatus::Matched) return MatchStatus::No_Match;
                if (status == MatchStatus::Budget_Exceeded) return status;
                unknown |= status == MatchStatus::Unknown;
            } else {
                auto args    = std::vector<const Def*>();
                bool missing = false;
                for (auto p : entry.inputs) {
                    args.push_back(captures_[p.index_]);
                    missing |= !args.back();
                }
                auto result = missing ? CheckResult::Unknown : entry.predicate(args);
                if (result == CheckResult::False) return MatchStatus::No_Match;
                unknown |= result == CheckResult::Unknown;
            }
        }
        return unknown ? MatchStatus::Unknown : MatchStatus::Matched;
    }

    const PatternProgram& program_;
    Budget& budget_;
    std::vector<const Def*> captures_;
    std::vector<size_t> undo_;
    std::vector<Obligation> types_;
    std::vector<Obligation> checks_;
    Continue on_match_;
    std::shared_ptr<const GraphSnapshot> snapshot_;
};

Anchor GraphPattern::anchor(Pattern p, AnchorDomain domain) {
    program_.check(p);
    auto result = Anchor(this, anchors_.size());
    anchors_.push_back({p, domain});
    return result;
}

void GraphPattern::check(Anchor anchor) const {
    if (anchor.graph_ != this || anchor.index_ >= anchors_.size()) throw std::invalid_argument("foreign graph anchor");
}

void GraphPattern::used_by(Anchor p, Anchor c, UseSite site) {
    check(p);
    check(c);
    relations_.push_back({p.index_, c.index_, false, std::move(site)});
}

void GraphPattern::only_used_by(Anchor p, Anchor c, UseSite site) {
    check(p);
    check(c);
    relations_.push_back({p.index_, c.index_, true, std::move(site)});
}

const Def* GraphMatch::at(Anchor anchor) const {
    if (!snapshot_->is_live()) throw std::invalid_argument("match snapshot is stale");
    if (anchor.graph_ != pattern_ || anchor.index_ >= anchors_.size())
        throw std::invalid_argument("foreign graph anchor");
    return anchors_[anchor.index_];
}

void GraphMatch::check_live() const {
    if (!snapshot_->is_live()) throw std::invalid_argument("match snapshot is stale");
}

const Def* GraphMatch::isa_capture(Pattern p) const {
    if (!snapshot_->is_live()) throw std::invalid_argument("match snapshot is stale");
    return captures_.isa_capture(p);
}

GraphMatches MatchEngine::search(const GraphPattern& pattern, const GraphView& graph, GraphMatchOptions options) {
    if (!program_.is_current()) throw std::invalid_argument("pattern references a rebuilt World");
    if (!graph.is_current()) throw std::invalid_argument("graph snapshot is stale");
    snapshot_ = graph.snapshot();
    if (pattern.anchors_.empty()) throw std::invalid_argument("graph pattern needs an anchor");
    const auto n = pattern.anchors_.size();
    if (n > options.match.max_depth || program_.entries_.size() > options.match.max_steps / n)
        return {{}, SearchStatus::Budget_Exceeded, 0, 0};
    auto capture_sets = std::vector<std::vector<bool>>(n, std::vector<bool>(program_.entries_.size()));
    bool broad_reads  = std::ranges::any_of(pattern.relations_, [](const auto& r) { return r.exclusive; });
    for (size_t i = 0; i != n; ++i) {
        auto pending = std::vector<Pattern>{pattern.anchors_[i].pattern};
        auto seen    = std::vector<bool>(program_.entries_.size());
        for (size_t j = 0; j != pending.size(); ++j) {
            if (is_exhausted(0)) return {{}, SearchStatus::Budget_Exceeded, budget_.steps, 0};
            auto p = pending[j];
            if (seen[p.index_]) continue;
            seen[p.index_]    = true;
            const auto& entry = program_.entries_[p.index_];
            if (entry.kind == Kind::Capture) capture_sets[i][p.index_] = true;
            if (entry.kind == Kind::Where) broad_reads = true; // Native graph reads are conservatively snapshot-wide.
            if (entry.kind != Kind::Not)
                for (auto child : entry.operands) {
                    if (is_exhausted(0)) return {{}, SearchStatus::Budget_Exceeded, budget_.steps, 0};
                    pending.push_back(child);
                }
        }
    }
    // NOT isolates bindings, not native graph reads. Analyze its probes separately.
    for (const auto& anchor : pattern.anchors_) {
        std::vector<Pattern> pending{anchor.pattern};
        std::vector<bool> seen(program_.entries_.size());
        for (size_t i = 0; i != pending.size(); ++i) {
            if (is_exhausted(0)) return {{}, SearchStatus::Budget_Exceeded, budget_.steps, 0};
            auto p = pending[i];
            if (seen[p.index_]) continue;
            seen[p.index_]    = true;
            const auto& entry = program_.entries_[p.index_];
            broad_reads |= entry.kind == Kind::Where;
            for (auto child : entry.operands) {
                if (is_exhausted(0)) return {{}, SearchStatus::Budget_Exceeded, budget_.steps, 0};
                pending.push_back(child);
            }
        }
    }
    auto connected = std::vector<bool>(n);
    connected[0]   = true;
    bool changed   = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i != n; ++i)
            if (connected[i])
                for (size_t j = 0; j != n; ++j)
                    if (!connected[j]) {
                        bool edge = false;
                        for (size_t k = 0; k != capture_sets[i].size(); ++k) {
                            if (is_exhausted(0)) return {{}, SearchStatus::Budget_Exceeded, budget_.steps, 0};
                            edge |= capture_sets[i][k] && capture_sets[j][k];
                        }
                        for (const auto& r : pattern.relations_)
                            edge |= (r.producer == i && r.consumer == j) || (r.producer == j && r.consumer == i);
                        if (edge) {
                            connected[j] = true;
                            changed      = true;
                        }
                    }
    }
    if (std::ranges::find(connected, false) != connected.end())
        throw std::invalid_argument("multi-root pattern must be connected by captures or data-use relations");

    GraphMatches output{{}, SearchStatus::Complete, 0, 0};
    auto candidates        = std::vector<std::vector<const Def*>>(n);
    bool prefilter_limited = false;
    // The cache belongs to this immutable snapshot, never to a mutable binding environment.
    std::map<std::pair<size_t, const Def*>, bool> filtered;
    std::function<bool(Pattern, const Def*, size_t)> may_match = [&](Pattern p, const Def* def, size_t depth) -> bool {
        if (is_exhausted(depth)) {
            prefilter_limited = true;
            return false;
        }
        if (!def) return false;
        auto key = std::pair{p.index_, def};
        if (options.mode == GraphSearchMode::Indexed)
            if (auto i = filtered.find(key); i != filtered.end()) return i->second;
        ++output.prefilter_checks;
        auto check = [&]() -> bool {
            const auto& e = program_.entries_[p.index_];
            switch (e.kind) {
                case Kind::Exact: return def == e.def;
                case Kind::Literal: return def->node() == Node::Lit && def->flags() == *e.flags && def->type() == e.def;
                case Kind::Node:
                case Kind::Range: {
                    if (def->node() != e.node || (e.flags && def->flags() != *e.flags)) return false;
                    if (def->is_mutable()) return true;
                    if (e.kind == Kind::Node && def->num_ops() != e.operands.size()) return false;
                    auto prefix = e.kind == Kind::Range ? e.operands.size() - 1 : e.operands.size();
                    if (e.kind == Kind::Range
                        && (def->num_ops() < prefix || def->num_ops() - prefix < e.min
                            || def->num_ops() - prefix > e.max))
                        return false;
                    for (size_t i = 0; i != def->num_ops(); ++i)
                        if (!may_match(e.operands[std::min(i, e.operands.size() - 1)], def->op(i), depth + 1))
                            return false;
                    return true;
                }
                case Kind::Capture:
                case Kind::Where: return may_match(e.operands[0], def, depth + 1);
                case Kind::All:
                    return may_match(e.operands[0], def, depth + 1) && may_match(e.operands[1], def, depth + 1);
                case Kind::Either:
                    return may_match(e.operands[0], def, depth + 1) || may_match(e.operands[1], def, depth + 1);
                default: return true;
            }
        };
        auto result = check();
        if (!prefilter_limited && options.mode == GraphSearchMode::Indexed) filtered.emplace(key, result);
        return result;
    };
    for (size_t i = 0; i != n; ++i) {
        auto domain = pattern.anchors_[i].domain == AnchorDomain::Operations ? graph.operations() : graph.nodes();
        for (auto def : domain) {
            if (may_match(pattern.anchors_[i].pattern, def, 0)) candidates[i].push_back(def);
            if (prefilter_limited) {
                output.status = SearchStatus::Budget_Exceeded;
                output.steps  = budget_.steps;
                return output;
            }
        }
        if (options.reverse_candidates) std::ranges::reverse(candidates[i]);
    }
    if (!options.max_results) return {{}, SearchStatus::Result_Limit, budget_.steps, 0};

    // Required raw captures occur on every successful structural branch. Type/Not captures
    // cannot imply raw ancestry; an OR only retains captures guaranteed by both alternatives.
    std::vector<std::vector<size_t>> required(program_.entries_.size());
    if (options.mode == GraphSearchMode::Indexed) {
        for (size_t i = 0; i != required.size(); ++i) {
            const auto& e = program_.entries_[i];
            if (e.kind == Kind::Not || e.kind == Kind::Type) continue;
            if (e.kind == Kind::Either) {
                auto& a = required[e.operands[0].index_];
                auto& b = required[e.operands[1].index_];
                for (auto capture : a) {
                    if (is_exhausted(0)) {
                        output.status = SearchStatus::Budget_Exceeded;
                        output.steps  = budget_.steps;
                        return output;
                    }
                    if (std::ranges::binary_search(b, capture)) required[i].push_back(capture);
                }
            } else {
                for (size_t j = 0; j != e.operands.size(); ++j) {
                    if (e.kind == Kind::Range && j + 1 == e.operands.size() && !e.min) continue;
                    for (auto capture : required[e.operands[j].index_]) {
                        if (is_exhausted(0)) {
                            output.status = SearchStatus::Budget_Exceeded;
                            output.steps  = budget_.steps;
                            return output;
                        }
                        required[i].push_back(capture);
                    }
                }
                if (e.kind == Kind::Capture) required[i].push_back(i);
                std::ranges::sort(required[i]);
                required[i].erase(std::unique(required[i].begin(), required[i].end()), required[i].end());
            }
        }
    }
    std::vector<DefMap<size_t>> ranks(n);
    for (size_t i = 0; i != n; ++i)
        for (size_t j = 0; j != candidates[i].size(); ++j)
            ranks[i].emplace(candidates[i][j], j);
    using OperandPath = std::vector<size_t>;
    std::map<std::pair<size_t, size_t>, std::optional<OperandPath>> paths;
    bool index_limited = false;
    size_t path_depth  = 0;
    std::function<std::optional<OperandPath>(Pattern, size_t)> path_to
        = [&](Pattern p, size_t capture) -> std::optional<OperandPath> {
        if (is_exhausted(path_depth)) {
            index_limited = true;
            return {};
        }
        auto active = fe::Restore(path_depth, path_depth + 1);
        auto key    = std::pair{p.index_, capture};
        if (auto i = paths.find(key); i != paths.end()) return i->second;
        const auto& e = program_.entries_[p.index_];
        std::optional<OperandPath> result;
        if (e.kind == Kind::Capture && p.index_ == capture)
            result = OperandPath{};
        else if (e.kind == Kind::Either) {
            auto a = path_to(e.operands[0], capture), b = path_to(e.operands[1], capture);
            if (a && b && *a == *b) result = std::move(a);
        } else if (e.kind != Kind::Type && e.kind != Kind::Not) {
            for (size_t i = 0; i != e.operands.size(); ++i) {
                if (e.kind == Kind::Range && i + 1 == e.operands.size() && !e.min) continue;
                if (!std::ranges::binary_search(required[e.operands[i].index_], capture)) continue;
                auto child = path_to(e.operands[i], capture);
                if (!child) continue;
                if (e.kind == Kind::Node || e.kind == Kind::Range) child->insert(child->begin(), i);
                result = std::move(child);
                break;
            }
        }
        paths.emplace(key, result);
        return result;
    };
    std::map<std::pair<const Def*, OperandPath>, std::shared_ptr<DefSet>> ancestors;
    auto index_work = [&] {
        if (is_exhausted(0)) {
            index_limited = true;
            return false;
        }
        ++output.index_steps;
        return true;
    };
    std::vector<const Def*> opaque;
    if (options.mode == GraphSearchMode::Indexed)
        for (auto def : graph.nodes()) {
            if (!index_work()) {
                output.status = SearchStatus::Budget_Exceeded;
                output.steps  = budget_.steps;
                return output;
            }
            if (def->is_mutable() || def->node() == Node::Lam || def->node() == Node::Pi) opaque.push_back(def);
        }
    auto include_opaque = [&](DefSet& set) {
        for (auto def : opaque) {
            if (!index_work()) return false;
            set.insert(def);
        }
        return true;
    };
    auto parents = [&](const Def* value, const std::optional<OperandPath>& path) -> std::shared_ptr<DefSet> {
        auto key = std::pair{value, path ? *path : OperandPath{size_t(-1)}};
        if (auto i = ancestors.find(key); i != ancestors.end()) return i->second;
        auto result = std::make_shared<DefSet>();
        result->insert(value);
        if (path) {
            for (auto index : *path | std::views::reverse) {
                auto next = std::make_shared<DefSet>();
                for (auto def : *result) {
                    if (!index_work()) return {};
                    for (auto edge : graph.uses(def)) {
                        if (!index_work()) return {};
                        if (edge.kind != EdgeKind::Type && edge.index == index) next->insert(edge.owner);
                    }
                }
                // Missing operands at a graph boundary cannot disprove this binding path.
                if (!include_opaque(*next)) return {};
                result = std::move(next);
            }
        } else {
            if (!include_opaque(*result)) return {};
            std::vector<const Def*> pending(result->begin(), result->end());
            for (size_t i = 0; i != pending.size(); ++i) {
                auto def = pending[i];
                if (!index_work()) return {};
                for (auto edge : graph.uses(def)) {
                    if (!index_work()) return {};
                    if (edge.kind != EdgeKind::Type && result->insert(edge.owner).second) pending.push_back(edge.owner);
                }
            }
        }
        ancestors.emplace(std::move(key), result);
        return result;
    };
    auto narrow = [&](size_t anchor) -> std::optional<std::vector<const Def*>> {
        if (options.mode == GraphSearchMode::Exhaustive) return {};
        std::vector<std::shared_ptr<DefSet>> constraints;
        for (auto index : required[pattern.anchors_[anchor].pattern.index_])
            if (auto value = captures_[index]) {
                auto set = parents(value, path_to(pattern.anchors_[anchor].pattern, index));
                if (!set) return std::vector<const Def*>();
                constraints.push_back(std::move(set));
            }
        if (constraints.empty()) return {};
        auto smallest = *std::ranges::min_element(constraints, {}, [](const auto& set) { return set->size(); });
        std::vector<const Def*> result;
        for (auto def : *smallest) {
            if (is_exhausted(0)) {
                index_limited = true;
                break;
            }
            ++output.index_steps;
            if (!ranks[anchor].contains(def)) continue;
            bool allowed = true;
            for (const auto& constraint : constraints) {
                if (is_exhausted(0)) {
                    index_limited = true;
                    return std::vector<const Def*>();
                }
                ++output.index_steps;
                if (!constraint->contains(def)) {
                    allowed = false;
                    break;
                }
            }
            if (allowed) result.push_back(def);
        }
        std::ranges::sort(result, {}, [&](auto def) { return ranks[anchor].at(def); });
        return result;
    };
    auto assigned  = std::vector<const Def*>(n, nullptr);
    bool limited   = false;
    auto relations = [&](bool final) -> MatchStatus {
        bool unknown = false;
        for (const auto& r : pattern.relations_) {
            auto p = assigned[r.producer], c = assigned[r.consumer];
            if (!p || !c) continue;
            auto used = r.exclusive ? graph.data_only_used_by(p, c) : graph.data_used_by(p, c);
            if (used == CheckResult::False) return MatchStatus::No_Match;
            bool site = false;
            for (const auto& edge : graph.data_uses(p))
                if (edge.consumer == c && (!r.site.group || edge.group == *r.site.group)
                    && (!r.site.path || edge.path == *r.site.path)
                    && (!r.site.result_path || edge.result_path == *r.site.result_path))
                    site = true;
            if (!site && graph.isa_operation(c) && graph.isa_operation(c)->complete) return MatchStatus::No_Match;
            unknown |= used == CheckResult::Unknown || !site;
        }
        return final && unknown ? MatchStatus::Unknown : MatchStatus::Matched;
    };
    std::set<std::vector<const Def*>> emitted, unresolved;
    std::shared_ptr<const std::vector<const Def*>> broad;
    on_match_ = [&] {
        auto status = relations(true);
        if (status != MatchStatus::Matched) return status;
        if (emitted.contains(assigned)) return MatchStatus::No_Match;
        DefSet matched;
        for (auto def : assigned)
            matched.insert(def);
        for (auto def : captures_)
            if (def && graph.is_contained(def)) matched.insert(def);
        auto nodes = std::vector<const Def*>(matched.begin(), matched.end());
        std::ranges::sort(nodes, {}, [](auto def) { return def->gid(); });
        std::shared_ptr<const std::vector<const Def*>> reads;
        if (broad_reads) {
            if (!broad)
                broad = std::make_shared<const std::vector<const Def*>>(graph.nodes().begin(), graph.nodes().end());
            reads = broad;
        } else {
            DefSet read;
            auto pending = assigned;
            auto values  = std::make_shared<std::vector<const Def*>>();
            for (size_t i = 0; i != pending.size(); ++i) {
                if (is_exhausted(0)) return MatchStatus::Budget_Exceeded;
                auto def = pending[i];
                if (!read.insert(def).second) continue;
                if (graph.is_contained(def)) values->push_back(def);
                if (def->is_mutable() || def->node() == Node::Var || def->node() == Node::Axm
                    || def->node() == Node::Lam || def->node() == Node::Pi)
                    continue;
                for (auto op : def->ops())
                    if (op) pending.push_back(op);
            }
            reads = values;
        }
        auto match = GraphMatch(&pattern, graph, assigned, result(), std::move(reads), std::move(nodes));
        if (options.accept && !options.accept(match)) return MatchStatus::No_Match;
        emitted.insert(assigned);
        unresolved.erase(assigned);
        output.matches.push_back(std::move(match));
        limited = output.matches.size() >= options.max_results;
        return MatchStatus::Matched;
    };
    std::function<MatchStatus(size_t)> enumerate = [&](size_t count) -> MatchStatus {
        if (count == n) return complete_types(0, count);
        size_t next = n;
        std::optional<std::vector<const Def*>> narrowed;
        for (size_t i = 0; i != n; ++i) {
            if (assigned[i]) continue;
            auto values = narrow(i);
            if (index_limited) return MatchStatus::Budget_Exceeded;
            auto size = values ? values->size() : candidates[i].size();
            auto best = next == n ? size_t(-1) : narrowed ? narrowed->size() : candidates[next].size();
            if (next == n || size < best) {
                next     = i;
                narrowed = std::move(values);
            }
        }
        auto status = MatchStatus::No_Match;
        for (auto def : narrowed ? *narrowed : candidates[next]) {
            if (output.candidates >= options.max_candidates) return MatchStatus::Budget_Exceeded;
            ++output.candidates;
            if (pattern.distinct_ && std::ranges::find(assigned, def) != assigned.end()) continue;
            assigned[next] = def;
            if (count + 1 == n && emitted.contains(assigned)) {
                assigned[next] = nullptr;
                continue;
            }
            if (relations(false) != MatchStatus::No_Match) {
                auto mark = undo_.size(), types = types_.size(), checks = checks_.size();
                auto attempt = visit(pattern.anchors_[next].pattern, def, count, [&] { return enumerate(count + 1); });
                // A later binding alternative can prove this same tuple. Keep its Unknown
                // separate from partial assignments, which remain conservatively unknown.
                if (attempt == MatchStatus::Unknown && count + 1 == n) {
                    if (!emitted.contains(assigned)) unresolved.insert(assigned);
                    attempt = MatchStatus::No_Match;
                }
                while (undo_.size() != mark) {
                    captures_[undo_.back()] = nullptr;
                    undo_.pop_back();
                }
                types_.erase(types_.begin() + types, types_.end());
                checks_.erase(checks_.begin() + checks, checks_.end());
                status = alternatives(status, attempt == MatchStatus::Matched ? MatchStatus::No_Match : attempt);
                if (limited || status == MatchStatus::Budget_Exceeded) {
                    assigned[next] = nullptr;
                    return status;
                }
            }
            assigned[next] = nullptr;
        }
        // Continue earlier alternatives, but a successful complete tuple stops its own OR.
        return status == MatchStatus::Matched ? MatchStatus::No_Match : status;
    };
    auto status   = enumerate(0);
    output.status = limited                                               ? SearchStatus::Result_Limit
                  : status == MatchStatus::Budget_Exceeded                ? SearchStatus::Budget_Exceeded
                  : status == MatchStatus::Unknown || !unresolved.empty() ? SearchStatus::Unknown
                                                                          : SearchStatus::Complete;
    output.steps  = budget_.steps;
    if (!graph.is_current()) throw std::invalid_argument("graph snapshot changed during matching");
    return output;
}

GraphMatches find_matches(const GraphPattern& pattern, const GraphView& graph, GraphMatchOptions options) {
    // GraphPattern exposes its program only to the shared matcher engine.
    return MatchEngine::find(pattern, graph, options);
}

MatchOutcome match(const PatternProgram& program, Pattern p, const Def* def, MatchOptions options) {
    Budget budget{options};
    MatchEngine engine(program, budget);
    auto status = engine.run(p, def);
    return {status, status == MatchStatus::Matched ? std::optional(engine.result()) : std::nullopt, budget.steps};
}

} // namespace mim::pattern
