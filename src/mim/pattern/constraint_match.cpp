#include <numeric>
#include <stdexcept>

#include "mim/world.h"

#include "mim/pattern/graph_match.h"

namespace mim::pattern {

// Portmatching-inspired single-pattern backend: compile local tests and binding dependencies,
// evaluate root relations once, then join them through indexed shared capture columns. It is
// deliberately separate from the continuation interpreter, which remains the semantic fallback.
class ConstraintMatcher {
    using Kind                      = PatternProgram::Kind;
    static constexpr size_t missing = size_t(-1);
    enum class Code { Check, Operand, Type, Capture };
    struct Instruction {
        Code code;
        size_t slot, arg;
        size_t index = 0;
    };
    struct Root {
        std::vector<Instruction> code;
        std::vector<size_t> columns;
        size_t slots = 1, depth = 0;
    };
    struct Table {
        std::vector<const Def*> roots, values;
        std::vector<DefMap<std::vector<size_t>>> indexes;
    };
    struct Exhausted {};
    struct Unsupported {};

public:
    explicit ConstraintMatcher(const GraphPattern& pattern)
        : pattern_(pattern)
        , program_(pattern.program_)
        , entries_(program_.entries_.size())
        , relations_(pattern.relations_.size())
        , anchor_count_(pattern.anchors_.size())
        , distinct_(pattern.distinct_) {
        if (!program_.is_current()) throw std::invalid_argument("pattern references a rebuilt World");
        if (pattern.anchors_.empty()) throw std::invalid_argument("graph pattern needs an anchor");
        if (anchor_count_ > 256 || entries_ > 100000) {
            supported_ = false;
            return;
        }
        capture_ids_.assign(entries_, missing);
        for (size_t i = 0; i < entries_; ++i)
            if (program_.entries_[i].kind == Kind::Capture) {
                capture_ids_[i] = captures_.size();
                captures_.push_back(i);
            }
        try {
            for (const auto& anchor : pattern.anchors_) {
                roots_.emplace_back();
                std::vector<bool> expanded(entries_);
                compile(anchor.pattern, 0, roots_.back(), expanded, 0);
                auto& columns = roots_.back().columns;
                std::ranges::sort(columns);
                columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
            }
        } catch (Unsupported) { supported_ = false; }
        if (!supported_) return;
        frequency_.resize(captures_.size());
        for (const auto& root : roots_)
            for (auto column : root.columns)
                ++frequency_[column];
        // A shared capture is a join key; semantic use relations may also connect roots.
        std::vector<bool> connected(roots_.size());
        connected[0] = true;
        std::vector<size_t> pending{0};
        for (size_t i = 0; i < pending.size(); ++i) {
            auto a = pending[i];
            for (size_t b = 0; b < roots_.size(); ++b) {
                if (!connected[b]) {
                    bool edge = false;
                    for (auto column : roots_[a].columns)
                        edge |= std::ranges::binary_search(roots_[b].columns, column);
                    for (const auto& relation : pattern.relations_)
                        edge |= (relation.producer == a && relation.consumer == b)
                             || (relation.producer == b && relation.consumer == a);
                    if (edge) {
                        connected[b] = true;
                        pending.push_back(b);
                    }
                }
            }
        }
        if (std::ranges::find(connected, false) != connected.end())
            throw std::invalid_argument("multi-root pattern must be connected by captures or data-use relations");
    }

    bool is_compiled() const { return supported_; }

    GraphMatches find(const GraphView& graph, GraphMatchOptions options) const {
        if (!program_.is_current() || entries_ != program_.entries_.size() || anchor_count_ != pattern_.anchors_.size()
            || relations_ != pattern_.relations_.size() || distinct_ != pattern_.distinct_)
            throw std::invalid_argument("compiled graph pattern is stale");
        if (!graph.is_current()) throw std::invalid_argument("graph snapshot is stale");
        if (!supported_) return fallback(graph, options, 0);
        GraphMatches output{{}, SearchStatus::Complete, 0, 0};
        output.used_compiled = true;
        if (!options.max_results) {
            output.status = SearchStatus::Result_Limit;
            return output;
        }
        auto tick = [&] {
            if (output.steps >= options.match.max_steps) throw Exhausted{};
            ++output.steps;
        };
        try {
            if (roots_.size() >= options.match.max_depth) throw Exhausted{};
            std::vector<Table> tables(roots_.size());
            std::vector<const Def*> local(captures_.size());
            for (size_t r = 0; r < roots_.size(); ++r) {
                const auto& root = roots_[r];
                if (root.depth >= options.match.max_depth) throw Exhausted{};
                auto& table = tables[r];
                auto domain
                    = pattern_.anchors_[r].domain == AnchorDomain::Operations ? graph.operations() : graph.nodes();
                std::vector<const Def*> slots(root.slots);
                // No per-candidate allocations: values are stored in a flat row-major array.
                for (auto candidate : domain) {
                    tick();
                    std::ranges::fill(local, nullptr);
                    slots[0]      = candidate;
                    bool accepted = true;
                    for (auto op : root.code) {
                        tick();
                        auto value = slots[op.slot];
                        if (op.code == Code::Capture) {
                            if (local[op.arg] && local[op.arg] != value) {
                                accepted = false;
                                break;
                            }
                            local[op.arg] = value;
                        } else if (op.code == Code::Operand) {
                            // Its preceding Node check proved arity/immutability.
                            if (!(slots[op.arg] = value->op(op.index))) {
                                accepted = false;
                                break;
                            }
                        } else if (op.code == Code::Type) {
                            if (auto var = value->isa<Var>();
                                var && (var->binder()->node() == Node::Arr || var->binder()->node() == Node::Pack))
                                throw Unsupported{};
                            if (!(slots[op.arg] = value->type())) {
                                accepted = false;
                                break;
                            }
                        } else {
                            ++output.prefilter_checks;
                            const auto& check = program_.entries_[op.arg];
                            if (check.kind == Kind::Exact)
                                accepted = value == check.def;
                            else if (check.kind == Kind::Literal)
                                accepted = value->node() == Node::Lit && value->flags() == *check.flags
                                        && value->type() == check.def;
                            else {
                                accepted
                                    = value->node() == check.node && (!check.flags || value->flags() == *check.flags);
                                if (accepted && value->is_mutable()) throw Unsupported{};
                                accepted = accepted && value->num_ops() == check.operands.size();
                            }
                            if (!accepted) break;
                        }
                    }
                    if (!accepted) continue;
                    table.roots.push_back(candidate);
                    for (auto column : root.columns)
                        table.values.push_back(local[column]);
                }
                if (table.roots.empty()) return output;
            }
            for (size_t r = 0; r < roots_.size(); ++r) {
                auto& table         = tables[r];
                const auto& columns = roots_[r].columns;
                table.indexes.resize(columns.size());
                for (size_t col = 0; col < columns.size(); ++col)
                    if (frequency_[columns[col]] > 1)
                        for (size_t row = 0; row < table.roots.size(); ++row) {
                            tick();
                            ++output.index_steps;
                            table.indexes[col][table.values[row * columns.size() + col]].push_back(row);
                        }
            }
            // Choose one join order per search. Runtime only selects a bound column's bucket;
            // it never builds ancestor sets, intersects them or sorts a fresh candidate vector.
            std::vector<size_t> order;
            std::vector<bool> planned(roots_.size()), known(captures_.size());
            while (order.size() < roots_.size()) {
                size_t best = missing, best_links = 0;
                for (size_t r = 0; r < roots_.size(); ++r) {
                    tick();
                    if (planned[r]) continue;
                    size_t links = 0;
                    for (auto col : roots_[r].columns)
                        links += known[col];
                    if (best == missing || links > best_links
                        || (links == best_links && tables[r].roots.size() < tables[best].roots.size())) {
                        best       = r;
                        best_links = links;
                    }
                }
                order.push_back(best);
                planned[best] = true;
                for (auto col : roots_[best].columns)
                    known[col] = true;
            }
            std::vector<const Def*> bound(captures_.size()), assigned(roots_.size());
            std::vector<size_t> trail;
            std::shared_ptr<const std::vector<const Def*>> broad;
            bool unknown = false, limited = false;
            auto relations = [&](bool final) {
                bool uncertain = false;
                for (const auto& relation : pattern_.relations_) {
                    tick();
                    auto p = assigned[relation.producer], c = assigned[relation.consumer];
                    if (!p || !c) continue;
                    auto used = relation.exclusive ? graph.data_only_used_by(p, c) : graph.data_used_by(p, c);
                    if (used == CheckResult::False) return CheckResult::False;
                    bool site = false;
                    for (const auto& edge : graph.data_uses(p)) {
                        tick();
                        if (edge.consumer == c && (!relation.site.group || edge.group == *relation.site.group)
                            && (!relation.site.path || edge.path == *relation.site.path)
                            && (!relation.site.result_path || edge.result_path == *relation.site.result_path))
                            site = true;
                    }
                    if (!site && graph.isa_operation(c) && graph.isa_operation(c)->complete) return CheckResult::False;
                    uncertain |= used == CheckResult::Unknown || !site;
                }
                return final && uncertain ? CheckResult::Unknown : CheckResult::True;
            };
            auto emit = [&] {
                auto state = relations(true);
                if (state == CheckResult::False) return;
                bool uncertain = state == CheckResult::Unknown;
                for (auto guard : guards_) {
                    tick();
                    std::vector<const Def*> args;
                    bool missing = false;
                    for (auto input : program_.entries_[guard].inputs) {
                        auto value = bound[capture_ids_[input.index_]];
                        args.push_back(value);
                        missing |= !value;
                    }
                    auto result = missing ? CheckResult::Unknown : program_.entries_[guard].predicate(args);
                    if (result == CheckResult::False) return;
                    uncertain |= result == CheckResult::Unknown;
                }
                if (uncertain) {
                    unknown = true;
                    return;
                }
                std::vector<const Def*> captures(entries_);
                for (size_t i = 0; i < bound.size(); ++i)
                    captures[captures_[i]] = bound[i];
                DefSet matched;
                for (auto value : assigned)
                    matched.insert(value);
                for (auto value : bound)
                    if (value && graph.is_contained(value)) matched.insert(value);
                std::vector<const Def*> nodes(matched.begin(), matched.end());
                std::ranges::sort(nodes, {}, [](auto value) { return value->gid(); });
                std::shared_ptr<const std::vector<const Def*>> reads;
                if (broad_reads_) {
                    if (!broad)
                        broad = std::make_shared<const std::vector<const Def*>>(graph.nodes().begin(),
                                                                                graph.nodes().end());
                    reads = broad;
                } else {
                    auto values = std::make_shared<std::vector<const Def*>>();
                    DefSet seen;
                    auto pending = assigned;
                    for (auto value : bound)
                        if (value) pending.push_back(value);
                    for (size_t i = 0; i < pending.size(); ++i) {
                        tick();
                        auto value = pending[i];
                        if (!seen.insert(value).second) continue;
                        if (graph.is_contained(value)) values->push_back(value);
                        if (value->is_mutable() || value->node() == Node::Var || value->node() == Node::Axm
                            || value->node() == Node::Lam || value->node() == Node::Pi)
                            continue;
                        for (auto op : value->ops())
                            if (op) pending.push_back(op);
                    }
                    reads = std::move(values);
                }
                output.matches.push_back(GraphMatch(&pattern_, graph, assigned,
                                                    MatchResult(&program_, std::move(captures), graph.snapshot()),
                                                    std::move(reads), std::move(nodes)));
                limited = output.matches.size() >= options.max_results;
            };
            auto join = [&](auto&& self, size_t depth) -> void {
                if (depth == order.size()) {
                    emit();
                    return;
                }
                if (depth >= options.match.max_depth) throw Exhausted{};
                auto r                            = order[depth];
                const auto& table                 = tables[r];
                const auto& columns               = roots_[r].columns;
                const std::vector<size_t>* bucket = nullptr;
                for (size_t col = 0; col < columns.size(); ++col)
                    if (auto value = bound[columns[col]]) {
                        tick();
                        auto found = table.indexes[col].find(value);
                        if (found == table.indexes[col].end()) return;
                        if (!bucket || found->second.size() < bucket->size()) bucket = &found->second;
                    }
                auto try_row = [&](size_t row) {
                    tick();
                    if (output.candidates >= options.max_candidates) throw Exhausted{};
                    ++output.candidates;
                    auto value = table.roots[row];
                    if (distinct_ && std::ranges::find(assigned, value) != assigned.end()) return;
                    auto mark     = trail.size();
                    bool accepted = true;
                    for (size_t col = 0; col < columns.size(); ++col) {
                        tick();
                        auto id      = columns[col];
                        auto current = table.values[row * columns.size() + col];
                        if (bound[id] && bound[id] != current) {
                            accepted = false;
                            break;
                        }
                        if (!bound[id]) {
                            bound[id] = current;
                            trail.push_back(id);
                        }
                    }
                    assigned[r] = value;
                    if (accepted && relations(false) != CheckResult::False) self(self, depth + 1);
                    assigned[r] = nullptr;
                    while (trail.size() > mark) {
                        bound[trail.back()] = nullptr;
                        trail.pop_back();
                    }
                };
                auto count = bucket ? bucket->size() : table.roots.size();
                for (size_t i = 0; i < count && !limited; ++i) {
                    auto pos = options.reverse_candidates ? count - i - 1 : i;
                    try_row(bucket ? (*bucket)[pos] : pos);
                }
            };
            join(join, 0);
            output.status = limited ? SearchStatus::Result_Limit
                          : unknown ? SearchStatus::Unknown
                                    : SearchStatus::Complete;
        } catch (Exhausted) { output.status = SearchStatus::Budget_Exceeded; } catch (Unsupported) {
            return fallback(graph, options, output.steps);
        }
        if (!graph.is_current()) throw std::invalid_argument("graph snapshot changed during matching");
        return output;
    }

private:
    void compile(Pattern p, size_t slot, Root& root, std::vector<bool>& expanded, size_t depth) {
        if (++compile_steps_ > 100000 || depth >= 256 || root.code.size() >= 100000 || root.slots >= 32768)
            throw Unsupported{};
        root.depth        = std::max(root.depth, depth);
        const auto& entry = program_.entries_[p.index_];
        switch (entry.kind) {
            case Kind::Any: break;
            case Kind::Exact:
            case Kind::Literal: root.code.push_back({Code::Check, slot, p.index_}); break;
            case Kind::Node:
                root.code.push_back({Code::Check, slot, p.index_});
                for (size_t i = 0; i < entry.operands.size(); ++i) {
                    auto child = root.slots++;
                    root.code.push_back({Code::Operand, slot, child, i});
                    compile(entry.operands[i], child, root, expanded, depth + 1);
                }
                break;
            case Kind::Capture:
                root.code.push_back({Code::Capture, slot, capture_ids_[p.index_]});
                root.columns.push_back(capture_ids_[p.index_]);
                if (!expanded[p.index_]) {
                    expanded[p.index_] = true;
                    compile(entry.operands[0], slot, root, expanded, depth + 1);
                }
                break;
            case Kind::All:
                for (auto child : entry.operands)
                    compile(child, slot, root, expanded, depth + 1);
                break;
            case Kind::Type: {
                auto child = root.slots++;
                root.code.push_back({Code::Type, slot, child});
                compile(entry.operands[0], child, root, expanded, depth + 1);
                break;
            }
            case Kind::Where:
                compile(entry.operands[0], slot, root, expanded, depth + 1);
                if (std::ranges::find(guards_, p.index_) == guards_.end()) guards_.push_back(p.index_);
                broad_reads_ = true;
                break;
            default: throw Unsupported{};
        }
        for (const auto& relation : pattern_.relations_)
            broad_reads_ |= relation.exclusive;
    }

    GraphMatches fallback(const GraphView& graph, GraphMatchOptions options, size_t spent) const {
        options.mode = GraphSearchMode::Indexed;
        options.match.max_steps -= std::min(spent, options.match.max_steps);
        auto result = find_matches(pattern_, graph, options);
        result.steps += spent;
        return result;
    }

    const GraphPattern& pattern_;
    const PatternProgram& program_;
    size_t entries_, relations_, anchor_count_;
    size_t compile_steps_ = 0;
    bool distinct_, supported_ = true, broad_reads_ = false;
    std::vector<size_t> capture_ids_, captures_, frequency_, guards_;
    std::vector<Root> roots_;
};

CompiledGraphPattern::CompiledGraphPattern(const GraphPattern& pattern)
    : matcher_(std::make_unique<ConstraintMatcher>(pattern)) {}
CompiledGraphPattern::~CompiledGraphPattern() = default;
bool CompiledGraphPattern::is_compiled() const { return matcher_->is_compiled(); }
GraphMatches CompiledGraphPattern::find(const GraphView& graph, GraphMatchOptions options) const {
    return matcher_->find(graph, options);
}

} // namespace mim::pattern
