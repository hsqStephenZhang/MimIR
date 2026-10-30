#include "mim/pattern/graph.h"

#include <algorithm>
#include <stdexcept>

#include "mim/axm.h"
#include "mim/lam.h"
#include "mim/tuple.h"
#include "mim/world.h"

namespace mim::pattern {

bool GraphSnapshot::is_live() const { return !published_ && (!world_ || world_->data_.storage_token == generation_); }

GraphView::GraphView(Defs roots, UseCoverage coverage)
    : GraphView(roots, GraphOptions{.coverage = coverage}) {}

GraphView::GraphView(Defs roots, GraphOptions options)
    : coverage_(options.coverage)
    , scope_(options.scope) {
    if (options.operators) operators_ = *options.operators;
    const World* world = nullptr;
    for (auto root : roots) {
        if (!root) throw std::invalid_argument("graph root must not be null");
        if (world && world != &root->world()) throw std::invalid_argument("graph roots belong to different Worlds");
        world = &root->world();
        if (outputs_.insert(root).second) roots_.push_back(root);
    }
    world_ = const_cast<World*>(world);
    if (scope_) {
        if (!scope_->is_mutable() || !scope_->is_set() || !scope_->is_closed()
            || (world_ && &scope_->world() != world_))
            throw std::invalid_argument("graph scope must be a fully defined, closed mutable Lam of the same World");
        world_       = &scope_->world();
        auto pending = std::vector<const Def*>{scope_->body()};
        DefSet body;
        for (size_t i = 0; i != pending.size(); ++i) {
            auto def = pending[i];
            if (!body.insert(def).second || def->is_mutable() || def->node() == Node::Var || def->node() == Node::Lam
                || def->node() == Node::Pi || def->node() == Node::Axm)
                continue;
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
        for (auto root : roots_)
            if (!body.contains(root)) throw std::invalid_argument("graph root is outside its scope body");
    }

    auto enqueue = [&](const Def* def) {
        if (indices_.emplace(def, nodes_.size()).second) nodes_.push_back(def);
    };
    for (auto root : roots_)
        enqueue(root);
    for (size_t i = 0; i != nodes_.size(); ++i) {
        auto def = nodes_[i];
        // A sequence Var's type may need World::type_indices, which can construct IR.
        if (def->node() != Node::Var)
            if (auto type = def->type()) uses_[type].push_back({def, 0, EdgeKind::Type});
        if (def->is_mutable() || def->node() == Node::Lam || def->node() == Node::Pi || def->node() == Node::Var
            || def->node() == Node::Axm)
            continue;
        for (size_t j = 0; j != def->num_ops(); ++j) {
            if (auto op = def->op(j)) {
                auto kind = def->node() == Node::App && j == 0 ? EdgeKind::Callee : EdgeKind::Operand;
                uses_[op].push_back({def, j, kind});
                enqueue(op);
            }
        }
    }
    // Track mutable dependencies as well as the scope. Captured type/binder changes invalidate a plan.
    snapshot_->world_ = world_;
    if (world_) snapshot_->generation_ = world_->data_.storage_token;
    auto pending = nodes_;
    if (scope_) pending.push_back(scope_);
    DefSet seen;
    auto record = [&](bool unknown_uses) {
        for (size_t i = 0; i != pending.size(); ++i) {
            auto def = pending[i];
            if (!seen.insert(def).second) continue;
            if (auto var = def->isa<Var>()) {
                pending.push_back(var->binder());
                continue;
            }
            if (def->is_mutable())
                mutables_.push_back({
                    def,
                    def->type(),
                    {def->ops().begin(), def->ops().end()}
                });
            if (unknown_uses && def != scope_ && !indices_.contains(def))
                for (auto op : def->ops())
                    if (op && indices_.contains(op)) unknown_uses_.insert(op);
            for (auto dep : def->deps())
                if (dep) pending.push_back(dep);
        }
    };
    record(true);
    pending.clear();
    if (world_)
        for (auto [name, ext] : world_->externals()) {
            externals_.emplace_back(name, ext);
            pending.push_back(ext);
        }
    record(false);
    build_semantics(options.max_inputs);
}

std::span<const UseEdge> GraphView::uses(const Def* def) const {
    if (!snapshot_->is_live()) throw std::invalid_argument("graph snapshot is stale");
    if (!def) return {};
    auto i = uses_.find(def);
    return i == uses_.end() ? std::span<const UseEdge>() : std::span<const UseEdge>(i->second);
}

size_t GraphView::use_count(const Def* def) const {
    return std::ranges::count_if(uses(def), [](const UseEdge& edge) { return edge.kind == EdgeKind::Operand; });
}

size_t GraphView::consumer_count(const Def* def) const {
    auto owners = DefSet();
    for (auto edge : uses(def))
        if (edge.kind == EdgeKind::Operand) owners.insert(edge.owner);
    return owners.size();
}

bool GraphView::is_used_by(const Def* producer, const Def* owner) const {
    return std::ranges::any_of(
        uses(producer), [owner](const UseEdge& edge) { return edge.kind == EdgeKind::Operand && edge.owner == owner; });
}

CheckResult GraphView::only_used_by(const Def* producer, const Def* owner) const {
    if (!is_contained(producer) || !is_contained(owner)) return CheckResult::Unknown;
    if (outputs_.contains(producer)) return CheckResult::False;
    if (!is_used_by(producer, owner)) return CheckResult::False;
    for (auto edge : uses(producer))
        if (edge.kind == EdgeKind::Operand && edge.owner != owner) return CheckResult::False;
    return coverage_ == UseCoverage::Closed ? CheckResult::True : CheckResult::Unknown;
}

World& GraphView::world() const {
    if (!world_) throw std::invalid_argument("empty graph has no World");
    return *world_;
}

bool GraphView::is_current() const {
    if (!snapshot_->is_live()) return false;
    if (world_ && world_->externals().size() != externals_.size()) return false;
    for (auto [name, ext] : externals_)
        if (world_->externals()[name] != ext) return false;
    for (const auto& state : mutables_)
        if (state.def->type() != state.type || state.def->num_ops() != state.ops.size()
            || !std::ranges::equal(state.def->ops(), state.ops))
            return false;
    return true;
}

void GraphView::build_semantics(size_t limit) {
    if (!operators_) return;
    for (auto def : nodes_)
        if (auto view = operators_->decode(def, limit)) {
            if (view->inputs.size() > limit) throw std::invalid_argument("operator decoder exceeded its input limit");
            operations_.push_back(def);
            size_t indexed = 0;
            for (const auto& input : view->inputs) {
                if (!input.value || !is_contained(input.value))
                    throw std::invalid_argument("operator input is outside the raw graph");
                const Def* producer = input.value;
                auto result_path    = std::vector<size_t>();
                auto index_use      = [&] {
                    auto cost = result_path.size() + 1;
                    if (cost > limit - indexed) {
                        view->complete = false;
                        return false;
                    }
                    indexed += cost;
                    auto path = result_path;
                    std::ranges::reverse(path);
                    data_uses_[producer].push_back(
                        {input.value, producer, def, input.group, input.path, std::move(path)});
                    return true;
                };
                if (!index_use()) break;
                while (auto extract = producer->isa<Extract>()) {
                    auto index = Lit::isa(extract->index());
                    if (!index) {
                        view->complete = false;
                        break;
                    }
                    result_path.push_back(*index);
                    producer = extract->tuple();
                    if (!index_use()) break;
                }
            }
            op_views_.emplace(def, std::move(*view));
            for (auto cur = def->as<App>()->callee(); cur->isa<App>(); cur = cur->as<App>()->callee())
                curry_prefixes_.insert(cur);
        }
}

const OpView* GraphView::isa_operation(const Def* def) const {
    if (!snapshot_->is_live()) throw std::invalid_argument("graph snapshot is stale");
    if (!def) return nullptr;
    auto i = op_views_.find(def);
    return i == op_views_.end() ? nullptr : &i->second;
}

bool GraphView::is_pure(flags_t axm) const { return operators_ && operators_->is_pure(axm); }

bool GraphView::is_pure(const Def* def) const {
    if (!isa_operation(def)) return false;
    return operators_->is_pure(def);
}

bool GraphView::has_purity(flags_t axm) const { return operators_ && operators_->has_purity(axm); }

bool GraphView::is_pure_call(flags_t axm, std::span<const Def* const> args) const {
    return operators_ && operators_->is_pure_call(axm, args);
}

std::span<const DataEdge> GraphView::data_uses(const Def* def) const {
    if (!snapshot_->is_live()) throw std::invalid_argument("graph snapshot is stale");
    if (!def) return {};
    auto i = data_uses_.find(def);
    return i == data_uses_.end() ? std::span<const DataEdge>() : std::span<const DataEdge>(i->second);
}

CheckResult GraphView::data_used_by(const Def* producer, const Def* consumer) const {
    if (!is_contained(producer) || !isa_operation(consumer)) return CheckResult::Unknown;
    for (const auto& edge : data_uses(producer))
        if (edge.consumer == consumer) return CheckResult::True;
    return isa_operation(consumer)->complete ? CheckResult::False : CheckResult::Unknown;
}

CheckResult GraphView::data_boundary(const Def* producer) const {
    auto pending = std::vector<const Def*>{producer};
    DefSet seen;
    bool unknown = false;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        unknown |= unknown_uses_.contains(def);
        if (outputs_.contains(def)) return CheckResult::False;
        for (auto edge : uses(def)) {
            if (edge.kind == EdgeKind::Type) continue;
            if (auto op = isa_operation(edge.owner)) {
                if (!op->complete || data_used_by(producer, edge.owner) != CheckResult::True
                    || (edge.kind == EdgeKind::Callee && !curry_prefixes_.contains(def)))
                    unknown = true;
            } else if ((edge.owner->node() == Node::Tuple || edge.owner->node() == Node::Pack)
                       || (edge.owner->node() == Node::Extract && edge.index == 0)
                       || (curry_prefixes_.contains(edge.owner) && edge.kind == EdgeKind::Operand)
                       || (curry_prefixes_.contains(def) && edge.kind == EdgeKind::Callee)) {
                pending.push_back(edge.owner);
            } else {
                unknown = true;
            }
        }
    }
    return unknown || coverage_ == UseCoverage::Open ? CheckResult::Unknown : CheckResult::True;
}

CheckResult GraphView::data_only_used_by(const Def* producer, const Def* consumer) const {
    auto used = data_used_by(producer, consumer);
    if (used != CheckResult::True) return used;
    for (const auto& edge : data_uses(producer))
        if (edge.consumer != consumer) return CheckResult::False;
    return data_boundary(producer);
}

CheckResult GraphView::no_external_users(Defs nodes, Defs outputs) const {
    DefSet inside, allowed;
    for (auto def : nodes) {
        if (!is_contained(def)) return CheckResult::Unknown;
        inside.insert(def);
    }
    for (auto def : outputs) {
        if (!inside.contains(def)) throw std::invalid_argument("output is not a matched node");
        allowed.insert(def);
    }
    bool unknown = false;
    for (auto def : nodes) {
        if (allowed.contains(def)) continue;
        for (const auto& edge : data_uses(def))
            if (!inside.contains(edge.consumer)) return CheckResult::False;
        auto boundary = data_boundary(def);
        if (boundary == CheckResult::False) return boundary;
        unknown |= boundary == CheckResult::Unknown;
    }
    return unknown ? CheckResult::Unknown : CheckResult::True;
}

size_t GraphView::data_use_count(const Def* def) const { return data_uses(def).size(); }

size_t GraphView::data_consumer_count(const Def* def) const {
    DefSet owners;
    for (const auto& edge : data_uses(def))
        owners.insert(edge.consumer);
    return owners.size();
}

} // namespace mim::pattern
