#include "mim/pattern/graph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "mim/axm.h"
#include "mim/driver.h"
#include "mim/lam.h"
#include "mim/tuple.h"
#include "mim/world.h"

namespace mim::pattern {

namespace {
DefVec program_bodies(World& world) {
    DefVec pending, result;
    for (auto ext : world.externals().muts())
        pending.push_back(ext);
    DefSet seen;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        if (auto lam = def->isa<Lam>(); lam && lam->is_set())
            result.push_back(lam->body());
        else if (def->is_mutable() && !def->isa<Pi>() && !def->isa<Rule>())
            for (auto op : def->ops())
                if (op) result.push_back(op);
        if (def->isa<Var>()) continue;
        for (auto dep : def->deps())
            if (dep) pending.push_back(dep);
    }
    return result;
}

std::optional<u64> multiply(std::optional<u64> a, std::optional<u64> b) {
    if (!a || !b || (*b && *a > std::numeric_limits<u64>::max() / *b)) return {};
    return *a * *b;
}
} // namespace

GraphView::GraphView(World& world, const OpRegistry& ops)
    : GraphView(program_bodies(world), {.coverage = UseCoverage::Closed, .operators = &ops}) {
    program_ = true;
    world_   = &world;
    unknown_uses_.clear();
    for (const auto& state : mutables_)
        for (auto op : state.ops)
            if (op && contained_.contains(op)) {
                auto fn = state.def->isa<Lam>();
                if (!fn || op != fn->body()) unknown_uses_.insert(op);
            }
    index_occurrences();
}

void GraphView::index_occurrences() {
    DefSet entered, recursive;
    DefMap<DefSet> region_owners;
    for (auto def : operations_)
        for (auto region : isa_operation(def)->regions) {
            region_owners[region.body].insert(def);
            auto outer = region.body->isa<Lam>();
            while (outer && outer->is_set()) {
                auto inner = outer->body()->isa<Lam>();
                if (!inner) break;
                region_owners[inner].insert(outer);
                outer = inner;
            }
        }
    auto visit = [&](this auto&& visit, const Def* body, std::optional<u64> trips, std::vector<const Def*> regions,
                     DefSet active, bool opaque) -> void {
        DefVec pending{body};
        DefSet seen;
        for (size_t i = 0; i != pending.size(); ++i) {
            if (i > 64 * (world_->curr_gid() + 1))
                throw std::invalid_argument("program occurrence traversal budget exceeded");
            auto def = pending[i];
            if (!seen.insert(def).second) continue;
            if (def->is_mutable() || def->isa<Var>() || def->isa<Axm>()) continue;
            if (auto op = isa_operation(def)) {
                occurrences_[def].push_back({def, trips, regions});
                if (opaque)
                    for (auto input : op->inputs)
                        unknown_uses_.insert(input.value);
                for (auto region : op->regions) {
                    auto fn = region.body->isa<Lam>();
                    if (!fn || !fn->is_set()) continue;
                    if (active.contains(fn)) {
                        recursive.insert(fn);
                        continue;
                    }
                    entered.insert(fn);
                    auto nested = regions;
                    nested.push_back(def);
                    auto next = active;
                    next.insert(fn);
                    while (auto inner = fn->body()->isa<Lam>()) {
                        if (!inner->is_set() || next.contains(inner)) break;
                        entered.insert(inner);
                        next.insert(inner);
                        fn = inner;
                    }
                    visit(fn->body(), multiply(trips, region.trips), std::move(nested), std::move(next), opaque);
                }
            }
            for (auto op : def->ops())
                if (op) pending.push_back(op);
        }
    };
    for (auto ext : world_->externals().muts())
        if (auto fn = ext->isa<Lam>(); fn && fn->is_set()) {
            entered.insert(fn);
            visit(fn->body(), 1, {}, {fn}, false);
        }
    for (const auto& state : mutables_)
        if (auto fn = state.def->isa<Lam>(); fn && fn->is_set()) {
            auto opaque = !entered.contains(fn) || recursive.contains(fn);
            for (auto edge : uses(fn))
                if (edge.kind != EdgeKind::Type && !region_owners[fn].contains(edge.owner)) opaque = true;
            if (opaque) visit(fn->body(), {}, {}, {fn}, true);
        }
}

std::span<const DataOccurrence> GraphView::occurrences(const Def* consumer) const {
    check_live();
    auto i = occurrences_.find(consumer);
    return i == occurrences_.end() ? std::span<const DataOccurrence>() : std::span<const DataOccurrence>(i->second);
}

CheckResult GraphView::complete_data_uses(const Def* producer, bool include_outputs) const {
    if (!program_ || !is_contained(producer)) return CheckResult::Unknown;
    return data_boundary(producer, include_outputs) == CheckResult::True ? CheckResult::True : CheckResult::Unknown;
}

bool GraphSnapshot::is_live() const { return !published_ && (!world_ || world_->data_.storage_token == generation_); }

std::shared_ptr<GraphSnapshot> GraphSnapshot::capture(World& world) {
    auto result         = std::make_shared<GraphSnapshot>();
    result->world_      = &world;
    result->generation_ = world.data_.storage_token;
    return result;
}

void GraphView::check_live() const {
    if (!snapshot_->is_live()) throw std::invalid_argument("graph snapshot is stale");
}

GraphView::GraphView(Defs roots, UseCoverage coverage)
    : GraphView(roots, GraphOptions{.coverage = coverage}) {}

GraphView::GraphView(Defs roots, GraphOptions options)
    : coverage_(options.coverage)
    , scope_(options.scope)
    , region_path_(std::move(options.region_path))
    , shared_region_(options.shared_region) {
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
        world_      = &scope_->world();
        auto region = scope_->body();
        for (auto index : region_path_) {
            if (index >= region->num_ops()) throw std::invalid_argument("invalid region occurrence path");
            if (auto lam = region->isa<Lam>()) {
                if (index != 1 || !lam->is_set() || lam->is_external())
                    throw std::invalid_argument("invalid region lambda occurrence");
            } else if (!region->isa<App>() && !region->isa<Tuple>() && !region->isa_imm<Pack>()) {
                throw std::invalid_argument("unsupported region occurrence owner");
            }
            region = region->op(index);
        }
        auto pending = std::vector<const Def*>{region};
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
        if (contained_.insert(def).second) nodes_.push_back(def);
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
            if (unknown_uses && def != scope_ && !contained_.contains(def))
                for (auto op : def->ops())
                    if (op && contained_.contains(op)) unknown_uses_.insert(op);
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

bool GraphView::is_pure_function(const Def* fn) const {
    auto root = fn->isa<Lam>();
    if (!root || !root->is_set() || root->is_external()) return false;
    DefVec pending{root};
    DefSet seen;
    LamSet locals;
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i > 64 * (world_->curr_gid() + 1)) return false;
        auto def = pending[i];
        if (!seen.insert(def).second || def->isa<Var>() || def->isa<Axm>() || def->isa<Pi>()) continue;
        if (auto lam = def->isa_mut<Lam>(); lam && !lam->is_external()) locals.emplace(lam);
        for (auto op : def->ops())
            if (op) pending.push_back(op);
    }
    auto local = [&](const Def* def) {
        while (auto extract = def->isa<Extract>())
            def = extract->tuple();
        if (auto var = def->isa<Var>())
            if (auto binder = var->binder()->isa_mut<Lam>()) return locals.contains(binder);
        if (auto lam = def->isa_mut<Lam>()) return locals.contains(lam);
        return false;
    };
    // Only computations that depend on a local parameter run when @p fn is called.
    auto evaluated = [&](const Def* def) {
        return std::ranges::any_of(locals,
                                   [&](Lam* lam) { return lam->has_var() && def->has_free_var(lam->has_var()); });
    };
    for (auto def : seen)
        if (auto app = def->isa<App>(); app && evaluated(app)) {
            if (app->type()->isa<Type>()) continue;
            auto callee = App::uncurry_callee(app);
            if (local(callee)) continue;
            auto axm = callee->isa<Axm>();
            if (axm && app->type()->isa<Pi>() && is_pure(axm->flags())) continue;
            DefVec args;
            for (auto cur = app; cur; cur = cur->callee()->isa<App>())
                args.push_back(cur->arg());
            std::ranges::reverse(args);
            if (!axm || !is_pure_call(axm->flags(), args)) {
                world_->log().d("graph pure-function proof rejected {}", app);
                return false;
            }
        }
    return true;
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

CheckResult GraphView::data_boundary(const Def* producer, bool include_outputs) const {
    auto pending = std::vector<const Def*>{producer};
    DefSet seen;
    bool unknown = false;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second) continue;
        unknown |= unknown_uses_.contains(def);
        if (outputs_.contains(def) && !include_outputs) return CheckResult::False;
        for (auto edge : uses(def)) {
            if (edge.kind == EdgeKind::Type) continue;
            if (include_outputs && outputs_.contains(edge.owner) && edge.kind == EdgeKind::Operand)
                if (auto call = edge.owner->isa<App>(); call && Pi::isa_cn(call->callee()->type())) {
                    auto parameter = call->callee();
                    u64 index      = 0;
                    if (auto ex = parameter->isa<Extract>(); ex && Lit::isa(ex->index())) {
                        index     = Lit::as(ex->index());
                        parameter = ex->tuple();
                    }
                    if (auto var = parameter->isa<Var>())
                        if (auto fn = var->binder()->isa<Lam>();
                            fn && fn->is_external() && index + 1 == fn->type()->dom()->num_projs()) {
                            pending.push_back(edge.owner);
                            continue;
                        }
                }
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
