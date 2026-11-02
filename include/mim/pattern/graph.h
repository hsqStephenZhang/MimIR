#pragma once

#include <memory>
#include <span>
#include <vector>

#include "mim/pattern/op.h"

namespace mim {
class Lam;
}

namespace mim::pattern {

enum class EdgeKind { Operand, Callee, Type };

struct UseEdge {
    const Def* owner;
    size_t index;
    EdgeKind kind;
};

/// Closed is an explicit caller guarantee that roots cover all relevant data uses.
/// A locally selected region is Open unless its caller has proved that guarantee.
enum class UseCoverage { Open, Closed };

struct GraphOptions {
    UseCoverage coverage            = UseCoverage::Open;
    const OpRegistry* operators     = nullptr; // Copied into the snapshot.
    Lam* scope                      = nullptr; // The enclosing closed function of the selected occurrence.
    size_t max_inputs               = 4096;
    std::vector<size_t> region_path = {};
    bool shared_region              = false;
};

struct DataEdge {
    const Def* value;
    const Def* producer;
    const Def* consumer;
    size_t group;
    std::vector<size_t> path;
    std::vector<size_t> result_path;
};

struct DataOccurrence {
    const Def* consumer;
    std::optional<u64> trips;
    std::vector<const Def*> regions;
};

class GraphSnapshot {
public:
    bool is_live() const;

private:
    static std::shared_ptr<GraphSnapshot> capture(World&);
    mutable bool published_ = false;
    World* world_           = nullptr;
    std::shared_ptr<const void> generation_;
    friend class GraphRewrite;
    friend class GraphView;
    friend class MatchEngine;
};

/// A snapshot of raw Def edges, not semantic operator inputs. Tuple/Pack are real owners here.
/// Traversal stops at every mutable, Lam, Pi, Var and Axm. Types are indexed but not traversed;
/// Var type edges are omitted because computing a sequence Var's type can construct IR.
/// The source World must outlive this object and must not be changed while the view is in use.
class GraphView {
public:
    explicit GraphView(Defs roots, UseCoverage = UseCoverage::Open);
    GraphView(Defs roots, GraphOptions);
    GraphView(World&, const OpRegistry&);

    std::span<const Def* const> nodes() const {
        check_live();
        return nodes_;
    }
    std::span<const Def* const> roots() const {
        check_live();
        return roots_;
    }
    std::span<const UseEdge> uses(const Def*) const;
    bool is_contained(const Def* def) const { return snapshot_->is_live() && def && indices_.contains(def); }
    /// Number of raw operand occurrences, excluding callee and type edges.
    size_t use_count(const Def*) const;
    size_t consumer_count(const Def*) const;
    bool is_used_by(const Def* producer, const Def* owner) const;
    /// Exclusivity among raw operand owners. This is not semantic operator exclusivity:
    /// in particular, a Pack body edge does not describe its dynamic read multiplicity.
    CheckResult only_used_by(const Def* producer, const Def* owner) const;

    World& world() const;
    Lam* scope() const {
        check_live();
        return scope_;
    }
    std::shared_ptr<const GraphSnapshot> snapshot() const { return snapshot_; }
    bool is_current() const;
    std::span<const size_t> region_path() const { return region_path_; }
    bool shared_region() const { return shared_region_; }
    std::span<const Def* const> operations() const {
        check_live();
        return operations_;
    }
    const OpView* isa_operation(const Def*) const;
    bool is_pure(const Def*) const;
    bool is_pure(flags_t axm) const;
    bool has_purity(flags_t axm) const;
    bool is_pure_call(flags_t axm, std::span<const Def* const> argument_groups) const;
    std::span<const DataEdge> data_uses(const Def*) const;
    CheckResult data_used_by(const Def* producer, const Def* consumer) const;
    CheckResult data_only_used_by(const Def* producer, const Def* consumer) const;
    CheckResult no_external_users(Defs nodes, Defs outputs) const;
    size_t data_use_count(const Def*) const;
    size_t data_consumer_count(const Def*) const;
    bool is_program() const { return program_; }
    CheckResult complete_data_uses(const Def*) const;
    std::span<const DataOccurrence> occurrences(const Def*) const;

private:
    void check_live() const;
    struct MutableState {
        const Def* def;
        const Def* type;
        std::vector<const Def*> ops;
    };
    void build_semantics(size_t);
    void index_occurrences();
    CheckResult data_boundary(const Def*) const;
    std::vector<const Def*> roots_;
    DefSet outputs_;
    std::vector<const Def*> nodes_;
    DefMap<size_t> indices_;
    DefMap<std::vector<UseEdge>> uses_;
    UseCoverage coverage_;
    World* world_ = nullptr;
    Lam* scope_   = nullptr;
    std::vector<size_t> region_path_;
    bool shared_region_;
    std::optional<OpRegistry> operators_;
    std::vector<const Def*> operations_;
    DefMap<OpView> op_views_;
    DefMap<std::vector<DataEdge>> data_uses_;
    DefSet curry_prefixes_;
    DefSet unknown_uses_;
    bool program_ = false;
    DefMap<std::vector<DataOccurrence>> occurrences_;
    std::vector<MutableState> mutables_;
    std::vector<std::pair<Sym, const Def*>> externals_;
    std::shared_ptr<GraphSnapshot> snapshot_ = std::make_shared<GraphSnapshot>();
};

} // namespace mim::pattern
