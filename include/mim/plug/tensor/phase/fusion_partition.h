#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <mim/phase.h>

#include "mim/plug/tensor/phase/op_pattern.h"

namespace mim::plug::tensor::phase {

/// TVM-style fusion partitioning for the lowered tensor dataflow DAG.
///
/// The analysis decides which operations belong to one fusion group.  It does
/// not prescribe how a group is materialized; Fuse separately consumes the
/// resulting evidence with the capabilities of its continuation rewriter.
class FusionPartitionAnalysis final : public Analysis {
public:
    explicit FusionPartitionAnalysis(World&, size_t max_fuse_nodes = 16);

    bool same_group(const App* lhs, const App* rhs) const;

    /// Whether the current tree-oriented continuation rewriter can inline this
    /// edge without duplicating a shared or externally visible producer.
    bool can_inline(const App* producer, const App* consumer) const;

private:
    struct CollectedNode {
        const App* app = nullptr;
        Vector<const App*> inputs;
        bool external = false;
    };

    struct Node {
        const App* app = nullptr;
        Vector<size_t> outputs;
        OpPatternKind pattern = OpPatternKind::kOpaque;
        bool external         = false;
    };

    struct Group {
        size_t parent         = 0;
        size_t size           = 1;
        OpPatternKind pattern = OpPatternKind::kOpaque;
    };

    const Def* rewrite_imm_App(const App*) final;
    const Def* rewrite_imm_Extract(const Extract*) final;
    const Def* rewrite_imm_Insert(const Insert*) final;
    void prepare() final;
    void finalize() final;

    void collect_tensor_refs(const Def*, Vector<const App*>&) const;
    void build_graph();
    void build_post_dominators();
    void partition();

    size_t find(size_t) const;
    size_t find(size_t);
    bool check_path(size_t src, size_t sink, bool allow_reduction_sink) const;
    bool check_output_path(size_t src, size_t sink) const;
    size_t count_path_groups(size_t src, size_t sink) const;
    void commit_fuse(size_t src, size_t sink);

    size_t max_fuse_nodes_;
    absl::flat_hash_map<const App*, CollectedNode> collected_;
    absl::flat_hash_set<const App*> external_refs_;

    Vector<Node> nodes_;
    absl::flat_hash_map<const App*, size_t> node_ids_;
    Vector<int> post_dom_parent_;
    Vector<int> post_dom_depth_;
    Vector<Group> groups_;
};

} // namespace mim::plug::tensor::phase
