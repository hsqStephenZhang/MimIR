#include "mim/plug/tensor/phase/fusion_partition.h"

#include <algorithm>
#include <functional>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

FusionPartitionAnalysis::FusionPartitionAnalysis(World& world, size_t max_fuse_nodes)
    : Analysis(world, "FusionPartitionAnalysis")
    , max_fuse_nodes_(max_fuse_nodes) {}

void FusionPartitionAnalysis::prepare() {
    collected_.clear();
    external_refs_.clear();
    nodes_.clear();
    node_ids_.clear();
    post_dom_parent_.clear();
    post_dom_depth_.clear();
    groups_.clear();
}

void FusionPartitionAnalysis::collect_tensor_refs(const Def* def, Vector<const App*>& refs) const {
    if (auto app = Axm::isa<tensor::map_reduce>(def)) {
        refs.emplace_back(app);
        return;
    }
    if (auto tuple = def->isa<Tuple>())
        for (auto op : tuple->ops())
            collect_tensor_refs(op, refs);
}

const Def* FusionPartitionAnalysis::rewrite_imm_App(const App* app) {
    if (auto mr = Axm::isa<tensor::map_reduce>(app)) {
        auto [nis, _meta, _shapes, _types, _comb, _out_map, _maps, inputs] = mr->uncurry_args<8>();
        (void)_meta;
        (void)_shapes;
        (void)_types;
        (void)_comb;
        (void)_out_map;
        (void)_maps;

        auto& node = collected_[mr];
        node.app   = mr;
        if (auto n = Lit::isa<u64>(nis))
            for (u64 i = 0; i < *n; ++i)
                collect_tensor_refs(inputs->proj(*n, i), node.inputs);
        else
            node.external = true;
    } else {
        // A tensor result consumed outside the lowered tensor graph is a graph
        // output.  Such a node may receive fused producers, but must not itself
        // be folded into a later tensor consumer.
        Vector<const App*> refs;
        collect_tensor_refs(app->arg(), refs);
        external_refs_.insert(refs.begin(), refs.end());
    }
    return Analysis::rewrite_imm_App(app);
}

const Def* FusionPartitionAnalysis::rewrite_imm_Extract(const Extract* extract) {
    Vector<const App*> refs;
    collect_tensor_refs(extract->tuple(), refs);
    external_refs_.insert(refs.begin(), refs.end());
    return Analysis::rewrite_imm_Extract(extract);
}

const Def* FusionPartitionAnalysis::rewrite_imm_Insert(const Insert* insert) {
    Vector<const App*> refs;
    collect_tensor_refs(insert->tuple(), refs);
    collect_tensor_refs(insert->value(), refs);
    external_refs_.insert(refs.begin(), refs.end());
    return Analysis::rewrite_imm_Insert(insert);
}

void FusionPartitionAnalysis::finalize() {
    build_graph();
    build_post_dominators();
    partition();
}

void FusionPartitionAnalysis::build_graph() {
    Vector<const App*> roots;
    roots.reserve(collected_.size());
    for (const auto& [app, _] : collected_)
        roots.emplace_back(app);
    std::ranges::sort(roots, {}, [](const App* app) { return app->gid(); });

    absl::flat_hash_set<const App*> visited;
    std::function<void(const App*)> visit = [&](const App* app) {
        if (!visited.emplace(app).second) return;
        auto it = collected_.find(app);
        if (it == collected_.end()) return;
        for (auto input : it->second.inputs)
            visit(input);
        node_ids_[app] = nodes_.size();
        nodes_.push_back({app, {}, classify_map_reduce(app), it->second.external || external_refs_.contains(app)});
    };
    for (auto app : roots)
        visit(app);

    for (size_t consumer = 0; consumer < nodes_.size(); ++consumer) {
        for (auto input : collected_.at(nodes_[consumer].app).inputs) {
            auto it = node_ids_.find(input);
            if (it == node_ids_.end()) continue;
            auto& outputs = nodes_[it->second].outputs;
            if (std::ranges::find(outputs, consumer) == outputs.end()) outputs.emplace_back(consumer);
        }
    }
    for (auto& node : nodes_)
        if (node.outputs.empty()) node.external = true;
}

void FusionPartitionAnalysis::build_post_dominators() {
    post_dom_parent_.assign(nodes_.size(), -1);
    post_dom_depth_.assign(nodes_.size(), 1);

    auto lca = [&](int lhs, int rhs) {
        while (lhs != rhs && lhs >= 0 && rhs >= 0) {
            if (post_dom_depth_[lhs] > post_dom_depth_[rhs])
                lhs = post_dom_parent_[lhs];
            else if (post_dom_depth_[rhs] > post_dom_depth_[lhs])
                rhs = post_dom_parent_[rhs];
            else {
                lhs = post_dom_parent_[lhs];
                rhs = post_dom_parent_[rhs];
            }
        }
        return lhs == rhs ? lhs : -1;
    };

    for (size_t i = nodes_.size(); i-- > 0;) {
        if (nodes_[i].external) continue;
        int parent = -1;
        for (auto output : nodes_[i].outputs)
            parent = parent < 0 ? int(output) : lca(parent, int(output));
        post_dom_parent_[i] = parent;
        if (parent >= 0) post_dom_depth_[i] = post_dom_depth_[parent] + 1;
    }
}

size_t FusionPartitionAnalysis::find(size_t id) const {
    while (groups_[id].parent != id)
        id = groups_[id].parent;
    return id;
}

size_t FusionPartitionAnalysis::find(size_t id) {
    auto root = static_cast<const FusionPartitionAnalysis*>(this)->find(id);
    while (groups_[id].parent != id) {
        auto parent        = groups_[id].parent;
        groups_[id].parent = root;
        id                 = parent;
    }
    return root;
}

bool FusionPartitionAnalysis::check_path(size_t src, size_t sink, bool allow_reduction_sink) const {
    absl::flat_hash_set<size_t> visited;
    std::function<bool(size_t)> check = [&](size_t id) {
        if (!visited.emplace(id).second) return true;
        auto pattern = groups_[find(id)].pattern;
        if (id == sink)
            return pattern <= OpPatternKind::kInjective
                || (allow_reduction_sink && pattern == OpPatternKind::kCommReduce);
        if (pattern > OpPatternKind::kInjective) return false;
        for (auto output : nodes_[id].outputs)
            if (!check(output)) return false;
        return true;
    };
    for (auto output : nodes_[src].outputs)
        if (!check(output)) return false;
    return true;
}

bool FusionPartitionAnalysis::check_output_path(size_t src, size_t sink) const {
    absl::flat_hash_set<size_t> visited;
    std::function<bool(size_t)> check = [&](size_t id) {
        if (!visited.emplace(id).second) return true;
        if (groups_[find(id)].pattern > OpPatternKind::kBroadcast) return false;
        if (id == sink) return true;
        for (auto output : nodes_[id].outputs)
            if (!check(output)) return false;
        return true;
    };
    for (auto output : nodes_[src].outputs)
        if (!check(output)) return false;
    return true;
}

size_t FusionPartitionAnalysis::count_path_groups(size_t src, size_t sink) const {
    absl::flat_hash_set<size_t> visited_nodes;
    absl::flat_hash_set<size_t> visited_groups;
    std::function<void(size_t)> count = [&](size_t id) {
        if (id == sink || !visited_nodes.emplace(id).second) return;
        visited_groups.emplace(find(id));
        for (auto output : nodes_[id].outputs)
            count(output);
    };
    count(src);
    size_t result = groups_[find(sink)].size;
    for (auto group : visited_groups)
        result += groups_[group].size;
    return result;
}

void FusionPartitionAnalysis::commit_fuse(size_t src, size_t sink) {
    auto target = find(sink);
    absl::flat_hash_set<size_t> visited;
    std::function<void(size_t)> commit = [&](size_t id) {
        if (id == sink || !visited.emplace(id).second) return;
        auto child = find(id);
        target     = find(target);
        if (child != target) {
            groups_[target].size += groups_[child].size;
            if (groups_[child].pattern > groups_[target].pattern) groups_[target].pattern = groups_[child].pattern;
            groups_[child].parent = target;
        }
        for (auto output : nodes_[id].outputs)
            commit(output);
    };
    commit(src);
}

void FusionPartitionAnalysis::partition() {
    groups_.resize(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i)
        groups_[i] = {i, 1, nodes_[i].pattern};

    // Phase 0: fuse elementwise/broadcast producers through injective paths,
    // including map producers feeding a reduction sink.
    // Phase 1: fuse the remaining injective chains.  Reduction producers and
    // opaque operations remain group boundaries.
    for (int phase = 0; phase < 2; ++phase) {
        for (size_t id = 0; id < nodes_.size(); ++id) {
            if (nodes_[id].external || post_dom_parent_[id] < 0) continue;
            auto src_pattern = groups_[find(id)].pattern;
            auto sink        = size_t(post_dom_parent_[id]);
            if (find(id) == find(sink)) continue;

            bool candidate
                = phase == 0 ? src_pattern <= OpPatternKind::kBroadcast : src_pattern == OpPatternKind::kInjective;
            bool valid_path = candidate && check_path(id, sink, phase == 0);
            if (phase == 0 && src_pattern == OpPatternKind::kOutEWiseFusable) {
                candidate  = true;
                valid_path = check_output_path(id, sink);
            }
            if (!candidate || !valid_path) continue;
            if (count_path_groups(id, sink) > max_fuse_nodes_) continue;
            commit_fuse(id, sink);
        }
    }
}

bool FusionPartitionAnalysis::same_group(const App* lhs, const App* rhs) const {
    auto l = node_ids_.find(lhs);
    auto r = node_ids_.find(rhs);
    return l != node_ids_.end() && r != node_ids_.end() && find(l->second) == find(r->second);
}

bool FusionPartitionAnalysis::can_inline(const App* producer, const App* consumer) const {
    auto p = node_ids_.find(producer);
    auto c = node_ids_.find(consumer);
    if (p == node_ids_.end() || c == node_ids_.end() || !same_group(producer, consumer)) return false;
    return !nodes_[p->second].external && nodes_[p->second].outputs.size() == 1
        && nodes_[p->second].outputs.front() == c->second;
}

} // namespace mim::plug::tensor::phase
