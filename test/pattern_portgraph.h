#pragma once

// Test/benchmark adapter for acyclic port graphs with three input/output ports.
// Each vertex has a private tag to prevent IR hash-consing from merging vertices.
// An output port is an explicit unary operation; input ports are tuple positions.
// Open pattern ports impose no condition. This is a positive-edge embedding,
// not an induced-subgraph matcher or a representation of cyclic IR.
#include <array>
#include <stdexcept>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph_match.h>

namespace port_test {
using namespace mim;
using namespace mim::pattern;

struct Edge {
    size_t source, output, target, input;
};

struct Graph {
    size_t size;
    std::vector<Edge> edges;
};

// Adapted topology from portmatching's single_pattern_match_complex/add_pattern.
inline Graph complex_pattern() {
    return {
        4,
        {{0, 0, 2, 1}, {1, 1, 2, 0}, {2, 0, 3, 1}, {1, 2, 3, 2}}
    };
}

inline void append(Graph& graph, const Graph& pattern, std::span<const size_t> vertices) {
    if (vertices.size() != pattern.size) throw std::invalid_argument("vertex count");
    for (auto e : pattern.edges)
        graph.edges.push_back({vertices[e.source], e.output, vertices[e.target], e.input});
}

inline auto incoming(const Graph& graph) {
    std::vector<std::array<std::optional<Edge>, 3>> result(graph.size);
    std::vector<std::array<bool, 3>> outputs(graph.size);
    for (auto e : graph.edges) {
        if (e.source >= e.target || e.target >= graph.size || e.input >= 3 || e.output >= 3)
            throw std::invalid_argument("expected topologically ordered three-port DAG");
        if (result[e.target][e.input] || outputs[e.source][e.output])
            throw std::invalid_argument("a port may have only one link");
        result[e.target][e.input]   = e;
        outputs[e.source][e.output] = true;
    }
    return result;
}

struct IR {
    Driver driver;
    World& w          = driver.world();
    const Def* nat    = w.type_nat();
    const Axm* vertex = w.axm(w.pi(w.arr(4, nat), nat));
    std::array<const Axm*, 3> outputs;
    OpRegistry ops;
    std::vector<const Def*> nodes;

    explicit IR(const Graph& graph) {
        ops.add(vertex->flags(), 1,
                {
                    {0, 1}
        },
                true);
        for (auto& output : outputs) {
            output = w.axm(w.pi(nat, nat));
            ops.add(output->flags(), 1, {{0}}, true);
        }
        for (auto& ports : incoming(graph)) {
            std::array<const Def*, 4> args{w.lit_nat(nodes.size() + 1), w.lit_nat(0), w.lit_nat(0), w.lit_nat(0)};
            for (size_t i = 0; i < 3; ++i)
                if (auto e = ports[i]) args[i + 1] = w.app(outputs[e->output], nodes[e->source]);
            nodes.push_back(w.app(vertex, w.tuple(args)));
        }
    }
};

struct Compiled {
    PatternProgram p;
    GraphPattern graph{p};
    std::vector<Pattern> vertices;
    std::vector<Anchor> anchors;

    Compiled(const Graph& pattern, const IR& ir, bool flat = false) {
        if (flat)
            for (size_t i = 0; i < pattern.size; ++i)
                vertices.push_back(p.capture());
        for (auto& ports : incoming(pattern)) {
            std::vector<Pattern> args(4, p.any());
            for (size_t i = 0; i < 3; ++i)
                if (auto e = ports[i])
                    args[i + 1] = p.node(Node::App, {p.exact(ir.outputs[e->output]), vertices[e->source]});
            auto value = p.node(Node::App, {p.exact(ir.vertex), p.node(Node::Tuple, std::move(args))});
            if (flat) {
                // Describe each vertex once. Edges refer only to shared identity captures.
                value = p.all(vertices[anchors.size()], value);
            } else {
                value = p.capture(value);
                vertices.push_back(value);
            }
            anchors.push_back(graph.anchor(value)); // Distinct anchors enforce vertex injectivity.
        }
    }
};

inline GraphMatchOptions options(GraphSearchMode mode, bool reverse = false) {
    GraphMatchOptions options;
    options.mode               = mode;
    options.reverse_candidates = reverse;
    options.max_results        = 1'000'000;
    options.max_candidates     = 100'000'000;
    options.match.max_steps    = 100'000'000;
    return options;
}
} // namespace port_test
