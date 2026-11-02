#include "pattern_portgraph.h"

#include <algorithm>
#include <set>

#include <doctest/doctest.h>

using namespace port_test;

namespace {
using Mappings = std::set<std::vector<size_t>>;

Mappings search(const Graph& host, const Graph& pattern, GraphSearchMode mode, bool reverse, bool flat = false) {
    IR ir(host);
    GraphView view(ir.nodes, {.operators = &ir.ops});
    Compiled compiled(pattern, ir, flat);
    auto found = find_matches(compiled.graph, view, options(mode, reverse));
    REQUIRE(found.status == SearchStatus::Complete);
    Mappings result;
    for (auto& match : found.matches) {
        std::vector<size_t> mapping;
        for (auto anchor : compiled.anchors) {
            auto it = std::find(ir.nodes.begin(), ir.nodes.end(), match.at(anchor));
            REQUIRE(it != ir.nodes.end());
            mapping.push_back(it - ir.nodes.begin());
        }
        CHECK(result.insert(mapping).second);
    }
    return result;
}
} // namespace

TEST_CASE("Portmatching: overlapping four-vertex port patterns retain exact mappings") {
    // portmatching/src/concrete/portgraph.rs::single_pattern_match_complex:
    // same 100 vertices, four labelled edges, and three overlapping embeddings.
    Graph host{100, {}};
    auto pattern = complex_pattern();
    for (auto vertices : {
             std::array<size_t, 4>{ 0, 10, 30, 55},
             { 3, 12, 23, 44},
             {12, 55, 98, 99}
    })
        append(host, pattern, vertices);
    Mappings expected{
        { 0, 10, 30, 55},
        { 3, 12, 23, 44},
        {12, 55, 98, 99}
    };
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool reverse : {false, true})
            for (bool flat : {false, true})
                CHECK(search(host, pattern, mode, reverse, flat) == expected);
}

TEST_CASE("Portmatching: both endpoint port offsets and shared vertex identity matter") {
    auto pattern = complex_pattern();
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool flat : {false, true}) {
            CHECK(search(pattern, pattern, mode, false, flat)
                  == Mappings{
                      {0, 1, 2, 3}
            });
            auto wrong_output                = pattern;
            wrong_output.edges.back().output = 0; // Unused port of vertex 1.
            CHECK(search(wrong_output, pattern, mode, false, flat).empty());
            auto wrong_input               = pattern;
            wrong_input.edges.back().input = 0;
            CHECK(search(wrong_input, pattern, mode, false, flat).empty());
            Graph split{
                5,
                {{0, 0, 2, 1}, {1, 1, 2, 0}, {2, 0, 4, 1}, {3, 2, 4, 2}}
            };
            CHECK(search(split, pattern, mode, false, flat).empty()); // The two occurrences of vertex 1 differ.
        }
}

TEST_CASE("Portmatching: vertex injectivity and unconstrained boundary ports") {
    // A fork has two distinct destinations, even if captures would otherwise permit aliasing.
    Graph fork{
        3,
        {{0, 0, 1, 0}, {0, 1, 2, 1}}
    };
    Graph collapsed{
        2,
        {{0, 0, 1, 0}, {0, 1, 1, 1}}
    };
    // Extra links at open pattern ports are allowed (positive-edge embedding).
    Graph extended{
        4,
        {{0, 0, 1, 0}, {0, 1, 2, 1}, {1, 2, 3, 2}}
    };
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool flat : {false, true}) {
            CHECK(search(collapsed, fork, mode, false, flat).empty());
            CHECK(search(extended, fork, mode, false, flat)
                  == Mappings{
                      {0, 1, 2}
            });
        }
}

TEST_CASE("Portmatching: factored vertex constraints match long chains within the default depth budget") {
    Graph host{12, {}}, pattern{8, {}};
    for (size_t i = 1; i < host.size; ++i)
        host.edges.push_back({i - 1, 0, i, 0});
    for (size_t i = 1; i < pattern.size; ++i)
        pattern.edges.push_back({i - 1, 0, i, 0});
    Mappings expected;
    for (size_t start = 0; start < 5; ++start) {
        std::vector<size_t> mapping;
        for (size_t i = 0; i < pattern.size; ++i)
            mapping.push_back(start + i);
        expected.insert(mapping);
    }
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool reverse : {false, true})
            CHECK(search(host, pattern, mode, reverse, true) == expected);
}

TEST_CASE("GHL: positive-edge matching permits extra internal links unlike induced isomorphism") {
    Graph pattern{
        3,
        {{0, 0, 1, 0}, {1, 0, 2, 0}}
    };
    Graph host{
        3,
        {{0, 0, 1, 0}, {1, 0, 2, 0}, {0, 1, 2, 1}}
    };
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool flat : {false, true})
            CHECK(search(host, pattern, mode, false, flat)
                  == Mappings{
                      {0, 1, 2}
            });
}

TEST_CASE("GHL: vertex predicate and validation filter before selecting the first match") {
    // Adapt the selection part of InsertVertexAfterID (VF2/VF3 tests), including
    // rejecting already-transformed vertices. Graph mutation itself is not compared.
    Graph host{
        4,
        {{0, 0, 1, 0}, {2, 0, 3, 0}}
    };
    IR ir(host);
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    auto tag           = p.capture();
    auto value         = p.node(Node::App, {p.exact(ir.vertex), p.node(Node::Tuple, {tag, p.any(), p.any(), p.any()})});
    value              = p.where(value, {tag}, [](std::span<const Def* const> values) {
        auto id = values[0]->as<Lit>()->get<nat_t>() - 1;
        return id == 0 || id == 2 ? CheckResult::True : CheckResult::False;
    });
    size_t validations = 0;
    value              = p.where(value, {tag}, [&](std::span<const Def* const> values) {
        ++validations;
        auto id = values[0]->as<Lit>()->get<nat_t>() - 1;
        // The vertex comparator accepts IDs 0 and 2; validation rejects ID 0.
        return id == 2 ? CheckResult::True : CheckResult::False;
    });
    GraphPattern graph(p);
    auto anchor      = graph.anchor(value);
    auto opts        = options(GraphSearchMode::Indexed);
    opts.max_results = 1;
    auto found       = find_matches(graph, view, opts);
    REQUIRE(found.matches.size() == 1);
    CHECK(found.matches[0].at(anchor) == ir.nodes[2]);
    CHECK(validations >= 1);
    CHECK(found.status == SearchStatus::Result_Limit);
}
