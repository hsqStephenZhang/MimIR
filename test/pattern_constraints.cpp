#include <map>
#include <random>
#include <set>

#include <doctest/doctest.h>

#include "pattern_portgraph.h"

using namespace port_test;

namespace {
using Key = std::vector<const Def*>;
struct Record {
    Key captures;
    std::set<const Def*> reads, nodes;
    bool operator==(const Record&) const = default;
};
auto records(const GraphMatches& found, std::span<const Pattern> captures) {
    std::map<Key, Record> result;
    for (const auto& match : found.matches) {
        Record record;
        for (auto capture : captures)
            record.captures.push_back(match.isa_capture(capture));
        record.reads = {match.reads().begin(), match.reads().end()};
        record.nodes = {match.nodes().begin(), match.nodes().end()};
        CHECK(result.emplace(Key(match.anchors().begin(), match.anchors().end()), std::move(record)).second);
    }
    return result;
}

void compare(const GraphPattern& pattern,
             const GraphView& view,
             std::span<const Pattern> captures,
             bool expected_compiled = true) {
    CompiledGraphPattern plan(pattern);
    CHECK(plan.is_compiled() == expected_compiled);
    auto gid       = view.world().curr_gid();
    auto reference = find_matches(pattern, view, options(GraphSearchMode::Exhaustive));
    for (bool reverse : {false, true}) {
        auto found = plan.find(view, options(GraphSearchMode::Compiled, reverse));
        CHECK(found.status == reference.status);
        CHECK(found.used_compiled == expected_compiled);
        CHECK(records(found, captures) == records(reference, captures));
        CHECK(view.world().curr_gid() == gid);
    }
}
} // namespace

TEST_CASE("Compiled graph: random small port DAGs preserve complete mappings and rewrite metadata") {
    std::mt19937 rng(0x504f5254);
    for (size_t trial = 0; trial < 32; ++trial) {
        Graph host{8, {}};
        std::array<std::array<bool, 3>, 8> inputs{}, outputs{};
        for (size_t s = 0; s < 8; ++s)
            for (size_t t = s + 1; t < 8; ++t)
                if (rng() % 3 == 0) {
                    auto o = rng() % 3, i = rng() % 3;
                    if (!outputs[s][o] && !inputs[t][i]) {
                        host.edges.push_back({s, o, t, i});
                        outputs[s][o] = inputs[t][i] = true;
                    }
                }
        IR ir(host);
        GraphView view(ir.nodes, {.operators = &ir.ops});
        for (auto shape : {
                 Graph{2, {{0, 0, 1, 0}}},
                 complex_pattern()
        }) {
            Compiled pattern(shape, ir);
            compare(pattern.graph, view, pattern.vertices);
        }
    }
}

TEST_CASE("Compiled graph: type-only capture joins and guards use the shared binding environment") {
    Driver driver;
    auto& w  = driver.world();
    auto nat = w.type_nat();
    auto m = w.axm(nat), k = w.axm(nat), n = w.axm(nat);
    auto make_a  = w.axm(w.pi(nat, w.arr(Defs{m, k}, nat)));
    auto make_b  = w.axm(w.pi(nat, w.arr(Defs{k, n}, nat)));
    auto wrong_b = w.axm(w.pi(nat, w.arr(Defs{m, n}, nat)));
    auto a = w.app(make_a, w.lit_nat(1)), b = w.app(make_b, w.lit_nat(2));
    auto wrong = w.app(wrong_b, w.lit_nat(3));
    OpRegistry ops;
    for (auto op : {make_a, make_b, wrong_b})
        ops.add(op->flags(), 1, {{0}}, true);
    GraphView view({a, b, wrong}, {.operators = &ops});
    PatternProgram p;
    auto pm = p.capture(), pk = p.capture(), pn = p.capture(), pt = p.capture();
    auto matrix = [&](Pattern x, Pattern y) { return p.node(Node::Arr, {p.node(Node::Tuple, {x, y}), pt}); };
    auto pa     = p.capture(p.all(p.node(Node::App, {p.exact(make_a), p.any()}), p.type(matrix(pm, pk))));
    auto pb     = p.capture(p.all(p.node(Node::App, {p.any(), p.any()}), p.type(matrix(pk, pn))));
    pb          = p.where(pb, {pm, pk, pn, pt}, [&](auto values) {
        return values[0] == m && values[1] == k && values[2] == n && values[3] == nat ? CheckResult::True
                                                                                      : CheckResult::False;
    });
    GraphPattern pattern(p);
    pattern.anchor(pa);
    pattern.anchor(pb);
    std::array captures{pm, pk, pn, pt, pa};
    compare(pattern, view, captures);
    auto found = CompiledGraphPattern(pattern).find(view);
    REQUIRE(found.matches.size() == 1);
    CHECK(found.matches[0].anchors()[0] == a);
    CHECK(found.matches[0].anchors()[1] == b);
}

TEST_CASE("Compiled graph: unsupported alternatives, negation and ranges use the interpreter") {
    IR ir(Graph{2, {{0, 0, 1, 0}}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    auto cap = p.capture();
    for (auto value : {p.either(cap, p.any()), p.negate(p.exact(ir.nodes[0])),
                       p.node_range(Node::App, {p.exact(ir.vertex)}, p.any(), 1, 1)}) {
        GraphPattern graph(p);
        graph.anchor(value);
        compare(graph, view, std::array{cap}, false);
    }
}

TEST_CASE("Compiled graph: type keys prune root pairs before joining") {
    Driver driver;
    auto& w                = driver.world();
    auto nat               = w.type_nat();
    constexpr size_t count = 24;
    DefVec roots;
    OpRegistry ops;
    for (size_t i = 0; i < count; ++i) {
        auto k = w.lit_nat(i + 5);
        auto a = w.axm(w.pi(nat, w.arr(Defs{w.lit_nat(2), k}, nat)));
        auto b = w.axm(w.pi(w.type_bool(), w.arr(Defs{k, w.lit_nat(4)}, nat)));
        for (auto op : {a, b})
            ops.add(op->flags(), 1, {{0}}, true);
        roots.push_back(w.app(a, w.lit_nat(1)));
        roots.push_back(w.app(b, w.lit_ff()));
    }
    GraphView view(roots, {.operators = &ops});
    PatternProgram p;
    auto k      = p.capture();
    auto matrix = [&](Pattern x, Pattern y) { return p.node(Node::Arr, {p.node(Node::Tuple, {x, y}), p.exact(nat)}); };
    GraphPattern graph(p);
    graph.anchor(p.all(p.node(Node::App, {p.any(), p.exact(w.lit_nat(1))}), p.type(matrix(p.exact(w.lit_nat(2)), k))));
    graph.anchor(p.all(p.node(Node::App, {p.any(), p.exact(w.lit_ff())}), p.type(matrix(k, p.exact(w.lit_nat(4))))));
    auto indexed  = find_matches(graph, view);
    auto compiled = CompiledGraphPattern(graph).find(view);
    REQUIRE(indexed.status == SearchStatus::Complete);
    REQUIRE(compiled.status == SearchStatus::Complete);
    CHECK(compiled.used_compiled);
    CHECK(records(compiled, std::array{k}) == records(indexed, std::array{k}));
    CHECK(compiled.matches.size() == count);
    CHECK(compiled.candidates == 2 * count);
    CHECK(indexed.candidates >= count * count);
}

TEST_CASE("Compiled graph: opaque nodes retain Unknown through fallback") {
    Driver driver;
    auto& w  = driver.world();
    auto lam = w.mut_lam(w.pi(w.type_nat(), w.type_nat()));
    lam->set(false, lam->var());
    GraphView view({lam});
    PatternProgram p;
    GraphPattern graph(p);
    graph.anchor(p.node(Node::Lam, {p.any(), p.any()}), AnchorDomain::Nodes);
    auto found = CompiledGraphPattern(graph).find(view);
    CHECK(!found.used_compiled);
    CHECK(found.status == find_matches(graph, view).status);
    CHECK(found.status == SearchStatus::Unknown);
}

TEST_CASE("Compiled graph: guards with missing inputs never expose partial matches") {
    IR ir(Graph{1, {}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    auto missing = p.capture();
    GraphPattern graph(p);
    graph.anchor(p.where(p.any(), {missing}, [](auto) { return CheckResult::True; }));
    compare(graph, view, std::array{missing});
    auto found = CompiledGraphPattern(graph).find(view);
    CHECK(found.status == SearchStatus::Unknown);
    CHECK(found.matches.empty());
}

TEST_CASE("Compiled graph: a disproven type can reject an otherwise opaque candidate") {
    Driver driver;
    auto& w  = driver.world();
    auto lam = w.mut_lam(w.pi(w.type_nat(), w.type_nat()));
    lam->set(false, lam->var());
    GraphView view({lam});
    PatternProgram p;
    GraphPattern graph(p);
    graph.anchor(p.all(p.type(p.exact(w.type_nat())), p.node(Node::Lam, {p.any(), p.any()})), AnchorDomain::Nodes);
    // The interpreter stops at opaque structure before its deferred type check. The
    // compiled engine can prove failure from the type alone, without inspecting Lam operands.
    CHECK(find_matches(graph, view).status == SearchStatus::Unknown);
    auto found = CompiledGraphPattern(graph).find(view);
    CHECK(found.used_compiled);
    CHECK(found.status == SearchStatus::Complete);
    CHECK(found.matches.empty());
}

TEST_CASE("Compiled graph: plans can be reused but mutations invalidate them") {
    IR ir(Graph{2, {{0, 0, 1, 0}}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    GraphPattern graph(p);
    auto capture = p.capture();
    graph.anchor(capture);
    CompiledGraphPattern plan(graph);
    CHECK(plan.find(view).matches.size() == view.operations().size());
    CHECK(plan.find(view).matches.size() == view.operations().size());
    graph.allow_aliases();
    CHECK_THROWS_AS(plan.find(view), std::invalid_argument);
    CompiledGraphPattern aliases(graph);
    graph.anchor(capture);
    CHECK_THROWS_AS(aliases.find(view), std::invalid_argument);
    CompiledGraphPattern two(graph);
    p.any();
    CHECK_THROWS_AS(two.find(view), std::invalid_argument);
}

TEST_CASE("Compiled graph: work, depth, candidate and result limits remain explicit") {
    IR ir(Graph{2, {{0, 0, 1, 0}}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    Compiled pattern(Graph{1, {}}, ir);
    CompiledGraphPattern plan(pattern.graph);
    for (auto opts : {GraphMatchOptions{.match = {.max_steps = 0}}, GraphMatchOptions{.match = {.max_depth = 0}},
                      GraphMatchOptions{.max_candidates = 0}}) {
        auto found = plan.find(view, opts);
        CHECK(found.status == SearchStatus::Budget_Exceeded);
        CHECK(found.matches.empty());
    }
    auto limited = plan.find(view, {.max_results = 1});
    CHECK(limited.used_compiled);
    CHECK(limited.status == SearchStatus::Result_Limit);
    CHECK(limited.matches.size() == 1);
    CHECK(plan.find(view, {.max_results = 0}).status == SearchStatus::Result_Limit);
}

TEST_CASE("Compiled graph: semantic use relations preserve port and coverage constraints") {
    IR ir(Graph{
        3,
        {{0, 0, 1, 0}, {1, 0, 2, 1}}
    });
    PatternProgram p;
    auto producer = p.capture(p.node(Node::App, {p.exact(ir.outputs[0]), p.any()}));
    auto consumer = p.capture(p.node(Node::App, {p.exact(ir.vertex), p.any()}));
    GraphPattern pattern(p);
    auto from = pattern.anchor(producer), to = pattern.anchor(consumer);
    pattern.used_by(from, to, {.group = 0, .path = std::vector<size_t>{1}});
    pattern.only_used_by(from, to);
    for (auto coverage : {UseCoverage::Open, UseCoverage::Closed}) {
        GraphView view(ir.nodes, {.coverage = coverage, .operators = &ir.ops});
        compare(pattern, view, std::array{producer, consumer});
        auto found = CompiledGraphPattern(pattern).find(view);
        CHECK(found.matches.size() == (coverage == UseCoverage::Closed ? 1 : 0));
    }
}

TEST_CASE("Compiled graph: distinct and aliased anchors follow the same capture identity") {
    IR ir(Graph{2, {{0, 0, 1, 0}}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    auto value = p.capture();
    GraphPattern graph(p);
    graph.anchor(value);
    graph.anchor(value);
    compare(graph, view, std::array{value});
    CHECK(CompiledGraphPattern(graph).find(view).matches.empty());
    graph.allow_aliases();
    compare(graph, view, std::array{value});
    CHECK(CompiledGraphPattern(graph).find(view).matches.size() == view.operations().size());
}

TEST_CASE("Compiled graph: shared pattern expansion is bounded even without instructions") {
    IR ir(Graph{1, {}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    PatternProgram p;
    auto value = p.any();
    for (size_t i = 0; i < 20; ++i)
        value = p.all(value, value);
    GraphPattern graph(p);
    graph.anchor(value);
    CompiledGraphPattern plan(graph);
    CHECK(!plan.is_compiled());
    CHECK(plan.find(view, {.match = {.max_steps = 0}}).status == SearchStatus::Budget_Exceeded);
}

TEST_CASE("Compiled graph: a World swap invalidates plans, snapshots and returned matches") {
    IR ir(Graph{1, {}});
    GraphView view(ir.nodes, {.operators = &ir.ops});
    Compiled pattern(Graph{1, {}}, ir);
    CompiledGraphPattern plan(pattern.graph);
    auto found = plan.find(view);
    REQUIRE(found.matches.size() == 1);
    auto fresh = ir.w.inherit();
    swap(ir.w, *fresh);
    CHECK_THROWS_AS(plan.find(view), std::invalid_argument);
    CHECK_THROWS_AS(found.matches[0].reads(), std::invalid_argument);
}
