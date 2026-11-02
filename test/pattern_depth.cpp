#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph_match.h>
#include <mim/pattern/match.h>

using namespace mim;
using namespace mim::pattern;

TEST_CASE("Pattern: wide operands do not consume nesting depth") {
    Driver driver;
    auto& w    = driver.world();
    auto nat   = w.type_nat();
    auto unary = w.axm(w.pi(nat, nat));
    for (size_t count : {32, 256, 2048}) {
        PatternProgram p;
        std::vector<Pattern> patterns;
        DefVec values;
        for (size_t i = 0; i != count; ++i) {
            auto value = w.lit_nat(i);
            patterns.push_back(p.node(Node::App, {p.exact(unary), p.exact(value)}));
            values.push_back(w.app(unary, value));
        }
        auto pattern = p.node(Node::Tuple, patterns);
        auto value   = w.tuple(values);
        auto outcome = match(p, pattern, value, {.max_depth = 3});
        INFO(count);
        CHECK(outcome.status == MatchStatus::Matched);
        CHECK(outcome.steps == 1 + 3 * count);
        auto limited = match(p, pattern, value, {.max_steps = 3 * count});
        CHECK(limited.status == MatchStatus::Budget_Exceeded);
        CHECK(limited.steps == 3 * count);
        CHECK_FALSE(limited.result);
    }
}

TEST_CASE("Pattern: deep structures still exhaust the nesting budget") {
    Driver driver;
    auto& w    = driver.world();
    auto unary = w.axm(w.pi(w.type_nat(), w.type_nat()));
    PatternProgram p;
    auto pattern     = p.any();
    const Def* value = w.lit_nat_0();
    for (size_t i = 0; i != 512; ++i) {
        pattern = p.node(Node::App, {p.exact(unary), pattern});
        value   = w.app(unary, value);
    }
    auto outcome = match(p, pattern, value);
    CHECK(outcome.status == MatchStatus::Budget_Exceeded);
    CHECK_FALSE(outcome.result);
    CHECK(outcome.steps == 511);
}

TEST_CASE("Pattern: wide choices backtrack across siblings without recursive frames") {
    Driver driver;
    auto& w    = driver.world();
    auto unary = w.axm(w.pi(w.type_nat(), w.type_nat()));
    PatternProgram p;
    std::vector<Pattern> patterns, captures;
    DefVec values;
    for (size_t i = 0; i != 1024; ++i) {
        auto capture = p.capture();
        auto inside  = p.node(Node::App, {p.exact(unary), capture});
        patterns.push_back(p.either(capture, inside));
        patterns.push_back(capture);
        captures.push_back(capture);
        auto value = w.lit_nat(i);
        values.push_back(w.app(unary, value));
        values.push_back(value);
    }
    auto outcome = match(p, p.node(Node::Tuple, patterns), w.tuple(values), {.max_depth = 8});
    REQUIRE(outcome.status == MatchStatus::Matched);
    for (size_t i = 0; i != captures.size(); ++i)
        CHECK(outcome.result->isa_capture(captures[i]) == w.lit_nat(i));
}

TEST_CASE("Pattern: wide range and deferred obligations keep sibling depth constant") {
    Driver driver;
    auto& w = driver.world();
    PatternProgram p;
    auto capture = p.capture();
    auto typed   = p.all(p.any(), p.type(p.exact(w.type_nat())));
    auto checked = p.where(typed, {capture},
                           [](auto args) { return Lit::isa(args[0]) == 1 ? CheckResult::True : CheckResult::False; });
    auto tail    = p.all(checked, p.negate(p.literal(w.type_nat(), 0)));
    DefVec values;
    for (size_t i = 0; i != 2048; ++i)
        values.push_back(w.lit_nat(i + 1));
    auto pattern = p.node_range(Node::Tuple, {capture}, tail, 2047, 2047);
    auto outcome = match(p, pattern, w.tuple(values), {.max_depth = 8});
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(capture) == w.lit_nat_1());
}

TEST_CASE("Pattern: wide binder operands do not retain native frames") {
    Driver driver;
    auto& w = driver.world();
    PatternProgram p;
    auto binder = p.binder(p.any(), p.exact(w.lit_ff()), p.any());
    std::vector<Pattern> patterns;
    DefVec values;
    for (size_t i = 0; i != 256; ++i) {
        auto lam = w.mut_lam(w.pi(w.type_nat(), w.type_nat()));
        lam->set(false, w.lit_nat(i));
        values.push_back(lam);
        patterns.push_back(binder);
    }
    CHECK(match(p, p.node(Node::Tuple, patterns), w.tuple(values), {.max_depth = 3}).status == MatchStatus::Matched);
}

TEST_CASE("Pattern: graph search matches wide structures in both enumeration modes") {
    Driver driver;
    auto& w = driver.world();
    PatternProgram p;
    std::vector<Pattern> patterns;
    DefVec values;
    for (size_t i = 0; i != 2048; ++i) {
        values.push_back(w.lit_nat(i));
        patterns.push_back(p.literal(w.type_nat(), i));
    }
    auto tuple = w.tuple(values);
    GraphView view({tuple});
    GraphPattern graph(p);
    auto anchor = graph.anchor(p.node(Node::Tuple, patterns), AnchorDomain::Nodes);
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive}) {
        auto outcome = find_matches(graph, view, {.match = {.max_depth = 3}, .mode = mode});
        CHECK(outcome.status == SearchStatus::Complete);
        REQUIRE(outcome.matches.size() == 1);
        CHECK(outcome.matches.front().at(anchor) == tuple);
    }
}
