#include <array>
#include <set>
#include <stdexcept>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/phase.h>
#include <mim/world.h>

#include <mim/pattern/graph_rule.h>

using namespace mim;
using namespace mim::pattern;

namespace {
struct IR {
    Driver driver;
    World& w       = driver.world();
    const Def* nat = w.type_nat();
    const Axm *q = w.axm(w.pi(nat, nat)), *k = w.axm(w.pi(nat, nat)), *v = w.axm(w.pi(nat, nat));
    const Axm* combined = w.axm(w.pi(nat, w.arr(3, nat)));
    const Axm* pair     = w.axm(w.pi(nat, w.pi(nat, nat)));
    OpRegistry ops;
    IR() {
        for (auto axm : {q, k, v, combined})
            ops.add(axm->flags(), 1, {{0}}, true);
        ops.add(pair->flags(), 2, {{0}, {1}}, true);
    }
    const Def* call(const Def* op, const Def* x) { return w.app(op, x); }
    const Def* both(const Def* a, const Def* b) { return w.app(w.app(pair, a), b); }
    const Rule* rule(bool accept = true, bool identity = false) {
        auto r      = w.mut_rule(w.reform(nat));
        auto x      = r->var();
        auto lhs    = w.tuple({call(q, x), call(k, x), call(v, x)});
        auto shared = call(combined, x);
        auto rhs    = w.tuple({w.extract(shared, 3, 0), w.extract(shared, 3, 1), w.extract(shared, 3, 2)});
        return r->set(lhs, identity ? lhs : rhs, w.lit_bool(accept));
    }
};

std::set<std::vector<const Def*>> tuples(const GraphMatches& matches) {
    std::set<std::vector<const Def*>> result;
    for (auto& m : matches.matches)
        result.emplace(m.anchors().begin(), m.anchors().end());
    return result;
}
} // namespace

TEST_CASE("P5 range: prefix, bounds and repeated-capture identity") {
    IR ir;
    auto a = ir.w.axm(ir.nat), b = ir.w.axm(ir.nat), c = ir.w.axm(ir.nat);
    auto value = ir.w.tuple({a, b, c});
    PatternProgram p;
    auto pat = p.node_range(Node::Tuple, {p.exact(a)}, p.any(), 1, 3);
    CHECK(match(p, pat, value).status == MatchStatus::Matched);
    CHECK(match(p, p.node_range(Node::Tuple, {}, p.any(), 4, 7), value).status == MatchStatus::No_Match);
    auto same = p.capture();
    CHECK(match(p, p.node_range(Node::Tuple, {p.exact(a)}, same, 2, 2), ir.w.tuple({a, b, b})).status
          == MatchStatus::Matched);
    CHECK(match(p, p.node_range(Node::Tuple, {p.exact(a)}, same, 2, 2), value).status == MatchStatus::No_Match);
    CHECK_THROWS_AS(p.node_range(Node::Tuple, {}, p.any(), 3, 2), std::invalid_argument);
    PatternProgram foreign;
    CHECK_THROWS_AS(p.node_range(Node::Tuple, {}, foreign.any(), 0, 3), std::invalid_argument);
    CHECK(match(p, pat, value, {.max_steps = 1}).status == MatchStatus::Budget_Exceeded);
}

TEST_CASE("P5 join: indexed and exhaustive enumeration agree on shared-input triples") {
    IR ir;
    std::vector<const Def*> roots;
    for (size_t i = 0; i != 40; ++i) {
        auto x = ir.w.axm(ir.nat);
        for (auto op : {ir.q, ir.k, ir.v})
            roots.push_back(ir.call(op, x));
    }
    GraphView view({roots.data(), roots.size()}, {.operators = &ir.ops});
    PatternProgram p;
    auto x = p.capture();
    GraphPattern pattern(p);
    for (auto op : {ir.q, ir.k, ir.v})
        pattern.anchor(p.node(Node::App, {p.exact(op), x}));
    auto indexed   = find_matches(pattern, view);
    auto reference = find_matches(pattern, view, {.mode = GraphSearchMode::Exhaustive});
    REQUIRE(indexed.status == SearchStatus::Complete);
    REQUIRE(reference.status == SearchStatus::Complete);
    CHECK(indexed.matches.size() == 40);
    CHECK(tuples(indexed) == tuples(reference));
    CHECK(indexed.candidates * 5 < reference.candidates);
    CHECK(indexed.prefilter_checks < reference.prefilter_checks);
    auto reversed = find_matches(pattern, view, {.reverse_candidates = true});
    CHECK(tuples(reversed) == tuples(reference));
    auto bounded = find_matches(pattern, view, {.max_candidates = 200});
    CHECK(bounded.status == SearchStatus::Complete);
    CHECK(find_matches(pattern, view, {.max_candidates = 200, .mode = GraphSearchMode::Exhaustive}).status
          == SearchStatus::Budget_Exceeded);
}

TEST_CASE("P5 join: optional OR captures and type-only captures never imply raw ancestry") {
    IR ir;
    auto a = ir.w.axm(ir.nat), b = ir.w.axm(ir.nat);
    auto qa = ir.call(ir.q, a), kb = ir.call(ir.k, b), va = ir.call(ir.v, a);
    GraphView view({qa, kb, va}, {.operators = &ir.ops});
    PatternProgram p;
    auto x    = p.capture();
    auto type = p.capture();
    auto q    = p.node(Node::App, {p.exact(ir.q), p.either(x, p.any())});
    auto k    = p.node(Node::App, {p.exact(ir.k), x});
    GraphPattern pattern(p);
    pattern.anchor(p.all(q, p.type(type)));
    pattern.anchor(p.all(k, p.type(type)));
    auto ares = find_matches(pattern, view);
    auto bres = find_matches(pattern, view, {.mode = GraphSearchMode::Exhaustive});
    CHECK(tuples(ares) == tuples(bres));
    REQUIRE(ares.matches.size() == 1);
    CHECK(ares.matches[0].isa_capture(x) == b);
}

TEST_CASE("P5 graph Rule: tuple templates match independent roots and share RHS computation") {
    IR ir;
    auto rule = ir.rule();
    auto fn   = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("p5_graph_rule");
    auto x    = fn->var();
    fn->set(false, ir.both(ir.call(ir.q, x), ir.both(ir.call(ir.k, x), ir.call(ir.v, x))))->externalize();
    GraphRulePattern compiled(rule);
    auto gid = ir.w.curr_gid();
    GraphView view({fn->body()}, {.operators = &ir.ops, .scope = fn});
    auto found = compiled.find(view);
    REQUIRE(found.status == SearchStatus::Complete);
    REQUIRE(found.matches.size() == 1);
    CHECK(ir.w.curr_gid() == gid);
    auto plan = compiled.plan(found.matches[0], view);
    REQUIRE(plan.status == MatchStatus::Matched);
    std::array plans{plan.plan.get()};
    auto rewritten = rewrite_graph(view, plans);
    GraphView target({rewritten.scope()->body()}, {.operators = &ir.ops});
    size_t combined = 0;
    for (auto def : target.operations())
        if (std::get<0>(Axm::get(def))->flags() == ir.combined->flags()) ++combined;
    CHECK(combined == 1);
    rewritten.publish(view);
    CHECK_THROWS_AS(compiled.find(view), std::invalid_argument);
    {
        Cleanup cleanup(ir.w, "review_published_result");
        cleanup.run();
    }
    CHECK_THROWS_AS(rewritten.scope(), std::invalid_argument);
    CHECK_THROWS_AS(rewritten.roots(), std::invalid_argument);
    CHECK_THROWS_AS(rewritten.world(), std::invalid_argument);
    CHECK_THROWS_AS(rewritten.is_changed(), std::invalid_argument);
}

TEST_CASE("P5 graph Rule: rejected guards and identity templates do not publish") {
    IR ir;
    const Rule* rule = nullptr;
    SUBCASE("false guard") { rule = ir.rule(false); }
    SUBCASE("identity RHS") { rule = ir.rule(true, true); }
    auto x = ir.w.axm(ir.nat);
    GraphRulePattern compiled(rule);
    GraphView view({ir.call(ir.q, x), ir.call(ir.k, x), ir.call(ir.v, x)}, {.operators = &ir.ops});
    auto found = compiled.find(view);
    REQUIRE(found.matches.size() == 1);
    auto plan = compiled.plan(found.matches[0], view);
    CHECK_FALSE(plan.plan);
    CHECK(plan.status == MatchStatus::No_Match);
    CHECK(view.is_current());
}

TEST_CASE("P5 graph Rule: root limits and unsupported RHSs are explicit failures") {
    IR ir;
    auto rule = ir.rule();
    CHECK_THROWS_AS(GraphRulePattern(rule, 2), std::invalid_argument);
    auto extra = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat})));
    auto x = extra->var(2, 0), y = extra->var(2, 1);
    extra->set(ir.w.tuple({ir.call(ir.q, x), ir.call(ir.k, x)}), ir.w.tuple({y, y}), ir.w.lit_tt());
    GraphRulePattern compiled(extra);
    auto value = ir.w.axm(ir.nat);
    GraphView view({ir.call(ir.q, value), ir.call(ir.k, value)}, {.operators = &ir.ops});
    auto found = compiled.find(view);
    REQUIRE(found.matches.size() == 1);
    auto plan = compiled.plan(found.matches[0], view);
    CHECK(plan.status == MatchStatus::Unknown);
    CHECK_FALSE(plan.plan);
}

TEST_CASE("P5 graph Rule: captured boolean guards are evaluated in the isolated World") {
    IR ir;
    auto boolean = ir.w.type_bool();
    auto h       = ir.w.axm(ir.w.pi(ir.w.sigma({boolean, ir.nat}), ir.nat));
    ir.ops.add(h->flags(), 1,
               {
                   {0, 1}
    },
               true);
    auto rule      = ir.w.mut_rule(ir.w.reform(ir.w.sigma({boolean, ir.nat, ir.nat})));
    auto condition = rule->var(3, 0), value = rule->var(3, 1);
    rule->set(ir.w.tuple({ir.w.app(h, {condition, value}), ir.call(ir.k, value)}),
              ir.w.tuple({ir.w.lit_nat(0), ir.w.lit_nat(0)}), condition);
    const Def* flag      = nullptr;
    MatchStatus expected = MatchStatus::Unknown;
    SUBCASE("true") {
        flag     = ir.w.lit_tt();
        expected = MatchStatus::Matched;
    }
    SUBCASE("false") {
        flag     = ir.w.lit_ff();
        expected = MatchStatus::No_Match;
    }
    SUBCASE("symbolic") { flag = ir.w.axm(boolean); }
    auto x = ir.w.axm(ir.nat);
    GraphView view({ir.w.app(h, {flag, x}), ir.call(ir.k, x)}, {.operators = &ir.ops});
    GraphRulePattern compiled(rule);
    auto found = compiled.find(view);
    REQUIRE(found.matches.size() == 1);
    auto gid  = ir.w.curr_gid();
    auto plan = compiled.plan(found.matches[0], view);
    CHECK(plan.status == expected);
    CHECK(bool(plan.plan) == (expected == MatchStatus::Matched));
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Review lifetime: single-root results and graph collections reject rebuilt storage") {
    IR ir;
    auto x     = ir.w.axm(ir.nat);
    auto value = ir.call(ir.q, x);
    PatternProgram p;
    auto capture = p.capture();
    auto single  = match(p, capture, value);
    GraphView view({value}, {.operators = &ir.ops});
    GraphPattern graph(p);
    graph.anchor(capture);
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    {
        Cleanup cleanup(ir.w, "review_lifetime");
        cleanup.run();
    }
    CHECK_THROWS_AS(single.result->isa_capture(capture), std::invalid_argument);
    CHECK_THROWS_AS(view.nodes(), std::invalid_argument);
    CHECK_THROWS_AS(view.roots(), std::invalid_argument);
    CHECK_THROWS_AS(view.operations(), std::invalid_argument);
    CHECK_THROWS_AS(view.scope(), std::invalid_argument);
    CHECK_THROWS_AS(found.matches[0].anchors(), std::invalid_argument);
    CHECK_THROWS_AS(found.matches[0].reads(), std::invalid_argument);
    CHECK_THROWS_AS(found.matches[0].nodes(), std::invalid_argument);
}

TEST_CASE("Review reads: native conditions nested in NOT participate in conflicts") {
    IR ir;
    auto x = ir.w.axm(ir.nat);
    auto a = ir.call(ir.q, x), b = ir.call(ir.k, x);
    GraphView view({a, b}, {.operators = &ir.ops});
    PatternProgram p;
    GraphPattern graph(p);
    graph.anchor(p.all(p.exact(b), p.negate(p.where(p.any(), {}, [](auto) { return CheckResult::False; }))));
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto reads = found.matches[0].reads();
    CHECK(std::ranges::find(reads, a) != reads.end());
    RewritePlan first(found.matches[0]);
    first.replace(b, first.literal(ir.nat, 0));
    GraphPattern other(p);
    other.anchor(p.exact(a));
    auto matcha = find_matches(other, view);
    RewritePlan second(matcha.matches[0]);
    second.replace(a, second.literal(ir.nat, 1));
    std::array plans{&first, &second};
    CHECK_THROWS_AS(rewrite_graph(view, plans), std::invalid_argument);
}

TEST_CASE("Review budget: zero-output nested wrapper expansion has bounded work") {
    IR ir;
    const Def *a = ir.w.axm(ir.nat), *b = ir.w.axm(ir.nat);
    for (size_t i = 0; i != 40; ++i) {
        auto next = ir.w.tuple({a, b});
        b         = ir.w.tuple({b, a});
        a         = next;
    }
    auto op = ir.w.axm(ir.w.pi(a->type(), ir.nat));
    OpRegistry ops;
    ops.add(op->flags(), 1,
            {
                {0, 41}
    },
            false);
    auto value   = ir.call(op, a);
    auto decoded = ops.decode(value, 64);
    REQUIRE(decoded);
    CHECK_FALSE(decoded->complete);
    CHECK(decoded->inputs.empty());
}

TEST_CASE("Review budget: unrelated external imports obey source and depth limits") {
    IR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("review_small");
    f->set(false, ir.call(ir.q, f->var()))->externalize();
    auto g          = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("review_deep");
    const Def* body = g->var();
    for (size_t i = 0; i != 60; ++i)
        body = ir.call(ir.q, body);
    g->set(false, body)->externalize();
    GraphView view({f->body()}, {.operators = &ir.ops, .scope = f});
    PatternProgram p;
    GraphPattern graph(p);
    auto a     = graph.anchor(p.any());
    auto found = find_matches(graph, view);
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(a), plan.literal(ir.nat, 0));
    std::array plans{&plan};
    CHECK_THROWS_AS(rewrite_graph(view, plans, {.max_depth = 12}), std::invalid_argument);
    CHECK_THROWS_AS(rewrite_graph(view, plans, {.max_source_nodes = 30}), std::invalid_argument);
    CHECK(view.is_current());
}

TEST_CASE("Review purity: a pure parent cannot erase an unproved computation") {
    IR ir;
    auto tick = ir.w.axm(ir.w.pi(ir.nat, ir.nat));
    auto x    = ir.w.axm(ir.nat);
    auto side = ir.call(tick, x), outer = ir.call(ir.q, side);
    GraphView view({outer}, {.operators = &ir.ops});
    PatternProgram p;
    GraphPattern graph(p);
    auto a     = graph.anchor(p.exact(outer));
    auto found = find_matches(graph, view);
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(a), plan.literal(ir.nat, 0));
    std::array plans{&plan};
    CHECK_THROWS_AS(rewrite_graph(view, plans), std::invalid_argument);
    RewritePlan keep(found.matches[0]);
    keep.replace(outer, keep.import(side));
    std::array kept{&keep};
    CHECK_NOTHROW(rewrite_graph(view, kept));
}

TEST_CASE("Review literals: equal result types do not legalize aggregate bit literals") {
    IR ir;
    auto value = ir.call(ir.combined, ir.w.axm(ir.nat));
    GraphView view({value}, {.operators = &ir.ops});
    PatternProgram p;
    GraphPattern graph(p);
    graph.anchor(p.exact(value));
    auto found = find_matches(graph, view);
    RewritePlan plan(found.matches[0]);
    plan.replace(value, plan.literal(value->type(), 0));
    std::array plans{&plan};
    CHECK_THROWS_AS(rewrite_graph(view, plans), std::invalid_argument);
    CHECK(view.is_current());
}

TEST_CASE("Review dependencies: simultaneous replacement cuts old operand dependencies") {
    IR ir;
    auto first = ir.call(ir.q, ir.w.axm(ir.nat)), second = ir.call(ir.k, first);
    GraphView view({second}, {.operators = &ir.ops});
    PatternProgram p;
    auto firstcap = p.capture(p.exact(first));
    GraphPattern graph(p);
    auto a     = graph.anchor(firstcap);
    auto b     = graph.anchor(p.node(Node::App, {p.exact(ir.k), firstcap}));
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(a), plan.import(second));
    plan.replace(found.matches[0].at(b), plan.literal(ir.nat, 0));
    std::array plans{&plan};
    auto result = rewrite_graph(view, plans);
    CHECK(Lit::isa(result.roots()[0]) == 0);
}

TEST_CASE("Review OR: successful anchor tuples dominate unknown alternatives") {
    IR ir;
    auto value = ir.call(ir.q, ir.w.axm(ir.nat));
    GraphView view({value}, {.operators = &ir.ops});
    PatternProgram p;
    GraphPattern graph(p);
    auto unknown = p.where(p.any(), {}, [](auto) { return CheckResult::Unknown; });
    graph.anchor(p.either(p.any(), unknown));
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive}) {
        auto found = find_matches(graph, view, {.mode = mode});
        CHECK(found.status == SearchStatus::Complete);
        CHECK(found.matches.size() == 1);
    }
    auto other = ir.call(ir.q, ir.w.axm(ir.nat));
    GraphView two({value, other}, {.operators = &ir.ops});
    GraphPattern partial(p);
    partial.anchor(p.either(p.exact(value), unknown));
    auto found = find_matches(partial, two);
    CHECK(found.matches.size() == 1);
    CHECK(found.status == SearchStatus::Unknown); // An unresolved different tuple stays unresolved.
}

TEST_CASE("Review join: skipped mutable operands preserve indexed and exhaustive uncertainty") {
    IR ir;
    auto a      = ir.w.axm(ir.nat);
    auto qa     = ir.call(ir.q, a);
    auto opaque = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    opaque->set(false, a);
    const Def* second  = opaque;
    const Axm* wrapper = nullptr;
    SUBCASE("mutable anchor is the exact review trigger") {}
    SUBCASE("mutable boundary under an immutable caller also stays possible") {
        wrapper = ir.w.axm(ir.w.pi(opaque->type(), ir.nat));
        ir.ops.add(wrapper->flags(), 1, {{0}}, true);
        second = ir.call(wrapper, opaque);
    }
    GraphView view({qa, second}, {.operators = &ir.ops});
    CHECK(view.uses(a).size() == 1);
    PatternProgram p;
    auto x = p.capture();
    GraphPattern graph(p);
    graph.anchor(p.node(Node::App, {p.exact(ir.q), x}));
    auto lam = p.node(Node::Lam, {p.any(), x});
    if (wrapper)
        graph.anchor(p.node(Node::App, {p.exact(wrapper), lam}));
    else
        graph.anchor(lam, AnchorDomain::Nodes);
    auto gid = ir.w.curr_gid();
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool reverse : {false, true}) {
            auto found = find_matches(graph, view, {.reverse_candidates = reverse, .mode = mode});
            CHECK(found.status == SearchStatus::Unknown);
            CHECK(found.matches.empty());
        }
    CHECK(ir.w.curr_gid() == gid);
    CHECK(opaque->body() == a);
    CHECK(view.is_current());
}

TEST_CASE("Review opacity: mutable arity never proves structural mismatch") {
    IR ir;
    auto opaque = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    opaque->set(false, ir.w.lit_nat(0));
    GraphView view({opaque}, {.operators = &ir.ops});
    PatternProgram p;
    for (auto structural : {p.node(Node::Lam, {}), p.node_range(Node::Lam, {}, p.any(), 3, 4)}) {
        CHECK(match(p, structural, opaque).status == MatchStatus::Unknown);
        GraphPattern graph(p);
        graph.anchor(structural, AnchorDomain::Nodes);
        for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
            for (bool reverse : {false, true}) {
                auto found = find_matches(graph, view, {.reverse_candidates = reverse, .mode = mode});
                CHECK(found.status == SearchStatus::Unknown);
                CHECK(found.matches.empty());
            }
    }
    CHECK(view.is_current());
}
