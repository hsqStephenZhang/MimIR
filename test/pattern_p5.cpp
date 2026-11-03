#include <array>
#include <set>
#include <stdexcept>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/phase.h>
#include <mim/world.h>

#include <mim/pattern/graph_rule.h>
#include <mim/pattern/guard.h>
#include <mim/pattern/rule.h>

using namespace mim;
using namespace mim::pattern;

namespace {
thread_local size_t witness_calls = 0;

WitnessResult test_witness(const GraphView& graph, Defs inputs, World& target) {
    ++witness_calls;
    if (&inputs[0]->world() != &graph.world()) return {CheckResult::Unknown, nullptr, "input was not frozen"};
    auto value = Lit::as(inputs[0]);
    if (value == 8) return {CheckResult::Unknown, nullptr, "test witness is unknown"};
    if (value == 9) return {CheckResult::False, nullptr, "test witness disproved"};
    if (value == 10) return {CheckResult::True, target.lit_nat(0), {}};
    if (value == 11) return {CheckResult::True, target.mut_lam(target.pi(target.type_nat(), target.type_nat())), {}};
    return {CheckResult::True, target.lit_tt(), {}};
}

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

TEST_CASE("P5 search: rejected matches do not consume the accepted result limit") {
    IR ir;
    DefVec roots;
    for (size_t i = 0; i != 8; ++i)
        roots.push_back(ir.call(ir.q, ir.w.lit_nat(i)));
    GraphView view(roots, {.operators = &ir.ops});
    PatternProgram p;
    auto x = p.capture();
    GraphPattern pattern(p);
    auto anchor    = pattern.anchor(p.node(Node::App, {p.exact(ir.q), x}));
    size_t visited = 0;
    GraphMatchOptions options;
    options.max_results = 1;
    options.accept      = [&](const GraphMatch&) { return ++visited == roots.size(); };
    auto found          = find_matches(pattern, view, options);
    CHECK(found.status == SearchStatus::Result_Limit);
    CHECK(visited == roots.size());
    REQUIRE(found.matches.size() == 1);
    CHECK(found.matches[0].at(anchor) == roots.back());
    CHECK(view.is_current());

    visited        = 0;
    options.accept = [&](const GraphMatch&) {
        ++visited;
        return false;
    };
    found = find_matches(pattern, view, options);
    CHECK(found.status == SearchStatus::Complete);
    CHECK(found.matches.empty());
    CHECK(visited == roots.size());

    options.max_candidates = 2;
    found                  = find_matches(pattern, view, options);
    CHECK(found.status == SearchStatus::Budget_Exceeded);
    CHECK(found.matches.empty());
}

TEST_CASE("P5 search: rejected orientation preserves alternative bindings and Unknown") {
    IR ir;
    auto a = ir.w.lit_nat(1), b = ir.w.lit_nat(2);
    auto value = ir.both(a, b);
    GraphView view({value}, {.operators = &ir.ops});
    PatternProgram p;
    auto x = p.capture(), y = p.capture();
    auto pair = [&](Pattern first, Pattern second) {
        return p.node(Node::App, {p.node(Node::App, {p.exact(ir.pair), first}), second});
    };
    auto unknown = p.where(pair(x, y), {x}, [](auto) { return CheckResult::Unknown; });
    GraphPattern pattern(p);
    pattern.anchor(p.either(unknown, p.either(pair(x, y), pair(y, x))));
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive}) {
        size_t visits = 0;
        GraphMatchOptions options;
        options.mode   = mode;
        options.accept = [&](const GraphMatch& match) {
            ++visits;
            return match.isa_capture(x) == b;
        };
        auto found = find_matches(pattern, view, options);
        CHECK(found.status == SearchStatus::Complete);
        CHECK(visits == 2);
        REQUIRE(found.matches.size() == 1);
        CHECK(found.matches[0].isa_capture(x) == b);
        CHECK(found.matches[0].isa_capture(y) == a);
        options.accept = [](const GraphMatch&) { return false; };
        found          = find_matches(pattern, view, options);
        CHECK(found.status == SearchStatus::Unknown);
        CHECK(found.matches.empty());
    }
}

TEST_CASE("Rule types: imported dependent Pi binders match without inference") {
    IR ir;
    auto type = ir.w.mut_pi(ir.w.type());
    type->set_dom(ir.nat);
    type->set_codom(ir.w.arr(type->var(), ir.nat));
    auto op   = ir.w.axm(type);
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto lhs  = ir.call(op, rule->var());
    rule->set(lhs, lhs, ir.w.lit_tt());
    auto templates = ir.w.inherit();
    Rewriter save(*templates);
    auto saved = save.rewrite(rule);
    Rewriter restore(ir.w);
    auto copied = restore.rewrite(saved)->as<Rule>();
    REQUIRE(copied->lhs()->as<App>()->callee()->type() != type);
    RulePattern compiled(copied);
    auto candidate = ir.call(op, ir.w.lit_nat(2));
    auto gid       = ir.w.curr_gid();
    auto matched   = compiled.match_lhs(candidate);
    CHECK(matched.status == MatchStatus::Matched);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(compiled.match_lhs(ir.call(ir.q, ir.w.lit_nat(2))).status == MatchStatus::No_Match);
}

TEST_CASE("Graph Rule: type-only dimensions and element type construct a shared RHS") {
    Driver driver;
    auto& w            = driver.world();
    const Def* nat     = w.type_nat();
    const Def* boolean = w.type_bool();
    bool symbolic      = false;
    SUBCASE("literal dimensions") {}
    SUBCASE("symbolic dimensions without equation solving") { symbolic = true; }

    auto dimension
        = [&](u64 value) -> const Def* { return symbolic ? static_cast<const Def*>(w.axm(nat)) : w.lit_nat(value); };
    auto m = dimension(2), k = dimension(3), n = dimension(4);
    for (auto element : {nat, boolean}) {
        auto matrix
            = [&](const Def* rows, const Def* columns, const Def* type) { return w.arr(Defs{rows, columns}, type); };
        auto at = matrix(m, k, element), bt = matrix(k, n, element);
        auto make_a = w.axm(w.pi(nat, at)), make_b = w.axm(w.pi(w.type_bool(), bt));
        // Distinct inputs; the only joins are k and T in the result types.
        auto a = w.app(make_a, w.axm(nat)), b = w.app(make_b, w.axm(w.type_bool()));

        auto signature = w.mut_pi(w.type<1>());
        signature->set_dom(w.sigma({w.type(), nat, nat, nat}));
        auto mt = signature->var(4, 0), mm = signature->var(4, 1);
        auto mk = signature->var(4, 2), mn = signature->var(4, 3);
        signature->set_codom(w.sigma({matrix(mm, mk, mt), matrix(mk, mn, mt)}));
        auto summary = w.axm(signature);

        auto parameters = w.mut_sigma<1>(6);
        parameters->set(0, w.type())->set(1, nat)->set(2, nat)->set(3, nat);
        auto pt = parameters->var(6, 0), pm = parameters->var(6, 1);
        auto pk = parameters->var(6, 2), pn = parameters->var(6, 3);
        parameters->set(4, matrix(pm, pk, pt))->set(5, matrix(pk, pn, pt));
        auto rule = w.mut_rule(w.reform(parameters));
        auto rt = rule->var(6, 0), rm = rule->var(6, 1);
        auto rk = rule->var(6, 2), rn = rule->var(6, 3);
        rule->set(w.tuple({rule->var(6, 4), rule->var(6, 5)}), w.app(summary, w.tuple({rt, rm, rk, rn})), w.lit_tt());

        RulePattern single(rule);
        auto candidate  = w.tuple({a, b});
        auto uncaptured = w.lit_nat(99);
        auto gid        = w.curr_gid();
        auto matched    = single.match_lhs(candidate);
        REQUIRE(matched.status == MatchStatus::Matched);
        CHECK(matched.bindings.at(rt) == element);
        CHECK(matched.bindings.at(rm) == m);
        CHECK(matched.bindings.at(rk) == k);
        CHECK(matched.bindings.at(rn) == n);
        CHECK(w.curr_gid() == gid);

        OpRegistry ops;
        ops.add(make_a->flags(), 1, {{0}}, true);
        ops.add(make_b->flags(), 1, {{0}}, true);
        ops.add(summary->flags(), 1, {}, true);
        GraphView view({a, b}, {.operators = &ops});
        for (auto capture : {element, m, k, n})
            CHECK_FALSE(view.is_contained(capture));
        GraphRulePattern graph(rule);
        for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
            for (bool reverse : {false, true}) {
                auto found = graph.find(view, {.reverse_candidates = reverse, .mode = mode});
                REQUIRE(found.status == SearchStatus::Complete);
                REQUIRE(found.matches.size() == 1);
                auto planned = graph.plan(found.matches.front(), view);
                REQUIRE(planned.status == MatchStatus::Matched);
                REQUIRE(planned.plan);
                std::array plans{planned.plan.get()};
                auto rebuilt = rewrite_graph(view, plans);
                REQUIRE(rebuilt.roots().size() == 2);
                auto first = rebuilt.roots()[0]->as<Extract>(), second = rebuilt.roots()[1]->as<Extract>();
                CHECK(first->tuple() == second->tuple());
                auto args = first->tuple()->as<App>()->arg()->as<Tuple>();
                CHECK(args->op(0) == (element == nat ? rebuilt.world().type_nat() : rebuilt.world().type_bool()));
                std::array dimensions{m, k, n};
                for (size_t i = 0; i != dimensions.size(); ++i) {
                    CHECK(args->op(i + 1)->node() == dimensions[i]->node());
                    CHECK(args->op(i + 1)->flags() == dimensions[i]->flags());
                    CHECK(args->op(i + 1)->type() == rebuilt.world().type_nat());
                }
                CHECK(rebuilt.is_changed());
                CHECK(view.is_current());
                CHECK(w.curr_gid() == gid);

                // Captured type dependencies are allowed; unrelated graph-external values are not.
                RewritePlan bad(found.matches.front());
                auto metadata = bad.tuple({bad.import(element), bad.import(m), bad.import(k), bad.import(uncaptured)});
                bad.replace(a, bad.extract(bad.call(summary, {metadata}), 0));
                std::array rejected{&bad};
                CHECK_THROWS_AS(rewrite_graph(view, rejected), std::invalid_argument);
                CHECK(w.curr_gid() == gid);
            }

        auto other_k = dimension(5);
        for (auto bad_type : {matrix(other_k, n, element), matrix(k, n, element == nat ? boolean : nat)}) {
            auto bad_op = w.axm(w.pi(w.type_bool(), bad_type));
            ops.add(bad_op->flags(), 1, {{0}}, true);
            GraphView bad({a, w.app(bad_op, w.axm(w.type_bool()))}, {.operators = &ops});
            CHECK(graph.find(bad).matches.empty());
            CHECK(graph.find(bad, {.mode = GraphSearchMode::Exhaustive}).matches.empty());
        }
    }
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

TEST_CASE("P5 witnesses: one frozen evaluation feeds the guard and both RHS frontends") {
    Driver driver;
    auto& w     = driver.world();
    auto op     = w.axm(w.pi(w.type_nat(), w.type_bool()));
    auto native = w.axm(w.pi(w.type_nat(), w.type_bool()));
    driver.register_witness(native->flags(), {1, test_witness});
    CHECK(driver.normalizer(native->flags()) == nullptr);
    OpRegistry ops;
    ops.add(op->flags(), 1, {{0}}, true);
    auto rule    = w.mut_rule(w.reform(w.type_nat()));
    auto witness = w.app(native, rule->var());
    rule->set(w.app(op, rule->var()), witness, witness);
    auto source = w.app(op, w.lit_nat(7));
    GraphView graph({source}, {.operators = &ops});
    GuardContext guards(graph);
    auto gid = w.curr_gid();

    auto unavailable = w.inherit();
    witness_calls    = 0;
    CHECK(RulePattern(rule).build_replacement(source, *unavailable).status == MatchStatus::Unknown);
    CHECK(witness_calls == 0);

    witness_calls = 0;
    auto target   = w.inherit();
    Rewriter importer(*target);
    auto replacement = RulePattern(rule).build_replacement(source, importer, {}, &guards);
    REQUIRE(replacement.status == MatchStatus::Matched);
    CHECK(replacement.value == target->lit_tt());
    CHECK(witness_calls == 1);
    CHECK(w.curr_gid() == gid);
    CHECK(graph.is_current());

    witness_calls = 0;
    GraphRulePattern pattern(rule);
    auto found = pattern.find(graph);
    REQUIRE(found.matches.size() == 1);
    auto plan = pattern.plan(found.matches[0], graph, &guards);
    REQUIRE(plan.status == MatchStatus::Matched);
    REQUIRE(plan.plan);
    CHECK(witness_calls == 1);
    std::array plans{plan.plan.get()};
    auto rebuilt = rewrite_graph(graph, plans);
    CHECK(rebuilt.roots()[0] == rebuilt.world().lit_tt());
    CHECK(witness_calls == 1);
    CHECK(w.curr_gid() == gid);
    CHECK(graph.is_current());
}

TEST_CASE("P5 witnesses: Unknown, disproved and invalid results never fire a Rule") {
    Driver driver;
    auto& w     = driver.world();
    auto op     = w.axm(w.pi(w.type_nat(), w.type_bool()));
    auto native = w.axm(w.pi(w.type_nat(), w.type_bool()));
    driver.register_witness(native->flags(), {1, test_witness});
    OpRegistry ops;
    ops.add(op->flags(), 1, {{0}}, true);
    auto rule = w.mut_rule(w.reform(w.type_nat()));
    rule->set(w.app(op, rule->var()), w.app(native, rule->var()), w.lit_tt());
    for (u64 value : {8, 9, 10, 11}) {
        auto source = w.app(op, w.lit_nat(value));
        GraphView graph({source}, {.operators = &ops});
        GuardContext guards(graph);
        auto gid      = w.curr_gid();
        auto expected = value == 9 ? MatchStatus::No_Match : MatchStatus::Unknown;
        auto target   = w.inherit();
        Rewriter importer(*target);
        witness_calls    = 0;
        auto replacement = RulePattern(rule).build_replacement(source, importer, {}, &guards);
        CHECK(replacement.status == expected);
        CHECK_FALSE(replacement.reason.empty());
        CHECK(replacement.value == nullptr);
        CHECK(witness_calls == 1);
        GraphRulePattern pattern(rule);
        auto found = pattern.find(graph);
        REQUIRE(found.matches.size() == 1);
        witness_calls = 0;
        auto plan     = pattern.plan(found.matches[0], graph, &guards);
        CHECK(plan.status == expected);
        CHECK_FALSE(plan.reason.empty());
        CHECK_FALSE(plan.plan);
        CHECK(witness_calls == 1);
        CHECK(w.curr_gid() == gid);
        CHECK(graph.is_current());
    }
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

TEST_CASE("Review budget: unrelated deep external imports obey work limits without a depth cliff") {
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
    CHECK_NOTHROW(rewrite_graph(view, plans, {.max_depth = 12}));
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

TEST_CASE("Graph purity: local lambda parameters are parametric but free calls remain protected") {
    IR ir;
    auto callback = ir.w.pi(ir.nat, ir.nat);
    auto closure  = ir.w.mut_lam(ir.w.pi(callback, ir.nat));
    closure->set(false, ir.w.app(closure->var(), ir.w.lit_nat(3)));
    auto scope    = ir.w.mut_lam(ir.w.pi(ir.w.sigma({ir.nat, callback}), ir.nat));
    auto captured = ir.w.mut_lam(closure->type());
    captured->set(false, ir.w.app(captured->var(), scope->var(2, 0)));
    const Def* input = closure;
    bool allowed     = true;
    SUBCASE("closed closure value") {}
    SUBCASE("open closure value") { input = captured; }
    SUBCASE("nested closure calls outer parameter") {
        auto outer = ir.w.mut_lam(ir.w.pi(callback, callback));
        auto inner = ir.w.mut_lam(callback);
        inner->set(false, ir.w.app(outer->var(), inner->var()));
        outer->set(false, inner);
        input = outer;
    }
    SUBCASE("closure calls free unknown function") {
        auto free = ir.w.mut_lam(callback);
        free->set(false, ir.w.app(scope->var(2, 1), free->var()));
        input   = free;
        allowed = false;
    }
    SUBCASE("closure calls projected local parameter") {
        auto local = ir.w.mut_lam(ir.w.pi(ir.w.sigma({callback, ir.nat}), ir.nat));
        local->set(false, ir.w.app(local->var(2, 0), local->var(2, 1)));
        input = local;
    }
    SUBCASE("closure calls unregistered axiom") {
        auto effect    = ir.w.axm(callback);
        auto effectful = ir.w.mut_lam(callback);
        effectful->set(false, ir.w.app(effect, effectful->var()));
        input   = effectful;
        allowed = false;
    }
    SUBCASE("evaluated unknown variable application") {
        input   = ir.w.app(scope->var(2, 1), ir.w.lit_nat(3));
        allowed = false;
    }
    SUBCASE("evaluated closure calls unknown function") {
        auto unknown = ir.w.axm(callback);
        input        = ir.w.app(closure, unknown);
        allowed      = false;
    }
    SUBCASE("closure passed to unregistered axiom") {
        auto unknown = ir.w.axm(ir.w.pi(closure->type(), ir.nat));
        input        = ir.w.app(unknown, closure);
        allowed      = false;
    }
    SUBCASE("closure passed to registered impure axiom") {
        auto impure = ir.w.axm(ir.w.pi(closure->type(), ir.nat));
        ir.ops.add(impure->flags(), 1, {{0}}, false);
        input   = ir.w.app(impure, closure);
        allowed = false;
    }
    auto parent = ir.w.axm(ir.w.pi(input->type(), ir.nat));
    ir.ops.add(parent->flags(), 1, {{0}}, true);
    auto outer = ir.w.app(parent, input);
    scope->set(false, outer);
    GraphView view({outer}, {.operators = &ir.ops, .scope = scope});
    PatternProgram p;
    GraphPattern graph(p);
    graph.anchor(p.exact(outer));
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    RewritePlan plan(found.matches[0]);
    plan.replace(outer, plan.literal(ir.nat, 0));
    std::array plans{&plan};
    if (allowed)
        CHECK_NOTHROW(rewrite_graph(view, plans));
    else
        CHECK_THROWS_AS(rewrite_graph(view, plans), PurityFailure);
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

TEST_CASE("Graph purity: captured functions are callable when their evaluated calls are pure") {
    IR ir;
    auto callback    = ir.w.pi(ir.nat, ir.nat);
    auto effect      = ir.w.axm(callback);
    auto scope       = ir.w.mut_lam(ir.w.pi(ir.w.sigma({ir.nat, callback}), ir.nat));
    auto fn          = ir.w.mut_lam(callback);
    const Def* input = fn;
    bool allowed     = true;
    SUBCASE("registered pure call on the parameter") { fn->set(false, ir.call(ir.q, fn->var())); }
    SUBCASE("impure outer value is not evaluated by the call") {
        fn->set(false, ir.call(ir.q, ir.w.app(effect, scope->var(2, 0))));
    }
    SUBCASE("own function parameter") {
        auto higher = ir.w.mut_lam(ir.w.pi(callback, ir.nat));
        higher->set(false, ir.w.app(higher->var(), ir.w.lit_nat(3)));
        input = higher;
    }
    SUBCASE("nested lambda calls the outer parameter") {
        auto outer = ir.w.mut_lam(ir.w.pi(callback, callback));
        auto inner = ir.w.mut_lam(callback);
        inner->set(false, ir.w.app(outer->var(), inner->var()));
        outer->set(false, inner);
        input = outer;
    }
    SUBCASE("free unknown function") {
        fn->set(false, ir.w.app(scope->var(2, 1), fn->var()));
        allowed = false;
    }
    SUBCASE("unregistered axiom on the parameter") {
        fn->set(false, ir.w.app(effect, fn->var()));
        allowed = false;
    }
    SUBCASE("not a lambda") {
        fn->set(false, fn->var());
        input   = effect;
        allowed = false;
    }
    GraphView view({ir.call(ir.q, ir.w.lit_nat(1))}, {.operators = &ir.ops});
    CHECK(view.is_pure_function(input) == allowed);
}
