#include <array>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph_rule.h>
#include <mim/phase/graph_rule_rewrite.h>

using namespace mim;
using namespace mim::pattern;

namespace {

struct DPLIR {
    Driver driver;
    World& w            = driver.world();
    const Def* nat      = w.type_nat();
    const Def* x        = w.axm(nat);
    const Def* y        = w.axm(nat);
    const Def* z        = w.axm(nat);
    const Axm* q        = w.axm(w.pi(nat, nat));
    const Axm* k        = w.axm(w.pi(nat, nat));
    const Axm* v        = w.axm(w.pi(nat, nat));
    const Axm* sum      = w.axm(w.pi(nat, w.pi(nat, nat)));
    const Axm* product  = w.axm(w.pi(nat, w.pi(nat, nat)));
    const Axm* combined = w.axm(w.pi(nat, w.arr(3, nat)));
    OpRegistry ops;

    DPLIR() {
        for (auto op : {q, k, v, combined})
            ops.add(op->flags(), 1, {{0}}, true);
        for (auto op : {sum, product})
            ops.add(op->flags(), 2, {{0}, {1}}, true);
    }

    const Def* call(const Def* op, const Def* input) { return w.app(op, input); }
    const Def* pair(const Def* op, const Def* a, const Def* b) { return w.app(w.app(op, a), b); }
    const Def* add(const Def* a, const Def* b) { return pair(sum, a, b); }
    GraphOptions options() { return {.coverage = UseCoverage::Closed, .operators = &ops}; }

    const Def* raw_tuple(Defs values) {
        auto type = w.mut_sigma(w.type(), values.size());
        DefVec types;
        for (auto value : values)
            types.push_back(value->type());
        type->set(types);
        return w.tuple(type, values);
    }
};

Pattern call(PatternProgram& p, const Def* op, Pattern x) { return p.node(Node::App, {p.exact(op), x}); }

Pattern pair(PatternProgram& p, const Def* op, Pattern x, Pattern y) { return p.node(Node::App, {call(p, op, x), y}); }

using Tuples = std::set<std::vector<const Def*>>;

Tuples tuples(const GraphMatches& matches) {
    Tuples result;
    for (const auto& m : matches.matches)
        result.emplace(m.anchors().begin(), m.anchors().end());
    return result;
}

void check_matches(const GraphPattern& pattern,
                   const GraphView& view,
                   const Tuples& expected,
                   std::vector<std::pair<Pattern, const Def*>> captures = {},
                   SearchStatus status                                  = SearchStatus::Complete) {
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive}) {
        for (bool reverse : {false, true}) {
            auto found = find_matches(pattern, view, {.reverse_candidates = reverse, .mode = mode});
            CHECK(found.status == status);
            CHECK(found.matches.size() == expected.size());
            CHECK(tuples(found) == expected);
            for (const auto& m : found.matches)
                for (auto [capture, value] : captures)
                    CHECK(m.isa_capture(capture) == value);
        }
    }
}

struct NestedDiamond {
    const Def *a, *b, *left, *right, *bridge, *join_left, *join_right, *output;
    explicit NestedDiamond(DPLIR& ir)
        : a(ir.call(ir.q, ir.x))
        , b(ir.call(ir.q, ir.y))
        , left(ir.call(ir.k, a))
        , right(ir.call(ir.k, b))
        , bridge(ir.add(a, b))
        , join_left(ir.add(left, bridge))
        , join_right(ir.add(right, bridge))
        , output(ir.add(join_left, join_right)) {}
};

} // namespace

// TVM test_diamond: shared producer identity, not just four compatible operator kinds.
TEST_CASE("DPL diamond: edges select the shared fork and both exclusive branches") {
    DPLIR ir;
    auto fork = ir.call(ir.q, ir.x), left = ir.call(ir.k, fork), right = ir.call(ir.v, fork);
    auto good  = ir.add(left, right);
    auto other = ir.call(ir.q, ir.y), other_right = ir.call(ir.v, other);
    auto parallel = ir.add(left, other_right);
    PatternProgram p;
    GraphPattern pattern(p);
    auto input = p.capture();
    auto a     = pattern.anchor(call(p, ir.q, input));
    auto b     = pattern.anchor(call(p, ir.k, p.any()));
    auto c     = pattern.anchor(call(p, ir.v, p.any()));
    auto d     = pattern.anchor(pair(p, ir.sum, p.any(), p.any()));
    pattern.used_by(a, b);
    pattern.used_by(a, c);
    pattern.only_used_by(b, d, {.group = 0});
    pattern.only_used_by(c, d, {.group = 1});
    GraphView closed({good}, ir.options());
    check_matches(pattern, closed,
                  {
                      {fork, left, right, good}
    },
                  {{input, ir.x}});
    GraphView wrong({parallel}, ir.options());
    check_matches(pattern, wrong, {});
    GraphView shared_branch({good, parallel}, ir.options());
    check_matches(pattern, shared_branch, {});
}

// TVM test_diamond_counter_oub: a fork cannot be exclusive to either of two consumers.
TEST_CASE("DPL diamond: adding a second branch invalidates exclusive fork use") {
    DPLIR ir;
    auto fork = ir.call(ir.q, ir.x), left = ir.call(ir.k, fork), right = ir.call(ir.v, fork);
    PatternProgram p;
    GraphPattern pattern(p);
    auto a = pattern.anchor(call(p, ir.q, p.any()));
    auto b = pattern.anchor(call(p, ir.k, p.any()));
    pattern.only_used_by(a, b);
    GraphView single({left}, ir.options());
    check_matches(pattern, single,
                  {
                      {fork, left}
    });
    GraphView forked({ir.add(left, right)}, ir.options());
    check_matches(pattern, forked, {});
    GraphView escaping({left, fork}, ir.options());
    check_matches(pattern, escaping, {});
    auto options     = ir.options();
    options.coverage = UseCoverage::Open;
    GraphView open({left}, options);
    check_matches(pattern, open, {}, {}, SearchStatus::Unknown);
}

// TVM test_distinguish_diamond_and_parallel: repeated slots and distinct anchors are separate facts.
TEST_CASE("DPL identity: two slots distinguish one producer from parallel producers") {
    DPLIR ir;
    auto a = ir.call(ir.q, ir.x), b = ir.call(ir.q, ir.y);
    auto diamond = ir.add(a, a), parallel = ir.add(a, b);
    PatternProgram p;
    GraphPattern repeated(p);
    auto producer = repeated.anchor(call(p, ir.q, p.any()));
    auto join     = repeated.anchor(pair(p, ir.sum, p.any(), p.any()));
    repeated.only_used_by(producer, join, {.group = 0});
    repeated.only_used_by(producer, join, {.group = 1});
    GraphView one({diamond}, ir.options()), two({parallel}, ir.options());
    check_matches(repeated, one,
                  {
                      {a, diamond}
    });
    check_matches(repeated, two, {});

    GraphPattern branches(p);
    auto first  = branches.anchor(call(p, ir.q, p.any()));
    auto second = branches.anchor(call(p, ir.q, p.any()));
    auto output = branches.anchor(pair(p, ir.sum, p.any(), p.any()));
    branches.only_used_by(first, output, {.group = 0});
    branches.only_used_by(second, output, {.group = 1});
    check_matches(branches, one, {});
    check_matches(branches, two,
                  {
                      {a, b, parallel}
    });
    branches.allow_aliases();
    check_matches(branches, one,
                  {
                      {a, a, diamond}
    });
    check_matches(branches, two,
                  {
                      {a, b, parallel}
    });
    branches.allow_aliases(false);
    check_matches(branches, one, {});
}

// TVM test_nested_diamond: the middle join is shared by the two inner diamonds.
TEST_CASE("DPL nested diamond: enumerate both inner forks without claiming bridge exclusivity") {
    DPLIR ir;
    NestedDiamond n(ir);
    GraphView view({n.output}, ir.options());
    PatternProgram p;
    GraphPattern pattern(p);
    auto a = pattern.anchor(call(p, ir.q, p.any()));
    auto b = pattern.anchor(call(p, ir.k, p.any()));
    auto c = pattern.anchor(pair(p, ir.sum, p.any(), p.any()));
    auto d = pattern.anchor(pair(p, ir.sum, p.any(), p.any()));
    pattern.used_by(a, b);
    pattern.used_by(a, c);
    pattern.only_used_by(b, d, {.group = 0});
    pattern.used_by(c, d, {.group = 1});
    check_matches(pattern, view,
                  {
                      {n.a,  n.left, n.bridge,  n.join_left},
                      {n.b, n.right, n.bridge, n.join_right}
    });
    pattern.only_used_by(c, d);
    check_matches(pattern, view, {});
}

// TVM test_nested_diamond: the outer diamond has exclusive arms but a non-exclusive fork.
TEST_CASE("DPL nested diamond: the outer diamond preserves the common bridge capture") {
    DPLIR ir;
    NestedDiamond n(ir);
    PatternProgram p;
    auto bridge = p.capture(pair(p, ir.sum, call(p, ir.q, p.any()), call(p, ir.q, p.any())));
    GraphPattern pattern(p);
    auto a = pattern.anchor(bridge);
    auto b = pattern.anchor(pair(p, ir.sum, call(p, ir.k, p.any()), bridge));
    auto c = pattern.anchor(pair(p, ir.sum, call(p, ir.k, p.any()), bridge));
    auto d = pattern.anchor(pair(p, ir.sum, p.any(), p.any()));
    pattern.used_by(a, b);
    pattern.used_by(a, c);
    pattern.only_used_by(b, d, {.group = 0});
    pattern.only_used_by(c, d, {.group = 1});
    GraphView view({n.output}, ir.options());
    check_matches(pattern, view,
                  {
                      {n.bridge, n.join_left, n.join_right, n.output}
    },
                  {{bridge, n.bridge}});
    auto wrong = ir.add(n.join_left, ir.add(n.right, ir.add(n.b, n.a)));
    GraphView disconnected({wrong}, ir.options());
    check_matches(pattern, disconnected, {});
}

// TVM test_attention_fake_qkv: equal types and compatible weights do not prove shared input identity.
TEST_CASE("DPL QKV: weight-constrained products reject a fake shared-input triple") {
    DPLIR ir;
    auto wq = ir.w.lit_nat(11), wk = ir.w.lit_nat(12), wv = ir.w.lit_nat(13);
    auto q = ir.pair(ir.product, ir.x, wq), k = ir.pair(ir.product, ir.y, wk);
    auto v = ir.pair(ir.product, ir.y, wv), real_q = ir.pair(ir.product, ir.y, wq);
    PatternProgram p;
    auto input = p.capture(), cq = p.capture(p.exact(wq)), ck = p.capture(p.exact(wk));
    auto cv = p.capture(p.exact(wv));
    GraphPattern pattern(p);
    pattern.anchor(pair(p, ir.product, input, cq));
    pattern.anchor(pair(p, ir.product, input, ck));
    pattern.anchor(pair(p, ir.product, input, cv));
    GraphView fake({q, k, v}, ir.options());
    check_matches(pattern, fake, {});
    GraphView mixed({q, real_q, k, v}, ir.options());
    check_matches(pattern, mixed,
                  {
                      {real_q, k, v}
    },
                  {{input, ir.y}, {cq, wq}, {ck, wk}, {cv, wv}});
}

// TVM test_repeated_pattern_match: capture a whole reused subpattern, not just its leaves.
TEST_CASE("DPL repeated pattern: whole-expression captures survive success failure success") {
    DPLIR ir;
    auto shared = ir.add(ir.x, ir.y), other = ir.pair(ir.product, ir.x, ir.y);
    auto good = ir.pair(ir.product, shared, shared), bad = ir.pair(ir.product, shared, other);
    PatternProgram p;
    auto x = p.capture(), y = p.capture();
    auto subexpression = p.capture(pair(p, ir.sum, x, y));
    auto pattern       = pair(p, ir.product, subexpression, subexpression);
    auto gid           = ir.w.curr_gid();
    for (auto candidate : {good, bad, good}) {
        auto outcome = match(p, pattern, candidate);
        if (candidate == bad) {
            CHECK(outcome.status == MatchStatus::No_Match);
            CHECK_FALSE(outcome.result);
        } else {
            REQUIRE(outcome.status == MatchStatus::Matched);
            REQUIRE(outcome.result);
            CHECK(outcome.result->isa_capture(subexpression) == shared);
            CHECK(outcome.result->isa_capture(x) == ir.x);
            CHECK(outcome.result->isa_capture(y) == ir.y);
        }
    }
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_commutative_pattern_match: MimIR spells both orientations explicitly.
TEST_CASE("DPL commutativity: indexed joins retain captures on different raw operand paths") {
    DPLIR ir;
    auto marker = ir.w.lit_nat(7);
    auto left = ir.add(ir.x, marker), right = ir.add(marker, ir.x);
    auto wrong = ir.add(ir.y, marker), root = ir.call(ir.q, ir.x);
    PatternProgram p;
    auto x      = p.capture();
    auto choice = p.either(pair(p, ir.sum, x, p.exact(marker)), pair(p, ir.sum, p.exact(marker), x));
    GraphPattern pattern(p);
    pattern.anchor(choice);
    pattern.anchor(call(p, ir.q, x));
    GraphView view({left, right, wrong, root}, ir.options());
    check_matches(pattern, view,
                  {
                      { left, root},
                      {right, root}
    },
                  {{x, ir.x}});
    CHECK(match(p, pair(p, ir.sum, x, p.exact(marker)), right).status == MatchStatus::No_Match);
}

// TVM test_commutative_pattern_match and test_backtrack_if_rewriter_returns_no_op.
TEST_CASE("DPL alternatives: a whole-root optional capture cannot prune the capture-free branch") {
    DPLIR ir;
    auto marker = ir.w.lit_nat(7);
    auto first = ir.add(ir.x, marker), optional = ir.add(ir.y, marker);
    auto witness = ir.call(ir.q, ir.x), wrong = ir.add(ir.z, marker);
    PatternProgram p;
    auto x      = p.capture();
    auto choice = p.either(pair(p, ir.sum, x, p.exact(marker)), p.exact(optional));
    GraphPattern pattern(p);
    pattern.anchor(choice);
    pattern.anchor(call(p, ir.q, x));
    GraphView view({first, optional, wrong, witness}, ir.options());
    check_matches(pattern, view,
                  {
                      {   first, witness},
                      {optional, witness}
    },
                  {{x, ir.x}});
}

// TVM test_repeated_pattern_match: type identity is not a raw data-operand ancestry proof.
TEST_CASE("DPL type captures: a prebound type anchor joins values through type edges only") {
    DPLIR ir;
    auto qa = ir.call(ir.q, ir.x), kb = ir.call(ir.k, ir.y);
    PatternProgram p;
    auto type = p.capture(p.exact(ir.nat));
    GraphPattern pattern(p);
    pattern.anchor(type, AnchorDomain::Nodes);
    pattern.anchor(p.all(call(p, ir.q, p.any()), p.type(type)));
    GraphView view({ir.nat, qa, kb}, ir.options());
    check_matches(pattern, view,
                  {
                      {ir.nat, qa}
    },
                  {{type, ir.nat}});
    GraphPattern wrong(p);
    auto boolean = p.capture(p.exact(ir.w.type_bool()));
    wrong.anchor(boolean, AnchorDomain::Nodes);
    wrong.anchor(p.all(call(p, ir.q, p.any()), p.type(boolean)));
    GraphView mismatch({ir.w.type_bool(), qa}, ir.options());
    check_matches(wrong, mismatch, {});
}

// TVM test_commutative_pattern_match: an OR capture can denote a raw value or that value's type.
TEST_CASE("DPL alternatives: a capture shared by raw and type branches has no guaranteed raw path") {
    DPLIR ir;
    auto qa = ir.call(ir.q, ir.x), qb = ir.call(ir.q, ir.y);
    PatternProgram p;
    auto type = p.capture(p.exact(ir.nat));
    GraphPattern pattern(p);
    pattern.anchor(type, AnchorDomain::Nodes);
    pattern.anchor(call(p, ir.q, p.either(type, p.type(type))));
    GraphView view({ir.nat, qa, qb}, ir.options());
    check_matches(pattern, view,
                  {
                      {ir.nat, qa},
                      {ir.nat, qb}
    },
                  {{type, ir.nat}});
}

// TVM test_varg_default_wildcard: zero tail operands do not require the tail capture.
TEST_CASE("DPL ranges: a zero-length tail still joins a capture bound by another root") {
    DPLIR ir;
    auto empty_tail = ir.raw_tuple({ir.y});
    auto one_tail   = ir.raw_tuple({ir.y, ir.x});
    auto wrong_tail = ir.raw_tuple({ir.y, ir.z});
    auto witness    = ir.call(ir.q, ir.x);
    REQUIRE(empty_tail->node() == Node::Tuple);
    PatternProgram p;
    auto x = p.capture();
    GraphPattern pattern(p);
    pattern.anchor(p.node_range(Node::Tuple, {p.exact(ir.y)}, x, 0, 2), AnchorDomain::Nodes);
    pattern.anchor(call(p, ir.q, x));
    GraphView view({empty_tail, one_tail, wrong_tail, witness}, ir.options());
    check_matches(pattern, view,
                  {
                      {empty_tail, witness},
                      {  one_tail, witness}
    },
                  {{x, ir.x}});
    CHECK(match(p, p.node_range(Node::Tuple, {p.exact(ir.y)}, p.any(), 2, 2), one_tail).status
          == MatchStatus::No_Match);
}

// TVM test_varg_default_wildcard and test_commutative_pattern_match: range operands keep OR continuations.
TEST_CASE("DPL ranges: prefix alternatives roll back after repeated tail identity fails") {
    DPLIR ir;
    auto good = ir.raw_tuple({ir.x, ir.y, ir.y});
    auto bad  = ir.raw_tuple({ir.x, ir.y, ir.z});
    PatternProgram p;
    auto tail    = p.capture();
    auto pattern = p.node_range(Node::Tuple, {p.either(tail, p.exact(ir.x))}, tail, 2, 2);
    auto outcome = match(p, pattern, good);
    REQUIRE(outcome.status == MatchStatus::Matched);
    REQUIRE(outcome.result);
    CHECK(outcome.result->isa_capture(tail) == ir.y);
    auto rejected = match(p, pattern, bad);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK_FALSE(rejected.result);
    auto normalized = ir.w.tuple({ir.y, ir.y, ir.y});
    REQUIRE(normalized->node() == Node::Pack);
    CHECK(match(p, p.node_range(Node::Tuple, {}, p.any(), 0, 4), normalized).status == MatchStatus::No_Match);
}

// TVM test_diamond and test_attention_fake_qkv: projected slots retain the full original-result path.
TEST_CASE("DPL projections: nested result paths filter repeated uses of one producer") {
    DPLIR ir;
    // A Sigma boundary preserves two scalar Extracts instead of one fused multidimensional index.
    auto results = ir.w.mut_sigma(ir.w.type(), 2);
    results->set({ir.w.arr(3, ir.nat), ir.w.arr(3, ir.nat)});
    auto grid = ir.w.axm(ir.w.pi(ir.nat, results));
    ir.ops.add(grid->flags(), 1, {{0}}, true);
    auto producer = ir.call(grid, ir.x);
    auto a        = ir.w.extract(ir.w.extract(producer, 2, 1), 3, 2);
    auto b        = ir.w.extract(ir.w.extract(producer, 2, 0), 3, 1);
    auto output   = ir.add(a, b);
    GraphView view({output}, ir.options());
    REQUIRE(view.data_uses(producer).size() == 2);
    std::set<std::vector<size_t>> paths;
    for (const auto& edge : view.data_uses(producer)) {
        CHECK(edge.consumer == output);
        CHECK(edge.path.empty());
        if (edge.group == 0) {
            CHECK(edge.value == a);
            CHECK(edge.result_path == std::vector<size_t>{1, 2});
        } else {
            CHECK(edge.group == 1);
            CHECK(edge.value == b);
            CHECK(edge.result_path == std::vector<size_t>{0, 1});
        }
        paths.insert(edge.result_path);
    }
    CHECK(paths
          == std::set<std::vector<size_t>>{
              {1, 2},
              {0, 1}
    });
    PatternProgram p;
    GraphPattern pattern(p);
    auto source = pattern.anchor(call(p, grid, p.any()));
    auto sink   = pattern.anchor(pair(p, ir.sum, p.any(), p.any()));
    pattern.only_used_by(source, sink,
                         {
                             .group       = 0,
                             .result_path = std::vector<size_t>{1, 2}
    });
    pattern.only_used_by(source, sink,
                         {
                             .group       = 1,
                             .result_path = std::vector<size_t>{0, 1}
    });
    check_matches(pattern, view,
                  {
                      {producer, output}
    });
    GraphView escaping({output, b}, ir.options());
    check_matches(pattern, escaping, {});
    GraphView extra_consumer({output, ir.call(ir.q, b)}, ir.options());
    check_matches(pattern, extra_consumer, {});
    pattern.used_by(source, sink,
                    {
                        .group       = 1,
                        .result_path = std::vector<size_t>{1, 2}
    });
    check_matches(pattern, view, {});
}

// TVM test_distinguish_diamond_and_parallel and test_varg_default_wildcard: semantic wrapper slots.
TEST_CASE("DPL wrapper paths: repeated Pack slots count as one exclusive consumer") {
    DPLIR ir;
    auto wrapper_op = ir.w.axm(ir.w.pi(ir.w.arr(2, ir.w.arr(2, ir.nat)), ir.nat));
    ir.ops.add(wrapper_op->flags(), 1,
               {
                   {0, 2}
    },
               true);
    auto producer = ir.call(ir.q, ir.x);
    auto packed   = ir.w.tuple({producer, producer});
    REQUIRE(packed->node() == Node::Pack);
    auto wrapper = ir.w.tuple({packed, ir.w.tuple({producer, ir.y})});
    auto output  = ir.call(wrapper_op, wrapper);
    GraphView view({output}, ir.options());
    REQUIRE(view.data_uses(producer).size() == 3);
    std::set<std::vector<size_t>> paths;
    for (const auto& edge : view.data_uses(producer)) {
        CHECK(edge.group == 0);
        CHECK(edge.consumer == output);
        CHECK(edge.value == producer);
        CHECK(edge.result_path.empty());
        paths.insert(edge.path);
    }
    CHECK(paths
          == std::set<std::vector<size_t>>{
              {0, 0},
              {0, 1},
              {1, 0}
    });
    PatternProgram p;
    GraphPattern pattern(p);
    auto source = pattern.anchor(call(p, ir.q, p.any()));
    auto sink   = pattern.anchor(call(p, wrapper_op, p.any()));
    for (auto path : {
             std::vector<size_t>{0, 0},
             std::vector<size_t>{0, 1},
             std::vector<size_t>{1, 0}
    })
        pattern.only_used_by(source, sink, {.group = 0, .path = path});
    check_matches(pattern, view,
                  {
                      {producer, output}
    });
    GraphView extra({output, ir.call(ir.k, producer)}, ir.options());
    check_matches(pattern, extra, {});
    pattern.used_by(source, sink,
                    {
                        .group = 0,
                        .path  = std::vector<size_t>{1, 1}
    });
    check_matches(pattern, view, {});
}

// TVM test_combine_matmul_twice: disjoint matches become two independently shared recipe DAGs.
TEST_CASE("DPL checked rewrite: two QKV groups merge without sharing their new computations") {
    DPLIR ir;
    auto a = ir.w.lit_nat(11), b = ir.w.lit_nat(22);
    std::array roots{ir.call(ir.q, a), ir.call(ir.k, a), ir.call(ir.v, a),
                     ir.call(ir.q, b), ir.call(ir.k, b), ir.call(ir.v, b)};
    GraphView view(roots, ir.options());
    PatternProgram p;
    auto input = p.capture();
    GraphPattern pattern(p);
    auto q = pattern.anchor(call(p, ir.q, input));
    auto k = pattern.anchor(call(p, ir.k, input));
    auto v = pattern.anchor(call(p, ir.v, input));
    check_matches(pattern, view,
                  {
                      {roots[0], roots[1], roots[2]},
                      {roots[3], roots[4], roots[5]}
    });
    auto found = find_matches(pattern, view);
    REQUIRE(found.matches.size() == 2);
    std::set<const Def*> inputs;
    std::vector<std::unique_ptr<RewritePlan>> owned;
    std::vector<const RewritePlan*> plans;
    for (const auto& m : found.matches) {
        inputs.insert(m.isa_capture(input));
        CHECK(m.isa_capture(input) == (m.at(q) == roots[0] ? a : b));
        CHECK(m.at(k)->as<App>()->arg() == m.isa_capture(input));
        CHECK(m.at(v)->as<App>()->arg() == m.isa_capture(input));
        auto plan   = std::make_unique<RewritePlan>(m);
        auto merged = plan->call(ir.combined, {plan->import(m.isa_capture(input))});
        plan->replace(m.at(q), plan->extract(merged, 0));
        plan->replace(m.at(k), plan->extract(merged, 1));
        plan->replace(m.at(v), plan->extract(merged, 2));
        plans.push_back(plan.get());
        owned.push_back(std::move(plan));
    }
    CHECK(inputs == std::set<const Def*>{a, b});
    auto gid    = ir.w.curr_gid();
    auto result = rewrite_graph(view, plans);
    REQUIRE(result.roots().size() == 6);
    CHECK(result.replacements() == 6);
    std::array<const Def*, 2> merged{};
    for (size_t group = 0; group != 2; ++group) {
        for (size_t slot = 0; slot != 3; ++slot) {
            auto ex = result.roots()[group * 3 + slot]->isa<Extract>();
            REQUIRE(ex);
            CHECK(Lit::isa(ex->index()) == slot);
            if (!merged[group]) merged[group] = ex->tuple();
            CHECK(ex->tuple() == merged[group]);
        }
        auto op = merged[group]->as<App>();
        CHECK(op->axm()->flags() == ir.combined->flags());
        CHECK(op->arg() == result.world().lit_nat(group == 0 ? 11 : 22));
    }
    CHECK(merged[0] != merged[1]);
    GraphView rewritten(result.roots(), ir.options());
    CHECK(std::set<const Def*>(rewritten.operations().begin(), rewritten.operations().end())
          == std::set<const Def*>{merged[0], merged[1]});
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// TVM test_backtrack_if_rewriter_returns_no_op: the ordered multi-root phase skips identity/rejected Rules.
TEST_CASE("DPL graph Rule fallback: identity and false guards do not hide the first productive Rule") {
    DPLIR ir;
    auto make_rule = [&](bool identity, bool accept, bool last) {
        auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
        auto x    = rule->var();
        auto lhs  = ir.w.tuple({ir.call(ir.q, x), ir.call(ir.k, x)});
        auto rhs  = last ? ir.w.tuple({ir.w.lit_nat(90), ir.w.lit_nat(91)})
                         : ir.w.tuple({ir.call(ir.v, x), ir.call(ir.v, x)});
        return rule->set(lhs, identity ? lhs : rhs, ir.w.lit_bool(accept));
    };
    auto identity   = make_rule(true, true, false);
    auto rejected   = make_rule(false, false, false);
    auto productive = make_rule(false, true, false);
    auto last       = make_rule(false, true, true);
    auto fn         = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("dpl_fallback");
    fn->set(false, ir.add(ir.call(ir.q, fn->var()), ir.call(ir.k, fn->var())))->externalize();
    auto name      = fn->sym();
    auto body      = fn->body();
    auto sum_flags = ir.sum->flags(), v_flags = ir.v->flags();
    bool fallback = true;
    SUBCASE("first productive Rule wins") {}
    SUBCASE("only identity and rejected Rules leave the function unchanged") { fallback = false; }
    auto rules = fallback ? ir.w.tuple({identity, rejected, productive, last}) : ir.w.tuple({identity, rejected});
    auto configuration = ir.w.tuple({rules, ir.w.tuple({ir.q, ir.k, ir.v, ir.sum})});
    auto configure     = ir.w.axm(ir.w.pi(configuration->type(), ir.nat));
    GraphRuleRewrite phase(ir.w, "dpl_ordered_fallback");
    phase.apply(ir.call(configure, configuration)->as<App>());
    phase.run();
    auto rebuilt = ir.w.externals()[name]->as<Lam>();
    if (fallback) {
        CHECK(phase.todo());
        auto outer = rebuilt->body()->isa<App>();
        REQUIRE(outer);
        auto inner = outer->callee()->isa<App>();
        REQUIRE(inner);
        CHECK(outer->axm()->flags() == sum_flags);
        CHECK(inner->arg() == outer->arg());
        auto chosen = outer->arg()->isa<App>();
        REQUIRE(chosen);
        CHECK(chosen->axm()->flags() == v_flags);
        CHECK(chosen->arg() == rebuilt->var());
    } else {
        CHECK_FALSE(phase.todo());
        CHECK(rebuilt == fn);
        CHECK(rebuilt->body() == body);
    }
}

// TVM test_commutative_pattern_match: an unproved branch must roll back, not export partial captures.
TEST_CASE("DPL deferred alternatives: successful tuples roll back unknown branch-local captures") {
    DPLIR ir;
    auto value = ir.add(ir.x, ir.y);
    PatternProgram p;
    auto local = p.capture(), shared = p.capture();
    auto unknown = p.where(pair(p, ir.sum, local, p.any()), {local}, [](auto) { return CheckResult::Unknown; });
    GraphPattern pattern(p);
    pattern.anchor(p.either(unknown, pair(p, ir.sum, p.exact(ir.x), shared)));
    pattern.anchor(call(p, ir.q, shared));
    auto witness = ir.call(ir.q, ir.y);
    GraphView view({value, witness}, ir.options());
    auto gid = ir.w.curr_gid();
    check_matches(pattern, view,
                  {
                      {value, witness}
    },
                  {{local, nullptr}, {shared, ir.y}});
    CHECK(ir.w.curr_gid() == gid);
    GraphView unresolved({value, ir.call(ir.q, ir.x)}, ir.options());
    check_matches(pattern, unresolved, {}, {}, SearchStatus::Unknown);
    GraphView mixed({value, witness, ir.call(ir.q, ir.x)}, ir.options());
    check_matches(pattern, mixed,
                  {
                      {value, witness}
    },
                  {{local, nullptr}, {shared, ir.y}}, SearchStatus::Unknown);
}
