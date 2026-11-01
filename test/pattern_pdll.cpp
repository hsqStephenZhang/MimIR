#include <array>
#include <set>
#include <stdexcept>
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

const Def* add_zero(const Def*, const Def* callee, const Def* arg) {
    auto args = arg->isa<Tuple>();
    if (callee->isa<Axm>() && args && args->num_ops() == 2) {
        auto value = Lit::isa(args->op(1));
        if (value && *value == 0) return args->op(0);
    }
    return nullptr;
}

const Def* selective_guard(const Def*, const Def*, const Def* arg) {
    if (auto value = Lit::isa(arg)) {
        if (*value == 1) return nullptr;
        return arg->world().lit_bool(*value != 2);
    }
    return nullptr;
}

struct PDLLIR {
    Driver driver;
    World& w              = driver.world();
    const Def* nat        = w.type_nat();
    const Def* boolean    = w.type_bool();
    const Def* mixed_type = w.sigma({nat, boolean});
    const Def* x          = w.axm(nat);
    const Def* y          = w.axm(nat);
    const Def* flag       = w.axm(boolean);
    const Axm* n          = w.axm(w.pi(nat, nat));
    const Axm* b          = w.axm(w.pi(nat, boolean));
    const Axm* tag        = w.axm(w.pi(boolean, boolean));
    const Axm* mixed      = w.axm(w.pi(nat, mixed_type));
    const Axm* wrap       = w.axm(w.pi(mixed_type, mixed_type));
    const Axm* take_nat   = w.axm(w.pi(mixed_type, nat));
    const Axm* take_bool  = w.axm(w.pi(mixed_type, boolean));
    const Axm* choose     = w.axm(w.pi(nat, w.pi(boolean, nat)));
    OpRegistry ops;

    PDLLIR() {
        for (auto op : {n, b, tag, mixed})
            ops.add(op->flags(), 1, {{0}}, true);
        for (auto op : {wrap, take_nat, take_bool})
            ops.add(op->flags(), 1,
                    {
                        {0, 1}
            },
                    true);
        ops.add(choose->flags(), 2, {{0}, {1}}, true);
    }

    const Def* call(const Def* op, const Def* input) { return w.app(op, input); }
    const Def* pair(const Def* op, const Def* first, const Def* second) { return w.app(w.app(op, first), second); }
    GraphOptions options(Lam* scope = nullptr) {
        return {.coverage = UseCoverage::Closed, .operators = &ops, .scope = scope};
    }
    const Def* raw_tuple(Defs values) {
        auto type = w.mut_sigma(w.type(), values.size());
        DefVec fields;
        for (auto value : values)
            fields.push_back(value->type());
        type->set(fields);
        return w.tuple(type, values);
    }
};

Pattern call(PatternProgram& p, const Def* op, Pattern input) { return p.node(Node::App, {p.exact(op), input}); }

Pattern pair(PatternProgram& p, const Def* op, Pattern first, Pattern second) {
    return p.node(Node::App, {call(p, op, first), second});
}

Pattern projection(PatternProgram& p, Pattern producer, const Def* index) {
    return p.node(Node::Extract, {producer, p.exact(index)});
}

CheckResult small_nat(std::span<const Def* const> values) {
    auto value = Lit::isa(values[0]);
    if (!value) return CheckResult::Unknown;
    return *value < 4 ? CheckResult::True : CheckResult::False;
}

CheckResult true_flag(std::span<const Def* const> values) {
    auto value = Lit::isa(values[0]);
    if (!value) return CheckResult::Unknown;
    return *value == 1 ? CheckResult::True : CheckResult::False;
}

Pattern small_enabled(PatternProgram& p, Pattern root, Pattern number, Pattern flag) {
    return p.where(p.where(root, {number}, small_nat), {flag}, true_flag);
}

using Tuples = std::set<std::vector<const Def*>>;

Tuples tuples(const GraphMatches& found) {
    Tuples result;
    for (const auto& m : found.matches)
        result.emplace(m.anchors().begin(), m.anchors().end());
    return result;
}

template<class Find, class Inspect>
void check_search(Find find, const Tuples& expected, Inspect inspect, SearchStatus status = SearchStatus::Complete) {
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool reverse : {false, true}) {
            auto found = find(GraphMatchOptions{.reverse_candidates = reverse, .mode = mode});
            REQUIRE(found.status == status);
            CHECK(found.matches.size() == expected.size());
            CHECK(tuples(found) == expected);
            for (const auto& m : found.matches)
                inspect(m);
        }
}

void check_matches(const GraphPattern& pattern,
                   const GraphView& view,
                   const Tuples& expected,
                   std::vector<std::pair<Pattern, const Def*>> captures = {},
                   SearchStatus status                                  = SearchStatus::Complete) {
    auto gid = view.world().curr_gid();
    check_search([&](auto options) { return find_matches(pattern, view, options); }, expected,
                 [&](const GraphMatch& m) {
                     for (auto [capture, value] : captures)
                         CHECK(m.isa_capture(capture) == value);
                 },
                 status);
    CHECK(view.world().curr_gid() == gid);
    CHECK(view.is_current());
}

GraphRewrite build(const GraphView& view, const RewritePlan& plan) {
    std::array plans{&plan};
    return rewrite_graph(view, plans);
}

std::multiset<flags_t> operation_flags(const GraphView& view) {
    std::multiset<flags_t> result;
    for (auto op : view.operations())
        result.insert(std::get<0>(Axm::get(op))->flags());
    return result;
}

struct MixedRoots {
    PatternProgram p;
    Pattern input;
    GraphPattern graph{p};
    Anchor number, flag;

    explicit MixedRoots(PDLLIR& ir)
        : input(p.capture(p.type(p.exact(ir.nat))))
        , number(graph.anchor(p.all(call(p, ir.n, input), p.type(p.exact(ir.nat)))))
        , flag(graph.anchor(p.all(call(p, ir.b, input), p.type(p.exact(ir.boolean))))) {}
};

} // namespace

// PDLL: test/mlir-pdll/CodeGen/MLIR/decl.pdll VarWithNamedOpConstraint / VarWithConstraints;
// test/lib/Tools/PDLL/TestPDLL.pdll TestConstraintWithNamedOpOperand.
TEST_CASE("PDLL adapted named operation: operand and heterogeneous result types are conjunctive") {
    PDLLIR ir;
    auto other = ir.w.axm(ir.mixed->type());
    auto good = ir.call(ir.mixed, ir.x), wrong_name = ir.call(other, ir.x);
    PatternProgram p;
    auto callee     = p.capture(p.exact(ir.mixed));
    auto input      = p.capture(p.type(p.exact(ir.nat)));
    auto first_type = p.capture(p.exact(ir.nat)), second_type = p.capture(p.exact(ir.boolean));
    auto root
        = p.capture(p.all(p.node(Node::App, {callee, input}), p.type(p.node(Node::Sigma, {first_type, second_type}))));
    auto gid   = ir.w.curr_gid();
    auto found = match(p, root, good);
    REQUIRE(found.status == MatchStatus::Matched);
    REQUIRE(found.result);
    CHECK(found.result->isa_capture(root) == good);
    CHECK(found.result->isa_capture(callee) == ir.mixed);
    CHECK(found.result->isa_capture(input) == ir.x);
    CHECK(found.result->isa_capture(first_type) == ir.nat);
    CHECK(found.result->isa_capture(second_type) == ir.boolean);
    auto rejected = match(p, root, wrong_name);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK_FALSE(rejected.result);
    auto wrong_operand = p.all(root, call(p, ir.mixed, p.type(p.exact(ir.boolean))));
    auto wrong_result  = p.all(root, p.type(p.node(Node::Sigma, {p.exact(ir.boolean), p.exact(ir.nat)})));
    for (auto pattern : {wrong_operand, wrong_result}) {
        auto result = match(p, pattern, good);
        CHECK(result.status == MatchStatus::No_Match);
        CHECK_FALSE(result.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: docs/PDLL.md Core Constraints, Value<valueType> shares the root result type;
// test/mlir-pdll/CodeGen/MLIR/expr.pdll TypeExpr. PDL lowering corroboration:
// test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir @inputs / @results.
TEST_CASE("PDLL adapted type equality: concrete and captured constraints reject either half alone") {
    PDLLIR ir;
    auto first_only           = ir.w.axm(ir.w.pi(ir.nat, ir.w.pi(ir.boolean, ir.boolean)));
    auto result_only          = ir.w.axm(ir.w.pi(ir.boolean, ir.w.pi(ir.boolean, ir.nat)));
    auto equal_wrong_concrete = ir.w.axm(ir.w.pi(ir.boolean, ir.w.pi(ir.boolean, ir.boolean)));
    auto bool_to_bool         = ir.w.axm(ir.w.pi(ir.boolean, ir.w.pi(ir.nat, ir.boolean)));
    auto good                 = ir.pair(ir.choose, ir.x, ir.flag);
    std::array bad{ir.pair(first_only, ir.x, ir.flag), ir.pair(result_only, ir.flag, ir.flag),
                   ir.pair(equal_wrong_concrete, ir.flag, ir.flag)};
    PatternProgram p;
    auto type         = p.capture();
    auto callee       = p.capture(p.node(Node::Axm, {}));
    auto first        = p.capture(p.all(p.type(type), p.type(p.exact(ir.nat))));
    auto second       = p.capture(p.type(p.exact(ir.boolean)));
    auto root         = p.capture(p.all(p.node(Node::App, {p.node(Node::App, {callee, first}), second}), p.type(type)));
    auto boolean_good = ir.pair(bool_to_bool, ir.flag, ir.y);
    auto bool_type    = p.capture(p.exact(ir.boolean));
    auto bool_input = p.capture(p.type(bool_type)), nat_input = p.capture(p.type(p.exact(ir.nat)));
    auto bool_pattern = p.all(pair(p, bool_to_bool, bool_input, nat_input), p.type(bool_type));
    auto gid          = ir.w.curr_gid();
    auto found        = match(p, root, good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(root) == good);
    CHECK(found.result->isa_capture(callee) == ir.choose);
    CHECK(found.result->isa_capture(type) == ir.nat);
    CHECK(found.result->isa_capture(first) == ir.x);
    CHECK(found.result->isa_capture(second) == ir.flag);
    for (auto candidate : bad) {
        auto rejected = match(p, root, candidate);
        CHECK(rejected.status == MatchStatus::No_Match);
        CHECK_FALSE(rejected.result);
    }
    found = match(p, bool_pattern, boolean_good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(bool_type) == ir.boolean);
    CHECK(found.result->isa_capture(bool_input) == ir.flag);
    CHECK(found.result->isa_capture(nat_input) == ir.y);
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/decl.pdll VarWithUserConstraint (NestedArgCst, NestedResCst, OpCst);
// docs/PDLL.md Native Constraints. Only read-only checks compose here, not returned bindings.
TEST_CASE("PDLL adapted composed native constraints: reuse preserves false and unknown distinctions") {
    PDLLIR ir;
    auto other = ir.w.axm(ir.choose->type());
    auto one = ir.w.lit_nat(1), seven = ir.w.lit_nat(7), yes = ir.w.lit_tt(), no = ir.w.lit_ff();
    std::array good{ir.pair(ir.choose, one, yes), ir.pair(other, one, yes)};
    auto bad_number = ir.pair(ir.choose, seven, yes), bad_flag = ir.pair(ir.choose, one, no);
    auto unknown_number = ir.pair(ir.choose, ir.x, yes), unknown_flag = ir.pair(ir.choose, one, ir.flag);
    PatternProgram p;
    auto number = p.capture(p.type(p.exact(ir.nat))), flag = p.capture(p.type(p.exact(ir.boolean)));
    auto type = p.capture(p.exact(ir.nat)), root = p.capture();
    auto first   = small_enabled(p, p.all(root, p.all(pair(p, ir.choose, number, flag), p.type(type))), number, flag);
    auto second  = small_enabled(p, p.all(root, p.all(pair(p, other, number, flag), p.type(type))), number, flag);
    auto pattern = p.either(first, second);
    auto gid     = ir.w.curr_gid();
    for (auto candidate : good) {
        auto found = match(p, pattern, candidate);
        REQUIRE(found.status == MatchStatus::Matched);
        CHECK(found.result->isa_capture(root) == candidate);
        CHECK(found.result->isa_capture(number) == one);
        CHECK(found.result->isa_capture(flag) == yes);
        CHECK(found.result->isa_capture(type) == ir.nat);
    }
    for (auto candidate : {bad_number, bad_flag}) {
        auto found = match(p, pattern, candidate);
        CHECK(found.status == MatchStatus::No_Match);
        CHECK_FALSE(found.result);
    }
    for (auto candidate : {unknown_number, unknown_flag}) {
        auto found = match(p, pattern, candidate);
        CHECK(found.status == MatchStatus::Unknown);
        CHECK_FALSE(found.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/expr.pdll TestExternalNegatedCall;
// test/mlir-pdll/Parser/expr-failure.pdll rejects negating a non-native constraint.
// PDL runtime corroboration: test/Rewrite/pdl-bytecode.mlir test.apply_constraint_3.
TEST_CASE("PDLL adapted native negation: typed probes cannot export bindings or manufacture proof") {
    PDLLIR ir;
    auto one = ir.w.lit_nat(1), seven = ir.w.lit_nat(7);
    auto good = ir.call(ir.mixed, seven), bad = ir.call(ir.mixed, one), symbolic = ir.call(ir.mixed, ir.x);
    PatternProgram p;
    auto input = p.capture(p.type(p.exact(ir.nat))), local = p.capture(p.type(p.exact(ir.nat)));
    auto not_small     = p.negate(p.where(local, {local}, small_nat));
    auto root          = p.capture(p.all(call(p, ir.mixed, p.all(input, not_small)), p.type(p.exact(ir.mixed_type))));
    auto missing       = p.capture(p.type(p.exact(ir.boolean)));
    auto missing_proof = p.all(root, p.negate(p.where(p.any(), {missing}, true_flag)));
    auto gid           = ir.w.curr_gid();
    auto found         = match(p, root, good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(root) == good);
    CHECK(found.result->isa_capture(input) == seven);
    CHECK(found.result->isa_capture(local) == nullptr);
    CHECK(found.result->isa_capture(missing) == nullptr);
    auto rejected = match(p, root, bad);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK_FALSE(rejected.result);
    for (auto [pattern, candidate] : {
             std::pair{         root, symbolic},
             std::pair{missing_proof,     good}
    }) {
        auto unknown = match(p, pattern, candidate);
        CHECK(unknown.status == MatchStatus::Unknown);
        CHECK_FALSE(unknown.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/decl.pdll VarWithRangeConstraints and docs/PDLL.md ValueRange.
// Partial adaptation: a Bool prefix and 1..3 physical Nat operands, not ValueRange capture.
TEST_CASE("PDLL partial typed operand range: bounds apply to physical fields and share only their type") {
    PDLLIR ir;
    auto one = ir.w.lit_nat(1), two = ir.w.lit_nat(2), three = ir.w.lit_nat(3);
    std::array good{ir.raw_tuple({ir.flag, one}), ir.raw_tuple({ir.flag, one, two}),
                    ir.raw_tuple({ir.flag, one, two, three})};
    std::array bad{ir.raw_tuple({ir.flag}), ir.raw_tuple({ir.flag, one, two, three, ir.x}),
                   ir.raw_tuple({ir.flag, one, ir.w.lit_tt()}), ir.raw_tuple({one, two}), ir.w.tuple({one, one})};
    PatternProgram p;
    auto prefix = p.capture(p.type(p.exact(ir.boolean))), type = p.capture(p.exact(ir.nat));
    auto root = p.capture(p.node_range(Node::Tuple, {prefix}, p.type(type), 1, 3));
    auto gid  = ir.w.curr_gid();
    for (auto candidate : good) {
        REQUIRE(candidate->node() == Node::Tuple);
        auto found = match(p, root, candidate);
        REQUIRE(found.status == MatchStatus::Matched);
        CHECK(found.result->isa_capture(root) == candidate);
        CHECK(found.result->isa_capture(prefix) == ir.flag);
        CHECK(found.result->isa_capture(type) == ir.nat);
    }
    REQUIRE(bad.back()->node() == Node::Pack);
    for (auto candidate : bad) {
        auto found = match(p, root, candidate);
        CHECK(found.status == MatchStatus::No_Match);
        CHECK_FALSE(found.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: docs/PDLL.md HasSameElementType / Native Constraints;
// test/mlir-pdll/CodeGen/MLIR/expr.pdll TestExternalNegatedCall. PDL lowering corroboration:
// test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir @constraints / @negated_constraint.
TEST_CASE("PDLL adapted graph constraints: mixed result roots join before read-only native checks") {
    PDLLIR ir;
    auto one = ir.w.lit_nat(1), seven = ir.w.lit_nat(7);
    auto n1 = ir.call(ir.n, one), b1 = ir.call(ir.b, one);
    auto n7 = ir.call(ir.n, seven), b7 = ir.call(ir.b, seven);
    auto ns = ir.call(ir.n, ir.x), bs = ir.call(ir.b, ir.x);
    PatternProgram p;
    auto input  = p.capture(p.type(p.exact(ir.nat)));
    auto number = p.capture(call(p, ir.n, input)), flag = p.capture(call(p, ir.b, input));
    GraphPattern graph(p);
    graph.anchor(p.all(number, p.negate(p.where(p.any(), {input}, small_nat))));
    graph.anchor(p.where(flag, {number, flag}, [](auto values) {
        return values[0]->type() != values[1]->type() ? CheckResult::True : CheckResult::False;
    }));
    GraphView known({n1, b1, n7, b7}, ir.options());
    check_matches(graph, known,
                  {
                      {n7, b7}
    },
                  {{input, seven}, {number, n7}, {flag, b7}});
    GraphView mismatched({n1, b7}, ir.options());
    check_matches(graph, mismatched, {});
    GraphView unresolved({ns, bs}, ir.options());
    check_matches(graph, unresolved, {}, {}, SearchStatus::Unknown);
    GraphView partial({n1, b1, n7, b7, ns, bs}, ir.options());
    check_matches(graph, partial,
                  {
                      {n7, b7}
    },
                  {{input, seven}, {number, n7}, {flag, b7}}, SearchStatus::Unknown);
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/expr.pdll UnregisteredOpResultIndexing / OpResultMemberAccess.
// PDL corroboration: test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-matcher.mlir @results_as_operands;
// test/Rewrite/pdl-bytecode.mlir test.get_defining_op_1.
TEST_CASE("PDLL adapted result identity: Nat and Bool projections must belong to the same producer") {
    PDLLIR ir;
    auto px = ir.call(ir.mixed, ir.x), py = ir.call(ir.mixed, ir.y);
    auto x0 = ir.w.extract(px, 2, 0), x1 = ir.w.extract(px, 2, 1);
    auto y0 = ir.w.extract(py, 2, 0), y1 = ir.w.extract(py, 2, 1);
    auto nx = ir.call(ir.n, x0), bx = ir.call(ir.tag, x1);
    auto ny = ir.call(ir.n, y0), by = ir.call(ir.tag, y1);
    PatternProgram p;
    auto input = p.capture(), producer = p.capture(call(p, ir.mixed, input));
    auto nat_type = p.capture(p.exact(ir.nat)), bool_type = p.capture(p.exact(ir.boolean));
    auto number = p.capture(p.all(projection(p, producer, ir.w.lit_ff()), p.type(nat_type)));
    auto flag   = p.capture(p.all(projection(p, producer, ir.w.lit_tt()), p.type(bool_type)));
    GraphPattern graph(p);
    auto a = graph.anchor(producer), n = graph.anchor(call(p, ir.n, number)), b = graph.anchor(call(p, ir.tag, flag));
    graph.used_by(a, n, {.group = 0, .result_path = std::vector<size_t>{0}});
    graph.used_by(a, b, {.group = 0, .result_path = std::vector<size_t>{1}});
    GraphView view({nx, bx, ny, by}, ir.options());
    auto gid = ir.w.curr_gid();
    check_search(
        [&](auto options) {
            return find_matches(graph, view, options);
    },
        {{px, nx, bx}, {py, ny, by}},
        [&](const GraphMatch& m) {
            bool first = m.at(a) == px;
            CHECK(m.isa_capture(producer) == (first ? px : py));
            CHECK(m.isa_capture(input) == (first ? ir.x : ir.y));
            CHECK(m.isa_capture(number) == (first ? x0 : y0));
            CHECK(m.isa_capture(flag) == (first ? x1 : y1));
            CHECK(m.isa_capture(nat_type) == ir.nat);
            CHECK(m.isa_capture(bool_type) == ir.boolean);
        });
    GraphView crossed({nx, by}, ir.options());
    check_matches(graph, crossed, {});
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: docs/PDLL.md Replacing Multiple Operations / Multi-result operations.
// PDL runtime: test/Integration/Dialect/PDL/CPU/multiroot.mlir
// softmax_cross_entropy (test.mlp_split), replace op5 before op2.
TEST_CASE("PDLL adapted multi-root replacement: heterogeneous ordinals survive reversed registration") {
    PDLLIR ir;
    auto nx = ir.call(ir.n, ir.x), bx = ir.call(ir.b, ir.x), unmatched = ir.call(ir.n, ir.y);
    GraphView view({nx, bx, unmatched}, ir.options());
    MixedRoots pattern(ir);
    check_matches(pattern.graph, view,
                  {
                      {nx, bx}
    },
                  {{pattern.input, ir.x}});
    auto found = find_matches(pattern.graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    for (bool reverse : {false, true}) {
        RewritePlan plan(found.matches[0]);
        auto shared = plan.call(ir.mixed, {plan.import(found.matches[0].isa_capture(pattern.input))});
        for (size_t i = 0; i != 2; ++i) {
            auto slot = reverse ? 1 - i : i;
            plan.replace(slot == 0 ? nx : bx, plan.extract(shared, slot));
        }
        auto result = build(view, plan);
        REQUIRE(result.roots().size() == 3);
        auto number = result.roots()[0]->as<Extract>(), flag = result.roots()[1]->as<Extract>();
        CHECK(Lit::isa(number->index()) == 0);
        CHECK(Lit::isa(flag->index()) == 1);
        CHECK(number->type() == result.world().type_nat());
        CHECK(flag->type() == result.world().type_bool());
        CHECK(number->tuple() == flag->tuple());
        auto producer = number->tuple()->as<App>();
        CHECK(producer->axm()->flags() == ir.mixed->flags());
        CHECK(producer->arg()->as<Axm>()->flags() == ir.x->flags());
        CHECK(producer->type() == result.world().sigma({result.world().type_nat(), result.world().type_bool()}));
        auto kept = result.roots()[2]->as<App>();
        CHECK(kept->axm()->flags() == ir.n->flags());
        CHECK(kept->arg()->as<Axm>()->flags() == ir.y->flags());
        GraphView target(result.roots(), ir.options());
        CHECK(operation_flags(target) == std::multiset<flags_t>{ir.mixed->flags(), ir.n->flags()});
        CHECK(result.replacements() == 2);
        CHECK(ir.w.curr_gid() == gid);
        CHECK(view.is_current());
    }
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/stmt.pdll ReplaceStmt / ReplaceStmtNested.
// PDL runtime: test/Rewrite/pdl-bytecode.mlir test.replace_op_1, consumers receive an existing operand.
TEST_CASE("PDLL adapted replace with existing operand: mixed tuple projections normalize without new calls") {
    PDLLIR ir;
    auto input = ir.w.tuple({ir.x, ir.flag}), wrapped = ir.call(ir.wrap, input);
    auto number = ir.call(ir.n, ir.w.extract(wrapped, 2, 0));
    auto flag   = ir.call(ir.tag, ir.w.extract(wrapped, 2, 1));
    GraphView view({number, flag}, ir.options());
    PatternProgram p;
    auto operand = p.capture(p.type(p.exact(ir.mixed_type)));
    GraphPattern graph(p);
    auto root = graph.anchor(call(p, ir.wrap, operand));
    check_matches(graph, view,
                  {
                      {wrapped}
    },
                  {{operand, input}});
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(root), plan.import(found.matches[0].isa_capture(operand)));
    auto result = build(view, plan);
    REQUIRE(result.roots().size() == 2);
    auto rebuilt_number = result.roots()[0]->as<App>(), rebuilt_flag = result.roots()[1]->as<App>();
    CHECK(rebuilt_number->axm()->flags() == ir.n->flags());
    CHECK(rebuilt_flag->axm()->flags() == ir.tag->flags());
    CHECK(rebuilt_number->arg()->as<Axm>()->flags() == ir.x->flags());
    CHECK(rebuilt_flag->arg()->as<Axm>()->flags() == ir.flag->flags());
    CHECK(rebuilt_number->arg()->type() == result.world().type_nat());
    CHECK(rebuilt_flag->arg()->type() == result.world().type_bool());
    CHECK_FALSE(rebuilt_number->arg()->isa<Extract>());
    CHECK_FALSE(rebuilt_flag->arg()->isa<Extract>());
    GraphView target(result.roots(), ir.options());
    CHECK(operation_flags(target) == std::multiset<flags_t>{ir.n->flags(), ir.tag->flags()});
    CHECK(result.replacements() == 1);
    CHECK(wrapped->as<App>()->arg() == input);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/expr.pdll TupleMemberAccessNumber / TupleMemberAccessName;
// docs/PDLL.md Tuples.
// MimIR recipe tuples contain typed Def values, not PDLL Op/Attr/Type language entities.
TEST_CASE("PDLL adapted tuple member access: reversed mixed fields route existing values by type") {
    PDLLIR ir;
    auto number = ir.pair(ir.choose, ir.x, ir.flag), flag = ir.call(ir.tag, ir.flag);
    GraphView view({number, flag}, ir.options());
    PatternProgram p;
    auto value = p.capture(p.type(p.exact(ir.nat))), condition = p.capture(p.type(p.exact(ir.boolean)));
    GraphPattern graph(p);
    graph.anchor(pair(p, ir.choose, value, condition));
    graph.anchor(call(p, ir.tag, condition));
    check_matches(graph, view,
                  {
                      {number, flag}
    },
                  {{value, ir.x}, {condition, ir.flag}});
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan plan(found.matches[0]);
    auto members = plan.tuple(
        {plan.import(found.matches[0].isa_capture(condition)), plan.import(found.matches[0].isa_capture(value))});
    plan.replace(flag, plan.extract(members, 0));
    plan.replace(number, plan.extract(members, 1));
    auto result = build(view, plan);
    REQUIRE(result.roots().size() == 2);
    CHECK(result.roots()[0]->as<Axm>()->flags() == ir.x->flags());
    CHECK(result.roots()[1]->as<Axm>()->flags() == ir.flag->flags());
    CHECK(result.roots()[0]->type() == result.world().type_nat());
    CHECK(result.roots()[1]->type() == result.world().type_bool());
    for (auto root : result.roots())
        CHECK(&root->world() == &result.world());
    GraphView target(result.roots(), ir.options());
    CHECK(target.operations().empty());
    CHECK(result.replacements() == 2);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// PDLL: docs/PDLL.md BuildFooOp / ReshapeReshapeOptPattern, replace with a new operation.
// PDL lowering corroboration: test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-rewriter.mlir
// @operation_operands, producer precedes consumer.
TEST_CASE("PDLL adapted dependent creation: both typed consumers reuse one new mixed producer") {
    PDLLIR ir;
    auto number = ir.call(ir.n, ir.x), flag = ir.call(ir.b, ir.x);
    GraphView view({number, flag}, ir.options());
    MixedRoots pattern(ir);
    check_matches(pattern.graph, view,
                  {
                      {number, flag}
    },
                  {{pattern.input, ir.x}});
    auto found = find_matches(pattern.graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    for (bool reverse : {false, true}) {
        RewritePlan plan(found.matches[0]);
        auto producer   = plan.call(ir.mixed, {plan.import(ir.x)});
        auto new_flag   = plan.call(ir.tag, {plan.extract(producer, 1)});
        auto new_number = plan.call(ir.n, {plan.extract(producer, 0)});
        if (reverse) {
            plan.replace(flag, new_flag);
            plan.replace(number, new_number);
        } else {
            plan.replace(number, new_number);
            plan.replace(flag, new_flag);
        }
        auto result = build(view, plan);
        REQUIRE(result.roots().size() == 2);
        auto n = result.roots()[0]->as<App>(), b = result.roots()[1]->as<App>();
        CHECK(n->axm()->flags() == ir.n->flags());
        CHECK(b->axm()->flags() == ir.tag->flags());
        CHECK(n->type() == result.world().type_nat());
        CHECK(b->type() == result.world().type_bool());
        auto n_input = n->arg()->as<Extract>(), b_input = b->arg()->as<Extract>();
        CHECK(Lit::isa(n_input->index()) == 0);
        CHECK(Lit::isa(b_input->index()) == 1);
        CHECK(n_input->tuple() == b_input->tuple());
        CHECK(n_input->tuple()->as<App>()->axm()->flags() == ir.mixed->flags());
        CHECK(n_input->tuple()->as<App>()->arg()->as<Axm>()->flags() == ir.x->flags());
        GraphView target(result.roots(), ir.options());
        CHECK(operation_flags(target) == std::multiset<flags_t>{ir.mixed->flags(), ir.n->flags(), ir.tag->flags()});
        CHECK(result.replacements() == 2);
        CHECK(ir.w.curr_gid() == gid);
        CHECK(view.is_current());
    }
}

// PDLL: test/mlir-pdll/Parser/expr-failure.pdll invalid tuple member 2 / "result index is too large";
// test/mlir-pdll/CodeGen/MLIR/expr.pdll UnregisteredOpResultIndexing.
// This is recipe validation, not a PDLL diagnostic.
TEST_CASE("PDLL adapted invalid member: failure in a dependent Bool recipe leaves the scoped source atomic") {
    PDLLIR ir;
    auto fn     = ir.w.mut_lam(ir.w.pi(ir.nat, ir.mixed_type))->set("pdll_invalid_member");
    auto number = ir.call(ir.n, fn->var()), flag = ir.call(ir.b, fn->var());
    fn->set(false, ir.w.tuple({number, flag}))->externalize();
    auto name = fn->sym();
    auto body = fn->body();
    GraphView view({body}, ir.options(fn));
    MixedRoots pattern(ir);
    check_matches(pattern.graph, view,
                  {
                      {number, flag}
    },
                  {{pattern.input, fn->var()}});
    auto found = find_matches(pattern.graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan invalid(found.matches[0]);
    auto shared = invalid.call(ir.mixed, {invalid.import(found.matches[0].isa_capture(pattern.input))});
    invalid.replace(number, invalid.extract(shared, 0));
    invalid.replace(flag, invalid.call(ir.tag, {invalid.extract(shared, 2)}));
    CHECK_THROWS_WITH_AS(build(view, invalid), "recipe projection has unknown or invalid arity", std::invalid_argument);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(ir.w.externals()[name] == fn);
    CHECK(fn->body() == body);
    CHECK(view.is_current());
    CHECK(found.matches[0].at(pattern.number) == number);
    RewritePlan valid(found.matches[0]);
    auto corrected = valid.call(ir.mixed, {valid.import(found.matches[0].isa_capture(pattern.input))});
    valid.replace(flag, valid.extract(corrected, 1));
    valid.replace(number, valid.extract(corrected, 0));
    auto result = build(view, valid);
    REQUIRE(result.scope());
    CHECK(result.scope()->body()->as<App>()->axm()->flags() == ir.mixed->flags());
    CHECK_FALSE(result.is_published());
    CHECK(fn->body() == body);
    CHECK(ir.w.externals()[name] == fn);
    CHECK(ir.w.curr_gid() == gid);
}

// PDLL: docs/PDLL.md replace statement; test/mlir-pdll/CodeGen/MLIR/expr.pdll OpResultMemberAccess.
// PDL lowering corroboration: test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-rewriter.mlir
// @replace_with_values; MimIR additionally rejects type-changing recipes.
TEST_CASE("PDLL adapted incompatible member: one wrong heterogeneous ordinal rejects the entire scoped plan") {
    PDLLIR ir;
    auto fn     = ir.w.mut_lam(ir.w.pi(ir.nat, ir.mixed_type))->set("pdll_incompatible_member");
    auto number = ir.call(ir.n, fn->var()), flag = ir.call(ir.b, fn->var());
    fn->set(false, ir.w.tuple({number, flag}))->externalize();
    auto name = fn->sym();
    auto body = fn->body();
    GraphView view({body}, ir.options(fn));
    MixedRoots pattern(ir);
    check_matches(pattern.graph, view,
                  {
                      {number, flag}
    },
                  {{pattern.input, fn->var()}});
    auto found = find_matches(pattern.graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    for (size_t wrong_slot : {size_t(0), size_t(1)}) {
        RewritePlan plan(found.matches[0]);
        auto producer = plan.call(ir.mixed, {plan.import(found.matches[0].isa_capture(pattern.input))});
        plan.replace(flag, plan.extract(producer, wrong_slot == 1 ? 0 : 1));
        plan.replace(number, plan.extract(producer, wrong_slot == 0 ? 1 : 0));
        CHECK_THROWS_WITH_AS(build(view, plan), "replacement does not preserve the source type", std::invalid_argument);
        CHECK(ir.w.externals()[name] == fn);
        CHECK(fn->body() == body);
        CHECK(ir.w.curr_gid() == gid);
        CHECK(view.is_current());
        CHECK(found.matches[0].isa_capture(pattern.input) == fn->var());
    }
}

// PDLL: docs/PDLL.md Replacing Multiple Operations / accessing operation results.
// PDL runtime: test/Integration/Dialect/PDL/CPU/multiroot.mlir softmax_cross_entropy;
// aggregate RHS lowering is MimIR-specific.
TEST_CASE("PDLL adapted graph Rule: aggregate heterogeneous RHS maps each independent root to its ordinal") {
    PDLLIR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto meta = rule->var();
    rule->set(ir.w.tuple({ir.call(ir.n, meta), ir.call(ir.b, meta)}), ir.call(ir.mixed, meta), ir.w.lit_tt());
    auto nx = ir.call(ir.n, ir.x), bx = ir.call(ir.b, ir.x);
    auto ny = ir.call(ir.n, ir.y), by = ir.call(ir.b, ir.y);
    GraphView view({nx, bx, ny, by}, ir.options());
    GraphRulePattern pattern(rule);
    auto gid = ir.w.curr_gid();
    check_search(
        [&](auto options) {
            return pattern.find(view, options);
    },
        {{nx, bx}, {ny, by}},
        [&](const GraphMatch& m) {
            auto planned = pattern.plan(m, view);
            REQUIRE(planned.status == MatchStatus::Matched);
            REQUIRE(planned.plan);
            auto result = build(view, *planned.plan);
            REQUIRE(result.roots().size() == 4);
            size_t slot = m.anchors()[0] == nx ? 0 : 2;
            auto number = result.roots()[slot]->as<Extract>(), flag = result.roots()[slot + 1]->as<Extract>();
            CHECK(number->tuple() == flag->tuple());
            CHECK(Lit::isa(number->index()) == 0);
            CHECK(Lit::isa(flag->index()) == 1);
            CHECK(number->type() == result.world().type_nat());
            CHECK(flag->type() == result.world().type_bool());
            auto shared = number->tuple()->as<App>();
            CHECK(shared->axm()->flags() == ir.mixed->flags());
            CHECK(shared->arg()->as<Axm>()->flags() == (slot == 0 ? ir.x->flags() : ir.y->flags()));
            size_t kept = slot == 0 ? 2 : 0;
            CHECK(result.roots()[kept]->as<App>()->axm()->flags() == ir.n->flags());
            CHECK(result.roots()[kept + 1]->as<App>()->axm()->flags() == ir.b->flags());
            CHECK(result.roots()[kept]->as<App>()->arg() == result.roots()[kept + 1]->as<App>()->arg());
            CHECK(result.roots()[kept]->as<App>()->arg()->as<Axm>()->flags()
                  == (kept == 0 ? ir.x->flags() : ir.y->flags()));
            GraphView target(result.roots(), ir.options());
            CHECK(operation_flags(target) == std::multiset<flags_t>{ir.mixed->flags(), ir.n->flags(), ir.b->flags()});
            CHECK(result.replacements() == 2);
        });
    GraphView crossed({nx, by}, ir.options());
    check_search([&](auto options) { return pattern.find(crossed, options); }, {}, [](const GraphMatch&) {});
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// PDLL: test/mlir-pdll/CodeGen/MLIR/stmt.pdll ReplaceStmtNested;
// docs/PDLL.md replace with multiple input values. PDL lowering corroboration:
// test/Conversion/PDLToPDLInterp/pdl-to-pdl-interp-rewriter.mlir @replace_with_values.
// Guard proof remains three-valued in MimIR.
TEST_CASE("PDLL adapted pass-through Rule: captured Nat and Bool values survive only a proven guard") {
    PDLLIR ir;
    auto rule  = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.boolean, ir.nat})));
    auto value = rule->var(3, 0), condition = rule->var(3, 1), unused = rule->var(3, 2);
    auto operands = ir.w.tuple({value, condition});
    rule->set(ir.w.tuple({ir.call(ir.take_nat, operands), ir.call(ir.take_bool, operands)}),
              ir.w.tuple({value, condition}), condition);
    const Def* flag = nullptr;
    auto expected   = MatchStatus::Unknown;
    SUBCASE("true captured Bool") {
        flag     = ir.w.lit_tt();
        expected = MatchStatus::Matched;
    }
    SUBCASE("false captured Bool") {
        flag     = ir.w.lit_ff();
        expected = MatchStatus::No_Match;
    }
    SUBCASE("symbolic captured Bool") { flag = ir.flag; }
    auto one    = ir.w.lit_nat(1);
    auto inputs = ir.w.tuple({one, flag});
    auto number = ir.call(ir.take_nat, inputs), boolean = ir.call(ir.take_bool, inputs);
    auto candidate = ir.w.tuple({number, boolean});
    RulePattern single(rule);
    GraphRulePattern graph(rule);
    GraphView view({number, boolean}, ir.options());
    auto gid      = ir.w.curr_gid();
    auto bindings = single.match_lhs(candidate);
    REQUIRE(bindings.status == MatchStatus::Matched);
    REQUIRE(bindings.bindings.size() == 2);
    CHECK(bindings.bindings.at(value) == one);
    CHECK(bindings.bindings.at(condition) == flag);
    CHECK_FALSE(bindings.bindings.contains(unused));
    check_search(
        [&](auto options) {
            return graph.find(view, options);
    },
        {{number, boolean}},
        [&](const GraphMatch& m) {
            auto planned = graph.plan(m, view);
            CHECK(planned.status == expected);
            CHECK(bool(planned.plan) == (expected == MatchStatus::Matched));
            if (expected != MatchStatus::Matched) return;
            auto result = build(view, *planned.plan);
            REQUIRE(result.roots().size() == 2);
            CHECK(result.roots()[0] == result.world().lit_nat(1));
            CHECK(result.roots()[1] == result.world().lit_tt());
            CHECK(result.roots()[0]->type() == result.world().type_nat());
            CHECK(result.roots()[1]->type() == result.world().type_bool());
            GraphView target(result.roots(), ir.options());
            CHECK(target.operations().empty());
            CHECK(result.replacements() == 2);
        });
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// PDLL: docs/PDLL.md Variable "Binding", unbound arg cannot be used as a replacement;
// test/mlir-pdll/Parser/expr-failure.pdll undefined reference behavior is NOT a MimIR parser adaptation.
TEST_CASE("PDLL adapted missing binding: an absent Bool in the guard or mixed RHS rejects before creation") {
    PDLLIR ir;
    auto rule  = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.boolean, ir.nat})));
    auto value = rule->var(3, 0), missing = rule->var(3, 1);
    auto lhs = ir.w.tuple({ir.call(ir.n, value), ir.call(ir.b, value)});
    SUBCASE("Bool required only by the mixed RHS") { rule->set(lhs, ir.w.tuple({value, missing}), ir.w.lit_tt()); }
    SUBCASE("Bool required only by the guard of a new mixed operation") {
        rule->set(lhs, ir.call(ir.mixed, value), missing);
    }
    auto one    = ir.w.lit_nat(1);
    auto number = ir.call(ir.n, one), flag = ir.call(ir.b, one);
    auto fn = ir.w.mut_lam(ir.w.pi(ir.w.sigma(), ir.mixed_type))->set("pdll_missing_binding");
    fn->set(false, ir.w.tuple({number, flag}))->externalize();
    auto name = fn->sym();
    auto body = fn->body();
    RulePattern single(rule);
    GraphRulePattern graph(rule);
    GraphView view({body}, ir.options(fn));
    auto gid      = ir.w.curr_gid();
    auto bindings = single.match_lhs(body);
    REQUIRE(bindings.status == MatchStatus::Matched);
    REQUIRE(bindings.bindings.size() == 1);
    CHECK(bindings.bindings.at(value) == one);
    CHECK_FALSE(bindings.bindings.contains(missing));
    check_search(
        [&](auto options) {
            return graph.find(view, options);
    },
        {{number, flag}},
        [&](const GraphMatch& m) {
            auto planned = graph.plan(m, view);
            CHECK(planned.status == MatchStatus::Unknown);
            CHECK_FALSE(planned.plan);
            CHECK(planned.reason == "guard or RHS contains unbound metavariables");
        });
    CHECK(ir.w.curr_gid() == gid);
    CHECK(ir.w.externals()[name] == fn);
    CHECK(fn->body() == body);
    CHECK(view.is_current());
}

// PDLL: test/lib/Tools/PDLL/TestPDLL.pdll TestSimplePattern;
// test/mlir-pdll/Integration/test-pdll.mlir @simpleTest / @testWithConstraint.
// This executes MimIR's GraphRuleRewrite phase, not MLIR's generated PDLL pass or interfaces.
TEST_CASE("PDLL adapted phase: mixed graph Rule publishes one producer and preserves a mismatched function") {
    PDLLIR ir;
    auto rule  = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.boolean, ir.nat})));
    auto value = rule->var(3, 0), condition = rule->var(3, 1);
    auto operands = ir.w.tuple({value, condition});
    rule->set(ir.w.tuple({ir.call(ir.take_nat, operands), ir.call(ir.take_bool, operands)}), ir.call(ir.wrap, operands),
              ir.w.lit_tt());
    auto good  = ir.w.mut_lam(ir.w.pi(ir.w.sigma({ir.nat, ir.boolean, ir.nat}), ir.mixed_type))->set("pdll_phase_good");
    auto input = ir.w.tuple({good->var(3, 0), good->var(3, 1)});
    good->set(false, ir.w.tuple({ir.call(ir.take_nat, input), ir.call(ir.take_bool, input)}))->externalize();
    auto bad    = ir.w.mut_lam(ir.w.pi(ir.w.sigma({ir.nat, ir.nat, ir.boolean}), ir.mixed_type))->set("pdll_phase_bad");
    auto first  = ir.w.tuple({bad->var(3, 0), bad->var(3, 2)});
    auto second = ir.w.tuple({bad->var(3, 1), bad->var(3, 2)});
    bad->set(false, ir.w.tuple({ir.call(ir.take_nat, first), ir.call(ir.take_bool, second)}))->externalize();
    auto good_name = good->sym(), bad_name = bad->sym();
    auto n_flags = ir.take_nat->flags(), b_flags = ir.take_bool->flags(), wrap_flags = ir.wrap->flags();
    auto configuration = ir.w.tuple({rule, ir.w.tuple({ir.take_nat, ir.take_bool, ir.wrap})});
    auto configure     = ir.w.axm(ir.w.pi(configuration->type(), ir.nat));
    GraphRuleRewrite phase(ir.w, "pdll_mixed_phase");
    phase.apply(ir.call(configure, configuration)->as<App>());
    phase.run();
    CHECK(phase.todo());
    auto current_good = ir.w.externals()[good_name]->as<Lam>();
    auto current_bad  = ir.w.externals()[bad_name]->as<Lam>();
    auto producer     = current_good->body()->as<App>();
    CHECK(producer->axm()->flags() == wrap_flags);
    CHECK(producer->type() == ir.w.sigma({ir.w.type_nat(), ir.w.type_bool()}));
    auto captured = producer->arg()->as<Tuple>();
    REQUIRE(captured->num_ops() == 2);
    CHECK(captured->op(0) == current_good->var(3, 0));
    CHECK(captured->op(1) == current_good->var(3, 1));
    CHECK(captured->op(0)->type() == ir.w.type_nat());
    CHECK(captured->op(1)->type() == ir.w.type_bool());
    auto preserved = current_bad->body()->as<Tuple>();
    REQUIRE(preserved->num_ops() == 2);
    auto number = preserved->op(0)->as<App>(), flag = preserved->op(1)->as<App>();
    CHECK(number->axm()->flags() == n_flags);
    CHECK(flag->axm()->flags() == b_flags);
    CHECK(number->arg()->as<Tuple>()->op(0) == current_bad->var(3, 0));
    CHECK(flag->arg()->as<Tuple>()->op(0) == current_bad->var(3, 1));
    CHECK(number->arg()->as<Tuple>()->op(1) == current_bad->var(3, 2));
    CHECK(flag->arg()->as<Tuple>()->op(1) == current_bad->var(3, 2));
    CHECK(current_good->is_closed());
    CHECK(current_bad->is_closed());
    OpRegistry current_ops;
    for (auto flags : {n_flags, b_flags, wrap_flags})
        current_ops.add(flags, 1,
                        {
                            {0, 1}
        },
                        true);
    GraphView rewritten({current_good->body()}, {.operators = &current_ops});
    GraphView untouched({current_bad->body()}, {.operators = &current_ops});
    CHECK(operation_flags(rewritten) == std::multiset<flags_t>{wrap_flags});
    CHECK(operation_flags(untouched) == std::multiset<flags_t>{n_flags, b_flags});
}

TEST_CASE("Review graph Rule: an instantiated add-zero recipe is a canonical scoped no-op") {
    PDLLIR ir;
    auto add = ir.w.axm(add_zero, 1, Axm::Trip_End, ir.w.pi(ir.w.arr(2, ir.nat), ir.nat));
    auto k   = ir.w.axm(ir.w.pi(ir.w.arr(2, ir.nat), ir.nat));
    ir.ops.add(add->flags(), 1,
               {
                   {0, 1}
    },
               true);
    ir.ops.add(k->flags(), 1,
               {
                   {0, 1}
    },
               true);
    auto rule = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat, ir.nat})));
    auto x = rule->var(3, 0), y = rule->var(3, 1);
    auto qx = ir.call(ir.n, x), kxy = ir.w.app(k, {x, y});
    rule->set(ir.w.tuple({qx, kxy}), ir.w.tuple({ir.w.app(add, {qx, y}), kxy}), ir.w.lit_tt());
    REQUIRE(rule->lhs() != rule->rhs());
    auto fn = ir.w.mut_lam(ir.w.pi(ir.nat, ir.w.arr(2, ir.nat)))->set("normalized_noop");
    auto q = ir.call(ir.n, fn->var()), key = ir.w.app(k, {fn->var(), ir.w.lit_nat(0)});
    auto body = ir.w.tuple({q, key});
    fn->set(false, body)->externalize();
    auto name = fn->sym();
    CHECK(ir.w.app(add, {q, ir.w.lit_nat(0)}) == q);
    GraphRulePattern compiled(rule);
    GraphView view({body}, ir.options(fn));
    auto found = compiled.find(view);
    REQUIRE(found.status == SearchStatus::Complete);
    REQUIRE(found.matches.size() == 1);
    auto gid     = ir.w.curr_gid();
    auto planned = compiled.plan(found.matches[0], view);
    REQUIRE(planned.status == MatchStatus::Matched);
    REQUIRE(planned.plan);
    {
        auto result = build(view, *planned.plan);
        CHECK_FALSE(result.is_changed());
        CHECK_FALSE(result.is_published());
        REQUIRE(result.scope()->body()->isa<Tuple>());
        auto fields = result.scope()->body()->as<Tuple>();
        CHECK(fields->op(0)->as<App>()->axm()->flags() == ir.n->flags());
        CHECK(fields->op(0)->as<App>()->arg() == result.scope()->var());
        auto inputs = fields->op(1)->as<App>()->arg()->as<Tuple>();
        CHECK(inputs->op(0) == result.scope()->var());
        CHECK(inputs->op(1) == result.world().lit_nat(0));
    }
    CHECK(ir.w.curr_gid() == gid);
    CHECK(ir.w.externals()[name] == fn);
    CHECK(fn->body() == body);
    CHECK(view.is_current());
    CHECK(found.matches[0].anchors()[0] == q);
}

TEST_CASE("P5 graph phase: a normalized no-op Rule does not hide a productive Rule") {
    PDLLIR ir;
    auto add  = ir.w.axm(add_zero, 1, Axm::Trip_End, ir.w.pi(ir.w.arr(2, ir.nat), ir.nat));
    auto k    = ir.w.axm(ir.w.pi(ir.w.arr(2, ir.nat), ir.nat));
    auto noop = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat, ir.nat})));
    auto x = noop->var(3, 0), y = noop->var(3, 1);
    auto qx = ir.call(ir.n, x), kxy = ir.w.app(k, {x, y});
    noop->set(ir.w.tuple({qx, kxy}), ir.w.tuple({ir.w.app(add, {qx, y}), kxy}), ir.w.lit_tt());
    auto productive = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto z          = productive->var();
    productive->set(ir.call(ir.n, z), z, ir.w.lit_tt());
    auto fn = ir.w.mut_lam(ir.w.pi(ir.nat, ir.w.arr(2, ir.nat)))->set("phase_normalized_noop");
    fn->set(false, ir.w.tuple({ir.call(ir.n, fn->var()), ir.w.app(k, {fn->var(), ir.w.lit_nat(0)})}))->externalize();
    auto name          = fn->sym();
    auto k_flags       = k->flags();
    auto configuration = ir.w.tuple({ir.w.tuple({noop, productive}), ir.w.tuple({ir.n, k, add})});
    auto configure     = ir.w.axm(ir.w.pi(configuration->type(), ir.nat));
    GraphRuleRewrite phase(ir.w, "normalized_noop_fallback");
    phase.apply(ir.call(configure, configuration)->as<App>());
    phase.run();
    CHECK(phase.todo());
    auto result = ir.w.externals()[name]->as<Lam>();
    auto fields = result->body()->as<Tuple>();
    CHECK(fields->op(0) == result->var());
    CHECK(fields->op(1)->as<App>()->axm()->flags() == k_flags);
}

TEST_CASE("P5 graph phase: search continues past no-op, Unknown and rejected guard candidates") {
    PDLLIR ir;
    auto add   = ir.w.axm(add_zero, 1, Axm::Trip_End, ir.w.pi(ir.w.arr(2, ir.nat), ir.nat));
    auto guard = ir.w.axm(selective_guard, 1, Axm::Trip_End, ir.w.pi(ir.nat, ir.boolean));
    auto rule  = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x     = rule->var();
    auto qx    = ir.call(ir.n, x);
    rule->set(qx, ir.w.app(add, {qx, x}), ir.call(guard, x));
    auto fn = ir.w.mut_lam(ir.w.pi(ir.w.sigma(), ir.w.arr(4, ir.nat)))->set("candidate_fallback");
    DefVec roots;
    for (size_t i = 0; i != 4; ++i)
        roots.push_back(ir.call(ir.n, ir.w.lit_nat(i)));
    fn->set(false, ir.w.tuple(roots))->externalize();
    auto name    = fn->sym();
    auto n_flags = ir.n->flags(), add_flags = add->flags();
    auto configuration = ir.w.tuple({rule, ir.w.tuple({ir.n, add})});
    auto configure     = ir.w.axm(ir.w.pi(configuration->type(), ir.nat));
    GraphRuleRewrite phase(ir.w, "candidate_fallback");
    phase.apply(ir.call(configure, configuration)->as<App>());
    phase.run();
    CHECK(phase.todo());
    auto fields = ir.w.externals()[name]->as<Lam>()->body()->as<Tuple>();
    for (size_t i = 0; i != 3; ++i) {
        auto retained = fields->op(i)->as<App>();
        CHECK(retained->axm()->flags() == n_flags);
        CHECK(Lit::isa(retained->arg()) == i);
    }
    auto changed = fields->op(3)->as<App>();
    CHECK(changed->axm()->flags() == add_flags);
    auto [value, offset] = changed->args<2>();
    CHECK(value->as<App>()->axm()->flags() == n_flags);
    CHECK(Lit::isa(value->as<App>()->arg()) == 3);
    CHECK(Lit::isa(offset) == 3);
}
