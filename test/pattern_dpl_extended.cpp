#include <algorithm>
#include <array>
#include <set>
#include <utility>
#include <vector>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph_rule.h>

using namespace mim;
using namespace mim::pattern;

namespace {

struct ExtendedIR {
    Driver driver;
    World& w       = driver.world();
    const Def* nat = w.type_nat();
    const Def *x = w.axm(nat), *y = w.axm(nat), *z = w.axm(nat);
    const Axm *q = unary(nat, nat), *k = unary(nat, nat), *v = unary(nat, nat);
    const Axm *add = binary(nat, nat, nat), *mm = binary(nat, nat, nat);
    OpRegistry ops;

    ExtendedIR() {
        for (auto op : {q, k, v})
            ops.add(op->flags(), 1, {{0}}, true);
        for (auto op : {add, mm})
            ops.add(op->flags(), 2, {{0}, {1}}, true);
    }

    const Axm* unary(const Def* input, const Def* output) { return w.axm(w.pi(input, output)); }
    const Axm* binary(const Def* a, const Def* b, const Def* output) { return w.axm(w.pi(a, w.pi(b, output))); }
    const Def* call(const Def* op, const Def* a) { return w.app(op, a); }
    const Def* pair(const Def* op, const Def* a, const Def* b) { return w.app(w.app(op, a), b); }
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
    const Def* matrix(const Def* rows, const Def* columns, const Def* element = nullptr) {
        return w.arr(Defs{rows, columns}, element ? element : nat);
    }
};

Pattern call(PatternProgram& p, const Def* op, Pattern a) { return p.node(Node::App, {p.exact(op), a}); }
Pattern pair(PatternProgram& p, const Def* op, Pattern a, Pattern b) { return p.node(Node::App, {call(p, op, a), b}); }
Pattern shape_type(PatternProgram& p, Pattern shape, Pattern element) {
    return p.type(p.node(Node::Arr, {shape, element}));
}

using Tuples = std::set<std::vector<const Def*>>;

Tuples tuples(const GraphMatches& found) {
    Tuples result;
    for (const auto& m : found.matches)
        result.emplace(m.anchors().begin(), m.anchors().end());
    return result;
}

template<class Find, class Inspect>
void check_search(Find find, const Tuples& expected, Inspect inspect) {
    for (auto mode : {GraphSearchMode::Indexed, GraphSearchMode::Exhaustive})
        for (bool reverse : {false, true}) {
            auto found = find(GraphMatchOptions{.reverse_candidates = reverse, .mode = mode});
            REQUIRE(found.status == SearchStatus::Complete);
            CHECK(found.matches.size() == expected.size());
            CHECK(tuples(found) == expected);
            for (const auto& m : found.matches)
                inspect(m);
        }
}

void check_matches(const GraphPattern& pattern,
                   const GraphView& view,
                   const Tuples& expected,
                   std::vector<std::pair<Pattern, const Def*>> captures = {}) {
    auto gid = view.world().curr_gid();
    check_search([&](auto options) { return find_matches(pattern, view, options); }, expected,
                 [&](const auto& m) {
                     for (auto [capture, value] : captures)
                         CHECK(m.isa_capture(capture) == value);
                 });
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

struct CBR {
    const Def *conv0, *bias0, *relu0, *conv1, *bias1, *relu1, *out;
    CBR(ExtendedIR& ir, const Def* input0, const Def* input1)
        : conv0(ir.pair(ir.mm, input0, ir.w.lit_nat(11)))
        , bias0(ir.pair(ir.add, conv0, ir.w.lit_nat(21)))
        , relu0(ir.call(ir.q, bias0))
        , conv1(ir.pair(ir.mm, input1, ir.w.lit_nat(12)))
        , bias1(ir.pair(ir.add, conv1, ir.w.lit_nat(22)))
        , relu1(ir.call(ir.q, bias1))
        , out(ir.pair(ir.add, relu0, relu1)) {}
};

void cbr_pattern(ExtendedIR& ir, PatternProgram& p, GraphPattern& graph, Pattern input) {
    auto c0  = graph.anchor(pair(p, ir.mm, input, p.literal(ir.nat, 11)));
    auto b0  = graph.anchor(pair(p, ir.add, p.any(), p.literal(ir.nat, 21)));
    auto r0  = graph.anchor(call(p, ir.q, p.any()));
    auto c1  = graph.anchor(pair(p, ir.mm, input, p.literal(ir.nat, 12)));
    auto b1  = graph.anchor(pair(p, ir.add, p.any(), p.literal(ir.nat, 22)));
    auto r1  = graph.anchor(call(p, ir.q, p.any()));
    auto out = graph.anchor(pair(p, ir.add, p.any(), p.any()));
    graph.only_used_by(c0, b0, {.group = 0});
    graph.only_used_by(b0, r0);
    graph.only_used_by(c1, b1, {.group = 0});
    graph.only_used_by(b1, r1);
    graph.only_used_by(r0, out, {.group = 0});
    graph.only_used_by(r1, out, {.group = 1});
}

} // namespace

// TVM test_unordered_tuple_pattern: explicit alternatives, not an unordered-tuple API.
TEST_CASE("DPL extended unordered: two explicit orders preserve literal and value captures") {
    ExtendedIR ir;
    auto marker  = ir.w.lit_nat(7);
    auto forward = ir.raw_tuple({marker, ir.x}), backward = ir.raw_tuple({ir.x, marker});
    auto wrong = ir.raw_tuple({ir.x, ir.y}), short_tuple = ir.raw_tuple({marker});
    auto wrong_type = ir.raw_tuple({marker, ir.w.lit_tt()});
    PatternProgram p;
    auto value    = p.capture(p.all(p.node(Node::Axm, {}), p.type(p.exact(ir.nat))));
    auto constant = p.capture(p.literal(ir.nat, 7));
    auto pattern  = p.either(p.node(Node::Tuple, {constant, value}), p.node(Node::Tuple, {value, constant}));
    auto gid      = ir.w.curr_gid();
    for (auto candidate : {forward, backward}) {
        REQUIRE(candidate->node() == Node::Tuple);
        auto found = match(p, pattern, candidate);
        REQUIRE(found.status == MatchStatus::Matched);
        CHECK(found.result->isa_capture(value) == ir.x);
        CHECK(found.result->isa_capture(constant) == marker);
    }
    for (auto candidate : {wrong, short_tuple, wrong_type}) {
        auto found = match(p, pattern, candidate);
        CHECK(found.status == MatchStatus::No_Match);
        CHECK_FALSE(found.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_unordered_tuple_pattern and test_repeated_pattern_match: shared leaves across six explicit orders.
TEST_CASE("DPL extended unordered: all six orders enforce the shared chain input") {
    ExtendedIR ir;
    auto a = ir.call(ir.q, ir.x), b = ir.call(ir.k, ir.x);
    auto marker = ir.w.lit_nat(9);
    std::array<const Def*, 3> fields{a, b, marker};
    std::array<size_t, 3> order{0, 1, 2};
    std::vector<const Def*> good;
    do {
        good.push_back(ir.raw_tuple({fields[order[0]], fields[order[1]], fields[order[2]]}));
    } while (std::next_permutation(order.begin(), order.end()));
    auto wrong_input = ir.raw_tuple({a, ir.call(ir.k, ir.y), marker});
    auto duplicate   = ir.raw_tuple({a, a, marker});
    PatternProgram p;
    auto input = p.capture(), qa = p.capture(call(p, ir.q, input)), kb = p.capture(call(p, ir.k, input));
    std::array patterns{qa, kb, p.literal(ir.nat, 9)};
    auto pattern = p.node(Node::Tuple, {patterns[0], patterns[1], patterns[2]});
    order        = {0, 1, 2};
    while (std::next_permutation(order.begin(), order.end()))
        pattern = p.either(pattern, p.node(Node::Tuple, {patterns[order[0]], patterns[order[1]], patterns[order[2]]}));
    auto gid = ir.w.curr_gid();
    REQUIRE(good.size() == 6);
    for (auto candidate : good) {
        auto found = match(p, pattern, candidate);
        REQUIRE(found.status == MatchStatus::Matched);
        CHECK(found.result->isa_capture(input) == ir.x);
        CHECK(found.result->isa_capture(qa) == a);
        CHECK(found.result->isa_capture(kb) == b);
    }
    for (auto candidate : {wrong_input, duplicate}) {
        auto found = match(p, pattern, candidate);
        CHECK(found.status == MatchStatus::No_Match);
        CHECK_FALSE(found.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_unordered_tuple_pattern and test_simple_used_by: permutations join a separate operation root.
TEST_CASE("DPL extended unordered graph: independently enumerated tuples join both input families") {
    ExtendedIR ir;
    auto marker = ir.w.lit_nat(7);
    auto tx = ir.raw_tuple({ir.x, marker}), xt = ir.raw_tuple({marker, ir.x});
    auto ty = ir.raw_tuple({ir.y, marker}), yt = ir.raw_tuple({marker, ir.y});
    auto qx = ir.call(ir.q, ir.x), qy = ir.call(ir.q, ir.y);
    auto bad = ir.raw_tuple({ir.z, marker});
    GraphView view({tx, xt, ty, yt, bad, qx, qy}, ir.options());
    PatternProgram p;
    auto input = p.capture();
    GraphPattern graph(p);
    auto tuple   = graph.anchor(p.either(p.node(Node::Tuple, {input, p.literal(ir.nat, 7)}),
                                         p.node(Node::Tuple, {p.literal(ir.nat, 7), input})),
                                AnchorDomain::Nodes);
    auto witness = graph.anchor(call(p, ir.q, input));
    auto gid     = ir.w.curr_gid();
    check_search(
        [&](auto options) {
            return find_matches(graph, view, options);
    },
        {{tx, qx}, {xt, qx}, {ty, qy}, {yt, qy}},
        [&](const GraphMatch& m) {
            auto expected = m.at(tuple) == tx || m.at(tuple) == xt ? ir.x : ir.y;
            CHECK(m.isa_capture(input) == expected);
            CHECK(m.at(witness) == (expected == ir.x ? qx : qy));
        });
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_op_attr: attributes are explicit typed literal operands in this adaptation.
TEST_CASE("DPL extended operation constraints: flags callee type and stride literals are conjunctive") {
    ExtendedIR ir;
    auto op     = ir.w.axm(ir.w.pi(ir.nat, ir.w.pi(ir.nat, ir.w.pi(ir.nat, ir.nat))));
    auto other  = ir.w.axm(op->type());
    auto invoke = [&](const Def* callee, u64 a, u64 b) {
        return ir.w.app(ir.w.app(ir.w.app(callee, ir.w.lit_nat(a)), ir.w.lit_nat(b)), ir.x);
    };
    auto good = invoke(op, 3, 3), bad_stride = invoke(op, 4, 3), bad_op = invoke(other, 3, 3);
    PatternProgram p;
    auto callee  = p.capture(p.all(p.node(Node::Axm, {}, op->flags()), p.type(p.exact(op->type()))));
    auto input   = p.capture(p.type(p.exact(ir.nat)));
    auto prefix  = p.node(Node::App, {callee, p.literal(ir.nat, 3)});
    auto pattern = p.node(Node::App, {p.node(Node::App, {prefix, p.literal(ir.nat, 3)}), input});
    auto gid     = ir.w.curr_gid();
    auto found   = match(p, pattern, good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(callee) == op);
    CHECK(found.result->isa_capture(input) == ir.x);
    for (auto candidate : {bad_stride, bad_op}) {
        auto rejected = match(p, pattern, candidate);
        CHECK(rejected.status == MatchStatus::No_Match);
        CHECK_FALSE(rejected.result);
    }
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_extern_fn_pattern and test_simple_call_packed: explicit callee identity, not symbol lookup.
TEST_CASE("DPL extended callee identity: same signature does not authorize another external operation") {
    ExtendedIR ir;
    auto good          = ir.pair(ir.mm, ir.x, ir.w.lit_nat(2));
    auto wrong_callee  = ir.pair(ir.add, ir.x, ir.w.lit_nat(2));
    auto wrong_literal = ir.pair(ir.mm, ir.x, ir.w.lit_nat(3));
    PatternProgram p;
    auto input = p.capture(), callee = p.capture(p.exact(ir.mm)), constant = p.capture(p.literal(ir.nat, 2));
    auto pattern = p.node(Node::App, {p.node(Node::App, {callee, input}), constant});
    auto gid     = ir.w.curr_gid();
    auto found   = match(p, pattern, good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(callee) == ir.mm);
    CHECK(found.result->isa_capture(input) == ir.x);
    CHECK(found.result->isa_capture(constant) == ir.w.lit_nat(2));
    CHECK(match(p, pattern, wrong_callee).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, wrong_literal).status == MatchStatus::No_Match);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_same_shape_pattern: compare declared static shapes, never infer a result shape.
TEST_CASE("DPL extended static shape: equal declared shapes do not require equal values") {
    ExtendedIR ir;
    auto rows = ir.w.lit_nat(1024), columns = ir.w.lit_nat(128);
    auto type = ir.matrix(rows, columns);
    auto a = ir.w.axm(type), b = ir.w.axm(type);
    auto narrow       = ir.w.axm(ir.matrix(ir.w.lit_nat(2), columns));
    auto other_dtype  = ir.w.axm(ir.matrix(rows, columns, ir.w.type_bool()));
    auto unknown_rows = ir.w.axm(ir.matrix(ir.w.axm(ir.nat), columns));
    auto root = ir.raw_tuple({a, b}), bad = ir.raw_tuple({a, narrow});
    auto bad_dtype = ir.raw_tuple({a, other_dtype}), symbolic = ir.raw_tuple({a, unknown_rows});
    PatternProgram p;
    auto shape   = p.capture(p.exact(*type->as<Arr>()->shape()));
    auto lhs     = p.capture(shape_type(p, shape, p.exact(ir.nat)));
    auto rhs     = p.capture(shape_type(p, shape, p.exact(ir.nat)));
    auto pattern = p.node(Node::Tuple, {lhs, rhs});
    auto gid     = ir.w.curr_gid();
    auto found   = match(p, pattern, root);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(lhs) == a);
    CHECK(found.result->isa_capture(rhs) == b);
    CHECK(found.result->isa_capture(shape) == *type->as<Arr>()->shape());
    CHECK(a != b);
    auto rejected = match(p, pattern, bad);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK_FALSE(rejected.result);
    CHECK(match(p, pattern, bad_dtype).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, symbolic).status == MatchStatus::No_Match);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_not_pattern: negating a declared shape is independent of the element type.
TEST_CASE("DPL extended negated shape: declared dimensions determine the result independently of dtype") {
    ExtendedIR ir;
    auto shape       = ir.w.arr(Defs{ir.w.lit_nat(2), ir.w.lit_nat(3), ir.w.lit_nat(3)}, ir.nat);
    auto same        = ir.w.axm(shape);
    auto other_dtype = ir.w.axm(ir.w.arr(Defs{ir.w.lit_nat(2), ir.w.lit_nat(3), ir.w.lit_nat(3)}, ir.w.type_bool()));
    auto different   = ir.w.axm(ir.w.arr(Defs{ir.w.lit_nat(3), ir.w.lit_nat(3), ir.w.lit_nat(3)}, ir.nat));
    PatternProgram p;
    auto pattern = p.negate(shape_type(p, p.exact(*shape->as<Arr>()->shape()), p.any()));
    auto gid     = ir.w.curr_gid();
    CHECK(match(p, pattern, same).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, other_dtype).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, different).status == MatchStatus::Matched);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_same_shape_pattern: symbolic dimension identity is read-only, with no arithmetic solver.
TEST_CASE("DPL extended symbolic shape: shared symbols match and distinct symbols reject") {
    ExtendedIR ir;
    auto n = ir.w.axm(ir.nat), m = ir.w.axm(ir.nat);
    auto columns = ir.w.lit_nat(128);
    auto type    = ir.matrix(n, columns);
    auto a = ir.w.axm(type), b = ir.w.axm(type), c = ir.w.axm(ir.matrix(m, columns));
    auto good = ir.raw_tuple({a, b}), bad = ir.raw_tuple({a, c});
    PatternProgram p;
    auto rows = p.capture(), cols = p.capture(p.literal(ir.nat, 128));
    auto shape = p.capture(p.node(Node::Tuple, {rows, cols}));
    auto lhs = p.capture(shape_type(p, shape, p.any())), rhs = p.capture(shape_type(p, shape, p.any()));
    auto pattern = p.node(Node::Tuple, {lhs, rhs});
    auto gid     = ir.w.curr_gid();
    auto found   = match(p, pattern, good);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(rows) == n);
    CHECK(found.result->isa_capture(cols) == columns);
    CHECK(found.result->isa_capture(shape) == *type->as<Arr>()->shape());
    CHECK(found.result->isa_capture(lhs) == a);
    CHECK(found.result->isa_capture(rhs) == b);
    CHECK(match(p, pattern, bad).status == MatchStatus::No_Match);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_wildcard_struct_info_for_unknown_dtype: omit the element-type constraint, not type inference.
TEST_CASE("DPL extended shape restriction: unspecified element type accepts two declared dtypes") {
    ExtendedIR ir;
    auto rows = ir.w.lit_nat(2), columns = ir.w.lit_nat(3);
    auto nt = ir.matrix(rows, columns), bt = ir.matrix(rows, columns, ir.w.type_bool());
    auto a = ir.w.axm(nt), b = ir.w.axm(bt);
    auto wrong = ir.w.axm(ir.matrix(ir.w.lit_nat(16), ir.w.lit_nat(32)));
    PatternProgram p;
    auto element = p.capture();
    auto pattern = p.capture(shape_type(p, p.exact(*nt->as<Arr>()->shape()), element));
    auto gid     = ir.w.curr_gid();
    for (auto value : {a, b}) {
        auto found = match(p, pattern, value);
        REQUIRE(found.status == MatchStatus::Matched);
        CHECK(found.result->isa_capture(pattern) == value);
        CHECK(found.result->isa_capture(element) == (value == a ? ir.nat : ir.w.type_bool()));
    }
    CHECK(match(p, pattern, wrong).status == MatchStatus::No_Match);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_wildcard_struct_info_with_symbolic_vars: bind explicit dimensions and reject broadcasting.
TEST_CASE("DPL extended shape graph: dimension captures select only the nonbroadcast pair") {
    ExtendedIR ir;
    auto rows = ir.w.lit_nat(64), columns = ir.w.lit_nat(128);
    auto type = ir.matrix(rows, columns);
    auto a = ir.w.axm(type), b = ir.w.axm(type);
    auto narrow = ir.w.axm(ir.matrix(ir.w.lit_nat(2), columns));
    auto left = ir.unary(type, type), right = ir.unary(type, type), broadcast = ir.unary(narrow->type(), type);
    for (auto op : {left, right, broadcast})
        ir.ops.add(op->flags(), 1, {{0}}, true);
    auto la = ir.call(left, a), rb = ir.call(right, b), bc = ir.call(broadcast, narrow);
    PatternProgram p;
    auto r = p.capture(), c = p.capture();
    auto dimensions = p.node(Node::Tuple, {r, c});
    auto lhs        = p.capture(shape_type(p, dimensions, p.any()));
    auto rhs        = p.capture(shape_type(p, dimensions, p.any()));
    GraphPattern graph(p);
    graph.anchor(call(p, left, lhs));
    graph.anchor(p.either(call(p, right, rhs), call(p, broadcast, rhs)));
    GraphView view({la, rb, bc}, ir.options());
    check_matches(graph, view,
                  {
                      {la, rb}
    },
                  {{r, rows}, {c, columns}, {lhs, a}, {rhs, b}});
}

// TVM test_prim_arr_pattern: match shape values themselves, independently of tensor type matching.
TEST_CASE("DPL extended shape values: static Pack and symbolic Tuple dimensions match directly") {
    ExtendedIR ir;
    auto n = ir.w.axm(ir.nat), m = ir.w.axm(ir.nat);
    auto static_shape    = ir.w.tuple({ir.w.lit_nat(32), ir.w.lit_nat(32)});
    auto wrong_static    = ir.w.tuple({ir.w.lit_nat(32), ir.w.lit_nat(31)});
    auto symbolic_shape  = ir.raw_tuple({n, m, ir.pair(ir.add, n, m)});
    auto wrong_operation = ir.raw_tuple({n, m, ir.pair(ir.mm, n, m)});
    auto wrong_order     = ir.raw_tuple({m, n, ir.pair(ir.add, n, m)});
    REQUIRE(static_shape->node() == Node::Pack);
    PatternProgram p;
    auto rows = p.capture(p.exact(n)), columns = p.capture(p.exact(m));
    auto sum       = p.capture(pair(p, ir.add, rows, columns));
    auto symbolic  = p.node(Node::Tuple, {rows, columns, sum});
    auto pattern   = p.either(p.exact(static_shape), symbolic);
    auto gid       = ir.w.curr_gid();
    auto constants = match(p, pattern, static_shape);
    REQUIRE(constants.status == MatchStatus::Matched);
    CHECK(constants.result->isa_capture(rows) == nullptr);
    CHECK(constants.result->isa_capture(columns) == nullptr);
    CHECK(constants.result->isa_capture(sum) == nullptr);
    auto found = match(p, pattern, symbolic_shape);
    REQUIRE(found.status == MatchStatus::Matched);
    CHECK(found.result->isa_capture(rows) == n);
    CHECK(found.result->isa_capture(columns) == m);
    CHECK(found.result->isa_capture(sum) == symbolic_shape->op(2));
    CHECK(match(p, pattern, wrong_operation).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, wrong_order).status == MatchStatus::No_Match);
    CHECK(match(p, pattern, wrong_static).status == MatchStatus::No_Match);
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_two_cbr: duplicated chains are connected by an explicitly shared input and concat.
TEST_CASE("DPL extended two CBR: both complete chains join the same input and ordered output") {
    ExtendedIR ir;
    CBR n(ir, ir.x, ir.x);
    PatternProgram p;
    auto input = p.capture();
    GraphPattern graph(p);
    cbr_pattern(ir, p, graph, input);
    GraphView view({n.out}, ir.options());
    check_matches(graph, view,
                  {
                      {n.conv0, n.bias0, n.relu0, n.conv1, n.bias1, n.relu1, n.out}
    },
                  {{input, ir.x}});
    GraphView escaping({n.out, ir.call(ir.k, n.bias0)}, ir.options());
    check_matches(graph, escaping, {});
}

// TVM test_two_cbr: compatible operators alone do not prove a common fork.
TEST_CASE("DPL extended two CBR counter: different inputs and reversed concat slots reject") {
    ExtendedIR ir;
    CBR different(ir, ir.x, ir.y), shared(ir, ir.x, ir.x);
    PatternProgram p;
    auto input = p.capture();
    GraphPattern graph(p);
    cbr_pattern(ir, p, graph, input);
    GraphView wrong_input({different.out}, ir.options());
    check_matches(graph, wrong_input, {});
    auto reversed = ir.pair(ir.add, shared.relu1, shared.relu0);
    GraphView wrong_order({reversed}, ir.options());
    check_matches(graph, wrong_order, {});
    GraphView good({shared.out}, ir.options());
    check_matches(graph, good,
                  {
                      {shared.conv0, shared.bias0, shared.relu0, shared.conv1, shared.bias1, shared.relu1, shared.out}
    },
                  {{input, ir.x}});
}

// TVM test_nested_context: separate arenas preserve a full chain while a direct-edge shortcut fails.
TEST_CASE("DPL extended nested programs: failing shortcut leaves full CBR and suffix matches unchanged") {
    ExtendedIR ir;
    CBR n(ir, ir.x, ir.x);
    GraphView view({n.out}, ir.options());
    PatternProgram outer;
    auto input = outer.capture();
    GraphPattern full(outer);
    auto conv = full.anchor(pair(outer, ir.mm, input, outer.literal(ir.nat, 11)));
    auto bias = full.anchor(pair(outer, ir.add, outer.any(), outer.literal(ir.nat, 21)));
    auto relu = full.anchor(call(outer, ir.q, outer.any()));
    full.only_used_by(conv, bias, {.group = 0});
    full.only_used_by(bias, relu);
    check_matches(full, view,
                  {
                      {n.conv0, n.bias0, n.relu0}
    },
                  {{input, ir.x}});
    {
        PatternProgram middle;
        GraphPattern shortcut(middle);
        auto c = shortcut.anchor(pair(middle, ir.mm, middle.any(), middle.literal(ir.nat, 11)));
        auto r = shortcut.anchor(call(middle, ir.q, middle.any()));
        shortcut.only_used_by(c, r);
        check_matches(shortcut, view, {});
        {
            PatternProgram inner;
            auto producer = inner.capture(pair(inner, ir.add, inner.any(), inner.literal(ir.nat, 21)));
            GraphPattern suffix(inner);
            auto b = suffix.anchor(producer), r = suffix.anchor(call(inner, ir.q, inner.any()));
            suffix.only_used_by(b, r);
            check_matches(suffix, view,
                          {
                              {n.bias0, n.relu0}
            },
                          {{producer, n.bias0}});
        }
        check_matches(shortcut, view, {});
    }
    check_matches(full, view,
                  {
                      {n.conv0, n.bias0, n.relu0}
    },
                  {{input, ir.x}});
}

// TVM test_two_matmul: typed pure axioms encode the declared intermediate dimensions.
TEST_CASE("DPL extended two matmul: typed chain accepts two stages but not a third or wrong shape") {
    ExtendedIR ir;
    auto a_type      = ir.matrix(ir.w.lit_nat(32), ir.w.lit_nat(16));
    auto b_type      = ir.matrix(ir.w.lit_nat(16), ir.w.lit_nat(48));
    auto c_type      = ir.matrix(ir.w.lit_nat(48), ir.w.lit_nat(32));
    auto middle_type = ir.matrix(ir.w.lit_nat(32), ir.w.lit_nat(48));
    auto output_type = ir.matrix(ir.w.lit_nat(32), ir.w.lit_nat(32));
    auto first_op = ir.binary(a_type, b_type, middle_type), second_op = ir.binary(middle_type, c_type, output_type);
    for (auto op : {first_op, second_op})
        ir.ops.add(op->flags(), 2, {{0}, {1}}, true);
    auto a = ir.w.axm(a_type), b = ir.w.axm(b_type), c = ir.w.axm(c_type);
    auto first = ir.pair(first_op, a, b), second = ir.pair(second_op, first, c);
    PatternProgram p;
    auto ca = p.capture(), cb = p.capture(), cc = p.capture();
    GraphPattern graph(p);
    auto f = graph.anchor(p.all(pair(p, first_op, ca, cb), p.type(p.exact(middle_type))));
    auto s = graph.anchor(p.all(pair(p, second_op, p.any(), cc), p.type(p.exact(output_type))));
    graph.only_used_by(f, s, {.group = 0});
    GraphView view({second}, ir.options());
    check_matches(graph, view,
                  {
                      {first, second}
    },
                  {{ca, a}, {cb, b}, {cc, c}});
    GraphPattern wrong_shape(p);
    auto wf = wrong_shape.anchor(p.all(pair(p, first_op, p.any(), p.any()), p.type(p.exact(output_type))));
    auto ws = wrong_shape.anchor(pair(p, second_op, p.any(), p.any()));
    wrong_shape.used_by(wf, ws);
    check_matches(wrong_shape, view, {});
    auto third = graph.anchor(pair(p, second_op, p.any(), p.any()));
    graph.used_by(s, third, {.group = 0});
    check_matches(graph, view, {});
}

// TVM test_concat_mm_split: scalar Extracts retain the split producer, both result indices and types.
TEST_CASE("DPL extended concat matmul split: typed chain and projected reconvergence match exactly") {
    ExtendedIR ir;
    auto half   = ir.matrix(ir.w.lit_nat(16), ir.w.lit_nat(32));
    auto whole  = ir.matrix(ir.w.lit_nat(32), ir.w.lit_nat(32));
    auto concat = ir.binary(half, half, whole), mm = ir.binary(whole, whole, whole);
    auto result_type = ir.w.mut_sigma(ir.w.type(), 2);
    result_type->set({half, half});
    auto split = ir.unary(whole, result_type), add = ir.binary(half, half, half);
    for (auto op : {concat, mm, add})
        ir.ops.add(op->flags(), 2, {{0}, {1}}, true);
    ir.ops.add(split->flags(), 1, {{0}}, true);
    auto a = ir.w.axm(whole), b = ir.w.axm(half), c = ir.w.axm(half);
    auto joined = ir.pair(concat, b, c), product = ir.pair(mm, a, joined), chunks = ir.call(split, product);
    auto left = ir.w.extract(chunks, 2, 0), right = ir.w.extract(chunks, 2, 1);
    auto output = ir.pair(add, left, right);
    PatternProgram p;
    auto cb = p.capture(), cc = p.capture(), ca = p.capture(), source = p.capture(call(p, split, p.any()));
    GraphPattern graph(p);
    auto j = graph.anchor(pair(p, concat, cb, cc));
    auto m = graph.anchor(pair(p, mm, ca, p.any()));
    auto s = graph.anchor(source);
    auto l = graph.anchor(
        p.all(p.node(Node::Extract, {source, p.exact(left->as<Extract>()->index())}), p.type(p.exact(half))),
        AnchorDomain::Nodes);
    auto r = graph.anchor(
        p.all(p.node(Node::Extract, {source, p.exact(right->as<Extract>()->index())}), p.type(p.exact(half))),
        AnchorDomain::Nodes);
    auto o = graph.anchor(pair(p, add, p.any(), p.any()));
    graph.only_used_by(j, m, {.group = 1});
    graph.only_used_by(m, s);
    graph.only_used_by(l, o, {.group = 0});
    graph.only_used_by(r, o, {.group = 1});
    GraphView view({output}, ir.options());
    check_matches(graph, view,
                  {
                      {joined, product, chunks, left, right, output}
    },
                  {{cb, b}, {cc, c}, {ca, a}, {source, chunks}});
    GraphView wrong({ir.pair(add, left, left)}, ir.options());
    check_matches(graph, wrong, {});
}

// TVM test_self_attention: all three projection chains share x, and their downstream motif is explicit.
TEST_CASE("DPL extended self attention: three shared chains connect scale softmax and final product") {
    ExtendedIR ir;
    auto softmax = ir.unary(ir.nat, ir.nat);
    ir.ops.add(softmax->flags(), 1, {{0}}, true);
    auto wq = ir.w.lit_nat(11), wk = ir.w.lit_nat(12), wv = ir.w.lit_nat(13), scale = ir.w.lit_nat(2);
    auto fq = ir.pair(ir.mm, ir.x, wq), fk = ir.pair(ir.mm, ir.x, wk), fv = ir.pair(ir.mm, ir.x, wv);
    auto tq = ir.call(ir.q, fq), tk = ir.call(ir.q, fk), tv = ir.call(ir.q, fv);
    auto score = ir.pair(ir.mm, tq, tk), scaled = ir.pair(ir.mm, score, scale);
    auto normalized = ir.call(softmax, scaled), output = ir.pair(ir.mm, normalized, tv);
    PatternProgram p;
    auto input = p.capture();
    GraphPattern graph(p);
    auto aq = graph.anchor(pair(p, ir.mm, input, p.exact(wq)));
    auto bq = graph.anchor(call(p, ir.q, p.any()));
    auto ak = graph.anchor(pair(p, ir.mm, input, p.exact(wk)));
    auto bk = graph.anchor(call(p, ir.q, p.any()));
    auto av = graph.anchor(pair(p, ir.mm, input, p.exact(wv)));
    auto bv = graph.anchor(call(p, ir.q, p.any()));
    auto s  = graph.anchor(pair(p, ir.mm, p.any(), p.any()));
    auto t  = graph.anchor(pair(p, ir.mm, p.any(), p.literal(ir.nat, 2)));
    auto n  = graph.anchor(call(p, softmax, p.any()));
    auto o  = graph.anchor(pair(p, ir.mm, p.any(), p.any()));
    graph.only_used_by(aq, bq);
    graph.only_used_by(ak, bk);
    graph.only_used_by(av, bv);
    graph.only_used_by(bq, s, {.group = 0});
    graph.only_used_by(bk, s, {.group = 1});
    graph.only_used_by(s, t, {.group = 0});
    graph.only_used_by(t, n);
    graph.only_used_by(n, o, {.group = 0});
    graph.only_used_by(bv, o, {.group = 1});
    GraphView view({output}, ir.options());
    check_matches(graph, view,
                  {
                      {fq, tq, fk, tk, fv, tv, score, scaled, normalized, output}
    },
                  {{input, ir.x}});
    auto fake_v = ir.call(ir.q, ir.pair(ir.mm, ir.y, wv));
    GraphView fake({ir.pair(ir.mm, normalized, fake_v)}, ir.options());
    check_matches(graph, fake, {});
    auto wrong_scale = ir.call(softmax, ir.pair(ir.mm, score, ir.w.lit_nat(3)));
    GraphView wrong({ir.pair(ir.mm, wrong_scale, tv)}, ir.options());
    check_matches(graph, wrong, {});
}

// TVM test_incremental_solving: transfer an exact witness between separate programs, without PatternContext.
TEST_CASE("DPL extended incremental constraints: independently matched prefix selects the suffix witness") {
    ExtendedIR ir;
    auto a = ir.call(ir.q, ir.x), b = ir.call(ir.k, a), c = ir.call(ir.v, b);
    auto other_b = ir.call(ir.k, ir.y), other_c = ir.call(ir.v, other_b);
    GraphView view({c, other_c}, ir.options());
    PatternProgram prefix;
    auto middle = prefix.capture(call(prefix, ir.k, prefix.any()));
    GraphPattern first(prefix);
    auto f = first.anchor(call(prefix, ir.q, prefix.any())), s = first.anchor(middle);
    first.only_used_by(f, s);
    check_matches(first, view,
                  {
                      {a, b}
    },
                  {{middle, b}});
    auto found = find_matches(first, view);
    REQUIRE(found.matches.size() == 1);
    PatternProgram suffix;
    GraphPattern second(suffix);
    auto witness = second.anchor(suffix.exact(found.matches[0].isa_capture(middle)));
    auto tail    = second.anchor(call(suffix, ir.v, suffix.any()));
    second.only_used_by(witness, tail);
    check_matches(second, view,
                  {
                      {b, c}
    });
    PatternProgram combined;
    GraphPattern conjunction(combined);
    auto x = conjunction.anchor(call(combined, ir.q, combined.any()));
    auto y = conjunction.anchor(call(combined, ir.k, combined.any()));
    auto z = conjunction.anchor(call(combined, ir.v, combined.any()));
    conjunction.only_used_by(x, y);
    conjunction.only_used_by(y, z);
    check_matches(conjunction, view,
                  {
                      {a, b, c}
    });
}

// TVM test_incremental_solving_counter: a fresh suffix succeeds while the full conjunction rejects.
TEST_CASE("DPL extended incremental counter: suffix success cannot erase the absent prefix constraint") {
    ExtendedIR ir;
    auto b = ir.call(ir.k, ir.x), c = ir.call(ir.v, b);
    GraphView view({c}, ir.options());
    PatternProgram suffix;
    GraphPattern independent(suffix);
    auto s = independent.anchor(call(suffix, ir.k, suffix.any()));
    auto t = independent.anchor(call(suffix, ir.v, suffix.any()));
    independent.only_used_by(s, t);
    check_matches(independent, view,
                  {
                      {b, c}
    });
    PatternProgram prefix;
    GraphPattern first(prefix);
    auto a      = first.anchor(call(prefix, ir.q, prefix.any()));
    auto middle = first.anchor(call(prefix, ir.k, prefix.any()));
    first.only_used_by(a, middle);
    check_matches(first, view, {});
    auto tail = first.anchor(call(prefix, ir.v, prefix.any()));
    first.only_used_by(middle, tail);
    check_matches(first, view, {});
    check_matches(independent, view,
                  {
                      {b, c}
    });
}

// TVM test_combine_matmul_emit_order: imported transpose dependencies must precede shared replacement use.
TEST_CASE("DPL extended emission order: nested weight dependencies feed exactly one new recipe chain") {
    ExtendedIR ir;
    auto concat = ir.unary(ir.w.arr(3, ir.nat), ir.nat), split = ir.unary(ir.nat, ir.w.arr(3, ir.nat));
    ir.ops.add(concat->flags(), 1,
               {
                   {0, 1}
    },
               true);
    ir.ops.add(split->flags(), 1, {{0}}, true);
    auto w0 = ir.call(ir.q, ir.w.lit_nat(11));
    auto w1 = ir.call(ir.q, ir.call(ir.q, ir.w.lit_nat(12)));
    auto w2 = ir.call(ir.q, ir.w.lit_nat(13));
    std::array roots{ir.pair(ir.mm, ir.x, w0), ir.pair(ir.mm, ir.x, w1), ir.pair(ir.mm, ir.x, w2)};
    PatternProgram p;
    auto input = p.capture();
    GraphPattern graph(p);
    graph.anchor(pair(p, ir.mm, input, p.exact(w0)));
    graph.anchor(pair(p, ir.mm, input, p.exact(w1)));
    graph.anchor(pair(p, ir.mm, input, p.exact(w2)));
    GraphView view(roots, ir.options());
    check_matches(graph, view,
                  {
                      {roots[0], roots[1], roots[2]}
    },
                  {{input, ir.x}});
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    for (bool reverse : {false, true}) {
        RewritePlan plan(found.matches[0]);
        auto weights = plan.call(concat, {plan.tuple({plan.import(w0), plan.import(w1), plan.import(w2)})});
        auto product = plan.call(ir.mm, {plan.import(found.matches[0].isa_capture(input)), weights});
        auto chunks  = plan.call(split, {product});
        for (size_t i = 0; i != 3; ++i) {
            auto slot = reverse ? 2 - i : i;
            plan.replace(roots[slot], plan.extract(chunks, slot));
        }
        auto rewritten = build(view, plan);
        REQUIRE(rewritten.roots().size() == 3);
        const Def* shared = nullptr;
        for (size_t i = 0; i != 3; ++i) {
            auto ex = rewritten.roots()[i]->as<Extract>();
            CHECK(Lit::isa(ex->index()) == i);
            if (!shared) shared = ex->tuple();
            CHECK(ex->tuple() == shared);
        }
        auto new_mm = shared->as<App>()->arg()->as<App>();
        CHECK(shared->as<App>()->axm()->flags() == split->flags());
        CHECK(new_mm->axm()->flags() == ir.mm->flags());
        auto weights_call = new_mm->arg()->as<App>();
        CHECK(weights_call->axm()->flags() == concat->flags());
        auto values = weights_call->arg()->as<Tuple>();
        REQUIRE(values->num_ops() == 3);
        for (size_t i = 0; i != 3; ++i) {
            auto weight = values->op(i)->as<App>();
            CHECK(weight->axm()->flags() == ir.q->flags());
            if (i == 1) {
                weight = weight->arg()->as<App>();
                CHECK(weight->axm()->flags() == ir.q->flags());
            }
            CHECK(Lit::isa(weight->arg()) == 11 + i);
        }
        GraphView target(rewritten.roots(), ir.options());
        CHECK(operation_flags(target)
              == std::multiset<flags_t>{ir.q->flags(), ir.q->flags(), ir.q->flags(), ir.q->flags(), concat->flags(),
                                        ir.mm->flags(), split->flags()});
        CHECK(rewritten.replacements() == 3);
        CHECK(ir.w.curr_gid() == gid);
        CHECK(view.is_current());
    }
}

// TVM test_iterative_rewrite_with_removed_intermediates: imported replacement keys resolve transitively.
TEST_CASE("DPL extended chained replacement: three replaced keys resolve to the terminal recipe") {
    ExtendedIR ir;
    auto a = ir.call(ir.q, ir.x), b = ir.call(ir.k, a), c = ir.call(ir.v, b);
    GraphView view({a, b, c}, ir.options());
    PatternProgram p;
    GraphPattern graph(p);
    auto x = graph.anchor(p.exact(a)), y = graph.anchor(p.exact(b)), z = graph.anchor(p.exact(c));
    graph.used_by(x, y);
    graph.used_by(y, z);
    check_matches(graph, view,
                  {
                      {a, b, c}
    });
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan plan(found.matches[0]);
    plan.replace(a, plan.import(b));
    plan.replace(b, plan.import(c));
    plan.replace(c, plan.literal(ir.nat, 42));
    auto rewritten = build(view, plan);
    REQUIRE(rewritten.roots().size() == 3);
    for (auto root : rewritten.roots())
        CHECK(root == rewritten.world().lit_nat(42));
    GraphView target(rewritten.roots(), ir.options());
    CHECK(target.operations().empty());
    CHECK(rewritten.replacements() == 3);
    CHECK(view.is_current());
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_iterative_rewrite_with_removed_intermediates: two explicit checked rounds, not automatic convergence.
TEST_CASE("DPL extended removed intermediates: tuple replacement exposes a second checked rewrite") {
    ExtendedIR ir;
    auto concat = ir.binary(ir.nat, ir.nat, ir.w.arr(2, ir.nat));
    auto split  = ir.unary(ir.w.arr(2, ir.nat), ir.w.arr(2, ir.nat));
    ir.ops.add(concat->flags(), 2, {{0}, {1}}, true);
    ir.ops.add(split->flags(), 1, {{0}}, true);
    auto keeper = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("dpl_removed_keep_op");
    keeper->set(false, ir.pair(ir.mm, keeper->var(), ir.w.lit_nat(3)))->externalize();
    auto keeper_name = keeper->sym();
    auto fn = ir.w.mut_lam(ir.w.pi(ir.w.sigma({ir.nat, ir.nat, ir.w.type_bool()}), ir.nat))->set("dpl_removed");
    auto a = fn->var(3, 0), b = fn->var(3, 1);
    auto chunks = ir.call(split, ir.pair(concat, a, b));
    auto left = ir.w.extract(chunks, 2, 0), right = ir.w.extract(chunks, 2, 1);
    auto doubled = ir.pair(ir.add, a, left);
    fn->set(false, ir.pair(ir.add, right, doubled))->externalize();
    auto name      = fn->sym();
    auto add_flags = ir.add->flags(), mm_flags = ir.mm->flags();
    auto original_body = fn->body();
    PatternProgram first;
    auto ca = first.capture(), cb = first.capture();
    GraphPattern unwrap(first);
    auto root = unwrap.anchor(call(first, split, pair(first, concat, ca, cb)));
    GraphView before({fn->body()}, ir.options(fn));
    check_matches(unwrap, before,
                  {
                      {chunks}
    },
                  {{ca, a}, {cb, b}});
    auto found = find_matches(unwrap, before);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(root), plan.tuple({plan.import(found.matches[0].isa_capture(ca)),
                                                        plan.import(found.matches[0].isa_capture(cb))}));
    auto rewritten = build(before, plan);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(fn->body() == original_body);
    auto prospective = rewritten.scope();
    auto outer       = prospective->body()->as<App>();
    CHECK(outer->arg()->as<App>()->arg() == prospective->var(3, 0));
    CHECK(outer->arg()->as<App>()->callee()->as<App>()->arg() == prospective->var(3, 0));
    CHECK(outer->callee()->as<App>()->arg() == prospective->var(3, 1));
    GraphView target({prospective->body()}, ir.options());
    CHECK(operation_flags(target) == std::multiset<flags_t>{add_flags, add_flags});
    rewritten.publish(before);

    auto current = ir.w.externals()[name]->as<Lam>();
    auto body = current->body()->as<App>(), inner = body->arg()->as<App>();
    auto current_add = inner->axm();
    CHECK(current_add->flags() == add_flags);
    auto current_mm = ir.w.externals()[keeper_name]->as<Lam>()->body()->as<App>()->axm();
    REQUIRE(current_mm);
    CHECK(current_mm->flags() == mm_flags);
    PatternProgram second;
    auto input = second.capture();
    GraphPattern twice(second);
    auto repeated = twice.anchor(pair(second, current_add, input, input));
    GraphView after({current->body()}, ir.options(current));
    check_matches(twice, after,
                  {
                      {inner}
    },
                  {{input, current->var(3, 0)}});
    auto next = find_matches(twice, after);
    REQUIRE(next.matches.size() == 1);
    auto next_gid = ir.w.curr_gid();
    RewritePlan multiply(next.matches[0]);
    multiply.replace(next.matches[0].at(repeated),
                     multiply.call(current_mm, {multiply.import(next.matches[0].isa_capture(input)),
                                                multiply.literal(ir.w.type_nat(), 2)}));
    auto final = build(after, multiply);
    CHECK(ir.w.curr_gid() == next_gid);
    auto final_scope = final.scope();
    auto final_body = final_scope->body()->as<App>(), product = final_body->arg()->as<App>();
    CHECK(final_body->axm()->flags() == add_flags);
    CHECK(final_body->callee()->as<App>()->arg() == final_scope->var(3, 1));
    CHECK(product->axm()->flags() == mm_flags);
    CHECK(Lit::isa(product->arg()) == 2);
    CHECK(product->callee()->as<App>()->arg() == final_scope->var(3, 0));
    GraphView final_view({final_scope->body()}, ir.options());
    CHECK(operation_flags(final_view) == std::multiset<flags_t>{add_flags, mm_flags});
    final.publish(after);
    auto published     = ir.w.externals()[name]->as<Lam>();
    auto published_add = published->body()->as<App>()->axm();
    PatternProgram exhausted;
    auto capture = exhausted.capture();
    GraphPattern no_more(exhausted);
    no_more.anchor(pair(exhausted, published_add, capture, capture));
    GraphView last({published->body()}, ir.options(published));
    check_matches(no_more, last, {});
}

// TVM test_rewrite_attention: capture the wrapper chains and replace the complete motif in one checked recipe.
TEST_CASE("DPL extended attention rewrite: exact leaf captures survive wrapper removal and operand routing") {
    ExtendedIR ir;
    auto attention = ir.w.axm(ir.w.pi(ir.nat, ir.w.pi(ir.nat, ir.w.pi(ir.nat, ir.nat))));
    ir.ops.add(attention->flags(), 3, {{0}, {1}, {2}}, true);
    auto prepare       = [&](const Def* x) { return ir.call(ir.k, ir.call(ir.q, x)); };
    auto score         = ir.pair(ir.mm, prepare(ir.x), ir.call(ir.q, prepare(ir.y)));
    auto softmax       = ir.call(ir.v, ir.pair(ir.mm, score, ir.w.lit_nat(2)));
    auto output        = ir.call(ir.q, ir.call(ir.k, ir.pair(ir.mm, softmax, prepare(ir.z))));
    auto wrong_softmax = ir.call(ir.v, ir.pair(ir.mm, score, ir.w.lit_nat(3)));
    auto wrong         = ir.call(ir.q, ir.call(ir.k, ir.pair(ir.mm, wrong_softmax, prepare(ir.z))));
    PatternProgram p;
    auto q = p.capture(), k = p.capture(), v = p.capture();
    auto prep       = [&](Pattern x) { return call(p, ir.k, call(p, ir.q, x)); };
    auto scores     = pair(p, ir.mm, prep(q), call(p, ir.q, prep(k)));
    auto normalized = call(p, ir.v, pair(p, ir.mm, scores, p.literal(ir.nat, 2)));
    auto pattern    = call(p, ir.q, call(p, ir.k, pair(p, ir.mm, normalized, prep(v))));
    GraphPattern graph(p);
    auto root = graph.anchor(pattern);
    GraphView view({output}, ir.options());
    check_matches(graph, view,
                  {
                      {output}
    },
                  {{q, ir.x}, {k, ir.y}, {v, ir.z}});
    GraphView counter({wrong}, ir.options());
    check_matches(graph, counter, {});
    auto found = find_matches(graph, view);
    REQUIRE(found.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan plan(found.matches[0]);
    plan.replace(found.matches[0].at(root), plan.call(attention, {plan.import(found.matches[0].isa_capture(q)),
                                                                  plan.import(found.matches[0].isa_capture(v)),
                                                                  plan.import(found.matches[0].isa_capture(k))}));
    auto rewritten = build(view, plan);
    auto result    = rewritten.roots()[0]->as<App>();
    CHECK(result->axm()->flags() == attention->flags());
    auto second = result->callee()->as<App>(), first = second->callee()->as<App>();
    CHECK(result->arg()->as<Axm>()->flags() == ir.y->flags());
    CHECK(second->arg()->as<Axm>()->flags() == ir.z->flags());
    CHECK(first->arg()->as<Axm>()->flags() == ir.x->flags());
    GraphView target(rewritten.roots(), ir.options());
    CHECK(operation_flags(target) == std::multiset<flags_t>{attention->flags()});
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.roots()[0] == output);
}

// TVM test_rewrite_simple: a nested repeated subpattern flattens two doubling stages, without iteration.
TEST_CASE("DPL extended single Rule: nested double-add binds one leaf and emits the factor-four RHS") {
    ExtendedIR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x = rule->var(), double_x = ir.pair(ir.add, x, x);
    rule->set(ir.pair(ir.add, double_x, double_x), ir.pair(ir.mm, x, ir.w.lit_nat(4)), ir.w.lit_tt());
    auto shared = ir.pair(ir.add, ir.x, ir.x), good = ir.pair(ir.add, shared, shared);
    auto other = ir.pair(ir.add, ir.y, ir.y), wrong = ir.pair(ir.add, shared, other);
    auto wrong_inner = ir.pair(ir.mm, ir.x, ir.x), wrong_op = ir.pair(ir.add, wrong_inner, wrong_inner);
    RulePattern pattern(rule);
    auto target = ir.w.inherit();
    auto gid    = ir.w.curr_gid();
    auto found  = pattern.match_lhs(good);
    REQUIRE(found.status == MatchStatus::Matched);
    REQUIRE(found.bindings.size() == 1);
    CHECK(found.bindings.at(x) == ir.x);
    for (auto candidate : {wrong, wrong_op}) {
        auto rejected = pattern.match_lhs(candidate);
        CHECK(rejected.status == MatchStatus::No_Match);
        CHECK(rejected.bindings.empty());
    }
    auto replacement = pattern.build_replacement(good, *target);
    REQUIRE(replacement.status == MatchStatus::Matched);
    auto result = replacement.value->as<App>();
    CHECK(result->axm()->flags() == ir.mm->flags());
    CHECK(result->arg() == target->lit_nat(4));
    CHECK(result->callee()->as<App>()->arg()->as<Axm>()->flags() == ir.x->flags());
    CHECK(ir.w.curr_gid() == gid);
}

// TVM test_combine_transposed_matmul_twice: current multi-root Rule semantics can keep an identity RHS slot.
TEST_CASE("DPL extended multi-root Rule: unchanged slot and projected weight capture remain independent") {
    ExtendedIR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat, ir.nat})));
    auto x = rule->var(3, 0), weight = rule->var(3, 1);
    auto first = ir.pair(ir.mm, x, ir.call(ir.q, weight)), second = ir.call(ir.k, x);
    rule->set(ir.w.tuple({first, second}), ir.w.tuple({first, ir.call(ir.v, weight)}), ir.w.lit_tt());
    auto a = ir.w.lit_nat(11), b = ir.w.lit_nat(12);
    auto xa = ir.pair(ir.mm, ir.x, ir.call(ir.q, a)), xb = ir.call(ir.k, ir.x);
    auto ya = ir.pair(ir.mm, ir.y, ir.call(ir.q, b)), yb = ir.call(ir.k, ir.y);
    auto fake = ir.call(ir.k, ir.z);
    GraphView view({xa, xb, ya, yb, fake}, ir.options());
    GraphRulePattern pattern(rule);
    auto gid = ir.w.curr_gid();
    check_search(
        [&](auto options) {
            return pattern.find(view, options);
    },
        {{xa, xb}, {ya, yb}},
        [&](const GraphMatch& m) {
            auto planned = pattern.plan(m, view);
            REQUIRE(planned.status == MatchStatus::Matched);
            REQUIRE(planned.plan);
            auto rewritten = build(view, *planned.plan);
            CHECK(rewritten.replacements() == 1);
            auto slot    = m.anchors()[0] == xa ? 1 : 3;
            auto changed = rewritten.roots()[slot]->as<App>();
            CHECK(changed->axm()->flags() == ir.v->flags());
            CHECK(Lit::isa(changed->arg()) == (slot == 1 ? 11 : 12));
            CHECK(rewritten.roots()[slot - 1]->as<App>()->axm()->flags() == ir.mm->flags());
            CHECK(rewritten.roots()[slot - 1]->as<App>()->arg()->as<App>()->axm()->flags() == ir.q->flags());
            CHECK(rewritten.roots()[4]->as<App>()->axm()->flags() == ir.k->flags());
        });
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}

// TVM test_combine_transposed_matmul_twice: an aggregate RHS operation supplies both scalar result recipes.
TEST_CASE("DPL extended aggregate Rule RHS: two transposed matmuls share one typed result operation") {
    ExtendedIR ir;
    auto concat = ir.unary(ir.w.arr(2, ir.nat), ir.nat), split = ir.unary(ir.nat, ir.w.arr(2, ir.nat));
    ir.ops.add(concat->flags(), 1,
               {
                   {0, 1}
    },
               true);
    ir.ops.add(split->flags(), 1, {{0}}, true);
    auto rule = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat, ir.nat, ir.nat})));
    auto x = rule->var(4, 0), a = rule->var(4, 1), b = rule->var(4, 2);
    auto lhs0 = ir.pair(ir.mm, x, ir.call(ir.q, a)), lhs1 = ir.pair(ir.mm, x, ir.call(ir.q, b));
    auto joined = ir.call(concat, ir.w.tuple({a, b}));
    auto rhs    = ir.call(split, ir.pair(ir.mm, x, ir.call(ir.q, joined)));
    rule->set(ir.w.tuple({lhs0, lhs1}), rhs, ir.w.lit_tt());
    auto w0 = ir.w.lit_nat(11), w1 = ir.w.lit_nat(12);
    auto m0 = ir.pair(ir.mm, ir.x, ir.call(ir.q, w0)), m1 = ir.pair(ir.mm, ir.x, ir.call(ir.q, w1));
    auto fake = ir.pair(ir.mm, ir.y, ir.call(ir.q, ir.w.lit_nat(13)));
    GraphRulePattern pattern(rule);
    GraphView view({m0, m1, fake}, ir.options());
    auto gid = ir.w.curr_gid();
    check_search(
        [&](auto options) {
            return pattern.find(view, options);
    },
        {{m0, m1}, {m1, m0}},
        [&](const GraphMatch& m) {
            auto plan = pattern.plan(m, view);
            REQUIRE(plan.status == MatchStatus::Matched);
            REQUIRE(plan.plan);
            auto rewritten = build(view, *plan.plan);
            auto e0 = rewritten.roots()[0]->as<Extract>(), e1 = rewritten.roots()[1]->as<Extract>();
            CHECK(e0->tuple() == e1->tuple());
            CHECK(Lit::isa(e0->index()) == (m.anchors()[0] == m0 ? 0 : 1));
            CHECK(Lit::isa(e1->index()) == (m.anchors()[0] == m0 ? 1 : 0));
            auto product   = e0->tuple()->as<App>()->arg()->as<App>();
            auto transpose = product->arg()->as<App>();
            auto weights   = transpose->arg()->as<App>()->arg()->as<Tuple>();
            CHECK(Lit::isa(weights->op(0)) == (m.anchors()[0] == m0 ? 11 : 12));
            CHECK(Lit::isa(weights->op(1)) == (m.anchors()[0] == m0 ? 12 : 11));
            CHECK(e0->tuple()->as<App>()->axm()->flags() == split->flags());
            CHECK(product->axm()->flags() == ir.mm->flags());
            CHECK(transpose->axm()->flags() == ir.q->flags());
            GraphView target(rewritten.roots(), ir.options());
            CHECK(operation_flags(target)
                  == std::multiset<flags_t>{concat->flags(), split->flags(), ir.mm->flags(), ir.q->flags(),
                                            ir.mm->flags(), ir.q->flags()});
            CHECK(rewritten.replacements() == 2);
        });
    CHECK(ir.w.curr_gid() == gid);
    CHECK(view.is_current());
}
