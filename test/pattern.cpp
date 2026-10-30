#include <stdexcept>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph.h>
#include <mim/pattern/match.h>
#include <mim/pattern/rule.h>
#include <mim/phase/rule_rewrite.h>

using namespace mim;
using namespace mim::pattern;

namespace {

struct IR {
    Driver driver;
    World& w          = driver.world();
    const Def* nat    = w.type_nat();
    const Def* a      = w.lit_nat(1);
    const Def* b      = w.lit_nat(2);
    const Def* unary  = w.axm(w.pi(nat, nat));
    const Def* binary = w.axm(w.pi(nat, w.pi(nat, nat)));

    const Def* call(const Def* x) { return w.app(unary, x); }
    const Def* pair(const Def* x, const Def* y) { return w.app(w.app(binary, x), y); }
};

class PatternCleanup : public Cleanup {
public:
    explicit PatternCleanup(World& world)
        : Cleanup(world, "pattern_cleanup") {}
    std::unique_ptr<Phase> recreate() override { return std::make_unique<PatternCleanup>(old_world()); }
};

Pattern app(PatternProgram& p, Pattern callee, Pattern arg) { return p.node(Node::App, {callee, arg}, 0); }

Pattern pair(PatternProgram& p, const Def* callee, Pattern x, Pattern y) {
    return app(p, app(p, p.exact(callee), x), y);
}

const Def* normalize_is_one(const Def*, const Def*, const Def* arg) {
    if (auto value = Lit::isa(arg)) return arg->world().lit_bool(*value == 1);
    return nullptr;
}

} // namespace

TEST_CASE("Pattern: captures preserve shared DAG identity") {
    IR ir;
    PatternProgram p;
    auto x       = p.capture();
    auto pattern = pair(p, ir.binary, x, x);
    auto shared  = ir.call(ir.a);
    auto same    = match(p, pattern, ir.pair(shared, shared));
    REQUIRE(same.status == MatchStatus::Matched);
    REQUIRE(same.result);
    CHECK(same.result->isa_capture(x) == shared);
    auto different = match(p, pattern, ir.pair(shared, ir.call(ir.b)));
    CHECK(different.status == MatchStatus::No_Match);
    CHECK_FALSE(different.result);

    auto y           = p.capture();
    auto independent = pair(p, ir.binary, x, y);
    CHECK(match(p, independent, ir.pair(ir.a, ir.a)).status == MatchStatus::Matched);
}

TEST_CASE("Pattern: typed literals and semantic flags") {
    IR ir;
    PatternProgram p;
    CHECK(match(p, p.literal(ir.nat, 1), ir.a).status == MatchStatus::Matched);
    CHECK(match(p, p.literal(ir.nat, 2), ir.a).status == MatchStatus::No_Match);
    CHECK(match(p, p.literal(ir.w.type_bool(), 1), ir.a).status == MatchStatus::No_Match);
    CHECK(match(p, p.node(Node::Axm, {}, ir.unary->flags()), ir.unary).status == MatchStatus::Matched);
    CHECK(match(p, p.node(Node::Axm, {}, ir.unary->flags()), ir.binary).status == MatchStatus::No_Match);
    CHECK(match(p, p.node(Node::App, {p.any()}), ir.call(ir.a)).status == MatchStatus::No_Match);
    CHECK(match(p, p.exact(ir.a), ir.b).status == MatchStatus::No_Match);
}

TEST_CASE("Pattern: failed alternatives roll back captures and constraints") {
    IR ir;
    PatternProgram p;
    auto x       = p.capture();
    auto left    = p.where(pair(p, ir.binary, x, p.exact(ir.b)), {x}, [](auto) { return CheckResult::False; });
    auto right   = pair(p, ir.binary, p.exact(ir.a), x);
    auto outcome = match(p, p.either(left, right), ir.pair(ir.a, ir.b));
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(x) == ir.b);
}

TEST_CASE("Pattern: alternatives backtrack after later operands fail") {
    IR ir;
    PatternProgram p;
    auto x = p.capture();
    // The first alternative binds x to the inner App. Only the second can also match the outer argument.
    auto callee  = p.either(x, app(p, p.exact(ir.binary), x));
    auto pattern = app(p, callee, x);
    auto outcome = match(p, pattern, ir.pair(ir.a, ir.a));
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(x) == ir.a);
}

TEST_CASE("Pattern: deferred predicates see all structural captures") {
    IR ir;
    PatternProgram p;
    auto x           = p.capture();
    auto y           = p.capture();
    auto constrained = p.where(
        x, {x, y}, [](auto values) { return values[0] == values[1] ? CheckResult::True : CheckResult::False; });
    auto pattern = pair(p, ir.binary, constrained, y);
    CHECK(match(p, pattern, ir.pair(ir.a, ir.a)).status == MatchStatus::Matched);
    CHECK(match(p, pattern, ir.pair(ir.a, ir.b)).status == MatchStatus::No_Match);
    auto missing = match(p, p.where(x, {y}, [](auto) { return CheckResult::True; }), ir.a);
    CHECK(missing.status == MatchStatus::Unknown);
    CHECK_FALSE(missing.result);
}

TEST_CASE("Pattern: downstream predicates can select another alternative") {
    IR ir;
    PatternProgram p;
    auto x      = p.capture();
    auto choice = p.either(pair(p, ir.binary, x, p.any()), pair(p, ir.binary, p.any(), x));
    auto pattern
        = p.where(choice, {x}, [&](auto values) { return values[0] == ir.b ? CheckResult::True : CheckResult::False; });
    auto outcome = match(p, pattern, ir.pair(ir.a, ir.b));
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(x) == ir.b);
}

TEST_CASE("Pattern: deferred type matching shares captures") {
    IR ir;
    PatternProgram p;
    auto t       = p.capture();
    auto value   = p.capture();
    auto pattern = p.where(p.all(p.type(t), value), {t, value}, [](auto args) {
        return args[0] == args[1]->type() ? CheckResult::True : CheckResult::False;
    });
    auto outcome = match(p, pattern, ir.a);
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(t) == ir.nat);
    CHECK(outcome.result->isa_capture(value) == ir.a);
    CHECK(match(p, p.type(p.exact(ir.w.type_bool())), ir.a).status == MatchStatus::No_Match);

    // A type check may itself enqueue further type checks.
    CHECK(match(p, p.type(p.type(p.any())), ir.a).status == MatchStatus::Matched);
}

TEST_CASE("Pattern: conjunction checks capture consistency") {
    IR ir;
    PatternProgram p;
    auto x   = p.capture();
    auto lhs = pair(p, ir.binary, x, p.any());
    auto rhs = pair(p, ir.binary, p.any(), x);
    CHECK(match(p, p.all(lhs, rhs), ir.pair(ir.a, ir.a)).status == MatchStatus::Matched);
    CHECK(match(p, p.all(lhs, rhs), ir.pair(ir.a, ir.b)).status == MatchStatus::No_Match);
}

TEST_CASE("Pattern: negation reads final captures but never exports bindings") {
    IR ir;
    PatternProgram p;
    auto x       = p.capture();
    auto not_x   = p.negate(x);
    auto pattern = pair(p, ir.binary, not_x, x);
    CHECK(match(p, pattern, ir.pair(ir.a, ir.a)).status == MatchStatus::No_Match);
    auto outcome = match(p, pattern, ir.pair(ir.a, ir.b));
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(x) == ir.b);

    auto local = p.capture();
    auto probe = p.all(local, p.exact(ir.b));
    outcome    = match(p, p.negate(probe), ir.a);
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(local) == nullptr);
    CHECK(match(p, p.negate(p.negate(p.exact(ir.a))), ir.a).status == MatchStatus::Matched);
}

TEST_CASE("Pattern: unknown and budget exhaustion survive negation") {
    IR ir;
    PatternProgram p;
    auto unknown = p.where(p.any(), {}, [](auto) { return CheckResult::Unknown; });
    CHECK(match(p, p.negate(unknown), ir.a).status == MatchStatus::Unknown);
    CHECK(match(p, p.either(unknown, p.exact(ir.a)), ir.a).status == MatchStatus::Matched);
    CHECK(match(p, p.either(unknown, p.exact(ir.b)), ir.a).status == MatchStatus::Unknown);
    CHECK(match(p, p.all(unknown, p.exact(ir.b)), ir.a).status == MatchStatus::No_Match);
    auto exhausted = match(p, p.negate(p.any()), ir.a, {.max_steps = 2});
    CHECK(exhausted.status == MatchStatus::Budget_Exceeded);
    CHECK_FALSE(exhausted.result);
    CHECK(exhausted.steps <= 2);
    CHECK(match(p, p.any(), ir.a, {.max_steps = 0}).status == MatchStatus::Budget_Exceeded);
}

TEST_CASE("Pattern: depth limits bound nested choices and deferred probes") {
    IR ir;
    PatternProgram p;
    auto nested = p.any();
    for (size_t i = 0; i != 100; ++i)
        nested = p.negate(nested);
    CHECK(match(p, nested, ir.a, {.max_depth = 16}).status == MatchStatus::Budget_Exceeded);
    CHECK(match(p, nested, ir.a).status == MatchStatus::Matched);
    auto types = p.any();
    for (size_t i = 0; i != 100; ++i)
        types = p.all(types, p.type(p.any()));
    CHECK(match(p, types, ir.a, {.max_depth = 16}).status == MatchStatus::Budget_Exceeded);
}

TEST_CASE("Pattern: mutable binders stay opaque and matching is read-only") {
    IR ir;
    auto lam = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    lam->set(false, ir.w.app(lam, lam->var()));
    PatternProgram p;
    auto gid = ir.w.curr_gid();
    CHECK(match(p, p.exact(lam), lam).status == MatchStatus::Matched);
    CHECK(match(p, p.node(Node::Lam, {p.any(), p.any()}), lam).status == MatchStatus::Unknown);
    GraphView graph({lam});
    CHECK(graph.nodes().size() == 1);
    CHECK_FALSE(graph.is_contained(lam->body()));
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Pattern: programs reject foreign handles and invalid constraints") {
    IR ir;
    PatternProgram p, other;
    auto foreign = other.capture();
    CHECK_THROWS_AS(p.capture(foreign), std::invalid_argument);
    CHECK_THROWS_AS(match(p, foreign, ir.a), std::invalid_argument);
    CHECK_THROWS_AS(p.exact(nullptr), std::invalid_argument);
    CHECK_THROWS_AS(p.literal(nullptr, 0), std::invalid_argument);
    CHECK_THROWS_AS(p.where(p.any(), {p.any()}, [](auto) { return CheckResult::True; }), std::invalid_argument);
    CHECK_THROWS_AS(p.where(p.any(), {}, {}), std::invalid_argument);
    auto outcome = match(p, p.capture(), ir.a);
    REQUIRE(outcome.result);
    CHECK_THROWS_AS(outcome.result->isa_capture(foreign), std::invalid_argument);
    CHECK(match(p, p.any(), nullptr).status == MatchStatus::No_Match);
}

TEST_CASE("Pattern: sequence variable types are not materialized by queries") {
    IR ir;
    auto shape = ir.w.axm(ir.nat);
    auto arr   = ir.w.mut_arr(ir.w.type());
    arr->set_shape(shape);
    auto var = arr->var();
    auto gid = ir.w.curr_gid();
    PatternProgram p;
    CHECK(match(p, p.type(p.any()), var).status == MatchStatus::Unknown);
    GraphView graph({var});
    CHECK(graph.nodes().size() == 1);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Pattern graph: closed lambdas are scope boundaries too") {
    IR ir;
    auto body = ir.call(ir.a);
    auto lam  = ir.w.lam(ir.w.pi(ir.nat, ir.nat), false, body);
    REQUIRE_FALSE(lam->is_mutable());
    GraphView graph({lam});
    CHECK(graph.nodes().size() == 1);
    CHECK_FALSE(graph.is_contained(body));
}

TEST_CASE("Pattern graph: roots and shared nodes are visited once") {
    IR ir;
    auto producer = ir.call(ir.a);
    auto root     = ir.pair(producer, producer);
    auto gid      = ir.w.curr_gid();
    GraphView graph({root, root}, UseCoverage::Closed);
    CHECK(graph.roots().size() == 1);
    CHECK(std::ranges::count(graph.nodes(), producer) == 1);
    CHECK(graph.use_count(producer) == 2);
    CHECK(graph.consumer_count(producer) == 2); // inner and outer App are distinct raw owners
    CHECK(graph.only_used_by(producer, root) == CheckResult::False);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Pattern graph: types and callees are separate from operand uses") {
    IR ir;
    auto root = ir.call(ir.a);
    GraphView graph({root}, UseCoverage::Closed);
    CHECK(graph.use_count(ir.a) == 1);
    CHECK(graph.consumer_count(ir.a) == 1);
    CHECK(graph.only_used_by(ir.a, root) == CheckResult::True);
    CHECK(graph.use_count(ir.unary) == 0);
    CHECK_FALSE(graph.is_used_by(ir.unary, root));
    CHECK(graph.uses(ir.unary)[0].kind == EdgeKind::Callee);
    CHECK(graph.use_count(ir.nat) == 0);
    CHECK_FALSE(graph.is_contained(ir.nat));
    REQUIRE_FALSE(graph.uses(ir.nat).empty());
    CHECK(std::ranges::all_of(graph.uses(ir.nat), [](auto e) { return e.kind == EdgeKind::Type; }));
}

TEST_CASE("Pattern graph: repeated operands and aggregate wrappers keep raw identity") {
    IR ir;
    auto type = ir.w.mut_sigma(ir.w.type(), 2);
    type->set(Defs{ir.nat, ir.nat});
    auto root = ir.w.tuple(type, {ir.a, ir.a});
    REQUIRE(root->node() == Node::Tuple);
    GraphView graph({root}, UseCoverage::Closed);
    CHECK(graph.use_count(ir.a) == 2);
    CHECK(graph.consumer_count(ir.a) == 1);
    CHECK(graph.only_used_by(ir.a, root) == CheckResult::True);
    CHECK(graph.uses(ir.a)[0].index == 0);
    CHECK(graph.uses(ir.a)[1].index == 1);

    // Uniform tuples normalize to Pack. The raw index counts its body edge, not expanded elements.
    auto pack = ir.w.tuple({ir.a, ir.a, ir.a});
    REQUIRE(pack->node() == Node::Pack);
    GraphView packed({pack}, UseCoverage::Closed);
    CHECK(packed.use_count(ir.a) == 1);
    CHECK(packed.only_used_by(ir.a, pack) == CheckResult::True);
}

TEST_CASE("Pattern: finite DAG choices agree with exhaustive capture candidates") {
    IR ir;
    auto a = ir.call(ir.a);
    auto b = ir.call(ir.b);
    std::array values{ir.a, ir.b, a, b};
    PatternProgram p;
    auto x      = p.capture();
    auto choose = p.either(pair(p, ir.binary, x, p.any()), pair(p, ir.binary, p.any(), x));
    for (auto lhs : values)
        for (auto rhs : values)
            for (auto wanted : values) {
                auto pattern = p.where(choose, {x}, [wanted](auto args) {
                    return args[0] == wanted ? CheckResult::True : CheckResult::False;
                });
                auto outcome = match(p, pattern, ir.pair(lhs, rhs));
                CHECK((outcome.status == MatchStatus::Matched) == (wanted == lhs || wanted == rhs));
                if (outcome.result) CHECK(outcome.result->isa_capture(x) == wanted);
            }
}

TEST_CASE("Pattern graph: open regions cannot prove exclusive usage") {
    IR ir;
    auto root = ir.call(ir.a);
    GraphView open({root});
    CHECK(open.only_used_by(ir.a, root) == CheckResult::Unknown);
    GraphView returned({root, ir.a}, UseCoverage::Closed);
    CHECK(returned.only_used_by(ir.a, root) == CheckResult::False);
    CHECK(returned.only_used_by(ir.b, root) == CheckResult::Unknown);
    auto other = ir.pair(ir.a, ir.b);
    GraphView shared({root, other}, UseCoverage::Closed);
    CHECK(shared.only_used_by(ir.a, root) == CheckResult::False);
}

TEST_CASE("Pattern graph: graph constraints participate in alternative selection") {
    IR ir;
    auto producer = ir.call(ir.a);
    auto root     = ir.pair(producer, ir.b);
    GraphView graph({root}, UseCoverage::Closed);
    PatternProgram p;
    auto input   = p.capture();
    auto owner   = p.capture();
    auto pattern = p.where(p.all(owner, app(p, p.any(), input)), {input, owner},
                           [&](auto args) { return graph.only_used_by(args[0], args[1]); });
    auto outcome = match(p, pattern, root);
    REQUIRE(outcome.status == MatchStatus::Matched);
    CHECK(outcome.result->isa_capture(input) == ir.b);
    CHECK(outcome.result->isa_capture(owner) == root);
}

TEST_CASE("Pattern graph: empty views and cross-World roots") {
    IR ir;
    GraphView empty({});
    CHECK(empty.nodes().empty());
    CHECK(empty.uses(ir.a).empty());
    CHECK(empty.uses(nullptr).empty());
    CHECK_FALSE(empty.is_contained(nullptr));
    CHECK(empty.only_used_by(ir.a, ir.b) == CheckResult::Unknown);
    CHECK_THROWS_AS(GraphView({nullptr}), std::invalid_argument);
    Driver other;
    CHECK_THROWS_AS(GraphView({ir.a, other.world().lit_nat(1)}), std::invalid_argument);
}

TEST_CASE("Pattern Rule: existing metavariables compile to shared captures") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(ir.pair(x, x), x, ir.w.lit_tt());
    RulePattern pattern(rule);
    auto result = pattern.match_lhs(ir.pair(ir.a, ir.a));
    REQUIRE(result.status == MatchStatus::Matched);
    CHECK(result.bindings.at(x) == ir.a);
    CHECK(pattern.match_lhs(ir.pair(ir.a, ir.b)).status == MatchStatus::No_Match);
    CHECK(pattern.match_lhs(rule->lhs()).status == MatchStatus::No_Match);
    CHECK(pattern.match_lhs(nullptr).status == MatchStatus::No_Match);
    // Constructing candidates may add nodes; compiling/matching an existing candidate does not.
    auto candidate = ir.pair(ir.b, ir.b);
    auto gid       = ir.w.curr_gid();
    RulePattern again(rule);
    CHECK(again.match_lhs(candidate).status == MatchStatus::Matched);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Pattern Rule: projections bind dependent types after structure") {
    IR ir;
    auto telescope = ir.w.mut_sigma(ir.w.type<1>(), 2);
    telescope->set(0, ir.w.type());
    telescope->set(1, telescope->var(2, 0));
    auto rule = ir.w.mut_rule(ir.w.reform(telescope));
    auto t    = rule->var(2, 0);
    auto x    = rule->var(2, 1);
    rule->set(x, x, ir.w.lit_tt());
    RulePattern pattern(rule);
    auto result = pattern.match_lhs(ir.a);
    REQUIRE(result.status == MatchStatus::Matched);
    CHECK(result.bindings.at(t) == ir.nat);
    CHECK(result.bindings.at(x) == ir.a);
    result = pattern.match_lhs(ir.w.lit_tt());
    REQUIRE(result.status == MatchStatus::Matched);
    CHECK(result.bindings.at(t) == ir.w.type_bool());
}

TEST_CASE("Pattern Rule: whole telescopes and projections must agree") {
    IR ir;
    auto domain = ir.w.sigma({ir.nat, ir.nat});
    auto rule   = ir.w.mut_rule(ir.w.reform(domain));
    auto whole  = rule->var();
    auto x      = rule->var(2, 0);
    auto lhs    = ir.w.tuple({whole, x});
    rule->set(lhs, lhs, ir.w.lit_tt());
    RulePattern pattern(rule);
    auto tuple        = ir.w.tuple({ir.a, ir.b});
    auto consistent   = ir.w.tuple({tuple, ir.a});
    auto inconsistent = ir.w.tuple({tuple, ir.b});
    auto uniform      = ir.w.tuple({ir.a, ir.a});
    CHECK(pattern.match_lhs(consistent).status == MatchStatus::Matched);
    auto rejected = pattern.match_lhs(inconsistent);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK(rejected.bindings.empty());
    CHECK(pattern.match_lhs(ir.w.tuple({uniform, ir.a})).status == MatchStatus::Matched);
    auto opaque = ir.w.axm(domain);
    CHECK(pattern.match_lhs(ir.w.tuple({opaque, ir.a})).status == MatchStatus::Unknown);
}

TEST_CASE("Pattern Rule: only the owning Rule's variables are metavariables") {
    IR ir;
    auto rule    = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto lam     = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    auto foreign = lam->var();
    lam->set(false, foreign);
    rule->set(foreign, ir.a, ir.w.lit_tt());
    RulePattern pattern(rule);
    CHECK(pattern.match_lhs(ir.a).status == MatchStatus::No_Match);
    auto result = pattern.match_lhs(foreign);
    CHECK(result.status == MatchStatus::Matched);
    CHECK(result.bindings.empty());
}

TEST_CASE("Pattern Rule: constant rules and guards have distinct matching semantics") {
    IR ir;
    auto constant = ir.w.mut_rule(ir.w.reform(ir.w.sigma()));
    constant->set(ir.a, ir.b, ir.w.lit_ff());
    RulePattern pattern(constant);
    // Like Rule::its_a_match, this API matches the LHS only, without accepting the guard.
    auto result = pattern.match_lhs(ir.a);
    CHECK(result.status == MatchStatus::Matched);
    CHECK(result.bindings.empty());
    CHECK(pattern.match_lhs(ir.b).status == MatchStatus::No_Match);
}

TEST_CASE("Pattern Rule: budget failures never leak partial bindings") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    rule->set(ir.pair(rule->var(), rule->var()), rule->var(), ir.w.lit_tt());
    RulePattern pattern(rule);
    auto result = pattern.match_lhs(ir.pair(ir.a, ir.a), {.max_steps = 3});
    CHECK(result.status == MatchStatus::Budget_Exceeded);
    CHECK(result.bindings.empty());
    auto incomplete = ir.w.mut_rule(ir.w.reform(ir.nat));
    CHECK_THROWS_AS(RulePattern{incomplete}, std::invalid_argument);
    CHECK_THROWS_AS(RulePattern{nullptr}, std::invalid_argument);
    Driver other;
    CHECK_THROWS_AS(pattern.match_lhs(other.world().lit_nat(1)), std::invalid_argument);
}

TEST_CASE("Pattern Rule: replacements are constructed in the target World") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(ir.pair(x, x), ir.call(x), ir.w.lit_tt());
    auto candidate = ir.pair(ir.a, ir.a);
    auto target    = ir.w.inherit();
    auto gid       = ir.w.curr_gid();
    RulePattern pattern(rule);
    auto result = pattern.build_replacement(candidate, *target);
    REQUIRE(result.status == MatchStatus::Matched);
    REQUIRE(result.value);
    CHECK(&result.value->world() == target.get());
    CHECK(result.value->as<App>()->arg() == target->lit_nat(1));
    CHECK(ir.w.curr_gid() == gid);
    CHECK_THROWS_AS(pattern.build_replacement(candidate, ir.w), std::invalid_argument);
}

TEST_CASE("Pattern Rule: no-capture replacements and guard rejection") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.w.sigma()));
    rule->set(ir.a, ir.b, ir.w.lit_tt());
    auto target = ir.w.inherit();
    RulePattern pattern(rule);
    auto result = pattern.build_replacement(ir.a, *target);
    REQUIRE(result.status == MatchStatus::Matched);
    CHECK(result.value == target->lit_nat(2));

    auto rejected = ir.w.mut_rule(ir.w.reform(ir.nat));
    rejected->set(rejected->var(), ir.b, ir.w.lit_ff());
    result = RulePattern(rejected).build_replacement(ir.a, *target);
    CHECK(result.status == MatchStatus::No_Match);
    CHECK(result.value == nullptr);
    CHECK(result.reason == "guard is false");

    auto unknown = ir.w.mut_rule(ir.w.reform(ir.nat));
    unknown->set(unknown->var(), ir.b, ir.w.axm(ir.w.type_bool()));
    result = RulePattern(unknown).build_replacement(ir.a, *target);
    CHECK(result.status == MatchStatus::Unknown);
    CHECK(result.value == nullptr);
}

TEST_CASE("Pattern Rule: guard evaluation uses captures in the target World") {
    IR ir;
    auto gate = ir.w.axm(normalize_is_one, 1, Axm::Trip_End, ir.w.pi(ir.nat, ir.w.type_bool()));
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(x, ir.b, ir.w.app(gate, x));
    auto target = ir.w.inherit();
    auto gid    = ir.w.curr_gid();
    RulePattern pattern(rule);
    auto accepted = pattern.build_replacement(ir.a, *target);
    REQUIRE(accepted.status == MatchStatus::Matched);
    CHECK(accepted.value == target->lit_nat(2));
    auto rejected = pattern.build_replacement(ir.b, *target);
    CHECK(rejected.status == MatchStatus::No_Match);
    CHECK(rejected.value == nullptr);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Pattern Rule: unbound captures are rejected without Hole construction") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.w.sigma({ir.nat, ir.nat})));
    auto x    = rule->var(2, 0);
    auto y    = rule->var(2, 1);
    rule->set(x, y, ir.w.lit_tt());
    RulePattern pattern(rule);
    auto target = ir.w.inherit();
    auto gid    = target->curr_gid();
    auto result = pattern.build_replacement(ir.a, *target);
    CHECK(result.status == MatchStatus::Unknown);
    CHECK(result.value == nullptr);
    CHECK(target->curr_gid() == gid);
}

TEST_CASE("Pattern Rule: RHS type changes are rejected") {
    IR ir;
    // The low-level immutable Rule constructor can receive an unchecked template.
    auto rule   = ir.w.rule(ir.w.reform(ir.w.sigma()), ir.a, ir.w.lit_tt(), ir.w.lit_tt());
    auto target = ir.w.inherit();
    auto result = RulePattern(rule).build_replacement(ir.a, *target);
    CHECK(result.status == MatchStatus::Unknown);
    CHECK(result.value == nullptr);
    CHECK(result.reason == "RHS does not preserve the matched type");
}

TEST_CASE("Pattern Rule: open replacements require an explicit binder environment") {
    IR ir;
    auto lam = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    auto var = lam->var();
    lam->set(false, var);
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    rule->set(rule->var(), rule->var(), ir.w.lit_tt());
    auto target = ir.w.inherit();
    auto gid    = target->curr_gid();
    auto result = RulePattern(rule).build_replacement(var, *target);
    CHECK(result.status == MatchStatus::Unknown);
    CHECK(result.value == nullptr);
    CHECK(result.reason == "open candidate requires a binder mapping");
    CHECK(target->curr_gid() == gid);

    auto new_lam = target->mut_lam(target->pi(target->type_nat(), target->type_nat()));
    Rewriter context(*target);
    context.map(lam, new_lam);
    result = RulePattern(rule).build_replacement(var, context);
    REQUIRE(result.status == MatchStatus::Matched);
    CHECK(result.value == new_lam->var());
    CHECK(result.value->has_free_var(new_lam->var()->as<Var>()));
    new_lam->set(false, result.value);
}

TEST_CASE("Pattern phase: rules rewrite open function bodies with the correct binder") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(ir.call(ir.call(x)), ir.call(x), ir.w.lit_tt());
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("pattern_f");
    f->set(false, ir.call(ir.call(f->var())))->externalize();
    auto name = f->sym();
    RuleRewrite phase(ir.w, {rule});
    phase.run();
    auto rebuilt = ir.w.externals()[name]->as<Lam>();
    REQUIRE(rebuilt->body()->isa<App>());
    CHECK(rebuilt->body()->as<App>()->arg() == rebuilt->var());
    CHECK(phase.todo());
    CHECK(rebuilt->body()->has_free_var(rebuilt->var()->as<Var>()));
}

TEST_CASE("Pattern phase: rule templates survive cleanup and fixed-point recreation") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(ir.call(ir.call(x)), ir.call(x), ir.w.lit_tt());
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("pattern_fp");
    f->set(false, ir.call(ir.call(ir.call(f->var()))))->externalize();
    auto name = f->sym();
    Phases phases;
    phases.push_back(std::make_unique<PatternCleanup>(ir.w));
    phases.push_back(std::make_unique<RuleRewrite>(ir.w, std::vector<const Rule*>{rule}));
    phases.push_back(std::make_unique<PatternCleanup>(ir.w));
    PhaseMan pipeline(ir.w, "pattern_pipeline");
    pipeline.apply(true, std::move(phases));
    pipeline.run();
    auto rebuilt = ir.w.externals()[name]->as<Lam>();
    CHECK(rebuilt->body()->as<App>()->arg() == rebuilt->var());
}

TEST_CASE("Pattern phase: identity rules do not request another iteration") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    auto x    = rule->var();
    rule->set(ir.call(x), ir.call(x), ir.w.lit_tt());
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("pattern_identity");
    f->set(false, ir.call(f->var()))->externalize();
    auto name = f->sym();
    RuleRewrite phase(ir.w, {rule});
    phase.run();
    CHECK_FALSE(phase.todo());
    auto rebuilt = ir.w.externals()[name]->as<Lam>();
    CHECK(rebuilt->body()->as<App>()->arg() == rebuilt->var());
}

TEST_CASE("Pattern Rule: rebuilt Worlds invalidate compiled Rule handles before accessing old Defs") {
    IR ir;
    auto rule = ir.w.mut_rule(ir.w.reform(ir.nat));
    rule->set(rule->var(), rule->var(), ir.w.lit_tt());
    RulePattern pattern(rule);
    {
        Cleanup cleanup(ir.w, "invalidate_compiled_rules");
        cleanup.run();
    }
    auto candidate = ir.w.lit_nat(1);
    auto target    = ir.w.inherit();
    CHECK_THROWS_AS(pattern.match_lhs(candidate), std::invalid_argument);
    CHECK_THROWS_AS(pattern.build_replacement(candidate, *target), std::invalid_argument);
}
