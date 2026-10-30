#include <array>
#include <set>
#include <stdexcept>

#include <doctest/doctest.h>

#include <mim/driver.h>
#include <mim/phase.h>
#include <mim/world.h>

#include <mim/pattern/rewrite.h>

using namespace mim;
using namespace mim::pattern;

namespace {

struct GraphIR {
    Driver driver;
    World& w            = driver.world();
    const Def* nat      = w.type_nat();
    const Def* x        = w.axm(nat);
    const Def* y        = w.axm(nat);
    const Axm* q        = w.axm(w.pi(nat, nat));
    const Axm* k        = w.axm(w.pi(nat, nat));
    const Axm* v        = w.axm(w.pi(nat, nat));
    const Axm* sum      = w.axm(w.pi(nat, w.pi(nat, nat)));
    const Axm* combined = w.axm(w.pi(nat, w.arr(3, nat)));
    OpRegistry ops;

    GraphIR() {
        for (auto op : {q, k, v, combined})
            ops.add(op->flags(), 1, {{0}}, true);
        ops.add(sum->flags(), 2, {{0}, {1}}, true);
    }
    const Def* call(const Axm* op, const Def* input) { return w.app(op, input); }
    const Def* add(const Def* a, const Def* b) { return w.app(w.app(sum, a), b); }
    GraphOptions options(Lam* scope = nullptr) {
        return {.coverage = UseCoverage::Closed, .operators = &ops, .scope = scope};
    }
};

Pattern app(PatternProgram& p, Pattern callee, Pattern arg) { return p.node(Node::App, {callee, arg}); }
Pattern call(PatternProgram& p, const Def* callee, Pattern arg) { return app(p, p.exact(callee), arg); }
Pattern pair(PatternProgram& p, const Def* callee, Pattern a, Pattern b) { return app(p, call(p, callee, a), b); }

struct QKV {
    PatternProgram p;
    Pattern x = p.capture();
    GraphPattern graph{p};
    Anchor q, k, v;
    explicit QKV(GraphIR& ir)
        : q(graph.anchor(call(p, ir.q, x)))
        , k(graph.anchor(call(p, ir.k, x)))
        , v(graph.anchor(call(p, ir.v, x))) {}
};

void combine(RewritePlan& plan, const GraphMatch& match, QKV& pattern, GraphIR& ir) {
    auto merged = plan.call(ir.combined, {plan.import(match.isa_capture(pattern.x))});
    plan.replace(match.at(pattern.q), plan.extract(merged, 0));
    plan.replace(match.at(pattern.k), plan.extract(merged, 1));
    plan.replace(match.at(pattern.v), plan.extract(merged, 2));
}

GraphRewrite build(const GraphView& graph, const RewritePlan& plan, RewriteOptions options = {}) {
    std::array plans{&plan};
    return rewrite_graph(graph, plans, options);
}

size_t combined_builds = 0;
const Def* counted_combined(const Def*, const Def*, const Def* arg) {
    ++combined_builds;
    return arg->world().tuple({arg, arg, arg});
}

} // namespace

TEST_CASE("Graph pattern: semantic uses retain curried argument positions") {
    GraphIR ir;
    auto producer = ir.call(ir.q, ir.x);
    auto consumer = ir.add(producer, producer);
    GraphView graph({consumer}, ir.options());
    REQUIRE(graph.data_use_count(producer) == 2);
    CHECK(graph.data_consumer_count(producer) == 1);
    CHECK(graph.data_used_by(producer, consumer) == CheckResult::True);
    CHECK(graph.data_only_used_by(producer, consumer) == CheckResult::True);
    CHECK(graph.data_uses(producer)[0].group == 0);
    CHECK(graph.data_uses(producer)[1].group == 1);
    CHECK(graph.consumer_count(producer) == 2); // The raw owners are different App stages.
    CHECK(graph.data_only_used_by(ir.x, consumer) == CheckResult::False);
}

TEST_CASE("Graph pattern: wrapper schemas distinguish argument Packs from tensor values") {
    GraphIR ir;
    auto packed = ir.w.axm(ir.w.pi(ir.w.arr(3, ir.nat), ir.nat));
    ir.ops.add(packed->flags(), 1,
               {
                   {0, 1}
    });
    auto wrapper  = ir.w.tuple({ir.x, ir.x, ir.x});
    auto consumer = ir.call(packed, wrapper);
    GraphView graph({consumer}, ir.options());
    CHECK(graph.data_use_count(ir.x) == 3);
    CHECK(graph.data_consumer_count(ir.x) == 1);
    CHECK(graph.data_only_used_by(ir.x, consumer) == CheckResult::True);
    CHECK(graph.data_uses(ir.x)[2].path == std::vector<size_t>{2});

    OpRegistry tensor_ops;
    tensor_ops.add(packed->flags(), 1,
                   {
                       {0, 0}
    });
    GraphView tensor({consumer}, {.coverage = UseCoverage::Closed, .operators = &tensor_ops});
    CHECK(tensor.data_use_count(wrapper) == 1);
    CHECK(tensor.data_use_count(ir.x) == 0);
    CHECK(tensor.data_only_used_by(ir.x, consumer) == CheckResult::False);
}

TEST_CASE("Graph pattern: projection edges identify the original producer and result path") {
    GraphIR ir;
    auto producer  = ir.call(ir.combined, ir.x);
    auto projected = ir.w.extract(producer, 3, 1);
    auto consumer  = ir.call(ir.q, projected);
    GraphView graph({consumer}, ir.options());
    CHECK(graph.data_only_used_by(producer, consumer) == CheckResult::True);
    REQUIRE(graph.data_uses(producer).size() == 1);
    CHECK(graph.data_uses(producer)[0].value == projected);
    CHECK(graph.data_used_by(projected, consumer) == CheckResult::True);
    CHECK(graph.data_only_used_by(projected, consumer) == CheckResult::True);
    CHECK(graph.data_uses(producer)[0].result_path == std::vector<size_t>{1});
    auto escaped = ir.w.extract(producer, 3, 2);
    GraphView escaping({consumer, escaped}, ir.options());
    CHECK(escaping.data_only_used_by(producer, consumer) == CheckResult::False);
}

TEST_CASE("Graph pattern: partial decoders and input budgets do not certify exclusivity") {
    GraphIR ir;
    auto count = ir.w.axm(ir.nat);
    auto op    = ir.w.axm(ir.w.pi(ir.w.arr(count, ir.nat), ir.nat));
    ir.ops.add(op->flags(), 1,
               {
                   {0, 1}
    });
    auto consumer = ir.call(op, ir.w.pack(count, ir.x));
    auto gid      = ir.w.curr_gid();
    GraphView symbolic({consumer}, ir.options());
    CHECK_FALSE(symbolic.isa_operation(consumer)->complete);
    CHECK(symbolic.data_used_by(ir.x, consumer) == CheckResult::Unknown);
    CHECK(symbolic.data_only_used_by(ir.x, consumer) == CheckResult::Unknown);
    CHECK(ir.w.curr_gid() == gid);

    auto large = ir.w.axm(ir.w.pi(ir.w.arr(10000, ir.nat), ir.nat));
    ir.ops.add(large->flags(), 1,
               {
                   {0, 1}
    });
    auto huge          = ir.call(large, ir.w.pack(10000, ir.x));
    auto options       = ir.options();
    options.max_inputs = 4;
    GraphView bounded({huge}, options);
    CHECK_FALSE(bounded.isa_operation(huge)->complete);
    CHECK(bounded.data_use_count(ir.x) == 0);
    CHECK(bounded.data_only_used_by(ir.x, huge) == CheckResult::Unknown);
}

TEST_CASE("Graph pattern: open coverage, output escape and unknown consumers stay separate") {
    GraphIR ir;
    auto producer         = ir.call(ir.q, ir.x);
    auto consumer         = ir.call(ir.k, producer);
    auto open_options     = ir.options();
    open_options.coverage = UseCoverage::Open;
    GraphView open({consumer}, open_options);
    CHECK(open.data_only_used_by(producer, consumer) == CheckResult::Unknown);
    GraphView output({producer, consumer}, ir.options());
    CHECK(output.data_only_used_by(producer, consumer) == CheckResult::False);
    auto unknown = ir.w.axm(ir.w.pi(ir.nat, ir.nat));
    GraphView opaque({consumer, ir.call(unknown, producer)}, ir.options());
    CHECK(opaque.data_only_used_by(producer, consumer) == CheckResult::Unknown);
    CHECK(opaque.no_external_users({producer, consumer}, {consumer}) == CheckResult::Unknown);
    GraphView closed({consumer}, ir.options());
    CHECK(closed.no_external_users({producer, consumer}, {consumer}) == CheckResult::True);
    auto extra = ir.call(ir.v, producer);
    GraphView shared({consumer, extra}, ir.options());
    CHECK(shared.no_external_users({producer, consumer}, {consumer}) == CheckResult::False);
}

TEST_CASE("Graph pattern: independent roots join through a shared capture") {
    GraphIR ir;
    std::array roots{ir.call(ir.q, ir.x), ir.call(ir.k, ir.x), ir.call(ir.v, ir.x),
                     ir.call(ir.q, ir.y), ir.call(ir.k, ir.y), ir.call(ir.v, ir.y)};
    GraphView graph(roots, ir.options());
    QKV pattern(ir);
    auto gid     = ir.w.curr_gid();
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.status == SearchStatus::Complete);
    REQUIRE(matches.matches.size() == 2);
    std::set<const Def*> inputs;
    for (const auto& m : matches.matches) {
        inputs.insert(m.isa_capture(pattern.x));
        CHECK(m.at(pattern.q)->as<App>()->arg() == m.isa_capture(pattern.x));
        CHECK(m.at(pattern.k)->as<App>()->arg() == m.isa_capture(pattern.x));
        CHECK(m.at(pattern.v)->as<App>()->arg() == m.isa_capture(pattern.x));
    }
    CHECK(inputs == std::set<const Def*>{ir.x, ir.y});
    CHECK(ir.w.curr_gid() == gid);
    auto reversed = find_matches(pattern.graph, graph, {.reverse_candidates = true});
    REQUIRE(reversed.status == SearchStatus::Complete);
    std::set<const Def*> reversed_inputs;
    for (const auto& m : reversed.matches)
        reversed_inputs.insert(m.isa_capture(pattern.x));
    CHECK(inputs == reversed_inputs);
}

TEST_CASE("Graph pattern: later roots backtrack into an earlier root's alternative") {
    GraphIR ir;
    auto a = ir.w.lit_nat(1), b = ir.w.lit_nat(2);
    auto binary = ir.add(a, b);
    auto unary  = ir.call(ir.q, b);
    GraphView graph({binary, unary}, ir.options());
    PatternProgram p;
    auto x = p.capture();
    GraphPattern pattern(p);
    auto first   = pattern.anchor(p.either(pair(p, ir.sum, x, p.any()), pair(p, ir.sum, p.any(), x)));
    auto second  = pattern.anchor(call(p, ir.q, x));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.status == SearchStatus::Complete);
    REQUIRE(matches.matches.size() == 1);
    CHECK(matches.matches[0].isa_capture(x) == b);
    CHECK(matches.matches[0].at(first) == binary);
    CHECK(matches.matches[0].at(second) == unary);
}

TEST_CASE("Graph pattern: edge constraints filter consumer, argument and result positions") {
    GraphIR ir;
    auto producer = ir.call(ir.combined, ir.x);
    auto consumer = ir.call(ir.q, ir.w.extract(producer, 3, 1));
    GraphView graph({consumer}, ir.options());
    PatternProgram p;
    GraphPattern pattern(p);
    auto a = pattern.anchor(call(p, ir.combined, p.any()));
    auto b = pattern.anchor(call(p, ir.q, p.any()));
    pattern.only_used_by(a, b, {.group = 0, .path = std::vector<size_t>{}, .result_path = std::vector<size_t>{1}});
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.status == SearchStatus::Complete);
    CHECK(matches.matches.size() == 1);
    GraphPattern wrong(p);
    auto wa = wrong.anchor(call(p, ir.combined, p.any()));
    auto wb = wrong.anchor(call(p, ir.q, p.any()));
    wrong.used_by(wa, wb, {.result_path = std::vector<size_t>{0}});
    CHECK(find_matches(wrong, graph).matches.empty());
    auto options     = ir.options();
    options.coverage = UseCoverage::Open;
    GraphView open({consumer}, options);
    auto unknown = find_matches(pattern, open);
    CHECK(unknown.status == SearchStatus::Unknown);
    CHECK(unknown.matches.empty());
}

TEST_CASE("Graph pattern: alias policy, connectivity and handle validation") {
    GraphIR ir;
    GraphView graph({ir.call(ir.q, ir.x)}, ir.options());
    PatternProgram p, other;
    GraphPattern disconnected(p);
    disconnected.anchor(p.any());
    disconnected.anchor(p.any());
    CHECK_THROWS_AS(find_matches(disconnected, graph), std::invalid_argument);
    GraphPattern aliases(p);
    auto x = p.capture();
    aliases.anchor(x);
    aliases.anchor(x);
    CHECK(find_matches(aliases, graph).matches.empty());
    aliases.allow_aliases();
    CHECK(find_matches(aliases, graph).matches.size() == 1);
    CHECK_THROWS_AS(aliases.anchor(other.any()), std::invalid_argument);
    GraphPattern foreign(p);
    auto a = foreign.anchor(p.any());
    CHECK_THROWS_AS(aliases.used_by(a, a), std::invalid_argument);
    GraphPattern empty(p);
    CHECK_THROWS_AS(find_matches(empty, graph), std::invalid_argument);
}

TEST_CASE("Graph pattern: result and work limits mark incomplete enumeration") {
    GraphIR ir;
    GraphView graph({ir.call(ir.q, ir.x), ir.call(ir.k, ir.x), ir.call(ir.v, ir.x)}, ir.options());
    QKV pattern(ir);
    CHECK(find_matches(pattern.graph, graph, {.max_results = 0}).status == SearchStatus::Result_Limit);
    auto limited = find_matches(pattern.graph, graph, {.max_results = 1});
    CHECK(limited.status == SearchStatus::Result_Limit);
    CHECK(limited.matches.size() == 1);
    auto steps = find_matches(pattern.graph, graph, {.match = {.max_steps = 0}});
    CHECK(steps.status == SearchStatus::Budget_Exceeded);
    CHECK(steps.matches.empty());
    auto candidates = find_matches(pattern.graph, graph, {.max_candidates = 0});
    CHECK(candidates.status == SearchStatus::Budget_Exceeded);
    CHECK(candidates.matches.empty());
}

TEST_CASE("Graph pattern: exhaustive tiny DAG joins agree with graph matching") {
    GraphIR ir;
    std::array values{ir.x, ir.y, ir.call(ir.q, ir.x), ir.call(ir.q, ir.y)};
    DefVec roots;
    for (auto value : values)
        for (auto op : {ir.q, ir.k, ir.v})
            roots.push_back(ir.call(op, value));
    GraphView graph(roots, ir.options());
    QKV pattern(ir);
    std::set<std::array<const Def*, 3>> expected, found;
    for (auto q : graph.operations())
        for (auto k : graph.operations())
            for (auto v : graph.operations()) {
                if (q->as<App>()->callee() != ir.q || k->as<App>()->callee() != ir.k || v->as<App>()->callee() != ir.v)
                    continue;
                if (q->as<App>()->arg() == k->as<App>()->arg() && k->as<App>()->arg() == v->as<App>()->arg())
                    expected.insert({q, k, v});
            }
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.status == SearchStatus::Complete);
    for (const auto& m : matches.matches)
        found.insert({m.at(pattern.q), m.at(pattern.k), m.at(pattern.v)});
    CHECK(found == expected);
}

TEST_CASE("Graph rewrite: three roots share one new computation") {
    GraphIR ir;
    GraphView graph({ir.call(ir.q, ir.x), ir.call(ir.k, ir.x), ir.call(ir.v, ir.x)}, ir.options());
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    combine(plan, matches.matches.front(), pattern, ir);
    auto gid    = ir.w.curr_gid();
    auto result = build(graph, plan);
    REQUIRE(result.roots().size() == 3);
    const Def* shared = nullptr;
    for (size_t i = 0; i != 3; ++i) {
        auto extract = result.roots()[i]->as<Extract>();
        CHECK(Lit::as(extract->index()) == i);
        if (!shared) shared = extract->tuple();
        CHECK(extract->tuple() == shared);
        CHECK(&extract->world() == &result.world());
    }
    CHECK(shared->as<App>()->axm()->flags() == ir.combined->flags());
    CHECK(result.replacements() == 3);
    CHECK(ir.w.curr_gid() == gid);
    CHECK_THROWS_AS(result.publish(graph), std::invalid_argument); // Explicit roots have no function publication scope.
}

TEST_CASE("Graph rewrite: open multi-root replacements preserve the function binder") {
    GraphIR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("qkv_scope");
    auto x = f->var();
    f->set(false, ir.add(ir.call(ir.q, x), ir.add(ir.call(ir.k, x), ir.call(ir.v, x))))->externalize();
    auto name = f->sym();
    GraphView graph({f->body()}, ir.options(f));
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    combine(plan, matches.matches.front(), pattern, ir);
    auto result = build(graph, plan);
    auto scope  = result.scope();
    REQUIRE(scope);
    const Def* combined = nullptr;
    DefSet seen;
    std::vector<const Def*> pending{scope->body()};
    size_t merged = 0;
    for (size_t i = 0; i != pending.size(); ++i) {
        auto def = pending[i];
        if (!seen.insert(def).second || def->is_mutable()) continue;
        if (auto app = def->isa<App>(); app && app->axm() && app->axm()->flags() == ir.combined->flags()) {
            ++merged;
            combined = app;
        }
        for (auto op : def->ops())
            pending.push_back(op);
    }
    CHECK(merged == 1);
    REQUIRE(combined);
    CHECK(combined->as<App>()->arg() == scope->var());
    CHECK(scope->is_closed());
    CHECK(result.world().externals()[name] == scope);
    CHECK(ir.w.externals()[name] == f);
    result.publish(graph);
    CHECK(result.is_published());
    CHECK(ir.w.externals()[name] == scope);
    CHECK_FALSE(graph.is_current());
    CHECK_THROWS_AS(result.publish(graph), std::invalid_argument);
    CHECK_THROWS_AS(find_matches(pattern.graph, graph), std::invalid_argument);
    CHECK_THROWS_AS(RewritePlan{matches.matches.front()}, std::invalid_argument);
}

TEST_CASE("Graph rewrite: a shared Def is replaced only in the selected function") {
    GraphIR ir;
    auto value = ir.call(ir.q, ir.x);
    auto f     = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("selected");
    auto g     = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("other");
    f->set(false, value)->externalize();
    g->set(false, value)->externalize();
    auto fn = f->sym(), gn = g->sym();
    GraphView graph({value}, ir.options(f));
    PatternProgram p;
    GraphPattern pattern(p);
    auto a       = pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(a), plan.literal(ir.nat, 42));
    auto result = build(graph, plan);
    CHECK(result.world().externals()[fn]->as<Lam>()->body() == result.world().lit_nat(42));
    auto other = result.world().externals()[gn]->as<Lam>()->body()->as<App>();
    CHECK(other->axm()->flags() == ir.q->flags());
    result.publish(graph);
    CHECK(ir.w.externals()[fn]->as<Lam>()->body() == ir.w.lit_nat(42));
    CHECK(ir.w.externals()[gn]->as<Lam>()->body()->isa<App>());
}

TEST_CASE("Graph rewrite: cycles including imported descendants are rejected") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x), k = ir.call(ir.k, ir.x), v = ir.call(ir.v, ir.x);
    auto sum = ir.add(q, k);
    GraphView graph({sum, v}, ir.options());
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan direct(matches.matches.front());
    direct.replace(q, direct.import(k));
    direct.replace(k, direct.import(q));
    auto gid = ir.w.curr_gid();
    CHECK_THROWS_AS(build(graph, direct), std::invalid_argument);
    RewritePlan hidden(matches.matches.front());
    hidden.replace(q, hidden.import(sum));
    CHECK_THROWS_AS(build(graph, hidden), std::invalid_argument);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Graph rewrite: mismatched types and invalid result projections are atomic failures") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x), k = ir.call(ir.k, ir.x), v = ir.call(ir.v, ir.x);
    GraphView graph({q, k, v}, ir.options());
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    auto gid = ir.w.curr_gid();
    RewritePlan bad_type(matches.matches.front());
    bad_type.replace(q, bad_type.literal(ir.w.type_bool(), 1));
    CHECK_THROWS_AS(build(graph, bad_type), std::invalid_argument);
    RewritePlan bad_index(matches.matches.front());
    auto merged = bad_index.call(ir.combined, {bad_index.import(ir.x)});
    bad_index.replace(q, bad_index.extract(merged, 3));
    CHECK_THROWS_AS(build(graph, bad_index), std::invalid_argument);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(graph.is_current());
}

TEST_CASE("Graph rewrite: conflicting plans and foreign snapshots cannot be combined") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x), k = ir.call(ir.k, ir.x), v = ir.call(ir.v, ir.x);
    GraphView graph({q, k, v}, ir.options());
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan a(matches.matches.front()), b(matches.matches.front());
    a.replace(q, a.literal(ir.nat, 1));
    b.replace(k, b.literal(ir.nat, 2));
    std::array plans{&a, &b};
    CHECK_THROWS_AS(rewrite_graph(graph, plans), std::invalid_argument);
    GraphView other({q, k, v}, ir.options());
    CHECK_THROWS_AS(build(other, a), std::invalid_argument);
    CHECK_THROWS_AS(b.replace(q, a.literal(ir.nat, 3)), std::invalid_argument);
}

TEST_CASE("Graph rewrite: independent plans may share read-only inputs") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x), k = ir.call(ir.k, ir.x);
    GraphView graph({q, k}, ir.options());
    PatternProgram p;
    GraphPattern pq(p), pk(p);
    pq.anchor(call(p, ir.q, p.any()));
    pk.anchor(call(p, ir.k, p.any()));
    auto mq = find_matches(pq, graph), mk = find_matches(pk, graph);
    REQUIRE(mq.matches.size() == 1);
    REQUIRE(mk.matches.size() == 1);
    RewritePlan a(mq.matches.front()), b(mk.matches.front());
    a.replace(q, a.literal(ir.nat, 1));
    b.replace(k, b.literal(ir.nat, 2));
    std::array plans{&a, &b};
    auto result = rewrite_graph(graph, plans);
    CHECK(result.roots()[0] == result.world().lit_nat(1));
    CHECK(result.roots()[1] == result.world().lit_nat(2));
}

TEST_CASE("Graph rewrite: unresolved purity, foreign values and exhausted recipe budgets are rejected") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x);
    GraphView graph({q}, ir.options());
    PatternProgram p;
    GraphPattern pattern(p);
    pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    auto unknown = ir.w.axm(ir.w.pi(ir.nat, ir.nat));
    plan.replace(q, plan.call(unknown, {plan.import(ir.x)}));
    CHECK_THROWS_AS(build(graph, plan), std::invalid_argument);
    RewritePlan bounded(matches.matches.front());
    bounded.replace(q, bounded.literal(ir.nat, 1));
    CHECK_THROWS_AS(build(graph, bounded, {.max_recipes = 0}), std::invalid_argument);
    Driver other;
    RewritePlan foreign(matches.matches.front());
    foreign.replace(q, foreign.import(other.world().lit_nat(1)));
    CHECK_THROWS_AS(build(graph, foreign), std::invalid_argument);
}

TEST_CASE("Graph rewrite: source mutation invalidates a built replacement before publication") {
    GraphIR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("stale_scope");
    f->set(false, ir.call(ir.q, f->var()))->externalize();
    auto name = f->sym();
    GraphView graph({f->body()}, ir.options(f));
    PatternProgram p;
    GraphPattern pattern(p);
    auto a       = pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(a), plan.literal(ir.nat, 42));
    auto result = build(graph, plan);
    f->set(false, ir.w.lit_nat(7));
    CHECK_FALSE(graph.is_current());
    CHECK_THROWS_AS(result.publish(graph), std::invalid_argument);
    CHECK_THROWS_AS(build(graph, plan), std::invalid_argument);
    CHECK(ir.w.externals()[name] == f);
    CHECK(f->body() == ir.w.lit_nat(7));
}

TEST_CASE("Graph pattern: opaque nested users cannot certify exclusive consumption") {
    GraphIR ir;
    auto producer = ir.call(ir.q, ir.x);
    auto nested   = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    nested->set(false, ir.call(ir.k, producer));
    auto f    = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat));
    auto body = ir.add(producer, ir.w.app(nested, ir.w.lit_nat(0)));
    f->set(false, body);
    GraphView graph({body}, ir.options(f));
    CHECK(graph.data_only_used_by(producer, body) == CheckResult::Unknown);
}

TEST_CASE("Graph rewrite: changes to other externals prevent lost updates at publication") {
    GraphIR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("transaction_f");
    auto g = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("transaction_g");
    f->set(false, ir.call(ir.q, f->var()))->externalize();
    g->set(false, ir.w.lit_nat(0))->externalize();
    GraphView graph({f->body()}, ir.options(f));
    PatternProgram p;
    GraphPattern pattern(p);
    auto a       = pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(a), plan.literal(ir.nat, 42));
    auto result = build(graph, plan);
    g->set(false, ir.w.lit_nat(9));
    CHECK_FALSE(graph.is_current());
    CHECK_THROWS_AS(result.publish(graph), std::invalid_argument);
    CHECK(g->body() == ir.w.lit_nat(9));
}

TEST_CASE("Graph rewrite: adding an external after construction invalidates publication") {
    GraphIR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("transaction_f");
    f->set(false, ir.call(ir.q, f->var()))->externalize();
    GraphView graph({f->body()}, ir.options(f));
    PatternProgram p;
    GraphPattern pattern(p);
    auto a       = pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(a), plan.literal(ir.nat, 42));
    auto result = build(graph, plan);
    auto added  = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("added_external");
    added->set(false, ir.w.lit_nat(9))->externalize();
    CHECK_FALSE(graph.is_current());
    CHECK_THROWS_AS(result.publish(graph), std::invalid_argument);
    CHECK(ir.w.externals()[added->sym()] == added);
}

TEST_CASE("Graph rewrite: an intervening World rebuild invalidates all graph handles") {
    GraphIR ir;
    auto f = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("rebuild_scope");
    f->set(false, ir.call(ir.q, f->var()))->externalize();
    GraphView graph({f->body()}, ir.options(f));
    PatternProgram p;
    auto input = p.capture();
    GraphPattern pattern(p);
    auto a       = pattern.anchor(call(p, ir.q, input));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(a), plan.literal(ir.nat, 42));
    {
        Cleanup cleanup(ir.w, "invalidate_graph");
        cleanup.run();
    }
    CHECK_FALSE(graph.is_current());
    CHECK_THROWS_AS(matches.matches.front().at(a), std::invalid_argument);
    CHECK_THROWS_AS(matches.matches.front().isa_capture(input), std::invalid_argument);
    CHECK_THROWS_AS(RewritePlan{matches.matches.front()}, std::invalid_argument);
    CHECK_THROWS_AS(build(graph, plan), std::invalid_argument);
    CHECK_FALSE(p.is_current());
    auto rebuilt = ir.w.externals()[ir.w.sym("rebuild_scope")]->as<Lam>();
    GraphView fresh({rebuilt->body()}, ir.options(rebuilt));
    CHECK_THROWS_AS(find_matches(pattern, fresh), std::invalid_argument);
}

TEST_CASE("Graph pattern: deferred cross-anchor predicates see complete bindings") {
    GraphIR ir;
    GraphView graph({ir.call(ir.q, ir.x), ir.call(ir.k, ir.x)}, ir.options());
    PatternProgram p;
    auto x = p.capture();
    auto k = p.capture(call(p, ir.k, x));
    auto q = p.where(call(p, ir.q, x), {x, k}, [](std::span<const Def* const> values) {
        return values[1]->as<App>()->arg() == values[0] ? CheckResult::True : CheckResult::False;
    });
    GraphPattern pattern(p);
    pattern.anchor(q);
    pattern.anchor(k);
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.status == SearchStatus::Complete);
    REQUIRE(matches.matches.size() == 1);
    CHECK(matches.matches.front().isa_capture(x) == ir.x);
    GraphPattern unknown(p);
    unknown.anchor(p.where(call(p, ir.q, x), {x}, [](auto) { return CheckResult::Unknown; }));
    auto uncertain = find_matches(unknown, graph);
    CHECK(uncertain.status == SearchStatus::Unknown);
    CHECK(uncertain.matches.empty());
}

TEST_CASE("Graph rewrite: source purity is explicit and registries are frozen in snapshots") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x);
    OpRegistry registry;
    registry.add(ir.q->flags(), 1, {{0}});
    GraphView graph({q}, {.coverage = UseCoverage::Closed, .operators = &registry});
    CHECK_FALSE(graph.is_pure(q));
    CHECK_FALSE(graph.is_pure(nullptr));
    registry.add(ir.k->flags(), 1, {{0}}, true);
    CHECK_FALSE(graph.is_pure(ir.k->flags()));
    PatternProgram p;
    GraphPattern pattern(p);
    pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(q, plan.literal(ir.nat, 1));
    CHECK_THROWS_AS(build(graph, plan), std::invalid_argument);
    CHECK_THROWS_AS(registry.add(ir.q->flags(), 1, {{0}}, true), std::invalid_argument);
    CHECK_THROWS_AS(registry.add(ir.v->flags(), 1, {{0}, {0}}, true), std::invalid_argument);
}

TEST_CASE("Graph rewrite: a shared recipe is evaluated once even when its call normalizes") {
    GraphIR ir;
    auto fused = ir.w.axm(counted_combined, 1, Axm::Trip_End, ir.w.pi(ir.nat, ir.w.arr(3, ir.nat)));
    ir.ops.add(fused->flags(), 1, {{0}}, true);
    GraphView graph({ir.call(ir.q, ir.x), ir.call(ir.k, ir.x), ir.call(ir.v, ir.x)}, ir.options());
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    auto merged = plan.call(fused, {plan.import(ir.x)});
    plan.replace(matches.matches.front().at(pattern.q), plan.extract(merged, 0));
    plan.replace(matches.matches.front().at(pattern.k), plan.extract(merged, 1));
    plan.replace(matches.matches.front().at(pattern.v), plan.extract(merged, 2));
    combined_builds = 0;
    auto result     = build(graph, plan);
    CHECK(combined_builds == 1);
    CHECK(result.roots()[0] == result.roots()[1]);
    CHECK(result.roots()[1] == result.roots()[2]);
}

TEST_CASE("Graph rewrite: evaluation and source dependency work have explicit bounds") {
    GraphIR ir;
    auto q = ir.call(ir.q, ir.x);
    GraphView graph({q}, ir.options());
    PatternProgram p;
    GraphPattern pattern(p);
    pattern.anchor(call(p, ir.q, p.any()));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(q, plan.call(ir.k, {plan.import(ir.x)}));
    CHECK_THROWS_AS(build(graph, plan, {.max_source_nodes = 0}), std::invalid_argument);
    CHECK_THROWS_AS(build(graph, plan, {.max_depth = 2}), std::invalid_argument);
    CHECK(graph.is_current());
}

TEST_CASE("Graph rewrite: continuation arguments retain their binder and share the combined result") {
    GraphIR ir;
    auto f            = ir.w.mut_con({ir.nat, ir.w.cn(ir.w.arr(3, ir.nat))})->set("qkv_cps");
    auto [input, ret] = f->vars<2>();
    f->app(false, ret, ir.w.tuple({ir.call(ir.q, input), ir.call(ir.k, input), ir.call(ir.v, input)}))->externalize();
    GraphView graph({f->body()}, ir.options(f));
    QKV pattern(ir);
    auto matches = find_matches(pattern.graph, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    combine(plan, matches.matches.front(), pattern, ir);
    auto result = build(graph, plan);
    auto scope  = result.scope();
    auto body   = scope->body()->as<App>();
    CHECK(body->callee() == scope->var(2, 1));
    auto combined = body->arg()->as<App>();
    CHECK(combined->axm()->flags() == ir.combined->flags());
    CHECK(combined->arg() == scope->var(2, 0));
    CHECK(scope->is_closed());
}

TEST_CASE("Graph rewrite: existing recursive references resolve to the rebuilt scope stub") {
    GraphIR ir;
    auto f         = ir.w.mut_lam(ir.w.pi(ir.nat, ir.nat))->set("recursive_scope");
    auto recursive = ir.w.app(f, f->var());
    f->set(false, ir.call(ir.q, recursive))->externalize();
    GraphView graph({f->body()}, ir.options(f));
    PatternProgram p;
    auto input = p.capture();
    GraphPattern pattern(p);
    auto anchor  = pattern.anchor(call(p, ir.q, input));
    auto matches = find_matches(pattern, graph);
    REQUIRE(matches.matches.size() == 1);
    RewritePlan plan(matches.matches.front());
    plan.replace(matches.matches.front().at(anchor),
                 plan.call(ir.k, {plan.import(matches.matches.front().isa_capture(input))}));
    auto result = build(graph, plan);
    auto scope  = result.scope();
    auto body   = scope->body()->as<App>();
    CHECK(body->axm()->flags() == ir.k->flags());
    auto self = body->arg()->as<App>();
    CHECK(self->callee() == scope);
    CHECK(self->arg() == scope->var());
    CHECK(scope->is_closed());
}
