#include "mim/plug/tensor/phase/combine_products.h"

#include <array>
#include <limits>

#include <mim/pattern/rewrite.h>

#include "mim/plug/tensor/tensor.h"

namespace mim::plug::tensor::phase {

namespace {

using namespace mim::pattern;

struct Products {
    PatternProgram p;
    Pattern ring  = p.capture();
    Pattern input = p.capture();
    std::vector<Pattern> meta, weights, values;
    GraphPattern graph{p};
    std::vector<Anchor> anchors;

    Products() {
        for (size_t i = 0; i != 3; ++i) {
            meta.push_back(p.capture());
            weights.push_back(p.capture());
            auto pair = p.node(Node::Tuple, {input, weights.back()});
            // Two equal operands may normalize into a Pack. An opaque argument wrapper is
            // conservatively left alone; such cases can be handled by a future semantic pattern.
            values.push_back(p.capture(graph::call(p, Annex::base<product_2d>(), {ring, meta.back(), pair})));
        }
        auto checked
            = p.where(values[2], {values[0], values[1], values[2], ring}, [](std::span<const Def* const> args) {
                  if (!graph::is_standard_ring(args[3])) return CheckResult::False;
                  if (!(args[0]->gid() < args[1]->gid() && args[1]->gid() < args[2]->gid())) return CheckResult::False;
                  const Def *rows = nullptr, *contraction = nullptr;
                  u64 total = 0;
                  for (size_t i = 0; i != 3; ++i) {
                      auto dims = args[i]->as<App>()->decurry()->arg();
                      auto m = graph::slot(dims, 3, 0), k = graph::slot(dims, 3, 1), n = graph::slot(dims, 3, 2);
                      if (!m || !k || !n) return CheckResult::False;
                      if (i && (m != rows || k != contraction)) return CheckResult::False;
                      rows = m, contraction = k;
                      auto width = Lit::isa(n);
                      if (!width || !*width || *width > std::numeric_limits<u64>::max() - total)
                          return CheckResult::False;
                      total += *width;
                  }
                  return CheckResult::True;
              });
        anchors.push_back(graph.anchor(values[0]));
        anchors.push_back(graph.anchor(values[1]));
        anchors.push_back(graph.anchor(checked));
    }

    void recipe(RewritePlan& plan, const GraphMatch& match, World& w) const {
        auto nat  = [&](u64 value) { return plan.literal(w.type_nat(), value); };
        auto dims = match.isa_capture(meta[0]);
        auto m = plan.import(graph::slot(dims, 3, 0)), k = plan.import(graph::slot(dims, 3, 1));
        auto type = plan.import(graph::slot(match.isa_capture(ring), 4, 0));
        std::vector<Recipe> shapes, inputs;
        std::array<u64, 3> widths;
        u64 total = 0;
        for (size_t i = 0; i != 3; ++i) {
            widths[i] = *Lit::isa(graph::slot(match.isa_capture(meta[i]), 3, 2));
            total += widths[i];
            shapes.push_back(plan.tuple({k, nat(widths[i])}));
            inputs.push_back(plan.import(match.isa_capture(weights[i])));
        }
        auto concat = plan.call(w.annex<tensor::concat>()->as<Axm>(),
                                {plan.tuple({type, nat(3), nat(2)}), plan.literal(w.type_idx(2), 1),
                                 plan.tuple(std::move(shapes)), plan.tuple(std::move(inputs))});
        auto merged = plan.call(w.annex<product_2d>()->as<Axm>(),
                                {plan.import(match.isa_capture(ring)), plan.tuple({m, k, nat(total)}),
                                 plan.tuple({plan.import(match.isa_capture(input)), concat})});
        auto slice  = w.annex<tensor::slice>()->as<Axm>();
        for (u64 i = 0, offset = 0; i != 3; offset += widths[i++]) {
            auto result = plan.call(slice, {plan.tuple({type, nat(2)}), plan.tuple({m, nat(total)}),
                                            plan.tuple({plan.tuple({nat(0), nat(offset)}), plan.tuple({nat(1), nat(1)}),
                                                        plan.tuple({m, nat(widths[i])})}),
                                            merged});
            plan.replace(match.at(anchors[i]), result);
        }
    }
};

} // namespace

void CombineProducts::start() {
    OpRegistry ops;
    graph::register_operations(ops);
    // Publication rebuilds every external. Retain names rather than pointers across functions.
    std::vector<Sym> names;
    for (auto [name, def] : world().externals())
        if (def->isa<Lam>()) names.push_back(name);
    for (auto name : names) {
        auto fn = world().externals()[name]->isa_mut<Lam>();
        if (!fn || !fn->is_set() || !fn->is_closed()) continue;
        Products patterns;
        GraphView view({fn->body()}, {.operators = &ops, .scope = fn});
        auto found = find_matches(patterns.graph, view);
        if (found.status != SearchStatus::Complete) {
            profile_count("tensor.combine.incomplete");
            log().d("combine_products skipped {}: incomplete search", fn);
            continue;
        }
        if (found.matches.empty()) continue;
        // Where conservatively records the whole snapshot. Publish one triple per function
        // per run, so no batch plan can invalidate another triple's native legality analysis.
        RewritePlan plan(found.matches.front());
        patterns.recipe(plan, found.matches.front(), world());
        std::array plans{&plan};
        auto result = rewrite_graph(view, plans);
        result.publish(view);
        profile_count("tensor.combine.triples");
        invalidate();
    }
}

} // namespace mim::plug::tensor::phase
