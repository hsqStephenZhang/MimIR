#include "mim/phase/graph_rule_rewrite.h"

#include <array>
#include <set>
#include <stdexcept>
#include <utility>

#include "mim/driver.h"

#include "mim/pattern/graph_rule.h"

namespace mim {

namespace {
std::string_view search_status(pattern::SearchStatus status) {
    switch (status) {
        case pattern::SearchStatus::Complete: return "Complete";
        case pattern::SearchStatus::Unknown: return "Unknown";
        case pattern::SearchStatus::Budget_Exceeded: return "Budget_Exceeded";
        case pattern::SearchStatus::Result_Limit: return "Result_Limit";
    }
    std::unreachable();
}

class FunctionPublisher : public pattern::DAGRewriter {
public:
    FunctionPublisher(World& target, const fe::SymMap<const Def*>& selected)
        : DAGRewriter(target, 64 * (target.curr_gid() + 1))
        , selected_(selected) {}

    const Def* lookup(const Def* def) override {
        if (auto value = Rewriter::lookup(def)) return value;
        if (def->is_external())
            if (auto i = selected_.find(def->sym()); i != selected_.end() && i->second != def)
                return map(def, rewrite(i->second));
        if (auto axm = def->isa<Axm>(); axm && axm->plugin() == Annex::Global_Plugin) return nullptr;
        if (auto flags = def->world().annexes().opaque(def)) return map(def, world().annex(*flags));
        return nullptr;
    }

    const Def* rewrite_mut_Lam(Lam* fn) override {
        auto stub = world().mut_lam(rewrite(fn->type())->as<Pi>())->set(fn->dbg_key());
        if (fn->is_external()) stub->externalize();
        return rewrite_stub(fn, stub);
    }

private:
    const fe::SymMap<const Def*>& selected_;
};

struct Region {
    const Def* body;
    std::vector<size_t> path;
};

std::vector<Region> regions(Lam* fn) {
    std::vector<Region> result{
        {fn->body(), {}}
    };
    struct Work {
        const Def* def;
        size_t parent, index;
        DefSet ancestors;
    };
    std::vector<Work> pending{
        {fn->body(), size_t(-1), 0, {fn}}
    };
    size_t limit = 64 * (fn->world().curr_gid() + 1);
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i >= limit) throw pattern::RebuildLimit("graph region discovery work budget exceeded");
        auto [def, parent, index, ancestors] = pending[i];
        if (auto lam = def->isa<Lam>()) {
            if (!lam->is_set() || lam->is_external() || !ancestors.insert(lam).second) continue;
            std::vector<size_t> path;
            for (size_t cursor = i; pending[cursor].parent != size_t(-1); cursor = pending[cursor].parent)
                path.push_back(pending[cursor].index);
            std::ranges::reverse(path);
            path.push_back(1);
            result.push_back({lam->body(), path});
            pending.push_back({lam->body(), i, 1, std::move(ancestors)});
            continue;
        }
        if (!def->isa<App>() && !def->isa<Tuple>() && !def->isa_imm<Pack>()) continue;
        for (size_t j = 0; j != def->num_ops(); ++j)
            pending.push_back({def->op(j), i, j, ancestors});
    }
    return result;
}
} // namespace

GraphRuleRewrite::GraphRuleRewrite(World& world,
                                   std::vector<const Rule*> rules,
                                   std::vector<const Axm*> pure,
                                   bool skip_unproved)
    : Phase(world, "graph_rule_app")
    , skip_unproved_(skip_unproved) {
    templates_ = world.inherit();
    Rewriter importer(*templates_);
    for (auto rule : rules) {
        if (!rule || !rule->is_set() || !rule->is_closed() || &rule->driver() != &driver())
            throw std::invalid_argument("graph_rule_app expects closed Rules of the same Driver");
        pattern::RulePattern(rule).check_definition();
        rules_.push_back(importer.rewrite(rule)->as<Rule>());
    }
    for (auto axm : pure) {
        if (!axm || &axm->driver() != &driver())
            throw std::invalid_argument("graph_rule_app expects pure axioms of the same Driver");
        pure_.push_back(importer.rewrite(axm)->as<Axm>());
    }
}

void GraphRuleRewrite::apply(const App* app) {
    if (!app) throw std::invalid_argument("graph_rule_app requires (Rules, pure axioms)");
    auto pair       = app->arg()->isa<Tuple>();
    auto configured = App::uncurry_callee(app)->sym() == driver().sym("compile.graph_rule_app_with");
    if (!pair || pair->num_ops() != (configured ? 3 : 2))
        app->blame("graph_rule_app requires (Rules, pure axioms) and optional data_ops").bail();
    schemas_.clear();
    if (configured) schemas_ = pattern::parse_data_ops(pair->op(2));
    rules_.clear();
    pure_.clear();
    templates_ = world().inherit();
    Rewriter importer(*templates_);
    std::function<void(const Def*, bool)> collect = [&](const Def* def, bool rules) {
        if (!rules) {
            auto name = tuple2str(def);
            if (!name.empty()) {
                auto annex = world().annex(driver().sym(name));
                auto axm   = annex ? annex->isa<Axm>() : nullptr;
                if (!axm) def->blame("graph_rule_app purity reference must name an axiom: {}", name).bail();
                pure_.push_back(importer.rewrite(axm)->as<Axm>());
                return;
            }
        }
        if (auto tuple = def->isa<Tuple>()) {
            for (auto op : tuple->ops())
                collect(op, rules);
        } else if (auto pack = def->isa_imm<Pack>()) {
            auto count = Lit::isa(*pack->shape());
            if (!count) def->blame("graph_rule_app requires static lists").bail();
            if (*count) collect(pack->body(), rules);
        } else if (rules) {
            auto rule = def->isa<Rule>();
            if (!rule || !rule->is_set() || !rule->is_closed())
                def->blame("graph_rule_app expects closed Rules").bail();
            pattern::RulePattern(rule).check_definition();
            rules_.push_back(importer.rewrite(rule)->as<Rule>());
        } else {
            auto axm = def->isa<Axm>();
            if (!axm) def->blame("graph_rule_app purity declarations must be axioms").bail();
            pure_.push_back(importer.rewrite(axm)->as<Axm>());
        }
    };
    collect(pair->op(0), true);
    collect(pair->op(1), false);
}

std::unique_ptr<Phase> GraphRuleRewrite::recreate() {
    auto result              = annex() ? std::make_unique<GraphRuleRewrite>(world(), annex())
                                       : std::make_unique<GraphRuleRewrite>(world(), std::string(name()));
    result->templates_       = templates_;
    result->rules_           = rules_;
    result->pure_            = pure_;
    result->schemas_         = schemas_;
    result->rewrite_options_ = rewrite_options_;
    result->skip_unproved_   = skip_unproved_;
    return result;
}

void GraphRuleRewrite::start() {
    if (rules_.empty()) return;
    Rewriter importer(world());
    auto ops = driver().graph_operations();
    std::set<flags_t> registered;
    for (auto source : pure_) {
        auto axm = importer.rewrite(source)->as<Axm>();
        if (!registered.insert(axm->flags()).second) continue;
        if (ops.contains(axm->flags())) {
            ops.assert_pure(axm->flags());
            continue;
        }
        if (auto schema = std::ranges::find(schemas_, axm->flags(), &pattern::DataOp::axm); schema != schemas_.end()) {
            ops.add(axm->flags(), schema->num_groups, schema->inputs, true);
            continue;
        }
        size_t groups = 0;
        for (auto type = axm->type(); auto pi = type->isa<Pi>(); type = pi->codom())
            if (++groups > 64) throw std::invalid_argument("graph_rule_app axiom curry depth exceeds 64");
        ops.add(
            axm->flags(),
            [groups](const Def* def, size_t) -> std::optional<pattern::OpView> {
                size_t count = 0;
                for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee())
                    if (++count > groups) return {};
                if (count != groups) return {};
                return pattern::OpView{.inputs = {}, .complete = false};
            },
            true);
    }
    for (const auto& schema : schemas_)
        if (registered.insert(schema.axm).second) {
            pattern::OpRegistry explicit_ops;
            explicit_ops.add(schema.axm, schema.num_groups, schema.inputs);
            ops.overlay(explicit_ops);
        }
    auto bodies = pattern::function_bodies(world());
    if (bodies.empty()) return;
    pattern::GraphView facts(world(), ops);
    pattern::GuardContext guards(facts);
    std::vector<pattern::GraphRewrite> proposals;
    fe::SymMap<const Def*> selected;
    for (auto [name, ext] : world().externals())
        selected.emplace(name, ext);
    std::vector<Sym> names;
    for (auto [name, def] : world().externals())
        if (def->isa<Lam>()) names.push_back(name);
    for (auto name : names) {
        auto fn = world().externals()[name]->isa_mut<Lam>();
        if (!fn || !fn->is_set() || !fn->is_closed()) continue;
        std::vector<Region> instances;
        try {
            instances = regions(fn);
        } catch (const pattern::RebuildLimit& error) {
            log().d("graph_rule_app skipped region discovery in {}: {}", fn, error.what());
            continue;
        }
        bool changed = false;
        for (auto source : rules_) {
            pattern::GraphRulePattern compiled(importer.rewrite(source)->as<Rule>());
            for (const auto& region : instances) {
                pattern::GraphView view({region.body}, {.operators     = &ops,
                                                        .scope         = fn,
                                                        .region_path   = region.path,
                                                        .shared_region = !compiled.uses_graph_facts()});
                std::optional<pattern::GraphRewrite> accepted;
                std::string reason = "no accepted match visited";
                pattern::GraphMatchOptions options;
                options.max_results = 1;
                options.accept      = [&](const pattern::GraphMatch& match) {
                    auto candidate = compiled.plan(match, view, &guards);
                    if (!candidate.plan) {
                        reason = candidate.reason;
                        if (candidate.status == pattern::MatchStatus::Unknown) {
                            profile_count("graph_rule.unknown");
                            log().d("graph_rule_app skipped a match in {}: {}", fn, candidate.reason);
                            if (arg_flag(args(), "report-unknown")) {
                                auto report = log();
                                report.set(fe::Log::Level::Warn)
                                    .w("graph_rule_app skipped a match in {}: {}", fn, candidate.reason);
                            }
                        }
                        return false;
                    }
                    std::array plans{candidate.plan.get()};
                    std::optional<pattern::GraphRewrite> rebuilt;
                    try {
                        rebuilt.emplace(pattern::rewrite_graph(view, plans, rewrite_options_));
                    } catch (const pattern::RebuildLimit& error) {
                        reason = error.what();
                        log().d("graph_rule_app skipped a match in {}: {}", fn, reason);
                        profile_count("graph_rule.budget");
                        return false;
                    } catch (const pattern::PurityFailure& error) {
                        if (!skip_unproved_) throw;
                        reason = error.what();
                        log().d("graph_rule_app skipped unproved replacement in {}: {}", fn, reason);
                        if (arg_flag(args(), "report-unknown")) {
                            auto report = log();
                            report.set(fe::Log::Level::Warn)
                                .w("graph_rule_app skipped unproved replacement in {}: {}", fn, reason);
                        }
                        profile_count("graph_rule.unknown");
                        return false;
                    } catch (const pattern::InstantiationFailure& error) {
                        reason = error.what();
                        log().d("graph_rule_app skipped RHS instantiation in {}: {}", fn, reason);
                        if (reason == "replacement does not preserve the source type"
                            || arg_flag(args(), "report-unknown")) {
                            auto report = log();
                            report.set(fe::Log::Level::Warn)
                                .w("graph_rule_app skipped RHS instantiation in {}: {}", fn, reason);
                        }
                        profile_count("graph_rule.unknown");
                        return false;
                    }
                    auto next = std::move(*rebuilt);
                    if (!next.is_changed()) {
                        profile_count("graph_rule.noop");
                        reason = "canonical no-op";
                        return false;
                    }
                    accepted.emplace(std::move(next));
                    reason = "first accepted changed rewrite";
                    return true;
                };
                auto matches = compiled.find(view, options);
                log().d("graph_rule_app search in {}: {} ({})", fn, search_status(matches.status), reason);
                if (accepted) {
                    selected[name] = accepted->scope();
                    proposals.push_back(std::move(*accepted));
                    profile_count("graph_rule.rewrites");
                    changed = true;
                    break; // The next invocation rebuilds matches and processes further groups.
                }
                if (matches.status != pattern::SearchStatus::Complete) profile_count("graph_rule.incomplete");
            }
            if (changed) break;
        }
    }
    if (!proposals.empty()) {
        if (!facts.is_current()) throw std::invalid_argument("graph Rule source changed before publication");
        auto target = world().inherit();
        FunctionPublisher publisher(*target, selected);
        try {
            for (auto ext : world().externals().muts()) {
                auto copy = publisher.rewrite(ext)->isa_mut();
                if (!copy) throw std::invalid_argument("graph Rule publication lost an external mutable");
                if (!copy->is_external()) copy->externalize();
            }
        } catch (const pattern::RebuildLimit& error) {
            log().d("graph_rule_app skipped publication: {}", error.what());
            profile_count("graph_rule.budget");
            return;
        }
        target->verify();
        swap(world(), *target);
        invalidate();
    }
}

} // namespace mim
