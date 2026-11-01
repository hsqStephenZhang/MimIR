#include "mim/phase/rule_rewrite.h"

#include <stdexcept>

#include "mim/driver.h"

namespace mim {

RuleRewrite::RuleRewrite(World& world, std::vector<const Rule*> rules)
    : RuleRewrite(world) {
    set_rules(std::move(rules));
}

void RuleRewrite::set_rules(std::vector<const Rule*> rules) {
    for (auto rule : rules)
        if (!rule || !rule->is_set() || &rule->driver() != &driver() || !rule->is_closed())
            throw std::invalid_argument("rule_app requires fully defined, closed Rules of the same Driver");
    for (auto rule : rules)
        pattern::RulePattern(rule).check_definition();
    template_world_ = old_world().inherit();
    Rewriter importer(*template_world_);
    templates_.clear();
    for (auto rule : rules)
        templates_.push_back(importer.rewrite(rule)->as<Rule>());
}

void RuleRewrite::apply(const App* app) {
    if (!app) throw std::invalid_argument("rule_app requires explicit Rules");
    const Def* supplied = app->arg();
    schemas_.clear();
    if (App::uncurry_callee(app)->sym() == driver().sym("compile.rule_app_with")) {
        auto pair = supplied->isa<Tuple>();
        if (!pair || pair->num_ops() != 2) supplied->blame("rule_app_with requires (Rules, data_ops)").bail();
        supplied = pair->op(0);
        schemas_ = pattern::parse_data_ops(pair->op(1));
    }
    auto rules   = std::vector<const Rule*>();
    auto collect = [&](this auto&& collect, const Def* def) -> void {
        if (auto rule = def->isa<Rule>())
            rules.push_back(rule);
        else if (auto tuple = def->isa<Tuple>())
            for (auto op : tuple->ops())
                collect(op);
        else if (auto pack = def->isa_imm<Pack>()) {
            auto count = Lit::isa(*pack->shape());
            if (!count) def->blame("rule_app expects a statically known tuple of Rules").bail();
            if (*count) collect(pack->body()); // Repeating the same rule does not change ordered selection.
        } else
            def->blame("rule_app expects a Rule or a tuple of Rules").bail();
    };
    collect(supplied);
    set_rules(std::move(rules));
}

std::unique_ptr<Phase> RuleRewrite::recreate() {
    auto result             = std::make_unique<RuleRewrite>(old_world());
    result->name_           = name_;
    result->template_world_ = template_world_;
    result->templates_      = templates_;
    result->schemas_        = schemas_;
    return result;
}

void RuleRewrite::start() {
    if (templates_.empty()) return;
    patterns_.clear();
    Rewriter importer(old_world());
    for (auto rule : templates_)
        patterns_.push_back(std::make_unique<pattern::RulePattern>(importer.rewrite(rule)->as<Rule>()));
    auto ops    = pattern::data_operations(schemas_);
    auto bodies = pattern::function_bodies(old_world());
    if (!bodies.empty()) {
        facts_ = std::make_unique<pattern::GraphView>(
            bodies, pattern::GraphOptions{.coverage = pattern::UseCoverage::Closed, .operators = &ops});
        guards_ = std::make_unique<pattern::GuardContext>(*facts_);
    }
    RWPhase::start();
    guards_.reset();
    facts_.reset();
    patterns_.clear();
}

const Def* RuleRewrite::rewrite_imm(const Def* old_def) {
    // Install the ordinary rebuilt value first. Capturing the whole candidate can then import it
    // through this rewriter without recursively attempting the same rule again.
    auto baseline = Rewriter::rewrite_imm(old_def);
    if (old_def->node() == Node::Rule) return baseline;
    for (auto& compiled : patterns_) {
        auto result = compiled->build_replacement(old_def, *this, {}, guards_.get());
        if (result.status == pattern::MatchStatus::Budget_Exceeded)
            old_def->blame("rule_app exceeded its pattern matching budget").bail();
        if (result.status == pattern::MatchStatus::Unknown) {
            profile_count("rule.unknown");
            log().d("rule_app skipped {}: {}", old_def, result.reason);
            if (result.reason == "RHS does not preserve the matched type" || arg_flag(args(), "report-unknown")) {
                auto report = log();
                report.set(fe::Log::Level::Warn).w("rule_app skipped {}: {}", old_def, result.reason);
            }
        }
        if (result.status != pattern::MatchStatus::Matched) continue;
        profile_count("rule.matches");
        if (result.value != baseline) {
            profile_count("rule.rewrites");
            invalidate();
        }
        return map(old_def, result.value);
    }
    return baseline;
}

const Def* RuleRewrite::rewrite_imm_Rule(const Rule* rule) {
    Rewriter importer(new_world());
    return importer.rewrite(rule);
}

const Def* RuleRewrite::rewrite_mut_Rule(Rule* rule) { return map(rule, rewrite_imm_Rule(rule)); }

} // namespace mim
