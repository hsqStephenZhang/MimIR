#include "mim/pattern/guard.h"

#include <stdexcept>

#include <fe/restore.h>

#include "mim/driver.h"
#include "mim/rewrite.h"
#include "mim/rule.h"

namespace mim::pattern {

namespace {

const Def* slot(const Def* def, size_t count, size_t index) {
    if (count == 1) return def;
    if (auto tuple = def->isa<Tuple>(); tuple && tuple->num_ops() == count) return tuple->op(index);
    if (auto pack = def->isa_imm<Pack>(); pack && Lit::isa(*pack->shape()) == count) return pack->body();
    return nullptr;
}

class DataOps {
public:
    std::vector<DataOp> result;

    void collect(const Def* def) {
        if (++work_ > 10000 || depth_ >= 256) def->blame("data schema list exceeds its traversal budget").bail();
        auto depth = fe::Restore(depth_, depth_ + 1);
        if (auto tuple = def->isa<Tuple>()) {
            for (auto op : tuple->ops())
                collect(op);
            return;
        }
        if (auto pack = def->isa_imm<Pack>()) {
            auto count = Lit::isa(*pack->shape());
            if (!count) def->blame("data schemas require a static list").bail();
            if (*count) collect(pack->body());
            return;
        }
        auto app = def->isa<App>();
        auto axm = app ? App::uncurry_callee(app)->isa<Axm>() : nullptr;
        if (!axm || axm->sym() != def->driver().sym("compile.data_op"))
            def->blame("expected compile.data_op declarations").bail();
        auto name_def   = slot(app->arg(), 3, 0);
        auto groups_def = slot(app->arg(), 3, 1);
        auto inputs_def = slot(app->arg(), 3, 2);
        if (!name_def || !groups_def || !inputs_def) def->blame("invalid data_op declaration").bail();
        auto name  = tuple2str(name_def);
        auto named = name.empty() ? nullptr : def->world().annex(def->driver().sym(name));
        auto op    = name_def->isa<Axm>();
        if (!op && named) op = named->isa<Axm>();
        if (!op) def->blame("data_op reference must name an axiom: {}", name).bail();
        auto groups    = Lit::isa(groups_def);
        auto implicit  = app->callee()->isa<App>();
        auto count_def = implicit ? slot(implicit->arg(), 2, 1) : nullptr;
        auto count     = count_def ? Lit::isa(count_def) : std::nullopt;
        if (!groups || !*groups || *groups > 64 || !count || *count > 64)
            def->blame("data_op requires at most 64 static groups and inputs").bail();
        size_t actual = 0;
        for (auto type = op->type(); auto pi = type->isa<Pi>(); type = pi->codom())
            if (++actual > 64) def->blame("data_op axiom curry depth exceeds 64").bail();
        if (actual != *groups) def->blame("data_op group count does not match the axiom").bail();
        DataOp schema{op->flags(), size_t(*groups), {}};
        for (size_t i = 0; i != *count; ++i) {
            auto pair      = slot(inputs_def, *count, i);
            auto group_def = pair ? slot(pair, 2, 0) : nullptr;
            auto depth_def = pair ? slot(pair, 2, 1) : nullptr;
            auto group     = group_def ? Lit::isa(group_def) : std::nullopt;
            auto depth     = depth_def ? Lit::isa(depth_def) : std::nullopt;
            if (!group || !depth || *group >= *groups || *depth > 64)
                def->blame("invalid data_op input group or tuple depth").bail();
            for (auto input : schema.inputs)
                if (input.index == *group) def->blame("duplicate data_op input group").bail();
            schema.inputs.push_back({size_t(*group), size_t(*depth)});
        }
        for (const auto& prior : result)
            if (prior.axm == schema.axm) def->blame("duplicate data_op declaration").bail();
        result.push_back(std::move(schema));
    }

private:
    size_t work_ = 0, depth_ = 0;
};

class GuardImporter : public Rewriter {
public:
    GuardImporter(World& target, const Def2Def& frozen, const GuardContext* context)
        : Rewriter(target)
        , frozen_(frozen)
        , context_(context) {}

    const Def* rewrite(const Def* def) override {
        if (++work_ > 10000 || depth_ >= 256) throw std::invalid_argument("Rule guard evaluation budget exceeded");
        auto depth = fe::Restore(depth_, depth_ + 1);
        if (context_)
            if (auto app = def->isa<App>())
                if (auto axm = App::uncurry_callee(app)->isa<Axm>()) {
                    auto flags = axm->flags();
                    if (flags == context_->used_by || flags == context_->only_used_by) {
                        auto p = source(slot(app->arg(), 2, 0)), c = source(slot(app->arg(), 2, 1));
                        auto result = !p || !c                   ? CheckResult::Unknown
                                    : flags == context_->used_by ? context_->graph.data_used_by(p, c)
                                                                 : context_->graph.data_only_used_by(p, c);
                        if (result != CheckResult::Unknown) return world().lit_bool(result == CheckResult::True);
                    }
                }
        return Rewriter::rewrite(def);
    }

private:
    const Def* source(const Def* def) const {
        if (!def) return nullptr;
        if (auto i = frozen_.find(def); i != frozen_.end()) return i->second;
        return context_->graph.is_contained(def) ? def : nullptr;
    }
    const Def2Def& frozen_;
    const GuardContext* context_;
    size_t work_ = 0, depth_ = 0;
};

} // namespace

std::vector<DataOp> parse_data_ops(const Def* def) {
    DataOps collector;
    collector.collect(def);
    return std::move(collector.result);
}

OpRegistry data_operations(std::span<const DataOp> schemas) {
    OpRegistry result;
    for (const auto& schema : schemas)
        result.add(schema.axm, schema.num_groups, schema.inputs);
    return result;
}

DefVec function_bodies(World& world) {
    DefVec result;
    for (auto ext : world.externals().muts())
        if (auto fn = ext->isa<Lam>(); fn && fn->is_set()) result.push_back(fn->body());
    return result;
}

GuardContext::GuardContext(const GraphView& view)
    : graph(view) {
    auto& world = graph.world();
    if (auto op = world.annex(world.driver().sym("compile.used_by"))) used_by = op->flags();
    if (auto op = world.annex(world.driver().sym("compile.only_used_by"))) only_used_by = op->flags();
}

GuardResult
evaluate_guard(const Rule* rule, const Def2Def& bindings, Defs lhs, Defs matched, const GuardContext* context) {
    auto& world = rule->world();
    if (rule->guard() == world.lit_tt()) return {MatchStatus::Matched, {}};
    if (rule->guard() == world.lit_ff()) return {MatchStatus::No_Match, "guard is false"};
    if (lhs.size() != matched.size()) throw std::invalid_argument("guard source roots do not agree");
    if (context && !context->graph.is_current()) throw std::invalid_argument("guard source snapshot is stale");
    Def2Def frozen = bindings;
    std::vector<std::pair<const Def*, const Def*>> pending;
    for (size_t i = 0; i != lhs.size(); ++i)
        pending.emplace_back(lhs[i], matched[i]);
    for (size_t i = 0; i != pending.size(); ++i) {
        if (i >= 10000) return {MatchStatus::Unknown, "guard source correspondence budget exceeded"};
        auto [term, value] = pending[i];
        if (auto [it, inserted] = frozen.emplace(term, value); !inserted) {
            if (it->second != value) return {MatchStatus::Unknown, "guard source correspondence is ambiguous"};
            continue;
        }
        if (term->node() != value->node() || term->is_mutable() || value->is_mutable()
            || term->num_ops() != value->num_ops())
            continue;
        for (size_t j = 0; j != term->num_ops(); ++j)
            pending.emplace_back(term->op(j), value->op(j));
    }
    auto target = world.inherit();
    GuardImporter importer(*target, frozen, context);
    try {
        for (auto [meta, value] : bindings)
            importer.map(meta, importer.rewrite(value));
        auto condition = importer.rewrite(rule->guard());
        if (condition == target->lit_tt()) return {MatchStatus::Matched, {}};
        if (condition == target->lit_ff()) return {MatchStatus::No_Match, "guard is false"};
        return {MatchStatus::Unknown, "guard is not true"};
    } catch (const std::invalid_argument& error) { return {MatchStatus::Unknown, error.what()}; }
}

} // namespace mim::pattern
