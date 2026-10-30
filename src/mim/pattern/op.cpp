#include "mim/pattern/op.h"

#include <stdexcept>

#include "mim/axm.h"
#include "mim/lam.h"
#include "mim/tuple.h"

namespace mim::pattern {

void OpRegistry::add(flags_t axm, size_t num_groups, std::vector<InputGroup> inputs, bool pure) {
    auto seen = std::vector<bool>(num_groups, false);
    for (auto input : inputs) {
        if (input.index >= num_groups || seen[input.index] || input.tuple_depth > 64)
            throw std::invalid_argument("invalid or duplicate operator argument group");
        seen[input.index] = true;
    }
    add(
        axm,
        [num_groups, inputs = std::move(inputs)](const Def* def, size_t limit) -> std::optional<OpView> {
            auto args = std::vector<const Def*>();
            for (auto cur = def; cur->isa<App>(); cur = cur->as<App>()->callee()) {
                args.push_back(cur->as<App>()->arg());
                if (args.size() > num_groups) return {};
            }
            OpView view;
            if (args.size() != num_groups) return {};
            std::ranges::reverse(args);
            auto expand = [&](this auto&& expand, const Def* value, size_t depth, size_t group,
                              std::vector<size_t> path) -> void {
                if (view.inputs.size() >= limit) {
                    view.complete = false;
                    return;
                }
                if (!depth) {
                    view.inputs.push_back({value, group, std::move(path)});
                    return;
                }
                if (auto tuple = value->isa<Tuple>()) {
                    for (size_t i = 0; i != tuple->num_ops(); ++i) {
                        if (view.inputs.size() >= limit) {
                            view.complete = false;
                            break;
                        }
                        auto next = path;
                        next.push_back(i);
                        expand(tuple->op(i), depth - 1, group, std::move(next));
                    }
                } else if (auto pack = value->isa_imm<Pack>()) {
                    auto count = Lit::isa(*pack->shape());
                    if (!count || *count > limit - view.inputs.size()) {
                        view.complete = false;
                        return;
                    }
                    for (size_t i = 0; i != *count; ++i) {
                        auto next = path;
                        next.push_back(i);
                        expand(pack->body(), depth - 1, group, std::move(next));
                    }
                } else {
                    view.complete = false;
                }
            };
            for (auto input : inputs)
                expand(args[input.index], input.tuple_depth, input.index, {});
            return view;
        },
        pure);
}

void OpRegistry::add(flags_t axm, Decoder decode, bool pure) {
    if (!decode || !specs_.emplace(axm, Spec{std::move(decode), pure}).second)
        throw std::invalid_argument("duplicate operator or missing decoder");
}

void OpRegistry::add(flags_t axm, Decoder decode, Purity purity) {
    if (!decode || !purity || !specs_.emplace(axm, Spec{std::move(decode), false, std::move(purity)}).second)
        throw std::invalid_argument("duplicate operator or missing decoder/purity proof");
}

std::optional<OpView> OpRegistry::decode(const Def* def, size_t limit) const {
    if (!def || def->node() != Node::App) return {};
    auto [axm, curry, trip] = Axm::get(def);
    if (!axm) return {};
    auto i = specs_.find(axm->flags());
    return i == specs_.end() ? std::optional<OpView>() : i->second.decode(def, limit);
}

bool OpRegistry::is_pure(flags_t axm) const {
    auto i = specs_.find(axm);
    return i != specs_.end() && i->second.pure;
}

bool OpRegistry::has_purity(flags_t axm) const {
    auto i = specs_.find(axm);
    return i != specs_.end() && (i->second.pure || bool(i->second.purity));
}

bool OpRegistry::is_pure_call(flags_t axm, std::span<const Def* const> args) const {
    auto i = specs_.find(axm);
    return i != specs_.end() && (i->second.pure || (i->second.purity && i->second.purity(args)));
}

bool OpRegistry::is_pure(const Def* def) const {
    if (!def) return false;
    auto [axm, curry, trip] = Axm::get(def);
    if (!axm) return false;
    std::vector<const Def*> args;
    for (auto cur = def; auto app = cur->isa<App>(); cur = app->callee())
        args.push_back(app->arg());
    std::ranges::reverse(args);
    return is_pure_call(axm->flags(), args);
}

} // namespace mim::pattern
