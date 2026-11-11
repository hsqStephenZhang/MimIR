#include "mim/plug/gpu/phase/check_sync.h"

#include <mim/check.h>
#include <mim/nom.h>

#include <mim/plug/mem/mem.h>

namespace mim::plug::gpu::phase {

namespace {

const Def* forward(const Def* def) {
    while (auto op = Hole::isa_set(def))
        def = op;
    return def;
}

bool has_token(const Def* type) {
    if (Axm::isa<gpu::GM>(type)) return true;
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return has_token(op); });
    return false;
}

std::optional<nat_t> visibility(const Def* def) {
    if (auto buf = Axm::isa<gpu::SharedBuf>(def->type()))
        if (auto name = buf->arg(6, 2)->isa<Name>())
            if (auto inj = name->op()->isa<Inj>()) return inj->index();
    return {};
}

bool has_panel(const Def* type) {
    if (Axm::isa<gpu::SharedBuf>(type)) return true;
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return has_panel(op); });
    return false;
}

} // namespace

// A token is consumed where it is made, never inside a mutable nested there, which may run more than once.
// Divergent use is not checked.
void CheckSync::start() {
    DefMap<DefVec> users;
    DefSet counted, linear;
    MutMap<Def*> parent, inlined;
    ankerl::unordered_dense::map<nat_t, const Def*> entered;
    // A `mem.fresh` continuation runs exactly once in place, so it belongs to the mutable that requests it.
    auto outer = [&](Def* mut) {
        while (auto at = fe::lookup(inlined, mut))
            mut = at;
        return mut;
    };
    auto depth = [&](Def* mut) {
        size_t n = 0;
        for (; mut; mut = fe::lookup(parent, mut))
            ++n;
        return n;
    };
    auto muts = fe::Vector<Def*>(world().externals().muts().begin(), world().externals().muts().end());
    for (MutSet done; !muts.empty();) {
        auto mut = muts.back();
        muts.pop_back();
        if (!done.insert(mut).second) continue;
        if (auto lam = mut->isa<Lam>(); lam && (has_token(lam->dom()) || has_panel(lam->dom())))
            lam->blame("a work-group phase token or panel must not cross into `{}`", lam).bail();
        DefSet seen;
        auto stack = DefVec();
        for (auto op : mut->ops())
            if (op) stack.push_back(op);
        while (!stack.empty()) {
            auto def = stack.back();
            stack.pop_back();
            if (auto m = def->isa_mut(); m && !def->isa<Hole>()) {
                if (m != mut) parent.emplace(m, mut);
                muts.push_back(m);
                continue;
            }
            if (!seen.insert(def).second) continue;
            // Universes and levels have no type.
            if (def->type() && (has_token(def->type()) || visibility(def))) linear.insert(def);
            if (auto fresh = Axm::isa<mem::fresh>(def))
                if (auto k = fresh->arg()->proj(2, 1)->isa_mut()) inlined.emplace(k, mut);
            if (auto enter = Axm::isa<gpu::group_enter>(def)) {
                auto g = Lit::as(enter->decurry()->arg());
                if (auto [at, fresh] = entered.emplace(g, def); !fresh && at->second != def)
                    def->blame("work-group construct {} is entered twice", g).bail();
            }
            auto first = counted.insert(def).second;
            for (auto op : def->ops()) {
                if (!op) continue;
                if (first && !def->isa<Hole>()) users[forward(op)].push_back(def);
                stack.push_back(op);
            }
        }
    }

    // The innermost mutable whose variable @p def depends on.
    auto placed = [&](const Def* def) {
        Def* inner = nullptr;
        for (auto var : def->local_vars())
            if (auto at = outer(var->binder()); !inner || depth(at) > depth(inner)) inner = at;
        return inner;
    };
    auto nested = [&](Def* mut, Def* in) {
        for (mut = fe::lookup(parent, mut); mut; mut = fe::lookup(parent, mut))
            if (outer(mut) == in) return true;
        return false;
    };
    auto users_of = [&](const Def* def) {
        auto i = users.find(def);
        return i == users.end() ? DefVec() : i->second;
    };
    auto witness = [&](const Def* user) {
        if (!user->isa<Tuple>()) return false;
        auto by = users_of(user);
        return !by.empty() && std::ranges::all_of(by, [](const Def* u) {
            return Axm::isa<gpu::publish_view>(u) || Axm::isa<gpu::claim>(u);
        });
    };
    // Tokens first, so a duplicated token is reported before the panels it duplicates.
    auto order = DefVec(linear.begin(), linear.end());
    std::ranges::stable_partition(order, [](const Def* def) { return Axm::isa<gpu::GM>(def->type()); });
    for (auto def : order) {
        if (def->isa<Hole>()) continue;
        auto by = users_of(def);
        if (Axm::isa<gpu::GM>(def->type())) {
            auto consumers = DefVec();
            std::ranges::copy_if(by, std::back_inserter(consumers), [&](const Def* u) { return !witness(u); });
            if (consumers.size() != 1)
                def->blame("work-group phase token `{}` is consumed {} times", def, consumers.size()).bail();
            if (auto home = placed(def))
                if (auto at = placed(consumers.front()); at && at != home && nested(at, home))
                    def->blame("work-group phase token `{}` is made in `{}` but consumed inside `{}`", def, home, at)
                        .bail();
        } else if (has_token(def->type())) {
            if (by.size() > 1 && !std::ranges::all_of(by, [](const Def* u) { return u->isa<Extract>(); }))
                def->blame("`{}` holds a work-group phase token and is used {} times", def, by.size()).bail();
        } else if (auto vis = visibility(def); *vis != gpu::Vis_ctor::Published) {
            if (by.size() != 1) def->blame("panel `{}` is used {} times", def, by.size()).bail();
        } else if (by.size() > 1 && std::ranges::any_of(by, [&](const Def* u) {
                       return std::ranges::any_of(users_of(u), [](const Def* c) { return Axm::isa<gpu::claim>(c); });
                   })) {
            def->blame("published panel `{}` is claimed and used elsewhere in its superstep", def).bail();
        }
    }
}

} // namespace mim::plug::gpu::phase
