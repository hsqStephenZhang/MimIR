#include "mim/plug/gpu/phase/check_sync.h"

#include <mim/check.h>
#include <mim/nom.h>

#include <mim/plug/buffer/buffer.h>
#include <mim/plug/cps/cps.h>
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

bool is_slot(const Def* type) { return Axm::isa<gpu::Held>(type) || Axm::isa<gpu::Vacant>(type); }

bool has_slot(const Def* type) {
    if (is_slot(type)) return true;
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return has_slot(op); });
    if (auto arr = type->isa<Arr>()) return has_slot(arr->body());
    return false;
}

bool is_slot_token(const Def* def) { return def->type() && has_slot(def->type()) && !Axm::isa<gpu::unheld>(def); }

} // namespace

// A token is consumed where it is made, never inside a mutable nested there, which may run more than once.
// A slot token may be used once in each case of a selection among mutables; other divergent use is not checked.
void CheckSync::start() {
    DefMap<DefVec> users;
    DefSet counted, linear;
    DefVec slots, branches;
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
            if (is_slot_token(def)) slots.push_back(def);
            if (auto extract = def->isa<Extract>();
                extract && extract->tuple()->isa<Tuple>() && !Lit::isa(extract->index()))
                branches.push_back(extract->tuple());
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
    auto inside = [&](const Def* def, Def* mut) {
        for (auto at = placed(def); at; at = fe::lookup(parent, at))
            if (at == mut || outer(at) == mut) return true;
        return false;
    };
    // Two uses in different cases of one selection among mutables never both run.
    auto exclusive = [&](const Def* u, const Def* v) {
        for (auto tuple : branches) {
            if (!std::ranges::all_of(tuple->ops(), [](const Def* op) { return op->isa_mut(); })) continue;
            auto case_of = [&](const Def* d) {
                std::optional<size_t> found;
                for (size_t i = 0, n = tuple->num_ops(); i != n; ++i)
                    if (inside(d, tuple->op(i)->isa_mut())) {
                        if (found) return std::optional<size_t>();
                        found = i;
                    }
                return found;
            };
            if (auto i = case_of(u), j = case_of(v); i && j && *i != *j) return true;
        }
        return false;
    };
    auto pairwise = [&](const DefVec& uses) {
        for (size_t i = 0; i != uses.size(); ++i)
            for (size_t j = 0; j != i; ++j)
                if (!exclusive(uses[i], uses[j])) return false;
        return true;
    };
    auto called_once = [&](const Def* callee) {
        auto by = users_of(callee);
        return by.size() == 1 && by.front()->isa<App>() && by.front()->as<App>()->callee() == callee;
    };
    // A mutable runs at most once if each use calls it once, directly, through `cps2ds` or as a case of a selection,
    // and no two uses run together.
    auto repeats = [&](Def* mut) {
        auto by = users_of(mut);
        return !pairwise(by) || !std::ranges::all_of(by, [&](const Def* user) {
            if (inside(user, mut)) return false;
            if (auto app = user->isa<App>())
                return app->callee() == mut || (Axm::isa<cps::cps2ds_dep>(app) && called_once(app));
            if (!user->isa<Tuple>() || !std::ranges::all_of(user->ops(), [](const Def* op) { return op->isa_mut(); }))
                return false;
            auto picks = users_of(user);
            return picks.size() == 1 && picks.front()->isa<Extract>() && called_once(picks.front());
        });
    };
    // A ring's last slot may stay held when its fold ends, so a slot token is used at most once.
    for (auto def : slots) {
        if (def->isa<Hole>()) continue;
        if (def->isa<Pack>()) def->blame("ring slot token `{}` is packed into copies", def).bail();
        auto by = users_of(def);
        if (!is_slot(def->type())) {
            // A value that holds a slot token is taken apart by literal indices or handed on whole, never both where
            // both run.
            DefVec whole, consumers;
            for (auto u : by) {
                auto part = u->isa<Extract>();
                if (!part) {
                    whole.push_back(u);
                    continue;
                }
                if (!Lit::isa(part->index()))
                    u->blame("a ring slot token is selected from `{}` by a computed index", def).bail();
                if (has_slot(part->type())) std::ranges::copy(users_of(part), std::back_inserter(consumers));
            }
            for (auto u : whole)
                for (auto c : consumers)
                    if (!exclusive(u, c))
                        def->blame("`{}` holds a ring slot token and is both taken apart and used whole", def).bail();
            by = whole;
        }
        if (!pairwise(by)) {
            if (is_slot(def->type())) def->blame("ring slot token `{}` is used {} times", def, by.size()).bail();
            def->blame("`{}` holds a ring slot token and is used {} times", def, by.size()).bail();
        }
        if (auto home = placed(def))
            for (auto user : by)
                if (auto at = placed(user); at && at != home && nested(at, home))
                    for (auto m = at; m && outer(m) != home; m = fe::lookup(parent, m))
                        if (repeats(m))
                            def->blame("ring slot token `{}` is made in `{}` but used inside `{}`, which may repeat",
                                       def, home, m)
                                .bail();
    }
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
        } else if (auto vis = visibility(def); *vis != buffer::Vis_ctor::Published) {
            if (by.size() != 1) def->blame("panel `{}` is used {} times", def, by.size()).bail();
        } else if (by.size() > 1 && std::ranges::any_of(by, [&](const Def* u) {
                       return std::ranges::any_of(users_of(u), [](const Def* c) { return Axm::isa<gpu::claim>(c); });
                   })) {
            def->blame("published panel `{}` is claimed and used elsewhere in its superstep", def).bail();
        }
    }
}

} // namespace mim::plug::gpu::phase
