#include "mim/plug/gpu/phase/check_sync.h"

#include <bit>

#include <mim/check.h>
#include <mim/nom.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/buffer/buffer.h>
#include <mim/plug/core/core.h>
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

bool is_slot(const Def* type) {
    return Axm::isa<gpu::Held>(type) || Axm::isa<gpu::Vacant>(type) || Axm::isa<gpu::Ring>(type);
}

bool has_slot(const Def* type) {
    if (is_slot(type)) return true;
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return has_slot(op); });
    if (auto arr = type->isa<Arr>()) return has_slot(arr->body());
    return false;
}

/// Whether @p type mentions a slot token anywhere, also inside a function type, where a conversion could hide.
bool mentions_slot(const Def* type) {
    if (has_slot(type)) return true;
    if (auto pi = type->isa<Pi>()) return mentions_slot(pi->dom()) || mentions_slot(pi->codom());
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return mentions_slot(op); });
    if (auto arr = type->isa<Arr>()) return mentions_slot(arr->body());
    return false;
}

/// An unset Hole that @p def reaches through its dependencies, stopping at mutables.
const Def* unresolved(const Def* def) {
    auto todo = DefVec{def};
    auto seen = DefSet{def};
    while (!todo.empty()) {
        auto d = todo.back();
        todo.pop_back();
        if (auto hole = d->isa<Hole>()) {
            if (!Hole::isa_set(hole)) return hole;
            continue;
        }
        if (d->isa_mut()) continue;
        for (auto op : d->deps())
            if (op && seen.insert(op).second) todo.push_back(op);
    }
    return nullptr;
}

bool has_fn(const Def* type) {
    if (type->isa<Pi>()) return true;
    if (auto sigma = type->isa<Sigma>())
        return std::ranges::any_of(sigma->ops(), [](const Def* op) { return has_fn(op); });
    if (auto arr = type->isa<Arr>()) return has_fn(arr->body());
    return false;
}

/// What a step holds before its first `await`, which any number of steps may start from.
bool is_idle(const Def* def) {
    if (Axm::isa<gpu::unheld>(def)) return true;
    if (auto tuple = def->isa<Tuple>()) return std::ranges::all_of(tuple->ops(), is_idle);
    return false;
}

bool is_slot_token(const Def* def) { return def->type() && has_slot(def->type()) && !is_idle(def); }

/// A natural number as literal multiples of opaque atoms plus a constant, all modulo @p depth; an `Idx` read as a
/// number is its index. `std::nullopt` if a literal is so large that the sum might wrap around, which would break the
/// congruence.
std::optional<std::pair<DefMap<u64>, u64>> nat_form(const Def* n, u64 depth) {
    auto small = [&](u64 lit) { return std::has_single_bit(depth) || lit < (u64(1) << 32); };
    if (auto lit = Lit::isa(n)) {
        if (!small(*lit)) return {};
        return std::pair{DefMap<u64>{}, *lit % depth};
    }
    if (auto add = Axm::isa(core::nat::add, n)) {
        auto lhs = nat_form(add->arg(2, 0), depth), rhs = nat_form(add->arg(2, 1), depth);
        if (!lhs || !rhs) return {};
        for (auto [atom, k] : rhs->first)
            lhs->first[atom] = (lhs->first[atom] + k) % depth;
        lhs->second = (lhs->second + rhs->second) % depth;
        return lhs;
    }
    if (auto mul = Axm::isa(core::nat::mul, n))
        for (size_t i = 0; i != 2; ++i)
            if (auto k = Lit::isa(mul->arg(2, i))) {
                auto form = small(*k) ? nat_form(mul->arg(2, 1 - i), depth) : std::nullopt;
                if (!form) return {};
                for (auto& [_, c] : form->first)
                    c = c * (*k % depth) % depth;
                form->second = form->second * (*k % depth) % depth;
                return form;
            }
    if (auto conv = Axm::isa<core::idx2nat>(n)) return std::pair{DefMap<u64>{{conv->arg(), 1 % depth}}, u64(0)};
    return std::pair{DefMap<u64>{{n, 1 % depth}}, u64(0)};
}

/// Whether slot index @p t of a ring of @p depth slots is @p s: alike, or `idx (x % depth)` where `x` is `s` up to
/// multiples of `depth`, assuming no atom of `x` is large enough to wrap it around.
bool same_slot(const Def* s, const Def* t, u64 depth) {
    if (s == t) return true;
    auto idx = Axm::isa<core::idx>(t);
    if (!idx) return false;
    auto rem = Axm::isa(core::nat::rem, idx->arg());
    if (!rem || Lit::isa(rem->arg(2, 1)) != depth) return false;
    auto form = nat_form(rem->arg(2, 0), depth);
    if (!form) return false;
    auto& [atoms, constant] = *form;
    if (auto lit = Lit::isa(s))
        constant = (constant + depth - *lit % depth) % depth;
    else if (auto i = atoms.find(s); i != atoms.end())
        i->second = (i->second + depth - 1) % depth;
    else
        return false;
    return constant == 0 && std::ranges::all_of(atoms, [](auto atom) { return atom.second == 0; });
}

/// Checks that @p value fits @p want rigidly wherever @p want holds a slot token; a dependent Sigma is instantiated
/// with @p value's own components first.
void rigid(const Def* want, const Def* value, const Def* at) {
    if (!mentions_slot(want)) return;
    if (auto arr = want->isa<Arr>())
        if (auto n = Lit::isa(arr->arity()); n && Lit::isa(value->type()->arity()) == n && *n < 64) {
            for (size_t i = 0; i != *n; ++i)
                rigid(want->proj(*n, i), value->proj(*n, i), at);
            return;
        }
    if (auto sigma = want->isa<Sigma>())
        if (auto n = Lit::isa(sigma->arity()); n && Lit::isa(value->type()->arity()) == n) {
            auto parts = DefVec(sigma->ops().begin(), sigma->ops().end());
            if (sigma->isa_mut()) {
                auto reduced = sigma->reduce(value);
                parts        = DefVec(reduced.begin(), reduced.end());
            }
            for (size_t i = 0; i != *n; ++i)
                rigid(parts[i], value->proj(*n, i), at);
            return;
        }
    if (!Check::test_alpha(want, value->type()))
        at->blame("a ring slot token of type `{}` is passed where `{}` is expected", value->type(), want).bail();
}

} // namespace

// A token is consumed where it is made, never inside a mutable nested there, which may run more than once.
// A slot token may be used once in each case of a selection among mutables; other divergent use is not checked.
void CheckSync::start() {
    DefMap<DefVec> users;
    DefSet counted, linear;
    DefVec slots, branches;
    fe::Vector<const App*> reads, calls;
    fe::Vector<Lam*> bodies;
    DefMap<fe::Vector<Lam*>> returns;
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
        if (auto lam = mut->isa<Lam>(); lam && lam->is_set()) {
            returns[lam->body()].push_back(lam);
            if (mentions_slot(lam->codom())) bodies.push_back(lam);
        }
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
            if (auto read = Axm::isa<gpu::read_slot>(def)) reads.push_back(read);
            if (auto freed = Axm::isa<gpu::retired>(def)) {
                auto [k, d] = freed->decurry()->args<2>([](const Def* n) { return Lit::isa(n); });
                if (!k || !d) def->blame("`{}` gives a slot up on a count of MMAs it cannot compute", def).bail();
                if (*k > *d)
                    def->blame("a slot is given up {} steps after its read while {} groups of MMAs may be in flight",
                               *d, *k)
                        .bail();
            }
            if (auto app = def->isa<App>())
                if (auto pi = app->callee()->type()->isa<Pi>(); pi && mentions_slot(pi->dom())) calls.push_back(app);
            // A hole or bottom that would stand for a slot token, also one a function would return, comes from no sync.
            if (def->type() && mentions_slot(def->type())
                && (def->isa<Bot>() || (def->isa<Hole>() && !Hole::isa_set(def))))
                def->blame("`{}` stands for a ring slot token that no ring sync made", def).bail();
            if (is_slot_token(def))
                if (auto hole = unresolved(def->type()))
                    def->blame("the type of ring slot token `{}` has the unresolved index `{}`", def, hole).bail();
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
    // The calls a function value reaches through tuples, literal projections, returns and the parameters of functions
    // called once; `std::nullopt` if it may escape anywhere else.
    auto calls_of = [&](const Def* fn) -> std::optional<DefVec> {
        auto sites = DefVec();
        auto todo  = DefVec{fn};
        auto done  = DefSet{fn};
        // The positions a tuple holds the function at, so its other components are not taken for it.
        auto at   = DefMap<fe::Vector<u64>>();
        auto push = [&](const Def* def) {
            if (done.insert(def).second) todo.push_back(def);
        };
        while (!todo.empty()) {
            auto value = todo.back();
            todo.pop_back();
            if (auto i = returns.find(value); i != returns.end())
                for (auto lam : i->second)
                    for (auto user : users_of(lam)) {
                        auto app = user->isa<App>();
                        if (!app || app->callee() != lam) return {};
                        if (auto i = at.find(value); i != at.end()) {
                            auto positions = i->second;
                            at[app]        = std::move(positions);
                        }
                        push(app);
                    }
            for (auto user : users_of(value)) {
                if (auto app = user->isa<App>()) {
                    if (app->callee() == value) {
                        // A call that yields another function only hands the token on to it.
                        if (app->type()->isa<Pi>())
                            push(app);
                        else
                            sites.push_back(app);
                        continue;
                    }
                    if (Axm::isa<cps::cps2ds_dep>(app) && app->arg() == value) {
                        push(app);
                        continue;
                    }
                    auto lam = app->callee()->isa_mut<Lam>();
                    if (!lam || !called_once(lam)) return {};
                    auto n = Lit::isa(app->arg()->arity()).value_or(0);
                    if (app->arg() == value) {
                        if (auto i = at.find(value); i != at.end()) {
                            auto positions = i->second;
                            at[lam->var()] = std::move(positions);
                        }
                        push(lam->var());
                    } else if (auto tuple = app->arg()->isa<Tuple>(); tuple && n) {
                        for (size_t j = 0; j != n; ++j)
                            if (tuple->op(j) == value) push(lam->var(n, j));
                    } else {
                        return {};
                    }
                } else if (auto part = user->isa<Extract>()) {
                    auto index = Lit::isa(part->index());
                    if (!index) return {};
                    if (auto i = at.find(value);
                        i != at.end() ? std::ranges::contains(i->second, *index) : has_fn(part->type()))
                        push(part);
                } else if (auto tuple = user->isa<Tuple>()) {
                    for (size_t j = 0, n = tuple->num_ops(); j != n; ++j)
                        if (tuple->op(j) == value) at[tuple].push_back(j);
                    push(tuple);
                } else {
                    return {};
                }
            }
        }
        return sites;
    };
    // The exit of an `affine.For`, which runs once per run of the loop.
    auto loop_exit = [&](const App* app, const Def* mut) {
        auto body = app->callee()->isa<App>();
        auto loop = body ? body->callee()->isa<App>() : nullptr;
        return app->arg() == mut && loop && loop->callee() == world().annex<affine::For>() && called_once(app);
    };
    // A mutable runs at most once if each use calls it once, directly, through `cps2ds`, as a case of a selection or
    // as a loop's exit, and no two uses run together.
    auto repeats_here = [&](Def* mut) {
        auto by = users_of(mut);
        return !pairwise(by) || !std::ranges::all_of(by, [&](const Def* user) {
            if (inside(user, mut)) return false;
            if (auto app = user->isa<App>())
                return app->callee() == mut || (Axm::isa<cps::cps2ds_dep>(app) && called_once(app))
                    || loop_exit(app, mut);
            if (!user->isa<Tuple>() || !std::ranges::all_of(user->ops(), [](const Def* op) { return op->isa_mut(); }))
                return false;
            auto picks = users_of(user);
            return picks.size() == 1 && picks.front()->isa<Extract>() && called_once(picks.front());
        });
    };
    // A call runs once per run of @p home if no mutable between them may repeat.
    auto once_within = [&](const Def* site, Def* home) {
        for (auto m = placed(site); m && m != home && outer(m) != home; m = fe::lookup(parent, m))
            if (repeats_here(m)) return false;
        return true;
    };
    // As above, or as a value that only one call reaches, which itself runs once per run of the mutable around.
    auto repeats = [&](Def* mut) {
        auto home = fe::lookup(parent, mut);
        if (auto sites = calls_of(mut); sites && pairwise(*sites) && std::ranges::all_of(*sites, [&](const Def* site) {
                                            return !inside(site, mut) && once_within(site, home);
                                        }))
            return false;
        return repeats_here(mut);
    };
    // The core check lets two different free variables stand for each other, so slot tokens are compared rigidly.
    for (auto app : calls)
        rigid(app->callee()->type()->as<Pi>()->dom(), app->arg(), app);
    for (auto lam : bodies)
        rigid(lam->codom(), lam->body(), lam);
    // A partial application that takes a slot token is a closure holding it, which runs at most once.
    for (auto app : calls)
        if (app->type()->isa<Pi>() && has_slot(app->arg()->type()))
            if (auto sites = calls_of(app);
                !sites || !pairwise(*sites)
                || !std::ranges::all_of(*sites, [&](const Def* site) { return once_within(site, placed(app)); }))
                app->blame("`{}` holds a ring slot token and may be called more than once", app).bail();
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
    for (auto read : reads) {
        auto [ring, s, t, token] = read->args<4>();
        auto depth               = Lit::isa(read->decurry()->arg(3, 2));
        if (!depth || !same_slot(s, t, *depth))
            read->blame("ring slot `{}` is read with the token of slot `{}`", s, t).bail();
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
