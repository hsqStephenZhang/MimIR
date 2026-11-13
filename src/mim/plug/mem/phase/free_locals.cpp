#include "mim/plug/mem/phase/free_locals.h"

#include <fe/worklist.h>

#include <mim/plug/core/core.h>

namespace mim::plug::mem::phase {

bool FreeLocals::analyze() {
    if (RWPhase::analyze()) return true;

    auto shared = fe::Vector<std::pair<Lam*, const Def*>>();
    for (const auto& [lam, chain] : escape_.chains()) {
        for (size_t i = 0, n = chain.ops.size(); i != n; ++i) {
            auto alloc = chain.ops[i];
            if (!EscapeAnalysis::allocates(alloc) || escape_.is_top(alloc) || escape_.freed(alloc)) continue;
            if (escape_.confined(alloc)) {
                auto last = fe::lookup(chain.last, alloc);
                auto j    = (last ? *last : i) + 1;
                frees_[j < n ? chain.ops[j] : chain.tail].emplace_back(alloc);
            } else {
                shared.emplace_back(lam, alloc);
            }
        }
    }

    for (auto [param, home] : owned_params())
        shared.emplace_back(home, param);

    if (!shared.empty()) {
        auto homes = LamSet();
        for (auto [lam, _] : shared)
            homes.emplace(lam);
        // A home nested in another must hang below it, not below the virtual root.
        auto roots = fe::Vector<Def*>();
        for (auto lam : homes)
            if (std::ranges::none_of(homes, [lam](Lam* home) { return home->nests(lam); })) roots.emplace_back(lam);

        const auto nest = Nest(old_world(), roots);
        mentions(nest.root());
        for (auto [lam, alloc] : shared)
            place(nest[lam], nest[lam], alloc);
    }

    log().d("free_locals: frees before {} ops, on {} edges", frees_.size(), edges_.size());
    return false;
}

const DefSet& FreeLocals::mentions(const Nest::Node* node) {
    auto res = DefSet();
    if (auto lam = node->mut() ? node->mut()->isa_mut<Lam>() : nullptr) {
        if (auto i = escape_.chains().find(lam); i != escape_.chains().end())
            for (auto [alloc, _] : i->second.last)
                res.emplace(alloc);
        for (auto& [param, _] : owned_)
            if (uses_directly(lam, param)) res.emplace(param);
    }
    for (auto child : node->children().nodes()) {
        const auto& sub = mentions(child);
        res.insert(sub.begin(), sub.end());
    }
    return mentions_[node] = std::move(res);
}

const DefSet& FreeLocals::reach(const Nest::Node* node) {
    if (auto i = reach_.find(node); i != reach_.end()) return i->second;

    auto res  = DefSet();
    auto todo = fe::BFSWorklist<GIDSet<const Nest::Node*>>();
    todo.push(node->sibl_deps().nodes());
    while (!todo.empty()) {
        auto s          = todo.pop();
        const auto& sub = mentions_.at(s);
        res.insert(sub.begin(), sub.end());
        todo.push(s->sibl_deps().nodes());
    }
    return reach_[node] = std::move(res);
}

void FreeLocals::place(const Nest::Node* home, const Nest::Node* node, const Def* alloc) {
    for (auto child : node->children().nodes())
        place(home, child, alloc);

    auto lam = node->mut()->isa_mut<Lam>();
    if (!lam || escape_.escaping(lam) || (node != home && !live(home, node, alloc))) return;
    auto i = escape_.chains().find(lam);
    if (i == escape_.chains().end()) return;

    const auto& chain = i->second;
    auto deads        = DefVec();
    for (auto target : chain.targets)
        if (dead(home, target, alloc)) deads.emplace_back(target);

    if (deads.size() == chain.targets.size())
        frees_[chain.tail].emplace_back(alloc);
    else
        for (auto target : deads)
            edges_[chain.tail][target].emplace_back(alloc);
}

bool FreeLocals::live(const Nest::Node* home, const Nest::Node* node, const Def* alloc) {
    if (mentions_.at(node).contains(alloc)) return true;
    for (auto x = node; x != home; x = x->inest())
        if (reach(x).contains(alloc)) return true;
    return false;
}

bool FreeLocals::dead(const Nest::Node* home, const Def* target, const Def* alloc) {
    auto scope = home->mut();
    if (!scope->nests(target)) return true;
    if (auto lam = target->isa_mut<Lam>()) return !live(home, home->nest()[lam], alloc);
    return false;
}

const Def* FreeLocals::rewrite_imm_App(const App* app) {
    auto allocs = fe::lookup(frees_, app);
    auto edges  = fe::lookup(edges_, app);
    if (!allocs && !edges) return RWPhase::rewrite_imm_App(app);

    auto& w  = new_world();
    auto mem = allocs ? mem_def(app->arg()) : nullptr;
    if (allocs && !mem) log().d("free_locals: `{}` threads no memory, leaking {} allocations", app, allocs->size());
    auto new_arg = mem ? thread(app->arg(), mem, *allocs) : rewrite(app->arg());

    if (!edges) return w.app(rewrite(app->callee()), new_arg);
    auto args = DefVec(app->num_targs(), [&](size_t i) {
        auto arg = app->targ(i);
        return Pi::isa_cn(arg->type()) ? rewrite_target(arg, *edges) : new_arg->proj(app->num_targs(), i);
    });
    return w.app(rewrite_target(app->callee(), *edges), w.tuple(args));
}

const Def* FreeLocals::thread(const Def* arg, const Def* mem, const DefVec& allocs) {
    if (arg == mem) return free(rewrite(mem), allocs);
    if (mem_def(arg) != mem) return rewrite(arg);
    return new_world().tuple(arg->projs([&](const Def* proj) { return thread(proj, mem, allocs); }));
}

// A continuation's pointer parameter owns its buffer when every jump back into the continuation passes a fresh
// allocation that nothing else keeps, and it is only ever read and written through: all but the one value entering from
// outside are then the continuation's to free, which a comparison with that value tells at run time.
std::vector<std::pair<const Def*, Lam*>> FreeLocals::owned_params() {
    owned_.clear();
    auto users = DefMap<DefVec>();
    auto calls = LamMap<std::vector<const App*>>();
    auto seen  = DefSet();
    auto todo  = DefVec();
    for (auto mut : old_world().externals().muts())
        todo.push_back(mut);
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (!seen.emplace(def).second) continue;
        if (auto app = def->isa<App>())
            if (auto lam = app->callee()->isa_mut<Lam>()) calls[lam].push_back(app);
        for (auto op : def->deps())
            if (op) {
                users[op].push_back(def);
                todo.push_back(op);
            }
    }
    auto none     = DefVec();
    auto users_of = [&](const Def* def) -> const DefVec& {
        auto i = users.find(def);
        return i == users.end() ? none : i->second;
    };
    // Whether @p ptr is only an address that `lea`, `load` and `store` access, or the argument of @p jump.
    auto accessed = [&](const Def* ptr, const Def* jump) {
        for (auto user : users_of(ptr)) {
            auto tuple = user->isa<Tuple>();
            if (!tuple) return false;
            if (tuple == jump) {
                if (users_of(tuple).size() != 1) return false;
                continue;
            }
            for (auto op : users_of(tuple)) {
                auto app = op->isa<App>();
                if (!app || app->arg() != tuple) return false;
                if (Axm::isa<mem::lea>(app) && tuple->op(0) == ptr) continue;
                if ((Axm::isa<mem::load>(app) || Axm::isa<mem::store>(app)) && tuple->op(1) == ptr) continue;
                return false;
            }
        }
        return true;
    };
    auto result = std::vector<std::pair<const Def*, Lam*>>();
    for (auto& [lam, sites] : calls) {
        auto var = lam->has_var();
        if (!var || !lam->is_set() || !Pi::isa_cn(lam->type())) continue;
        auto n = lam->num_vars();
        for (size_t k = 0; k != n; ++k) {
            auto param = lam->var(n, k);
            auto ptr   = Axm::isa<mem::Ptr>(param->type());
            if (!ptr || Lit::isa(ptr->arg(2, 1)) != 0 || !accessed(param, nullptr)) continue;
            const Def* entry = nullptr;
            auto owned       = true;
            auto loops       = false;
            for (auto site : sites) {
                auto arg = site->arg()->proj(n, k);
                if (!site->has_free_var(var)) {
                    owned &= !entry && arg->in_scope(lam);
                    entry = arg;
                    continue;
                }
                auto made  = arg->isa<Extract>();
                auto alloc = made ? made->tuple()->isa<App>() : nullptr;
                owned &= alloc && Lit::isa(made->index()) == 1 && EscapeAnalysis::allocates(alloc)
                      && !escape_.freed(alloc);
                loops = true;
                if (!owned) break;
                // The allocation reaches nothing but its reads and writes and this jump.
                owned &= accessed(arg, site->arg());
            }
            if (owned && loops && entry) {
                owned_[param] = entry;
                result.emplace_back(param, lam);
            }
        }
    }
    return result;
}

bool FreeLocals::uses_directly(Lam* lam, const Def* param) {
    auto seen = DefSet();
    auto todo = DefVec{lam->body()};
    while (!todo.empty()) {
        auto def = todo.back();
        todo.pop_back();
        if (!def || !seen.emplace(def).second || def->isa_mut()) continue;
        if (def == param) return true;
        for (auto op : def->ops())
            todo.push_back(op);
    }
    return false;
}

const Def* FreeLocals::free(const Def* mem, const DefVec& allocs) {
    // TODO: address spaces other than 0 are barely tested; some, like gpu shared memory, cannot be freed.
    auto space = [](const Def* m) { return Lit::isa(Axm::as<mem::M>(m->type())->arg()); };
    auto& w    = new_world();
    // A memory converted from another address space, like a kernel's exit, frees that space's allocations before.
    if (auto conv = mem->isa<App>(); conv && Axm::isa<mem::M>(conv->arg()->type())) {
        auto inner = DefVec();
        for (auto alloc : allocs)
            if (!owned_.contains(alloc) && space(alloc->proj(2, 0)) == space(conv->arg())
                && space(conv->arg()) != space(mem))
                inner.emplace_back(alloc);
        if (!inner.empty()) {
            mem        = w.app(conv->callee(), free(conv->arg(), inner));
            auto outer = DefVec();
            for (auto alloc : allocs)
                if (std::ranges::find(inner, alloc) == inner.end()) outer.emplace_back(alloc);
            return free(mem, outer);
        }
    }
    for (auto alloc : allocs) {
        if (auto entry = fe::lookup(owned_, alloc)) {
            auto ptr  = rewrite(alloc);
            auto i64  = w.type_i64();
            auto same = w.call(core::icmp::e,
                               Defs{w.call<core::bitcast>(i64, ptr), w.call<core::bitcast>(i64, rewrite(entry))});
            mem = w.call<mem::free>(Defs{mem, w.select(same, w.call<core::bitcast>(ptr->type(), w.lit_i64(0)), ptr)});
            continue;
        }
        if (space(alloc->proj(2, 0)) != space(mem)) {
            log().d("free_locals: `{}` lives in another address space than `{}`, leaking it", alloc, mem);
            continue;
        }
        mem = w.call<mem::free>(Defs{mem, rewrite(alloc->proj(2, 1))});
    }
    return mem;
}

const Def* FreeLocals::rewrite_target(const Def* target, const DefMap<DefVec>& edges) {
    auto& w = new_world();
    if (auto allocs = fe::lookup(edges, target)) return trampoline(rewrite(target), *allocs);
    if (auto tuple = target->isa<Tuple>())
        return w.tuple(DefVec(tuple->num_ops(), [&](size_t i) { return rewrite_target(tuple->op(i), edges); }));
    if (auto sw = Switch(target)) return w.extract(rewrite_target(sw.tuple(), edges), rewrite(sw.index()));
    return rewrite(target);
}

const Def* FreeLocals::trampoline(const Def* cont, const DefVec& allocs) {
    auto& w   = new_world();
    auto lam  = w.mut_con(Pi::isa_cn(cont->type())->dom())->set("free_then");
    auto args = DefVec(lam->vars());
    auto mem  = std::ranges::find_if(args, [](const Def* arg) { return isa_mem(arg); });
    if (mem == args.end()) {
        log().d("free_locals: `{}` takes no memory, leaking {} allocations", cont, allocs.size());
        return cont;
    }
    *mem = free(*mem, allocs);
    lam->app(false, cont, args);
    return lam;
}

} // namespace mim::plug::mem::phase
