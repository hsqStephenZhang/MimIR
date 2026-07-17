#include "mim/plug/regex/dfa2matcher.h"

#include <algorithm>

#include <automaton/dfa.h>
#include <automaton/range_helper.h>

#include <mim/plug/core/core.h>
#include <mim/plug/mem/mem.h>

template<>
struct std::formatter<automaton::DFA> : fe::ostream_formatter {};

using namespace mim;
using namespace automaton;

using Range  = automaton::Range;
using Ranges = Vector<Range>;

// nomenclature:
// c is the character from the string we want to match

// see lit/regex/match_manual.mim for a hand written state machine impl that this is based on

// the main idea is:
// for every state in the DFA, we create a lam that checks if the char c at pos is the end of the string \0
// if not, jump to a checker lam, that consists of a few bit-wise or-ed in-range checks to verify if we can transition
// to a certain other state with c if any of the checks is true, we jump to the corresponding state lam with the updated
// position if the check fails, we jump to the next checker lam or the exit lam if we checked all possible transitions
// Note, since the DFA stores the transitions as chars, not ranges, we have to merge the transitions to ranges first
// (cf. transitions_to_ranges)

namespace {

namespace core = plug::core;
namespace mem  = plug::mem;

// Name states by their stable id - never by pointer value, which would change from run to run.
std::string state_to_name(const DFANode* state) { return "state_" + std::to_string(state->id()); }

DFAMap<Ranges> transitions_to_ranges(World& w, const DFANode* state) {
    DFAMap<Ranges> state2ranges;
    state->for_transitions([&](std::uint16_t transition, const DFANode* next_state) {
        if (!state2ranges.contains(next_state))
            state2ranges.try_emplace(next_state, Ranges{
                                                     {transition, transition}
            });
        else
            state2ranges[next_state].emplace_back(transition, transition);
    });
    Range any_range{0, 255};
    for (auto& [state, ranges] : state2ranges) {
        if (std::ranges::contains(ranges, any_range)) {
            ranges = {any_range};
            continue;
        }

        std::sort(ranges.begin(), ranges.end(), RangeCompare{});
        ranges = merge_ranges(ranges, [&w](std::string_view msg) { w.DLOG("{}", msg); });
    }
    return state2ranges;
}

const Def* match_range(const Def* c, nat_t lo, nat_t hi) {
    World& w = c->world();
    if (lo == 0 && hi == 255) return w.lit_tt();

    // let in_range     = %core.bit2.and_ 0 (%core.icmp.uge (char, lower),  %core.icmp.ule (char, upper));
    auto below_hi = w.call(core::icmp::ule, w.tuple({c, w.lit_i8(hi)}));
    auto above_lo = w.call(core::icmp::uge, w.tuple({c, w.lit_i8(lo)}));
    return w.call(core::bit2::and_, w.lit_nat(2), w.tuple({below_hi, above_lo}));
}

DFAMap<const Def*> create_check_match_transitions_from(const Def* c, const DFANode* state) {
    World& w = c->world();
    DFAMap<const Def*> state2check;

    auto state2ranges = transitions_to_ranges(w, state);

    for (auto& [state, ranges] : state2ranges) {
        for (auto& [lo, hi] : ranges)
            if (!state2check.contains(state))
                state2check.try_emplace(state, match_range(c, lo, hi));
            else
                state2check[state]
                    = w.call(core::bit2::or_, w.lit_nat(2), w.tuple({state2check[state], match_range(c, lo, hi)}));
    }
    return state2check;
}

} // namespace

extern "C" const Def* dfa2matcher(World& w, const DFA& dfa, const Def* n) {
    w.DLOG("dfa to match: {}", dfa);

    auto states = dfa.get_reachable_states();
    DFAMap<Lam*> state2matcher;

    // ((mem: %mem.M 0, string: Str n, pos: Idx n), Cn [%mem.M 0, Bool, Idx n])
    auto matcher = w.mut_fun({w.call<mem::M>(0), w.call<mem::Ptr0>(w.arr(n, w.type_i8())), w.type_idx(n)},
                             {w.call<mem::M>(0), w.type_bool(), w.type_idx(n)});
    matcher->debug_prefix(std::string("match_regex"));
    auto [args, exit] = matcher->vars<2>();
    exit->debug_prefix(std::string("exit"));
    auto [mem, string, pos] = args->projs<3>();
    mem->debug_prefix(std::string("mem"));
    string->debug_prefix(std::string("string"));
    pos->debug_prefix(std::string("pos"));

    auto error = mem::mut_con(w.type_idx(n));
    error->debug_prefix("error");
    {
        auto [mem, pos] = error->vars<2>();
        mem->debug_prefix(std::string("mem"));
        pos->debug_prefix(std::string("pos"));
        error->app(false, exit, {mem, w.lit_ff(), pos});
    }

    auto accept = mem::mut_con(w.type_idx(n));
    accept->debug_prefix("accept");
    {
        auto [mem, pos] = accept->vars<2>();
        mem->debug_prefix(std::string("mem"));
        pos->debug_prefix(std::string("pos"));
        accept->app(false, exit, {mem, w.lit_tt(), pos});
    }

    auto exiting = [error, accept](const DFANode* state) { return state->is_accepting() ? accept : error; };

    for (auto state : states) {
        auto lam = mem::mut_con(w.type_idx(n));
        lam->debug_prefix(state_to_name(state));
        state2matcher.emplace(state, lam);
    }

    for (auto [state, lam] : state2matcher) {
        auto [mem, i] = lam->vars<2>();

        if (state->is_erroring()) {
            lam->app(true, error, {mem, i});
            continue;
        }

        auto lea       = w.call<mem::lea>(Defs{string, i});
        auto [mem2, c] = w.call<mem::load>(Defs{mem, lea})->projs<2>();

        auto is_end  = w.call(core::icmp::e, Defs({c, w.lit_i8(0)}));
        auto not_end = mem::mut_con(w.type_idx(n));
        not_end->debug_prefix("not_end_" + state_to_name(state));

        auto new_i = w.call(core::wrap::add, core::Mode::nsuw, w.tuple({i, w.call(core::conv::u, n, w.lit_i64(1))}));
        lam->app(false, w.select(is_end, exiting(state), not_end), {mem2, i});

        auto transitions = create_check_match_transitions_from(c, state);
        auto next_check  = exiting(state); // if we want to check full string only, use error instead of exiting(state)c
        for (auto [next_state, check] : transitions) {
            auto next_lam = state2matcher[next_state];
            auto checker  = mem::mut_con(w.type_idx(n));
            checker->debug_prefix("check_" + state_to_name(state) + "_to_" + state_to_name(next_state));
            auto [mem3, pos] = checker->vars<2>();
            checker->app(false, w.select(check, next_lam, next_check), {mem3, w.select(check, new_i, pos)});
            next_check = checker;
        }
        {
            auto [mem, pos] = not_end->vars<2>();
            not_end->app(true, next_check, {mem, pos});
        }
    }

    matcher->app(false, state2matcher[dfa.get_start()], {mem, pos});
    return matcher;
}

namespace mim::plug::regex {

DFATable dfa_to_table(World& w, const automaton::DFA& dfa) {
    auto reachable = dfa.get_reachable_states();

    DFAMap<nat_t> state2id;
    std::vector<const DFANode*> id2state;
    id2state.reserve(reachable.size() + 1);
    for (auto state : reachable) {
        state2id.emplace(state, id2state.size());
        id2state.emplace_back(state);
    }

    auto error_it = std::ranges::find_if(id2state, [](const DFANode* state) { return state->is_erroring(); });
    nat_t error_id;
    if (error_it == id2state.end()) {
        // Some incomplete automata do not carry an explicit error node. The
        // table interpreter still wants one so unmatched input has a concrete
        // rejecting target.
        error_id = id2state.size();
    } else {
        error_id = std::distance(id2state.begin(), error_it);
    }

    DFATable table{state2id[dfa.get_start()], error_id, {}};
    table.states.reserve(id2state.size() + (error_it == id2state.end() ? 1 : 0));
    nat_t max_transitions = 0;

    for (auto state : id2state) {
        TableState row{state->is_accepting(), error_id, {}};
        if (!state->is_erroring()) {
            auto state2ranges = transitions_to_ranges(w, state);
            for (auto& [target, ranges] : state2ranges) {
                auto found = state2id.find(target);
                nat_t target_id = found == state2id.end() ? error_id : found->second;
                if (target_id == error_id) continue;
                for (auto [lo, hi] : ranges)
                    row.transitions.push_back({static_cast<std::uint8_t>(lo), static_cast<std::uint8_t>(hi), target_id});
            }
        }
        max_transitions = std::max(max_transitions, nat_t(row.transitions.size()));
        table.states.emplace_back(std::move(row));
    }

    if (error_it == id2state.end()) table.states.push_back(TableState{false, error_id, {}});

    // Keep at least two slots: a one-element pack collapses to the element in
    // the current C++ tuple builder, while `%regex.DFA ns k` needs an array.
    max_transitions = std::max<nat_t>(max_transitions, 2);
    for (auto& state : table.states) {
        state.transitions.resize(max_transitions, TableTransition{255, 0, error_id});
    }

    return table;
}

const Def* encode_dfa_table(World& w, const DFATable& dfa) {
    auto ns     = nat_t(dfa.states.size());
    auto k      = dfa.states.empty() ? nat_t(1) : nat_t(dfa.states.front().transitions.size());
    auto ns_def = w.lit_nat(ns);
    auto k_def  = w.lit_nat(k);

    auto transition_ty = w.call(dfa::Transition, ns_def);
    auto state_ty      = w.call(dfa::State, ns_def, k_def);
    auto dfa_ty        = w.call<regex::DFA>(ns_def, k_def);

    DefVec state_defs;
    state_defs.reserve(ns);
    for (auto& state : dfa.states) {
        DefVec transition_defs;
        transition_defs.reserve(k);
        for (auto transition : state.transitions) {
            transition_defs.emplace_back(w.tuple(transition_ty,
                                                 Defs{w.lit_i8(transition.lo), w.lit_i8(transition.hi),
                                                      w.lit_idx(ns, transition.target)}));
        }

        state_defs.emplace_back(w.tuple(state_ty,
                                        Defs{state.accepting ? w.lit_tt() : w.lit_ff(), w.lit_idx(ns, state.fallback),
                                             w.tuple(w.arr(k_def, transition_ty), transition_defs)}));
    }

    return w.tuple(dfa_ty, Defs{w.lit_idx(ns, dfa.entry), w.lit_idx(ns, dfa.error),
                                w.tuple(w.arr(ns_def, state_ty), state_defs)});
}

const Def* dfa_table2matcher(World& w, const DFATable& dfa, const Def* n) {
    assert(dfa.entry < dfa.states.size());
    assert(dfa.error < dfa.states.size());

    auto matcher = w.mut_fun({w.call<mem::M>(0), w.call<mem::Ptr0>(w.arr(n, w.type_i8())), w.type_idx(n)},
                             {w.call<mem::M>(0), w.type_bool(), w.type_idx(n)});
    matcher->debug_prefix("match_dfa_table");
    auto [args, exit]          = matcher->vars<2>();
    auto [memory, string, pos] = args->projs<3>();

    auto reject = mem::mut_con(w.type_idx(n));
    reject->debug_prefix("reject");
    {
        auto [mem, i] = reject->vars<2>();
        reject->app(false, exit, {mem, w.lit_ff(), i});
    }

    auto accept = mem::mut_con(w.type_idx(n));
    accept->debug_prefix("accept");
    {
        auto [mem, i] = accept->vars<2>();
        accept->app(false, exit, {mem, w.lit_tt(), i});
    }

    // Allocate all state continuations before emitting bodies. This is the
    // finite memo table that makes cyclic DFA specialization terminate.
    std::vector<Lam*> states;
    states.reserve(dfa.states.size());
    for (nat_t i = 0; i != dfa.states.size(); ++i) {
        auto state = mem::mut_con(w.type_idx(n));
        state->debug_prefix("state_" + std::to_string(i));
        states.emplace_back(state);
    }

    for (nat_t state_id = 0; state_id != dfa.states.size(); ++state_id) {
        auto state_lam          = states[state_id];
        const auto& table_state = dfa.states[state_id];
        auto [mem, i]           = state_lam->vars<2>();

        if (state_id == dfa.error) {
            state_lam->app(true, reject, {mem, i});
            continue;
        }

        auto ptr       = w.call<mem::lea>(Defs{string, i});
        auto [mem2, c] = w.call<mem::load>(Defs{mem, ptr})->projs<2>();
        auto advanced = w.call(core::wrap::add, core::Mode::nsuw, w.tuple({i, w.call(core::conv::u, n, w.lit_i64(1))}));

        // Keep EOF handling on the cold transition-miss path. In particular,
        // an accepting self-loop should test its hot byte range and branch
        // back directly instead of materializing `(c == 0)` on every byte.
        auto miss = mem::mut_con(w.type_idx(n));
        miss->debug_prefix("miss_state_" + std::to_string(state_id));
        {
            auto [miss_mem, miss_pos] = miss->vars<2>();
            auto is_end               = w.call(core::icmp::e, Defs{c, w.lit_i8(0)});
            auto on_end               = table_state.accepting ? accept : reject;
            miss->app(false, w.select(is_end, on_end, states[table_state.fallback]),
                      {miss_mem, w.select(is_end, miss_pos, advanced)});
        }

        const Def* next = miss;
        for (nat_t transition_id = 0; transition_id != table_state.transitions.size(); ++transition_id) {
            const auto& transition = table_state.transitions[transition_id];
            // NUL terminates `Input`; it is never part of the matched byte
            // stream, even if a closed table contains a range starting at 0.
            auto lo = std::max<std::uint8_t>(transition.lo, 1);
            if (lo > transition.hi) continue;

            auto checker = mem::mut_con(w.type_idx(n));
            checker->debug_prefix("check_state_" + std::to_string(state_id) + "_transition_"
                                  + std::to_string(transition_id));
            auto [check_mem, check_pos] = checker->vars<2>();
            auto in_range               = match_range(c, lo, transition.hi);
            checker->app(false, w.select(in_range, states[transition.target], next),
                         {check_mem, w.select(in_range, advanced, check_pos)});
            next = checker;
        }

        // State continuations are recursive CFG nodes and must remain opaque to
        // partial evaluation; the checker chain itself may still inline.
        state_lam->app(false, next, {mem2, i});
    }

    matcher->app(false, states[dfa.entry], {memory, pos});
    return matcher;
}

} // namespace mim::plug::regex
