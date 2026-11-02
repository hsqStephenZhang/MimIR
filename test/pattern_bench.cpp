#include <chrono>
#include <cstdlib>

#include <iostream>
#include <string>

#include <mim/driver.h>
#include <mim/world.h>

#include <mim/pattern/graph_match.h>

using namespace mim;
using namespace mim::pattern;
using Clock = std::chrono::steady_clock;

// Build explicitly: cmake --build <build> --target mim-pattern-bench.
// CSV counters are deterministic; timings are process-local observations, not test assertions.
int main(int argc, char** argv) {
    auto size = argc > 1 ? std::stoul(argv[1]) : 100;
    if (!size || size > 10000) return 2;
    std::cout << "workload,size,mode,matches,status,candidates,steps,prefilter_checks,index_steps,build_ns,match_ns\n";
    for (auto workload : {"parallel", "chain", "diamond", "spine", "fanout"}) {
        Driver driver;
        auto& w      = driver.world();
        auto nat     = w.type_nat();
        bool fanout  = std::string(workload) == "fanout";
        auto type    = fanout ? w.pi(nat, w.pi(nat, nat)) : w.pi(nat, nat);
        const Axm *q = w.axm(type), *k = w.axm(type), *v = w.axm(type);
        OpRegistry ops;
        for (auto op : {q, k, v})
            ops.add(op->flags(), fanout ? 2 : 1, {{fanout ? 1u : 0u}}, true);
        std::vector<const Def*> roots;
        const Def* x = w.axm(nat);
        for (size_t i = 0; i != size; ++i) {
            if (std::string(workload) == "parallel") x = w.lit_nat(i);
            if (std::string(workload) == "chain") {
                x = w.app(q, x);
                roots.push_back(x);
            } else if (fanout) {
                for (auto op : {q, k, v})
                    roots.push_back(w.app(w.app(op, w.lit_nat(i)), x));
            } else {
                auto input = std::string(workload) == "diamond" ? w.app(q, x) : x;
                roots.push_back(w.app(q, input));
                roots.push_back(w.app(k, input));
                roots.push_back(w.app(v, input));
                if (std::string(workload) == "diamond") x = w.lit_nat(i + 1);
                if (std::string(workload) == "spine") x = w.app(q, x);
            }
        }
        auto begin = Clock::now();
        GraphView view({roots.data(), roots.size()}, {.operators = &ops});
        auto build_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
        PatternProgram p;
        auto input = p.capture();
        auto tag   = p.capture();
        GraphPattern pattern(p);
        for (auto op : std::string(workload) == "chain" ? std::vector{q} : std::vector{q, k, v})
            pattern.anchor(p.node(Node::App, {fanout ? p.node(Node::App, {p.exact(op), tag}) : p.exact(op), input}));
        for (auto mode : {GraphSearchMode::Exhaustive, GraphSearchMode::Indexed, GraphSearchMode::Compiled}) {
            GraphMatchOptions options;
            options.match.max_steps = 100000000;
            options.max_candidates  = 100000000;
            options.max_results     = size * 4 + 1;
            options.mode            = mode;
            auto start              = Clock::now();
            auto found              = find_matches(pattern, view, options);
            auto elapsed   = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
            auto mode_name = mode == GraphSearchMode::Compiled ? "compiled"
                           : mode == GraphSearchMode::Indexed  ? "indexed"
                                                               : "exhaustive";
            std::cout << workload << ',' << size << ',' << mode_name << ',' << found.matches.size() << ','
                      << int(found.status) << ',' << found.candidates << ',' << found.steps << ','
                      << found.prefilter_checks << ',' << found.index_steps << ',' << build_ns << ',' << elapsed
                      << '\n';
            if (found.status != SearchStatus::Complete) return 1;
        }
    }
}
