#include <memory>

#include "../../test/pattern_portgraph.h"
#include "common.h"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    auto corpus  = common::read(argv[1]);
    auto convert = [](const common::Graph& graph) {
        port_test::Graph result{graph.size, {}};
        for (auto e : graph.edges)
            result.edges.push_back({e.source, e.output, e.target, e.input});
        return result;
    };
    auto begin = common::Clock::now();
    port_test::IR ir(convert(corpus[0]));
    auto host_ns = common::ns(begin);
    begin        = common::Clock::now();
    mim::pattern::GraphView view(ir.nodes, {.operators = &ir.ops});
    std::vector<std::unique_ptr<port_test::Compiled>> patterns;
    for (size_t i = 1; i < corpus.size(); ++i)
        patterns.push_back(
            std::make_unique<port_test::Compiled>(convert(corpus[i]), ir, std::string(argv[3]).starts_with("flat-")));
    auto prepare_ns = common::ns(begin);
    // Identity decoding is outside the timed search, as it is for GHL/portmatching.
    mim::DefMap<size_t> vertex_ids;
    for (size_t i = 0; i < ir.nodes.size(); ++i)
        vertex_ids[ir.nodes[i]] = i;
    auto mode               = std::string(argv[3]).ends_with("exhaustive") ? mim::pattern::GraphSearchMode::Exhaustive
                                                                           : mim::pattern::GraphSearchMode::Indexed;
    auto options            = port_test::options(mode);
    options.match.max_depth = 4096; // Multi-anchor continuations exceed the default at chain length eight.
    std::vector<mim::pattern::GraphMatches> results;
    begin = common::Clock::now();
    for (auto& pattern : patterns)
        results.push_back(mim::pattern::find_matches(pattern->graph, view, options));
    auto search_ns = common::ns(begin);
    common::Mappings mappings;
    size_t steps = 0, candidates = 0;
    for (size_t i = 0; i < patterns.size(); ++i) {
        auto& result = results[i];
        if (result.status != mim::pattern::SearchStatus::Complete) {
            std::cerr << "incomplete pattern " << i << ": status=" << int(result.status) << ", steps=" << result.steps
                      << ", candidates=" << result.candidates << '\n';
            return 1;
        }
        steps += result.steps;
        candidates += result.candidates;
        for (auto& match : result.matches) {
            common::Mapping mapping{i};
            for (auto anchor : patterns[i]->anchors)
                mapping.push_back(vertex_ids.at(match.at(anchor)));
            mappings.push_back(std::move(mapping));
        }
    }
    common::write(argv[2], mappings);
    std::cout << "mimir-" << argv[3] << ',' << host_ns << ',' << prepare_ns << ',' << search_ns << ','
              << mappings.size() << ',' << candidates << ',' << steps << '\n';
}
