#include <ghl/isomorphism/graph_isomorphisms.h>

#include "common.h"

// Use GHL's actual discover() API and map collection, rather than a direct Boost call.
using Vertex = std::shared_ptr<ghl::BaseNode>;
using Edge   = std::shared_ptr<ghl::PrimitiveEdge>;
using Graph  = ghl::DirectedGraph<Vertex, Edge>;
using Base   = ghl::DirectedIsomorphism<Vertex, Edge>;
struct Matcher : Base {
    using Base::Base;
    EdgeCompFunction edge_comp_function() override {
        return [](const Graph& a, const Graph& b, auto x, auto y) { return a[x]->id() == b[y]->id(); };
    }
};

Graph convert(const common::Graph& input) {
    Graph graph;
    for (size_t i = 0; i < input.size; ++i)
        boost::add_vertex(std::make_shared<ghl::BaseNode>(0), graph);
    for (auto e : input.edges)
        boost::add_edge(e.source, e.target, std::make_shared<ghl::PrimitiveEdge>(3 * e.output + e.input), graph);
    return graph;
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    auto corpus  = common::read(argv[1]);
    auto begin   = common::Clock::now();
    auto host    = convert(corpus[0]);
    auto host_ns = common::ns(begin);
    begin        = common::Clock::now();
    std::vector<Graph> patterns;
    for (size_t i = 1; i < corpus.size(); ++i)
        patterns.push_back(convert(corpus[i]));
    Matcher matcher(Graph{});
    auto prepare_ns = common::ns(begin);
    std::vector<std::vector<Base::IsoMap>> results;
    begin = common::Clock::now();
    for (auto& pattern : patterns)
        results.push_back(matcher.discover(pattern, host, false, false, std::string(argv[3]) == "vf3"));
    auto search_ns = common::ns(begin);
    common::Mappings mappings;
    for (size_t i = 0; i < results.size(); ++i)
        for (auto& match : results[i]) {
            common::Mapping mapping{i};
            for (size_t v = 0; v < corpus[i + 1].size; ++v)
                mapping.push_back(match.at(v).at(0));
            mappings.push_back(std::move(mapping));
        }
    common::write(argv[2], mappings);
    std::cout << "ghl-" << argv[3] << ',' << host_ns << ',' << prepare_ns << ',' << search_ns << ',' << mappings.size()
              << ",0,0\n";
}
