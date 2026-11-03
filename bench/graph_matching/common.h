#pragma once

#include <chrono>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace common {
using Clock    = std::chrono::steady_clock;
using Mapping  = std::vector<size_t>; // pattern ID followed by host vertex IDs in pattern order.
using Mappings = std::vector<Mapping>;
inline auto ns(Clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}
struct Edge {
    size_t source, output, target, input;
};
struct Graph {
    size_t size;
    std::vector<Edge> edges;
};
inline std::vector<Graph> read(const char* path) {
    std::ifstream input(path);
    size_t count = 0;
    input >> count;
    std::vector<Graph> graphs;
    for (size_t i = 0; i < count; ++i) {
        size_t n = 0, e = 0;
        input >> n >> e;
        Graph graph{n, {}};
        for (size_t j = 0; j < e; ++j) {
            Edge edge{};
            input >> edge.source >> edge.output >> edge.target >> edge.input;
            graph.edges.push_back(edge);
        }
        graphs.push_back(std::move(graph));
    }
    if (!input || graphs.size() < 2) throw std::runtime_error("invalid benchmark corpus");
    return graphs;
}
inline void write(const char* path, const Mappings& matches) {
    std::ofstream output(path);
    for (auto& match : matches) {
        for (auto vertex : match)
            output << vertex << ' ';
        output << '\n';
    }
    if (!output) throw std::runtime_error("cannot write mappings");
}
} // namespace common
