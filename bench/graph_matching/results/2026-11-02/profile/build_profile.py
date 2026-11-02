import json
from pathlib import Path
import shlex
import subprocess

repo = Path('/workspaces/compile/MimIR')
root = Path('/tmp/mimir-pattern-profile')
build = Path('/tmp/mimir-reference-bench/mim-release')
source = (repo / 'src/mim/pattern/match.cpp').read_text()

support = r'''
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
namespace match_profile {
using Clock = std::chrono::steady_clock;
enum Stage { Analysis, Prefilter, Index, Enumerate, Narrow, Output, Cleanup, Count };
constexpr const char* names[] = {"analysis", "prefilter", "index_setup", "enumerate", "narrow", "output", "cleanup"};
struct Stats {
    Stage stage = Analysis;
    Clock::time_point last = Clock::now();
    uint64_t ns[Count]{}, allocations[Count]{}, bytes[Count]{}, frees[Count]{};
    uint64_t visits=0, bound_captures=0, new_captures=0, narrow_calls=0, parent_calls=0, parent_hits=0;
    uint64_t read_pops=0, read_nodes=0, capture_slots=0, copied_slots=0, filter_states=0;
    uint64_t filter_calls=0, filter_hits=0, enumerate_48=0;
    bool no_cache = std::getenv("MIM_PROFILE_NO_CACHE");
    bool no_metadata = std::getenv("MIM_PROFILE_NO_METADATA");
    void set(Stage next) {
        auto now=Clock::now(); ns[stage] += std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count();
        stage=next; last=now;
    }
};
thread_local Stats* active __attribute__((tls_model("initial-exec"))) = nullptr;
struct Scope {
    Stats& stats; Stage saved;
    Scope(Stage next):stats(*active),saved(stats.stage){ stats.set(next); }
    ~Scope(){ stats.set(saved); }
};
struct Session {
    Stats stats;
    size_t anchors=0, entries=0, nodes=0, ops=0, matches=0, candidates=0, steps=0;
    bool complete=false;
    Session(){ active=&stats; }
    void finish(size_t a, size_t e, size_t n, size_t o, size_t m, size_t c, size_t s) {
        anchors=a;entries=e;nodes=n;ops=o;matches=m;candidates=c;steps=s;complete=true;
        stats.set(Cleanup);
    }
    ~Session() {
        stats.set(stats.stage); active=nullptr;
        if (auto path=std::getenv("MIM_PROFILE_JSON"); path && complete) {
            std::ofstream out(path,std::ios::app);
            out << "{\"anchors\":" << anchors << ",\"entries\":" << entries << ",\"nodes\":" << nodes
                << ",\"operations\":" << ops << ",\"matches\":" << matches << ",\"candidates\":" << candidates
                << ",\"steps\":" << steps << ",\"no_cache\":" << stats.no_cache << ",\"no_metadata\":" << stats.no_metadata;
            uint64_t total=0;
            for (int i=0;i<Count;++i) {
                total+=stats.ns[i];
                out << ",\"" << names[i] << "_ns\":" << stats.ns[i]
                    << ",\"" << names[i] << "_allocations\":" << stats.allocations[i]
                    << ",\"" << names[i] << "_bytes\":" << stats.bytes[i];
            }
            out << ",\"total_ns\":" << total << ",\"visits\":" << stats.visits
                << ",\"bound_captures\":" << stats.bound_captures << ",\"new_captures\":" << stats.new_captures
                << ",\"narrow_calls\":" << stats.narrow_calls << ",\"parent_calls\":" << stats.parent_calls
                << ",\"parent_hits\":" << stats.parent_hits << ",\"read_pops\":" << stats.read_pops
                << ",\"read_nodes\":" << stats.read_nodes << ",\"capture_slots\":" << stats.capture_slots
                << ",\"filter_calls\":" << stats.filter_calls << ",\"filter_hits\":" << stats.filter_hits << ",\"enumerate_48\":" << stats.enumerate_48
                << ",\"copied_slots\":" << stats.copied_slots << ",\"filter_states\":" << stats.filter_states << "}\n";
        }
    }
};
}
extern "C" void* __libc_malloc(size_t);
extern "C" void __libc_free(void*);
extern "C" void* malloc(size_t n) noexcept {
    auto value=__libc_malloc(n);
    if (auto p=match_profile::active) {++p->allocations[p->stage];p->bytes[p->stage]+=n;if(p->stage==match_profile::Enumerate && n==48) ++p->enumerate_48;}
    return value;
}
extern "C" void free(void* p) noexcept {
    if (auto stats=match_profile::active) ++stats->frees[stats->stage];
    __libc_free(p);
}
'''

def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)

replace('namespace mim::pattern {', support+'\nnamespace mim::pattern {')
replace('    snapshot_ = graph.snapshot();', '    match_profile::Session profile;\n    snapshot_ = graph.snapshot();')
replace('    GraphMatches output{{}, SearchStatus::Complete, 0, 0};',
        '    profile.stats.set(match_profile::Prefilter);\n    GraphMatches output{{}, SearchStatus::Complete, 0, 0};')
# Restrict the cache ablation to this recursive coarse filter; preserve all candidate indexes.
replace('if (options.mode == GraphSearchMode::Indexed)\n            if (auto i = filtered.find(key)',
        'if (options.mode == GraphSearchMode::Indexed && !profile.stats.no_cache)\n            if (auto i = filtered.find(key)')
replace('if (!prefilter_limited && options.mode == GraphSearchMode::Indexed) filtered.emplace(key, result);',
        'if (!prefilter_limited && options.mode == GraphSearchMode::Indexed && !profile.stats.no_cache) filtered.emplace(key, result);')
replace('        auto key = std::pair{p.index_, def};',
        '        ++profile.stats.filter_calls;\n        auto key = std::pair{p.index_, def};')
replace('if (auto i = filtered.find(key); i != filtered.end()) return i->second;',
        'if (auto i = filtered.find(key); i != filtered.end()) {++profile.stats.filter_hits; return i->second;}')
replace('    // Required raw captures occur',
        '    profile.stats.filter_states = filtered.size();\n    profile.stats.set(match_profile::Index);\n    // Required raw captures occur')
replace('    auto narrow = [&](size_t anchor) -> std::optional<std::vector<const Def*>> {',
        '    auto narrow = [&](size_t anchor) -> std::optional<std::vector<const Def*>> {\n'
        '        match_profile::Scope measured(match_profile::Narrow);\n        ++profile.stats.narrow_calls;')
replace('        auto key = std::pair{value, path ? *path : OperandPath{size_t(-1)}};',
        '        ++profile.stats.parent_calls;\n        auto key = std::pair{value, path ? *path : OperandPath{size_t(-1)}};')
replace('if (auto i = ancestors.find(key); i != ancestors.end()) return i->second;',
        'if (auto i = ancestors.find(key); i != ancestors.end()) {++profile.stats.parent_hits; return i->second;}')
replace('    on_match_ = [&] {', '    on_match_ = [&] {\n        match_profile::Scope measured(match_profile::Output);')
replace('        DefSet matched;','''        profile.stats.copied_slots += captures_.size();
        for (auto value:captures_) if (value) ++profile.stats.capture_slots;
        if (profile.stats.no_metadata) {
            output.matches.push_back(GraphMatch(&pattern,graph,assigned,result(),{},{}));
            limited=output.matches.size()>=options.max_results;
            return MatchStatus::Matched;
        }
        DefSet matched;''')
replace('                auto def = pending[i];\n                if (!read.insert(def).second) continue;',
        '                ++profile.stats.read_pops;\n                auto def = pending[i];\n                if (!read.insert(def).second) continue;\n                ++profile.stats.read_nodes;')
replace('    auto status   = enumerate(0);',
        '    profile.stats.set(match_profile::Enumerate);\n    auto status   = enumerate(0);')
replace('    output.steps  = budget_.steps;\n    if (!graph.is_current())',
        '    output.steps  = budget_.steps;\n'
        '    profile.finish(n, program_.entries_.size(), graph.nodes().size(), graph.operations().size(),'
        'output.matches.size(),output.candidates,output.steps);\n    if (!graph.is_current())')
replace('        const auto& entry = program_.entries_[p.index_];\n        switch (entry.kind)',
        '        if (match_profile::active) ++match_profile::active->visits;\n'
        '        const auto& entry = program_.entries_[p.index_];\n        switch (entry.kind)')
replace('            case Kind::Capture:\n                if (auto bound',
        '            case Kind::Capture:\n                if (auto p=match_profile::active) {if(captures_[p_index_placeholder]) ++p->bound_captures;else ++p->new_captures;}\n                if (auto bound'.replace('p_index_placeholder','p.index_').replace('auto p=match_profile::active','auto stats=match_profile::active').replace('++p->','++stats->'))

root.joinpath('match.cpp').write_text(source)
commands=json.loads((build/'compile_commands.json').read_text())
entry=next(c for c in commands if c['file'].endswith('/src/mim/pattern/match.cpp'))
args=shlex.split(entry['command']); args.pop(args.index('-o')+1);args.remove('-o');args.remove('-c');args.remove(entry['file'])
args += ['-g1','-shared',str(root/'match.cpp'),'-L'+str(build/'lib'),'-Wl,-rpath,'+str(build/'lib'),'-lmim','-o',str(root/'libmatch-profile.so')]
print('Building',root/'libmatch-profile.so',flush=True)
subprocess.run(args,cwd=entry['directory'],check=True)
