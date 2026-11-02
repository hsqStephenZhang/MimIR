#[link(name="allocprobe")]
unsafe extern "C" {fn bench_profile_begin(); fn bench_profile_end();}
use portgraph::{LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView};
use portmatching::{
    concrete::portgraph::{
        indexing::{PGIndexKey, PGIndexingScheme},
        PGManyPatternMatcher, PGNaiveManyPatternMatcher, PGPattern, PGSinglePatternMatcher,
    },
    PortMatcher,
};
use std::{collections::HashMap, env, fs, time::Instant};

fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 4);
    let input = fs::read_to_string(&args[1]).unwrap();
    let mut tokens = input
        .split_whitespace()
        .map(|v| v.parse::<usize>().unwrap());
    let count = tokens.next().unwrap();
    let mut graphs = Vec::new();
    let mut host_ns = 0;
    let mut prepare_ns = 0;
    for i in 0..count {
        let begin = Instant::now();
        let n = tokens.next().unwrap();
        let e = tokens.next().unwrap();
        let mut graph = PortGraph::new();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node(3, 3)).collect();
        for _ in 0..e {
            let s = tokens.next().unwrap();
            let o = tokens.next().unwrap();
            let t = tokens.next().unwrap();
            let p = tokens.next().unwrap();
            let from = graph
                .port_index(nodes[s], PortOffset::new_outgoing(o))
                .unwrap();
            let to = graph
                .port_index(nodes[t], PortOffset::new_incoming(p))
                .unwrap();
            graph.link_ports(from, to).unwrap();
        }
        if i == 0 {
            host_ns = begin.elapsed().as_nanos();
        } else {
            prepare_ns += begin.elapsed().as_nanos();
        }
        graphs.push(graph);
    }
    let begin = Instant::now();
    let patterns: Vec<_> = graphs[1..]
        .iter()
        .map(|g| PGPattern::from_host_with_root(g.clone(), NodeIndex::new(0)))
        .collect();
    prepare_ns += begin.elapsed().as_nanos();
    // Recover the public matcher's internal indexing keys in pattern vertex order.
    // This decoding setup is excluded from preparation/search in every engine.
    let keys: Vec<HashMap<PGIndexKey, Option<NodeIndex>>> = patterns
        .iter()
        .zip(&graphs[1..])
        .map(|(p, g)| {
            let m =
                PGSinglePatternMatcher::try_from_pattern::<PGIndexingScheme, _>(p.clone()).unwrap();
            let found: Vec<_> = m.find_matches(g).collect();
            assert_eq!(
                found.len(),
                1,
                "corpus patterns must have a unique self embedding"
            );
            assert_eq!(found[0].match_data.len(), g.node_count());
            found
                .into_iter()
                .next()
                .unwrap()
                .match_data
                .into_iter()
                .collect()
        })
        .collect();
    let begin = Instant::now();
    let (results, search_ns) = if args[3] == "many" {
        let m = PGManyPatternMatcher::try_from_patterns::<PGIndexingScheme>(
            patterns,
            Default::default(),
        )
        .unwrap();
        prepare_ns += begin.elapsed().as_nanos();
        unsafe {bench_profile_begin();}
        let start = Instant::now();
        let found: Vec<_> = m.find_matches(&graphs[0]).collect();
        unsafe {bench_profile_end();}
        (found, start.elapsed().as_nanos())
    } else {
        let m =
            PGNaiveManyPatternMatcher::try_from_patterns::<PGIndexingScheme, _>(patterns).unwrap();
        prepare_ns += begin.elapsed().as_nanos();
        unsafe {bench_profile_begin();}
        let start = Instant::now();
        let found: Vec<_> = m.find_matches(&graphs[0]).collect();
        unsafe {bench_profile_end();}
        (found, start.elapsed().as_nanos())
    };
    let mut output = String::new();
    for m in &results {
        let id = m.pattern.0;
        let mut mapping = vec![usize::MAX; graphs[id + 1].node_count()];
        for (key, original) in &keys[id] {
            mapping[original.unwrap().index()] = m.match_data[key].unwrap().index();
        }
        assert!(!mapping.contains(&usize::MAX));
        output.push_str(&id.to_string());
        for v in mapping {
            output.push_str(&format!(" {v}"));
        }
        output.push('\n');
    }
    fs::write(&args[2], output).unwrap();
    println!(
        "portmatching-{},{host_ns},{prepare_ns},{search_ns},{},0,0",
        args[3],
        results.len()
    );
}
