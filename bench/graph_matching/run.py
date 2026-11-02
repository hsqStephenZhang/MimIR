#!/usr/bin/env python3
"""Compare exact port-DAG embeddings, then collect process-local median timings."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import subprocess


def chain(n):
    return n, [(i, 0, i + 1, 0) for i in range(n - 1)]


def copies(graph, count):
    n, edges = graph
    return n * count, [(s + i * n, o, t + i * n, p)
                       for i in range(count) for s, o, t, p in edges]


def oracle(host, pattern):
    """Small independent injective positive-edge matcher; no compiler/library API."""
    hn, he = host
    pn, pe = pattern
    incoming = {(t, p): (s, o) for s, o, t, p in he}
    outgoing = {(s, o): (t, p) for s, o, t, p in he}
    bound = {}
    used = set()
    results = set()

    def visit():
        if len(bound) == pn:
            results.add(tuple(bound[i] for i in range(pn)))
            return
        v = max((v for v in range(pn) if v not in bound), key=lambda v: (
            sum((s == v and t in bound) or (t == v and s in bound) for s, _, t, _ in pe),
            sum(s == v or t == v for s, _, t, _ in pe), v))
        candidates = None
        for s, o, t, p in pe:
            if s == v and t in bound:
                endpoint = incoming.get((bound[t], p))
                allowed = {endpoint[0]} if endpoint and endpoint[1] == o else set()
            elif t == v and s in bound:
                endpoint = outgoing.get((bound[s], o))
                allowed = {endpoint[0]} if endpoint and endpoint[1] == p else set()
            else:
                continue
            candidates = allowed if candidates is None else candidates & allowed
        for candidate in range(hn) if candidates is None else candidates:
            if candidate not in used:
                bound[v] = candidate
                used.add(candidate)
                visit()
                used.remove(candidate)
                del bound[v]

    visit()
    return results


def write_corpus(path, graphs):
    lines = [str(len(graphs))]
    for n, edges in graphs:
        lines.append(f'{n} {len(edges)}')
        lines.extend(' '.join(map(str, e)) for e in edges)
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mimir', 'ghl', 'portmatching'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--components', default='8,32,128')
    parser.add_argument('--repeats', default=5, type=int)
    parser.add_argument('--mimir-modes', default='indexed,exhaustive,flat-indexed,flat-exhaustive')
    parser.add_argument('--baseline', type=Path, help='optional frozen original MimIR runner (indexed mode)')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    complex_pattern = (4, [(0, 0, 2, 1), (1, 1, 2, 0), (2, 0, 3, 1), (1, 2, 3, 2)])
    broken = (4, complex_pattern[1][:-1] + [(1, 0, 3, 2)])
    overlap = (100, [(vertices[s], o, vertices[t], p)
                     for vertices in ((0, 10, 30, 55), (3, 12, 23, 44), (12, 55, 98, 99))
                     for s, o, t, p in complex_pattern[1]])
    cases = [('overlap', 1, overlap, [complex_pattern])]
    for count in map(int, args.components.split(',')):
        cases.extend([
            ('chain', count, copies(chain(12), count), [chain(4)]),
            ('shared-dag', count, copies(complex_pattern, count), [complex_pattern]),
            ('wrong-port', count, copies(broken, count), [complex_pattern]),
            ('many-chain', count, copies(chain(12), count), [chain(n) for n in range(2, 9)]),
        ])
    engines = [(args.mimir, mode) for mode in args.mimir_modes.split(',')]
    if args.baseline:
        engines.insert(0, (args.baseline, 'indexed'))
    engines += [(args.ghl, 'vf2'), (args.ghl, 'vf3'),
               (args.portmatching, 'naive'), (args.portmatching, 'many')]
    samples = []
    for workload, count, host, patterns in cases:
        case = f'{workload}-{count}'
        corpus = args.output / (case + '.txt')
        write_corpus(corpus, [host] + patterns)
        expected = {(i,) + mapping for i, pattern in enumerate(patterns) for mapping in oracle(host, pattern)}
        # GHL uses induced matching. Verify this corpus does not exercise the difference.
        he = set(host[1])
        for mapping in expected:
            pattern = patterns[mapping[0]]
            vertices = mapping[1:]
            selected = set(vertices)
            assert sum(s in selected and t in selected for s, _, t, _ in he) == len(pattern[1])
        for binary, mode in engines:
            for repeat in range(args.repeats):
                mappings = args.output / (case + '-' + binary.name + '-' + mode + '.matches')
                output = subprocess.check_output([str(binary.resolve()), str(corpus.resolve()),
                                                  str(mappings.resolve()), mode], text=True, timeout=60)
                fields = output.strip().split(',')
                if len(fields) != 7:
                    raise RuntimeError(output)
                engine = fields[0]
                if args.baseline and binary == args.baseline:
                    engine = 'baseline-' + engine
                host_ns, prepare_ns, search_ns, matches, candidates, steps = map(int, fields[1:])
                actual = [tuple(map(int, line.split())) for line in mappings.read_text().splitlines()]
                if len(actual) != matches or len(set(actual)) != matches or set(actual) != expected:
                    missing, extra = expected - set(actual), set(actual) - expected
                    raise RuntimeError(f'{case}/{engine}: missing={list(missing)[:3]}, extra={list(extra)[:3]}, '
                                       f'expected={len(expected)}, got={matches}, unique={len(set(actual))}')
                samples.append(dict(workload=workload, components=count, nodes=host[0], patterns=len(patterns),
                                    engine=engine, repeat=repeat, matches=matches, host_ns=host_ns,
                                    prepare_ns=prepare_ns, search_ns=search_ns,
                                    ready_search_ns=prepare_ns + search_ns,
                                    total_ns=host_ns + prepare_ns + search_ns, candidates=candidates, steps=steps))
            print(f'{case}: {engine}, {len(expected)} verified mappings', flush=True)
        # Keep complete mappings for reproducibility; CSV does not approximate equality by a hash.
    def write_csv(path, rows):
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    write_csv(args.output / 'samples.csv', samples)
    medians = []
    samples.sort(key=lambda r: (r['workload'], r['components'], r['engine']))
    for _, rows in itertools.groupby(samples, key=lambda r: (r['workload'], r['components'], r['engine'])):
        rows = list(rows)
        result = dict(rows[0])
        del result['repeat']
        result['repeats'] = args.repeats
        for key in ('host_ns', 'prepare_ns', 'search_ns', 'ready_search_ns', 'total_ns'):
            result[key] = int(statistics.median(r[key] for r in rows))
        medians.append(result)
    write_csv(args.output / 'medians.csv', medians)
    (args.output / 'invocation.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')


if __name__ == '__main__':
    main()
