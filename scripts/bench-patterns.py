#!/usr/bin/env python3
"""Collect median local pattern-search timings alongside deterministic work counters."""

import argparse
import csv
import io
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', help='path to mim-pattern-bench')
    parser.add_argument('--sizes', default='80,160,320')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    writer = None
    for size in args.sizes.split(','):
        samples = {}
        for _ in range(args.repeats):
            output = subprocess.check_output([args.binary, str(int(size))], text=True)
            for row in csv.DictReader(io.StringIO(output)):
                if row['status'] != '0':
                    raise RuntimeError(f'incomplete search: {row}')
                if int(row['matches']) != int(size):
                    raise RuntimeError(f'unexpected synthetic graph cardinality: {row}')
                samples.setdefault((row['workload'], row['mode']), []).append(row)
        for (workload, mode), rows in samples.items():
            other = samples[(workload, 'indexed' if mode == 'exhaustive' else 'exhaustive')]
            if rows[0]['matches'] != other[0]['matches']:
                raise RuntimeError(f'match counts differ: {workload}')
            result = dict(rows[0], repeats=args.repeats)
            for key in ('build_ns', 'match_ns'):
                result[key] = int(statistics.median(int(row[key]) for row in rows))
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=list(result), lineterminator='\n')
                writer.writeheader()
            writer.writerow(result)


if __name__ == '__main__':
    main()
