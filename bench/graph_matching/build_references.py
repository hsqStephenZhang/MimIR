#!/usr/bin/env python3
"""Build external reference runners outside their read-only checkouts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--portmatching', required=True, type=Path)
    p.add_argument('--ghl', required=True, type=Path)
    p.add_argument('--build', required=True, type=Path)
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    root = args.build.resolve()
    crate = root / 'port'
    (crate / 'src').mkdir(parents=True, exist_ok=True)
    manifest = ('[package]\nname="mimir-port-bench"\nversion="0.0.0"\nedition="2021"\n'
                '[dependencies]\nportmatching={path=' + json.dumps(str(args.portmatching.resolve())) +
                ',features=["portgraph"]}\nportgraph="0.13"\n')
    (crate / 'Cargo.toml').write_text(manifest)
    shutil.copyfile(source / 'portmatching.Cargo.lock', crate / 'Cargo.lock')
    shutil.copyfile(source / 'portmatching.rs', crate / 'src/main.rs')
    subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path', str(crate / 'Cargo.toml')] +
                   (['--offline'] if args.offline else []), check=True)
    # Extract only the VF3 addition. Other changes in the patch are unrelated to this runner.
    patch = (args.ghl / 'vf3_patch.patch').read_text().split('diff --git ')[1]
    header = root / 'boost/boost/graph/vf3_sub_graph_iso.hpp'
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(''.join(line[1:] + '\n' for line in patch.splitlines()
                              if line.startswith('+') and not line.startswith('+++')))
    subprocess.run(['c++', '-O3', '-DNDEBUG', '-std=c++23', '-I' + str(root / 'boost'),
                    '-I' + str(args.ghl.resolve() / 'include'), str(source / 'ghl.cpp'),
                    '-o', str(root / 'ghl-bench')], check=True)


if __name__ == '__main__':
    main()
