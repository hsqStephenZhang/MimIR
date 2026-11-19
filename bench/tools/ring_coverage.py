"""Classify LowerToMem rings in compile_all.sh logs produced with -X tensor:ring-coverage=1."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def classify(path):
    placements = defaultdict(set)
    rings = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ring-layout "):
            group, placement = line.split()[1:]
            placements[group].add(placement.removeprefix("placement="))
        elif line.startswith("ring-coverage "):
            rings.append(dict(field.split("=", 1) for field in line.split()[1:]))
    for ring in rings:
        scope = ring["scope"]
        placement = placements.get("group=" + ring.get("group", ""), set())
        if scope == "check-layout":
            if "overlaid" in placement:
                scope = "overlaid"
            elif placement == {"disjoint"}:
                scope = "covered"
            else:
                scope = "unknown-layout"
        yield ring["rid"], scope, sorted(placement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="output directory or a single compiler log")
    path = parser.parse_args().path
    files = sorted(path.glob("*.log")) if path.is_dir() else [path]
    counts = Counter(covered=0)
    for log in files:
        for rid, scope, placement in classify(log):
            layout = ",".join(placement) or "n/a"
            print(log.stem, f"rid={rid}", f"scope={scope}", f"layout={layout}")
            counts[scope] += 1
    for scope, count in sorted(counts.items()):
        print(scope, count)
    print("total", sum(counts.values()))


if __name__ == "__main__":
    main()
