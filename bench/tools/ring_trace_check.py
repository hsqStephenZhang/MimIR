"""Check the ring events of -X gpu:ring-trace=1 -X tensor:ring-coverage=1 compiler logs with proofs/ring's checkers."""
import argparse
from pathlib import Path
import subprocess
import sys


def rings(path):
    coverage = []
    traces = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ring-coverage "):
            coverage.append(dict(field.split("=", 1) for field in line.split()[1:]))
        elif line.startswith("ring-trace: "):
            raise ValueError(f"{path}: {line}")
        elif line.startswith("ring-trace ") and "depth=" in line:
            traces.append([])
        elif line.startswith("ring-trace "):
            traces[-1].append(line.removeprefix("ring-trace "))
    if len(coverage) != len(traces):
        raise ValueError(f"{path}: {len(coverage)} rings but {len(traces)} traces")
    return zip(coverage, traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs", type=Path, help="proofs/ring")
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    failed = 0
    for log in args.logs:
        for ring, events in rings(log):
            if ring["model"] != "async":
                print(log.stem, f"rid={ring['rid']}", f"model={ring['model']}", "unchecked")
                continue
            proxy = "1" if ring["proxy"] == "true" else "0"
            run = subprocess.run(["lake", "exe", "ring_async_check", ring["depth"], ring["count"], proxy],
                                 cwd=args.proofs, input="\n".join(events) + "\n", capture_output=True, text=True)
            verdict = (run.stdout + run.stderr).strip().splitlines()[-1]
            print(log.stem, f"rid={ring['rid']}", "model=async", f"events={len(events)}", verdict)
            failed += run.returncode != 0
    sys.exit(failed != 0)


if __name__ == "__main__":
    main()
