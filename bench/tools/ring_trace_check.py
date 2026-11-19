"""Check the ring events of -X gpu:ring-trace=3 -X tensor:ring-coverage=1 compiler logs with proofs/ring's checkers."""
import argparse
from pathlib import Path
import subprocess
import sys

# The mbarrier buffers in the order a ring sets them up: the full ones first, then a producer's empty ones.
BARRIERS = {"0": "full", "1": "empty"}


def rings(path):
    """Each ring's coverage fields and its traces by work-item; a failed trace raises."""
    coverage = []
    traces = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ring-coverage "):
            coverage.append(dict(field.split("=", 1) for field in line.split()[1:]))
        elif line.startswith("ring-trace: "):
            raise ValueError(f"{path}: {line}")
        elif line.startswith("ring-trace ") and "depth=" in line:
            fields = dict(field.split("=", 1) for field in line.split()[2:])
            current = traces.setdefault(int(fields.get("thread", "0")), [])
        elif line.startswith("ring-trace "):
            current.append(line.removeprefix("ring-trace "))
    if len(coverage) > 1:
        raise ValueError(f"{path}: {len(coverage)} rings in one log")
    if coverage and not traces:
        raise ValueError(f"{path}: a ring without a trace")
    return [(ring, traces) for ring in coverage]


def barrier(name):
    kind, slot = name.split(".")
    return f"{BARRIERS[kind]} {slot}"


def tma(events):
    """A TMA role's events in the checkers' format."""
    out, fill = [], None
    for event in events:
        op, *args = event.split()
        if op == "fill":
            fill = args[1], args[2]
        elif op == "init":
            out.append(f"init {barrier(args[0])} {args[1]}")
        elif op == "expect":
            out.append(f"expect_tx {barrier(args[0])} {args[1]}")
        elif op == "tma":
            if fill is None:
                raise ValueError("a TMA load outside a fill")
            out.append(f"tma_load {fill[0]} {barrier(args[0])} {fill[1]} {args[1]}")
        elif op == "mbar_wait":
            out.append(f"mbar_wait {barrier(args[0])} {args[1]}")
        elif op == "arrive":
            out.append(f"mbar_arrive {barrier(args[0])}")
        elif op in ("wgmma_fence", "wgmma_commit"):
            out.append(op)
        elif op == "read":
            out.append(f"read {args[0]} {args[1]}")
        elif op == "wgmma_wait0":
            out.append("wgmma_wait 0")
        elif op == "wgmma_wait":
            out.append(f"wgmma_wait {args[0]}")
        elif op in ("bar", "settle"):
            out.append(op)
        else:
            raise ValueError(f"an event {event!r} the TMA checkers do not know")
    return out


def check(proofs, ring, traces):
    """The checker's verdict on a ring, and whether it accepted."""
    model, depth, count = ring["model"], ring["depth"], ring.get("count", "0")
    if model == "async":
        cmd = ["ring_async_check", depth, count, "1" if ring["proxy"] == "true" else "0"]
        text = traces[0]
    elif model == "cooperative":
        groups = int(ring["consumers"])
        roles = [("leader", 0)] + [(f"consumer {g}", 128 * g) for g in range(1, groups)]
        cmd = ["ring_coop_check", depth, count]
        text = [line for role, t in roles for line in [f"role {role}", *tma(traces[t])]]
    elif model == "warp_specialized":
        groups = int(ring["consumers"])
        roles = [(f"consumer {g}", 128 * g) for g in range(groups)] + [("producer", 128 * groups)]
        cmd = ["ring_ws_check", depth, count]
        text = [line for role, t in roles for line in [f"role {role}", *tma(traces[t])]]
    else:
        return f"model={model} unchecked", None
    run = subprocess.run(["lake", "exe", *cmd], cwd=proofs, input="\n".join(text) + "\n", capture_output=True,
                         text=True)
    return (run.stdout + run.stderr).strip().splitlines()[-1], run.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proofs", type=Path, help="proofs/ring")
    parser.add_argument("logs", type=Path, nargs="+")
    args = parser.parse_args()
    failed = 0
    for log in args.logs:
        for ring, traces in rings(log):
            verdict, ok = check(args.proofs, ring, traces)
            print(log.stem, f"rid={ring['rid']}", f"model={ring['model']}", f"events={len(traces[0])}", verdict)
            failed += ok is False
    sys.exit(failed != 0)


if __name__ == "__main__":
    main()
