"""Searches a tritonbench template's schedule space: the tile rules prune it at elaboration, the rest is timed.

usage: tune.py <op> key=v1,v2,... ...   e.g. tune.py softmax M=4096 N=4096 R=1,2,4 I=4,8,16,32 P=0,1
Every combination is rendered and elaborated first; a schedule the rules reject costs no compile and no run.
The survivors run through `tb.py`, which checks them against PyTorch and times them with nsys.
"""
import itertools
import os
import re
import subprocess
import sys
import time

HERE  = os.path.dirname(os.path.abspath(__file__))
TB    = os.path.join(HERE, "..", "tritonbench")
MIMIR = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT   = os.path.join(HERE, "out")
MIM   = os.environ.get("MIM", os.path.join(MIMIR, "build-h200", "bin", "mim"))
PY    = os.environ.get("PY", os.path.expanduser("~/workspaces/tritonbench/.venv/bin/python"))

sys.path.insert(0, TB)
from tb import ENV, render  # noqa: E402


def space(args):
    keys, values = [], []
    for arg in args:
        key, vals = arg.split("=", 1)
        keys.append(key)
        values.append([int(v) if v.isdigit() else v for v in vals.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def legal(op, params):
    """Whether elaboration, where the tile rules run, accepts the schedule; the first error otherwise."""
    name = op + "_" + "_".join(f"{k}{v}" for k, v in params.items())
    path = os.path.join(OUT, f"{name}.mim")
    open(path, "w").write(render(op, params))
    log = subprocess.run([MIM, "-p", "gpu", path, "-o", "/dev/null"], capture_output=True, text=True, env=ENV)
    if log.returncode == 0:
        return True, ""
    line = next((l for l in log.stderr.splitlines() if "error:" in l), log.stderr[-200:])
    return False, line.split("error: ")[-1]


def items(op, params):
    """The work-items one work-group runs, for templates whose levels say so; beyond 1024 they run in turns."""
    if op in ("softmax", "rms_norm", "layer_norm", "cross_entropy", "swiglu") and "R" in params and "I" in params:
        n = params.get("N", params.get("V", 0))
        return params["R"] * max(1, n // params["I"])
    return 0


def timed(op, params):
    args = [f"{k}={v}" for k, v in params.items()]
    out  = subprocess.run([PY, os.path.join(TB, "tb.py"), op, *args], capture_output=True, text=True, env=ENV)
    m    = re.search(r": (OK|FAIL) \(err [^)]*\) ([0-9.]+) ms/call, ([0-9]+) kernel", out.stdout)
    if not m:
        return None, ((out.stdout + out.stderr).strip().splitlines() or ["?"])[-1][:120]
    if m.group(1) != "OK" or m.group(3) == "0":
        return None, m.group(0)
    return float(m.group(2)), m.group(0)


def main():
    op, candidates = sys.argv[1], space(sys.argv[2:])
    os.makedirs(OUT, exist_ok=True)
    start, reasons, survivors = time.time(), {}, []
    for p in candidates:
        if items(op, p) > 1024:
            # Legal, but the work-items beyond a work-group's 1024 run in turns, a fallback far off the fast path.
            reasons["more than 1024 work-items run in turns (pruned by cost, not by a rule)"] = \
                reasons.get("more than 1024 work-items run in turns (pruned by cost, not by a rule)", 0) + 1
            continue
        ok, why = legal(op, p)
        if ok:
            survivors.append(p)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    pruned = time.time() - start
    print(f"{len(candidates)} schedules, {len(survivors)} legal ({pruned:.1f} s of elaboration)")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d} rejected: {why[:110]}")
    results = []
    for p in survivors:
        ms, line = timed(op, p)
        print(f"  {p}: {f'{ms:.4f} ms' if ms is not None else line}")
        if ms is not None:
            results.append((ms, p))
    results.sort(key=lambda r: r[0])
    if results:
        best, worst = results[0], results[-1]
        print(f"best {best[1]}: {best[0]:.4f} ms; slowest {worst[0]:.4f} ms ({worst[0] / best[0]:.1f}x)")
    print(f"{len(results)} of {len(survivors)} legal schedules ran correctly, {time.time() - start:.0f} s in all")


if __name__ == "__main__":
    main()
