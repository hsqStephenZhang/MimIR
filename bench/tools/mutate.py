"""Inject protocol bugs into the GPU plugins' Mim sources and report which check catches each in the ring benchmarks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
import ring_trace_check  # noqa: E402

# id, file, old, new, occurrences, what the bug is
MUTATIONS = [
    ("wait_pending", "gpu.mim", "let waited = async_wait_group pending m;",
     "let waited = async_wait_group (pending + 1) m;", 1, "cp.async step waits for one group too few"),
    ("no_proxy_fence", "gpu.mim", "(waited, fence_proxy_async waited)#proxy", "(waited, waited)#proxy", 1,
     "async-proxy readers miss the proxy fence"),
    ("no_refill_barrier", "gpu.mim", "release (fence m, s, live, h, tt, r)", "release (m, s, live, h, tt, r)", 2,
     "refill without the CTA barrier"),
    ("refill_read_slot", "gpu.mim", "slots#later),", "slots#j),", 1, "refill into the slot being read"),
    ("read_other_slot", "gpu.mim", "(panels @(rid, depth, P) slots, j, here, mine)",
     "(panels @(rid, depth, P) slots, later, here, mine)", 1, "step reads another slot than it awaited"),
    ("no_settle", "gpu.mim", "return (settle m, a);", "return (m, a);", 2, "fold ends with copies in flight"),
    ("early_release", "gpu.mim", "(early, late)#(lag != 0) m;", "(early, early)#(lag != 0) m;", 3,
     "lag 1 gives a slot up before its MMAs retire"),
    ("prologue_slot", "gpu.mim", "ring#(core.idx 0 depth n)", "ring#(core.idx 0 depth (n + 1))", 1,
     "prologue fills the wrong slot"),
    ("hold_slot", "gpu.mim", "let (m, h) = hold (landed (m, n), core.idx 0 depth (n % depth));",
     "let (m, h) = hold (landed (m, n), core.idx 0 depth ((n + 1) % depth));", 1, "await vouches for the wrong slot"),
    ("wgmma_wait", "gpu.mim", "let (m, x) = wgmma_wait (m, pending, x);", "let (m, x) = wgmma_wait (m, pending + 1, x);", 1,
     "WGMMA batch leaves one more group in flight than it vouches for"),
    ("ws_release_current", "gpu.mim", "release (retired @(lag, 1) (m, now), s, live, h, tt, r);",
     "release (retired @(lag, 1) (m, now), here, tt, mine, tt, r);", 1, "ws_fold gives up the slot it just read"),
    ("full_parity", "tensor.mim", "full_parity (fill: Nat) (depth: Nat): Nat = (fill / depth) % 2;",
     "full_parity (fill: Nat) (depth: Nat): Nat = (fill / depth + 1) % 2;", 1, "full mbarrier waited with the wrong parity"),
    ("empty_parity", "tensor.mim", "empty_parity (fill: Nat) (depth: Nat): Nat = (fill / depth + 1) % 2;",
     "empty_parity (fill: Nat) (depth: Nat): Nat = (fill / depth) % 2;", 1, "empty mbarrier waited with the wrong parity"),
    ("refill_block", "tensor.mim", "ring_block (iteration, depth - 1, blocks)", "ring_block (iteration, depth - 2, blocks)",
     1, "refill loads the wrong block"),
    ("pending_formula", "tensor.mim", "pub anx lam pending (depth: Nat): Nat = depth - 2;",
     "pub anx lam pending (depth: Nat): Nat = depth - 1;", 1, "cp.async ring keeps one group too many in flight"),
]

BENCHES = ["gpu_attn/attn_law", "gpu_attn/attn_roll", "gpu_attn/attn_warp", "gpu_attn/attn_wg", "gpu_fusion/attn_warps",
           "gpu_tiling/gemm_tiled_wg_tma", "gpu_tiling/gemm_tiled_wg_ws"]

def canonicalize(text):
    """Device IR with gids, metadata ids and value names renumbered, so equal code compares equal."""
    ids = {}
    text = re.sub(r"_(\d{4,})(?!\d)", lambda m: "_" + ids.setdefault(m.group(1), str(len(ids))), text)
    text = re.sub(r"%[A-Za-z_.]*\d+", "%V", text)
    return re.sub(r"!\d+", "!N", text)


SUBST = [(r"\bHH\b", "2"), (r"\bLL\b", "128"), (r"\bDD\b", "64"), (r"\bBM\b", "64"), (r"\bBK\b", "64"),
         (r"SCALE", "0.125"), (r"\bPIPE\b", "2"), (r"\bCL\b", "2")]


def library(root, out, mutation):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lib = root / "build-debug/lib/mim"
    for f in lib.iterdir():
        if f.suffix == ".mim":
            shutil.copy(f, out / f.name)
        else:
            (out / f.name).symlink_to(f)
    if mutation:
        _, file, old, new, count, _ = mutation
        text = (out / file).read_text()
        if text.count(old) != count:
            raise ValueError(f"{mutation[0]}: expected {count} of {old!r}, found {text.count(old)}")
        (out / file).write_text(text.replace(old, new))


def compile_bench(root, lib, work, bench):
    name = bench.replace("/", "_")
    src = (root / "bench" / f"{bench}.mim").read_text()
    for pat, rep in SUBST:
        src = re.sub(pat, rep, src)
    work.mkdir(parents=True, exist_ok=True)
    (work / f"{name}.mim").write_text(src)
    cmd = [str(root / "build-debug/bin/mim"), "-P", str(lib), "-p", "opt", "-p", "gpu", "-X", "tensor:tile-fuse=1",
           "-X", "tensor:ring-coverage=1", "-X", "gpu:ring-trace=3", f"{name}.mim", "-p", "nvptx", "-X", "nvptx:sm=90a",
           "-o", "/dev/null"]
    run = subprocess.run(["bash", "-c", "ulimit -v 16000000; exec \"$@\"", "_", *cmd], cwd=work, capture_output=True,
                         text=True, timeout=900)
    (work / f"{name}.log").write_text(run.stdout + run.stderr)
    dev = work / f"{name}_dev.ll"
    ir = canonicalize(dev.read_text()) if run.returncode == 0 and dev.exists() else None
    error = next((line for line in (run.stdout + run.stderr).splitlines() if "error" in line), "")
    return name, run.returncode, ir, error


def verdict(root, work, name, code, ir, error, base):
    if code != 0:
        return "static", error.split("error:", 1)[-1].strip()[:90]
    if ir == base:
        return "same", ""
    log = work / f"{name}.log"
    try:
        checked = ring_trace_check.rings(log)
    except ValueError as e:
        return "trace", str(e).split(": ", 1)[-1][:90]
    for ring, traces in checked:
        try:
            verdict, ok = ring_trace_check.check(root / "proofs/ring", ring, traces)
        except (KeyError, ValueError) as e:
            return "trace", str(e)[:90]
        if ok is None:
            return "unchecked", verdict
        if not ok:
            return "lean", verdict[:90]
    return "MISSED", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("-j", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()

    library(root, out / "lib-base", None)
    with ThreadPoolExecutor(args.j) as pool:
        base = {n: ir for n, code, ir, _ in pool.map(lambda b: compile_bench(root, out / "lib-base", out / "base", b), BENCHES)}
    for n, ir in base.items():
        if ir is None:
            sys.exit(f"baseline {n} does not compile")

    rows = []
    for mutation in MUTATIONS:
        if args.only and mutation[0] not in args.only:
            continue
        lib = out / f"lib-{mutation[0]}"
        library(root, lib, mutation)
        work = out / mutation[0]
        with ThreadPoolExecutor(args.j) as pool:
            results = list(pool.map(lambda b: compile_bench(root, lib, work, b), BENCHES))
        for name, code, ir, error in results:
            kind, detail = verdict(root, work, name, code, ir, error, base[name])
            rows.append((mutation[0], name, kind, detail))
            print(f"{mutation[0]:20} {name:36} {kind:9} {detail}", flush=True)
    (out / "mutations.tsv").write_text("".join("\t".join(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
