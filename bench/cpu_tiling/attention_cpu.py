"""Local native CPU SDPA benchmark, with isolated and memory-bounded samples."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MASKS = {1: [0], 4: [0, 2, 4, 6], 6: [0, 2, 4, 6, 8, 10], 8: [0, 2, 4, 6, 8, 10, 12, 13]}
VARIANTS = {"mimir_default": 0, "mimir_online": 1}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def environment(threads):
    cpus = MASKS[threads]
    return dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                OPENBLAS_NUM_THREADS=str(threads), OMP_PROC_BIND="TRUE",
                GOMP_CPU_AFFINITY=" ".join(map(str, cpus)), OMP_WAIT_POLICY="PASSIVE",
                GOMP_SPINCOUNT="0", MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE",
                OMP_MAX_ACTIVE_LEVELS="1", ATEN_CPU_CAPABILITY="avx2",
                MKL_ENABLE_INSTRUCTIONS="AVX2", DNNL_MAX_CPU_ISA="AVX2",
                ONEDNN_MAX_CPU_ISA="AVX2", MKL_VERBOSE="0")


def program(case):
    h, m, s, d, n = case["shape"]
    q, k, v, out = (h, m, d), (h, s, d), (h, s, n), (h, m, n)
    ty = lambda shape: f"tensor.Tensor 3 {shape} F"
    header = "plugin tensor;\nplugin math;\nplugin core;\nlet F = math.F32;\n"
    body = ""
    op = "sdpa_masked" if case["causal"] else "sdpa"
    if case["causal"]:
        body = (f"    let mask = tensor.generate {(h, m, s)} (λ (_ i j: I64): F = "
                "core.select (core.icmp.ule (j, i), 0:F, math.neg_inf math.f32));\n")
    return (header + f"extern fun mim_attention (q: {ty(q)}, k: {ty(k)}, v: {ty(v)}): {ty(out)} =\n" + body
            + f"    return (tensor.{op} math.mode.contract @{(h, m, s, d, n)} "
            + f"({d ** -.5!r}:F, tensor.buf q, tensor.buf k, tensor.buf v"
            + (", mask" if case["causal"] else "") + "));\n")


def command_run(command, log):
    start = time.monotonic()
    with log.open("w") as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=900)
    return dict(command=command, seconds=time.monotonic() - start)


def tools(args, cases):
    import numpy as np
    import torch

    lib = Path(torch.__file__).parent / "lib"
    include = lib.parent / "include"
    args.root.mkdir(parents=True, exist_ok=True)
    command = ["clang++", "-O3", "-march=haswell", "-mno-avx512f", "-std=c++20", "-shared", "-fPIC",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
               "-I", str(include), "-I", str(include / "torch/csrc/api/include"),
               str(HERE / "attention_torch.cpp"), "-L", str(lib), "-ltorch_cpu", "-lc10",
               "-Wl,-rpath," + str(lib), "-o", str(args.root / "torch_attention.so")]
    record = command_run(command, args.root / "torch-compile.log")
    for suffix, extra in [("", []), ("-audit", ["-Dmim_par_for=mim_par_for_impl"])]:
        subprocess.run(["gcc", "-O3", "-fPIC", "-fopenmp", *extra, "-c",
                        str(SOURCE / "src/mim/plug/omp/rt/mim_omp_rt.c"),
                        "-o", str(args.root / ("omp" + suffix + ".o"))], check=True)
    subprocess.run(["clang++", "-O3", "-std=c++20", str(HERE / "attention_sample.cpp"), "-ldl",
                    "-o", str(args.root / "sample")], check=True)
    for case in cases:
        h, m, s, d, n = case["shape"]
        path = args.root / "inputs" / case["name"]
        path.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(np.random.SeedSequence([42, *case["shape"], int(case["causal"])]))
        with (path / "qkv.bin").open("wb") as raw:
            for name, shape in [("q", (h, m, d)), ("k", (h, s, d)), ("v", (h, s, n))]:
                value = rng.uniform(-1, 1, shape).astype("float32")
                np.save(path / (name + ".npy"), value)
                value.tofile(raw)
    save(args.root / "metadata.json", dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
        source_dirty=subprocess.check_output(["git", "diff", "--name-only"], cwd=SOURCE, text=True).splitlines(),
        source_sha256={p.name: digest(p) for p in [HERE / "attention_cpu.py", HERE / "attention_native.cpp",
                                                  HERE / "attention_torch.cpp", HERE / "attention_sample.cpp"]},
        compiler_sha256=digest(SOURCE / "build/bin/mim"),
        plugin_sha256={p.name: digest(p) for p in sorted((SOURCE / "build/lib/mim").glob("*.so"))},
        torch=torch.__version__, torch_config=torch.__config__.show(), clang=subprocess.check_output(
            ["clang++", "--version"], text=True), lscpu=subprocess.check_output(["lscpu"], text=True),
        torch_compile=record, torch_timer_sha256=digest(args.root / "torch_attention.so"),
        omp_runtime=str(lib / "libgomp.so.1"), cases=cases))


def prepare(args, cases):
    import resource
    resource.setrlimit(resource.RLIMIT_STACK, (64 * 1024 * 1024, resource.getrlimit(resource.RLIMIT_STACK)[1]))
    metadata = json.loads((args.root / "metadata.json").read_text())
    runtime = Path(metadata["omp_runtime"])
    root = args.root / f"t{args.threads}"
    root.mkdir(exist_ok=True)

    def compile_one(item):
        case, variant = item
        path = root / variant / case["name"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "kernel.mim").write_text(program(case))
        flags = dict(zip(["mr", "nr", "mc", "nc", "kc", "order"], [6, 16, 320, 104512, 320, 1]))
        command = [str(SOURCE / "build/bin/mim"), "-p", "opt", "-p", "omp", str(path / "kernel.mim"), "-p", "ll",
                   "-X", "tensor:strict=1", "-X", "omp:strict=1", "-X", f"omp:threads={args.threads}",
                   "-X", f"tensor:attention-online={VARIANTS[variant]}", "-X", "btensor:registers=128",
                   "-X", "ll:rt=extern", "-X", "ll:o=" + str(path / "kernel.ll")]
        for key, value in flags.items():
            command.extend(["-X", f"tensor:cpu-{key}={value}"])
        record = dict(case=case, variant=variant)
        try:
            record["lower"] = command_run(command, path / "lower.log")
            record["ir_sha256"] = digest(path / "kernel.ll")
            record["launches"] = (path / "kernel.ll").read_text().count("call void @mim_par_for")
            command = ["clang++", "-O3", "-march=haswell", "-mno-avx512f", "-std=c++20", "-shared", "-fPIC",
                       "-Wno-override-module", str(path / "kernel.ll"), str(HERE / "attention_native.cpp"),
                       str(args.root / "omp.o"), str(runtime), "-Wl,-rpath," + str(runtime.parent),
                       "-o", str(path / "kernel.so")]
            record["compile"] = command_run(command, path / "compile.log")
            record.update(status="ok", binary_sha256=digest(path / "kernel.so"))
        except Exception as error:
            record.update(status="failed", error=str(error))
        save(path / "record.json", record)
        print(json.dumps(dict(threads=args.threads, case=case["name"], variant=variant, status=record["status"])), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(compile_one, [(c, v) for c in cases for v in VARIANTS]))
    save(root / "preparation.json", records)


def validate(args, cases):
    os.sched_setaffinity(0, MASKS[args.threads])
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {MASKS[args.threads][0]})
    ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib = ctypes.CDLL(str(args.root / "torch_attention.so"))
    lib.create_attention.argtypes = [ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.int64), ptr, ptr, ptr, ptr]
    lib.create_attention.restype = ctypes.c_void_p
    lib.measure_attention.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.measure_attention.restype = ctypes.c_double
    lib.destroy_attention.argtypes = [ctypes.c_void_p]
    records = []
    for case in cases:
        path = args.root / "inputs" / case["name"]
        q, k, v = [np.load(path / (name + ".npy")) for name in ["q", "k", "v"]]
        h, m, s, d, n = case["shape"]
        score = q.astype("float64") @ k.astype("float64").swapaxes(-2, -1) * float(np.float32(d ** -.5))
        if case["causal"]:
            score = np.where(np.triu(np.ones((m, s), dtype=bool), 1), -np.inf, score)
        p = np.exp(score - score.max(axis=-1, keepdims=True))
        reference = p / p.sum(axis=-1, keepdims=True) @ v.astype("float64")
        out = np.empty((h, m, n), dtype=np.float32)
        errors = {}
        for variant in VARIANTS:
            directory = args.root / f"t{args.threads}" / variant / case["name"]
            record = json.loads((directory / "record.json").read_text())
            if record["status"] != "ok":
                continue
            kernel = ctypes.CDLL(str(directory / "kernel.so"))
            kernel.run_attention.argtypes = [ptr] * 4 + [ctypes.c_size_t]
            assert kernel.run_attention(q, k, v, out, out.size) == 0
            np.testing.assert_allclose(out, reference, rtol=1e-3, atol=1e-3)
            errors[variant] = float(np.max(abs(out - reference)))
        for kind, name in [(0, "torch_sdpa"), (1, "torch_explicit")]:
            dims = np.array([*case["shape"], case["causal"]], dtype=np.int64)
            handle = lib.create_attention(kind, dims, q, k, v, out)
            assert handle and lib.measure_attention(handle, 1) > 0
            np.testing.assert_allclose(out, reference, rtol=1e-3, atol=1e-3)
            errors[name] = float(np.max(abs(out - reference)))
            lib.destroy_attention(handle)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            F.scaled_dot_product_attention(*[torch.from_numpy(a).unsqueeze(0) for a in [q, k, v]],
                                           is_causal=case["causal"], scale=float(np.float32(d ** -.5)))
        dispatch = [event.key for event in profile.key_averages() if "scaled_dot_product" in event.key]
        records.append(dict(case=case, max_abs_error=errors, torch_dispatch=dispatch))
        print(json.dumps(records[-1]), flush=True)
    save(args.root / f"t{args.threads}" / "validation.json", records)


def run(args, cases):
    root = args.root / f"t{args.threads}"
    validation = json.loads((root / "validation.json").read_text())
    assert [row["case"] for row in validation] == cases, "Validate every case before timing"
    metadata = json.loads((args.root / "metadata.json").read_text())
    assert digest(args.root / "torch_attention.so") == metadata["torch_timer_sha256"]
    records = []
    for case in cases:
        kernels = {}
        for variant in VARIANTS:
            record = json.loads((root / variant / case["name"] / "record.json").read_text())
            if record["status"] == "ok":
                directory = root / variant / case["name"]
                assert digest(directory / "kernel.ll") == record["ir_sha256"]
                assert digest(directory / "kernel.so") == record["binary_sha256"]
                kernels[variant] = (root / variant / case["name"] / "kernel.so", 2, record["launches"])
        kernels.update(torch_sdpa=(args.root / "torch_attention.so", 0, 0),
                       torch_explicit=(args.root / "torch_attention.so", 1, 0))
        samples = {name: [] for name in kernels}
        orders = []
        rng = random.Random(42)
        commands, workers = {}, {}
        try:
            for name, (library, kind, launches) in kernels.items():
                bound = math.prod(case["shape"][:3]) * 16 if name == "mimir_default" and launches else 0
                command = [str(args.root / "sample"), str(library), str(args.root / "inputs" / case["name"] / "qkv.bin"),
                           *map(str, case["shape"]), str(int(case["causal"])), str(args.threads),
                           ",".join(map(str, MASKS[args.threads])), str(kind), str(bound)]
                commands[name] = command
                if not bound:
                    worker = subprocess.Popen([*command, "stream"], env=environment(args.threads),
                                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, text=True)
                    workers[name] = worker
                    assert json.loads(worker.stdout.readline())["ready"]
            for repeat in range(9):
                order = list(kernels)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    if name in workers:
                        worker = workers[name]
                        worker.stdin.write("sample\n")
                        worker.stdin.flush()
                        value = json.loads(worker.stdout.readline())
                    else:
                        result = subprocess.run(commands[name], env=environment(args.threads), capture_output=True,
                                                text=True, check=True, timeout=180)
                        value = json.loads(result.stdout)
                    samples[name].append(value)
        finally:
            for worker in workers.values():
                worker.stdin.close()
                worker.wait(timeout=30)
                if worker.returncode:
                    raise RuntimeError(worker.stderr.read())
        row = dict(case=case, ms={name: statistics.median(s["seconds"] for s in values) * 1000
                                 for name, values in samples.items()}, samples=samples, orders=orders)
        records.append(row)
        save(root / "results.json", dict(threads=args.threads, cpus=MASKS[args.threads], rows=records))
        print(json.dumps(dict(case=case["name"], ms=row["ms"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["tools", "prepare", "validate", "run"])
    parser.add_argument("--root", type=Path, default=Path("/tmp/mimir-attention-local"))
    parser.add_argument("--threads", type=int, choices=MASKS, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.root = args.root.resolve()
    cases = json.loads((HERE / "attention-cases.json").read_text())
    globals()[args.phase](args, cases)


if __name__ == "__main__":
    main()
