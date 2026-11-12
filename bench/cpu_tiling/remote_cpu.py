"""Compile prepared MimIR IR and compare native gemm-bench timers on a remote CPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--openblas", type=Path, required=True)
    parser.add_argument("--clang", default="/usr/lib/llvm-20/bin/clang++")
    parser.add_argument("--cpu", type=int, help="Primary thread CPU (default: first selected CPU)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpus", help="Comma-separated physical CPUs (one NUMA node)")
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--omp-runtime", type=Path, help="Explicit shared OpenMP runtime; use PyTorch's libgomp")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--calibrate-once", action="store_true")
    parser.add_argument("--isa", choices=["native", "avx2"], default="native")
    parser.add_argument("--mcpu", default="icelake-server")
    parser.add_argument("--compile-cpus", help="Compilation CPU mask, before any measurements")
    parser.add_argument("--bind-openblas-workers", action="store_true")
    args = parser.parse_args()
    cpus = {int(cpu) for cpu in args.cpus.split(",")} if args.cpus else {args.cpu or 0}
    args.cpu = min(cpus) if args.cpu is None else args.cpu
    if args.cpu not in cpus:
        parser.error("--cpu must belong to --cpus")
    if args.threads < 1 or len(cpus) != args.threads:
        parser.error("Provide exactly --threads physical CPUs")
    if bool(args.runtime_source) != bool(args.omp_runtime):
        parser.error("--runtime-source and --omp-runtime must be used together")
    # Every native pool must inherit the full mask, including pools created at import.
    os.sched_setaffinity(0, cpus)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != str(args.threads):
            parser.error(f"{name}={args.threads} must be set before Python starts")
    required_env = {
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_ENABLE_INSTRUCTIONS": "AVX2",
        "OPENBLAS_CORETYPE": "Haswell",
    }
    if args.isa == "avx2":
        for name, value in required_env.items():
            if os.environ.get(name) != value:
                parser.error(f"--isa avx2 requires {name}={value} before Python starts")
    target_flags = ["-march=" + args.mcpu]
    if args.isa == "avx2":
        target_flags.append("-mno-avx512f")
    root = args.root.resolve()
    sys.path.insert(0, str(root / "harness"))
    import numpy as np
    import torch
    from common.config import load
    from common.openblas import OpenBLAS, OpenBLASKernel
    from mimir.benchmark import Kernel as MimKernel
    from pytorch.benchmark import PyTorch, Kernel as TorchKernel

    if args.isa == "avx2" and torch.backends.cpu.get_cpu_capability() != "AVX2":
        raise RuntimeError("PyTorch did not select the requested AVX2 dispatch")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # OpenMP binding can narrow its primary thread at initialization. Pthread
    # OpenBLAS workers must instead inherit the entire benchmark CPU budget.
    os.sched_setaffinity(0, cpus)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("threads", 1) != args.threads:
        raise RuntimeError("IR thread hint differs from requested runtime thread count")
    config = dict(
        benchmark=dict(device="cpu", threads=args.threads, mcpu=args.mcpu),
        artifacts={},
        openblas=dict(
            library=str(args.openblas),
            compiler=args.clang.removesuffix("++"),
            compile_flags=["-O3", "-shared", "-fPIC", "-I", str(args.openblas.parent)],
            link_flags=[str(args.openblas), "-Wl,-rpath," + str(args.openblas.parent)],
        ),
    )
    output = root / "results"
    output.mkdir(exist_ok=True)
    settings = load(root / "harness/config/pytorch.toml")
    settings["pytorch"]["compiler"] = args.clang
    blas = None

    if args.phase == "prepare":
        # Compile on cores away from the later timing core, then finish every compilation before timing.
        os.sched_setaffinity(0, {int(c) for c in args.compile_cpus.split(",")} if args.compile_cpus else set(range(2, 12)))
        if args.runtime_source:
            subprocess.run(["gcc", "-O3", "-fPIC", "-fopenmp", "-c", str(args.runtime_source),
                            "-o", str(root / "omp-runtime.o")], check=True)
        jobs = []
        for variant in manifest["variants"]:
            for case in manifest["cases"]:
                stem = "_".join(map(str, case["shape"]))
                directory = root / "ir" / variant / stem
                path = directory / "kernel.ll"
                relative = str(path.relative_to(root))
                if digest(path) != manifest["files"][relative]:
                    raise RuntimeError("IR hash mismatch: " + relative)
                jobs.append(directory)

        def compile_one(directory):
            command = [
                args.clang,
                "-O3",
                *target_flags,
                "-shared",
                "-fPIC",
                "-std=c++20",
                "-Wno-override-module",
                str(directory / "kernel.ll"),
                str(root / "mimir/native_timer.cpp"),
                *([str(root / "omp-runtime.o"), str(args.omp_runtime),
                   "-Wl,-rpath," + str(args.omp_runtime.parent)] if args.omp_runtime else []),
                "-o",
                str(directory / "kernel.so"),
            ]
            previous = directory / "compile.json"
            cached = (
                previous.is_file()
                and json.loads(previous.read_text())["command"] == command
            )
            if not (directory / "kernel.so").is_file() or not cached:
                start = time.monotonic()
                with (directory / "compile.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                save(
                    directory / "compile.json",
                    dict(command=command, seconds=time.monotonic() - start),
                )
            return str(directory.relative_to(root))

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for directory in pool.map(compile_one, jobs):
                print(json.dumps(dict(compiled=directory)), flush=True)
        pytorch = PyTorch(config, settings)
        os.sched_setaffinity(0, cpus)
        pytorch.build_timer(output)
        blas = OpenBLAS(config, output)
        metadata = dict(
            manifest=manifest,
            clang=subprocess.check_output([args.clang, "--version"], text=True),
            lscpu=subprocess.check_output(["lscpu"], text=True),
            python=sys.version,
            pytorch=pytorch.metadata,
            openblas=blas.metadata,
            openblas_sha256=digest(args.openblas.resolve()),
            rng="np.random.default_rng(np.random.SeedSequence([42,*case['shape']])); uniform(-1,1) cast float32",
            cpu=args.cpu,
            threads=args.threads,
            cpus=sorted(cpus),
            omp_runtime=str(args.omp_runtime) if args.omp_runtime else None,
            runtime_source_sha256=digest(args.runtime_source) if args.runtime_source else None,
            thread_environment={name: value for name, value in os.environ.items()
                                if name.startswith(("OMP_", "GOMP_", "MKL_", "OPENBLAS_"))},
            isa=args.isa,
            target_flags=target_flags,
            isa_environment={
                name: os.environ.get(name)
                for name in [
                    *required_env,
                    "NPY_DISABLE_CPU_FEATURES",
                    "DNNL_MAX_CPU_ISA",
                    "ONEDNN_MAX_CPU_ISA",
                ]
            },
        )
        try:
            metadata["openblas_blocks_before_gemm"] = {
                n: ctypes.c_int.in_dll(blas.blas, "sgemm_" + n).value
                for n in ["p", "q", "r"]
            }
        except ValueError:
            metadata["openblas_blocks_before_gemm"] = None
        save(output / "metadata.json", metadata)
    else:
        prepared = json.loads((output / "metadata.json").read_text())
        if prepared.get("threads", 1) != args.threads or prepared.get("cpus", [args.cpu]) != sorted(cpus):
            raise RuntimeError("Prepared thread count or affinity differs")
        if prepared.get("isa", "native") != args.isa:
            raise RuntimeError(
                "Prepared ISA differs; use a separate output root and prepare it first"
            )
        for variant in manifest["variants"]:
            for case in manifest["cases"]:
                directory = root / "ir" / variant / "_".join(map(str, case["shape"]))
                command = json.loads((directory / "compile.json").read_text())[
                    "command"
                ]
                if ("-mno-avx512f" in command) != (args.isa == "avx2"):
                    raise RuntimeError(
                        "Kernel ISA differs from requested run: " + str(directory)
                    )
        # The wrappers were built and verified during preparation; loading them does not compile anything.
        blaslib = ctypes.CDLL(str(args.openblas))
        os.sched_setaffinity(0, cpus)
        blaslib.openblas_set_num_threads(args.threads)
        if blaslib.openblas_get_num_threads() != args.threads:
            raise RuntimeError("OpenBLAS thread count differs")
        blas = object.__new__(OpenBLAS)
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        blas.library_path = output / "openblas_timer.so"
        timer = ctypes.CDLL(str(blas.library_path)).measure
        timer.argtypes = [ctypes.c_int] * 3 + [ptr] * 3 + [ctypes.c_int]
        timer.restype = ctypes.c_double
        blas.timers = {"gemm": timer}
        if any(case["op"] == "bmm" for case in manifest["cases"]):
            lib = ctypes.CDLL(str(blas.library_path))
            for op, symbol in [("bmm", "measure_bmm"), ("bmm_loop", "measure_bmm_loop")]:
                fn = getattr(lib, symbol)
                fn.argtypes = [ctypes.c_int] * 4 + [ptr] * 3 + [ctypes.c_int]
                fn.restype = ctypes.c_double
                blas.timers[op] = fn
            blas.metadata = prepared["openblas"]
        ptlib = ctypes.CDLL(str(output / "pytorch_timer.so"))
        ptlib.create_kernel.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ptr,
            ptr,
            ptr,
        ]
        ptlib.create_kernel.restype = ctypes.c_void_p
        ptlib.measure.argtypes = [ctypes.c_void_p, ctypes.c_int]
        ptlib.measure.restype = ctypes.c_double
        ptlib.destroy_kernel.argtypes = [ctypes.c_void_p]
        ptlib.destroy_kernel.restype = None
        ptlib.last_error.restype = ctypes.c_char_p

    # OpenMP workers use GOMP_CPU_AFFINITY. Keep the primary thread on the first
    # physical core after the independent pthread pool has been initialized.
    openblas_affinity = []
    if args.bind_openblas_workers:
        library = blas.blas if args.phase == "prepare" else blaslib
        for symbol in ("openblas_setaffinity", "openblas_getaffinity"):
            getattr(library, symbol).argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        for index in range(args.threads):
            cpu = args.cpu if index == args.threads - 1 else sorted(cpus - {args.cpu})[index]
            mask = (ctypes.c_ubyte * 128)()
            mask[cpu // 8] = 1 << (cpu % 8)
            if library.openblas_setaffinity(index, ctypes.sizeof(mask), mask):
                raise RuntimeError("Cannot bind OpenBLAS worker")
            actual = (ctypes.c_ubyte * 128)()
            if library.openblas_getaffinity(index, ctypes.sizeof(actual), actual) or bytes(actual) != bytes(mask):
                raise RuntimeError("OpenBLAS worker affinity differs")
            openblas_affinity.append(cpu)
    os.sched_setaffinity(0, {args.cpu})
    records = []
    for case in manifest["cases"]:
        m, n, k = case["shape"][-3:]
        batch = case["shape"][0] if case["op"] == "bmm" else None
        a_shape = (batch, m, k) if batch is not None else (m, k)
        b_shape = (batch, k, n) if batch is not None else (k, n)
        c_shape = (batch, m, n) if batch is not None else (m, n)
        stem = "_".join(map(str, case["shape"]))
        inputs = output / "inputs" / stem
        inputs.mkdir(parents=True, exist_ok=True)
        if args.phase == "prepare" and not (
            (inputs / "A.npy").is_file() and (inputs / "B.npy").is_file()
        ):
            rng = np.random.default_rng(np.random.SeedSequence([42, *case["shape"]]))
            a = rng.uniform(-1, 1, a_shape).astype("float32")
            b = rng.uniform(-1, 1, b_shape).astype("float32")
            np.save(inputs / "A.npy", a)
            np.save(inputs / "B.npy", b)
        else:
            a = np.load(inputs / "A.npy")
            b = np.load(inputs / "B.npy")
        reference = a.astype("float64") @ b.astype("float64")
        kernels = {}
        identities = {}
        for variant in manifest["variants"]:
            directory = root / "ir" / variant / stem
            binary = directory / "kernel.so"
            kernels[variant] = MimKernel(
                binary, a, b, c_shape, directory, 0, args.clang, [], []
            )
            identities[variant] = dict(binary=str(binary), sha256=digest(binary))
        kernels["openblas"] = blas.prepare(case, a, b)
        if batch is not None:
            kernels["openblas_loop"] = OpenBLASKernel(blas.timers["bmm_loop"], case, a, b, blas.library_path)
        kernels["pytorch"] = (
            pytorch.prepare(case, a, b, inputs, {})
            if args.phase == "prepare"
            else TorchKernel(ptlib, case, a, b, 0, output, {})
        )
        errors = {}
        for name, kernel in kernels.items():
            value = kernel.run()
            np.testing.assert_allclose(value, reference, rtol=0.001, atol=0.001)
            errors[name] = float(np.max(abs(value - reference)))
        record = dict(**case, max_abs_error=errors, identities=identities)
        if args.phase == "run":
            for kernel in kernels.values():
                for _ in range(5):
                    kernel.run()
            times = {name: [] for name in kernels}
            counts = {name: [] for name in kernels}
            orders = []
            calibrated_counts = {}
            if args.calibrate_once:
                for name, kernel in kernels.items():
                    kernel.measure(100, case["reps"])
                    calibrated_counts[name] = kernel.last_iterations

            def fixed_time(name, kernel, count):
                if name in ("openblas", "openblas_loop"):
                    seconds = kernel.timer(*kernel.arguments, count)
                elif name == "pytorch":
                    seconds = kernel.library.measure(kernel.handle, count)
                else:
                    seconds = kernel.time(count)
                if seconds <= 0:
                    raise RuntimeError("Native measurement failed: " + name)
                return seconds

            rng = random.Random(42)
            for _ in range(9):
                order = list(kernels)
                rng.shuffle(order)
                orders.append(order)
                for name in order:
                    kernel = kernels[name]
                    if args.calibrate_once:
                        count = calibrated_counts[name]
                        seconds = fixed_time(name, kernel, count)
                        while seconds * count < 0.1:
                            count = max(count + 1, int(0.12 / seconds))
                            seconds = fixed_time(name, kernel, count)
                        calibrated_counts[name] = kernel.last_iterations = count
                    else:
                        seconds = kernel.measure(100, case["reps"])
                    times[name].append(seconds * 1000)
                    counts[name].append(kernel.last_iterations)
            record.update(
                ms={name: statistics.median(v) for name, v in times.items()},
                batches_ms=times,
                counts=counts,
                orders=orders,
            )
        records.append(record)
        print(
            json.dumps(
                dict(
                    name=case["name"],
                    **(
                        {"ms": record["ms"]}
                        if args.phase == "run"
                        else {"max_abs_error": errors}
                    ),
                )
            ),
            flush=True,
        )
        save(
            output
            / ("paired-native.json" if args.phase == "run" else "validation.json"),
            dict(
                affinity=sorted(os.sched_getaffinity(0)),
                cpus=sorted(cpus),
                openblas_worker_cpus=openblas_affinity,
                threads=args.threads,
                warmup=5,
                repeats=9,
                min_repeat_ms=100,
                isa=args.isa,
                target_flags=target_flags,
                calibrate_once=args.calibrate_once,
                rows=records,
            ),
        )

    if args.phase == "prepare":
        metadata["openblas_worker_cpus"] = openblas_affinity
        metadata["thread_affinities_after_validation"] = {
            path.name: sorted(os.sched_getaffinity(int(path.name)))
            for path in Path("/proc/self/task").iterdir()
        }
        try:
            metadata["openblas_blocks_after_gemm"] = {
                n: ctypes.c_int.in_dll(blas.blas, "sgemm_" + n).value
                for n in ["p", "q", "r"]
            }
        except ValueError:
            metadata["openblas_blocks_after_gemm"] = None
        save(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
