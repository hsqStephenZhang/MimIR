import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts"
OUTPUT = ARTIFACTS / "math-toolchain"


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def checkout(spec, source):
    if not source.exists():
        sparse = ["--sparse"] if "component" in spec else []
        run("git", "clone", "--depth", "1", "--filter=blob:none", *sparse,
            "--branch", spec["ref"], spec["repository"], source)
        if "component" in spec:
            run("git", "-C", source, "sparse-checkout", "set", spec["component"])
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["commit"]:
        raise RuntimeError(f"Expected {spec['commit']}, found {revision}; use a clean dependency checkout")
    run("git", "-C", source, "diff", "HEAD", "--exit-code", "--", spec.get("component", "."))
    return revision


def main():
    spec = json.loads((HERE / "sources.json").read_text())
    source = ARTIFACTS / "llvm-project"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    revision = checkout(spec, source)
    openlibm = ARTIFACTS / "openlibm"
    openlibm_revision = checkout(spec["openlibm"], openlibm)
    compiler = shutil.which("clang++")
    if not compiler:
        raise RuntimeError("clang++ is required")
    flags = shlex.split(subprocess.check_output(
        ["llvm-config", "--cxxflags", "--ldflags", "--libs", "core", "codegen", "passes"], text=True
    ))
    run(compiler, "-shared", "-fPIC", *flags, HERE / "lower_math.cpp", "-o", OUTPUT / "lower_math.so")
    objects = []
    for name in spec["entrypoints"]:
        target = OUTPUT / f"{name}.bc"
        implementation = source / f"libc/src/math/generic/{name}.cpp"
        if name == "pow":
            patched = OUTPUT / "pow.cpp"
            run("patch", "--batch", "--fuzz=0", "-o", patched, implementation, HERE / spec["patches"][0])
            implementation = patched
        run(compiler, "--target=nvptx64-nvidia-cuda", "-march=sm_75", "-nogpulib", "-ffreestanding",
            "-fno-exceptions", "-fno-rtti", "-fno-math-errno", "-std=c++17", "-O2", "-emit-llvm", "-c",
            "-DLIBC_NAMESPACE=__llvm_libc", "-DLIBC_COPT_PUBLIC_PACKAGING", "-DLIBC_FULL_BUILD",
            "-I", source / "libc", "-I", source / "libc/include",
            implementation, "-o", target)
        objects.append(target)
    target = OUTPUT / "erf.bc"
    run(compiler, "-x", "c", "--target=nvptx64-nvidia-cuda", "-march=sm_75", "-nogpulib",
        "-std=c99", "-ffreestanding", "-fno-math-errno", "-ffp-contract=off",
        "-D__weak_reference(sym,alias)=", "-O2", "-emit-llvm", "-c",
        "-I", openlibm / "include", "-I", openlibm / "src",
        openlibm / spec["openlibm"]["source"], "-o", target)
    objects.append(target)
    run("llvm-link", *objects, "-o", OUTPUT / "runtime.bc")
    undefined = subprocess.check_output(["llvm-nm", "--undefined-only", OUTPUT / "runtime.bc"], text=True)
    if undefined.strip():
        raise RuntimeError(f"Software math runtime has unresolved dependencies:\n{undefined}")
    dependencies = subprocess.check_output(["llvm-dis", OUTPUT / "runtime.bc", "-o", "-"], text=True)
    if "__nv_" in dependencies:
        raise RuntimeError("Software math runtime must not depend on libdevice")
    plugin = f"-load-pass-plugin={OUTPUT / 'lower_math.so'}"
    run("opt", plugin, "-passes=mim-nvptx-math,verify", "-S", HERE / "pow.ll", "-o", OUTPUT / "lowered.ll")
    lowered = (OUTPUT / "lowered.ll").read_text()
    if (not re.search(r"call float @powf\(", lowered) or not re.search(r"call double @pow\(", lowered)
            or re.search(r"call (float|double) @llvm\.pow\.", lowered)):
        raise RuntimeError("LLVM intrinsic lowering did not produce the expected ordinary powf call")
    run("llvm-link", HERE / "pow.ll", OUTPUT / "runtime.bc", "-o", OUTPUT / "probe.bc")
    run("opt", plugin, "-passes=default<O2>,mim-nvptx-math,cgscc(inline),globaldce,nvvm-reflect",
        OUTPUT / "probe.bc", "-o", OUTPUT / "probe-opt.bc")
    run("llc", "-march=nvptx64", "-mcpu=sm_75", OUTPUT / "probe-opt.bc", "-o", OUTPUT / "probe.ptx")
    run("ptxas", "-arch=sm_75", OUTPUT / "probe.ptx", "-o", OUTPUT / "probe.cubin")
    if "__nv_" in (OUTPUT / "probe.ptx").read_text():
        raise RuntimeError("Standalone pow probe must not reference libdevice")
    files = [HERE / "sources.json", HERE / "build.py", HERE / "lower_math.cpp", HERE / "pow.ll",
             OUTPUT / "lower_math.so", OUTPUT / "runtime.bc"]
    files += [HERE / name for name in spec["patches"]]
    manifest = {
        "llvm_commit": revision,
        "openlibm_commit": openlibm_revision,
        "compiler": subprocess.check_output([compiler, "--version"], text=True).strip(),
        "sha256": {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in files},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Non-libdevice math toolchain ready: {OUTPUT}")


if __name__ == "__main__":
    main()
