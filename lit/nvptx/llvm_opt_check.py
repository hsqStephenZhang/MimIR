import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


TOOLS = ("llvm-link", "opt", "llc", "ptxas", "fatbinary")


def fake_tool(tool, args):
    with open(os.environ["NVPTX_TOOL_LOG"], "a") as log:
        log.write(json.dumps([tool, *args]) + "\n")
    if tool == "fatbinary":
        output = next(arg.removeprefix("--create=") for arg in args if arg.startswith("--create="))
    else:
        output = args[args.index("-o") + 1]
    Path(output).write_bytes(b"fake device artifact\n")


def check(args):
    source = Path(args.source).resolve()
    stem = source.stem
    with tempfile.TemporaryDirectory(prefix="mim-nvptx-opt-") as tmp:
        root = Path(tmp)
        tools = root / "bin"
        tools.mkdir()
        # Only the external tool boundary is mocked; MimIR emits both LLVM modules normally.
        for tool in TOOLS:
            wrapper = tools / tool
            wrapper.write_text(
                f"#!{sys.executable}\nimport runpy, sys\n"
                f"runpy.run_path({str(Path(__file__).resolve())!r})['fake_tool']({tool!r}, sys.argv[1:])\n"
            )
            wrapper.chmod(0o755)
        libdevice = root / "libdevice.bc"
        if args.libdevice:
            libdevice.write_bytes(b"fake libdevice\n")
        # The no-libdevice case intentionally passes a nonexistent library path.
        for mode in ("default", "custom", "no-embed"):
            work = root / mode
            work.mkdir()
            log = work / "tools.jsonl"
            env = dict(
                os.environ,
                PATH=str(tools) + os.pathsep + os.environ["PATH"],
                NVPTX_TOOL_LOG=str(log),
            )
            flags = [
                "-Xnvptx:embed", "-Xnvptx:sm=75", "-Xnvptx:rt=extern",
                f"-Xnvptx:libdevice={libdevice}",
            ]
            if mode == "custom":
                flags.append("-Xnvptx:Xopt=-passes=verify")
            if mode == "no-embed":
                flags.append("-Xnvptx:no-embed")
            subprocess.run([*args.mim, str(source), "-p", "nvptx", *flags], cwd=work, env=env, check=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            assert (work / f"{stem}.ll").exists()
            assert (work / f"{stem}_dev.ll").exists()
            if mode == "no-embed":
                assert calls == [], calls
                assert not (work / f"{stem}_dev_opt.bc").exists()
                continue
            expected_tools = (["llvm-link"] if args.libdevice else []) + ["opt", "llc", "ptxas", "fatbinary"]
            assert [call[0] for call in calls] == expected_tools, calls
            if args.libdevice:
                assert calls.pop(0) == [
                    "llvm-link", f"{stem}_dev.ll", str(libdevice), "-o", f"{stem}_dev_raw.bc",
                ]
            passes = "verify" if mode == "custom" else "default<O2>,nvvm-reflect"
            opt_input = f"{stem}_dev_raw.bc" if args.libdevice else f"{stem}_dev.ll"
            assert calls[0] == [
                "opt", "-mtriple=nvptx64-nvidia-cuda", f"-passes={passes}",
                opt_input, "-o", f"{stem}_dev_opt.bc",
            ], calls[0]
            assert calls[1] == [
                "llc", "-march=nvptx64", "-mcpu=sm_75", f"{stem}_dev_opt.bc",
                "-o", f"{stem}_dev.ptx",
            ], calls[1]
            assert (work / f"{stem}_dev_opt.bc").exists()
            assert (work / f"{stem}_dev.fatbin").exists()
    print(f"{source.name}: default optimization, Xopt override and no-embed passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libdevice", action="store_true")
    parser.add_argument("source")
    parser.add_argument("mim", nargs="+", help="MimIR command, after --")
    check(parser.parse_args())
