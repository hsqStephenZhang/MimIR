import pathlib
import subprocess
import sys


mim, source, output = sys.argv[1:]
directory = pathlib.Path(output)
directory.mkdir(exist_ok=True)
results = []
for group in (0, 1):
    target = directory / str(group)
    target.mkdir(exist_ok=True)
    subprocess.run(
        [mim, "-p", "opt", "-p", "gpu", "-X", f"tensor:group={group}", source, "-p", "nvptx"],
        cwd=target,
        check=True,
    )
    results.append({path.name: path.read_bytes() for path in target.glob("*.ll")})
if not results[0]:
    raise RuntimeError("NVPTX compilation produced no LLVM output")
if results[0] != results[1]:
    raise RuntimeError("GPU output changed with tensor:group=1")
