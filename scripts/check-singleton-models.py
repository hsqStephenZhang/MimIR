#!/usr/bin/env python3
"""Compile the Task 22A standalone model reproducers and check scalar references without timing."""

import argparse
import ctypes
import json
import math
import pathlib
import re
import subprocess


root = pathlib.Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reproducers", type=pathlib.Path)
parser.add_argument("--output", type=pathlib.Path, default=root / "build/task22-model-check")
args = parser.parse_args()
directory = args.output.resolve()
directory.mkdir(parents=True, exist_ok=True)
scalar = ctypes.c_float
pointer = ctypes.POINTER(scalar)
libc = ctypes.CDLL(None)
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None


def linear(x, weights, bias, rows, inputs, outputs):
    return [
        sum(x[i * inputs + k] * weights[j * inputs + k] for k in range(inputs)) + bias[j]
        for i in range(rows) for j in range(outputs)
    ]


def convolution(x, weights, bias):
    result = []
    for channel in range(2):
        for y in range(5):
            for x_coord in range(5):
                value = bias[channel]
                for source in range(2):
                    for ky in range(3):
                        for kx in range(3):
                            iy, ix = y + ky - 1, x_coord + kx - 1
                            if 0 <= iy < 5 and 0 <= ix < 5:
                                value += x[(source * 5 + iy) * 5 + ix] * weights[((channel * 2 + source) * 3 + ky) * 3 + kx]
                result.append(value)
    return result


def normalize(x, weights, bias):
    result = []
    for i in range(3):
        row = x[i * 8:(i + 1) * 8]
        mean = sum(row) / 8
        variance = sum((value - mean) ** 2 for value in row) / 8
        result.extend((value - mean) / math.sqrt(variance + 0.00001) * weights[j] + bias[j]
                      for j, value in enumerate(row))
    return result


def reference(name, values):
    x, *p = values
    if name.startswith("resnet"):
        hidden = [max(0, v) for v in convolution(x, p[0], p[1])]
        output = convolution(hidden, p[2], p[3])
        return [max(0, v + residual) for v, residual in zip(output, x)]
    qkv = linear(x, p[0], p[1], 3, 8, 24)
    context = [0.0] * 24
    for head in range(2):
        for i in range(3):
            scores = [
                sum(qkv[i * 24 + head * 4 + k] * qkv[j * 24 + 8 + head * 4 + k] for k in range(4)) * 0.5
                for j in range(3)
            ]
            weights = [math.exp(value - max(scores)) for value in scores]
            denominator = sum(weights)
            for k in range(4):
                context[i * 8 + head * 4 + k] = sum(
                    weights[j] / denominator * qkv[j * 24 + 16 + head * 4 + k] for j in range(3)
                )
    output = linear(context, p[2], p[3], 3, 8, 8)
    if name.startswith("attention"):
        return output
    hidden = normalize([a + b for a, b in zip(x, output)], p[4], p[5])
    expanded = [max(0, v) for v in linear(hidden, p[6], p[7], 3, 8, 12)]
    projected = linear(expanded, p[8], p[9], 3, 12, 8)
    return normalize([a + b for a, b in zip(hidden, projected)], p[10], p[11])


print("reproducer,group,trials,max_abs_error")
for name in ("resnet_mimir_group0", "resnet_mimir_group1", "attention_mimir_group1", "transformer_mimir_group1"):
    source = args.reproducers.resolve() / name / "kernel.mim"
    text = source.read_text()
    signature = text.split("extern fun mim_model (", 1)[1].split("):", 1)[0]
    sizes = [math.prod(int(axis.strip()) for axis in dims.split(","))
             for dims in re.findall(r": «([^;]+); F32»", signature)]
    if not sizes:
        raise ValueError(f"No model arguments found in {source}")
    for group in (0, 1):
        output = directory / f"{name}-g{group}"
        command = json.loads((source.parent / "mimir_command.json").read_text())
        command[0] = str(root / "build/bin/mim")
        command[command.index("-P") + 1] = str(root / "build/lib/mim")
        command[command.index("-X") + 1] = f"tensor:group={group}"
        command[command.index("-p") + 2] = str(source)
        command[command.index("--output-mim") + 1] = str(output.with_suffix(".mim"))
        command += ["-X", f"ll:o={output.with_suffix('.ll')}"]
        output.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(command, cwd=directory, stdout=log, stderr=log, check=True)
        library = output.with_suffix(".dylib")
        subprocess.run(
            ["/opt/homebrew/opt/llvm/bin/clang++", "-O3", "-mcpu=native", "-dynamiclib", "-Wno-override-module",
             str(output.with_suffix(".ll")), "-o", str(library)], check=True,
        )
        module = ctypes.CDLL(str(library))
        function = module.mim_model
        function.argtypes = [pointer] * len(sizes)
        function.restype = pointer
        maximum_error = 0.0
        for trial in range(3):
            values, buffers = [], []
            for arg, size in enumerate(sizes):
                data = [float((i * (arg + 3) + trial * 7) % 23 - 11) / 32 for i in range(size)]
                if name.startswith("transformer") and arg in (5, 11):
                    data = [value + 1 for value in data]
                if trial == 2:
                    data = [0.0 if arg == 0 else value for value in data]
                buffer = (scalar * size)(*data)
                buffers.append(buffer)
                values.append(list(buffer))
            expected = reference(name, values)
            actual = function(*buffers)
            if not actual:
                raise RuntimeError(f"{name} group={group}: null output")
            try:
                for i, wanted in enumerate(expected):
                    got = actual[i]
                    error = abs(got - wanted)
                    if not math.isfinite(got) or error > 0.00002 + 0.0002 * abs(wanted):
                        raise AssertionError(f"{name} group={group} trial={trial} index={i}: {got} != {wanted}")
                    maximum_error = max(maximum_error, error)
                for buffer, before in zip(buffers, values):
                    if list(buffer) != before:
                        raise AssertionError(f"{name} group={group}: input modified")
            finally:
                libc.free(actual)
        print(f"{name},{group},3,{maximum_error:.9g}", flush=True)
