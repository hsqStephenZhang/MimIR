# pyright: reportMissingModuleSource=false

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence, cast

from . import _mim_core as _core
from ._mim_core import AST, Def, Driver, Lam, Level, Lit, Log, Parser, Pi, PyParser, World
from ._mim_core import *

__all__ = [
    "AST",
    "Def",
    "Driver",
    "Lam",
    "Level",
    "Lit",
    "Log",
    "Parser",
    "Pi",
    "PyParser",
    "World",
    "configure_driver",
    "emit_llvm",
    "build_native_main_i32",
    "clang_compile",
    "build_native_executable",
    "matmul_i32",
    "matrix_i32",
    "matrix_to_list",
    "plugin_search_paths",
    "run_native_executable",
    "transpose_2d",
]


def plugin_search_paths() -> list[Path]:
    core_file = _core.__file__
    assert core_file is not None
    pkg_dir = Path(core_file).resolve().parent
    package_plugins = pkg_dir / "plugins"
    if package_plugins.exists():
        return [package_plugins]

    repo_root = Path(__file__).resolve().parent.parent.parent
    fallback_candidates = [
        repo_root / "build" / "cp313-cp313-linux_x86_64" / "mim" / "plugins",
        repo_root / "build-nopy" / "lib" / "mim",
        repo_root / "build-pr300" / "lib" / "mim",
        repo_root / "build" / "cp313-cp313-linux_x86_64" / "lib" / "mim",
        repo_root / "build" / "lib" / "mim",
    ]

    return [path for path in fallback_candidates if path.exists()]


def configure_driver(driver: Driver) -> Driver:
    for path in plugin_search_paths():
        libmim_candidates = []
        if path.name == "plugins":
            libmim_candidates.append(path.parent / "libmim.so")
            libmim_candidates.append(path.parents[1] / "lib" / "libmim.so")
        else:
            libmim_candidates.append(path.parent / "libmim.so")

        for libmim in libmim_candidates:
            if libmim.exists():
                ctypes.CDLL(str(libmim), mode=ctypes.RTLD_GLOBAL)
                break
        driver.add_search_path(path)
    return driver


def build_native_main_i32(world: World, result: Def, name: str = "main") -> Lam:
    mem_t = world.call("%mem.M", world.lit_nat_0())
    argv_t = world.call("%mem.Ptr0", [world.call("%mem.Ptr0", [world.type_i8()])])
    i32_t = world.type_i32()

    if result.type().to_string() != i32_t.to_string():
        raise TypeError(f"build_native_main_i32 expects an I32 result, got {result.type().to_string()}")

    main = world.mut_fun2([mem_t, i32_t, argv_t], [mem_t, i32_t]).set(name)
    args = main.var().proj(0)
    ret = main.var().proj(1)
    mem = args.proj(0)

    main.app(False, ret, [mem, result])
    main.externalize()
    return main


def emit_llvm(driver: Driver, world: World, output_path: str | Path) -> Path:
    output = Path(output_path)
    driver.backend("ll", str(output), world)
    return output


def clang_compile(
    llvm_ir: str | Path,
    output_path: str | Path,
    *,
    clang: str = "clang",
    extra_args: Sequence[str] = (),
) -> Path:
    clang_path = shutil.which(clang)
    if clang_path is None:
        raise FileNotFoundError(f"clang executable not found: {clang}")

    llvm_path = Path(llvm_ir)
    output = Path(output_path)
    subprocess.run(
        [clang_path, str(llvm_path), "-o", str(output), "-Wno-override-module", *extra_args],
        check=True,
        text=True,
    )
    return output


def build_native_executable(
    driver: Driver,
    world: World,
    stem: str | Path,
    *,
    clang: str = "clang",
    extra_args: Sequence[str] = (),
) -> Path:
    stem_path = Path(stem)
    llvm_path = stem_path.with_suffix(".ll")
    emit_llvm(driver, world, llvm_path)
    return clang_compile(llvm_path, stem_path, clang=clang, extra_args=extra_args)


def run_native_executable(
    executable: str | Path,
    args: Sequence[str] = (),
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    executable_path = Path(executable).resolve()
    return subprocess.run(
        [str(executable_path), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _resolve_annex(self: World, callee: str | Def) -> Def:
    if isinstance(callee, str):
        return self.annex(self.sym(callee))
    return callee


def _world_call(self: World, callee: str | Def, *stages: object, implicit: bool = False) -> Def:
    target = _resolve_annex(self, callee)
    apply: Callable[[Def, list[Def]],
                    Def] = self.implicit_app if implicit else self.app

    if not stages:
        return target

    for stage in stages:
        if isinstance(stage, tuple):
            stage_args = cast(list[Def], list(stage))
        elif isinstance(stage, list):
            stage_args = cast(list[Def], stage)
        else:
            stage_args = [cast(Def, stage)]
        target = apply(target, stage_args)

    return target


def _annex_name(self: World, name: str) -> Def:
    return self.annex(self.sym(name))


setattr(World, "call", _world_call)
setattr(World, "annex_name", _annex_name)
if not hasattr(Driver, "load_plugins"):
    Driver.load_plugins = Driver.load_pluins


def matrix_i32(world: World, rows: list[list[int]]) -> Def:
    return world.tuple([
        world.tuple([world.lit_i32(value) for value in row])
        for row in rows
    ])


def matrix_to_list(matrix: Def) -> list[list[int]]:
    return [
        [matrix.proj(i).proj(j).value()
         for j in range(matrix.proj(i).num_projs())]
        for i in range(matrix.num_projs())
    ]


def transpose_2d(world: World, matrix: Def) -> Def:
    rows = matrix.num_projs()
    cols = matrix.proj(0).num_projs()
    return world.tuple([
        world.tuple([matrix.proj(i).proj(j) for i in range(rows)])
        for j in range(cols)
    ])


def matmul_i32(world: World, lhs: Def, rhs: Def) -> Def:
    lhs_rows = matrix_to_list(lhs)
    rhs_rows = matrix_to_list(rhs)

    inner = len(lhs_rows[0])
    assert inner == len(rhs_rows)

    rhs_t = list(zip(*rhs_rows))
    result = []
    for row in lhs_rows:
        result.append([sum(a * b for a, b in zip(row, col)) for col in rhs_t])

    return matrix_i32(world, result)
