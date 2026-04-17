# pyright: reportMissingModuleSource=false

import ctypes
from pathlib import Path
from typing import Callable, cast

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
    "matmul_i32",
    "matrix_i32",
    "matrix_to_list",
    "plugin_search_paths",
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
