import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mim

if TYPE_CHECKING:
    from mim._mim_core import AST, Def, Driver, Lam, Level, Lit, Log, Parser, PyParser, World
else:
    AST = mim.AST
    Def = mim.Def
    Driver = mim.Driver
    Lam = mim.Lam
    Level = mim.Level
    Lit = mim.Lit
    Log = mim.Log
    Parser = mim.Parser
    PyParser = mim.PyParser
    World = mim.World


def _run_in_clean_python(script: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd="/tmp",
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_driver_log_and_world_access() -> None:
    driver = Driver()
    log = driver.log().set_stdout().set(Level.Debug)
    world = driver.world()

    assert isinstance(log, Log)
    assert isinstance(world, World)
    assert world is driver.world()


def test_world_builds_basic_mutable_function() -> None:
    world = Driver().world()

    lam = world.mut_fun2([world.type_i32()], [world.type_i32()])
    assert isinstance(lam, Lam)

    var = lam.var()
    assert isinstance(var, Def)
    assert var.num_projs() == 2

    args = var.proj(0)
    ret = var.proj(1)
    assert isinstance(args, Def)
    assert isinstance(ret, Def)

    assert len(var.projs(2)) == 2
    assert len(args.projs(1)) == 1

    assert lam.set("py_test_main") is lam
    assert lam.externalize() is None


def test_world_primitive_constructors() -> None:
    world = Driver().world()

    top_nat = world.top_nat()
    idx_type = world.type_idx(top_nat)
    zero = world.lit_nat_0()
    lit_i8 = world.lit_i8(65)
    arr = world.arr(top_nat, world.type_i8())
    cn = world.cn([world.type_i32()])

    assert isinstance(top_nat, Def)
    assert isinstance(idx_type, Def)
    assert isinstance(zero, Lit)
    assert isinstance(lit_i8, Lit)
    assert isinstance(arr, Def)
    assert isinstance(cn, Def)


def test_world_builds_mutable_continuation() -> None:
    world = Driver().world()

    lam = world.mut_con([world.type_i32()])

    assert isinstance(lam, Lam)
    assert isinstance(lam.var(), Def)
    assert lam.var().num_projs() == 1
    assert lam.set("py_test_cont") is lam


def test_ast_and_parser_classes_construct() -> None:
    world = Driver().world()
    ast = AST(world)
    parser = Parser(ast)
    py_parser = PyParser(parser)

    assert isinstance(ast, AST)
    assert isinstance(parser, Parser)
    assert isinstance(py_parser, PyParser)


def test_matrix_transpose_helper() -> None:
    world = Driver().world()

    matrix = mim.matrix_i32(world, [[1, 2, 3], [4, 5, 6]])
    transposed = mim.transpose_2d(world, matrix)

    assert mim.matrix_to_list(matrix) == [[1, 2, 3], [4, 5, 6]]
    assert mim.matrix_to_list(transposed) == [[1, 4], [2, 5], [3, 6]]


def test_matrix_matmul_helper() -> None:
    world = Driver().world()

    lhs = mim.matrix_i32(world, [[1, 2], [3, 4]])
    rhs = mim.matrix_i32(world, [[5, 6], [7, 8]])
    product = mim.matmul_i32(world, lhs, rhs)

    assert mim.matrix_to_list(product) == [[19, 22], [43, 50]]


def test_installed_package_loads_plugins_from_clean_cwd() -> None:
    script = """
import mim

d = mim.configure_driver(mim.Driver())
d.load_plugins(['core', 'mem', 'math', 'matrix', 'tensor'])
print(mim.__file__)
print(mim.plugin_search_paths()[0])
print('loaded')
"""
    lines = _run_in_clean_python(script)
    assert lines[-1] == "loaded"
    assert lines[0].endswith("python/mim/__init__.py") or "site-packages/mim/__init__.py" in lines[0]
    assert lines[1].endswith("build/cp313-cp313-linux_x86_64/mim/plugins") or "site-packages/mim/plugins" in lines[1]


def test_matrix_plugin_normalizes_shape_and_read() -> None:
    lines = _run_in_clean_python(
        """
import mim

driver = mim.configure_driver(mim.Driver())
driver.load_plugins(['core', 'mem', 'math', 'matrix'])
world = driver.world()

mem = world.bot(world.call('%mem.M', world.lit_nat_0()))
matrix_type = world.tuple([world.lit_nat(2), world.tuple([world.lit_nat(3), world.lit_nat(5)]), world.type_i32()])
matrix = world.call('%matrix.constMat', matrix_type, [mem, world.lit_i32(5)])

shape_0 = world.call('%matrix.shape', matrix_type, [matrix.proj(1), world.lit_idx(2, 0)])
read_00 = world.call(
    '%matrix.read',
    matrix_type,
    [mem, matrix.proj(1), world.tuple([world.lit_idx(3, 0), world.lit_idx(5, 0)])],
)

print(shape_0.value())
read_00.dump()
print(read_00.proj(1).value())
"""
    )

    assert sorted(lines) == sorted(["3", "(⊥:(%mem.M 0), 5I32)", "5"])


def test_axiom_wrappers_list_and_call_plugins() -> None:
    axioms = mim.list_axioms()

    assert "%mem.Ptr0" in axioms["mem"]
    assert "%matrix.read" in axioms["matrix"]
    assert "%core.wrap.add" in axioms["core"]

    driver = mim.configure_driver(Driver())
    driver.load_plugins(["core", "mem", "matrix"])
    world = driver.world()

    mem_t = mim.Mem.M(world, world.lit_nat_0())
    ptr_t = mim.Mem.Ptr0(world, world.type_i8())

    mem = world.bot(mem_t)
    matrix_type = world.tuple([world.lit_nat(2), world.tuple([world.lit_nat(3), world.lit_nat(5)]), world.type_i32()])
    matrix = mim.Matrix.constMat(world, matrix_type, [mem, world.lit_i32(5)])
    shape_0 = mim.Matrix.shape(world, matrix_type, [matrix.proj(1), world.lit_idx(2, 0)])

    assert mem_t.to_string().strip() == "%mem.M 0"
    assert ptr_t.to_string().strip() == "%mem.Ptr (I8, 0)"
    assert shape_0.value() == 3


def test_make_driver_and_bound_axioms() -> None:
    driver = mim.make_driver("core", "mem", "matrix", log_level=Level.Info)
    world = driver.world()
    ax = mim.bind(world)

    mem = world.bot(ax.Mem.M(world.lit_nat_0()))
    matrix_type = world.tuple([world.lit_nat(2), world.tuple([world.lit_nat(3), world.lit_nat(5)]), world.type_i32()])
    matrix = ax.Matrix.constMat(matrix_type, [mem, world.lit_i32(5)])
    shape_0 = ax.Matrix.shape(matrix_type, [matrix.proj(1), world.lit_idx(2, 0)])

    assert shape_0.value() == 3
    assert ax.Matrix.read.symbol == "%matrix.read"


def test_tensor_plugin_transpose_builds_expected_type() -> None:
    lines = _run_in_clean_python(
        """
import mim

driver = mim.configure_driver(mim.Driver())
driver.load_plugins(['core', 'tensor'])
world = driver.world()

tensor = world.tuple([
    world.tuple([world.lit_i32(1), world.lit_i32(2), world.lit_i32(3)]),
    world.tuple([world.lit_i32(4), world.lit_i32(5), world.lit_i32(6)]),
])
permutation = world.tuple([world.lit_idx(2, 1), world.lit_idx(2, 0)])
shape = world.tuple([world.lit_nat(2), world.lit_nat(3)])

callee = world.app(world.annex_name('%tensor.transpose'), [world.type_i32()])
callee = world.app(callee, [world.lit_nat(2), shape])
transposed = world.app(callee, [tensor, permutation])

transposed.dump()
"""
    )

    assert lines == ["%tensor.transpose I32 (2, (2, 3)) (((1I32, 2I32, 3I32), (4I32, 5I32, 6I32)), (tt, ff))"]


def test_backend_helpers_emit_llvm_and_optionally_compile() -> None:
    driver = mim.configure_driver(Driver())
    driver.load_plugins(["core", "mem", "math", "matrix", "tensor"])
    world = driver.world()

    mem = world.bot(world.call("%mem.M", world.lit_nat_0()))
    matrix_type = world.tuple([world.lit_nat(2), world.tuple([world.lit_nat(3), world.lit_nat(5)]), world.type_i32()])
    matrix = world.call("%matrix.constMat", matrix_type, [mem, world.lit_i32(5)])
    read_00 = world.call(
        "%matrix.read",
        matrix_type,
        [mem, matrix.proj(1), world.tuple([world.lit_idx(3, 0), world.lit_idx(5, 0)])],
    )

    mim.build_native_main_i32(world, read_00.proj(1))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        ll_path = mim.emit_llvm(driver, world, tmp / "shape_main.ll")
        assert ll_path.exists()
        assert "define" in ll_path.read_text()

        if shutil.which("clang") is not None:
            exe_path = mim.build_native_executable(driver, world, tmp / "shape_main")
            assert exe_path.exists()
            result = mim.run_native_executable(exe_path, check=False)
            assert result.returncode == 5
