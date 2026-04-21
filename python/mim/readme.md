# MimIR Python API

Python bindings for the [MimIR](https://github.com/AnyDSL/MimIR) compiler
intermediate representation.  The package lets you construct, inspect, optimise,
and compile MimIR programs from Python.

## Installation

```bash
pip install .          # from the repository root
# or, for development:
pip install -e .
```

The build requires CMake, a C++20 compiler, and pybind11 (vendored).

## Quick start

```python
import mim

# Create a driver with plugins loaded
driver = mim.make_driver("core", "mem", "matrix")
world  = driver.world()

# Build a 3x5 constant matrix filled with 42
mem = world.bot(mim.Mem.M(world, world.lit_nat_0()))
mat_desc = world.tuple([
    world.lit_nat(2),
    world.tuple([world.lit_nat(3), world.lit_nat(5)]),
    world.type_i32(),
])
result = mim.Matrix.constMat(world, mat_desc, [mem, world.lit_i32(42)])
mat    = result.proj(1)

# Query the shape
shape_0 = mim.Matrix.shape(world, mat_desc, [mat, world.lit_idx(2, 0)])
print(shape_0.value())  # 3
```

---

## Core classes

All core classes live in `mim._mim_core` and are re-exported from `mim`.

### Driver

Entry point.  Owns the `World` and manages plugin loading.

```python
driver = mim.Driver()
```

| Method | Description |
|--------|-------------|
| `world() -> World` | Return the world owned by this driver. |
| `load_plugins(names: list[str])` | Load the named plugins (e.g. `["core", "mem", "matrix"]`). |
| `add_search_path(path: Path)` | Add a directory to the plugin search path. |
| `log() -> Log` | Return the logger. Chain with `.set_stdout().set(Level.Debug)`. |
| `backend(name, output_file, world)` | Emit code via a backend. `name` is typically `"ll"` for LLVM IR. |
| `add_import(path, name) -> Path` | Register an import file. |

### World

The universe of all definitions.  Everything you build belongs to a `World`.

**Type constructors**

| Method | Returns | Description |
|--------|---------|-------------|
| `type_i8()` | `Def` | The `I8` type. |
| `type_i32()` | `Def` | The `I32` type. |
| `type_i8()` | `Def` | The `I8` type. |
| `type_bool()` | `Def` | The `Bool` type. |
| `type_idx(size)` | `Def` | Index type `Idx size`. `size` can be a `Def` or `int`. |
| `top_nat()` | `Def` | Top element of Nat (infinity). |
| `cn(domains: list[Def])` | `Def` | Continuation type with the given domain types. |
| `arr(arity, body)` | `Def` | Dependent array type. |

**Literal constructors**

| Method | Returns | Description |
|--------|---------|-------------|
| `lit_nat(n)` | `Lit` | Natural-number literal. |
| `lit_nat_0()` | `Lit` | Shorthand for `lit_nat(0)`. |
| `lit_i8(n)` | `Lit` | 8-bit integer literal. |
| `lit_i32(n)` | `Lit` | 32-bit integer literal. |
| `lit(type, value)` | `Lit` | Literal of arbitrary type. |
| `lit_idx(size, value)` | `Lit` | Index literal: value of type `Idx size`. |

**Building terms**

| Method | Returns | Description |
|--------|---------|-------------|
| `call(callee, *stages, implicit=False)` | `Def` | The main way to invoke axioms.  `callee` is a string like `"%matrix.constMat"` or a `Def`.  Each positional argument after it is one "stage" (curried application).  A stage can be a single `Def` or a `list[Def]` (packed into a tuple automatically). |
| `app(callee, args)` | `Def` | Explicit application: `callee(args...)`. |
| `implicit_app(callee, args)` | `Def` | Application using implicit (type-level) arguments. |
| `tuple(elements: list[Def])` | `Def` | Build a tuple from a list of defs. |
| `bot(type)` | `Def` | Bottom value of the given type (undefined / poison). |
| `sym(name) -> Sym` | `Sym` | Intern a string as a symbol. |
| `annex_name(name: str)` | `Def` | Look up an axiom by its full name string. |

**Mutables (functions, continuations)**

| Method | Returns | Description |
|--------|---------|-------------|
| `mut_fun2(dom, codom)` | `Lam` | Create a mutable function.  `dom` and `codom` are lists of types. |
| `mut_fun(dom, codom)` | `Lam` | Like `mut_fun2` but `dom` is a single `Def`. |
| `mut_con(domains)` | `Lam` | Create a mutable continuation (no return). |

**Compilation**

| Method | Description |
|--------|-------------|
| `optimize()` | Run the full optimization pipeline on the world. |
| `write()` | Dump the world to stdout as `.mim` source. |
| `dot(filename, show_types, show_bbs)` | Write a Graphviz `.dot` file. |

### Def

The base class for all MimIR definitions (types, terms, axioms, ...).

| Method | Returns | Description |
|--------|---------|-------------|
| `type()` | `Def` | The type of this def. |
| `proj(index)` | `Def` | Project the i-th component from a tuple/sigma. |
| `proj(arity, index)` | `Def` | Project with an explicit arity. |
| `projs(n)` | `list[Def]` | Return the first `n` projections as a list. |
| `num_projs()` | `int` | Number of projections (tuple width). |
| `var()` | `Def` | The variable bound by this def (for mutables). |
| `set(name: str)` | `Def` | Set the debug name.  Returns `self` for chaining. |
| `externalize()` | `None` | Mark this def as externally visible (exported). |
| `to_string()` | `str` | Pretty-print the def as a MimIR string. |
| `dump()` | `None` | Print to stdout (for debugging). |

### Lit

Subclass of `Def` for literal values.

| Method | Returns | Description |
|--------|---------|-------------|
| `value()` | `int` | The raw integer value of the literal. |

Note: `value()` is only available on `Lit` instances.  If an operation does not
normalize to a literal (e.g. `read(insert(...))` without lowering), the result
is a plain `Def` and you must compile or inspect it as IR.

### Lam

Subclass of `Def` for lambda / function nodes.

| Method | Returns | Description |
|--------|---------|-------------|
| `var()` | `Def` | The bound variable (parameter pack). |
| `app(filter, callee, args)` | `Lam` | Set the body to an application. `filter` is a `bool` for the inliner. |
| `externalize()` | `None` | Mark as externally visible. |

### Pi, AST, Parser, PyParser, Log, Level

- `Pi` -- subclass of `Def` for function-type nodes.
- `AST` -- constructed from a `World`, needed to create a `Parser`.
- `Parser` -- the MimIR parser.  `parser.plugin("core")` loads a plugin by name.
- `PyParser` -- wraps a `Parser` for use from Python callbacks.
- `Log` -- logger.  `log.set_stdout()` sends output to stdout; `log.set(Level.Debug)` sets verbosity.
- `Level` -- log levels: `Error`, `Warn`, `Info`, `Verbose`, `Debug`.

---

## Convenience functions

These are defined in `mim.__init__` and provide higher-level workflows.

### Driver setup

```python
# One-liner: create a driver, configure paths, load plugins
driver = mim.make_driver("core", "mem", "matrix", log_level=mim.Level.Info)

# Or step by step:
driver = mim.configure_driver(mim.Driver())
driver.load_plugins(["core", "mem", "matrix"])
```

`make_driver(*plugins, log_level=None, set_stdout=False)` -- creates a
`Driver`, configures plugin search paths, optionally sets log level, and loads
the given plugins.

`configure_driver(driver)` -- adds the standard plugin search paths to an
existing driver and preloads `libmim.so`.

`plugin_search_paths()` -- returns the list of directories where plugins are
searched.

### Compilation pipeline

```python
world = driver.world()

# ... build your program ...

# Wrap a result value into a main(argc, argv) -> i32 function
main = mim.build_native_main_i32(world, some_i32_value)

# Emit LLVM IR
ll_path = mim.emit_llvm(driver, world, "output.ll")

# Compile to a native binary (requires clang)
exe_path = mim.clang_compile("output.ll", "output")

# Or do both in one step
exe_path = mim.build_native_executable(driver, world, "output")

# Run it
result = mim.run_native_executable(exe_path)
print(result.returncode)
```

| Function | Description |
|----------|-------------|
| `build_native_main_i32(world, result, name="main")` | Wrap an I32 def into a `main(mem, argc, argv) -> (mem, i32)` function and externalize it. |
| `emit_llvm(driver, world, path)` | Run the LLVM backend, writing IR to `path`. Returns the `Path`. |
| `clang_compile(ll_path, output, clang="clang", extra_args=())` | Invoke clang to compile LLVM IR to a native binary. |
| `build_native_executable(driver, world, stem, clang="clang", extra_args=())` | Emit LLVM IR and compile in one step.  `stem` is the output path without extension. |
| `run_native_executable(exe, args=(), check=False)` | Run a compiled binary and return a `CompletedProcess`. |

### Tuple-level matrix helpers

These work on MimIR tuples directly (no plugin required).  Useful for building
constant matrices from Python lists.

```python
world = mim.Driver().world()
mat   = mim.matrix_i32(world, [[1, 2], [3, 4]])
vals  = mim.matrix_to_list(mat)          # [[1, 2], [3, 4]]
t     = mim.transpose_2d(world, mat)     # [[1, 3], [2, 4]]
prod  = mim.matmul_i32(world, mat, mat)  # [[7, 10], [15, 22]]
```

| Function | Description |
|----------|-------------|
| `matrix_i32(world, rows)` | Build a nested tuple of I32 literals from a list of lists. |
| `matrix_to_list(matrix)` | Extract a nested tuple back into a Python list of lists. |
| `transpose_2d(world, matrix)` | Transpose a 2D nested tuple. |
| `matmul_i32(world, lhs, rhs)` | Integer matrix multiplication on nested tuples. |

---

## Axiom wrappers

Every MimIR plugin (core, mem, matrix, tensor, ...) exports named axioms like
`%matrix.constMat` or `%mem.M`.  The Python package provides three ways to call
them, from low-level to high-level.

### 1. Raw string calls

Pass the axiom name as a string to `world.call()`:

```python
result = world.call("%matrix.constMat", mat_type, [mem, world.lit_i32(5)])
```

Each positional argument after the name is a "stage" -- one curried application.
A list is automatically packed into a tuple.

### 2. Namespace objects (recommended)

Module-level objects mirror the plugin hierarchy.  Pass `world` as the first
argument:

```python
import mim

# mim.Matrix.constMat calls "%matrix.constMat" under the hood
result = mim.Matrix.constMat(world, mat_type, [mem, world.lit_i32(5)])

# Also available: mim.Mem.M, mim.Core.wrap.add, mim.Math.arith.mul, ...
```

Autocomplete works via `__dir__`.  Use `node.symbol` to see the underlying
axiom name:

```python
mim.Matrix.read.symbol  # "%matrix.read"
```

Available namespaces: `Affine`, `Autodiff`, `Clos`, `Compile`, `Core`, `Demo`,
`Direct`, `Gpu`, `Math`, `Matrix`, `Mem`, `Opt`, `Ord`, `Refly`, `Regex`,
`Tensor`, `Tuple`, `Vec`.

### 3. Bound axioms (optional shortcut)

If you are making many calls against the same world, `mim.bind()` saves you
from passing `world` every time:

```python
ax = mim.bind(world)

mem_type = ax.Mem.M(world.lit_nat_0())      # no world argument needed
matrix   = ax.Matrix.constMat(mat_type, [mem, world.lit_i32(5)])
shape    = ax.Matrix.shape(mat_type, [mat, world.lit_idx(2, 0)])
```

`list_axioms()` returns a dict mapping plugin names to their axiom symbols:

```python
axioms = mim.list_axioms()
axioms["matrix"]  # ("%matrix.Mat", "%matrix.constMat", "%matrix.read", ...)
```

---

## Matrix plugin operations

The matrix plugin (`%matrix.*`) operates on n-dimensional tensors.  Most
operations are effectful and thread a memory token.

The examples below use the module-level wrappers (`mim.Matrix.*`).  You can
also use `mim.bind(world)` to avoid repeating `world`, or raw strings
(`world.call("%matrix.constMat", ...)`) -- all three forms are equivalent.

All matrix operations take a type descriptor as their first stage:

```python
# Type descriptor for a 2D matrix: n=2, shape=(rows, cols), element type
mat_type = world.tuple([
    world.lit_nat(2),                                         # n dimensions
    world.tuple([world.lit_nat(rows), world.lit_nat(cols)]),  # shape
    world.type_i32(),                                         # element type
])
```

### constMat -- constant matrix

```python
result = mim.Matrix.constMat(world, mat_type, [mem, world.lit_i32(val)])
mem_out = result.proj(0)
mat     = result.proj(1)
```

### shape -- query dimension size

```python
dim_size = mim.Matrix.shape(world, mat_type, [mat, world.lit_idx(2, dim)])
# Returns a Lit whose value() is the size along that dimension.
```

### read -- element access

```python
idx    = world.tuple([world.lit_idx(rows, r), world.lit_idx(cols, c)])
result = mim.Matrix.read(world, mat_type, [mem, mat, idx])
mem_out = result.proj(0)
value   = result.proj(1)
```

### insert -- element write

```python
idx    = world.tuple([world.lit_idx(rows, r), world.lit_idx(cols, c)])
result = mim.Matrix.insert(world, mat_type, [mem, mat, idx, world.lit_i32(val)])
mem_out = result.proj(0)
mat_out = result.proj(1)
```

### transpose

The first stage is `[k, l]` (the original dimensions), plus the element type:

```python
kl = world.tuple([world.lit_nat(rows), world.lit_nat(cols)])
result = mim.Matrix.transpose(world, [kl, world.type_i32()], [mem, mat])
```

### prod -- matrix multiplication (floating point)

For F64 matrices (`%math.F (52, 11)`):

```python
pe  = world.tuple([world.lit_nat(52), world.lit_nat(11)])  # precision/exponent
f64 = mim.Math.F(world, pe)
result = mim.Matrix.prod(world,
    [world.lit_nat(m), world.lit_nat(k), world.lit_nat(l), pe],
    [mem, mat_a, mat_b])
```

---

## Lowering and optimization

Matrix and tensor operations exist only at the IR level.  The LLVM backend does
not understand them directly.  Before emitting code, you must lower them:

```python
world.optimize()  # runs the full pass pipeline, including matrix lowering
```

For operations that normalize at construction time (like `shape` on a constant
matrix, or `read` on a `constMat`), no lowering is needed -- the result is
already a literal.

Operations that do not normalize (like `read(insert(...))` or
`read(transpose(...))`) produce symbolic IR nodes.  These require the full
`optimize()` pipeline before they can be compiled to LLVM.

---

## Complete example: build, compile, and run

```python
import tempfile
from pathlib import Path
import mim

driver = mim.make_driver("core", "mem", "math", "matrix")
world  = driver.world()

# Build a 3x5 constant matrix filled with 5
mem = world.bot(mim.Mem.M(world, world.lit_nat_0()))
mt  = world.tuple([
    world.lit_nat(2),
    world.tuple([world.lit_nat(3), world.lit_nat(5)]),
    world.type_i32(),
])
cm  = mim.Matrix.constMat(world, mt, [mem, world.lit_i32(5)])

# Read element at (0, 0)
idx = world.tuple([world.lit_idx(3, 0), world.lit_idx(5, 0)])
rd  = mim.Matrix.read(world, mt, [cm.proj(0), cm.proj(1), idx])
val = rd.proj(1)  # normalized to Lit(5)

# Wrap into main and compile
mim.build_native_main_i32(world, val)

with tempfile.TemporaryDirectory() as tmp:
    exe = mim.build_native_executable(driver, world, Path(tmp) / "example")
    result = mim.run_native_executable(exe)
    print(result.returncode)  # 5
```

---

## Error handling

- `mim.MIM_Error` -- raised when the compiler detects a type error or invalid
  IR construction (e.g. applying an argument to a callee with an incompatible
  type).
- `TypeError` -- raised by `build_native_main_i32` if the result def is not
  `I32`.
- `FileNotFoundError` -- raised by `clang_compile` if clang is not found.
- `subprocess.CalledProcessError` -- raised by compilation or execution helpers
  when a subprocess fails.
