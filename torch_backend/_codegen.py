"""FX graph → MimIR World → compiled shared library → Python callable.

Calling convention
------------------
The externalized MimIR function has the signature (after MimIR's CPS→direct
lowering and Mem-token elimination):

    void mimir_compute(T0* in0, T1* in1, ..., To0* out0, ...)

All inputs and outputs are flat, row-major arrays of the appropriate C type.
Output buffers are pre-allocated by the Python caller.

Design limits (first version)
------------------------------
* Only float32 tensors.
* Only ops in _ELEMENTWISE_OPS and aten.mm.
* Elementwise loops are unrolled at IR build time (total element count ≤
  _UNROLL_LIMIT).  aten.mm uses %matrix.prod (loop limit is output M×N only).
  Large tensors raise UnsupportedGraph.
* Multi-output graphs are supported; intermediate buffers are stack-allocated
  via %mem.alloc.
"""

from __future__ import annotations

import ctypes
import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx

import mim
from mim import _mim_core as _core

log = logging.getLogger(__name__)

# Max elements we'll unroll in one loop.
_UNROLL_LIMIT = 8192

# ATen target → MimIR float arithmetic axiom symbol.
_ELEMENTWISE_OPS: dict[Any, str] = {
    torch.ops.aten.add.Tensor: "%math.arith.add",
    torch.ops.aten.sub.Tensor: "%math.arith.sub",
    torch.ops.aten.mul.Tensor: "%math.arith.mul",
}

# ATen target → MimIR float unary axiom symbol.
_UNARY_OPS: dict[Any, str] = {
    torch.ops.aten.neg.default: "%math.minus",
    torch.ops.aten.abs.default: "%math.abs",
    torch.ops.aten.exp.default: "%math.exp.exp",
    torch.ops.aten.log.default: "%math.exp.log",
    torch.ops.aten.sin.default: "%math.tri.sin",
    torch.ops.aten.cos.default: "%math.tri.cos",
}


class UnsupportedGraph(Exception):
    pass


class GraphCodegen:
    """Translates a torch.fx.GraphModule into a compiled MimIR function."""

    def __init__(self, gm: fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs
        self._setup_world()

    # ------------------------------------------------------------------
    # World / type setup
    # ------------------------------------------------------------------

    def _setup_world(self) -> None:
        self.driver = mim.make_driver("core", "mem", "math", "matrix", "clos")
        self.w = self.driver.world()
        self._mem_t = mim.Mem.M(self.w, self.w.lit_nat_0())
        self._f32_t = self.w.call("%math.F32")
        self._ptr_f32 = mim.Mem.Ptr0(self.w, self._f32_t)

    def _ptr_type(self, dtype: torch.dtype) -> _core.Def:
        if dtype == torch.float32:
            return self._ptr_f32
        raise UnsupportedGraph(f"Unsupported dtype: {dtype}")

    def _elem_type(self, dtype: torch.dtype) -> _core.Def:
        if dtype == torch.float32:
            return self._f32_t
        raise UnsupportedGraph(f"Unsupported dtype: {dtype}")

    def _zero_lit(self, dtype: torch.dtype) -> _core.Def:
        if dtype == torch.float32:
            # 0.0f as IEEE-754 bits = 0
            return self.w.lit(self._f32_t, 0)
        raise UnsupportedGraph(f"Unsupported dtype: {dtype}")

    def _mat_desc(self, shape: tuple[int, ...], dtype: torch.dtype) -> _core.Def:
        """(n, S, T) descriptor tuple used as stage arg for %matrix.* ops."""
        sizes = self.w.tuple([self.w.lit_nat(s) for s in shape])
        return self.w.tuple([self.w.lit_nat(len(shape)), sizes, self._elem_type(dtype)])

    def _mat_t(self, shape: tuple[int, ...], dtype: torch.dtype) -> _core.Def:
        """Returns the %matrix.Mat (n, S, T) type Def."""
        return self.w.call("%matrix.Mat", self._mat_desc(shape, dtype))

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _placeholders(self) -> list[fx.Node]:
        return [n for n in self.gm.graph.nodes if n.op == "placeholder"]

    def _output_args(self) -> list[fx.Node]:
        for n in self.gm.graph.nodes:
            if n.op == "output":
                args = n.args[0]
                return list(args) if isinstance(args, (list, tuple)) else [args]
        return []

    def _node_shape(self, node: fx.Node) -> tuple[int, ...]:
        val = node.meta.get("val")
        if val is not None:
            return tuple(val.shape)
        raise UnsupportedGraph(f"No shape metadata on node '{node.name}'")

    def _node_dtype(self, node: fx.Node) -> torch.dtype:
        val = node.meta.get("val")
        if val is not None:
            return val.dtype
        for t in self.example_inputs:
            if isinstance(t, torch.Tensor):
                return t.dtype
        raise UnsupportedGraph(f"Cannot determine dtype for node '{node.name}'")

    def _n_elems(self, shape: tuple[int, ...]) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    # ------------------------------------------------------------------
    # Main compilation entry
    # ------------------------------------------------------------------

    def compile(self) -> callable:
        fn = self._build_mimir_fn()
        return self._lower(fn)

    # ------------------------------------------------------------------
    # MimIR function builder
    # ------------------------------------------------------------------

    def _build_mimir_fn(self) -> _core.Lam:
        w = self.w
        placeholders = self._placeholders()
        output_nodes = self._output_args()

        # Use only tensor inputs; dynamo may pass SymInts when recompiling with
        # dynamic shapes after seeing multiple concrete shapes for the same function.
        tensor_inputs = [t for t in self.example_inputs if isinstance(t, torch.Tensor)]
        in_dtypes = [t.dtype for t in tensor_inputs]
        out_dtypes = [self._node_dtype(n) for n in output_nodes]

        # Validate dtypes before computing shapes for clearer error messages.
        in_ptr_types = [self._ptr_type(d) for d in in_dtypes]
        out_ptr_types = [self._ptr_type(d) for d in out_dtypes]

        out_shapes = [self._node_shape(n) for n in output_nodes]

        # Signature: (mem, *in_ptrs, *out_ptrs) → (mem,)
        #
        # MimIR's LLVM backend eliminates Mem tokens and applies CPS→direct
        # conversion, producing:  void fn(T0* in0, ..., To0* out0, ...)
        dom_types = [self._mem_t] + in_ptr_types + out_ptr_types
        fn = w.mut_fun2(dom_types, [self._mem_t])
        fn_args = fn.var().proj(0)   # input bundle
        fn_ret = fn.var().proj(1)    # return continuation

        self._mem: _core.Def = fn_args.proj(0)
        self._fn_ret = fn_ret

        # Bind placeholders → input pointers
        self._defs: dict[str, _core.Def] = {}
        for i, ph in enumerate(placeholders):
            self._defs[ph.name] = fn_args.proj(1 + i)

        # Pre-allocated output pointers from caller
        n_in = len(in_ptr_types)
        self._out_ptrs: list[_core.Def] = [
            fn_args.proj(1 + n_in + i) for i in range(len(out_ptr_types))
        ]

        # Walk the graph
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                break
            if node.op == "call_function":
                self._emit_node(node)
            else:
                raise UnsupportedGraph(f"Unsupported node op: {node.op!r}")

        # Copy computation results into the caller's output buffers
        for i, out_node in enumerate(output_nodes):
            shape = out_shapes[i]
            dtype = out_dtypes[i]
            n = self._n_elems(shape)
            result_ptr = self._defs[out_node.name]
            self._emit_copy(result_ptr, self._out_ptrs[i], n)

        fn.app(False, fn_ret, [self._mem])
        fn.externalize()
        fn.set("mimir_compute")
        return fn

    # ------------------------------------------------------------------
    # Per-node emission
    # ------------------------------------------------------------------

    def _emit_node(self, node: fx.Node) -> None:
        target = node.target
        shape = self._node_shape(node)
        dtype = self._node_dtype(node)
        n = self._n_elems(shape)

        if target in _ELEMENTWISE_OPS:
            if n > _UNROLL_LIMIT:
                raise UnsupportedGraph(
                    f"{target}: {n} elements exceeds unroll limit {_UNROLL_LIMIT}"
                )
            op_sym = _ELEMENTWISE_OPS[target]
            a_ptr = self._arg_ptr(node, 0)
            b_ptr = self._arg_ptr(node, 1)
            out_ptr = self._alloc_temp(shape, dtype)
            for i in range(n):
                idx = self.w.lit_nat(i)
                a_val = self._load(self._lea(a_ptr, idx))
                b_val = self._load(self._lea(b_ptr, idx))
                result = self.w.call(op_sym, [a_val, b_val])
                self._store(self._lea(out_ptr, idx), result)
            self._defs[node.name] = out_ptr

        elif target in _UNARY_OPS:
            if n > _UNROLL_LIMIT:
                raise UnsupportedGraph(
                    f"{target}: {n} elements exceeds unroll limit {_UNROLL_LIMIT}"
                )
            op_sym = _UNARY_OPS[target]
            a_ptr = self._arg_ptr(node, 0)
            out_ptr = self._alloc_temp(shape, dtype)
            for i in range(n):
                idx = self.w.lit_nat(i)
                a_val = self._load(self._lea(a_ptr, idx))
                result = self.w.call(op_sym, [a_val])
                self._store(self._lea(out_ptr, idx), result)
            self._defs[node.name] = out_ptr

        elif target == torch.ops.aten.mm.default:
            self._emit_mm(node, shape, dtype)

        else:
            raise UnsupportedGraph(f"Unsupported op: {target}")

    def _emit_mm(self, node: fx.Node, out_shape: tuple[int, ...], dtype: torch.dtype) -> None:
        a_node, b_node = node.args[0], node.args[1]
        M, K = self._node_shape(a_node)
        K2, N = self._node_shape(b_node)
        if K != K2:
            raise UnsupportedGraph(f"mm: shape mismatch {(M,K)} @ {(K2,N)}")
        if M * N > _UNROLL_LIMIT:
            raise UnsupportedGraph(
                f"mm [{M}x{K}]@[{K}x{N}]: output {M*N} elements exceeds limit {_UNROLL_LIMIT}"
            )

        a_ptr = self._defs[a_node.name]
        b_ptr = self._defs[b_node.name]

        # Bitcast flat F32 pointers to matrix types for %matrix.prod
        mat_a = self.w.call("%core.bitcast", self._ptr_f32, self._mat_t((M, K), dtype), [a_ptr])
        mat_b = self.w.call("%core.bitcast", self._ptr_f32, self._mat_t((K, N), dtype), [b_ptr])

        # %matrix.prod stage: (M, K, N, (mantissa_bits, exponent_bits)) for F32
        pe = self.w.tuple([self.w.lit_nat(23), self.w.lit_nat(8)])
        stage = self.w.tuple([self.w.lit_nat(M), self.w.lit_nat(K), self.w.lit_nat(N), pe])
        prod_r = self.w.call("%matrix.prod", stage, [self._mem, mat_a, mat_b])
        self._mem = prod_r.proj(0)
        result_mat = prod_r.proj(1)

        # Read matrix elements into a flat output buffer
        out_ptr = self._alloc_temp(out_shape, dtype)
        result_desc = self._mat_desc((M, N), dtype)
        for i in range(M):
            for j in range(N):
                idx = self.w.tuple([self.w.lit_idx(M, i), self.w.lit_idx(N, j)])
                rd = self.w.call("%matrix.read", result_desc, [self._mem, result_mat, idx])
                self._mem = rd.proj(0)
                self._store(self._lea(out_ptr, self.w.lit_nat(i * N + j)), rd.proj(1))

        self._defs[node.name] = out_ptr

    # ------------------------------------------------------------------
    # Memory helpers (thread self._mem through each op)
    # ------------------------------------------------------------------

    def _arg_ptr(self, node: fx.Node, idx: int) -> _core.Def:
        arg = node.args[idx]
        if isinstance(arg, fx.Node):
            return self._defs[arg.name]
        raise UnsupportedGraph(f"Non-node argument at position {idx}: {arg!r}")

    def _alloc_temp(self, shape: tuple[int, ...], dtype: torch.dtype) -> _core.Def:
        """Allocate a flat temporary buffer; returns a typed Ptr."""
        n = self._n_elems(shape)
        elem_t = self._elem_type(dtype)
        # %mem.alloc : [T] (Mem, count: Nat) → (Mem, Ptr(T))
        result = self.w.call("%mem.alloc", elem_t, [self._mem, self.w.lit_nat(n)])
        self._mem = result.proj(0)
        return result.proj(1)

    def _lea(self, ptr: _core.Def, idx: _core.Def) -> _core.Def:
        # %mem.lea : Ptr(T) × Nat → Ptr(T)  (pointer arithmetic, no Mem)
        return self.w.call("%mem.lea", [ptr, idx])

    def _load(self, ptr: _core.Def) -> _core.Def:
        # %mem.load : (Mem, Ptr(T)) → (Mem, T)
        result = self.w.call("%mem.load", [self._mem, ptr])
        self._mem = result.proj(0)
        return result.proj(1)

    def _store(self, ptr: _core.Def, val: _core.Def) -> None:
        # %mem.store : (Mem, Ptr(T), T) → Mem
        self._mem = self.w.call("%mem.store", [self._mem, ptr, val])

    def _emit_copy(self, src: _core.Def, dst: _core.Def, n: int) -> None:
        for i in range(n):
            idx = self.w.lit_nat(i)
            val = self._load(self._lea(src, idx))
            self._store(self._lea(dst, idx), val)

    # ------------------------------------------------------------------
    # Compilation: MimIR → LLVM IR → shared library → ctypes callable
    # ------------------------------------------------------------------

    def _lower(self, fn: _core.Lam) -> callable:
        tmpdir = Path(tempfile.mkdtemp(prefix="mimir_backend_"))
        ll_path = tmpdir / "compute.ll"
        so_path = tmpdir / "compute.so"

        mim.emit_llvm(self.driver, self.w, str(ll_path))
        log.debug("MimIR LLVM IR written to %s", ll_path)

        mim.clang_compile(ll_path, so_path, extra_args=["-shared", "-fPIC", "-O2"])
        log.debug("Compiled shared library: %s", so_path)

        lib = ctypes.CDLL(str(so_path))
        c_fn = lib.mimir_compute

        # After Mem-token elimination and CPS→direct lowering the exported
        # symbol has signature:  void mimir_compute(T0* in0, ..., To0* out0, ...)
        n_inputs = len(self._placeholders())
        n_args = n_inputs + len(self._output_args())
        c_fn.argtypes = [ctypes.c_void_p] * n_args
        c_fn.restype = None

        out_specs = [
            (self._node_shape(n), self._node_dtype(n))
            for n in self._output_args()
        ]

        return _build_python_wrapper(c_fn, n_inputs, out_specs)


# ------------------------------------------------------------------
# Runtime wrapper
# ------------------------------------------------------------------

def _build_python_wrapper(
    c_fn,
    n_inputs: int,
    out_specs: list[tuple[tuple[int, ...], torch.dtype]],
) -> callable:
    """Returns a Python callable: (*torch.Tensor) → torch.Tensor | tuple."""

    def call(*inputs: torch.Tensor):
        if len(inputs) != n_inputs:
            raise ValueError(f"Expected {n_inputs} inputs, got {len(inputs)}")
        outputs = [
            torch.empty(shape, dtype=dtype, device="cpu")
            for shape, dtype in out_specs
        ]
        ptrs = [ctypes.c_void_p(t.data_ptr()) for t in inputs]
        ptrs += [ctypes.c_void_p(t.data_ptr()) for t in outputs]
        c_fn(*ptrs)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return call
