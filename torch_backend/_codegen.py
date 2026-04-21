"""FX graph → MimIR World → compiled shared library → Python callable.

Calling convention
------------------
The externalized MimIR function has the signature (after MimIR's CPS→direct
lowering and Mem-token elimination):

    void mimir_compute(T0* in0, T1* in1, ..., Tc0* const0, ..., To0* out0, ...)

All inputs, constants (get_attr), and outputs are flat, row-major arrays.
Output buffers are pre-allocated by the Python caller.

Design limits
-------------
* Only float32 tensors.
* Elementwise loops are unrolled at IR build time (total element count ≤
  _UNROLL_LIMIT).  aten.mm / aten.addmm use %matrix.prod (limit is output
  M×N only).  Large tensors raise UnsupportedGraph and fall back to eager.
* Multi-output graphs are supported; intermediate buffers are stack-allocated
  via %mem.alloc.
"""

from __future__ import annotations

import ctypes
import logging
import struct
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

# ATen target → MimIR float binary axiom symbol.
_ELEMENTWISE_OPS: dict[Any, str] = {
    torch.ops.aten.add.Tensor: "%math.arith.add",
    torch.ops.aten.sub.Tensor: "%math.arith.sub",
    torch.ops.aten.mul.Tensor: "%math.arith.mul",
    torch.ops.aten.div.Tensor: "%math.arith.div",
    torch.ops.aten.pow.Tensor_Tensor: "%math.pow",
    torch.ops.aten.maximum.default: "%math.extrema.fmax",
    torch.ops.aten.minimum.default: "%math.extrema.fmin",
}

# ATen target → MimIR float unary axiom symbol.
_UNARY_OPS: dict[Any, str] = {
    torch.ops.aten.neg.default: "%math.minus",
    torch.ops.aten.abs.default: "%math.abs",
    torch.ops.aten.exp.default: "%math.exp.exp",
    torch.ops.aten.exp2.default: "%math.exp.exp2",
    torch.ops.aten.log.default: "%math.exp.log",
    torch.ops.aten.log2.default: "%math.exp.log2",
    torch.ops.aten.sin.default: "%math.tri.sin",
    torch.ops.aten.cos.default: "%math.tri.cos",
    torch.ops.aten.tan.default: "%math.tri.tan",
    torch.ops.aten.tanh.default: "%math.tri.tanh",
    torch.ops.aten.sigmoid.default: "%math.slf",
    torch.ops.aten.sqrt.default: "%math.rt.sq",
    torch.ops.aten.rsqrt.default: "%math.rrt",
    torch.ops.aten.erf.default: "%math.er.f",
    torch.ops.aten.floor.default: "%math.round.f",
    torch.ops.aten.ceil.default: "%math.round.c",
    torch.ops.aten.round.default: "%math.round.r",
    torch.ops.aten.trunc.default: "%math.round.t",
}

# ATen target → MimIR float binary axiom symbol (tensor op scalar).
_SCALAR_OPS: dict[Any, str] = {
    torch.ops.aten.add.Scalar: "%math.arith.add",
    torch.ops.aten.sub.Scalar: "%math.arith.sub",
    torch.ops.aten.mul.Scalar: "%math.arith.mul",
    torch.ops.aten.div.Scalar: "%math.arith.div",
    torch.ops.aten.pow.Tensor_Scalar: "%math.pow",
}

# Ops that are pure shape changes — the flat data layout is unchanged.
_NOOP_OPS: set[Any] = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.flatten.using_ints,
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
            return self.w.lit(self._f32_t, 0)
        raise UnsupportedGraph(f"Unsupported dtype: {dtype}")

    def _float_lit(self, val: float) -> _core.Def:
        bits = struct.unpack("I", struct.pack("f", float(val)))[0]
        return self.w.lit(self._f32_t, bits)

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

    def _get_attr_nodes(self) -> list[fx.Node]:
        return [n for n in self.gm.graph.nodes if n.op == "get_attr"]

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

    def _resolve_get_attr(self, node: fx.Node) -> torch.Tensor:
        parts = node.target.split(".")
        obj = self.gm
        for part in parts:
            obj = getattr(obj, part)
        if not isinstance(obj, torch.Tensor):
            raise UnsupportedGraph(f"get_attr {node.target!r} is not a tensor")
        return obj

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
        get_attr_nodes = self._get_attr_nodes()
        output_nodes = self._output_args()

        # Use only tensor inputs; dynamo may pass SymInts when recompiling with
        # dynamic shapes after seeing multiple concrete shapes for the same function.
        tensor_inputs = [t for t in self.example_inputs if isinstance(t, torch.Tensor)]
        in_dtypes = [t.dtype for t in tensor_inputs]
        out_dtypes = [self._node_dtype(n) for n in output_nodes]

        # Resolve constant tensors from get_attr nodes.
        self._const_tensors: list[torch.Tensor] = [
            self._resolve_get_attr(n) for n in get_attr_nodes
        ]
        const_dtypes = [t.dtype for t in self._const_tensors]

        # Validate dtypes before computing shapes for clearer error messages.
        in_ptr_types = [self._ptr_type(d) for d in in_dtypes]
        const_ptr_types = [self._ptr_type(d) for d in const_dtypes]
        out_ptr_types = [self._ptr_type(d) for d in out_dtypes]

        out_shapes = [self._node_shape(n) for n in output_nodes]

        # Signature: (mem, *in_ptrs, *const_ptrs, *out_ptrs) → (mem,)
        #
        # MimIR's LLVM backend eliminates Mem tokens and applies CPS→direct
        # conversion, producing:
        #   void fn(T0* in0, ..., Tc0* const0, ..., To0* out0, ...)
        dom_types = [self._mem_t] + in_ptr_types + const_ptr_types + out_ptr_types
        fn = w.mut_fun2(dom_types, [self._mem_t])
        fn_args = fn.var().proj(0)   # input bundle
        fn_ret = fn.var().proj(1)    # return continuation

        self._mem: _core.Def = fn_args.proj(0)
        self._fn_ret = fn_ret

        self._defs: dict[str, _core.Def] = {}

        # Bind placeholders → input pointers
        for i, ph in enumerate(placeholders):
            self._defs[ph.name] = fn_args.proj(1 + i)

        # Bind get_attr nodes → constant pointers
        n_in = len(in_ptr_types)
        for i, ga in enumerate(get_attr_nodes):
            self._defs[ga.name] = fn_args.proj(1 + n_in + i)

        # Pre-allocated output pointers from caller
        n_in_const = n_in + len(const_ptr_types)
        self._out_ptrs: list[_core.Def] = [
            fn_args.proj(1 + n_in_const + i) for i in range(len(out_ptr_types))
        ]

        # Walk the graph
        for node in self.gm.graph.nodes:
            if node.op in ("placeholder", "get_attr"):
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

        elif target in _SCALAR_OPS:
            if n > _UNROLL_LIMIT:
                raise UnsupportedGraph(
                    f"{target}: {n} elements exceeds unroll limit {_UNROLL_LIMIT}"
                )
            op_sym = _SCALAR_OPS[target]
            a_ptr = self._arg_ptr(node, 0)
            b_lit = self._float_lit(float(node.args[1]))
            out_ptr = self._alloc_temp(shape, dtype)
            for i in range(n):
                idx = self.w.lit_nat(i)
                a_val = self._load(self._lea(a_ptr, idx))
                result = self.w.call(op_sym, [a_val, b_lit])
                self._store(self._lea(out_ptr, idx), result)
            self._defs[node.name] = out_ptr

        elif target == torch.ops.aten.relu.default:
            if n > _UNROLL_LIMIT:
                raise UnsupportedGraph(
                    f"relu: {n} elements exceeds unroll limit {_UNROLL_LIMIT}"
                )
            a_ptr = self._arg_ptr(node, 0)
            zero = self._zero_lit(dtype)
            out_ptr = self._alloc_temp(shape, dtype)
            for i in range(n):
                idx = self.w.lit_nat(i)
                a_val = self._load(self._lea(a_ptr, idx))
                result = self.w.call("%math.extrema.fmax", [zero, a_val])
                self._store(self._lea(out_ptr, idx), result)
            self._defs[node.name] = out_ptr

        elif target in _NOOP_OPS:
            # Shape-only ops: flat data layout is unchanged, alias the pointer.
            self._defs[node.name] = self._defs[node.args[0].name]

        elif target == torch.ops.aten.t.default:
            self._emit_t(node, shape, dtype, n)

        elif target == torch.ops.aten.mm.default:
            self._emit_mm(node, shape, dtype)

        elif target == torch.ops.aten.addmm.default:
            self._emit_addmm(node, shape, dtype)

        else:
            raise UnsupportedGraph(f"Unsupported op: {target}")

    def _emit_t(self, node: fx.Node, out_shape: tuple[int, ...], dtype: torch.dtype, n: int) -> None:
        in_node = node.args[0]
        in_shape = self._node_shape(in_node)
        if len(in_shape) != 2:
            raise UnsupportedGraph(f"t() only supported for 2D tensors, got shape {in_shape}")
        if n > _UNROLL_LIMIT:
            raise UnsupportedGraph(f"t() {in_shape}: {n} elements exceeds unroll limit")
        M, N = in_shape  # input rows, cols
        in_ptr = self._defs[in_node.name]
        out_ptr = self._alloc_temp(out_shape, dtype)  # shape = (N, M)
        for i in range(M):
            for j in range(N):
                val = self._load(self._lea(in_ptr, self.w.lit_nat(i * N + j)))
                self._store(self._lea(out_ptr, self.w.lit_nat(j * M + i)), val)
        self._defs[node.name] = out_ptr

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

        result_mat = self._matrix_prod(
            self._defs[a_node.name], (M, K),
            self._defs[b_node.name], (K, N),
            dtype,
        )

        out_ptr = self._alloc_temp(out_shape, dtype)
        result_desc = self._mat_desc((M, N), dtype)
        for i in range(M):
            for j in range(N):
                idx = self.w.tuple([self.w.lit_idx(M, i), self.w.lit_idx(N, j)])
                rd = self.w.call("%matrix.read", result_desc, [self._mem, result_mat, idx])
                self._mem = rd.proj(0)
                self._store(self._lea(out_ptr, self.w.lit_nat(i * N + j)), rd.proj(1))

        self._defs[node.name] = out_ptr

    def _emit_addmm(self, node: fx.Node, out_shape: tuple[int, ...], dtype: torch.dtype) -> None:
        bias_node, mat1_node, mat2_node = node.args[0], node.args[1], node.args[2]
        beta = node.kwargs.get("beta", 1)
        alpha = node.kwargs.get("alpha", 1)
        if beta != 1 or alpha != 1:
            raise UnsupportedGraph(f"addmm with beta={beta}, alpha={alpha} not supported")

        B, N = out_shape
        mat1_shape = self._node_shape(mat1_node)
        mat2_shape = self._node_shape(mat2_node)
        K = mat1_shape[1]

        if B * N > _UNROLL_LIMIT:
            raise UnsupportedGraph(
                f"addmm output {B}x{N}={B*N} elements exceeds limit {_UNROLL_LIMIT}"
            )

        result_mat = self._matrix_prod(
            self._defs[mat1_node.name], mat1_shape,
            self._defs[mat2_node.name], mat2_shape,
            dtype,
        )

        out_ptr = self._alloc_temp(out_shape, dtype)
        bias_ptr = self._defs[bias_node.name]
        result_desc = self._mat_desc((B, N), dtype)
        for i in range(B):
            for j in range(N):
                idx = self.w.tuple([self.w.lit_idx(B, i), self.w.lit_idx(N, j)])
                rd = self.w.call("%matrix.read", result_desc, [self._mem, result_mat, idx])
                self._mem = rd.proj(0)
                mm_val = rd.proj(1)
                bias_val = self._load(self._lea(bias_ptr, self.w.lit_nat(j)))
                result = self.w.call("%math.arith.add", [mm_val, bias_val])
                self._store(self._lea(out_ptr, self.w.lit_nat(i * N + j)), result)

        self._defs[node.name] = out_ptr

    def _matrix_prod(
        self,
        a_ptr: _core.Def, a_shape: tuple[int, int],
        b_ptr: _core.Def, b_shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> _core.Def:
        """Call %matrix.prod on two flat F32 pointers; returns the result Mat."""
        M, K = a_shape
        K2, N = b_shape
        mat_a = self.w.call("%core.bitcast", self._ptr_f32, self._mat_t((M, K), dtype), [a_ptr])
        mat_b = self.w.call("%core.bitcast", self._ptr_f32, self._mat_t((K2, N), dtype), [b_ptr])
        pe = self.w.tuple([self.w.lit_nat(23), self.w.lit_nat(8)])
        stage = self.w.tuple([self.w.lit_nat(M), self.w.lit_nat(K), self.w.lit_nat(N), pe])
        prod_r = self.w.call("%matrix.prod", stage, [self._mem, mat_a, mat_b])
        self._mem = prod_r.proj(0)
        return prod_r.proj(1)

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
        result = self.w.call("%mem.alloc", elem_t, [self._mem, self.w.lit_nat(n)])
        self._mem = result.proj(0)
        return result.proj(1)

    def _lea(self, ptr: _core.Def, idx: _core.Def) -> _core.Def:
        return self.w.call("%mem.lea", [ptr, idx])

    def _load(self, ptr: _core.Def) -> _core.Def:
        result = self.w.call("%mem.load", [self._mem, ptr])
        self._mem = result.proj(0)
        return result.proj(1)

    def _store(self, ptr: _core.Def, val: _core.Def) -> None:
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
        # symbol has signature:
        #   void mimir_compute(T0* in0, ..., Tc0* const0, ..., To0* out0, ...)
        n_inputs = len(self._placeholders())
        n_consts = len(self._get_attr_nodes())
        n_args = n_inputs + n_consts + len(self._output_args())
        c_fn.argtypes = [ctypes.c_void_p] * n_args
        c_fn.restype = None

        out_specs = [
            (self._node_shape(n), self._node_dtype(n))
            for n in self._output_args()
        ]

        return _build_python_wrapper(c_fn, n_inputs, self._const_tensors, out_specs)


# ------------------------------------------------------------------
# Runtime wrapper
# ------------------------------------------------------------------

def _build_python_wrapper(
    c_fn,
    n_inputs: int,
    const_tensors: list[torch.Tensor],
    out_specs: list[tuple[tuple[int, ...], torch.dtype]],
) -> callable:
    """Returns a Python callable: (*torch.Tensor) → torch.Tensor | tuple."""
    # Pre-compute constant pointers; keep tensor refs alive via closure.
    const_ptrs = [ctypes.c_void_p(t.data_ptr()) for t in const_tensors]

    def call(*inputs: torch.Tensor):
        if len(inputs) != n_inputs:
            raise ValueError(f"Expected {n_inputs} inputs, got {len(inputs)}")
        outputs = [
            torch.empty(shape, dtype=dtype, device="cpu")
            for shape, dtype in out_specs
        ]
        ptrs = [ctypes.c_void_p(t.data_ptr()) for t in inputs]
        ptrs += const_ptrs
        ptrs += [ctypes.c_void_p(t.data_ptr()) for t in outputs]
        c_fn(*ptrs)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    return call
