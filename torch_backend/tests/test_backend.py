"""Tests for the MimIR torch.compile backend.

Run with:
    cd /workspaces/compile/MimIR
    uv run pytest torch_backend/tests/ -v
"""

import shutil

import pytest
import torch

import torch_backend  # noqa: F401  registers "mimir" backend
from torch_backend._codegen import GraphCodegen, UnsupportedGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _requires_clang():
    return pytest.mark.skipif(
        shutil.which("clang") is None,
        reason="clang not available",
    )


def _compile(fn, *example_inputs, **compile_kwargs):
    return torch.compile(fn, backend="mimir", **compile_kwargs)(*example_inputs)


# ---------------------------------------------------------------------------
# Codegen unit tests (no clang needed)
# ---------------------------------------------------------------------------

class TestGraphCodegenInit:
    def test_setup_world(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        gm, _ = torch._dynamo.export(lambda x, y: x + y)(a, b)
        cg = GraphCodegen(gm, [a, b])
        assert cg.w is not None
        assert cg.driver is not None


class TestUnsupportedOps:
    def test_unsupported_dtype_raises(self):
        a = torch.randn(4, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        gm, _ = torch._dynamo.export(lambda x, y: x + y)(a, b)
        cg = GraphCodegen(gm, [a, b])
        with pytest.raises(UnsupportedGraph, match="dtype"):
            cg._build_mimir_fn()

    def test_unsupported_op_raises(self):
        a = torch.randn(4)
        gm, _ = torch._dynamo.export(lambda x: torch.sigmoid(x))(a)
        cg = GraphCodegen(gm, [a])
        with pytest.raises(UnsupportedGraph):
            cg._build_mimir_fn()


# ---------------------------------------------------------------------------
# End-to-end tests (clang required)
# ---------------------------------------------------------------------------

@_requires_clang()
class TestElementwiseAdd:
    def test_1d_add(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])
        result = _compile(torch.add, a, b)
        torch.testing.assert_close(result, a + b)

    def test_2d_add(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        result = _compile(torch.add, a, b)
        torch.testing.assert_close(result, a + b)

    def test_add_zeros(self):
        a = torch.zeros(8)
        b = torch.zeros(8)
        result = _compile(torch.add, a, b)
        torch.testing.assert_close(result, torch.zeros(8))

    def test_add_ones(self):
        a = torch.ones(6)
        b = torch.ones(6)
        result = _compile(torch.add, a, b)
        torch.testing.assert_close(result, torch.full((6,), 2.0))


@_requires_clang()
class TestElementwiseMul:
    def test_1d_mul(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = _compile(torch.mul, a, b)
        torch.testing.assert_close(result, a * b)

    def test_2d_mul(self):
        a = torch.randn(3, 5)
        b = torch.randn(3, 5)
        result = _compile(torch.mul, a, b)
        torch.testing.assert_close(result, a * b)


@_requires_clang()
class TestElementwiseSub:
    def test_sub(self):
        a = torch.tensor([5.0, 4.0, 3.0])
        b = torch.tensor([1.0, 2.0, 1.0])
        result = _compile(torch.sub, a, b)
        torch.testing.assert_close(result, a - b)


@_requires_clang()
class TestMatmul:
    def test_2x2_matmul(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = _compile(torch.mm, a, b)
        torch.testing.assert_close(result, torch.mm(a, b))

    def test_2x3_times_3x2(self):
        a = torch.randn(2, 3)
        b = torch.randn(3, 2)
        result = _compile(torch.mm, a, b)
        torch.testing.assert_close(result, torch.mm(a, b), atol=1e-5, rtol=1e-5)

    def test_identity_matmul(self):
        a = torch.eye(3)
        b = torch.randn(3, 3)
        result = _compile(torch.mm, a, b)
        torch.testing.assert_close(result, b, atol=1e-6, rtol=1e-6)

    def test_larger_matmul(self):
        """Matmul that exceeds old M*K*N unroll limit but fits new M*N limit."""
        a = torch.randn(32, 32)
        b = torch.randn(32, 32)
        result = _compile(torch.mm, a, b)
        torch.testing.assert_close(result, torch.mm(a, b), atol=1e-4, rtol=1e-4)

    def test_non_square_matmul(self):
        a = torch.randn(16, 64)
        b = torch.randn(64, 16)
        result = _compile(torch.mm, a, b)
        torch.testing.assert_close(result, torch.mm(a, b), atol=1e-4, rtol=1e-4)


@_requires_clang()
class TestComposedOps:
    def test_add_then_mul(self):
        def f(x, y, z):
            return (x + y) * z

        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([1.0, 1.0, 1.0, 1.0])
        c = torch.tensor([2.0, 2.0, 2.0, 2.0])
        result = _compile(f, a, b, c)
        torch.testing.assert_close(result, f(a, b, c))

    def test_linear_no_bias(self):
        """Single matmul as a linear layer."""
        def linear(x, w):
            return torch.mm(x, w.t())

        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        result = _compile(linear, x, w)
        torch.testing.assert_close(result, linear(x, w), atol=1e-5, rtol=1e-5)


@_requires_clang()
class TestFallback:
    def test_unsupported_op_falls_back(self):
        """Unsupported ops should fall back to eager without error."""
        a = torch.randn(4)
        result = torch.compile(torch.sigmoid, backend="mimir")(a)
        torch.testing.assert_close(result, torch.sigmoid(a))

    def test_large_tensor_falls_back(self):
        """Tensors over the unroll limit fall back to eager."""
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        result = torch.compile(torch.add, backend="mimir")(a, b)
        torch.testing.assert_close(result, a + b)
