# test_linop.py
import numpy as np
from scipy.linalg import cholesky
import pytest

from probpipe.linalg.linop import (
    DenseLinOp, DiagonalLinOp, TriangularLinOp, TransposedLinOp,
    ProductLinOp, SumLinOp, ScaledLinOp, ALLOWED_FLAGS
)


def approx(a, b, tol=1e-12):
    return np.allclose(a, b, atol=tol, rtol=0)

class TestDenseLinOp:
    @classmethod
    def setup_class(cls):
        cls.n = 3
        cls.arr = 2 * np.identity(cls.n, dtype=np.float64)
        cls.op = DenseLinOp(cls.arr, copy=True)

    def test_array_properties(self):
        assert self.op.shape == (self.n, self.n)
        assert self.op.dtype == self.arr.dtype
        assert self.op.is_dense
        assert np.array_equal(self.op.to_dense(), self.arr)

    def test_transformation_views(self):
        assert np.array_equal(self.op.T.to_dense(), self.arr.T)
        assert np.array_equal((2 * self.op).to_dense(), 2 * self.arr)
        assert np.array_equal((self.op + self.op).to_dense(), self.arr + self.arr)
        assert np.array_equal((self.op @ self.op).to_dense(), self.arr @ self.arr)

    def test_linalg_operations(self):
        assert approx(self.op.cholesky(lower=True).to_dense(), cholesky(self.arr, lower=True))
        assert np.array_equal(self.op.trace(), np.linalg.trace(self.arr))
        assert approx(self.op.logdet(), np.linalg.slogdet(self.arr)[1])
        assert np.array_equal(self.op.diag(), np.diag(self.arr))

        b1 = np.ones((self.n,))
        b2 = np.ones((self.n, 1))
        B = np.ones((self.n, 4))
        assert approx(self.op.solve(b1), np.linalg.solve(self.arr, b1))
        assert approx(self.op.solve(b2), np.linalg.solve(self.arr, b2))
        assert approx(self.op.solve(B), np.linalg.solve(self.arr, B))

    def test_matvec_matmat(self):
        b1 = np.ones((self.n,))
        b2 = np.ones((self.n, 1))
        B = np.ones((self.n, 4))

        assert np.array_equal(self.op.matvec(b1), self.arr @ b1)
        assert np.array_equal(self.op.matvec(b2), (self.arr @ b2).ravel())
        assert np.array_equal(self.op.matmat(b1), (self.arr @ b1).reshape(-1, 1))
        assert np.array_equal(self.op.matmat(b2), self.arr @ b2)
        assert np.array_equal(self.op.matmat(B), self.arr @ B)


def test_dense_matvec_matmat_dtype():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    op = DenseLinOp(A)
    x = np.array([1.0, 1.0])
    assert approx(op.matvec(x), A @ x)
    X = np.stack([x, x], axis=1)
    assert approx(op.matmat(X), A @ X)
    assert op.dtype == A.dtype
    assert "dense" in op.flags


def test_diagonal_matvec_solve_cholesky_flags():
    diag = np.array([2.0, 3.0])
    d = DiagonalLinOp(diag)
    x = np.array([1.0, 2.0])
    assert approx(d.matvec(x), diag * x)
    b = np.array([2.0, 6.0])
    sol = d.solve(b)
    assert approx(sol.squeeze(), np.array([1.0, 2.0]))
    # cholesky of diagonal is diag(sqrt)
    L = d.cholesky()
    assert isinstance(L, DiagonalLinOp)
    assert approx(L.diag(), np.sqrt(diag))
    assert "diagonal" in d.flags
    assert "symmetric" in d.flags
    assert "positive_definite" in d.flags


def test_triangular_solve_and_flags():
    L = np.array([[1.0, 0.0], [2.0, 3.0]])
    tri = TriangularLinOp(L, lower=True)
    b = np.array([1.0, 5.0])
    x = tri.solve(b)
    # verify forward solve L x = b
    assert approx(tri.matvec(x.squeeze()), b)
    assert "triangular_lower" in tri.flags
    # test rmatvec equivalence
    v = np.array([1.0, 2.0])
    assert approx(tri.rmatvec(v), L.T @ v)


def test_transpose_flags_and_behavior():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    op = DenseLinOp(A)
    op.add_flag("symmetric")
    t = TransposedLinOp(op)
    assert t.shape == (2, 2)
    assert "dense" in t.flags
    assert "symmetric" in t.flags
    v = np.array([1.0, 0.0])
    assert approx(t.matvec(v), A.T @ v)


def test_sum_propagation_and_matmat():
    d1 = DiagonalLinOp(np.array([1.0, 2.0]))
    d2 = DiagonalLinOp(np.array([3.0, 4.0]))
    s = SumLinOp([d1, d2])
    X = np.eye(2)
    # sum of diagonals -> diagonal operator, so matmat matches to_dense @ X
    assert approx(s.matmat(X), s.to_dense() @ X)
    assert "diagonal" in s.flags and "symmetric" in s.flags
    # sum dtype matches promoted
    assert s.dtype == np.result_type(d1.dtype, d2.dtype)


def test_product_dtype_promotion_and_diagonal_product_flag():
    d1 = DiagonalLinOp(np.array([1.0, 2.0], dtype=np.float32))
    d2 = DiagonalLinOp(np.array([3.0, 4.0], dtype=np.float64))
    p = ProductLinOp(d2, d1)
    # dtype should be promoted float64
    assert p.dtype == np.result_type(d2.dtype, d1.dtype)
    # product of two diagonal -> diagonal flag
    assert "diagonal" in p.flags


def test_scaled_preserves_positive_definite_when_scalar_positive():
    d = DiagonalLinOp(np.array([2.0, 3.0]))
    s = ScaledLinOp(d, 2.0)
    assert "positive_definite" in s.flags
    # negative scalar removes positive_definite
    s2 = ScaledLinOp(d, -1.0)
    assert "positive_definite" not in s2.flags


def test_logdet_sign_error():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    op = DenseLinOp(A)
    # determinant is -1 so slogdet sign = -1 -> logdet should raise
    with pytest.raises(np.linalg.LinAlgError):
        _ = op.logdet()
