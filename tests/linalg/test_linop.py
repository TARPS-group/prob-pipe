# test_linop.py
import numpy as np
import pytest

from probpipe.linalg.linop import (
    DenseLinOp, DiagonalLinOp, TriangularLinOp, TransposedLinOp,
    ProductLinOp, SumLinOp, ScaledLinOp, ALLOWED_FLAGS
)


def approx(a, b, tol=1e-12):
    return np.allclose(a, b, atol=tol, rtol=0)


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
