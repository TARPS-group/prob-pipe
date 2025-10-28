# linop.py
from __future__ import annotations

from typing import Tuple, Any, Iterable
import numpy as np
from abc import ABC, abstractmethod
from functools import singledispatch
from scipy.linalg import cholesky, solve_triangular
import math

from ..custom_types import Array, ArrayLike
from .utils import (
    _ensure_scalar,
    _ensure_vector,
    _ensure_matrix,
    _ensure_square_matrix
)

# TODO:
# - Add **kwargs to methods in ABC
# - Improve dtype promotion (e.g., in ProductLinOp)
# - Standardize tags / propagate tags

# ---- Core abstract class ----

class LinOp(ABC):
    """Abstract base class for a linear operator.

    Concrete subclasses must provide `shape`, `dtype`, `matvec`, `rmatvec`, and
    `to_dense`. LinOp provides convenience arithmetic (__matmul__, __add__, __mul__, ...)
    that constructs composed operators without densifying where possible. The
    `matvec`/`rmatvec` and `matmat`/`rmatmat` methods provide the option to implement matrix-vector
    and matrix-matrix products without densifying the operator.

    Attributes:
        _flags: a set of semantic flags (e.g., "symmetric", "positive_definite").
                See `add_flag`, `has_flag`.
    """

    def __init__(self) -> None:
        self._flags: set[str] = set()

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Return (n_out, n_in)."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """Return dtype (e.g., np.float64)."""
        ...

    # ---- Minimal numeric primitives ----
    @abstractmethod
    def to_dense(self) -> Array:
        """Return dense array representation of the operator.

        Implementations may raise if dense form is unaffordable.
        """
        ...

    @property
    def is_dense(self) -> bool:
        """Default is_dense method only treats DenseLinOp as dense. Composite
           LinOps should overwrite this."""
        return isinstance(op, DenseLinOp)

    # ---- Optional convenience methods that implementors may override for speed ----
    def matvec(self, x: ArrayLike) -> Array:
        """Return A @ x for x shape (n_in,) -> (n_out,)."""
        x = _ensure_vector(x)
        return self.to_dense() @ x

    def rmatvec(self, x: ArrayLike) -> Array:
        """Return A^T @ x for x shape (n_out,) -> (n_in,)."""
        x = _ensure_vector(x)
        return self.to_dense().T @ x

    def matmat(self, X: ArrayLike) -> Array:
        """Return A @ X for X shape (n_in, k) or (n_in,)"""
        X = _ensure_matrix(X)
        return self.to_dense() @ X

    def rmatmat(self, X: ArrayLike) -> Array:
        """Return A.T @ X for X shape (n_out, k) or (n_out,)"""
        X = _ensure_matrix(X)
        return self.to_dense().T @ X

    def solve(self, b: ArrayLike) -> Array:
        """Solve A x = b; default uses dense fallback."""
        return np.linalg.solve(self.to_dense(), b)

    def cholesky(self, lower: bool = True) -> LinOp:
        """Return triangular LinOp L (lower) or L.T (upper) such that A = L @ L.T. Default: dense path."""
        L = cholesky(self.to_dense(), lower=lower)
        return TriangularLinOp(L, lower=lower)

    def diag(self) -> Array:
        """Return diagonal of operator; default uses dense fallback."""
        A = self.to_dense()
        return np.diag(A)

    def logdet(self) -> float:
        """Return log determinant; default uses dense fallback."""
        A = self.to_dense()
        sign, log_det = np.linalg.slogdet(A)
        return float(log_det)

    def trace(self) -> float:
        """Return trace of operator; default uses dense fallback."""
        return np.trace(self.to_dense())

    # ---- Flags API ----
    def add_flag(self, flag: str) -> None:
        """Attach a semantic flag (e.g., 'symmetric', 'positive_definite')."""
        self._flags.add(flag)

    def remove_flag(self, flag: str) -> None:
        """Remove a previously attached flag (if present)."""
        self._flags.discard(flag)

    def has_flag(self, flag: str) -> bool:
        """Return True if flag is attached."""
        return flag in self._flags

    # ---- Basic operator algebra builds (do not densify unless unavoidable) ----
    def __matmul__(self, other: LinOp) -> LinOp:
        """Return operator representing self @ other (composition)."""
        return ProductLinOp(self, other)

    def __rmatmul__(self, other: LinOp) -> LinOp:
        # other @ self
        return ProductLinOp(other, self)

    def __add__(self, other: LinOp) -> LinOp:
        return SumLinOp([self, other])

    def __radd__(self, other: LinOp) -> LinOp:
        return SumLinOp([other, self])

    def __sub__(self, other: LinOp) -> LinOp:
        return SumLinOp([self, ScaledLinOp(other, -1.0)])

    def __rsub__(self, other: LinOp) -> LinOp:
        return SumLinOp([other, ScaledLinOp(self, -1.0)])

    def __mul__(self, scalar: float) -> LinOp:
        return ScaledLinOp(self, scalar)

    def __rmul__(self, scalar: float) -> LinOp:
        return ScaledLinOp(self, scalar)

    @property
    def T(self) -> LinOp:
        """Return a transposed operator view."""
        return TransposedLinOp(self)

    # ---- Default representations ----
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"


# ---- Structural composite operator classes ----

class DenseLinOp(LinOp):
    """Dense linear operator backed by a numpy array."""

    def __init__(self, arr: ArrayLike, copy: bool = True) -> None:
        super().__init__()

        matrix = _as_matrix(arr)

        # TODO: copying should probably be handled in _as_matrix().
        # i.e., matrix = _as_matrix(arr, copy=copy)
        if copy:
            self.array = matrix.copy()
        else:
            self.array = matrix

        self._dtype = self.array.dtype

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def dtype(self) -> Any:
        return self._dtype

    def to_dense(self) -> Array:
        return self.array


class DiagonalLinOp(LinOp):
    """Diagonal operator represented by a 1D array of diagonal entries."""

    def __init__(self, diag: ArrayLike) -> None:
        """`diag` may be higher-dimensional array, but will be flattened."""
        super().__init__()
        self.diagonal = np.asarray(diag).ravel()
        self._n = int(self.diagonal.size)
        self._dtype = self.diagonal.dtype

        self.add_flag("symmetric")
        if np.all(self.diagonal > 0):
            self.add_flag("positive_definite")

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> Any:
        return self._dtype

    def matvec(self, x: Array) -> Array:
        x = _ensure_vector(x)
        return self.diagonal * x

    def rmatvec(self, x: Array) -> Array:
        return self.matvec(x)

    def matmat(self, X: Array) -> Array:
        X = _ensure_matrix(X)
        return X * self.diagonal[:,np.newaxis]

    def rmatmat(self, X: Array) -> Array:
        return self.matmat(X)

    def to_dense(self) -> Array:
        return np.diag(self.diagonal)

    def solve(self, b: Array) -> Array:
        """
        For consistency with np.linalg.solve(), b can be (n,) or (n,b).
        """
        if np.any(self.diagonal == 0):
            raise np.linalg.LinAlgError("Diagonal contains zero entries; not invertible.")

        if b.ndim < 2:
            b = _ensure_vector(b, as_column=True)
        b = _ensure_matrix(b, num_rows=self._n)

        return b / self.diagonal[:,np.newaxis]

    def cholesky(self) -> LinOp:
        if np.any(self.diagonal <= 0):
            raise np.linalg.LinAlgError("Diagonal has non-positive entries; cholesky not defined.")
        return DiagonalLinOp(np.sqrt(self.diagonal))

    def diag(self) -> Array:
        return self.diagonal.copy()

    def logdet(self) -> float:
        if np.any(self.diagonal <= 0):
            raise np.linalg.LinAlgError("Non-positive diagonal entries; logdet undefined.")
        return float(np.sum(np.log(self.diagonal)))


class TriangularLinOp(LinOp):
    """Triangular operator represented by lower or upper triangular matrix L (2D array).

    The operator interprets stored matrix `tri` so that:
      - if lower==True: stored tri is lower triangular and operator is L @ x
      - if lower==False: stored tri is upper triangular and operator is U @ x
    """

    def __init__(self, tri: Array, lower: bool = True) -> None:
        super().__init__()
        tri = _ensure_square_matrix(tri)
        self.tri = tri
        self.lower = bool(lower)
        self._n = self.tri.shape[0]
        self._dtype = self.tri.dtype

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> Any:
        return self._dtype

    def matvec(self, x: Array) -> Array:
        x = np.asarray(x)
        return self.tri @ x

    def rmatvec(self, x: Array) -> Array:
        x = np.asarray(x)
        return self.tri.T @ x

    def matmat(self, X: Array) -> Array:
        X = np.asarray(X)
        return self.tri @ X

    def rmatmat(self, X: Array) -> Array:
        X = np.asarray(X)
        return self.tri.T @ X

    def to_dense(self) -> Array:
        return np.array(self.tri)

    def solve(self, b: Array, unit_diagonal: bool = False,
              overwrite_b: bool = False, check_finite: bool = True) -> Array:
        """
        Solve triangular system Lx=b or Ux=b. Optional arguments are forwarded
        to scipy.solve_triangular.
        """

        return solve_triangular(self.tri, b, lower=self.lower,
                                unit_diagonal=unit_diagonal,
                                overwrite_b=overwrite_b, check_finite=check_finite)


class TransposedLinOp(LinOp):
    """View representing the transpose of an existing operator (no densify)."""

    def __init__(self, op: LinOp) -> None:
        super().__init__()
        self.op = op

    @property
    def shape(self) -> Tuple[int, int]:
        n_out, n_in = self.op.shape
        return (n_in, n_out)

    @property
    def dtype(self) -> Any:
        return self.op.dtype

    def matvec(self, x: Array) -> Array:
        # matvec of transpose is rmatvec of base
        return self.op.rmatvec(x)

    def rmatvec(self, x: Array) -> Array:
        return self.op.matvec(x)

    def matmat(self, X: Array) -> Array:
        return self.op.rmatmat(X)

    def rmatmat(self, X: Array) -> Array:
        return self.op.matmat(X)

    def to_dense(self) -> Array:
        return self.op.to_dense().T


# ---- Composite operator types: product, sum, scaled ----

class ProductLinOp(LinOp):
    """Operator representing composition A @ B where A and B are LinOp and shapes agree."""

    def __init__(self, A: LinOp, B: LinOp) -> None:
        super().__init__()
        if A.shape[1] != B.shape[0]:
            raise ValueError("Shapes incompatible for product: A.shape[1] != B.shape[0]")
        self.A = A
        self.B = B

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.A.shape[0], self.B.shape[1])

    @property
    def dtype(self) -> Any:
        return self.A.dtype

    def matvec(self, x: Array) -> Array:
        return self.A.matvec(self.B.matvec(x))

    def rmatvec(self, x: Array) -> Array:
        return self.B.rmatvec(self.A.rmatvec(x))

    def matmat(self, X: Array) -> Array:
        return self.A.matmat(self.B.matmat(X))

    def rmatmat(self, X: Array) -> Array:
        return self.B.rmatmat(self.A.rmatmat(X))

    def to_dense(self) -> Array:
        return self.A.to_dense() @ self.B.to_dense()


class SumLinOp(LinOp):
    """Sum of multiple LinOps."""

    def __init__(self, ops: Iterable[LinOp]) -> None:
        ops = list(ops)
        if not ops:
            raise ValueError("SumLinOp needs at least one operator.")
        # shapes must align
        first_shape = ops[0].shape
        for op in ops:
            if op.shape != first_shape:
                raise ValueError("All operands to SumLinOp must have same shape.")
        super().__init__()
        self.ops = ops

    @property
    def shape(self) -> Tuple[int, int]:
        return self.ops[0].shape

    @property
    def dtype(self) -> Any:
        return self.ops[0].dtype

    @property
    def is_dense(self) -> bool:
        return all(op.is_dense for op in self.ops)

    def matvec(self, x: Array) -> Array:
        out_vec = None
        for op in self.ops:
            if out_vec is None:
                out_vec = op.matvec(x)
            else:
                out_vec = out_vec + op.matvec(x)
        return out_vec

    def rmatvec(self, x: Array) -> Array:
        out_vec = None
        for op in self.ops:
            if out_vec is None:
                out_vec = op.rmatvec(x)
            else:
                out_vec = out_vec + op.rmatvec(x)
        return out_vec

    def matmat(self, X: Array) -> Array:
        """
        Avoids densifying unless all summands are already dense, in which case
        it will typically be more efficient to density the sum operator and then
        multiply with the array.

        TODO: may want to only take this fast pass if X has many columns; would
        need to set some heuristic threshold.
        """
        X = _ensure_matrix(X, as_row_matrix=False)

        # Fast path: all DenseLinOp and X has multiple columns -> densify once
        if X.shape[1] > 1 and self.is_dense:
            return self.to_dense() @ X

        # General path: sum per-op matmat (respects structured/sparse ops)
        X_is_vector = (X.shape[1] == 1)
        out = None
        for op in self.ops:
            term = op.matvec(X) if X_is_vector else op.matmat(X)
            out = term if out is None else out + term
        return out

    def rmatmat(self, X: Array) -> Array:
        X = _ensure_matrix(X, as_row_matrix=False)

        # Fast path: all DenseLinOp and X is matrix -> densify once
        if X.shape[1] > 1 and self.is_dense:
            return self.to_dense().T @ X

        # General path: sum per-op matmat (respects structured/sparse ops)
        X_is_vector = (X.shape[1] == 1)
        out = None
        for op in self.ops:
            term = op.rmatvec(X) if X_is_vector else op.rmatmat(X)
            out = term if out is None else out + term
        return out

    def to_dense(self) -> Array:
        return sum(op.to_dense() for op in self.ops)


class ScaledLinOp(LinOp):
    """Scalar multiple of an operator."""

    def __init__(self, op: LinOp, scalar: float) -> None:
        super().__init__()
        self.op = op
        self.scalar = float(scalar)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.op.shape

    @property
    def dtype(self) -> Any:
        return self.op.dtype

    def matvec(self, x: Array) -> Array:
        return self.scalar * self.op.matvec(x)

    def rmatvec(self, x: Array) -> Array:
        return self.scalar * self.op.rmatvec(x)

    def matmat(self, X: Array) -> Array:
        return self.scalar * self.op.matmat(X)

    def rmatmat(self, X: Array) -> Array:
        return self.scalar * self.op.rmatmat(X)

    def to_dense(self) -> Array:
        return self.scalar * self.op.to_dense()
