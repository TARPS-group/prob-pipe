# linop.py
from __future__ import annotations

from typing import Any, Iterable, FrozenSet
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import cholesky, solve_triangular
import math

from ..custom_types import Array, ArrayLike
from ..array_backend.utils import (
    _ensure_real_scalar,
    _ensure_vector,
    _ensure_matrix,
    _ensure_square_matrix
)

__all__ = [
    "LinOp",
    "DenseLinOp",
    "DiagonalLinOp",
    "RootLinOp",
    "CholeskyLinOp"
]

# TODO:
# - Update so that matmat/rmatmat always outputs shape (n,k)?
# - Seems that tags are sometimes not used when they could be; e.g., unit_diagonal in TriangularLinOp


# --- Flags: canonical set and helpers ----------------------------------------
ALLOWED_FLAGS = frozenset({
    "symmetric",
    "positive_definite",
    "diagonal",
    "triangular_lower",
    "triangular_upper",
    "unit_diagonal",
    "dense",
})

def _promote_dtype(*dtypes: Any) -> Any:
    """Return numpy result_type for given dtypes/arrays."""
    # Use numpy's result_type semantics (works with numpy scalar dtypes).
    return np.result_type(*dtypes)


# ---- Core abstract class ----

class LinOp(ABC):
    """Abstract base class for a linear operator.

    Concrete subclasses must provide `shape`, `dtype`, `matvec`, `rmatvec`, and
    `to_dense`. LinOp provides convenience arithmetic (__matmul__, __add__, __mul__, ...)
    that constructs composed operators without densifying where possible. The
    `matvec`/`rmatvec` and `matmat`/`rmatmat` methods provide the option to implement matrix-vector
    and matrix-matrix products without densifying the operator.

    Flags are restricted to ALLOWED_FLAGS. Use `.flags` (frozenset) to inspect.

    Attributes:
        _flags: a set of semantic flags (e.g., "symmetric", "positive_definite").
                See `add_flag`, `has_flag`.
    """

    def __init__(self) -> None:
        self._flags: set[str] = set()

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
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
        """Default is False; DenseLinOp overrides to True"""
        return False
    
    # ---- Utility methods for validation ----

    def _check_square(self) -> None:
        """ Throw error if operator is not square """
        n_out, n_in = self.shape
        if n_out != n_in:
            raise np.linalg.LinAlgError(f"Linear operator is not square. Has shape ({n_out}, {n_in})")
    
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

    def solve(self, b: ArrayLike, **kwargs) -> Array:
        """Solve A x = b; default uses dense fallback."""
        self._check_square()
        n, _ = self.shape
        dense_op = self.to_dense()
        
        b = np.asarray(b)
        if b.ndim < 2:
            b = _ensure_vector(b, as_column=True)
        b = _ensure_matrix(b, num_rows=n)

        return np.linalg.solve(dense_op, b)

    def cholesky(self, lower: bool = True, **kwargs) -> TriangularLinOp | DiagonalLinOp:
        """Return triangular LinOp L (lower) or L.T (upper) such that A = L @ L.T. Default: dense path."""
        self._check_square()
        L = cholesky(self.to_dense(), lower=lower)
        return TriangularLinOp(L, lower=lower)

    def diag(self) -> Array:
        """Return diagonal of operator; default uses dense fallback."""
        A = self.to_dense()
        return np.diag(A)

    def logdet(self) -> float:
        """Return log determinant; default uses dense fallback. Raises if sign <= 0"""
        self._check_square()
        A = self.to_dense()
        sign, log_det = np.linalg.slogdet(A)
        if sign <= 0:
            raise np.linalg.LinAlgError("Log-determinant undefined: matrix has non-positive determinant.")
        return float(log_det)

    def trace(self) -> float:
        """Return trace of operator; default uses dense fallback."""
        self._check_square()
        return float(np.linalg.trace(self.to_dense()))

    # ---- Flags API ----
    def add_flag(self, flag: str) -> None:
        """Attach a semantic flag (must be one of ALLOWED_FLAGS)."""
        if flag not in ALLOWED_FLAGS:
            raise ValueError(f"Unknown flag {flag!r}. Allowed: {sorted(ALLOWED_FLAGS)}")
        self._flags.add(flag)

    def remove_flag(self, flag: str) -> None:
        """Remove an attached flag (no-op if missing)."""
        self._flags.discard(flag)

    def has_flag(self, flag: str) -> bool:
        """Return True if flag attached."""
        return flag in self._flags

    @property
    def flags(self) -> FrozenSet[str]:
        """Return frozenset of current flags (read-only view)."""
        return frozenset(self._flags)


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


# ---- Concrete linear operator subclasses ----

class DenseLinOp(LinOp):
    """Dense linear operator backed by a numpy array."""

    def __init__(self, arr: ArrayLike, copy: bool = True) -> None:
        super().__init__()
        self.array = _ensure_matrix(arr, copy=copy)
        self._dtype = self.array.dtype
        self.add_flag("dense")

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    @property
    def dtype(self) -> Any:
        return self._dtype
    
    @property
    def is_dense(self) -> bool:
        return True

    def to_dense(self) -> Array:
        return self.array


class DiagonalLinOp(LinOp):
    """Diagonal operator represented by a 1D array of diagonal entries."""

    def __init__(self, diag: ArrayLike, copy: bool = True) -> None:
        """`diag` may be higher-dimensional array, but will be flattened."""
        super().__init__()
        self.diagonal = _ensure_vector(np.asarray(diag).ravel(), copy=copy)
        self._n = int(self.diagonal.size)
        self._dtype = self.diagonal.dtype

        self.add_flag("diagonal")
        self.add_flag("symmetric")
        if np.all(self.diagonal > 0):
            self.add_flag("positive_definite")

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> Any:
        return self._dtype

    def matvec(self, x: ArrayLike) -> Array:
        x = _ensure_vector(x)
        return self.diagonal * x

    def rmatvec(self, x: ArrayLike) -> Array:
        return self.matvec(x)

    def matmat(self, X: ArrayLike) -> Array:
        X = _ensure_matrix(X)
        return X * self.diagonal[:, np.newaxis]

    def rmatmat(self, X: ArrayLike) -> Array:
        return self.matmat(X)

    def to_dense(self) -> Array:
        return np.diag(self.diagonal)

    def solve(self, b: ArrayLike) -> Array:
        """
        For consistency with np.linalg.solve(), b can be (n,) or (n,k).
        """
        if np.any(self.diagonal == 0):
            raise np.linalg.LinAlgError("Diagonal contains zero entries; not invertible.")

        b = np.asarray(b)
        if b.ndim < 2:
            b = _ensure_vector(b, as_column=True)
        b = _ensure_matrix(b, num_rows=self._n)

        return b / self.diagonal[:, np.newaxis]

    def cholesky(self, lower: bool = True) -> DiagonalLinOp:
        """Note that `lower` has no effect on Cholesky decomposition of diagonal matrix"""
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

    def __init__(self, tri: Array, *, lower: bool = True, copy: bool = True) -> None:
        super().__init__()
        tri = _ensure_square_matrix(tri, copy=copy)
        self.tri = tri
        self.lower = bool(lower)
        self._n = self.tri.shape[0]
        self._dtype = self.tri.dtype

        if self.lower:
            self.add_flag("triangular_lower")
        else:
            self.add_flag("triangular_upper")
        # if strictly triangular with ones on diagonal, mark unit_diagonal
        if np.allclose(np.diag(self.tri), 1.0):
            self.add_flag("unit_diagonal")

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> Any:
        return self._dtype

    def matvec(self, x: ArrayLike) -> Array:
        x = _ensure_vector(x)
        return self.tri @ x

    def rmatvec(self, x: ArrayLike) -> Array:
        x = _ensure_vector(x)
        return self.tri.T @ x

    def matmat(self, X: ArrayLike) -> Array:
        X = _ensure_matrix(X)
        return self.tri @ X

    def rmatmat(self, X: ArrayLike) -> Array:
        X = _ensure_matrix(X)
        return self.tri.T @ X

    def to_dense(self) -> Array:
        return np.array(self.tri)

    def solve(self, b: ArrayLike, *, unit_diagonal: bool = False,
              overwrite_b: bool = False, check_finite: bool = True) -> Array:
        """
        Solve triangular system Lx=b or Ux=b. Optional arguments are forwarded
        to scipy.solve_triangular.
        """
        b_arr = np.asarray(b)
        if b_arr.ndim < 2:
            b_arr = _ensure_vector(b_arr, as_column=True)
        b_arr = _ensure_matrix(b_arr, num_rows=self._n)

        return solve_triangular(self.tri, b_arr, lower=self.lower,
                                unit_diagonal=unit_diagonal,
                                overwrite_b=overwrite_b, check_finite=check_finite)


class RootLinOp(LinOp):
    """A linear operator A represented by its square root S such that A = S @ S.T
       A is guaranteed to be symmetric positive semidefinite, but not necessarily 
       positive definite."""

    def __init__(self, root: LinOp | ArrayLike) -> None:
        super().__init__()

        if isinstance(root, LinOp):
            self.root = root
        else:
            self.root = DenseLinOp(root)

        self._n = self.root.shape[0]
        self.add_flag("symmetric")

    @property
    def shape(self) -> tuple[int, int]:
        return (self._n, self._n)

    @property
    def dtype(self) -> Any:
        return self.root.dtype

    def matvec(self, x: ArrayLike) -> Array:
        return self.root.matvec(self.root.rmatvec(x))

    def rmatvec(self, x: ArrayLike) -> Array:
        # Operator is symmetric
        return self.matvec(x)

    def matmat(self, X: ArrayLike) -> Array:
        return self.root.matmat(self.root.rmatmat(X))

    def rmatmat(self, X: ArrayLike) -> Array:
        return self.matmat(X)
    
    def solve(self, b: ArrayLike) -> Array:
        """Linear solve using forward backward substitution
        
        To compute A^{-1}b = (S @ S.T)^{-1}b first compute y = S^{-1}b
        then S.T^{-1}y.
        """
        b = np.asarray(b)
        if b.ndim < 2:
            b = _ensure_vector(b, as_column=True)
        b = _ensure_matrix(b, num_rows=self._n)

        S = self.root.to_dense()
        y = np.linalg.solve(S, b)
        return np.linalg.solve(S.T, y)
    
    def diag(self) -> Array:
        S = self.root.to_dense()
        return np.einsum('ij,ij->i', S, S)
    
    def trace(self) -> float:
        S = self.root.to_dense()
        return np.sum(S**2)

    def to_dense(self) -> Array:
        S = self.root.to_dense()
        return S @ S.T
    

class CholeskyLinOp(RootLinOp):
    """A positive definite linear operator A represented by its lower 
       L or upper L.T Cholesky factor such that A = L @ L.T"""

    def __init__(self, root: TriangularLinOp) -> None:
        if not isinstance(root, TriangularLinOp):
            raise ValueError("CholeskyLinOp requires initialization via a TriangularLinOp object.")

        super().__init__(root)
        self.add_flag("positive_definite")


    def solve(self, b: ArrayLike) -> Array:
        """Linear solve using forward backward triangular substitution
        
        To compute A^{-1}b = (L @ L.T)^{-1}b first compute y = L^{-1}b
        then L.T^{-1}y.
        """
        b = np.asarray(b)
        if b.ndim < 2:
            b = _ensure_vector(b, as_column=True)
        b = _ensure_matrix(b, num_rows=self._n)

        S = self.root.to_dense()
        y = solve_triangular(S, b, lower=self.root.lower)
        return solve_triangular(S, y, trans=1, lower=self.root.lower)
    

    def cholesky(self, lower: bool = True) -> TriangularLinOp:
        cholesky_factor = self.root
        if lower == cholesky_factor.lower:
            return cholesky_factor
        else:
            return cholesky_factor.T


class TransposedLinOp(LinOp):
    """A lazy transpose view of an existing linear operator."""

    def __init__(self, op: LinOp) -> None:
        super().__init__()
        self.op = op

        # Flags that are invariant under transpose
        if "dense" in op.flags:
            self.add_flag("dense")
        if "diagonal" in op.flags:
            self.add_flag("diagonal")
        if "symmetric" in op.flags:
            self.add_flag("symmetric")
        if "positive_definite" in op.flags:
            self.add_flag("positive_definite")
        if "unit_diagonal" in op.flags:
            self.add_flag("unit_diagonal")

        # Triangular flags swap sides
        if "triangular_lower" in op.flags:
            self.add_flag("triangular_upper")
        if "triangular_upper" in op.flags:
            self.add_flag("triangular_lower")

    @property
    def shape(self) -> tuple[int, int]:
        n_out, n_in = self.op.shape
        return (n_in, n_out)

    @property
    def dtype(self) -> Any:
        return self.op.dtype

    def matvec(self, x: ArrayLike) -> Array:
        # matvec of transpose is rmatvec of base
        return self.op.rmatvec(x)

    def rmatvec(self, x: ArrayLike) -> Array:
        return self.op.matvec(x)

    def matmat(self, X: ArrayLike) -> Array:
        return self.op.rmatmat(X)

    def rmatmat(self, X: ArrayLike) -> Array:
        return self.op.matmat(X)

    def to_dense(self) -> Array:
        return self.op.to_dense().T


# ---- Composite linear operator types ----------------------------------------

class ProductLinOp(LinOp):
    """Operator representing composition A @ B where A and B are LinOp and shapes agree."""

    def __init__(self, A: LinOp, B: LinOp) -> None:
        super().__init__()
        if not isinstance(A, LinOp) or not isinstance(B, LinOp):
            raise ValueError("ProductLinOp requires LinOp operands.")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Shapes incompatible for product: A.shape[1] != B.shape[0]")
        self.A = A
        self.B = B

        if "dense" in A.flags and "dense" in B.flags:
            self.add_flag("dense")
        if "diagonal" in A.flags and "diagonal" in B.flags:
            # product of two diagonal operators is diagonal
            self.add_flag("diagonal")
            self.add_flag("symmetric")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.A.shape[0], self.B.shape[1])

    @property
    def dtype(self) -> Any:
        return _promote_dtype(self.A.dtype, self.B.dtype)

    def matvec(self, x: ArrayLike) -> Array:
        return self.A.matvec(self.B.matvec(x))

    def rmatvec(self, x: ArrayLike) -> Array:
        return self.B.rmatvec(self.A.rmatvec(x))

    def matmat(self, X: ArrayLike) -> Array:
        return self.A.matmat(self.B.matmat(X))

    def rmatmat(self, X: ArrayLike) -> Array:
        return self.B.rmatmat(self.A.rmatmat(X))

    def to_dense(self) -> Array:
        return self.A.to_dense() @ self.B.to_dense()


class SumLinOp(LinOp):
    """Sum of multiple LinOps."""

    def __init__(self, ops: Iterable[LinOp]) -> None:
        ops = list(ops)
        if not ops:
            raise ValueError("SumLinOp needs at least one operator.")
        if not all(isinstance(op, LinOp) for op in ops):
            raise ValueError("SumLinOp requires all sumands to be LinOps.")

        first_shape = ops[0].shape
        for op in ops:
            if op.shape != first_shape:
                raise ValueError("All operands to SumLinOp must have same shape.")
        super().__init__()
        self.ops = ops

        if all("dense" in op.flags for op in ops):
            self.add_flag("dense")
        if all("diagonal" in op.flags for op in ops):
            self.add_flag("diagonal")
            self.add_flag("symmetric")
        if all("symmetric" in op.flags for op in ops):
            self.add_flag("symmetric")
        if all("positive_definite" in op.flags for op in ops):
            # sum of PD matrices is PD
            self.add_flag("positive_definite")

    @property
    def shape(self) -> tuple[int, int]:
        return self.ops[0].shape

    @property
    def dtype(self) -> Any:
        return _promote_dtype(*(op.dtype for op in self.ops))

    @property
    def is_dense(self) -> bool:
        return all(op.is_dense for op in self.ops)

    def matvec(self, x: ArrayLike) -> Array:
        x = _ensure_vector(x)
        arr = np.zeros((self.shape[0],), dtype=self.dtype)
        for op in self.ops:
            arr = arr + op.matvec(x)
        return arr

    def rmatvec(self, x: ArrayLike) -> Array:
        x = _ensure_vector(x)
        arr = np.zeros((self.shape[1],), dtype=self.dtype)
        for op in self.ops:
            arr = arr + op.rmatvec(x)
        return arr

    def matmat(self, X: ArrayLike) -> Array:
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
        if X.shape[1] == 1:
            return self.matvec(X)
        
        arr = np.zeros((self.shape[0], X.shape[1]), dtype=self.dtype)
        for op in self.ops:
            arr = arr + op.matmat(X)
        return arr

    def rmatmat(self, X: ArrayLike) -> Array:
        X = _ensure_matrix(X, as_row_matrix=False)

        # Fast path: all DenseLinOp and X is matrix -> densify once
        if X.shape[1] > 1 and self.is_dense:
            return self.to_dense().T @ X

        # General path: sum per-op matmat (respects structured/sparse ops)
        if X.shape[1] == 1:
            return self.rmatvec(X)
        
        arr = np.zeros((self.shape[1], X.shape[1]), dtype=self.dtype)
        for op in self.ops:
            arr = arr + op.rmatmat(X)
        return arr

    def to_dense(self) -> Array:
        arr_sum = np.zeros(self.shape, dtype=self.dtype)
        for op in self.ops:
            arr_sum = arr_sum + op.to_dense()
        return arr_sum


class ScaledLinOp(LinOp):
    """Scalar multiple of a linear operator."""

    def __init__(self, op: LinOp, scalar: float) -> None:
        super().__init__()
        if not isinstance(op, LinOp):
            raise ValueError("ScaledLinOp requires a LinOp object.")

        self.op = op
        self.scalar = float(_ensure_real_scalar(scalar, as_array=False))

        if "dense" in op.flags:
            self.add_flag("dense")
        if "diagonal" in op.flags:
            self.add_flag("diagonal")
            self.add_flag("symmetric")
        if "symmetric" in op.flags:
            self.add_flag("symmetric")
        if self.scalar > 0 and "positive_definite" in op.flags:
            self.add_flag("positive_definite")

    @property
    def shape(self) -> tuple[int, int]:
        return self.op.shape

    @property
    def dtype(self) -> Any:
        # scalar might be Python float; use array dtype promotion
        return _promote_dtype(self.op.dtype, np.array(self.scalar).dtype)

    def matvec(self, x: ArrayLike) -> Array:
        return self.scalar * self.op.matvec(x)

    def rmatvec(self, x: ArrayLike) -> Array:
        return self.scalar * self.op.rmatvec(x)

    def matmat(self, X: ArrayLike) -> Array:
        return self.scalar * self.op.matmat(X)

    def rmatmat(self, X: ArrayLike) -> Array:
        return self.scalar * self.op.rmatmat(X)

    def to_dense(self) -> Array:
        return self.scalar * self.op.to_dense()
