# linalg/operations.py
"""
Functions that accept linear operator inputs, potentially also returning 
linear operator outputs. This includes the core basic operations like 
`solve()` that are also implemented as `LinOp` methods. This allows users
to call the method via `solve(linop, ...)` rather than `linop.solve(...)`.
Also included are more specialized operations like `mah_dist_squared()`
that are not in one-to-one correspondence with `LinOp` methods.
"""

from typing import TypeAlias

from ..custom_types import Array, ArrayLike
from ..array_backend.utils import _is_array
from .linop import LinOp, DenseLinOp, TriangularLinOp, DiagonalLinOp

LinOpLike: TypeAlias = LinOp | ArrayLike

def _as_linear_operator(A: LinOpLike) -> LinOp:
    """
    Wraps arrays as a DenseLinOp, and returns existing LinOp objects untouched.
    """
    if isinstance(A, LinOp):
        return A
    else:
        try:
            return DenseLinOp(A)
        except Exception as e:
            raise TypeError(
                f"Could not convert A to linear operator\n"
                f"DenseLinOp error: {e}"
            )

def shape(A: LinOpLike):
    return A.shape

def dtype(A: LinOpLike):
    return A.dtype

def diag(A: LinOpLike) -> Array:
    return _as_linear_operator(A).diag()

def to_dense(A: LinOpLike) -> Array:
    return _as_linear_operator(A).to_dense()
    
def solve(A: LinOpLike, b: ArrayLike, **kwargs) -> Array:
    return _as_linear_operator(A).solve(b, **kwargs)

def cholesky(A: LinOpLike, lower: bool = True, **kwargs) -> TriangularLinOp | DiagonalLinOp:
    return _as_linear_operator(A).cholesky(lower=lower, **kwargs)

def logdet(A: LinOpLike) -> float:
    return _as_linear_operator(A).logdet()

def trace(A: LinOpLike) -> float:
    return _as_linear_operator(A).trace()

