# linalg/operations.py
"""
Functions that accept linear operator inputs, potentially also returning 
linear operator outputs. This includes the core basic operations like 
`solve()` that are also implemented as `LinOp` methods. This allows users
to call the method via `solve(linop, ...)` rather than `linop.solve(...)`.
Also included are more specialized operations like `mah_dist_squared()`
that are not in one-to-one correspondence with `LinOp` methods.
"""

import numpy as np
from typing import Any, TypeAlias

from ..custom_types import Array, ArrayLike
from ..array_backend.utils import _is_array, _ensure_matrix
from .linear_operator import (
    _as_linear_operator, 
    LinOpLike,
    CholeskyFactor
)


# -----------------------------------------------------------------------------
# Expose LinOp methods as functions
# -----------------------------------------------------------------------------

def shape(A: LinOpLike) -> tuple[int, int]:
    return _as_linear_operator(A).shape

def dtype(A: LinOpLike) -> Any:
    return _as_linear_operator(A).dtype

def diag(A: LinOpLike) -> Array:
    return _as_linear_operator(A).diag()

def to_dense(A: LinOpLike) -> Array:
    return _as_linear_operator(A).to_dense()
    
def solve(A: LinOpLike, b: ArrayLike, **kwargs) -> Array:
    return _as_linear_operator(A).solve(b, **kwargs)

def cholesky(A: LinOpLike, lower: bool = True, **kwargs) -> CholeskyFactor:
    return _as_linear_operator(A).cholesky(lower=lower, **kwargs)

def logdet(A: LinOpLike) -> float:
    return _as_linear_operator(A).logdet()

def trace(A: LinOpLike) -> float:
    return _as_linear_operator(A).trace()

# -----------------------------------------------------------------------------
# Other specialized operations, not core LinOp methods
# -----------------------------------------------------------------------------

def trace_Ainv_B(A: LinOpLike, B: LinOpLike) -> float:
    """ Compute the trace of the product of an inverse of a matrix times another matrix

    For square matrices :math:`A` and :math:`B` of equal dimension, with :math:`A`
    invertible, this function computes :math:`\text{trace}(A^{-1}B)`. At present,
    this function will exploit structure in the linear operator :math:`A`, but 
    :math:`B` will be handled as a dense matrix.

    Args:
        A: LinOpLike, square (d, d) and invertible matrix.
        B: LinOpLik, square (d, d) matrix.
    
    Returns:
        float, the trace.
    """
    A = _as_linear_operator(A)
    B = _as_linear_operator(B)
    A._check_square()
    B._check_square()
    if A.shape[1] != B.shape[0]:
        raise np.linalg.LinAlgError(
            f"trace_Ainv_B requires A and B to be square and of equal dimension.\n" 
            f"Got A and B with shapes {A.shape} and {B.shape}, respectively."
        )
     
    return trace(solve(A, to_dense(B)))


def mah_dist_squared(x: ArrayLike,
                     A: LinOpLike, 
                     y: ArrayLike | None = None) -> Array:
    """ Compute squared Mahalanobis distance(s) between one or more vectors

    The squared Mahalanobis distance between vectors :math:`x` and :math:`y`
    with respect to the invertible weight matrix :math:`A` is defined as:

    .. math::

        D^2(x, y; A) = (x - y)^\\top A^{-1} (x - m)

    If multiple observations are provided as rows in the matrix 
    :math:`x \\in \\mathbb{R}^{n \\times d}` or :math:`y \\in \\mathbb{R}^{n \\times d}`
    the function computes:

    .. math::

        D_i^2 = (x_i - y_i)^\\top A^{-1} (x_i - y_i), \\quad i = 1, \\ldots, n.

    In this case, either `x` and `y` must both contain the same number of rows, or 
    one of them must specify a single point (in which case the point will be 
    repeated to match the length of the other input). If `y` is `None`, then it 
    will be set to the zero vector.
    
    Args:
        x: ArrayLike, of shape (d,) or (n,d).
        A: LinOpLike, invertible and shape (d,d).
        y: ArrayLike or None, shape (d,) or (d,n).

    Notes:
        Strictly speaking, the Mahalanobis distance requires :math:`A` to be positive definite.
        However, the quadratic form can be computed so long as :math:`A` is invertible. This
        function only requires this weaker condition. 

    Returns:
        Array of shape (n,)
    """
    A = _as_linear_operator(A)
    A._check_square()
    d = A.shape[0]
    X = _ensure_matrix(x, as_row_matrix=True, num_cols=d)
    if y is not None:
        Y = _ensure_matrix(y, as_row_matrix=True, num_cols=d)
        if Y.shape[0] not in (1, X.shape[0]):
            raise ValueError("y must have same batch dimension `n` as x, or have batch dimension one.")
        X = X - Y

    Ainv_Xt = solve(A, X.T).T # (n, d)
    return np.sum(X * Ainv_Xt, axis=1)
