# linalg/utils.py

from __future__ import annotations

import numpy as np
from typing import Any, Tuple, TypeAlias

from .linop import LinOp, DenseLinOp
from ..custom_types import Array, ArrayLike
from ..array_backend.utils import (
    _ensure_real_scalar,
    _ensure_square_matrix
)


def _check_square(A: LinOpLike) -> None:
    n_out, n_in = _as_linear_operator(A).shape
    if n_out != n_in:
        raise np.linalg.LinAlgError(f"Linear operator is not square. Has shape ({n_out}, {n_in})") 



def add_diag_jitter(matrix: ArrayLike, jitter: float | ArrayLike = 1e-6, *, copy: bool = True) -> Array:
    """
    Return a new matrix = matrix + jitter * I.

    Args:
      matrix: 2D square array-like
      jitter: scalar or array-like of length n (interpreted elementwise)
      copy: if True (default) operate on and return a copy; if False, attempt to
            operate in-place and return the (possibly mutated) input. If dtype promotion
            is required and copy=False, a promoted-copy will be returned (since dtype
            cannot be changed in-place).

    Returns:
      Array with jitter added to diagonal.

    Raises:
        ValueError on invalid shapes or non-real jitter values.
    """
    mat = _ensure_square_matrix(matrix, copy=copy)
    n = mat.shape[0]

    # Normalize jitter into a 1D real array of length n (or scalar)
    try:
        jitter_scalar = _ensure_real_scalar(jitter)
        jitter_arr = np.full((n,), jitter_scalar, dtype=np.result_type(mat.dtype, np.array(jitter_scalar).dtype))
    except ValueError:
        # Not a scalar real: try array-like
        jitter_arr = np.asarray(jitter)
        if jitter_arr.ndim == 0:
            jitter_scalar = _ensure_real_scalar(jitter_arr)
            jitter_arr = np.full((n,), jitter_scalar, dtype=np.result_type(mat.dtype, np.array(jitter_scalar).dtype))
        elif jitter_arr.ndim == 1:
            if jitter_arr.shape != (n,):
                raise ValueError(f"add_diag_jitter: jitter must be scalar or shape ({n},). Got {jitter_arr.shape}.")
            if np.iscomplexobj(jitter_arr):
                raise ValueError("add_diag_jitter: jitter contains complex values.")
        else:
            raise ValueError(f"add_diag_jitter: jitter must be scalar or 1D array. Got ndim={jitter_arr.ndim}.")

    # Determine result dtype (promote matrix dtype and jitter dtype)
    result_dtype = np.result_type(mat.dtype, jitter_arr.dtype)

    # Decide whether to perform in-place update.
    if copy:
        out = mat.astype(result_dtype, copy=True)
    else:
        if result_dtype == mat.dtype:
            out = mat  # safe to update in-place
        else:
            # cannot change dtype in-place => must return a promoted copy
            out = mat.astype(result_dtype, copy=True)

    # Add jitter to diagonal
    diag_idcs = np.diag_indices(n)
    out[diag_idcs] = out[diag_idcs] + jitter_arr.astype(result_dtype, copy=False)
    return out


def symmetrize_pd(matrix: ArrayLike, 
                  *,
                  symmetrize: bool = True,
                  add_jitter: bool = True,
                  jitter: float = 1e-6,
                  copy: bool = True) -> Array:
    """
    Modify a square matrix to promote (but not guarantee) numerical positive
    definiteness.

    Optionally symmetrize the matrix by taking the average of the matrix with 
    its transpose. In addition, optionally add a constant to the diagonal 
    of the matrix. If both true (default), performs these two steps in this 
    order.

    Args:
    matrix : array-like, shape (d, d)
        Input square matrix intended to be positive definite.
    jitter : float, default 1e-6
        Constant to add to matrix diagonal.

    Returns:
        Array, the modified matrix. A copy of the original matrix if `copy` is True.
    """

    C = _ensure_square_matrix(matrix, copy=copy)

    if symmetrize:
        C = 0.5 * (C + C.T)
    if add_jitter:
        C = add_diag_jitter(C, jitter=jitter)

    return C
