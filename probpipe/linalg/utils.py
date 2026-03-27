# linalg/utils.py
from __future__ import annotations

import jax.numpy as jnp
from ..custom_types import Array, ArrayLike
from .._array_utils import (
    _ensure_real_scalar,
    _ensure_square_matrix
)


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
        jitter_arr = jnp.full((n,), jitter_scalar, dtype=jnp.result_type(mat.dtype, jnp.array(jitter_scalar).dtype))
    except ValueError:
        # Not a scalar real: try array-like
        jitter_arr = jnp.asarray(jitter)
        if jitter_arr.ndim == 0:
            jitter_scalar = _ensure_real_scalar(jitter_arr)
            jitter_arr = jnp.full((n,), jitter_scalar, dtype=jnp.result_type(mat.dtype, jnp.array(jitter_scalar).dtype))
        elif jitter_arr.ndim == 1:
            if jitter_arr.shape != (n,):
                raise ValueError(f"add_diag_jitter: jitter must be scalar or shape ({n},). Got {jitter_arr.shape}.")
            if jnp.iscomplexobj(jitter_arr):
                raise ValueError("add_diag_jitter: jitter contains complex values.")
        else:
            raise ValueError(f"add_diag_jitter: jitter must be scalar or 1D array. Got ndim={jitter_arr.ndim}.")

    # Add jitter to diagonal (JAX arrays are immutable; always returns new array)
    idx = jnp.arange(n)
    return mat.at[idx, idx].add(jitter_arr)


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
