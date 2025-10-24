# linalg/utils.py

import numpy as np
from typing import Any

from .types import Array, ArrayLike

def _ensure_scalar(x: Any) -> Any:
    """
    Returns x as a Python scalar (float, int, etc.).
    Accepts:
        - Python scalars (float, int, etc.)
        - NumPy 0d arrays
        - NumPy 1d arrays of length one

    Raises:
        ValueError for arrays/lists with more than one element.
    """
    # Python scalar
    if isinstance(x, (int, float, complex)):
        return x

    arr = np.asarray(x)

    if arr.ndim == 0: # 0D array
        return arr.item()
    elif arr.ndim == 1 and arr.shape[0] == 1: # 1D array of length 1
        return arr.item()
    elif arr.ndim == 2 and arr.shape == (1, 1): # (1,1) array
        return arr.item()
    else:
        raise ValueError(f"Input cannot be converted to scalar. Shape {arr.shape}")


def _ensure_vector(x: ArrayLike, as_column: bool = True) -> Array:
    """
    Ensure an input is returned as a (column) vector.

    Args:
        x: Input object to be converted (list, array, etc.)
        as_column: If True, output shape will be (n,1). If False, shape will be (n,).
    Returns:
        Array: Shape (n,1) if as_column, otherwise (n,)
    """
    arr = np.asarray(x)

    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr if as_column else arr.squeeze()
    elif arr.ndim == 1:
        return arr.reshape(-1, 1) if as_column else arr
    elif arr.ndim == 0:
        out = arr.reshape(1, 1)
        return out if as_column else out.squeeze()
    else:
        raise ValueError(f"Input cannot be converted to a 1D vector. Shape {arr.shape}")


def _ensure_matrix(x: ArrayLike) -> Array:
    """ Ensure input is a 2D matrix

    - Scalar inputs (0D) become arrays of shape (1, 1)
    - 1D inputs become row matrices of shape (1, n)
    - 2D inputs are passed through as is
    - Other shapes raise an error
    """
    arr = np.asarray(x)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim == 0:
        return arr.reshape(1, 1)
    else:
        raise ValueError(f"Input cannot be converted to a 2D matrix. Shape {arr.shape}")


def _ensure_square_matrix(x: ArrayLike) -> Array:
    """Ensure input is a 2d square matrix"""
    matrix = _ensure_matrix(x)
    shp = matrix.shape
    if shp[0] != shp[1]:
        raise ValueError(f"Array is not square. Shape {shp}")
    return matrix


def add_diag_jitter(matrix: Array, jitter: float|Array = 1e-6) -> Array:
    """Add constant to diagonal of square matrix.

    This function adds a scalar constant to the diagonal of a square matrix
    (2d array). The constant is typically a small "jitter" intended to
    promote numerical stability and ensure numerical positive definiteness
    for Cholesky factorizations and matrix inversion.

    Args:
        matrix: A 2d square array.
        jitter: The value to add to the diagonal. Defaults to 1e-6.

    Returns:
        The matrix with jitter added to the diagonal.

    Notes:
        This function was inspired by the `gpjax` function `add_jitter`.
    """
    matrix = _ensure_square_matrix(matrix)
    jitter = _ensure_scalar(jitter)

    return np.fill_diagonal(matrix, matrix.diagonal() + jitter)
