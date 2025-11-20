# array_backend/utils.py
"""
Utility functions for array canonicalization used by probpipe.

Notes for backend support
-------------------------
This module currently imports numpy as `np`. When adding a JAX backend,
replace `np` with a small backend shim (e.g., `from .backend import xp as np`)
where `backend.xp` resolves to either `numpy` or `jax.numpy` depending on the
configured backend. The functions here are written so that the only change
needed for backend switching is the `np` import.

All functions that return arrays accept `copy: bool = True`. When `copy=True`
the returned array is guaranteed to be a different object from the input.
This makes behavior explicit and is compatible with JAX-style functional APIs.

When jax backend support is added, will need to replace
`return out.copy() if copy else out` statements with something like
`copy(out)`, a shim for np.array(x, copy=True)/x.copy() [numpy]
and jnp.array(x) [jax]. See Jumpy for example of package supporting both
backends.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Tuple

from ..custom_types import Array, ArrayLike


def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)

def _as_array(x: Any) -> Array:
    try:
        return np.asarray(x)
    except Exception as e:
        raise TypeError(
            f"Could not convert input to array.\n"
            f"Input type: {type(x).__name__}\n"
            f"Input value: {repr(x)}\n"
            f"Original error: {e}"
        ) from e  

def _is_numpy_scalar(x: Any) -> bool:
    """Return true if object is a numpy generic or Python scalar"""
    return np.isscalar(x) or isinstance(x, np.generic)


def _ensure_real_scalar(x: Any, *, as_array: bool = False) -> float|int|Array:
    """
    Return a Python scalar or 0d array for inputs that contain a single real value.

    Accepts:
      - Python scalars (int, float)
      - numpy scalar types (np.float64(...), np.int32(...))
      - 0-D numpy arrays (shape == ())

    Returns:
        If as_array=False (default), returns a Python scalar (float or int).
        If as_array=True, returns a 0-D array.

    Raises:
      ValueError if input contains more than one element or is not a float/int.
    """
    # fast path for Python/numpy scalar
    if _is_numpy_scalar(x):
        # np.iscomplexobj handles python numbers too (returns False for ints/floats)
        if np.iscomplexobj(x):
            raise ValueError(f"_ensure_real_scalar: input is complex-valued: {x!r}")
        if isinstance(x, np.generic) and not as_array:
            return x.item()
        if as_array:
            return np.array(x)
        # Python scalar
        return x

    arr = _as_array(x)
    if arr.size != 1:
        raise ValueError(f"_ensure_real_scalar: input must contain exactly one element; got size={arr.size}, shape={arr.shape}")
    if np.iscomplexobj(arr):
        raise ValueError(f"_ensure_real_scalar: input is complex-valued (shape={arr.shape}).")

    if as_array:
        return np.array(arr.reshape(()))  # 0-D array
    return arr.item()


def _ensure_scalar(x: Any) -> Any:
    """
    Return a Python scalar (via np.ndarray.item()) for inputs that contain a single value.

    Accepts:
      - Python scalars (int, float, complex)
      - numpy scalar types (np.float64(...), np.int32(...))
      - 0-D numpy arrays (shape == ())
      - 1-D numpy arrays of length 1 (shape == (1,))
      - 2-D numpy arrays of shape (1,1)

    Raises:
      ValueError if input contains more than one element.
    """
    # handle Python/numpy scalar quickly
    if _is_numpy_scalar(x):
        if isinstance(x, np.generic):
            return x.item()
        return x  # Python scalar

    arr = _as_array(x)

    if arr.size == 1:
        return arr.item()
    else:
        raise ValueError(f"_ensure_scalar: input cannot be converted to a scalar (size={arr.size}, shape={arr.shape})")


def _ensure_vector(x: ArrayLike, *, as_column: bool = False,
                   length: int | None = None, copy: bool = True) -> Array:
    """
    Ensure input is returned as a 1-D vector (canonical shape (n,)) by default.
    If as_column=True, return shape (n,1).

    Accepts:
      - 1D arrays -> (n,) (or (n,1) if as_column)
      - 2D arrays shaped (n,1) or (1,n) -> converted appropriately
      - 0D scalar -> treated as length-1 vector (1,) or (1,1) if as_column

    Raises:
      ValueError for incompatible shapes (ndim > 2 or 2D with both dims >1)
    """
    arr = _as_array(x)

    if arr.ndim == 0:
        v = arr.reshape((1,))
        out = v.reshape((-1, 1)) if as_column else v
    elif arr.ndim == 1:
        out = arr.reshape((-1, 1)) if as_column else arr
    elif arr.ndim == 2:
        num_rows, num_cols = arr.shape
        if num_rows == 1 or num_cols == 1:
            v = np.ravel(arr)
            out = v.reshape((-1, 1)) if as_column else v
        else:
            raise ValueError(f"_ensure_vector: 2D input has shape {arr.shape}, which is not a vector (expected (n,1) or (1,n)).")
    else:
        raise ValueError(f"_ensure_vector: input has too many dimensions (ndim={arr.ndim}).")

    # validate vector length
    if length is not None and out.size != length:
        raise ValueError(f"_ensure_vector: required length {length}. Got {out.size}.")

    return out.copy() if copy else out


def _ensure_matrix(x: ArrayLike, *, as_row_matrix: bool = False,
                   num_rows: int | None = None, num_cols: int | None = None,
                   copy: bool = True) -> Array:
    """ Ensure input is a 2D matrix

    - Scalar inputs (0D) become arrays of shape (1, 1)
    - 1D inputs become:
        - shape (1, n) if as_row_matrix is True
        - shape (n, 1) if as_row_matrix is False
    - 2D inputs are passed through as is
    - Other shapes raise an error
    """
    arr = _as_array(x)

    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 1:
        if as_row_matrix:
            out = arr.reshape(1, -1)
        else:
            out = arr.reshape(-1, 1)
    elif arr.ndim == 0:
        out = arr.reshape(1, 1)
    else:
        raise ValueError(f"_ensure_matrix: Input cannot be converted to a 2D matrix. Shape {arr.shape}")

    if num_rows is not None and out.shape[0] != num_rows:
        raise ValueError(f"_ensure_matrix: Required {num_rows} rows. Got {out.shape[0]}.")

    if num_cols is not None and out.shape[1] != num_cols:
        raise ValueError(f"_ensure_matrix: Required {num_cols} columns. Got {out.shape[1]}.")

    return out.copy() if copy else out


def _ensure_square_matrix(x: ArrayLike, n: int | None = None, *, copy: bool = True) -> Array:
    """Ensure input is a 2d square matrix"""
    matrix = _ensure_matrix(x, copy=copy)
    num_rows, num_cols = matrix.shape
    if num_rows != num_cols:
        raise ValueError(f"Array is not square. Shape {matrix.shape}")

    if n is not None and matrix.shape[0] != n:
        raise ValueError(f"Required matrix dimension {n}. Got {matrix.shape[0]}.")

    return matrix


# ------------------------------------------------------------------------------
# Batch arrays
# ------------------------------------------------------------------------------

def _ensure_batch_array(x: ArrayLike, value_shape: Tuple[int, ...] | None = None,
                        *, copy: bool = True) -> Array:
    """Ensure `x` has a leading batch axis and optionally enforce value shape.

    A batch array is defined as an array with dimension at least two, where the
    leading dimension is interpreted as the batch dimension, while the remaining
    dimensions describe the shape of a single value in the batch.

    This function converts single values to singleton batches, and optionally
    validates a batch array against a specified value shape.

    The logic for converting single value to singleton batches is as follows:
    - 0d scalar is treated as a single value, expanded as:
        - batch scalar with shape (1,), if `value_shape` is None
        - batch (1, *value_shape), if `value_shape` is provided
    - 1d array (d,) is treated as single vector and expanded to batch shape (1,d)
    - array of >= 2 dimensions treated as already having batch axis, the first axis
        - One exception: if array has number of axes matching number of axes implied
          by `value_shape`, then treated as a single value of that shape.

    If `value_shape` is not None, validation is performed. After the above
    conversion step, the resulting array must have value shape (the shape implied
    by all but the first dimension) equal to `value_shape`. Note that at present,
    singleton trailing dimensions are not squeezed, implying that (b,n,m) is
    treated as different than (b,n,m,1).

    Args:
        x: Array-like input.
        value_shape: Optional tuple specifying the expected shape of a single
                     value (e.g., (n,) for vectors, (n, m) for matrices). If
                     provided the function will raise if the value shape does
                     not match.

    Returns:
        Array: an array with a leading batch axis. If `value_shape` was provided,
        shape will be (B, *value_shape). Otherwise shape will be (B, ...)
        where "..." are the original trailing dims.

    Raises:
        ValueError: If `value_shape` is provided and the per-value shape doesn't
        match after canonicalization.
    """
    arr = _as_array(x)

    # Convert single values to singleton batch
    if value_shape is not None:
        # If input is exactly single-value shaped, expand to batch of size 1
        if arr.ndim == len(value_shape):
            arr = arr[np.newaxis, ...]
    else:
        if arr.ndim == 0:
            arr = arr.reshape((1,))  # () -> (1,)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :] # (d,) -> (1,d)

    # Optionally validate value shape
    if value_shape is not None:
        arr_value_shape = arr.shape[1:]
        if arr_value_shape != tuple(value_shape):
            raise ValueError(
                f"Batch array with value shape {arr_value_shape} does not match required value shape {tuple(value_shape)}."
            )

    return arr.copy() if copy else arr


def _ensure_batch_real_scalar(x: ArrayLike, *, copy: bool = True) -> Array:
    """
    Ensure `x` is a batch of real scalars with shape (B,).

    - scalar -> (1,)
    - 1D array (B,) -> returned (B,)
    - higher-dimensional inputs raise an error

    Args:
        x: scalar or array-like
        copy: if True, return a copy (new array). If False, may return a view.

    Returns:
        Array with shape (B,) containing real numbers.

    Raises:
        ValueError if the input contains complex numbers or has ndim >= 2.
    """
    if _is_numpy_scalar(x):
        out = _ensure_real_scalar(x, as_array=True).reshape((1,))
        return out.copy() if copy else out

    arr = _as_array(x)
    if arr.ndim == 0:
        out = _ensure_real_scalar(arr, as_array=True).reshape((1,))
        return out.copy() if copy else out
    if arr.ndim == 1:
        if np.iscomplexobj(arr):
            raise ValueError("_ensure_batch_real_scalar: input contains complex values.")
        return arr.copy() if copy else arr
    raise ValueError(f"_ensure_batch_real_scalar: expected scalar or 1D array. Got shape={arr.shape}.")


def _ensure_batch_vector(x: ArrayLike, length: int | None = None,
                         *, copy: bool = True) -> Array:
    """Ensure `x` is a batch of vectors and return shape (B, d).

    This function returns an array of shape (B, d) encoding a batch vector,
    where B is the batch size and d the length of a single vector. Two-dimensional
    arrays are returned unchanged. Lower dimensional arrays are converted to
    canonical vector shape (d,) via `_ensure_vector`, which is then reshaped as
    to (1,d) (representing a singleton batch). Higher dimensional arrays raise
    an error.

    Examples:
      - Input shape (d,) -> returned shape (1, d)
      - Input shape (B, d) -> returned shape (B, d)
      - other shapes -> raises error

    Returns:
        Array of shape (B, d).

    Raises:
        ValueError: If the input cannot be interpreted as a batch of vectors.
    """
    arr = _as_array(x)

    # If single vector value, standardize to (1,d)
    if arr.ndim < 2:
        v = _ensure_vector(arr, as_column=False, length=length, copy=copy)
        return _ensure_batch_array(v, value_shape=v.shape, copy=copy)

    # Batch vector must be two dimensional
    if arr.ndim != 2:
        raise ValueError(
            f"_ensure_batch_vector: Array of shape {arr.shape} is not a batch vector. Require shape (n_batch, d)."
        )

    # Validate vector length
    if length is not None and arr.shape[1] != length:
        raise ValueError(f"_ensure_batch_vector: Required vector length {length}. Got {arr.shape[1]}.")

    return arr.copy() if copy else arr


def _ensure_batch_matrix(x: ArrayLike, num_rows: int | None = None, num_cols: int | None = None,
                         as_row_matrix: bool = True, *, copy: bool = True) -> Array:
    """Ensure `x` is a batch of matrices and return shape (B, n, m).

    This function returns an array of shape (B, n, m) encoding a batch matrix,
    where B is the batch size and (n, m) is the shape of each matrix in the
    batch. Three-dimensional arrays are returned unchanged. Lower dimensional
    arrays are converted to matrices via `_ensure_matrix`, which is then reshaped
    to (1,n,m) (representing a singleton batch). Higher dimensional arrays raise
    an error.

    Examples:
      - Input shape (n,) -> returned shape (1, 1, n) (as_row_matrix = True) or (1, n, 1) (False)
      - Input shape (n, m) -> returned shape (1, n, m)
      - Input shape (B, n, m) -> returned shape (B, n, m)
      - other Shapes -> raises error

    Returns:
        Array of shape (B, n, m).

    Raises:
        ValueError: If the input cannot be interpreted as a batch of matrices.
    """
    arr = _as_array(x)

    # If single matrix value, standardize to (1,n,m)
    if arr.ndim < 3:
        mat = _ensure_matrix(arr, as_row_matrix=as_row_matrix, copy=copy)
        return _ensure_batch_array(mat, value_shape=mat.shape, copy=copy)

    # Batch matrix must be three dimensional
    if arr.ndim != 3:
        raise ValueError(
            f"Array of shape {arr.shape} is not a batch matrix. Require shape (n_batch, n_row, n_col)."
        )

    if num_rows is not None and arr.shape[1] != num_rows:
        raise ValueError(f"_ensure_batch_matrix: Required {num_rows} rows. Got {arr.shape[1]}.")

    if num_cols is not None and arr.shape[2] != num_cols:
        raise ValueError(f"_ensure_batch_matrix: Required {num_cols} rows. Got {arr.shape[2]}.")

    return arr.copy() if copy else arr