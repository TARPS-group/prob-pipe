# linalg/utils.py

import numpy as np
from typing import Any

from .types import Array, ArrayLike

# This code uses numpy, but is written to be robust to future suport for
# other backend arrays (e.g., jax). For such support, replace `np` by
# an import from a small backend shim; e.g.,
#   from .backend import xp as np
# where backend.xp is either numpy or jax.numpy.

def _is_numpy_scalar(x: Any) -> bool:
    # handles np.generic and Python scalars
    return np.isscalar(x) or isinstance(x, np.generic)


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

    arr = np.asarray(x)

    if arr.size == 1:
        return arr.item()
    else:
        raise ValueError(f"_ensure_scalar: input cannot be converted to a scalar (size={arr.size}, shape={arr.shape})")


def _ensure_vector(x: ArrayLike, *, as_column: bool = False,
                   length: int = None) -> Array:
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
    arr = np.asarray(x)

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
    if length is not None:
        if out.size != length:
            raise ValueError(f"_ensure_vector: required length {length}. Got {out.size}.")

    return out


def _ensure_matrix(x: ArrayLike, *, as_row_matrix: bool = False,
                   num_rows: int = None, num_cols: int = None) -> Array:
    """ Ensure input is a 2D matrix

    - Scalar inputs (0D) become arrays of shape (1, 1)
    - 1D inputs become:
        - shape (1, n) if as_row_matrix is True
        - shape (n, 1) if as_row_matrix is False
    - 2D inputs are passed through as is
    - Other shapes raise an error
    """
    arr = np.asarray(x)

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

    if num_rows is not None:
        if out.shape[0] != num_rows:
            raise ValueError(f"_ensure_matrix: Required {num_rows} rows. Got {out.shape[0]}.")

    if num_cols is not None:
        if out.shape[1] != num_cols:
            raise ValueError(f"_ensure_matrix: Required {num_cols} rows. Got {out.shape[1]}.")

    return out


def _ensure_square_matrix(x: ArrayLike, n: int = None) -> Array:
    """Ensure input is a 2d square matrix"""
    matrix = _ensure_matrix(x)
    num_rows, num_cols = matrix.shape
    if num_rows != num_cols:
        raise ValueError(f"Array is not square. Shape {matrix.shape}")

    if n is not None:
        if matrix.shape[0] != n:
            raise ValueError(f"Required matrix dimension {n}. Got {matrix.shape[0]}.")

    return matrix


# ------------------------------------------------------------------------------
# Batch arrays
# ------------------------------------------------------------------------------

def _ensure_batch_array(x: ArrayLike, value_shape: Optional[Tuple[int, ...]] = None) -> Array:
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
    x = np.asarray(x)

    # Convert single values to batch with batch size one
    if value_shape is not None:
        # If input is exactly single-value shaped, expand to batch of size 1
        if x.ndim == len(value_shape):
            x = x[np.newaxis, ...]
    else:
        if x.ndim == 0:
            x = x.reshape((1,))  # () -> (1,)
        elif x.ndim == 1:
            x = x[np.newaxis, :] # (d,) -> (1,d)

    # Optionally validate value shape
    if value_shape is not None:
        x_value_shape = x.shape[1:]
        if x_value_shape != tuple(value_shape):
            raise ValueError(
                f"Batch array with value shape {x_value_shape} does not match required value shape {tuple(value_shape)}."
            )

    return x


def _ensure_batch_vector(x: ArrayLike, length: int = None):
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
      - Input shape (B, d, 1) -> raises error

    Returns:
        Array of shape (B, d).

    Raises:
        ValueError: If the input cannot be interpreted as a batch of vectors.
    """
    x = np.asarray(x)

    # If single vector value, standardize to (1,d)
    if x.ndim < 2:
        x = _ensure_vector(x, as_column=False)
        return _ensure_batch_array(x, value_shape=x.shape)

    # Batch vector must be two dimensional
    if x.ndim != 2:
        raise ValueError(
            f"_ensure_batch_vector: Array of shape {x.shape} is not a batch vector. Require shape (n_batch, d)."
        )

    # Validate vector length
    if length is not None:
        if x.shape[1] != length:
            raise ValueError(f"_ensure_batch_vector: Required vector length {length}. Got {x.shape[1]}.")

    return x


def _ensure_batch_matrix(x: ArrayLike, num_rows: int = None, num_cols: int = None,
                         as_row_matrix: bool = True):
    """Ensure `x` is a batch of matrices and return shape (B, n, m).

    This function returns an array of shape (B, n, m) encoding a batch matrix,
    where B is the batch size and (n, m) is the shape of each matrix in the
    batch. Three-dimensional arrays are returned unchanged. Lower dimensional
    arrays are converted to matrices via `_ensure_matrix`, which is then reshaped
    to (1,n,m) (representing a singleton batch). Higher dimensional arrays raise
    an error.

    Examples:
      - Input shape (n,) -> returned shape (1, 1, n)
      - Input shape (n, m) -> returned shape (1, n, m)
      - Input shape (B, n, m) -> returned shape (B, n, m)
      - Input shape (B, n, m, 1) -> raises error

    Returns:
        Array of shape (B, n, m).

    Raises:
        ValueError: If the input cannot be interpreted as a batch of matrices.
    """
    x = np.asarray(x)

    # If single matrix value, standardize to (1,n,m)
    if x.ndim < 3:
        x = _ensure_matrix(x, as_row_matrix=as_row_matrix)
        x = _ensure_batch_array(x, value_shape=x.shape)

    # Batch matrix must be three dimensional
    if x.ndim != 3:
        raise ValueError(
            f"Array of shape {x.shape} is not a batch matrix. Require shape (n_batch, n_row, n_col)."
        )

    if num_rows is not None:
        if x.shape[1] != num_rows:
            raise ValueError(f"_ensure_batch_matrix: Required {num_rows} rows. Got {x.shape[1]}.")

    if num_cols is not None:
        if x.shape[2] != num_cols:
            raise ValueError(f"_ensure_batch_matrix: Required {num_cols} rows. Got {x.shape[2]}.")

    return x


# ------------------------------------------------------------------------------
# Other utility functions
# ------------------------------------------------------------------------------

def add_diag_jitter(matrix: ArrayLike, jitter: float|ArrayLike = 1e-6, *, copy: bool = True) -> Array:
    """
    Return a new matrix = matrix + jitter * I.

    Args:
      matrix: 2D square array-like
      jitter: scalar or array-like of length n (interpreted elementwise)
      copy: if True (default) operate on a copy; if False, modify input in-place and return it.

    Returns:
      New Array (or the same array if copy=False) with jitter added to diagonal.
    """
    mat = _ensure_square_matrix(matrix)
    n = mat.shape[0]

    # ensure jitter is scalar or 1D length n
    try:
        jitter = _ensure_scalar(jitter)
        jitter = np.full((n,), jitter, dtype=mat.dtype)
    except ValueError:
        jitter = np.asarray(jitter)
        if jitter.ndim == 0:
            jitter = np.full((n,), jitter.item(), dtype=mat.dtype)
        elif jitter.shape != (n,):
            raise ValueError(f"add_diag_jitter: jitter must be scalar or shape ({n},); got {jarr.shape}")

    if copy:
        out = mat.copy()
    else:
        out = mat

    # Add jitter to diagonal
    diag_idcs = np.diag_indices(n)
    out[diag_idcs] = out[diag_idcs] + jitter
    return out
