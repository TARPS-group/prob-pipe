from numpy.typing import NDArray

import numpy as np

    
def _as_2d(x: NDArray) -> NDArray:
    """Converts input to a 2-D float array.

    A 1-D array is reshaped to (1, n). Higher-dimensional arrays are kept
    unchanged except for dtype casting to float.

    Args:
        x (NDArray): Input array of shape (n,), (n, d), or higher.

    Returns:
        NDArray: Float array. If input was 1-D, returns shape (1, n).
    """
    x = np.asarray(x, dtype=float)
    return x.reshape(1, -1) if x.ndim == 1 else x

def _symmetrize_spd(C: NDArray, jitter: float = 1e-9) -> NDArray:
    """Symmetrizes a matrix and ensures positive semi-definiteness.

    The matrix is symmetrized as (C + C.T)/2, and if the smallest eigenvalue
    is near or below zero, a small diagonal jitter is added to maintain
    numerical stability (useful for Cholesky and covariance operations).

    Args:
        C (NDArray): Square matrix of shape (d, d).
        jitter (float, optional): Minimum diagonal regularization added when
            eigenvalues are too small. Defaults to 1e-9.

    Returns:
        NDArray: Symmetric positive-definite matrix adjusted for stability.
    """
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    eigmin = np.linalg.eigvalsh(C).min()
    if eigmin < 1e-12:
        C = C + (max(jitter, 1e-12 - eigmin) + 1e-12) * np.eye(C.shape[0])
    return C

def _clip_unit_interval(x: NDArray[np.floating], eps: float = 0.0) -> NDArray[np.floating]:
    """Clips values to the [0, 1] interval, optionally padding to an open range.

    Args:
        x (NDArray[np.floating]): Values to clip.
        eps (float, optional): If 0, clips to [0, 1]. If >0, clips to
            (eps, 1 − eps) using `np.nextafter` to avoid exact endpoints.
            Defaults to 0.0.

    Returns:
        NDArray[np.floating]: Array with clipped values.
    """
    if eps <= 0.0:
        return np.clip(x, 0.0, 1.0)
    lo = np.nextafter(0.0 + eps, 1.0)
    hi = np.nextafter(1.0 - eps, 0.0)
    return np.clip(x, lo, hi)


def _to_1d_vector(values: NDArray) -> NDArray[np.floating]:
    """Normalizes input to a 1-D float vector of shape (n,).

    Accepts scalars, 1-D arrays, or 2-D column vectors and converts them
    to a standardized 1-D float array.

    Args:
        values (NDArray): Input values as scalar, (n,), or (n, 1).

    Returns:
        NDArray[np.floating]: Flattened 1-D array.

    Raises:
        ValueError: If the input is not scalar, (n,), or (n, 1).
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError("values must be scalar, (n,), or (n,1).")


