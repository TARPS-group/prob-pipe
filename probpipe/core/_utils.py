from numpy.typing import NDArray

import numpy as np

    
def _as_2d(x: NDArray) -> NDArray:
    """
    Convert input to a 2-D float array.

    A 1-D input is promoted to shape (1, n); higher-dimensional inputs are left
    unchanged except for dtype casting to float.

    Parameters
    ----------
    x : array-like
        Input data of shape (n,), (n, d), or higher.

    Returns
    -------
    NDArray
        Array with dtype float. If the original was 1-D (n,), returns shape (1, n);
        otherwise returns the float-cast array as-is.
    """
    x = np.asarray(x, dtype=float)
    return x.reshape(1, -1) if x.ndim == 1 else x

def _symmetrize_spd(C: NDArray, jitter: float = 1e-9) -> NDArray:
    """
    Symmetrize a matrix and nudge it toward positive definiteness.

    The matrix is first symmetrized via (C + C.T)/2. If the smallest eigenvalue
    is near or below zero (numerical PSD violation), a diagonal jitter is added
    to ensure a safely positive spectrum.

    Parameters
    ----------
    C : array-like, shape (d, d)
        Input square matrix intended to be SPD/PSD.
    jitter : float, default 1e-9
        Baseline diagonal regularization used when eigenvalues are too small.

    Returns
    -------
    NDArray
        A symmetric matrix with eigenvalues pushed above ~1e-12 (plus jitter)
        to avoid numerical issues in Cholesky and similar factorizations.
    """
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    eigmin = np.linalg.eigvalsh(C).min()
    if eigmin < 1e-12:
        C = C + (max(jitter, 1e-12 - eigmin) + 1e-12) * np.eye(C.shape[0])
    return C

def _clip_unit_interval(x: NDArray[np.floating], eps: float = 0.0) -> NDArray[np.floating]:
    """
    Clip values to the unit interval, with optional open-interval padding.

    Parameters
    ----------
    x : array-like of float
        Values to clip.
    eps : float, default 0.0
        If 0, clip to the closed interval [0, 1]. If >0, clip to the open
        interval (eps, 1 - eps) using `np.nextafter` to avoid returning exact
        endpoints, which is useful before applying `log`, `logit`, etc.

    Returns
    -------
    NDArray[np.floating]
        Clipped array.
    """
    if eps <= 0.0:
        return np.clip(x, 0.0, 1.0)
    lo = np.nextafter(0.0 + eps, 1.0)
    hi = np.nextafter(1.0 - eps, 0.0)
    return np.clip(x, lo, hi)


def _to_1d_vector(values: NDArray) -> NDArray[np.floating]:
    """
    Normalize input to a 1-D float vector of shape (n,).

    Rules
    -----
    - scalar -> (1,)
    - (n,)   -> (n,)
    - (n,1)  -> (n,)

    Parameters
    ----------
    values : array-like
        Scalar, 1-D array, or 2-D column vector.

    Returns
    -------
    NDArray[np.floating]
        1-D float array.

    Raises
    ------
    ValueError
        If the input is not a scalar, (n,), or (n,1).
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError("values must be scalar, (n,), or (n,1).")


