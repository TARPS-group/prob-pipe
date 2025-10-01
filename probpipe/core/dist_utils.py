from numpy.typing import NDArray

import numpy as np


    
def _as_2d(x: NDArray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(1, -1) if x.ndim == 1 else x

def _symmetrize_spd(C: np.ndarray, jitter: float = 1e-9) -> NDArray:
    C = np.asarray(C, dtype=float)
    C = 0.5 * (C + C.T)
    eigmin = np.linalg.eigvalsh(C).min()
    if eigmin < 1e-12:
        C = C + (max(jitter, 1e-12 - eigmin) + 1e-12) * np.eye(C.shape[0])
    return C

def _clip_unit_interval(x: NDArray[np.floating], eps: float = 0.0) -> NDArray[np.floating]:
    """
    Ensure values lie in [0,1] (or (eps, 1-eps) if eps>0).
    """
    if eps <= 0.0:
        return np.clip(x, 0.0, 1.0)
    lo = np.nextafter(0.0 + eps, 1.0)
    hi = np.nextafter(1.0 - eps, 0.0)
    return np.clip(x, lo, hi)


def _to_1d_vector(values: NDArray) -> NDArray[np.floating]:
    """
    Normalize input to a 1-D float vector (n,):
      - scalar -> (1,)
      - (n,)   -> (n,)
      - (n,1)  -> (n,)
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError("values must be scalar, (n,), or (n,1) for Beta (event dim = 1).")


