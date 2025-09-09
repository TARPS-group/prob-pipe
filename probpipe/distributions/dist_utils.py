from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
from numpy.typing import NDArray

import numpy as np


def _as_2d(x: NDArray) -> NDArray[np.floating]:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

def _symmetrize_spd(cov: NDArray[np.floating], jitter: float = 1e-6) -> NDArray[np.floating]:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    # minimal jitter for numerical stability
    eigmin = np.linalg.eigvalsh(cov).min()
    if eigmin <= 0:
        cov = cov + (jitter - eigmin + 1e-12) * np.eye(cov.shape[0])
    else:
        cov = cov + jitter * np.eye(cov.shape[0])
    return cov