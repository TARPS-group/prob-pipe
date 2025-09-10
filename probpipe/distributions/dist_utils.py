from typing import Generic, TypeVar, Callable, Any, Optional, Union, Tuple
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