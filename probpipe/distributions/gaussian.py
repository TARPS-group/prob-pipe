# gaussian.py
from __future__ import annotations

import math
import numpy as np

from ..custom_types import Array, ArrayLike
from ..core.distributions import Distribution
from ..core.multivariate import Multivariate
from ..linalg.linop import LinOp, DenseLinOp

from ..linalg.utils import (
    _ensure_real_scalar,
    _ensure_vector,
    _ensure_matrix,
    _ensure_square_matrix
)


# Notes/Questions/TODOs:
# - Currently Multivariate.cov defined to return type Array; this implementation 
#   returns LinOp.
# - We probably want `n_samples` to default to 1.
# - Shorten `dimension` property to `dim`?
# - Should we have log_density, etc. return (n,)?



def GaussianDistribution(Multivariate[np.floating]):

    def __init__(self, mean: Array[np.floating], cov: LinOp | Array[np.floating],
                 *, rng: np.random.Generator | None = None):
        mean = _ensure_vector(mean)
        self._dim = len(mean)

        if not isinstance(cov, LinOp):
            cov = DenseLinOp(cov)
        cov._check_square()
        if cov.shape[0] != self._dim:
            raise ValueError(f"Dimension mismatch between mean {mean.shape} and covariance {cov.shape}.")

        self._mean = mean
        self._cov = cov
        self._rng = rng or np.random.default_rng()


    @property
    def dimension(self) -> int:
        """
        Number of coordinates d. Default infers from mean(). Subclasses may override.
        """
        m = self.mean()
        if m.ndim != 1:
            raise ValueError("mean() must return a 1D array of shape (d,).")
        return int(m.shape[0])
    

    def sample(self, n_samples: int) -> Array[np.floating]:
        """
        Draw (n, d) samples.
        """
        L = self.cov.cholesky(lower=True)
        Z = self._rng.normal(size=(self.dimension, n_samples))

        samp = self.mean + L.matmat(Z).T
        return samp
    
    def log_density(self, values: Array) -> Array[np.floating]:
        """
        Return (n,1) column of log-pdf values for rows in `values`.
        Accepts (d,), (n,d).
        """
        x = _ensure_matrix(values, as_row_matrix=True, num_cols=self.dimension)

        x_centered = x - self.mean # (n,d)
        logdet_term = (2 * math.pi * self.cov).logdet() # TODO: update
        L_inv_x_centered = self.cov.cholesky(lower=True).solve(x_centered.T)
        quadratic_term = np.sum(L_inv_x_centered ** 2, axis=0)
        log_dens = -0.5 * (logdet_term + quadratic_term)
        
        return log_dens.reshape(-1, 1) # (n,1)
