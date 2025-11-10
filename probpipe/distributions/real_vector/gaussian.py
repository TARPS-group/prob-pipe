# gaussian.py
from __future__ import annotations

import math
import numpy as np
from typing import Any

from ...custom_types import Array, ArrayLike, Float, PRNG
from ..distribution import Distribution
from .real_vector import RealVectorDistribution
from ...linalg.linop import LinOp, DenseLinOp, CholeskyLinOp, CholeskyFactor
from ...linalg.operations import _as_linear_operator, LinOpLike, cholesky, logdet

from ...array_backend.utils import (
    _ensure_real_scalar,
    _ensure_vector,
    _ensure_matrix,
    _ensure_square_matrix,
    _ensure_batch_array
)


class Gaussian(RealVectorDistribution[Float]):

    def __init__(self, mean: ArrayLike, cov: LinOpLike,
                 *, rng: PRNG | None = None):
        mean = _ensure_vector(mean)
        self._dim = len(mean)
        cov = _as_linear_operator(cov)
        cov._check_square()
        if cov.shape[0] != self._dim:
            raise ValueError(f"Dimension mismatch between mean {mean.shape} and covariance {cov.shape}.")

        # Covariance is represented via the lower root L such that C = L @ L.T.
        self._cov_from_chol = cov.to_cholesky_representation(lower=True)

        self._mean = mean
        self._cov = cov
        self._rng = rng or np.random.default_rng()

    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def mean(self) -> Array[Float]:
        return self._mean

    @property
    def cov(self) -> LinOp:
        return self._cov
    
    @property 
    def lower_chol(self) -> CholeskyFactor:
        return self._cov_from_chol.root
    
    def sample(self, n_samples: int = 1) -> Array[Float]:
        """
        Draw (n, d) samples.
        """
        L = self.lower_chol
        Z = self._rng.normal(size=(self.dim, n_samples))

        samp = self.mean + L.matmat(Z).T
        return samp
    

    def log_density(self, values: ArrayLike) -> Array[Float]:
        x = _ensure_matrix(values, as_row_matrix=True, num_cols=self.dim)
        L = self.lower_chol

        x_centered = x - self.mean # (n,d)
        logdet_term = self.dim * np.log(2 * math.pi) + logdet(self._cov_from_chol)
        L_inv_x_centered = L.solve(x_centered.T)
        quadratic_term = np.sum(L_inv_x_centered ** 2, axis=0)
        log_dens = -0.5 * (logdet_term + quadratic_term)
        
        return log_dens


    def density(self, values: ArrayLike) -> Array[Float]:
        return np.exp(self.log_density(values))
    
    
    def cdf(self, x: ArrayLike) -> Array[Float]:
        """
        Joint CDF F(x) = P[X1 ≤ x1, ..., Xd ≤ xd].
        Accepts x with shape (..., d) and returns shape (...,).
        """
        raise NotImplementedError


    def inv_cdf(self, u: Array) -> Array[Float]:
        """
        Inverse CDF (Rosenblatt inverse) mapping u ∈ (0,1)^d to x ∈ R^d.
        Accepts u with shape (..., d) and returns x with shape (..., d).
        """
        raise NotImplementedError
    

    @classmethod
    def from_distribution(cls, convert_from: Distribution, num_samples: int = 1024, *, 
                          conversion_by_KDE: bool = False, **fit_kwargs: Any) -> Gaussian:
        """
        Fit mean and covariance from samples drawn from another distribution-like object.
        """
        if not isinstance(convert_from, Distribution):
            raise ValueError("`convert_from` must be a Distribution object.")

        samp = convert_from.sample(num_samples)
        samp = _ensure_matrix(samp, num_rows=num_samples)

        mean = samp.mean(axis=0)
        cov = np.cov(samp, rowvar=False, ddof=1)
 
        return cls(mean=mean, cov=cov)