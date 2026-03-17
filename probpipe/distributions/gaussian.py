"""
Multivariate Gaussian distribution backed by TFP.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .distribution import (
    TFPDistribution,
    Distribution,
    EmpiricalDistribution,
    Provenance,
)
from ..custom_types import Array, ArrayLike, PRNGKey


class Gaussian(TFPDistribution):
    """
    Multivariate Gaussian (normal) distribution.

    Parameters
    ----------
    loc : array-like, shape ``(d,)``
        Mean vector.
    scale_tril : array-like, shape ``(d, d)``, optional
        Lower-triangular Cholesky factor of the covariance.  Exactly one of
        *scale_tril* or *cov* must be provided.
    cov : array-like, shape ``(d, d)``, optional
        Covariance matrix (Cholesky-decomposed internally).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        cov: ArrayLike | None = None,
        name: str | None = None,
    ):
        loc = jnp.asarray(loc, dtype=jnp.float32)
        if loc.ndim == 0:
            loc = loc.reshape(1)

        if scale_tril is not None and cov is not None:
            raise ValueError("Provide exactly one of scale_tril or cov, not both.")

        if scale_tril is not None:
            scale_tril = jnp.asarray(scale_tril, dtype=jnp.float32)
        elif cov is not None:
            cov = jnp.asarray(cov, dtype=jnp.float32)
            if cov.shape != (loc.shape[0], loc.shape[0]):
                raise ValueError(
                    f"cov shape {cov.shape} does not match loc length {loc.shape[0]}."
                )
            scale_tril = jnp.linalg.cholesky(cov)
        else:
            raise ValueError("One of scale_tril or cov must be provided.")

        self._loc = loc
        self._scale_tril = scale_tril
        self._tfp_dist = tfd.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril
        )
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale_tril(self) -> Array:
        return self._scale_tril

    @property
    def cov(self) -> Array:
        """Full covariance matrix (computed from Cholesky factor)."""
        return self._scale_tril @ self._scale_tril.T

    @property
    def dim(self) -> int:
        return self._loc.shape[0]

    # -- conversion ---------------------------------------------------------

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        num_samples: int = 1024,
        name: str | None = None,
        **kwargs: Any,
    ) -> Gaussian:
        """Fit a Gaussian by moment-matching samples from *other*."""
        if key is None:
            key = jax.random.PRNGKey(0)

        if isinstance(other, EmpiricalDistribution):
            loc = other.mean()
            cov_mat = other.cov()
        else:
            samples = other.sample(key, sample_shape=(num_samples,))
            loc = jnp.mean(samples, axis=0)
            diff = samples - loc
            cov_mat = jnp.einsum("ni,nj->ij", diff, diff) / samples.shape[0]

        # Ensure PSD via jitter
        cov_mat = 0.5 * (cov_mat + cov_mat.T)
        cov_mat = cov_mat + 1e-6 * jnp.eye(cov_mat.shape[0])

        g = cls(loc=loc, cov=cov_mat, name=name or other.name)
        g._source = Provenance("from_distribution", parents=(other,))
        return g
