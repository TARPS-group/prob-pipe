"""GLM likelihood wrapper for TFP exponential family models."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from .._utils import _auto_key

__all__ = ["GLMLikelihood"]


class GLMLikelihood:
    """Wraps a TFP GLM family + design matrix into a Likelihood and GenerativeLikelihood.

    Given a TFP GLM family (e.g., ``tfp.glm.Poisson()``) and a design
    matrix ``X``, this class computes the linear predictor
    ``X @ params`` and delegates to the family for log-probability
    and data generation.

    Satisfies both the ``Likelihood[Array, Array]`` and
    ``GenerativeLikelihood[Array, Array]`` protocols.

    Parameters
    ----------
    family : tfp.glm.ExponentialFamily
        TFP GLM family (e.g., ``Poisson()``, ``Bernoulli()``,
        ``NegativeBinomial()``).
    x : array-like
        Design matrix of shape ``(n, p)`` or covariate vector of shape
        ``(n,)``.  If 1-D, an intercept column is **not** added
        automatically — include it in ``x`` or in the parameter vector
        as needed.
    seed : int
        Random seed for data generation.

    Examples
    --------
    >>> import tensorflow_probability.substrates.jax.glm as tfp_glm
    >>> lik = GLMLikelihood(tfp_glm.Poisson(), x=X_design)
    >>> lik.log_likelihood(params, y_obs)  # scalar log-prob
    >>> lik.generate_data(params, n_samples=100)  # Poisson draws
    """

    def __init__(
        self,
        family: Any,
        x: Array,
        *,
        seed: int = 0,
    ):
        self.family = family
        self._x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
        if self._x.ndim == 2 and self._x.shape[0] == 1 and self._x.shape[1] > 1:
            # atleast_2d turned (n,) into (1, n) — transpose to (n, 1)
            self._x = self._x.T
        self._key = jax.random.PRNGKey(seed)

    def log_likelihood(self, params: Array, data: Array) -> float:
        """Log-likelihood: sum of per-observation log-probs."""
        eta = self._x @ params
        return jnp.sum(self.family.log_prob(data, eta))

    def generate_data(self, params: Array, n_samples: int) -> Array:
        """Generate synthetic data from the GLM."""
        self._key, subkey = jax.random.split(self._key)
        eta = self._x[:n_samples] @ params
        dist = self.family.as_distribution(eta)
        return dist.sample(seed=subkey)
