"""GLM likelihood wrapper for TFP exponential family models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.glm as tfp_glm

from ..core.values import Values
from ..custom_types import Array, ArrayLike, PRNGKey
from .._utils import _auto_key

__all__ = ["GLMLikelihood"]


def _coerce_array(x: ArrayLike | Values) -> jnp.ndarray:
    """Extract a JAX array from a Values field or raw array-like.

    Single-field Values: extract the field.
    Multi-field Values with scalar fields: stack into a vector
    (preserving any leading batch dimensions).
    """
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, Values):
        fields = x.fields()
        if len(fields) == 1:
            return x[fields[0]]
        # Stack scalar fields along a new trailing axis.
        # E.g., Values(a=array(100,), b=array(100,)) → array(100, 2)
        # E.g., Values(a=scalar, b=scalar) → array(2,)
        arrays = [jnp.asarray(x[f]) for f in fields]
        return jnp.stack(arrays, axis=-1)
    return jnp.asarray(x)


class GLMLikelihood:
    """Wraps a TFP GLM family + design matrix into a Likelihood and GenerativeLikelihood.

    The design matrix ``X`` is stored at construction.  Both
    ``log_likelihood`` and ``generate_data`` accept raw arrays or
    :class:`~probpipe.Values` for params and data; ``Values`` fields
    are extracted automatically.

    Parameters
    ----------
    family : tfp.glm.ExponentialFamily
        TFP GLM family (e.g., ``Poisson()``, ``Bernoulli()``,
        ``NegativeBinomial()``).
    x : array-like
        Design matrix of shape ``(n, p)`` or covariate vector of shape
        ``(n,)``.
    seed : int
        Random seed for data generation.
    """

    def __init__(
        self,
        family: tfp_glm.ExponentialFamily,
        x: ArrayLike,
        *,
        seed: int = 0,
    ):
        self.family = family
        self._x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
        if self._x.ndim == 2 and self._x.shape[0] == 1 and self._x.shape[1] > 1:
            self._x = self._x.T
        self._key = jax.random.PRNGKey(seed)

    def log_likelihood(self, params: ArrayLike | Values, data: ArrayLike | Values) -> float:
        """Log-likelihood: sum of per-observation log-probs.

        *params* and *data* can be raw arrays or ``Values`` objects.
        """
        eta = self._x @ _coerce_array(params)
        return jnp.sum(self.family.log_prob(_coerce_array(data), eta))

    def generate_data(
        self,
        params: ArrayLike | Values,
        n_samples: int,
        *,
        key: PRNGKey | None = None,
    ) -> Array:
        """Generate synthetic data from the GLM.

        Parameters
        ----------
        params : Array or Values
            Parameter vector of shape ``(p,)`` or a batch ``(*batch, p)``.
        n_samples : int
            Number of observations to generate (per batch element).
        key : PRNGKey, optional
            JAX PRNG key.
        """
        if key is None:
            self._key, key = jax.random.split(self._key)
        eta = _coerce_array(params) @ self._x[:n_samples].T
        dist = self.family.as_distribution(eta)
        return dist.sample(seed=key)
