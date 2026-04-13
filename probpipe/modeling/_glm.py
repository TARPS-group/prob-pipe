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
        arrays = [jnp.asarray(x[f]) for f in fields]
        return jnp.stack(arrays, axis=-1)
    return jnp.asarray(x)


class GLMLikelihood:
    """Wraps a TFP GLM family + design matrix into a Likelihood and GenerativeLikelihood.

    The design matrix ``X`` can be provided at construction (default)
    or per-call via ``data=Values(X=..., y=...)``.  When ``data``
    is a ``Values`` with ``X`` and ``y`` fields, both are extracted;
    otherwise ``data`` is treated as the response and the stored ``X``
    is used.

    This enables joint bootstrapping of covariates and response::

        Xy = Values(X=X_design, y=y_observed)
        bootstrap = BootstrapReplicateDistribution(EmpiricalDistribution(Xy))
        bagged = condition_on(model, bootstrap, n_broadcast_samples=16)

    Parameters
    ----------
    family : tfp.glm.ExponentialFamily
        TFP GLM family (e.g., ``Poisson()``, ``Bernoulli()``,
        ``NegativeBinomial()``).
    x : array-like or None
        Default design matrix of shape ``(n, p)``.  If ``None``, must
        be provided per-call via ``data=Values(X=..., y=...)``.
    seed : int
        Random seed for data generation.
    """

    def __init__(
        self,
        family: tfp_glm.ExponentialFamily,
        x: ArrayLike | None = None,
        *,
        seed: int = 0,
    ):
        self.family = family
        if x is not None:
            self._x = jnp.atleast_2d(jnp.asarray(x, dtype=jnp.float32))
            if self._x.ndim == 2 and self._x.shape[0] == 1 and self._x.shape[1] > 1:
                self._x = self._x.T
        else:
            self._x = None
        self._key = jax.random.PRNGKey(seed)

    @property
    def data_template(self) -> Values:
        """Named structure of GLM data: ``X`` (design matrix) and ``y`` (response)."""
        return Values(X=jnp.zeros((0, 0)), y=jnp.zeros(0))

    def _extract_X_y(self, data):
        """Extract design matrix and response from data.

        Resolution order:

        1. ``data = Values(X=..., y=...)`` → extract both fields.
        2. ``data`` is an array with more columns than ``p`` (the number
           of coefficients, inferred from the stored ``X``) → last column
           is the response, preceding columns are the design matrix.
        3. ``data`` is a response array → use the stored ``X``.
        """
        if isinstance(data, Values) and "X" in data and "y" in data:
            return jnp.asarray(data["X"]), _coerce_array(data["y"])
        data_arr = _coerce_array(data)
        if self._x is not None:
            p = self._x.shape[1]
            if data_arr.ndim == 2 and data_arr.shape[1] > p:
                # Stacked (X, y) array — split columns
                return data_arr[:, :p], data_arr[:, p]
            return self._x, data_arr
        # No stored X: data must be a stacked (X, y) array
        if data_arr.ndim == 2 and data_arr.shape[1] > 1:
            return data_arr[:, :-1], data_arr[:, -1]
        raise ValueError(
            "No design matrix: pass X at construction or "
            "via data=Values(X=..., y=...)"
        )

    def log_likelihood(self, params: ArrayLike | Values, data: ArrayLike | Values) -> float:
        """Log-likelihood: sum of per-observation log-probs.

        *params* and *data* can be raw arrays or ``Values`` objects.
        When *data* is ``Values(X=..., y=...)``, both the design matrix
        and response are extracted.
        """
        X, y = self._extract_X_y(data)
        eta = X @ _coerce_array(params)
        return jnp.sum(self.family.log_prob(y, eta))

    def generate_data(
        self,
        params: ArrayLike | Values,
        n_samples: int,
        *,
        key: PRNGKey | None = None,
    ) -> Array:
        """Generate synthetic data from the GLM.

        Uses the stored design matrix (first ``n_samples`` rows).

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
        X = self._x
        if X is None:
            raise ValueError(
                "generate_data requires a stored design matrix (pass x at construction)"
            )
        eta = _coerce_array(params) @ X[:n_samples].T
        dist = self.family.as_distribution(eta)
        return dist.sample(seed=key)
