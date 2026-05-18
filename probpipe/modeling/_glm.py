"""GLM likelihood wrapper for TFP exponential family models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.glm as tfp_glm

from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike, PRNGKey
from .._utils import _auto_key

__all__ = ["GLMLikelihood"]


def _coerce_array(x: ArrayLike | Record) -> jnp.ndarray:
    """Extract a JAX array from a Record, RecordArray, or raw array-like.

    Single-field Record/RecordArray: extract the field.
    Multi-field: stack fields into a vector (preserving leading batch dims).
    """
    from .._utils import prod
    from ..core._record_array import RecordArray
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, (Record, RecordArray)):
        fields = x.fields
        if len(fields) == 1:
            return jnp.asarray(x[fields[0]])
        arrays = [jnp.asarray(x[f]) for f in fields]
        return jnp.stack(arrays, axis=-1)
    return jnp.asarray(x)


class GLMLikelihood:
    """Wraps a TFP GLM family + design matrix into a Likelihood and GenerativeLikelihood.

    Two accepted data forms:

    * ``data = Record(X=X_design, y=y_observed)`` — both fields
      explicit; the canonical form.
    * ``data = y_observed`` (a bare response array) when ``X`` was
      supplied at construction time — the construction-time ``X`` is
      used.

    Stacked ``(X, y)`` arrays (a single matrix whose last column is
    the response) are **intentionally not accepted**: ProbPipe uses
    named Records precisely to avoid axis-position ambiguity. Pass a
    ``Record(X=..., y=...)`` explicitly.

    Joint bootstrapping of covariates and response uses the Record
    form::

        Xy = Record(X=X_design, y=y_observed)
        bootstrap = BootstrapReplicateDistribution(EmpiricalDistribution(Xy))
        bagged = condition_on(model, bootstrap, n_broadcast_samples=16)

    Parameters
    ----------
    family : tfp.glm.ExponentialFamily
        TFP GLM family (e.g., ``Poisson()``, ``Bernoulli()``,
        ``NegativeBinomial()``).
    x : array-like or None
        Default design matrix of shape ``(n, p)``.  If ``None``, must
        be provided per-call via ``data=Record(X=..., y=...)``.
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
            self._x = jnp.atleast_2d(jnp.asarray(x))
            if self._x.ndim == 2 and self._x.shape[0] == 1 and self._x.shape[1] > 1:
                self._x = self._x.T
        else:
            self._x = None
        self._key = jax.random.PRNGKey(seed)

    @property
    def data_template(self) -> RecordTemplate:
        """Named structure of GLM data: ``X`` (design matrix) and ``y`` (response)."""
        return RecordTemplate(X=(0, 0), y=(0,))

    def _extract_X_y(self, data):
        """Extract design matrix and response from data.

        Two accepted forms:

        1. ``data = Record(X=..., y=...)`` — both fields explicit; the
           canonical form, free of axis-position ambiguity.
        2. ``data`` is a response array, and ``X`` was supplied at
           construction time — use the construction-time ``X``.

        A bare array without a construction-time ``X`` is rejected —
        ProbPipe uses named Records precisely to avoid "is column N
        the response or a covariate?" guessing.
        """
        if isinstance(data, Record) and "X" in data and "y" in data:
            return jnp.asarray(data["X"]), _coerce_array(data["y"])
        if self._x is not None:
            return self._x, _coerce_array(data)
        raise ValueError(
            "GLMLikelihood data must be either a Record(X=..., y=...) "
            "or a response array paired with an X passed at "
            "construction time. Stacked (X, y) arrays are intentionally "
            "rejected; use Record(X=..., y=...) instead."
        )

    def log_likelihood(self, params: ArrayLike | Record, data: ArrayLike | Record) -> float:
        """Log-likelihood: sum of per-observation log-probs.

        *params* and *data* can be raw arrays or ``Record`` objects.
        When *data* is ``Record(X=..., y=...)``, both the design matrix
        and response are extracted.
        """
        X, y = self._extract_X_y(data)
        eta = X @ _coerce_array(params)
        return jnp.sum(self.family.log_prob(y, eta))

    def per_datum_log_likelihood(
        self, params: ArrayLike | Record, datum: Record,
    ) -> Array:
        """Log-density of a single observation given parameters.

        Satisfies :class:`~probpipe.ConditionallyIndependentLikelihood`.
        Evaluates ``family.log_prob(y_i, x_i @ params)`` directly on a
        scalar response, bypassing the length-1-batch reshape that the
        default fallback (``log_likelihood(params, datum[None, ...])``)
        would add. The saved per-call overhead matters when this method
        is called inside a stochastic-gradient inner loop.

        Parameters
        ----------
        params : Array or Record
            Coefficient vector of shape ``(p,)``.
        datum : Record
            ``Record(X=x_i, y=y_i)`` with ``x_i`` of shape ``(p,)``
            and ``y_i`` scalar. Stacked ``(x, y)`` arrays are
            intentionally not supported — use the named-field Record
            form.

        Raises
        ------
        TypeError
            If ``datum`` is not a ``Record(X=..., y=...)``.
        """
        if not (isinstance(datum, Record) and "X" in datum and "y" in datum):
            raise TypeError(
                "GLMLikelihood.per_datum_log_likelihood requires "
                "datum=Record(X=x_i, y=y_i). Stacked arrays are "
                "intentionally not supported; use the named-field "
                "Record form to avoid axis-position ambiguity."
            )
        x_i = jnp.asarray(datum["X"])
        y_i = jnp.asarray(datum["y"])
        eta = x_i @ _coerce_array(params)
        return self.family.log_prob(y_i, eta)

    def generate_data(
        self,
        params: ArrayLike | Record,
        n_samples: int,
        *,
        key: PRNGKey | None = None,
    ) -> Array:
        """Generate synthetic data from the GLM.

        Uses the stored design matrix (first ``n_samples`` rows).

        Parameters
        ----------
        params : Array or Record
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
