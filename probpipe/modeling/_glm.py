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

    * ``data = Record(X=X_covariates, y=y_observed)`` — both fields
      explicit; the canonical form. ``X`` is the covariate matrix only;
      *do not* include a constant column for the intercept.
    * ``data = y_observed`` (a bare response array) when ``X`` was
      supplied at construction time — the construction-time ``X`` is
      used.

    Joint bootstrapping of covariates and response uses the Record
    form::

        Xy = Record(X=X_covariates, y=y_observed)
        bootstrap = BootstrapReplicateDistribution(EmpiricalDistribution(Xy))
        bagged = condition_on(model, bootstrap, n_broadcast_samples=16)

    Parameters
    ----------
    family : tfp.glm.ExponentialFamily
        TFP GLM family (e.g., ``Poisson()``, ``Bernoulli()``,
        ``NegativeBinomial()``).
    x : array-like or None
        Default covariate matrix of shape ``(n, p)``. If ``None``, must
        be provided per-call via ``data=Record(X=..., y=...)``. Should
        contain **only the covariates** — the intercept is fit
        separately when ``fit_intercept=True``.
    fit_intercept : bool, default True
        When True, the likelihood expects ``params`` to flatten to
        ``(intercept, *slopes)`` of length ``p + 1`` and computes
        ``eta = intercept + X @ slopes``. When False, ``params``
        flattens to length ``p`` and the likelihood computes
        ``eta = X @ params`` directly — useful when the user wants to
        carry the intercept as a constant column in ``X`` themselves
        (the classical "model matrix" convention).
    seed : int
        Random seed for data generation.
    """

    def __init__(
        self,
        family: tfp_glm.ExponentialFamily,
        x: ArrayLike | None = None,
        *,
        fit_intercept: bool = True,
        seed: int = 0,
    ):
        self.family = family
        if x is not None:
            self._x = jnp.atleast_2d(jnp.asarray(x))
            if self._x.ndim == 2 and self._x.shape[0] == 1 and self._x.shape[1] > 1:
                self._x = self._x.T
        else:
            self._x = None
        self._fit_intercept = bool(fit_intercept)
        self._key = jax.random.PRNGKey(seed)

    def _linear_predictor(self, X: Array, params: ArrayLike | Record) -> Array:
        """``eta = intercept + X @ slopes`` (or ``X @ params`` if fit_intercept=False)."""
        beta = _coerce_array(params)
        if self._fit_intercept:
            return beta[0] + X @ beta[1:]
        return X @ beta

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
            X_raw = jnp.asarray(data["X"])
            # Single-covariate convenience: a 1-D X array is the natural
            # form when there is only one covariate (no need for a
            # gratuitous trailing axis). The linear-predictor math
            # downstream wants 2-D, so reshape here once.
            if X_raw.ndim == 1:
                X_raw = X_raw[:, None]
            return X_raw, _coerce_array(data["y"])
        if self._x is not None:
            return self._x, _coerce_array(data)
        raise ValueError(
            "GLMLikelihood data must be either a Record(X=..., y=...) "
            "or a response array paired with an X passed at "
            "construction time."
        )

    def log_likelihood(self, params: ArrayLike | Record, data: ArrayLike | Record) -> float:
        """Log-likelihood: sum of per-observation log-probs.

        *params* and *data* can be raw arrays or ``Record`` objects.
        When *data* is ``Record(X=..., y=...)``, both the covariate
        matrix and response are extracted. The linear predictor
        ``eta = intercept + X @ slopes`` is computed via
        :meth:`_linear_predictor` so the ``fit_intercept`` convention is
        respected uniformly across the public methods.
        """
        X, y = self._extract_X_y(data)
        eta = self._linear_predictor(X, params)
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
            and ``y_i`` scalar.

        Raises
        ------
        TypeError
            If ``datum`` is not a ``Record(X=..., y=...)``.
        """
        if not (isinstance(datum, Record) and "X" in datum and "y" in datum):
            raise TypeError(
                "GLMLikelihood.per_datum_log_likelihood requires "
                "datum=Record(X=x_i, y=y_i)."
            )
        # `atleast_1d` accommodates the single-covariate case where the
        # per-observation X leaf is naturally scalar — the matmul below
        # still needs a 1-D vector.
        x_i = jnp.atleast_1d(jnp.asarray(datum["X"]))
        y_i = jnp.asarray(datum["y"])
        beta = _coerce_array(params)
        if self._fit_intercept:
            eta = beta[0] + x_i @ beta[1:]
        else:
            eta = x_i @ beta
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
        # Linear predictor — same convention as log_likelihood:
        #   fit_intercept=True  → eta = intercept + X @ slopes
        #   fit_intercept=False → eta = X @ params (classical model-matrix form)
        # Batched params come in as ``(*batch, p_total)``; broadcast against
        # ``X[:n_samples]`` of shape ``(n_samples, p)``.
        beta = _coerce_array(params)
        Xn = X[:n_samples]
        if self._fit_intercept:
            slopes = beta[..., 1:]
            intercept = beta[..., 0:1]  # keep last axis for broadcasting
            eta = intercept + slopes @ Xn.T
        else:
            eta = beta @ Xn.T
        dist = self.family.as_distribution(eta)
        return dist.sample(seed=key)
