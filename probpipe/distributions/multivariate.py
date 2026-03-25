"""
Multivariate distributions backed by TFP.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .distribution import (
    TFPDistribution,
    ArrayDistribution,
    EmpiricalDistribution,
    Provenance,
    Constraint,
    real,
    positive_definite,
    simplex,
    non_negative_integer,
    sphere,
)
from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = [
    "MultivariateNormal",
    "Dirichlet",
    "Multinomial",
    "Wishart",
    "VonMisesFisher",
]


# ---------------------------------------------------------------------------
# MultivariateNormal
# ---------------------------------------------------------------------------


class MultivariateNormal(TFPDistribution):
    """
    Multivariate normal (Gaussian) distribution.

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

    # -- support ------------------------------------------------------------

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

    @property
    def support(self) -> Constraint:
        return real

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> MultivariateNormal:
        """Fit a multivariate normal by moment-matching samples from *other*.

        Keyword Args
        -------------
        num_samples : int
            Samples drawn for moment-matching (default 1024).
        """
        num_samples = kwargs.pop("num_samples", 1024)

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
        g.with_source(Provenance("from_distribution", parents=(other,)))
        return g


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------


class Dirichlet(TFPDistribution):
    """
    Dirichlet distribution over the probability simplex.

    Parameters
    ----------
    concentration : array-like, shape ``(k,)``
        Positive concentration (alpha) parameters.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        concentration: ArrayLike,
        *,
        name: str | None = None,
    ):
        concentration = jnp.asarray(concentration, dtype=jnp.float32)
        if concentration.ndim == 0:
            raise ValueError("concentration must be at least 1-D.")

        self._concentration = concentration
        self._tfp_dist = tfd.Dirichlet(concentration=concentration)
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def dim(self) -> int:
        return self._concentration.shape[-1]

    # -- support ------------------------------------------------------------

    @classmethod
    def _default_support(cls) -> Constraint:
        return simplex

    @property
    def support(self) -> Constraint:
        return simplex

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Dirichlet:
        """Fit a Dirichlet by method-of-moments from *other*."""
        num_samples = kwargs.pop("num_samples", 1024)

        if isinstance(other, Dirichlet):
            result = cls(concentration=other.concentration, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result

        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples, axis=0)
        var = jnp.var(samples, axis=0)
        # Method of moments: concentration_0 = m[0]*(1 - m[0])/var[0] - 1
        concentration_0 = m[0] * (1.0 - m[0]) / (var[0] + 1e-8) - 1.0
        concentration_0 = jnp.maximum(concentration_0, 0.01)
        concentration = m * concentration_0
        concentration = jnp.maximum(concentration, 0.01)

        result = cls(concentration=concentration, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------


class Multinomial(TFPDistribution):
    """
    Multinomial distribution over count vectors.

    Exactly one of *probs* or *logits* must be provided.

    Parameters
    ----------
    total_count : int or array-like
        Number of trials.
    probs : array-like, shape ``(k,)``, optional
        Event probabilities (need not be normalised).
    logits : array-like, shape ``(k,)``, optional
        Log-odds of each event.
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        total_count: int | ArrayLike,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        *,
        name: str | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")

        total_count = jnp.asarray(total_count, dtype=jnp.float32)

        if probs is not None:
            probs = jnp.asarray(probs, dtype=jnp.float32)
            self._probs = probs
            self._logits = None
            self._tfp_dist = tfd.Multinomial(
                total_count=total_count, probs=probs
            )
        else:
            logits = jnp.asarray(logits, dtype=jnp.float32)
            self._logits = logits
            self._probs = None
            self._tfp_dist = tfd.Multinomial(
                total_count=total_count, logits=logits
            )

        self._total_count = total_count
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def total_count(self) -> Array:
        return self._total_count

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative_integer

    @property
    def support(self) -> Constraint:
        return non_negative_integer

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Multinomial:
        """Fit a Multinomial from *other*.

        Keyword Args
        -------------
        total_count : int
            Number of trials. **Required** unless *other* is already a
            ``Multinomial`` (in which case its ``total_count`` is reused).
        num_samples : int
            Samples drawn for moment-matching (default 1024).
        """
        num_samples = kwargs.pop("num_samples", 1024)
        total_count = kwargs.pop("total_count", None)

        if isinstance(other, Multinomial):
            if total_count is None:
                total_count = other.total_count
            probs = other.probs
            if probs is None:
                # Convert logits to probs
                probs = jax.nn.softmax(other.logits)
            result = cls(
                total_count=total_count, probs=probs, name=name or other.name
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result

        if total_count is None:
            raise ValueError(
                "total_count must be provided via kwargs when fitting "
                "Multinomial from a non-Multinomial distribution."
            )

        samples = other.sample(key, sample_shape=(num_samples,))
        mean = jnp.mean(samples, axis=0)
        total_count = jnp.asarray(total_count, dtype=jnp.float32)
        probs = mean / total_count
        # Normalise to ensure valid probabilities
        probs = probs / jnp.sum(probs)

        result = cls(total_count=total_count, probs=probs, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


# ---------------------------------------------------------------------------
# Wishart
# ---------------------------------------------------------------------------


class Wishart(TFPDistribution):
    """
    Wishart distribution over positive-definite matrices.

    Exactly one of *scale_tril* or *scale* must be provided.

    Parameters
    ----------
    df : float or array-like
        Degrees of freedom (must be >= dimension).
    scale_tril : array-like, shape ``(d, d)``, optional
        Lower-triangular Cholesky factor of the scale matrix.
    scale : array-like, shape ``(d, d)``, optional
        Full scale matrix (Cholesky-decomposed internally).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        df: float | ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        scale: ArrayLike | None = None,
        name: str | None = None,
    ):
        if scale_tril is not None and scale is not None:
            raise ValueError(
                "Provide exactly one of scale_tril or scale, not both."
            )
        if scale_tril is None and scale is None:
            raise ValueError("One of scale_tril or scale must be provided.")

        df = jnp.asarray(df, dtype=jnp.float32)

        if scale is not None:
            scale = jnp.asarray(scale, dtype=jnp.float32)
            scale_tril = jnp.linalg.cholesky(scale)
        else:
            scale_tril = jnp.asarray(scale_tril, dtype=jnp.float32)

        self._df = df
        self._scale_tril = scale_tril
        self._tfp_dist = tfd.WishartTriL(df=df, scale_tril=scale_tril)
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def df(self) -> Array:
        return self._df

    @property
    def scale_tril(self) -> Array:
        return self._scale_tril

    @property
    def scale(self) -> Array:
        """Full scale matrix (computed from Cholesky factor)."""
        return self._scale_tril @ self._scale_tril.T

    @property
    def dim(self) -> int:
        return self._scale_tril.shape[-1]

    # -- support ------------------------------------------------------------

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive_definite

    @property
    def support(self) -> Constraint:
        return positive_definite

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> Wishart:
        """Fit a Wishart by moment-matching from *other*."""
        num_samples = kwargs.pop("num_samples", 1024)

        if isinstance(other, Wishart):
            result = cls(
                df=other.df, scale_tril=other.scale_tril, name=name or other.name
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result

        samples = other.sample(key, sample_shape=(num_samples,))
        mean_mat = jnp.mean(samples, axis=0)
        # Reasonable default: df = event_dim + 2
        event_dim = mean_mat.shape[-1]
        df = jnp.asarray(event_dim + 2, dtype=jnp.float32)
        scale_mat = mean_mat / df
        # Ensure PSD via symmetrisation + jitter
        scale_mat = 0.5 * (scale_mat + scale_mat.T)
        scale_mat = scale_mat + 1e-6 * jnp.eye(scale_mat.shape[0])

        result = cls(df=df, scale=scale_mat, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


# ---------------------------------------------------------------------------
# VonMisesFisher
# ---------------------------------------------------------------------------


class VonMisesFisher(TFPDistribution):
    """
    Von Mises-Fisher distribution on the unit hypersphere.

    Parameters
    ----------
    mean_direction : array-like, shape ``(d,)``
        Unit vector giving the mean direction.
    concentration : float or array-like
        Scalar concentration parameter (kappa >= 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        mean_direction: ArrayLike,
        concentration: float | ArrayLike,
        *,
        name: str | None = None,
    ):
        mean_direction = jnp.asarray(mean_direction, dtype=jnp.float32)
        concentration = jnp.asarray(concentration, dtype=jnp.float32)

        self._mean_direction = mean_direction
        self._concentration = concentration
        self._tfp_dist = tfd.VonMisesFisher(
            mean_direction=mean_direction, concentration=concentration
        )
        self._name = name

    # -- convenient accessors -----------------------------------------------

    @property
    def mean_direction(self) -> Array:
        return self._mean_direction

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def dim(self) -> int:
        return self._mean_direction.shape[-1]

    # -- support ------------------------------------------------------------

    @classmethod
    def _default_support(cls) -> Constraint:
        return sphere

    @property
    def support(self) -> Constraint:
        return sphere

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> VonMisesFisher:
        """Fit a VonMisesFisher from *other* by estimating mean direction and concentration."""
        num_samples = kwargs.pop("num_samples", 1024)

        if isinstance(other, VonMisesFisher):
            result = cls(
                mean_direction=other.mean_direction,
                concentration=other.concentration,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result

        samples = other.sample(key, sample_shape=(num_samples,))
        mean_vector = jnp.mean(samples, axis=0)
        norm = jnp.linalg.norm(mean_vector)
        mean_direction = mean_vector / jnp.maximum(norm, 1e-8)

        # Estimate concentration from resultant length
        # R = |mean_vector| / n (but mean already divides by n, so R = norm)
        R = norm
        d = mean_vector.shape[-1]
        # Approximation: kappa ~ R * (d - R^2) / (1 - R^2)
        R2 = R ** 2
        concentration = R * (d - R2) / jnp.maximum(1.0 - R2, 1e-8)
        concentration = jnp.maximum(concentration, 0.0)

        result = cls(
            mean_direction=mean_direction,
            concentration=concentration,
            name=name or other.name,
        )
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result
