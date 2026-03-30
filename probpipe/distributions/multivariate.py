"""
Multivariate distributions backed by TFP.
"""

from __future__ import annotations

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.constraints import (
    Constraint,
    real,
    positive_definite,
    simplex,
    non_negative_integer,
    sphere,
)
from ..custom_types import Array, ArrayLike

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

