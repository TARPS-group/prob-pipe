"""
Multivariate distributions backed by TFP.
"""

from __future__ import annotations

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from .._dtype import _as_float_array, _promote_floats
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
    name : str
        Distribution name.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        cov: ArrayLike | None = None,
        name: str,
    ):
        if scale_tril is not None and cov is not None:
            raise ValueError("Provide exactly one of scale_tril or cov, not both.")

        if scale_tril is not None:
            _, (loc, scale_tril) = _promote_floats(loc, scale_tril)
        elif cov is not None:
            _, (loc, cov) = _promote_floats(loc, cov)
        else:
            raise ValueError("One of scale_tril or cov must be provided.")

        if loc.ndim == 0:
            loc = loc.reshape(1)

        if cov is not None:
            if cov.shape != (loc.shape[0], loc.shape[0]):
                raise ValueError(
                    f"cov shape {cov.shape} does not match loc length {loc.shape[0]}."
                )
            scale_tril = jnp.linalg.cholesky(cov)

        self._loc = loc
        self._scale_tril = scale_tril
        self._tfp_dist = tfd.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril
        )
        super().__init__(name=name)

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
    name : str
        Distribution name.
    """

    def __init__(
        self,
        concentration: ArrayLike,
        *,
        name: str,
    ):
        concentration = _as_float_array(concentration)
        if concentration.ndim == 0:
            raise ValueError("concentration must be at least 1-D.")

        self._concentration = concentration
        self._tfp_dist = tfd.Dirichlet(concentration=concentration)
        super().__init__(name=name)

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
    name : str
        Distribution name.
    """

    def __init__(
        self,
        total_count: int | ArrayLike,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        *,
        name: str,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("Exactly one of probs or logits must be provided.")

        if probs is not None:
            _, (total_count, probs) = _promote_floats(total_count, probs)
            self._probs = probs
            self._logits = None
            self._tfp_dist = tfd.Multinomial(
                total_count=total_count, probs=probs
            )
        else:
            _, (total_count, logits) = _promote_floats(total_count, logits)
            self._logits = logits
            self._probs = None
            self._tfp_dist = tfd.Multinomial(
                total_count=total_count, logits=logits
            )

        self._total_count = total_count
        super().__init__(name=name)

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
    name : str
        Distribution name.
    """

    def __init__(
        self,
        df: float | ArrayLike,
        scale_tril: ArrayLike | None = None,
        *,
        scale: ArrayLike | None = None,
        name: str,
    ):
        if scale_tril is not None and scale is not None:
            raise ValueError(
                "Provide exactly one of scale_tril or scale, not both."
            )
        if scale_tril is None and scale is None:
            raise ValueError("One of scale_tril or scale must be provided.")

        if scale is not None:
            _, (df, scale) = _promote_floats(df, scale)
            scale_tril = jnp.linalg.cholesky(scale)
        else:
            _, (df, scale_tril) = _promote_floats(df, scale_tril)

        self._df = df
        self._scale_tril = scale_tril
        self._tfp_dist = tfd.WishartTriL(df=df, scale_tril=scale_tril)
        super().__init__(name=name)

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
    name : str
        Distribution name.
    """

    def __init__(
        self,
        mean_direction: ArrayLike,
        concentration: float | ArrayLike,
        *,
        name: str,
    ):
        _, (mean_direction, concentration) = _promote_floats(
            mean_direction, concentration
        )

        self._mean_direction = mean_direction
        self._concentration = concentration
        self._tfp_dist = tfd.VonMisesFisher(
            mean_direction=mean_direction, concentration=concentration
        )
        super().__init__(name=name)

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

