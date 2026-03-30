"""
Continuous univariate distributions backed by TFP.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ._tfp_base import TFPDistribution
from ..core.distribution import (
    ArrayDistribution,
    EmpiricalDistribution,
)
from ..core.provenance import Provenance
from ..core.constraints import (
    Constraint,
    real,
    positive,
    non_negative,
    unit_interval,
    interval,
    greater_than,
)
from ..custom_types import Array, ArrayLike, PRNGKey

__all__ = [
    "Normal",
    "Beta",
    "Gamma",
    "InverseGamma",
    "Exponential",
    "LogNormal",
    "StudentT",
    "Uniform",
    "Cauchy",
    "Laplace",
    "HalfNormal",
    "HalfCauchy",
    "Pareto",
    "TruncatedNormal",
]


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


class Normal(TFPDistribution):
    """Univariate normal (Gaussian) distribution.

    Parameters
    ----------
    loc : array-like
        Mean of the distribution.
    scale : array-like
        Standard deviation (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.Normal(loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real

    @classmethod
    def _default_support(cls) -> Constraint:
        return real



# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class Beta(TFPDistribution):
    """Beta distribution on [0, 1].

    Parameters
    ----------
    alpha : array-like
        First concentration parameter (> 0).
    beta : array-like
        Second concentration parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        alpha: ArrayLike,
        beta: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._alpha = jnp.asarray(alpha, dtype=jnp.float32)
        self._beta = jnp.asarray(beta, dtype=jnp.float32)
        self._tfp_dist = tfd.Beta(concentration1=self._alpha, concentration0=self._beta)
        self._name = name

    @property
    def alpha(self) -> Array:
        return self._alpha

    @property
    def beta(self) -> Array:
        return self._beta

    @property
    def support(self) -> Constraint:
        return unit_interval

    @classmethod
    def _default_support(cls) -> Constraint:
        return unit_interval



# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


class Gamma(TFPDistribution):
    """Gamma distribution.

    Parameters
    ----------
    concentration : array-like
        Shape parameter (> 0).
    rate : array-like
        Rate (inverse scale) parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        concentration: ArrayLike,
        rate: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self._rate = jnp.asarray(rate, dtype=jnp.float32)
        self._tfp_dist = tfd.Gamma(concentration=self._concentration, rate=self._rate)
        self._name = name

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def rate(self) -> Array:
        return self._rate

    @property
    def support(self) -> Constraint:
        return positive

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive



# ---------------------------------------------------------------------------
# InverseGamma
# ---------------------------------------------------------------------------


class InverseGamma(TFPDistribution):
    """Inverse-gamma distribution.

    Parameters
    ----------
    concentration : array-like
        Shape parameter (> 0).
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.InverseGamma(
            concentration=self._concentration, scale=self._scale
        )
        self._name = name

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return positive

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive



# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


class Exponential(TFPDistribution):
    """Exponential distribution.

    Parameters
    ----------
    rate : array-like
        Rate parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        rate: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._rate = jnp.asarray(rate, dtype=jnp.float32)
        self._tfp_dist = tfd.Exponential(rate=self._rate)
        self._name = name

    @property
    def rate(self) -> Array:
        return self._rate

    @property
    def support(self) -> Constraint:
        return positive

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive



# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------


class LogNormal(TFPDistribution):
    """Log-normal distribution.

    Parameters
    ----------
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.LogNormal(loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return positive

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive



# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------


class StudentT(TFPDistribution):
    """Student's t-distribution.

    Parameters
    ----------
    df : array-like
        Degrees of freedom (> 0).
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        df: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._df = jnp.asarray(df, dtype=jnp.float32)
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.StudentT(df=self._df, loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def df(self) -> Array:
        return self._df

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real

    @classmethod
    def _default_support(cls) -> Constraint:
        return real



# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


class Uniform(TFPDistribution):
    """Uniform distribution on [low, high].

    Parameters
    ----------
    low : array-like
        Lower bound.
    high : array-like
        Upper bound (> low).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        low: ArrayLike,
        high: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._low = jnp.asarray(low, dtype=jnp.float32)
        self._high = jnp.asarray(high, dtype=jnp.float32)
        self._tfp_dist = tfd.Uniform(low=self._low, high=self._high)
        self._name = name

    @property
    def low(self) -> Array:
        return self._low

    @property
    def high(self) -> Array:
        return self._high

    @property
    def support(self) -> Constraint:
        return interval(float(self._low), float(self._high))

    @classmethod
    def _default_support(cls) -> Constraint:
        return real



# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------


class Cauchy(TFPDistribution):
    """Cauchy distribution.

    Parameters
    ----------
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.Cauchy(loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real

    @classmethod
    def _default_support(cls) -> Constraint:
        return real



# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------


class Laplace(TFPDistribution):
    """Laplace distribution.

    Parameters
    ----------
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.Laplace(loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real

    @classmethod
    def _default_support(cls) -> Constraint:
        return real



# ---------------------------------------------------------------------------
# HalfNormal
# ---------------------------------------------------------------------------


class HalfNormal(TFPDistribution):
    """Half-normal distribution (support on [0, inf)).

    Parameters
    ----------
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.HalfNormal(scale=self._scale)
        self._name = name

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return non_negative

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative



# ---------------------------------------------------------------------------
# HalfCauchy
# ---------------------------------------------------------------------------


class HalfCauchy(TFPDistribution):
    """Half-Cauchy distribution (support on [loc, inf)).

    Parameters
    ----------
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.HalfCauchy(loc=self._loc, scale=self._scale)
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return greater_than(float(self._loc))

    @classmethod
    def _default_support(cls) -> Constraint:
        return non_negative



# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class Pareto(TFPDistribution):
    """Pareto distribution.

    Parameters
    ----------
    concentration : array-like
        Tail index (shape parameter, > 0).
    scale : array-like
        Minimum value (scale parameter, > 0).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._concentration = jnp.asarray(concentration, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._tfp_dist = tfd.Pareto(
            concentration=self._concentration, scale=self._scale
        )
        self._name = name

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return greater_than(float(self._scale))

    @classmethod
    def _default_support(cls) -> Constraint:
        return positive



# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------


class TruncatedNormal(TFPDistribution):
    """Truncated normal distribution on [low, high].

    Parameters
    ----------
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    low : array-like
        Lower truncation bound.
    high : array-like
        Upper truncation bound (> low).
    name : str, optional
        Distribution name for provenance / JointDistribution.
    """

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        *,
        name: str | None = None,
    ):
        self._loc = jnp.asarray(loc, dtype=jnp.float32)
        self._scale = jnp.asarray(scale, dtype=jnp.float32)
        self._low = jnp.asarray(low, dtype=jnp.float32)
        self._high = jnp.asarray(high, dtype=jnp.float32)
        self._tfp_dist = tfd.TruncatedNormal(
            loc=self._loc, scale=self._scale, low=self._low, high=self._high
        )
        self._name = name

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def low(self) -> Array:
        return self._low

    @property
    def high(self) -> Array:
        return self._high

    @property
    def support(self) -> Constraint:
        return interval(float(self._low), float(self._high))

    @classmethod
    def _default_support(cls) -> Constraint:
        return real

