"""
Continuous univariate distributions backed by TFP.
"""

from __future__ import annotations

import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array, _promote_floats
from ..core.constraints import (
    Constraint,
    greater_than,
    interval,
    non_negative,
    positive,
    real,
    unit_interval,
)
from ..custom_types import Array, ArrayLike
from ._tfp_base import TFPDistribution

__all__ = [
    "Beta",
    "Cauchy",
    "Exponential",
    "Gamma",
    "HalfCauchy",
    "HalfNormal",
    "InverseGamma",
    "Laplace",
    "LogNormal",
    "Normal",
    "Pareto",
    "StudentT",
    "TruncatedNormal",
    "Uniform",
]


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


class Normal(TFPDistribution):
    """Univariate normal (Gaussian) distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Mean of the distribution.
    scale : array-like
        Standard deviation (> 0).
    """

    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        self._tfp_dist = tfd.Normal(loc=self._loc, scale=self._scale)
        super().__init__(name=name)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class Beta(TFPDistribution):
    """Beta distribution on [0, 1].

    Parameters
    ----------
    name : str
        Distribution name.
    alpha : array-like
        First concentration parameter (> 0).
    beta : array-like
        Second concentration parameter (> 0).
    """

    def __init__(self, name: str, alpha: ArrayLike, beta: ArrayLike):
        _, (self._alpha, self._beta) = _promote_floats(alpha, beta)
        self._tfp_dist = tfd.Beta(concentration1=self._alpha, concentration0=self._beta)
        super().__init__(name=name)

    @property
    def alpha(self) -> Array:
        return self._alpha

    @property
    def beta(self) -> Array:
        return self._beta

    @property
    def support(self) -> Constraint:
        return unit_interval


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


class Gamma(TFPDistribution):
    """Gamma distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    concentration : array-like
        Shape parameter (> 0).
    rate : array-like
        Rate (inverse scale) parameter (> 0).
    """

    def __init__(self, name: str, concentration: ArrayLike, rate: ArrayLike):
        _, (self._concentration, self._rate) = _promote_floats(concentration, rate)
        self._tfp_dist = tfd.Gamma(concentration=self._concentration, rate=self._rate)
        super().__init__(name=name)

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def rate(self) -> Array:
        return self._rate

    @property
    def support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# InverseGamma
# ---------------------------------------------------------------------------


class InverseGamma(TFPDistribution):
    """Inverse-gamma distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    concentration : array-like
        Shape parameter (> 0).
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, concentration: ArrayLike, scale: ArrayLike):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        self._tfp_dist = tfd.InverseGamma(concentration=self._concentration, scale=self._scale)
        super().__init__(name=name)

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


class Exponential(TFPDistribution):
    """Exponential distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    rate : array-like
        Rate parameter (> 0).
    """

    def __init__(self, name: str, rate: ArrayLike):
        self._rate = _as_float_array(rate)
        self._tfp_dist = tfd.Exponential(rate=self._rate)
        super().__init__(name=name)

    @property
    def rate(self) -> Array:
        return self._rate

    @property
    def support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------


class LogNormal(TFPDistribution):
    """Log-normal distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    """

    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        self._tfp_dist = tfd.LogNormal(loc=self._loc, scale=self._scale)
        super().__init__(name=name)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------


class StudentT(TFPDistribution):
    """Student's t-distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    df : array-like
        Degrees of freedom (> 0).
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, df: ArrayLike, loc: ArrayLike, scale: ArrayLike):
        _, (self._df, self._loc, self._scale) = _promote_floats(df, loc, scale)
        self._tfp_dist = tfd.StudentT(df=self._df, loc=self._loc, scale=self._scale)
        super().__init__(name=name)

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


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


class Uniform(TFPDistribution):
    """Uniform distribution on [low, high].

    Parameters
    ----------
    name : str
        Distribution name.
    low : array-like
        Lower bound.
    high : array-like
        Upper bound (> low).
    """

    def __init__(self, name: str, low: ArrayLike, high: ArrayLike):
        _, (self._low, self._high) = _promote_floats(low, high)
        self._tfp_dist = tfd.Uniform(low=self._low, high=self._high)
        super().__init__(name=name)

    @property
    def low(self) -> Array:
        return self._low

    @property
    def high(self) -> Array:
        return self._high

    @property
    def support(self) -> Constraint:
        return interval(self._low, self._high)


# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------


class Cauchy(TFPDistribution):
    """Cauchy distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        self._tfp_dist = tfd.Cauchy(loc=self._loc, scale=self._scale)
        super().__init__(name=name)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------


class Laplace(TFPDistribution):
    """Laplace distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        self._tfp_dist = tfd.Laplace(loc=self._loc, scale=self._scale)
        super().__init__(name=name)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# HalfNormal
# ---------------------------------------------------------------------------


class HalfNormal(TFPDistribution):
    """Half-normal distribution (support on [0, inf)).

    Parameters
    ----------
    name : str
        Distribution name.
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, scale: ArrayLike):
        self._scale = _as_float_array(scale)
        self._tfp_dist = tfd.HalfNormal(scale=self._scale)
        super().__init__(name=name)

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return non_negative


# ---------------------------------------------------------------------------
# HalfCauchy
# ---------------------------------------------------------------------------


class HalfCauchy(TFPDistribution):
    """Half-Cauchy distribution (support on [loc, inf)).

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    """

    def __init__(self, name: str, loc: ArrayLike, scale: ArrayLike):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        self._tfp_dist = tfd.HalfCauchy(loc=self._loc, scale=self._scale)
        super().__init__(name=name)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return greater_than(self._loc)


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class Pareto(TFPDistribution):
    """Pareto distribution.

    Parameters
    ----------
    name : str
        Distribution name.
    concentration : array-like
        Tail index (shape parameter, > 0).
    scale : array-like
        Minimum value (scale parameter, > 0).
    """

    def __init__(self, name: str, concentration: ArrayLike, scale: ArrayLike):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        self._tfp_dist = tfd.Pareto(concentration=self._concentration, scale=self._scale)
        super().__init__(name=name)

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    @property
    def support(self) -> Constraint:
        return greater_than(self._scale)


# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------


class TruncatedNormal(TFPDistribution):
    """Truncated normal distribution on [low, high].

    Parameters
    ----------
    name : str
        Distribution name.
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    low : array-like
        Lower truncation bound.
    high : array-like
        Upper truncation bound (> low).
    """

    def __init__(
        self, name: str, loc: ArrayLike, scale: ArrayLike, low: ArrayLike, high: ArrayLike
    ):
        _, (self._loc, self._scale, self._low, self._high) = _promote_floats(loc, scale, low, high)
        self._tfp_dist = tfd.TruncatedNormal(
            loc=self._loc, scale=self._scale, low=self._low, high=self._high
        )
        super().__init__(name=name)

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
        return interval(self._low, self._high)
