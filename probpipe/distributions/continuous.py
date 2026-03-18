"""
Continuous univariate distributions backed by TFP.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .distribution import (
    TFPDistribution,
    Distribution,
    EmpiricalDistribution,
    Provenance,
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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Normal:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Normal):
            result = cls(loc=other._loc, scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        result = cls(loc=m, scale=jnp.sqrt(v), name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Beta:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Beta):
            result = cls(alpha=other._alpha, beta=other._beta, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        # Method of moments
        common = m * (1.0 - m) / v - 1.0
        alpha = jnp.maximum(m * common, 0.01)
        beta = jnp.maximum((1.0 - m) * common, 0.01)
        result = cls(alpha=alpha, beta=beta, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Gamma:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Gamma):
            result = cls(
                concentration=other._concentration,
                rate=other._rate,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        concentration = m**2 / v
        rate = m / v
        result = cls(concentration=concentration, rate=rate, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> InverseGamma:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, InverseGamma):
            result = cls(
                concentration=other._concentration,
                scale=other._scale,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        concentration = m**2 / v + 2.0
        scale = m * (m**2 / v + 1.0)
        result = cls(concentration=concentration, scale=scale, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Exponential:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Exponential):
            result = cls(rate=other._rate, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        rate = 1.0 / m
        result = cls(rate=rate, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> LogNormal:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, LogNormal):
            result = cls(loc=other._loc, scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        scale = jnp.sqrt(jnp.log(1.0 + v / m**2))
        loc = jnp.log(m) - scale**2 / 2.0
        result = cls(loc=loc, scale=scale, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> StudentT:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, StudentT):
            result = cls(
                df=other._df,
                loc=other._loc,
                scale=other._scale,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        result = cls(df=5.0, loc=m, scale=jnp.sqrt(v), name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Uniform:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Uniform):
            result = cls(low=other._low, high=other._high, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        low = m - jnp.sqrt(3.0 * v)
        high = m + jnp.sqrt(3.0 * v)
        result = cls(low=low, high=high, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Cauchy:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Cauchy):
            result = cls(loc=other._loc, scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        # Rough heuristic: loc=mean, scale=sqrt(var)/2
        result = cls(loc=m, scale=jnp.sqrt(v) / 2.0, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Laplace:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Laplace):
            result = cls(loc=other._loc, scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        result = cls(loc=m, scale=jnp.sqrt(v / 2.0), name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> HalfNormal:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, HalfNormal):
            result = cls(scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        v = jnp.var(samples)
        # var = scale^2 * (1 - 2/pi), so scale = sqrt(var / (1 - 2/pi))
        scale = jnp.sqrt(2.0 * v / (1.0 - 2.0 / jnp.pi))
        result = cls(scale=scale, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> HalfCauchy:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, HalfCauchy):
            result = cls(loc=other._loc, scale=other._scale, name=name or other.name)
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        # HalfCauchy has infinite moments; use sample-based estimation
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        loc = jnp.float32(0.0)
        median = jnp.median(samples)
        # For HalfCauchy(0, scale), median = scale. Use median as scale estimate.
        scale = jnp.maximum(median - loc, 0.01)
        result = cls(loc=loc, scale=scale, name=name or other.name)
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> Pareto:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, Pareto):
            result = cls(
                concentration=other._concentration,
                scale=other._scale,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        # Sample-based MLE for Pareto
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        # MLE: scale = min(samples), concentration = n / sum(log(samples/scale))
        scale = jnp.min(samples)
        scale = jnp.maximum(scale, 1e-6)
        concentration = num_samples / jnp.sum(jnp.log(samples / scale))
        concentration = jnp.maximum(concentration, 0.01)
        result = cls(
            concentration=concentration, scale=scale, name=name or other.name
        )
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result


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

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> TruncatedNormal:
        if check_support:
            cls._check_support_compatible(other)
        if isinstance(other, TruncatedNormal):
            result = cls(
                loc=other._loc,
                scale=other._scale,
                low=other._low,
                high=other._high,
                name=name or other.name,
            )
            result.with_source(Provenance("from_distribution", parents=(other,)))
            return result
        num_samples = kwargs.pop("num_samples", 1024)
        if key is None:
            key = jax.random.PRNGKey(0)
        samples = other.sample(key, sample_shape=(num_samples,))
        m = jnp.mean(samples)
        v = jnp.var(samples)
        low = jnp.min(samples)
        high = jnp.max(samples)
        result = cls(
            loc=m, scale=jnp.sqrt(v), low=low, high=high, name=name or other.name
        )
        result.with_source(Provenance("from_distribution", parents=(other,)))
        return result
