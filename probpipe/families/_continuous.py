"""The continuous parametric families.

``Normal``, ``Beta``, ``Gamma``, ``InverseGamma``, ``Exponential``,
``LogNormal``, ``StudentT``, ``Uniform``, ``Cauchy``, ``Laplace``,
``HalfNormal``, ``HalfCauchy``, ``Pareto``, and ``TruncatedNormal`` each derive
their event term spec from their parameters and take an ``event_spec``
declaration that names the event's component.

The parameters broadcast against one another. Scalar parameters give a scalar
draw, and parameters with axes give one draw of independent coordinates of the
broadcast shape. Each family claims the mean, the variance, the covariance,
and, except ``Pareto``, whose backend has no quantile function, the quantile of
each coordinate.

A moment that diverges is the extended real ``inf``, and a moment known to be
undefined raises ``MathematicalDomainError``. The mean of a ``HalfCauchy`` is
``inf``, and so is the mean of an ``InverseGamma`` or a ``Pareto`` for a
concentration at most one. The mean of a ``Cauchy``, and of a ``StudentT`` for
degrees of freedom at most one, is undefined. A variance is ``inf`` where the
tail parameter lies between one and two, and undefined where the mean is
infinite or undefined, and a covariance follows its variance. Where the
answer depends on a parameter that is traced, the capability's guard reports
that it needs values not yet known.
"""

from __future__ import annotations

from typing import NoReturn

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array, _promote_floats
from ..core._dispatch import MathematicalDomainError
from ..core._specs import OutputSpec
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
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)
from ..linalg import DiagonalLinOp, LinOp
from ._backend import TFPDistribution

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

#: The capabilities the backend computes for the families with a quantile function.
_CLOSED_FORM = frozenset({SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile})


# ---------------------------------------------------------------------------
# Moments that diverge or are undefined
# ---------------------------------------------------------------------------


def _no_moment(law: TFPDistribution, moment: str, reason: str) -> NoReturn:
    """Raise that the *moment* of *law* is undefined, for *reason*.

    Raises
    ------
    MathematicalDomainError
        Always.
    """
    raise MathematicalDomainError(f"the {moment} of {law.label!r} does not exist: {reason}")


def _require_moment(law: TFPDistribution, holds: Array, moment: str, reason: str) -> None:
    """Raise unless *holds* for every coordinate, when *holds* is concrete.

    A traced *holds* decides nothing before the computation runs, so the
    capability then returns the backend's value.

    Raises
    ------
    MathematicalDomainError
        If *holds* is concrete and false for a coordinate.
    """
    if not isinstance(holds, jax.core.Tracer) and not bool(jnp.all(holds)):
        _no_moment(law, moment, reason)


def _decided(*parameters: Array) -> bool | None:
    """``True`` when every parameter is concrete, and ``None`` while one is traced."""
    return None if any(isinstance(p, jax.core.Tracer) for p in parameters) else True


class _TailBoundedMoments:
    """The moments of a family whose tail parameter decides which converge.

    The mean is finite where the parameter named by ``_tail_parameter``
    exceeds one. At most one it is ``inf`` when ``_mean_diverges`` is set, and
    undefined otherwise. The variance, with the covariance, is finite where the
    parameter exceeds two, ``inf`` where it lies between one and two, and
    undefined at most one, where the mean is infinite or undefined. Each guard
    decides once the parameter is concrete.
    """

    _tail_parameter: str
    _tail_description: str
    _mean_diverges: bool

    def _tail(self) -> Array:
        return getattr(self, self._tail_parameter)

    def _mean(self) -> Array:
        """The backend's mean, ``inf`` at each coordinate where it diverges.

        Raises
        ------
        MathematicalDomainError
            If the mean is undefined at a coordinate, where the parameter is at
            most one and the mean does not diverge.
        """
        tail = self._tail()
        if self._mean_diverges:
            return jnp.where(tail > 1, self._tfp_dist.mean(), jnp.inf)
        reason = f"it is defined only for {self._tail_description} above one"
        _require_moment(self, tail > 1, "mean", reason)
        return jnp.where(tail > 1, self._tfp_dist.mean(), jnp.nan)

    def _mean_guard(self) -> bool | None:
        """The parameter that decides whether the mean is defined is concrete."""
        return _decided(self._tail())

    def _variance_values(self, moment: str, reason: str) -> Array:
        """The variance, ``inf`` where the parameter lies between one and two.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most one for a coordinate, naming *moment*
            and *reason*.
        """
        tail = self._tail()
        _require_moment(self, tail > 1, moment, reason)
        diverging = jnp.where(tail > 1, jnp.inf, jnp.nan)
        return jnp.where(tail > 2, self._tfp_dist.variance(), diverging)

    def _variance(self) -> Array:
        """The backend's variance, ``inf`` at each coordinate where it diverges.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most one for a coordinate, where the mean is
            infinite or undefined.
        """
        reason = f"it is defined only for {self._tail_description} above one"
        return self._variance_values("variance", reason)

    def _variance_guard(self) -> bool | None:
        """The parameter that decides whether the variance is defined is concrete."""
        return _decided(self._tail())

    def _cov(self) -> LinOp:
        """The diagonal covariance of the independent coordinates, ``inf`` where the variance is.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most one for a coordinate, where the
            variance is undefined.
        """
        reason = f"its variance is defined only for {self._tail_description} above one"
        return DiagonalLinOp(jnp.reshape(self._variance_values("covariance", reason), (-1,)))

    def _cov_guard(self) -> bool | None:
        """The parameter that decides whether the covariance is defined is concrete."""
        return _decided(self._tail())


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


class Normal(TFPDistribution):
    """Univariate normal (Gaussian) distribution.

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Mean of the distribution.
    scale : array-like
        Standard deviation (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(label, tfd.Normal(loc=self._loc, scale=self._scale), event_spec=event_spec)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


class Beta(TFPDistribution):
    """Beta distribution on [0, 1].

    Parameters
    ----------
    label : str
        Distribution name.
    alpha : array-like
        First concentration parameter (> 0).
    beta : array-like
        Second concentration parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, alpha: ArrayLike, beta: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._alpha, self._beta) = _promote_floats(alpha, beta)
        super().__init__(
            label,
            tfd.Beta(concentration1=self._alpha, concentration0=self._beta),
            event_spec=event_spec,
        )

    @property
    def alpha(self) -> Array:
        return self._alpha

    @property
    def beta(self) -> Array:
        return self._beta

    def _event_support(self) -> Constraint:
        return unit_interval


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


class Gamma(TFPDistribution):
    """Gamma distribution.

    Parameters
    ----------
    label : str
        Distribution name.
    concentration : array-like
        Shape parameter (> 0).
    rate : array-like
        Rate (inverse scale) parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self,
        label: str,
        concentration: ArrayLike,
        rate: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._rate) = _promote_floats(concentration, rate)
        super().__init__(
            label,
            tfd.Gamma(concentration=self._concentration, rate=self._rate),
            event_spec=event_spec,
        )

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def rate(self) -> Array:
        return self._rate

    def _event_support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# InverseGamma
# ---------------------------------------------------------------------------


class InverseGamma(_TailBoundedMoments, TFPDistribution):
    """Inverse-gamma distribution.

    The mean is finite for a concentration above one and ``inf`` at most one.
    The variance is finite above two, ``inf`` between one and two, and
    undefined at most one, where it raises ``MathematicalDomainError``.

    Parameters
    ----------
    label : str
        Distribution name.
    concentration : array-like
        Shape parameter (> 0).
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM
    _tail_parameter = "_concentration"
    _tail_description = "a concentration"
    _mean_diverges = True

    def __init__(
        self,
        label: str,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        super().__init__(
            label,
            tfd.InverseGamma(concentration=self._concentration, scale=self._scale),
            event_spec=event_spec,
        )

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------


class Exponential(TFPDistribution):
    """Exponential distribution.

    Parameters
    ----------
    label : str
        Distribution name.
    rate : array-like
        Rate parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(self, label: str, rate: ArrayLike, *, event_spec: OutputSpec | None = None):
        self._rate = _as_float_array(rate)
        super().__init__(label, tfd.Exponential(rate=self._rate), event_spec=event_spec)

    @property
    def rate(self) -> Array:
        return self._rate

    def _event_support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# LogNormal
# ---------------------------------------------------------------------------


class LogNormal(TFPDistribution):
    """Log-normal distribution.

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(
            label,
            tfd.LogNormal(loc=self._loc, scale=self._scale),
            event_spec=event_spec,
        )

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return positive


# ---------------------------------------------------------------------------
# StudentT
# ---------------------------------------------------------------------------


class StudentT(_TailBoundedMoments, TFPDistribution):
    """Student's t-distribution.

    The mean is the location for degrees of freedom above one and undefined at
    most one, where it raises ``MathematicalDomainError``. The variance is
    finite above two, ``inf`` between one and two, and undefined at most one.

    Parameters
    ----------
    label : str
        Distribution name.
    df : array-like
        Degrees of freedom (> 0).
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM
    _tail_parameter = "_df"
    _tail_description = "degrees of freedom"
    _mean_diverges = False

    def __init__(
        self,
        label: str,
        df: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._df, self._loc, self._scale) = _promote_floats(df, loc, scale)
        super().__init__(
            label,
            tfd.StudentT(df=self._df, loc=self._loc, scale=self._scale),
            event_spec=event_spec,
        )

    @property
    def df(self) -> Array:
        return self._df

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


class Uniform(TFPDistribution):
    """Uniform distribution on [low, high].

    Parameters
    ----------
    label : str
        Distribution name.
    low : array-like
        Lower bound.
    high : array-like
        Upper bound (> low).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, low: ArrayLike, high: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._low, self._high) = _promote_floats(low, high)
        super().__init__(label, tfd.Uniform(low=self._low, high=self._high), event_spec=event_spec)

    @property
    def low(self) -> Array:
        return self._low

    @property
    def high(self) -> Array:
        return self._high

    def _event_support(self) -> Constraint:
        return interval(self._low, self._high)


# ---------------------------------------------------------------------------
# Cauchy
# ---------------------------------------------------------------------------


class Cauchy(TFPDistribution):
    """Cauchy distribution.

    Its mean, variance, and covariance do not exist, and each raises
    ``MathematicalDomainError``.

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(label, tfd.Cauchy(loc=self._loc, scale=self._scale), event_spec=event_spec)

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return real

    def _mean(self) -> NoReturn:
        """The mean, which does not exist.

        Raises
        ------
        MathematicalDomainError
            Always, since ``E|X|`` is infinite.
        """
        _no_moment(self, "mean", "E|X| is infinite")

    def _variance(self) -> NoReturn:
        """The variance, which does not exist.

        Raises
        ------
        MathematicalDomainError
            Always, since the law has no mean.
        """
        _no_moment(self, "variance", "the law has no mean")

    def _cov(self) -> NoReturn:
        """The covariance, which does not exist.

        Raises
        ------
        MathematicalDomainError
            Always, since the law has no variance.
        """
        _no_moment(self, "covariance", "its variance does not exist, since the law has no mean")


# ---------------------------------------------------------------------------
# Laplace
# ---------------------------------------------------------------------------


class Laplace(TFPDistribution):
    """Laplace distribution.

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(
            label, tfd.Laplace(loc=self._loc, scale=self._scale), event_spec=event_spec
        )

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return real


# ---------------------------------------------------------------------------
# HalfNormal
# ---------------------------------------------------------------------------


class HalfNormal(TFPDistribution):
    """Half-normal distribution (support on [0, inf)).

    Parameters
    ----------
    label : str
        Distribution name.
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(self, label: str, scale: ArrayLike, *, event_spec: OutputSpec | None = None):
        self._scale = _as_float_array(scale)
        super().__init__(label, tfd.HalfNormal(scale=self._scale), event_spec=event_spec)

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return non_negative


# ---------------------------------------------------------------------------
# HalfCauchy
# ---------------------------------------------------------------------------


class HalfCauchy(TFPDistribution):
    """Half-Cauchy distribution (support on [loc, inf)).

    Its mean is ``inf``, and its variance and covariance are undefined, since
    the mean is infinite, so each raises ``MathematicalDomainError``.

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Location parameter.
    scale : array-like
        Scale parameter (> 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self, label: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(
            label,
            tfd.HalfCauchy(loc=self._loc, scale=self._scale),
            event_spec=event_spec,
        )

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return greater_than(self._loc)

    def _mean(self) -> Array:
        """The mean, ``inf`` at each coordinate, since ``E[X]`` diverges."""
        return jnp.full(jnp.shape(self._tfp_dist.mean()), jnp.inf, self._tfp_dist.dtype)

    def _variance(self) -> NoReturn:
        """The variance, which does not exist.

        Raises
        ------
        MathematicalDomainError
            Always, since the mean is infinite.
        """
        _no_moment(self, "variance", "the mean is infinite")

    def _cov(self) -> NoReturn:
        """The covariance, which does not exist.

        Raises
        ------
        MathematicalDomainError
            Always, since the mean is infinite.
        """
        _no_moment(self, "covariance", "its variance does not exist, since the mean is infinite")


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


class Pareto(_TailBoundedMoments, TFPDistribution):
    """Pareto distribution.

    The mean is finite for a concentration above one and ``inf`` at most one.
    The variance is finite above two, ``inf`` between one and two, and
    undefined at most one, where it raises ``MathematicalDomainError``.

    Parameters
    ----------
    label : str
        Distribution name.
    concentration : array-like
        Tail index (shape parameter, > 0).
    scale : array-like
        Minimum value (scale parameter, > 0).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})
    _tail_parameter = "_concentration"
    _tail_description = "a concentration"
    _mean_diverges = True

    def __init__(
        self,
        label: str,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        super().__init__(
            label,
            tfd.Pareto(concentration=self._concentration, scale=self._scale),
            event_spec=event_spec,
        )

    @property
    def concentration(self) -> Array:
        return self._concentration

    @property
    def scale(self) -> Array:
        return self._scale

    def _event_support(self) -> Constraint:
        return greater_than(self._scale)


# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------


class TruncatedNormal(TFPDistribution):
    """Truncated normal distribution on [low, high].

    Parameters
    ----------
    label : str
        Distribution name.
    loc : array-like
        Mean of the underlying normal distribution.
    scale : array-like
        Standard deviation of the underlying normal distribution (> 0).
    low : array-like
        Lower truncation bound.
    high : array-like
        Upper truncation bound (> low).
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The family
        fills a pending type, as in ``OutputSpec(theta=None)``. By default the
        component is ``name``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If *event_spec* declares a type that one draw does not conform to.
    """

    _backend_capabilities = _CLOSED_FORM

    def __init__(
        self,
        label: str,
        loc: ArrayLike,
        scale: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._loc, self._scale, self._low, self._high) = _promote_floats(loc, scale, low, high)
        super().__init__(
            label,
            tfd.TruncatedNormal(loc=self._loc, scale=self._scale, low=self._low, high=self._high),
            event_spec=event_spec,
        )

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

    def _event_support(self) -> Constraint:
        return interval(self._low, self._high)
