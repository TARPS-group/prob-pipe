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

A moment known not to exist raises ``MathematicalDomainError``: the mean and
the variance of a ``Cauchy`` and a ``HalfCauchy``, the mean of a ``StudentT``
for degrees of freedom at most one and its variance at most two, and the mean
of an ``InverseGamma`` or a ``Pareto`` for a concentration at most one and its
variance at most two, a covariance with its variance. A moment that diverges
to infinity is reported as not existing too, since infinity is not a value of
the event. Where existence depends on a parameter that is traced, the
capability's guard reports that it needs values not yet known.
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
from ..linalg import LinOp
from ._backend import TFPDistribution, _backend_cov

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
# Moments that do not exist
# ---------------------------------------------------------------------------


def _no_moment(law: TFPDistribution, moment: str, reason: str) -> NoReturn:
    """Raise that the *moment* of *law* does not exist, for *reason*.

    Raises
    ------
    MathematicalDomainError
        Always.
    """
    raise MathematicalDomainError(f"the {moment} of {law.name!r} does not exist: {reason}")


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
    """The moments of a family whose tail parameter decides which exist.

    The mean exists where the parameter named by ``_tail_parameter`` exceeds
    one, and the variance, with the covariance, where it exceeds two. Each
    guard decides existence once the parameter is concrete.
    """

    _tail_parameter: str
    _tail_description: str

    def _tail(self) -> Array:
        return getattr(self, self._tail_parameter)

    def _mean(self) -> Array:
        """The backend's mean, where it exists.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most one for a coordinate.
        """
        reason = f"it is finite only for {self._tail_description} above one"
        _require_moment(self, self._tail() > 1, "mean", reason)
        return self._tfp_dist.mean()

    def _mean_guard(self) -> bool | None:
        """The parameter that decides whether the mean exists is concrete."""
        return _decided(self._tail())

    def _variance(self) -> Array:
        """The backend's variance, where it exists.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most two for a coordinate.
        """
        reason = f"it is finite only for {self._tail_description} above two"
        _require_moment(self, self._tail() > 2, "variance", reason)
        return self._tfp_dist.variance()

    def _variance_guard(self) -> bool | None:
        """The parameter that decides whether the variance exists is concrete."""
        return _decided(self._tail())

    def _cov(self) -> LinOp:
        """The diagonal covariance of the independent coordinates, where the variance exists.

        Raises
        ------
        MathematicalDomainError
            If the parameter is at most two for a coordinate.
        """
        reason = f"its variance is finite only for {self._tail_description} above two"
        _require_moment(self, self._tail() > 2, "covariance", reason)
        return _backend_cov(self)

    def _cov_guard(self) -> bool | None:
        """The parameter that decides whether the covariance exists is concrete."""
        return _decided(self._tail())


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
        self, name: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(name, tfd.Normal(loc=self._loc, scale=self._scale), event_spec=event_spec)

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
    name : str
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
        self, name: str, alpha: ArrayLike, beta: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._alpha, self._beta) = _promote_floats(alpha, beta)
        super().__init__(
            name,
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
    name : str
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
        name: str,
        concentration: ArrayLike,
        rate: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._rate) = _promote_floats(concentration, rate)
        super().__init__(
            name,
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

    The mean exists for a concentration above one and the variance above two;
    elsewhere each raises ``MathematicalDomainError``.

    Parameters
    ----------
    name : str
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

    def __init__(
        self,
        name: str,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        super().__init__(
            name,
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
    name : str
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

    def __init__(self, name: str, rate: ArrayLike, *, event_spec: OutputSpec | None = None):
        self._rate = _as_float_array(rate)
        super().__init__(name, tfd.Exponential(rate=self._rate), event_spec=event_spec)

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
    name : str
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
        self, name: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(
            name,
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

    The mean exists for degrees of freedom above one and the variance above
    two; elsewhere each raises ``MathematicalDomainError``.

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

    def __init__(
        self,
        name: str,
        df: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._df, self._loc, self._scale) = _promote_floats(df, loc, scale)
        super().__init__(
            name,
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
    name : str
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
        self, name: str, low: ArrayLike, high: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._low, self._high) = _promote_floats(low, high)
        super().__init__(name, tfd.Uniform(low=self._low, high=self._high), event_spec=event_spec)

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
    name : str
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
        self, name: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(name, tfd.Cauchy(loc=self._loc, scale=self._scale), event_spec=event_spec)

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
    name : str
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
        self, name: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(name, tfd.Laplace(loc=self._loc, scale=self._scale), event_spec=event_spec)

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
    name : str
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

    def __init__(self, name: str, scale: ArrayLike, *, event_spec: OutputSpec | None = None):
        self._scale = _as_float_array(scale)
        super().__init__(name, tfd.HalfNormal(scale=self._scale), event_spec=event_spec)

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

    Its mean is infinite, and its variance and covariance do not exist, so each
    raises ``MathematicalDomainError``.

    Parameters
    ----------
    name : str
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
        self, name: str, loc: ArrayLike, scale: ArrayLike, *, event_spec: OutputSpec | None = None
    ):
        _, (self._loc, self._scale) = _promote_floats(loc, scale)
        super().__init__(
            name,
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

    def _mean(self) -> NoReturn:
        """The mean, which is infinite.

        Raises
        ------
        MathematicalDomainError
            Always, since ``E[X]`` diverges.
        """
        _no_moment(self, "mean", "E[X] is infinite")

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

    The mean exists for a concentration above one and the variance above two;
    elsewhere each raises ``MathematicalDomainError``.

    Parameters
    ----------
    name : str
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

    def __init__(
        self,
        name: str,
        concentration: ArrayLike,
        scale: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._concentration, self._scale) = _promote_floats(concentration, scale)
        super().__init__(
            name,
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
        name: str,
        loc: ArrayLike,
        scale: ArrayLike,
        low: ArrayLike,
        high: ArrayLike,
        *,
        event_spec: OutputSpec | None = None,
    ):
        _, (self._loc, self._scale, self._low, self._high) = _promote_floats(loc, scale, low, high)
        super().__init__(
            name,
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
