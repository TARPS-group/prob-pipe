"""The discrete parametric families.

``Bernoulli``, ``Binomial``, ``Poisson``, ``Categorical``, and
``NegativeBinomial`` each derive their event term spec from their parameters
and take an ``event_spec`` declaration that names the event's component.

The parameters broadcast against one another, a categorical's along all but
its last axis, which indexes the categories. Scalar parameters give a scalar
draw, and parameters with more axes give one draw of independent coordinates
of the broadcast shape. Each family claims the mean, the variance, and the
covariance; the backend has no quantile function for them. A law over one
coordinate of finite support, a Bernoulli, binomial, or categorical one, also
claims the exact expectation, which enumerates the support.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .._dtype import _as_float_array, _promote_floats
from ..core._specs import OutputSpec
from ..core.constraints import (
    Constraint,
    boolean,
    integer_interval,
    non_negative_integer,
)
from ..custom_types import Array, ArrayLike
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsVariance,
    _capability_subclass,
)
from ._backend import TFPDistribution

__all__ = [
    "Bernoulli",
    "Binomial",
    "Categorical",
    "NegativeBinomial",
    "Poisson",
]

#: The capabilities the backend computes for every discrete family.
_MOMENTS = frozenset({SupportsMean, SupportsVariance, SupportsCovariance})


def _one_coordinate(*parameters: Any) -> bool:
    """Whether *parameters* broadcast to a single coordinate, the scalar shape."""
    return np.broadcast_shapes(*(np.shape(p) for p in parameters if p is not None)) == ()


def _finite_support_instance(cls: type, one_coordinate: bool) -> Any:
    """An instance of *cls* that claims the exact expectation when its law is over one coordinate.

    Enumerating the product support of several coordinates grows exponentially
    with their number, so a law over independent coordinates claims no exact
    expectation.
    """
    base = vars(cls).get("_capability_base", cls)
    claimed = (SupportsExpectation,) if one_coordinate else ()
    return object.__new__(_capability_subclass(base, claimed))


def _bernoulli_expectation(self: Bernoulli, f: Callable[[Any], Array]) -> Array:
    """The exact expectation of *f* over the two-point support ``{0, 1}``."""
    p = self._tfp_dist.probs_parameter()
    f0 = f(jnp.zeros((), dtype=self.dtype))
    f1 = f(jnp.ones((), dtype=self.dtype))
    return (1 - p) * f0 + p * f1


def _binomial_expectation(self: Binomial, f: Callable[[Any], Array]) -> Array:
    """The exact expectation of *f* over the support ``{0, ..., total_count}``."""
    support = jnp.arange(int(self._total_count) + 1, dtype=self.dtype)
    probs = jnp.exp(self._tfp_dist.log_prob(support))
    return jnp.einsum("n,n...->...", probs, jax.vmap(f)(support))


def _categorical_expectation(self: Categorical, f: Callable[[Any], Array]) -> Array:
    """The exact expectation of *f* over the categories ``{0, ..., k - 1}``."""
    probs = self._tfp_dist.probs_parameter()
    support = jnp.arange(self._num_categories(), dtype=self.dtype)
    return jnp.einsum("n,n...->...", probs, jax.vmap(f)(support))


class Bernoulli(TFPDistribution):
    """
    Bernoulli distribution.

    Parameters
    ----------
    component : str
        The component of the law's event.
    probs : array-like, optional
        Probability of a 1 outcome.  Exactly one of *probs* or *logits*
        must be provided.
    logits : array-like, optional
        Log-odds of a 1 outcome.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = _MOMENTS
    _capability_table: ClassVar = {SupportsExpectation: {"_expectation": _bernoulli_expectation}}

    def __new__(
        cls,
        component: str,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> Bernoulli:
        return _finite_support_instance(cls, _one_coordinate(probs, logits))

    def __init__(
        self,
        component: str,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("exactly one of probs or logits must be provided")
        if probs is not None:
            self._probs = _as_float_array(probs)
            self._logits = None
            backend = tfd.Bernoulli(probs=self._probs)
        else:
            self._logits = _as_float_array(logits)
            self._probs = None
            backend = tfd.Bernoulli(logits=self._logits)
        super().__init__(component, backend, label=label, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return boolean


class Binomial(TFPDistribution):
    """
    Binomial distribution.

    Parameters
    ----------
    component : str
        The component of the law's event.
    total_count : array-like
        Number of trials.
    probs : array-like, optional
        Probability of success per trial.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Log-odds of success per trial.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = _MOMENTS
    _capability_table: ClassVar = {SupportsExpectation: {"_expectation": _binomial_expectation}}

    def __new__(
        cls,
        component: str,
        total_count: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> Binomial:
        return _finite_support_instance(cls, _one_coordinate(total_count, probs, logits))

    def __init__(
        self,
        component: str,
        total_count: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("exactly one of probs or logits must be provided")
        if probs is not None:
            _, (self._total_count, self._probs) = _promote_floats(total_count, probs)
            self._logits = None
            backend = tfd.Binomial(total_count=self._total_count, probs=self._probs)
        else:
            _, (self._total_count, self._logits) = _promote_floats(total_count, logits)
            self._probs = None
            backend = tfd.Binomial(total_count=self._total_count, logits=self._logits)
        super().__init__(component, backend, label=label, event_spec=event_spec)

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

    def _event_support(self) -> Constraint:
        return integer_interval(0, self._total_count)


class Poisson(TFPDistribution):
    """
    Poisson distribution.

    Parameters
    ----------
    component : str
        The component of the law's event.
    rate : array-like
        Rate parameter (must be positive).
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *component* is not a string, or *event_spec* is not an
        :class:`~probpipe.OutputSpec` or exposes a record.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component or declares a type that one draw does not conform to.
    """

    _backend_capabilities = _MOMENTS

    def __init__(
        self,
        component: str,
        rate: ArrayLike,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        self._rate = _as_float_array(rate)
        backend = tfd.Poisson(rate=self._rate)
        super().__init__(component, backend, label=label, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def rate(self) -> Array:
        return self._rate

    # -- support ------------------------------------------------------------

    def _event_support(self) -> Constraint:
        return non_negative_integer


class Categorical(TFPDistribution):
    """
    Categorical distribution over ``k`` classes.

    Parameters
    ----------
    component : str
        The component of the law's event.
    probs : array-like, optional
        Probabilities for each category.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Unnormalized log-probabilities for each category.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = _MOMENTS
    _capability_table: ClassVar = {SupportsExpectation: {"_expectation": _categorical_expectation}}

    def __new__(
        cls,
        component: str,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> Categorical:
        parameters = probs if probs is not None else logits
        return _finite_support_instance(cls, len(np.shape(parameters)) <= 1)

    def __init__(
        self,
        component: str,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("exactly one of probs or logits must be provided")
        if probs is not None:
            self._probs = _as_float_array(probs)
            self._logits = None
            backend = tfd.Categorical(probs=self._probs)
        else:
            self._logits = _as_float_array(logits)
            self._probs = None
            backend = tfd.Categorical(logits=self._logits)
        super().__init__(component, backend, label=label, event_spec=event_spec)

    # -- convenient accessors -----------------------------------------------

    @property
    def probs(self) -> Array | None:
        return self._probs

    @property
    def logits(self) -> Array | None:
        return self._logits

    # -- support ------------------------------------------------------------

    def _num_categories(self) -> int:
        """Return the number of categories, the size of the parameters' last axis."""
        params = self._probs if self._probs is not None else self._logits
        return int(params.shape[-1])

    def _event_support(self) -> Constraint:
        return integer_interval(0, self._num_categories() - 1)


class NegativeBinomial(TFPDistribution):
    """
    Negative binomial distribution.

    Parameters
    ----------
    component : str
        The component of the law's event.
    total_count : array-like
        Number of successes before stopping.
    probs : array-like, optional
        Probability of success per trial.  Exactly one of *probs* or
        *logits* must be provided.
    logits : array-like, optional
        Log-odds of success per trial.
    label : str, optional
        The law's label, the family's class name by default.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the family completes, as ``OutputSpec(theta=NumericArraySpec((3,)))``.

    Raises
    ------
    TypeError
        If *event_spec* is not an :class:`~probpipe.OutputSpec` or exposes a
        record.
    ValueError
        If not exactly one of *probs* and *logits* is given, or *event_spec*
        declares a type that one draw does not conform to.
    """

    _backend_capabilities = _MOMENTS

    def __init__(
        self,
        component: str,
        total_count: ArrayLike,
        *,
        probs: ArrayLike | None = None,
        logits: ArrayLike | None = None,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError("exactly one of probs or logits must be provided")
        if probs is not None:
            _, (self._total_count, self._probs) = _promote_floats(total_count, probs)
            self._logits = None
            backend = tfd.NegativeBinomial(total_count=self._total_count, probs=self._probs)
        else:
            _, (self._total_count, self._logits) = _promote_floats(total_count, logits)
            self._probs = None
            backend = tfd.NegativeBinomial(total_count=self._total_count, logits=self._logits)
        super().__init__(component, backend, label=label, event_spec=event_spec)

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

    def _event_support(self) -> Constraint:
        return non_negative_integer
