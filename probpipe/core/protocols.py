"""Protocol definitions for distribution capabilities.

Each protocol declares a capability that a distribution may support.
Operations in :mod:`probpipe.core.ops` check these protocols via
``isinstance`` to determine what computations are valid.

All protocols are ``@runtime_checkable`` so that external distribution
types (TFP, scipy) can satisfy them via structural subtyping without
inheriting from ProbPipe base classes.

**Naming convention:** Protocol methods use an underscore prefix
(``_sample``, ``_log_prob``, ``_mean``, …) to distinguish the
primitive implementation from the public workflow-function API in
:mod:`probpipe.core.ops`.

**Orchestration hints:** ``SupportsSampling`` defines class-attribute
defaults for orchestration preferences.  Distribution subclasses
override as needed.

Protocol hierarchy
------------------

::

    SupportsSampling          standalone; single _sample protocol method

    SupportsExpectation
        ↑ inherits
    SupportsMean, SupportsVariance, SupportsCovariance

    SupportsUnnormalizedLogProb
        ↑ inherits
    SupportsLogProb           provides _unnormalized_log_prob via _log_prob

"""

from __future__ import annotations

import functools
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable

import jax.numpy as jnp

from ..custom_types import Array, PRNGKey


# ---------------------------------------------------------------------------
# Decorator for default moment implementations via expectation
# ---------------------------------------------------------------------------

def compute_expectation(method):
    """Decorator providing a default moment implementation via ``expectation``.

    The decorated method should return the function ``f`` to pass to
    ``self.expectation(f, return_dist=False)``.  Any setup (e.g.
    computing the mean before computing the variance) can be done in
    the method body before the ``return``.

    Example::

        @compute_expectation
        def _mean(self):
            return lambda x: x
    """

    @functools.wraps(method)
    def wrapper(self):
        f = method(self)
        return self._expectation(f, return_dist=False)

    return wrapper


# ---------------------------------------------------------------------------
# Expectation & sampling
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsExpectation(Protocol):
    """Distribution that can compute ``E[f(X)]``."""

    def _expectation(self, f: Any, *, key: Any, num_evaluations: Any,
                     return_dist: Any) -> Any: ...


@runtime_checkable
class SupportsSampling(Protocol):
    """Distribution that can produce samples via ``_sample(key, sample_shape)``.

    Only requires ``_sample(key, sample_shape)``; concrete classes choose
    their own implementation strategy (vmap over ``_sample_one``, TFP
    batched sampling, index resampling, etc.).

    Does NOT extend :class:`SupportsExpectation` — not all samplable
    distributions support array-valued expectations (e.g., random functions).
    Classes that support both should inherit both protocols.
    """

    _sampling_cost: ClassVar[str]  # "low", "medium", "high"
    _preferred_orchestration: ClassVar[str | None]  # "task", "flow", or None

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw sample(s) from this distribution.

        Parameters
        ----------
        key : PRNGKey
            JAX PRNG key.
        sample_shape : tuple of int
            Shape prefix for independent draws.

        Returns
        -------
        Any
            A single sample when ``sample_shape == ()``, or a batched
            representation when ``sample_shape`` is non-empty.
        """
        ...


# ---------------------------------------------------------------------------
# Density evaluation
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    """Distribution with an unnormalized log-density.

    Provides ``_unnormalized_log_prob(value)`` and
    ``_unnormalized_prob(value)`` (computed as ``exp`` of the log form).
    """

    def _unnormalized_log_prob(self, value: Any) -> Array: ...

    def _unnormalized_prob(self, value: Any) -> Array:
        """Default: ``exp(_unnormalized_log_prob(value))``."""
        return jnp.exp(self._unnormalized_log_prob(value))


@runtime_checkable
class SupportsLogProb(SupportsUnnormalizedLogProb, Protocol):
    """Distribution with a (normalized) log-density.

    Extends :class:`SupportsUnnormalizedLogProb` because any distribution
    with a normalized density also has an unnormalized one (they coincide).
    The base :class:`~probpipe.core.distribution.Distribution` class
    provides ``_unnormalized_log_prob`` defaulting to ``_log_prob``.

    Also provides ``_prob`` computed as ``exp(_log_prob)``.
    """

    def _log_prob(self, value: Any) -> Array: ...

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """Default: delegates to ``_log_prob``."""
        return self._log_prob(value)

    def _unnormalized_prob(self, value: Any) -> Array:
        """Default: ``exp(_log_prob(value))``."""
        return jnp.exp(self._log_prob(value))

    def _prob(self, value: Any) -> Array:
        """Default: ``exp(_log_prob(value))``."""
        return jnp.exp(self._log_prob(value))


# ---------------------------------------------------------------------------
# Moment protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsMean(SupportsExpectation, Protocol):
    """Distribution with a mean via ``_mean()``.

    Extends :class:`SupportsExpectation`.  The default implementation
    computes ``E[x]`` via ``expectation``; subclasses with exact
    moments should override.
    """

    @compute_expectation
    def _mean(self):
        return lambda x: x


@runtime_checkable
class SupportsVariance(SupportsExpectation, Protocol):
    """Distribution with a variance via ``_variance()``.

    Extends :class:`SupportsExpectation`.  The default implementation
    computes ``E[(x - mean)^2]`` via ``expectation``; subclasses with
    exact moments should override.
    """

    @compute_expectation
    def _variance(self):
        mu = self._expectation(lambda x: x, return_dist=False)
        return lambda x: (x - mu) ** 2


@runtime_checkable
class SupportsCovariance(SupportsExpectation, Protocol):
    """Distribution with a covariance via ``_cov()``.

    Extends :class:`SupportsExpectation`.  The default implementation
    computes ``E[(x - mean)(x - mean)^T]`` via ``expectation``;
    subclasses with exact moments should override.
    """

    @compute_expectation
    def _cov(self):
        mu = self._expectation(lambda x: x, return_dist=False)

        def _outer_diff(x):
            d = jnp.ravel(x) - jnp.ravel(mu)
            return jnp.outer(d, d)

        return _outer_diff


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsConditioning(Protocol):
    """Distribution that supports conditioning on observed values."""

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Named components (joint distributions)
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsNamedComponents(Protocol):
    """Distribution with named sub-components (e.g., joint distributions)."""

    @property
    def component_names(self) -> tuple: ...

    def __getitem__(self, key: Any) -> Any: ...


__all__ = [
    "compute_expectation",
    "SupportsExpectation",
    "SupportsSampling",
    "SupportsUnnormalizedLogProb",
    "SupportsLogProb",
    "SupportsMean",
    "SupportsVariance",
    "SupportsCovariance",
    "SupportsConditioning",
    "SupportsNamedComponents",
]
