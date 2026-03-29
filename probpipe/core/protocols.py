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

    SupportsExpectation       standalone; E[f(X)] computation

    SupportsMean              standalone; exact _mean()
    SupportsVariance          standalone; exact _variance()
    SupportsCovariance        standalone; exact _cov()

    SupportsUnnormalizedLogProb
        ↑ inherits
    SupportsLogProb           provides _unnormalized_log_prob via _log_prob

The moment protocols (SupportsMean, SupportsVariance, SupportsCovariance)
are independent of SupportsExpectation.  The ops layer falls back to
MC estimation via SupportsExpectation when the exact protocol is absent.
Concrete classes that want default MC implementations can use the
``@compute_expectation`` decorator on their ``_mean``/``_variance``/
``_cov`` methods — but this is opt-in, not required by the protocol.

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
class SupportsMean(Protocol):
    """Distribution with an exact mean via ``_mean()``.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_mean`` implementation.
    """

    def _mean(self) -> Any: ...


@runtime_checkable
class SupportsVariance(Protocol):
    """Distribution with an exact variance via ``_variance()``.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_variance`` implementation.
    """

    def _variance(self) -> Any: ...


@runtime_checkable
class SupportsCovariance(Protocol):
    """Distribution with an exact covariance via ``_cov()``.

    Independent of :class:`SupportsExpectation`.  The ops layer falls
    back to MC estimation via ``SupportsExpectation`` when this protocol
    is absent.  Concrete classes that want the MC default can apply
    ``@compute_expectation`` to their ``_cov`` implementation.
    """

    def _cov(self) -> Any: ...


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


# ---------------------------------------------------------------------------
# Conditionable components (probabilistic models)
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsConditionableComponents(SupportsNamedComponents, SupportsConditioning, Protocol):
    """Model with named components, some of which can/must be conditioned on.

    Combines :class:`SupportsNamedComponents` (component access) with
    :class:`SupportsConditioning` (conditioning via ``_condition_on``).
    Adds metadata about which components accept or require observed data.
    """

    @property
    def conditionable_components(self) -> dict[str, bool]:
        """Map component name to whether conditioning is required.

        Returns a dict where keys are component names and values indicate
        whether conditioning on that component is required (``True``) or
        optional (``False``).
        """
        ...

    @property
    def required_observations(self) -> tuple[str, ...]:
        """Component names that must be conditioned on."""
        ...


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
    "SupportsConditionableComponents",
]
