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
    their own implementation strategy (TFP batched sampling, index
    resampling, vmap over a local single-draw helper, etc.).

    Does NOT extend :class:`SupportsExpectation` — not all samplable
    distributions support array-valued expectations (e.g., random functions).
    Classes that support both should inherit both protocols.

    Return-type convention
    ----------------------
    The shape of the return value depends on whether the distribution
    emits structured samples and whether the caller asks for a batch:

    =====================  =======================  =========================================
    Distribution kind      ``sample_shape == ()``   ``sample_shape == (S1, S2, ...)``
    =====================  =======================  =========================================
    Numeric (raw array)    ``Array[*event_shape]``  ``Array[*sample_shape, *event_shape]``
    ``RecordDistribution`` ``Record`` / ``NumericRecord``  ``NumericRecordArray(batch_shape=sample_shape)``
    =====================  =======================  =========================================

    To draw a single sample, call ``_sample(key, ())``. Implementations
    that find it clearer to factor out a single-draw helper should
    define it as a private method (e.g. ``_one_bootstrap``) and have
    ``_sample`` dispatch on ``sample_shape`` internally.
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
            See class-level "Return-type convention".
        """
        ...


# ---------------------------------------------------------------------------
# Density evaluation
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    """Distribution with an unnormalized log-density.

    Provides ``_unnormalized_log_prob(value)``.
    """

    def _unnormalized_log_prob(self, value: Any) -> Array: ...


@runtime_checkable
class SupportsLogProb(SupportsUnnormalizedLogProb, Protocol):
    """Distribution with a (normalized) log-density.

    Extends :class:`SupportsUnnormalizedLogProb` because any distribution
    with a normalized density also has an unnormalized one (they coincide).
    The base :class:`~probpipe.core.distribution.Distribution` class
    provides ``_unnormalized_log_prob`` defaulting to ``_log_prob``.

    """

    def _log_prob(self, value: Any) -> Array: ...

    def _unnormalized_log_prob(self, value: Any) -> Array:
        """Default: delegates to ``_log_prob``."""
        return self._log_prob(value)


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
    """Distribution that supports exact (closed-form) conditioning.

    Reserved for distributions where ``_condition_on`` computes an
    **exact** posterior — e.g., conjugate updates or joint distribution
    marginalization.  When ``condition_on(dist, observed)`` is called
    and *dist* implements this protocol, the exact path is used
    directly; otherwise the inference method registry selects an
    approximate algorithm (NUTS, RWMH, etc.).

    Probabilistic models that require MCMC or variational inference
    should **not** implement this protocol — let the registry handle
    algorithm selection instead.
    """

    def _condition_on(self, observed: Any, /, **kwargs: Any) -> Any: ...


def protocols_supported_by_all(
    leaves: list, candidates: tuple[type, ...],
) -> tuple[type, ...]:
    """Return the subset of *candidates* that every leaf satisfies.

    Used by dynamic-protocol factories (``ProductDistribution``,
    ``SequentialJointDistribution``, ``TransformedDistribution``,
    ``_RecordDistributionView``, ``FlattenedView``) when building a
    cached subclass whose protocol bases track the capabilities of the
    underlying distribution(s). Pass in the leaves to check and the
    tuple of ``SupportsFoo`` protocols to test against; get back the
    protocols that are satisfied by every leaf, in the given order.
    """
    return tuple(p for p in candidates if all(isinstance(l, p) for l in leaves))


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
    "protocols_supported_by_all",
]
