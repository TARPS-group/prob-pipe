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

**Orchestration hints:** Protocols define class-attribute defaults
for orchestration preferences.  Distribution subclasses override
as needed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from ..custom_types import Array, PRNGKey


# ---------------------------------------------------------------------------
# Sampling & expectation
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsSampling(Protocol):
    """Distribution that can produce samples via ``_sample(key)``."""

    _sampling_cost: ClassVar[str]  # "low", "medium", "high"
    _preferred_orchestration: ClassVar[str | None]  # "task", "flow", or None

    def _sample(self, key: PRNGKey) -> Any: ...


@runtime_checkable
class SupportsExpectation(Protocol):
    """Distribution that can compute ``E[f(X)]``."""

    def expectation(self, f: Any, *, key: Any, num_evaluations: Any,
                    return_dist: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Density evaluation
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsLogProb(Protocol):
    """Distribution with a (normalized) log-density ``_log_prob(value)``."""

    def _log_prob(self, value: Any) -> Array: ...


@runtime_checkable
class SupportsProb(Protocol):
    """Distribution with a density ``_prob(value)``."""

    def _prob(self, value: Any) -> Array: ...


@runtime_checkable
class SupportsUnnormalizedLogProb(Protocol):
    """Distribution with an unnormalized log-density."""

    def _unnormalized_log_prob(self, value: Any) -> Array: ...


# ---------------------------------------------------------------------------
# Moment protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SupportsMean(Protocol):
    """Distribution with an exact (non-MC) mean via ``_mean()``."""

    def _mean(self) -> Any: ...


@runtime_checkable
class SupportsVariance(Protocol):
    """Distribution with an exact (non-MC) variance via ``_variance()``."""

    def _variance(self) -> Any: ...


@runtime_checkable
class SupportsCovariance(Protocol):
    """Distribution with an exact (non-MC) covariance via ``_cov()``."""

    def _cov(self) -> Array: ...


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
    "SupportsSampling",
    "SupportsExpectation",
    "SupportsLogProb",
    "SupportsProb",
    "SupportsUnnormalizedLogProb",
    "SupportsMean",
    "SupportsVariance",
    "SupportsCovariance",
    "SupportsConditioning",
    "SupportsNamedComponents",
]
