"""Built-in operations for distribution computation.

Each public function (``sample``, ``mean``, ``log_prob``, …) is a
:class:`~probpipe.core.node.WorkflowFunction` created via the
``@workflow_function`` decorator.  This means every call automatically
participates in broadcasting and Prefect orchestration when a
distribution argument is passed where a concrete value is expected.

Usage::

    from probpipe import sample, mean, log_prob, condition_on

    dist = Normal(loc=0.0, scale=1.0)
    s = sample(dist, key=jax.random.PRNGKey(0), sample_shape=(100,))
    m = mean(dist)
    lp = log_prob(dist, jnp.array(1.5))
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..custom_types import Array, PRNGKey
from .._utils import _auto_key
from .distribution import Distribution
from .node import workflow_function
from .protocols import (
    SupportsConditioning,
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsUnnormalizedLogProb,
    SupportsVariance,
)

__all__ = [
    "sample",
    "log_prob",
    "prob",
    "unnormalized_log_prob",
    "unnormalized_prob",
    "mean",
    "variance",
    "cov",
    "expectation",
    "condition_on",
    "from_distribution",
]


# ---------------------------------------------------------------------------
# Public API — each function is a WorkflowFunction via @workflow_function
# ---------------------------------------------------------------------------


@workflow_function
def sample(
    dist: SupportsSampling,
    *,
    key: PRNGKey | None = None,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draw samples from a distribution.

    Parameters
    ----------
    dist : SupportsSampling
        Distribution to sample from.
    key : PRNGKey, optional
        JAX PRNG key.  Auto-generated if ``None``.
    sample_shape : tuple of int
        Shape prefix for independent draws.
    """
    if not isinstance(dist, SupportsSampling):
        raise TypeError(
            f"{type(dist).__name__} does not support sampling "
            f"(does not implement SupportsSampling)"
        )
    if key is None:
        key = _auto_key()
    return dist._sample(key, sample_shape)


@workflow_function
def log_prob(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the normalized log-density at *value*."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob"
        )
    return dist._log_prob(value)


@workflow_function
def prob(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the density at *value* (``exp(log_prob)``)."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support prob "
            f"(missing _log_prob method)"
        )
    return dist._prob(value)


@workflow_function
def unnormalized_log_prob(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    """Evaluate the unnormalized log-density at *value*."""
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_log_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return dist._unnormalized_log_prob(value)


@workflow_function
def unnormalized_prob(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    """Evaluate the unnormalized density at *value* (``exp(unnormalized_log_prob)``)."""
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return dist._unnormalized_prob(value)


@workflow_function
def mean(dist: SupportsMean) -> Any:
    """Compute E[X].

    Requires the distribution to implement :class:`SupportsMean`.
    """
    if not isinstance(dist, SupportsMean):
        raise TypeError(
            f"{type(dist).__name__} does not support mean "
            f"(does not implement SupportsMean)"
        )
    return dist._mean()


@workflow_function
def variance(dist: SupportsVariance) -> Any:
    """Compute Var[X].

    Requires the distribution to implement :class:`SupportsVariance`.
    """
    if not isinstance(dist, SupportsVariance):
        raise TypeError(
            f"{type(dist).__name__} does not support variance "
            f"(does not implement SupportsVariance)"
        )
    return dist._variance()


@workflow_function
def cov(dist: SupportsCovariance) -> Array:
    """Compute the covariance matrix.

    Requires the distribution to implement :class:`SupportsCovariance`.
    """
    if not isinstance(dist, SupportsCovariance):
        raise TypeError(
            f"{type(dist).__name__} does not support covariance "
            f"(does not implement SupportsCovariance)"
        )
    return dist._cov()


@workflow_function
def expectation(
    dist: SupportsExpectation,
    f: Any,
    *,
    key: PRNGKey | None = None,
    num_evaluations: int | None = None,
    return_dist: bool | None = None,
) -> Any:
    """Compute E[f(X)] where X ~ dist."""
    if not isinstance(dist, SupportsExpectation):
        raise TypeError(
            f"{type(dist).__name__} does not support expectation"
        )
    return dist._expectation(
        f, key=key, num_evaluations=num_evaluations, return_dist=return_dist,
    )


@workflow_function
def condition_on(
    dist: SupportsConditioning,
    observed: Any = None,
    **kwargs: Any,
) -> Distribution:
    """Condition a joint distribution on observed values."""
    if not isinstance(dist, SupportsConditioning):
        raise TypeError(
            f"{type(dist).__name__} does not support conditioning"
        )
    return dist._condition_on(observed, **kwargs)


@workflow_function
def from_distribution(
    source: Distribution,
    target_type: type,
    *,
    key: Any | None = None,
    check_support: bool = True,
    **kwargs: Any,
) -> Any:
    """Convert *source* into an instance of *target_type*.

    Delegates to the global converter registry.

    Parameters
    ----------
    source : Distribution
        Source distribution to convert.
    target_type : type
        The target distribution class.
    key : PRNGKey, optional
        JAX PRNG key for sampling-based conversion.
    check_support : bool
        If ``True`` (default), verify the supports are compatible.
    **kwargs
        Additional keyword arguments passed to the converter.
    """
    from ..converters import converter_registry
    if key is None:
        key = _auto_key()
    return converter_registry.convert(
        source, target_type, key=key, check_support=check_support, **kwargs
    )
