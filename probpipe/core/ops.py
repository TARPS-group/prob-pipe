"""Built-in operations for distribution computation.

This module provides two layers of API:

1. **Plain functions** — ``sample(dist, ...)``, ``mean(dist)``, etc.
   These check protocol compliance and delegate to the distribution's
   private protocol methods.  Use these for direct computation.

2. **WorkflowFunction wrappers** — accessible via :data:`wf_sample`,
   :data:`wf_mean`, etc.  These participate in Prefect orchestration
   and broadcasting.  Use these when you want to pass a distribution
   through a ``WorkflowFunction`` pipeline.

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

# ---------------------------------------------------------------------------
# Plain-function API
# ---------------------------------------------------------------------------


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
            f"(missing _sample method)"
        )
    return dist.sample(key=key, sample_shape=sample_shape)


def log_prob(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the normalized log-density at *value*."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob"
        )
    return dist._log_prob(value)


def prob(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the density at *value* (computed as ``exp(log_prob)``)."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support prob "
            f"(missing _log_prob method)"
        )
    return jnp.exp(dist._log_prob(value))


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


def unnormalized_prob(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    """Evaluate the unnormalized density at *value* (computed as ``exp(unnormalized_log_prob)``)."""
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return jnp.exp(dist._unnormalized_log_prob(value))


def mean(dist: Any) -> Any:
    """Compute E[X].

    Uses exact moments if the distribution supports :class:`SupportsMean`,
    otherwise falls back to Monte Carlo via :class:`SupportsExpectation`.
    """
    if isinstance(dist, SupportsMean):
        return dist._mean()
    if isinstance(dist, SupportsExpectation):
        return dist.expectation(lambda x: x, return_dist=False)
    raise TypeError(
        f"{type(dist).__name__} does not support mean "
        f"(implements neither SupportsMean nor SupportsExpectation)"
    )


def variance(dist: Any) -> Any:
    """Compute Var[X].

    Uses exact variance if available, otherwise MC fallback.
    """
    if isinstance(dist, SupportsVariance):
        return dist._variance()
    if isinstance(dist, SupportsExpectation):
        mu = mean(dist)
        return dist.expectation(
            lambda x: (x - mu) ** 2, return_dist=False,
        )
    raise TypeError(
        f"{type(dist).__name__} does not support variance "
        f"(implements neither SupportsVariance nor SupportsExpectation)"
    )


def cov(dist: Any) -> Array:
    """Compute the covariance matrix.

    Uses exact covariance if available, otherwise MC fallback.
    """
    if isinstance(dist, SupportsCovariance):
        return dist._cov()
    if isinstance(dist, SupportsExpectation):
        mu = mean(dist)

        def _outer_diff(x):
            d = jnp.ravel(x) - jnp.ravel(mu)
            return jnp.outer(d, d)

        return dist.expectation(_outer_diff, return_dist=False)
    raise TypeError(
        f"{type(dist).__name__} does not support covariance "
        f"(implements neither SupportsCovariance nor SupportsExpectation)"
    )


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
    return dist.expectation(
        f, key=key, num_evaluations=num_evaluations, return_dist=return_dist,
    )


def condition_on(
    dist: SupportsConditioning,
    observed: Any = None,
    /,
    **kwargs: Any,
) -> Any:
    """Condition a joint distribution on observed values."""
    if not isinstance(dist, SupportsConditioning):
        raise TypeError(
            f"{type(dist).__name__} does not support conditioning"
        )
    return dist._condition_on(observed, **kwargs)


# ---------------------------------------------------------------------------
# Single source of truth for op names
# ---------------------------------------------------------------------------

# All plain-function op names.  WorkflowFunction wrappers, __all__, and
# module __getattr__ are derived from this list automatically.
_OP_NAMES: list[str] = [
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
]

# Map name → function object (populated from module globals)
_OP_FUNCS: dict[str, Any] = {name: globals()[name] for name in _OP_NAMES}


# ---------------------------------------------------------------------------
# WorkflowFunction wrappers (lazy to avoid circular imports)
# ---------------------------------------------------------------------------

_wf_ops: dict | None = None


def _make_wf_ops() -> dict:
    """Create WorkflowFunction instances wrapping the plain functions."""
    from .node import WorkflowFunction

    return {
        name: WorkflowFunction(func=func, name=name)
        for name, func in _OP_FUNCS.items()
    }


def __getattr__(name: str):
    """Lazy access to WorkflowFunction wrappers (``wf_*`` names)."""
    global _wf_ops
    if name.startswith("wf_"):
        op_name = name.removeprefix("wf_")
        if op_name in _OP_FUNCS:
            if _wf_ops is None:
                _wf_ops = _make_wf_ops()
            return _wf_ops[op_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    _OP_NAMES
    + [f"wf_{name}" for name in _OP_NAMES]
)
