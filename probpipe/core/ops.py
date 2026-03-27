"""Built-in operations for distribution computation.

Each public function (``sample``, ``mean``, ``log_prob``, …) is a
lightweight positional-arg wrapper around an internal
:class:`~probpipe.core.node.WorkflowFunction`.  This means every call
automatically participates in broadcasting and Prefect orchestration
when a distribution argument is passed where a concrete value is
expected.

Usage::

    from probpipe import sample, mean, log_prob, condition_on

    dist = Normal(loc=0.0, scale=1.0)
    s = sample(dist, key=jax.random.PRNGKey(0), sample_shape=(100,))
    m = mean(dist)
    lp = log_prob(dist, jnp.array(1.5))
"""

from __future__ import annotations

import functools
import inspect
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
# Implementation functions (private)
#
# Each function contains the actual protocol-check + delegation logic.
# They are wrapped by WorkflowFunction instances (for broadcasting /
# Prefect) and then by lightweight positional-arg adapters that form
# the public API.
# ---------------------------------------------------------------------------


def _sample_impl(
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


def _log_prob_impl(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the normalized log-density at *value*."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support log_prob"
        )
    return dist._log_prob(value)


def _prob_impl(dist: SupportsLogProb, value: Any) -> Array:
    """Evaluate the density at *value* (``exp(log_prob)``)."""
    if not isinstance(dist, SupportsLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support prob "
            f"(missing _log_prob method)"
        )
    return dist._prob(value)


def _unnormalized_log_prob_impl(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    """Evaluate the unnormalized log-density at *value*."""
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_log_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return dist._unnormalized_log_prob(value)


def _unnormalized_prob_impl(
    dist: SupportsUnnormalizedLogProb, value: Any,
) -> Array:
    """Evaluate the unnormalized density at *value* (``exp(unnormalized_log_prob)``)."""
    if not isinstance(dist, SupportsUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support unnormalized_prob "
            f"(missing _unnormalized_log_prob method)"
        )
    return dist._unnormalized_prob(value)


def _mean_impl(dist: SupportsExpectation) -> Any:
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


def _variance_impl(dist: SupportsExpectation) -> Any:
    """Compute Var[X].

    Uses exact variance if available, otherwise MC fallback.
    """
    if isinstance(dist, SupportsVariance):
        return dist._variance()
    if isinstance(dist, SupportsExpectation):
        mu = _mean_impl(dist)
        return dist.expectation(
            lambda x: (x - mu) ** 2, return_dist=False,
        )
    raise TypeError(
        f"{type(dist).__name__} does not support variance "
        f"(implements neither SupportsVariance nor SupportsExpectation)"
    )


def _cov_impl(dist: SupportsExpectation) -> Array:
    """Compute the covariance matrix.

    Uses exact covariance if available, otherwise MC fallback.
    """
    if isinstance(dist, SupportsCovariance):
        return dist._cov()
    if isinstance(dist, SupportsExpectation):
        mu = _mean_impl(dist)

        def _outer_diff(x):
            d = jnp.ravel(x) - jnp.ravel(mu)
            return jnp.outer(d, d)

        return dist.expectation(_outer_diff, return_dist=False)
    raise TypeError(
        f"{type(dist).__name__} does not support covariance "
        f"(implements neither SupportsCovariance nor SupportsExpectation)"
    )


def _expectation_impl(
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


def _condition_on_impl(
    dist: SupportsConditioning,
    observed: Any = None,
    **kwargs: Any,
) -> Any:
    """Condition a joint distribution on observed values."""
    if not isinstance(dist, SupportsConditioning):
        raise TypeError(
            f"{type(dist).__name__} does not support conditioning"
        )
    return dist._condition_on(observed, **kwargs)


# ---------------------------------------------------------------------------
# Op registry — single source of truth
# ---------------------------------------------------------------------------

# Maps public name → implementation function.
_OP_REGISTRY: dict[str, Any] = {
    "sample": _sample_impl,
    "log_prob": _log_prob_impl,
    "prob": _prob_impl,
    "unnormalized_log_prob": _unnormalized_log_prob_impl,
    "unnormalized_prob": _unnormalized_prob_impl,
    "mean": _mean_impl,
    "variance": _variance_impl,
    "cov": _cov_impl,
    "expectation": _expectation_impl,
    "condition_on": _condition_on_impl,
}


# ---------------------------------------------------------------------------
# WorkflowFunction wrappers (lazy to avoid circular imports)
# ---------------------------------------------------------------------------

_wf_ops: dict | None = None


def _ensure_wf_ops() -> dict:
    """Lazily create WorkflowFunction instances wrapping the impl functions."""
    global _wf_ops
    if _wf_ops is None:
        from .node import WorkflowFunction

        _wf_ops = {
            name: WorkflowFunction(func=impl, name=name)
            for name, impl in _OP_REGISTRY.items()
        }
    return _wf_ops


# ---------------------------------------------------------------------------
# Public API — positional-arg wrappers routing through WorkflowFunction
# ---------------------------------------------------------------------------

def _make_op(name: str, impl):
    """Build a public wrapper that accepts positional args and routes
    through the corresponding :class:`WorkflowFunction`.

    The wrapper preserves the implementation's signature, docstring,
    and type annotations so that IDE tooling and ``help()`` work as
    expected.
    """
    sig = inspect.signature(impl)

    # Detect if the impl has a **kwargs parameter.
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    @functools.wraps(impl)
    def wrapper(*args, **kwargs):
        wf_ops = _ensure_wf_ops()
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call_kwargs = dict(bound.arguments)
        # If the impl has **kwargs, sig.bind nests them under the
        # parameter name (e.g. "kwargs": {...}).  Unpack so that
        # WorkflowFunction receives them as top-level keyword args.
        if has_var_keyword:
            for p in sig.parameters.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD and p.name in call_kwargs:
                    extra = call_kwargs.pop(p.name)
                    call_kwargs.update(extra)
        return wf_ops[name](**call_kwargs)

    return wrapper


# Generate the public functions and populate __all__.
__all__: list[str] = []

for _name, _impl in _OP_REGISTRY.items():
    globals()[_name] = _make_op(_name, _impl)
    __all__.append(_name)
