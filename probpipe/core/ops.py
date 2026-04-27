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
    SupportsExpectedDistribution,
    SupportsLogProb,
    SupportsMean,
    SupportsRandomLogProb,
    SupportsRandomUnnormalizedLogProb,
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
    "expected_distribution",
    "random_log_prob",
    "random_unnormalized_log_prob",
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
    return jnp.exp(dist._log_prob(value))


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
    return jnp.exp(dist._unnormalized_log_prob(value))


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
def expected_distribution(dist: SupportsExpectedDistribution) -> Distribution:
    """Compute the expected distribution of a random measure.

    For a :class:`~probpipe.core._random_measures.RandomMeasure[T]`
    ``M``, returns the marginalised ``Distribution[T]``
    ``D̄(A) = ∫ D(A) dM(D)``.  Requires the distribution to implement
    :class:`~probpipe.core.protocols.SupportsExpectedDistribution`.
    """
    if not isinstance(dist, SupportsExpectedDistribution):
        raise TypeError(
            f"{type(dist).__name__} does not support expected_distribution "
            f"(does not implement SupportsExpectedDistribution)"
        )
    return dist._expected_distribution()


@workflow_function
def random_log_prob(dist: SupportsRandomLogProb) -> Any:
    """Return the random (normalized) log-density of a random measure.

    For a ``RandomMeasure[T]`` ``M``, returns the random function
    ``x ↦ log D(x)`` where ``D ~ M``, as a
    :class:`~probpipe.core._random_functions.RandomFunction`.
    """
    if not isinstance(dist, SupportsRandomLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_log_prob "
            f"(does not implement SupportsRandomLogProb)"
        )
    return dist._random_log_prob()


@workflow_function
def random_unnormalized_log_prob(dist: SupportsRandomUnnormalizedLogProb) -> Any:
    """Return the random unnormalized log-density of a random measure.

    For a ``RandomMeasure[T]`` ``M``, returns the random function
    ``x ↦ log D̃(x)`` where ``D̃`` is the unnormalized density of a
    draw ``D ~ M``, as a
    :class:`~probpipe.core._random_functions.RandomFunction`.
    """
    if not isinstance(dist, SupportsRandomUnnormalizedLogProb):
        raise TypeError(
            f"{type(dist).__name__} does not support random_unnormalized_log_prob "
            f"(does not implement SupportsRandomUnnormalizedLogProb)"
        )
    return dist._random_unnormalized_log_prob()


def _split_data_kwargs(
    dist: Distribution,
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate named data kwargs from inference kwargs.

    Uses the distribution's ``fields`` as the sole signal:
    any kwarg whose name matches a component name is data (conditioning
    target); everything else is an inference parameter.

    Returns ``(data_kwargs, inference_kwargs)``.
    """
    comp_names = frozenset(
        dist.fields if hasattr(dist, 'fields') else ()
    )
    data_kwargs = {k: v for k, v in kwargs.items() if k in comp_names}
    inference_kwargs = {k: v for k, v in kwargs.items() if k not in comp_names}
    return data_kwargs, inference_kwargs


@workflow_function
def condition_on(
    dist: Distribution,
    observed: Any = None,
    *,
    method: str | None = None,
    **kwargs: Any,
) -> Distribution:
    """Condition a distribution on observed values.

    Observed data can be passed positionally or as named keyword
    arguments::

        # Positional (backward compatible):
        condition_on(model, y_obs)

        # Named data kwargs — bundled into Record(X=..., y=...):
        condition_on(model, X=bootstrap["X"], y=bootstrap["y"],
                     n_broadcast_samples=16)

    When named data kwargs are distribution views from the same parent,
    the workflow function broadcasting machinery samples the parent once
    and distributes the fields, preserving joint correlation.

    Dispatch priority:

    1. **Explicit override** — ``method="tfp_nuts"`` (or any registered
       name) routes directly to the named inference method.
    2. **Exact conditioning** — if *dist* implements
       ``SupportsConditioning``, its ``_condition_on`` is called for a
       closed-form result (e.g., conjugate updates, joint marginalization).
    3. **Registry auto-select** — the inference method registry picks
       the highest-priority feasible algorithm (NUTS, HMC, RWMH, etc.).

    Parameters
    ----------
    dist : Distribution
        Distribution or model to condition.  Need not implement
        ``SupportsConditioning`` — the registry provides inference
        methods for common model types.
    observed : Any
        Observed values to condition on.
    method : str or None
        If provided, use the named inference method from the registry
        instead of the default dispatch.
    **kwargs
        Inference parameters (e.g., ``num_results``, ``num_warmup``,
        ``random_seed``) and/or named data kwargs.  Any kwarg whose
        name matches a distribution component name is treated as
        observed data; everything else is an inference parameter.
    """
    from ..inference import inference_method_registry
    from .record import Record

    # Separate data kwargs (names matching fields) from
    # inference kwargs (everything else like num_results, num_warmup).
    data_kwargs, inference_kwargs = _split_data_kwargs(dist, kwargs)

    # Explicit method override → always use the registry
    if method is not None:
        if data_kwargs:
            if observed is not None:
                raise ValueError(
                    "Cannot provide both positional `observed` and named "
                    f"data kwargs ({', '.join(data_kwargs)})"
                )
            observed = Record(data_kwargs)
        return inference_method_registry.execute(
            dist, observed, method=method, **inference_kwargs
        )

    # Exact conditioning (conjugate updates, joint marginalization, etc.)
    # All kwargs pass through to _condition_on — it handles its own
    # validation (e.g., ProductDistribution raises KeyError on unknown names).
    if isinstance(dist, SupportsConditioning):
        return dist._condition_on(observed, **data_kwargs, **inference_kwargs)

    # Registry auto-selects the best approximate inference algorithm.
    # Data kwargs are bundled into observed as a Record object.
    if data_kwargs:
        if observed is not None:
            raise ValueError(
                "Cannot provide both positional `observed` and named "
                f"data kwargs ({', '.join(data_kwargs)})"
            )
        observed = Record(data_kwargs)
    return inference_method_registry.execute(dist, observed, **inference_kwargs)


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
