"""Iterative distribution transformation abstractions.

Provides utilities for algorithms that transform a distribution over a
sequence of steps — incremental conditioning, tempering/annealing,
filtering, active learning, etc.

The central pattern is a **fold over distributions**: starting from an
initial distribution, a step function is applied repeatedly with
successive inputs, producing a sequence of distributions.

Core API::

    from probpipe import iterate, with_conversion, with_resampling

    dists = iterate(step_fn, initial_dist, inputs)
    dists[-1]   # final distribution
    dists[0]    # initial distribution
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .distribution import Distribution
from .node import WorkflowFunction, workflow_function
from .provenance import Provenance

__all__ = [
    "iterate",
    "with_conversion",
    "with_resampling",
]


# ---------------------------------------------------------------------------
# iterate — the fold WorkflowFunction
# ---------------------------------------------------------------------------


@workflow_function
def iterate[T, S](
    step_fn: Callable[[Distribution[T], S], Distribution[T]],
    initial: Distribution[T],
    inputs: Iterable[S],
    *,
    callback: Callable[[int, Distribution[T]], Any] | None = None,
) -> list[Distribution[T]]:
    """Fold a step function over inputs, accumulating a distribution sequence.

    Starting from *initial*, applies ``step_fn(dist, inp)`` for each
    element of *inputs*, collecting the resulting distributions into a
    list.  The returned list includes the initial distribution at
    index 0.

    Provenance is automatically attached to each output distribution
    (linking it to the previous distribution) unless the step function
    has already set provenance.

    Parameters
    ----------
    step_fn : callable
        ``(Distribution[T], S) -> Distribution[T]``.
        Any callable matching this signature — plain functions,
        :class:`WorkflowFunction` instances, or bound methods.
    initial : Distribution[T]
        The starting distribution.
    inputs : Iterable[S]
        Sequence of inputs to pass to the step function.
    callback : callable or None
        Called as ``callback(i, dist)`` after each step, where *i* is
        the step index and *dist* is the newly produced distribution.
        If it returns exactly ``False``, iteration stops early.

    Returns
    -------
    list[Distribution[T]]
        The full sequence: ``[initial, dist_1, dist_2, ...]``.
    """
    dists: list[Distribution[T]] = [initial]
    current = initial

    for i, inp in enumerate(inputs):
        result = step_fn(current, inp)
        if not isinstance(result, Distribution):
            raise TypeError(
                f"Step function at index {i} returned "
                f"{type(result).__name__}, expected Distribution."
            )

        # Auto-attach provenance if not already set
        if result.source is None:
            try:
                result.with_source(
                    Provenance(
                        "iterate",
                        parents=(current,),
                        metadata={"step": i},
                    )
                )
            except RuntimeError:
                pass  # write-once guard

        dists.append(result)
        current = result

        if callback is not None:
            cont = callback(i, result)
            if cont is False:
                break

    return dists


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


def _step_fn_name(step_fn: Callable) -> str:
    """Extract a human-readable name from a step function."""
    if isinstance(step_fn, WorkflowFunction):
        return step_fn._name
    return getattr(step_fn, "__name__", type(step_fn).__name__)


def with_conversion(
    step_fn: Callable,
    target_type: type,
    **convert_kwargs: Any,
) -> WorkflowFunction:
    """Wrap a step function to convert its output after each step.

    After calling *step_fn*, converts the resulting distribution to
    *target_type* using ProbPipe's standard ``from_distribution``
    operation (which dispatches through the converter registry).
    The pre-conversion distribution is accessible via the converted
    distribution's provenance parents (set by the converter).

    This is useful when the step function produces samples (e.g.,
    MCMC output) but the next iteration needs a parametric
    distribution as input.

    The returned wrapper is a :class:`WorkflowFunction`, so it appears
    as a node in the ProbPipe workflow DAG.

    Parameters
    ----------
    step_fn : callable
        The underlying step function.
    target_type : type
        Distribution type to convert to (e.g., ``MultivariateNormal``).
        Can also be a protocol (e.g., ``SupportsLogProb``).
    **convert_kwargs
        Extra keyword arguments passed to ``from_distribution``.

    Returns
    -------
    WorkflowFunction
        A new step function with the same call signature.
    """
    inner_name = _step_fn_name(step_fn)

    def _with_conversion_impl(dist: Distribution, inp: Any) -> Distribution:
        from .ops import from_distribution

        result = step_fn(dist, inp)
        return from_distribution(result, target_type, **convert_kwargs)

    return WorkflowFunction(
        func=_with_conversion_impl,
        name=f"with_conversion({inner_name}, {target_type.__name__})",
    )


def with_resampling(
    step_fn: Callable,
    *,
    ess_threshold: float = 0.5,
    seed: int = 0,
) -> WorkflowFunction:
    """Wrap a step function to resample when particle weights degenerate.

    After calling *step_fn*, if the result is an
    :class:`~probpipe.core.distribution.EmpiricalDistribution` with
    ``ESS / N < ess_threshold``, performs multinomial resampling to
    produce equally-weighted particles.

    The pre-resampling ESS is stored in provenance metadata of the
    resampled distribution (``dist.source.metadata["ess"]``) since
    this information would otherwise be lost after resampling to
    uniform weights.

    The returned wrapper is a :class:`WorkflowFunction`, so it appears
    as a node in the ProbPipe workflow DAG.

    Parameters
    ----------
    step_fn : callable
        The underlying step function.
    ess_threshold : float
        Resample when ``ESS / N`` drops below this value (default 0.5).
    seed : int
        Base random seed; combined with a call counter for
        deterministic reproducibility.

    Returns
    -------
    WorkflowFunction
        A new step function with the same call signature.

    Notes
    -----
    This API is likely to evolve as typical use cases become clearer.
    A future direction is a ``SupportsResampling`` protocol that would
    decouple this combinator from the concrete
    :class:`~probpipe.core.distribution.EmpiricalDistribution` type.
    """
    import jax
    import jax.numpy as jnp

    inner_name = _step_fn_name(step_fn)
    call_count = 0

    def _with_resampling_impl(dist: Distribution, inp: Any) -> Distribution:
        nonlocal call_count
        from .distribution import EmpiricalDistribution

        out_dist = step_fn(dist, inp)

        if isinstance(out_dist, EmpiricalDistribution):
            n = out_dist.n
            ess = float(out_dist.effective_sample_size)
            ess_ratio = ess / n

            if ess_ratio < ess_threshold:
                key = jax.random.PRNGKey(seed + call_count)
                call_count += 1
                indices = out_dist._w.choice(key, shape=(n,))
                new_samples = out_dist.samples[indices]
                resampled = EmpiricalDistribution(new_samples)
                resampled.with_source(
                    Provenance(
                        "resample",
                        parents=(out_dist,),
                        metadata={"ess": ess, "ess_ratio": ess_ratio},
                    )
                )
                return resampled

        return out_dist

    return WorkflowFunction(
        func=_with_resampling_impl,
        name=f"with_resampling({inner_name})",
    )
