"""Iterative distribution transformation abstractions.

Provides utilities for algorithms that transform a distribution over a
sequence of steps — incremental conditioning, tempering/annealing,
filtering, active learning, etc.

The central pattern is a **fold over distributions**: starting from an
initial distribution, a step function is applied repeatedly with
successive inputs, producing the batch of the distributions the fold visits.

Core API::

    from probpipe import iterate, with_conversion, with_resampling

    dists = iterate(step_fn, initial_dist, inputs)
    dists[-1]   # final distribution
    dists[0]    # initial distribution
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from typing import Any

from .._weights import Weights, weighted_choice
from ..distributions._batches import DistributionBatch
from ..distributions._conversion import converter_registry
from ..distributions._distribution import Distribution
from ..distributions._empirical import EmpiricalDistribution, _batch_form
from ..functions import function
from ..values import Function
from .provenance import Provenance

__all__ = [
    "iterate",
    "with_conversion",
    "with_resampling",
]


# ---------------------------------------------------------------------------
# iterate — the fold Function
# ---------------------------------------------------------------------------


#: The level of the batch ``iterate`` returns, named after the operation that mints it.
_ITERATE_LEVEL = "iterate"


@function
def iterate[S](
    step_fn: Callable[[Distribution, S], Distribution],
    initial: Distribution,
    inputs: Iterable[S],
    *,
    callback: Callable[[int, Distribution], Any] | None = None,
) -> DistributionBatch:
    """Fold a step function over inputs, returning the batch of the laws the fold visits.

    Starting from *initial*, applies ``step_fn(dist, inp)`` for each element
    of *inputs*. The laws visited, *initial* first, are the elements of a
    ``DistributionBatch`` on one level named ``iterate``, so they share one
    event declaration. Each element is a view of the law the fold produced at
    that step, and its provenance records the batch and that law.

    A step's law that has no provenance receives a record of the step, whose
    parent is the previous law and whose metadata holds the step's index; a
    law that already has provenance keeps it.

    Parameters
    ----------
    step_fn : callable
        ``(Distribution, S) -> Distribution``: a plain function, a
        :class:`Function`, or a bound method.
    initial : Distribution
        The starting distribution.
    inputs : Iterable[S]
        Sequence of inputs to pass to the step function.
    callback : callable or None
        Called as ``callback(i, dist)`` after each step, where *i* is
        the step index and *dist* is the newly produced distribution.
        If it returns exactly ``False``, iteration stops early.

    Returns
    -------
    DistributionBatch
        The laws ``[initial, dist_1, dist_2, ...]`` on the level ``iterate``.

    Raises
    ------
    TypeError
        If a step returns something other than a ``Distribution``, or a law
        whose event declaration does not match *initial*'s.
    """
    laws: list[Distribution] = [initial]
    current = initial

    for i, inp in enumerate(inputs):
        result = step_fn(current, inp)
        if not isinstance(result, Distribution):
            raise TypeError(
                f"Step function at index {i} returned "
                f"{type(result).__name__}, expected Distribution."
            )

        # Auto-attach provenance if not already set
        if result.provenance is None:
            # write-once guard: re-sourcing an already-sourced result raises
            with contextlib.suppress(RuntimeError):
                result.with_provenance(
                    Provenance.create(
                        "iterate",
                        parents=[current],
                        metadata={"step": i},
                    )
                )

        laws.append(result)
        current = result

        if callback is not None:
            cont = callback(i, result)
            if cont is False:
                break

    return DistributionBatch(_ITERATE_LEVEL, laws, _ITERATE_LEVEL)


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


def _step_fn_name(step_fn: Callable) -> str:
    """Extract a human-readable name from a step function."""
    if isinstance(step_fn, Function):
        return step_fn._label
    return getattr(step_fn, "__name__", type(step_fn).__name__)


def with_conversion(
    step_fn: Callable,
    target_type: type,
    **convert_kwargs: Any,
) -> Function:
    """Wrap a step function to convert its output after each step.

    After calling *step_fn*, converts the resulting distribution to
    *target_type* through the converter registry, which returns a law that
    already satisfies the target as it is. The pre-conversion distribution
    is the converted distribution's provenance parent, which the registry
    records with the converter it selected.

    This is useful when the step function produces samples (e.g.,
    MCMC output) but the next iteration needs a parametric
    distribution as input.

    The returned wrapper is a :class:`Function`, so it appears
    as a node in the ProbPipe workflow DAG.

    Parameters
    ----------
    step_fn : callable
        The underlying step function.
    target_type : type
        Distribution type to convert to (e.g., ``MultivariateNormal``).
        Can also be a protocol (e.g., ``SupportsLogProb``).
    **convert_kwargs
        The registry's controls ``method`` and ``exact_only`` and the
        converter's options, passed to ``converter_registry.convert``.

    Returns
    -------
    Function
        A new step function with the same call signature.
    """
    inner_name = _step_fn_name(step_fn)

    def _with_conversion_impl(dist: Distribution, inp: Any) -> Distribution:
        result = step_fn(dist, inp)
        return converter_registry.convert(result, target_type, **convert_kwargs)

    return Function(
        fn=_with_conversion_impl,
        label=f"with_conversion({inner_name}, {target_type.__name__})",
    )


def with_resampling(
    step_fn: Callable,
    *,
    ess_threshold: float = 0.5,
    seed: int = 0,
) -> Function:
    """Wrap a step function to resample when particle weights degenerate.

    After calling *step_fn*, if the result is an
    :class:`~probpipe.EmpiricalDistribution` with
    ``ESS / N < ess_threshold``, performs multinomial resampling to
    produce equally-weighted particles.

    When resampling occurs, the raw result from
    ``wrapper.apply(...)`` carries ``"resample"`` provenance whose
    metadata stores the pre-resampling ``"ess"`` and ``"ess_ratio"``.
    A normal ``wrapper(...)`` call returns an independent Function result
    whose provenance describes the wrapper invocation; it does not retain
    that implementation-level resampling metadata.

    The returned wrapper is a :class:`Function`, so it appears
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
    Function
        A new step function with the same call signature.

    Notes
    -----
    This API is likely to evolve as typical use cases become clearer.
    A future direction is a ``SupportsResampling`` protocol that would
    decouple this combinator from the concrete
    :class:`~probpipe.EmpiricalDistribution` type.
    """
    import jax

    inner_name = _step_fn_name(step_fn)
    call_count = 0

    def _with_resampling_impl(dist: Distribution, inp: Any) -> Distribution:
        nonlocal call_count

        out_dist = step_fn(dist, inp)

        if isinstance(out_dist, EmpiricalDistribution):
            n = out_dist.num_atoms
            ess = float(Weights(n=n, weights=out_dist.weights).effective_sample_size)
            ess_ratio = ess / n

            if ess_ratio < ess_threshold:
                key = jax.random.PRNGKey(seed + call_count)
                call_count += 1
                indices = weighted_choice(key, n, weights=out_dist.weights, shape=(n,))
                # The drawn atoms, equally weighted, on the one level resampling mints.
                atoms = _batch_form(
                    out_dist.label,
                    out_dist._atoms_at(indices),
                    "resample",
                    out_dist.event_spec.spec,
                )
                resampled = EmpiricalDistribution(
                    out_dist.label, atoms, event_spec=out_dist.event_spec
                )
                resampled.with_provenance(
                    Provenance.create(
                        "resample",
                        parents=[out_dist],
                        metadata={"ess": ess, "ess_ratio": ess_ratio},
                    )
                )
                return resampled

        return out_dist

    return Function(
        fn=_with_resampling_impl,
        label=f"with_resampling({inner_name})",
    )
