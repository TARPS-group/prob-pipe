"""Iterative distribution transformation abstractions.

Provides core types and utilities for algorithms that transform a
distribution over a sequence of steps — incremental conditioning,
tempering/annealing, filtering, active learning, etc.

The central pattern is a **fold over distributions**: starting from an
initial distribution, a step function is applied repeatedly with
successive inputs, producing a trajectory of distributions.

Core API::

    from probpipe import iterate, StepResult, TransitionTrace

    trace = iterate(step_fn, initial_dist, inputs)
    trace.final          # final distribution
    trace.distributions  # full trajectory (including initial)
    trace.infos          # per-step auxiliary info dicts
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .distribution import Distribution
from .node import WorkflowFunction, workflow_function
from .provenance import Provenance

__all__ = [
    "StepResult",
    "TransitionTrace",
    "DistributionTransition",
    "iterate",
    "with_approximation",
    "with_resampling",
]


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepResult[T]:
    """Result of one transition step.

    Pairs a distribution with an optional info dict carrying auxiliary
    data such as diagnostics, ESS, or log-normalizing constants.
    """

    distribution: Distribution[T]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionTrace[T]:
    """Trajectory of distributions produced by iterating a transition.

    Stores the initial distribution and a list of :class:`StepResult`
    objects, one per step.  Provides convenient accessors for the full
    sequence of distributions, the final distribution, and per-step
    auxiliary info.

    Supports iteration (``for result in trace:``) and indexing
    (``trace[i]``).
    """

    initial: Distribution[T]
    results: tuple[StepResult[T], ...]

    # -- Accessors ----------------------------------------------------------

    @property
    def distributions(self) -> list[Distribution[T]]:
        """All distributions in the trajectory, including the initial.

        Length is ``len(self) + 1``.
        """
        return [self.initial] + [r.distribution for r in self.results]

    @property
    def final(self) -> Distribution[T]:
        """The last distribution in the trajectory.

        Returns the initial distribution if no steps were taken.
        """
        if self.results:
            return self.results[-1].distribution
        return self.initial

    @property
    def infos(self) -> list[dict[str, Any]]:
        """Per-step info dicts (one per step, excludes initial)."""
        return [r.info for r in self.results]

    def info_values(self, key: str) -> list[Any]:
        """Extract a single key from each step's info dict.

        Raises :class:`KeyError` if any step is missing the key.
        """
        return [r.info[key] for r in self.results]

    # -- Container protocol -------------------------------------------------

    def __len__(self) -> int:
        """Number of steps (not counting the initial distribution)."""
        return len(self.results)

    def __getitem__(self, index: int) -> StepResult[T]:
        """Index into the results list."""
        return self.results[index]

    def __iter__(self) -> Iterator[StepResult[T]]:
        """Iterate over step results."""
        return iter(self.results)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DistributionTransition[T, S](Protocol):
    """Protocol for a single step of an iterative distribution transformation.

    Any callable with signature
    ``(Distribution[T], S) -> Distribution[T] | StepResult[T]``
    satisfies this protocol.  The protocol exists for documentation and
    optional type-checking; :func:`iterate` accepts any matching callable.
    """

    def __call__(
        self, dist: Distribution[T], input: S, /
    ) -> Distribution[T] | StepResult[T]: ...


# ---------------------------------------------------------------------------
# iterate — the fold WorkflowFunction
# ---------------------------------------------------------------------------


def _normalize_step_result(result: Any, step_context: int | str) -> StepResult:
    """Wrap a bare Distribution return in StepResult; validate type."""
    if isinstance(result, StepResult):
        return result
    if isinstance(result, Distribution):
        return StepResult(distribution=result)
    raise TypeError(
        f"Step function {step_context} returned "
        f"{type(result).__name__}, expected Distribution or StepResult."
    )


def _maybe_attach_provenance(
    result: StepResult, prev_dist: Distribution, step_index: int
) -> None:
    """Attach provenance to the result distribution if not already set."""
    dist = result.distribution
    if dist.source is None:
        try:
            dist.with_source(
                Provenance(
                    "iterate",
                    parents=(prev_dist,),
                    metadata={"step": step_index},
                )
            )
        except RuntimeError:
            pass  # write-once guard; should not happen if source is None


@workflow_function
def iterate(
    step_fn: Callable,
    initial: Distribution,
    inputs: Iterable,
    *,
    callback: Callable | None = None,
) -> TransitionTrace:
    """Fold a step function over inputs, accumulating a distribution trajectory.

    Starting from *initial*, applies ``step_fn(dist, inp)`` for each
    element of *inputs*, collecting the results into a
    :class:`TransitionTrace`.

    Parameters
    ----------
    step_fn : callable
        ``(Distribution[T], S) -> Distribution[T] | StepResult[T]``.
        Any callable matching this signature — plain functions,
        :class:`WorkflowFunction` instances, or bound methods.
    initial : Distribution[T]
        The starting distribution.
    inputs : Iterable[S]
        Sequence of inputs to pass to the step function.
    callback : callable or None
        Called as ``callback(i, step_result)`` after each step.
        If it returns exactly ``False``, iteration stops early.

    Returns
    -------
    TransitionTrace[T]
        The full trajectory including initial and all step results.
    """
    results: list[StepResult] = []
    current = initial

    for i, inp in enumerate(inputs):
        raw = step_fn(current, inp)
        result = _normalize_step_result(raw, f"at index {i}")
        _maybe_attach_provenance(result, current, i)
        results.append(result)
        current = result.distribution

        if callback is not None:
            cont = callback(i, result)
            if cont is False:
                break

    return TransitionTrace(initial=initial, results=tuple(results))


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


def _step_fn_name(step_fn: Callable) -> str:
    """Extract a human-readable name from a step function."""
    if isinstance(step_fn, WorkflowFunction):
        return step_fn._name
    return getattr(step_fn, "__name__", type(step_fn).__name__)


def with_approximation(
    step_fn: Callable,
    target_type: type,
    **convert_kwargs: Any,
) -> WorkflowFunction:
    """Wrap a step function to convert its output after each step.

    After calling *step_fn*, converts the resulting distribution to
    *target_type* using ProbPipe's standard ``from_distribution``
    operation (which dispatches through the converter registry).
    The pre-conversion distribution is stored in
    ``info["pre_approximation"]``.

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

    def _with_approximation_impl(dist: Distribution, inp: Any) -> StepResult:
        from .ops import from_distribution

        raw = step_fn(dist, inp)
        result = _normalize_step_result(
            raw, f"(inside with_approximation wrapping {inner_name})"
        )
        pre = result.distribution
        approximated = from_distribution(pre, target_type, **convert_kwargs)
        info = {**result.info, "pre_approximation": pre}
        return StepResult(distribution=approximated, info=info)

    return WorkflowFunction(
        func=_with_approximation_impl,
        name=f"with_approximation({inner_name}, {target_type.__name__})",
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

    Records ``"ess"``, ``"ess_ratio"``, and ``"resampled"`` in the
    step info dict.

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

    def _with_resampling_impl(dist: Distribution, inp: Any) -> StepResult:
        nonlocal call_count
        from .distribution import EmpiricalDistribution

        raw = step_fn(dist, inp)
        result = _normalize_step_result(
            raw, f"(inside with_resampling wrapping {inner_name})"
        )
        out_dist = result.distribution
        info = dict(result.info)

        if isinstance(out_dist, EmpiricalDistribution):
            n = out_dist.n
            ess = float(out_dist.effective_sample_size)
            ess_ratio = ess / n
            info["ess"] = ess
            info["ess_ratio"] = ess_ratio

            if ess_ratio < ess_threshold:
                # Multinomial resampling to uniform weights
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
                info["resampled"] = True
                return StepResult(distribution=resampled, info=info)
            else:
                info["resampled"] = False
                return StepResult(distribution=out_dist, info=info)

        # Not an EmpiricalDistribution — pass through unchanged
        return StepResult(distribution=out_dist, info=info)

    return WorkflowFunction(
        func=_with_resampling_impl,
        name=f"with_resampling({inner_name})",
    )
