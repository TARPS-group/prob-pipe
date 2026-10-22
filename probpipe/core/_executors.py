"""Executors, the registry that chooses among them, and the function that runs evaluations on them (V.9).

An *executor* decides how a set of evaluation points runs: one at a time, on
threads, in worker processes, compiled by JAX, or on a user's cluster. It
never decides what they compute, so changing the executor changes no result
beyond the last bit of rounding. Executors are the dispatch methods of
:data:`executor_registry`, chosen as inference methods are: by name, with the
controls ``executor`` and ``executor_options``, or by automatic selection,
which tries the registered executors from the highest priority down, skips the
opt-in ones, and runs the first that can run the set.

Provides:
  - :class:`EvaluationExecutor` – the base class of an executor.
  - :data:`executor_registry` – the registry of executors.
  - :func:`evaluate_at_points` – the one function through which code asks for
    the evaluations of a callable at many arguments.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing
import os
import threading
import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np

from .._messages import unknown_names
from ._array_backend import _is_numeric_leaf
from ._call_context import (
    _EvaluationLog,
    _EvaluationSetRecord,
    _ExecutionChoice,
    _jax_settings_snapshot,
    _merge_noted_into,
    current_call_context,
)
from ._catalog import registry_catalog
from ._dispatch import (
    Feasibility,
    ResolutionError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ._evaluation_points import (
    EvaluationChunk,
    EvaluationError,
    EvaluationOutcomes,
    EvaluationPoints,
    SharedPayload,
    _assembled_outcomes,
    _checked_stacked_output,
    _holds_only_values,
    _InvalidPointOutput,
    _output_path,
    _point_failure,
    _warn_lossy_casts,
)
from ._execution_claims import NO_CLAIMS, ExecutionClaims, FalseTraceableClaimError
from ._specs import OutputSpec
from .config import _checked_count

__all__ = ["EvaluationExecutor", "evaluate_at_points", "executor_registry"]


class EvaluationExecutor(UnaryDispatchMethod):
    """The base class of an executor: a named way to run a set of evaluation points.

    A subclass sets ``name`` and ``description``, sets ``priority`` when
    automatic selection may choose it, and implements :meth:`check` and
    :meth:`execute`. Its options are the keyword-only parameters of
    :meth:`execute`, which :meth:`check` takes too; ProbPipe refuses an option
    that :meth:`execute` does not take before it calls either. An executor
    passes ``probpipe.testing.check_executor`` when it runs every evaluation
    point through its chunks' ``run``, returns the outcomes in order with what
    the evaluation points noted, and changes no argument.

    Attributes
    ----------
    name : str
        The name the control ``executor`` selects it by.
    exact : bool
        ``True``: an executor returns the requested result itself.
    priority : int or None
        Its rank in automatic selection, higher first; ``None``, the default,
        makes it opt-in, run only when named.
    description : str
        One line, which the registry catalog shows.

    Examples
    --------
    An executor that runs the chunks of a set on a pool the user gives::

        class PoolExecutor(EvaluationExecutor):
            name = "my_pool"
            description = "Run chunks of evaluation points on my pool."

            def check(self, evaluation_points, /, *, num_chunks=8):
                return evaluation_points.shipping_check()

            def execute(self, evaluation_points, /, *, num_chunks=8):
                shared_payload = evaluation_points.shared_payload()
                chunks = evaluation_points.chunks(num_chunks=num_chunks)
                return EvaluationOutcomes.merge(
                    my_pool.map(lambda chunk: chunk.run(shared_payload), chunks)
                )

        executor_registry.register(PoolExecutor())
    """

    @property
    def exact(self) -> bool:
        """``True``: an executor returns the requested result itself, whatever it runs on."""
        return True

    def supported_types(self) -> tuple[type, ...]:
        """``(EvaluationPoints,)``: an executor admits every set of evaluation points."""
        return (EvaluationPoints,)

    @abstractmethod
    def check(self, evaluation_points: EvaluationPoints, /, **executor_options: Any) -> Feasibility:
        """Whether the executor can run *evaluation_points*, read from their claims and the environment.

        It runs none of the evaluation points.

        Parameters
        ----------
        evaluation_points : EvaluationPoints
            The set of evaluation points.
        **executor_options : Any
            The executor's options.

        Returns
        -------
        Feasibility
            Feasible when the executor can run the set; otherwise infeasible,
            with the reason in its description.
        """

    @abstractmethod
    def execute(
        self, evaluation_points: EvaluationPoints, /, **executor_options: Any
    ) -> EvaluationOutcomes:
        """Run every evaluation point and return their outcomes, in the order of the evaluation points.

        Parameters
        ----------
        evaluation_points : EvaluationPoints
            The set of evaluation points.
        **executor_options : Any
            The executor's options.

        Returns
        -------
        EvaluationOutcomes
            One outcome per evaluation point, starting at point index 0.
        """


class _ExecutorRegistry(UnaryDispatchRegistry[EvaluationExecutor]):
    """The registry of executors, whose messages call its methods executors."""

    def _unknown(self, name: str) -> str:
        """The message for a request of the executor *name*, which is not registered."""
        return unknown_names("executor", [name], sorted(self._by_name))


executor_registry: _ExecutorRegistry = _ExecutorRegistry(
    name="executors",
    description="Executors that run sets of evaluation points: in order, on threads, in "
    "worker processes, or compiled by JAX",
)
"""The registry of executors, from which a call chooses the one that runs its evaluations.

It holds ``jax``, ``sequential``, ``thread``, and ``process``; register an
executor of your own with ``executor_registry.register``.
"""


def evaluate_at_points(
    fn: Callable[..., Any],
    /,
    *,
    varying: Mapping[str, Any],
    shared: Mapping[str, Any],
    origin: str,
    output_spec: OutputSpec | None,
    stacked: bool = False,
) -> Any:
    """Evaluate *fn* at many arguments, on the executor the running call chooses.

    Each evaluation point calls ``fn(**shared, **arguments)``, where
    ``arguments`` holds each varying argument's values at that evaluation
    point. With concrete arguments the evaluation points form one
    :class:`EvaluationPoints`, which runs under the running call's settings,
    and a draw made inside one takes its random key from its place in the
    workflow, so changing the executor changes no result. A ``Function`` that
    claims ``traceable=True`` is mapped by ``jax`` whatever executor the call
    names. With traced arguments, as inside a function that JAX traces, *fn*
    is mapped inline with ``jax.vmap``.

    Parameters
    ----------
    fn : callable
        The callable. A ``Function``'s execution claims say how it may run.
    varying : Mapping of str to Any
        Each argument that differs between evaluation points, its values
        stacked on a leading axis with one row per evaluation point.
    shared : Mapping of str to Any
        Each argument that every evaluation point receives unchanged.
    origin : str
        The name of the callable or of the site that asks for the
        evaluations, which messages use.
    output_spec : OutputSpec or None
        The callable's output spec, which each output is checked against;
        ``None`` checks nothing.
    stacked : bool, optional
        Whether to return the outputs as one tree stacked on a leading axis of
        evaluation points, in place of one output per evaluation point.
        ``jax`` returns its mapped output whole. The outputs of any other
        executor are stacked leaf by leaf when every leaf of every output is
        numeric and the outputs share one tree structure, and are otherwise
        returned as a one-dimensional object array that holds each output.
        ``False`` by default.

    Returns
    -------
    tuple or Any
        Each evaluation point's output, in point order, or under *stacked* the
        outputs stacked as one tree.

    Raises
    ------
    EvaluationError
        If an evaluation point failed, naming every failed one.
    ResolutionError
        If the executor named for the callable cannot run its evaluation
        points, or no registered executor can.
    ValueError
        If the varying arguments do not have one row per evaluation point, or
        the chosen executor does not take one of its options.
    RuntimeError
        If it is called outside the ProbPipe call that asks for the
        evaluations, such as inside a function's body, where no place in the
        workflow gives the evaluation points their random keys.

    Examples
    --------
    A route of a ProbPipe call evaluates a simulator at 200 parameter values,
    stacked in ``thetas``, against the observed data ``y``::

        simulations = evaluate_at_points(
            simulate,
            varying={"theta": thetas},
            shared={"data": y},
            origin="simulate",
            output_spec=simulate.output_spec,
        )
    """
    if _holds_tracer((varying, shared)):
        return _mapped_inline(fn, varying, shared, stacked=stacked)
    evaluation_points = EvaluationPoints(
        fn,
        varying=varying,
        shared=shared,
        origin=origin,
        output_spec=output_spec,
        claims=_execution_claims_of(fn),
    )
    return _evaluated_outputs(evaluation_points, stacked=stacked)


def _evaluated_outputs(
    evaluation_points: EvaluationPoints,
    *,
    stacked: bool = False,
    execution_choice: _ExecutionChoice | None = None,
) -> Any:
    """Each evaluation point's output, or under *stacked* the outputs stacked, run on the chosen executor.

    The choice follows V.9's rules. A traceable callable is compiled by
    ``jax`` whatever executor the call names, and receives none of the named
    options. Otherwise an executor named for this callable runs the set, or
    raises when it cannot; an inherited executor that cannot run the set
    leaves it to automatic selection without its options, and warns once per
    call; and with no executor named, automatic selection runs the set, with
    the options the call's own ``Function`` gave.

    The call's evaluation log records the set, and what its evaluation points
    noted merges into the call's accumulators.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of evaluation points, built under the running call.
    stacked : bool, optional
        Whether to return the outputs stacked, as :func:`evaluate_at_points`
        states. ``jax`` then returns its mapped output whole, never cut into
        rows.
    execution_choice : _ExecutionChoice or None, optional
        The choice to follow in place of the running call's; ``None`` follows
        the call's.

    Returns
    -------
    tuple or Any
        Each evaluation point's output, in point order, or the stacked
        outputs.

    Raises
    ------
    ResolutionError
        If the executor named for this callable is not registered or cannot
        run the set, or no executor can.
    ValueError
        If the chosen executor does not take one of its options.
    EvaluationError
        If an evaluation point failed, naming every failed one.
    """
    executor, executor_options = _chosen_executor_closing_on_error(
        evaluation_points, execution_choice
    )
    if stacked and isinstance(executor, _JaxExecutor):
        return _mapped_outputs_or_error(evaluation_points)
    evaluation_outcomes = _run_on_executor(evaluation_points, executor, executor_options)
    return _outputs_of(evaluation_points, evaluation_outcomes, stacked=stacked)


def _evaluated_outcomes(
    evaluation_points: EvaluationPoints, *, execution_choice: _ExecutionChoice | None = None
) -> EvaluationOutcomes:
    """The outcomes of *evaluation_points*, run on the executor :func:`_evaluated_outputs` chooses.

    A requester that decides itself what a failure means reads the outcomes,
    failures included, in place of the outputs.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of evaluation points, built under the running call.
    execution_choice : _ExecutionChoice or None, optional
        The choice to follow in place of the running call's; ``None`` follows
        the call's.

    Returns
    -------
    EvaluationOutcomes
        One outcome per evaluation point, failures included.
    """
    executor, executor_options = _chosen_executor_closing_on_error(
        evaluation_points, execution_choice
    )
    return _run_on_executor(evaluation_points, executor, executor_options)


def _chosen_executor_closing_on_error(
    evaluation_points: EvaluationPoints, execution_choice: _ExecutionChoice | None
) -> tuple[EvaluationExecutor, dict[str, Any]]:
    """The executor that runs *evaluation_points*, which release what they hold when none is chosen."""
    call_context = current_call_context()
    try:
        return _chosen_executor(
            evaluation_points,
            call_context.execution_choice if execution_choice is None else execution_choice,
            call_context.evaluation_log,
        )
    except BaseException:
        evaluation_points._point_scopes.close()
        raise


def _outputs_of(
    evaluation_points: EvaluationPoints, evaluation_outcomes: EvaluationOutcomes, *, stacked: bool
) -> Any:
    """The outputs in *evaluation_outcomes*, stacked under *stacked*, or the error naming its failures.

    A failed evaluation point raises :class:`EvaluationError`, chained to the
    first failure's error.
    """
    if evaluation_outcomes.failures:
        failures = tuple(evaluation_outcomes.failures.values())
        first_error = min(failures, key=lambda failure: failure.point_index).error
        raise EvaluationError(
            failures,
            evaluation_outcomes.outputs,
            origin=evaluation_points.origin,
            point_noun=evaluation_points.point_noun,
        ) from first_error
    if stacked:
        return _stacked_outputs(evaluation_outcomes.outputs)
    return evaluation_outcomes.outputs


def _run_on_executor(
    evaluation_points: EvaluationPoints,
    executor: EvaluationExecutor,
    executor_options: Mapping[str, Any],
) -> EvaluationOutcomes:
    """The outcomes of *evaluation_points* run on *executor*, under the running call.

    It checks that the executor returned one outcome per evaluation point,
    accepts the draws that runs in other processes claimed, merges what the
    evaluation points noted into the call's accumulators, records the set in
    the call's evaluation log, and warns about casts that lost precision.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of evaluation points, built under the running call.
    executor : EvaluationExecutor
        The executor that runs it.
    executor_options : Mapping of str to Any
        The options *executor* receives.

    Returns
    -------
    EvaluationOutcomes
        One outcome per evaluation point, failures included.

    Raises
    ------
    RuntimeError
        If the executor did not return one outcome per evaluation point,
        starting at point index 0.
    """
    call_context = current_call_context()
    try:
        evaluation_outcomes = executor.execute(evaluation_points, **executor_options)
        _check_outcomes_cover(evaluation_outcomes, evaluation_points, executor.name)
        evaluation_points._point_scopes.accept_claim_reports(evaluation_outcomes._claim_reports)
    finally:
        evaluation_points._point_scopes.close()
    _merge_noted_into(
        call_context, evaluation_outcomes._noted_num_samples, evaluation_outcomes._noted_set_records
    )
    _record_set_run(
        evaluation_points,
        executor.name,
        executor_options,
        failed_point_indices=tuple(sorted(evaluation_outcomes.failures)),
        lossy_casts=evaluation_outcomes._lossy_casts,
    )
    return evaluation_outcomes


def _mapped_outputs_or_error(evaluation_points: EvaluationPoints) -> Any:
    """The outputs of a set that ``jax`` maps, kept whole, recorded as :func:`_run_on_executor` records a run.

    A map that raised, or whose output does not conform to the output spec,
    fails every evaluation point, and the call raises :class:`EvaluationError`.
    """
    try:
        stacked_output, lossy_casts = _mapped_outputs(evaluation_points)
    except _MappingFailure as mapping_failure:
        evaluation_points._point_scopes.close()
        failures = mapping_failure.point_failures(evaluation_points)
        _record_set_run(
            evaluation_points,
            "jax",
            {},
            failed_point_indices=tuple(range(len(evaluation_points))),
            lossy_casts=frozenset(),
        )
        raise EvaluationError(
            failures,
            (None,) * len(evaluation_points),
            origin=evaluation_points.origin,
            point_noun=evaluation_points.point_noun,
        ) from mapping_failure.error
    evaluation_points._point_scopes.close()
    _record_set_run(evaluation_points, "jax", {}, failed_point_indices=(), lossy_casts=lossy_casts)
    return stacked_output


def _record_set_run(
    evaluation_points: EvaluationPoints,
    executor_name: str,
    executor_options: Mapping[str, Any],
    *,
    failed_point_indices: tuple[int, ...],
    lossy_casts: frozenset[tuple[str, str]],
) -> None:
    """Record a run of *evaluation_points* in the call's evaluation log, and warn about lossy casts."""
    call_context = current_call_context()
    if call_context.evaluation_log is not None:
        call_context.evaluation_log.record(
            _EvaluationSetRecord(
                origin=evaluation_points.origin,
                set_ordinal=evaluation_points._set_ordinal,
                executor_name=executor_name,
                executor_options=dict(executor_options),
                num_evaluation_points=len(evaluation_points),
                failed_point_indices=failed_point_indices,
            )
        )
    _warn_lossy_casts(
        lossy_casts, origin=evaluation_points.origin, evaluation_log=call_context.evaluation_log
    )


# ---------------------------------------------------------------------------
# The built-in executors
# ---------------------------------------------------------------------------


class _JaxExecutor(EvaluationExecutor):
    """Compile a traceable callable over every evaluation point with one ``jax.vmap``."""

    @property
    def name(self) -> str:
        return "jax"

    @property
    def priority(self) -> int | None:
        return 20

    @property
    def description(self) -> str:
        return "Map a traceable callable over every evaluation point with jax.vmap."

    def check(self, evaluation_points: EvaluationPoints, /) -> Feasibility:
        if not evaluation_points.traceable:
            untraceable_reason = evaluation_points._untraceable_reason
            return Feasibility(
                feasible=False,
                description=(
                    f"{evaluation_points.origin!r} is not known to be traceable by JAX"
                    if untraceable_reason is None
                    else f"JAX cannot trace {evaluation_points.origin!r}: {untraceable_reason}"
                ),
            )
        output_spec = evaluation_points.output_spec
        if output_spec is not None and not _holds_only_values(output_spec.term_spec):
            return Feasibility(
                feasible=False,
                description=(
                    f"{evaluation_points.origin!r} returns a distribution, a function, or a "
                    "batch, which jax.vmap cannot return"
                ),
            )
        return Feasibility(feasible=True)

    def execute(self, evaluation_points: EvaluationPoints, /) -> EvaluationOutcomes:
        try:
            stacked_output, lossy_casts = _mapped_outputs(evaluation_points)
        except _MappingFailure as mapping_failure:
            return _assembled_outcomes(0, mapping_failure.point_failures(evaluation_points))
        return _assembled_outcomes(
            0,
            [
                (jax.tree.map(lambda leaf, row=point_index: leaf[row], stacked_output), lossy_casts)
                for point_index in range(len(evaluation_points))
            ],
        )


class _SequentialExecutor(EvaluationExecutor):
    """Run the evaluation points one after another, in the calling thread."""

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def priority(self) -> int | None:
        return 0

    @property
    def description(self) -> str:
        return "Run the evaluation points one after another, in the calling thread."

    def check(
        self, evaluation_points: EvaluationPoints, /, *, chunk_size: int | None = None
    ) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(
        self, evaluation_points: EvaluationPoints, /, *, chunk_size: int | None = None
    ) -> EvaluationOutcomes:
        shared_payload = evaluation_points.shared_payload()
        return EvaluationOutcomes.merge(
            chunk.run(shared_payload) for chunk in evaluation_points.chunks(chunk_size=chunk_size)
        )


class _ThreadExecutor(EvaluationExecutor):
    """Run chunks of evaluation points on a pool of threads; opt-in, since user code may not be thread-safe."""

    @property
    def name(self) -> str:
        return "thread"

    @property
    def description(self) -> str:
        return "Run chunks of evaluation points on a pool of threads."

    def check(
        self,
        evaluation_points: EvaluationPoints,
        /,
        *,
        max_workers: int | None = None,
        chunk_size: int | None = None,
    ) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(
        self,
        evaluation_points: EvaluationPoints,
        /,
        *,
        max_workers: int | None = None,
        chunk_size: int | None = None,
    ) -> EvaluationOutcomes:
        worker_count = _checked_worker_count(max_workers)
        chunks = _parallel_chunks(evaluation_points, chunk_size, worker_count)
        shared_payload = evaluation_points.shared_payload()
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            return EvaluationOutcomes.merge(
                pool.map(lambda chunk: chunk.run(shared_payload), chunks)
            )


class _ProcessExecutor(EvaluationExecutor):
    """Run chunks of evaluation points in spawned worker processes, under the parent's JAX settings."""

    @property
    def name(self) -> str:
        return "process"

    @property
    def description(self) -> str:
        return "Run chunks of evaluation points in worker processes."

    def check(
        self,
        evaluation_points: EvaluationPoints,
        /,
        *,
        max_workers: int | None = None,
        chunk_size: int | None = None,
        start_method: str = "spawn",
    ) -> Feasibility:
        return evaluation_points.shipping_check()

    def execute(
        self,
        evaluation_points: EvaluationPoints,
        /,
        *,
        max_workers: int | None = None,
        chunk_size: int | None = None,
        start_method: str = "spawn",
    ) -> EvaluationOutcomes:
        worker_count = _checked_worker_count(max_workers)
        jax_settings = evaluation_points._call_context.jax_settings or _jax_settings_snapshot()
        pool = _process_pool(jax_settings, start_method, worker_count)
        shipped_payload = cloudpickle.dumps(evaluation_points.shared_payload())
        chunks = _parallel_chunks(evaluation_points, chunk_size, worker_count)
        submitted_chunks = [
            (chunk, pool.submit(_run_shipped_chunk, cloudpickle.dumps(chunk), shipped_payload))
            for chunk in chunks
        ]
        chunk_outcomes = []
        try:
            for chunk, outcomes_future in submitted_chunks:
                chunk_outcomes.append(_received_outcomes(chunk, outcomes_future, pool))
        finally:
            for _, outcomes_future in submitted_chunks:
                outcomes_future.cancel()
        return EvaluationOutcomes.merge(chunk_outcomes)


for _built_in_executor in (
    _JaxExecutor(),
    _SequentialExecutor(),
    _ThreadExecutor(),
    _ProcessExecutor(),
):
    executor_registry.register(_built_in_executor)
registry_catalog.register(executor_registry)


# ---------------------------------------------------------------------------
# Choosing the executor
# ---------------------------------------------------------------------------


def _chosen_executor(
    evaluation_points: EvaluationPoints,
    execution_choice: _ExecutionChoice,
    evaluation_log: _EvaluationLog | None,
) -> tuple[EvaluationExecutor, dict[str, Any]]:
    """The executor that runs *evaluation_points* under *execution_choice*, with its options."""
    jax_executor = executor_registry.get_method("jax")
    if jax_executor.check(evaluation_points).feasible:
        return jax_executor, {}
    automatic_options: Mapping[str, Any] = {}
    if execution_choice.names_executor():
        executor_name = execution_choice.executor_name
        named_executor = _requested_executor(cast(str, executor_name))
        _check_executor_reads_options(named_executor, execution_choice.executor_options)
        feasibility = named_executor.check(evaluation_points, **execution_choice.executor_options)
        if feasibility.feasible:
            return named_executor, dict(execution_choice.executor_options)
        if _is_named_for_the_callable(execution_choice):
            raise ResolutionError(
                f"executor {executor_name!r} cannot run the evaluation points of "
                f"{evaluation_points.origin!r}: {feasibility.description}. Name another executor"
            )
        fallback_executor = _automatically_selected_executor(evaluation_points)
        how_named = "inherited from" if execution_choice.source == "inherited" else "named on"
        warning_key = ("fallback", executor_name, evaluation_points.origin)
        if evaluation_log is None or evaluation_log.is_first_warning(warning_key):
            warnings.warn(
                f"executor {executor_name!r}, {how_named} {execution_choice.named_on}, cannot "
                f"run the evaluation points of {evaluation_points.origin!r} "
                f"({feasibility.description}), so they run on {fallback_executor.name!r} instead",
                UserWarning,
                stacklevel=3,
            )
        return fallback_executor, {}
    if execution_choice.source == "own":
        automatic_options = execution_choice.executor_options
    selected_executor = _automatically_selected_executor(evaluation_points)
    _check_executor_reads_options(selected_executor, automatic_options)
    return selected_executor, dict(automatic_options)


def _requested_executor(executor_name: str) -> EvaluationExecutor:
    """The executor a caller named, raising ``ResolutionError`` when none is registered by that name."""
    return executor_registry._named(executor_name).method


def _automatically_selected_executor(evaluation_points: EvaluationPoints) -> EvaluationExecutor:
    """The executor automatic selection picks for *evaluation_points*."""
    selection = executor_registry.check(evaluation_points)
    if not selection.feasible or selection.method_name is None:
        raise ResolutionError(selection.description)
    return executor_registry.get_method(selection.method_name)


def _is_named_for_the_callable(execution_choice: _ExecutionChoice) -> bool:
    """Whether the choice was named for the callable whose call asks for the evaluation points.

    That holds for a choice set on the running call's own ``Function``; a
    choice named on an operation covers every opaque callable of its call,
    and one passed down by an enclosing call is inherited.
    """
    return execution_choice.source == "own" and execution_choice.applies_only_to is not None


def _executor_option_names(executor: EvaluationExecutor) -> frozenset[str] | None:
    """The options *executor* takes, read from its ``execute``; ``None`` when it takes any."""
    parameters = list(inspect.signature(executor.execute).parameters.values())[1:]
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return frozenset(
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _check_executor_reads_options(
    executor: EvaluationExecutor, executor_options: Mapping[str, Any]
) -> None:
    """Raise if *executor* does not take one of *executor_options*."""
    option_names = _executor_option_names(executor)
    if option_names is None:
        return
    unread_options = sorted(set(executor_options) - option_names)
    if not unread_options:
        return
    listed_options = ", ".join(sorted(option_names)) if option_names else "none"
    raise ValueError(
        f"executor {executor.name!r} does not read the option {unread_options[0]!r}; "
        f"its options are {listed_options}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_outcomes_cover(
    evaluation_outcomes: EvaluationOutcomes, evaluation_points: EvaluationPoints, executor_name: str
) -> None:
    """Raise if an executor returned other than one outcome per evaluation point, from point index 0."""
    if evaluation_outcomes.start != 0 or len(evaluation_outcomes.outputs) != len(evaluation_points):
        raise RuntimeError(
            f"executor {executor_name!r} returned {len(evaluation_outcomes.outputs)} outcomes "
            f"starting at {evaluation_outcomes.start} for {len(evaluation_points)} evaluation "
            "points. An executor must return one outcome per evaluation point, starting at 0"
        )


def _checked_worker_count(max_workers: Any) -> int:
    """*max_workers*, or the number of processors when it is ``None``.

    Parameters
    ----------
    max_workers : Any
        The executor's ``max_workers`` option.

    Returns
    -------
    int
        The number of workers.

    Raises
    ------
    TypeError, ValueError
        If *max_workers* is not a positive integer.
    """
    if max_workers is None:
        return os.cpu_count() or 1
    return _checked_count(max_workers, "max_workers")


def _parallel_chunks(
    evaluation_points: EvaluationPoints, chunk_size: int | None, worker_count: int
) -> list[EvaluationChunk]:
    """The chunks a parallel executor runs: of *chunk_size*, or one for each worker."""
    if chunk_size is not None:
        return evaluation_points.chunks(chunk_size=chunk_size)
    return evaluation_points.chunks(num_chunks=worker_count)


#: The process pools of the ``process`` executor, one for each JAX settings
#: snapshot, start method, and worker count, reused across calls.
_PROCESS_POOLS: dict[tuple[Any, ...], concurrent.futures.ProcessPoolExecutor] = {}
_PROCESS_POOLS_LOCK = threading.Lock()


def _process_pool(
    jax_settings: Mapping[str, Any], start_method: str, worker_count: int
) -> concurrent.futures.ProcessPoolExecutor:
    """The pool whose workers run under *jax_settings*, started by *start_method*."""
    pool_key = (tuple(sorted(jax_settings.items())), start_method, worker_count)
    with _PROCESS_POOLS_LOCK:
        pool = _PROCESS_POOLS.get(pool_key)
        if pool is None:
            try:
                process_context = multiprocessing.get_context(start_method)
            except ValueError as start_method_error:
                raise ValueError(
                    f"executor 'process' cannot start workers by {start_method!r}: "
                    f"{start_method_error}. Use start_method='spawn'"
                ) from start_method_error
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=process_context,
                initializer=_apply_jax_settings,
                initargs=(dict(jax_settings),),
            )
            _PROCESS_POOLS[pool_key] = pool
        return pool


def _discard_process_pool(pool: concurrent.futures.ProcessPoolExecutor) -> None:
    """Forget *pool*, whose workers died, so the next call starts a new one."""
    with _PROCESS_POOLS_LOCK:
        for pool_key, cached_pool in list(_PROCESS_POOLS.items()):
            if cached_pool is pool:
                del _PROCESS_POOLS[pool_key]
    pool.shutdown(wait=False, cancel_futures=True)


def _apply_jax_settings(jax_settings: Mapping[str, Any]) -> None:
    """Apply the parent's JAX settings in a new worker process, before it unpickles any work."""
    for setting_name, setting_value in jax_settings.items():
        jax.config.update(setting_name, setting_value)


def _run_shipped_chunk(shipped_chunk: bytes, shipped_payload: bytes) -> bytes:
    """Run a pickled chunk in a worker process, and return its pickled outcomes.

    An output that cannot be sent back fails its evaluation point, so the
    other evaluation points' outcomes still reach the calling process.
    """
    chunk = cloudpickle.loads(shipped_chunk)
    shared_payload: SharedPayload = cloudpickle.loads(shipped_payload)
    chunk_outcomes = chunk.run(shared_payload)
    try:
        return cloudpickle.dumps(chunk_outcomes)
    except Exception:
        return cloudpickle.dumps(_with_unshippable_outputs_failed(chunk, chunk_outcomes))


def _with_unshippable_outputs_failed(
    chunk: EvaluationChunk, chunk_outcomes: EvaluationOutcomes
) -> EvaluationOutcomes:
    """*chunk_outcomes* with each output that cannot be pickled replaced by its evaluation point's failure."""
    outputs = list(chunk_outcomes.outputs)
    failures = dict(chunk_outcomes.failures)
    for point_index, output in enumerate(chunk_outcomes.outputs, start=chunk_outcomes.start):
        if point_index in failures:
            continue
        try:
            cloudpickle.dumps(output)
        except Exception as pickling_error:
            outputs[point_index - chunk_outcomes.start] = None
            failures[point_index] = chunk._failure(
                point_index,
                "error",
                TypeError(
                    f"its output, a {type(output).__name__}, cannot be sent back from the worker "
                    f"process ({type(pickling_error).__name__}: {pickling_error}). Return arrays "
                    "or other values that pickle, or use an executor that runs in the calling "
                    "process"
                ),
            )
    return replace(chunk_outcomes, outputs=tuple(outputs), failures=MappingProxyType(failures))


def _received_outcomes(
    chunk: EvaluationChunk,
    outcomes_future: concurrent.futures.Future[bytes],
    pool: concurrent.futures.ProcessPoolExecutor,
) -> EvaluationOutcomes:
    """The outcomes of *chunk*, which a worker process ran, or its evaluation points lost.

    A worker that died loses the chunk and takes its pool with it, so the next
    call starts a new pool.
    """
    try:
        return cloudpickle.loads(outcomes_future.result())
    except concurrent.futures.process.BrokenProcessPool as pool_error:
        _discard_process_pool(pool)
        return EvaluationOutcomes.lost(chunk, f"a worker process died: {pool_error}")
    except Exception as receiving_error:
        return EvaluationOutcomes.lost(
            chunk,
            "its outcomes could not be received from the worker process "
            f"({type(receiving_error).__name__}: {receiving_error})",
        )


class _MappingFailure(Exception):
    """A ``jax.vmap`` over a set of evaluation points raised, or returned an output that does not conform.

    Either fails every evaluation point of the set, since the map computes
    them together.

    Parameters
    ----------
    kind : {"error", "invalid"}
        ``"error"`` when the map raised; ``"invalid"`` when its output did not
        conform to the output spec.
    error : Exception
        The error.
    """

    def __init__(self, kind: str, error: Exception) -> None:
        super().__init__(str(error))
        self.kind = kind
        self.error = error

    def point_failures(self, evaluation_points: EvaluationPoints) -> list[Any]:
        """The failure of each evaluation point of *evaluation_points*, all caused by this one."""
        return [
            _point_failure(
                evaluation_points.origin,
                evaluation_points._set_ordinal,
                point_index,
                cast(Any, self.kind),
                self.error,
            )
            for point_index in range(len(evaluation_points))
        ]


def _mapped_outputs(
    evaluation_points: EvaluationPoints,
) -> tuple[Any, frozenset[tuple[str, str]]]:
    """The callable mapped over every evaluation point with ``jax.vmap``, checked once against the output spec.

    It returns the outputs stacked on a leading axis, with the pairs of dtype
    names of each cast to a declared dtype that lost precision. A map that
    raised, or whose output does not conform, raises :class:`_MappingFailure`,
    except a false ``traceable=True`` claim, whose error reaches the caller as
    itself.
    """
    shared_payload = evaluation_points.shared_payload()

    def evaluate_one_point(varying_arguments: Mapping[str, Any]) -> Any:
        return shared_payload.fn(**shared_payload.shared_arguments, **varying_arguments)

    try:
        with evaluation_points._point_scopes.mapped_scope():
            stacked_output = jax.vmap(evaluate_one_point)(dict(evaluation_points._varying))
    except FalseTraceableClaimError:
        # A false claim is an error of the callable's declaration, not of any
        # evaluation point, so it reaches the caller as itself.
        raise
    except Exception as error:
        raise _MappingFailure("error", error) from error
    try:
        return _checked_stacked_output(
            stacked_output,
            evaluation_points.output_spec,
            len(evaluation_points),
            _output_path(evaluation_points.origin),
        )
    except _InvalidPointOutput as invalid_output:
        raise _MappingFailure("invalid", invalid_output.check_error) from invalid_output


def _mapped_inline(
    fn: Callable[..., Any], varying: Mapping[str, Any], shared: Mapping[str, Any], *, stacked: bool
) -> Any:
    """*fn* mapped over traced varying arguments with ``jax.vmap``, in the trace that holds them."""
    stacked_output = jax.vmap(lambda varying_arguments: fn(**shared, **varying_arguments))(
        dict(varying)
    )
    if stacked:
        return stacked_output
    num_points = next(np.shape(leaf)[0] for leaf in jax.tree.leaves(varying) if np.ndim(leaf) > 0)
    return tuple(
        jax.tree.map(lambda leaf, row=point_index: leaf[row], stacked_output)
        for point_index in range(num_points)
    )


def _stacked_outputs(outputs: Sequence[Any]) -> Any:
    """*outputs*, one per evaluation point, as one tree stacked on a leading axis.

    When every leaf of every output is numeric and the outputs share one tree
    structure, each leaf is stacked with ``jnp.stack``, as ``jax.vmap`` stacks
    a mapped output. Otherwise, as for outputs that are laws, arrays of
    different shapes, or batches, the result is a one-dimensional object array
    that holds each output.
    """
    flattened_outputs = [jax.tree.flatten(output) for output in outputs]
    tree_definitions = {tree_definition for _, tree_definition in flattened_outputs}
    output_leaves = [leaves for leaves, _ in flattened_outputs]
    if (
        len(tree_definitions) == 1
        and output_leaves
        and output_leaves[0]
        and all(_is_numeric_leaf(leaf) for leaves in output_leaves for leaf in leaves)
    ):
        try:
            return jax.tree.unflatten(
                next(iter(tree_definitions)),
                [jnp.stack(leaf_rows) for leaf_rows in zip(*output_leaves)],
            )
        except (TypeError, ValueError):
            # Leaves of different shapes, or a container that cannot hold an
            # added axis, such as a batch, whose shape is fixed.
            pass
    held_outputs = np.empty(len(outputs), dtype=object)
    for point_index, output in enumerate(outputs):
        held_outputs[point_index] = output
    return held_outputs


def _holds_tracer(tree: Any) -> bool:
    """Whether any leaf of *tree* is a value JAX is tracing."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(tree))


def _execution_claims_of(fn: Callable[..., Any]) -> ExecutionClaims:
    """The execution claims a ``Function`` states, or none for any other callable."""
    claims = getattr(fn, "_execution_claims", None)
    return claims if isinstance(claims, ExecutionClaims) else NO_CLAIMS
