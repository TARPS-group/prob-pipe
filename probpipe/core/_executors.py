"""Executors, the registry that chooses among them, and the dispatcher that runs a set of evaluation points (V.9).

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
  - :func:`_run_on_chosen_executor` – the one function that runs a set of
    evaluation points on the executor its call chose.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing
import os
import threading
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import cloudpickle
import jax

from .._messages import unknown_names
from ._call_context import _ExecutionChoice, _jax_settings_snapshot, current_call_context
from ._catalog import registry_catalog
from ._dispatch import (
    Feasibility,
    ResolutionError,
    UnaryDispatchMethod,
    UnaryDispatchRegistry,
)
from ._evaluation_points import (
    EvaluationError,
    EvaluationFailure,
    EvaluationOutcomes,
    EvaluationPoints,
    SharedPayload,
    _checked_point_output,
    _holds_only_values,
    _InvalidPointOutput,
    _warn_lossy_casts,
)

__all__ = ["EvaluationExecutor", "executor_registry"]


class EvaluationExecutor(UnaryDispatchMethod):
    """The base class of an executor: a named way to run a set of evaluation points.

    A subclass sets ``name`` and ``description``, sets ``priority`` when
    automatic selection may choose it, and implements :meth:`check` and
    :meth:`execute`. Its options are the keyword-only parameters of
    :meth:`execute`, which :meth:`check` takes too; ProbPipe refuses an option
    that :meth:`execute` does not take before it calls either. An executor
    passes ``probpipe.testing.check_executor`` when it runs every evaluation
    point once, returns the outcomes in order, and changes no argument.

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

    exact = True
    priority: int | None = None
    description = ""

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


#: The registry of executors, which the registry catalog lists.
executor_registry: _ExecutorRegistry = _ExecutorRegistry(
    name="executor",
    description="Executors that run sets of evaluation points: in order, on threads, in "
    "worker processes, or compiled by JAX",
)


def _run_on_chosen_executor(evaluation_points: EvaluationPoints) -> tuple[Any, ...]:
    """Each evaluation point's output, in point order, run on the executor the running call chose.

    The choice follows V.9's rules. A traceable callable is compiled by
    ``jax`` whatever executor the call names, and receives none of the named
    options. Otherwise an executor named for this callable runs the set, or
    raises when it cannot; an inherited executor that cannot run the set
    leaves it to automatic selection without its options, and warns once per
    call; and with no executor named, automatic selection runs the set, with
    the options the call's own ``Function`` gave.

    The call's provenance records the set, and what its evaluation points
    noted merges into the call's record.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of evaluation points, built under the running call.

    Returns
    -------
    tuple
        Each evaluation point's output, in point order.

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
    call_context = current_call_context()
    executor, executor_options = _chosen_executor(
        evaluation_points, call_context.execution_choice, call_context.evaluation_log
    )
    try:
        evaluation_outcomes = executor.execute(evaluation_points, **executor_options)
        _check_outcomes_cover(evaluation_outcomes, evaluation_points, executor.name)
        evaluation_points._point_scopes.accept_claim_reports(evaluation_outcomes._claim_reports)
    finally:
        evaluation_points._point_scopes.close()
    _merge_outcome_accumulators(evaluation_outcomes, call_context)
    if call_context.evaluation_log is not None:
        call_context.evaluation_log.record(
            _set_record(evaluation_points, executor.name, executor_options, evaluation_outcomes)
        )
    _warn_lossy_casts(
        evaluation_outcomes._lossy_casts,
        origin=evaluation_points.origin,
        evaluation_log=call_context.evaluation_log,
    )
    if evaluation_outcomes.failures:
        failures = tuple(evaluation_outcomes.failures.values())
        first_error = min(failures, key=lambda failure: failure.point_index).error
        raise EvaluationError(
            failures,
            evaluation_outcomes.outputs,
            origin=evaluation_points.origin,
            point_noun=evaluation_points.point_noun,
        ) from first_error
    return evaluation_outcomes.outputs


# ---------------------------------------------------------------------------
# The built-in executors
# ---------------------------------------------------------------------------


class _JaxExecutor(EvaluationExecutor):
    """Compile a traceable callable over every evaluation point with one ``jax.vmap``."""

    name = "jax"
    priority = 20
    description = "Map a traceable callable over every evaluation point with jax.vmap."

    def check(self, evaluation_points: EvaluationPoints, /) -> Feasibility:
        if not evaluation_points.traceable:
            return Feasibility(
                feasible=False,
                description=f"{evaluation_points.origin!r} is not known to be traceable by JAX",
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
        shared_payload = evaluation_points.shared_payload()
        num_points = len(evaluation_points)

        def evaluate_one_point(varying_arguments: Mapping[str, Any]) -> Any:
            return shared_payload.fn(**shared_payload.shared_arguments, **varying_arguments)

        try:
            with evaluation_points._point_scopes.mapped_scope():
                stacked_outputs = jax.vmap(evaluate_one_point)(dict(evaluation_points._varying))
        except Exception as error:
            failures = {
                point_index: _point_failure(evaluation_points, point_index, "error", error)
                for point_index in range(num_points)
            }
            return EvaluationOutcomes(start=0, outputs=(None,) * num_points, failures=failures)
        outputs: list[Any] = []
        failures = {}
        lossy_casts: set[tuple[str, str]] = set()
        for point_index in range(num_points):
            output = jax.tree.map(lambda leaf, row=point_index: leaf[row], stacked_outputs)
            try:
                checked_output, point_lossy_casts = _checked_point_output(
                    output,
                    evaluation_points.output_spec,
                    f"the output of {evaluation_points.origin!r} at "
                    f"{evaluation_points.point_noun} {point_index}",
                )
            except _InvalidPointOutput as invalid_output:
                failures[point_index] = _point_failure(
                    evaluation_points, point_index, "invalid", invalid_output.check_error
                )
                outputs.append(None)
                continue
            outputs.append(checked_output)
            lossy_casts |= point_lossy_casts
        return EvaluationOutcomes(
            start=0,
            outputs=tuple(outputs),
            failures=failures,
            _lossy_casts=frozenset(lossy_casts),
        )


class _SequentialExecutor(EvaluationExecutor):
    """Run the evaluation points one after another, in the calling thread."""

    name = "sequential"
    priority = 0
    description = "Run the evaluation points one after another, in the calling thread."

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

    name = "thread"
    description = "Run chunks of evaluation points on a pool of threads."

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
        worker_count = _checked_worker_count(max_workers, self.name)
        chunks = _parallel_chunks(evaluation_points, chunk_size, worker_count)
        shared_payload = evaluation_points.shared_payload()
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            return EvaluationOutcomes.merge(
                pool.map(lambda chunk: chunk.run(shared_payload), chunks)
            )


class _ProcessExecutor(EvaluationExecutor):
    """Run chunks of evaluation points in spawned worker processes, under the parent's JAX settings."""

    name = "process"
    description = "Run chunks of evaluation points in worker processes."

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
        worker_count = _checked_worker_count(max_workers, self.name)
        jax_settings = evaluation_points._call_context.jax_settings or _jax_settings_snapshot()
        pool = _process_pool(jax_settings, start_method, worker_count)
        shipped_payload = cloudpickle.dumps(evaluation_points.shared_payload())
        chunks = _parallel_chunks(evaluation_points, chunk_size, worker_count)
        submitted_chunks = [
            (chunk, pool.submit(_run_shipped_chunk, cloudpickle.dumps(chunk), shipped_payload))
            for chunk in chunks
        ]
        chunk_outcomes = []
        for chunk, shipped_outcomes in submitted_chunks:
            try:
                chunk_outcomes.append(cloudpickle.loads(shipped_outcomes.result()))
            except concurrent.futures.process.BrokenProcessPool as pool_error:
                _discard_process_pool(pool)
                chunk_outcomes.append(
                    EvaluationOutcomes.lost(chunk, f"a worker died: {pool_error}")
                )
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
    evaluation_log: Any,
) -> tuple[EvaluationExecutor, dict[str, Any]]:
    """The executor that runs *evaluation_points* under *execution_choice*, with its options."""
    jax_executor = executor_registry.get_method("jax")
    if jax_executor.check(evaluation_points).feasible:
        return jax_executor, {}
    automatic_options: Mapping[str, Any] = {}
    if execution_choice.names_executor():
        executor_name = execution_choice.executor_name
        named_executor = executor_registry.get_method(executor_name)
        _check_executor_reads_options(named_executor, execution_choice.executor_options)
        feasibility = named_executor.check(evaluation_points, **execution_choice.executor_options)
        if feasibility.feasible:
            return named_executor, dict(execution_choice.executor_options)
        fallback_executor = _automatically_selected_executor(evaluation_points)
        if _is_named_for_the_callable(execution_choice):
            raise ResolutionError(
                f"executor {executor_name!r} cannot run the evaluation points of "
                f"{evaluation_points.origin!r}: {feasibility.description}. Name another executor"
            )
        warning_key = ("fallback", executor_name, evaluation_points.origin)
        if evaluation_log is None or evaluation_log.is_first_warning(warning_key):
            warnings.warn(
                f"executor {executor_name!r}, inherited from {execution_choice.named_on}, cannot "
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


def _merge_outcome_accumulators(evaluation_outcomes: EvaluationOutcomes, call_context: Any) -> None:
    """Merge what the evaluation points noted into the call's accumulators."""
    if (
        call_context.num_samples_used is not None
        and evaluation_outcomes._num_samples_used is not None
    ):
        call_context.num_samples_used.merge_from(evaluation_outcomes._num_samples_used)
    if call_context.evaluation_log is not None and evaluation_outcomes._evaluation_log is not None:
        call_context.evaluation_log.merge_from(evaluation_outcomes._evaluation_log)


def _set_record(
    evaluation_points: EvaluationPoints,
    executor_name: str,
    executor_options: Mapping[str, Any],
    evaluation_outcomes: EvaluationOutcomes,
) -> Any:
    """The record of one set of evaluation points that the call's provenance reports."""
    from ._call_context import _EvaluationSetRecord

    return _EvaluationSetRecord(
        origin=evaluation_points.origin,
        set_ordinal=evaluation_points._set_ordinal,
        executor_name=executor_name,
        executor_options=dict(executor_options),
        num_evaluation_points=len(evaluation_points),
        num_runs=len(evaluation_points),
        failed_point_indices=tuple(sorted(evaluation_outcomes.failures)),
    )


def _point_failure(
    evaluation_points: EvaluationPoints, point_index: int, kind: Any, error: Exception
) -> EvaluationFailure:
    """The failure of evaluation point *point_index* of *evaluation_points*."""
    return EvaluationFailure(
        point_index=point_index,
        point_identity=evaluation_points._point_identity(point_index),
        kind=kind,
        error_type=type(error).__qualname__,
        message=str(error),
        error=error,
    )


def _checked_worker_count(max_workers: Any, executor_name: str) -> int:
    """*max_workers*, or the number of processors when it is ``None``; raises if it is not a positive integer."""
    if max_workers is None:
        return os.cpu_count() or 1
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError(
            f"executor {executor_name!r} needs max_workers to be a positive integer, "
            f"got {max_workers!r}"
        )
    return max_workers


def _parallel_chunks(
    evaluation_points: EvaluationPoints, chunk_size: int | None, worker_count: int
) -> list[Any]:
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
    """Run a pickled chunk in a worker process, and return its pickled outcomes."""
    chunk = cloudpickle.loads(shipped_chunk)
    shared_payload: SharedPayload = cloudpickle.loads(shipped_payload)
    return cloudpickle.dumps(chunk.run(shared_payload))
