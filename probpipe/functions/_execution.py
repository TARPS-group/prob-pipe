"""How the function engine runs the evaluation points its calls ask for, and the execution contract.

A lifted call, a sweep, and a plain call each ask for one set of evaluation
points, which :func:`evaluate_requested_points` runs on the executor the call
chooses, or as Prefect tasks when the call's ``workflow_kind`` selects
Prefect, which then records the call. Each evaluation point runs in a
workflow scope of its own, which gives a draw made inside it its random key;
this module builds those scopes and installs them into ``core``.

The module also holds the versioned execution contract: the versions of what
decides a call's random keys, which a replay record stores and a replay
checks. The executor is not part of it, since no executor changes a result.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from time import sleep
from typing import Any

try:
    from prefect import flow, task
    from prefect.cache_policies import NO_CACHE
except ImportError:
    task = flow = NO_CACHE = None

from ..core._call_context import (
    _AUTOMATIC_SELECTION,
    _jax_tracing_active,
    _jax_transformation_active,
    current_call_context,
)
from ..core._dispatch import Feasibility
from ..core._evaluation_points import (
    EvaluationOutcomes,
    EvaluationPoints,
    _install_evaluation_point_scopes,
    _UnscopedEvaluationPoints,
)
from ..core._executors import (
    EvaluationExecutor,
    _evaluated_outcomes,
    _evaluated_outputs,
    _outputs_of,
    _run_on_executor,
    executor_registry,
)
from ..core.config import WorkflowKind, config
from ..values._function_base import _WARNING_SKIP_PREFIXES
from . import _context, _evaluation_keys
from ._evaluation_keys import (
    _DISTRIBUTION_SAMPLING_ABI,
    _PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
    EvaluationPointAttempt,
    EvaluationPointClaimReport,
    IssuedEvaluationPoint,
    issue_evaluation_points,
    lifted_evaluation_point_segment,
    plain_call_point_segment,
    sweep_cell_point_segment,
)
from ._plan import StochasticPlan

__all__ = (
    "evaluate_plain_call",
    "evaluate_requested_points",
    "issue_evaluation_points",
    "lifted_evaluation_point_segment",
    "plain_call_point_segment",
    "preflight_prefect_route",
    "recorded_evaluations",
    "sweep_cell_point_segment",
    "warn_if_named_executor_ran_nothing",
)

#: The orchestration modes in which Prefect runs a call's evaluation points.
_PREFECT_WORKFLOW_KINDS = frozenset({WorkflowKind.TASK, WorkflowKind.FLOW})


def evaluate_requested_points(
    evaluation_points: EvaluationPoints,
    *,
    stacked: bool,
    workflow_kind: WorkflowKind,
    function_label: str,
    stochastic_plan: StochasticPlan | None,
) -> Any:
    """The outputs of the evaluation points a call of the engine asks for, as ``evaluate_at_points`` gives them.

    The set runs on the executor the call chooses. When *workflow_kind*
    selects Prefect, Prefect runs it instead, as one task per evaluation
    point, or as one task that maps a traceable callable with ``jax.vmap``,
    and a call that names an executor warns that the executor runs nothing.
    Inside a route probe or a body that JAX maps, the set runs in the calling
    thread under automatic selection, with no Prefect, since nothing may
    start threads, processes, or tasks there. The call records its execution
    contract.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of evaluation points, built under the running call.
    stacked : bool
        Whether to return the outputs stacked on a leading axis, as
        ``evaluate_at_points`` states.
    workflow_kind : WorkflowKind
        The call's resolved orchestration mode.
    function_label : str
        The label of the called ``Function``, which names Prefect's flows and
        tasks.
    stochastic_plan : StochasticPlan or None
        The plan of the laws the call draws from, which the execution contract
        versions; ``None`` for a call that draws from none.

    Returns
    -------
    Any
        Each evaluation point's output, in point order, or the stacked
        outputs.

    Raises
    ------
    EvaluationError
        If an evaluation point failed, naming every failed one.
    """
    side_effects_forbidden = _context._workflow_side_effects_forbidden()
    if not side_effects_forbidden:
        _record_execution_contract(stochastic_plan)
        if workflow_kind in _PREFECT_WORKFLOW_KINDS:
            prefect_outcomes = _prefect_outcomes(evaluation_points, workflow_kind, function_label)
            return _outputs_of(evaluation_points, prefect_outcomes, stacked=stacked)
    return _evaluated_outputs(
        evaluation_points,
        stacked=stacked,
        execution_choice=_AUTOMATIC_SELECTION if side_effects_forbidden else None,
    )


def evaluate_plain_call(
    evaluation_points: EvaluationPoints,
    *,
    workflow_kind: WorkflowKind,
    function_label: str,
) -> Any:
    """The output of a plain call's one evaluation point, which runs as :func:`evaluate_requested_points` runs a set.

    A plain call's failure is its body's: the call raises the exception its
    body raised, as if the body had run in the calling thread. ``jax`` never
    maps a plain call, so a call that names ``jax`` runs its body in the
    calling thread, as automatic selection does. So does a call made inside a
    JAX transformation, whose traced arguments no other executor can run.

    Parameters
    ----------
    evaluation_points : EvaluationPoints
        The set of the call's one evaluation point.
    workflow_kind : WorkflowKind
        The call's resolved orchestration mode.
    function_label : str
        The label of the called ``Function``.

    Returns
    -------
    Any
        The body's result.

    Raises
    ------
    Exception
        Whatever the body raised.
    EvaluationError
        If the evaluation point failed with an exception that did not come
        back from the worker that ran it.
    """
    side_effects_forbidden = _context._workflow_side_effects_forbidden()
    if not side_effects_forbidden:
        _record_execution_contract(None)
    if not side_effects_forbidden and workflow_kind in _PREFECT_WORKFLOW_KINDS:
        evaluation_outcomes = _prefect_outcomes(evaluation_points, workflow_kind, function_label)
    else:
        execution_choice = current_call_context().execution_choice
        names_jax = execution_choice.executor_name == "jax"
        if names_jax and execution_choice.named_executor_use is not None:
            execution_choice.named_executor_use.note_ran_a_set()
        in_calling_thread = side_effects_forbidden or names_jax or _jax_transformation_active()
        evaluation_outcomes = _evaluated_outcomes(
            evaluation_points,
            execution_choice=_AUTOMATIC_SELECTION if in_calling_thread else None,
        )
    body_failure = evaluation_outcomes.failures.get(0)
    if body_failure is not None and body_failure.error is not None:
        raise body_failure.error
    (body_result,) = _outputs_of(evaluation_points, evaluation_outcomes, stacked=False)
    return body_result


def recorded_evaluations(*, origin: str, set_ordinal: int | None = None) -> list[dict[str, Any]]:
    """The sets of evaluation points of *origin* that the running call ran, as its provenance records them.

    Each set is read from the call's evaluation log, in the order it ran.

    Parameters
    ----------
    origin : str
        The callable whose sets are recorded.
    set_ordinal : int or None, optional
        The one set to record, by its ordinal among the call's sets of
        *origin*; ``None`` records every set of *origin*.

    Returns
    -------
    list of dict
        One entry per set: the ``callable``, the ``executor`` that ran it,
        the ``executor_options`` it received, the ``num_evaluation_points``,
        and the ``failed_point_indices``. Empty for a call that keeps no log.
    """
    evaluation_log = current_call_context().evaluation_log
    if evaluation_log is None:
        return []
    return [
        {
            "callable": set_record.origin,
            "executor": set_record.executor_name,
            "executor_options": dict(set_record.executor_options),
            "num_evaluation_points": set_record.num_evaluation_points,
            "failed_point_indices": list(set_record.failed_point_indices),
        }
        for set_record in evaluation_log.set_records
        if set_record.origin == origin
        and (set_ordinal is None or set_record.set_ordinal == set_ordinal)
    ]


def preflight_prefect_route(
    workflow_kind: WorkflowKind, function_label: str, *, maps_with_jax: bool
) -> None:
    """Build the Prefect flow and tasks a call will run its evaluation points in, before it draws.

    A route Prefect cannot build then fails before the call takes any random
    draw or commits its place in the workflow, so a failed route shifts no key
    of a later call. Outside Prefect orchestration, and inside a route probe or
    a mapped body, it does nothing.

    Parameters
    ----------
    workflow_kind : WorkflowKind
        The call's resolved orchestration mode.
    function_label : str
        The label of the called ``Function``, which names the flows and tasks.
    maps_with_jax : bool
        Whether ``jax`` will map the call's evaluation points.

    Raises
    ------
    RuntimeError
        If Prefect is not installed.
    """
    if workflow_kind not in _PREFECT_WORKFLOW_KINDS or _context._workflow_side_effects_forbidden():
        return
    _ensure_prefect_available()
    prefect_transport = _PrefectTransport(
        workflow_kind, function_label, maps_with_jax=maps_with_jax
    )
    _, chunks_flow = prefect_transport._decorated_runs(lambda chunk=None: None)
    if chunks_flow is not None:
        chunks_flow(lambda: None)


def _prefect_outcomes(
    evaluation_points: EvaluationPoints, workflow_kind: WorkflowKind, function_label: str
) -> EvaluationOutcomes:
    """The outcomes of a set of evaluation points that Prefect runs, as :func:`evaluate_requested_points` states."""
    _warn_executor_under_prefect(workflow_kind, function_label)
    prefect_transport = _PrefectTransport(
        workflow_kind,
        function_label,
        maps_with_jax=executor_registry.get_method("jax").check(evaluation_points).feasible,
    )
    return _run_on_executor(evaluation_points, prefect_transport, {})


def _record_execution_contract(stochastic_plan: StochasticPlan | None) -> None:
    """Record the execution contract of a call that runs evaluation points, for its replay record."""
    _evaluation_keys._record_active_execution_contract(
        make_execution_contract(stochastic_plan=stochastic_plan)
    )


def _warn_executor_under_prefect(workflow_kind: WorkflowKind, function_label: str) -> None:
    """Warn, once per call, that the executor a call names runs nothing, since Prefect runs its evaluations."""
    call_context = current_call_context()
    execution_choice = call_context.execution_choice
    named_executor_use = execution_choice.named_executor_use
    if execution_choice.source != "own":
        if named_executor_use is not None:
            named_executor_use.note_ran_elsewhere()
        return
    if named_executor_use is not None:
        named_executor_use.note_warned()
    evaluation_log = call_context.evaluation_log
    if evaluation_log is not None and not evaluation_log.is_first_warning(
        ("executor under Prefect", function_label)
    ):
        return
    if execution_choice.names_executor():
        named_control = f"executor {execution_choice.executor_name!r}"
        fix = "Remove the executor, or set workflow_kind=WorkflowKind.OFF to run them on it"
    else:
        named_control = "executor_options"
        fix = (
            "Remove executor_options, or set workflow_kind=WorkflowKind.OFF to run them on "
            "an executor"
        )
    warnings.warn(
        f"{named_control} on {function_label!r} has no effect: workflow_kind="
        f"WorkflowKind.{workflow_kind.name} runs its evaluations as Prefect tasks. {fix}",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=_WARNING_SKIP_PREFIXES,
    )


def warn_if_named_executor_ran_nothing() -> None:
    """Warn that the executor the running call named ran no set of evaluation points, in it or in any call under it.

    Only a call that named its executor itself warns, when it succeeds. It
    does not warn when its evaluations ran elsewhere, such as on an executor
    an inner call named itself or as Prefect tasks, or when a warning about
    the executor was already given, such as a fallback's. A call made inside
    a route probe or inside a body that JAX maps runs under automatic
    selection, so it does not warn. A call opened while JAX stages a program
    only stages its evaluations, which run when the program runs, so it does
    not warn either. When ``jax`` compiled the call's code in place of the
    executor, or the call ran inside a JAX transformation, the warning says so.
    """
    call_context = current_call_context()
    execution_choice = call_context.execution_choice
    named_executor_use = execution_choice.named_executor_use
    executor_name = execution_choice.executor_name
    if (
        execution_choice.source != "own"
        or executor_name is None
        or named_executor_use is None
        or named_executor_use.ran_a_set
        or named_executor_use.ran_elsewhere
        or named_executor_use.warned
        or _context._workflow_side_effects_forbidden()
        or _jax_tracing_active()
    ):
        return
    if named_executor_use.jax_compiled_a_set:
        why_nothing_ran = "every callable in the call is traceable and was compiled by jax"
    elif _jax_transformation_active():
        why_nothing_ran = (
            "the call ran inside a JAX transformation, which runs its body in the calling thread"
        )
    else:
        why_nothing_ran = "the call evaluated no user callable"
    warnings.warn(
        f"{call_context.function_label} named executor={executor_name!r}, but "
        f"{why_nothing_ran}, so {executor_name!r} ran nothing",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=_WARNING_SKIP_PREFIXES,
    )


# -- Prefect ------------------------------------------------------------------


class _PrefectTransport(EvaluationExecutor):
    """Run a set of evaluation points as Prefect tasks, which record the call.

    Each evaluation point is one chunk, run as one task, inside one flow. A
    set that ``jax`` can map runs as one task, or under ``FLOW`` one flow,
    that maps it, and the transport's name is then ``"jax"``, which the
    call's evaluation log records; otherwise it is ``"prefect"``. Prefect's
    default task retry policy runs a failed evaluation point again, with the
    same random keys. The transport is no entry of the executor registry:
    ``workflow_kind`` selects it.

    Parameters
    ----------
    workflow_kind : WorkflowKind
        ``TASK`` or ``FLOW``.
    function_label : str
        The label of the called ``Function``, which names the flows and tasks.
    maps_with_jax : bool
        Whether ``jax`` maps the set.
    """

    def __init__(
        self, workflow_kind: WorkflowKind, function_label: str, *, maps_with_jax: bool
    ) -> None:
        self._workflow_kind = workflow_kind
        self._function_label = function_label
        self._maps_with_jax = maps_with_jax

    @property
    def name(self) -> str:
        return "jax" if self._maps_with_jax else "prefect"

    @property
    def description(self) -> str:
        return "Run evaluation points as Prefect tasks."

    def check(self, evaluation_points: EvaluationPoints, /) -> Feasibility:
        return Feasibility(feasible=True)

    def execute(self, evaluation_points: EvaluationPoints, /) -> EvaluationOutcomes:
        _ensure_prefect_available()
        if self._maps_with_jax:
            jax_executor = executor_registry.get_method("jax")
            return self._mapped_run(lambda: jax_executor.execute(evaluation_points))
        return self._chunk_runs(evaluation_points)

    def _decorated_runs(self, run: Any) -> tuple[Any, Any]:
        """*run* as the Prefect task that runs the set, and the decorator of the flow around the tasks.

        A mapped set runs as one task, or under ``FLOW`` one flow, and has no
        flow around it; any other set runs as one task per chunk, which the
        decorated flow submits. The names are those the class states. No task
        caches its result: Prefect would pickle each chunk to compute a cache
        key, which reserves the chunk's evaluation points as if it were sent to
        another process.
        """
        function_label = self._function_label
        task_runner = config.prefect.resolve_task_runner()
        runner_options = {"task_runner": task_runner} if task_runner is not None else {}
        if self._maps_with_jax:
            mapped_name = f"{function_label}_vmap"
            if self._workflow_kind is WorkflowKind.TASK:
                return task(name=mapped_name, cache_policy=NO_CACHE)(run), None
            return flow(name=mapped_name, **runner_options)(run), None
        if self._workflow_kind is WorkflowKind.TASK:
            flow_name, task_name = f"{function_label}_map", function_label
        else:
            flow_name, task_name = function_label, f"{function_label}_run"
        return (
            task(name=task_name, retries=0, cache_policy=NO_CACHE)(run),
            flow(name=flow_name, **runner_options),
        )

    def _mapped_run(self, run_mapped: Any) -> EvaluationOutcomes:
        """*run_mapped* run as one Prefect task, or under ``FLOW`` one flow, named for the map."""
        mapped_task_or_flow, _ = self._decorated_runs(run_mapped)
        return mapped_task_or_flow()

    def _chunk_runs(self, evaluation_points: EvaluationPoints) -> EvaluationOutcomes:
        """The outcomes of every evaluation point, each run as one Prefect task inside one flow."""
        shared_payload = evaluation_points.shared_payload()
        run_chunk, chunks_flow = self._decorated_runs(lambda chunk: chunk.run(shared_payload))

        def run_chunks() -> EvaluationOutcomes:
            chunks = evaluation_points.chunks(chunk_size=1)
            outcomes_by_start = _submitted_chunk_outcomes(run_chunk, chunks)
            max_retries, retry_delays = _prefect_retry_policy()
            for retry_ordinal in range(1, max_retries + 1):
                failed_chunks = [
                    chunk for chunk in chunks if outcomes_by_start[chunk.start].failures
                ]
                if not failed_chunks:
                    break
                retry_delay = _retry_delay_for_ordinal(retry_delays, retry_ordinal)
                if retry_delay > 0:
                    sleep(retry_delay)
                outcomes_by_start.update(_submitted_chunk_outcomes(run_chunk, failed_chunks))
            return EvaluationOutcomes.merge(outcomes_by_start.values())

        return chunks_flow(run_chunks)()


def _submitted_chunk_outcomes(
    run_chunk: Any, chunks: Sequence[Any]
) -> dict[int, EvaluationOutcomes]:
    """The outcomes of *chunks*, each run as one task of *run_chunk*, by the start of each chunk.

    A task that did not finish loses its chunk's evaluation points.
    """
    futures = run_chunk.map(chunk=list(chunks))
    outcomes_by_start: dict[int, EvaluationOutcomes] = {}
    for chunk, future in zip(chunks, futures, strict=True):
        try:
            outcomes_by_start[chunk.start] = future.result()
        except Exception as task_error:
            outcomes_by_start[chunk.start] = EvaluationOutcomes.lost(
                chunk,
                f"its Prefect task did not finish ({type(task_error).__name__}: {task_error})",
            )
    return outcomes_by_start


# -- The execution contract ----------------------------------------------------

_EXECUTION_CONTRACT_ABI = "probpipe.workflow_rng_execution/v2"
_RNG_ABI = "ProbPipe-RNG-v1"
_DESCENDANT_ADAPTER_ABI = "probpipe.transformed_descendant/v1"
_DESCENDANT_PROVIDER_ABI = "tensorflow_probability.substrates.jax.bijector.forward/v1"
_JAX_KEY_ABI = "jax.random.wrap_key_data/threefry2x32/v1"


@dataclass(frozen=True)
class WorkflowRngExecutionContract:
    """The versions of what decides the random keys of a call, which a replay checks.

    Attributes
    ----------
    abi : str
        The version of the contract itself.
    rng_abi : str
        The version of the derivation of keys from a seed and a place in the
        workflow.
    sampling_abi : str
        The version of the sampling algorithms.
    provider_abis : tuple of str
        The versions of the laws the call draws from, sorted.
    descendant_adapter_abis : tuple of str
        The versions of the adapters of transformed laws, sorted.
    jax_key_abi : str
        The version of the format of JAX keys.
    plan_evaluation_mode : str or None
        Whether the call's lift samples or enumerates; ``None`` for a call
        that lifts nothing.
    """

    abi: str
    rng_abi: str
    sampling_abi: str
    provider_abis: tuple[str, ...]
    descendant_adapter_abis: tuple[str, ...]
    jax_key_abi: str
    plan_evaluation_mode: str | None


def make_execution_contract(
    *, stochastic_plan: StochasticPlan | None
) -> WorkflowRngExecutionContract:
    """The execution contract of a call that draws from the laws of *stochastic_plan*."""
    provider_abis, descendant_adapter_abis = _stochastic_plan_abis(stochastic_plan)
    return WorkflowRngExecutionContract(
        abi=_EXECUTION_CONTRACT_ABI,
        rng_abi=_RNG_ABI,
        sampling_abi=_DISTRIBUTION_SAMPLING_ABI,
        provider_abis=provider_abis,
        descendant_adapter_abis=descendant_adapter_abis,
        jax_key_abi=_JAX_KEY_ABI,
        plan_evaluation_mode=(None if stochastic_plan is None else stochastic_plan.evaluation_mode),
    )


def supports_execution_contract(
    contract: WorkflowRngExecutionContract, stochastic_plan: StochasticPlan | None
) -> bool:
    """Whether *contract* is the contract this ProbPipe version gives a call of *stochastic_plan*."""
    if not isinstance(contract, WorkflowRngExecutionContract):
        return False
    if not _is_canonical_abi_sequence(contract.provider_abis):
        return False
    if not _is_canonical_abi_sequence(contract.descendant_adapter_abis):
        return False
    try:
        expected_contract = make_execution_contract(stochastic_plan=stochastic_plan)
    except (TypeError, ValueError):
        return False
    return contract == expected_contract


def execution_diagnostics(contract: WorkflowRngExecutionContract) -> dict[str, Any]:
    """What a replay record notes of how a call ran under *contract*: the executor the call named, and its options.

    The executor changes no result, so a replay notes it and never requires
    it. An option whose value JSON cannot hold is noted by its ``repr``.

    Parameters
    ----------
    contract : WorkflowRngExecutionContract
        The execution contract the call ran under.

    Returns
    -------
    dict of str to Any
        ``executor``, the name the call's executor choice holds, ``None`` for
        automatic selection; ``executor_options``; and ``contract_abi``.
    """
    execution_choice = current_call_context().execution_choice
    return {
        "executor": execution_choice.executor_name,
        "executor_options": {
            option_name: _json_or_repr(option_value)
            for option_name, option_value in execution_choice.executor_options.items()
        },
        "contract_abi": contract.abi,
    }


def execution_contract_abi() -> str:
    """The version of the execution contract that this ProbPipe version records and replays."""
    return _EXECUTION_CONTRACT_ABI


def key_adapter_abi() -> str:
    """The version of the format of JAX keys that this ProbPipe version records and replays."""
    return _JAX_KEY_ABI


def _stochastic_plan_abis(
    stochastic_plan: StochasticPlan | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    provider_abis = {_PROBPIPE_DISTRIBUTION_PROVIDER_ABI}
    descendant_adapter_abis: set[str] = set()
    sampling_abis: set[str] = set()
    if stochastic_plan is not None:
        for group in stochastic_plan.source_groups:
            for consumer in group.consumers:
                summary = consumer._descriptor_abi_summary
                sampling_abis.update(summary.sampling_abis)
                provider_abis.update(summary.provider_abis)
                descendant_adapter_abis.update(summary.descendant_adapter_abis)
    if sampling_abis and sampling_abis != {_DISTRIBUTION_SAMPLING_ABI}:
        raise ValueError("stochastic plan requires an unsupported sampling ABI")
    return tuple(sorted(provider_abis)), tuple(sorted(descendant_adapter_abis))


def _json_or_repr(value: Any) -> Any:
    """*value* when JSON can hold it, and otherwise its ``repr``."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _is_canonical_abi_sequence(values: object) -> bool:
    if not isinstance(values, tuple):
        return False
    if any(not isinstance(value, str) or not value for value in values):
        return False
    return values == tuple(sorted(set(values)))


@contextmanager
def in_place_evaluation_point_scope(
    broker: _evaluation_keys._AutomaticKeyBroker,
) -> Generator[None, None, None]:
    """Run the enclosed evaluation in place as the one evaluation point of *broker*'s invocation.

    Inside the scope, each workflow-owned draw claims the next child occurrence
    of the invocation, in program order, as each draw of a plain call's
    evaluation point does. *broker* commits the invocation's own occurrence at
    the first such draw, so an evaluation that draws nothing commits none.
    Under a route probe or inside a JAX body the scope installs nothing, as no
    evaluation point's scope does there, so a draw claims its key from
    *broker* directly, which signals a probe before committing an occurrence.

    Parameters
    ----------
    broker : _AutomaticKeyBroker
        The broker of the invocation, installed in the active workflow frame.
    """
    if _context._workflow_side_effects_forbidden():
        yield
        return
    evaluation_points = issue_evaluation_points((plain_call_point_segment(),))
    broker.register_evaluation_points(evaluation_points)
    try:
        with _evaluation_point_scope(
            evaluation_points[0], _context._capture_active_workflow_frame(), broker
        ):
            yield
    finally:
        broker.cancel_unstarted_evaluation_points(evaluation_points)
        broker.assert_evaluation_points_joined(evaluation_points)


@contextmanager
def _evaluation_point_scope(
    evaluation_point: IssuedEvaluationPoint,
    parent_frame: _context._WorkflowFrame | None,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> Generator[None, None, None]:
    """Install the workflow frame and the attempt of one evaluation point under *parent_broker*.

    Inside, each workflow-owned draw claims the next child occurrence of
    *parent_broker*'s invocation. Without a parent frame nothing is installed,
    and without a parent broker only the frame is.
    """
    if parent_frame is None:
        yield
        return
    with _context._evaluation_point_frame_scope(
        parent_frame,
        evaluation_point.frame.point_segment,
    ):
        if parent_broker is None:
            yield
            return
        with _evaluation_keys._evaluation_point_stochastic_scope(
            parent_broker,
            evaluation_point.frame,
        ):
            yield


def _prefect_retry_policy() -> tuple[int, tuple[float, ...]]:
    """Return Prefect's configured default task retry policy."""
    try:
        from prefect.settings import get_current_settings
    except ImportError:
        return 0, ()

    task_settings = get_current_settings().tasks
    return (
        task_settings.default_retries,
        _normalize_prefect_retry_delays(task_settings.default_retry_delay_seconds),
    )


def _normalize_prefect_retry_delays(value: Any) -> tuple[float, ...]:
    """Normalize Prefect's scalar-or-list retry delay into one schedule."""
    if value is None:
        return ()
    if isinstance(value, bool):
        raise TypeError(
            "Prefect default_retry_delay_seconds must be a number, a list of numbers, or None"
        )
    if isinstance(value, (int, float)):
        return (float(value),)
    if not isinstance(value, list):
        raise TypeError(
            "Prefect default_retry_delay_seconds must be a number, a list of numbers, or None"
        )
    delays = []
    for index, delay in enumerate(value):
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError(
                f"Prefect default_retry_delay_seconds list item {index} must be a number"
            )
        delays.append(float(delay))
    return tuple(delays)


def _retry_delay_for_ordinal(delays: tuple[float, ...], ordinal: int) -> float:
    """Return a one-based retry delay, repeating the final configured value."""
    if not delays:
        return 0.0
    return delays[min(ordinal - 1, len(delays) - 1)]


def _ensure_prefect_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )


# ---------------------------------------------------------------------------
# The workflow scopes of a set of evaluation points (core._evaluation_points)
# ---------------------------------------------------------------------------


class _WorkflowEvaluationPointScopes:
    """The workflow scope of each evaluation point of one set, under the call that built the set.

    A ProbPipe draw made inside an evaluation point takes its random key from
    the evaluation point's segment of the workflow. The first time an
    evaluation point runs in a scope of its own, or is shipped, this issues
    every evaluation point of the set and registers them with the call's key
    broker, so a set that ``jax`` maps whole issues none. In the calling
    process an evaluation point runs in a workflow scope of its own under that
    broker. A chunk sent to another process reserves an attempt for each of
    its evaluation points with the broker, so a draw there takes the key it
    takes here, and the claims the worker makes return with the chunk's
    outcomes.

    Parameters
    ----------
    parent_broker : _AutomaticKeyBroker
        The key broker of the call that asks for the evaluation points.
    num_evaluation_points : int
        The size of the set.
    point_segment : callable
        The segment of the workflow of the evaluation point at a point index.
    """

    def __init__(
        self,
        parent_broker: _evaluation_keys._AutomaticKeyBroker,
        num_evaluation_points: int,
        point_segment: Callable[[int], Any],
    ) -> None:
        self._parent_frame = _context._capture_active_workflow_frame()
        self._parent_broker = parent_broker
        self._num_evaluation_points = num_evaluation_points
        self._point_segment = point_segment
        self._issuance_lock = threading.Lock()
        self._issued_points_or_none: tuple[IssuedEvaluationPoint, ...] | None = None
        self._attempts_lock = threading.Lock()
        self._shipping_lock = threading.Lock()
        self._outstanding_attempts: dict[int, EvaluationPointAttempt] = {}

    @property
    def _issued_points(self) -> tuple[IssuedEvaluationPoint, ...]:
        """Every evaluation point of the set, issued and registered with the broker on first use."""
        with self._issuance_lock:
            if self._issued_points_or_none is None:
                self._issued_points_or_none = issue_evaluation_points(
                    [
                        self._point_segment(point_index)
                        for point_index in range(self._num_evaluation_points)
                    ]
                )
                self._parent_broker.register_evaluation_points(self._issued_points_or_none)
            return self._issued_points_or_none

    @contextmanager
    def point_scope(self, point_index: int) -> Generator[None, None, None]:
        """The scope evaluation point *point_index* runs in, in the calling process.

        A reservation the evaluation point holds from a chunk sent elsewhere
        is released first, since the evaluation point now runs here.
        """
        self._release_outstanding_attempt(point_index)
        with _evaluation_point_scope(
            self._issued_points[point_index], self._parent_frame, self._parent_broker
        ):
            yield

    def vectorized_scope(self) -> AbstractContextManager[None]:
        """The scope one call on stacked evaluation points runs in, which refuses workflow draws."""
        return _context._workflow_jax_runtime_guard()

    def mapped_scope(self) -> AbstractContextManager[None]:
        """The scope a ``jax.vmap`` over the evaluation points runs in, which refuses workflow draws."""
        return _context._workflow_jax_runtime_guard()

    def shipped(self, point_indices: range) -> _ShippedEvaluationPointScopes:
        """Scopes for *point_indices* that pickle, each reserved as an attempt with the call's broker.

        Shipping an evaluation point again, as an executor that retries a
        chunk does, releases its earlier reservation, so the new attempt is a
        retry with the same keys.
        """
        reserved_points: dict[int, tuple[IssuedEvaluationPoint, EvaluationPointAttempt, Any]] = {}
        for point_index in point_indices:
            issued_point = self._issued_points[point_index]
            attempt = EvaluationPointAttempt.create(issued_point.frame.token)
            # One shipping at a time, so two threads pickling one chunk never
            # reserve an evaluation point between the other's release and record.
            with self._shipping_lock:
                self._release_outstanding_attempt(point_index)
                parent_envelope = self._parent_broker.reserve_remote_evaluation_point_attempt(
                    issued_point.frame, attempt
                )
                with self._attempts_lock:
                    self._outstanding_attempts[point_index] = attempt
            reserved_points[point_index] = (issued_point, attempt, parent_envelope)
        return _ShippedEvaluationPointScopes(reserved_points)

    def claim_reports(self) -> tuple[Any, ...]:
        """No reports: the evaluation points of the calling process claim with the broker directly."""
        return ()

    def accept_claim_reports(self, claim_reports: Iterable[EvaluationPointClaimReport]) -> None:
        """Add the claims that runs in other processes made to the call's record of draws.

        A report of an attempt that a later shipping replaced is ignored, since
        its reservation was released.
        """
        for claim_report in claim_reports:
            with self._attempts_lock:
                reported_point = next(
                    (
                        point_index
                        for point_index, attempt in self._outstanding_attempts.items()
                        if attempt.attempt_token == claim_report.attempt.attempt_token
                    ),
                    None,
                )
                if reported_point is None:
                    continue
                del self._outstanding_attempts[reported_point]
            self._parent_broker.accept_remote_claim_report(claim_report)

    def close(self) -> None:
        """Release every reservation no report answered, and check that every evaluation point joined."""
        if self._issued_points_or_none is None:
            return
        for point_index in list(self._outstanding_attempts):
            self._release_outstanding_attempt(point_index)
        self._parent_broker.cancel_unstarted_evaluation_points(self._issued_points)
        self._parent_broker.assert_evaluation_points_joined(self._issued_points)

    def _release_outstanding_attempt(self, point_index: int) -> None:
        """Release the reservation evaluation point *point_index* holds for a run elsewhere, if any."""
        with self._attempts_lock:
            outstanding_attempt = self._outstanding_attempts.pop(point_index, None)
        if outstanding_attempt is not None:
            self._parent_broker.abort_remote_evaluation_point_attempt(outstanding_attempt)


class _ShippedEvaluationPointScopes:
    """The workflow scopes of a chunk's evaluation points, as a worker process receives them.

    Each evaluation point runs in a frame rebuilt from its call's workflow
    and in the attempt its call reserved, and its claims are reported back
    with the chunk's outcomes.

    Parameters
    ----------
    reserved_points : Mapping of int to tuple
        For each point index, the issued evaluation point, its reserved
        attempt, and the envelope of its call's authority.
    """

    def __init__(
        self,
        reserved_points: Mapping[int, tuple[IssuedEvaluationPoint, EvaluationPointAttempt, Any]],
    ) -> None:
        self._reserved_points = dict(reserved_points)
        self._claim_reports: list[EvaluationPointClaimReport] = []

    @contextmanager
    def point_scope(self, point_index: int) -> Generator[None, None, None]:
        """The scope evaluation point *point_index* runs in, in the worker process."""
        _, attempt, parent_envelope = self._reserved_points[point_index]
        with _context._transported_workflow_frame(parent_envelope.root_words):
            remote_parent = None
            try:
                with _evaluation_keys._remote_evaluation_point_stochastic_scope(
                    parent_envelope, attempt
                ) as remote_parent:
                    yield
            finally:
                if remote_parent is not None:
                    self._claim_reports.append(remote_parent.report())

    def vectorized_scope(self) -> AbstractContextManager[None]:
        """The scope one call on stacked evaluation points runs in, which refuses workflow draws."""
        return _context._workflow_jax_runtime_guard()

    def mapped_scope(self) -> AbstractContextManager[None]:
        """The scope a ``jax.vmap`` over the evaluation points runs in, which refuses workflow draws."""
        return _context._workflow_jax_runtime_guard()

    def shipped(self, point_indices: range) -> _ShippedEvaluationPointScopes:
        """These scopes, cut to *point_indices*."""
        return _ShippedEvaluationPointScopes(
            {point_index: self._reserved_points[point_index] for point_index in point_indices}
        )

    def claim_reports(self) -> tuple[EvaluationPointClaimReport, ...]:
        """The claims the evaluation points made in this process, for the call to accept."""
        return tuple(self._claim_reports)

    def accept_claim_reports(self, claim_reports: Iterable[EvaluationPointClaimReport]) -> None:
        """Nothing to accept in a worker process."""
        return None

    def close(self) -> None:
        """Nothing to release in a worker process."""
        return None

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the reservations, without the reports, which a worker makes afresh."""
        return (_ShippedEvaluationPointScopes, (self._reserved_points,))


@contextmanager
def _requester_workflow_scope() -> Generator[None, None, None]:
    """The workflow position from which code asks for sets of evaluation points, as a lifted call does.

    Inside it a key broker of its own is active under the running call, as
    for a nested call, so a set of evaluation points built there issues its
    evaluation points' keys from it. A set built inside a function body
    without it is refused, since no key broker is active there.
    """
    with _context._ephemeral_workflow_run(), _evaluation_keys._function_stochastic_scope():
        yield


def _evaluation_point_scopes(
    origin: str,
    num_evaluation_points: int,
    set_position: int,
    point_segment: Callable[[int], Any] | None,
) -> _WorkflowEvaluationPointScopes | _UnscopedEvaluationPoints:
    """The workflow scopes of a new set of evaluation points, built under the running call.

    Without *point_segment*, each evaluation point takes the segment of a
    sweep cell at its set's position and its point index. Inside a route probe
    or a JAX body, where workflow draws are refused, the evaluation points run
    in no scope. Where no key broker is active, as inside a function's body or
    outside every ProbPipe call, it raises ``RuntimeError``.
    """
    if _context._workflow_side_effects_forbidden():
        return _UnscopedEvaluationPoints()
    parent_broker = _evaluation_keys._capture_active_broker()
    if parent_broker is None:
        raise RuntimeError(
            f"cannot build evaluation points of {origin!r} where no ProbPipe call gives them a "
            "place in the workflow, such as directly in a function's body. Ask for the "
            "evaluations with evaluate_at_points or through a ProbPipe call, such as a lifted call"
        )
    if point_segment is None:

        def point_segment(point_index: int) -> Any:
            return sweep_cell_point_segment((set_position, point_index))

    return _WorkflowEvaluationPointScopes(parent_broker, num_evaluation_points, point_segment)


_install_evaluation_point_scopes(_evaluation_point_scopes, _requester_workflow_scope)
