"""Function execution dispatch helpers.

This private module owns ordered execution of plain call dictionaries.
It knows nothing about ProbPipe value semantics: callers assemble inputs
and interpret outputs outside this module. Each worker runs the work
function under the caller's workflow frame and under the controls that a
derived route passes down.

The module also holds the versioned execution contract of each route: its
evaluator, its transport, and the random-key ABIs it requires, which a
replay record stores and a replay checks.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import sleep
from typing import Any, Literal

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from ..core._call_context import _carrying_call_context
from ..core.config import WorkflowKind
from . import _context, _evaluation_keys
from ._evaluation_keys import (
    _DISTRIBUTION_SAMPLING_ABI,
    _PROBPIPE_DISTRIBUTION_PROVIDER_ABI,
    EvaluationPointAttempt,
    EvaluationPointClaimReport,
    EvaluationPointOutcome,
    EvaluationPointPrefectPayload,
    IssuedEvaluationPoint,
    _validated_evaluation_point_outcome_snapshot,
    _validated_evaluation_point_prefect_payload_snapshot,
    _validated_issued_evaluation_point_snapshot,
    issue_evaluation_points,
    lifted_evaluation_point_segment,
    plain_call_point_segment,
    sweep_cell_point_segment,
)
from ._plan import StochasticPlan

__all__ = (
    "issue_evaluation_points",
    "lifted_evaluation_point_segment",
    "plain_call_point_segment",
    "sweep_cell_point_segment",
)

WorkflowExecutionMode = Literal[
    "sequential",
    "thread",
    "prefect_task",
    "prefect_flow",
]


@dataclass(frozen=True)
class WorkflowExecutionConfig:
    """Resolved execution settings for ordered workflow calls."""

    mode: WorkflowExecutionMode
    max_workers: int | None = None
    name: str = "workflow"
    prefect_task_runner: Any | None = None


@dataclass(frozen=True)
class WorkflowExecutionRequest:
    """A backend-neutral request to execute ordered evaluation points."""

    func: Callable[..., Any]
    evaluation_points: tuple[IssuedEvaluationPoint, ...]
    execution: WorkflowExecutionConfig
    contract: WorkflowRngExecutionContract | None = None
    stochastic_plan: StochasticPlan | None = None


# -- The execution contract --------------------------------------------------

_EXECUTION_CONTRACT_ABI = "probpipe.workflow_rng_execution/v1"
_RNG_ABI = "ProbPipe-RNG-v1"
_DESCENDANT_ADAPTER_ABI = "probpipe.transformed_descendant/v1"
_DESCENDANT_PROVIDER_ABI = "tensorflow_probability.substrates.jax.bijector.forward/v1"
_JAX_KEY_ABI = "jax.random.wrap_key_data/threefry2x32/v1"

WorkflowEvaluator = Literal["rowwise", "jax_vmap"]
WorkflowTransport = Literal[
    "local_inline",
    "local_thread",
    "prefect_task",
    "prefect_flow",
]

_SUPPORTED_ROUTES: frozenset[tuple[WorkflowEvaluator, WorkflowTransport]] = frozenset(
    {
        ("rowwise", "local_inline"),
        ("rowwise", "local_thread"),
        ("rowwise", "prefect_task"),
        ("rowwise", "prefect_flow"),
        ("jax_vmap", "local_inline"),
        ("jax_vmap", "prefect_task"),
        ("jax_vmap", "prefect_flow"),
    }
)


@dataclass(frozen=True)
class WorkflowRngExecutionContract:
    """Canonical evaluator/transport and stochastic ABI requirements."""

    abi: str
    evaluator: WorkflowEvaluator
    transport: WorkflowTransport
    rng_abi: str
    sampling_abi: str
    provider_abis: tuple[str, ...]
    descendant_adapter_abis: tuple[str, ...]
    jax_key_abi: str
    plan_evaluation_mode: str | None


def make_execution_contract(
    *,
    evaluator: WorkflowEvaluator,
    transport: WorkflowTransport,
    stochastic_plan: StochasticPlan | None,
) -> WorkflowRngExecutionContract:
    """Build the exact execution contract for one planned route."""
    provider_abis, descendant_adapter_abis = _stochastic_plan_abis(stochastic_plan)
    return WorkflowRngExecutionContract(
        abi=_EXECUTION_CONTRACT_ABI,
        evaluator=evaluator,
        transport=transport,
        rng_abi=_RNG_ABI,
        sampling_abi=_DISTRIBUTION_SAMPLING_ABI,
        provider_abis=provider_abis,
        descendant_adapter_abis=descendant_adapter_abis,
        jax_key_abi=_JAX_KEY_ABI,
        plan_evaluation_mode=(None if stochastic_plan is None else stochastic_plan.evaluation_mode),
    )


def supports_execution_contract(
    contract: WorkflowRngExecutionContract,
    stochastic_plan: StochasticPlan | None,
    *,
    jax_structure_supported: bool = True,
) -> bool:
    """Return whether one route satisfies all versioned RNG capabilities."""
    if not isinstance(contract, WorkflowRngExecutionContract):
        return False
    if (
        contract.abi != _EXECUTION_CONTRACT_ABI
        or contract.rng_abi != _RNG_ABI
        or contract.sampling_abi != _DISTRIBUTION_SAMPLING_ABI
        or contract.jax_key_abi != _JAX_KEY_ABI
        or not isinstance(contract.evaluator, str)
        or not isinstance(contract.transport, str)
        or (contract.evaluator, contract.transport) not in _SUPPORTED_ROUTES
    ):
        return False
    if not _is_canonical_abi_sequence(contract.provider_abis):
        return False
    if not _is_canonical_abi_sequence(contract.descendant_adapter_abis):
        return False
    if any(
        provider not in (_PROBPIPE_DISTRIBUTION_PROVIDER_ABI, _DESCENDANT_PROVIDER_ABI)
        for provider in contract.provider_abis
    ):
        return False
    if any(adapter != _DESCENDANT_ADAPTER_ABI for adapter in contract.descendant_adapter_abis):
        return False
    has_descendant_provider = _DESCENDANT_PROVIDER_ABI in contract.provider_abis
    has_descendant_adapter = _DESCENDANT_ADAPTER_ABI in contract.descendant_adapter_abis
    if has_descendant_provider != has_descendant_adapter:
        return False
    try:
        expected = make_execution_contract(
            evaluator=contract.evaluator,
            transport=contract.transport,
            stochastic_plan=stochastic_plan,
        )
    except (TypeError, ValueError):
        return False
    if contract != expected:
        return False
    return contract.evaluator != "jax_vmap" or jax_structure_supported


def execution_contract_abi() -> str:
    """Return the immutable route-neutral execution contract ABI."""
    return _EXECUTION_CONTRACT_ABI


def key_adapter_abi() -> str:
    """Return the immutable JAX key adapter ABI used by admitted routes."""
    return _JAX_KEY_ABI


def transport_for_execution_mode(mode: str) -> WorkflowTransport:
    """Map row-wise execution configuration to its canonical transport."""
    match mode:
        case "sequential":
            return "local_inline"
        case "thread":
            return "local_thread"
        case "prefect_task":
            return "prefect_task"
        case "prefect_flow":
            return "prefect_flow"
        case _:
            raise ValueError(f"unsupported workflow execution mode: {mode!r}")


def transport_for_workflow_kind(kind: WorkflowKind) -> WorkflowTransport:
    """Map a JAX evaluator's orchestration mode to canonical transport."""
    match kind:
        case WorkflowKind.OFF:
            return "local_inline"
        case WorkflowKind.TASK:
            return "prefect_task"
        case WorkflowKind.FLOW:
            return "prefect_flow"
        case _:
            raise ValueError(f"transport requires a resolved workflow kind; got {kind!r}")


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


def _is_canonical_abi_sequence(values: object) -> bool:
    if not isinstance(values, tuple):
        return False
    if any(not isinstance(value, str) or not value for value in values):
        return False
    return values == tuple(sorted(set(values)))


def _preflight_execution_config(execution: WorkflowExecutionConfig) -> None:
    """Validate a row-wise execution route before stochastic work starts."""
    match execution.mode:
        case "sequential":
            return
        case "thread":
            _validate_max_workers(execution.max_workers)
        case "prefect_task" | "prefect_flow":
            _ensure_prefect_available()
            runner = execution.prefect_task_runner
            flow_name = (
                f"{execution.name}_map" if execution.mode == "prefect_task" else execution.name
            )
            task_name = (
                execution.name if execution.mode == "prefect_task" else f"{execution.name}_run"
            )
            flow(
                name=flow_name,
                **({"task_runner": runner} if runner is not None else {}),
            )(lambda: None)
            task(name=task_name, retries=0)(lambda payload: payload)
        case unknown:
            raise ValueError(f"Unknown workflow execution mode: {unknown!r}")


def execute_many(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute all call dictionaries using the configured dispatch mode."""
    if not request.evaluation_points:
        return []
    request = replace(
        request,
        evaluation_points=tuple(
            _validated_issued_evaluation_point_snapshot(evaluation_point)
            for evaluation_point in request.evaluation_points
        ),
    )
    contract = request.contract or make_execution_contract(
        evaluator="rowwise",
        transport=transport_for_execution_mode(request.execution.mode),
        stochastic_plan=request.stochastic_plan,
    )
    if not supports_execution_contract(
        contract,
        request.stochastic_plan,
    ):
        raise RuntimeError("workflow execution route does not satisfy the RNG contract")
    if _context._workflow_side_effects_forbidden():
        return [
            request.func(**evaluation_point.call_arguments())
            for evaluation_point in request.evaluation_points
        ]
    _evaluation_keys._record_active_execution_contract(contract)
    # A worker starts without the caller's context, so the work function carries
    # the call context.
    request = replace(request, func=_carrying_call_context(request.func))
    parent_frame = _context._capture_active_workflow_frame()
    parent_broker = _evaluation_keys._capture_active_broker()
    if parent_broker is not None:
        parent_broker.register_evaluation_points(request.evaluation_points)

    try:
        match request.execution.mode:
            case "sequential":
                return [
                    _execute_evaluation_point(
                        request.func,
                        evaluation_point,
                        parent_frame,
                        parent_broker,
                    )
                    for evaluation_point in request.evaluation_points
                ]
            case "thread":
                return execute_many_threaded(
                    request,
                    parent_frame=parent_frame,
                    parent_broker=parent_broker,
                )
            case "prefect_task":
                return execute_many_prefect_task(
                    request,
                    parent_broker=parent_broker,
                )
            case "prefect_flow":
                return execute_many_prefect_flow(
                    request,
                    parent_broker=parent_broker,
                )
            case unknown:
                raise ValueError(f"Unknown workflow execution mode: {unknown!r}")
    finally:
        if parent_broker is not None:
            parent_broker.cancel_unstarted_evaluation_points(request.evaluation_points)
            parent_broker.assert_evaluation_points_joined(request.evaluation_points)


@contextmanager
def in_place_evaluation_point_scope(
    broker: _evaluation_keys._AutomaticKeyBroker,
) -> Generator[None, None, None]:
    """Run the enclosed evaluation in place as the one evaluation point of *broker*'s invocation.

    Inside the scope, each workflow-owned draw claims the next child occurrence
    of the invocation, in program order, as each draw of a plain call's
    evaluation point does under :func:`execute_many`. *broker* commits the invocation's own
    occurrence at the first such draw, so an evaluation that draws nothing
    commits none. Under a route probe or inside a JAX body the scope installs
    nothing, as :func:`execute_many` installs nothing there, so a draw claims
    its key from *broker* directly, which signals a probe before committing an
    occurrence.

    Parameters
    ----------
    broker : _AutomaticKeyBroker
        The broker of the invocation, installed in the active workflow frame.
    """
    if _context._workflow_side_effects_forbidden():
        yield
        return
    evaluation_points = issue_evaluation_points([{}], point_segments=(plain_call_point_segment(),))
    broker.register_evaluation_points(evaluation_points)
    try:
        with _evaluation_point_scope(
            evaluation_points[0], _context._capture_active_workflow_frame(), broker
        ):
            yield
    finally:
        broker.cancel_unstarted_evaluation_points(evaluation_points)
        broker.assert_evaluation_points_joined(evaluation_points)


def execute_many_threaded(
    request: WorkflowExecutionRequest,
    *,
    parent_frame: _context._WorkflowFrame | None = None,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Execute call dictionaries through ``ThreadPoolExecutor``."""
    if not request.evaluation_points:
        return []

    _context._guard_evaluation_point_submission()
    max_workers = _validate_max_workers(request.execution.max_workers)
    if parent_frame is None:
        parent_frame = _context._capture_active_workflow_frame()
    if parent_broker is None:
        parent_broker = _evaluation_keys._capture_active_broker()
        if parent_broker is not None:
            parent_broker.register_evaluation_points(request.evaluation_points)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda evaluation_point: _execute_evaluation_point(
                    request.func,
                    evaluation_point,
                    parent_frame,
                    parent_broker,
                ),
                request.evaluation_points,
            )
        )


def map_task(
    request: WorkflowExecutionRequest,
    *,
    task_name: str | None = None,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Create a Prefect task, map keyword arguments over calls, and resolve futures."""
    if not request.evaluation_points:
        return []

    _context._guard_evaluation_point_submission()
    _ensure_prefect_available()

    func = request.func

    @task(name=task_name or request.execution.name, retries=0)
    def run_func(payload):
        return _execute_prefect_payload(func, payload)

    payloads_by_index: dict[int, EvaluationPointPrefectPayload] = {}
    try:
        for evaluation_point in request.evaluation_points:
            payloads_by_index[evaluation_point.index] = _make_prefect_payload(
                evaluation_point,
                parent_broker=parent_broker,
                parent_authority=False,
            )
        outcomes = _run_prefect_payloads(run_func, list(payloads_by_index.values()))
        outcomes = _coordinate_prefect_randomness(
            run_func,
            outcomes,
            payloads_by_index=payloads_by_index,
            parent_broker=parent_broker,
        )
        outcomes = _retry_prefect_failures(
            run_func,
            outcomes,
            payloads_by_index=payloads_by_index,
            parent_broker=parent_broker,
        )

        return _resolve_prefect_outcomes(outcomes, parent_broker=parent_broker)
    except BaseException:
        _abort_prefect_payloads(
            tuple(payloads_by_index.values()),
            parent_broker=parent_broker,
        )
        raise


def _run_prefect_payloads(
    run_func: Any,
    payloads: list[EvaluationPointPrefectPayload],
) -> list[EvaluationPointOutcome]:
    """Submit one attempt for each payload and collect its evaluation point outcome."""
    _context._guard_evaluation_point_submission()
    validated_payloads = [
        _validated_evaluation_point_prefect_payload_snapshot(payload) for payload in payloads
    ]
    payloads_by_index = {payload.evaluation_point.index: payload for payload in validated_payloads}
    if len(payloads_by_index) != len(validated_payloads):
        raise RuntimeError("Prefect payloads contain duplicate evaluation point outcome indexes")
    futures = list(run_func.map(payload=validated_payloads))
    if len(futures) != len(validated_payloads):
        raise RuntimeError("Prefect returned a different number of futures than payloads")
    outcomes = []
    errors = []
    for submitted_payload, future in zip(validated_payloads, futures, strict=True):
        try:
            outcome = _validated_evaluation_point_outcome_snapshot(future.result())
            payload = payloads_by_index.get(outcome.index)
            if payload is None:
                raise RuntimeError(
                    "Prefect outcome index does not belong to the submitted payload batch"
                )
            _validate_prefect_outcome_for_payload(outcome, payload)
            outcomes.append(outcome)
        except BaseException as error:
            errors.append((submitted_payload.evaluation_point.index, error))
    if errors:
        raise min(errors, key=lambda indexed_error: indexed_error[0])[1]
    if len({outcome.index for outcome in outcomes}) != len(outcomes):
        raise RuntimeError("Prefect returned duplicate evaluation point outcome indexes")
    if {outcome.index for outcome in outcomes} != set(payloads_by_index):
        raise RuntimeError("Prefect outcomes do not cover the submitted payload batch")
    return outcomes


def _validate_prefect_outcome_for_payload(
    outcome: EvaluationPointOutcome,
    payload: EvaluationPointPrefectPayload,
) -> None:
    """Bind one deeply validated outcome to its original submitted authority."""
    if outcome.index != payload.evaluation_point.index:
        raise RuntimeError("Prefect outcome index does not match its evaluation point")
    report = outcome.report
    if report is None:
        raise RuntimeError("Prefect outcome must contain an evaluation point claim report")
    if report.frame != payload.evaluation_point.frame:
        raise RuntimeError("Prefect outcome report frame does not match its payload")
    if report.attempt != payload.attempt:
        raise RuntimeError("Prefect outcome report attempt does not match its payload")
    if not outcome.coordination_required:
        return
    if payload.parent_envelope is not None:
        raise RuntimeError("an authorized Prefect outcome cannot request coordination again")
    if (
        report.child_count != 0
        or report.effects
        or report.successful_effects
        or outcome.output is not None
    ):
        raise RuntimeError(
            "a rootless coordination outcome must contain an empty claim report and no value"
        )


def _make_prefect_payload(
    evaluation_point: IssuedEvaluationPoint,
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
    parent_authority: bool,
) -> EvaluationPointPrefectPayload:
    """Create and parent-reserve one fresh remote execution attempt."""
    attempt = EvaluationPointAttempt.create(evaluation_point.frame.token)
    parent_envelope = None
    if parent_broker is not None:
        parent_envelope = parent_broker.reserve_remote_evaluation_point_attempt(
            evaluation_point.frame,
            attempt,
            parent_authority=parent_authority,
        )
    elif parent_authority:
        raise RuntimeError("remote workflow randomness requires an active parent Function broker")
    return EvaluationPointPrefectPayload(
        evaluation_point=evaluation_point,
        attempt=attempt,
        provenance_mode=_context._active_provenance_mode(),
        parent_envelope=parent_envelope,
    )


def _abort_prefect_payloads(
    payloads: tuple[EvaluationPointPrefectPayload, ...],
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> None:
    """Release every current parent reservation after a transport failure."""
    if parent_broker is None:
        return
    for payload in payloads:
        parent_broker.abort_remote_evaluation_point_attempt(payload.attempt)


def _coordinate_prefect_randomness(
    run_func: Any,
    outcomes: list[EvaluationPointOutcome],
    *,
    payloads_by_index: dict[int, EvaluationPointPrefectPayload],
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> list[EvaluationPointOutcome]:
    """Re-submit remote evaluation points that lazily request parent RNG authority."""
    coordination = [outcome for outcome in outcomes if outcome.coordination_required]
    if not coordination:
        return outcomes
    if parent_broker is None:
        raise RuntimeError("remote workflow randomness requires an active parent Function broker")

    coordinated_payloads = []
    for outcome in coordination:
        payload = payloads_by_index[outcome.index]
        if payload.parent_envelope is not None:
            raise RuntimeError("coordinated Prefect attempt requested RNG authority twice")
        _accept_prefect_claim_report(outcome, parent_broker=parent_broker)
        payload = _make_prefect_payload(
            payload.evaluation_point,
            parent_broker=parent_broker,
            parent_authority=True,
        )
        payloads_by_index[outcome.index] = payload
        coordinated_payloads.append(payload)

    coordinated_outcomes = _run_prefect_payloads(run_func, coordinated_payloads)
    if any(outcome.coordination_required for outcome in coordinated_outcomes):
        raise RuntimeError("coordinated Prefect attempt requested RNG authority twice")
    return _replace_prefect_outcomes(outcomes, coordinated_outcomes)


def _retry_prefect_failures(
    run_func: Any,
    outcomes: list[EvaluationPointOutcome],
    *,
    payloads_by_index: dict[int, EvaluationPointPrefectPayload],
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> list[EvaluationPointOutcome]:
    """Coordinate configured retries while preserving evaluation point ownership."""
    max_retries, retry_delays = _prefect_retry_policy()
    retries_by_index = dict.fromkeys(payloads_by_index, 0)

    while True:
        retryable = [
            outcome
            for outcome in sorted(outcomes, key=lambda candidate: candidate.index)
            if outcome.error is not None and retries_by_index[outcome.index] < max_retries
        ]
        if not retryable:
            return outcomes

        retry_payloads = []
        for outcome in retryable:
            _accept_prefect_claim_report(outcome, parent_broker=parent_broker)
            retries_by_index[outcome.index] += 1
            payload = payloads_by_index[outcome.index]
            if payload.parent_envelope is not None:
                if parent_broker is None:
                    raise RuntimeError("coordinated Prefect retry lost its parent Function broker")
            payload = _make_prefect_payload(
                payload.evaluation_point,
                parent_broker=parent_broker,
                parent_authority=payload.parent_envelope is not None,
            )
            payloads_by_index[outcome.index] = payload
            retry_payloads.append(payload)

        retry_ordinal = max(retries_by_index[outcome.index] for outcome in retryable)
        retry_delay = _retry_delay_for_ordinal(retry_delays, retry_ordinal)
        if retry_delay > 0:
            sleep(retry_delay)
        retried = _run_prefect_payloads(run_func, retry_payloads)
        retried = _coordinate_prefect_randomness(
            run_func,
            retried,
            payloads_by_index=payloads_by_index,
            parent_broker=parent_broker,
        )
        outcomes = _replace_prefect_outcomes(outcomes, retried)


def _replace_prefect_outcomes(
    outcomes: list[EvaluationPointOutcome],
    replacements: list[EvaluationPointOutcome],
) -> list[EvaluationPointOutcome]:
    """Replace selected outcomes without relying on remote completion order."""
    replacements_by_index = {outcome.index: outcome for outcome in replacements}
    return [replacements_by_index.get(outcome.index, outcome) for outcome in outcomes]


def _accept_prefect_claim_report(
    outcome: EvaluationPointOutcome,
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> None:
    """Join one completed remote attempt before it is retried."""
    if parent_broker is None:
        return
    if outcome.report is None:
        raise RuntimeError("Prefect evaluation point returned no evaluation point claim report")
    parent_broker.accept_remote_claim_report(outcome.report)


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


def execute_many_prefect_task(
    request: WorkflowExecutionRequest,
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Use Prefect ``task.map()`` inside a lightweight flow."""
    if not request.evaluation_points:
        return []

    _context._guard_evaluation_point_submission()
    _ensure_prefect_available()
    runner = request.execution.prefect_task_runner

    @flow(
        name=f"{request.execution.name}_map",
        **({"task_runner": runner} if runner is not None else {}),
    )
    def _task_map_flow():
        return map_task(request, parent_broker=parent_broker)

    return _task_map_flow()


def execute_many_prefect_flow(
    request: WorkflowExecutionRequest,
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Wrap a mapped task inside a named Prefect flow."""
    if not request.evaluation_points:
        return []

    _context._guard_evaluation_point_submission()
    _ensure_prefect_available()
    runner = request.execution.prefect_task_runner

    @flow(
        name=request.execution.name,
        **({"task_runner": runner} if runner is not None else {}),
    )
    def mapped_flow():
        return map_task(
            request,
            task_name=f"{request.execution.name}_run",
            parent_broker=parent_broker,
        )

    return mapped_flow()


def _validate_max_workers(max_workers: int | None) -> int | None:
    if max_workers is None:
        return None

    if not isinstance(max_workers, int):
        raise TypeError(f"max_workers must be None or a positive int; got {max_workers!r}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be None or a positive int; got {max_workers!r}")
    return max_workers


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


def _execute_evaluation_point(
    func: Callable[..., Any],
    evaluation_point: IssuedEvaluationPoint,
    parent_frame: _context._WorkflowFrame | None = None,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None = None,
) -> Any:
    """Execute one frozen evaluation point without changing its canonical identity."""
    with _evaluation_point_scope(evaluation_point, parent_frame, parent_broker):
        return func(**evaluation_point.call_arguments())


def _execute_prefect_payload(
    func: Callable[..., Any],
    payload: EvaluationPointPrefectPayload,
) -> EvaluationPointOutcome:
    """Execute one serializable Prefect payload and return its claim report."""
    payload = _validated_evaluation_point_prefect_payload_snapshot(payload)
    evaluation_point = payload.evaluation_point
    attempt = payload.attempt
    if payload.parent_envelope is None:
        with (
            _context._transported_workflow_frame(
                None,
                payload.provenance_mode,
            ),
            _evaluation_keys._remote_coordination_probe_scope() as observation,
        ):
            output = None
            execution_error = None
            try:
                output = func(**evaluation_point.call_arguments())
            except Exception as error:
                execution_error = error
        if observation.effect_observed:
            return EvaluationPointOutcome(
                index=evaluation_point.index,
                coordination_required=True,
                report=EvaluationPointClaimReport(evaluation_point.frame, attempt, 0),
            )
        if execution_error is not None:
            return EvaluationPointOutcome(
                index=evaluation_point.index,
                error=execution_error,
                report=EvaluationPointClaimReport(evaluation_point.frame, attempt, 0),
            )
        return EvaluationPointOutcome(
            index=evaluation_point.index,
            output=output,
            report=EvaluationPointClaimReport(evaluation_point.frame, attempt, 0),
        )

    with _context._transported_workflow_frame(
        payload.parent_envelope.root_words,
        payload.provenance_mode,
    ):
        remote_parent = None
        try:
            with _evaluation_keys._remote_evaluation_point_stochastic_scope(
                payload.parent_envelope,
                attempt,
            ) as remote_parent:
                output = func(**evaluation_point.call_arguments())
        except Exception as error:
            if remote_parent is None:
                raise
            return EvaluationPointOutcome(
                index=evaluation_point.index,
                error=error,
                report=remote_parent.report(),
            )
        return EvaluationPointOutcome(
            index=evaluation_point.index,
            output=output,
            report=remote_parent.report(),
        )


def _resolve_prefect_outcomes(
    outcomes: list[EvaluationPointOutcome],
    *,
    parent_broker: _evaluation_keys._AutomaticKeyBroker | None,
) -> list[Any]:
    """Join remote claims, restore canonical order, then raise the first error."""
    ordered = sorted(outcomes, key=lambda outcome: outcome.index)
    for outcome in ordered:
        if outcome.coordination_required:
            raise RuntimeError("coordinated Prefect attempt requested RNG authority twice")
        if parent_broker is not None:
            if outcome.report is None:
                raise RuntimeError(
                    "Prefect evaluation point returned no evaluation point claim report"
                )
            parent_broker.accept_remote_claim_report(outcome.report)
    for outcome in ordered:
        if outcome.error is not None:
            raise outcome.error
    return [outcome.output for outcome in ordered]


def _ensure_prefect_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
