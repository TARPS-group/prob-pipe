"""Function execution dispatch helpers.

This private module owns ordered execution of plain call dictionaries.
It deliberately knows nothing about ProbPipe value semantics; callers
assemble inputs and interpret outputs outside this module.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from time import sleep
from typing import Any, Literal

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from . import _workflow_broker, _workflow_context, _workflow_execution_contract, _workflow_plan
from ._workflow_managed import (
    ManagedAttemptState,
    ManagedClaimReport,
    ManagedExecutionOutcome,
    ManagedPrefectPayload,
    ManagedWorkItem,
    _validated_managed_execution_outcome_snapshot,
    _validated_managed_prefect_payload_snapshot,
    _validated_managed_work_item_snapshot,
    lifted_evaluation_unit_segment,
    make_managed_work_items,
    point_unit_segment,
    sweep_unit_segment,
)

__all__ = (
    "lifted_evaluation_unit_segment",
    "make_managed_work_items",
    "point_unit_segment",
    "sweep_unit_segment",
)

WorkflowExecutionMode = Literal[
    "sequential",
    "thread",
    "prefect_task",
    "prefect_flow",
]


@dataclass(frozen=True, slots=True)
class WorkflowExecutionConfig:
    """Resolved execution settings for ordered workflow calls."""

    mode: WorkflowExecutionMode
    max_workers: int | None = None
    name: str = "workflow"
    prefect_task_runner: Any | None = None


@dataclass(frozen=True, slots=True)
class WorkflowExecutionRequest:
    """A backend-neutral request to execute ordered managed work items."""

    func: Callable[..., Any]
    work_items: tuple[ManagedWorkItem, ...]
    execution: WorkflowExecutionConfig
    contract: _workflow_execution_contract.WorkflowRngExecutionContract | None = None
    stochastic_plan: _workflow_plan.StochasticPlan | None = None


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
    if not request.work_items:
        return []
    request = replace(
        request,
        work_items=tuple(
            _validated_managed_work_item_snapshot(item) for item in request.work_items
        ),
    )
    contract = request.contract or _workflow_execution_contract.make_execution_contract(
        evaluator="rowwise",
        transport=_workflow_execution_contract.transport_for_execution_mode(request.execution.mode),
        stochastic_plan=request.stochastic_plan,
    )
    if not _workflow_execution_contract.supports_execution_contract(
        contract,
        request.stochastic_plan,
    ):
        raise RuntimeError("workflow execution route does not satisfy the RNG contract")
    if _workflow_context._workflow_side_effects_forbidden():
        return [request.func(**item.call_values()) for item in request.work_items]
    _workflow_broker._record_active_execution_contract(contract)
    parent_frame = _workflow_context._capture_active_workflow_frame()
    parent_broker = _workflow_broker._capture_active_broker()
    if parent_broker is not None:
        parent_broker.register_managed_work_items(request.work_items)

    try:
        match request.execution.mode:
            case "sequential":
                return [
                    _execute_work_item(
                        request.func,
                        item,
                        parent_frame,
                        parent_broker,
                    )
                    for item in request.work_items
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
            parent_broker.cancel_unstarted_managed_items(request.work_items)
            parent_broker.assert_managed_items_joined(request.work_items)


def execute_many_threaded(
    request: WorkflowExecutionRequest,
    *,
    parent_frame: _workflow_context._WorkflowFrame | None = None,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Execute call dictionaries through ``ThreadPoolExecutor``."""
    if not request.work_items:
        return []

    _workflow_context._guard_managed_submission()
    max_workers = _validate_max_workers(request.execution.max_workers)
    if parent_frame is None:
        parent_frame = _workflow_context._capture_active_workflow_frame()
    if parent_broker is None:
        parent_broker = _workflow_broker._capture_active_broker()
        if parent_broker is not None:
            parent_broker.register_managed_work_items(request.work_items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda item: _execute_work_item(
                    request.func,
                    item,
                    parent_frame,
                    parent_broker,
                ),
                request.work_items,
            )
        )


def map_task(
    request: WorkflowExecutionRequest,
    *,
    task_name: str | None = None,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Create a Prefect task, map keyword arguments over calls, and resolve futures."""
    if not request.work_items:
        return []

    _workflow_context._guard_managed_submission()
    _ensure_prefect_available()

    func = request.func

    @task(name=task_name or request.execution.name, retries=0)
    def run_func(payload):
        return _execute_prefect_payload(func, payload)

    payloads_by_index: dict[int, ManagedPrefectPayload] = {}
    try:
        for item in request.work_items:
            payloads_by_index[item.index] = _make_prefect_payload(
                item,
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
    payloads: list[ManagedPrefectPayload],
) -> list[ManagedExecutionOutcome]:
    """Submit one attempt for each payload and collect its managed outcome."""
    _workflow_context._guard_managed_submission()
    validated_payloads = [
        _validated_managed_prefect_payload_snapshot(payload) for payload in payloads
    ]
    payloads_by_index = {payload.item.index: payload for payload in validated_payloads}
    if len(payloads_by_index) != len(validated_payloads):
        raise RuntimeError("Prefect payloads contain duplicate managed outcome indexes")
    futures = list(run_func.map(payload=validated_payloads))
    if len(futures) != len(validated_payloads):
        raise RuntimeError("Prefect returned a different number of futures than payloads")
    outcomes = []
    errors = []
    for submitted_payload, future in zip(validated_payloads, futures, strict=True):
        try:
            outcome = _validated_managed_execution_outcome_snapshot(future.result())
            payload = payloads_by_index.get(outcome.index)
            if payload is None:
                raise RuntimeError(
                    "Prefect outcome index does not belong to the submitted payload batch"
                )
            _validate_prefect_outcome_for_payload(outcome, payload)
            outcomes.append(outcome)
        except BaseException as error:
            errors.append((submitted_payload.item.index, error))
    if errors:
        raise min(errors, key=lambda item: item[0])[1]
    if len({outcome.index for outcome in outcomes}) != len(outcomes):
        raise RuntimeError("Prefect returned duplicate managed outcome indexes")
    if {outcome.index for outcome in outcomes} != set(payloads_by_index):
        raise RuntimeError("Prefect outcomes do not cover the submitted payload batch")
    return outcomes


def _validate_prefect_outcome_for_payload(
    outcome: ManagedExecutionOutcome,
    payload: ManagedPrefectPayload,
) -> None:
    """Bind one deeply validated outcome to its original submitted authority."""
    if outcome.index != payload.item.index:
        raise RuntimeError("Prefect outcome index does not match its managed work item")
    report = outcome.report
    if report is None:
        raise RuntimeError("Prefect outcome must contain a managed claim report")
    if report.frame != payload.item.frame:
        raise RuntimeError("Prefect outcome report frame does not match its payload")
    if report.attempt != payload.attempt:
        raise RuntimeError("Prefect outcome report attempt does not match its payload")
    if not outcome.coordination_required:
        return
    if payload.parent is not None:
        raise RuntimeError("an authorized Prefect outcome cannot request coordination again")
    if (
        report.child_count != 0
        or report.effects
        or report.successful_effects
        or outcome.value is not None
    ):
        raise RuntimeError(
            "a rootless coordination outcome must contain an empty claim report and no value"
        )


def _make_prefect_payload(
    item: ManagedWorkItem,
    *,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
    parent_authority: bool,
) -> ManagedPrefectPayload:
    """Create and parent-reserve one fresh remote execution attempt."""
    attempt = ManagedAttemptState.create(item.frame.token)
    parent = None
    if parent_broker is not None:
        parent = parent_broker.reserve_remote_managed_attempt(
            item.frame,
            attempt,
            parent_authority=parent_authority,
        )
    elif parent_authority:
        raise RuntimeError("remote workflow randomness requires an active parent Function broker")
    return ManagedPrefectPayload(
        item=item,
        attempt=attempt,
        provenance_mode=_workflow_context._active_provenance_mode(),
        parent=parent,
    )


def _abort_prefect_payloads(
    payloads: tuple[ManagedPrefectPayload, ...],
    *,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
) -> None:
    """Release every current parent reservation after a transport failure."""
    if parent_broker is None:
        return
    for payload in payloads:
        parent_broker.abort_remote_managed_attempt(payload.attempt)


def _coordinate_prefect_randomness(
    run_func: Any,
    outcomes: list[ManagedExecutionOutcome],
    *,
    payloads_by_index: dict[int, ManagedPrefectPayload],
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
) -> list[ManagedExecutionOutcome]:
    """Re-submit remote units that lazily request parent RNG authority."""
    coordination = [outcome for outcome in outcomes if outcome.coordination_required]
    if not coordination:
        return outcomes
    if parent_broker is None:
        raise RuntimeError("remote workflow randomness requires an active parent Function broker")

    coordinated_payloads = []
    for outcome in coordination:
        payload = payloads_by_index[outcome.index]
        if payload.parent is not None:
            raise RuntimeError("coordinated Prefect attempt requested RNG authority twice")
        _accept_prefect_claim_report(outcome, parent_broker=parent_broker)
        payload = _make_prefect_payload(
            payload.item,
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
    outcomes: list[ManagedExecutionOutcome],
    *,
    payloads_by_index: dict[int, ManagedPrefectPayload],
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
) -> list[ManagedExecutionOutcome]:
    """Coordinate configured retries while preserving work-item ownership."""
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
            if payload.parent is not None:
                if parent_broker is None:
                    raise RuntimeError("coordinated Prefect retry lost its parent Function broker")
            payload = _make_prefect_payload(
                payload.item,
                parent_broker=parent_broker,
                parent_authority=payload.parent is not None,
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
    outcomes: list[ManagedExecutionOutcome],
    replacements: list[ManagedExecutionOutcome],
) -> list[ManagedExecutionOutcome]:
    """Replace selected outcomes without relying on remote completion order."""
    replacements_by_index = {outcome.index: outcome for outcome in replacements}
    return [replacements_by_index.get(outcome.index, outcome) for outcome in outcomes]


def _accept_prefect_claim_report(
    outcome: ManagedExecutionOutcome,
    *,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
) -> None:
    """Join one completed remote attempt before it is retried."""
    if parent_broker is None:
        return
    if outcome.report is None:
        raise RuntimeError("Prefect work item returned no managed claim report")
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
    parent_broker: _workflow_broker._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Use Prefect ``task.map()`` inside a lightweight flow."""
    if not request.work_items:
        return []

    _workflow_context._guard_managed_submission()
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
    parent_broker: _workflow_broker._AutomaticKeyBroker | None = None,
) -> list[Any]:
    """Wrap a mapped task inside a named Prefect flow."""
    if not request.work_items:
        return []

    _workflow_context._guard_managed_submission()
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


def _execute_work_item(
    func: Callable[..., Any],
    item: ManagedWorkItem,
    parent_frame: _workflow_context._WorkflowFrame | None = None,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None = None,
) -> Any:
    """Execute one frozen work item without changing its canonical identity."""
    if parent_frame is None:
        return func(**item.call_values())
    with _workflow_context._managed_work_item_scope(
        parent_frame,
        item.frame.unit_segment,
    ):
        if parent_broker is None:
            return func(**item.call_values())
        with _workflow_broker._managed_work_item_stochastic_scope(
            parent_broker,
            item.frame,
        ):
            return func(**item.call_values())


def _execute_prefect_payload(
    func: Callable[..., Any],
    payload: ManagedPrefectPayload,
) -> ManagedExecutionOutcome:
    """Execute one serializable Prefect payload and return its claim report."""
    payload = _validated_managed_prefect_payload_snapshot(payload)
    item = payload.item
    attempt = payload.attempt
    if payload.parent is None:
        with (
            _workflow_context._transported_workflow_frame(
                None,
                payload.provenance_mode,
            ),
            _workflow_broker._remote_coordination_probe_scope() as observation,
        ):
            value = None
            execution_error = None
            try:
                value = func(**item.call_values())
            except Exception as error:
                execution_error = error
        if observation.effect_observed:
            return ManagedExecutionOutcome(
                index=item.index,
                coordination_required=True,
                report=ManagedClaimReport(item.frame, attempt, 0),
            )
        if execution_error is not None:
            return ManagedExecutionOutcome(
                index=item.index,
                error=execution_error,
                report=ManagedClaimReport(item.frame, attempt, 0),
            )
        return ManagedExecutionOutcome(
            index=item.index,
            value=value,
            report=ManagedClaimReport(item.frame, attempt, 0),
        )

    with _workflow_context._transported_workflow_frame(
        payload.parent.root_words,
        payload.provenance_mode,
    ):
        remote_parent = None
        try:
            with _workflow_broker._remote_managed_work_item_stochastic_scope(
                payload.parent,
                attempt,
            ) as remote_parent:
                value = func(**item.call_values())
        except Exception as error:
            if remote_parent is None:
                raise
            return ManagedExecutionOutcome(
                index=item.index,
                error=error,
                report=remote_parent.report(),
            )
        return ManagedExecutionOutcome(
            index=item.index,
            value=value,
            report=remote_parent.report(),
        )


def _resolve_prefect_outcomes(
    outcomes: list[ManagedExecutionOutcome],
    *,
    parent_broker: _workflow_broker._AutomaticKeyBroker | None,
) -> list[Any]:
    """Join remote claims, restore canonical order, then raise the first error."""
    ordered = sorted(outcomes, key=lambda outcome: outcome.index)
    for outcome in ordered:
        if outcome.coordination_required:
            raise RuntimeError("coordinated Prefect attempt requested RNG authority twice")
        if parent_broker is not None:
            if outcome.report is None:
                raise RuntimeError("Prefect work item returned no managed claim report")
            parent_broker.accept_remote_claim_report(outcome.report)
    for outcome in ordered:
        if outcome.error is not None:
            raise outcome.error
    return [outcome.value for outcome in ordered]


def _ensure_prefect_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
