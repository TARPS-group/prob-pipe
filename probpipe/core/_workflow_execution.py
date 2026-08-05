"""Function execution dispatch helpers.

This private module owns ordered execution of plain call dictionaries.
It deliberately knows nothing about ProbPipe value semantics; callers
assemble inputs and interpret outputs outside this module.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import sleep
from typing import Any, Literal

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from . import _workflow_broker, _workflow_context, _workflow_execution_contract
from ._workflow_managed import (
    ManagedAttemptState,
    ManagedClaimReport,
    ManagedExecutionOutcome,
    ManagedPrefectPayload,
    ManagedWorkItem,
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


def execute_many(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute all call dictionaries using the configured dispatch mode."""
    if not request.work_items:
        return []
    if _workflow_context._workflow_side_effects_forbidden():
        return [request.func(**item.call_values()) for item in request.work_items]
    contract = request.contract or _workflow_execution_contract.make_execution_contract(
        evaluator="rowwise",
        transport=_workflow_execution_contract.transport_for_execution_mode(request.execution.mode),
        stochastic_plan=None,
    )
    if not _workflow_execution_contract.supports_execution_contract(contract, None):
        raise RuntimeError("workflow execution route does not satisfy the RNG contract")
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

    @task(name=task_name or request.execution.name)
    def run_func(payload):
        return _execute_prefect_payload(func, payload)

    payloads_by_index = {
        item.index: ManagedPrefectPayload(item=item) for item in request.work_items
    }
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


def _run_prefect_payloads(
    run_func: Any,
    payloads: list[ManagedPrefectPayload],
) -> list[ManagedExecutionOutcome]:
    """Submit one attempt for each payload and collect its managed outcome."""
    _workflow_context._guard_managed_submission()
    return [future.result() for future in run_func.map(payload=payloads)]


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
        payload = ManagedPrefectPayload(
            item=payload.item,
            parent=parent_broker.prepare_remote_managed_unit(payload.item.frame),
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
    max_retries, retry_delay = _prefect_retry_policy()
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
                payload = ManagedPrefectPayload(
                    item=payload.item,
                    parent=parent_broker.prepare_remote_managed_unit(payload.item.frame),
                )
                payloads_by_index[outcome.index] = payload
            retry_payloads.append(payload)

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


def _prefect_retry_policy() -> tuple[int, float]:
    """Return Prefect's configured default task retry policy."""
    try:
        from prefect.settings import get_current_settings
    except ImportError:
        return 0, 0.0

    task_settings = get_current_settings().tasks
    return (
        task_settings.default_retries,
        task_settings.default_retry_delay_seconds,
    )


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
    item = payload.item
    attempt = ManagedAttemptState.create(item.frame.token)
    if payload.parent is None:
        with (
            _workflow_context._transported_workflow_frame(None),
            _workflow_broker._remote_coordination_probe_scope(),
        ):
            try:
                value = func(**item.call_values())
            except _workflow_broker._ManagedCoordinationRequired:
                return ManagedExecutionOutcome(
                    index=item.index,
                    coordination_required=True,
                )
            except Exception as error:
                return ManagedExecutionOutcome(
                    index=item.index,
                    error=error,
                    report=ManagedClaimReport(item.frame, attempt, 0),
                )
        return ManagedExecutionOutcome(
            index=item.index,
            value=value,
            report=ManagedClaimReport(item.frame, attempt, 0),
        )

    with _workflow_context._transported_workflow_frame(payload.parent.root_words):
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
