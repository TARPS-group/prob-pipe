"""Function execution dispatch helpers.

This private module owns ordered execution of plain call dictionaries.
It deliberately knows nothing about ProbPipe value semantics; callers
assemble inputs and interpret outputs outside this module.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

try:
    from prefect import flow, task
except ImportError:
    task = flow = None

from . import _workflow_context
from ._workflow_managed import (
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


@dataclass(frozen=True)
class WorkflowExecutionConfig:
    """Resolved execution settings for ordered workflow calls."""

    mode: WorkflowExecutionMode
    max_workers: int | None = None
    name: str = "workflow"
    prefect_task_runner: Any | None = None


@dataclass(frozen=True)
class WorkflowExecutionRequest:
    """A backend-neutral request to execute ordered managed work items."""

    func: Callable[..., Any]
    work_items: tuple[ManagedWorkItem, ...]
    execution: WorkflowExecutionConfig


def execute_many(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute all call dictionaries using the configured dispatch mode."""
    if not request.work_items:
        return []
    parent_frame = _workflow_context._capture_active_workflow_frame()

    match request.execution.mode:
        case "sequential":
            return [
                _execute_work_item(request.func, item, parent_frame) for item in request.work_items
            ]
        case "thread":
            return execute_many_threaded(request, parent_frame=parent_frame)
        case "prefect_task":
            return execute_many_prefect_task(request)
        case "prefect_flow":
            return execute_many_prefect_flow(request)
        case unknown:
            raise ValueError(f"Unknown workflow execution mode: {unknown!r}")


def execute_many_threaded(
    request: WorkflowExecutionRequest,
    *,
    parent_frame: _workflow_context._WorkflowFrame | None = None,
) -> list[Any]:
    """Execute call dictionaries through ``ThreadPoolExecutor``."""
    if not request.work_items:
        return []

    max_workers = _validate_max_workers(request.execution.max_workers)
    if parent_frame is None:
        parent_frame = _workflow_context._capture_active_workflow_frame()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(
            pool.map(
                lambda item: _execute_work_item(request.func, item, parent_frame),
                request.work_items,
            )
        )


def map_task(
    request: WorkflowExecutionRequest,
    *,
    task_name: str | None = None,
) -> list[Any]:
    """Create a Prefect task, map keyword arguments over calls, and resolve futures."""
    if not request.work_items:
        return []

    _ensure_prefect_available()

    func = request.func

    @task(name=task_name or request.execution.name)
    def run_func(work_item):
        return _execute_work_item(func, work_item)

    futures = run_func.map(work_item=list(request.work_items))
    return [future.result() for future in futures]


def execute_many_prefect_task(request: WorkflowExecutionRequest) -> list[Any]:
    """Use Prefect ``task.map()`` inside a lightweight flow."""
    if not request.work_items:
        return []

    _ensure_prefect_available()
    runner = request.execution.prefect_task_runner

    @flow(
        name=f"{request.execution.name}_map",
        **({"task_runner": runner} if runner is not None else {}),
    )
    def _task_map_flow():
        return map_task(request)

    return _task_map_flow()


def execute_many_prefect_flow(request: WorkflowExecutionRequest) -> list[Any]:
    """Wrap a mapped task inside a named Prefect flow."""
    if not request.work_items:
        return []

    _ensure_prefect_available()
    runner = request.execution.prefect_task_runner

    @flow(
        name=request.execution.name,
        **({"task_runner": runner} if runner is not None else {}),
    )
    def mapped_flow():
        return map_task(request, task_name=f"{request.execution.name}_run")

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
) -> Any:
    """Execute one frozen work item without changing its canonical identity."""
    if parent_frame is None:
        return func(**item.call_values())
    with _workflow_context._managed_work_item_scope(
        parent_frame,
        item.frame.unit_segment,
    ):
        return func(**item.call_values())


def _ensure_prefect_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
