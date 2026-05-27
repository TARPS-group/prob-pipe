"""WorkflowFunction execution dispatch helpers.

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
    parallel: bool | int = False
    name: str = "workflow"
    prefect_task_runner: Any | None = None


@dataclass(frozen=True)
class WorkflowExecutionRequest:
    """A backend-neutral request to run a function over call dictionaries."""

    func: Callable[..., Any]
    call_value_list: list[dict[str, Any]]
    execution: WorkflowExecutionConfig


def execute_many(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute all call dictionaries using the configured dispatch mode."""
    if not request.call_value_list:
        return []

    if request.execution.mode == "sequential":
        return [request.func(**values) for values in request.call_value_list]
    if request.execution.mode == "thread":
        return execute_many_threaded(request)
    if request.execution.mode == "prefect_task":
        return execute_many_prefect_task(request)
    if request.execution.mode == "prefect_flow":
        return execute_many_prefect_flow(request)
    raise ValueError(f"Unknown workflow execution mode: {request.execution.mode!r}")


def execute_many_threaded(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute call dictionaries through ``ThreadPoolExecutor``."""
    if not request.call_value_list:
        return []

    max_workers = _resolve_max_workers(request.execution.parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda kwargs: request.func(**kwargs), request.call_value_list))


def map_task(
    request: WorkflowExecutionRequest,
    *,
    task_name: str | None = None,
) -> list[Any]:
    """Create a Prefect task, map keyword arguments over calls, and resolve futures."""
    if not request.call_value_list:
        return []

    _ensure_prefect_task_available()

    func = request.func

    @task(name=task_name or request.execution.name)
    def run_func(**kwargs):
        return func(**kwargs)

    # The first row defines the mapped parameter columns. Prefect maps keyword
    # columns, so Ray via Prefect relies on this rectangular kwargs shape.
    keys = request.call_value_list[0].keys()
    kwargs_by_param = {
        key: [call_values[key] for call_values in request.call_value_list]
        for key in keys
    }
    futures = run_func.map(**kwargs_by_param)
    return [future.result() for future in futures]


def execute_many_prefect_task(request: WorkflowExecutionRequest) -> list[Any]:
    """Use Prefect ``task.map()`` inside a lightweight flow."""
    if not request.call_value_list:
        return []

    _ensure_prefect_flow_available()
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
    if not request.call_value_list:
        return []

    _ensure_prefect_flow_available()
    runner = request.execution.prefect_task_runner

    @flow(
        name=request.execution.name,
        **({"task_runner": runner} if runner is not None else {}),
    )
    def mapped_flow():
        return map_task(request, task_name=f"{request.execution.name}_run")

    return mapped_flow()


def _resolve_max_workers(parallel: bool | int) -> int | None:
    if isinstance(parallel, int) and not isinstance(parallel, bool):
        if parallel < 1:
            raise ValueError(
                f"self._parallel must be True, False, or a positive int; got {parallel!r}"
            )
        return parallel

    if parallel is True:
        return None

    raise TypeError(
        f"parallel must be True, False, or a positive int; got {parallel!r}"
    )


def _ensure_prefect_task_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )


def _ensure_prefect_flow_available() -> None:
    if task is None or flow is None:
        raise RuntimeError(
            "Prefect task execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
