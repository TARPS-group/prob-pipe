"""WorkflowFunction execution dispatch helpers.

This private module owns ordered execution of plain call dictionaries.
It deliberately knows nothing about ProbPipe value semantics; callers
assemble inputs and interpret outputs outside this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from ._fingerprint import environment_salt, fingerprint

try:
    from prefect import flow, task
    from prefect.cache_policies import CacheKeyFnPolicy
except ImportError:
    task = flow = None
    CacheKeyFnPolicy = None

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
    func_fingerprint: str | None = None
    """Content fingerprint of the wrapped user function, or ``None``.

    Populated only on the Prefect paths when caching is enabled; it identifies
    the user-function body for the cache key so a change to the function
    invalidates cached results even though the task wrapper source is
    unchanged.  ``None`` disables caching for this execution.
    """
    cache_result_storage: Any | None = None
    """Shared storage for persisted cache results and cache-key records.

    ``None`` uses Prefect's local default (single-machine).  When set, it is
    passed to both the task's ``result_storage`` and the cache policy's
    ``key_storage`` so other workers see both the result and its key record.
    """


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

    match request.execution.mode:
        case "sequential":
            return [request.func(**values) for values in request.call_value_list]
        case "thread":
            return execute_many_threaded(request)
        case "prefect_task":
            return execute_many_prefect_task(request)
        case "prefect_flow":
            return execute_many_prefect_flow(request)
        case unknown:
            raise ValueError(f"Unknown workflow execution mode: {unknown!r}")


def execute_many_threaded(request: WorkflowExecutionRequest) -> list[Any]:
    """Execute call dictionaries through ``ThreadPoolExecutor``."""
    if not request.call_value_list:
        return []

    max_workers = _validate_max_workers(request.execution.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda kwargs: request.func(**kwargs), request.call_value_list))


def _cache_task_options(execution: WorkflowExecutionConfig) -> dict[str, Any]:
    """Return Prefect ``@task`` caching options for this execution.

    Caching is active exactly when ``func_fingerprint`` is set — ``node.py``
    populates it only on the Prefect paths under an opted-in ``CacheMode`` — so
    an empty dict here means "no caching", leaving the task decorated exactly as
    before.

    The cache key combines three parts so a change to any of them is a cache
    *miss* rather than a stale *hit*:

    - the user-function fingerprint (detects a changed function body, which the
      generic ``run_func`` wrapper source would otherwise hide),
    - the environment salt (probpipe/jax versions + x64 flag), and
    - a content fingerprint of the per-row inputs.

    Prefect binds a ``**kwargs`` task's arguments as ``{"kwargs": {...}}``, and
    ``fingerprint`` hashes dict keys order-independently, so the whole
    ``parameters`` mapping is fingerprinted directly.
    """
    func_fp = execution.func_fingerprint
    if func_fp is None:
        return {}

    salt = environment_salt()
    storage = execution.cache_result_storage

    def cache_key_fn(context: Any, parameters: dict[str, Any]) -> str:
        material = f"{func_fp}|{salt}|{fingerprint(parameters)}"
        return hashlib.sha256(material.encode()).hexdigest()

    options: dict[str, Any] = {
        "cache_policy": CacheKeyFnPolicy(cache_key_fn=cache_key_fn, key_storage=storage),
        "persist_result": True,
    }
    if storage is not None:
        options["result_storage"] = storage
    return options


def map_task(
    request: WorkflowExecutionRequest,
    *,
    task_name: str | None = None,
) -> list[Any]:
    """Create a Prefect task, map keyword arguments over calls, and resolve futures."""
    if not request.call_value_list:
        return []

    _ensure_prefect_available()

    func = request.func

    @task(
        name=task_name or request.execution.name,
        **_cache_task_options(request.execution),
    )
    def run_func(**kwargs):
        return func(**kwargs)

    # The first row defines the mapped parameter columns. Prefect maps keyword
    # columns, so Ray via Prefect relies on this rectangular kwargs shape.
    keys = request.call_value_list[0].keys()
    kwargs_by_param = {
        key: [call_values[key] for call_values in request.call_value_list] for key in keys
    }
    futures = run_func.map(**kwargs_by_param)
    return [future.result() for future in futures]


def execute_many_prefect_task(request: WorkflowExecutionRequest) -> list[Any]:
    """Use Prefect ``task.map()`` inside a lightweight flow."""
    if not request.call_value_list:
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
    if not request.call_value_list:
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


def _ensure_prefect_available() -> None:
    if task is None:
        raise RuntimeError(
            "Prefect task or flow execution was requested, but Prefect is not installed. "
            "Install with: pip install probpipe[prefect]"
        )
