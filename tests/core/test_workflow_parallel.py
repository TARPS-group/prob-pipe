from __future__ import annotations

from typing import ClassVar

import pytest

import probpipe.core._workflow_execution as execution_mod
import probpipe.core.node as node_mod
from probpipe.core.config import WorkflowKind
from probpipe.core.node import WorkflowFunction


def add_one(x):
    return x + 1


def add_xy(x, y):
    return x + y


class RecordingExecutor:
    instances: ClassVar[list[RecordingExecutor]] = []

    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.items = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable):
        self.items = list(iterable)
        return [fn(item) for item in self.items]


class FakeFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class FakeMappedTask:
    created_names: ClassVar[list[str | None]] = []

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name
        self.__class__.created_names.append(name)

    def map(self, **kwargs_by_param):
        count = len(next(iter(kwargs_by_param.values())))
        futures = []
        for index in range(count):
            kwargs = {name: values[index] for name, values in kwargs_by_param.items()}
            futures.append(FakeFuture(self.fn(**kwargs)))
        return futures


class RecordingFlow:
    calls: ClassVar[list[dict[str, object]]] = []


def fake_task(name=None):
    def decorator(fn):
        return FakeMappedTask(fn, name)

    return decorator


def fake_flow(name=None, **flow_kwargs):
    def decorator(fn):
        def wrapper():
            RecordingFlow.calls.append({"name": name, "kwargs": flow_kwargs})
            return fn()

        return wrapper

    return decorator


def make_request(
    *,
    mode="sequential",
    parallel=False,
    calls=None,
    func=add_one,
    name="add_one",
    prefect_task_runner=None,
):
    return execution_mod.WorkflowExecutionRequest(
        func=func,
        call_value_list=calls if calls is not None else [{"x": 1}, {"x": 2}],
        execution=execution_mod.WorkflowExecutionConfig(
            mode=mode,
            parallel=parallel,
            name=name,
            prefect_task_runner=prefect_task_runner,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_fakes():
    RecordingExecutor.instances.clear()
    FakeMappedTask.created_names.clear()
    RecordingFlow.calls.clear()
    yield
    RecordingExecutor.instances.clear()
    FakeMappedTask.created_names.clear()
    RecordingFlow.calls.clear()


class TestExecutionRequestShape:
    def test_workflow_function_request_contains_plain_function(self, monkeypatch):
        seen = {}

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**request.call_value_list[0])]

        monkeypatch.setattr(execution_mod, "execute_many", fake_execute_many)
        wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=False)

        assert wf._execute_many([{"x": 1}]) == [2]
        assert seen["request"].func is add_one
        assert not isinstance(seen["request"].func, WorkflowFunction)
        assert seen["request"].execution.mode == "sequential"


class TestSequentialExecution:
    def test_execute_many_parallel_false_runs_sequentially(self):
        request = make_request(mode="sequential", parallel=False)

        assert execution_mod.execute_many(request) == [2, 3]
        assert RecordingExecutor.instances == []

    def test_execute_many_empty_input_returns_empty_without_executor(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", parallel=True, calls=[])

        assert execution_mod.execute_many(request) == []
        assert RecordingExecutor.instances == []


class TestThreadExecution:
    def test_execute_many_parallel_true_uses_executor_default_workers(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", parallel=True)

        assert execution_mod.execute_many(request) == [2, 3]
        assert len(RecordingExecutor.instances) == 1
        assert RecordingExecutor.instances[0].max_workers is None

    def test_execute_many_parallel_int_uses_explicit_worker_count(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", parallel=3)

        assert execution_mod.execute_many(request) == [2, 3]
        assert len(RecordingExecutor.instances) == 1
        assert RecordingExecutor.instances[0].max_workers == 3

    @pytest.mark.parametrize("parallel", [0, -1])
    def test_execute_many_rejects_non_positive_parallel_int(self, parallel):
        request = make_request(mode="thread", parallel=parallel)

        with pytest.raises(ValueError, match="positive int"):
            execution_mod.execute_many(request)

    def test_execute_many_rejects_invalid_parallel_value(self):
        request = make_request(mode="thread", parallel=None)

        with pytest.raises(TypeError, match="positive int"):
            execution_mod.execute_many(request)


class TestPrefectMapping:
    def test_map_task_empty_input_returns_empty_before_prefect_guard(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", None)
        monkeypatch.setattr(execution_mod, "flow", None)
        request = make_request(mode="prefect_task", calls=[])

        assert execution_mod.map_task(request) == []

    def test_map_task_raises_clear_error_when_prefect_missing(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", None)
        monkeypatch.setattr(execution_mod, "flow", None)
        request = make_request(mode="prefect_task")

        with pytest.raises(RuntimeError, match="Prefect task execution was requested"):
            execution_mod.map_task(request)

    def test_map_task_maps_keyword_arguments_and_resolves_futures(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", None)
        request = make_request(
            mode="prefect_task",
            calls=[{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            func=add_xy,
        )

        assert execution_mod.map_task(request, task_name="add-xy") == [3, 7]
        assert FakeMappedTask.created_names == ["add-xy"]

    def test_prefect_task_executor_uses_flow_wrapper_and_runner(self, monkeypatch):
        runner = object()
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(
            mode="prefect_task",
            prefect_task_runner=runner,
            name="plus_one",
        )

        assert execution_mod.execute_many(request) == [2, 3]
        assert RecordingFlow.calls == [
            {"name": "plus_one_map", "kwargs": {"task_runner": runner}},
        ]
        assert FakeMappedTask.created_names == ["plus_one"]

    def test_prefect_flow_executor_uses_named_flow_and_task_name(self, monkeypatch):
        runner = object()
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(
            mode="prefect_flow",
            prefect_task_runner=runner,
            name="plus_one",
        )

        assert execution_mod.execute_many(request) == [2, 3]
        assert RecordingFlow.calls == [
            {"name": "plus_one", "kwargs": {"task_runner": runner}},
        ]
        assert FakeMappedTask.created_names == ["plus_one_run"]


class TestWorkflowFunctionCompatibility:
    def test_execute_many_wrapper_preserves_private_call_behavior(self):
        wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=False)

        assert wf._execute_many([{"x": 1}, {"x": 2}]) == [2, 3]

    def test_private_wrappers_return_empty_before_building_request(self, monkeypatch):
        wf = WorkflowFunction(func=add_one, workflow_kind=WorkflowKind.TASK, vectorize="loop")

        def fail_request(*args, **kwargs):
            raise AssertionError("empty calls should not build an execution request")

        monkeypatch.setattr(wf, "_make_execution_request", fail_request)

        assert wf._execute_many([]) == []
        assert wf._execute_many_threaded([]) == []
        assert wf._map_task([]) == []
        assert wf._execute_many_prefect_task([]) == []
        assert wf._execute_many_prefect_flow([]) == []

    def test_execute_many_wrapper_resolves_task_and_flow_modes(self, monkeypatch):
        seen_modes = []

        def fake_execute_many(request):
            seen_modes.append(request.execution.mode)
            return ("ok", request.call_value_list)

        monkeypatch.setattr(node_mod, "task", object())
        monkeypatch.setattr(execution_mod, "execute_many", fake_execute_many)
        task_wf = WorkflowFunction(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            vectorize="loop",
        )
        flow_wf = WorkflowFunction(
            func=add_one,
            workflow_kind=WorkflowKind.FLOW,
            vectorize="loop",
        )

        calls = [{"x": 1}]
        assert task_wf._execute_many(calls) == ("ok", calls)
        assert flow_wf._execute_many(calls) == ("ok", calls)
        assert seen_modes == ["prefect_task", "prefect_flow"]

    def test_threaded_wrapper_delegates_to_execution_module(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=2)

        assert wf._execute_many_threaded([{"x": 1}, {"x": 2}]) == [2, 3]
        assert RecordingExecutor.instances[0].max_workers == 2

    def test_map_task_wrapper_delegates_to_execution_module(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", None)
        wf = WorkflowFunction(func=add_one, vectorize="loop")

        assert wf._map_task([{"x": 1}, {"x": 2}], task_name="add-one") == [2, 3]
        assert FakeMappedTask.created_names == ["add-one"]

    def test_map_task_wrapper_does_not_resolve_task_runner(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", None)

        def fail_resolve_task_runner():
            raise AssertionError("direct _map_task should not resolve a task runner")

        monkeypatch.setattr(node_mod.prefect_config, "resolve_task_runner", fail_resolve_task_runner)
        wf = WorkflowFunction(func=add_one, vectorize="loop")

        assert wf._map_task([{"x": 1}], task_name="add-one") == [2]
