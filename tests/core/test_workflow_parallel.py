from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

import pytest

import probpipe.core._workflow_execution as execution_mod
import probpipe.core.node as node_mod
from probpipe import Normal
from probpipe.core.config import WorkflowKind, prefect_config
from probpipe.core.node import Function


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
    max_workers=None,
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
            max_workers=max_workers,
            name=name,
            prefect_task_runner=prefect_task_runner,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_fakes():
    prefect_config.workflow_kind = WorkflowKind.OFF
    prefect_config.task_runner = None
    RecordingExecutor.instances.clear()
    FakeMappedTask.created_names.clear()
    RecordingFlow.calls.clear()
    yield
    prefect_config.workflow_kind = WorkflowKind.OFF
    prefect_config.task_runner = None
    RecordingExecutor.instances.clear()
    FakeMappedTask.created_names.clear()
    RecordingFlow.calls.clear()


class TestExecutionRequestShape:
    def test_execution_config_has_resolved_execution_fields_only(self):
        field_names = {field.name for field in fields(execution_mod.WorkflowExecutionConfig)}

        assert "dispatch" not in field_names
        assert "parallel" not in field_names
        assert field_names == {
            "mode",
            "max_workers",
            "name",
            "prefect_task_runner",
        }

    def test_function_request_contains_plain_function(self, monkeypatch):
        seen = {}

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**request.call_value_list[0])]

        monkeypatch.setattr(execution_mod, "execute_many", fake_execute_many)
        wf = Function(func=add_one, dispatch="sequential")

        result = wf(x=1)

        assert float(result["add_one"]) == 2.0
        assert seen["request"].func is add_one
        assert not isinstance(seen["request"].func, Function)
        assert seen["request"].execution.mode == "sequential"


class TestSequentialExecution:
    def test_execute_many_sequential_mode_preserves_order(self):
        request = make_request(mode="sequential")

        assert execution_mod.execute_many(request) == [2, 3]
        assert RecordingExecutor.instances == []

    def test_execute_many_empty_input_returns_empty_without_executor(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", calls=[])

        assert execution_mod.execute_many(request) == []
        assert RecordingExecutor.instances == []


class TestThreadExecution:
    def test_execute_many_thread_mode_uses_executor_default_workers(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread")

        assert execution_mod.execute_many(request) == [2, 3]
        assert len(RecordingExecutor.instances) == 1
        assert RecordingExecutor.instances[0].max_workers is None

    def test_execute_many_max_workers_uses_explicit_worker_count(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", max_workers=3)

        assert execution_mod.execute_many(request) == [2, 3]
        assert len(RecordingExecutor.instances) == 1
        assert RecordingExecutor.instances[0].max_workers == 3

    def test_execute_many_accepts_true_max_workers_as_positive_int(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        request = make_request(mode="thread", max_workers=True)

        assert execution_mod.execute_many(request) == [2, 3]
        assert len(RecordingExecutor.instances) == 1
        assert RecordingExecutor.instances[0].max_workers is True

    @pytest.mark.parametrize("max_workers", [0, -1, False])
    def test_execute_many_rejects_non_positive_max_workers(self, max_workers):
        request = make_request(mode="thread", max_workers=max_workers)

        with pytest.raises(ValueError, match="positive int"):
            execution_mod.execute_many(request)

    @pytest.mark.parametrize("max_workers", ["3"])
    def test_execute_many_rejects_invalid_max_workers_value(self, max_workers):
        request = make_request(mode="thread", max_workers=max_workers)

        with pytest.raises(TypeError, match="max_workers"):
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

        with pytest.raises(RuntimeError, match=r"Prefect task.*execution was requested"):
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


class TestFunctionExecutionConfig:
    def test_make_execution_config_resolves_thread_dispatch_to_thread_mode(self):
        wf = Function(
            func=add_one,
            dispatch="thread",
            workflow_kind=WorkflowKind.OFF,
        )

        assert wf._make_execution_config().mode == "thread"

    def test_thread_dispatch_passes_max_workers_to_execution_config(self):
        wf = Function(func=add_one, dispatch="thread", max_workers=3)

        execution = wf._make_execution_config()

        assert execution.mode == "thread"
        assert execution.max_workers == 3

    def test_thread_dispatch_accepts_true_max_workers_as_positive_int(self):
        wf = Function(func=add_one, dispatch="thread", max_workers=True)

        execution = wf._make_execution_config()

        assert execution.mode == "thread"
        assert execution.max_workers is True

    def test_auto_dispatch_does_not_use_max_workers_as_mode_switch(self):
        with pytest.warns(UserWarning, match="max_workers configures only"):
            wf = Function(func=add_one, dispatch="auto", max_workers=3)

        execution = wf._make_execution_config()

        assert execution.mode == "sequential"
        assert execution.max_workers is None

    @pytest.mark.parametrize("dispatch", ["loop", "python", "map", None, 1])
    def test_function_rejects_invalid_dispatch(self, dispatch):
        with pytest.raises(ValueError, match="dispatch must be one of"):
            Function(func=add_one, dispatch=dispatch)

    @pytest.mark.parametrize("max_workers", [0, -1, False])
    def test_function_rejects_non_positive_max_workers(self, max_workers):
        with pytest.raises(ValueError, match="max_workers"):
            Function(func=add_one, dispatch="sequential", max_workers=max_workers)

    @pytest.mark.parametrize("max_workers", ["3"])
    def test_function_rejects_invalid_max_workers(self, max_workers):
        with pytest.raises(TypeError, match="max_workers"):
            Function(func=add_one, dispatch="sequential", max_workers=max_workers)

    def test_non_thread_dispatch_warns_and_ignores_max_workers(self):
        with pytest.warns(UserWarning, match="max_workers configures only"):
            wf = Function(func=add_one, dispatch="sequential", max_workers=3)

        execution = wf._make_execution_config()

        assert execution.mode == "sequential"
        assert execution.max_workers is None

    def test_explicit_prefect_warns_and_ignores_local_thread_options(self, monkeypatch):
        # The mode resolver only yields prefect_* when Prefect is installed
        # (effective_workflow_kind also gates on node_mod.flow, not just task).
        pytest.importorskip("prefect")
        monkeypatch.setattr(node_mod, "task", object())
        wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="thread",
            max_workers=3,
        )

        with pytest.warns(UserWarning, match="do not control Prefect scheduling"):
            execution = wf._make_execution_config()

        assert execution.mode == "prefect_task"
        assert execution.max_workers is None

    def test_global_prefect_warns_and_ignores_max_workers_with_sequential_dispatch(
        self,
        monkeypatch,
    ):
        # The mode resolver only yields prefect_* when Prefect is installed
        # (effective_workflow_kind also gates on node_mod.flow, not just task).
        pytest.importorskip("prefect")
        monkeypatch.setattr(node_mod, "task", object())
        prefect_config.workflow_kind = WorkflowKind.FLOW
        with pytest.warns(UserWarning, match="max_workers configures only"):
            wf = Function(
                func=add_one,
                dispatch="sequential",
                max_workers=3,
            )

        with pytest.warns(UserWarning, match="do not control Prefect scheduling"):
            execution = wf._make_execution_config()

        assert execution.mode == "prefect_flow"
        assert execution.max_workers is None

    def test_jax_dispatch_warns_and_ignores_max_workers(self):
        with pytest.warns(UserWarning, match="max_workers configures only"):
            wf = Function(
                func=add_one,
                dispatch="jax",
                workflow_kind=WorkflowKind.OFF,
                max_workers=3,
            )

        execution = wf._make_execution_config()

        assert execution.mode == "sequential"
        assert execution.max_workers is None

    def test_jax_broadcast_ignores_max_workers_after_warning(self, monkeypatch):
        def fail_executor(*args, **kwargs):
            raise AssertionError("JAX broadcast should not use ThreadPoolExecutor")

        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", fail_executor)
        with pytest.warns(UserWarning, match="max_workers configures only"):
            wf = Function(
                func=add_one,
                dispatch="jax",
                workflow_kind=WorkflowKind.OFF,
                max_workers=3,
                n_broadcast_samples=8,
                seed=0,
            )

        result = wf(x=Normal(loc=0.0, scale=1.0, name="x"))

        assert result.num_atoms == 8

    def test_public_call_resolves_task_and_flow_modes(self, monkeypatch):
        # The mode resolver only yields prefect_* when Prefect is installed
        # (effective_workflow_kind also gates on node_mod.flow, not just task).
        pytest.importorskip("prefect")
        seen_modes = []

        def fake_execute_many(request):
            seen_modes.append(request.execution.mode)
            return [request.func(**request.call_value_list[0])]

        monkeypatch.setattr(node_mod, "task", object())
        monkeypatch.setattr(execution_mod, "execute_many", fake_execute_many)
        task_wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )
        flow_wf = Function(
            func=add_one,
            workflow_kind=WorkflowKind.FLOW,
            dispatch="sequential",
        )

        assert float(task_wf(x=1)["add_one"]) == 2.0
        assert float(flow_wf(x=1)["add_one"]) == 2.0
        assert seen_modes == ["prefect_task", "prefect_flow"]
