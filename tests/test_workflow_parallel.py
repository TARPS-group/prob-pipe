import pytest

import probpipe.core.node as node_mod
from probpipe.core.config import WorkflowKind
from probpipe.core.node import WorkflowFunction


def add_one(x):
    return x + 1


class RecordingExecutor:
    instances = []

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
    def __init__(self, fn):
        self.fn = fn

    def map(self, **kwargs_by_param):
        count = len(next(iter(kwargs_by_param.values())))
        futures = []
        for index in range(count):
            kwargs = {name: values[index] for name, values in kwargs_by_param.items()}
            futures.append(FakeFuture(self.fn(**kwargs)))
        return futures


def fake_task(name=None):
    def decorator(fn):
        return FakeMappedTask(fn)

    return decorator


@pytest.fixture(autouse=True)
def _reset_executor_instances():
    RecordingExecutor.instances.clear()
    yield
    RecordingExecutor.instances.clear()


def test_execute_many_parallel_false_runs_sequentially():
    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=False)

    assert wf._execute_many([{"x": 1}, {"x": 2}]) == [2, 3]


def test_execute_many_parallel_true_uses_executor_default_workers(monkeypatch):
    monkeypatch.setattr(node_mod, "ThreadPoolExecutor", RecordingExecutor)

    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=True)

    assert wf._execute_many([{"x": 1}, {"x": 2}]) == [2, 3]
    assert len(RecordingExecutor.instances) == 1
    assert RecordingExecutor.instances[0].max_workers is None


def test_execute_many_parallel_int_uses_explicit_worker_count(monkeypatch):
    monkeypatch.setattr(node_mod, "ThreadPoolExecutor", RecordingExecutor)

    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=3)

    assert wf._execute_many([{"x": 1}, {"x": 2}]) == [2, 3]
    assert len(RecordingExecutor.instances) == 1
    assert RecordingExecutor.instances[0].max_workers == 3


def test_execute_many_parallel_true_empty_input_returns_empty_without_executor(monkeypatch):
    monkeypatch.setattr(node_mod, "ThreadPoolExecutor", RecordingExecutor)

    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=True)

    assert wf._execute_many([]) == []
    assert RecordingExecutor.instances == []


def test_execute_many_rejects_non_positive_parallel_int():
    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=0)

    with pytest.raises(ValueError, match="positive int"):
        wf._execute_many([{"x": 1}])


def test_execute_many_rejects_invalid_parallel_value():
    wf = WorkflowFunction(func=add_one, vectorize="loop", parallel=None)

    with pytest.raises(TypeError, match="positive int"):
        wf._execute_many([{"x": 1}])


def test_execute_many_dispatches_task_and_flow_without_running_prefect(monkeypatch):
    monkeypatch.setattr(node_mod, "task", object())

    task_wf = WorkflowFunction(func=add_one, workflow_kind=WorkflowKind.TASK, vectorize="loop")
    flow_wf = WorkflowFunction(func=add_one, workflow_kind=WorkflowKind.FLOW, vectorize="loop")

    monkeypatch.setattr(task_wf, "_execute_many_prefect_task", lambda values: ("task", values))
    monkeypatch.setattr(flow_wf, "_execute_many_prefect_flow", lambda values: ("flow", values))

    calls = [{"x": 1}]
    assert task_wf._execute_many(calls) == ("task", calls)
    assert flow_wf._execute_many(calls) == ("flow", calls)


def test_map_task_empty_input_returns_empty_before_prefect_guard(monkeypatch):
    monkeypatch.setattr(node_mod, "task", None)

    wf = WorkflowFunction(func=add_one, vectorize="loop")

    assert wf._map_task([]) == []


def test_map_task_raises_clear_error_when_prefect_missing(monkeypatch):
    monkeypatch.setattr(node_mod, "task", None)

    wf = WorkflowFunction(func=add_one, vectorize="loop")

    with pytest.raises(RuntimeError, match="Prefect task execution was requested"):
        wf._map_task([{"x": 1}])


def test_map_task_maps_keyword_arguments_and_resolves_futures(monkeypatch):
    monkeypatch.setattr(node_mod, "task", fake_task)

    wf = WorkflowFunction(func=add_one, vectorize="loop")

    assert wf._map_task([{"x": 1}, {"x": 2}], task_name="add-one") == [2, 3]
