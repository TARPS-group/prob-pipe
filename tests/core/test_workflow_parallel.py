from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError, fields
from threading import Event
from typing import ClassVar
from unittest.mock import patch

import jax
import pytest

import probpipe.core._workflow_broker as broker_mod
import probpipe.core._workflow_execution as execution_mod
import probpipe.core._workflow_managed as managed_mod
import probpipe.core.node as node_mod
from probpipe import (
    Normal,
    Provenance,
    ReplayCompatibilityError,
    replay_run,
    workflow_run,
)
from probpipe.core.config import WorkflowKind, prefect_config
from probpipe.core.node import Function


def add_one(x):
    return x + 1


def add_xy(x, y):
    return x + y


def _claim_automatic_words():
    key = broker_mod._resolve_automatic_key(
        None,
        broker_mod._singleton_effect_plan(
            operation_kind="managed-test",
            execution_mode="sampled",
            sample_shape=(),
        ),
    )
    return tuple(int(word) for word in jax.random.key_data(key))


def _claim_automatic_scalar():
    key = broker_mod._resolve_automatic_key(
        None,
        broker_mod._singleton_effect_plan(
            operation_kind="managed-test",
            execution_mode="sampled",
            sample_shape=(),
        ),
    )
    return jax.random.key_data(key)[0]


def _claim_nested_seeded_scalar():
    with workflow_run(seed=42):
        return _claim_automatic_scalar()


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
    reverse_results: ClassVar[bool] = False
    map_calls: ClassVar[int] = 0

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name
        self.__class__.created_names.append(name)

    def map(self, **kwargs_by_param):
        self.__class__.map_calls += 1
        count = len(next(iter(kwargs_by_param.values())))
        futures = []
        for index in range(count):
            kwargs = {name: values[index] for name, values in kwargs_by_param.items()}
            futures.append(FakeFuture(self.fn(**kwargs)))
        return list(reversed(futures)) if self.__class__.reverse_results else futures


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
    call_values = calls if calls is not None else [{"x": 1}, {"x": 2}]
    unit_segments = (
        (execution_mod.point_unit_segment(),)
        if len(call_values) == 1
        else tuple(execution_mod.sweep_unit_segment((index,)) for index in range(len(call_values)))
    )
    return execution_mod.WorkflowExecutionRequest(
        func=func,
        work_items=execution_mod.make_managed_work_items(
            call_values,
            unit_segments=unit_segments,
        ),
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
    FakeMappedTask.reverse_results = False
    FakeMappedTask.map_calls = 0
    RecordingFlow.calls.clear()
    yield
    prefect_config.workflow_kind = WorkflowKind.OFF
    prefect_config.task_runner = None
    RecordingExecutor.instances.clear()
    FakeMappedTask.created_names.clear()
    FakeMappedTask.reverse_results = False
    FakeMappedTask.map_calls = 0
    RecordingFlow.calls.clear()


class TestExecutionRequestShape:
    def test_managed_work_items_are_frozen_tuple_only_and_pickleable(self):
        request = make_request(calls=[{"x": 1}])

        assert isinstance(request.work_items, tuple)
        assert request.work_items[0].values == (("x", 1),)
        assert request.work_items[0].frame.unit_segment == (
            "managed-unit",
            "probpipe.managed_work_item/v1",
            "point",
            0,
        )
        assert len(request.work_items[0].frame.token.value) == 16
        assert pickle.loads(pickle.dumps(request.work_items)) == request.work_items
        with pytest.raises(FrozenInstanceError):
            request.work_items[0].index = 9

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

    def test_function_request_contains_resolved_point_callable(self, monkeypatch):
        seen = {}

        def fake_execute_many(request):
            seen["request"] = request
            return [request.func(**request.work_items[0].call_values())]

        monkeypatch.setattr(execution_mod, "execute_many", fake_execute_many)
        wf = Function(func=add_one, dispatch="sequential")

        result = wf(x=1)

        assert float(result["add_one"]) == 2.0
        assert callable(seen["request"].func)
        assert seen["request"].func is not add_one
        assert not isinstance(seen["request"].func, Function)
        assert seen["request"].execution.mode == "sequential"


class TestManagedRetryClaims:
    def test_same_work_item_token_retries_the_same_child_claim(self):
        request = make_request(calls=[{}], func=_claim_automatic_words)

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            first = execution_mod.execute_many(request)[0]
            second = execution_mod.execute_many(request)[0]

            state = parent._managed_claims.by_unit[request.work_items[0].frame.unit_segment]
            occurrence_path = state.child_invocations[0].occurrence_path

        assert first == second
        assert occurrence_path == (
            ("invocation", 0),
            request.work_items[0].frame.unit_segment,
            ("child", 0),
        )

    def test_different_token_cannot_reuse_a_managed_unit(self):
        first_request = make_request(calls=[{}], func=lambda: None)
        second_request = make_request(calls=[{}], func=lambda: None)

        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            execution_mod.execute_many(first_request)
            with pytest.raises(RuntimeError, match="different token"):
                execution_mod.execute_many(second_request)

    def test_same_attempt_cannot_enter_twice(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            with (
                broker_mod._managed_work_item_stochastic_scope(
                    parent,
                    item.frame,
                    attempt=attempt,
                ),
                pytest.raises(RuntimeError, match="active attempt"),
                broker_mod._managed_work_item_stochastic_scope(
                    parent,
                    item.frame,
                    attempt=attempt,
                ),
            ):
                pass

    @pytest.mark.parametrize("mode", ["sequential", "prefect_task"])
    def test_same_attempt_cannot_claim_one_effect_twice(self, mode, monkeypatch):
        if mode == "prefect_task":
            monkeypatch.setattr(execution_mod, "task", fake_task)
            monkeypatch.setattr(execution_mod, "flow", fake_flow)

        def duplicate_effect_claim():
            with broker_mod._managed_stochastic_scope():
                _claim_automatic_words()
                _claim_automatic_words()

        request = make_request(
            mode=mode,
            calls=[{}],
            func=duplicate_effect_claim,
        )
        with (
            workflow_run(seed=17),
            broker_mod._function_stochastic_scope(),
            pytest.raises(RuntimeError, match="duplicated a stochastic effect claim"),
        ):
            execution_mod.execute_many(request)

    def test_deterministic_attempt_does_not_consume_child_ordinal(self):
        state = {"random": False}

        def maybe_claim():
            return _claim_automatic_words() if state["random"] else None

        request = make_request(calls=[{}], func=maybe_claim)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            assert execution_mod.execute_many(request) == [None]
            state["random"] = True
            after_deterministic = execution_mod.execute_many(request)[0]

        baseline = make_request(calls=[{}], func=_claim_automatic_words)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            first_child = execution_mod.execute_many(baseline)[0]

        assert after_deterministic == first_child

    def test_failed_attempt_reuses_existing_child_claim(self):
        seen = []

        def fail_once_after_claim():
            words = _claim_automatic_words()
            seen.append(words)
            if len(seen) == 1:
                raise RuntimeError("retry me")
            return words

        request = make_request(calls=[{}], func=fail_once_after_claim)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                execution_mod.execute_many(request)
            retried = execution_mod.execute_many(request)[0]

        assert seen == [retried, retried]

    def test_failed_attempt_reuses_nested_scope_claim(self):
        seen = []

        def fail_once_after_nested_claim():
            with workflow_run(seed=42):
                words = _claim_automatic_words()
            seen.append(words)
            if len(seen) == 1:
                raise RuntimeError("retry me")
            return words

        request = make_request(calls=[{}], func=fail_once_after_nested_claim)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                execution_mod.execute_many(request)
            retried = execution_mod.execute_many(request)[0]

        assert seen == [retried, retried]

    def test_failed_attempt_only_effect_is_not_persisted_by_later_success(self):
        state = {"claim": True, "fail": True}

        def conditional_attempt():
            if state["claim"]:
                _claim_automatic_words()
            if state["fail"]:
                raise RuntimeError("retry me")
            return 1

        request = make_request(calls=[{}], func=conditional_attempt)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                execution_mod.execute_many(request)
            state.update(claim=False, fail=False)
            assert execution_mod.execute_many(request) == [1]
            snapshot = broker_mod._snapshot_active_recipe_state()

        assert snapshot is None

    def test_caught_failed_nested_function_effect_is_not_persisted(self):
        def fail_after_claim():
            _claim_automatic_words()
            raise RuntimeError("nested failure")

        nested = Function(func=fail_after_claim)

        def catch_nested_failure():
            with pytest.raises(RuntimeError, match="nested failure"):
                nested()
            return 1

        request = make_request(calls=[{}], func=catch_nested_failure)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            assert execution_mod.execute_many(request) == [1]
            snapshot = broker_mod._snapshot_active_recipe_state()

        assert snapshot is None

    def test_parent_cancels_unstarted_items_before_releasing(self):
        def fail(value):
            if value == 1:
                raise RuntimeError("stop")
            return value

        request = make_request(calls=[{"value": 1}, {"value": 2}], func=fail)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="stop"):
                execution_mod.execute_many(request)
            parent.assert_managed_items_joined(request.work_items)


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

    def test_reverse_completion_preserves_keys_and_canonical_result_order(self):
        second_completed = Event()
        completion_order = []

        def claim_in_order(value):
            return value, _claim_automatic_words()

        def claim_in_reverse(value):
            if value == 0:
                assert second_completed.wait(timeout=5)
            else:
                completion_order.append(value)
                second_completed.set()
            result = value, _claim_automatic_words()
            if value == 0:
                completion_order.append(value)
            return result

        def run(mode, func):
            request = make_request(
                mode=mode,
                max_workers=2 if mode == "thread" else None,
                calls=[{"value": 0}, {"value": 1}],
                func=func,
            )
            with workflow_run(seed=17), broker_mod._function_stochastic_scope():
                return execution_mod.execute_many(request)

        expected = run("sequential", claim_in_order)
        actual = run("thread", claim_in_reverse)

        assert completion_order == [1, 0]
        assert actual == expected

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

    def test_prefect_outcomes_restore_canonical_order(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        FakeMappedTask.reverse_results = True
        request = make_request(
            mode="prefect_task",
            calls=[{"x": 1}, {"x": 2}, {"x": 3}],
        )

        assert execution_mod.map_task(request) == [2, 3, 4]

    def test_deterministic_prefect_items_do_not_materialize_anonymous_root(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(mode="prefect_task")

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
            broker_mod._function_stochastic_scope(),
        ):
            assert execution_mod.execute_many(request) == [2, 3]

        urandom.assert_not_called()
        assert FakeMappedTask.map_calls == 1

    def test_prefect_randomness_uses_lazy_parent_coordination(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(
            mode="prefect_task",
            calls=[{}],
            func=_claim_automatic_words,
        )

        with (
            patch(
                "probpipe.core._workflow_context._os_urandom",
                return_value=bytes.fromhex("0123456789abcdef"),
            ) as urandom,
            workflow_run(),
            broker_mod._function_stochastic_scope() as parent,
        ):
            result = execution_mod.execute_many(request)[0]
            envelope = parent.prepare_remote_managed_unit(request.work_items[0].frame)

        assert result
        assert envelope.parent_occurrence_path == (("invocation", 0),)
        assert len(parent._effects_by_identity) == 1
        urandom.assert_called_once_with(8)
        assert FakeMappedTask.map_calls == 2

    def test_failed_prefect_effect_remains_transient(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (0, 0.0))

        def fail_after_claim():
            _claim_automatic_words()
            raise RuntimeError("remote failure")

        request = make_request(
            mode="prefect_task",
            calls=[{}],
            func=fail_after_claim,
        )
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="remote failure"):
                execution_mod.execute_many(request)
            snapshot = broker_mod._snapshot_active_recipe_state()
            claim_state = parent._managed_claims.by_token[request.work_items[0].frame.token]

        assert snapshot is None
        assert len(claim_state.effect_claims_by_identity) == 1

    def test_prefect_retry_reuses_work_item_token_and_key(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (1, 0.0))
        seen = []

        def fail_once_after_claim():
            words = _claim_automatic_words()
            seen.append(words)
            if len(seen) == 1:
                raise RuntimeError("retry me")
            return words

        request = make_request(
            mode="prefect_task",
            calls=[{}],
            func=fail_once_after_claim,
        )
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            retried = execution_mod.execute_many(request)[0]
            snapshot = broker_mod._snapshot_active_recipe_state()
            claim_state = parent._managed_claims.by_token[request.work_items[0].frame.token]

        assert seen == [retried, retried]
        assert snapshot is not None
        assert len(snapshot.effects) == 1
        assert len(claim_state.seen_attempts) == 2
        assert FakeMappedTask.map_calls == 3

    def test_deterministic_prefect_retry_does_not_materialize_root(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (1, 0.0))
        calls = 0

        def fail_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("retry me")
            return 1

        request = make_request(mode="prefect_task", calls=[{}], func=fail_once)
        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
            broker_mod._function_stochastic_scope(),
        ):
            assert execution_mod.execute_many(request) == [1]
            snapshot = broker_mod._snapshot_active_recipe_state()

        assert snapshot is None
        urandom.assert_not_called()
        assert FakeMappedTask.map_calls == 2

    def test_managed_keys_match_across_sequential_thread_and_prefect(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)

        def run(mode):
            request = make_request(
                mode=mode,
                calls=[{}],
                func=_claim_automatic_words,
            )
            with workflow_run(seed=17), broker_mod._function_stochastic_scope():
                return execution_mod.execute_many(request)[0]

        assert run("sequential") == run("thread") == run("prefect_task")

    def test_nested_seeded_scope_matches_across_local_and_prefect(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)

        def run(mode, outer_seed):
            workflow = Function(
                func=_claim_nested_seeded_scalar,
                workflow_kind=(WorkflowKind.TASK if mode == "prefect" else WorkflowKind.OFF),
                dispatch="sequential",
            )
            with workflow_run(seed=outer_seed):
                return int(workflow()["_claim_nested_seeded_scalar"])

        local = run("local", 1)
        assert run("local", 2) == local
        assert run("prefect", 1) == local
        assert run("prefect", 2) == local

    def test_prefect_transport_payload_is_pickleable(self):
        request = make_request(calls=[{}], func=_claim_automatic_words)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            envelope = parent.prepare_remote_managed_unit(request.work_items[0].frame)
            payload = managed_mod.ManagedPrefectPayload(
                item=request.work_items[0],
                parent=envelope,
            )
            parent.cancel_unstarted_managed_items(request.work_items)

        assert pickle.loads(pickle.dumps(payload)) == payload

    def test_replay_prefect_worker_uses_assigned_expected_event_namespace(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        local = Function(
            func=_claim_automatic_scalar,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
        )
        remote = Function(
            func=_claim_automatic_scalar,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )
        with workflow_run(seed=17):
            original = local()

        with replay_run(original.provenance):
            replayed = remote()

        assert (
            replayed.provenance.controls["randomness"] == original.provenance.controls["randomness"]
        )
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True
        assert FakeMappedTask.map_calls == 2

    def test_replay_prefect_worker_rejects_effect_drift_before_key_derivation(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        local = Function(
            func=_claim_automatic_scalar,
            workflow_kind=WorkflowKind.OFF,
            dispatch="sequential",
        )
        remote = Function(
            func=_claim_automatic_scalar,
            workflow_kind=WorkflowKind.TASK,
            dispatch="sequential",
        )
        with workflow_run(seed=17):
            original = local()
        payload = original.provenance.to_dict()
        payload["controls"]["replay"]["plan"]["expected_effects"][0]["provider_abi"] = (
            "unknown-provider/v99"
        )
        payload["controls"]["replay"]["compatibility"]["provider_abi"] = ["unknown-provider/v99"]
        changed = Provenance.from_dict(payload)

        with (
            patch(
                "probpipe.core._workflow_context.derive_event_key_words",
                side_effect=AssertionError("derived key before replay validation"),
            ),
            pytest.raises(ReplayCompatibilityError, match="assigned event namespace"),
            replay_run(changed),
        ):
            remote()

        assert FakeMappedTask.map_calls == 2


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
            )

        with workflow_run(seed=0):
            result = wf(x=Normal(loc=0.0, scale=1.0, name="x"))

        assert result.num_atoms == 8

    def test_public_call_resolves_task_and_flow_modes(self, monkeypatch):
        # The mode resolver only yields prefect_* when Prefect is installed
        # (effective_workflow_kind also gates on node_mod.flow, not just task).
        pytest.importorskip("prefect")
        seen_modes = []

        def fake_execute_many(request):
            seen_modes.append(request.execution.mode)
            return [request.func(**request.work_items[0].call_values())]

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
