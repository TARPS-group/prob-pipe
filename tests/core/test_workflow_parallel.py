from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields, replace
from threading import Barrier, Event
from typing import ClassVar
from unittest.mock import Mock, patch

import jax
import pytest

import probpipe.core._workflow_broker as broker_mod
import probpipe.core._workflow_context as context_mod
import probpipe.core._workflow_execution as execution_mod
import probpipe.core._workflow_managed as managed_mod
import probpipe.core._workflow_replay as replay_mod
import probpipe.core.node as node_mod
from probpipe import (
    Normal,
    Provenance,
    ReplayCompatibilityError,
    replay_run,
    sample,
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


def fake_task(name=None, **task_kwargs):
    del task_kwargs

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


def make_managed_frame():
    return managed_mod.ManagedUnitFrame(
        unit_segment=managed_mod.point_unit_segment(),
        token=managed_mod.ManagedWorkItemToken(b"w" * 16),
    )


def make_managed_attempt():
    return managed_mod.ManagedAttemptState(
        work_item_token=managed_mod.ManagedWorkItemToken(b"w" * 16),
        attempt_token=b"a" * 16,
    )


def make_managed_effect(**overrides):
    values = {
        "occurrence_path": (("invocation", 0),),
        "occurrence_kind": "invocation",
        "stochastic_source_id": ("source-group", 0),
        "logical_unit_id": ("singleton",),
        "operation_kind": "sample",
        "execution_mode": "sampled",
        "sample_shape": (),
        "sampling_abi": "probpipe.distribution_sampling/v1",
        "provider_abi": "probpipe.distribution/v1",
    }
    values.update(overrides)
    return managed_mod.ManagedEffectClaim(**values)


def make_remote_effect(
    envelope,
    *,
    source_index=0,
    child_segment=("child", 0),
    parent_path=None,
    unit_segment=None,
):
    return make_managed_effect(
        occurrence_path=(
            *(envelope.parent_occurrence_path if parent_path is None else parent_path),
            envelope.frame.unit_segment if unit_segment is None else unit_segment,
            child_segment,
        ),
        stochastic_source_id=("source-group", source_index),
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

    def test_execute_many_revalidates_work_items_before_broker_or_callable(self):
        func = Mock(return_value=2)
        request = make_request(calls=[{"x": 1}], func=func)
        object.__setattr__(
            request.work_items[0],
            "values",
            (("x", 1), ("x", 2)),
        )

        with (
            patch.object(broker_mod, "_record_active_execution_contract") as record,
            pytest.raises(ValueError, match="duplicate parameter names"),
        ):
            execution_mod.execute_many(request)

        func.assert_not_called()
        record.assert_not_called()


class TestManagedPayloadValidation:
    @pytest.mark.parametrize(
        ("factory", "exception", "match"),
        [
            pytest.param(
                lambda: managed_mod.ManagedWorkItemToken(b"short"),
                TypeError,
                "exactly 16 bytes",
                id="work-item-token",
            ),
            pytest.param(
                lambda: managed_mod.ManagedUnitFrame(
                    unit_segment=[],
                    token=managed_mod.ManagedWorkItemToken(b"w" * 16),
                ),
                TypeError,
                "unit segments must be tuples",
                id="unit-segment",
            ),
            pytest.param(
                lambda: managed_mod.ManagedUnitFrame(
                    unit_segment=("scope", 0),
                    token=managed_mod.ManagedWorkItemToken(b"w" * 16),
                ),
                ValueError,
                "canonical managed unit segment",
                id="colliding-unit-segment",
            ),
            pytest.param(
                lambda: managed_mod.ManagedUnitFrame(
                    unit_segment=managed_mod.point_unit_segment(),
                    token=managed_mod.ManagedWorkItemToken(b"w" * 16),
                    derivation_abi="unknown-managed/v99",
                ),
                ValueError,
                "unsupported managed work-item ABI",
                id="managed-abi",
            ),
            pytest.param(
                lambda: managed_mod.ManagedWorkItem(
                    index=True,
                    values=(("x", 1),),
                    frame=make_managed_frame(),
                ),
                TypeError,
                "indexes must be non-negative integers",
                id="work-item-index",
            ),
            pytest.param(
                lambda: managed_mod.ManagedWorkItem(
                    index=0,
                    values=[("x", 1)],
                    frame=make_managed_frame(),
                ),
                TypeError,
                "values must be name/value tuples",
                id="work-item-values",
            ),
            pytest.param(
                lambda: managed_mod.ManagedWorkItem(
                    index=0,
                    values=(("x", 1), ("x", 2)),
                    frame=make_managed_frame(),
                ),
                ValueError,
                "duplicate parameter names",
                id="work-item-duplicate-values",
            ),
            pytest.param(
                lambda: managed_mod.ManagedWorkItem(
                    index=0,
                    values=(("x", 1),),
                    frame=object(),
                ),
                TypeError,
                "valid managed unit frame",
                id="work-item-frame",
            ),
            pytest.param(
                lambda: managed_mod.ManagedAttemptState(
                    work_item_token=managed_mod.ManagedWorkItemToken(b"w" * 16),
                    attempt_token=b"short",
                ),
                TypeError,
                "attempt tokens must contain exactly 16 bytes",
                id="attempt-token",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(True, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                ),
                TypeError,
                "root words must be two uint32 integers",
                id="parent-root-words",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=[],
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                ),
                TypeError,
                "parent occurrence paths must be tuples",
                id="parent-occurrence-path",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    replay_expected_effects=[],
                ),
                TypeError,
                "replay expectations must be effect tuples or None",
                id="parent-replay-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    retry_effects=[],
                ),
                TypeError,
                "retry effects must be an effect tuple",
                id="parent-retry-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    replay_expected_effects=(make_managed_effect(), make_managed_effect()),
                ),
                ValueError,
                "replay expectations contains a duplicate effect identity",
                id="parent-duplicate-replay-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    retry_effects=(make_managed_effect(), make_managed_effect()),
                ),
                ValueError,
                "retry effects contains a duplicate effect identity",
                id="parent-duplicate-retry-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_managed_frame(),
                    attempt=managed_mod.ManagedAttemptState(
                        work_item_token=managed_mod.ManagedWorkItemToken(b"z" * 16),
                        attempt_token=b"a" * 16,
                    ),
                ),
                ValueError,
                "attempt must own its frame",
                id="parent-attempt-frame",
            ),
            pytest.param(
                lambda: managed_mod.ManagedPrefectPayload(
                    item=managed_mod.ManagedWorkItem(
                        index=0,
                        values=(),
                        frame=make_managed_frame(),
                    ),
                    attempt=managed_mod.ManagedAttemptState(
                        work_item_token=managed_mod.ManagedWorkItemToken(b"z" * 16),
                        attempt_token=b"a" * 16,
                    ),
                ),
                ValueError,
                "attempt must own its work item",
                id="payload-attempt-item",
            ),
            pytest.param(
                lambda: make_managed_effect(occurrence_path=[]),
                TypeError,
                "effect occurrence paths must be tuples",
                id="effect-occurrence-path",
            ),
            pytest.param(
                lambda: make_managed_effect(stochastic_source_id=[]),
                TypeError,
                "source and unit identities must be tuples",
                id="effect-source",
            ),
            pytest.param(
                lambda: make_managed_effect(occurrence_kind=1),
                TypeError,
                "occurrence_kind must be a string",
                id="effect-occurrence-kind-type",
            ),
            pytest.param(
                lambda: make_managed_effect(occurrence_kind="nested"),
                ValueError,
                "occurrence_kind must be 'invocation' or 'operation'",
                id="effect-occurrence-kind-value",
            ),
            pytest.param(
                lambda: make_managed_effect(operation_kind=1),
                TypeError,
                "operation_kind must be a string",
                id="effect-operation-kind-type",
            ),
            pytest.param(
                lambda: make_managed_effect(operation_kind=""),
                ValueError,
                "operation_kind must be non-empty",
                id="effect-operation-kind-empty",
            ),
            pytest.param(
                lambda: make_managed_effect(execution_mode=None),
                TypeError,
                "execution_mode must be a string",
                id="effect-execution-mode-type",
            ),
            pytest.param(
                lambda: make_managed_effect(execution_mode=""),
                ValueError,
                "execution_mode must be non-empty",
                id="effect-execution-mode-empty",
            ),
            pytest.param(
                lambda: make_managed_effect(sampling_abi=1),
                TypeError,
                "sampling_abi must be a string",
                id="effect-sampling-abi-type",
            ),
            pytest.param(
                lambda: make_managed_effect(sampling_abi=""),
                ValueError,
                "sampling_abi must be non-empty",
                id="effect-sampling-abi-empty",
            ),
            pytest.param(
                lambda: make_managed_effect(provider_abi=None),
                TypeError,
                "provider_abi must be a string",
                id="effect-provider-abi-type",
            ),
            pytest.param(
                lambda: make_managed_effect(provider_abi=""),
                ValueError,
                "provider_abi must be non-empty",
                id="effect-provider-abi-empty",
            ),
            pytest.param(
                lambda: make_managed_effect(sample_shape=[]),
                TypeError,
                "sample shapes must be tuples or None",
                id="effect-sample-shape",
            ),
            pytest.param(
                lambda: make_managed_effect(sample_shape=(True,)),
                TypeError,
                "sample shape dimensions must be non-boolean integers",
                id="effect-sample-shape-bool",
            ),
            pytest.param(
                lambda: make_managed_effect(sample_shape=(-1,)),
                ValueError,
                "sample shape dimensions must be non-negative",
                id="effect-sample-shape-negative",
            ),
            pytest.param(
                lambda: make_managed_effect(record_path=[]),
                TypeError,
                "record paths must be tuples",
                id="effect-record-path",
            ),
            pytest.param(
                lambda: make_managed_effect(record_path=("field", 1)),
                TypeError,
                "record path fields must be strings",
                id="effect-record-path-item",
            ),
            pytest.param(
                lambda: make_managed_effect(descendant_descriptor=[]),
                TypeError,
                "descendant descriptors must be tuples or None",
                id="effect-descendant",
            ),
            pytest.param(
                lambda: make_managed_effect(descendant_descriptor=("descriptor", ["mutable"])),
                TypeError,
                "canonical tuple values",
                id="effect-descendant-nested-list",
            ),
            pytest.param(
                lambda: make_managed_effect(descendant_descriptor=("descriptor", 1.5)),
                TypeError,
                "canonical tuple values",
                id="effect-descendant-float",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=True,
                ),
                TypeError,
                "child counts must be non-negative integers",
                id="report-child-count",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=0,
                    effects=[],
                ),
                TypeError,
                "reports must contain a tuple of effects",
                id="report-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=0,
                    successful_effects=[],
                ),
                TypeError,
                "successful effects must contain a tuple",
                id="report-successful-effects",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=0,
                    successful_effects=(make_managed_effect(),),
                ),
                ValueError,
                "must be claimed by the same attempt",
                id="report-successful-subset",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=managed_mod.ManagedAttemptState(
                        work_item_token=managed_mod.ManagedWorkItemToken(b"z" * 16),
                        attempt_token=b"a" * 16,
                    ),
                    child_count=0,
                ),
                ValueError,
                "attempt must own its frame",
                id="report-attempt-frame",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=1,
                    effects=(make_managed_effect(), make_managed_effect()),
                ),
                ValueError,
                "duplicate effect identity",
                id="report-duplicate-effect",
            ),
            pytest.param(
                lambda: managed_mod.ManagedClaimReport(
                    frame=make_managed_frame(),
                    attempt=make_managed_attempt(),
                    child_count=1,
                    effects=(make_managed_effect(),),
                    successful_effects=(make_managed_effect(), make_managed_effect()),
                ),
                ValueError,
                "duplicate effect identity",
                id="report-duplicate-successful-effect",
            ),
            pytest.param(
                lambda: managed_mod.make_managed_work_items(
                    [{"x": 1}],
                    unit_segments=(),
                ),
                ValueError,
                "must have equal lengths",
                id="work-item-count",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(index=True),
                TypeError,
                "outcome indexes must be non-negative integers",
                id="outcome-index",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(
                    index=0,
                    coordination_required=1,
                ),
                TypeError,
                "coordination flag must be a bool",
                id="outcome-coordination-flag",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(index=0, error="failure"),
                TypeError,
                "outcome errors must be Exception values or None",
                id="outcome-error",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(index=0, report=object()),
                TypeError,
                "outcome reports must be claim reports or None",
                id="outcome-report",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(
                    index=0,
                    error=RuntimeError("failure"),
                    coordination_required=True,
                ),
                ValueError,
                "coordination outcomes cannot contain errors",
                id="outcome-coordination-error",
            ),
            pytest.param(
                lambda: managed_mod.ManagedExecutionOutcome(
                    index=0,
                    value="success",
                    error=RuntimeError("failure"),
                ),
                ValueError,
                "error outcomes cannot contain successful values",
                id="outcome-error-value",
            ),
            pytest.param(
                lambda: managed_mod.sweep_unit_segment(()),
                ValueError,
                "at least one coordinate",
                id="empty-sweep-segment",
            ),
            pytest.param(
                lambda: managed_mod.sweep_unit_segment((True,)),
                TypeError,
                "non-boolean integers",
                id="bool-sweep-coordinate",
            ),
            pytest.param(
                lambda: managed_mod.sweep_unit_segment((-1,)),
                ValueError,
                "non-negative",
                id="negative-sweep-coordinate",
            ),
        ],
    )
    def test_invalid_transport_payloads_fail_at_construction(self, factory, exception, match):
        with pytest.raises(exception, match=match):
            factory()


class TestManagedRetryClaims:
    def test_child_exports_preseal_snapshot_while_parent_is_open(self, monkeypatch):
        operation_brokers = []
        lifecycle_pairs = []
        original_publish = broker_mod._AutomaticKeyBroker._publish_managed_effects

        def observe_publish(broker, effects):
            if broker._managed_attempt is not None:
                lifecycle_pairs.append(
                    (
                        broker._lifecycle,
                        broker._managed_attempt.parent_broker._lifecycle,
                        effects,
                    )
                )
            return original_publish(broker, effects)

        monkeypatch.setattr(
            broker_mod._AutomaticKeyBroker,
            "_publish_managed_effects",
            observe_publish,
        )

        def claim_inside_operation():
            with broker_mod._managed_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()

        request = make_request(calls=[{}], func=claim_inside_operation)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            execution_mod.execute_many(request)

        assert len(operation_brokers) == 1
        assert len(lifecycle_pairs) == 1
        child_status, parent_status, effects = lifecycle_pairs[0]
        assert child_status is broker_mod._BrokerLifecycle.FINALIZING
        assert parent_status is broker_mod._BrokerLifecycle.OPEN
        assert len(effects) == 1
        assert operation_brokers[0]._lifecycle is broker_mod._BrokerLifecycle.SEALED
        with pytest.raises(RuntimeError, match="finalizing"):
            operation_brokers[0]._publish_managed_effects(effects)

    def test_failed_operation_scope_seals_without_publishing_effects(self):
        operation_brokers = []

        def fail_after_claim():
            with broker_mod._managed_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()
                raise RuntimeError("operation failed")

        request = make_request(calls=[{}], func=fail_after_claim)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="operation failed"):
                execution_mod.execute_many(request)
            assert parent._effects_by_identity == {}

        assert len(operation_brokers) == 1
        assert operation_brokers[0]._lifecycle is broker_mod._BrokerLifecycle.SEALED

    def test_managed_operation_broker_closes_before_publishing_effects(self):
        operation_brokers = []

        def leave_nested_item_unjoined():
            with broker_mod._managed_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()
                nested_request = make_request(calls=[{}], func=lambda: None)
                operation_broker.register_managed_work_items(nested_request.work_items)

        request = make_request(calls=[{}], func=leave_nested_item_unjoined)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="before all managed work items join"):
                execution_mod.execute_many(request)

            assert parent._effects_by_identity == {}

        assert len(operation_brokers) == 1

    def test_successful_managed_operation_broker_is_sealed(self):
        operation_brokers = []

        def capture_operation_broker():
            with broker_mod._managed_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)

        request = make_request(calls=[{}], func=capture_operation_broker)
        with workflow_run(seed=17), broker_mod._function_stochastic_scope():
            execution_mod.execute_many(request)

        assert len(operation_brokers) == 1
        assert operation_brokers[0]._lifecycle is broker_mod._BrokerLifecycle.SEALED

    def test_successful_effects_require_the_active_claiming_attempt(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        claimed = make_managed_effect()
        unclaimed = replace(
            claimed,
            stochastic_source_id=("source-group", 1),
        )

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            parent.begin_managed_attempt(attempt, item.frame)
            parent.claim_managed_effect(
                claimed,
                frame=item.frame,
                attempt=attempt,
            )
            stale_attempt = managed_mod.ManagedAttemptState.create(item.frame.token)

            with pytest.raises(RuntimeError, match="active attempt"):
                parent.accept_successful_managed_effects(
                    (claimed,),
                    frame=item.frame,
                    attempt=stale_attempt,
                )
            assert parent._effects_by_identity == {}

            with pytest.raises(RuntimeError, match="was not claimed"):
                parent.accept_successful_managed_effects(
                    (unclaimed,),
                    frame=item.frame,
                    attempt=attempt,
                )
            assert parent._effects_by_identity == {}

            parent.accept_successful_managed_effects(
                (claimed,),
                frame=item.frame,
                attempt=attempt,
            )
            parent.finish_managed_attempt(attempt)

        assert tuple(parent._effects_by_identity.values()) == (claimed,)

    def test_sealed_parent_rejects_successful_effect_injection(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        effect = make_managed_effect()
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)

        with workflow_run(seed=17):
            with broker_mod._function_stochastic_scope() as parent:
                parent.register_managed_work_items(request.work_items)
                parent.begin_managed_attempt(attempt, item.frame)
                parent.claim_managed_effect(
                    effect,
                    frame=item.frame,
                    attempt=attempt,
                )
                parent.finish_managed_attempt(attempt)

            with pytest.raises(RuntimeError, match="sealed"):
                parent.accept_successful_managed_effects(
                    (effect,),
                    frame=item.frame,
                    attempt=attempt,
                )

        assert parent._effects_by_identity == {}

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
            root_frame = context_mod._capture_active_workflow_frame()
            assert root_frame is not None
            with (
                context_mod._managed_work_item_scope(
                    root_frame,
                    item.frame.unit_segment,
                ),
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

    def test_cancelled_item_cannot_start_later(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)

        with workflow_run(seed=17):
            parent = broker_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_managed_work_items(request.work_items)
            parent.cancel_unstarted_managed_items(request.work_items)
            state = parent._managed_claims.by_token[item.frame.token]
            assert state.status is broker_mod._ManagedUnitStatus.CANCELLED

            with pytest.raises(RuntimeError, match="cancelled"):
                parent.begin_managed_attempt(attempt, item.frame)

    def test_managed_unit_uses_explicit_lifecycle_status(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17):
            parent = broker_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_managed_work_items(request.work_items)
            state = parent._managed_claims.by_token[item.frame.token]
            assert state.status is broker_mod._ManagedUnitStatus.ISSUED

            first_attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            parent.begin_managed_attempt(first_attempt, item.frame)
            assert state.status is broker_mod._ManagedUnitStatus.ACTIVE
            assert state.active_attempt == first_attempt.attempt_token

            parent.finish_managed_attempt(first_attempt)
            assert state.status is broker_mod._ManagedUnitStatus.JOINED
            assert state.active_attempt is None
            assert state.seen_attempts == {first_attempt.attempt_token}

            retry_attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            parent.begin_managed_attempt(retry_attempt, item.frame)
            assert state.status is broker_mod._ManagedUnitStatus.ACTIVE
            assert state.active_attempt == retry_attempt.attempt_token
            assert state.seen_attempts == {
                first_attempt.attempt_token,
                retry_attempt.attempt_token,
            }
            parent.finish_managed_attempt(retry_attempt)
            assert state.status is broker_mod._ManagedUnitStatus.JOINED

    def test_cancel_and_begin_race_has_one_terminal_outcome(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
        barrier = Barrier(2)

        with workflow_run(seed=17):
            parent = broker_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_managed_work_items(request.work_items)

            def cancel():
                barrier.wait()
                parent.cancel_unstarted_managed_items(request.work_items)

            def begin():
                barrier.wait()
                try:
                    parent.begin_managed_attempt(attempt, item.frame)
                except RuntimeError as error:
                    assert "cancelled" in str(error)
                    return broker_mod._ManagedUnitStatus.CANCELLED
                parent.finish_managed_attempt(attempt)
                return broker_mod._ManagedUnitStatus.JOINED

            with ThreadPoolExecutor(max_workers=2) as pool:
                cancel_future = pool.submit(cancel)
                begin_future = pool.submit(begin)
                cancel_future.result()
                outcome = begin_future.result()

            state = parent._managed_claims.by_token[item.frame.token]

        assert state.status is outcome

    def test_function_scope_seals_its_broker(self):
        request = make_request(calls=[{}], func=lambda: None)

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            execution_mod.execute_many(request)

        with pytest.raises(RuntimeError, match="sealed"):
            parent.register_managed_work_items(request.work_items)
        with pytest.raises(RuntimeError, match="sealed"):
            parent.begin_managed_attempt(
                managed_mod.ManagedAttemptState.create(request.work_items[0].frame.token),
                request.work_items[0].frame,
            )

    def test_frame_mismatch_fails_before_attempt_reservation(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
        wrong_frame = managed_mod.ManagedUnitFrame(
            unit_segment=managed_mod.sweep_unit_segment((0,)),
            token=item.frame.token,
        )

        with workflow_run(seed=17):
            parent = broker_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_managed_work_items(request.work_items)

            with (
                pytest.raises(RuntimeError, match="frame does not match"),
                broker_mod._managed_work_item_stochastic_scope(
                    parent,
                    wrong_frame,
                    attempt=attempt,
                ),
            ):
                pass

            with (
                context_mod._managed_work_item_scope(
                    context_mod._capture_active_workflow_frame(),
                    item.frame.unit_segment,
                ),
                broker_mod._managed_work_item_stochastic_scope(
                    parent,
                    item.frame,
                    attempt=attempt,
                ),
            ):
                pass

    def test_missing_managed_frame_does_not_reserve_attempt(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        parent = broker_mod._AutomaticKeyBroker("invocation")
        parent.register_managed_work_items(request.work_items)
        with (
            pytest.raises(RuntimeError, match="active workflow frame"),
            broker_mod._managed_work_item_stochastic_scope(parent, item.frame),
        ):
            pass

        parent.cancel_unstarted_managed_items(request.work_items)
        parent.assert_managed_items_joined(request.work_items)

    def test_context_installation_failure_releases_attempt(self, monkeypatch):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        class BrokenContextVar:
            def set(self, value):
                del value
                raise RuntimeError("context installation failed")

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            root_frame = context_mod._capture_active_workflow_frame()
            assert root_frame is not None
            with context_mod._managed_work_item_scope(
                root_frame,
                item.frame.unit_segment,
            ):
                monkeypatch.setattr(
                    broker_mod,
                    "_ACTIVE_MANAGED_ATTEMPT",
                    BrokenContextVar(),
                )
                with (
                    pytest.raises(RuntimeError, match="context installation failed"),
                    broker_mod._managed_work_item_stochastic_scope(parent, item.frame),
                ):
                    pass

            parent.assert_managed_items_joined(request.work_items)

    def test_remote_attempt_is_reserved_before_submission(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        first = managed_mod.ManagedAttemptState.create(item.frame.token)
        second = managed_mod.ManagedAttemptState.create(item.frame.token)

        with workflow_run(seed=17):
            parent = broker_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_managed_work_items(request.work_items)

            assert (
                parent.reserve_remote_managed_attempt(
                    item.frame,
                    first,
                    parent_authority=False,
                )
                is None
            )
            with pytest.raises(RuntimeError, match="active attempt"):
                parent.reserve_remote_managed_attempt(
                    item.frame,
                    second,
                    parent_authority=False,
                )

            parent.abort_remote_managed_attempt(first)
            assert (
                parent.reserve_remote_managed_attempt(
                    item.frame,
                    second,
                    parent_authority=False,
                )
                is None
            )
            parent.abort_remote_managed_attempt(second)
            parent.assert_managed_items_joined(request.work_items)


class TestRemoteReportTransactions:
    def test_worker_rejects_mismatched_transported_root_authority(self):
        frame = make_managed_frame()
        attempt = make_managed_attempt()
        envelope = managed_mod.ManagedParentEnvelope(
            root_words=(0, 17),
            parent_occurrence_path=(("invocation", 0),),
            frame=frame,
            attempt=attempt,
        )

        with (
            context_mod._transported_workflow_frame((0, 99)),
            pytest.raises(RuntimeError, match="root authority"),
            broker_mod._remote_managed_work_item_stochastic_scope(envelope, attempt),
        ):
            pass

    def test_sealed_parent_rejects_remote_report(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
        report = managed_mod.ManagedClaimReport(
            frame=item.frame,
            attempt=attempt,
            child_count=0,
        )

        with workflow_run(seed=17):
            with broker_mod._function_stochastic_scope() as parent:
                parent.register_managed_work_items(request.work_items)
                parent.reserve_remote_managed_attempt(
                    item.frame,
                    attempt,
                    parent_authority=False,
                )
                parent.abort_remote_managed_attempt(attempt)

            with pytest.raises(RuntimeError, match="sealed"):
                parent.accept_remote_claim_report(report)

        state = parent._managed_claims.by_token[item.frame.token]
        assert state.effect_claims_by_identity == {}
        assert parent._effects_by_identity == {}

    @pytest.mark.parametrize(
        "violation",
        [
            "attempt",
            "frame",
            "prefix",
            "unit",
            "child-segment",
            "child-range",
        ],
    )
    def test_invalid_remote_report_leaves_parent_ledgers_unchanged(self, violation):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            report_attempt = attempt
            report_frame = item.frame
            child_segment = ("child", 0)
            parent_path = None
            unit_segment = None
            if violation == "attempt":
                report_attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            elif violation == "frame":
                report_frame = managed_mod.ManagedUnitFrame(
                    unit_segment=managed_mod.sweep_unit_segment((9,)),
                    token=item.frame.token,
                )
            elif violation == "prefix":
                parent_path = (("invocation", 9),)
            elif violation == "unit":
                unit_segment = managed_mod.sweep_unit_segment((9,))
            elif violation == "child-segment":
                child_segment = ("scope", 0)
            elif violation == "child-range":
                child_segment = ("child", 1)

            effect = make_remote_effect(
                envelope,
                child_segment=child_segment,
                parent_path=parent_path,
                unit_segment=unit_segment,
            )
            report = managed_mod.ManagedClaimReport(
                frame=report_frame,
                attempt=report_attempt,
                child_count=1,
                effects=(effect,),
                successful_effects=(effect,),
            )
            state = parent._managed_claims.by_token[item.frame.token]
            before_seen = set(state.seen_attempts)

            with pytest.raises(RuntimeError, match=r"remote|occurrence|child"):
                parent.accept_remote_claim_report(report)

            assert state.effect_claims_by_identity == {}
            assert state.child_invocations == []
            assert parent._effects_by_identity == {}
            assert state.seen_attempts == before_seen
            assert state.active_attempt == attempt.attempt_token
            parent.abort_remote_managed_attempt(attempt)

    def test_rootless_probe_report_cannot_inject_claims(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            assert (
                parent.reserve_remote_managed_attempt(
                    item.frame,
                    attempt,
                    parent_authority=False,
                )
                is None
            )
            effect = make_managed_effect()
            report = managed_mod.ManagedClaimReport(
                frame=item.frame,
                attempt=attempt,
                child_count=1,
                effects=(effect,),
            )

            with pytest.raises(RuntimeError, match=r"rootless.*cannot report"):
                parent.accept_remote_claim_report(report)

            state = parent._managed_claims.by_token[item.frame.token]
            assert state.effect_claims_by_identity == {}
            assert state.child_invocations == []
            assert parent._effects_by_identity == {}
            parent.abort_remote_managed_attempt(attempt)

    @pytest.mark.parametrize(
        ("violation", "exception", "match"),
        [
            pytest.param(
                "successful-subset",
                ValueError,
                "claimed by the same attempt",
                id="successful-subset",
            ),
            pytest.param(
                "namespace",
                RuntimeError,
                "outside its managed unit namespace",
                id="namespace",
            ),
            pytest.param(
                "duplicate-identity",
                ValueError,
                "duplicate effect identity",
                id="duplicate-identity",
            ),
            pytest.param(
                "duplicate-successful-identity",
                ValueError,
                "duplicate effect identity",
                id="duplicate-successful-identity",
            ),
            pytest.param(
                "descriptor-drift",
                ValueError,
                "claimed by the same attempt",
                id="descriptor-drift",
            ),
            pytest.param(
                "frame",
                RuntimeError,
                "not registered",
                id="frame",
            ),
            pytest.param(
                "attempt",
                RuntimeError,
                "active reservation",
                id="attempt",
            ),
            pytest.param(
                "child-count",
                TypeError,
                "non-negative integers",
                id="child-count",
            ),
            pytest.param(
                "effect-field",
                ValueError,
                "sample shape dimensions must be non-negative",
                id="effect-field",
            ),
            pytest.param(
                "identity-field",
                TypeError,
                "source and unit identities must be tuples",
                id="identity-field",
            ),
        ],
    )
    def test_parent_revalidates_tampered_report_before_mutation(
        self,
        violation,
        exception,
        match,
    ):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            first = make_remote_effect(envelope, source_index=0)
            second = make_remote_effect(envelope, source_index=1)
            report = managed_mod.ManagedClaimReport(
                frame=item.frame,
                attempt=attempt,
                child_count=1,
                effects=(first, second),
                successful_effects=(first, second),
            )

            if violation == "successful-subset":
                forged = make_remote_effect(envelope, source_index=2)
                object.__setattr__(report, "successful_effects", (forged,))
            elif violation == "namespace":
                object.__setattr__(
                    first,
                    "occurrence_path",
                    (("invocation", 99), item.frame.unit_segment, ("child", 0)),
                )
            elif violation == "duplicate-identity":
                object.__setattr__(report, "effects", (first, first))
            elif violation == "duplicate-successful-identity":
                object.__setattr__(report, "successful_effects", (first, first))
            elif violation == "descriptor-drift":
                object.__setattr__(
                    report,
                    "successful_effects",
                    (replace(first, provider_abi="drifted-provider/v1"),),
                )
            elif violation == "frame":
                object.__setattr__(
                    report,
                    "frame",
                    managed_mod.ManagedUnitFrame(
                        unit_segment=managed_mod.sweep_unit_segment((9,)),
                        token=item.frame.token,
                    ),
                )
            elif violation == "attempt":
                object.__setattr__(
                    report,
                    "attempt",
                    managed_mod.ManagedAttemptState.create(item.frame.token),
                )
            elif violation == "child-count":
                object.__setattr__(report, "child_count", -1)
            elif violation == "effect-field":
                object.__setattr__(second, "sample_shape", (-1,))
            else:
                object.__setattr__(second, "stochastic_source_id", ["invalid"])

            state = parent._managed_claims.by_token[item.frame.token]
            before_managed = (
                dict(state.effect_claims_by_identity),
                tuple(state.child_invocations),
                set(state.active_effect_identities),
                state.active_attempt,
                state.active_transport,
                state.active_parent_occurrence_path,
                state.status,
            )
            before_recipe = dict(parent._effects_by_identity)

            with pytest.raises(exception, match=match):
                parent.accept_remote_claim_report(report)

            assert (
                dict(state.effect_claims_by_identity),
                tuple(state.child_invocations),
                set(state.active_effect_identities),
                state.active_attempt,
                state.active_transport,
                state.active_parent_occurrence_path,
                state.status,
            ) == before_managed
            assert parent._effects_by_identity == before_recipe
            parent.abort_remote_managed_attempt(attempt)

    def test_report_construction_revalidates_existing_effect_instances(self):
        frame = make_managed_frame()
        attempt = make_managed_attempt()
        effect = make_managed_effect()
        object.__setattr__(effect, "sample_shape", (-1,))

        with pytest.raises(ValueError, match="sample shape dimensions must be non-negative"):
            managed_mod.ManagedClaimReport(
                frame=frame,
                attempt=attempt,
                child_count=0,
                effects=(effect,),
            )

    def test_pickled_remote_report_is_revalidated_and_committed(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            effect = make_remote_effect(envelope)
            report = managed_mod.ManagedClaimReport(
                frame=item.frame,
                attempt=attempt,
                child_count=1,
                effects=(effect,),
                successful_effects=(effect,),
            )

            transported = pickle.loads(pickle.dumps(report))
            parent.accept_remote_claim_report(transported)

            state = parent._managed_claims.by_token[item.frame.token]
            assert state.status is broker_mod._ManagedUnitStatus.JOINED
            assert tuple(state.effect_claims_by_identity.values()) == (effect,)
            assert tuple(parent._effects_by_identity.values()) == (effect,)

    def test_valid_remote_report_commits_all_ledgers_together(self):
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            effects = (
                make_remote_effect(envelope, source_index=0),
                make_remote_effect(envelope, source_index=1),
            )
            report = managed_mod.ManagedClaimReport(
                frame=item.frame,
                attempt=attempt,
                child_count=1,
                effects=effects,
                successful_effects=effects,
            )

            parent.accept_remote_claim_report(report)

            state = parent._managed_claims.by_token[item.frame.token]
            assert state.status is broker_mod._ManagedUnitStatus.JOINED
            assert state.active_attempt is None
            assert len(state.effect_claims_by_identity) == 2
            assert len(state.child_invocations) == 1
            assert tuple(parent._effects_by_identity.values()) == effects

    def test_second_replay_effect_failure_rolls_back_every_ledger(self):
        with workflow_run(seed=17):
            recorded = sample(Normal(loc=0.0, scale=1.0, name="value"))
        replay_state = replay_mod._validate_provenance(recorded.provenance)
        expected = replay_state.expected_events[0].managed_effect()
        unexpected = replace(
            expected,
            stochastic_source_id=("source-group", 99),
        )
        request = make_request(calls=[{}], func=lambda: None)
        item = request.work_items[0]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent._replay_state = replay_state
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            report = managed_mod.ManagedClaimReport(
                frame=item.frame,
                attempt=attempt,
                child_count=1,
                effects=(expected, unexpected),
                successful_effects=(expected, unexpected),
            )

            with pytest.raises(ReplayCompatibilityError, match="unexpected replay event"):
                parent.accept_remote_claim_report(report)

            state = parent._managed_claims.by_token[item.frame.token]
            assert state.effect_claims_by_identity == {}
            assert state.child_invocations == []
            assert parent._effects_by_identity == {}
            assert all(
                claim.work_item_token is None and not claim.attempt_tokens
                for claim in replay_state.claims.values()
            )
            parent.abort_remote_managed_attempt(attempt)


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


class TestManagedSubmissionGuards:
    @pytest.mark.parametrize(
        "entry_name",
        ["thread", "map_task", "prefect_task", "prefect_flow"],
    )
    @pytest.mark.parametrize(
        ("guard_factory", "exception", "message"),
        [
            pytest.param(
                context_mod._workflow_probe,
                context_mod._StochasticProbeSignal,
                "managed submission",
                id="probe",
            ),
            pytest.param(
                context_mod._workflow_jax_runtime_guard,
                TypeError,
                "managed submission",
                id="jax-runtime",
            ),
        ],
    )
    def test_direct_transport_entry_rejects_side_effect_forbidden_callers(
        self,
        entry_name,
        guard_factory,
        exception,
        message,
        monkeypatch,
    ):
        monkeypatch.setattr(execution_mod, "ThreadPoolExecutor", RecordingExecutor)
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(mode=entry_name if entry_name != "map_task" else "prefect_task")
        entry = {
            "thread": execution_mod.execute_many_threaded,
            "map_task": execution_mod.map_task,
            "prefect_task": execution_mod.execute_many_prefect_task,
            "prefect_flow": execution_mod.execute_many_prefect_flow,
        }[entry_name]

        with pytest.raises(exception, match=message), guard_factory():
            entry(request)

        assert RecordingExecutor.instances == []
        assert FakeMappedTask.map_calls == 0
        assert RecordingFlow.calls == []


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

    def test_prefect_worker_revalidates_pickled_payload_before_callable(self):
        item = make_request(calls=[{"x": 1}]).work_items[0]
        payload = managed_mod.ManagedPrefectPayload(
            item=item,
            attempt=managed_mod.ManagedAttemptState.create(item.frame.token),
        )
        object.__setattr__(item, "values", (("x", 1), ("x", 2)))
        transported = pickle.loads(pickle.dumps(payload))
        func = Mock(return_value=2)

        with pytest.raises(ValueError, match="duplicate parameter names"):
            execution_mod._execute_prefect_payload(func, transported)

        func.assert_not_called()

    def test_prefect_worker_revalidates_nested_parent_authority_before_callable(self):
        item = make_request(calls=[{"x": 1}]).work_items[0]
        attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
        effect = make_managed_effect()
        payload = managed_mod.ManagedPrefectPayload(
            item=item,
            attempt=attempt,
            parent=managed_mod.ManagedParentEnvelope(
                root_words=(0, 17),
                parent_occurrence_path=(("invocation", 0),),
                frame=item.frame,
                attempt=attempt,
                retry_effects=(effect,),
            ),
        )
        object.__setattr__(effect, "sample_shape", (-1,))
        transported = pickle.loads(pickle.dumps(payload))
        func = Mock(return_value=2)

        with pytest.raises(ValueError, match="sample shape dimensions must be non-negative"):
            execution_mod._execute_prefect_payload(func, transported)

        func.assert_not_called()

    @pytest.mark.parametrize(
        "violation",
        [
            "index",
            "frame",
            "attempt",
            "effect",
            "missing-report",
            "rootless-coordination",
            "authorized-coordination",
        ],
    )
    def test_parent_revalidates_pickled_outcome_against_payload(self, violation):
        request = make_request(calls=[{"x": 1}])
        item = request.work_items[0]

        class ReturningTask:
            def __init__(self, outcome):
                self.outcome = outcome

            def map(self, **kwargs_by_param):
                assert kwargs_by_param["payload"]
                return [FakeFuture(self.outcome)]

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            parent.register_managed_work_items(request.work_items)
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=violation == "authorized-coordination",
            )
            payload = managed_mod.ManagedPrefectPayload(
                item=item,
                attempt=attempt,
                parent=envelope,
            )
            report = managed_mod.ManagedClaimReport(item.frame, attempt, 0)
            outcome = managed_mod.ManagedExecutionOutcome(
                index=item.index,
                value=2,
                report=report,
            )
            if violation == "index":
                object.__setattr__(outcome, "index", 9)
            elif violation == "frame":
                other_frame = managed_mod.ManagedUnitFrame(
                    unit_segment=managed_mod.sweep_unit_segment((9,)),
                    token=item.frame.token,
                )
                object.__setattr__(
                    outcome,
                    "report",
                    managed_mod.ManagedClaimReport(other_frame, attempt, 0),
                )
            elif violation == "attempt":
                other_attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
                object.__setattr__(
                    outcome,
                    "report",
                    managed_mod.ManagedClaimReport(item.frame, other_attempt, 0),
                )
            elif violation == "effect":
                effect = make_managed_effect()
                effect_report = managed_mod.ManagedClaimReport(
                    item.frame,
                    attempt,
                    0,
                    effects=(effect,),
                )
                object.__setattr__(effect, "sample_shape", (-1,))
                object.__setattr__(outcome, "report", effect_report)
            elif violation == "missing-report":
                object.__setattr__(outcome, "report", None)
            elif violation == "rootless-coordination":
                object.__setattr__(outcome, "value", None)
                object.__setattr__(outcome, "coordination_required", True)
                object.__setattr__(
                    outcome,
                    "report",
                    managed_mod.ManagedClaimReport(item.frame, attempt, 1),
                )
            else:
                object.__setattr__(outcome, "value", None)
                object.__setattr__(outcome, "coordination_required", True)

            transported = pickle.loads(pickle.dumps(outcome))
            state = parent._managed_claims.by_token[item.frame.token]
            before = (
                state.status,
                state.active_attempt,
                dict(state.effect_claims_by_identity),
                tuple(state.child_invocations),
                dict(parent._effects_by_identity),
            )

            try:
                with pytest.raises(
                    (RuntimeError, ValueError),
                    match=r"outcome|report|coordination|sample shape",
                ):
                    execution_mod._run_prefect_payloads(ReturningTask(transported), [payload])

                assert (
                    state.status,
                    state.active_attempt,
                    dict(state.effect_claims_by_identity),
                    tuple(state.child_invocations),
                    dict(parent._effects_by_identity),
                ) == before
            finally:
                parent.abort_remote_managed_attempt(attempt)

    def test_prefect_worker_failure_aborts_parent_reservation(self, monkeypatch):
        class CrashingFuture:
            def result(self):
                raise RuntimeError("worker crashed")

        class CrashingTask:
            def map(self, **kwargs_by_param):
                assert kwargs_by_param["payload"]
                return [CrashingFuture()]

        def crashing_task(name=None, **task_kwargs):
            del name, task_kwargs

            def decorator(func):
                del func
                return CrashingTask()

            return decorator

        monkeypatch.setattr(execution_mod, "task", crashing_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        request = make_request(mode="prefect_task", calls=[{}], func=lambda: None)

        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="worker crashed"):
                execution_mod.execute_many(request)
            state = parent._managed_claims.by_token[request.work_items[0].frame.token]

        assert state.status is broker_mod._ManagedUnitStatus.JOINED
        assert state.active_attempt is None
        assert len(state.seen_attempts) == 1

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
            item = request.work_items[0]
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            parent.abort_remote_managed_attempt(attempt)

        assert result
        assert envelope.parent_occurrence_path == (("invocation", 0),)
        assert len(parent._effects_by_identity) == 1
        urandom.assert_called_once_with(8)
        assert FakeMappedTask.map_calls == 2

    @pytest.mark.parametrize(
        "handler",
        [
            "runtime-return",
            "exception-return",
            "runtime-reraise",
            "exception-reraise",
        ],
    )
    def test_prefect_probe_observation_cannot_be_swallowed(self, handler, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        catch_type = RuntimeError if handler.startswith("runtime") else Exception

        def handle_probe():
            try:
                return _claim_automatic_words()
            except catch_type:
                if handler.endswith("return"):
                    return "non-authoritative fallback"
                raise ValueError("replacement probe error") from None

        request = make_request(
            mode="prefect_task",
            calls=[{}],
            func=handle_probe,
        )
        with workflow_run(seed=17), broker_mod._function_stochastic_scope() as parent:
            result = execution_mod.execute_many(request)[0]
            state = parent._managed_claims.by_token[request.work_items[0].frame.token]

        assert isinstance(result, tuple)
        assert len(parent._effects_by_identity) == 1
        assert len(state.seen_attempts) == 2
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
        assert len(claim_state.seen_attempts) == 3
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
            item = request.work_items[0]
            attempt = managed_mod.ManagedAttemptState.create(item.frame.token)
            envelope = parent.reserve_remote_managed_attempt(
                item.frame,
                attempt,
                parent_authority=True,
            )
            assert envelope is not None
            payload = managed_mod.ManagedPrefectPayload(
                item=item,
                attempt=attempt,
                parent=envelope,
            )
            parent.abort_remote_managed_attempt(attempt)

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
                "probpipe.core._workflow_context.derive_event_key_words_from_encoded",
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
