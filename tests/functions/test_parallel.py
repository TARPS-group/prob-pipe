"""Tests for the key broker of evaluation points, and for running a call's evaluation points on threads and as Prefect tasks.

The key broker gives each evaluation point its random keys, whatever runs it,
such as a pool of threads or a Prefect task. Its tests issue evaluation
points and run each in the workflow scope the broker gives it. The Prefect tests replace Prefect's
``task`` and ``flow`` with recording fakes, so they run without a Prefect
server.
"""

from __future__ import annotations

import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Barrier, Event
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import probpipe.core._executors as executors_mod
import probpipe.functions._context as context_mod
import probpipe.functions._evaluation_keys as evaluation_keys_mod
import probpipe.functions._execution as execution_mod
import probpipe.functions._replay as replay_mod
from probpipe import (
    EvaluationError,
    EvaluationPoints,
    Function,
    Normal,
    OutputSpec,
    Provenance,
    ReplayCompatibilityError,
    executor_registry,
    replay_run,
    sample,
    workflow_run,
)
from probpipe.core.config import ProvenanceMode, WorkflowKind, config
from tests.core._requester_calls import requester, requester_call

from ._design_helpers import atom_leaves, standard_normal


def add_one(x):
    return x + 1


def _claim_automatic_key():
    return evaluation_keys_mod._resolve_automatic_key(
        None,
        evaluation_keys_mod._singleton_effect_plan(
            operation_kind="evaluation-point-test",
            execution_mode="sampled",
            sample_shape=(),
        ),
    )


def _claim_automatic_words():
    return tuple(int(word) for word in jax.random.key_data(_claim_automatic_key()))


def _claim_automatic_scalar():
    return jax.random.key_data(_claim_automatic_key())[0]


def _claim_nested_seeded_scalar():
    with workflow_run(seed=42):
        return _claim_automatic_scalar()


def _claim_words_at_point(x):
    return int(x), _claim_automatic_words()


def _noisy_draw(z):
    return jnp.stack(
        [jnp.asarray(z, dtype=jnp.float32), jax.random.uniform(_claim_automatic_key())]
    )


class FakeFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class FakeMappedTask:
    """A Prefect task that runs each chunk it maps in place, or pickled in a fresh thread as a worker does."""

    created_names: ClassVar[list[str | None]] = []
    run_in_reverse: ClassVar[bool] = False
    ship_chunks: ClassVar[bool] = False
    map_calls: ClassVar[int] = 0

    def __init__(self, fn, name):
        self.fn = fn
        self.name = name
        self.__class__.created_names.append(name)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def map(self, *, chunk):
        self.__class__.map_calls += 1
        point_order = range(len(chunk))
        outcomes = [None] * len(chunk)
        for index in reversed(point_order) if self.__class__.run_in_reverse else point_order:
            outcomes[index] = self._run(chunk[index])
        return [FakeFuture(chunk_outcomes) for chunk_outcomes in outcomes]

    def _run(self, chunk):
        if not self.__class__.ship_chunks:
            return self.fn(chunk=chunk)
        shipped_chunk = cloudpickle.dumps(chunk)
        # A fresh thread starts with no context, as a worker process does.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: self.fn(chunk=cloudpickle.loads(shipped_chunk))).result()


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


def issue_points(num_points=1):
    """*num_points* issued evaluation points: one plain-call point, or one sweep cell each."""
    point_segments = (
        (execution_mod.plain_call_point_segment(),)
        if num_points == 1
        else tuple(execution_mod.sweep_cell_point_segment((index,)) for index in range(num_points))
    )
    return execution_mod.issue_evaluation_points(point_segments)


def run_at_points(evaluation_points, func):
    """Run *func* at each issued evaluation point, in point order, in the scope the active key broker gives it.

    The evaluation points are registered with the active broker first. Once
    they ran, or one raised, every unstarted one is cancelled and the broker
    checks that every one joined.
    """
    parent_frame = context_mod._capture_active_workflow_frame()
    parent_broker = evaluation_keys_mod._capture_active_broker()
    parent_broker.register_evaluation_points(evaluation_points)
    try:
        outputs = []
        for evaluation_point in evaluation_points:
            with execution_mod._evaluation_point_scope(
                evaluation_point, parent_frame, parent_broker
            ):
                outputs.append(func())
        return outputs
    finally:
        parent_broker.cancel_unstarted_evaluation_points(evaluation_points)
        parent_broker.assert_evaluation_points_joined(evaluation_points)


def points_of(fn, num_points=1, *, origin="plus_one"):
    """A set of evaluation points of *fn* at ``x = 1, ..., num_points``, built under the running call."""
    return EvaluationPoints(fn, varying={"x": np.arange(1, num_points + 1)}, origin=origin)


def issued_point(evaluation_points, point_index=0):
    """The issued evaluation point *point_index* of a set that ran in scopes of its own."""
    return evaluation_points._point_scopes._issued_points[point_index]


def run_on(executor_name, evaluation_points, *, workflow_kind=WorkflowKind.TASK):
    """The outputs of *evaluation_points* on the named executor, or as Prefect tasks for ``"prefect"``."""
    executor = (
        execution_mod._PrefectTransport(
            workflow_kind, evaluation_points.origin, maps_with_jax=False
        )
        if executor_name == "prefect"
        else executor_registry.get_method(executor_name)
    )
    evaluation_outcomes = executors_mod._run_on_executor(evaluation_points, executor, {})
    return executors_mod._outputs_of(evaluation_points, evaluation_outcomes, stacked=False)


def make_evaluation_point_frame():
    return evaluation_keys_mod.EvaluationPointFrame(
        point_segment=evaluation_keys_mod.plain_call_point_segment(),
        token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
    )


def make_evaluation_point_attempt():
    return evaluation_keys_mod.EvaluationPointAttempt(
        evaluation_point_token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
        attempt_token=b"a" * 16,
    )


def make_effect_claim(**overrides):
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
    return evaluation_keys_mod.StochasticEffectClaim(**values)


def make_remote_effect(
    envelope,
    *,
    source_index=0,
    child_segment=("child", 0),
    parent_path=None,
    point_segment=None,
):
    return make_effect_claim(
        occurrence_path=(
            *(envelope.parent_occurrence_path if parent_path is None else parent_path),
            envelope.frame.point_segment if point_segment is None else point_segment,
            child_segment,
        ),
        stochastic_source_id=("source-group", source_index),
    )


def _reset_fakes():
    config.prefect.workflow_kind = WorkflowKind.OFF
    config.prefect.task_runner = None
    FakeMappedTask.created_names.clear()
    FakeMappedTask.run_in_reverse = False
    FakeMappedTask.ship_chunks = False
    FakeMappedTask.map_calls = 0
    RecordingFlow.calls.clear()


@pytest.fixture(autouse=True)
def _fresh_fakes():
    _reset_fakes()
    yield
    _reset_fakes()


@pytest.fixture
def prefect_fakes(monkeypatch):
    """Prefect's ``task`` and ``flow`` replaced by the recording fakes, with no retries."""
    monkeypatch.setattr(execution_mod, "task", fake_task)
    monkeypatch.setattr(execution_mod, "flow", fake_flow)
    monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (0, ()))


class TestIssuedEvaluationPoints:
    def test_issued_evaluation_points_are_frozen_tuple_only_and_pickleable(self):
        evaluation_points = issue_points()

        assert isinstance(evaluation_points, tuple)
        assert evaluation_points[0].frame.point_segment == (
            "evaluation-point",
            "probpipe.evaluation_point/v1",
            "point",
            0,
        )
        assert len(evaluation_points[0].frame.token.value) == 16
        assert pickle.loads(pickle.dumps(evaluation_points)) == evaluation_points
        with pytest.raises(FrozenInstanceError):
            evaluation_points[0].index = 9

    def test_evaluation_point_registration_failure_is_transactional_and_preserves_cause(self):
        evaluation_points = issue_points(2)
        duplicate_token = replace(
            evaluation_points[1],
            frame=replace(evaluation_points[1].frame, token=evaluation_points[0].frame.token),
        )

        with (
            workflow_run(seed=17),
            pytest.raises(
                RuntimeError,
                match="evaluation point token cannot own multiple point segments",
            ),
            evaluation_keys_mod._function_stochastic_scope() as parent,
        ):
            parent.register_evaluation_points((evaluation_points[0], duplicate_token))

        assert parent._evaluation_point_claims.by_point_segment == {}
        assert parent._evaluation_point_claims.by_token == {}
        assert parent._lifecycle is evaluation_keys_mod._BrokerLifecycle.SEALED


class TestEvaluationPointValueValidation:
    @pytest.mark.parametrize(
        ("factory", "exception", "match"),
        [
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointToken(b"short"),
                TypeError,
                "exactly 16 bytes",
                id="evaluation-point-token",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointFrame(
                    point_segment=[],
                    token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
                ),
                TypeError,
                "point segments must be tuples",
                id="point-segment",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointFrame(
                    point_segment=("scope", 0),
                    token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
                ),
                ValueError,
                "canonical evaluation point segment",
                id="colliding-point-segment",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointFrame(
                    point_segment=evaluation_keys_mod.plain_call_point_segment(),
                    token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
                    derivation_abi="unknown-evaluation-point/v99",
                ),
                ValueError,
                "unsupported evaluation point ABI",
                id="evaluation-point-abi",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.IssuedEvaluationPoint(
                    index=True,
                    frame=make_evaluation_point_frame(),
                ),
                TypeError,
                "indexes must be non-negative integers",
                id="evaluation-point-index",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.IssuedEvaluationPoint(
                    index=0,
                    frame=object(),
                ),
                TypeError,
                "valid evaluation point frame",
                id="evaluation-point-frame",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointAttempt(
                    evaluation_point_token=evaluation_keys_mod.EvaluationPointToken(b"w" * 16),
                    attempt_token=b"short",
                ),
                TypeError,
                "attempt tokens must contain exactly 16 bytes",
                id="attempt-token",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(True, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                ),
                TypeError,
                "root words must be two uint32 integers",
                id="parent-root-words",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=[],
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                ),
                TypeError,
                "parent occurrence paths must be tuples",
                id="parent-occurrence-path",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    replay_expected_effects=[],
                ),
                TypeError,
                "replay expectations must be effect tuples or None",
                id="parent-replay-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    retry_effects=[],
                ),
                TypeError,
                "retry effects must be an effect tuple",
                id="parent-retry-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    replay_expected_effects=(make_effect_claim(), make_effect_claim()),
                ),
                ValueError,
                "replay expectations contains a duplicate effect identity",
                id="parent-duplicate-replay-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    retry_effects=(make_effect_claim(), make_effect_claim()),
                ),
                ValueError,
                "retry effects contains a duplicate effect identity",
                id="parent-duplicate-retry-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointParentEnvelope(
                    root_words=(0, 1),
                    parent_occurrence_path=(("invocation", 0),),
                    frame=make_evaluation_point_frame(),
                    attempt=evaluation_keys_mod.EvaluationPointAttempt(
                        evaluation_point_token=evaluation_keys_mod.EvaluationPointToken(b"z" * 16),
                        attempt_token=b"a" * 16,
                    ),
                ),
                ValueError,
                "attempt must own its frame",
                id="parent-attempt-frame",
            ),
            pytest.param(
                lambda: make_effect_claim(occurrence_path=[]),
                TypeError,
                "effect occurrence paths must be tuples",
                id="effect-occurrence-path",
            ),
            pytest.param(
                lambda: make_effect_claim(stochastic_source_id=[]),
                TypeError,
                "source and unit identities must be tuples",
                id="effect-source",
            ),
            pytest.param(
                lambda: make_effect_claim(occurrence_kind=1),
                TypeError,
                "occurrence_kind must be a string",
                id="effect-occurrence-kind-type",
            ),
            pytest.param(
                lambda: make_effect_claim(occurrence_kind="nested"),
                ValueError,
                "occurrence_kind must be 'invocation' or 'operation'",
                id="effect-occurrence-kind-value",
            ),
            pytest.param(
                lambda: make_effect_claim(operation_kind=1),
                TypeError,
                "operation_kind must be a string",
                id="effect-operation-kind-type",
            ),
            pytest.param(
                lambda: make_effect_claim(operation_kind=""),
                ValueError,
                "operation_kind must be non-empty",
                id="effect-operation-kind-empty",
            ),
            pytest.param(
                lambda: make_effect_claim(execution_mode=None),
                TypeError,
                "execution_mode must be a string",
                id="effect-execution-mode-type",
            ),
            pytest.param(
                lambda: make_effect_claim(execution_mode=""),
                ValueError,
                "execution_mode must be non-empty",
                id="effect-execution-mode-empty",
            ),
            pytest.param(
                lambda: make_effect_claim(sampling_abi=1),
                TypeError,
                "sampling_abi must be a string",
                id="effect-sampling-abi-type",
            ),
            pytest.param(
                lambda: make_effect_claim(sampling_abi=""),
                ValueError,
                "sampling_abi must be non-empty",
                id="effect-sampling-abi-empty",
            ),
            pytest.param(
                lambda: make_effect_claim(provider_abi=None),
                TypeError,
                "provider_abi must be a string",
                id="effect-provider-abi-type",
            ),
            pytest.param(
                lambda: make_effect_claim(provider_abi=""),
                ValueError,
                "provider_abi must be non-empty",
                id="effect-provider-abi-empty",
            ),
            pytest.param(
                lambda: make_effect_claim(sample_shape=[]),
                TypeError,
                "sample shapes must be tuples or None",
                id="effect-sample-shape",
            ),
            pytest.param(
                lambda: make_effect_claim(sample_shape=(True,)),
                TypeError,
                "sample shape dimensions must be non-boolean integers",
                id="effect-sample-shape-bool",
            ),
            pytest.param(
                lambda: make_effect_claim(sample_shape=(-1,)),
                ValueError,
                "sample shape dimensions must be non-negative",
                id="effect-sample-shape-negative",
            ),
            pytest.param(
                lambda: make_effect_claim(record_path=[]),
                TypeError,
                "record paths must be tuples",
                id="effect-record-path",
            ),
            pytest.param(
                lambda: make_effect_claim(record_path=("field", 1)),
                TypeError,
                "record path fields must be strings",
                id="effect-record-path-item",
            ),
            pytest.param(
                lambda: make_effect_claim(descendant_descriptor=[]),
                TypeError,
                "descendant descriptors must be tuples or None",
                id="effect-descendant",
            ),
            pytest.param(
                lambda: make_effect_claim(descendant_descriptor=("descriptor", ["mutable"])),
                TypeError,
                "canonical tuple values",
                id="effect-descendant-nested-list",
            ),
            pytest.param(
                lambda: make_effect_claim(descendant_descriptor=("descriptor", 1.5)),
                TypeError,
                "canonical tuple values",
                id="effect-descendant-float",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=True,
                ),
                TypeError,
                "child counts must be non-negative integers",
                id="report-child-count",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=0,
                    effects=[],
                ),
                TypeError,
                "reports must contain a tuple of effects",
                id="report-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=0,
                    successful_effects=[],
                ),
                TypeError,
                "successful evaluation point effects must contain a tuple",
                id="report-successful-effects",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=0,
                    successful_effects=(make_effect_claim(),),
                ),
                ValueError,
                "must be claimed by the same attempt",
                id="report-successful-subset",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=evaluation_keys_mod.EvaluationPointAttempt(
                        evaluation_point_token=evaluation_keys_mod.EvaluationPointToken(b"z" * 16),
                        attempt_token=b"a" * 16,
                    ),
                    child_count=0,
                ),
                ValueError,
                "attempt must own its frame",
                id="report-attempt-frame",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=1,
                    effects=(make_effect_claim(), make_effect_claim()),
                ),
                ValueError,
                "duplicate effect identity",
                id="report-duplicate-effect",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.EvaluationPointClaimReport(
                    frame=make_evaluation_point_frame(),
                    attempt=make_evaluation_point_attempt(),
                    child_count=1,
                    effects=(make_effect_claim(),),
                    successful_effects=(make_effect_claim(), make_effect_claim()),
                ),
                ValueError,
                "duplicate effect identity",
                id="report-duplicate-successful-effect",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.sweep_cell_point_segment(()),
                ValueError,
                "at least one coordinate",
                id="empty-sweep-segment",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.sweep_cell_point_segment((True,)),
                TypeError,
                "non-boolean integers",
                id="bool-sweep-coordinate",
            ),
            pytest.param(
                lambda: evaluation_keys_mod.sweep_cell_point_segment((-1,)),
                ValueError,
                "non-negative",
                id="negative-sweep-coordinate",
            ),
        ],
    )
    def test_invalid_evaluation_point_values_fail_at_construction(self, factory, exception, match):
        with pytest.raises(exception, match=match):
            factory()


class TestEvaluationPointRetryClaims:
    def test_child_exports_preseal_snapshot_while_parent_is_open(self, monkeypatch):
        operation_brokers = []
        lifecycle_pairs = []
        original_publish = (
            evaluation_keys_mod._AutomaticKeyBroker._publish_effects_to_evaluation_point
        )

        def observe_publish(broker, effects):
            if broker._evaluation_point_attempt is not None:
                lifecycle_pairs.append(
                    (
                        broker._lifecycle,
                        broker._evaluation_point_attempt.parent_broker._lifecycle,
                        effects,
                    )
                )
            return original_publish(broker, effects)

        monkeypatch.setattr(
            evaluation_keys_mod._AutomaticKeyBroker,
            "_publish_effects_to_evaluation_point",
            observe_publish,
        )

        def claim_inside_operation():
            with evaluation_keys_mod._operation_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            run_at_points(evaluation_points, claim_inside_operation)

        assert len(operation_brokers) == 1
        assert len(lifecycle_pairs) == 1
        child_status, parent_status, effects = lifecycle_pairs[0]
        assert child_status is evaluation_keys_mod._BrokerLifecycle.FINALIZING
        assert parent_status is evaluation_keys_mod._BrokerLifecycle.OPEN
        assert len(effects) == 1
        assert operation_brokers[0]._lifecycle is evaluation_keys_mod._BrokerLifecycle.SEALED
        with pytest.raises(RuntimeError, match="finalizing"):
            operation_brokers[0]._publish_effects_to_evaluation_point(effects)

    def test_failed_operation_scope_seals_without_publishing_effects(self):
        operation_brokers = []

        def fail_after_claim():
            with evaluation_keys_mod._operation_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()
                raise RuntimeError("operation failed")

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="operation failed"):
                run_at_points(evaluation_points, fail_after_claim)
            assert parent._effects_by_identity == {}

        assert len(operation_brokers) == 1
        assert operation_brokers[0]._lifecycle is evaluation_keys_mod._BrokerLifecycle.SEALED

    def test_operation_broker_in_an_evaluation_point_closes_before_publishing_effects(self):
        operation_brokers = []

        def leave_nested_item_unjoined():
            with evaluation_keys_mod._operation_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)
                _claim_automatic_words()
                nested_points = issue_points()
                operation_broker.register_evaluation_points(nested_points)

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            with pytest.raises(RuntimeError, match="before all evaluation points join"):
                run_at_points(evaluation_points, leave_nested_item_unjoined)

            assert parent._effects_by_identity == {}

        assert len(operation_brokers) == 1

    def test_successful_operation_broker_in_an_evaluation_point_is_sealed(self):
        operation_brokers = []

        def capture_operation_broker():
            with evaluation_keys_mod._operation_stochastic_scope() as operation_broker:
                operation_brokers.append(operation_broker)

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            run_at_points(evaluation_points, capture_operation_broker)

        assert len(operation_brokers) == 1
        assert operation_brokers[0]._lifecycle is evaluation_keys_mod._BrokerLifecycle.SEALED

    def test_successful_effects_require_the_active_claiming_attempt(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        claimed = make_effect_claim()
        unclaimed = replace(
            claimed,
            stochastic_source_id=("source-group", 1),
        )

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            parent.begin_evaluation_point_attempt(attempt, evaluation_point.frame)
            parent.claim_evaluation_point_effect(
                claimed,
                frame=evaluation_point.frame,
                attempt=attempt,
            )
            stale_attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )

            with pytest.raises(RuntimeError, match="active attempt"):
                parent.accept_successful_evaluation_point_effects(
                    (claimed,),
                    frame=evaluation_point.frame,
                    attempt=stale_attempt,
                )
            assert parent._effects_by_identity == {}

            with pytest.raises(RuntimeError, match="was not claimed"):
                parent.accept_successful_evaluation_point_effects(
                    (unclaimed,),
                    frame=evaluation_point.frame,
                    attempt=attempt,
                )
            assert parent._effects_by_identity == {}

            parent.accept_successful_evaluation_point_effects(
                (claimed,),
                frame=evaluation_point.frame,
                attempt=attempt,
            )
            parent.finish_evaluation_point_attempt(attempt)

        assert tuple(parent._effects_by_identity.values()) == (claimed,)

    def test_sealed_parent_rejects_successful_effect_injection(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        effect = make_effect_claim()
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)

        with workflow_run(seed=17):
            with evaluation_keys_mod._function_stochastic_scope() as parent:
                parent.register_evaluation_points(evaluation_points)
                parent.begin_evaluation_point_attempt(attempt, evaluation_point.frame)
                parent.claim_evaluation_point_effect(
                    effect,
                    frame=evaluation_point.frame,
                    attempt=attempt,
                )
                parent.finish_evaluation_point_attempt(attempt)

            with pytest.raises(RuntimeError, match="sealed"):
                parent.accept_successful_evaluation_point_effects(
                    (effect,),
                    frame=evaluation_point.frame,
                    attempt=attempt,
                )

        assert parent._effects_by_identity == {}

    def test_same_evaluation_point_token_retries_the_same_child_claim(self):
        evaluation_points = issue_points()

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            first = run_at_points(evaluation_points, _claim_automatic_words)[0]
            second = run_at_points(evaluation_points, _claim_automatic_words)[0]

            state = parent._evaluation_point_claims.by_point_segment[
                evaluation_points[0].frame.point_segment
            ]
            occurrence_path = state.child_invocations[0].occurrence_path

        assert first == second
        assert occurrence_path == (
            ("invocation", 0),
            evaluation_points[0].frame.point_segment,
            ("child", 0),
        )

    def test_different_token_cannot_reuse_an_evaluation_point(self):
        first_evaluation_points = issue_points()
        second_evaluation_points = issue_points()

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            run_at_points(first_evaluation_points, lambda: None)
            with pytest.raises(RuntimeError, match="different token"):
                run_at_points(second_evaluation_points, lambda: None)

    def test_same_attempt_cannot_enter_twice(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            root_frame = context_mod._capture_active_workflow_frame()
            assert root_frame is not None
            with (
                context_mod._evaluation_point_frame_scope(
                    root_frame,
                    evaluation_point.frame.point_segment,
                ),
                evaluation_keys_mod._evaluation_point_stochastic_scope(
                    parent,
                    evaluation_point.frame,
                    attempt=attempt,
                ),
                pytest.raises(RuntimeError, match="active attempt"),
                evaluation_keys_mod._evaluation_point_stochastic_scope(
                    parent,
                    evaluation_point.frame,
                    attempt=attempt,
                ),
            ):
                pass

    @pytest.mark.parametrize("executor_name", ["sequential", "thread", "prefect"])
    def test_same_attempt_cannot_claim_one_effect_twice(self, executor_name, prefect_fakes):
        def duplicate_effect_claim(x):
            del x
            with evaluation_keys_mod._operation_stochastic_scope():
                _claim_automatic_words()
                _claim_automatic_words()

        with (
            workflow_run(seed=17),
            requester_call(requester()),
            pytest.raises(EvaluationError, match="duplicated a stochastic effect claim") as excinfo,
        ):
            run_on(executor_name, points_of(duplicate_effect_claim))

        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_deterministic_attempt_does_not_consume_child_ordinal(self):
        state = {"random": False}

        def maybe_claim():
            return _claim_automatic_words() if state["random"] else None

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            assert run_at_points(evaluation_points, maybe_claim) == [None]
            state["random"] = True
            after_deterministic = run_at_points(evaluation_points, maybe_claim)[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            first_child = run_at_points(issue_points(), _claim_automatic_words)[0]

        assert after_deterministic == first_child

    def test_failed_attempt_reuses_existing_child_claim(self):
        seen = []

        def fail_once_after_claim():
            words = _claim_automatic_words()
            seen.append(words)
            if len(seen) == 1:
                raise RuntimeError("retry me")
            return words

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                run_at_points(evaluation_points, fail_once_after_claim)
            retried = run_at_points(evaluation_points, fail_once_after_claim)[0]

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

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                run_at_points(evaluation_points, fail_once_after_nested_claim)
            retried = run_at_points(evaluation_points, fail_once_after_nested_claim)[0]

        assert seen == [retried, retried]

    def test_failed_attempt_only_effect_is_not_persisted_by_later_success(self):
        state = {"claim": True, "fail": True}

        def conditional_attempt():
            if state["claim"]:
                _claim_automatic_words()
            if state["fail"]:
                raise RuntimeError("retry me")
            return 1

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            with pytest.raises(RuntimeError, match="retry me"):
                run_at_points(evaluation_points, conditional_attempt)
            state.update(claim=False, fail=False)
            assert run_at_points(evaluation_points, conditional_attempt) == [1]
            snapshot = evaluation_keys_mod._snapshot_active_recipe_state()

        assert snapshot is None

    def test_caught_failed_nested_function_effect_is_not_persisted(self):
        def fail_after_claim():
            _claim_automatic_words()
            raise RuntimeError("nested failure")

        nested = Function(
            fail_after_claim,
            label="fail_after_claim",
        )

        def catch_nested_failure():
            with pytest.raises(RuntimeError, match="nested failure"):
                nested()
            return 1

        evaluation_points = issue_points()
        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope():
            assert run_at_points(evaluation_points, catch_nested_failure) == [1]
            snapshot = evaluation_keys_mod._snapshot_active_recipe_state()

        assert snapshot is None

    def test_parent_cancels_unstarted_items_before_releasing(self):
        evaluation_points = issue_points(2)
        started, unstarted = evaluation_points

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(started.frame.token)
            parent.begin_evaluation_point_attempt(attempt, started.frame)
            parent.finish_evaluation_point_attempt(attempt)
            with pytest.raises(RuntimeError, match="join"):
                parent.assert_evaluation_points_joined(evaluation_points)

            parent.cancel_unstarted_evaluation_points(evaluation_points)
            parent.assert_evaluation_points_joined(evaluation_points)
            states = parent._evaluation_point_claims.by_token

        assert (
            states[started.frame.token].status is evaluation_keys_mod._EvaluationPointStatus.JOINED
        )
        assert (
            states[unstarted.frame.token].status
            is evaluation_keys_mod._EvaluationPointStatus.CANCELLED
        )

    def test_cancelled_item_cannot_start_later(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)

        with workflow_run(seed=17):
            parent = evaluation_keys_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_evaluation_points(evaluation_points)
            parent.cancel_unstarted_evaluation_points(evaluation_points)
            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.CANCELLED

            with pytest.raises(RuntimeError, match="cancelled"):
                parent.begin_evaluation_point_attempt(attempt, evaluation_point.frame)

    def test_evaluation_point_uses_explicit_lifecycle_status(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17):
            parent = evaluation_keys_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_evaluation_points(evaluation_points)
            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.ISSUED

            first_attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            parent.begin_evaluation_point_attempt(first_attempt, evaluation_point.frame)
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.ACTIVE
            assert state.active_attempt == first_attempt.attempt_token

            parent.finish_evaluation_point_attempt(first_attempt)
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.JOINED
            assert state.active_attempt is None
            assert state.seen_attempts == {first_attempt.attempt_token}

            retry_attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            parent.begin_evaluation_point_attempt(retry_attempt, evaluation_point.frame)
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.ACTIVE
            assert state.active_attempt == retry_attempt.attempt_token
            assert state.seen_attempts == {
                first_attempt.attempt_token,
                retry_attempt.attempt_token,
            }
            parent.finish_evaluation_point_attempt(retry_attempt)
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.JOINED

    def test_cancel_and_begin_race_has_one_terminal_outcome(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)
        barrier = Barrier(2)

        with workflow_run(seed=17):
            parent = evaluation_keys_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_evaluation_points(evaluation_points)

            def cancel():
                barrier.wait()
                parent.cancel_unstarted_evaluation_points(evaluation_points)

            def begin():
                barrier.wait()
                try:
                    parent.begin_evaluation_point_attempt(attempt, evaluation_point.frame)
                except RuntimeError as error:
                    assert "cancelled" in str(error)
                    return evaluation_keys_mod._EvaluationPointStatus.CANCELLED
                parent.finish_evaluation_point_attempt(attempt)
                return evaluation_keys_mod._EvaluationPointStatus.JOINED

            with ThreadPoolExecutor(max_workers=2) as pool:
                cancel_future = pool.submit(cancel)
                begin_future = pool.submit(begin)
                cancel_future.result()
                outcome = begin_future.result()

            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]

        assert state.status is outcome

    def test_function_scope_seals_its_broker(self):
        evaluation_points = issue_points()

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            run_at_points(evaluation_points, lambda: None)

        with pytest.raises(RuntimeError, match="sealed"):
            parent.register_evaluation_points(evaluation_points)
        with pytest.raises(RuntimeError, match="sealed"):
            parent.begin_evaluation_point_attempt(
                evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_points[0].frame.token),
                evaluation_points[0].frame,
            )

    def test_frame_mismatch_fails_before_attempt_reservation(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)
        wrong_frame = evaluation_keys_mod.EvaluationPointFrame(
            point_segment=evaluation_keys_mod.sweep_cell_point_segment((0,)),
            token=evaluation_point.frame.token,
        )

        with workflow_run(seed=17):
            parent = evaluation_keys_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_evaluation_points(evaluation_points)

            with (
                pytest.raises(RuntimeError, match="frame does not match"),
                evaluation_keys_mod._evaluation_point_stochastic_scope(
                    parent,
                    wrong_frame,
                    attempt=attempt,
                ),
            ):
                pass

            with (
                context_mod._evaluation_point_frame_scope(
                    context_mod._capture_active_workflow_frame(),
                    evaluation_point.frame.point_segment,
                ),
                evaluation_keys_mod._evaluation_point_stochastic_scope(
                    parent,
                    evaluation_point.frame,
                    attempt=attempt,
                ),
            ):
                pass

    def test_missing_evaluation_point_frame_does_not_reserve_attempt(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        parent = evaluation_keys_mod._AutomaticKeyBroker("invocation")
        parent.register_evaluation_points(evaluation_points)
        with (
            pytest.raises(RuntimeError, match="active workflow frame"),
            evaluation_keys_mod._evaluation_point_stochastic_scope(parent, evaluation_point.frame),
        ):
            pass

        parent.cancel_unstarted_evaluation_points(evaluation_points)
        parent.assert_evaluation_points_joined(evaluation_points)

    def test_context_installation_failure_releases_attempt(self, monkeypatch):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        class BrokenContextVar:
            def set(self, value):
                del value
                raise RuntimeError("context installation failed")

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            root_frame = context_mod._capture_active_workflow_frame()
            assert root_frame is not None
            with context_mod._evaluation_point_frame_scope(
                root_frame,
                evaluation_point.frame.point_segment,
            ):
                monkeypatch.setattr(
                    evaluation_keys_mod,
                    "_ACTIVE_EVALUATION_POINT_ATTEMPT",
                    BrokenContextVar(),
                )
                with (
                    pytest.raises(RuntimeError, match="context installation failed"),
                    evaluation_keys_mod._evaluation_point_stochastic_scope(
                        parent, evaluation_point.frame
                    ),
                ):
                    pass

            parent.assert_evaluation_points_joined(evaluation_points)

    def test_remote_attempt_is_reserved_before_submission(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        first = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)
        second = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)

        with workflow_run(seed=17):
            parent = evaluation_keys_mod._AutomaticKeyBroker(
                "invocation",
                _frame=context_mod._capture_active_workflow_frame(),
            )
            parent.register_evaluation_points(evaluation_points)

            first_envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame, first
            )
            with pytest.raises(RuntimeError, match="active attempt"):
                parent.reserve_remote_evaluation_point_attempt(evaluation_point.frame, second)

            parent.abort_remote_evaluation_point_attempt(first)
            second_envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame, second
            )
            parent.abort_remote_evaluation_point_attempt(second)
            parent.assert_evaluation_points_joined(evaluation_points)

        assert (first_envelope.attempt, second_envelope.attempt) == (first, second)
        assert first_envelope.parent_occurrence_path == second_envelope.parent_occurrence_path
        assert first_envelope.root_words == second_envelope.root_words

    def test_a_shipped_chunk_carries_its_parents_authority(self):
        with workflow_run(seed=17), requester_call(requester()):
            expected = run_on("sequential", points_of(_claim_words_at_point, 2))

        with workflow_run(seed=17), requester_call(requester()):
            evaluation_points = points_of(_claim_words_at_point, 2)
            (chunk,) = evaluation_points.chunks()
            shipped_chunk = cloudpickle.loads(cloudpickle.dumps(chunk))
            reserved_points = shipped_chunk._point_scopes._reserved_points
            envelopes = [parent_envelope for _, _, parent_envelope in reserved_points.values()]
            # A fresh thread starts with no context, as a worker process does.
            with ThreadPoolExecutor(max_workers=1) as pool:
                outcomes = pool.submit(
                    shipped_chunk.run, evaluation_points.shared_payload()
                ).result()
            evaluation_points._point_scopes.accept_claim_reports(outcomes._claim_reports)
            evaluation_points._point_scopes.close()
            parent = evaluation_keys_mod._capture_active_broker()

        assert all(
            isinstance(parent_envelope, evaluation_keys_mod.EvaluationPointParentEnvelope)
            for parent_envelope in envelopes
        )
        assert [parent_envelope.frame for parent_envelope in envelopes] == [
            issued_point(evaluation_points, point_index).frame for point_index in range(2)
        ]
        assert outcomes.failures == {}
        assert outcomes.outputs == expected
        assert len(parent._effects_by_identity) == 2


class TestRemoteReportTransactions:
    def test_worker_rejects_a_transported_frame_root_mismatch(self):
        frame = make_evaluation_point_frame()
        attempt = make_evaluation_point_attempt()
        envelope = evaluation_keys_mod.EvaluationPointParentEnvelope(
            root_words=(0, 17),
            parent_occurrence_path=(("invocation", 0),),
            frame=frame,
            attempt=attempt,
        )

        with (
            context_mod._transported_workflow_frame((0, 99)),
            pytest.raises(RuntimeError, match="transport envelope"),
            evaluation_keys_mod._remote_evaluation_point_stochastic_scope(envelope, attempt),
        ):
            pass

    def test_sealed_parent_rejects_remote_report(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]
        attempt = evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token)
        report = evaluation_keys_mod.EvaluationPointClaimReport(
            frame=evaluation_point.frame,
            attempt=attempt,
            child_count=0,
        )

        with workflow_run(seed=17):
            with evaluation_keys_mod._function_stochastic_scope() as parent:
                parent.register_evaluation_points(evaluation_points)
                parent.reserve_remote_evaluation_point_attempt(
                    evaluation_point.frame,
                    attempt,
                )
                parent.abort_remote_evaluation_point_attempt(attempt)

            with pytest.raises(RuntimeError, match="sealed"):
                parent.accept_remote_claim_report(report)

        state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
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
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame,
                attempt,
            )
            assert envelope is not None
            report_attempt = attempt
            report_frame = evaluation_point.frame
            child_segment = ("child", 0)
            parent_path = None
            point_segment = None
            if violation == "attempt":
                report_attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                    evaluation_point.frame.token
                )
            elif violation == "frame":
                report_frame = evaluation_keys_mod.EvaluationPointFrame(
                    point_segment=evaluation_keys_mod.sweep_cell_point_segment((9,)),
                    token=evaluation_point.frame.token,
                )
            elif violation == "prefix":
                parent_path = (("invocation", 9),)
            elif violation == "unit":
                point_segment = evaluation_keys_mod.sweep_cell_point_segment((9,))
            elif violation == "child-segment":
                child_segment = ("scope", 0)
            elif violation == "child-range":
                child_segment = ("child", 1)

            effect = make_remote_effect(
                envelope,
                child_segment=child_segment,
                parent_path=parent_path,
                point_segment=point_segment,
            )
            report = evaluation_keys_mod.EvaluationPointClaimReport(
                frame=report_frame,
                attempt=report_attempt,
                child_count=1,
                effects=(effect,),
                successful_effects=(effect,),
            )
            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            before_seen = set(state.seen_attempts)

            with pytest.raises(RuntimeError, match=r"remote|occurrence|child"):
                parent.accept_remote_claim_report(report)

            assert state.effect_claims_by_identity == {}
            assert state.child_invocations == []
            assert parent._effects_by_identity == {}
            assert state.seen_attempts == before_seen
            assert state.active_attempt == attempt.attempt_token
            parent.abort_remote_evaluation_point_attempt(attempt)

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
                "outside its evaluation point namespace",
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
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame,
                attempt,
            )
            assert envelope is not None
            first = make_remote_effect(envelope, source_index=0)
            second = make_remote_effect(envelope, source_index=1)
            report = evaluation_keys_mod.EvaluationPointClaimReport(
                frame=evaluation_point.frame,
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
                    (("invocation", 99), evaluation_point.frame.point_segment, ("child", 0)),
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
                    evaluation_keys_mod.EvaluationPointFrame(
                        point_segment=evaluation_keys_mod.sweep_cell_point_segment((9,)),
                        token=evaluation_point.frame.token,
                    ),
                )
            elif violation == "attempt":
                object.__setattr__(
                    report,
                    "attempt",
                    evaluation_keys_mod.EvaluationPointAttempt.create(evaluation_point.frame.token),
                )
            elif violation == "child-count":
                object.__setattr__(report, "child_count", -1)
            elif violation == "effect-field":
                object.__setattr__(second, "sample_shape", (-1,))
            else:
                object.__setattr__(second, "stochastic_source_id", ["invalid"])

            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            before_evaluation_point = (
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
            ) == before_evaluation_point
            assert parent._effects_by_identity == before_recipe
            parent.abort_remote_evaluation_point_attempt(attempt)

    def test_report_construction_revalidates_existing_effect_instances(self):
        frame = make_evaluation_point_frame()
        attempt = make_evaluation_point_attempt()
        effect = make_effect_claim()
        object.__setattr__(effect, "sample_shape", (-1,))

        with pytest.raises(ValueError, match="sample shape dimensions must be non-negative"):
            evaluation_keys_mod.EvaluationPointClaimReport(
                frame=frame,
                attempt=attempt,
                child_count=0,
                effects=(effect,),
            )

    def test_pickled_remote_report_is_revalidated_and_committed(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame,
                attempt,
            )
            assert envelope is not None
            effect = make_remote_effect(envelope)
            report = evaluation_keys_mod.EvaluationPointClaimReport(
                frame=evaluation_point.frame,
                attempt=attempt,
                child_count=1,
                effects=(effect,),
                successful_effects=(effect,),
            )

            transported = pickle.loads(pickle.dumps(report))
            parent.accept_remote_claim_report(transported)

            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.JOINED
            assert tuple(state.effect_claims_by_identity.values()) == (effect,)
            assert tuple(parent._effects_by_identity.values()) == (effect,)

    def test_valid_remote_report_commits_all_ledgers_together(self):
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame,
                attempt,
            )
            assert envelope is not None
            effects = (
                make_remote_effect(envelope, source_index=0),
                make_remote_effect(envelope, source_index=1),
            )
            report = evaluation_keys_mod.EvaluationPointClaimReport(
                frame=evaluation_point.frame,
                attempt=attempt,
                child_count=1,
                effects=effects,
                successful_effects=effects,
            )

            parent.accept_remote_claim_report(report)

            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            assert state.status is evaluation_keys_mod._EvaluationPointStatus.JOINED
            assert state.active_attempt is None
            assert len(state.effect_claims_by_identity) == 2
            assert len(state.child_invocations) == 1
            assert tuple(parent._effects_by_identity.values()) == effects

    def test_second_replay_effect_failure_rolls_back_every_ledger(self):
        with workflow_run(seed=17):
            recorded = sample(Normal("value", loc=0.0, scale=1.0))
        replay_state = replay_mod._validate_provenance(recorded.provenance)
        expected = replay_state.expected_events[0].effect_claim()
        unexpected = replace(
            expected,
            stochastic_source_id=("source-group", 99),
        )
        evaluation_points = issue_points()
        evaluation_point = evaluation_points[0]

        with workflow_run(seed=17), evaluation_keys_mod._function_stochastic_scope() as parent:
            parent._replay_state = replay_state
            parent.register_evaluation_points(evaluation_points)
            attempt = evaluation_keys_mod.EvaluationPointAttempt.create(
                evaluation_point.frame.token
            )
            envelope = parent.reserve_remote_evaluation_point_attempt(
                evaluation_point.frame,
                attempt,
            )
            assert envelope is not None
            report = evaluation_keys_mod.EvaluationPointClaimReport(
                frame=evaluation_point.frame,
                attempt=attempt,
                child_count=1,
                effects=(expected, unexpected),
                successful_effects=(expected, unexpected),
            )

            with pytest.raises(
                ReplayCompatibilityError, match="that the recorded call did not make"
            ):
                parent.accept_remote_claim_report(report)

            state = parent._evaluation_point_claims.by_token[evaluation_point.frame.token]
            assert state.effect_claims_by_identity == {}
            assert state.child_invocations == []
            assert parent._effects_by_identity == {}
            assert all(
                claim.evaluation_point_token is None and not claim.attempt_tokens
                for claim in replay_state.claims.values()
            )
            parent.abort_remote_evaluation_point_attempt(attempt)


class TestThreadExecution:
    def test_evaluation_points_that_finish_out_of_order_keep_their_keys_and_order(self):
        num_draws = 6

        def lifted(body, **controls):
            return Function(
                body,
                traceable=False,
                max_evaluations=num_draws,
                label="noisy_draw",
                output_spec=OutputSpec(noisy_draw=None),
            ).using(**controls)

        with workflow_run(seed=17):
            expected = atom_leaves(lifted(_noisy_draw, executor="sequential")(standard_normal()))[0]
        first_draw = float(expected[0, 0])
        others_finished = Event()
        finish_order = []
        finish_lock = threading.Lock()

        def noisy_draw_first_point_last(z):
            if float(z) == first_draw:
                assert others_finished.wait(timeout=5)
            output = _noisy_draw(z)
            with finish_lock:
                finish_order.append(float(z))
                if len(finish_order) == num_draws - 1:
                    others_finished.set()
            return output

        threaded = lifted(
            noisy_draw_first_point_last,
            executor="thread",
            executor_options={"max_workers": num_draws, "chunk_size": 1},
        )
        with workflow_run(seed=17):
            result = threaded(standard_normal())

        assert finish_order[-1] == first_draw
        assert sorted(finish_order[:-1]) == sorted(float(draw) for draw in expected[1:, 0])
        assert result.provenance.metadata["dispatch"] == "thread"
        np.testing.assert_array_equal(atom_leaves(result)[0], expected)


class TestEvaluationsInsideAMappedBody:
    @pytest.mark.parametrize(
        "inner_controls",
        [
            pytest.param({"executor": "thread"}, id="thread-executor"),
            pytest.param({"workflow_kind": WorkflowKind.TASK}, id="prefect-task"),
        ],
    )
    def test_a_plain_call_in_a_mapped_body_runs_in_the_calling_thread(
        self, inner_controls, prefect_fakes
    ):
        threads = []

        def add_one_noting_the_thread(x):
            threads.append(threading.get_ident())
            return x + 1

        def mapped_calling(inner):
            return Function(
                lambda z: jnp.asarray(inner(x=z)),
                traceable=True,
                max_evaluations=8,
                label="outer",
                output_spec=OutputSpec(outer=None),
            )

        plain = Function(add_one_noting_the_thread, label="inner")
        named = Function(add_one_noting_the_thread, label="inner", **inner_controls)
        with workflow_run(seed=5):
            expected = mapped_calling(plain)(standard_normal())
        with workflow_run(seed=5):
            result = mapped_calling(named)(standard_normal())

        assert result.provenance.metadata["dispatch"] == "jax"
        np.testing.assert_allclose(atom_leaves(result)[0], atom_leaves(expected)[0], rtol=1e-6)
        assert set(threads) == {threading.get_ident()}
        assert FakeMappedTask.created_names == []
        assert RecordingFlow.calls == []


class TestPrefectTransport:
    def test_prefect_transport_raises_clear_error_when_prefect_missing(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", None)
        monkeypatch.setattr(execution_mod, "flow", None)

        with (
            workflow_run(seed=17),
            requester_call(requester()),
            pytest.raises(RuntimeError, match=r"Prefect task.*execution was requested"),
        ):
            run_on("prefect", points_of(add_one, 2))

    @pytest.mark.parametrize(
        ("workflow_kind", "flow_name", "task_name"),
        [
            pytest.param(WorkflowKind.TASK, "plus_one_map", "plus_one", id="task"),
            pytest.param(WorkflowKind.FLOW, "plus_one", "plus_one_run", id="flow"),
        ],
    )
    def test_prefect_runs_each_evaluation_point_as_a_named_task_in_one_flow(
        self, workflow_kind, flow_name, task_name, prefect_fakes
    ):
        runner = object()
        config.prefect.task_runner = runner
        plus_one = Function(add_one, label="plus_one", workflow_kind=workflow_kind)

        assert float(plus_one(x=1)) == 2.0
        assert RecordingFlow.calls == [{"name": flow_name, "kwargs": {"task_runner": runner}}]
        assert FakeMappedTask.created_names == [task_name]
        assert FakeMappedTask.map_calls == 1

    @pytest.mark.parametrize(
        ("traceable", "created_names", "executor_name"),
        [
            pytest.param(None, ["affine_vmap"], "jax", id="traceable"),
            pytest.param(False, ["affine"], "prefect", id="one-task-per-point"),
        ],
    )
    def test_a_lifted_call_under_prefect_records_what_ran_its_evaluations(
        self, traceable, created_names, executor_name, prefect_fakes
    ):
        affine = Function(
            lambda z: 2.0 * z + 1.0,
            traceable=traceable,
            max_evaluations=6,
            label="affine",
            output_spec=OutputSpec(affine=None),
            workflow_kind=WorkflowKind.TASK,
        )

        with workflow_run(seed=5):
            result = affine(standard_normal())

        assert result.num_atoms == 6
        # The route is built once before any draw, then again to run.
        assert FakeMappedTask.created_names == created_names * 2
        assert result.provenance.metadata["dispatch"] == executor_name

    def test_a_call_naming_an_executor_under_prefect_warns_once(self, prefect_fakes):
        affine = Function(
            lambda z: 2.0 * z + 1.0,
            traceable=False,
            max_evaluations=6,
            label="affine",
            output_spec=OutputSpec(affine=None),
            workflow_kind=WorkflowKind.TASK,
        ).using(executor="thread")

        with warnings.catch_warnings(record=True) as caught, workflow_run(seed=5):
            warnings.simplefilter("always")
            result = affine(standard_normal())

        executor_warnings = [
            str(warning.message) for warning in caught if "has no effect" in str(warning.message)
        ]
        assert executor_warnings == [
            "executor 'thread' on 'affine' has no effect: workflow_kind=WorkflowKind.TASK runs "
            "its evaluations as Prefect tasks. Remove the executor, or set "
            "workflow_kind=WorkflowKind.OFF to run them on it"
        ]
        assert result.provenance.metadata["dispatch"] == "prefect"

    def test_a_call_naming_only_executor_options_under_prefect_warns_to_remove_them(
        self, prefect_fakes
    ):
        affine = Function(
            lambda z: 2.0 * z + 1.0,
            traceable=False,
            max_evaluations=6,
            label="affine",
            output_spec=OutputSpec(affine=None),
            workflow_kind=WorkflowKind.TASK,
        ).using(executor_options={"chunk_size": 2})

        with warnings.catch_warnings(record=True) as caught, workflow_run(seed=5):
            warnings.simplefilter("always")
            affine(standard_normal())

        executor_warnings = [
            str(warning.message) for warning in caught if "has no effect" in str(warning.message)
        ]
        assert executor_warnings == [
            "executor_options on 'affine' has no effect: workflow_kind=WorkflowKind.TASK runs "
            "its evaluations as Prefect tasks. Remove executor_options, or set "
            "workflow_kind=WorkflowKind.OFF to run them on an executor"
        ]

    @pytest.mark.parametrize("ship_chunks", [False, True], ids=["in-place", "shipped"])
    def test_prefect_outcomes_keep_point_order_when_tasks_run_in_reverse(
        self, ship_chunks, prefect_fakes
    ):
        with workflow_run(seed=17), requester_call(requester()):
            expected = run_on("sequential", points_of(_claim_words_at_point, 3))
        FakeMappedTask.run_in_reverse = True
        FakeMappedTask.ship_chunks = ship_chunks

        with workflow_run(seed=17), requester_call(requester()):
            assert run_on("prefect", points_of(_claim_words_at_point, 3)) == expected

    @pytest.mark.parametrize("mode_source", ["block", "session snapshot"])
    def test_a_prefect_worker_takes_the_provenance_mode_of_the_carried_context(
        self, mode_source, prefect_fakes, monkeypatch
    ):
        FakeMappedTask.ship_chunks = True
        monkeypatch.setattr(config.provenance, "mode", ProvenanceMode.FULL)

        def provenance_mode(x):
            del x
            return context_mod._active_provenance_mode()

        if mode_source == "block":
            with (
                config.provenance.using(mode=ProvenanceMode.OFF),
                workflow_run(seed=17),
                requester_call(requester()),
            ):
                outputs = run_on("prefect", points_of(provenance_mode))
        else:
            config.provenance.mode = ProvenanceMode.OFF
            with workflow_run(seed=17), requester_call(requester()):
                evaluation_points = points_of(provenance_mode)
                # A thread with no open workflow run may change the session's mode, as a
                # worker process starts under a session of its own.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(setattr, config.provenance, "mode", ProvenanceMode.FULL).result()
                outputs = run_on("prefect", evaluation_points)

        assert config.provenance.mode is ProvenanceMode.FULL
        assert outputs == (ProvenanceMode.OFF,)

    def test_prefect_task_that_does_not_finish_releases_its_reservation(self, monkeypatch):
        class CrashingFuture:
            def result(self):
                raise RuntimeError("worker crashed")

        class CrashingTask:
            def map(self, *, chunk):
                # Sending a chunk to a worker pickles it, which reserves its evaluation points.
                for shipped_chunk in chunk:
                    cloudpickle.dumps(shipped_chunk)
                return [CrashingFuture() for _ in chunk]

        def crashing_task(name=None, **task_kwargs):
            del name, task_kwargs

            def decorator(func):
                del func
                return CrashingTask()

            return decorator

        monkeypatch.setattr(execution_mod, "task", crashing_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (0, ()))

        with workflow_run(seed=17), requester_call(requester()):
            evaluation_points = points_of(_claim_words_at_point)
            with pytest.raises(
                EvaluationError, match=r"its Prefect task did not finish.*worker crashed"
            ) as excinfo:
                run_on("prefect", evaluation_points)
            parent = evaluation_keys_mod._capture_active_broker()
            state = parent._evaluation_point_claims.by_token[
                issued_point(evaluation_points).frame.token
            ]

        assert excinfo.value.failures[0].kind == "lost"
        assert state.status is evaluation_keys_mod._EvaluationPointStatus.JOINED
        assert state.active_attempt is None
        assert len(state.seen_attempts) == 1

    def test_deterministic_prefect_items_do_not_materialize_anonymous_root(self, prefect_fakes):
        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            requester_call(requester()),
        ):
            assert run_on("prefect", points_of(add_one, 2)) == (2, 3)

        urandom.assert_not_called()
        assert FakeMappedTask.map_calls == 1

    def test_failed_prefect_effect_remains_transient(self, prefect_fakes):
        def fail_after_claim(x):
            del x
            _claim_automatic_words()
            raise RuntimeError("remote failure")

        with workflow_run(seed=17), requester_call(requester()):
            evaluation_points = points_of(fail_after_claim)
            with pytest.raises(EvaluationError, match="remote failure") as raised:
                run_on("prefect", evaluation_points)
            snapshot = evaluation_keys_mod._snapshot_active_recipe_state()
            claim_state = (
                evaluation_keys_mod._capture_active_broker()._evaluation_point_claims.by_token[
                    issued_point(evaluation_points).frame.token
                ]
            )

        assert isinstance(raised.value.__cause__, RuntimeError)
        assert snapshot is None
        assert len(claim_state.effect_claims_by_identity) == 1

    @pytest.mark.parametrize("ship_chunks", [False, True], ids=["in-place", "shipped"])
    def test_prefect_retry_reuses_evaluation_point_token_and_key(
        self, ship_chunks, prefect_fakes, monkeypatch
    ):
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (1, ()))
        FakeMappedTask.ship_chunks = ship_chunks
        seen = []

        def fail_once_after_claim(x):
            del x
            words = _claim_automatic_words()
            seen.append(words)
            if len(seen) == 1:
                raise RuntimeError("retry me")
            return words

        with workflow_run(seed=17), requester_call(requester()):
            evaluation_points = points_of(fail_once_after_claim)
            (retried,) = run_on("prefect", evaluation_points)
            snapshot = evaluation_keys_mod._snapshot_active_recipe_state()
            claim_state = (
                evaluation_keys_mod._capture_active_broker()._evaluation_point_claims.by_token[
                    issued_point(evaluation_points).frame.token
                ]
            )

        assert seen == [retried, retried]
        assert snapshot is not None
        assert len(snapshot.effects) == 1
        assert len(claim_state.seen_attempts) == 2
        assert FakeMappedTask.map_calls == 2

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            pytest.param(None, (), id="none"),
            pytest.param([], (), id="empty"),
            pytest.param(0.25, (0.25,), id="scalar"),
            pytest.param([0.1, 0.2], (0.1, 0.2), id="sequence"),
        ],
    )
    def test_prefect_retry_policy_normalizes_delay_schedule(
        self,
        configured,
        expected,
        monkeypatch,
    ):
        settings = SimpleNamespace(
            tasks=SimpleNamespace(
                default_retries=3,
                default_retry_delay_seconds=configured,
            )
        )
        monkeypatch.setattr("prefect.settings.get_current_settings", lambda: settings)

        assert execution_mod._prefect_retry_policy() == (3, expected)

    @pytest.mark.parametrize(
        "configured",
        [
            pytest.param("0.1", id="string"),
            pytest.param([0.1, "0.2"], id="invalid-list-item"),
            pytest.param(True, id="boolean"),
        ],
    )
    def test_prefect_retry_policy_rejects_invalid_delay_type(
        self,
        configured,
        monkeypatch,
    ):
        settings = SimpleNamespace(
            tasks=SimpleNamespace(
                default_retries=3,
                default_retry_delay_seconds=configured,
            )
        )
        monkeypatch.setattr("prefect.settings.get_current_settings", lambda: settings)

        with pytest.raises(TypeError, match="default_retry_delay_seconds"):
            execution_mod._prefect_retry_policy()

    def test_prefect_retry_delay_list_follows_retry_ordinal(self, monkeypatch):
        monkeypatch.setattr(execution_mod, "task", fake_task)
        monkeypatch.setattr(execution_mod, "flow", fake_flow)
        settings = SimpleNamespace(
            tasks=SimpleNamespace(
                default_retries=3,
                default_retry_delay_seconds=[0.1, 0.2],
            )
        )
        monkeypatch.setattr("prefect.settings.get_current_settings", lambda: settings)
        seen = []

        def fail_three_times_after_claim(x):
            del x
            words = _claim_automatic_words()
            seen.append(words)
            if len(seen) < 4:
                raise RuntimeError("retry me")
            return words

        with (
            patch.object(execution_mod, "sleep") as sleep,
            workflow_run(seed=17),
            requester_call(requester()),
        ):
            evaluation_points = points_of(fail_three_times_after_claim)
            (result,) = run_on("prefect", evaluation_points)
            parent = evaluation_keys_mod._capture_active_broker()
            claim_state = parent._evaluation_point_claims.by_token[
                issued_point(evaluation_points).frame.token
            ]

        assert seen == [result] * 4
        assert len(parent._effects_by_identity) == 1
        assert len(claim_state.seen_attempts) == 4
        assert FakeMappedTask.map_calls == 4
        assert [call.args[0] for call in sleep.call_args_list] == [0.1, 0.2, 0.2]

    def test_deterministic_prefect_retry_does_not_materialize_root(
        self, prefect_fakes, monkeypatch
    ):
        monkeypatch.setattr(execution_mod, "_prefect_retry_policy", lambda: (1, ()))
        calls = 0

        def fail_once(x):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("retry me")
            return x

        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            requester_call(requester()),
        ):
            assert run_on("prefect", points_of(fail_once)) == (1,)
            snapshot = evaluation_keys_mod._snapshot_active_recipe_state()

        assert snapshot is None
        urandom.assert_not_called()
        assert FakeMappedTask.map_calls == 2

    def test_evaluation_point_keys_match_across_sequential_thread_and_prefect(self, prefect_fakes):
        def run(executor_name):
            with workflow_run(seed=17), requester_call(requester()):
                return run_on(executor_name, points_of(_claim_words_at_point, 3))

        expected = run("sequential")
        assert run("thread") == expected
        assert run("prefect") == expected
        FakeMappedTask.ship_chunks = True
        assert run("prefect") == expected

    def test_nested_seeded_scope_matches_across_local_and_prefect(self, prefect_fakes):
        def run(workflow_kind, outer_seed):
            workflow = Function(
                _claim_nested_seeded_scalar,
                label="_claim_nested_seeded_scalar",
                workflow_kind=workflow_kind,
            )
            with workflow_run(seed=outer_seed):
                return int(workflow())

        local = run(WorkflowKind.OFF, 1)
        assert run(WorkflowKind.OFF, 2) == local
        assert run(WorkflowKind.TASK, 1) == local
        assert run(WorkflowKind.TASK, 2) == local

    def test_replay_prefect_worker_uses_assigned_expected_event_namespace(self, prefect_fakes):
        FakeMappedTask.ship_chunks = True
        local = Function(
            _claim_automatic_scalar,
            label="_claim_automatic_scalar",
            workflow_kind=WorkflowKind.OFF,
        ).using(executor="sequential")
        remote = Function(
            _claim_automatic_scalar,
            label="_claim_automatic_scalar",
            workflow_kind=WorkflowKind.TASK,
        )
        with workflow_run(seed=17):
            original = local()

        with replay_run(original.provenance):
            replayed = remote()

        assert (
            replayed.provenance.controls["randomness"] == original.provenance.controls["randomness"]
        )
        assert replayed.provenance.diagnostics["replay"]["execution_drift"] is True
        assert FakeMappedTask.map_calls == 1

    def test_replay_prefect_worker_rejects_effect_drift_before_key_derivation(self, prefect_fakes):
        FakeMappedTask.ship_chunks = True
        local = Function(
            _claim_automatic_scalar,
            label="_claim_automatic_scalar",
            workflow_kind=WorkflowKind.OFF,
        )
        remote = Function(
            _claim_automatic_scalar,
            label="_claim_automatic_scalar",
            workflow_kind=WorkflowKind.TASK,
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
                "probpipe.functions._context.derive_event_key_words_from_encoded",
                side_effect=AssertionError("derived key before replay validation"),
            ),
            pytest.raises(ReplayCompatibilityError, match=r"on a worker .* matches no draw"),
            replay_run(changed),
        ):
            remote()

        assert FakeMappedTask.map_calls == 1
