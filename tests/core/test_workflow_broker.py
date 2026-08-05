"""Tests for the private workflow automatic-key broker."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import jax
import pytest

import probpipe.core._workflow_managed as managed_mod
from probpipe import (
    Function,
    Normal,
    UnmanagedConcurrentWorkflowEntryError,
    workflow_run,
)
from probpipe.core._workflow_broker import (
    StochasticEffectPlan,
    _function_stochastic_scope,
    _managed_stochastic_scope,
    _resolve_automatic_key,
)
from probpipe.core._workflow_plan import PlannedRandomEvent


def _plan(source: int = 0) -> StochasticEffectPlan:
    return StochasticEffectPlan(
        operation_kind="test",
        execution_mode="sampled",
        event=PlannedRandomEvent(("source-group", source), ("singleton",)),
        sample_shape=(8,),
        sampling_abi="test-sampling/v1",
        provider_abi="test-provider/v1",
    )


def _key_words(key) -> tuple[int, int]:
    return tuple(int(word) for word in jax.random.key_data(key))


class TestStochasticEffectPlan:
    def test_plan_is_frozen_and_tuple_only(self):
        plan = _plan()

        assert plan.sample_shape == (8,)
        assert isinstance(plan.sample_shape, tuple)
        assert isinstance(plan.event.stochastic_source_id, tuple)
        assert isinstance(plan.event.logical_unit_id, tuple)
        assert plan.record_path == ()
        assert plan.descendant_descriptor is None
        with pytest.raises(FrozenInstanceError):
            plan.operation_kind = "changed"

        with pytest.raises(TypeError, match="record paths"):
            StochasticEffectPlan(
                operation_kind="test",
                execution_mode="sampled",
                event=plan.event,
                sample_shape=(1,),
                sampling_abi="test-sampling/v1",
                provider_abi="test-provider/v1",
                record_path=["field"],
            )

    def test_invalid_event_identity_does_not_consume_an_occurrence(self):
        class MutableEvent:
            stochastic_source_id = ("source-group", 0)
            logical_unit_id = ("singleton",)

        event = MutableEvent()
        invalid_plan = StochasticEffectPlan(
            operation_kind="test",
            execution_mode="sampled",
            event=event,
            sample_shape=(8,),
            sampling_abi="test-sampling/v1",
            provider_abi="test-provider/v1",
        )
        event.stochastic_source_id = ("source-group", True)
        object.__setattr__(invalid_plan, "event", event)

        with workflow_run(seed=7):
            with (
                _function_stochastic_scope() as invalid_broker,
                pytest.raises(TypeError, match="boolean values"),
            ):
                _resolve_automatic_key(None, invalid_plan)
            with _function_stochastic_scope():
                after_invalid = _key_words(_resolve_automatic_key(None, _plan()))

        with workflow_run(seed=7), _function_stochastic_scope():
            baseline = _key_words(_resolve_automatic_key(None, _plan()))

        assert invalid_broker._invocation is None
        assert after_invalid == baseline

    def test_event_identity_is_snapshotted_before_occurrence_commit(self):
        class ChangingEvent:
            source_reads = 0
            logical_unit_id = ("singleton",)

            @property
            def stochastic_source_id(self):
                self.source_reads += 1
                if self.source_reads <= 2:
                    return ("source-group", 0)
                return ("source-group", True)

        event = ChangingEvent()
        plan = StochasticEffectPlan(
            operation_kind="test",
            execution_mode="sampled",
            event=event,
            sample_shape=(8,),
            sampling_abi="test-sampling/v1",
            provider_abi="test-provider/v1",
        )

        with workflow_run(seed=7), _function_stochastic_scope() as broker:
            key = _resolve_automatic_key(None, plan)

        assert _key_words(key)
        assert broker._invocation is not None
        assert event.source_reads == 1


class TestAutomaticKeyOwnership:
    def test_explicit_key_is_returned_unchanged_without_context_or_validation(self):
        explicit = jax.random.key(9)

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
        ):
            actual = _resolve_automatic_key(explicit, None)

        assert actual is explicit
        urandom.assert_not_called()
        commit.assert_not_called()

    def test_function_scope_commits_lazily_and_reuses_one_invocation(self):
        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=__import__(
                    "probpipe.core._workflow_context",
                    fromlist=["_commit_stochastic_invocation"],
                )._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
            _function_stochastic_scope(),
        ):
            first = _resolve_automatic_key(None, _plan(0))
            second = _resolve_automatic_key(None, _plan(1))

        assert _key_words(first) != _key_words(second)
        commit.assert_called_once_with("invocation")

    def test_managed_scope_uses_a_tagged_operation_occurrence(self):
        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=__import__(
                    "probpipe.core._workflow_context",
                    fromlist=["_commit_stochastic_invocation"],
                )._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
            _managed_stochastic_scope(),
        ):
            _resolve_automatic_key(None, _plan())

        commit.assert_called_once_with("operation")

    def test_new_operation_scope_closes_its_managed_registry(self):
        with workflow_run(seed=7), _managed_stochastic_scope() as operation_broker:
            assert not operation_broker._managed_claims.closed

        assert operation_broker._managed_claims.closed

    def test_new_operation_scope_closes_after_body_error(self):
        with (
            workflow_run(seed=7),
            pytest.raises(ValueError, match="body failed"),
            _managed_stochastic_scope() as operation_broker,
        ):
            raise ValueError("body failed")

        assert operation_broker._managed_claims.closed

    def test_reused_operation_scope_does_not_close_its_owner(self):
        with workflow_run(seed=7), _function_stochastic_scope() as function_broker:
            with _managed_stochastic_scope() as reused_broker:
                assert reused_broker is function_broker

            assert not function_broker._managed_claims.closed
            _resolve_automatic_key(None, _plan())

        assert function_broker._managed_claims.closed

    def test_operation_scope_rejects_unjoined_nested_items(self):
        items = managed_mod.make_managed_work_items(
            [{}],
            unit_segments=(managed_mod.point_unit_segment(),),
        )

        with (
            workflow_run(seed=7),
            pytest.raises(RuntimeError, match="before all managed work items join"),
            _managed_stochastic_scope() as operation_broker,
        ):
            operation_broker.register_managed_work_items(items)

        assert operation_broker._effects_by_identity == {}

    def test_bare_managed_scopes_receive_independent_ephemeral_roots(self):
        with patch(
            "probpipe.core._workflow_context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            with _managed_stochastic_scope():
                first = _resolve_automatic_key(None, _plan())
            with _managed_stochastic_scope():
                second = _resolve_automatic_key(None, _plan())

        assert _key_words(first) != _key_words(second)
        assert urandom.call_count == 2

    def test_materialized_broker_rejects_a_copied_async_task(self):
        async def run():
            with workflow_run(seed=7), _function_stochastic_scope() as active_broker:
                first = _key_words(_resolve_automatic_key(None, _plan(0)))

                async def child():
                    return _resolve_automatic_key(None, _plan(1))

                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    await asyncio.create_task(child())

                assert len(active_broker._effects_by_identity) == 1
                second = _key_words(_resolve_automatic_key(None, _plan(1)))
                return first, second

        actual = asyncio.run(run())

        with workflow_run(seed=7), _function_stochastic_scope():
            expected = (
                _key_words(_resolve_automatic_key(None, _plan(0))),
                _key_words(_resolve_automatic_key(None, _plan(1))),
            )

        assert actual == expected

    def test_materialized_broker_rejects_a_copied_thread(self):
        with workflow_run(seed=7), _function_stochastic_scope() as active_broker:
            first = _key_words(_resolve_automatic_key(None, _plan(0)))
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(copied.run, _resolve_automatic_key, None, _plan(1))
                with pytest.raises(UnmanagedConcurrentWorkflowEntryError):
                    future.result()

            assert len(active_broker._effects_by_identity) == 1
            second = _key_words(_resolve_automatic_key(None, _plan(1)))

        with workflow_run(seed=7), _function_stochastic_scope():
            expected = (
                _key_words(_resolve_automatic_key(None, _plan(0))),
                _key_words(_resolve_automatic_key(None, _plan(1))),
            )

        assert (first, second) == expected


class TestFunctionBrokerScope:
    def test_deterministic_apply_does_not_materialize_entropy(self):
        deterministic = Function(func=lambda value: value + 1)

        with patch("probpipe.core._workflow_context._os_urandom") as urandom:
            assert deterministic.apply(2) == 3

        urandom.assert_not_called()

    def test_lifting_uses_the_active_function_broker(self):
        identity = Function(
            func=lambda value: value,
            n_broadcast_samples=8,
            dispatch="sequential",
        )

        with (
            patch(
                "probpipe.core._workflow_broker._AutomaticKeyBroker.key_for",
                autospec=True,
                wraps=__import__(
                    "probpipe.core._workflow_broker",
                    fromlist=["_AutomaticKeyBroker"],
                )._AutomaticKeyBroker.key_for,
            ) as key_for,
            workflow_run(seed=7),
        ):
            result = identity(Normal(loc=0.0, scale=1.0, name="x"))

        assert result.num_atoms == 8
        key_for.assert_called_once()
