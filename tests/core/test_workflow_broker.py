"""Tests for the private workflow automatic-key broker."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import jax
import pytest

import probpipe.core._workflow_broker as broker_mod
import probpipe.core._workflow_context as context_mod
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


def _plan(source: int = 0, **overrides) -> StochasticEffectPlan:
    values = {
        "operation_kind": "test",
        "execution_mode": "sampled",
        "event": PlannedRandomEvent(("source-group", source), ("singleton",)),
        "sample_shape": (8,),
        "sampling_abi": "test-sampling/v1",
        "provider_abi": "test-provider/v1",
    }
    values.update(overrides)
    return StochasticEffectPlan(**values)


def _key_words(key) -> tuple[int, int]:
    return tuple(int(word) for word in jax.random.key_data(key))


class TestStochasticEffectPlan:
    def test_plan_is_frozen_and_tuple_only(self):
        descriptor = ("descriptor", ("value", 1), None, True, b"bytes")
        plan = _plan(
            record_path=("nested", "leaf"),
            descendant_descriptor=descriptor,
        )

        assert plan.sample_shape == (8,)
        assert isinstance(plan.sample_shape, tuple)
        assert isinstance(plan.event.stochastic_source_id, tuple)
        assert isinstance(plan.event.logical_unit_id, tuple)
        assert plan.record_path == ("nested", "leaf")
        assert plan.descendant_descriptor == descriptor
        with pytest.raises(FrozenInstanceError):
            plan.operation_kind = "changed"

    @pytest.mark.parametrize(
        ("field", "value", "exception", "match"),
        [
            pytest.param(
                "operation_kind",
                1,
                TypeError,
                "operation_kind must be a string",
                id="operation-kind-type",
            ),
            pytest.param(
                "operation_kind",
                "",
                ValueError,
                "operation_kind must be non-empty",
                id="operation-kind-empty",
            ),
            pytest.param(
                "execution_mode",
                None,
                TypeError,
                "execution_mode must be a string",
                id="execution-mode-type",
            ),
            pytest.param(
                "execution_mode",
                "",
                ValueError,
                "execution_mode must be non-empty",
                id="execution-mode-empty",
            ),
            pytest.param(
                "sampling_abi",
                b"sampling/v1",
                TypeError,
                "sampling_abi must be a string",
                id="sampling-abi-type",
            ),
            pytest.param(
                "sampling_abi",
                "",
                ValueError,
                "sampling_abi must be non-empty",
                id="sampling-abi-empty",
            ),
            pytest.param(
                "provider_abi",
                1,
                TypeError,
                "provider_abi must be a string",
                id="provider-abi-type",
            ),
            pytest.param(
                "provider_abi",
                "",
                ValueError,
                "provider_abi must be non-empty",
                id="provider-abi-empty",
            ),
            pytest.param(
                "sample_shape",
                [1],
                TypeError,
                "sample shapes must be tuples or None",
                id="sample-shape-container",
            ),
            pytest.param(
                "sample_shape",
                (True,),
                TypeError,
                "sample shape dimensions must be non-boolean integers",
                id="sample-shape-bool",
            ),
            pytest.param(
                "sample_shape",
                (1.5,),
                TypeError,
                "sample shape dimensions must be non-boolean integers",
                id="sample-shape-item",
            ),
            pytest.param(
                "sample_shape",
                (-1,),
                ValueError,
                "sample shape dimensions must be non-negative",
                id="sample-shape-negative",
            ),
            pytest.param(
                "record_path",
                ["field"],
                TypeError,
                "record paths must be tuples",
                id="record-path-container",
            ),
            pytest.param(
                "record_path",
                ("field", 1),
                TypeError,
                "record path fields must be strings",
                id="record-path-item",
            ),
            pytest.param(
                "descendant_descriptor",
                ["descriptor"],
                TypeError,
                "descendant descriptors must be tuples or None",
                id="descriptor-container",
            ),
            pytest.param(
                "descendant_descriptor",
                ("descriptor", ["mutable"]),
                TypeError,
                "canonical tuple values",
                id="descriptor-nested-list",
            ),
            pytest.param(
                "descendant_descriptor",
                ("descriptor", 1.5),
                TypeError,
                "canonical tuple values",
                id="descriptor-float",
            ),
        ],
    )
    def test_invalid_effect_fields_fail_at_plan_construction(
        self,
        field,
        value,
        exception,
        match,
    ):
        with pytest.raises(exception, match=match):
            _plan(**{field: value})

    def test_corrupted_effect_fields_fail_before_stochastic_commit(self):
        invalid_plan = _plan()
        object.__setattr__(invalid_plan, "sample_shape", (True,))

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            patch("probpipe.core._workflow_context.derive_event_key_words_from_encoded") as derive,
            workflow_run(),
            _function_stochastic_scope() as broker,
            pytest.raises(TypeError, match="sample shape dimensions"),
        ):
            _resolve_automatic_key(None, invalid_plan)

        assert broker._invocation is None
        urandom.assert_not_called()
        commit.assert_not_called()
        derive.assert_not_called()

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

    @pytest.mark.parametrize(
        "scope",
        [_function_stochastic_scope, _managed_stochastic_scope],
        ids=["function", "operation"],
    )
    def test_one_scope_rejects_duplicate_effect_before_second_derivation(self, scope):
        derivations = 0
        original_key_for = context_mod._WorkflowInvocation.key_for

        def count_derivation(invocation, *, stochastic_source_id, logical_unit_id):
            nonlocal derivations
            derivations += 1
            return original_key_for(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        with (
            patch.object(
                context_mod._WorkflowInvocation,
                "key_for",
                new=count_derivation,
            ),
            workflow_run(seed=7),
            scope(),
        ):
            _resolve_automatic_key(None, _plan())
            with pytest.raises(RuntimeError, match="duplicated a stochastic effect claim"):
                _resolve_automatic_key(None, _plan())

        assert derivations == 1

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

    def test_new_operation_scope_seals_its_broker(self):
        with workflow_run(seed=7), _managed_stochastic_scope() as operation_broker:
            assert operation_broker._lifecycle is broker_mod._BrokerLifecycle.OPEN

        assert operation_broker._lifecycle is broker_mod._BrokerLifecycle.SEALED

    def test_new_operation_scope_seals_after_body_error(self):
        with (
            workflow_run(seed=7),
            pytest.raises(ValueError, match="body failed"),
            _managed_stochastic_scope() as operation_broker,
        ):
            raise ValueError("body failed")

        assert operation_broker._lifecycle is broker_mod._BrokerLifecycle.SEALED

    def test_reused_operation_scope_does_not_seal_its_owner(self):
        with workflow_run(seed=7), _function_stochastic_scope() as function_broker:
            with _managed_stochastic_scope() as reused_broker:
                assert reused_broker is function_broker

            assert function_broker._lifecycle is broker_mod._BrokerLifecycle.OPEN
            _resolve_automatic_key(None, _plan())

        assert function_broker._lifecycle is broker_mod._BrokerLifecycle.SEALED

    def test_sealed_broker_rejects_new_effect_before_occurrence_commit(self):
        with workflow_run(seed=7):
            with _function_stochastic_scope() as broker:
                pass

            with (
                patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
                pytest.raises(RuntimeError, match="sealed"),
            ):
                broker.key_for(_plan())

        commit.assert_not_called()

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
