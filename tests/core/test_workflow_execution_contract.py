"""Execution-contract and JAX side-effect guard tests."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    TransformedDistribution,
    sample,
    workflow_run,
)
from probpipe.core import (
    _workflow_broker,
    _workflow_call,
    _workflow_context,
    _workflow_descendants,
    _workflow_execution,
    _workflow_execution_contract,
)
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan
from probpipe.core.config import WorkflowKind


def _plan(values, n_broadcast_samples=8):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(signature)
    broadcast = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast, n_broadcast_samples)


def _record_batch():
    return NumericRecordBatch.stack(
        [NumericRecord("row", x=float(value)) for value in range(4)],
        level_name="draw",
    )


def _add_automatic_noise(row):
    noise = sample(Normal(loc=0.0, scale=1.0, name="noise"))
    return row["x"] + noise["sample"]


def _add_caller_keyed_noise(row):
    noise = sample(
        Normal(loc=0.0, scale=1.0, name="noise"),
        key=jax.random.key(3),
    )
    return row["x"] + noise["sample"]


class TestExecutionContract:
    def test_workflow_kind_transport_requires_a_resolved_kind(self):
        assert (
            _workflow_execution_contract.transport_for_workflow_kind(WorkflowKind.OFF)
            == "local_inline"
        )
        assert (
            _workflow_execution_contract.transport_for_workflow_kind(WorkflowKind.TASK)
            == "prefect_task"
        )
        assert (
            _workflow_execution_contract.transport_for_workflow_kind(WorkflowKind.FLOW)
            == "prefect_flow"
        )
        with pytest.raises(ValueError, match="resolved workflow kind"):
            _workflow_execution_contract.transport_for_workflow_kind(WorkflowKind.DEFAULT)

    def test_contract_is_frozen_and_uses_the_fixed_abi(self):
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        contract = _workflow_execution_contract.make_execution_contract(
            evaluator="jax_vmap",
            transport="local_inline",
            stochastic_plan=plan,
        )

        assert contract.abi == "probpipe.workflow_rng_execution/v1"
        assert _workflow_execution_contract.supports_execution_contract(
            contract,
            plan,
        )
        with pytest.raises(FrozenInstanceError):
            contract.transport = "local_thread"

    def test_exact_plan_is_not_jax_capable_but_is_rowwise_capable(self):
        plan = _plan(
            {"x": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="x")},
            n_broadcast_samples=8,
        )
        jax_contract = _workflow_execution_contract.make_execution_contract(
            evaluator="jax_vmap",
            transport="local_inline",
            stochastic_plan=plan,
        )
        rowwise_contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_thread",
            stochastic_plan=plan,
        )

        assert not _workflow_execution_contract.supports_execution_contract(
            jax_contract,
            plan,
        )
        assert _workflow_execution_contract.supports_execution_contract(
            rowwise_contract,
            plan,
        )

    def test_unknown_provider_or_key_abi_fails_the_single_predicate(self):
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="prefect_task",
            stochastic_plan=plan,
        )

        assert not _workflow_execution_contract.supports_execution_contract(
            replace(contract, provider_abis=("unknown",)),
            plan,
        )
        assert not _workflow_execution_contract.supports_execution_contract(
            replace(contract, jax_key_abi="unknown"),
            plan,
        )

    def test_contract_is_bound_to_the_exact_plan_requirements(self):
        sampled_plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        exact_plan = _plan(
            {"x": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="x")},
            n_broadcast_samples=8,
        )
        transformed_plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )
        sampled_contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=sampled_plan,
        )
        transformed_contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=transformed_plan,
        )

        assert not _workflow_execution_contract.supports_execution_contract(
            sampled_contract,
            exact_plan,
        )
        assert not _workflow_execution_contract.supports_execution_contract(
            sampled_contract,
            transformed_plan,
        )
        assert not _workflow_execution_contract.supports_execution_contract(
            transformed_contract,
            sampled_plan,
        )

    def test_nested_descriptor_fields_are_collected_without_overwrite(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )
        group = plan.source_groups[0]
        consumer = group.consumers[0]
        hidden_provider = "example.hidden/provider-v1"
        wrapped_consumer = replace(
            consumer,
            descendant_descriptor=(
                "test-wrapper",
                ("provider_abi", hidden_provider),
                ("nested", consumer.descendant_descriptor),
            ),
        )
        wrapped_plan = replace(
            plan,
            source_groups=(replace(group, consumers=(wrapped_consumer,)),),
        )

        contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=wrapped_plan,
        )

        assert hidden_provider in wrapped_consumer._descriptor_abi_summary.provider_abis
        assert hidden_provider in contract.provider_abis
        assert not _workflow_execution_contract.supports_execution_contract(
            contract,
            wrapped_plan,
        )

    def test_nested_descriptor_abi_summary_is_sorted_and_deduplicated(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    TransformedDistribution(
                        Normal(loc=0.0, scale=1.0, name="base"),
                        tfb.Exp(),
                    ),
                    tfb.Shift(1.0),
                )
            }
        )

        summary = plan.source_groups[0].consumers[0]._descriptor_abi_summary

        assert summary.sampling_abis == ("probpipe.distribution_sampling/v1",)
        assert summary.provider_abis == (
            "tensorflow_probability.substrates.jax.bijector.forward/v1",
        )
        assert summary.descendant_adapter_abis == ("probpipe.transformed_descendant/v1",)
        assert all(
            isinstance(values, tuple)
            for values in (
                summary.sampling_abis,
                summary.provider_abis,
                summary.descendant_adapter_abis,
            )
        )
        with pytest.raises(FrozenInstanceError):
            summary.provider_abis = ()

    def test_execution_contract_reuses_plan_cached_descriptor_summary(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )

        with patch.object(
            _workflow_descendants,
            "_summarize_descriptor_abis",
            side_effect=AssertionError("descriptor was scanned again"),
        ):
            contract = _workflow_execution_contract.make_execution_contract(
                evaluator="rowwise",
                transport="local_inline",
                stochastic_plan=plan,
            )
            assert _workflow_execution_contract.supports_execution_contract(contract, plan)
            assert _workflow_execution_contract.supports_execution_contract(contract, plan)

    def test_descriptor_summary_does_not_compute_an_unused_digest(self):
        with patch.object(
            _workflow_descendants,
            "descriptor_digest",
            side_effect=AssertionError("unexpected descriptor digest"),
        ):
            plan = _plan(
                {
                    "x": TransformedDistribution(
                        Normal(loc=0.0, scale=1.0, name="base"),
                        tfb.Exp(),
                    )
                }
            )
            contract = _workflow_execution_contract.make_execution_contract(
                evaluator="rowwise",
                transport="local_inline",
                stochastic_plan=plan,
            )

        assert _workflow_execution_contract.supports_execution_contract(contract, plan)

    def test_replaced_consumer_rebuilds_descriptor_summary(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )
        consumer = plan.source_groups[0].consumers[0]

        replaced_consumer = replace(
            consumer,
            descendant_descriptor=(
                "test-wrapper",
                ("provider_abi", "example.replaced/provider-v1"),
                ("nested", consumer.descendant_descriptor),
            ),
        )

        assert replaced_consumer._descriptor_abi_summary is not (consumer._descriptor_abi_summary)
        assert replaced_consumer._descriptor_abi_summary.provider_abis == (
            "example.replaced/provider-v1",
            "tensorflow_probability.substrates.jax.bijector.forward/v1",
        )

    def test_sampling_abi_drift_still_fails_before_contract_execution(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )
        group = plan.source_groups[0]
        consumer = group.consumers[0]
        drifted_consumer = replace(
            consumer,
            descendant_descriptor=(
                "test-wrapper",
                ("sampling_abi", "example.sampling/v2"),
                ("nested", consumer.descendant_descriptor),
            ),
        )
        drifted_plan = replace(
            plan,
            source_groups=(replace(group, consumers=(drifted_consumer,)),),
        )

        with pytest.raises(ValueError, match="unsupported sampling ABI"):
            _workflow_execution_contract.make_execution_contract(
                evaluator="rowwise",
                transport="local_inline",
                stochastic_plan=drifted_plan,
            )

    def test_noncanonical_abi_sequences_and_cross_field_drift_are_rejected(self):
        plan = _plan(
            {
                "x": TransformedDistribution(
                    Normal(loc=0.0, scale=1.0, name="base"),
                    tfb.Exp(),
                )
            }
        )
        contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=plan,
        )
        direct_plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        direct_contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=direct_plan,
        )

        invalid_transformed_contracts = (
            replace(contract, provider_abis=contract.provider_abis * 2),
            replace(contract, provider_abis=tuple(reversed(contract.provider_abis))),
            replace(
                contract,
                descendant_adapter_abis=contract.descendant_adapter_abis * 2,
            ),
            replace(contract, plan_evaluation_mode="exact"),
            replace(contract, descendant_adapter_abis=()),
        )
        assert all(
            not _workflow_execution_contract.supports_execution_contract(item, plan)
            for item in invalid_transformed_contracts
        )
        assert not _workflow_execution_contract.supports_execution_contract(
            replace(
                direct_contract,
                descendant_adapter_abis=("probpipe.transformed_descendant/v1",),
            ),
            direct_plan,
        )

    @pytest.mark.parametrize(
        ("evaluator", "transport", "expected"),
        [
            pytest.param("rowwise", "local_inline", True),
            pytest.param("rowwise", "local_thread", True),
            pytest.param("rowwise", "prefect_task", True),
            pytest.param("rowwise", "prefect_flow", True),
            pytest.param("jax_vmap", "local_inline", True),
            pytest.param("jax_vmap", "local_thread", False),
            pytest.param("jax_vmap", "prefect_task", True),
            pytest.param("jax_vmap", "prefect_flow", True),
        ],
    )
    def test_evaluator_transport_support_matrix(self, evaluator, transport, expected):
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        contract = _workflow_execution_contract.make_execution_contract(
            evaluator=evaluator,
            transport=transport,
            stochastic_plan=plan,
        )

        assert _workflow_execution_contract.supports_execution_contract(contract, plan) is expected

    def test_execution_request_rejects_plan_drift_before_broker_or_user_code(self):
        sampled_plan = _plan({"x": Normal(loc=0.0, scale=1.0, name="x")})
        exact_plan = _plan(
            {"x": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), name="x")},
            n_broadcast_samples=8,
        )
        contract = _workflow_execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=sampled_plan,
        )
        func = Mock(return_value=1)
        request = _workflow_execution.WorkflowExecutionRequest(
            func=func,
            work_items=_workflow_execution.make_managed_work_items(
                [{"x": 1}],
                unit_segments=(_workflow_execution.point_unit_segment(),),
            ),
            execution=_workflow_execution.WorkflowExecutionConfig(mode="sequential"),
            contract=contract,
            stochastic_plan=exact_plan,
        )

        with (
            patch.object(_workflow_broker, "_record_active_execution_contract") as record,
            pytest.raises(RuntimeError, match="RNG contract"),
        ):
            _workflow_execution.execute_many(request)

        func.assert_not_called()
        record.assert_not_called()


class TestJaxWorkflowGuards:
    def test_auto_falls_back_for_omitted_key_effect_without_shifting_results(self):
        auto = Function(func=_add_automatic_noise, dispatch="auto")
        rowwise = Function(func=_add_automatic_noise, dispatch="sequential")

        with workflow_run(seed=19):
            auto_result = auto(row=_record_batch())
        with workflow_run(seed=19):
            rowwise_result = rowwise(row=_record_batch())

        np.testing.assert_array_equal(auto_result["_add_automatic_noise"], rowwise_result)

    def test_explicit_jax_rejects_omitted_key_before_entropy(self):
        workflow = Function(func=_add_automatic_noise, dispatch="jax")

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
            pytest.raises(TypeError, match="workflow-owned randomness"),
        ):
            workflow(row=_record_batch())

        urandom.assert_not_called()

    def test_caller_keyed_effect_can_trace_and_execute_with_jax(self):
        workflow = Function(func=_add_caller_keyed_noise, dispatch="jax")

        first = workflow(row=_record_batch())
        second = workflow(row=_record_batch())

        np.testing.assert_array_equal(first, second)

    def test_actual_jax_guard_rejects_unprobed_dynamic_effect_before_commit(self):
        plan = _workflow_broker._singleton_effect_plan(
            operation_kind="dynamic-test",
            execution_mode="sampled",
            sample_shape=(),
        )
        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
            _workflow_broker._function_stochastic_scope() as broker,
            _workflow_context._workflow_jax_runtime_guard(),
            pytest.raises(TypeError, match="JAX workflow execution"),
        ):
            _workflow_broker._resolve_automatic_key(None, plan)

        assert broker._invocation is None
        urandom.assert_not_called()
