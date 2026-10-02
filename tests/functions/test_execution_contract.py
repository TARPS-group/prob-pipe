"""Execution-contract and JAX side-effect guard tests."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    sample,
    workflow_run,
)
from probpipe.core.config import WorkflowKind
from probpipe.functions import (
    _broker,
    _context,
    _execution,
    _execution_contract,
)
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding


def _plan(values, n_broadcast_samples=8):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(signature)
    broadcast = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast, n_broadcast_samples)


def _record_batch():
    return NumericRecordBatch.stack(
        [NumericRecord("row", x=float(value)) for value in range(4)],
        level_name="draw",
    )


def _add_automatic_noise(row):
    noise = sample(Normal(loc=0.0, scale=1.0, label="noise"))
    return row["x"] + noise


class TestExecutionContract:
    def test_workflow_kind_transport_requires_a_resolved_kind(self):
        assert _execution_contract.transport_for_workflow_kind(WorkflowKind.OFF) == "local_inline"
        assert _execution_contract.transport_for_workflow_kind(WorkflowKind.TASK) == "prefect_task"
        assert _execution_contract.transport_for_workflow_kind(WorkflowKind.FLOW) == "prefect_flow"
        with pytest.raises(ValueError, match="resolved workflow kind"):
            _execution_contract.transport_for_workflow_kind(WorkflowKind.DEFAULT)

    def test_contract_is_frozen_and_uses_the_fixed_abi(self):
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, label="x")})
        contract = _execution_contract.make_execution_contract(
            evaluator="jax_vmap",
            transport="local_inline",
            stochastic_plan=plan,
        )

        assert contract.abi == "probpipe.workflow_rng_execution/v1"
        assert _execution_contract.supports_execution_contract(
            contract,
            plan,
        )
        with pytest.raises(FrozenInstanceError):
            contract.transport = "local_thread"

    def test_exact_plan_is_not_jax_capable_but_is_rowwise_capable(self):
        plan = _plan(
            {"x": EmpiricalDistribution("x", jnp.asarray([1.0, 2.0]))},
            n_broadcast_samples=8,
        )
        jax_contract = _execution_contract.make_execution_contract(
            evaluator="jax_vmap",
            transport="local_inline",
            stochastic_plan=plan,
        )
        rowwise_contract = _execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_thread",
            stochastic_plan=plan,
        )

        assert not _execution_contract.supports_execution_contract(
            jax_contract,
            plan,
        )
        assert _execution_contract.supports_execution_contract(
            rowwise_contract,
            plan,
        )

    def test_unknown_provider_or_key_abi_fails_the_single_predicate(self):
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, label="x")})
        contract = _execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="prefect_task",
            stochastic_plan=plan,
        )

        assert not _execution_contract.supports_execution_contract(
            replace(contract, provider_abis=("unknown",)),
            plan,
        )
        assert not _execution_contract.supports_execution_contract(
            replace(contract, jax_key_abi="unknown"),
            plan,
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
        plan = _plan({"x": Normal(loc=0.0, scale=1.0, label="x")})
        contract = _execution_contract.make_execution_contract(
            evaluator=evaluator,
            transport=transport,
            stochastic_plan=plan,
        )

        assert _execution_contract.supports_execution_contract(contract, plan) is expected

    def test_execution_request_rejects_plan_drift_before_broker_or_user_code(self):
        sampled_plan = _plan({"x": Normal(loc=0.0, scale=1.0, label="x")})
        exact_plan = _plan(
            {"x": EmpiricalDistribution("x", jnp.asarray([1.0, 2.0]))},
            n_broadcast_samples=8,
        )
        contract = _execution_contract.make_execution_contract(
            evaluator="rowwise",
            transport="local_inline",
            stochastic_plan=sampled_plan,
        )
        func = Mock(return_value=1)
        request = _execution.WorkflowExecutionRequest(
            func=func,
            work_items=_execution.make_managed_work_items(
                [{"x": 1}],
                unit_segments=(_execution.point_unit_segment(),),
            ),
            execution=_execution.WorkflowExecutionConfig(mode="sequential"),
            contract=contract,
            stochastic_plan=exact_plan,
        )

        with (
            patch.object(_broker, "_record_active_execution_contract") as record,
            pytest.raises(RuntimeError, match="RNG contract"),
        ):
            _execution.execute_many(request)

        func.assert_not_called()
        record.assert_not_called()


class TestJaxWorkflowGuards:
    def test_auto_falls_back_for_omitted_key_effect_without_shifting_results(self):
        auto = Function(label="_add_automatic_noise", fn=_add_automatic_noise, dispatch="auto")
        rowwise = Function(
            label="_add_automatic_noise", fn=_add_automatic_noise, dispatch="sequential"
        )

        with workflow_run(seed=19):
            auto_result = auto(row=_record_batch())
        with workflow_run(seed=19):
            rowwise_result = rowwise(row=_record_batch())

        np.testing.assert_array_equal(auto_result, rowwise_result)

    def test_explicit_jax_rejects_omitted_key_before_entropy(self):
        workflow = Function(label="_add_automatic_noise", fn=_add_automatic_noise, dispatch="jax")

        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            pytest.raises(TypeError, match="workflow-owned randomness"),
        ):
            workflow(row=_record_batch())

        urandom.assert_not_called()

    def test_actual_jax_guard_rejects_unprobed_dynamic_effect_before_commit(self):
        plan = _broker._singleton_effect_plan(
            operation_kind="dynamic-test",
            execution_mode="sampled",
            sample_shape=(),
        )
        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            _broker._function_stochastic_scope() as broker,
            _context._workflow_jax_runtime_guard(),
            pytest.raises(TypeError, match="JAX workflow execution"),
        ):
            _broker._resolve_automatic_key(None, plan)

        assert broker._invocation is None
        urandom.assert_not_called()
