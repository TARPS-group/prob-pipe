"""Execution-contract and JAX side-effect guard tests."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordArray,
    sample,
    workflow_run,
)
from probpipe.core import (
    _workflow_broker,
    _workflow_call,
    _workflow_context,
    _workflow_execution_contract,
)
from probpipe.core._workflow_plan import build_broadcast_plan, build_stochastic_plan


def _plan(values, n_broadcast_samples=8):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _workflow_call.make_signature_info_from_signature(signature)
    broadcast = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast, n_broadcast_samples)


def _record_array():
    return NumericRecordArray.stack([NumericRecord("row", x=float(value)) for value in range(4)])


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


class TestJaxWorkflowGuards:
    def test_auto_falls_back_for_omitted_key_effect_without_shifting_results(self):
        auto = Function(func=_add_automatic_noise, dispatch="auto")
        rowwise = Function(func=_add_automatic_noise, dispatch="sequential")

        with workflow_run(seed=19):
            auto_result = auto(row=_record_array())
        with workflow_run(seed=19):
            rowwise_result = rowwise(row=_record_array())

        np.testing.assert_array_equal(auto_result["_add_automatic_noise"], rowwise_result)

    def test_explicit_jax_rejects_omitted_key_before_entropy(self):
        workflow = Function(func=_add_automatic_noise, dispatch="jax")

        with (
            patch("probpipe.core._workflow_context._os_urandom") as urandom,
            workflow_run(),
            pytest.raises(TypeError, match="workflow-owned randomness"),
        ):
            workflow(row=_record_array())

        urandom.assert_not_called()

    def test_caller_keyed_effect_can_trace_and_execute_with_jax(self):
        workflow = Function(func=_add_caller_keyed_noise, dispatch="jax")

        first = workflow(row=_record_array())
        second = workflow(row=_record_array())

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
