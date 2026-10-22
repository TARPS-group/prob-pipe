"""Tests of the execution contract and the JAX side-effect guards."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericRecord,
    NumericRecordBatch,
    ResolutionError,
    sample,
    workflow_run,
)
from probpipe.functions import (
    _context,
    _evaluation_keys,
    _execution,
)
from probpipe.functions._plan import build_broadcast_plan, build_stochastic_plan
from probpipe.values import _binding


def _plan(values, max_evaluations=8):
    signature = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in values]
    )
    signature_info = _binding.make_signature_info_from_signature(signature)
    broadcast = build_broadcast_plan(values=values, signature_info=signature_info)
    return build_stochastic_plan(values, broadcast, max_evaluations)


def _record_batch():
    return NumericRecordBatch.stack(
        [
            NumericRecord(
                {"x": float(value)},
                label="row",
            )
            for value in range(4)
        ],
        level_name="draw",
    )


def _add_automatic_noise(row):
    noise = sample(Normal("noise", loc=0.0, scale=1.0))
    return row["x"] + noise


class TestExecutionContract:
    def test_contract_is_frozen_and_versions_only_what_decides_the_keys(self):
        plan = _plan({"x": Normal("x", loc=0.0, scale=1.0)})
        contract = _execution.make_execution_contract(stochastic_plan=plan)

        assert contract.abi == "probpipe.workflow_rng_execution/v2"
        assert not hasattr(contract, "evaluator")
        assert not hasattr(contract, "transport")
        assert _execution.supports_execution_contract(contract, plan)
        with pytest.raises(FrozenInstanceError):
            contract.abi = "probpipe.workflow_rng_execution/v1"

    def test_an_exact_plan_and_a_sampled_plan_each_have_their_own_contract(self):
        sampled_plan = _plan({"x": Normal("x", loc=0.0, scale=1.0)})
        exact_plan = _plan(
            {"x": EmpiricalDistribution(jnp.asarray([1.0, 2.0]), output_name="x")},
            max_evaluations=8,
        )
        assert exact_plan.evaluation_mode == "exact"
        exact_contract = _execution.make_execution_contract(stochastic_plan=exact_plan)
        sampled_contract = _execution.make_execution_contract(stochastic_plan=sampled_plan)

        assert _execution.supports_execution_contract(exact_contract, exact_plan)
        assert not _execution.supports_execution_contract(sampled_contract, exact_plan)

    def test_unknown_provider_or_key_abi_fails_the_single_predicate(self):
        plan = _plan({"x": Normal("x", loc=0.0, scale=1.0)})
        contract = _execution.make_execution_contract(stochastic_plan=plan)

        assert not _execution.supports_execution_contract(
            replace(contract, provider_abis=("unknown",)),
            plan,
        )
        assert not _execution.supports_execution_contract(
            replace(contract, jax_key_abi="unknown"),
            plan,
        )

    def test_every_executor_records_the_same_contract_and_names_itself_beside_it(self):
        shifted = Function(lambda x: np.asarray(x) + 1.0, label="shifted", traceable=False)
        diagnostics_by_executor = {}
        for executor in ("sequential", "thread"):
            with workflow_run(seed=3):
                lifted = shifted.using(executor=executor)(Normal("x", loc=0.0, scale=1.0))
            diagnostics_by_executor[executor] = lifted.provenance.diagnostics["execution"]

        assert diagnostics_by_executor == {
            executor: [
                {
                    "executor": executor,
                    "executor_options": {},
                    "contract_abi": "probpipe.workflow_rng_execution/v2",
                }
            ]
            for executor in ("sequential", "thread")
        }


class TestJaxWorkflowGuards:
    def test_a_traceable_body_that_draws_without_a_key_runs_one_point_at_a_time(self):
        probed = Function(_add_automatic_noise, label="_add_automatic_noise")
        one_at_a_time = Function(
            _add_automatic_noise, label="_add_automatic_noise", traceable=False
        )

        with workflow_run(seed=19):
            probed_result = probed(row=_record_batch())
        with workflow_run(seed=19):
            one_at_a_time_result = one_at_a_time(row=_record_batch())

        np.testing.assert_array_equal(probed_result, one_at_a_time_result)

    def test_naming_jax_for_a_body_that_draws_without_a_key_raises_before_entropy(self):
        workflow = Function(_add_automatic_noise, label="_add_automatic_noise", executor="jax")

        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            pytest.raises(
                ResolutionError,
                match=(
                    "executor 'jax' cannot run the evaluation points of '_add_automatic_noise': "
                    "JAX cannot trace '_add_automatic_noise': it draws random numbers without an "
                    "explicit key"
                ),
            ),
        ):
            workflow(row=_record_batch())

        urandom.assert_not_called()

    def test_actual_jax_guard_rejects_unprobed_dynamic_effect_before_commit(self):
        plan = _evaluation_keys._singleton_effect_plan(
            operation_kind="dynamic-test",
            execution_mode="sampled",
            sample_shape=(),
        )
        with (
            patch("probpipe.functions._context._os_urandom") as urandom,
            workflow_run(),
            _evaluation_keys._function_stochastic_scope() as broker,
            _context._workflow_jax_runtime_guard(),
            pytest.raises(TypeError, match="cannot draw random numbers without an explicit key"),
        ):
            _evaluation_keys._resolve_automatic_key(None, plan)

        assert broker._invocation is None
        urandom.assert_not_called()
