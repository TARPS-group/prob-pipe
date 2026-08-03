"""End-to-end tests for workflow-owned randomness in Function lifting."""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from probpipe import EmpiricalDistribution, Function, Normal, function, workflow_run
from probpipe.core import _workflow_plan


class TestSequentialLiftingWorkflowRun:
    def test_function_builds_one_stochastic_plan_for_a_lifted_call(self):
        @function(n_broadcast_samples=8, dispatch="sequential")
        def identity(x):
            return x

        with (
            patch(
                "probpipe.core.node._workflow_plan.build_stochastic_plan",
                wraps=_workflow_plan.build_stochastic_plan,
            ) as build_plan,
            workflow_run(seed=7),
        ):
            result = identity(Normal(loc=0.0, scale=1.0, name="x"))

        assert result.num_atoms == 8
        build_plan.assert_called_once()

    def test_invalid_sample_count_fails_before_probe_or_event_commit(self):
        workflow = Function(
            func=lambda x: x,
            n_broadcast_samples=8,
            dispatch="auto",
        )

        with (
            patch.object(Function, "_resolve_dispatch") as resolve_dispatch,
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation"
            ) as commit_invocation,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="n_broadcast_samples must be a positive integer"),
        ):
            workflow.with_options(n_broadcast_samples=0)(Normal(loc=0.0, scale=1.0, name="x"))

        resolve_dispatch.assert_not_called()
        commit_invocation.assert_not_called()

    def test_seeded_run_reproduces_distinct_lifted_occurrences(self):
        @function(n_broadcast_samples=16, dispatch="sequential")
        def identity(x):
            return x

        dist = Normal(loc=0.0, scale=1.0, name="x")

        def run():
            with workflow_run(seed=7):
                return identity(dist), identity(dist)

        first_run = run()
        second_run = run()

        assert jnp.array_equal(first_run[0].samples, second_run[0].samples)
        assert jnp.array_equal(first_run[1].samples, second_run[1].samples)
        assert not jnp.array_equal(first_run[0].samples, first_run[1].samples)

    def test_bare_lifted_calls_receive_independent_ephemeral_roots(self):
        @function(n_broadcast_samples=16, dispatch="sequential")
        def identity(x):
            return x

        dist = Normal(loc=0.0, scale=1.0, name="x")
        with patch(
            "probpipe.core._workflow_context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            first = identity(dist)
            second = identity(dist)

        assert not jnp.array_equal(first.samples, second.samples)
        assert urandom.call_count == 2

    def test_non_stochastic_siblings_do_not_shift_later_lifting(self):
        @function(n_broadcast_samples=16, dispatch="sequential")
        def identity(x):
            return x

        @function(dispatch="sequential")
        def deterministic(x):
            return x + 1

        normal = Normal(loc=0.0, scale=1.0, name="x")
        empirical = EmpiricalDistribution(jnp.asarray([1.0, 2.0, 3.0]), name="x")

        def baseline():
            with workflow_run(seed=7):
                return identity(normal), identity(normal)

        def with_siblings():
            with workflow_run(seed=7):
                first = identity(normal)
                deterministic(1.0)
                identity(empirical)
                with pytest.raises(TypeError, match="Missing required input"):
                    identity()
                second = identity(normal)
                return first, second

        expected = baseline()
        actual = with_siblings()

        assert jnp.array_equal(actual[0].samples, expected[0].samples)
        assert jnp.array_equal(actual[1].samples, expected[1].samples)


def test_auto_probe_detects_nested_randomness_caught_by_user_code():
    @function(n_broadcast_samples=5, dispatch="sequential")
    def inner_identity(value):
        return value

    nested_dist = Normal(loc=0.0, scale=1.0, name="nested")

    def call_nested_and_catch(value):
        with suppress(Exception):
            inner_identity(nested_dist)
        return value

    auto = Function(
        func=call_nested_and_catch,
        n_broadcast_samples=5,
        dispatch="auto",
    )
    sequential = Function(
        func=call_nested_and_catch,
        n_broadcast_samples=5,
        dispatch="sequential",
    )

    @function(n_broadcast_samples=5, dispatch="sequential")
    def following_identity(value):
        return value

    outer_dist = Normal(loc=0.0, scale=1.0, name="outer")

    def following_samples(workflow):
        with workflow_run(seed=7):
            workflow(outer_dist)
            return following_identity(outer_dist).samples["marginal"]

    assert jnp.array_equal(following_samples(auto), following_samples(sequential))
