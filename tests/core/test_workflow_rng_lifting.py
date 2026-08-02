"""End-to-end tests for workflow-owned randomness in Function lifting."""

from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from probpipe import EmpiricalDistribution, Normal, function, workflow_run


class TestSequentialLiftingWorkflowRun:
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
            "probpipe.core._workflow_context.os.urandom",
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
