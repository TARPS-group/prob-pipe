"""Workflow RNG ownership tests for sampling and expectation operations."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import EmpiricalDistribution, Normal, expectation, sample, workflow_run
from probpipe.core import _workflow_context


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class TestAutomaticSample:
    def test_seeded_runs_reproduce_distinct_sample_occurrences(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        def run():
            with workflow_run(seed=7):
                return sample(dist, sample_shape=8), sample(dist, sample_shape=8)

        first = run()
        second = run()

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert not jnp.array_equal(first[0], first[1])

    def test_explicit_key_passes_through_and_does_not_shift_automatic_sample(self):
        calls = []
        dist = _RecordingNormal(calls)
        explicit = jax.random.key(11)

        with workflow_run(seed=7):
            expected = sample(dist, sample_shape=4)

        calls.clear()
        with workflow_run(seed=7):
            sample(dist, key=explicit, sample_shape=4)
            actual = sample(dist, sample_shape=4)

        assert calls[0][0] is explicit
        np.testing.assert_array_equal(actual, expected)

    def test_sample_shape_does_not_multiply_events(self):
        claims = []
        original = _workflow_context._WorkflowInvocation.key_for

        def record(invocation, *, stochastic_source_id, logical_unit_id):
            claims.append((stochastic_source_id, logical_unit_id))
            return original(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        with (
            patch.object(_workflow_context._WorkflowInvocation, "key_for", new=record),
            workflow_run(seed=7),
        ):
            result = sample(Normal(loc=0.0, scale=1.0, name="x"), sample_shape=(4, 5))

        assert result.shape == (4, 5)
        assert claims == [(("source-group", 0), ("singleton",))]

    @pytest.mark.parametrize("sample_shape", [True, -1, (2, -1), (2, 1.5), [2]])
    def test_invalid_sample_shape_fails_before_event_commit(self, sample_shape):
        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            sample(Normal(loc=0.0, scale=1.0, name="x"), sample_shape=sample_shape)

        commit.assert_not_called()

    def test_bare_samples_receive_independent_ephemeral_roots(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        with patch(
            "probpipe.core._workflow_context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            first = sample(dist, sample_shape=8)
            second = sample(dist, sample_shape=8)

        assert not jnp.array_equal(first, second)
        assert urandom.call_count == 2


class TestAutomaticExpectation:
    def test_exact_empirical_expectation_claims_no_event(self):
        dist = EmpiricalDistribution(jnp.asarray([1.0, 2.0, 3.0]), name="x")

        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
        ):
            result = expectation(dist, lambda value: value, return_dist=False)

        np.testing.assert_allclose(float(result), 2.0)
        commit.assert_not_called()

    def test_empirical_subsample_claims_one_event(self):
        dist = EmpiricalDistribution(jnp.arange(20.0), name="x")

        with (
            patch(
                "probpipe.core._workflow_context._commit_stochastic_invocation",
                wraps=_workflow_context._commit_stochastic_invocation,
            ) as commit,
            workflow_run(seed=7),
        ):
            expectation(
                dist,
                lambda value: value,
                num_evaluations=5,
                return_dist=False,
            )

        commit.assert_called_once_with("invocation")

    def test_monte_carlo_expectation_claims_one_batched_event(self):
        calls = []
        dist = _RecordingNormal(calls)

        with workflow_run(seed=7):
            result = expectation(
                dist,
                lambda value: value,
                num_evaluations=32,
                return_dist=False,
            )

        assert jnp.asarray(result).shape == ()
        assert [(shape) for _key, shape in calls] == [(32,)]

    @pytest.mark.parametrize("num_evaluations", [True, 0, -1, 1.5])
    def test_invalid_evaluation_count_fails_before_event_commit(self, num_evaluations):
        with (
            patch("probpipe.core._workflow_context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            expectation(
                Normal(loc=0.0, scale=1.0, name="x"),
                lambda value: value,
                num_evaluations=num_evaluations,
            )

        commit.assert_not_called()
