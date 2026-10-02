"""Workflow RNG ownership tests for sampling and expectation operations."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Normal,
    expectation,
    sample,
    workflow_run,
)
from probpipe.functions import _context


class _RecordingNormal(Normal):
    def __init__(self, calls):
        self.calls = calls
        super().__init__(loc=0.0, scale=1.0, name="x")

    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _FailOnceNormal(_RecordingNormal):
    def _sample(self, key, sample_shape=()):
        self.calls.append((key, tuple(sample_shape)))
        if len(self.calls) == 1:
            raise RuntimeError("planned sample failure")
        return Normal._sample(self, key, sample_shape)


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

    def test_sample_shape_does_not_multiply_events(self):
        claims = []
        original = _context._WorkflowInvocation.key_for

        def record(invocation, *, stochastic_source_id, logical_unit_id):
            claims.append((stochastic_source_id, logical_unit_id))
            return original(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )

        with (
            patch.object(_context._WorkflowInvocation, "key_for", new=record),
            workflow_run(seed=7),
        ):
            result = sample(Normal(loc=0.0, scale=1.0, name="x"), sample_shape=(4, 5))

        assert result.shape == (4, 5)
        assert claims == [(("source-group", 0), ("singleton",))]

    @pytest.mark.parametrize(
        "sample_shape",
        [
            pytest.param(np.int64(4), id="numpy-scalar"),
            pytest.param((np.int64(4),), id="numpy-tuple"),
            pytest.param(jnp.int32(4), id="jax-scalar"),
            pytest.param((jnp.int32(4),), id="jax-tuple"),
        ],
    )
    def test_integer_protocol_sample_shapes_match_python_ints(self, sample_shape):
        dist = Normal(loc=0.0, scale=1.0, name="x")

        def run(shape):
            with workflow_run(seed=7):
                return sample(dist, sample_shape=shape)

        expected = run((4,))
        actual = run(sample_shape)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("sample_shape", [True, -1, (2, -1), (2, 1.5), [2]])
    def test_invalid_sample_shape_fails_before_event_commit(self, sample_shape):
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            sample(Normal(loc=0.0, scale=1.0, name="x"), sample_shape=sample_shape)

        commit.assert_not_called()

    def test_bare_samples_receive_independent_ephemeral_roots(self):
        dist = Normal(loc=0.0, scale=1.0, name="x")
        with patch(
            "probpipe.functions._context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            first = sample(dist, sample_shape=8)
            second = sample(dist, sample_shape=8)

        assert not jnp.array_equal(first, second)
        assert urandom.call_count == 2

    def test_direct_reentry_after_post_commit_failure_uses_a_new_path(self):
        calls = []
        dist = _FailOnceNormal(calls)

        with workflow_run(seed=7):
            with pytest.raises(RuntimeError, match="planned sample failure"):
                sample(dist)
            sample(dist)

        words = [tuple(int(word) for word in jax.random.key_data(key)) for key, _shape in calls]
        assert len(words) == 2
        assert words[0] != words[1]


class TestAutomaticExpectation:
    def test_exact_empirical_expectation_claims_no_event(self):
        dist = EmpiricalDistribution("x", jnp.asarray([1.0, 2.0, 3.0]))

        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
        ):
            result = expectation(dist, lambda value: value)

        np.testing.assert_allclose(float(result), 2.0)
        commit.assert_not_called()

    def test_monte_carlo_expectation_claims_one_batched_event(self):
        calls = []
        dist = _RecordingNormal(calls)

        with workflow_run(seed=7):
            result = expectation.with_options(n_broadcast_samples=32)(dist, lambda value: value)

        assert jnp.asarray(result).shape == ()
        assert [(shape) for _key, shape in calls] == [(32,)]

    @pytest.mark.parametrize("num_evaluations", [True, 0, -1, 1.5])
    def test_invalid_evaluation_count_fails_before_event_commit(self, num_evaluations):
        with (
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit,
            workflow_run(seed=7),
            pytest.raises((TypeError, ValueError)),
        ):
            expectation.with_options(n_broadcast_samples=num_evaluations)(
                Normal(loc=0.0, scale=1.0, name="x"), lambda value: value
            )

        commit.assert_not_called()
