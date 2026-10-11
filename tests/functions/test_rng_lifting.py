"""End-to-end tests for workflow-owned randomness in Function lifting."""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArraySpec,
    NumericDistribution,
    NumericRecord,
    NumericRecordBatch,
    OutputSpec,
    SupportsSampling,
    function,
    workflow_run,
)
from probpipe.core.constraints import real
from probpipe.functions import _context, _plan


class _RecordingNormal(Normal):
    def __init__(self, sample_calls, *, label):
        self.sample_calls = sample_calls
        super().__init__(label, loc=0.0, scale=1.0, label=label)

    def _sample(self, key, sample_shape=()):
        words = tuple(int(word) for word in jax.random.key_data(key))
        self.sample_calls.append((words, tuple(sample_shape)))
        return super()._sample(key, sample_shape)


class _GoldenBitsDistribution(NumericDistribution, SupportsSampling):
    def __init__(self, sample_calls):
        self.sample_calls = sample_calls
        super().__init__(
            OutputSpec(bits=NumericArraySpec((), "float32", real)),
            label="bits",
        )

    def _sample(self, key, sample_shape=()):
        words = tuple(int(word) for word in jax.random.key_data(key))
        self.sample_calls.append((words, tuple(sample_shape)))
        bits = jax.random.bits(key, shape=sample_shape, dtype=jnp.uint16)
        return bits.astype(jnp.float32)


class TestSequentialLiftingWorkflowRun:
    def test_lifted_key_and_sample_match_v1_end_to_end_golden(self):
        # These values pin ProbPipe RNG v1 through the explicit Threefry key
        # adapter and random-bits sampling. Do not regenerate them for a JAX
        # upgrade without reviewing the intended RNG compatibility contract.
        sample_calls = []
        dist = _GoldenBitsDistribution(sample_calls)

        @function(
            n_broadcast_samples=5,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(value):
            return value

        with workflow_run(seed=17):
            result = identity(dist)

        assert sample_calls == [((3974922193, 721833970), (5,))]
        np.testing.assert_array_equal(
            np.asarray(result.atoms),
            np.asarray([6783, 11879, 26960, 21029, 10625], dtype=np.float32),
        )

    def test_function_builds_one_stochastic_plan_for_a_lifted_call(self):
        @function(
            n_broadcast_samples=8,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        with (
            patch(
                "probpipe.functions._function._plan.build_stochastic_plan",
                wraps=_plan.build_stochastic_plan,
            ) as build_plan,
            workflow_run(seed=7),
        ):
            result = identity(Normal("x", loc=0.0, scale=1.0))

        assert result.num_atoms == 8
        build_plan.assert_called_once()

    def test_invalid_sample_count_fails_before_probe_or_event_commit(self):
        workflow = Function(
            lambda x: x,
            label="function",
            n_broadcast_samples=8,
            dispatch="auto",
        )

        with (
            patch("probpipe.functions._function._resolve_dispatch") as resolve_dispatch,
            patch("probpipe.functions._context._commit_stochastic_invocation") as commit_invocation,
            workflow_run(seed=7),
            pytest.raises(ValueError, match="n_broadcast_samples must be a positive integer"),
        ):
            workflow.with_options(n_broadcast_samples=0)(Normal("x", loc=0.0, scale=1.0))

        resolve_dispatch.assert_not_called()
        commit_invocation.assert_not_called()

    def test_nested_sources_claim_raw_keys_by_row_major_logical_unit(self):
        rows = NumericRecordBatch.stack(
            [
                NumericRecord(
                    {"offset": float(index)},
                    label="row",
                )
                for index in range(3)
            ],
            level_name="draw",
        )

        @function(
            n_broadcast_samples=6,
            dispatch="sequential",
            output_spec=OutputSpec(combine=None),
        )
        def combine(row, first, second):
            return row["offset"] + first + second

        first_calls = []
        second_calls = []
        first = _RecordingNormal(first_calls, label="first")
        second = _RecordingNormal(second_calls, label="second")
        claims = []
        original_key_for = _context._WorkflowInvocation.key_for

        def recording_key_for(
            invocation,
            *,
            stochastic_source_id,
            logical_unit_id,
        ):
            key = original_key_for(
                invocation,
                stochastic_source_id=stochastic_source_id,
                logical_unit_id=logical_unit_id,
            )
            words = tuple(int(word) for word in jax.random.key_data(key))
            claims.append((stochastic_source_id, logical_unit_id, words))
            return key

        with (
            patch.object(
                _context._WorkflowInvocation,
                "key_for",
                new=recording_key_for,
            ),
            workflow_run(seed=17),
        ):
            result = combine(rows, first, second)

        expected_identities = [
            (("source-group", source), ("cell", cell)) for cell in range(3) for source in range(2)
        ]
        assert result.batch_shape == (3,)
        assert [(source, unit) for source, unit, _words in claims] == expected_identities
        assert [shape for _words, shape in first_calls] == [(6,)] * 3
        assert [shape for _words, shape in second_calls] == [(6,)] * 3
        sampled_words = [
            words for cell in range(3) for words in (first_calls[cell][0], second_calls[cell][0])
        ]
        assert sampled_words == [words for _source, _unit, words in claims]
        assert len(set(sampled_words)) == 6

        first_key_words = [words for words, _shape in first_calls]
        second_key_words = [words for words, _shape in second_calls]
        first_calls.clear()
        second_calls.clear()
        with workflow_run(seed=17):
            combine(rows, first, second)

        assert [words for words, _shape in first_calls] == first_key_words
        assert [words for words, _shape in second_calls] == second_key_words

    def test_seeded_run_reproduces_distinct_lifted_occurrences(self):
        @function(
            n_broadcast_samples=16,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        dist = Normal("x", loc=0.0, scale=1.0)

        def run():
            with workflow_run(seed=7):
                return identity(dist), identity(dist)

        first_run = run()
        second_run = run()

        assert jnp.array_equal(first_run[0].atoms, second_run[0].atoms)
        assert jnp.array_equal(first_run[1].atoms, second_run[1].atoms)
        assert not jnp.array_equal(first_run[0].atoms, first_run[1].atoms)

    def test_bare_lifted_calls_receive_independent_ephemeral_roots(self):
        @function(
            n_broadcast_samples=16,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        dist = Normal("x", loc=0.0, scale=1.0)
        with patch(
            "probpipe.functions._context._os_urandom",
            side_effect=[bytes(8), bytes.fromhex("0000000000000001")],
        ) as urandom:
            first = identity(dist)
            second = identity(dist)

        assert not jnp.array_equal(first.atoms, second.atoms)
        assert urandom.call_count == 2

    def test_non_stochastic_siblings_do_not_shift_later_lifting(self):
        @function(
            n_broadcast_samples=16,
            dispatch="sequential",
            output_spec=OutputSpec(identity=None),
        )
        def identity(x):
            return x

        @function(dispatch="sequential", output_spec=OutputSpec(deterministic=None))
        def deterministic(x):
            return x + 1

        normal = Normal("x", loc=0.0, scale=1.0)
        empirical = EmpiricalDistribution(jnp.asarray([1.0, 2.0, 3.0]), component="x")

        def baseline():
            with workflow_run(seed=7):
                return identity(normal), identity(normal)

        def with_siblings():
            with workflow_run(seed=7):
                first = identity(normal)
                deterministic(1.0)
                identity(empirical)
                with pytest.raises(TypeError, match=r"identity\(\) missing required argument 'x'"):
                    identity()
                second = identity(normal)
                return first, second

        expected = baseline()
        actual = with_siblings()

        assert jnp.array_equal(actual[0].atoms, expected[0].atoms)
        assert jnp.array_equal(actual[1].atoms, expected[1].atoms)


def test_auto_probe_detects_nested_randomness_caught_by_user_code():
    @function(
        n_broadcast_samples=5,
        dispatch="sequential",
        output_spec=OutputSpec(inner_identity=None),
    )
    def inner_identity(value):
        return value

    nested_dist = Normal("nested", loc=0.0, scale=1.0)

    def call_nested_and_catch(value):
        with suppress(Exception):
            inner_identity(nested_dist)
        return value

    auto = Function(
        call_nested_and_catch,
        label="call_nested_and_catch",
        n_broadcast_samples=5,
        dispatch="auto",
        output_spec=OutputSpec(call_nested_and_catch=None),
    )
    sequential = Function(
        call_nested_and_catch,
        label="call_nested_and_catch",
        n_broadcast_samples=5,
        dispatch="sequential",
        output_spec=OutputSpec(call_nested_and_catch=None),
    )

    @function(
        n_broadcast_samples=5,
        dispatch="sequential",
        output_spec=OutputSpec(following_identity=None),
    )
    def following_identity(value):
        return value

    outer_dist = Normal("outer", loc=0.0, scale=1.0)

    def following_samples(workflow):
        with workflow_run(seed=7):
            workflow(outer_dist)
            return following_identity(outer_dist)._rows

    assert jnp.array_equal(following_samples(auto), following_samples(sequential))
