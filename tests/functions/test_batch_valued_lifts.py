"""Lifted trajectory laws retain one whole batch per evaluation."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArrayBatch,
    Opaque,
    OpaqueBatch,
    OutputSpec,
    Record,
    RecordBatch,
    ResultSchemaError,
    function,
    replay_run,
    sample,
    workflow_run,
)


def _trajectory(beta):
    return NumericArrayBatch(beta * jnp.arange(1.0, 4.0), "time", label="path")


class TestBatchValuedLifts:
    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "thread"])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_exact_lifts_keep_whole_events_and_weighted_inputs(self, dispatch, include_inputs):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], component="beta")
        trajectory = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            output_label="display alias",
            dispatch=dispatch,
            include_inputs=include_inputs,
        )
        with workflow_run(seed=0):
            law = trajectory(source)
            draws = sample(law, 6)
        assert law.num_atoms == 2
        np.testing.assert_allclose(law.weights, [0.25, 0.75])
        assert law.atoms.level_names[0] == "trajectory"
        if include_inputs:
            assert set(law.event_spec.components) == {"beta", "trajectory"}
            paths = law.atoms["trajectory"]
            assert paths.level_names == ("trajectory", "time")
            for row in draws:
                np.testing.assert_array_equal(
                    row["trajectory"].raw(), row["beta"] * jnp.arange(1.0, 4.0)
                )
        else:
            assert isinstance(law.event_spec.spec, BatchSpec)
            assert law.event_spec.spec.level_names == ("time",)
            assert draws.level_names == ("sample", "time")
            np.testing.assert_array_equal(law.atoms.raw(), [[2, 4, 6], [5, 10, 15]])

    @pytest.mark.parametrize("sampled", [False, True])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_a_second_lift_receives_the_batch_kind(self, sampled, include_inputs):
        trajectory = Function(_trajectory, output_spec=OutputSpec(trajectory=None))

        @function(dispatch="auto", n_broadcast_samples=5, include_inputs=include_inputs)
        def total(path: NumericArrayBatch):
            assert isinstance(path, NumericArrayBatch)
            return jnp.sum(path.raw())

        source = (
            Normal("beta", 0.0, 1.0)
            if sampled
            else EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], component="beta")
        )
        with workflow_run(seed=2):
            first = trajectory(source)
            second = total(first)
        assert second.num_atoms == (5 if sampled else 2)
        assert second.atoms.level_names == ("total",)
        if not sampled:
            values = second.atoms.raw("total") if include_inputs else second.atoms.raw()
            np.testing.assert_array_equal(values, [12.0, 30.0])
            np.testing.assert_allclose(second.weights, [0.25, 0.75])

    def test_a_component_collision_requires_a_distinct_output_component(self):
        source = EmpiricalDistribution(jnp.array([2.0]), component="beta")
        bad = Function(_trajectory, output_spec=OutputSpec(time=None), dispatch="sequential")
        with workflow_run(seed=0), pytest.raises(ResultSchemaError, match="level"):
            bad(source)

    def test_non_batch_outputs_use_their_component_as_the_atom_level(self):
        wrapped = Function(lambda x: x + 1, output_spec=OutputSpec(score=None), label="alias")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), component="beta"))
        assert law.atoms.level_names == ("score",)


class TestBatchOutputComposition:
    def test_record_batch_outputs_keep_their_fields_in_joint_draws(self):
        def trajectory(beta):
            return RecordBatch.stack(
                [Record.from_fields(x=beta, y=beta * k) for k in (1.0, 2.0)], level_name="time"
            )

        wrapped = Function(trajectory, include_inputs=True, dispatch="sequential")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), component="beta"))
            draws = sample(law, 3)
        paths = law.atoms["trajectory"]
        assert paths.level_names == ("trajectory", "time")
        for row in draws:
            path = row["trajectory"]
            np.testing.assert_array_equal(path.raw("x"), jnp.repeat(row["beta"], 2))
            np.testing.assert_array_equal(path.raw("y"), row["beta"] * jnp.array([1.0, 2.0]))
        assert law.atoms[0:0]["trajectory"].batch_shape == (0, 2)

    def test_replay_preserves_batch_events_and_atom_level_under_a_display_alias(self):
        wrapped = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            dispatch="sequential",
            n_broadcast_samples=5,
        )
        source = Normal("beta", 0.0, 1.0)
        with workflow_run(seed=4):
            first = wrapped(source)
        with replay_run(first.provenance):
            second = wrapped.with_label("renamed display")(source)
        np.testing.assert_array_equal(first.atoms.raw(), second.atoms.raw())
        assert second.atoms.level_names == ("trajectory", "time")


class TestObjectBatchEvents:
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_object_batch_events_remain_whole_and_frozen(self, include_inputs):
        def payload(x):
            return OpaqueBatch(
                [Opaque((int(x), "a"), label="first"), Opaque((int(x), "b"), label="second")],
                "item",
                label="payload",
            )

        wrapped = Function(payload, dispatch="sequential", include_inputs=include_inputs)
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), component="x"))
            drawn = sample(law)
        assert law.num_atoms == 1
        if include_inputs:
            assert isinstance(drawn["payload"], OpaqueBatch)
            field = law.atoms["payload"]
            assert field.level_names == ("payload", "item")
            assert not field.raw().flags.writeable
        else:
            assert isinstance(drawn, OpaqueBatch)
            assert drawn.level_names == ("item",)
            assert drawn[0].raw() == (2, "a")
