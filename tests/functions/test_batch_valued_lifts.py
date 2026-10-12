"""Lifted trajectory laws retain one whole batch per evaluation."""

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    BatchSpec,
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    Opaque,
    OpaqueBatch,
    OutputSpec,
    Record,
    RecordBatch,
    ResultSchemaError,
    function,
    marginal,
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

    @pytest.mark.parametrize("include_inputs", [False, True])
    @pytest.mark.parametrize("declared", [False, True])
    def test_a_component_collision_requires_a_distinct_output_component(
        self, declared, include_inputs
    ):
        def time(beta):
            return _trajectory(beta)

        source = EmpiricalDistribution(jnp.array([2.0]), component="beta")
        bad = Function(
            time,
            output_spec=OutputSpec(time=None) if declared else None,
            dispatch="sequential",
            include_inputs=include_inputs,
        )
        message = r"level 'time'.*\['time'\].*OutputSpec with a distinct output component"
        with workflow_run(seed=0), pytest.raises(ResultSchemaError, match=message):
            bad(source)

    def test_array_atoms_on_an_event_level_name_both_levels(self):
        event = OutputSpec(trajectory=BatchSpec(NumericArraySpec(()), time=3))
        with pytest.raises(ValueError, match=r"atom level 'time'.*levels \['time'\]"):
            EmpiricalDistribution(jnp.ones((4, 3)), event_spec=event, level="time")

    def test_non_batch_outputs_use_their_component_as_the_atom_level(self):
        wrapped = Function(lambda x: x + 1, output_spec=OutputSpec(score=None), label="alias")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), component="beta"))
        assert law.atoms.level_names == ("score",)


def _trajectory_law(include_inputs):
    """A lift of a trajectory on the level time, from six draws of beta."""
    wrapped = Function(
        _trajectory,
        output_spec=OutputSpec(trajectory=None),
        include_inputs=include_inputs,
        n_broadcast_samples=6,
    )
    return wrapped(Normal("beta", 0.0, 1.0))


class TestBatchValuedFieldSampling:
    @pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_a_field_view_draws_what_the_marginal_draws(self, include_inputs, sample_shape):
        with workflow_run(seed=0):
            law = _trajectory_law(include_inputs)
            viewed = sample(law["trajectory"], sample_shape)
            expected = sample(marginal(law, "trajectory"), sample_shape)
        assert isinstance(viewed, NumericArrayBatch)
        assert viewed.spec == expected.spec
        levels = ("sample", "time") if sample_shape else ("time",)
        assert viewed.level_names == levels
        assert viewed.batch_shape == (*sample_shape, 3)
        paths = np.asarray(viewed.raw()).reshape(-1, 3)
        np.testing.assert_allclose(paths, paths[:, :1] * np.arange(1.0, 4.0), rtol=1e-6)

    @pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_a_selection_draws_what_the_marginal_draws(self, include_inputs, sample_shape):
        with workflow_run(seed=0):
            law = _trajectory_law(include_inputs)
            viewed = sample(law[("trajectory",)], sample_shape)
            expected = sample(marginal(law, ("trajectory",)), sample_shape)
        assert viewed.spec == expected.spec
        paths = viewed["trajectory"]
        assert isinstance(paths, NumericArrayBatch)
        levels = ("sample", "time") if sample_shape else ("time",)
        assert paths.level_names == levels
        assert paths.batch_shape == (*sample_shape, 3)

    @pytest.mark.parametrize("sample_shape", [(), (4,), (2, 3)])
    def test_joint_draws_pair_each_trajectory_with_its_beta(self, sample_shape):
        with workflow_run(seed=0):
            law = _trajectory_law(include_inputs=True)
            drawn = sample(law[("beta", "trajectory")], sample_shape)
        paths = drawn["trajectory"]
        assert paths.level_names == (("sample", "time") if sample_shape else ("time",))
        assert paths.batch_shape == (*sample_shape, 3)
        beta = np.asarray(drawn["beta"].raw())
        assert beta.shape == sample_shape
        np.testing.assert_allclose(
            np.asarray(paths.raw()), beta[..., None] * np.arange(1.0, 4.0), rtol=1e-6
        )

    @pytest.mark.parametrize("sample_shape", [(), (3,)])
    def test_record_batch_fields_draw_whole_through_every_view(self, sample_shape):
        def trajectory(beta):
            return RecordBatch.stack(
                [Record.from_fields(x=beta, y=beta * k) for k in (1.0, 2.0)], level_name="time"
            )

        wrapped = Function(trajectory, include_inputs=True, dispatch="sequential")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), component="beta"))
            single = sample(law["trajectory"], sample_shape)
            selected = sample(law[("beta", "trajectory")], sample_shape)
        levels = ("sample", "time") if sample_shape else ("time",)
        for paths in (single, selected["trajectory"]):
            assert paths.level_names == levels
            assert paths.batch_shape == (*sample_shape, 2)
        beta = np.asarray(selected["beta"].raw())
        np.testing.assert_array_equal(
            np.asarray(selected["trajectory"].raw("y")), beta[..., None] * np.array([1.0, 2.0])
        )


def _total(trajectory: NumericArrayBatch):
    assert isinstance(trajectory, NumericArrayBatch)
    return jnp.sum(trajectory.raw())


def _gap(beta, trajectory: NumericArrayBatch):
    assert isinstance(trajectory, NumericArrayBatch)
    return jnp.sum(trajectory.raw()) - 6.0 * beta


class TestLiftsOverBatchValuedFields:
    @pytest.mark.parametrize("dispatch", ["sequential", "auto"])
    @pytest.mark.parametrize("sampled", [False, True])
    def test_a_lift_over_a_field_reads_one_batch_per_draw(self, sampled, dispatch):
        source = (
            Normal("beta", 0.0, 1.0)
            if sampled
            else EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], component="beta")
        )
        joint = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            include_inputs=True,
            n_broadcast_samples=5,
        )
        total = Function(_total, dispatch=dispatch, n_broadcast_samples=5)
        gap = Function(_gap, dispatch=dispatch, n_broadcast_samples=5)
        with workflow_run(seed=0):
            law = joint(source)
            totals = total(law["trajectory"])
            gaps = gap(law["beta"], law["trajectory"])
        np.testing.assert_allclose(totals.atoms.raw(), 6.0 * law.atoms.raw("beta"), rtol=1e-6)
        np.testing.assert_allclose(totals.weights, law.weights)
        np.testing.assert_allclose(gaps.atoms.raw(), 0.0, atol=1e-5)

    def test_a_lift_over_a_batch_field_of_a_record_output_reads_its_batches(self):
        def summary(beta):
            return Record.from_fields(scale=2.0 * beta, traj=_trajectory(beta))

        def total(traj: NumericArrayBatch):
            assert isinstance(traj, NumericArrayBatch)
            return jnp.sum(traj.raw())

        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), component="beta")
        with workflow_run(seed=0):
            law = Function(summary, dispatch="sequential")(source)
            totals = Function(total, dispatch="sequential")(law["traj"])
        assert isinstance(law.event_spec.spec["traj"], BatchSpec)
        np.testing.assert_allclose(totals.atoms.raw(), [12.0, 30.0])


class TestSampleLevelCollision:
    @staticmethod
    def _on_sample(beta, scale=1.0):
        return NumericArrayBatch(beta * scale * jnp.arange(1.0, 4.0), "sample", label="path")

    def test_a_sample_shape_names_the_event_level_it_repeats(self):
        wrapped = Function(
            self._on_sample, output_spec=OutputSpec(trajectory=None), dispatch="sequential"
        )
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), component="beta"))
            assert sample(law).level_names == ("sample",)
            with pytest.raises(ApplicabilityError, match=r"level 'sample'.*Rename the event's"):
                sample(law, 2)

    def test_a_nested_sweep_names_the_event_level_it_repeats(self):
        wrapped = Function(
            self._on_sample, output_spec=OutputSpec(trajectory=None), dispatch="sequential"
        )
        scales = NumericArrayBatch(jnp.array([1.0, 10.0]), "horizon", label="scale")
        with workflow_run(seed=0):
            laws = wrapped(EmpiricalDistribution(jnp.array([2.0]), component="beta"), scales)
            assert isinstance(laws, DistributionBatch)
            assert sample(laws).level_names == ("horizon", "sample")
            with pytest.raises(ApplicabilityError, match=r"level 'sample'.*Rename the event's"):
                sample(laws, 2)


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


class TestBatchOutputDeclarations:
    def test_a_polymorphic_declaration_completes_without_mutating_the_function(self):
        declared = OutputSpec(trajectory=BatchSpec(NumericArraySpec(()), time="n"))
        wrapped = Function(_trajectory, output_spec=declared, dispatch="sequential")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), component="beta"))
        assert law.event_spec.spec.axis_groups == ((3,),)
        assert law.event_spec.spec.element_spec.dtype == jnp.float32
        assert wrapped.output_spec is declared
        assert wrapped.output_spec.spec.axis_groups == (("n",),)
        assert wrapped.output_spec.spec.element_spec.dtype is None

    @pytest.mark.parametrize("difference", ["shape", "level"])
    def test_all_evaluations_must_agree_on_one_event(self, difference):
        def trajectory(x):
            size = 2 + int(x) if difference == "shape" else 2
            level = "time" if difference == "shape" or int(x) == 0 else "other"
            return NumericArrayBatch(jnp.ones(size), level, label="path")

        wrapped = Function(trajectory, dispatch="sequential")
        source = EmpiricalDistribution(jnp.array([0.0, 1.0]), component="x")
        with workflow_run(seed=0), pytest.raises(ResultSchemaError):
            wrapped(source)
