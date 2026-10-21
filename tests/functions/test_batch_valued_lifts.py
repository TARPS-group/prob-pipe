"""Lifted trajectory laws retain one whole batch per evaluation."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    Batch,
    BatchSpec,
    BootstrapReplicateDistribution,
    DistributionBatch,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    Opaque,
    OpaqueBatch,
    OpaqueSpec,
    OutputSpec,
    Record,
    RecordBatch,
    ResultSchemaError,
    function,
    marginal,
    mean,
    replay_run,
    sample,
    variance,
    workflow_run,
)


def _trajectory(beta):
    return NumericArrayBatch(beta * jnp.arange(1.0, 4.0), "time", label="path")


class TestBatchValuedLifts:
    @pytest.mark.parametrize("dispatch", ["auto", "sequential", "thread"])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_exact_lifts_keep_whole_events_and_weighted_inputs(self, dispatch, include_inputs):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], output_name="beta")
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
            assert set(law.event_spec.outputs) == {"beta", "trajectory"}
            paths = law.atoms["trajectory"]
            assert paths.level_names == ("trajectory", "time")
            for row in draws:
                np.testing.assert_array_equal(
                    row["trajectory"].raw, row["beta"] * jnp.arange(1.0, 4.0)
                )
        else:
            assert isinstance(law.event_spec.term_spec, BatchSpec)
            assert law.event_spec.term_spec.level_names == ("time",)
            assert draws.level_names == ("sample", "time")
            np.testing.assert_array_equal(law.atoms.raw, [[2, 4, 6], [5, 10, 15]])

    @pytest.mark.parametrize("sampled", [False, True])
    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_a_second_lift_receives_the_batch_kind(self, sampled, include_inputs):
        trajectory = Function(_trajectory, output_spec=OutputSpec(trajectory=None))

        @function(dispatch="auto", max_evaluations=5, include_inputs=include_inputs)
        def total(path: NumericArrayBatch):
            assert isinstance(path, NumericArrayBatch)
            return jnp.sum(path.raw)

        source = (
            Normal("beta", 0.0, 1.0)
            if sampled
            else EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], output_name="beta")
        )
        with workflow_run(seed=2):
            first = trajectory(source)
            second = total(first)
        assert second.num_atoms == (5 if sampled else 2)
        assert second.atoms.level_names == ("total",)
        if not sampled:
            values = second.atoms["total"].raw if include_inputs else second.atoms.raw
            np.testing.assert_array_equal(values, [12.0, 30.0])
            np.testing.assert_allclose(second.weights, [0.25, 0.75])

    @pytest.mark.parametrize("include_inputs", [False, True])
    @pytest.mark.parametrize("declared", [False, True])
    def test_an_output_collision_requires_a_distinct_output(self, declared, include_inputs):
        def time(beta):
            return _trajectory(beta)

        source = EmpiricalDistribution(jnp.array([2.0]), output_name="beta")
        bad = Function(
            time,
            output_spec=OutputSpec(time=None) if declared else None,
            dispatch="sequential",
            include_inputs=include_inputs,
        )
        message = r"level 'time'.*\['time'\].*OutputSpec with a distinct output"
        with workflow_run(seed=0), pytest.raises(ResultSchemaError, match=message):
            bad(source)

    def test_array_atoms_on_an_event_level_name_both_levels(self):
        event = OutputSpec(trajectory=BatchSpec(NumericArraySpec(()), time=3))
        with pytest.raises(ValueError, match=r"atom level 'time'.*levels \['time'\]"):
            EmpiricalDistribution(jnp.ones((4, 3)), event_spec=event, level="time")

    def test_non_batch_outputs_use_their_output_name_as_the_atom_level(self):
        wrapped = Function(lambda x: x + 1, output_spec=OutputSpec(score=None), label="alias")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), output_name="beta"))
        assert law.atoms.level_names == ("score",)


class TestBatchValuedLiftMoments:
    def test_a_weighted_trajectory_law_has_elementwise_moments(self):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), [0.25, 0.75], output_name="beta")
        trajectory = Function(
            _trajectory, output_spec=OutputSpec(trajectory=None), dispatch="sequential"
        )
        with workflow_run(seed=0):
            law = trajectory(source)
            mu = mean(law)
            var = variance(law)
        for moment in (mu, var):
            assert isinstance(moment, NumericArrayBatch)
            assert moment.level_names == ("time",)
            assert moment.batch_shape == (3,)
        # The atoms 2·[1, 2, 3] and 5·[1, 2, 3] under the weights 0.25 and 0.75.
        np.testing.assert_allclose(mu.raw, [4.25, 8.5, 12.75], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(var.raw, [1.6875, 6.75, 15.1875], rtol=1e-6, atol=1e-6)


def _trajectory_law(include_inputs):
    """A lift of a trajectory on the level time, from six draws of beta."""
    wrapped = Function(
        _trajectory,
        output_spec=OutputSpec(trajectory=None),
        include_inputs=include_inputs,
        max_evaluations=6,
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
        paths = np.asarray(viewed.raw).reshape(-1, 3)
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
        beta = np.asarray(drawn["beta"].raw)
        assert beta.shape == sample_shape
        np.testing.assert_allclose(
            np.asarray(paths.raw), beta[..., None] * np.arange(1.0, 4.0), rtol=1e-6
        )

    @pytest.mark.parametrize("sample_shape", [(), (3,)])
    def test_record_batch_fields_draw_whole_through_every_view(self, sample_shape):
        def trajectory(beta):
            return RecordBatch.stack(
                [Record.from_fields(x=beta, y=beta * k) for k in (1.0, 2.0)], level_name="time"
            )

        wrapped = Function(trajectory, include_inputs=True, dispatch="sequential")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta"))
            single = sample(law["trajectory"], sample_shape)
            selected = sample(law[("beta", "trajectory")], sample_shape)
        levels = ("sample", "time") if sample_shape else ("time",)
        for paths in (single, selected["trajectory"]):
            assert paths.level_names == levels
            assert paths.batch_shape == (*sample_shape, 2)
        beta = np.asarray(selected["beta"].raw)
        np.testing.assert_array_equal(
            np.asarray(selected["trajectory"]["y"].raw), beta[..., None] * np.array([1.0, 2.0])
        )


def _total(trajectory: NumericArrayBatch):
    assert isinstance(trajectory, NumericArrayBatch)
    return jnp.sum(trajectory.raw)


def _gap(beta, trajectory: NumericArrayBatch):
    assert isinstance(trajectory, NumericArrayBatch)
    return jnp.sum(trajectory.raw) - 6.0 * beta


class TestLiftsOverBatchValuedFields:
    @pytest.mark.parametrize("dispatch", ["sequential", "auto"])
    @pytest.mark.parametrize("sampled", [False, True])
    def test_a_lift_over_a_field_reads_one_batch_per_draw(self, sampled, dispatch):
        source = (
            Normal("beta", 0.0, 1.0)
            if sampled
            else EmpiricalDistribution(jnp.array([2.0, 5.0]), [1.0, 3.0], output_name="beta")
        )
        joint = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            include_inputs=True,
            max_evaluations=5,
        )
        total = Function(_total, dispatch=dispatch, max_evaluations=5)
        gap = Function(_gap, dispatch=dispatch, max_evaluations=5)
        with workflow_run(seed=0):
            law = joint(source)
            totals = total(law["trajectory"])
            gaps = gap(law["beta"], law["trajectory"])
        np.testing.assert_allclose(totals.atoms.raw, 6.0 * law.atoms["beta"].raw, rtol=1e-6)
        np.testing.assert_allclose(totals.weights, law.weights)
        np.testing.assert_allclose(gaps.atoms.raw, 0.0, atol=1e-5)

    def test_a_lift_over_a_batch_field_of_a_record_output_reads_its_batches(self):
        def summary(beta):
            return Record.from_fields(scale=2.0 * beta, traj=_trajectory(beta))

        def total(traj: NumericArrayBatch):
            assert isinstance(traj, NumericArrayBatch)
            return jnp.sum(traj.raw)

        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta")
        with workflow_run(seed=0):
            law = Function(summary, dispatch="sequential")(source)
            totals = Function(total, dispatch="sequential")(law["traj"])
        assert isinstance(law.event_spec.term_spec["traj"], BatchSpec)
        np.testing.assert_allclose(totals.atoms.raw, [12.0, 30.0])


class TestSampleLevelCollision:
    @staticmethod
    def _on_sample(beta, scale=1.0):
        return NumericArrayBatch(beta * scale * jnp.arange(1.0, 4.0), "sample", label="path")

    def test_a_sample_shape_names_the_event_level_it_repeats(self):
        wrapped = Function(
            self._on_sample, output_spec=OutputSpec(trajectory=None), dispatch="sequential"
        )
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta"))
            assert sample(law).level_names == ("sample",)
            with pytest.raises(ApplicabilityError, match=r"level 'sample'.*Rename the event's"):
                sample(law, 2)

    def test_a_nested_sweep_names_the_event_level_it_repeats(self):
        wrapped = Function(
            self._on_sample, output_spec=OutputSpec(trajectory=None), dispatch="sequential"
        )
        scales = NumericArrayBatch(jnp.array([1.0, 10.0]), "horizon", label="scale")
        with workflow_run(seed=0):
            laws = wrapped(EmpiricalDistribution(jnp.array([2.0]), output_name="beta"), scales)
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
            law = wrapped(EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta"))
            draws = sample(law, 3)
        paths = law.atoms["trajectory"]
        assert paths.level_names == ("trajectory", "time")
        for row in draws:
            path = row["trajectory"]
            np.testing.assert_array_equal(path["x"].raw, jnp.repeat(row["beta"], 2))
            np.testing.assert_array_equal(path["y"].raw, row["beta"] * jnp.array([1.0, 2.0]))
        assert law.atoms[0:0]["trajectory"].batch_shape == (0, 2)

    def test_replay_preserves_batch_events_and_atom_level_under_a_display_alias(self):
        wrapped = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            dispatch="sequential",
            max_evaluations=5,
        )
        source = Normal("beta", 0.0, 1.0)
        with workflow_run(seed=4):
            first = wrapped(source)
        with replay_run(first.provenance):
            second = wrapped.with_label("renamed display")(source)
        np.testing.assert_array_equal(first.atoms.raw, second.atoms.raw)
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
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), output_name="x"))
            drawn = sample(law)
        assert law.num_atoms == 1
        if include_inputs:
            assert isinstance(drawn["payload"], OpaqueBatch)
            field = law.atoms["payload"]
            assert field.level_names == ("payload", "item")
            assert not field.raw.flags.writeable
        else:
            assert isinstance(drawn, OpaqueBatch)
            assert drawn.level_names == ("item",)
            assert drawn[0].raw == (2, "a")


class TestBatchOutputDeclarations:
    def test_a_polymorphic_declaration_completes_without_mutating_the_function(self):
        declared = OutputSpec(trajectory=BatchSpec(NumericArraySpec(()), time="n"))
        wrapped = Function(_trajectory, output_spec=declared, dispatch="sequential")
        with workflow_run(seed=0):
            law = wrapped(EmpiricalDistribution(jnp.array([2.0]), output_name="beta"))
        assert law.event_spec.term_spec.axis_groups == ((3,),)
        assert law.event_spec.term_spec.element_spec.dtype == jnp.float32
        assert wrapped.output_spec is declared
        assert wrapped.output_spec.term_spec.axis_groups == (("n",),)
        assert wrapped.output_spec.term_spec.element_spec.dtype is None

    @pytest.mark.parametrize("element", [NumericArraySpec((), jnp.float32), OpaqueSpec()])
    def test_an_empty_sweep_of_a_batch_output_has_no_batch_form(self, element):
        declared = OutputSpec(trajectory=BatchSpec(element, time=3))
        wrapped = Function(_trajectory, output_spec=declared, dispatch="sequential")
        with pytest.raises(ValueError, match="An empty sweep has no batch form for BatchSpec"):
            wrapped(NumericArrayBatch(jnp.zeros((0,)), "beta", label="beta"))

    @pytest.mark.parametrize("difference", ["shape", "level"])
    def test_all_evaluations_must_agree_on_one_event(self, difference):
        def trajectory(x):
            size = 2 + int(x) if difference == "shape" else 2
            level = "time" if difference == "shape" or int(x) == 0 else "other"
            return NumericArrayBatch(jnp.ones(size), level, label="path")

        wrapped = Function(trajectory, dispatch="sequential")
        source = EmpiricalDistribution(jnp.array([0.0, 1.0]), output_name="x")
        with workflow_run(seed=0), pytest.raises(ResultSchemaError):
            wrapped(source)


class TestBatchValuedDrawsArriveRaw:
    """A lifted body receives a batch-valued draw raw, unless its annotation names a batch class."""

    @staticmethod
    def _received(body, law, dispatch="sequential"):
        """The draws *body* received over *law*, and the lifted law."""
        seen = []

        def record(value):
            seen.append(value)
            return body(value)

        with workflow_run(seed=0):
            lifted = Function(record, dispatch=dispatch, max_evaluations=6)(law)
        return seen, lifted

    def test_a_bootstrap_replicate_arrives_as_an_array(self):
        source = EmpiricalDistribution(jnp.array([1.0, 5.0, 9.0]), output_name="y")
        replicate = BootstrapReplicateDistribution("b", source, 4)
        seen, means = self._received(lambda b: b.mean(), replicate)
        assert all(isinstance(b, jax.Array) and b.shape == (4,) for b in seen)
        drawn = np.asarray([np.asarray(b) for b in seen])
        np.testing.assert_allclose(means.atoms.raw, drawn.mean(axis=1), rtol=1e-6)

    @pytest.mark.parametrize("include_inputs", [False, True])
    def test_an_unannotated_parameter_receives_the_raw_trajectory(self, include_inputs):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), [0.25, 0.75], output_name="beta")
        trajectory = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            include_inputs=include_inputs,
            dispatch="sequential",
        )
        with workflow_run(seed=0):
            law = trajectory(source)
        seen, totals = self._received(lambda path: jnp.sum(path), law["trajectory"])
        assert all(isinstance(path, jax.Array) and path.shape == (3,) for path in seen)
        np.testing.assert_allclose(totals.atoms.raw, [12.0, 30.0], rtol=1e-6)

    @pytest.mark.parametrize("annotation", [NumericArrayBatch, Batch, NumericArrayBatch | None])
    def test_a_batch_annotation_receives_the_batch(self, annotation):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), [0.25, 0.75], output_name="beta")
        trajectory = Function(_trajectory, output_spec=OutputSpec(trajectory=None))
        seen = []

        def total(path):
            seen.append(path)
            return jnp.sum(path.raw)

        total.__annotations__["path"] = annotation
        with workflow_run(seed=0):
            totals = Function(total, dispatch="sequential")(trajectory(source))
        assert all(isinstance(path, NumericArrayBatch) for path in seen)
        assert all(path.level_names == ("time",) for path in seen)
        np.testing.assert_allclose(totals.atoms.raw, [12.0, 30.0], rtol=1e-6)

    def test_an_array_annotation_receives_the_raw_trajectory(self):
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta")
        trajectory = Function(_trajectory, output_spec=OutputSpec(trajectory=None))
        seen = []

        def total(path: jax.Array):
            seen.append(path)
            return jnp.sum(path)

        with workflow_run(seed=0):
            Function(total, dispatch="sequential")(trajectory(source))
        assert all(isinstance(path, jax.Array) and path.shape == (3,) for path in seen)

    @pytest.mark.parametrize("annotated", [False, True])
    def test_a_batch_of_records_arrives_as_its_columns_or_as_the_batch(self, annotated):
        def trajectory(beta):
            return RecordBatch.stack(
                [Record.from_fields(x=beta, y=beta * k) for k in (1.0, 2.0)], level_name="time"
            )

        seen = []

        def total(path):
            seen.append(path)
            columns = path.raw if annotated else path
            return jnp.sum(columns["y"])

        if annotated:
            total.__annotations__["path"] = RecordBatch
        source = EmpiricalDistribution(jnp.array([2.0, 5.0]), output_name="beta")
        with workflow_run(seed=0):
            law = Function(trajectory, dispatch="sequential")(source)
            totals = Function(total, dispatch="sequential")(law)
        for path in seen:
            if annotated:
                assert isinstance(path, RecordBatch)
                assert path.level_names == ("time",)
            else:
                assert isinstance(path, Mapping) and set(path) == {"x", "y"}
                assert np.shape(path["y"]) == (2,)
        np.testing.assert_allclose(totals.atoms.raw, [6.0, 15.0], rtol=1e-6)


class TestNestedBatchValuedInputs:
    @pytest.mark.parametrize("dispatch", ["auto", "jax", "sequential", "thread"])
    @pytest.mark.parametrize("annotated", [False, True])
    def test_joint_lifts_keep_a_whole_bootstrap_replicate(self, dispatch, annotated):
        source = EmpiricalDistribution(
            jnp.array([[2.0, 4.0, 6.0]]),
            event_spec=OutputSpec(path=BatchSpec(NumericArraySpec(()), time=3)),
        )
        replicate = BootstrapReplicateDistribution("rep", source, 2, level="replicate")

        def total(p):
            if annotated:
                assert isinstance(p, NumericArrayBatch)
                assert p.level_names == ("replicate", "time")
                p = p.raw
            return jnp.sum(p)

        if annotated:
            total.__annotations__["p"] = NumericArrayBatch
        wrapped = Function(total, dispatch=dispatch, include_inputs=True, max_evaluations=5)
        with workflow_run(seed=0):
            law = wrapped(replicate)
            drawn = sample(law)

        assert law.event_spec.term_spec["p"] == BatchSpec(
            NumericArraySpec((), jnp.float32), replicate=2, time=3
        )
        assert law.atoms["p"].level_names == ("total", "replicate", "time")
        assert drawn["p"].level_names == ("replicate", "time")
        np.testing.assert_array_equal(drawn["p"].raw, [[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]])
        np.testing.assert_array_equal(law.atoms["total"].raw, jnp.full((5,), 24.0))


def _record_trajectory(beta):
    return RecordBatch.stack(
        [Record.from_fields(x=beta, y=beta * k) for k in (1.0, 2.0)], level_name="time"
    )


def _assert_same(mapped, rowwise):
    """Assert two stored atom forms hold the same values, batch entries included."""
    if isinstance(mapped, Batch):
        assert mapped.level_names == rowwise.level_names
        _assert_same(mapped.raw, rowwise.raw)
    elif isinstance(mapped, Mapping):
        assert set(mapped) == set(rowwise)
        for key in mapped:
            _assert_same(mapped[key], rowwise[key])
    elif isinstance(mapped, np.ndarray) and mapped.dtype == object:
        assert mapped.shape == rowwise.shape
        for left, right in zip(mapped.flat, rowwise.flat, strict=True):
            _assert_same(left, right)
    else:
        np.testing.assert_array_equal(np.asarray(mapped), np.asarray(rowwise))


def _sources():
    return {
        "sampled": Normal("beta", 0.0, 1.0),
        "enumerated": EmpiricalDistribution(
            jnp.array([2.0, 5.0]), [0.25, 0.75], output_name="beta"
        ),
    }


class TestMappedBatchValuedLifts:
    """``auto`` and ``jax`` map a body over batch-valued outputs and draws, as row-wise runs do."""

    @staticmethod
    def _lift(body, law, dispatch, **controls):
        with workflow_run(seed=3):
            return Function(body, dispatch=dispatch, max_evaluations=7, **controls)(law)

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    @pytest.mark.parametrize("include_inputs", [False, True])
    @pytest.mark.parametrize("body", [_trajectory, _record_trajectory])
    @pytest.mark.parametrize("source", ["sampled", "enumerated"])
    def test_a_batch_returning_body_maps_as_the_row_wise_lift_runs(
        self, dispatch, include_inputs, body, source
    ):
        law = _sources()[source]
        mapped = self._lift(body, law, dispatch, include_inputs=include_inputs)
        rowwise = self._lift(body, law, "sequential", include_inputs=include_inputs)
        assert mapped.provenance.metadata["dispatch"] == "jax"
        assert mapped.event_spec == rowwise.event_spec
        _assert_same(mapped.atoms, rowwise.atoms)
        np.testing.assert_array_equal(mapped.weights, rowwise.weights)

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    @pytest.mark.parametrize("include_inputs", [False, True])
    @pytest.mark.parametrize("annotated", [False, True])
    @pytest.mark.parametrize("source", ["sampled", "enumerated"])
    def test_a_second_lift_maps_as_the_row_wise_lift_runs(
        self, dispatch, include_inputs, annotated, source
    ):
        def total(trajectory):
            values = trajectory.raw if annotated else trajectory
            return jnp.sum(values) - 6.0

        if annotated:
            total.__annotations__["trajectory"] = NumericArrayBatch
        first = Function(_trajectory, output_spec=OutputSpec(trajectory=None))
        with workflow_run(seed=1):
            law = first(_sources()[source])
        mapped = self._lift(total, law, dispatch, include_inputs=include_inputs)
        rowwise = self._lift(total, law, "sequential", include_inputs=include_inputs)
        assert mapped.provenance.metadata["dispatch"] == "jax"
        _assert_same(mapped.atoms, rowwise.atoms)
        np.testing.assert_array_equal(mapped.weights, rowwise.weights)

    def test_the_mapped_lift_traces_the_body_instead_of_calling_it_per_draw(self):
        calls = []

        def trajectory(beta):
            calls.append("first")
            return _trajectory(beta)

        def total(path):
            calls.append("second")
            return jnp.sum(path)

        source = Normal("beta", 0.0, 1.0)
        for dispatch in ("jax", "sequential"):
            calls.clear()
            with workflow_run(seed=0):
                law = Function(trajectory, dispatch=dispatch, max_evaluations=50)(source)
                totals = Function(total, dispatch=dispatch, max_evaluations=50)(law)
            if dispatch == "jax":
                # One probe trace and one mapped trace for each lift.
                assert calls.count("first") <= 2 and calls.count("second") <= 2
                assert law.provenance.metadata["dispatch"] == "jax"
                assert totals.provenance.metadata["dispatch"] == "jax"
            else:
                assert calls.count("first") == 50 and calls.count("second") == 50

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_a_joint_field_that_stores_batches_is_not_mapped(self, dispatch):
        joint = Function(
            _trajectory,
            output_spec=OutputSpec(trajectory=None),
            include_inputs=True,
            max_evaluations=5,
        )
        total = Function(lambda path: jnp.sum(path), dispatch=dispatch, max_evaluations=5)
        with workflow_run(seed=0):
            law = joint(Normal("beta", 0.0, 1.0))
            if dispatch == "jax":
                message = (
                    r"dispatch='jax' cannot map over the draws of '_trajectory', whose field "
                    r"'trajectory' stores a batch on the levels \['time'\].*"
                    r"marginal\(law, 'trajectory'\)"
                )
                with pytest.raises(ValueError, match=message):
                    total(law["trajectory"])
            else:
                assert total(law["trajectory"]).provenance.metadata["dispatch"] == "sequential"
                mapped = total(marginal(law, "trajectory"))
                assert mapped.provenance.metadata["dispatch"] == "jax"

    @pytest.mark.parametrize("dispatch", ["auto", "jax"])
    def test_a_returned_record_that_holds_a_batch_is_not_mapped(self, dispatch):
        def summary(beta):
            return Record.from_fields(scale=2.0 * beta, traj=_trajectory(beta))

        wrapped = Function(summary, dispatch=dispatch, max_evaluations=5)
        with workflow_run(seed=0):
            if dispatch == "jax":
                message = (
                    r"dispatch='jax' cannot stack a returned record whose field 'traj' is a "
                    r"batch on the levels \['time'\]"
                )
                with pytest.raises(ValueError, match=message):
                    wrapped(Normal("beta", 0.0, 1.0))
            else:
                law = wrapped(Normal("beta", 0.0, 1.0))
                assert law.provenance.metadata["dispatch"] == "sequential"
