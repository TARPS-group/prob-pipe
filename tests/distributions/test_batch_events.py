"""Empirical atoms that are whole batches retain their event multiplicity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    BootstrapReplicateDistribution,
    EmpiricalDistribution,
    Function,
    Normal,
    NumericArray,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
    OutputSpec,
    RecordBatch,
    RecordSpec,
    cov,
    mean,
    quantile,
    sample,
    variance,
    workflow_run,
)
from probpipe.core.constraints import non_negative
from probpipe.distributions._capabilities import (
    SupportsCovariance,
    SupportsMean,
    SupportsQuantile,
    SupportsVariance,
)


class TestBatchValuedAtoms:
    @pytest.mark.parametrize("raw", [False, True])
    @pytest.mark.parametrize("event_shape", [(3,), (2, 3), (0,)])
    def test_weights_and_samples_treat_a_whole_batch_as_one_atom(self, raw, event_shape):
        values = jnp.stack([jnp.full(event_shape, x) for x in (1.0, 7.0)])
        event = BatchSpec(NumericArraySpec(()), time=event_shape)
        atoms = (
            values
            if raw
            else NumericArrayBatch(
                values,
                ("trajectory", "time"),
                element_spec=NumericArraySpec(()),
                axes_per_level=(1, len(event_shape)),
                label="trajectories",
            )
        )
        law = EmpiricalDistribution(atoms, [0.0, 1.0], event_spec=OutputSpec(trajectory=event))

        assert law.num_atoms == 2
        np.testing.assert_array_equal(law.weights, [0.0, 1.0])
        for protocol in (SupportsMean, SupportsVariance, SupportsCovariance, SupportsQuantile):
            assert isinstance(law, protocol)
        np.testing.assert_array_equal(law._sample(jax.random.key(0)), values[1])
        with workflow_run(seed=3):
            single = sample(law)
            many = sample(law, (2, 4))
        assert single.level_names == ("time",)
        assert single.batch_shape == event_shape
        assert many.level_names == ("sample", "time")
        assert many.axis_groups == ((2, 4), event_shape)
        np.testing.assert_array_equal(many.raw, jnp.full((2, 4, *event_shape), 7.0))

    def test_exact_expectation_preserves_the_event_axes(self):
        atoms = NumericArrayBatch(
            jnp.array([[1.0, 2.0], [10.0, 20.0]]), ("trajectory", "time"), label="paths"
        )
        law = EmpiricalDistribution(
            atoms, [1.0, 3.0], event_spec=OutputSpec(path=BatchSpec(NumericArraySpec(()), time=2))
        )
        np.testing.assert_allclose(law._expectation(lambda x: x), [7.75, 15.5])
        assert law._marginal("path").num_atoms == 2

    def test_record_batch_atoms_sample_paired_columns(self):
        values = jnp.arange(6.0).reshape(2, 3)
        atoms = RecordBatch(
            {"x": values, "y": 10 * values},
            ("trajectory", "time"),
            element_spec=RecordSpec(x=(), y=()),
            label="paths",
        )
        event = BatchSpec(RecordSpec(x=(), y=()), time=3)
        law = EmpiricalDistribution(atoms, [0.0, 1.0], event_spec=OutputSpec(path=event))
        with workflow_run(seed=0):
            one = sample(law)
            many = sample(law, 4)
        assert one.level_names == ("time",)
        assert many.level_names == ("sample", "time")
        np.testing.assert_array_equal(one["x"].raw, values[1])
        np.testing.assert_array_equal(many["y"].raw, 10 * many["x"].raw)

    @pytest.mark.parametrize("levels", [("time",), ("trajectory", "other")])
    def test_the_declared_event_must_be_a_trailing_suffix(self, levels):
        values = jnp.zeros((2,) * len(levels))
        atoms = NumericArrayBatch(values, levels, label="paths")
        event = BatchSpec(NumericArraySpec(()), time=2)
        with pytest.raises(ValueError):
            EmpiricalDistribution(atoms, event_spec=OutputSpec(path=event))

    def test_weights_cannot_count_each_time_point_as_an_atom(self):
        atoms = NumericArrayBatch(jnp.zeros((2, 3)), ("trajectory", "time"), label="paths")
        event = BatchSpec(NumericArraySpec(()), time=3)
        with pytest.raises(ValueError):
            EmpiricalDistribution(atoms, jnp.ones((2, 3)), event_spec=OutputSpec(path=event))


class TestMultipleAtomLevels:
    def test_only_preceding_levels_index_the_weights(self):
        values = jnp.arange(12.0).reshape(2, 2, 3)
        atoms = NumericArrayBatch(values, ("chain", "trajectory", "time"), label="paths")
        event = BatchSpec(NumericArraySpec(()), time=3)
        law = EmpiricalDistribution(
            atoms, jnp.array([[1.0, 2.0], [3.0, 4.0]]), event_spec=OutputSpec(path=event)
        )
        assert law.num_atoms == 4
        np.testing.assert_allclose(law.weights, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(law._atoms_at(2), values[1, 0])
        selected = law._marginal(("path",))
        assert selected.num_atoms == 4
        assert selected.atoms.level_names == ("chain", "trajectory")
        assert selected.atoms["path"].level_names == ("chain", "trajectory", "time")

    def test_an_empty_atom_axis_is_refused_even_with_an_empty_event(self):
        atoms = NumericArrayBatch(jnp.empty((0, 0)), ("trajectory", "time"), label="paths")
        with pytest.raises(ValueError, match="at least one atom"):
            EmpiricalDistribution(
                atoms, event_spec=OutputSpec(path=BatchSpec(NumericArraySpec(()), time=0))
            )


class TestNestedBatchEvents:
    @pytest.mark.parametrize("raw", [False, True])
    def test_nested_event_levels_flatten_before_counting_atoms(self, raw):
        nested = BatchSpec(BatchSpec(NumericArraySpec(()), time=(2, 3)), replicate=2)
        declaration = OutputSpec(path=nested)
        values = jnp.stack([jnp.ones((2, 2, 3)), jnp.full((2, 2, 3), 5.0)])
        atoms = (
            values
            if raw
            else NumericArrayBatch(
                values, ("path", "replicate", "time"), axes_per_level=(1, 1, 2), label="paths"
            )
        )
        law = EmpiricalDistribution(atoms, [0.25, 0.75], event_spec=declaration)

        assert law.num_atoms == 2
        assert law.event_spec.term_spec == BatchSpec(
            NumericArraySpec((), jnp.float32), replicate=2, time=(2, 3)
        )
        assert declaration.term_spec is nested
        assert isinstance(nested.element_spec, BatchSpec)
        with workflow_run(seed=0):
            drawn = sample(law, 4)
            mu = mean(law)
        assert drawn.axis_groups == ((4,), (2,), (2, 3))
        assert drawn.level_names == ("sample", "replicate", "time")
        assert mu.level_names == ("replicate", "time")
        # Weight normalization gives a float32 error of 4.8e-7 at the analytic mean 4.
        np.testing.assert_allclose(mu.raw, jnp.full((2, 2, 3), 4.0), rtol=0, atol=1e-6)

    def test_repeated_nested_level_names_are_refused(self):
        nested = BatchSpec(BatchSpec(NumericArraySpec(()), time=3), time=2)
        with pytest.raises(ValueError, match="level 'time' would appear twice"):
            EmpiricalDistribution(jnp.ones((2, 2, 3)), event_spec=OutputSpec(path=nested))

    @pytest.mark.parametrize("moment", ["mean", "variance", "cov", "quantile", "quantiles"])
    def test_bootstrap_replicate_moments_keep_the_nested_event_axes(self, moment):
        path = jnp.array([2.0, 4.0, 6.0])
        source = EmpiricalDistribution(
            path[None, :], event_spec=OutputSpec(path=BatchSpec(NumericArraySpec(()), time=3))
        )
        replicate = BootstrapReplicateDistribution("rep", source, 2, level="replicate")
        operations = {
            "mean": mean,
            "variance": variance,
            "cov": cov,
            "quantile": lambda d: quantile(d, 0.5),
            "quantiles": lambda d: quantile(d, jnp.array([0.25, 0.75])),
        }
        with workflow_run(seed=0):
            result = operations[moment](replicate)

        # The source has one atom, so every replicate consists of two copies of path.
        expected = jnp.broadcast_to(path, (2, 3))
        if moment == "cov":
            np.testing.assert_array_equal(result.raw, jnp.zeros((6, 6)))
        else:
            if moment == "variance":
                expected = jnp.zeros((2, 3))
            if moment == "quantiles":
                expected = jnp.broadcast_to(expected, (2, 2, 3))
            assert result.level_names == (
                ("quantile", "replicate", "time")
                if moment == "quantiles"
                else ("replicate", "time")
            )
            np.testing.assert_array_equal(result.raw, expected)


#: The weights and the atoms 2·[1, 2, 3] and 5·[1, 2, 3] of a weighted trajectory law.
_WEIGHTS = jnp.array([0.25, 0.75])
_STEPS = jnp.arange(1.0, 4.0)
_ATOMS = jnp.stack([2.0 * _STEPS, 5.0 * _STEPS])
#: The weighted mean and variance of the scale 2 or 5: 0.25·2 + 0.75·5 and 0.25·4 + 0.75·25 − 4.25².
_SCALE_MEAN, _SCALE_VARIANCE = 4.25, 1.6875


def _trajectory_law():
    """The weighted law of the trajectories 2·[1, 2, 3] and 5·[1, 2, 3] on the level time."""
    atoms = NumericArrayBatch(_ATOMS, ("trajectory", "time"), label="paths")
    event = BatchSpec(NumericArraySpec(()), time=3)
    return EmpiricalDistribution(atoms, _WEIGHTS, event_spec=OutputSpec(trajectory=event))


class TestBatchValuedMoments:
    def test_mean_and_variance_are_elementwise_weighted_moments(self):
        law = _trajectory_law()
        with workflow_run(seed=0):
            mu = mean(law)
            var = variance(law)
        for moment in (mu, var):
            assert isinstance(moment, NumericArrayBatch)
            assert moment.level_names == ("time",)
        assert mu.spec == BatchSpec(NumericArraySpec((), jnp.float32), time=3)
        assert var.spec == BatchSpec(NumericArraySpec((), jnp.float32, non_negative), time=3)
        np.testing.assert_allclose(mu.raw, [4.25, 8.5, 12.75], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(var.raw, [1.6875, 6.75, 15.1875], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(law._mean(), _SCALE_MEAN * _STEPS, rtol=1e-6, atol=1e-6)

    def test_quantiles_are_elementwise_and_put_their_level_first(self):
        law = _trajectory_law()
        with workflow_run(seed=0):
            median = quantile(law, 0.5)
            both = quantile(law, jnp.array([0.1, 0.9]))
        assert isinstance(median, NumericArrayBatch)
        assert median.level_names == ("time",)
        np.testing.assert_array_equal(median.raw, 5.0 * _STEPS)
        assert both.level_names == ("quantile", "time")
        assert both.spec == BatchSpec(NumericArraySpec((), jnp.float32), quantile=2, time=3)
        np.testing.assert_array_equal(both.raw, _ATOMS)

    def test_the_covariance_is_over_the_flat_coordinates_of_one_draw(self):
        with workflow_run(seed=0):
            matrix = cov(_trajectory_law())
        assert isinstance(matrix, NumericArray)
        assert matrix.spec == NumericArraySpec((3, 3), jnp.float32)
        np.testing.assert_allclose(
            matrix.raw, _SCALE_VARIANCE * jnp.outer(_STEPS, _STEPS), rtol=1e-6, atol=1e-6
        )

    def test_two_event_levels_keep_their_order(self):
        second = jnp.arange(6.0).reshape(2, 3)
        atoms = NumericArrayBatch(
            jnp.stack([jnp.full((2, 3), 2.0), second]), ("path", "chain", "time"), label="paths"
        )
        event = BatchSpec(NumericArraySpec(()), chain=2, time=3)
        law = EmpiricalDistribution(atoms, _WEIGHTS, event_spec=OutputSpec(path=event))
        with workflow_run(seed=0):
            mu = mean(law)
            var = variance(law)
        assert mu.level_names == ("chain", "time")
        assert mu.spec == BatchSpec(NumericArraySpec((), jnp.float32), chain=2, time=3)
        # 0.25·2 + 0.75·x and 0.25·0.75·(x − 2)² at each entry x of the second atom.
        np.testing.assert_allclose(mu.raw, 0.5 + 0.75 * second, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(var.raw, 0.1875 * (second - 2.0) ** 2, rtol=1e-6, atol=1e-6)

    def test_a_level_of_two_axes_keeps_its_axes(self):
        atoms = jnp.stack([jnp.full((2, 3), 1.0), jnp.full((2, 3), 3.0)])
        event = BatchSpec(NumericArraySpec(()), time=(2, 3))
        law = EmpiricalDistribution(atoms, _WEIGHTS, event_spec=OutputSpec(path=event))
        with workflow_run(seed=0):
            mu = mean(law)
        assert mu.level_names == ("time",)
        assert mu.axis_groups == ((2, 3),)
        np.testing.assert_allclose(mu.raw, jnp.full((2, 3), 2.5), rtol=1e-6, atol=1e-6)

    def test_record_elements_give_a_batch_of_records(self):
        x = jnp.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
        atoms = RecordBatch(
            {"x": x, "y": 10 * x},
            ("trajectory", "time"),
            element_spec=RecordSpec(x=(), y=()),
            label="paths",
        )
        event = BatchSpec(RecordSpec(x=(), y=()), time=3)
        law = EmpiricalDistribution(atoms, _WEIGHTS, event_spec=OutputSpec(path=event))
        with workflow_run(seed=0):
            mu = mean(law)
            matrix = cov(law)
        assert isinstance(mu, NumericRecordBatch)
        assert mu.level_names == ("time",)
        assert mu.spec == BatchSpec(NumericRecordSpec(x=(), y=()), time=3)
        # 0.25·[1, 2, 3] + 0.75·[5, 6, 7], and ten times that for y.
        np.testing.assert_allclose(mu["x"].raw, [4.0, 5.0, 6.0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(mu["y"].raw, [40.0, 50.0, 60.0], rtol=1e-6, atol=1e-6)
        # The flat coordinates are x over time, then y over time; each moves by 4 between atoms.
        direction = jnp.concatenate([jnp.ones(3), 10.0 * jnp.ones(3)])
        np.testing.assert_allclose(
            matrix.raw, 3.0 * jnp.outer(direction, direction), rtol=1e-6, atol=1e-6
        )

    def test_a_view_of_a_batch_field_averages_its_draws(self):
        def trajectory(beta):
            return NumericArrayBatch(beta * _STEPS, "time", label="path")

        wrapped = Function(
            trajectory,
            output_spec=OutputSpec(trajectory=None),
            include_inputs=True,
            max_evaluations=20,
        )
        with workflow_run(seed=0):
            joint = wrapped(Normal("beta", 0.0, 1.0))
            mu = mean(joint["trajectory"])
        assert isinstance(mu, NumericArrayBatch)
        assert mu.level_names == ("time",)
        assert mu.provenance.metadata["route"] == "monte_carlo"
        # Every draw is a multiple of [1, 2, 3], so their average is too.
        values = np.asarray(mu.raw)
        np.testing.assert_allclose(values, values[0] * np.asarray(_STEPS), rtol=1e-5, atol=1e-6)
