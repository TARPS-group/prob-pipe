"""Empirical atoms that are whole batches retain their event multiplicity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    EmpiricalDistribution,
    NumericArrayBatch,
    NumericArraySpec,
    OutputSpec,
    RecordBatch,
    RecordSpec,
    sample,
    workflow_run,
)
from probpipe.distributions._capabilities import SupportsMean, SupportsVariance


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
        assert not isinstance(law, (SupportsMean, SupportsVariance))
        np.testing.assert_array_equal(law._sample(jax.random.key(0)), values[1])
        with workflow_run(seed=3):
            single = sample(law)
            many = sample(law, (2, 4))
        assert single.level_names == ("time",)
        assert single.batch_shape == event_shape
        assert many.level_names == ("sample", "time")
        assert many.axis_groups == ((2, 4), event_shape)
        np.testing.assert_array_equal(many.raw(), jnp.full((2, 4, *event_shape), 7.0))

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
        np.testing.assert_array_equal(one.raw("x"), values[1])
        np.testing.assert_array_equal(many.raw("y"), 10 * many.raw("x"))

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
