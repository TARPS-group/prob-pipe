"""Contract tests of sample: draws at the declared kind, batches on a sample level, raw forms."""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    ApplicabilityError,
    NumericArray,
    NumericArrayBatch,
    NumericRecordBatch,
    Record,
    RecordBatch,
    workflow_run,
)
from probpipe.core._batch import BatchSpec
from probpipe.core._dispatch import ResolutionError
from probpipe.distributions._batches import DistributionBatch
from probpipe.distributions._distribution import Distribution
from probpipe.operations._sample import sample

from ._laws import Bare, Gaussian, Kernel, Measure, OneField, Pair, Polymorphic, Vector


class TestOneDraw:
    def test_an_array_law_draws_a_numeric_array_at_its_declaration(self):
        draw = sample(Gaussian("g"))
        assert isinstance(draw, NumericArray)
        assert draw.spec == Gaussian("g").event_spec.spec

    def test_a_record_law_draws_a_record_with_its_schema(self):
        draw = sample(Pair("p"))
        assert isinstance(draw, Record)
        assert jnp.shape(draw["b"]) == (2,)

    def test_a_one_field_record_stays_a_record(self):
        assert isinstance(sample(OneField("o")), Record)
        assert isinstance(sample(OneField("o"), sample_shape=(3,)), RecordBatch)

    def test_a_measure_draws_a_law(self):
        assert isinstance(sample(Measure("m")), Distribution)

    def test_the_draw_is_labeled_by_the_law(self):
        assert sample(Gaussian("g")).label == "g"
        assert sample(Gaussian("g"), sample_shape=(3,)).label == "g"


class TestBatches:
    def test_a_sample_shape_prepends_axes_on_a_level_named_sample(self):
        draws = sample(Vector("v"), sample_shape=(4, 3))
        assert isinstance(draws, NumericArrayBatch)
        assert draws.batch_shape == (4, 3)
        assert draws.level_names == ("sample",)
        assert draws.element_spec.shape == (2,)

    def test_an_integer_sample_shape_is_one_axis(self):
        assert sample(Gaussian("g"), sample_shape=5).batch_shape == (5,)

    def test_a_record_law_draws_a_batch_of_records(self):
        draws = sample(Pair("p"), sample_shape=(6,))
        assert isinstance(draws, RecordBatch)
        assert draws.batch_shape == (6,)
        assert draws.level_names == ("sample",)

    def test_a_joint_draws_a_batch_of_records_under_its_declaration(self):
        joint = Gaussian("a") * Gaussian("b", 5.0)
        draws = sample(joint, sample_shape=(4,))
        assert isinstance(draws, NumericRecordBatch)
        assert (draws.batch_shape, draws.level_names) == ((4,), ("sample",))
        assert draws.element_spec == joint.event_spec.spec
        assert jnp.shape(draws["b"]) == (4,)

    def test_a_batch_of_draws_is_declared_under_the_event_components(self):
        whole = sample.check(Gaussian("g"), (4,)).result
        assert (list(whole.components), whole.exposes_record) == (["g"], False)
        assert isinstance(whole.spec, BatchSpec)
        record = sample.check(Pair("p"), (4,)).result
        assert list(record.components) == list(Pair("p").event_spec.components)
        assert record.spec == BatchSpec(Pair("p").event_spec.spec, sample=4)

    def test_a_measure_draws_a_batch_of_laws(self):
        draws = sample(Measure("m"), sample_shape=(3,))
        assert isinstance(draws, DistributionBatch)
        assert draws.batch_shape == (3,)

    def test_the_draws_of_a_batch_are_independent(self):
        values = np.asarray(sample(Gaussian("g"), sample_shape=(64,)).values)
        assert len(np.unique(values)) == 64

    def test_a_seeded_workflow_reproduces_the_batch(self):
        with workflow_run(seed=5):
            first = np.asarray(sample(Gaussian("g"), sample_shape=(8,)).values)
        with workflow_run(seed=5):
            second = np.asarray(sample(Gaussian("g"), sample_shape=(8,)).values)
        np.testing.assert_array_equal(first, second)


class TestRawDraws:
    def test_one_raw_draw_is_the_raw_value(self):
        draw = sample.with_options(raw=True)(Gaussian("g"))
        assert isinstance(draw, jnp.ndarray) and draw.shape == ()

    def test_a_raw_batch_is_the_stacked_array_with_the_batch_axes_leading(self):
        draws = sample.with_options(raw=True)(Vector("v"), sample_shape=(5,))
        assert isinstance(draws, jnp.ndarray) and draws.shape == (5, 2)

    def test_a_raw_draw_of_a_measure_is_the_law_itself(self):
        assert isinstance(sample.with_options(raw=True)(Measure("m")), Distribution)

    def test_a_raw_batch_of_laws_is_an_object_array(self):
        draws = sample.with_options(raw=True)(Measure("m"), sample_shape=(2,))
        assert isinstance(draws, np.ndarray) and draws.dtype == object

    def test_a_raw_batch_of_a_joint_is_the_mapping_of_its_columns(self):
        draws = sample.with_options(raw=True)(Gaussian("a") * Gaussian("b"), sample_shape=(3,))
        assert isinstance(draws, dict) and list(draws) == ["a", "b"]
        assert all(jnp.shape(column) == (3,) for column in draws.values())


class TestRequirements:
    def test_sampling_requires_a_concrete_declaration_and_names_the_free_dimensions(self):
        with pytest.raises(ApplicabilityError, match=r"\['n'\]"):
            sample(Polymorphic("poly"))

    @pytest.mark.parametrize("shape", [True, (2.5,), "3", (-1,)])
    def test_a_malformed_sample_shape_raises_applicability_error(self, shape):
        with pytest.raises(ApplicabilityError):
            sample(Gaussian("g"), sample_shape=shape)

    def test_a_law_that_does_not_sample_raises_resolution_error(self):
        with pytest.raises(ResolutionError, match="does not implement SupportsSampling"):
            sample(Bare("b"))

    def test_the_operation_takes_no_key(self):
        assert list(inspect.signature(sample).parameters) == ["d", "sample_shape"]

    def test_a_kernel_is_refused_without_its_given(self):
        with pytest.raises(ApplicabilityError, match="ConditionalDistributionSpec"):
            sample(Kernel())

    @pytest.mark.pending(
        reason="the fused conditional path adds given= to the operation", raises=TypeError
    )
    def test_the_fused_path_draws_from_the_kernel_at_its_given(self):
        assert isinstance(sample(Kernel(), given={"mu": 1.0}), NumericArray)
