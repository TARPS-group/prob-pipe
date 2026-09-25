"""Tests for KDEDistribution's template-aware sampling and log-density.

The single-field auto-template path is exercised throughout the
converter tests in ``tests/converters/test_converters.py``; this file
focuses on the ``event_spec=`` constructor parameter and the
:meth:`KDEDistribution.from_empirical` factory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    NumericArraySpec,
    NumericRecord,
    NumericRecordSpec,
    OutputSpec,
    Record,
    RecordSpec,
    real,
)
from probpipe.core._empirical import RecordEmpiricalDistribution
from probpipe.core._numeric_record import NumericRecord as _NumericRecord
from probpipe.core._numeric_record_batch import NumericRecordBatch
from probpipe.distributions.kde import KDEDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_field_template():
    return RecordSpec(intercept=(), slope=())


@pytest.fixture
def flat_samples():
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.randn(200, 2).astype("float32"))


# ---------------------------------------------------------------------------
# Constructor + validation
# ---------------------------------------------------------------------------


class TestRecordSpecConstructor:
    def test_multi_field_template_preserved(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        assert kde.event_spec.spec.fields == ("intercept", "slope")

    def test_mismatched_vector_size_raises(self, flat_samples):
        bad_tpl = RecordSpec(a=(), b=(), c=())  # vector_size=3, samples flat dim=2
        with pytest.raises(ValueError, match="vector_size"):
            KDEDistribution("bad", flat_samples, event_spec=bad_tpl)

    def test_a_one_field_record_draws_an_array(self, flat_samples):
        """A record with one field declares an array under the name, as no
        record does."""
        single = RecordSpec(theta=(2,))
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=single,
        )
        assert kde.event_spec == OutputSpec(post=NumericArraySpec((2,), "float32", real))

    def test_without_a_record_one_draw_is_an_array_under_the_name(self, flat_samples):
        kde = KDEDistribution("kde", flat_samples)
        assert tuple(kde.event_spec.components) == ("kde",)


class TestDeclaration:
    """A KDE declares one draw from its samples and template."""

    def test_samples_without_a_template_are_a_whole_array(self, flat_samples):
        kde = KDEDistribution("post", flat_samples)
        assert kde.event_spec == OutputSpec(post=NumericArraySpec((2,), "float32", real))
        assert kde.event_shape == (2,)

    def test_one_column_draws_scalars(self):
        kde = KDEDistribution("k", jnp.arange(5.0)[:, None])
        assert kde.event_shape == ()

    def test_a_record_template_is_the_declared_record(self, two_field_template, flat_samples):
        kde = KDEDistribution("post", flat_samples, event_spec=two_field_template)
        assert kde.event_spec == OutputSpec(
            RecordSpec(
                intercept=NumericArraySpec((), "float32", real),
                slope=NumericArraySpec((), "float32", real),
            )
        )
        assert kde.supports == {"intercept": real, "slope": real}
        with pytest.raises(TypeError, match="does not draw a single array"):
            _ = kde.event_shape
        assert "event_shape=(2,)" in repr(kde)


# ---------------------------------------------------------------------------
# _sample round-trip
# ---------------------------------------------------------------------------


class TestSampleRoundTrip:
    def test_sample_scalar_returns_numeric_record(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        s = kde._sample(jax.random.PRNGKey(0), ())
        assert isinstance(s, _NumericRecord)
        assert tuple(s.event_template.keys()) == ("intercept", "slope")

    def test_sample_batched_returns_record_batch(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        s = kde._sample(jax.random.PRNGKey(1), (8,))
        assert isinstance(s, NumericRecordBatch)
        assert s.batch_shape == (8,)
        assert tuple(s.event_template.keys()) == ("intercept", "slope")

    def test_sample_no_template_returns_raw_array(self, flat_samples):
        """With auto-build single-field template the sample stays a raw
        array (existing TFP-base behaviour)."""
        kde = KDEDistribution("post", flat_samples)
        s = kde._sample(jax.random.PRNGKey(2), (4,))
        assert isinstance(s, jnp.ndarray)
        assert s.shape == (4, 2)


# ---------------------------------------------------------------------------
# _log_prob dual input
# ---------------------------------------------------------------------------


class TestLogProbDualInput:
    def test_structured_and_flat_inputs_agree(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        nr = NumericRecord("nr", intercept=jnp.array(0.5), slope=jnp.array(-0.3))
        lp_struct = kde._log_prob(nr)
        lp_flat = kde._log_prob(jnp.array([0.5, -0.3]))
        assert jnp.allclose(lp_struct, lp_flat)

    def test_record_accepted(self, two_field_template, flat_samples):
        """Plain Record (not NumericRecord) also accepted."""
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        rec = Record("r", intercept=jnp.array(0.5), slope=jnp.array(-0.3))
        lp = kde._log_prob(rec)
        assert jnp.isfinite(lp)

    def test_batched_record_batch(self, two_field_template, flat_samples):
        """A NumericRecordBatch input is flattened to (batch, d) and the
        TFP mixture log-prob returns a (batch,) array."""
        kde = KDEDistribution(
            "post",
            flat_samples,
            event_spec=two_field_template,
        )
        # Build a 3-row NumericRecordBatch
        nrb = NumericRecordBatch(
            "batch",
            {"intercept": jnp.array([0.5, 0.6, 0.7]), "slope": jnp.array([-0.3, -0.4, -0.5])},
            "draw",
            element_spec=NumericRecordSpec(intercept=(), slope=()),
        )
        lp = kde._log_prob(nrb)
        assert lp.shape == (3,)
        # Matches the flat form
        lp_flat = kde._log_prob(
            jnp.stack(
                [
                    jnp.array([0.5, -0.3]),
                    jnp.array([0.6, -0.4]),
                    jnp.array([0.7, -0.5]),
                ]
            )
        )
        assert jnp.allclose(lp, lp_flat)


# ---------------------------------------------------------------------------
# from_empirical factory
# ---------------------------------------------------------------------------


class TestFromEmpirical:
    def test_multi_field_record_empirical(self, two_field_template):
        n = 200
        rec = Record(
            "r",
            intercept=jax.random.normal(jax.random.PRNGKey(0), (n,)),
            slope=jax.random.normal(jax.random.PRNGKey(1), (n,)),
        )
        emp = RecordEmpiricalDistribution("emp", rec)
        kde = KDEDistribution.from_empirical(emp, name="post")
        assert isinstance(kde, KDEDistribution)
        # The source's record is declared.
        assert kde.event_spec.spec.fields == ("intercept", "slope")
        # Samples come back structured
        s = kde._sample(jax.random.PRNGKey(2), ())
        assert isinstance(s, _NumericRecord)
        assert tuple(s.event_template.keys()) == ("intercept", "slope")

    def test_single_field_record_empirical(self):
        """Single-field empirical → single-field KDE (no multi-field
        template threading)."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        emp = RecordEmpiricalDistribution("theta", samples)
        kde = KDEDistribution.from_empirical(emp)
        assert isinstance(kde, KDEDistribution)
        # One array under the source's field name.
        assert tuple(kde.event_spec.components) == ("theta",)

    def test_rejects_non_record_empirical(self):
        """Generic (object-array) EmpiricalDistribution is rejected."""
        emp_generic = EmpiricalDistribution(
            "x",
            np.array([{"a": 1}, {"a": 2}], dtype=object),
        )
        with pytest.raises(TypeError, match="RecordEmpiricalDistribution"):
            KDEDistribution.from_empirical(emp_generic)
