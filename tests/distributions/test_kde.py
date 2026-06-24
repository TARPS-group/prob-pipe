"""Tests for KDEDistribution's template-aware sampling and log-density.

The single-field auto-template path is exercised throughout the
converter tests in ``tests/converters/test_converters.py``; this file
focuses on the multi-field ``event_template=`` constructor parameter
(issue #267) and the :meth:`KDEDistribution.from_empirical` factory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    EmpiricalDistribution,
    EventTemplate,
    NumericEventTemplate,
    NumericRecord,
    Record,
)
from probpipe.core._empirical import RecordEmpiricalDistribution
from probpipe.core._numeric_record import NumericRecord as _NumericRecord
from probpipe.core._record_array import NumericRecordArray
from probpipe.distributions.kde import KDEDistribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_field_template():
    return EventTemplate(intercept=(), slope=())


@pytest.fixture
def flat_samples():
    rng = np.random.RandomState(0)
    return jnp.asarray(rng.randn(200, 2).astype("float32"))


# ---------------------------------------------------------------------------
# Constructor + validation
# ---------------------------------------------------------------------------


class TestEventTemplateConstructor:
    def test_multi_field_template_preserved(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        assert kde.event_template is two_field_template
        assert kde.event_template.fields == ("intercept", "slope")

    def test_mismatched_vector_size_raises(self, flat_samples):
        bad_tpl = EventTemplate(a=(), b=(), c=())  # vector_size=3, samples flat dim=2
        with pytest.raises(ValueError, match="vector_size"):
            KDEDistribution(flat_samples, event_template=bad_tpl, name="bad")

    def test_single_field_template_unchanged(self, flat_samples):
        """A single-field template falls through to the auto-build path
        (the existing single-field behaviour is the baseline)."""
        single = EventTemplate(theta=(2,))
        kde = KDEDistribution(
            flat_samples,
            event_template=single,
            name="post",
        )
        # Single-field template still gets stored — but the special-case
        # multi-field path isn't taken.
        assert kde.event_template is not None

    def test_no_template_keeps_auto_build(self, flat_samples):
        """Without event_template= the auto-build keyed by name still
        fires — backward compatible with every existing call site."""
        kde = KDEDistribution(flat_samples, name="kde")
        assert kde.event_template.fields == ("kde",)


# ---------------------------------------------------------------------------
# _sample round-trip
# ---------------------------------------------------------------------------


class TestSampleRoundTrip:
    def test_sample_scalar_returns_numeric_record(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        s = kde._sample(jax.random.PRNGKey(0), ())
        assert isinstance(s, _NumericRecord)
        assert s.fields == ("intercept", "slope")

    def test_sample_batched_returns_record_array(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        s = kde._sample(jax.random.PRNGKey(1), (8,))
        assert isinstance(s, NumericRecordArray)
        assert s.batch_shape == (8,)
        assert s.fields == ("intercept", "slope")

    def test_sample_no_template_returns_raw_array(self, flat_samples):
        """With auto-build single-field template the sample stays a raw
        array (existing TFP-base behaviour)."""
        kde = KDEDistribution(flat_samples, name="post")
        s = kde._sample(jax.random.PRNGKey(2), (4,))
        assert isinstance(s, jnp.ndarray)
        assert s.shape == (4, 2)


# ---------------------------------------------------------------------------
# _log_prob dual input
# ---------------------------------------------------------------------------


class TestLogProbDualInput:
    def test_structured_and_flat_inputs_agree(self, two_field_template, flat_samples):
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        nr = NumericRecord(intercept=jnp.array(0.5), slope=jnp.array(-0.3))
        lp_struct = kde._log_prob(nr)
        lp_flat = kde._log_prob(jnp.array([0.5, -0.3]))
        assert jnp.allclose(lp_struct, lp_flat)

    def test_record_accepted(self, two_field_template, flat_samples):
        """Plain Record (not NumericRecord) also accepted."""
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        rec = Record(intercept=jnp.array(0.5), slope=jnp.array(-0.3))
        lp = kde._log_prob(rec)
        assert jnp.isfinite(lp)

    def test_batched_record_array(self, two_field_template, flat_samples):
        """A NumericRecordArray input is flattened to (batch, d) and the
        TFP mixture log-prob returns a (batch,) array."""
        kde = KDEDistribution(
            flat_samples,
            event_template=two_field_template,
            name="post",
        )
        # Build a 3-row NumericRecordArray
        nra = NumericRecordArray(
            {"intercept": jnp.array([0.5, 0.6, 0.7]), "slope": jnp.array([-0.3, -0.4, -0.5])},
            batch_shape=(3,),
            template=NumericEventTemplate(intercept=(), slope=()),
        )
        lp = kde._log_prob(nra)
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
            intercept=jax.random.normal(jax.random.PRNGKey(0), (n,)),
            slope=jax.random.normal(jax.random.PRNGKey(1), (n,)),
        )
        emp = RecordEmpiricalDistribution(rec)
        kde = KDEDistribution.from_empirical(emp, name="post")
        assert isinstance(kde, KDEDistribution)
        # Template preserved
        assert kde.event_template.fields == ("intercept", "slope")
        # Samples come back structured
        s = kde._sample(jax.random.PRNGKey(2), ())
        assert isinstance(s, _NumericRecord)
        assert s.fields == ("intercept", "slope")

    def test_single_field_record_empirical(self):
        """Single-field empirical → single-field KDE (no multi-field
        template threading)."""
        samples = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        emp = RecordEmpiricalDistribution(samples, name="theta")
        kde = KDEDistribution.from_empirical(emp)
        assert isinstance(kde, KDEDistribution)
        # Single-field template
        assert kde.event_template.fields == ("theta",)

    def test_rejects_non_record_empirical(self):
        """Generic (object-array) EmpiricalDistribution is rejected."""
        emp_generic = EmpiricalDistribution(
            np.array([{"a": 1}, {"a": 2}], dtype=object),
            name="x",
        )
        with pytest.raises(TypeError, match="RecordEmpiricalDistribution"):
            KDEDistribution.from_empirical(emp_generic)
