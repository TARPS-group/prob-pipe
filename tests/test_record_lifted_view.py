"""Tests for ``NumericRecordDistribution.as_record_distribution``.

The method lifts a single-field flat distribution to a Record-keyed view
under a user-supplied :class:`NumericRecordTemplate`. Tests mirror the
existing :class:`FlattenedView` test patterns (its inverse).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    MultivariateNormal,
    Normal,
    NumericRecordTemplate,
    ProductDistribution,
    RecordTemplate,
    cov,
    log_prob,
    mean,
    sample,
    variance,
)
from probpipe.core._numeric_record_distribution import _RecordLiftedView
from probpipe.core.protocols import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)


@pytest.fixture
def mvn4():
    """Single-field 4-D MVN with non-trivial loc / cov."""
    return MultivariateNormal(
        loc=jnp.array([1.0, -1.0, 2.0, 0.5]),
        cov=jnp.diag(jnp.array([0.5, 1.0, 1.5, 2.0])),
        name="theta",
    )


@pytest.fixture
def split_template():
    """Template that fragments a 4-vector into intercept (scalar) + slope (3-vec)."""
    return NumericRecordTemplate(intercept=(), slope=(3,))


# -- Construction / structure --------------------------------------------------


class TestConstruction:
    def test_as_record_returns_lifted_view(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        assert isinstance(rec, _RecordLiftedView)

    def test_record_template_carried_through(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        assert rec.record_template is split_template

    def test_event_shapes_match_template(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        assert rec.event_shapes == {"intercept": (), "slope": (3,)}

    def test_event_shape_single_field(self, mvn4):
        """Single-field template: event_shape returns that field's shape."""
        single = NumericRecordTemplate(theta=(4,))
        rec = mvn4.as_record_distribution(template=single)
        assert rec.event_shape == (4,)

    def test_event_shape_multi_field_raises(self, mvn4, split_template):
        """Multi-field template: event_shape raises, point user at event_shapes."""
        rec = mvn4.as_record_distribution(template=split_template)
        with pytest.raises(TypeError, match="event_shapes"):
            _ = rec.event_shape

    def test_base_distribution_accessor(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        assert rec.base_distribution is mvn4

    def test_name_defaults_to_source(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        assert rec.name == mvn4.name

    def test_name_override(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template, name="post")
        assert rec.name == "post"

    def test_repr_contains_base_and_template(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        r = repr(rec)
        assert "MultivariateNormal" in r
        assert "intercept" in r
        assert "slope" in r


# -- Sampling ------------------------------------------------------------------


class TestSampling:
    def test_sample_unbatched_is_numeric_record(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        draw = sample(rec, key=jax.random.PRNGKey(0))
        # sample() wraps with the function-name field. Unwrap to the
        # actual draw the distribution emitted.
        from probpipe import NumericRecord
        assert isinstance(draw, NumericRecord)
        assert draw.fields == ("intercept", "slope")
        assert draw["intercept"].shape == ()
        assert draw["slope"].shape == (3,)

    def test_sample_batched_is_record_array(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        draws = sample(rec, key=jax.random.PRNGKey(0), sample_shape=(5,))
        from probpipe import NumericRecordArray
        assert isinstance(draws, NumericRecordArray)
        assert draws.batch_shape == (5,)
        assert draws["intercept"].shape == (5,)
        assert draws["slope"].shape == (5, 3)

    def test_sample_roundtrip_with_flat(self, mvn4, split_template):
        """Same seed: lifted-view sample equals NumericRecord.unflatten(flat_sample)."""
        from probpipe import NumericRecord
        rec = mvn4.as_record_distribution(template=split_template)
        key = jax.random.PRNGKey(7)
        rec_draw = rec._sample(key)
        flat_draw = mvn4._sample(key)
        ref = NumericRecord.unflatten(flat_draw, template=split_template)
        np.testing.assert_allclose(jnp.asarray(rec_draw["intercept"]),
                                   jnp.asarray(ref["intercept"]))
        np.testing.assert_allclose(jnp.asarray(rec_draw["slope"]),
                                   jnp.asarray(ref["slope"]))

    def test_chain_flatten_then_lift_on_product(self):
        """``ProductDistribution.as_flat_distribution().as_record_distribution(...)``
        produces a Record view that samples consistently with the source.

        ``FlattenedView`` only carries Sampling + LogProb (per its factory),
        so the lifted view advertises the same subset.
        """
        joint = ProductDistribution(
            a=Normal(loc=1.0, scale=0.5, name="a"),
            b=Normal(loc=-2.0, scale=2.0, name="b"),
        )
        flat = joint.as_flat_distribution()
        rec = flat.as_record_distribution(template=joint.record_template)
        assert rec.record_template is joint.record_template

        # Capability passthrough: FlattenedView has Sampling + LogProb only.
        assert isinstance(rec, SupportsSampling)
        assert isinstance(rec, SupportsLogProb)
        assert not isinstance(rec, SupportsMean)

        # Sample through the chain; verify shape + field structure.
        from probpipe import NumericRecord
        draw = rec._sample(jax.random.PRNGKey(0))
        assert isinstance(draw, NumericRecord)
        assert draw.fields == ("a", "b")
        assert draw["a"].shape == ()
        assert draw["b"].shape == ()


# -- Log-prob ------------------------------------------------------------------


class TestLogProb:
    def test_log_prob_roundtrips_with_flat(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        key = jax.random.PRNGKey(3)
        flat_x = mvn4._sample(key)
        from probpipe import NumericRecord
        rec_x = NumericRecord.unflatten(flat_x, template=split_template)

        lp_rec = float(log_prob(rec, rec_x))
        lp_flat = float(log_prob(mvn4, flat_x))
        # Numerical equality up to JAX precision.
        np.testing.assert_allclose(lp_rec, lp_flat, rtol=1e-5)

    def test_log_prob_batched(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        key = jax.random.PRNGKey(5)
        flat_xs = mvn4._sample(key, sample_shape=(10,))
        from probpipe import NumericRecordArray
        rec_xs = NumericRecordArray.unflatten(
            flat_xs, template=split_template, batch_shape=(10,),
        )
        lp_rec = jnp.asarray(log_prob(rec, rec_xs))
        lp_flat = jnp.asarray(log_prob(mvn4, flat_xs))
        np.testing.assert_allclose(lp_rec, lp_flat, rtol=1e-5)


# -- Moments -------------------------------------------------------------------


class TestMoments:
    def test_mean_unflattens(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        m = mean(rec)
        np.testing.assert_allclose(jnp.asarray(m["intercept"]), 1.0, rtol=1e-5)
        np.testing.assert_allclose(jnp.asarray(m["slope"]),
                                   jnp.array([-1.0, 2.0, 0.5]), rtol=1e-5)

    def test_variance_unflattens(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        v = variance(rec)
        np.testing.assert_allclose(jnp.asarray(v["intercept"]), 0.5, rtol=1e-5)
        np.testing.assert_allclose(jnp.asarray(v["slope"]),
                                   jnp.array([1.0, 1.5, 2.0]), rtol=1e-5)

    def test_cov_stays_flat(self, mvn4, split_template):
        """cov(rec) is the same (event_size, event_size) matrix as cov(source)."""
        rec = mvn4.as_record_distribution(template=split_template)
        c_rec = jnp.asarray(cov(rec))
        c_src = jnp.asarray(cov(mvn4))
        assert c_rec.shape == (4, 4)
        np.testing.assert_allclose(c_rec, c_src, rtol=1e-5)


# -- Error paths + protocol passthrough ----------------------------------------


class TestErrors:
    def test_lifted_view_advertises_all_source_protocols(self, mvn4, split_template):
        """MVN supports everything; the lifted view should too."""
        rec = mvn4.as_record_distribution(template=split_template)
        assert isinstance(rec, SupportsSampling)
        assert isinstance(rec, SupportsLogProb)
        assert isinstance(rec, SupportsMean)
        assert isinstance(rec, SupportsVariance)
        assert isinstance(rec, SupportsCovariance)
        assert isinstance(rec, SupportsExpectation)

    def test_type_error_on_base_record_template(self, mvn4):
        """Passing a base RecordTemplate (allows None leaves) is rejected."""
        bad = RecordTemplate(intercept=(), label=None)
        with pytest.raises(TypeError, match="NumericRecordTemplate"):
            mvn4.as_record_distribution(template=bad)

    def test_value_error_on_size_mismatch(self, mvn4):
        bad = NumericRecordTemplate(a=(), b=(7,))  # flat_size=8 vs source flat_size=4
        with pytest.raises(ValueError, match="flat_size mismatch"):
            mvn4.as_record_distribution(template=bad)

    def test_value_error_on_non_flat_source(self, split_template):
        """Sources with event_shape having >1 dimensions are rejected.

        No public constructor builds a ``NumericRecordDistribution`` with a
        multi-dim event_shape — ``Normal`` is scalar, ``MultivariateNormal``
        is flat. We define a minimal in-test subclass with a hand-set
        event_shape, the pattern STYLE_GUIDE.md §8.4 recommends for
        backend-stubbing.
        """
        from probpipe.core._numeric_record_distribution import (
            NumericRecordDistribution,
        )

        class _MultiDimStub(NumericRecordDistribution):
            def __init__(self):
                self._name = "stub"
                self._record_template = NumericRecordTemplate(stub=(2, 3))

            @property
            def event_shape(self):
                return (2, 3)

        stub = _MultiDimStub()
        assert stub.event_shape == (2, 3)

        with pytest.raises(ValueError, match="at most one dimension"):
            stub.as_record_distribution(
                template=NumericRecordTemplate(a=(), b=(5,)),
            )

    def test_sanity_product_distribution_has_no_method(self):
        """Multi-field ``ProductDistribution`` is not a ``NumericRecordDistribution``
        and so does not expose ``as_record_distribution`` at all. Documents
        the intended class-hierarchy boundary."""
        joint = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Normal(loc=0.0, scale=1.0, name="b"),
        )
        assert not hasattr(joint, "as_record_distribution")


