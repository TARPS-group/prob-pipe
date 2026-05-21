"""Tests for :class:`NumericRecordDistributionView`.

Constructed via
:meth:`FlatNumericRecordDistribution.as_record_distribution`, the view
lifts a flat distribution to a Record-keyed structure under a
user-supplied :class:`NumericRecordTemplate`. Tests mirror the
existing :class:`FlattenedDistributionView` test patterns (its inverse).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Dirichlet,
    FlatNumericRecordDistribution,
    MultivariateNormal,
    Normal,
    NumericRecord,
    NumericRecordArray,
    NumericRecordTemplate,
    ProductDistribution,
    RecordTemplate,
    cov,
    expectation,
    log_prob,
    mean,
    sample,
    variance,
)
from probpipe.core._numeric_record_distribution import NumericRecordDistributionView
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
        assert isinstance(rec, NumericRecordDistributionView)

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
        assert isinstance(draw, NumericRecord)
        assert draw.fields == ("intercept", "slope")
        assert draw["intercept"].shape == ()
        assert draw["slope"].shape == (3,)

    def test_sample_batched_is_record_array(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        draws = sample(rec, key=jax.random.PRNGKey(0), sample_shape=(5,))
        assert isinstance(draws, NumericRecordArray)
        assert draws.batch_shape == (5,)
        assert draws["intercept"].shape == (5,)
        assert draws["slope"].shape == (5, 3)

    def test_sample_roundtrip_with_flat(self, mvn4, split_template):
        """Same seed: lifted-view sample equals NumericRecord.unflatten(flat_sample)."""
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

        ``FlattenedDistributionView`` only carries Sampling + LogProb (per its factory),
        so the lifted view advertises the same subset.
        """
        joint = ProductDistribution(
            a=Normal(loc=1.0, scale=0.5, name="a"),
            b=Normal(loc=-2.0, scale=2.0, name="b"),
        )
        flat = joint.as_flat_distribution()
        rec = flat.as_record_distribution(template=joint.record_template)
        assert rec.record_template is joint.record_template

        # Capability passthrough: FlattenedDistributionView has Sampling + LogProb only.
        assert isinstance(rec, SupportsSampling)
        assert isinstance(rec, SupportsLogProb)
        assert not isinstance(rec, SupportsMean)

        # Sample through the chain; verify shape + field structure.
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
        rec_x = NumericRecord.unflatten(flat_x, template=split_template)

        lp_rec = float(log_prob(rec, rec_x))
        lp_flat = float(log_prob(mvn4, flat_x))
        # Numerical equality up to JAX precision.
        np.testing.assert_allclose(lp_rec, lp_flat, rtol=1e-5)

    def test_log_prob_batched(self, mvn4, split_template):
        rec = mvn4.as_record_distribution(template=split_template)
        key = jax.random.PRNGKey(5)
        flat_xs = mvn4._sample(key, sample_shape=(10,))
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

    def test_expectation_through_unflatten(self, mvn4, split_template):
        """``expectation(rec, f)`` MC-estimates ``E[f(record)]``.

        f operates on a NumericRecord drawn from the lifted view; the
        estimate should match the analytic E[intercept^2 + sum(slope^2)]
        = (loc^2 + var) summed across fields to MC tolerance.
        """
        rec = mvn4.as_record_distribution(template=split_template)

        def f(r):
            return r["intercept"] ** 2 + jnp.sum(r["slope"] ** 2)

        est = expectation(
            rec, f, key=jax.random.PRNGKey(0),
            num_evaluations=5000, return_dist=False,
        )
        # E[X^2] = loc^2 + var per coordinate. 5k samples puts MC SE
        # around ~0.15 for these parameters; atol=0.2 stays meaningful.
        loc = jnp.array([1.0, -1.0, 2.0, 0.5])
        var = jnp.array([0.5, 1.0, 1.5, 2.0])
        analytic = float(jnp.sum(loc ** 2 + var))
        np.testing.assert_allclose(float(jnp.asarray(est)), analytic, atol=0.2)


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

    def test_type_error_on_non_flat_source(self, split_template):
        """Non-``FlatNumericRecordDistribution`` sources can't call
        ``as_record_distribution`` directly — the method is only on the
        flat subclass. The stub on ``NumericRecordDistribution`` raises
        ``TypeError`` with a migration hint pointing at
        ``as_flat_distribution()``.
        """
        n = Normal(loc=0.0, scale=1.0, name="x")  # not Flat (event_shape == ())
        assert not isinstance(n, FlatNumericRecordDistribution)
        # Match both the type name and the migration hint so a future
        # shortening of either part trips the test.
        with pytest.raises(TypeError, match="FlatNumericRecordDistribution"):
            n.as_record_distribution(template=NumericRecordTemplate(x=(1,)))
        with pytest.raises(TypeError, match="as_flat_distribution"):
            n.as_record_distribution(template=NumericRecordTemplate(x=(1,)))

    def test_no_method_on_record_distribution(self):
        """``ProductDistribution`` inherits from ``RecordDistribution``
        rather than ``NumericRecordDistribution``, so it doesn't expose
        ``as_record_distribution`` at all. Documents the intended
        class-hierarchy boundary.
        """
        joint = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Normal(loc=0.0, scale=1.0, name="b"),
        )
        assert not hasattr(joint, "as_record_distribution")


# -- FlatNumericRecordDistribution membership ----------------------------------


class TestFlatContract:
    """Pins the class-hierarchy promises that the refactor turned into
    type invariants. A regression here (an accidental MRO change, a
    deleted base, …) would silently undo the one-hop
    ``mvn.as_record_distribution(...)`` ergonomic the refactor was about.
    """

    def test_multivariate_normal_is_flat(self):
        mvn = MultivariateNormal(loc=jnp.zeros(3), cov=jnp.eye(3), name="x")
        assert isinstance(mvn, FlatNumericRecordDistribution)
        assert mvn.flat_size == 3

    def test_dirichlet_is_flat(self):
        d = Dirichlet(concentration=jnp.ones(4), name="p")
        assert isinstance(d, FlatNumericRecordDistribution)
        assert d.flat_size == 4

    def test_flattened_view_is_flat(self):
        """``as_flat_distribution()`` produces a ``FlatNumericRecordDistribution``
        even when the base is non-flat (e.g. a scalar ``Normal`` or a
        multi-field ``ProductDistribution``).
        """
        flat_scalar = Normal(loc=0.0, scale=1.0, name="x").as_flat_distribution()
        assert isinstance(flat_scalar, FlatNumericRecordDistribution)
        assert flat_scalar.flat_size == 1

        flat_joint = ProductDistribution(
            a=Normal(loc=0.0, scale=1.0, name="a"),
            b=Normal(loc=0.0, scale=1.0, name="b"),
        ).as_flat_distribution()
        assert isinstance(flat_joint, FlatNumericRecordDistribution)
        assert flat_joint.flat_size == 2

    def test_flat_size_raises_on_non_1d_event_shape(self):
        """A misdeclared subclass with rank-≥2 event_shape surfaces on
        ``flat_size`` access rather than silently truncating to the
        first dim.
        """
        from probpipe.core._numeric_record_distribution import (
            FlatNumericRecordDistribution as _FlatNRD,
        )

        class _BadFlat(_FlatNRD):
            def __init__(self):
                self._name = "bad"
                self._record_template = NumericRecordTemplate(bad=(2, 3))

            @property
            def event_shape(self):
                return (2, 3)

        bad = _BadFlat()
        with pytest.raises(TypeError, match="event_shape"):
            _ = bad.flat_size


# -- Cross-class smoke: lifting from Dirichlet (a non-MVN FlatNRD) -------------


class TestLiftFromOtherFlatParametrics:
    """The four multivariate parametrics inherit from
    ``FlatNumericRecordDistribution``, so each one can serve as a flat
    source for ``as_record_distribution`` — not just MVN. One smoke
    test ensures the type-system change actually unlocks the
    cross-family lift path.
    """

    def test_dirichlet_as_record_distribution_preserves_simplex(self):
        d = Dirichlet(concentration=jnp.array([1.0, 2.0, 3.0]), name="p")
        rec = d.as_record_distribution(template=NumericRecordTemplate(probs=(3,)))
        assert isinstance(rec, NumericRecordDistributionView)

        draw = sample(rec, key=jax.random.PRNGKey(0))
        # Simplex invariant must survive the lift.
        np.testing.assert_allclose(float(jnp.sum(draw["probs"])), 1.0, rtol=1e-5)


