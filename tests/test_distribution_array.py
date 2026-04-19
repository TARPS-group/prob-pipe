"""Tests for ``DistributionArray`` (issue #130 / PR 1 commit 2).

A ``DistributionArray`` is a shape-indexed collection of independent
distributions with leading ``batch_shape=(n,)`` — **not** a mixture.
Used by the RecordArray-broadcast layer when a WorkflowFunction sweep
returns Distribution-valued outputs, one per sweep row.

The tests cover: construction + invariants, Pattern B dynamic protocol
support (isinstance reflects all-components-support), sample/mean/
variance/log_prob for both scalar and Record component-sample types,
and the provenance slot inherited from ``Distribution``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    NumericEmpiricalDistribution,
    ProductDistribution,
    Provenance,
)
from probpipe.core._distribution_array import (
    DistributionArray,
    _make_distribution_array,
)
from probpipe.core.protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_list_of_normals(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        assert isinstance(da, DistributionArray)
        assert da.n == 4
        assert len(da) == 4
        assert da.batch_shape == (4,)
        assert da.event_shape == ()

    def test_indexing_returns_component(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        assert da[0] is comps[0]
        assert da[2] is comps[2]

    def test_negative_indexing(self):
        """Negative int indices work; Python wraparound applies per axis.

        Regression for a ``np.ravel_multi_index``-based fast path that
        initially rejected negatives (the common ``dists[-1]`` pattern
        — e.g. the last posterior out of ``iterate`` — must work).
        """
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        assert da[-1] is comps[3]
        assert da[-2] is comps[2]

    def test_iter_yields_components(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        assert list(da) == list(comps)

    def test_empty_components_raises(self):
        with pytest.raises(ValueError, match="at least one component"):
            _make_distribution_array([])

    def test_single_component(self):
        """Edge case: n=1 works; batch_shape=(1,), indexing, sampling,
        and reductions all behave uniformly with the n>1 path."""
        comp = Normal(loc=7.0, scale=0.5, name="only")
        da = _make_distribution_array([comp])
        assert da.n == 1
        assert da.batch_shape == (1,)
        assert da[0] is comp
        assert da._mean().shape == (1,)
        np.testing.assert_allclose(da._mean(), [7.0])

    def test_factory_returns_distribution_subclass(self):
        comps = [Normal(loc=0.0, scale=1.0, name="d0")]
        da = _make_distribution_array(comps)
        from probpipe.core._distribution_base import Distribution
        assert isinstance(da, Distribution)

    def test_repr(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        r = repr(da)
        assert "DistributionArray" in r
        assert "n=3" in r

    # The factory demands uniform shapes so the (n,) + inner_batch
    # batch_shape semantic holds. Mismatches raise at construction.

    def test_mismatched_event_shape_raises(self):
        c0 = Normal(loc=0.0, scale=1.0, name="d0")
        c1 = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        )
        with pytest.raises(ValueError, match="event_shape"):
            _make_distribution_array([c0, c1])


# ---------------------------------------------------------------------------
# Pattern B — dynamic protocol opt-in
# ---------------------------------------------------------------------------


class TestProtocolOptIn:
    """A DistributionArray satisfies SupportsX iff *every* component does.

    This is the Pattern B "protocols supported by all" rule used
    throughout ProbPipe's view / joint classes.
    """

    def test_all_protocols_when_all_components_support(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        assert isinstance(da, SupportsSampling)
        assert isinstance(da, SupportsMean)
        assert isinstance(da, SupportsVariance)
        assert isinstance(da, SupportsLogProb)
        # isinstance is necessary but not sufficient — verify the
        # methods actually dispatch and produce the expected shapes.
        assert da._sample(jax.random.PRNGKey(0)).shape == (3,)
        assert da._mean().shape == (3,)
        assert da._variance().shape == (3,)
        assert da._log_prob(jnp.zeros(3)).shape == (3,)

    def test_drops_log_prob_when_component_lacks_it(self):
        # NumericEmpiricalDistribution supports Sampling + Mean + Variance
        # but NOT LogProb. The DistributionArray should mirror that.
        raw = [jax.random.normal(jax.random.PRNGKey(i), (50,)) for i in range(3)]
        comps = [NumericEmpiricalDistribution(r) for r in raw]
        assert not isinstance(comps[0], SupportsLogProb)

        da = _make_distribution_array(comps)
        assert isinstance(da, SupportsSampling)
        assert isinstance(da, SupportsMean)
        assert isinstance(da, SupportsVariance)
        assert not isinstance(da, SupportsLogProb)
        # And _log_prob is not defined on the dynamic subclass —
        # the opt-out is structural, not just a lie.
        assert not hasattr(da, "_log_prob")

    def test_factory_cache_reuses_class(self):
        """Two DistributionArrays with the same protocol signature share
        a class — keeps isinstance cheap and JIT cache-friendly."""
        from probpipe.core._distribution_array import _distarray_class_cache
        # Clear cache so the assertion is deterministic
        _distarray_class_cache.clear()
        c0 = [Normal(loc=0.0, scale=1.0, name="x0")]
        c1 = [Normal(loc=1.0, scale=2.0, name="x1")]
        da0 = _make_distribution_array(c0)
        da1 = _make_distribution_array(c1)
        assert type(da0) is type(da1)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_scalar_sample_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0))
        assert s.shape == (4,)

    def test_sample_shape_leading_axis(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0), sample_shape=(7,))
        assert s.shape == (7, 4)

    def test_sample_shape_2d(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0), sample_shape=(5, 2))
        assert s.shape == (5, 2, 3)

    def test_components_drive_samples(self):
        """The i-th slice of the stacked sample must concentrate around
        the i-th component's mean — confirms per-component (not mixture)
        sampling."""
        comps = [Normal(loc=float(i) * 100, scale=1e-3, name=f"d{i}")
                 for i in range(3)]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0), sample_shape=(1000,))
        # column i should have mean ≈ i * 100
        means = s.mean(axis=0)
        np.testing.assert_allclose(means, jnp.array([0.0, 100.0, 200.0]),
                                   atol=0.2)

    def test_record_valued_components_scalar_sample(self):
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1e-3, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1e-3, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0))
        from probpipe import RecordArray
        assert isinstance(s, RecordArray)
        assert s.batch_shape == (3,)
        np.testing.assert_allclose(s["x"], [0.0, 1.0, 2.0], atol=1e-2)

    def test_record_valued_components_batched_sample(self):
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1e-3, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1e-3, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        s = da._sample(jax.random.PRNGKey(0), sample_shape=(5,))
        from probpipe import NumericRecordArray
        assert isinstance(s, NumericRecordArray)
        assert s.batch_shape == (5, 3)
        # Shape of x field: (5, 3) — leading sample axis then the n axis
        assert s["x"].shape == (5, 3)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------


class TestMean:
    def test_scalar_components_mean_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        m = da._mean()
        assert m.shape == (4,)
        np.testing.assert_allclose(m, [0.0, 1.0, 2.0, 3.0])

    def test_record_components_mean_is_recordarray(self):
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1.0, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1.0, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        m = da._mean()
        from probpipe import RecordArray
        assert isinstance(m, RecordArray)
        assert m.batch_shape == (3,)
        np.testing.assert_allclose(m["x"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(m["y"], [0.0, -1.0, -2.0])


class TestVariance:
    def test_scalar_components_variance_shape(self):
        comps = [Normal(loc=0.0, scale=float(i + 1), name=f"d{i}")
                 for i in range(3)]
        da = _make_distribution_array(comps)
        v = da._variance()
        assert v.shape == (3,)
        np.testing.assert_allclose(v, [1.0, 4.0, 9.0])


class TestLogProb:
    def test_per_component_log_prob(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        # value shape (n,) matches the batch axis
        value = jnp.array([0.0, 1.0, 2.0])  # each at its component's mean
        lp = da._log_prob(value)
        assert lp.shape == (3,)
        # All three log-probs equal log_prob(N(0,1), 0)
        np.testing.assert_allclose(lp, [lp[0]] * 3, rtol=1e-5)


# ---------------------------------------------------------------------------
# Provenance (inherited from Distribution)
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_initial_source_is_none(self):
        comps = [Normal(loc=0.0, scale=1.0, name="d0")]
        da = _make_distribution_array(comps)
        assert da.source is None

    def test_with_source_roundtrip(self):
        comps = [Normal(loc=0.0, scale=1.0, name="d0")]
        da = _make_distribution_array(comps)
        parent = Normal(loc=0.0, scale=1.0, name="parent")
        da.with_source(Provenance("sweep", parents=(parent,)))
        assert da.source.operation == "sweep"
        assert da.source.parents == (parent,)
