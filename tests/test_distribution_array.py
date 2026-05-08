"""Tests for ``DistributionArray``.

A ``DistributionArray`` is ``Array[Distribution]`` — an ordered
collection of scalar distributions indexed by a (multi-d)
``batch_shape``. Vectorized ops live at the ``WorkflowFunction``
sweep layer, not on the DistArray itself. This file covers:

- Construction + invariants + container surface (indexing, iteration,
  shape, slicing, ``len()``).
- Sweep-layer behavior for the four canonical ops (``sample``,
  ``mean``, ``variance``, ``log_prob``) on DistArray inputs — axis
  ordering, return-type contract, per-cell values.
- Provenance plumbing inherited from ``Distribution``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Normal,
    NumericRecordArray,
    ProductDistribution,
    Provenance,
    log_prob,
    mean,
    sample,
    variance,
)
from probpipe.core._distribution_array import (
    DistributionArray,
    _make_distribution_array,
)


# ---------------------------------------------------------------------------
# Construction & invariants
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_list_of_normals(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        assert isinstance(da, DistributionArray)
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
        """Edge case: a single-cell DA works; batch_shape=(1,), indexing,
        sampling, and reductions all behave uniformly with the multi-cell path."""
        comp = Normal(loc=7.0, scale=0.5, name="only")
        da = _make_distribution_array([comp])
        assert len(da) == 1
        assert da.batch_shape == (1,)
        assert da[0] is comp
        m = mean(da)
        np.testing.assert_allclose(jnp.asarray(m), [7.0])
        assert m.batch_shape == (1,)

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
        assert "batch_shape=(3,)" in r

    # The factory demands uniform event_shape across components so the
    # shared event_shape contract holds. Mismatches raise at construction.

    def test_mismatched_event_shape_raises(self):
        c0 = Normal(loc=0.0, scale=1.0, name="d0")
        c1 = ProductDistribution(
            x=Normal(loc=0.0, scale=1.0, name="x"),
            y=Normal(loc=0.0, scale=1.0, name="y"),
        )
        with pytest.raises(ValueError, match="event_shape"):
            _make_distribution_array([c0, c1])


# ---------------------------------------------------------------------------
# Container surface — len, size, iteration (numpy / jax alignment, #178)
# ---------------------------------------------------------------------------


class TestContainerSurface:
    """Pin the numpy / jax conventions on the container surface:

    * ``len(da)`` is the leading-axis dim (mirrors ``np.ndarray.__len__``).
    * ``da.size`` is the total cell count (mirrors ``np.ndarray.size``).
    * ``iter(da)`` walks the leading axis: 1-D yields scalar cells;
      multi-d yields sub-``DistributionArray`` of shape ``batch_shape[1:]``.
    """

    def test_len_is_leading_axis_1d(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        assert len(da) == 4

    def test_len_is_leading_axis_multi_d(self):
        from probpipe import DistributionArray
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="g",
        )
        assert len(da) == 2  # leading axis only — matches np.zeros((2,3))

    def test_size_is_total_cells_1d(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        assert da.size == 4

    def test_size_is_total_cells_multi_d(self):
        from probpipe import DistributionArray
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="g",
        )
        assert da.size == 6  # prod(batch_shape) — matches np.zeros((2,3)).size

    def test_size_matches_numpy_convention(self):
        """``da.size`` follows the numpy / jax rule ``size == prod(shape)``."""
        from probpipe import DistributionArray
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="g",
        )
        ref = np.zeros(da.batch_shape)
        assert da.size == ref.size
        assert len(da) == len(ref)

    def test_iter_yields_scalar_cells_1d(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        items = list(da)
        assert items == comps  # 1-D iteration yields the cells

    def test_iter_yields_subarrays_multi_d(self):
        """Multi-d iteration yields leading-axis slices as sub-arrays
        (mirrors ``iter(np.zeros((2, 3)))``)."""
        from probpipe import DistributionArray
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="g",
        )
        items = list(da)
        assert len(items) == 2  # one per leading-axis index
        for sub in items:
            assert isinstance(sub, DistributionArray)
            assert sub.batch_shape == (3,)
        # First sub-array has loc=[0,1,2], second has loc=[3,4,5].
        for i, sub in enumerate(items):
            for j in range(3):
                assert float(sub[j].loc) == float(i * 3 + j)

    def test_zero_d_iter_raises_like_numpy(self):
        """``iter(np.zeros(()))`` raises ``TypeError``; so does
        ``iter(da)`` on a 0-d ``DistributionArray``. Reach for
        ``_flat_component`` / ``components`` for the single cell.
        Fully-joint GRF predictions with no extra batch axes return
        a 0-d DA."""
        from probpipe import DistributionArray, MultivariateNormal as _MVN
        da = DistributionArray.from_batched_params(
            _MVN,
            batch_shape=(),
            loc=jnp.zeros(3),
            cov=jnp.eye(3),
            name="m",
        )
        with pytest.raises(TypeError, match="0-d"):
            iter(da)
        # Single-cell access still works.
        assert da.size == 1
        assert isinstance(da._flat_component(0), _MVN)

    def test_zero_d_len_raises_like_numpy(self):
        """``len(np.zeros(()))`` raises ``TypeError``; so does
        ``len(da)`` on a 0-d ``DistributionArray``. ``da.size``
        still works."""
        from probpipe import DistributionArray, MultivariateNormal as _MVN
        da = DistributionArray.from_batched_params(
            _MVN,
            batch_shape=(),
            loc=jnp.zeros(3),
            cov=jnp.eye(3),
            name="m",
        )
        with pytest.raises(TypeError, match="0-d"):
            len(da)
        assert da.size == 1


# ---------------------------------------------------------------------------
# WF sweep — ops dispatched cell-by-cell
# ---------------------------------------------------------------------------


class TestSampleViaSweep:
    """``sample(da)`` vectorizes over the DistArray's batch_shape.

    Each cell is a scalar Distribution; ``sample(component, sample_shape)``
    returns a leaf-shaped array; ``_make_stack`` assembles them into a
    ``NumericRecordArray`` with ``batch_shape = da.batch_shape`` and
    per-field leaf shape equal to ``sample_shape + event_shape``.
    """

    def test_scalar_components_no_sample_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        s = sample(da)
        assert isinstance(s, NumericRecordArray)
        assert s.batch_shape == (4,)
        assert s["sample"].shape == (4,)

    def test_scalar_components_with_sample_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        s = sample(da, sample_shape=(7,))
        assert isinstance(s, NumericRecordArray)
        # batch_shape is the DistArray's own shape; sample_shape is leaf.
        assert s.batch_shape == (4,)
        assert s["sample"].shape == (4, 7)

    def test_components_drive_samples(self):
        """Per-cell mean of the 1000-sample draw concentrates at each
        component's own mean — confirms cell-by-cell dispatch."""
        comps = [Normal(loc=float(i) * 100, scale=1e-3, name=f"d{i}")
                 for i in range(3)]
        da = _make_distribution_array(comps)
        s = sample(da, sample_shape=(1000,))
        # batch_shape (3,), leaf (1000,) → field shape (3, 1000).
        means = s["sample"].mean(axis=-1)
        np.testing.assert_allclose(means, jnp.array([0.0, 100.0, 200.0]),
                                   atol=0.2)

    def test_multi_d_batch_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(6)]
        da = _make_distribution_array(comps, batch_shape=(2, 3))
        s = sample(da, sample_shape=(5,))
        assert s.batch_shape == (2, 3)
        assert s["sample"].shape == (2, 3, 5)

    def test_record_valued_components_scalar_sample(self):
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1e-3, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1e-3, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        s = sample(da)
        from probpipe import RecordArray
        assert isinstance(s, RecordArray)
        assert s.batch_shape == (3,)
        np.testing.assert_allclose(s["x"], [0.0, 1.0, 2.0], atol=1e-2)

    def test_record_valued_components_batched_sample(self):
        """Record-valued inner returns carry their own ``batch_shape`` —
        the outer sweep prepends ``da.batch_shape`` to it. Scalar
        components land in the trailing ``sample_shape`` as leaf; Record
        components land in the batch (because their per-cell return is
        already a batched ``NumericRecordArray``, not a raw array). Under
        direct vectorization both are the concatenation of outer sweep
        axes with the inner return's shape.
        """
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1e-3, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1e-3, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        s = sample(da, sample_shape=(5,))
        assert isinstance(s, NumericRecordArray)
        # sweep (3,) + inner batch (5,) → (3, 5); fields carry no leaf
        # (scalar Normals inside the Product).
        assert s.batch_shape == (3, 5)
        assert s["x"].shape == (3, 5)
        assert s["y"].shape == (3, 5)


class TestMeanVianSweep:
    def test_scalar_components_mean(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(4)]
        da = _make_distribution_array(comps)
        m = mean(da)
        assert isinstance(m, NumericRecordArray)
        assert m.batch_shape == (4,)
        assert m["mean"].shape == (4,)
        np.testing.assert_allclose(m["mean"], [0.0, 1.0, 2.0, 3.0])

    def test_multi_d_mean_shape(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(6)]
        da = _make_distribution_array(comps, batch_shape=(3, 2))
        m = mean(da)
        assert m.batch_shape == (3, 2)
        assert m["mean"].shape == (3, 2)

    def test_record_components_mean_is_recordarray(self):
        comps = [
            ProductDistribution(
                x=Normal(loc=float(i), scale=1.0, name=f"x{i}"),
                y=Normal(loc=-float(i), scale=1.0, name=f"y{i}"),
            )
            for i in range(3)
        ]
        da = _make_distribution_array(comps)
        m = mean(da)
        from probpipe import RecordArray
        assert isinstance(m, RecordArray)
        assert m.batch_shape == (3,)
        np.testing.assert_allclose(m["x"], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(m["y"], [0.0, -1.0, -2.0])


class TestVarianceViaSweep:
    def test_scalar_components_variance(self):
        comps = [Normal(loc=0.0, scale=float(i + 1), name=f"d{i}")
                 for i in range(3)]
        da = _make_distribution_array(comps)
        v = variance(da)
        assert isinstance(v, NumericRecordArray)
        assert v.batch_shape == (3,)
        assert v["variance"].shape == (3,)
        np.testing.assert_allclose(v["variance"], [1.0, 4.0, 9.0])


class TestLogProbViaSweep:
    """``log_prob`` sweeps over the DistArray; ``value`` is ``Any``-hinted
    so it passes through to each cell as-is (the sweep broadcasts the
    single value across cells). Per-cell-value evaluation against a
    batched input is a user-level composition (wrap in a small
    ``@workflow_function``) and is tested via the broader sweep tests,
    not here.
    """

    def test_same_value_all_cells(self):
        comps = [Normal(loc=float(i), scale=1.0, name=f"d{i}") for i in range(3)]
        da = _make_distribution_array(comps)
        # Single scalar value broadcasts to every cell.
        value = jnp.asarray(0.0)
        lp = log_prob(da, value=value)
        assert isinstance(lp, NumericRecordArray)
        assert lp.batch_shape == (3,)
        # Cell i evaluates Normal(i, 1) at 0 → gaussian log-density at
        # distance ``i`` from the mean.
        expected = jnp.array(
            [Normal(loc=float(i), scale=1.0, name=f"d{i}")._log_prob(0.0)
             for i in range(3)]
        )
        np.testing.assert_allclose(lp["log_prob"], expected, rtol=1e-5)


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


# ---------------------------------------------------------------------------
# Backend-delegated storage (PR-C.1 commit 3)
# ---------------------------------------------------------------------------


class TestBackendDelegatedStorage:
    """Tests for the ``_from_backend`` private constructor + lazy
    component materialisation. The factory entry point
    (``from_batched_params``) lands in commit 4; this commit pins the
    storage refactor in isolation.
    """

    def _make_backend(self, n=5):
        from probpipe import Normal
        return Normal._make_array_backend(
            name="x",
            batch_shape=(n,),
            loc=jnp.arange(float(n)),
            scale=jnp.ones(n),
        )

    def test_from_backend_returns_distribution_array(self):
        from probpipe import DistributionArray
        backend = self._make_backend(n=5)
        da = DistributionArray._from_backend(backend, name="batched_x")
        assert isinstance(da, DistributionArray)
        assert da._backend is backend
        assert da.batch_shape == (5,)
        assert da.event_shape == ()
        assert len(da) == 5
        assert da.name == "batched_x"

    def test_from_backend_does_not_materialise_components_eagerly(self):
        from probpipe import DistributionArray
        backend = self._make_backend(n=5)
        da = DistributionArray._from_backend(backend, name="x")
        # Components stay None until first .components access; the
        # storage refactor's whole point is lazy materialisation.
        assert da._components is None

    def test_indexing_routes_through_backend_cell(self):
        from probpipe import DistributionArray, Normal
        backend = self._make_backend(n=5)
        da = DistributionArray._from_backend(backend, name="x")
        # Each int access fabricates fresh; identity differs.
        a = da[2]
        b = da[2]
        assert isinstance(a, Normal)
        assert a is not b
        # Per-cell params correctly sliced.
        assert float(a.loc) == 2.0
        # Components stayed None — int indexing doesn't materialise the
        # eager tuple.
        assert da._components is None

    def test_flat_component_routes_through_backend_cell(self):
        from probpipe import DistributionArray, Normal
        backend = self._make_backend(n=4)
        da = DistributionArray._from_backend(backend, name="x")
        for i in range(4):
            cell = da._flat_component(i)
            assert isinstance(cell, Normal)
            assert float(cell.loc) == float(i)
        assert da._components is None  # still lazy

    def test_iteration_lazy_via_backend(self):
        from probpipe import DistributionArray, Normal
        backend = self._make_backend(n=4)
        da = DistributionArray._from_backend(backend, name="x")
        cells = list(da)
        assert len(cells) == 4
        for i, cell in enumerate(cells):
            assert isinstance(cell, Normal)
            assert float(cell.loc) == float(i)
        # Iteration via backend.cell does NOT cache into _components;
        # only .components does that.
        assert da._components is None

    def test_components_property_materialises_lazily_and_caches(self):
        from probpipe import DistributionArray
        backend = self._make_backend(n=3)
        da = DistributionArray._from_backend(backend, name="x")
        c1 = da.components
        c2 = da.components
        assert isinstance(c1, tuple)
        assert len(c1) == 3
        # Cached: same tuple identity returned.
        assert c1 is c2
        # And now _components is populated.
        assert da._components is c1

    def test_slice_indexing_materialises_components(self):
        from probpipe import DistributionArray
        backend = self._make_backend(n=5)
        da = DistributionArray._from_backend(backend, name="x")
        sub = da[1:4]
        # Slicing forces materialisation (rare path).
        assert da._components is not None
        assert isinstance(sub, DistributionArray)
        assert len(sub) == 3

    def test_repr_indicates_backend_mode(self):
        from probpipe import DistributionArray
        backend = self._make_backend(n=3)
        backed = DistributionArray._from_backend(backend, name="x")
        assert "backend=True" in repr(backed)

    def test_literal_path_unchanged(self):
        """Construction via the explicit components list still works
        identically — no _backend, eager _components tuple."""
        from probpipe import DistributionArray, Normal
        comps = [Normal(loc=float(i), scale=1.0, name=f"c_{i}") for i in range(3)]
        da = DistributionArray(comps, name="literal")
        assert da._backend is None
        assert da._components == tuple(comps)
        assert da[1] is comps[1]
        assert da.components is da._components

    def test_multi_d_batch_shape_via_backend(self):
        from probpipe import DistributionArray, Normal
        loc = jnp.arange(6.0).reshape(2, 3)
        backend = Normal._make_array_backend(
            name="g", batch_shape=(2, 3), loc=loc, scale=1.0,
        )
        da = DistributionArray._from_backend(backend, name="g")
        assert da.batch_shape == (2, 3)
        # Row-major flat index 4 -> (1, 1) -> loc=4.0
        assert float(da[1, 1].loc) == 4.0
        assert float(da._flat_component(4).loc) == 4.0


# ---------------------------------------------------------------------------
# from_batched_params factory (PR-C.1 commit 4)
# ---------------------------------------------------------------------------


class TestFromBatchedParams:
    """Public entry point that dispatches on ``SupportsArrayBackend``."""

    def test_normal_uses_tfp_backend(self):
        from probpipe import Normal, DistributionArray
        from probpipe.distributions._tfp_base import _TFPArrayBackend
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.arange(5.0), scale=1.0, name="x",
        )
        assert isinstance(da, DistributionArray)
        assert isinstance(da._backend, _TFPArrayBackend)
        assert da.batch_shape == (5,)
        assert da.event_shape == ()
        assert len(da) == 5

    def test_per_cell_name_auto_suffix(self):
        from probpipe import Normal, DistributionArray
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.arange(3.0), scale=jnp.ones(3), name="weights",
        )
        assert da[0].name == "weights_0"
        assert da[1].name == "weights_1"
        assert da[2].name == "weights_2"

    def test_per_cell_params_correctly_sliced(self):
        from probpipe import Normal, DistributionArray
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.arange(4.0), scale=jnp.linspace(0.1, 0.4, 4),
            name="x",
        )
        for i in range(4):
            cell = da[i]
            assert float(cell.loc) == float(i)
            assert float(cell.scale) == pytest.approx(0.1 + 0.1 * i, abs=1e-6)

    def test_scalar_param_broadcasts_through_cells(self):
        from probpipe import Normal, DistributionArray
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.zeros(3), scale=2.5, name="x",
        )
        for i in range(3):
            assert float(da[i].scale) == 2.5

    def test_inferred_batch_shape_uses_broadcast(self):
        from probpipe import Normal, DistributionArray
        # Both arrays imply batch_shape=(4,) — broadcast convention.
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.arange(4.0), scale=jnp.ones(4), name="x",
        )
        assert da.batch_shape == (4,)

    def test_explicit_batch_shape_honored(self):
        from probpipe import Normal, DistributionArray
        da = DistributionArray.from_batched_params(
            Normal,
            batch_shape=(2, 3),
            loc=jnp.arange(6.0).reshape(2, 3),
            scale=1.0,
            name="g",
        )
        assert da.batch_shape == (2, 3)
        # Row-major: (1, 2) -> flat 5 -> loc=5.0, name="g_5".
        assert float(da[1, 2].loc) == 5.0
        assert da[1, 2].name == "g_5"

    def test_no_array_params_requires_explicit_batch_shape(self):
        from probpipe import Normal, DistributionArray
        with pytest.raises(ValueError, match="batch_shape"):
            DistributionArray.from_batched_params(
                Normal, loc=0.0, scale=1.0, name="x",
            )

    def test_multivariate_normal_via_factory(self):
        """MVN-style classes need explicit ``batch_shape`` because
        their per-param event ranks differ (``loc`` is
        ``(*batch, d)``, ``scale_tril`` is ``(*batch, d, d)``)."""
        from probpipe import MultivariateNormal, DistributionArray
        from probpipe.distributions._tfp_base import _TFPArrayBackend
        d = 3
        da = DistributionArray.from_batched_params(
            MultivariateNormal,
            batch_shape=(2,),
            loc=jnp.zeros((2, d)),
            scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
            name="z",
        )
        assert isinstance(da._backend, _TFPArrayBackend)
        assert da.batch_shape == (2,)
        assert da.event_shape == (d,)
        assert da[0].event_shape == (d,)

    def test_inference_punts_on_event_rank_mismatch(self):
        """Without explicit ``batch_shape``, MVN-style heterogeneous
        event-rank params raise a clear ``ValueError``."""
        from probpipe import MultivariateNormal, DistributionArray
        d = 3
        with pytest.raises(ValueError, match="batch_shape"):
            DistributionArray.from_batched_params(
                MultivariateNormal,
                loc=jnp.zeros((2, d)),
                scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
                name="z",
            )

    def test_non_protocol_class_falls_back_to_literal(self):
        """A Distribution subclass that doesn't implement
        ``SupportsArrayBackend`` gets the literal-array fallback path:
        eager construction, no backend.
        """
        from probpipe import DistributionArray
        from probpipe.core._distribution_base import Distribution

        class MyDist(Distribution):
            def __init__(self, value, *, name):
                self._value = float(value)
                super().__init__(name=name)

            @property
            def event_shape(self):
                return ()

            @property
            def batch_shape(self):
                return ()

        da = DistributionArray.from_batched_params(
            MyDist, value=jnp.arange(3.0), name="m",
        )
        # Fallback path: literal components, no backend.
        assert da._backend is None
        assert da._components is not None
        assert isinstance(da[0], MyDist)
        assert da[0]._value == 0.0
        assert da[2]._value == 2.0
        assert da[1].name == "m_1"

    def test_factory_results_match_native_tfp(self):
        """Result of ``from_batched_params(Normal, loc=arr, scale=...)``
        produces samples / means / variances matching ``tfd.Normal``
        constructed natively with the same batched params.

        Pre-PR-C.2 this test compared to ``Normal(loc=arr, scale=...)``
        directly; PR-C.2 rejects that form, so the comparison goes
        against TFP's own batched form (which is what
        ``_TFPArrayBackend`` wraps internally anyway).
        """
        from probpipe import Normal, DistributionArray
        import tensorflow_probability.substrates.jax.distributions as tfd
        loc = jnp.array([0.5, -1.2, 3.7, 0.0])
        scale = jnp.array([0.1, 0.2, 0.3, 0.4])

        native = tfd.Normal(loc=loc, scale=scale)
        da = DistributionArray.from_batched_params(
            Normal, loc=loc, scale=scale, name="x",
        )
        # Sample shapes match.
        key = jax.random.PRNGKey(42)
        native_samples = native.sample(seed=key)
        da_samples = da._backend._sample(key)
        np.testing.assert_allclose(
            np.asarray(native_samples), np.asarray(da_samples)
        )
        # Mean / variance match.
        np.testing.assert_allclose(
            np.asarray(native.mean()), np.asarray(da._backend._mean())
        )
        np.testing.assert_allclose(
            np.asarray(native.variance()),
            np.asarray(da._backend._variance()),
        )

    def test_iteration_matches_indexing(self):
        from probpipe import Normal, DistributionArray
        da = DistributionArray.from_batched_params(
            Normal, loc=jnp.arange(4.0), scale=jnp.ones(4), name="x",
        )
        for i, cell in enumerate(da):
            assert float(cell.loc) == float(da[i].loc)
            assert cell.name == da[i].name


# ---------------------------------------------------------------------------
# Distribution.from_batched_params alias (PR-C.1 commit 5)
# ---------------------------------------------------------------------------


class TestDistributionFromBatchedParamsAlias:
    """Ergonomic per-class alias on Distribution[T]."""

    def test_alias_dispatches_to_distribution_array_factory(self):
        from probpipe import Normal, DistributionArray
        from probpipe.distributions._tfp_base import _TFPArrayBackend

        da = Normal.from_batched_params(
            loc=jnp.arange(5.0), scale=1.0, name="x",
        )
        assert isinstance(da, DistributionArray)
        assert isinstance(da._backend, _TFPArrayBackend)
        assert da.batch_shape == (5,)
        assert da[0].name == "x_0"

    def test_alias_matches_classmethod_call(self):
        from probpipe import Normal, DistributionArray
        loc = jnp.arange(4.0)
        scale = jnp.linspace(0.1, 0.4, 4)

        via_alias = Normal.from_batched_params(
            loc=loc, scale=scale, name="x",
        )
        via_factory = DistributionArray.from_batched_params(
            Normal, loc=loc, scale=scale, name="x",
        )
        # Same backend type, same per-cell parameters / names.
        assert type(via_alias._backend) is type(via_factory._backend)
        for i in range(4):
            a = via_alias[i]
            f = via_factory[i]
            assert a.name == f.name
            assert float(a.loc) == float(f.loc)
            assert float(a.scale) == float(f.scale)

    def test_alias_inherited_by_all_distribution_subclasses(self):
        from probpipe.core._distribution_base import Distribution
        from probpipe import (
            Normal, Beta, Gamma, MultivariateNormal,
            EmpiricalDistribution,
        )
        # Every Distribution subclass inherits the alias.
        for cls in (Distribution, Normal, Beta, Gamma,
                    MultivariateNormal, EmpiricalDistribution):
            assert hasattr(cls, "from_batched_params")
            assert callable(cls.from_batched_params)

    def test_alias_with_explicit_batch_shape(self):
        from probpipe import MultivariateNormal
        d = 3
        da = MultivariateNormal.from_batched_params(
            batch_shape=(2,),
            loc=jnp.zeros((2, d)),
            scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
            name="z",
        )
        assert da.batch_shape == (2,)
        assert da.event_shape == (d,)

    def test_alias_falls_back_for_non_protocol_class(self):
        """The alias inherits the factory's protocol-vs-fallback
        dispatch, so non-protocol Distribution subclasses also get the
        literal-array fallback when invoking the alias."""
        from probpipe.core._distribution_base import Distribution

        class MyDist(Distribution):
            def __init__(self, value, *, name):
                self._value = float(value)
                super().__init__(name=name)

            @property
            def event_shape(self):
                return ()

            @property
            def batch_shape(self):
                return ()

        da = MyDist.from_batched_params(value=jnp.arange(3.0), name="m")
        assert da._backend is None
        assert isinstance(da[0], MyDist)
        assert da[1].name == "m_1"
