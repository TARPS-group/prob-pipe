"""Tests for ``DistributionArray``.

A ``DistributionArray`` is ``Array[Distribution]`` — an ordered
collection of scalar distributions indexed by a (multi-d)
``batch_shape``. Vectorized ops live at the ``WorkflowFunction``
sweep layer, not on the DistArray itself. This file covers:

- Construction + invariants + container surface (indexing, iteration,
  shape, slicing, ``.n``).
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
        assert "n=3" in r

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
        assert da.n == 5
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
        assert sub.n == 3

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
        assert da.n == 6
        # Row-major flat index 4 -> (1, 1) -> loc=4.0
        assert float(da[1, 1].loc) == 4.0
        assert float(da._flat_component(4).loc) == 4.0
