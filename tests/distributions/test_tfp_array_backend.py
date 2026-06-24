"""Tests for ``_TFPArrayBackend`` (PR-C.1 commit 2).

The backend is the fused-storage substrate that
:class:`~probpipe.DistributionArray` will dispatch onto in commits 3-4.
These tests pin the backend's behaviour in isolation:

* Per-cell materialisation (``cell(i)``) returns fresh scalar
  distributions with sliced parameters.
* Vectorised ops (``_sample`` / ``_log_prob`` / ``_mean`` / ``_variance``)
  are numerically equivalent to constructing the same TFP-batched
  distribution directly.
* Multi-d ``batch_shape`` works with both flat-int and tuple indexing.
* Scalar parameters that broadcast across the batch are passed through
  ``cell(i)`` unchanged.
* Mismatched ``batch_shape`` declarations are rejected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import Beta, Gamma, MultivariateNormal, Normal
from probpipe.distributions._tfp_base import _TFPArrayBackend

# ---------------------------------------------------------------------------
# Construction + minimum surface
# ---------------------------------------------------------------------------


class TestMakeArrayBackendConstruction:
    def test_normal_returns_tfp_array_backend(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(5,),
            loc=jnp.arange(5.0),
            scale=1.0,
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (5,)
        assert backend.event_shape == ()

    def test_beta_inherits_make_array_backend(self):
        backend = Beta._make_array_backend(
            name="b",
            batch_shape=(3,),
            alpha=jnp.array([1.0, 2.0, 3.0]),
            beta=jnp.array([1.0, 1.0, 1.0]),
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (3,)

    def test_gamma_inherits_make_array_backend(self):
        backend = Gamma._make_array_backend(
            name="g",
            batch_shape=(4,),
            concentration=jnp.array([1.0, 2.0, 3.0, 4.0]),
            rate=1.0,
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (4,)

    def test_mvn_inherits_make_array_backend(self):
        d = 3
        backend = MultivariateNormal._make_array_backend(
            name="z",
            batch_shape=(2,),
            loc=jnp.zeros((2, d)),
            scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (2,)
        assert backend.event_shape == (d,)

    def test_required_minimum_surface(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2,),
            loc=jnp.zeros(2),
            scale=1.0,
        )
        for attr in (
            "batch_shape",
            "event_shape",
            "cell",
            "_sample",
            "_log_prob",
            "_mean",
            "_variance",
            "_cov",
        ):
            assert hasattr(backend, attr), f"_TFPArrayBackend missing required attr {attr!r}"

    def test_mismatched_batch_shape_rejected(self):
        """Declaring ``batch_shape=(5,)`` with params that broadcast to
        ``(3,)`` raises ``ValueError`` at backend construction."""
        with pytest.raises(ValueError, match="batch_shape"):
            Normal._make_array_backend(
                name="x",
                batch_shape=(5,),
                loc=jnp.zeros(3),  # actually batch_shape=(3,)
                scale=1.0,
            )


# ---------------------------------------------------------------------------
# Per-cell materialisation
# ---------------------------------------------------------------------------


class TestCellMaterialisation:
    def test_cell_returns_fresh_scalar_normal(self):
        loc = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(5,),
            loc=loc,
            scale=1.0,
        )
        cell0 = backend.cell(0)
        cell2 = backend.cell(2)
        assert isinstance(cell0, Normal)
        assert isinstance(cell2, Normal)
        # Per-cell parameters are correctly sliced.
        assert float(cell0.loc) == 0.0
        assert float(cell2.loc) == 2.0
        # Per-cell scalar `scale` is broadcast through unchanged.
        assert float(cell0.scale) == 1.0
        assert float(cell2.scale) == 1.0

    def test_cell_value_correctness_independent_of_call(self):
        """Each ``cell(i)`` call returns a distribution that holds
        exactly its own per-cell parameters — even when called
        repeatedly with different indices.

        Pins observable behaviour rather than instance identity:
        a future caching optimisation could legitimately deduplicate
        but must not corrupt per-cell values.
        """
        loc = jnp.array([10.0, 20.0, 30.0])
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=loc,
            scale=1.0,
        )
        a = backend.cell(0)
        b = backend.cell(2)
        c = backend.cell(0)
        assert float(a.loc) == 10.0
        assert float(b.loc) == 30.0
        assert float(c.loc) == 10.0  # second cell(0) still gets loc[0]

    def test_cell_name_auto_suffixes(self):
        backend = Normal._make_array_backend(
            name="weights",
            batch_shape=(3,),
            loc=jnp.zeros(3),
            scale=jnp.ones(3),
        )
        for i in range(3):
            assert backend.cell(i).name == f"weights_{i}"

    def test_cell_returns_unbatched_distribution(self):
        """Cells materialise as scalar distributions
        (``tfd batch_shape == ()``)."""
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(4,),
            loc=jnp.arange(4.0),
            scale=jnp.ones(4),
        )
        for i in range(4):
            cell = backend.cell(i)
            assert tuple(cell._tfp_dist.batch_shape) == ()

    def test_cell_negative_index_rejected(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.zeros(3),
            scale=1.0,
        )
        with pytest.raises(IndexError):
            backend.cell(-1)
        with pytest.raises(IndexError):
            backend.cell(3)

    def test_cell_with_mvn_preserves_event_axis(self):
        d = 3
        loc = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scale_tril = jnp.broadcast_to(jnp.eye(d), (2, d, d))
        backend = MultivariateNormal._make_array_backend(
            name="z",
            batch_shape=(2,),
            loc=loc,
            scale_tril=scale_tril,
        )
        cell0 = backend.cell(0)
        cell1 = backend.cell(1)
        np.testing.assert_allclose(np.asarray(cell0.loc), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(cell1.loc), [4.0, 5.0, 6.0])
        # event_shape preserved on the per-cell scalar.
        assert cell0.event_shape == (d,)


# ---------------------------------------------------------------------------
# Multi-d batching
# ---------------------------------------------------------------------------


class TestMultiDimensionalBatch:
    def test_int_index_maps_to_row_major_position(self):
        """Flat ``int`` indices unravel row-major over ``batch_shape``."""
        loc = jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2, 3),
            loc=loc,
            scale=1.0,
        )
        # Row-major: index 0 -> (0, 0), index 4 -> (1, 1), index 5 -> (1, 2).
        assert float(backend.cell(0).loc) == 10.0
        assert float(backend.cell(4).loc) == 21.0
        assert float(backend.cell(5).loc) == 22.0

    def test_tuple_index_axis_aligned(self):
        loc = jnp.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2, 3),
            loc=loc,
            scale=1.0,
        )
        assert float(backend.cell((0, 0)).loc) == 10.0
        assert float(backend.cell((1, 2)).loc) == 22.0

    def test_int_and_tuple_indices_match(self):
        loc = jnp.arange(12.0).reshape(3, 4)
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3, 4),
            loc=loc,
            scale=1.0,
        )
        for flat in range(12):
            multi = np.unravel_index(flat, (3, 4))
            assert float(backend.cell(flat).loc) == float(
                backend.cell(tuple(int(x) for x in multi)).loc
            )

    def test_tuple_wrong_rank_rejected(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2, 3),
            loc=jnp.zeros((2, 3)),
            scale=1.0,
        )
        with pytest.raises(IndexError):
            backend.cell((0,))


# ---------------------------------------------------------------------------
# Vectorised ops match TFP-batched native
# ---------------------------------------------------------------------------


class TestBatchedOpsMatchTFPNative:
    def _make_pair(self, loc, scale):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=tuple(
                jnp.broadcast_shapes(
                    jnp.asarray(loc).shape,
                    jnp.asarray(scale).shape,
                )
            ),
            loc=loc,
            scale=scale,
        )
        tfp_native = tfd.Normal(loc=jnp.asarray(loc), scale=jnp.asarray(scale))
        return backend, tfp_native

    def test_sample_shape_matches_native(self):
        backend, native = self._make_pair(jnp.arange(5.0), 1.0)
        key = jax.random.PRNGKey(0)
        b_samples = backend._sample(key)
        n_samples = native.sample(seed=key)
        assert b_samples.shape == n_samples.shape == (5,)

    def test_sample_with_sample_shape_matches_native(self):
        backend, native = self._make_pair(jnp.arange(3.0), jnp.ones(3))
        key = jax.random.PRNGKey(7)
        b_samples = backend._sample(key, sample_shape=(10,))
        n_samples = native.sample(seed=key, sample_shape=(10,))
        np.testing.assert_allclose(np.asarray(b_samples), np.asarray(n_samples))

    def test_log_prob_matches_native(self):
        backend, native = self._make_pair(jnp.arange(4.0), 1.5)
        x = jnp.array([0.5, 1.0, 2.5, 3.5])
        np.testing.assert_allclose(
            np.asarray(backend._log_prob(x)),
            np.asarray(native.log_prob(x)),
            rtol=1e-6,
        )

    def test_mean_variance_match_native(self):
        backend, native = self._make_pair(jnp.array([1.0, 2.0, 3.0]), jnp.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(
            np.asarray(backend._mean()),
            np.asarray(native.mean()),
        )
        np.testing.assert_allclose(
            np.asarray(backend._variance()),
            np.asarray(native.variance()),
        )

    def test_multi_d_batch_sample_shape(self):
        loc = jnp.zeros((2, 3))
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2, 3),
            loc=loc,
            scale=1.0,
        )
        samples = backend._sample(jax.random.PRNGKey(0))
        assert samples.shape == (2, 3)


# ---------------------------------------------------------------------------
# Scalar broadcast in cell()
# ---------------------------------------------------------------------------


class TestScalarBroadcast:
    def test_scalar_param_passes_through_cell(self):
        """A param given as a Python float / 0-D array (broadcast across
        every cell) is preserved unchanged in ``cell(i)``."""
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(4,),
            loc=jnp.arange(4.0),
            scale=2.5,  # scalar
        )
        for i in range(4):
            cell = backend.cell(i)
            assert float(cell.scale) == 2.5

    def test_zero_d_jax_array_param_passes_through(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.arange(3.0),
            scale=jnp.array(0.5),
        )
        for i in range(3):
            assert float(backend.cell(i).scale) == 0.5


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


class TestPytreeRegistration:
    """``_TFPArrayBackend`` is registered as a JAX pytree node so it
    can flow through ``jit`` / ``vmap`` / ``tree_map``.

    Children are the batched parameter values (the JAX-array leaves
    the user passed); aux carries the distribution class, name,
    declared ``batch_shape``, and parameter keys. Reconstruction
    rebuilds the wrapped ``_batched_dist`` from the parameter dict.
    """

    def _backend(self):
        return Normal._make_array_backend(
            name="x",
            batch_shape=(5,),
            loc=jnp.arange(5.0),
            scale=1.0,
        )

    def test_flatten_yields_batched_params_in_insertion_order(self):
        backend = self._backend()
        leaves, _treedef = jax.tree_util.tree_flatten(backend)
        # Two children: ``loc`` then ``scale`` — insertion order from
        # ``Normal._make_array_backend``. Both end up as ``(5,)``-
        # shaped arrays after the constructor's scalar broadcast.
        assert len(leaves) == 2
        assert hasattr(leaves[0], "shape") and leaves[0].shape == (5,)
        assert hasattr(leaves[1], "shape") and leaves[1].shape == (5,)
        np.testing.assert_allclose(np.asarray(leaves[1]), np.ones(5))

    def test_round_trip_preserves_behaviour(self):
        backend = self._backend()
        leaves, treedef = jax.tree_util.tree_flatten(backend)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, _TFPArrayBackend)
        assert rebuilt.batch_shape == backend.batch_shape
        # Behavioural equivalence: same mean, same per-cell scalar.
        np.testing.assert_allclose(np.asarray(rebuilt._mean()), np.asarray(backend._mean()))
        assert float(rebuilt.cell(2).loc) == float(backend.cell(2).loc)

    def test_jit_through_backend(self):
        """A ``jit``-compiled function that consumes the backend
        traces cleanly: the backend is a valid pytree node, so JAX
        threads its leaves through the compilation."""

        @jax.jit
        def fn(b):
            return b._mean()

        backend = self._backend()
        out = fn(backend)
        np.testing.assert_allclose(np.asarray(out), np.arange(5.0))

    def test_tree_map_replaces_leaves(self):
        """``tree_map`` over the backend rebuilds it with transformed
        leaves; the surrounding aux is preserved."""
        backend = self._backend()
        scaled = jax.tree_util.tree_map(lambda x: jnp.asarray(x) * 2.0, backend)
        assert isinstance(scaled, _TFPArrayBackend)
        assert scaled.batch_shape == backend.batch_shape
        # ``loc`` doubled, ``scale`` doubled; per-cell mean = 2 * loc.
        np.testing.assert_allclose(np.asarray(scaled._mean()), 2.0 * np.arange(5.0))

    def test_vmap_over_leading_batch(self):
        """vmap-able through ``tree_map`` lifting a fresh axis on each
        leaf, then calling the backend's vectorised op under the lift."""
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.zeros(3),
            scale=jnp.ones(3),
        )
        # Stack two backends along a new leading axis via tree_map.
        stacked = jax.tree_util.tree_map(
            lambda x: jnp.stack([jnp.asarray(x), jnp.asarray(x) + 10.0]),
            backend,
        )
        # vmap pulls the new axis out and runs ``_mean`` per slice.
        means = jax.vmap(lambda b: b._mean())(stacked)
        # Two slices: original (zeros) and shifted by 10.
        np.testing.assert_allclose(
            np.asarray(means), np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        )


# ---------------------------------------------------------------------------
# Scalar parameter broadcasting (review finding C4)
# ---------------------------------------------------------------------------


class TestScalarParamBroadcasting:
    """Scalar parameters paired with an explicit ``batch_shape``
    broadcast across every cell. Without this, the sanity check on
    ``_TFPArrayBackend.__init__`` rejects the configuration with a
    cryptic shape-mismatch error.
    """

    def test_all_scalar_params_with_explicit_shape(self):
        """All-scalar params + ``batch_shape=(5,)`` produces five
        identical Normals."""
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(5,),
            loc=0.0,
            scale=1.0,
        )
        assert backend.batch_shape == (5,)
        for i in range(5):
            cell = backend.cell(i)
            assert float(cell.loc) == 0.0
            assert float(cell.scale) == 1.0

    def test_scalar_loc_array_scale(self):
        """``loc`` scalar + ``scale`` array broadcasts ``loc`` to
        match the batch axis."""
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=0.0,
            scale=jnp.array([0.1, 0.2, 0.3]),
        )
        for i in range(3):
            cell = backend.cell(i)
            assert float(cell.loc) == 0.0

    def test_multi_d_batch_with_scalar_params(self):
        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(2, 3),
            loc=0.0,
            scale=1.0,
        )
        assert backend.batch_shape == (2, 3)
        for i in range(6):
            assert float(backend.cell(i).loc) == 0.0


# ---------------------------------------------------------------------------
# Backend-derived approximation status (review finding C2)
# ---------------------------------------------------------------------------


class TestBackendApproximate:
    """``_from_backend`` propagates ``is_approximate`` from the
    backend rather than hardcoding ``False``. This is forward-
    compatible with a future ``_RecordArrayBackend`` over an
    empirical source whose samples are an approximation.
    """

    def test_tfp_backend_is_exact(self):
        """The shipping ``_TFPArrayBackend`` has no
        ``is_approximate`` attribute, so the ``DistributionArray``
        defaults to exact (``False``)."""
        from probpipe import DistributionArray

        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.zeros(3),
            scale=1.0,
        )
        da = DistributionArray._from_backend(backend, name="x")
        assert da.is_approximate is False

    def test_approximate_backend_propagates(self):
        """A backend reporting ``is_approximate=True`` flows through
        to the assembled DistributionArray."""
        from probpipe import DistributionArray

        class _ApproxBackend:
            batch_shape = (3,)
            event_shape = ()
            is_approximate = True

            def cell(self, i):
                return Normal(loc=0.0, scale=1.0, name=f"x_{i}")

        da = DistributionArray._from_backend(_ApproxBackend(), name="x")
        assert da.is_approximate is True


# ---------------------------------------------------------------------------
# Negative-index alignment (review finding C5)
# ---------------------------------------------------------------------------


class TestFlatComponentNegativeRejection:
    """``_flat_component`` rejects negatives in *both* the backend
    and literal-array paths. ``__getitem__`` wraps user-facing
    ``da[-1]`` before any flat-index call site sees it, so internal
    sweep code (which already only passes non-negatives) and
    direct ``_flat_component(-1)`` calls behave consistently.
    """

    def test_backed_path_rejects_negative(self):
        from probpipe import DistributionArray

        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.zeros(3),
            scale=1.0,
        )
        da = DistributionArray._from_backend(backend, name="x")
        with pytest.raises(IndexError):
            da._flat_component(-1)

    def test_literal_path_rejects_negative(self):
        """The literal path used to silently allow Python tuple
        wraparound; align with the backed path."""
        from probpipe import DistributionArray

        comps = [Normal(loc=float(i), scale=1.0, name=f"c_{i}") for i in range(3)]
        da = DistributionArray(comps, name="x")
        with pytest.raises(IndexError):
            da._flat_component(-1)

    def test_user_facing_da_minus_one_still_works(self):
        """``da[-1]`` continues to work via ``__getitem__`` wrap."""
        from probpipe import DistributionArray

        backend = Normal._make_array_backend(
            name="x",
            batch_shape=(3,),
            loc=jnp.array([10.0, 20.0, 30.0]),
            scale=1.0,
        )
        da = DistributionArray._from_backend(backend, name="x")
        assert float(da[-1].loc) == 30.0
        assert float(da[-2].loc) == 20.0
