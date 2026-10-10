"""Tests for ``_TFPArrayBackend``.

The backend is the fused storage of a family's batched parameters.
These tests pin the backend's behaviour in isolation:

* Vectorised ops (``_sample`` / ``_log_prob`` / ``_mean`` / ``_variance``)
  are numerically equivalent to constructing the same TFP-batched
  distribution directly.
* Scalar parameters broadcast across the declared ``batch_shape``.
* Mismatched ``batch_shape`` declarations are rejected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from probpipe import Beta, Gamma, MultivariateNormal, Normal
from probpipe.families._backend import _TFPArrayBackend

# ---------------------------------------------------------------------------
# Construction + minimum surface
# ---------------------------------------------------------------------------


class TestMakeArrayBackendConstruction:
    def test_normal_returns_tfp_array_backend(self):
        backend = Normal._make_array_backend(
            component="x",
            batch_shape=(5,),
            loc=jnp.arange(5.0),
            scale=1.0,
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (5,)
        assert backend.event_shape == ()

    def test_beta_inherits_make_array_backend(self):
        backend = Beta._make_array_backend(
            component="b",
            batch_shape=(3,),
            alpha=jnp.array([1.0, 2.0, 3.0]),
            beta=jnp.array([1.0, 1.0, 1.0]),
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (3,)

    def test_gamma_inherits_make_array_backend(self):
        backend = Gamma._make_array_backend(
            component="g",
            batch_shape=(4,),
            concentration=jnp.array([1.0, 2.0, 3.0, 4.0]),
            rate=1.0,
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (4,)

    def test_mvn_inherits_make_array_backend(self):
        d = 3
        backend = MultivariateNormal._make_array_backend(
            component="z",
            batch_shape=(2,),
            loc=jnp.zeros((2, d)),
            scale_tril=jnp.broadcast_to(jnp.eye(d), (2, d, d)),
        )
        assert isinstance(backend, _TFPArrayBackend)
        assert backend.batch_shape == (2,)
        assert backend.event_shape == (d,)

    def test_required_minimum_surface(self):
        backend = Normal._make_array_backend(
            component="x",
            batch_shape=(2,),
            loc=jnp.zeros(2),
            scale=1.0,
        )
        for attr in (
            "batch_shape",
            "event_shape",
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
                component="x",
                batch_shape=(5,),
                loc=jnp.zeros(3),  # actually batch_shape=(3,)
                scale=1.0,
            )


# ---------------------------------------------------------------------------
# Vectorised ops match TFP-batched native
# ---------------------------------------------------------------------------


class TestBatchedOpsMatchTFPNative:
    def _make_pair(self, loc, scale):
        backend = Normal._make_array_backend(
            component="x",
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
            component="x",
            batch_shape=(2, 3),
            loc=loc,
            scale=1.0,
        )
        samples = backend._sample(jax.random.PRNGKey(0))
        assert samples.shape == (2, 3)


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------


class TestPytreeRegistration:
    """``_TFPArrayBackend`` is registered as a JAX pytree node so it
    can flow through ``jit`` / ``vmap`` / ``tree_map``.

    Children are the batched parameter values (the JAX-array leaves
    the user passed); aux carries the distribution class, component,
    declared ``batch_shape``, and parameter keys. Reconstruction
    rebuilds the wrapped ``_batched_dist`` from the parameter dict.
    """

    def _backend(self):
        return Normal._make_array_backend(
            component="x",
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
        # Behavioural equivalence: same mean and variance.
        np.testing.assert_allclose(np.asarray(rebuilt._mean()), np.asarray(backend._mean()))
        np.testing.assert_allclose(np.asarray(rebuilt._variance()), np.asarray(backend._variance()))

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
            component="x",
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
# Scalar parameter broadcasting
# ---------------------------------------------------------------------------


class TestScalarParamBroadcasting:
    """Scalar parameters paired with an explicit ``batch_shape``
    broadcast across every cell. Without this, the sanity check on
    ``_TFPArrayBackend.__init__`` rejects the configuration with a
    cryptic shape-mismatch error.
    """

    def test_all_scalar_params_with_explicit_shape(self):
        """All-scalar params + ``batch_shape=(5,)`` produce five
        identical Normals."""
        backend = Normal._make_array_backend(
            component="x",
            batch_shape=(5,),
            loc=0.0,
            scale=1.0,
        )
        assert backend.batch_shape == (5,)
        np.testing.assert_allclose(np.asarray(backend._mean()), np.zeros(5))
        np.testing.assert_allclose(np.asarray(backend._variance()), np.ones(5))

    def test_scalar_loc_array_scale(self):
        """``loc`` scalar + ``scale`` array broadcasts ``loc`` to
        match the batch axis."""
        backend = Normal._make_array_backend(
            component="x",
            batch_shape=(3,),
            loc=0.0,
            scale=jnp.array([0.1, 0.2, 0.3]),
        )
        np.testing.assert_allclose(np.asarray(backend._mean()), np.zeros(3))
        np.testing.assert_allclose(
            np.asarray(backend._variance()), np.array([0.1, 0.2, 0.3]) ** 2, rtol=1e-6
        )

    def test_multi_d_batch_with_scalar_params(self):
        backend = Normal._make_array_backend(
            component="x",
            batch_shape=(2, 3),
            loc=0.0,
            scale=1.0,
        )
        assert backend.batch_shape == (2, 3)
        np.testing.assert_allclose(np.asarray(backend._mean()), np.zeros((2, 3)))
