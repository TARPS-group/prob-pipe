"""Tests for probpipe._weights: Weights class and utility functions."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from probpipe._weights import (
    Weights,
    _validate_to_log_weights,
    normalize_weights,
    normalized_log_weights,
    uniform_weights,
    weighted_mean,
    weighted_variance,
    weighted_covariance,
    weighted_choice,
)


# ---------------------------------------------------------------------------
# _validate_to_log_weights
# ---------------------------------------------------------------------------

class TestValidation:
    def test_uniform(self):
        log_w, is_uniform = _validate_to_log_weights(5)
        assert log_w is None
        assert is_uniform is True

    def test_from_weights(self):
        log_w, is_uniform = _validate_to_log_weights(
            3, weights=jnp.array([1.0, 2.0, 1.0])
        )
        assert is_uniform is False
        assert log_w.shape == (3,)

    def test_from_log_weights(self):
        log_w, is_uniform = _validate_to_log_weights(
            3, log_weights=jnp.array([-1.0, 0.0, -1.0])
        )
        assert is_uniform is False
        npt.assert_allclose(log_w, jnp.array([-1.0, 0.0, -1.0]))

    def test_mutual_exclusivity(self):
        with pytest.raises(ValueError, match="either weights or log_weights"):
            _validate_to_log_weights(
                3,
                weights=jnp.ones(3),
                log_weights=jnp.zeros(3),
            )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            _validate_to_log_weights(3, weights=jnp.ones(5))

    def test_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_to_log_weights(
                3, weights=jnp.array([1.0, -1.0, 1.0])
            )

    def test_zero_sum_weights(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_to_log_weights(
                3, weights=jnp.zeros(3)
            )

    def test_log_weights_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            _validate_to_log_weights(
                3, log_weights=jnp.zeros(5)
            )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_normalize_weights(self):
        log_w = jnp.array([0.0, 1.0, 0.0])
        w = normalize_weights(log_w)
        npt.assert_allclose(jnp.sum(w), 1.0, atol=1e-6)
        # Middle element should have highest weight
        assert w[1] > w[0]
        assert w[1] > w[2]

    def test_normalized_log_weights(self):
        log_w = jnp.array([0.0, 1.0, 0.0])
        log_norm = normalized_log_weights(log_w)
        npt.assert_allclose(jax.scipy.special.logsumexp(log_norm), 0.0, atol=1e-6)

    def test_uniform_weights(self):
        w = uniform_weights(4)
        npt.assert_allclose(w, jnp.ones(4) / 4)


class TestWeightedMean:
    def test_uniform(self):
        vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = weighted_mean(None, vals)
        npt.assert_allclose(result, jnp.array([3.0, 4.0]))

    def test_weighted(self):
        vals = jnp.array([[1.0], [3.0]])
        w = jnp.array([0.25, 0.75])
        result = weighted_mean(w, vals)
        npt.assert_allclose(result, jnp.array([2.5]))


class TestWeightedVariance:
    def test_uniform(self):
        vals = jnp.array([[0.0], [2.0], [4.0]])
        var = weighted_variance(None, vals)
        # Mean = 2, variance = mean((0-2)^2, (2-2)^2, (4-2)^2) = (4+0+4)/3
        npt.assert_allclose(var, jnp.array([8.0 / 3]), atol=1e-5)

    def test_weighted(self):
        vals = jnp.array([[0.0], [4.0]])
        w = jnp.array([0.5, 0.5])
        var = weighted_variance(w, vals)
        # Mean = 2, variance = 0.5*(0-2)^2 + 0.5*(4-2)^2 = 4
        npt.assert_allclose(var, jnp.array([4.0]))

    def test_precomputed_mean(self):
        vals = jnp.array([[0.0], [4.0]])
        w = jnp.array([0.5, 0.5])
        mu = jnp.array([2.0])
        var = weighted_variance(w, vals, mean=mu)
        npt.assert_allclose(var, jnp.array([4.0]))


class TestWeightedCovariance:
    def test_2d(self):
        vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cov = weighted_covariance(None, vals)
        assert cov.shape == (2, 2)
        # Diagonal should be variances
        var = weighted_variance(None, vals)
        npt.assert_allclose(jnp.diag(cov), var, atol=1e-5)


class TestWeightedChoice:
    def test_uniform(self):
        key = jax.random.PRNGKey(42)
        idx = weighted_choice(key, 10, shape=(100,))
        assert idx.shape == (100,)
        assert jnp.all(idx >= 0) and jnp.all(idx < 10)

    def test_weighted_concentrates(self):
        key = jax.random.PRNGKey(42)
        # Almost all weight on index 2
        w = jnp.array([0.001, 0.001, 0.998])
        idx = weighted_choice(key, 3, weights=w, shape=(100,))
        assert jnp.sum(idx == 2) > 90

    def test_scalar_output(self):
        key = jax.random.PRNGKey(42)
        idx = weighted_choice(key, 10)
        assert idx.shape == ()


# ---------------------------------------------------------------------------
# Weights class
# ---------------------------------------------------------------------------

class TestWeightsConstruction:
    def test_uniform_via_n(self):
        w = Weights(n=5)
        assert w.n == 5
        assert w.is_uniform is True
        npt.assert_allclose(w.normalized, jnp.ones(5) / 5)

    def test_uniform_via_factory(self):
        w = Weights.uniform(5)
        assert w.n == 5
        assert w.is_uniform is True

    def test_from_weights(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        assert w.n == 3
        assert w.is_uniform is False
        npt.assert_allclose(jnp.sum(w.normalized), 1.0, atol=1e-6)

    def test_from_log_weights(self):
        w = Weights(log_weights=jnp.array([-1.0, 0.0, -1.0]))
        assert w.n == 3
        assert w.is_uniform is False
        npt.assert_allclose(jnp.sum(w.normalized), 1.0, atol=1e-6)

    def test_uniform_log_weights(self):
        w = Weights(n=5)
        npt.assert_allclose(w.log_normalized, jnp.full(5, -jnp.log(5.0)), atol=1e-6)
        assert w.log_unnormalized is None

    def test_n_inferred_from_weights(self):
        w = Weights(weights=jnp.ones(7))
        assert w.n == 7

    def test_n_inferred_from_log_weights(self):
        w = Weights(log_weights=jnp.zeros(4))
        assert w.n == 4

    def test_at_least_one_required(self):
        with pytest.raises(ValueError, match="At least one"):
            Weights()  # none provided

    def test_n_with_weights_validates_length(self):
        # Matching n is OK
        w = Weights(n=3, weights=jnp.ones(3))
        assert w.n == 3
        # Mismatched n raises
        with pytest.raises(ValueError, match="does not match"):
            Weights(n=5, weights=jnp.ones(3))

    def test_weights_and_log_weights_exclusive(self):
        with pytest.raises(ValueError, match="either weights or log_weights"):
            Weights(weights=jnp.ones(3), log_weights=jnp.zeros(3))

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Weights(weights=jnp.array([1.0, -1.0]))

    def test_zero_sum_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Weights(weights=jnp.zeros(3))


class TestWeightsProperties:
    def test_normalized_cached(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        norm1 = w.normalized
        norm2 = w.normalized
        # Should be the exact same object (cached)
        assert norm1 is norm2

    def test_log_normalized(self):
        w = Weights(log_weights=jnp.array([-1.0, 0.0, -1.0]))
        log_n = w.log_normalized
        npt.assert_allclose(
            jax.scipy.special.logsumexp(log_n), 0.0, atol=1e-6
        )

    def test_ess_uniform(self):
        w = Weights(n=10)
        npt.assert_allclose(w.effective_sample_size, 10.0)

    def test_ess_nonuniform(self):
        # All weight on one item → ESS = 1
        w = Weights(weights=jnp.array([1e-10, 1e-10, 1.0]))
        assert float(w.effective_sample_size) == pytest.approx(1.0, abs=0.01)

    def test_ess_equal_weights(self):
        # Equal weights → ESS = n
        w = Weights(weights=jnp.array([1.0, 1.0, 1.0, 1.0]))
        npt.assert_allclose(w.effective_sample_size, 4.0, atol=1e-5)

    def test_ess_between_1_and_n(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 3.0]))
        ess = float(w.effective_sample_size)
        assert 1.0 <= ess <= 3.0

    def test_log_unnormalized(self):
        log_w = jnp.array([-1.0, 0.0, -1.0])
        w = Weights(log_weights=log_w)
        npt.assert_allclose(w.log_unnormalized, log_w)


class TestWeightsJaxArrayProtocol:
    def test_jax_array_returns_normalized(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        arr = w.__jax_array__()
        npt.assert_allclose(arr, w.normalized)
        npt.assert_allclose(jnp.sum(arr), 1.0, atol=1e-6)

    def test_jnp_sum(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        npt.assert_allclose(jnp.sum(w), 1.0, atol=1e-6)

    def test_einsum_works(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        vals = jnp.array([10.0, 20.0, 30.0])
        result = jnp.einsum("n,n->", w, vals)
        expected = jnp.einsum("n,n->", w.normalized, vals)
        npt.assert_allclose(result, expected, atol=1e-5)


class TestWeightsArrayDuckTyping:
    def test_shape(self):
        w = Weights(weights=jnp.ones(5))
        assert w.shape == (5,)

    def test_dtype(self):
        # Default for uniform Weights is JAX's default float dtype.
        w = Weights(n=5)
        assert w.dtype == jnp.zeros((), dtype=float).dtype

    def test_len(self):
        w = Weights(n=7)
        assert len(w) == 7


class TestWeightsMethods:
    def test_mean(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        vals = jnp.array([[1.0], [2.0], [3.0]])
        result = w.mean(vals)
        expected = weighted_mean(w.normalized, vals)
        npt.assert_allclose(result, expected)

    def test_variance(self):
        w = Weights(n=3)
        vals = jnp.array([[0.0], [2.0], [4.0]])
        result = w.variance(vals)
        expected = weighted_variance(None, vals)
        npt.assert_allclose(result, expected)

    def test_covariance(self):
        w = Weights(n=3)
        vals = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = w.covariance(vals)
        expected = weighted_covariance(None, vals)
        npt.assert_allclose(result, expected)

    def test_choice(self):
        w = Weights(n=10)
        key = jax.random.PRNGKey(0)
        idx = w.choice(key, shape=(50,))
        assert idx.shape == (50,)
        assert jnp.all(idx >= 0) and jnp.all(idx < 10)

    def test_subsample_uniform(self):
        w = Weights(n=10)
        sub = w.subsample(jnp.array([0, 2, 4]))
        assert sub.n == 3
        assert sub.is_uniform is True

    def test_subsample_weighted(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 3.0, 4.0]))
        sub = w.subsample(jnp.array([1, 3]))
        assert sub.n == 2
        assert sub.is_uniform is False
        npt.assert_allclose(jnp.sum(sub.normalized), 1.0, atol=1e-6)


class TestWeightsObjectPassthrough:
    """Test that passing a Weights object to the constructor adopts it."""

    def test_weights_passthrough(self):
        w = Weights(weights=jnp.array([1.0, 2.0]))
        result = Weights(weights=w)
        assert result._log_weights is w._log_weights  # same internal state

    def test_log_weights_passthrough(self):
        w = Weights(log_weights=jnp.array([-1.0, 0.0]))
        result = Weights(log_weights=w)
        assert result._log_weights is w._log_weights

    def test_weights_object_and_log_weights_raises(self):
        w = Weights(weights=jnp.ones(3))
        with pytest.raises(ValueError, match="either weights or log_weights"):
            Weights(weights=w, log_weights=jnp.zeros(3))

    def test_n_validation_with_weights_object(self):
        w = Weights(weights=jnp.array([1.0, 2.0]))
        # Matching n is fine
        result = Weights(n=2, weights=w)
        assert result.n == 2
        # Mismatched n raises
        with pytest.raises(ValueError, match="does not match"):
            Weights(n=5, weights=w)

    def test_n_only_gives_uniform(self):
        result = Weights(n=5)
        assert result.is_uniform
        assert result.n == 5

    def test_n_with_array_validates_length(self):
        result = Weights(n=3, weights=jnp.array([1.0, 2.0, 3.0]))
        assert result.n == 3
        with pytest.raises(ValueError, match="does not match"):
            Weights(n=5, weights=jnp.array([1.0, 2.0, 3.0]))

    def test_both_arrays_raises(self):
        with pytest.raises(ValueError, match="either weights or log_weights"):
            Weights(weights=jnp.ones(3), log_weights=jnp.zeros(3))


class TestWeightsPytree:
    def test_tree_flatten_produces_normalized_array(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        leaves = jax.tree.leaves(w)
        assert len(leaves) == 1
        npt.assert_allclose(leaves[0], w.normalized, atol=1e-6)

    def test_uniform_tree_leaves(self):
        w = Weights(n=5)
        leaves = jax.tree.leaves(w)
        assert len(leaves) == 1
        npt.assert_allclose(leaves[0], jnp.ones(5) / 5, atol=1e-6)

    def test_jit_compatible(self):
        w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
        vals = jnp.array([10.0, 20.0, 30.0])

        @jax.jit
        def compute(weights, values):
            return jnp.einsum("n,n->", weights, values)

        result = compute(w, vals)
        expected = jnp.einsum("n,n->", w.normalized, vals)
        npt.assert_allclose(result, expected, atol=1e-5)


class TestWeightsRepr:
    def test_uniform_repr(self):
        w = Weights(n=5)
        assert "uniform" in repr(w)
        assert "n=5" in repr(w)

    def test_weighted_repr(self):
        w = Weights(weights=jnp.ones(3))
        assert "n=3" in repr(w)


class TestWeightsEquality:
    def test_uniform_equal(self):
        assert Weights(n=5) == Weights(n=5)

    def test_uniform_different_n(self):
        assert Weights(n=3) != Weights(n=5)

    def test_weighted_equal(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        assert Weights(weights=arr) == Weights(weights=arr)

    def test_weighted_different(self):
        assert Weights(weights=jnp.array([1.0, 2.0])) != Weights(weights=jnp.array([1.0, 3.0]))

    def test_uniform_vs_weighted(self):
        assert Weights(n=3) != Weights(weights=jnp.ones(3))

    def test_not_equal_to_non_weights(self):
        assert Weights(n=3) != "not a Weights"
        assert Weights(n=3) != 3

    def test_hash_uniform(self):
        assert hash(Weights(n=5)) == hash(Weights(n=5))

    def test_hash_weighted(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        assert hash(Weights(weights=arr)) == hash(Weights(weights=arr))

    def test_hash_usable_in_set(self):
        w1 = Weights(n=5)
        w2 = Weights(n=5)
        w3 = Weights(n=3)
        s = {w1, w2, w3}
        assert len(s) == 2

    def test_hash_usable_as_dict_key(self):
        w = Weights(n=5)
        d = {w: "value"}
        assert d[Weights(n=5)] == "value"


# ---------------------------------------------------------------------------
# Factory dispatch tests
# ---------------------------------------------------------------------------

class TestFactoryDispatch:
    def test_empirical_numeric_returns_array_variant(self):
        from probpipe import EmpiricalDistribution, NumericEmpiricalDistribution
        dist = EmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]))
        assert isinstance(dist, NumericEmpiricalDistribution)

    def test_empirical_numpy_numeric_returns_array_variant(self):
        from probpipe import EmpiricalDistribution, NumericEmpiricalDistribution
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        assert isinstance(dist, NumericEmpiricalDistribution)

    def test_empirical_object_stays_generic(self):
        from probpipe import EmpiricalDistribution, NumericEmpiricalDistribution
        dist = EmpiricalDistribution(["hello", "world"])
        assert not isinstance(dist, NumericEmpiricalDistribution)
        assert isinstance(dist, EmpiricalDistribution)

    def test_empirical_numpy_object_stays_generic(self):
        from probpipe import EmpiricalDistribution, NumericEmpiricalDistribution
        dist = EmpiricalDistribution(np.array(["a", "b"], dtype=object))
        assert not isinstance(dist, NumericEmpiricalDistribution)

    def test_bootstrap_numeric_returns_array_variant(self):
        from probpipe import BootstrapReplicateDistribution, ArrayBootstrapReplicateDistribution
        dist = BootstrapReplicateDistribution(jnp.ones((5, 2)))
        assert isinstance(dist, ArrayBootstrapReplicateDistribution)

    def test_bootstrap_from_empirical_returns_array_variant(self):
        from probpipe import (
            EmpiricalDistribution,
            BootstrapReplicateDistribution,
            ArrayBootstrapReplicateDistribution,
        )
        emp = EmpiricalDistribution(jnp.ones((5, 2)))
        dist = BootstrapReplicateDistribution(emp)
        assert isinstance(dist, ArrayBootstrapReplicateDistribution)

    def test_bootstrap_object_stays_generic(self):
        from probpipe import BootstrapReplicateDistribution, ArrayBootstrapReplicateDistribution
        dist = BootstrapReplicateDistribution(["a", "b", "c"])
        assert not isinstance(dist, ArrayBootstrapReplicateDistribution)

    def test_subclass_not_redirected(self):
        from probpipe import NumericEmpiricalDistribution
        dist = NumericEmpiricalDistribution(jnp.array([1.0, 2.0, 3.0]))
        assert type(dist) is NumericEmpiricalDistribution


# ---------------------------------------------------------------------------
# sample_shape tests
# ---------------------------------------------------------------------------

class TestSampleShape:
    def test_default_single_axis(self):
        from probpipe import NumericEmpiricalDistribution
        samples = jnp.ones((100, 3))
        dist = NumericEmpiricalDistribution(samples)
        assert dist.n == 100
        assert dist.event_shape == (3,)

    def test_explicit_1d_sample_shape(self):
        from probpipe import NumericEmpiricalDistribution
        samples = jnp.ones((100, 3))
        dist = NumericEmpiricalDistribution(samples, sample_shape=(100,))
        assert dist.n == 100
        assert dist.event_shape == (3,)

    def test_2d_sample_shape(self):
        from probpipe import NumericEmpiricalDistribution
        samples = jnp.ones((10, 5, 3))
        dist = NumericEmpiricalDistribution(samples, sample_shape=(10, 5))
        assert dist.n == 50
        assert dist.event_shape == (3,)

    def test_sample_shape_mismatch_raises(self):
        from probpipe import NumericEmpiricalDistribution
        samples = jnp.ones((10, 3))
        with pytest.raises(ValueError, match="do not match"):
            NumericEmpiricalDistribution(samples, sample_shape=(20,))

    def test_moments_with_sample_shape(self):
        from probpipe import NumericEmpiricalDistribution, mean, variance
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (10, 5, 2))
        dist = NumericEmpiricalDistribution(samples, sample_shape=(10, 5))
        m = mean(dist)
        v = variance(dist)
        assert m.shape == (2,)
        assert v.shape == (2,)
