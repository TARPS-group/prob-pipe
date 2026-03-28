"""Tests to improve code coverage for distribution protocols, ops, and helpers.

Covers:
- BootstrapDistribution: weighted sampling, variance, repr, support, evaluations
- EmpiricalDistribution: weighted subsampled expectation
- FlattenedView: sampling, log_prob, expectation, support
- TFPDistribution._cov: scalar and multivariate
- ops error paths for unsupported protocols
- SupportsCovariance default implementation
- TransformedDistribution non-TFP paths
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    BootstrapDistribution,
    EmpiricalDistribution,
    Normal,
    TransformedDistribution,
    cov,
    expectation,
    log_prob,
    mean,
    prob,
    sample,
    variance,
)
from probpipe.distributions.multivariate import MultivariateNormal


# ---------------------------------------------------------------------------
# BootstrapDistribution
# ---------------------------------------------------------------------------


class TestBootstrapDistributionCoverage:
    """Cover weighted paths, repr, support, evaluations, sample shapes."""

    def test_scalar_0d_raises(self):
        with pytest.raises(ValueError, match="at least 1 dimension"):
            BootstrapDistribution(jnp.float32(1.0))

    def test_evaluations_property(self):
        evals = jnp.array([1.0, 2.0, 3.0])
        bd = BootstrapDistribution(evals)
        assert bd.evaluations.shape == (3,)
        np.testing.assert_allclose(bd.evaluations, evals)

    def test_repr(self):
        bd = BootstrapDistribution(jnp.array([1.0, 2.0, 3.0]))
        r = repr(bd)
        assert "BootstrapDistribution" in r
        assert "n=3" in r
        assert "event_shape=()" in r

    def test_support_is_real(self):
        from probpipe.core.distribution import real

        bd = BootstrapDistribution(jnp.array([1.0, 2.0]))
        assert bd.support is real

    def test_weighted_mean(self):
        evals = jnp.array([0.0, 10.0])
        weights = jnp.array([0.3, 0.7])
        bd = BootstrapDistribution(evals, weights=weights)
        np.testing.assert_allclose(float(mean(bd)), 7.0, atol=1e-5)

    def test_weighted_variance(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
        bd = BootstrapDistribution(evals, weights=weights)
        v = variance(bd)
        assert jnp.isfinite(v)
        # Weighted variance / n_eff should be positive and smaller than sample_var
        assert float(v) > 0

    def test_weighted_sample_one(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
        bd = BootstrapDistribution(evals, weights=weights)
        key = jax.random.PRNGKey(42)
        s = sample(bd, key=key)
        assert s.shape == ()
        assert jnp.isfinite(s)

    def test_unweighted_sample_one(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        key = jax.random.PRNGKey(42)
        s = sample(bd, key=key)
        assert s.shape == ()
        assert jnp.isfinite(s)

    def test_weighted_sample_batched(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
        bd = BootstrapDistribution(evals, weights=weights)
        key = jax.random.PRNGKey(42)
        s = sample(bd, key=key, sample_shape=(10,))
        assert s.shape == (10,)
        assert jnp.all(jnp.isfinite(s))

    def test_unweighted_sample_batched(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        key = jax.random.PRNGKey(42)
        s = sample(bd, key=key, sample_shape=(10,))
        assert s.shape == (10,)
        assert jnp.all(jnp.isfinite(s))

    def test_expectation_delegates_to_mc(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bd = BootstrapDistribution(evals)
        key = jax.random.PRNGKey(0)
        result = expectation(bd, lambda x: x, key=key, num_evaluations=100, return_dist=False)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_allclose(float(result), 3.0, atol=0.5)

    def test_multidimensional_evaluations(self):
        evals = jnp.ones((10, 3))
        bd = BootstrapDistribution(evals)
        assert bd.event_shape == (3,)
        assert bd.n == 10
        assert repr(bd) == "BootstrapDistribution(n=10, event_shape=(3,))"


# ---------------------------------------------------------------------------
# EmpiricalDistribution — weighted subsampling
# ---------------------------------------------------------------------------


class TestEmpiricalSubsampling:
    """Cover the weighted subsample paths in _expectation."""

    def test_weighted_subsample_returns_bootstrap(self):
        """Weighted EmpiricalDistribution with num_evaluations < n → Bootstrap."""
        samples = jnp.arange(100.0)
        weights = jax.random.uniform(jax.random.PRNGKey(0), (100,))
        weights = weights / jnp.sum(weights)
        ed = EmpiricalDistribution(samples, weights=weights)
        key = jax.random.PRNGKey(1)
        result = expectation(ed, lambda x: x, key=key, num_evaluations=10)
        assert isinstance(result, BootstrapDistribution)

    def test_weighted_subsample_returns_array(self):
        """Weighted EmpiricalDistribution with num_evaluations < n, return_dist=False."""
        samples = jnp.arange(100.0)
        weights = jax.random.uniform(jax.random.PRNGKey(0), (100,))
        weights = weights / jnp.sum(weights)
        ed = EmpiricalDistribution(samples, weights=weights)
        key = jax.random.PRNGKey(1)
        result = expectation(ed, lambda x: x, key=key, num_evaluations=10, return_dist=False)
        assert isinstance(result, jnp.ndarray)
        assert jnp.isfinite(result)

    def test_weighted_cov(self):
        """Weighted EmpiricalDistribution covariance."""
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (50, 2))
        weights = jax.random.uniform(jax.random.PRNGKey(1), (50,))
        weights = weights / jnp.sum(weights)
        ed = EmpiricalDistribution(samples, weights=weights)
        C = cov(ed)
        assert C.shape == (2, 2)
        assert jnp.all(jnp.isfinite(C))


# ---------------------------------------------------------------------------
# TFPDistribution._cov — scalar and multivariate
# ---------------------------------------------------------------------------


class TestTFPDistributionCov:
    """Cover the _cov method on TFPDistribution."""

    def test_scalar_cov_equals_variance(self):
        """For scalar distributions, _cov returns variance."""
        d = Normal(loc=0.0, scale=2.0)
        c = cov(d)
        v = variance(d)
        np.testing.assert_allclose(float(c), float(v), atol=1e-5)

    def test_multivariate_cov(self):
        """For multivariate distributions, _cov returns full covariance matrix."""
        loc = jnp.zeros(3)
        cov_matrix = jnp.eye(3) * 2.0
        d = MultivariateNormal(loc=loc, cov=cov_matrix)
        C = cov(d)
        np.testing.assert_allclose(C, cov_matrix, atol=1e-5)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ops error paths
# ---------------------------------------------------------------------------


class TestOpsErrorPaths:
    """Cover TypeError error paths in ops for unsupported protocols."""

    def test_prob_requires_log_prob(self):
        from probpipe import ArrayDistribution
        from probpipe.core.distribution import _vmap_sample
        from probpipe.core.protocols import SupportsSampling

        class NoLogProbDist(ArrayDistribution, SupportsSampling):
            _sampling_cost = "low"
            _preferred_orchestration = None

            @property
            def event_shape(self):
                return ()

            def _sample_one(self, key):
                return jnp.float32(0.0)

            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)

        d = NoLogProbDist()
        with pytest.raises(TypeError, match="does not support prob"):
            prob(d, jnp.float32(0.0))

    def test_expectation_requires_protocol(self):
        from probpipe import ArrayDistribution

        class MinimalDist(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist()
        with pytest.raises(TypeError, match="does not support expectation"):
            expectation(d, lambda x: x)

    def test_mean_requires_protocol(self):
        from probpipe import ArrayDistribution

        class MinimalDist(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist()
        with pytest.raises(TypeError, match="does not support mean"):
            mean(d)

    def test_variance_requires_protocol(self):
        from probpipe import ArrayDistribution

        class MinimalDist(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist()
        with pytest.raises(TypeError, match="does not support variance"):
            variance(d)

    def test_cov_requires_protocol(self):
        from probpipe import ArrayDistribution

        class MinimalDist(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist()
        with pytest.raises(TypeError, match="does not support covariance"):
            cov(d)


# ---------------------------------------------------------------------------
# FlattenedView
# ---------------------------------------------------------------------------


class TestFlattenedView:
    """Cover FlattenedView sampling, log_prob, expectation, support."""

    @pytest.fixture
    def joint_dist(self):
        """A simple joint distribution that can be flattened."""
        from probpipe.distributions.joint import ProductDistribution

        return ProductDistribution(
            x=Normal(loc=0.0, scale=1.0),
            y=Normal(loc=1.0, scale=0.5),
        )

    def test_sample_one(self, joint_dist):
        flat = joint_dist.as_flat_distribution()
        key = jax.random.PRNGKey(0)
        s = sample(flat, key=key)
        assert s.shape == (flat.event_shape[0],)
        assert jnp.all(jnp.isfinite(s))

    def test_sample_batched(self, joint_dist):
        flat = joint_dist.as_flat_distribution()
        key = jax.random.PRNGKey(0)
        s = sample(flat, key=key, sample_shape=(5,))
        assert s.shape == (5, flat.event_shape[0])

    def test_log_prob(self, joint_dist):
        flat = joint_dist.as_flat_distribution()
        key = jax.random.PRNGKey(0)
        s = sample(flat, key=key)
        lp = log_prob(flat, s)
        assert jnp.isfinite(lp)

    def test_support(self, joint_dist):
        from probpipe.core.distribution import real

        flat = joint_dist.as_flat_distribution()
        assert flat.support is real

    def test_expectation(self, joint_dist):
        flat = joint_dist.as_flat_distribution()
        key = jax.random.PRNGKey(0)
        result = expectation(flat, lambda x: x, key=key, num_evaluations=500, return_dist=False)
        assert result.shape == flat.event_shape
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# TransformedDistribution non-TFP paths
# ---------------------------------------------------------------------------


class TestTransformedNonTFP:
    """Cover TransformedDistribution with non-TFP base (EmpiricalDistribution)."""

    @pytest.fixture
    def td(self):
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (100, 2))
        emp = EmpiricalDistribution(samples)
        return TransformedDistribution(emp, tfb.Exp())

    def test_base_property(self, td):
        assert isinstance(td.base, EmpiricalDistribution)

    def test_bijector_property(self, td):
        assert td.bijector is not None

    def test_event_shape(self, td):
        assert td.event_shape == (2,)

    def test_sample_one(self, td):
        key = jax.random.PRNGKey(0)
        s = sample(td, key=key)
        assert s.shape == (2,)
        assert jnp.all(s > 0)  # Exp bijector

    def test_sample_batched(self, td):
        key = jax.random.PRNGKey(0)
        s = sample(td, key=key, sample_shape=(5,))
        assert s.shape == (5, 2)
        assert jnp.all(s > 0)

    def test_mean_mc_fallback(self, td):
        m = mean(td)
        assert jnp.all(jnp.isfinite(m))

    def test_variance_mc_fallback(self, td):
        v = variance(td)
        assert jnp.all(jnp.isfinite(v))

    def test_repr(self, td):
        r = repr(td)
        assert "TransformedDistribution" in r


# ---------------------------------------------------------------------------
# SupportsCovariance default implementation (protocol-level)
# ---------------------------------------------------------------------------


class TestCovarianceRequiresProtocol:
    """cov op requires SupportsCovariance — no MC fallback."""

    def test_cov_raises_without_supports_covariance(self):
        """A distribution with SupportsExpectation but not SupportsCovariance
        should raise TypeError from the cov op."""
        from probpipe.core.distribution import _mc_expectation, _vmap_sample
        from probpipe.core.protocols import SupportsSampling, SupportsExpectation
        from probpipe import ArrayDistribution, cov

        class NoCovDist(ArrayDistribution, SupportsSampling, SupportsExpectation):
            _sampling_cost = "low"
            _preferred_orchestration = None

            @property
            def event_shape(self):
                return (2,)

            def _sample_one(self, key):
                return jax.random.normal(key, (2,))

            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)

            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                return _mc_expectation(
                    self, f, key=key, num_evaluations=num_evaluations,
                    return_dist=return_dist,
                )

        d = NoCovDist()
        with pytest.raises(TypeError, match="does not support covariance"):
            cov(d)


# ---------------------------------------------------------------------------
# SupportsUnnormalizedLogProb._unnormalized_prob default
# ---------------------------------------------------------------------------


class TestUnnormalizedProbDefault:
    """Cover the _unnormalized_prob default (exp of _unnormalized_log_prob)."""

    def test_unnormalized_prob_default(self):
        from probpipe import unnormalized_prob, unnormalized_log_prob

        d = Normal(loc=0.0, scale=1.0)
        x = jnp.array(1.0)
        up = unnormalized_prob(d, x)
        ulp = unnormalized_log_prob(d, x)
        np.testing.assert_allclose(float(up), float(jnp.exp(ulp)), rtol=1e-5)
