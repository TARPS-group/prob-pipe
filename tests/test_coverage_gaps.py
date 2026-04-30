"""Regression tests for narrow code paths added to close historical coverage gaps.

Each test in this file targets a specific observable behavior that was
discovered to be missing coverage (weighted paths, error branches on
unsupported protocols, repr/alias fall-throughs).  Tests include real
value/shape assertions — they are not coverage-only touches.

Covers:
- BootstrapDistribution: weighted sampling, variance, repr, support, evaluations
- EmpiricalDistribution: weighted subsampled expectation
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
    RecordEmpiricalDistribution,
    NumericRecord,
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
        from probpipe.core.constraints import real

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
        assert float(v) > 0
        # Weighted variance should be smaller than unweighted sample variance
        sample_var = float(jnp.var(evals))
        assert float(v) < sample_var

    def test_weighted_sample_unbatched(self):
        evals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])
        bd = BootstrapDistribution(evals, weights=weights)
        key = jax.random.PRNGKey(42)
        s = sample(bd, key=key)
        assert s.shape == ()
        assert jnp.isfinite(s)

    def test_unweighted_sample_unbatched(self):
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
        assert isinstance(result, NumericRecord)
        np.testing.assert_allclose(float(result), 3.0, atol=0.5)

    def test_multidimensional_evaluations(self):
        evals = jnp.ones((10, 3))
        bd = BootstrapDistribution(evals)
        assert bd.event_shape == (3,)
        assert bd.n == 10
        r = repr(bd)
        assert "BootstrapDistribution" in r
        assert "n=10" in r
        assert "event_shape=(3,)" in r


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
        ed = EmpiricalDistribution(samples, weights=weights, name="x")
        key = jax.random.PRNGKey(1)
        result = expectation(ed, lambda x: x, key=key, num_evaluations=10)
        assert isinstance(result, BootstrapDistribution)

    def test_weighted_subsample_returns_array(self):
        """Weighted EmpiricalDistribution with num_evaluations < n, return_dist=False."""
        samples = jnp.arange(100.0)
        weights = jax.random.uniform(jax.random.PRNGKey(0), (100,))
        weights = weights / jnp.sum(weights)
        ed = EmpiricalDistribution(samples, weights=weights, name="x")
        key = jax.random.PRNGKey(1)
        result = expectation(ed, lambda x: x, key=key, num_evaluations=10, return_dist=False)
        assert isinstance(result, NumericRecord)
        assert jnp.isfinite(jnp.asarray(result))

    def test_weighted_cov(self):
        """Weighted RecordEmpiricalDistribution covariance."""
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (50, 2))
        weights = jax.random.uniform(jax.random.PRNGKey(1), (50,))
        weights = weights / jnp.sum(weights)
        ed = RecordEmpiricalDistribution(samples, weights=weights, name="x")
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
        d = Normal(loc=0.0, scale=2.0, name="x")
        c = cov(d)
        v = variance(d)
        np.testing.assert_allclose(float(c), float(v), atol=1e-5)

    def test_multivariate_cov(self):
        """For multivariate distributions, _cov returns full covariance matrix."""
        loc = jnp.zeros(3)
        cov_matrix = jnp.eye(3) * 2.0
        d = MultivariateNormal(loc=loc, cov=cov_matrix, name="z")
        C = cov(d)
        np.testing.assert_allclose(C, cov_matrix, atol=1e-5)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ops error paths
# ---------------------------------------------------------------------------


class TestOpsErrorPaths:
    """Cover TypeError error paths in ops for unsupported protocols."""

    def test_prob_requires_log_prob(self):
        from probpipe import NumericRecordDistribution

        class NoLogProbNoSampleDist(NumericRecordDistribution):
            """Has neither SupportsLogProb nor SupportsSampling."""

            @property
            def event_shape(self):
                return ()

        d = NoLogProbNoSampleDist(name="test")
        with pytest.raises(TypeError):
            prob(d, jnp.float32(0.0))

    def test_expectation_requires_protocol(self):
        from probpipe import NumericRecordDistribution

        class MinimalDist(NumericRecordDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist(name="test")
        with pytest.raises(TypeError, match="does not support expectation"):
            expectation(d, lambda x: x)

    def test_mean_requires_protocol(self):
        from probpipe import NumericRecordDistribution

        class MinimalDist(NumericRecordDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist(name="test")
        with pytest.raises(TypeError, match="does not support mean"):
            mean(d)

    def test_variance_requires_protocol(self):
        from probpipe import NumericRecordDistribution

        class MinimalDist(NumericRecordDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist(name="test")
        with pytest.raises(TypeError, match="does not support variance"):
            variance(d)

    def test_cov_requires_protocol(self):
        from probpipe import NumericRecordDistribution

        class MinimalDist(NumericRecordDistribution):
            @property
            def event_shape(self):
                return ()

        d = MinimalDist(name="test")
        with pytest.raises(TypeError, match="does not support covariance"):
            cov(d)


# ---------------------------------------------------------------------------
# TransformedDistribution non-TFP paths
# ---------------------------------------------------------------------------


class TestTransformedNonTFP:
    """Cover TransformedDistribution with non-TFP base (EmpiricalDistribution)."""

    @pytest.fixture
    def td(self):
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (100, 2))
        emp = RecordEmpiricalDistribution(samples, name="x")
        return TransformedDistribution(emp, tfb.Exp())

    def test_base_property(self, td):
        assert isinstance(td.base, EmpiricalDistribution)

    def test_bijector_property(self, td):
        assert td.bijector is not None

    def test_event_shape(self, td):
        assert td.event_shape == (2,)

    def test_sample_unbatched(self, td):
        key = jax.random.PRNGKey(0)
        s = jnp.asarray(sample(td, key=key))
        assert s.shape == (2,)
        assert jnp.all(s > 0)  # Exp bijector

    def test_sample_batched(self, td):
        key = jax.random.PRNGKey(0)
        s = jnp.asarray(sample(td, key=key, sample_shape=(5,)))
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
        from probpipe.core.distribution import _mc_expectation
        from probpipe.core.protocols import SupportsSampling, SupportsExpectation
        from probpipe import NumericRecordDistribution, cov

        class NoCovDist(NumericRecordDistribution, SupportsSampling, SupportsExpectation):
            _sampling_cost = "low"
            _preferred_orchestration = None

            @property
            def event_shape(self):
                return (2,)

            def _sample(self, key, sample_shape=()):
                return jax.random.normal(key, sample_shape + (2,))

            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                return _mc_expectation(
                    self, f, key=key, num_evaluations=num_evaluations,
                    return_dist=return_dist,
                )

        d = NoCovDist(name="test")
        with pytest.raises(TypeError, match="does not support covariance"):
            cov(d)


# ---------------------------------------------------------------------------
# SupportsUnnormalizedLogProb._unnormalized_prob default
# ---------------------------------------------------------------------------


class TestUnnormalizedProbDefault:
    """Cover the _unnormalized_prob default (exp of _unnormalized_log_prob)."""

    def test_unnormalized_prob_default(self):
        from probpipe import unnormalized_prob, unnormalized_log_prob

        d = Normal(loc=0.0, scale=1.0, name="x")
        x = jnp.array(1.0)
        up = unnormalized_prob(d, x)
        ulp = unnormalized_log_prob(d, x)
        np.testing.assert_allclose(float(up), float(jnp.exp(ulp)), rtol=1e-5)
