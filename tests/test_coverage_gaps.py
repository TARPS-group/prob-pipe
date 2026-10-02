"""Regression tests for narrow code paths added to close historical coverage gaps.

Each test in this file targets a specific observable behavior that was
discovered to be missing coverage (weighted paths, repr/alias
fall-throughs).  Tests include real value/shape assertions — they are not
coverage-only touches.

Covers:
- EmpiricalDistribution: weighted subsampled expectation
- TFPDistribution._cov: scalar and multivariate
- BijectorTransformedDistribution over a base outside the backend
- SupportsUnnormalizedLogProb._unnormalized_prob default
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb

from probpipe import (
    BijectorTransformedDistribution,
    EmpiricalDistribution,
    Normal,
    cov,
    sample,
    variance,
)
from probpipe.families._multivariate import MultivariateNormal

# ---------------------------------------------------------------------------
# EmpiricalDistribution — weighted subsampling
# ---------------------------------------------------------------------------


class TestEmpiricalSubsampling:
    """Cover the weighted subsample paths in _expectation."""

    def test_weighted_cov(self):
        """Weighted EmpiricalDistribution covariance."""
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (50, 2))
        weights = jax.random.uniform(jax.random.PRNGKey(1), (50,))
        weights = weights / jnp.sum(weights)
        ed = EmpiricalDistribution("x", samples, weights=weights)
        C = cov(ed)
        assert C.shape == (2, 2)
        assert jnp.all(jnp.isfinite(C))


# ---------------------------------------------------------------------------
# TFPDistribution._cov — scalar and multivariate
# ---------------------------------------------------------------------------


class TestTFPDistributionCov:
    """Cover the _cov method on TFPDistribution."""

    def test_scalar_cov_equals_variance(self):
        """For a scalar law, cov is the (1, 1) matrix holding the variance."""
        d = Normal(loc=0.0, scale=2.0, label="x")
        c = np.asarray(cov(d))
        v = variance(d)
        assert c.shape == (1, 1)
        np.testing.assert_allclose(c[0, 0], float(v), atol=1e-5)

    def test_multivariate_cov(self):
        """For multivariate distributions, _cov returns full covariance matrix."""
        loc = jnp.zeros(3)
        cov_matrix = jnp.eye(3) * 2.0
        d = MultivariateNormal(loc=loc, cov=cov_matrix, label="z")
        C = cov(d)
        np.testing.assert_allclose(C, cov_matrix, atol=1e-5)


# ---------------------------------------------------------------------------
# BijectorTransformedDistribution over a base outside the backend
# ---------------------------------------------------------------------------


class TestTransformedNonTFP:
    """Cover BijectorTransformedDistribution over an empirical base."""

    @pytest.fixture
    def td(self):
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (100, 2))
        emp = EmpiricalDistribution("x", samples)
        return BijectorTransformedDistribution("transformed", emp, tfb.Exp())

    def test_base_property(self, td):
        assert isinstance(td.base, EmpiricalDistribution)

    def test_bijector_property(self, td):
        assert td.bijector is not None

    def test_event_shape(self, td):
        assert td.event_shape == (2,)

    def test_sample_unbatched(self, td):
        s = jnp.asarray(sample(td))
        assert s.shape == (2,)
        assert jnp.all(s > 0)  # Exp bijector

    def test_sample_batched(self, td):
        s = jnp.asarray(sample(td, sample_shape=(5,)))
        assert s.shape == (5, 2)
        assert jnp.all(s > 0)

    def test_repr(self, td):
        r = repr(td)
        assert "BijectorTransformedDistribution" in r


# ---------------------------------------------------------------------------
# SupportsUnnormalizedLogProb._unnormalized_prob default
# ---------------------------------------------------------------------------


class TestUnnormalizedProbDefault:
    """Cover the _unnormalized_prob default (exp of _unnormalized_log_prob)."""

    def test_unnormalized_prob_default(self):
        from probpipe import unnormalized_log_prob, unnormalized_prob

        d = Normal(loc=0.0, scale=1.0, label="x")
        x = jnp.array(1.0)
        up = unnormalized_prob(d, x)
        ulp = unnormalized_log_prob(d, x)
        np.testing.assert_allclose(float(up), float(jnp.exp(ulp)), rtol=1e-5)
