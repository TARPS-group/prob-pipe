"""Tests for probpipe.modeling — Likelihood protocols, IncrementalConditioner."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from probpipe import MultivariateNormal, EmpiricalDistribution, ArrayEmpiricalDistribution, Provenance
from probpipe.modeling import (
    GenerativeLikelihood,
    IncrementalConditioner,
    Likelihood,
    SimpleModel,
)
from probpipe import log_prob, mean, prob
from probpipe.custom_types import ArrayLike


# ---------------------------------------------------------------------------
# Simple likelihood implementations for testing
# ---------------------------------------------------------------------------


class MultivariateNormalLikelihood:
    """Isotropic MultivariateNormal likelihood — JAX-traceable."""

    def log_likelihood(self, params, data):
        residuals = data - params[None, :]
        return -0.5 * jnp.sum(residuals ** 2)


class SimpleGenerativeLikelihood:
    """Generate data as params + noise. Satisfies both Likelihood and GenerativeLikelihood."""

    def log_likelihood(self, params, data):
        residuals = data - params[None, :]
        return -0.5 * jnp.sum(residuals ** 2)

    def generate_data(self, params, n_samples):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples, len(params)))
        return params[None, :] + noise


# ---------------------------------------------------------------------------
# Likelihood protocols
# ---------------------------------------------------------------------------


class TestLikelihood:
    def test_is_protocol(self):
        """Likelihood is a runtime-checkable protocol."""
        assert isinstance(MultivariateNormalLikelihood(), Likelihood)

    def test_concrete_likelihood(self):
        lik = MultivariateNormalLikelihood()
        params = jnp.array([1.0, 2.0])
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        ll = lik.log_likelihood(params=params, data=data)
        assert jnp.isfinite(ll)


class TestGenerativeLikelihood:
    def test_is_protocol(self):
        """GenerativeLikelihood is a runtime-checkable protocol."""
        gen = SimpleGenerativeLikelihood()
        assert isinstance(gen, GenerativeLikelihood)
        # Also satisfies Likelihood
        assert isinstance(gen, Likelihood)

    def test_concrete_generative(self):
        gen = SimpleGenerativeLikelihood()
        params = jnp.array([1.0, 2.0])
        data = gen.generate_data(params=params, n_samples=10)
        assert data.shape == (10, 2)


# ---------------------------------------------------------------------------
# IncrementalConditioner
# ---------------------------------------------------------------------------


def _simple_condition_fn(model, data):
    """A simple conditioning function for testing.

    Returns an EmpiricalDistribution near the data mean instead of
    running MCMC.
    """
    data_mean = jnp.mean(jnp.asarray(data), axis=0)
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(50, data_mean.shape[0]))
    samples = data_mean[None, :] + noise * 0.1
    return EmpiricalDistribution(samples)


class TestIncrementalConditioner:
    @pytest.fixture
    def dim(self):
        return 2

    @pytest.fixture
    def prior(self, dim):
        return MultivariateNormal(loc=jnp.zeros(dim), cov=jnp.eye(dim) * 10.0)

    @pytest.fixture
    def likelihood(self):
        return MultivariateNormalLikelihood()

    def test_incremental_update(self, prior, likelihood, dim):
        from probpipe.core.transition import TransitionTrace

        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
            condition_fn=_simple_condition_fn,
        )

        key = jax.random.PRNGKey(0)
        data1 = jax.random.normal(key, shape=(10, dim)) + 2.0
        data2 = jax.random.normal(jax.random.PRNGKey(1), shape=(10, dim)) + 3.0
        trace = conditioner.update(data_batches=[data1, data2])

        assert isinstance(trace, TransitionTrace)
        assert len(trace) == 2
        assert trace.distributions[0] is prior
        assert isinstance(trace.final, EmpiricalDistribution)

    def test_default_condition_fn(self, prior, likelihood):
        """IncrementalConditioner should work without explicit condition_fn."""
        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
        )
        # Just verify construction works; actual conditioning requires MCMC
        assert conditioner._prior is prior


# ---------------------------------------------------------------------------
# Distribution coverage gaps
# ---------------------------------------------------------------------------


class TestDistributionCoverageGaps:
    """Cover the few uncovered lines in distribution.py."""

    def test_batch_shape_default(self):
        """ArrayDistribution.batch_shape defaults to ()."""
        from probpipe.core.distribution import ArrayDistribution

        class Scalar(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = Scalar()
        assert d.batch_shape == ()

    def test_dtype_default(self):
        """ArrayDistribution.dtype defaults to float32."""
        from probpipe.core.distribution import ArrayDistribution

        class Scalar(ArrayDistribution):
            @property
            def event_shape(self):
                return ()

        d = Scalar()
        assert d.dtype == jnp.float32

    def test_prob_method(self):
        """prob(dist, x) returns exp(log_prob(dist, x))."""
        from probpipe import Normal

        d = Normal(loc=0.0, scale=1.0)
        x = jnp.array(0.0)
        np.testing.assert_allclose(prob(d, x), jnp.exp(log_prob(d, x)), atol=1e-6)

    def test_repr_with_batch_shape(self):
        """ArrayDistribution repr includes batch_shape when non-trivial."""
        from probpipe import Normal

        d = Normal(loc=jnp.array([0.0, 1.0]), scale=jnp.array([1.0, 1.0]))
        r = repr(d)
        assert "batch_shape" in r

    def test_empirical_dtype(self):
        """ArrayEmpiricalDistribution.dtype returns sample dtype."""
        samples = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
        ed = ArrayEmpiricalDistribution(samples)
        assert ed.dtype == jnp.float32
