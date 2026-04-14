"""Tests for probpipe.modeling — Likelihood protocols, IncrementalConditioner, lazy imports."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from probpipe import MultivariateNormal, EmpiricalDistribution
from probpipe.modeling import (
    GenerativeLikelihood,
    IncrementalConditioner,
    Likelihood,
    SimpleModel,
)


# ---------------------------------------------------------------------------
# Lazy imports — probpipe.modeling.__getattr__ branches
# ---------------------------------------------------------------------------


class TestLazyImports:
    """Exercise the lazy-import fallback in probpipe/modeling/__init__.py."""

    def test_stanmodel_lazy_load(self):
        from probpipe.modeling import StanModel
        assert StanModel is not None

    def test_pymcmodel_lazy_load(self):
        from probpipe.modeling import PyMCModel
        assert PyMCModel is not None

    def test_unknown_attr_raises(self):
        import probpipe.modeling as mod
        with pytest.raises(AttributeError, match="has no attribute"):
            mod.NonExistent


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

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
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
        return MultivariateNormal(loc=jnp.zeros(dim), cov=jnp.eye(dim) * 10.0, name="prior")

    @pytest.fixture
    def likelihood(self):
        return MultivariateNormalLikelihood()

    def test_incremental_update(self, prior, likelihood, dim):
        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
            condition_fn=_simple_condition_fn,
        )
        assert conditioner.curr_posterior is prior

        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, shape=(10, dim)) + 2.0
        posterior = conditioner.update(data=data)

        assert isinstance(posterior, EmpiricalDistribution)
        assert conditioner.curr_posterior is posterior

    def test_default_condition_fn(self, prior, likelihood):
        """update() with the default condition_fn returns a new posterior."""
        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
            method="tfp_nuts",
            num_results=50,
            num_warmup=25,
            num_chains=1,
            random_seed=0,
        )
        assert conditioner.curr_posterior is prior

        data = jax.random.normal(jax.random.PRNGKey(1), shape=(5, 2)) + 2.0
        posterior = conditioner.update(data=data)
        assert posterior is not prior
        assert conditioner.curr_posterior is posterior

    def test_update_all_chains_batches(self, prior, likelihood, dim):
        """update_all must condition on each batch in turn and return the chain."""
        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
            condition_fn=_simple_condition_fn,
        )
        key = jax.random.PRNGKey(0)
        batches = [
            jax.random.normal(key, shape=(4, dim)) + 1.0,
            jax.random.normal(jax.random.PRNGKey(1), shape=(4, dim)) + 2.0,
            jax.random.normal(jax.random.PRNGKey(2), shape=(4, dim)) + 3.0,
        ]
        dists = conditioner.update_all(data_batches=batches)
        # iterate returns the starting distribution followed by one posterior
        # per batch, so 3 batches -> 4 entries in the chain.
        assert len(dists) == len(batches) + 1
        assert dists[0] is prior
        for d in dists[1:]:
            assert isinstance(d, EmpiricalDistribution)
        # Conditioner state must track the final posterior.
        assert conditioner.curr_posterior is dists[-1]

    def test_step_property_exposes_conditioning_step(self, prior, likelihood):
        """The ``step`` property must return the underlying _ConditioningStep."""
        from probpipe.modeling._likelihood import _ConditioningStep

        conditioner = IncrementalConditioner(
            prior=prior,
            likelihood=likelihood,
            condition_fn=_simple_condition_fn,
        )
        step = conditioner.step
        assert isinstance(step, _ConditioningStep)
        # Calling step directly should yield a posterior without touching
        # the conditioner's internal state.
        data = jnp.zeros((3, 2))
        post = step(prior, data)
        assert isinstance(post, EmpiricalDistribution)
        # Internal state unchanged because we bypassed update().
        assert conditioner.curr_posterior is prior
