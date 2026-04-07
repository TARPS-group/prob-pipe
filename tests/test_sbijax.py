"""Tests for sbijax integration (amortized NPE + SMCABC).

These tests require sbijax to be installed: pip install probpipe[sbi]
"""

import jax
import jax.numpy as jnp
import pytest

sbijax = pytest.importorskip("sbijax")

from probpipe import (
    Normal,
    SimpleGenerativeModel,
    condition_on,
    train_sbi,
)
from probpipe.distributions.multivariate import MultivariateNormal
from probpipe.core.protocols import SupportsConditioning
from probpipe.inference import ApproximateDistribution
from probpipe.inference._sbijax_distribution import TrainedSBIModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class GaussianSimulator:
    """Simple: y = theta + noise."""

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples,) + jnp.atleast_1d(params).shape)
        return jnp.atleast_1d(params) + 0.1 * noise


@pytest.fixture
def prior():
    return Normal(loc=0.0, scale=1.0)


@pytest.fixture
def simulator():
    return GaussianSimulator()


@pytest.fixture
def generative_model(prior, simulator):
    return SimpleGenerativeModel(prior, simulator)


@pytest.fixture
def observed():
    return jnp.array([0.5])


class GaussianSimulator2D:
    """2D: y = theta + noise."""

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=(n_samples, 2))
        return params + 0.1 * noise


@pytest.fixture
def prior_2d():
    return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))


@pytest.fixture
def simulator_2d():
    return GaussianSimulator2D()


@pytest.fixture
def generative_model_2d(prior_2d, simulator_2d):
    return SimpleGenerativeModel(prior_2d, simulator_2d)


@pytest.fixture
def observed_2d():
    return jnp.array([0.5, -0.3])


# ---------------------------------------------------------------------------
# Amortized NPE via train_sbi
# ---------------------------------------------------------------------------


@pytest.mark.sbi
class TestTrainSBI:
    def test_returns_trained_model(self, prior, simulator):
        trained = train_sbi._func(
            prior, simulator,
            method="npe",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
            random_seed=42,
        )
        assert isinstance(trained, TrainedSBIModel)

    def test_supports_conditioning(self, prior, simulator):
        trained = train_sbi._func(
            prior, simulator,
            method="npe",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
        )
        assert isinstance(trained, SupportsConditioning)

    def test_condition_on_produces_approximate_dist(self, prior, simulator, observed):
        trained = train_sbi._func(
            prior, simulator,
            method="npe",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
            n_samples=100,
        )
        posterior = condition_on._func(trained, observed)
        assert isinstance(posterior, ApproximateDistribution)
        assert posterior.algorithm == "sbijax_npe"

    def test_nle_method(self, prior, simulator, observed):
        trained = train_sbi._func(
            prior, simulator,
            method="nle",
            n_simulations=200,
            n_iter=5,
            batch_size=32,
            n_samples=100,
        )
        posterior = condition_on._func(trained, observed)
        assert isinstance(posterior, ApproximateDistribution)
        assert posterior.algorithm == "sbijax_nle"

    def test_invalid_method(self, prior, simulator):
        with pytest.raises(ValueError, match="Unknown SBI method"):
            train_sbi._func(
                prior, simulator,
                method="invalid",
                n_simulations=100,
                n_iter=5,
            )


# ---------------------------------------------------------------------------
# SMCABC via registry
# ---------------------------------------------------------------------------


@pytest.mark.sbi
class TestSMCABC:
    def test_smcabc_via_method_override(self, generative_model_2d, observed_2d):
        from probpipe.inference import inference_method_registry

        result = inference_method_registry.execute(
            generative_model_2d, observed_2d,
            method="sbijax_smcabc",
            n_rounds=2,
            n_particles=100,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.algorithm == "sbijax_smcabc"

    def test_smcabc_check_feasibility(self, generative_model_2d, observed_2d):
        from probpipe.inference._sbijax_methods import SbiSMCABCMethod

        method = SbiSMCABCMethod()
        info = method.check(generative_model_2d, observed_2d)
        assert info.feasible

    def test_smcabc_rejects_non_generative_model(self, observed):
        from probpipe import SimpleModel
        from probpipe.inference._sbijax_methods import SbiSMCABCMethod

        class DummyLik:
            def log_likelihood(self, params, data):
                return 0.0

        model = SimpleModel(Normal(0.0, 1.0), DummyLik())
        method = SbiSMCABCMethod()
        info = method.check(model, observed)
        assert not info.feasible
