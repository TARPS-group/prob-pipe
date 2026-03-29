"""Tests for SimpleModel.

Covers:
- Construction from prior + likelihood
- Dynamic protocol detection (SupportsLogProb, SupportsSampling, SupportsMean)
- conditionable_components / required_observations
- condition_on → MCMCApproximateDistribution
- Event shape delegation
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    Likelihood,
    MCMCApproximateDistribution,
    SimpleModel,
    SupportsConditionableComponents,
    SupportsLogProb,
    SupportsNamedComponents,
    condition_on,
    log_prob,
    mean,
    sample,
)
from probpipe.modeling import Likelihood as LikelihoodBase
from probpipe.core.node import wf
from probpipe.distributions.multivariate import MultivariateNormal


# ---------------------------------------------------------------------------
# Test likelihood
# ---------------------------------------------------------------------------


class GaussianLikelihood(LikelihoodBase):
    """Simple Gaussian likelihood for testing."""

    @wf
    def log_likelihood(self, params, data):
        return -0.5 * jnp.sum((data - params) ** 2)


# ---------------------------------------------------------------------------
# SimpleModel tests
# ---------------------------------------------------------------------------


class TestSimpleModel:
    """Test SimpleModel construction and protocol support."""

    @pytest.fixture
    def prior(self):
        return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10)

    @pytest.fixture
    def likelihood(self):
        return GaussianLikelihood()

    @pytest.fixture
    def model(self, prior, likelihood):
        return SimpleModel(prior, likelihood, name="test_model")

    def test_construction(self, model):
        assert isinstance(model, SimpleModel)
        assert model.name == "test_model"

    def test_event_shape(self, model):
        assert model.event_shape == (2,)

    def test_component_names(self, model):
        names = model.component_names
        assert "data" in names
        assert isinstance(names, tuple)

    def test_parameter_names(self, model):
        names = model.parameter_names
        assert isinstance(names, tuple)

    def test_conditionable_components(self, model):
        cc = model.conditionable_components
        assert isinstance(cc, dict)
        # "data" should be required
        assert cc["data"] is True

    def test_required_observations(self, model):
        ro = model.required_observations
        assert "data" in ro

    def test_supports_named_components(self, model):
        assert isinstance(model, SupportsNamedComponents)

    def test_supports_conditionable_components(self, model):
        assert isinstance(model, SupportsConditionableComponents)

    def test_getitem_prior(self, model):
        p = model["prior"]
        assert p is not None

    def test_getitem_unknown_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_repr(self, model):
        r = repr(model)
        assert "SimpleModel" in r
        assert "MultivariateNormal" in r

    def test_log_prob_delegation(self, model):
        """SimpleModel log_prob delegates to prior."""
        x = jnp.zeros(2)
        lp = model._log_prob(x)
        assert jnp.isfinite(lp)

    def test_sample_delegation(self, model):
        """SimpleModel sample delegates to prior."""
        key = jax.random.PRNGKey(0)
        s = model._sample(key)
        assert s.shape == (2,)

    def test_condition_on(self, model):
        """condition_on returns MCMCApproximateDistribution."""
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        result = model._condition_on(
            data,
            num_results=50,
            num_warmup=20,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)
        assert result.event_shape == (2,)

    def test_condition_on_via_ops(self, model):
        """condition_on op works with SimpleModel."""
        data = jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])
        result = condition_on(
            model,
            data,
            num_results=50,
            num_warmup=20,
            step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)


class TestSimpleModelDynamicProtocol:
    """Test that SimpleModel raises when prior doesn't support protocol."""

    def test_log_prob_requires_supports_log_prob(self):
        """log_prob raises if prior doesn't support it."""
        from probpipe import EmpiricalDistribution

        emp = EmpiricalDistribution(jnp.ones((10, 2)))
        lik = GaussianLikelihood()
        model = SimpleModel(emp, lik)

        with pytest.raises(TypeError, match="does not support log_prob"):
            model._log_prob(jnp.zeros(2))

    def test_custom_data_names(self):
        """SimpleModel with custom data_names."""
        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2))
        lik = GaussianLikelihood()
        model = SimpleModel(prior, lik, data_names=("observations", "covariates"))

        assert "observations" in model.conditionable_components
        assert "covariates" in model.conditionable_components
        assert model.conditionable_components["observations"] is True
        assert model.conditionable_components["covariates"] is True

    def test_sample_requires_supports_sampling(self):
        """sample raises if prior doesn't support it."""
        from probpipe import ArrayDistribution

        class NoSampleDist(ArrayDistribution):
            @property
            def event_shape(self):
                return (2,)

        prior = NoSampleDist()
        lik = GaussianLikelihood()
        model = SimpleModel(prior, lik)

        with pytest.raises(TypeError, match="does not support sampling"):
            model._sample(jax.random.PRNGKey(0))

    def test_parameter_names_from_prior_components(self):
        """SimpleModel derives parameter_names from prior's component_names."""
        from probpipe import ProductDistribution, Normal

        prior = ProductDistribution(
            alpha=Normal(loc=0.0, scale=1.0),
            beta=Normal(loc=0.0, scale=1.0),
        )
        lik = GaussianLikelihood()
        model = SimpleModel(prior, lik)

        assert "alpha" in model.parameter_names
        assert "beta" in model.parameter_names


class TestSimpleModelConditioningPaths:
    """Test conditioning paths: HMC, zero warmup, bad algorithm, RWMH fallback."""

    @pytest.fixture
    def prior(self):
        return MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10)

    @pytest.fixture
    def likelihood(self):
        return GaussianLikelihood()

    @pytest.fixture
    def model(self, prior, likelihood):
        return SimpleModel(prior, likelihood)

    @pytest.fixture
    def data(self):
        return jnp.array([[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]])

    def test_condition_on_hmc(self, model, data):
        """Condition with algorithm='hmc'."""
        result = model._condition_on(
            data, num_results=30, num_warmup=10, step_size=0.3,
            random_seed=42, algorithm="hmc",
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_condition_on_zero_warmup(self, model, data):
        """Condition with num_warmup=0 (no adaptation)."""
        result = model._condition_on(
            data, num_results=30, num_warmup=0, step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_condition_on_bad_algorithm(self, model, data):
        """Bad algorithm name raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be"):
            model._condition_on(
                data, num_results=30, num_warmup=10, algorithm="bad",
            )

    def test_extract_diagnostics_step_size_branch(self):
        """Test _extract_diagnostics with step_size but no new_step_size."""
        from probpipe.modeling._simple import _extract_diagnostics
        from probpipe.inference._diagnostics import MCMCDiagnostics

        # Mock trace with step_size but no new_step_size
        trace = type("Trace", (), {
            "step_size": jnp.array(0.1),
            "log_accept_ratio": jnp.zeros(5),
            "is_accepted": jnp.ones(5, dtype=bool),
        })()
        diag = _extract_diagnostics(trace, "test")
        assert isinstance(diag, MCMCDiagnostics)
        np.testing.assert_allclose(diag.final_step_size, 0.1, atol=1e-5)

    def test_extract_diagnostics_no_step_size(self):
        """Test _extract_diagnostics with no step_size attributes."""
        from probpipe.modeling._simple import _extract_diagnostics

        trace = type("Trace", (), {
            "log_accept_ratio": jnp.zeros(5),
        })()
        diag = _extract_diagnostics(trace, "test")
        assert jnp.isnan(jnp.asarray(diag["step_size"]))

    def test_get_init_state_mean_exception_fallback(self, likelihood):
        """_get_init_state falls back to data mean when _mean() raises."""
        from probpipe import ArrayDistribution
        from probpipe.core.protocols import SupportsMean, SupportsExpectation

        class BrokenMeanDist(ArrayDistribution, SupportsMean):
            @property
            def event_shape(self):
                return (2,)

            def _mean(self):
                raise RuntimeError("broken")

            def _expectation(self, f, *, key=None, num_evaluations=None, return_dist=None):
                raise RuntimeError("broken")

        prior = BrokenMeanDist()
        model = SimpleModel(prior, likelihood)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        init = model._get_init_state(None, data)
        np.testing.assert_allclose(init, jnp.array([2.0, 3.0]), atol=1e-5)

    def test_get_init_state_no_mean_protocol(self, likelihood):
        """_get_init_state falls back to data mean when prior has no SupportsMean."""
        from probpipe import ArrayDistribution
        from probpipe.core.distribution import _vmap_sample
        from probpipe.core.protocols import SupportsSampling

        class NoMeanDist(ArrayDistribution, SupportsSampling):
            _sampling_cost = "low"
            _preferred_orchestration = None

            @property
            def event_shape(self):
                return (2,)

            def _sample_one(self, key):
                return jax.random.normal(key, (2,))

            def _sample(self, key, sample_shape=()):
                return _vmap_sample(self, key, sample_shape)

        prior = NoMeanDist()
        model = SimpleModel(prior, likelihood)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        init = model._get_init_state(None, data)
        np.testing.assert_allclose(init, jnp.array([2.0, 3.0]), atol=1e-5)
