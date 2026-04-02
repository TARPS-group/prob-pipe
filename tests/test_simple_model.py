"""Tests for SimpleModel.

Covers:
- Construction from prior + likelihood
- Joint log-prob over (params, data) pairs
- SupportsLogProb always satisfied (prior must support it)
- No event_shape, no _sample
- Named components: "parameters" and "data"
- conditionable_components / required_observations
- condition_on → MCMCApproximateDistribution
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
    SupportsSampling,
    condition_on,
)
from probpipe.distributions.multivariate import MultivariateNormal


# ---------------------------------------------------------------------------
# Test likelihood (plain class — satisfies Likelihood protocol)
# ---------------------------------------------------------------------------


class GaussianLikelihood:
    """Simple Gaussian likelihood for testing."""

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

    def test_requires_supports_log_prob_prior(self):
        """SimpleModel rejects priors that don't support SupportsLogProb."""
        from probpipe import EmpiricalDistribution

        emp = EmpiricalDistribution(jnp.ones((10, 2)))
        lik = GaussianLikelihood()
        with pytest.raises(TypeError, match="SupportsLogProb"):
            SimpleModel(emp, lik)

    def test_always_supports_log_prob(self, model):
        """SimpleModel always satisfies SupportsLogProb."""
        assert isinstance(model, SupportsLogProb)

    def test_no_event_shape(self, model):
        """SimpleModel does not define event_shape."""
        assert not hasattr(model, "event_shape")

    def test_no_sample(self, model):
        """SimpleModel does not define _sample even if prior supports sampling."""
        assert not isinstance(model, SupportsSampling)

    def test_component_names(self, model):
        names = model.component_names
        assert "parameters" in names
        assert "data" in names
        assert isinstance(names, tuple)

    def test_parameter_names(self, model):
        assert model.parameter_names == ("parameters",)

    def test_conditionable_components(self, model):
        cc = model.conditionable_components
        assert isinstance(cc, dict)
        assert cc["data"] is True
        assert cc["parameters"] is False

    def test_required_observations(self, model):
        ro = model.required_observations
        assert "data" in ro
        assert "parameters" not in ro

    def test_supports_named_components(self, model):
        assert isinstance(model, SupportsNamedComponents)

    def test_supports_conditionable_components(self, model):
        assert isinstance(model, SupportsConditionableComponents)

    def test_getitem_parameters(self, model, prior):
        assert model["parameters"] is prior

    def test_getitem_data(self, model, likelihood):
        assert model["data"] is likelihood

    def test_getitem_unknown_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_repr(self, model):
        r = repr(model)
        assert "SimpleModel" in r
        assert "MultivariateNormal" in r

    # -- Joint log-prob over (params, data) pairs --------------------------

    def test_log_prob_joint(self, model, prior):
        """_log_prob accepts (params, data) tuple and returns prior + likelihood."""
        params = jnp.zeros(2)
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        lp = model._log_prob((params, data))
        assert jnp.isfinite(lp)
        # Should equal prior log-prob + likelihood
        expected = prior._log_prob(params) + (-0.5 * jnp.sum((data - params) ** 2))
        np.testing.assert_allclose(lp, expected, atol=1e-5)

    # -- Conditioning ------------------------------------------------------

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


class TestSimpleModelConditioningPaths:
    """Test conditioning paths: HMC, zero warmup, bad algorithm, explicit init, RWMH fallback."""

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
        result = model._condition_on(
            data, num_results=30, num_warmup=10, step_size=0.3,
            random_seed=42, algorithm="hmc",
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_condition_on_zero_warmup(self, model, data):
        result = model._condition_on(
            data, num_results=30, num_warmup=0, step_size=0.3,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_condition_on_bad_algorithm(self, model, data):
        with pytest.raises(ValueError, match="algorithm must be"):
            model._condition_on(
                data, num_results=30, num_warmup=10, algorithm="bad",
            )

    def test_condition_on_explicit_init(self, model, data):
        result = model._condition_on(
            data, num_results=30, num_warmup=10, step_size=0.3,
            random_seed=42, init=jnp.ones(2),
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_rwmh_fallback(self):
        """RWMH fallback when likelihood is not JAX-traceable."""

        class NonTraceableLikelihood:
            def log_likelihood(self, params, data):
                # Python-side branching prevents JAX tracing
                if float(params[0]) > 100:
                    return jnp.float32(-1e10)
                return jnp.float32(-0.5 * np.sum((np.array(data) - np.array(params)) ** 2))

        prior = MultivariateNormal(loc=jnp.zeros(2), cov=jnp.eye(2) * 10)
        model = SimpleModel(prior, NonTraceableLikelihood())
        data = jnp.array([[1.0, 2.0], [1.5, 2.5]])
        result = model._condition_on(
            data, num_results=30, num_warmup=10, step_size=0.3, random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)

    def test_extract_diagnostics_step_size_branch(self):
        from probpipe.modeling._simple import _extract_diagnostics

        trace = type("Trace", (), {
            "step_size": jnp.array(0.1),
            "log_accept_ratio": jnp.array([-0.5]),
        })()
        diag = _extract_diagnostics(trace, "nuts")
        assert diag.algorithm == "nuts"

    def test_extract_diagnostics_no_step_size(self):
        from probpipe.modeling._simple import _extract_diagnostics

        trace = type("Trace", (), {
            "log_accept_ratio": jnp.array([-0.5]),
        })()
        diag = _extract_diagnostics(trace, "nuts")
        assert jnp.isnan(diag.final_step_size)
