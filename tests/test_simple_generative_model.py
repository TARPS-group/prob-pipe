"""Tests for SimpleGenerativeModel."""

import jax
import jax.numpy as jnp
import pytest

from probpipe import Normal, SimpleGenerativeModel
from probpipe.core.protocols import SupportsLogProb, SupportsSampling
from probpipe.modeling import GenerativeLikelihood, ProbabilisticModel


class GaussianSimulator:
    """Simple Gaussian simulator for testing."""

    def generate_data(self, params, n_samples, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        return params + jax.random.normal(key, shape=(n_samples,) + params.shape)


@pytest.fixture
def prior():
    return Normal(loc=0.0, scale=1.0)


@pytest.fixture
def simulator():
    return GaussianSimulator()


@pytest.fixture
def model(prior, simulator):
    return SimpleGenerativeModel(prior, simulator)


class TestConstruction:
    """SimpleGenerativeModel construction and input validation."""

    def test_default_name_is_none(self, model):
        assert model.name is None

    def test_with_name(self, prior, simulator):
        m = SimpleGenerativeModel(prior, simulator, name="test_model")
        assert m.name == "test_model"

    def test_rejects_non_sampling_prior(self, simulator):
        class BadPrior:
            pass

        with pytest.raises(TypeError, match="SupportsSampling"):
            SimpleGenerativeModel(BadPrior(), simulator)

    def test_rejects_non_generative_likelihood(self, prior):
        class BadLikelihood:
            def log_likelihood(self, params, data):
                return 0.0

        with pytest.raises(TypeError, match="GenerativeLikelihood"):
            SimpleGenerativeModel(prior, BadLikelihood())


class TestProtocols:
    def test_is_probabilistic_model(self, model):
        assert isinstance(model, ProbabilisticModel)

    def test_is_not_supports_log_prob(self, model):
        assert not isinstance(model, SupportsLogProb)

    def test_simulator_is_generative_likelihood(self, simulator):
        assert isinstance(simulator, GenerativeLikelihood)


class TestComponents:
    def test_component_names(self, model):
        assert model.component_names == ("parameters", "data")

    def test_parameter_names(self, model):
        assert model.parameter_names == ("parameters",)

    def test_getitem_parameters(self, model, prior):
        assert model["parameters"] is prior

    def test_getitem_data(self, model, simulator):
        assert model["data"] is simulator

    def test_getitem_invalid(self, model):
        with pytest.raises(KeyError):
            model["invalid"]


class TestRepr:
    def test_repr(self, model):
        r = repr(model)
        assert "SimpleGenerativeModel" in r
        assert "Normal" in r
        assert "GaussianSimulator" in r
