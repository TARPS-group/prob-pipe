"""Tests for PyMCModel.

These tests require pymc to be installed.
"""

import pytest

pm = pytest.importorskip("pymc")

import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock, patch

from probpipe import SupportsConditionableComponents, MCMCApproximateDistribution
from probpipe.modeling import PyMCModel


def simple_model_fn(y=None):
    """Simple PyMC model for testing."""
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", mu, sigma, observed=y)
    return m


class TestPyMCModel:
    """Test PyMCModel construction and protocol compliance."""

    @pytest.fixture
    def model(self):
        return PyMCModel(simple_model_fn, name="test_pymc")

    def test_construction(self, model):
        assert isinstance(model, PyMCModel)
        assert model.name == "test_pymc"

    def test_parameter_names(self, model):
        names = model.parameter_names
        assert "mu" in names
        assert "sigma" in names

    def test_component_names(self, model):
        names = model.component_names
        assert "mu" in names
        assert "sigma" in names
        assert "y" in names

    def test_conditionable_components(self, model):
        cc = model.conditionable_components
        assert cc["mu"] is False  # parameter, optional
        assert cc["sigma"] is False
        assert cc["y"] is True  # observed, required

    def test_required_observations(self, model):
        ro = model.required_observations
        assert "y" in ro

    def test_supports_conditionable_components(self, model):
        assert isinstance(model, SupportsConditionableComponents)

    def test_repr(self, model):
        r = repr(model)
        assert "PyMCModel" in r
        assert "mu" in r
        assert "sigma" in r

    def test_getitem(self, model):
        assert model["mu"] == "mu"

    def test_getitem_unknown(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_event_shape(self, model):
        es = model.event_shape
        assert isinstance(es, tuple)
        assert es[0] == 2  # mu (scalar) + sigma (scalar)

    def test_sample_scalar(self, model):
        import jax
        key = jax.random.PRNGKey(0)
        s = model._sample(key, sample_shape=())
        assert s.shape == (2,)  # 2 scalar params

    def test_sample_batched(self, model):
        import jax
        key = jax.random.PRNGKey(0)
        s = model._sample(key, sample_shape=(5,))
        assert s.shape == (5, 2)

    def test_pymc_model_no_data(self, model):
        m = model._pymc_model()
        assert m is not None

    def test_pymc_model_dict_data(self, model):
        data = np.random.randn(20)
        m = model._pymc_model(data={"y": data})
        # Should have observed data
        assert len(m.observed_RVs) > 0

    def test_pymc_model_array_data(self, model):
        data = np.random.randn(20)
        m = model._pymc_model(data=data)
        assert len(m.observed_RVs) > 0

    def test_condition_on(self, model):
        """condition_on runs PyMC sampling and returns MCMCApproximateDistribution."""
        data = np.random.randn(50)
        result = model._condition_on(
            {"y": data},
            num_results=20,
            num_warmup=10,
            num_chains=1,
            random_seed=42,
        )
        assert isinstance(result, MCMCApproximateDistribution)
        assert result.num_chains == 1
        assert result.num_draws == 20
        assert result.diagnostics is not None
        assert result.diagnostics.algorithm == "pymc_nuts"
        assert result.source is not None
        assert result.source.operation == "pymc_sample"
