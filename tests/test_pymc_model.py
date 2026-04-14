"""Tests for PyMCModel.

These tests require pymc to be installed.
"""

import pytest

pm = pytest.importorskip("pymc")

import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock, patch

from probpipe import ApproximateDistribution
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

    def test_supports_named_components(self, model):
        assert hasattr(model, 'component_names')
        assert len(model.component_names) > 0

    def test_repr(self, model):
        r = repr(model)
        assert "PyMCModel" in r
        assert "mu" in r
        assert "sigma" in r

    def test_getitem_returns_name_placeholder(self, model):
        """PyMCModel['mu'] returns the name — PyMC doesn't expose
        sub-distributions, so __getitem__ only validates the key. See the
        comment in PyMCModel.__getitem__.
        """
        assert model["mu"] == "mu"
        assert model["sigma"] == "sigma"

    def test_getitem_unknown_key_raises(self, model):
        with pytest.raises(KeyError):
            model["nonexistent"]

    def test_event_shape_values(self, model):
        """mu (scalar) + sigma (scalar) -> event_shape == (2,)."""
        assert model.event_shape == (2,)

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
        """condition_on runs PyMC sampling and returns ApproximateDistribution."""
        from probpipe import condition_on

        data = np.random.randn(50)
        result = condition_on(
            model,
            {"y": data},
            num_results=20,
            num_warmup=10,
            num_chains=1,
            random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 1
        assert result.num_draws == 20
        assert result.algorithm == "pymc_nuts"
        assert result.inference_data is not None
        assert hasattr(result.inference_data, "posterior")
        assert hasattr(result.inference_data, "sample_stats")
        assert result.source is not None
        assert result.source.operation == "pymc_nuts"
