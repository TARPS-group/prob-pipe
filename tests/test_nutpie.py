"""Tests for the nutpie workflow function."""

import sys
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from probpipe.inference._nutpie import (
    _compile_for_nutpie,
    _extract_chains,
    _condition_on_nutpie_impl,
)
from probpipe.inference import MCMCApproximateDistribution


@pytest.fixture(autouse=True)
def mock_nutpie():
    """Provide a mock nutpie module for all tests in this file."""
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"nutpie": mock_mod}):
        yield mock_mod


# ---------------------------------------------------------------------------
# _compile_for_nutpie
# ---------------------------------------------------------------------------


class TestCompileForNutpie:
    def test_bridgestan_path(self, mock_nutpie):
        """Models with _bridgestan_model use nutpie.compile_stan_model."""
        model = MagicMock()
        model._bridgestan_model.return_value = "bs_model"
        mock_nutpie.compile_stan_model.return_value = "compiled"

        result = _compile_for_nutpie(model, data={"N": 10})

        mock_nutpie.compile_stan_model.assert_called_once_with("bs_model")
        model._bridgestan_model.assert_called_once_with(data={"N": 10})
        assert result == "compiled"

    def test_pymc_path(self, mock_nutpie):
        """Models with _pymc_model use nutpie.compile_pymc_model."""
        model = MagicMock(spec=[])  # no _bridgestan_model
        model._pymc_model = MagicMock(return_value="pm_model")
        mock_nutpie.compile_pymc_model.return_value = "compiled"

        result = _compile_for_nutpie(model, data={"y": [1, 2]})

        mock_nutpie.compile_pymc_model.assert_called_once_with("pm_model")
        model._pymc_model.assert_called_once_with(data={"y": [1, 2]})
        assert result == "compiled"

    def test_unsupported_model_raises(self):
        """Models without _bridgestan_model or _pymc_model raise TypeError."""
        model = MagicMock(spec=[])  # no relevant attributes
        with pytest.raises(TypeError, match="does not support"):
            _compile_for_nutpie(model, data=None)


# ---------------------------------------------------------------------------
# _extract_chains
# ---------------------------------------------------------------------------


class TestExtractChains:
    def test_scalar_params(self):
        """Extract scalar parameters from mock InferenceData."""
        mock_trace = MagicMock()
        mock_posterior = MagicMock()

        # Two params, 2 chains, 10 draws each
        mu_vals = np.random.randn(2, 10)
        sigma_vals = np.random.randn(2, 10)
        mu_var = MagicMock()
        mu_var.values = mu_vals
        sigma_var = MagicMock()
        sigma_var.values = sigma_vals

        mock_posterior.data_vars = ["mu", "sigma"]
        mock_posterior.__getitem__ = lambda self, k: {"mu": mu_var, "sigma": sigma_var}[k]
        mock_trace.posterior = mock_posterior

        chains, param_names = _extract_chains(mock_trace, num_chains=2)

        assert param_names == ["mu", "sigma"]
        assert len(chains) == 2
        assert chains[0].shape == (10, 2)  # 10 draws, 2 params

    def test_multidim_params(self):
        """Extract multi-dimensional parameters."""
        mock_trace = MagicMock()
        mock_posterior = MagicMock()

        # One param with shape (chains=1, draws=5, dim=3)
        beta_vals = np.random.randn(1, 5, 3)
        beta_var = MagicMock()
        beta_var.values = beta_vals

        mock_posterior.data_vars = ["beta"]
        mock_posterior.__getitem__ = lambda self, k: beta_var
        mock_trace.posterior = mock_posterior

        chains, param_names = _extract_chains(mock_trace, num_chains=1)

        assert len(chains) == 1
        assert chains[0].shape == (5, 3)

    def test_no_posterior_raises(self):
        """Traces without .posterior raise TypeError."""
        mock_trace = MagicMock(spec=[])  # no .posterior
        with pytest.raises(TypeError, match="Cannot extract chains"):
            _extract_chains(mock_trace, num_chains=1)


# ---------------------------------------------------------------------------
# _condition_on_nutpie_impl
# ---------------------------------------------------------------------------


class TestNutpieSampleImpl:
    def test_full_sampling_path(self, mock_nutpie):
        """End-to-end test with mocked nutpie."""
        model = MagicMock()
        model._bridgestan_model.return_value = "bs_model"
        mock_nutpie.compile_stan_model.return_value = "compiled"

        # Mock trace with posterior
        mock_trace = MagicMock()
        mock_posterior = MagicMock()
        mu_vals = np.random.randn(2, 20)
        mu_var = MagicMock()
        mu_var.values = mu_vals
        mock_posterior.data_vars = ["mu"]
        mock_posterior.__getitem__ = lambda self, k: mu_var
        mock_trace.posterior = mock_posterior
        mock_nutpie.sample.return_value = mock_trace

        result = _condition_on_nutpie_impl(
            model,
            data={"N": 10},
            num_results=20,
            num_warmup=10,
            num_chains=2,
            random_seed=42,
        )

        assert isinstance(result, MCMCApproximateDistribution)
        assert result.num_chains == 2
        assert result.diagnostics.algorithm == "nutpie_nuts"
        assert result.source is not None
        assert result.source.operation == "condition_on_nutpie"

    def test_import_error(self):
        """Raises ImportError with install instructions when nutpie missing."""
        with patch.dict("sys.modules", {"nutpie": None}):
            with pytest.raises(ImportError, match="pip install nutpie"):
                _condition_on_nutpie_impl(MagicMock(), num_results=10)
