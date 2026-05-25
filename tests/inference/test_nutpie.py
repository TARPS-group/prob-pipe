"""Tests for the nutpie workflow function.

These tests require nutpie (and pymc, for the PyMC integration path) to
be installed.  Helper / error-path tests that don't require a compiled
model are isolated in ``TestHelpers``.
"""

import sys
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

nutpie = pytest.importorskip("nutpie")

from probpipe.inference._nutpie import (
    _compile_for_nutpie,
    _extract_chains,
    condition_on_nutpie,
)
from probpipe.inference import ApproximateDistribution


# ---------------------------------------------------------------------------
# Helpers (no model compilation needed)
# ---------------------------------------------------------------------------


class TestCompileForNutpie:
    """_compile_for_nutpie dispatch — still uses mocks since we only test
    which nutpie function is called, not that it produces a runnable model."""

    def test_bridgestan_path(self):
        """Models with _bridgestan_model use nutpie.compile_stan_model."""
        model = MagicMock()
        model._bridgestan_model.return_value = "bs_model"
        with patch.object(nutpie, "compile_stan_model",
                          return_value="compiled") as compile_stan:
            result = _compile_for_nutpie(model, data={"N": 10})
        compile_stan.assert_called_once_with("bs_model")
        model._bridgestan_model.assert_called_once_with(data={"N": 10})
        assert result == "compiled"

    def test_pymc_path(self):
        """Models with _pymc_model use nutpie.compile_pymc_model."""
        model = MagicMock(spec=[])
        model._pymc_model = MagicMock(return_value="pm_model")
        with patch.object(nutpie, "compile_pymc_model",
                          return_value="compiled") as compile_pymc:
            result = _compile_for_nutpie(model, data={"y": [1, 2]})
        compile_pymc.assert_called_once_with("pm_model")
        model._pymc_model.assert_called_once_with(data={"y": [1, 2]})
        assert result == "compiled"

    def test_unsupported_model_raises(self):
        model = MagicMock(spec=[])
        with pytest.raises(TypeError, match="does not support"):
            _compile_for_nutpie(model, data=None)


class TestImportError:
    """When nutpie is missing, condition_on_nutpie raises a helpful
    ImportError.  This path is exercised by temporarily hiding nutpie."""

    def test_import_error_message(self):
        with patch.dict("sys.modules", {"nutpie": None}):
            with pytest.raises(ImportError, match="pip install nutpie"):
                condition_on_nutpie._func(MagicMock(), num_results=10)


# ---------------------------------------------------------------------------
# _extract_chains — exercised with real ArviZ trace structure
# ---------------------------------------------------------------------------


class TestExtractChains:
    """Tests against mock arviz-like objects whose `values` attribute is a
    real numpy array, matching nutpie's actual trace shape."""

    def test_scalar_params_two_chains(self):
        mock_trace = MagicMock()
        mu_vals = np.random.randn(2, 10)
        sigma_vals = np.random.randn(2, 10)
        mu_var = MagicMock(); mu_var.values = mu_vals
        sigma_var = MagicMock(); sigma_var.values = sigma_vals

        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["mu", "sigma"]
        mock_posterior.__getitem__ = (
            lambda self, k: {"mu": mu_var, "sigma": sigma_var}[k]
        )
        mock_trace.posterior = mock_posterior

        chains, param_names = _extract_chains(mock_trace, num_chains=2)

        assert param_names == ["mu", "sigma"]
        assert len(chains) == 2
        assert chains[0].shape == (10, 2)
        np.testing.assert_allclose(chains[0][:, 0], mu_vals[0])
        np.testing.assert_allclose(chains[1][:, 1], sigma_vals[1])

    def test_multidim_params(self):
        mock_trace = MagicMock()
        beta_vals = np.random.randn(1, 5, 3)
        beta_var = MagicMock(); beta_var.values = beta_vals

        mock_posterior = MagicMock()
        mock_posterior.data_vars = ["beta"]
        mock_posterior.__getitem__ = lambda self, k: beta_var
        mock_trace.posterior = mock_posterior

        chains, _ = _extract_chains(mock_trace, num_chains=1)
        assert chains[0].shape == (5, 3)

    def test_no_posterior_raises(self):
        mock_trace = MagicMock(spec=[])
        with pytest.raises(TypeError, match="Cannot extract chains"):
            _extract_chains(mock_trace, num_chains=1)


# ---------------------------------------------------------------------------
# Real integration: nutpie + PyMCModel
# ---------------------------------------------------------------------------


pm = pytest.importorskip("pymc")

from probpipe.modeling import PyMCModel  # noqa: E402


def _gaussian_pymc_fn(y=None):
    """Known-conjugate Gaussian model.  Posterior mean for mu is
    (n * y_bar / sigma^2) / (1/tau_0^2 + n/sigma^2) with tau_0=10,
    sigma=1, known from closed form."""
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("y", mu, 1.0, observed=y)
    return m


class TestNutpieIntegration:
    """End-to-end sampling via real nutpie + PyMC compilation."""

    def test_samples_recover_posterior(self):
        """Nutpie recovers the analytical posterior mean for a simple Gaussian."""
        np.random.seed(0)
        y_obs = np.array([1.2, 0.8, 1.1, 0.9, 1.0], dtype=float)
        model = PyMCModel(_gaussian_pymc_fn, name="gaussian")
        result = condition_on_nutpie._func(
            model, data={"y": y_obs},
            num_results=500, num_warmup=200, num_chains=2, random_seed=42,
        )
        assert isinstance(result, ApproximateDistribution)
        assert result.num_chains == 2
        assert result.algorithm == "nutpie_nuts"
        assert result.source is not None
        assert result.source.operation == "nutpie_nuts"
        # Analytical posterior: prior N(0, 10), likelihood N(mu, 1) with n=5
        #   Precision: 1/100 + 5 = 5.01  ->  var = 0.1996
        #   Mean:      5.0 * y_bar / 5.01
        y_bar = float(y_obs.mean())
        post_mean = 5.0 * y_bar / (1.0 / 100.0 + 5.0)
        post_sd = np.sqrt(1.0 / (1.0 / 100.0 + 5.0))
        # PyMCModel.record_template is None, so draws() returns a flat
        # (num_draws, n_params) array. The only parameter is mu.
        draws = result.draws()
        assert draws.shape[1] == 1
        mu_draws = draws[:, 0]
        # With 1000 draws total, MC SE for mean ~ post_sd / sqrt(1000) ~ 0.014
        np.testing.assert_allclose(float(jnp.mean(mu_draws)), post_mean, atol=0.05)
        np.testing.assert_allclose(float(jnp.std(mu_draws)), post_sd, atol=0.05)

    def test_auxiliary_trace_attached(self):
        model = PyMCModel(_gaussian_pymc_fn, name="gaussian")
        y_obs = np.array([0.0, 1.0], dtype=float)
        result = condition_on_nutpie._func(
            model, data={"y": y_obs},
            num_results=50, num_warmup=50, num_chains=1, random_seed=0,
        )
        assert result.inference_data is not None
        # arviz-like trace exposes posterior as an xarray Dataset/DataTree
        assert hasattr(result.inference_data, "posterior")
