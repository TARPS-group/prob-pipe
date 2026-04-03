"""Tests for the nutpie workflow function."""

import sys
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from probpipe.inference._nutpie import (
    _compile_for_nutpie,
    _extract_chains,
    condition_on_nutpie,
)
from probpipe.inference import MCMCApproximateDistribution, InferenceDiagnostics
from probpipe.inference._diagnostics import extract_arviz_diagnostics


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
# extract_arviz_diagnostics (nutpie path)
# ---------------------------------------------------------------------------


def _make_mock_sample_stats(
    num_chains=2, num_draws=20, *, include=None, exclude=None,
):
    """Build a mock sample_stats xarray-like object."""
    all_fields = {
        "acceptance_rate": np.random.uniform(0.5, 1.0, (num_chains, num_draws)),
        "step_size": np.full((num_chains, num_draws), 0.05),
        "diverging": np.zeros((num_chains, num_draws), dtype=bool),
        "tree_depth": np.random.randint(1, 8, (num_chains, num_draws)),
        "n_steps": np.random.randint(1, 128, (num_chains, num_draws)),
        "energy": np.random.randn(num_chains, num_draws),
        "energy_error": np.random.randn(num_chains, num_draws) * 0.01,
        "lp": np.random.randn(num_chains, num_draws),
    }
    if include is not None:
        all_fields = {k: v for k, v in all_fields.items() if k in include}
    if exclude is not None:
        all_fields = {k: v for k, v in all_fields.items() if k not in exclude}

    stats = MagicMock()
    stats.__contains__ = lambda self, k: k in all_fields
    stats.__getitem__ = lambda self, k: MagicMock(values=all_fields[k])
    return stats


class TestExtractArvizDiagnosticsNutpie:
    def test_full_stats(self):
        """All sample_stats fields are extracted."""
        trace = MagicMock()
        trace.sample_stats = _make_mock_sample_stats(num_chains=2, num_draws=10)

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=10, num_chains=2)

        assert diag.algorithm == "nutpie_nuts"
        assert diag["log_accept_ratio"].shape == (20,)
        assert diag["step_size"].shape == (20,)
        assert 0.0 < diag.accept_rate <= 1.0
        # Extra diagnostics
        assert "diverging" in diag
        assert "n_divergences" in diag
        assert diag["n_divergences"] == 0
        assert "tree_depth" in diag
        assert diag["tree_depth"].shape == (20,)
        assert "n_steps" in diag
        assert "energy" in diag
        assert "energy_error" in diag
        assert "lp" in diag

    def test_no_sample_stats(self):
        """Falls back to zeros when sample_stats is missing."""
        trace = MagicMock(spec=["posterior"])

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=5, num_chains=2)

        assert diag.algorithm == "nutpie_nuts"
        assert diag["log_accept_ratio"].shape == (10,)
        assert float(jnp.sum(diag["log_accept_ratio"])) == 0.0

    def test_partial_stats(self):
        """Only available fields are extracted."""
        trace = MagicMock()
        trace.sample_stats = _make_mock_sample_stats(
            num_chains=1, num_draws=5,
            include={"acceptance_rate", "diverging"},
        )

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=5, num_chains=1)

        assert diag["log_accept_ratio"].shape == (5,)
        assert "diverging" in diag
        assert "tree_depth" not in diag
        assert "energy" not in diag
        assert "step_size" not in diag

    def test_divergences_counted(self):
        """n_divergences reflects actual divergent transitions."""
        trace = MagicMock()
        stats = _make_mock_sample_stats(num_chains=1, num_draws=10,
                                        include={"diverging"})
        # Inject 3 divergences
        div_arr = np.zeros((1, 10), dtype=bool)
        div_arr[0, [2, 5, 7]] = True
        stats.__getitem__ = lambda self, k: MagicMock(values=div_arr)
        trace.sample_stats = stats

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=10, num_chains=1)

        assert diag["n_divergences"] == 3
        assert int(jnp.sum(diag["diverging"])) == 3

    def test_summary_includes_extras(self):
        """summary() includes scalar extras."""
        trace = MagicMock()
        trace.sample_stats = _make_mock_sample_stats(num_chains=1, num_draws=5)

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=5, num_chains=1)

        s = diag.summary()
        assert "n_divergences=0" in s
        assert "tree_depth=" in s

    def test_dict_access(self):
        """Dict-style get/set/contains/keys work."""
        trace = MagicMock()
        trace.sample_stats = _make_mock_sample_stats(
            num_chains=1, num_draws=5, include={"acceptance_rate"},
        )

        diag = extract_arviz_diagnostics(trace, algorithm="nutpie_nuts", num_results=5, num_chains=1)

        # set
        diag["custom"] = 42
        assert "custom" in diag
        assert diag["custom"] == 42
        assert diag.get("custom") == 42
        assert diag.get("missing", -1) == -1
        assert "custom" in list(diag.keys())

        with pytest.raises(KeyError, match="No diagnostic named"):
            diag["nonexistent"]


# ---------------------------------------------------------------------------
# condition_on_nutpie._func
# ---------------------------------------------------------------------------


class TestNutpieSampleImpl:
    def test_full_sampling_path(self, mock_nutpie):
        """End-to-end test with mocked nutpie."""
        model = MagicMock()
        model._bridgestan_model.return_value = "bs_model"
        mock_nutpie.compile_stan_model.return_value = "compiled"

        # Mock trace with posterior and sample_stats
        mock_trace = MagicMock()
        mock_posterior = MagicMock()
        mu_vals = np.random.randn(2, 20)
        mu_var = MagicMock()
        mu_var.values = mu_vals
        mock_posterior.data_vars = ["mu"]
        mock_posterior.__getitem__ = lambda self, k: mu_var
        mock_trace.posterior = mock_posterior
        mock_trace.sample_stats = _make_mock_sample_stats(
            num_chains=2, num_draws=20,
        )
        mock_nutpie.sample.return_value = mock_trace

        result = condition_on_nutpie._func(
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
        assert 0.0 < result.diagnostics.accept_rate <= 1.0
        assert "diverging" in result.diagnostics
        assert result.source is not None
        assert result.source.operation == "nutpie_nuts"

    def test_import_error(self):
        """Raises ImportError with install instructions when nutpie missing."""
        with patch.dict("sys.modules", {"nutpie": None}):
            with pytest.raises(ImportError, match="pip install nutpie"):
                condition_on_nutpie._func(MagicMock(), num_results=10)
