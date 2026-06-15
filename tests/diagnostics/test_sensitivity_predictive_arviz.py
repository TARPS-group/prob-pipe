"""Focused coverage for sensitivity, predictive checks, and ArviZ bridge."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._arviz_bridge import extract_draws, to_arviz_dataset
from probpipe.diagnostics._predictive_check import (
    _predictive_check_batched,
    _predictive_check_loop,
    _supports_key_arg,
    predictive_check,
)
from probpipe.diagnostics._sensitivity import (
    _build_loo_warnings,
    _compare_summaries,
    _finite_diff_gradient,
    _kl_divergence_gaussian,
    _posterior_summary,
    _reweighted_summary,
    data_sensitivity,
    power_scale_sensitivity,
    prior_sensitivity,
)


class _Record(dict):
    @property
    def fields(self):
        return list(self.keys())


class _DrawsPosterior:
    def __init__(self, draws: dict[str, np.ndarray]):
        self._draws = _Record(draws)

    def draws(self):
        return self._draws


class _SamplesPosterior:
    def __init__(self, samples):
        self.samples = samples


class _SamplingDistribution:
    def __init__(self):
        self.validation_results = []

    def _sample(self, key, shape):
        if shape == ():
            return jax.random.normal(key, ())
        return jax.random.normal(key, shape)


class _KeyedLikelihood:
    def generate_data(self, params, n_samples, *, key=None):
        noise = jax.random.normal(key, params.shape + (n_samples,)) * 0.0
        return params[..., None] + noise


class _LoopLikelihood:
    def generate_data(self, params, n_samples):
        return np.full(n_samples, float(params))


def test_extract_draws_supports_draws_records_dicts_and_samples():
    post = _DrawsPosterior({"alpha": np.arange(3), "beta": np.ones(3)})
    assert set(extract_draws(post)) == {"alpha", "beta"}

    class _DictDraws:
        def draws(self):
            return {"theta": [1.0, 2.0]}

    np.testing.assert_array_equal(extract_draws(_DictDraws())["theta"], [1.0, 2.0])
    np.testing.assert_array_equal(extract_draws(_SamplesPosterior([4, 5]))["x"], [4, 5])

    with pytest.raises(TypeError, match="Cannot extract draws"):
        extract_draws(object())


def test_to_arviz_dataset_flat_empirical_and_filtering():
    post = _DrawsPosterior(
        {
            "alpha": np.array([1.0, 2.0, 3.0]),
            "beta": np.ones((1, 3, 2)),
        }
    )
    ds = to_arviz_dataset(post, var_names=["alpha"])
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"alpha"}
    assert ds["alpha"].dims == ("chain", "draw")
    assert ds["alpha"].shape == (1, 3)


def test_sensitivity_numeric_helpers_cover_edge_cases():
    summary = _posterior_summary({"x": np.array([[0.0, 1.0], [2.0, 3.0]])})
    assert summary["x"]["mean"] == pytest.approx(1.5)
    assert summary["x"]["hdi_3%"] == pytest.approx(np.percentile([0, 1, 2, 3], 3))

    compared = _compare_summaries(
        {"x": {"mean": 0.0, "std": 0.0}, "missing": {"mean": 1.0, "std": 1.0}},
        {"x": {"mean": 1.0, "std": 2.0}},
    )
    assert np.isnan(compared["x"]["std_ratio"])
    assert np.isnan(_kl_divergence_gaussian(0.0, 0.0, 0.0, 1.0))

    weighted = _reweighted_summary(
        np.array([[0.0, 10.0], [2.0, 20.0]]),
        ["a", "b"],
        np.array([0.0, 0.0]),
    )
    assert weighted["a"]["mean"] == pytest.approx(1.0)
    assert weighted["b"]["std"] == pytest.approx(5.0)

    arr = np.array([1.0, 2.0, 4.0])
    assert _finite_diff_gradient(arr, 0, 0.5) == pytest.approx(2.0)
    assert _finite_diff_gradient(arr, 1, 0.5) == pytest.approx(3.0)
    assert _finite_diff_gradient(arr, 2, 0.5) == pytest.approx(4.0)


def test_prior_sensitivity_compares_posteriors_and_warns():
    baseline = _DrawsPosterior({"theta": np.array([0.0, 0.1, -0.1, 0.0])})
    shifted = _DrawsPosterior({"theta": np.array([3.0, 3.1, 2.9, 3.0])})
    result = prior_sensitivity(
        {"base": baseline, "shifted": shifted},
        baseline="base",
        kl_threshold=0.01,
    )
    assert result["baseline"] == "base"
    assert "shifted vs base" in result["comparisons"]
    assert result["warnings"]

    with pytest.raises(ValueError, match="at least 2"):
        prior_sensitivity({"only": baseline})
    with pytest.raises(ValueError, match="Baseline"):
        prior_sensitivity({"base": baseline, "shifted": shifted}, baseline="nope")


def test_data_sensitivity_uses_pareto_k_thresholds_and_recommendations():
    fake_loo = MagicMock()
    fake_loo.pareto_k = xr.DataArray([0.2, 0.75, 1.2], dims=["obs"])

    with patch("arviz.loo", return_value=fake_loo) as mock_loo:
        result = data_sensitivity(
            object(),
            np.ones((5, 3)),
            var_name="obs",
            threshold_bad=0.7,
            threshold_very_bad=1.0,
        )

    assert mock_loo.call_args.kwargs["pointwise"] is True
    assert result["influential_indices"] == [1, 2]
    assert result["very_bad_indices"] == [2]
    assert result["n_observations"] == 3
    assert result["warnings"][0].startswith("1 observation")
    assert "heavier-tailed" in result["recommendations"][0]
    assert _build_loo_warnings(np.array([0.1, 0.8]))[0].startswith("1 observation")
    assert _build_loo_warnings(np.array([0.1, 0.2])) == []


def test_power_scale_sensitivity_handles_shapes_warnings_and_errors():
    post = _DrawsPosterior(
        {
            "theta": np.linspace(-1.0, 1.0, 6).reshape(2, 3),
            "sigma": np.linspace(1.0, 2.0, 6).reshape(2, 3),
        }
    )
    log_lik = np.column_stack([np.linspace(-2.0, 2.0, 6), np.zeros(6)])
    log_prior = np.linspace(2.0, -2.0, 6)

    result = power_scale_sensitivity(
        post,
        log_lik.reshape(2, 3, 2),
        log_prior.reshape(2, 3),
        lower=0.5,
        upper=1.5,
        n_steps=5,
        high_sensitivity_threshold=0.01,
    )

    assert result["alpha_grid"].shape == (5,)
    assert set(result["prior_sensitivity"]) == {"theta", "sigma"}
    assert result["warnings"]

    with pytest.raises(ValueError, match="log_likelihood must be 2D"):
        power_scale_sensitivity(post, np.ones(6), log_prior)
    with pytest.raises(ValueError, match="log_prior must be 1D"):
        power_scale_sensitivity(post, log_lik, np.ones((2, 3)))


def test_predictive_check_batched_loop_and_public_result_paths():
    dist = _SamplingDistribution()
    key = jax.random.PRNGKey(0)

    batched = _predictive_check_batched(
        dist, _KeyedLikelihood(), lambda y: jnp.mean(y), 4, 5, key,
    )
    assert batched.shape == (5,)

    with patch("jax.vmap", side_effect=Exception("not traceable")):
        fallback = _predictive_check_batched(
            dist, _KeyedLikelihood(), lambda y: float(np.asarray(y).mean()), 4, 5, key,
        )
    assert fallback.shape == (5,)

    looped = _predictive_check_loop(
        dist, _LoopLikelihood(), lambda y: np.mean(y), 3, 4, key,
    )
    assert looped.shape == (4,)

    assert _supports_key_arg(_KeyedLikelihood()) is True
    assert _supports_key_arg(_LoopLikelihood()) is False

    result = predictive_check(
        dist,
        _LoopLikelihood(),
        lambda y: float(np.mean(y)),
        observed_data=np.array([0.0, 0.0, 0.0]),
        n_replications=6,
        key=key,
    )
    assert result["test_fn_name"] == "<lambda>"
    assert "observed_statistic" in result
    assert 0.0 <= result["p_value"] <= 1.0
    assert dist.validation_results[-1]["p_value"] == pytest.approx(result["p_value"])
    assert dist.validation_results[-1]["test_fn_name"] == result["test_fn_name"]

    with pytest.raises(ValueError, match="n_samples is required"):
        predictive_check(dist, _LoopLikelihood(), np.mean, key=key)
