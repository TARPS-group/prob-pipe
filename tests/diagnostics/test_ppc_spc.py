"""Tests for probpipe.diagnostics._ppc_spc."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._ppc_spc import (
    _dataset_from_payload,
    _observed_data_to_dataset,
    _ppc_op,
    _replicated_data_to_dataset,
    _replicated_statistics_summary,
    _write_ppc_payload,
    add_ppc,
)
from probpipe.diagnostics._views import DiagnosticsView, PPCView

# conftest.py provides: posterior, posterior_3params


# ---------------------------------------------------------------------------
# Fake GenerativeLikelihood (pure NumPy — no JAX required)
# ---------------------------------------------------------------------------


class _NumpyLikelihood:
    """Generates i.i.d. Normal(0, 1) replicated data.

    Does NOT accept a ``key`` keyword so _predictive_check_loop is used.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def generate_data(self, params, n_samples: int) -> np.ndarray:
        return self._rng.standard_normal(n_samples)


class _KeyedLikelihood:
    def generate_data(self, params, n_samples: int, *, key=None):
        import jax.numpy as jnp

        params_arr = jnp.asarray(params["alpha"])
        return jnp.broadcast_to(params_arr[:, None], (params_arr.shape[0], n_samples))


class _NormalLocationPosterior:
    _annotations = None

    def _sample(self, key, shape):
        import jax

        return jax.random.normal(key, shape)


class _LocationLikelihood:
    def generate_data(self, params, n_samples: int) -> np.ndarray:
        return np.full(n_samples, float(params))


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


def _mean(y: np.ndarray) -> float:
    return float(np.mean(y))


def _std(y: np.ndarray) -> float:
    return float(np.std(y))


# ---------------------------------------------------------------------------
# _observed_data_to_dataset
# ---------------------------------------------------------------------------


class TestObservedDataToDataset:
    def test_scalar_input(self):
        ds = _observed_data_to_dataset(3.14)
        assert "y" in ds.data_vars
        assert ds["y"].shape == ()

    def test_1d_input(self):
        ds = _observed_data_to_dataset(np.array([1.0, 2.0, 3.0]))
        assert ds["y"].dims == ("obs",)
        assert ds["y"].shape == (3,)

    def test_2d_input(self):
        ds = _observed_data_to_dataset(np.ones((4, 5)))
        assert ds["y"].ndim == 2

    def test_custom_var_name(self):
        ds = _observed_data_to_dataset(np.array([1.0]), var_name="x")
        assert "x" in ds.data_vars

    def test_returns_dataset(self):
        assert isinstance(_observed_data_to_dataset(np.array([1.0])), xr.Dataset)


# ---------------------------------------------------------------------------
# _replicated_data_to_dataset
# ---------------------------------------------------------------------------


class TestReplicatedDataToDataset:
    def test_scalar_gets_chain_draw_dims(self):
        ds = _replicated_data_to_dataset(np.float64(1.0))
        assert "chain" in ds["y"].dims
        assert "draw" in ds["y"].dims

    def test_1d_adds_chain_dim(self):
        ds = _replicated_data_to_dataset(np.ones(10))
        assert ds["y"].shape == (1, 10)

    def test_2d_adds_chain_dim(self):
        ds = _replicated_data_to_dataset(np.ones((5, 3)))
        assert ds["y"].shape == (1, 5, 3)

    def test_3d_kept_as_is(self):
        ds = _replicated_data_to_dataset(np.ones((2, 5, 3)))
        assert ds["y"].shape == (2, 5, 3)
        assert ds["y"].dims == ("chain", "draw", "obs")

    def test_higher_dimensional_obs_dims_are_named(self):
        ds = _replicated_data_to_dataset(np.ones((2, 5, 3, 4)))
        assert ds["y"].dims == ("chain", "draw", "obs_dim_0", "obs_dim_1")


# ---------------------------------------------------------------------------
# _replicated_statistics_summary
# ---------------------------------------------------------------------------


class TestReplicatedStatisticsSummary:
    def test_basic_summary(self):
        stats = {"mean_fn": np.array([0.1, 0.2, 0.3])}
        result = _replicated_statistics_summary(stats)
        assert result is not None
        assert "replicated_stat_mean" in result
        assert "replicated_stat_sd" in result

    def test_empty_dict_returns_none(self):
        assert _replicated_statistics_summary({}) is None

    def test_none_values_produce_nan(self):
        result = _replicated_statistics_summary({"fn": None})
        assert result is None  # all-NaN → no available stats

    def test_empty_array_produces_nan(self):
        result = _replicated_statistics_summary({"fn": np.array([])})
        assert result is None

    def test_multiple_fns(self):
        stats = {
            "mean_fn": np.ones(100),
            "std_fn": np.ones(100) * 2,
        }
        result = _replicated_statistics_summary(stats)
        assert result is not None
        assert len(result["replicated_stat_mean"]) == 2

    def test_quantiles_present(self):
        stats = {"fn": np.linspace(0, 1, 100)}
        result = _replicated_statistics_summary(stats)
        assert "replicated_stat_q05" in result
        assert "replicated_stat_q95" in result


# ---------------------------------------------------------------------------
# add_ppc
# ---------------------------------------------------------------------------


class TestAddPpc:
    def test_writes_ppc_group(self, posterior):
        observed = np.random.default_rng(0).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        assert posterior._annotations is not None
        ppc_ds = posterior._annotations["diagnostics"]["runs"]["ppc"].to_dataset()
        assert "p_value" in ppc_ds.data_vars
        assert ppc_ds.attrs["plot_ready"] is False
        assert ppc_ds.attrs["plot_fn"] == ""
        assert ppc_ds.attrs["plot_groups"] == "[]"

    def test_p_value_in_range(self, posterior):
        observed = np.random.default_rng(1).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=50,
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        p = view.p_values.get("_mean")
        if isinstance(p, float):
            assert 0.0 <= p <= 1.0

    def test_p_value_matches_known_centered_case(self):
        import jax

        posterior = _NormalLocationPosterior()
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=np.zeros(5),
            generative_likelihood=_LocationLikelihood(),
            n_replications=1000,
            key=jax.random.PRNGKey(0),
        )

        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert view.p_values["_mean"] == pytest.approx(0.5, abs=0.06)

    def test_p_value_responds_to_shifted_observed_data(self):
        import jax

        posterior = _NormalLocationPosterior()
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=np.full(5, 3.0),
            generative_likelihood=_LocationLikelihood(),
            n_replications=1000,
            key=jax.random.PRNGKey(0),
        )

        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert view.p_values["_mean"] < 0.01

    def test_multiple_test_fns(self, posterior):
        observed = np.random.default_rng(2).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=[_mean, _std],
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert set(view.p_values.keys()) == {"_mean", "_std"}

    def test_observed_stored(self, posterior):
        observed = np.random.default_rng(3).standard_normal(30)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        obs = view.observed.get("_mean")
        if isinstance(obs, float):
            assert np.isfinite(obs)

    def test_returns_none(self, posterior):
        observed = np.ones(10)
        result = add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=5,
        )
        assert result is None

    def test_prior_predictive_without_observed_data(self, posterior):
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=None,
            num_observations=7,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=5,
        )

        ppc_ds = posterior._annotations["diagnostics"]["runs"]["ppc"].to_dataset()
        assert ppc_ds.attrs["has_observed_data"] is False
        assert bool(ppc_ds.attrs["wrote_arviz_observed_data"]) is False
        assert "replicated_stat_mean" in ppc_ds.data_vars
        assert np.isnan(float(ppc_ds["p_value"].sel(test_fn="_mean")))
        assert np.isnan(float(ppc_ds["observed"].sel(test_fn="_mean")))

    def test_num_observations_required_without_observed_data(self, posterior):
        with pytest.raises(ValueError, match="num_observations is required"):
            add_ppc(
                posterior,
                test_fns=_mean,
                observed_data=None,
                generative_likelihood=_NumpyLikelihood(),
                n_replications=5,
            )

    def test_num_observations_must_be_positive(self, posterior):
        with pytest.raises(ValueError, match="positive integer"):
            add_ppc(
                posterior,
                test_fns=_mean,
                observed_data=None,
                num_observations=0,
                generative_likelihood=_NumpyLikelihood(),
                n_replications=5,
            )

    def test_diagnostics_view_integration(self, posterior):
        observed = np.random.default_rng(4).standard_normal(40)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            generative_likelihood=_NumpyLikelihood(),
            n_replications=20,
        )
        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.ppc.exists

    def test_keyed_likelihood_uses_batched_path(self, posterior):
        payload = _ppc_op(
            posterior,
            _mean,
            observed_data=np.ones(5),
            generative_likelihood=_KeyedLikelihood(),
            n_replications=4,
        )

        ds = _dataset_from_payload(payload)
        assert "p_value" in ds

    def test_dataset_from_payload_rejects_non_dataset(self):

        with pytest.raises(TypeError, match=r"xarray\.Dataset"):
            _dataset_from_payload({"dataset": "not-a-dataset"})

    def test_write_ppc_payload_stores_optional_predictive_group(self, posterior):
        run_ds = xr.Dataset({"p_value": xr.DataArray([0.5], dims=["test_fn"])})
        pred_ds = xr.Dataset({"y": xr.DataArray(np.ones((1, 2)), dims=["chain", "draw"])})
        payload = {
            "posterior_predictive_dataset": pred_ds,
            "observed_data_dataset": None,
            "dataset": run_ds,
        }

        _write_ppc_payload(posterior, payload)

        assert "posterior_predictive" in posterior._annotations["arviz"].children
