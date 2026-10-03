"""Tests for probpipe.diagnostics._ppc_spc."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from scipy import stats

from probpipe import Normal, conditional_distribution
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
# The kernel of the observations
# ---------------------------------------------------------------------------


def _kernel(posterior, n: int = 50):
    """``y ~ Normal(alpha, 1)``, iid over *n* observations, given the posterior's slots."""
    return conditional_distribution(
        "y_given_alpha",
        lambda alpha, beta: Normal("y", alpha * jnp.ones(n), 1.0),
        given_spec=posterior.event_spec.components,
    )


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
            kernel=_kernel(posterior),
            n_replications=20,
            key=jax.random.key(0),
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
            kernel=_kernel(posterior),
            n_replications=50,
            key=jax.random.key(0),
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert 0.0 <= view.p_values["_mean"] <= 1.0

    @pytest.mark.parametrize("location", [0.0, 0.5, 3.0])
    def test_p_value_matches_the_closed_form(self, posterior, location):
        """At each atom of ``alpha``, the mean of five replicated observations is
        ``Normal(alpha, sqrt(1/5))``, so the p-value is the average of its tail
        probabilities over the posterior's equally weighted atoms.
        """
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=np.full(5, location),
            kernel=_kernel(posterior, n=5),
            n_replications=2000,
            key=jax.random.key(0),
        )

        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        atoms = np.ravel(np.asarray(posterior.atoms["alpha"].values))
        exact = float(np.mean(stats.norm.sf(location, atoms, np.sqrt(1.0 / 5.0))))
        # Four Monte Carlo standard errors of 2000 replications, plus float32 rounding.
        tolerance = 4.0 * np.sqrt(exact * (1.0 - exact) / 2000) + 2e-3
        assert view.p_values["_mean"] == pytest.approx(exact, abs=tolerance)

    def test_multiple_test_fns(self, posterior):
        observed = np.random.default_rng(2).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=[_mean, _std],
            observed_data=observed,
            kernel=_kernel(posterior),
            n_replications=20,
            key=jax.random.key(0),
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert set(view.p_values.keys()) == {"_mean", "_std"}

    def test_observed_stored(self, posterior):
        observed = np.random.default_rng(3).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            kernel=_kernel(posterior),
            n_replications=20,
            key=jax.random.key(0),
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert view.observed["_mean"] == pytest.approx(float(np.mean(observed)), rel=1e-5)

    def test_observed_data_may_map_the_kernel_components(self, posterior):
        observed = np.random.default_rng(3).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data={"y": observed},
            kernel=_kernel(posterior),
            n_replications=20,
            key=jax.random.key(0),
        )
        view = PPCView(posterior._annotations["diagnostics"]["runs"]["ppc"])
        assert view.observed["_mean"] == pytest.approx(float(np.mean(observed)), rel=1e-5)
        arviz_observed = posterior._annotations["arviz"]["observed_data"].to_dataset()
        assert arviz_observed["y"].shape == (50,)

    def test_returns_none(self, posterior):
        observed = np.ones(50)
        result = add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            kernel=_kernel(posterior),
            n_replications=5,
            key=jax.random.key(0),
        )
        assert result is None

    def test_prior_predictive_without_observed_data(self, posterior):
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=None,
            kernel=_kernel(posterior, n=7),
            n_replications=5,
            key=jax.random.key(0),
        )

        ppc_ds = posterior._annotations["diagnostics"]["runs"]["ppc"].to_dataset()
        assert ppc_ds.attrs["has_observed_data"] is False
        assert bool(ppc_ds.attrs["wrote_arviz_observed_data"]) is False
        assert "replicated_stat_mean" in ppc_ds.data_vars
        assert np.isnan(float(ppc_ds["p_value"].sel(test_fn="_mean")))
        assert np.isnan(float(ppc_ds["observed"].sel(test_fn="_mean")))

    def test_a_posterior_that_misses_a_given_slot_raises_naming_it(self, posterior):
        kernel = conditional_distribution(
            "y_given_alpha_gamma",
            lambda alpha, gamma: Normal("y", alpha * jnp.ones(5), 1.0),
            given_spec={
                "alpha": posterior.event_spec.components["alpha"],
                "gamma": posterior.event_spec.components["alpha"],
            },
        )
        with pytest.raises(ValueError, match=r"does not produce the given slots \['gamma'\]"):
            add_ppc(posterior, _mean, np.zeros(5), kernel=kernel, key=jax.random.key(0))

    def test_n_replications_must_be_positive(self, posterior):
        with pytest.raises(ValueError, match="positive integer"):
            add_ppc(
                posterior,
                test_fns=_mean,
                observed_data=None,
                kernel=_kernel(posterior),
                n_replications=0,
            )

    def test_diagnostics_view_integration(self, posterior):
        observed = np.random.default_rng(4).standard_normal(50)
        add_ppc(
            posterior,
            test_fns=_mean,
            observed_data=observed,
            kernel=_kernel(posterior),
            n_replications=20,
            key=jax.random.key(0),
        )
        view = DiagnosticsView(posterior._annotations["diagnostics"])
        assert view.ppc.exists

    def test_ppc_op_returns_the_payload(self, posterior):
        payload = _ppc_op(
            posterior,
            _mean,
            observed_data=np.ones(50),
            kernel=_kernel(posterior),
            n_replications=4,
            key=jax.random.key(0),
        )

        ds = _dataset_from_payload(payload)
        assert "p_value" in ds
        assert "diagnostics" not in posterior.annotations.children

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
