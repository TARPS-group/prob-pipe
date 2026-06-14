"""Tests for probpipe.diagnostics._datatree_store."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._datatree_store import (
    _add_group,
    _get_or_create_mcmc_ds,
    _mcmc_has_field,
    _write_mcmc_field,
    to_named_posterior_dataset,
)
from probpipe.diagnostics._view_base import NotComputed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalPosterior:
    """Minimal posterior stub — only _auxiliary matters for storage tests."""

    def __init__(self):
        self._auxiliary = None


# ---------------------------------------------------------------------------
# _add_group
# ---------------------------------------------------------------------------


class TestAddGroup:
    def test_creates_auxiliary_from_none(self):
        post = _MinimalPosterior()
        ds = xr.Dataset({"x": xr.DataArray(1.0)})
        _add_group(post, "diagnostics/mcmc", ds)
        assert post._auxiliary is not None

    def test_group_accessible_on_tree(self):
        post = _MinimalPosterior()
        ds = xr.Dataset({"rhat": xr.DataArray(1.001)})
        _add_group(post, "diagnostics/mcmc", ds)
        retrieved = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        assert "rhat" in retrieved.data_vars

    def test_second_group_preserves_first(self):
        post = _MinimalPosterior()
        ds1 = xr.Dataset({"rhat": xr.DataArray(1.0)})
        ds2 = xr.Dataset({"ess_bulk": xr.DataArray(500.0)})
        _add_group(post, "diagnostics/mcmc", ds1)
        _add_group(post, "diagnostics/runs/ppc", ds2)
        mcmc_ds = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        ppc_ds = post._auxiliary["diagnostics"]["runs"]["ppc"].to_dataset()
        assert "rhat" in mcmc_ds.data_vars
        assert "ess_bulk" in ppc_ds.data_vars

    def test_replace_existing_group(self):
        post = _MinimalPosterior()
        ds1 = xr.Dataset({"rhat": xr.DataArray(1.05)})
        ds2 = xr.Dataset({"rhat": xr.DataArray(1.001)})
        _add_group(post, "diagnostics/mcmc", ds1)
        _add_group(post, "diagnostics/mcmc", ds2)
        retrieved = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        val = float(retrieved["rhat"].values)
        assert val == pytest.approx(1.001)


# ---------------------------------------------------------------------------
# _get_or_create_mcmc_ds
# ---------------------------------------------------------------------------


class TestGetOrCreateMcmcDs:
    def test_returns_empty_when_no_auxiliary(self):
        post = _MinimalPosterior()
        ds = _get_or_create_mcmc_ds(post)
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0

    def test_returns_existing_ds(self):
        post = _MinimalPosterior()
        ds = xr.Dataset({"rhat": xr.DataArray(1.0)})
        _add_group(post, "diagnostics/mcmc", ds)
        result = _get_or_create_mcmc_ds(post)
        assert "rhat" in result.data_vars


# ---------------------------------------------------------------------------
# _write_mcmc_field
# ---------------------------------------------------------------------------


class TestWriteMcmcField:
    def test_writes_field(self):
        post = _MinimalPosterior()
        _write_mcmc_field(post, "rhat", {"alpha": 1.001, "beta": 1.0})
        ds = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        assert "rhat" in ds.data_vars
        da = ds["rhat"]
        coords = list(da.coords["param"].values)
        assert "alpha" in coords and "beta" in coords

    def test_not_computed_writes_nan(self):
        post = _MinimalPosterior()
        _write_mcmc_field(post, "rhat", {"alpha": NotComputed("single chain")})
        ds = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        da = ds["rhat"]
        val = float(da.sel(param="alpha").values)
        assert np.isnan(val)

    def test_attrs_stored(self):
        post = _MinimalPosterior()
        _write_mcmc_field(
            post, "rhat", {"alpha": 1.0},
            attrs={"rhat_method": "rank"},
        )
        ds = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        assert ds.attrs.get("rhat_method") == "rank"

    def test_second_write_accumulates_fields(self):
        post = _MinimalPosterior()
        _write_mcmc_field(post, "rhat", {"alpha": 1.001})
        _write_mcmc_field(post, "ess_bulk", {"alpha": 450.0})
        ds = post._auxiliary["diagnostics"]["mcmc"].to_dataset()
        assert "rhat" in ds.data_vars
        assert "ess_bulk" in ds.data_vars


# ---------------------------------------------------------------------------
# _mcmc_has_field
# ---------------------------------------------------------------------------


class TestMcmcHasField:
    def test_false_when_no_auxiliary(self):
        post = _MinimalPosterior()
        assert _mcmc_has_field(post, "rhat") is False

    def test_false_when_field_absent(self):
        post = _MinimalPosterior()
        _write_mcmc_field(post, "rhat", {"alpha": 1.0})
        assert _mcmc_has_field(post, "ess_bulk") is False

    def test_true_after_write(self):
        post = _MinimalPosterior()
        _write_mcmc_field(post, "rhat", {"alpha": 1.0})
        assert _mcmc_has_field(post, "rhat") is True


# ---------------------------------------------------------------------------
# to_named_posterior_dataset
# ---------------------------------------------------------------------------


class TestToNamedPosteriorDataset:
    def _posterior(self, params, n_chains=2, n_draws=100):
        rng = np.random.default_rng(42)

        class _FakeRecord(dict):
            @property
            def fields(self):
                return list(self.keys())

        class _Post:
            fields = params
            num_chains = n_chains

            def draws(self, *, chain):
                return _FakeRecord(
                    {p: rng.standard_normal(n_draws) for p in params}
                )

        return _Post()

    def test_output_is_dataset(self):
        post = self._posterior(["mu", "sigma"])
        ds = to_named_posterior_dataset(post)
        assert isinstance(ds, xr.Dataset)

    def test_all_params_present(self):
        post = self._posterior(["mu", "sigma", "nu"])
        ds = to_named_posterior_dataset(post)
        assert set(ds.data_vars) == {"mu", "sigma", "nu"}

    def test_shape_chain_draw(self):
        post = self._posterior(["mu"], n_chains=3, n_draws=150)
        ds = to_named_posterior_dataset(post)
        assert ds["mu"].dims == ("chain", "draw")
        assert ds["mu"].shape == (3, 150)

    def test_single_param(self):
        post = self._posterior(["alpha"], n_chains=2, n_draws=50)
        ds = to_named_posterior_dataset(post)
        assert "alpha" in ds.data_vars
        assert ds["alpha"].shape == (2, 50)
