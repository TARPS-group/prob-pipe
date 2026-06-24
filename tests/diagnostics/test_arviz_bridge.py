"""Focused coverage for diagnostics ArviZ bridge helpers."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys

import numpy as np
import pytest
import xarray as xr

import probpipe.diagnostics._arviz_bridge as arviz_bridge
from probpipe.diagnostics._arviz_bridge import (
    check_arviz_installed,
    extract_draws,
    to_arviz_dataset,
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


def test_extract_draws_supports_sample_records():
    post = _SamplesPosterior(_Record({"alpha": [1.0, 2.0], "beta": [3.0, 4.0]}))

    draws = extract_draws(post)

    assert set(draws) == {"alpha", "beta"}
    np.testing.assert_array_equal(draws["beta"], [3.0, 4.0])


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


def test_to_arviz_dataset_prepends_chain_for_matrix_valued_params():
    post = _DrawsPosterior({"omega": np.arange(24.0).reshape(4, 2, 3)})

    ds = to_arviz_dataset(post)

    assert ds["omega"].dims == ("chain", "draw", "dim_0", "dim_1")
    assert ds["omega"].shape == (1, 4, 2, 3)
    np.testing.assert_array_equal(ds["omega"].values[0], post.draws()["omega"])


def test_to_arviz_dataset_delegates_for_approximate_distribution(monkeypatch):
    class _ApproxPosterior:
        def __init__(self):
            self.chains = [object()]
            self.fields = ["alpha", "beta"]

    source = xr.Dataset(
        {
            "alpha": xr.DataArray(np.ones((1, 2)), dims=["chain", "draw"]),
            "beta": xr.DataArray(np.zeros((1, 2)), dims=["chain", "draw"]),
        }
    )

    def _fake_builder(posterior):
        assert isinstance(posterior, _ApproxPosterior)
        return source

    monkeypatch.setattr(
        "probpipe.diagnostics._datatree_store.to_named_posterior_dataset",
        _fake_builder,
    )

    ds = to_arviz_dataset(_ApproxPosterior(), var_names=["beta"])

    assert list(ds.data_vars) == ["beta"]


def test_to_arviz_dataset_requires_xarray(monkeypatch):
    monkeypatch.setattr(arviz_bridge, "xr", None)

    with pytest.raises(ImportError, match="xarray is required"):
        to_arviz_dataset(_DrawsPosterior({"x": [1.0]}))


def test_check_arviz_installed_reports_missing_dependencies(monkeypatch):
    real_import = builtins.__import__

    def _missing_arviz(name, *args, **kwargs):
        if name == "arviz":
            raise ImportError("missing arviz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_arviz)
    with pytest.raises(ImportError, match="ArviZ is required"):
        check_arviz_installed()

    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setattr(arviz_bridge, "xr", None)
    with pytest.raises(ImportError, match="xarray is required"):
        check_arviz_installed()


def test_arviz_bridge_import_sets_xarray_none_when_missing(monkeypatch):
    real_import = builtins.__import__

    def _missing_xarray(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("missing xarray")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_xarray)
    spec = importlib.util.spec_from_file_location(
        "_probpipe_arviz_bridge_missing_xarray",
        arviz_bridge.__file__,
    )
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.xr is None


def test_diagnostics_init_optional_import_paths(monkeypatch):
    import probpipe.diagnostics as diagnostics

    reloaded = importlib.reload(diagnostics)
    assert not hasattr(reloaded, "DiagnosticsModule")
    assert "DiagnosticsModule" not in reloaded.__all__

    monkeypatch.setitem(sys.modules, "probpipe.diagnostics.views", None)
    reloaded = importlib.reload(diagnostics)
    assert "DiagnosticsView" in reloaded.__all__

    monkeypatch.delitem(sys.modules, "probpipe.diagnostics.views", raising=False)
    importlib.import_module("probpipe.diagnostics.views")
    importlib.reload(diagnostics)
