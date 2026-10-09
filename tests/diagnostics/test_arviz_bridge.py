"""Focused coverage for diagnostics ArviZ bridge helpers."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import probpipe.diagnostics._arviz_bridge as arviz_bridge
from probpipe import EmpiricalDistribution, NumericRecordBatch, RecordSpec
from probpipe.diagnostics._arviz_bridge import (
    check_arviz_installed,
    extract_draws,
    to_arviz_dataset,
)
from tests._posterior import posterior_of


def test_extract_draws_reads_an_empirical_law_and_refuses_others():
    empirical = EmpiricalDistribution("x", jnp.array([4.0, 5.0]))
    np.testing.assert_array_equal(extract_draws(empirical)["x"], [4.0, 5.0])

    with pytest.raises(TypeError, match="cannot extract draws"):
        extract_draws(object())


def test_extract_draws_supports_record_atoms():
    atoms = NumericRecordBatch(
        "rows",
        {"alpha": jnp.array([1.0, 2.0]), "beta": jnp.array([3.0, 4.0])},
        "row",
        element_spec=RecordSpec(alpha=(), beta=()),
    )
    post = EmpiricalDistribution("post", atoms)

    draws = extract_draws(post)

    assert set(draws) == {"alpha", "beta"}
    np.testing.assert_array_equal(draws["beta"], [3.0, 4.0])


def test_to_arviz_dataset_flat_empirical_and_filtering():
    atoms = NumericRecordBatch(
        "rows",
        {"alpha": jnp.array([1.0, 2.0, 3.0]), "beta": jnp.ones((3, 2))},
        "row",
        element_spec=RecordSpec(alpha=(), beta=(2,)),
    )
    post = EmpiricalDistribution("post", atoms)
    ds = to_arviz_dataset(post, var_names=["alpha"])
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"alpha"}
    assert ds["alpha"].dims == ("chain", "draw")
    assert ds["alpha"].shape == (1, 3)


def _flat_posterior_of(**columns) -> EmpiricalDistribution:
    atoms = NumericRecordBatch(
        "rows",
        {name: jnp.asarray(column) for name, column in columns.items()},
        "row",
        element_spec=RecordSpec(**{name: () for name in columns}),
    )
    return EmpiricalDistribution("post", atoms)


def test_to_arviz_dataset_reads_a_single_variable_name_as_one():
    post = _flat_posterior_of(mu=[1.0, 2.0], mu_x=[3.0, 4.0])

    single = to_arviz_dataset(post, var_names="mu_x")

    assert isinstance(single, xr.Dataset)
    assert list(single.data_vars) == ["mu_x"]
    xr.testing.assert_identical(single, to_arviz_dataset(post, var_names=("mu_x",)))


def test_to_arviz_dataset_keeps_the_order_of_var_names():
    post = _flat_posterior_of(alpha=[1.0, 2.0], beta=[3.0, 4.0])

    assert list(to_arviz_dataset(post, var_names=["beta", "alpha"]).data_vars) == ["beta", "alpha"]


def test_to_arviz_dataset_refuses_an_unknown_variable():
    post = _flat_posterior_of(alpha=[1.0, 2.0])

    with pytest.raises(
        ValueError, match=r"unknown variable 'gamma'; available variables: \['alpha'\]"
    ):
        to_arviz_dataset(post, var_names=["alpha", "gamma"])


@pytest.mark.parametrize(
    ("var_names", "match"),
    [
        (3, "to_arviz_dataset var_names must be a str or a sequence of str, got int 3"),
        (b"alpha", "got bytes"),
        ({"alpha": 1}, "got dict"),
        ({"alpha"}, "got set"),
        (("alpha", 3), "to_arviz_dataset var_names entry must be a str, got int 3"),
    ],
)
def test_to_arviz_dataset_refuses_var_names_that_are_not_names(var_names, match):
    with pytest.raises(TypeError, match=match):
        to_arviz_dataset(_flat_posterior_of(alpha=[1.0, 2.0]), var_names=var_names)


def test_to_arviz_dataset_prepends_chain_for_matrix_valued_params():
    omega = np.arange(24.0).reshape(4, 2, 3)
    post = EmpiricalDistribution("omega", jnp.asarray(omega))

    ds = to_arviz_dataset(post)

    assert ds["omega"].dims == ("chain", "draw", "dim_0", "dim_1")
    assert ds["omega"].shape == (1, 4, 2, 3)
    np.testing.assert_array_equal(ds["omega"].values[0], omega)


def test_to_arviz_dataset_delegates_for_an_inference_result(monkeypatch):
    result = posterior_of(
        [np.stack([np.ones(2), np.zeros(2)], axis=-1)], event_spec=RecordSpec(alpha=(), beta=())
    )

    source = xr.Dataset(
        {
            "alpha": xr.DataArray(np.ones((1, 2)), dims=["chain", "draw"]),
            "beta": xr.DataArray(np.zeros((1, 2)), dims=["chain", "draw"]),
        }
    )

    def _fake_builder(posterior):
        assert posterior is result
        return source

    monkeypatch.setattr(
        "probpipe.diagnostics._datatree_store.to_named_posterior_dataset",
        _fake_builder,
    )

    ds = to_arviz_dataset(result, var_names=["beta"])

    assert list(ds.data_vars) == ["beta"]
    single = to_arviz_dataset(result, var_names="beta")
    assert isinstance(single, xr.Dataset)
    xr.testing.assert_identical(single, ds)
    with pytest.raises(ValueError, match=r"unknown variable 'gamma'; available variables"):
        to_arviz_dataset(result, var_names="gamma")


def test_to_arviz_dataset_requires_xarray(monkeypatch):
    monkeypatch.setattr(arviz_bridge, "xr", None)

    with pytest.raises(ImportError, match="xarray is required"):
        to_arviz_dataset(EmpiricalDistribution("x", jnp.array([1.0])))


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
