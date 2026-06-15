"""Tests for probpipe.diagnostics._view_base."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from probpipe.diagnostics._view_base import (
    DatasetView,
    DataTreeView,
    DiagnosticRunView,
    NotComputed,
    read_json_attr,
    read_indexed,
    read_scalar,
)


# ---------------------------------------------------------------------------
# NotComputed
# ---------------------------------------------------------------------------


class TestNotComputed:
    def test_repr(self):
        nc = NotComputed("not run yet")
        assert "not run yet" in repr(nc)

    def test_equality_same_reason(self):
        assert NotComputed("x") == NotComputed("x")

    def test_inequality_different_reason(self):
        assert NotComputed("x") != NotComputed("y")

    def test_not_equal_to_none(self):
        assert NotComputed("x") != None  # noqa: E711


# ---------------------------------------------------------------------------
# read_scalar
# ---------------------------------------------------------------------------


class TestReadScalar:
    def _da(self, value, **attrs):
        da = xr.DataArray(value)
        da.attrs.update(attrs)
        return da

    def test_plain_float(self):
        assert read_scalar(self._da(3.14)) == pytest.approx(3.14)

    def test_zero(self):
        assert read_scalar(self._da(0.0)) == 0.0

    def test_negative(self):
        assert read_scalar(self._da(-1.5)) == pytest.approx(-1.5)

    def test_none_returns_not_computed(self):
        result = read_scalar(None)
        assert isinstance(result, NotComputed)

    def test_nan_returns_not_computed(self):
        result = read_scalar(self._da(float("nan")))
        assert isinstance(result, NotComputed)

    def test_non_scalar_returns_not_computed(self):
        da = xr.DataArray([1.0, 2.0], dims=["x"])
        result = read_scalar(da)
        assert isinstance(result, NotComputed)

    def test_not_computed_attr_global(self):
        da = self._da(1.0, not_computed="some reason")
        result = read_scalar(da)
        assert isinstance(result, NotComputed)
        assert "some reason" in result.reason

    def test_not_computed_attr_labelled(self):
        da = self._da(1.0)
        da.attrs["not_computed_alpha"] = "chain too short"
        result = read_scalar(da, label="alpha")
        assert isinstance(result, NotComputed)

    def test_label_does_not_match_other_param(self):
        """Label-specific not_computed attr for 'beta' shouldn't affect 'alpha'."""
        da = self._da(2.0)
        da.attrs["not_computed_beta"] = "reason"
        result = read_scalar(da, label="alpha")
        assert result == pytest.approx(2.0)

    def test_non_numeric_returns_not_computed(self):
        result = read_scalar(xr.DataArray("not numeric"))
        assert isinstance(result, NotComputed)

    def test_not_computed_attr_without_reason_uses_unknown(self):
        da = self._da(1.0)
        da.attrs["not_computed_alpha"] = ""
        da.attrs["not_computed"] = "global reason"
        result = read_scalar(da, label="alpha")
        assert isinstance(result, NotComputed)
        assert result.reason == "global reason"


class TestReadJsonAttr:
    def test_valid_json(self):
        assert read_json_attr({"groups": '["a", "b"]'}, "groups") == ["a", "b"]

    def test_missing_uses_empty_list_default(self):
        assert read_json_attr({}, "missing") == []

    def test_invalid_json_uses_custom_default(self):
        assert read_json_attr({"groups": "not-json"}, "groups", default=["fallback"]) == [
            "fallback"
        ]


# ---------------------------------------------------------------------------
# read_indexed
# ---------------------------------------------------------------------------


class TestReadIndexed:
    def _ds(self, field: str, coords: list[str], values: list[float]) -> xr.Dataset:
        da = xr.DataArray(values, dims=["param"], coords={"param": coords})
        return xr.Dataset({field: da})

    def test_basic(self):
        ds = self._ds("rhat", ["alpha", "beta"], [1.0, 1.002])
        result = read_indexed(ds, "rhat", dim="param")
        assert result == {"alpha": pytest.approx(1.0), "beta": pytest.approx(1.002)}

    def test_missing_field_returns_empty(self):
        ds = self._ds("rhat", ["alpha"], [1.0])
        assert read_indexed(ds, "ess_bulk", dim="param") == {}

    def test_none_ds_returns_empty(self):
        assert read_indexed(None, "rhat", dim="param") == {}

    def test_wrong_dim_returns_empty(self):
        ds = self._ds("rhat", ["alpha"], [1.0])
        assert read_indexed(ds, "rhat", dim="chain") == {}

    def test_nan_value_becomes_not_computed(self):
        ds = self._ds("rhat", ["alpha", "beta"], [float("nan"), 1.001])
        result = read_indexed(ds, "rhat", dim="param")
        assert isinstance(result["alpha"], NotComputed)
        assert result["beta"] == pytest.approx(1.001)


# ---------------------------------------------------------------------------
# DataTreeView
# ---------------------------------------------------------------------------


class TestDataTreeView:
    def _make_tree(self, ds: xr.Dataset | None = None) -> xr.DataTree:
        if ds is None:
            ds = xr.Dataset()
        return xr.DataTree(dataset=ds)

    def test_exists_true(self):
        view = DataTreeView(self._make_tree())
        assert view.exists

    def test_exists_false(self):
        view = DataTreeView(None)
        assert not view.exists

    def test_attrs_empty_when_none(self):
        assert DataTreeView(None).attrs == {}

    def test_children_empty_when_none(self):
        assert DataTreeView(None).children == {}

    def test_attr_reads_value(self):
        tree = self._make_tree(xr.Dataset(attrs={"foo": "bar"}))
        view = DataTreeView(tree)
        assert view.attr("foo") == "bar"

    def test_attr_default(self):
        view = DataTreeView(self._make_tree())
        assert view.attr("missing", default=42) == 42

    def test_child_returns_none_when_absent(self):
        view = DataTreeView(self._make_tree())
        assert view.child("nonexistent") is None

    def test_has_child_false(self):
        view = DataTreeView(self._make_tree())
        assert not view.has_child("x")

    def test_child_returns_present_child(self):
        tree = xr.DataTree.from_dict({"child": xr.Dataset()})
        child = DataTreeView(tree).child("child")
        assert child is not None

    def test_child_returns_none_when_lookup_fails(self):
        class _TreeLike:
            children = {"child": object()}

            def __getitem__(self, key):
                raise RuntimeError("cannot read child")

        assert DataTreeView(_TreeLike()).child("child") is None

    def test_dataset_falls_back_to_ds_attribute(self):
        class _TreeLike:
            def __init__(self):
                self.ds = xr.Dataset({"x": xr.DataArray(1.0)})

            def to_dataset(self):
                raise RuntimeError("no direct dataset")

        ds = DataTreeView(_TreeLike()).dataset()
        assert "x" in ds

    def test_dataset_falls_back_to_dataset_attribute(self):
        class _TreeLike:
            def __init__(self):
                self.dataset = xr.Dataset({"x": xr.DataArray(2.0)})

            def to_dataset(self):
                raise RuntimeError("no direct dataset")

        ds = DataTreeView(_TreeLike()).dataset()
        assert ds["x"].item() == pytest.approx(2.0)

    def test_dataset_returns_none_when_all_accessors_fail(self):
        class _TreeLike:
            def to_dataset(self):
                raise RuntimeError("no direct dataset")

        assert DataTreeView(_TreeLike()).dataset() is None


# ---------------------------------------------------------------------------
# DatasetView
# ---------------------------------------------------------------------------


class TestDatasetView:
    def _view(self, **data_vars) -> DatasetView:
        arrays = {
            k: xr.DataArray(v) for k, v in data_vars.items()
        }
        tree = xr.DataTree(dataset=xr.Dataset(arrays))
        return DatasetView(tree)

    def test_scalar_reads_value(self):
        view = self._view(elpd_loo=xr.DataArray(-42.5))
        assert view.scalar("elpd_loo") == pytest.approx(-42.5)

    def test_scalar_missing_field(self):
        view = self._view()
        result = view.scalar("elpd_loo")
        assert isinstance(result, NotComputed)

    def test_scalar_none_tree(self):
        view = DatasetView(None)
        result = view.scalar("anything")
        assert isinstance(result, NotComputed)

    def test_indexed_reads_dict(self):
        da = xr.DataArray([1.01, 400.0], dims=["param"], coords={"param": ["mu", "sigma"]})
        tree = xr.DataTree(dataset=xr.Dataset({"rhat": da}))
        view = DatasetView(tree)
        result = view.indexed("rhat", dim="param")
        assert result["mu"] == pytest.approx(1.01)
        assert result["sigma"] == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# DiagnosticRunView
# ---------------------------------------------------------------------------


class TestDiagnosticRunView:
    def _view(self, name: str = "ppc", **scalars) -> DiagnosticRunView:
        arrays = {k: xr.DataArray(float(v)) for k, v in scalars.items()}
        tree = xr.DataTree(dataset=xr.Dataset(arrays))
        return DiagnosticRunView(name, tree)

    def test_name(self):
        view = self._view("loo")
        assert view.name == "loo"

    def test_result_scalar(self):
        view = self._view("ppc", p_value=0.43)
        assert view.result["p_value"] == pytest.approx(0.43)

    def test_result_one_dimensional_values(self):
        da = xr.DataArray([0.2, 0.8], dims=["test_fn"], coords={"test_fn": ["a", "b"]})
        tree = xr.DataTree(dataset=xr.Dataset({"p_value": da}))
        result = DiagnosticRunView("ppc", tree).result
        assert result["p_value"]["b"] == pytest.approx(0.8)

    def test_result_multidimensional_values_are_not_computed(self):
        da = xr.DataArray(np.ones((2, 2)), dims=["row", "col"])
        tree = xr.DataTree(dataset=xr.Dataset({"matrix": da}))
        result = DiagnosticRunView("run", tree).result
        assert isinstance(result["matrix"], NotComputed)

    def test_result_empty_when_no_tree(self):
        view = DiagnosticRunView("ppc", None)
        assert view.result == {}

    def test_plot_ready_default_false(self):
        view = self._view("ppc")
        assert view.plot_ready is False

    def test_timestamp_default_empty(self):
        view = self._view("ppc")
        assert view.timestamp == ""

    def test_plot_metadata_from_attrs(self):
        tree = xr.DataTree(
            dataset=xr.Dataset(
                attrs={
                    "plot_fn": "az.plot_ppc",
                    "plot_ready": True,
                    "plot_groups": '["posterior_predictive"]',
                    "timestamp": "2026-01-01T00:00:00",
                }
            )
        )
        view = DiagnosticRunView("ppc", tree)
        assert view.plot_fn == "az.plot_ppc"
        assert view.plot_ready is True
        assert view.plot_groups == ["posterior_predictive"]
        assert view.timestamp.startswith("2026")

    def test_repr_contains_name(self):
        view = self._view("loo")
        assert "loo" in repr(view)
