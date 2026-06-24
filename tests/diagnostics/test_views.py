"""Tests for probpipe.diagnostics._views."""

from __future__ import annotations

import json

import pytest
import xarray as xr

from probpipe.diagnostics._view_base import NotComputed
from probpipe.diagnostics._views import (
    DiagnosticsView,
    LOOView,
    MCMCView,
    PPCView,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mcmc_tree(
    *,
    rhat: dict[str, float] | None = None,
    ess_bulk: dict[str, float] | None = None,
    ess_tail: dict[str, float] | None = None,
    mcse_mean: dict[str, float] | None = None,
    n_divergences=None,
    rhat_warnings: list[str] | None = None,
) -> xr.DataTree:
    """Build a minimal /diagnostics/mcmc DataTree node."""
    data_vars: dict[str, xr.DataArray] = {}

    def _indexed(d):
        params = list(d.keys())
        return xr.DataArray(list(d.values()), dims=["param"], coords={"param": params})

    if rhat is not None:
        data_vars["rhat"] = _indexed(rhat)
    if ess_bulk is not None:
        data_vars["ess_bulk"] = _indexed(ess_bulk)
    if ess_tail is not None:
        data_vars["ess_tail"] = _indexed(ess_tail)
    if mcse_mean is not None:
        data_vars["mcse_mean"] = _indexed(mcse_mean)

    attrs: dict = {}
    if n_divergences is not None:
        attrs["n_divergences"] = json.dumps(n_divergences)
    if rhat_warnings is not None:
        attrs["rhat_warnings"] = json.dumps(rhat_warnings)

    ds = xr.Dataset(data_vars, attrs=attrs)
    return xr.DataTree(dataset=ds)


def _ppc_tree(
    fn_names: list[str],
    p_values: list[float],
    observed: list[float],
    *,
    plot_ready: bool = False,
    timestamp: str = "",
) -> xr.DataTree:
    da_p = xr.DataArray(p_values, dims=["test_fn"], coords={"test_fn": fn_names})
    da_o = xr.DataArray(observed, dims=["test_fn"], coords={"test_fn": fn_names})
    ds = xr.Dataset(
        {"p_value": da_p, "observed": da_o},
        attrs={
            "plot_ready": plot_ready,
            "timestamp": timestamp,
        },
    )
    return xr.DataTree(dataset=ds)


def _loo_tree(**scalars) -> xr.DataTree:
    data_vars = {k: xr.DataArray(float(v)) for k, v in scalars.items()}
    return xr.DataTree(dataset=xr.Dataset(data_vars))


def _diagnostics_tree(
    mcmc_ds: xr.Dataset | None = None,
    ppc_ds: xr.Dataset | None = None,
    loo_ds: xr.Dataset | None = None,
) -> xr.DataTree:
    """Build a /diagnostics DataTree with optional child nodes."""
    d: dict[str, xr.Dataset] = {}
    if mcmc_ds is not None:
        d["mcmc"] = mcmc_ds
    if ppc_ds is not None:
        d["runs/ppc"] = ppc_ds
    if loo_ds is not None:
        d["runs/loo"] = loo_ds
    return xr.DataTree.from_dict(d)


# ---------------------------------------------------------------------------
# MCMCView
# ---------------------------------------------------------------------------


class TestMCMCView:
    def test_rhat_reads_values(self):
        view = MCMCView(_mcmc_tree(rhat={"alpha": 1.001, "beta": 1.003}))
        assert view.rhat == {"alpha": pytest.approx(1.001), "beta": pytest.approx(1.003)}

    def test_ess_bulk_reads_values(self):
        view = MCMCView(_mcmc_tree(ess_bulk={"alpha": 450.0}))
        assert view.ess_bulk["alpha"] == pytest.approx(450.0)

    def test_n_divergences_integer(self):
        view = MCMCView(_mcmc_tree(n_divergences=3))
        assert view.n_divergences == 3

    def test_n_divergences_zero(self):
        view = MCMCView(_mcmc_tree(n_divergences=0))
        assert view.n_divergences == 0

    def test_n_divergences_not_recorded(self):
        view = MCMCView(_mcmc_tree())
        result = view.n_divergences
        assert isinstance(result, NotComputed)

    def test_n_divergences_none_tree(self):
        view = MCMCView(None)
        result = view.n_divergences
        assert isinstance(result, NotComputed)

    def test_n_divergences_not_computed_payload(self):
        tree = xr.DataTree(
            dataset=xr.Dataset(attrs={"n_divergences": json.dumps({"not_computed": "no stats"})})
        )
        result = MCMCView(tree).n_divergences
        assert isinstance(result, NotComputed)
        assert result.reason == "no stats"

    def test_n_divergences_unparseable(self):
        tree = xr.DataTree(dataset=xr.Dataset(attrs={"n_divergences": "not-an-int"}))
        result = MCMCView(tree).n_divergences
        assert isinstance(result, NotComputed)

    def test_warnings_from_attrs(self):
        view = MCMCView(_mcmc_tree(rhat_warnings=["R-hat > 1.01 for 'alpha'"]))
        assert any("alpha" in w for w in view.warnings)

    def test_warnings_empty_when_none(self):
        view = MCMCView(_mcmc_tree())
        assert view.warnings == []

    def test_warnings_ignore_invalid_json_attrs(self):
        tree = xr.DataTree(dataset=xr.Dataset(attrs={"rhat_warnings": "not-json"}))
        assert MCMCView(tree).warnings == []

    def test_repr_contains_params(self):
        view = MCMCView(_mcmc_tree(rhat={"mu": 1.0}))
        assert "mu" in repr(view)


# ---------------------------------------------------------------------------
# PPCView
# ---------------------------------------------------------------------------


class TestPPCView:
    def test_p_values(self):
        view = PPCView(_ppc_tree(["mean_fn"], [0.43], [3.2]))
        assert view.p_values["mean_fn"] == pytest.approx(0.43)

    def test_observed(self):
        view = PPCView(_ppc_tree(["mean_fn"], [0.43], [3.2]))
        assert view.observed["mean_fn"] == pytest.approx(3.2)

    def test_result_combines_p_and_observed(self):
        view = PPCView(_ppc_tree(["mean_fn"], [0.5], [1.0]))
        r = view.result["mean_fn"]
        assert r["p_value"] == pytest.approx(0.5)
        assert r["observed"] == pytest.approx(1.0)

    def test_result_multiple_fns(self):
        view = PPCView(_ppc_tree(["mean_fn", "var_fn"], [0.4, 0.7], [1.0, 2.0]))
        assert set(view.result.keys()) == {"mean_fn", "var_fn"}

    def test_result_reports_missing_observed(self):
        da = xr.DataArray([0.25], dims=["test_fn"], coords={"test_fn": ["fn"]})
        view = PPCView(xr.DataTree(dataset=xr.Dataset({"p_value": da})))
        assert isinstance(view.result["fn"]["observed"], NotComputed)

    def test_result_reports_missing_p_value(self):
        da = xr.DataArray([1.25], dims=["test_fn"], coords={"test_fn": ["fn"]})
        view = PPCView(xr.DataTree(dataset=xr.Dataset({"observed": da})))
        assert isinstance(view.result["fn"]["p_value"], NotComputed)

    def test_replicated_summaries(self):
        coords = {"test_fn": ["fn"]}
        ds = xr.Dataset(
            {
                "replicated_stat_mean": xr.DataArray([2.0], dims=["test_fn"], coords=coords),
                "replicated_stat_sd": xr.DataArray([0.5], dims=["test_fn"], coords=coords),
            }
        )
        view = PPCView(xr.DataTree(dataset=ds))
        assert view.replicated_stat_mean["fn"] == pytest.approx(2.0)
        assert view.replicated_stat_sd["fn"] == pytest.approx(0.5)

    def test_observed_backward_compat_field_name(self):
        """Falls back to 'observed_statistic' if 'observed' is absent."""
        da = xr.DataArray([9.9], dims=["test_fn"], coords={"test_fn": ["fn"]})
        ds = xr.Dataset({"observed_statistic": da})
        view = PPCView(xr.DataTree(dataset=ds))
        assert view.observed["fn"] == pytest.approx(9.9)

    def test_plot_ready_false_by_default(self):
        view = PPCView(_ppc_tree(["fn"], [0.5], [1.0]))
        assert view.plot_ready is False

    def test_plot_ready_true(self):
        view = PPCView(_ppc_tree(["fn"], [0.5], [1.0], plot_ready=True))
        assert view.plot_ready is True

    def test_timestamp(self):
        view = PPCView(_ppc_tree(["fn"], [0.5], [1.0], timestamp="2025-01-01T00:00:00"))
        assert "2025" in view.timestamp

    def test_repr_not_computed_when_none(self):
        assert "not computed" in repr(PPCView(None))

    def test_repr_with_data(self):
        view = PPCView(_ppc_tree(["fn"], [0.5], [1.0]))
        assert "fn" in repr(view)


# ---------------------------------------------------------------------------
# LOOView
# ---------------------------------------------------------------------------


class TestLOOView:
    def test_elpd_loo(self):
        view = LOOView(_loo_tree(elpd_loo=-42.5))
        assert view.elpd_loo == pytest.approx(-42.5)

    def test_se(self):
        view = LOOView(_loo_tree(se=3.1))
        assert view.se == pytest.approx(3.1)

    def test_looic(self):
        view = LOOView(_loo_tree(looic=85.0))
        assert view.looic == pytest.approx(85.0)

    def test_p_loo_and_pareto_k_mean(self):
        view = LOOView(_loo_tree(p_loo=3.0, pareto_k_mean=0.25))
        assert view.p_loo == pytest.approx(3.0)
        assert view.pareto_k_mean == pytest.approx(0.25)

    def test_pareto_k_bad_count_int(self):
        view = LOOView(_loo_tree(pareto_k_bad_count=2.0))
        result = view.pareto_k_bad_count
        assert result == 2
        assert isinstance(result, int)

    def test_warning_false(self):
        view = LOOView(_loo_tree(warning=0.0))
        assert view.warning is False

    def test_warning_not_computed_is_false(self):
        view = LOOView(_loo_tree())
        assert view.warning is False

    def test_plot_metadata(self):
        tree = xr.DataTree(
            dataset=xr.Dataset(attrs={"plot_ready": True, "plot_fn": "az.plot_loo_pit"})
        )
        view = LOOView(tree)
        assert view.plot_ready is True
        assert view.plot_fn == "az.plot_loo_pit"

    def test_warning_true(self):
        view = LOOView(_loo_tree(warning=1.0))
        assert view.warning is True

    def test_warnings_includes_arviz_warning(self):
        view = LOOView(_loo_tree(warning=1.0, pareto_k_max=0.5))
        assert any("reliability" in w.lower() or "LOO" in w for w in view.warnings)

    def test_warnings_includes_pareto_k_threshold(self):
        view = LOOView(_loo_tree(warning=0.0, pareto_k_max=0.75))
        assert any("0.7" in w for w in view.warnings)

    def test_warnings_empty_when_fine(self):
        view = LOOView(_loo_tree(warning=0.0, pareto_k_max=0.5))
        assert view.warnings == []

    def test_not_computed_when_none(self):
        view = LOOView(None)
        assert isinstance(view.elpd_loo, NotComputed)

    def test_repr_not_computed(self):
        assert "not computed" in repr(LOOView(None))

    def test_repr_with_data(self):
        view = LOOView(_loo_tree(elpd_loo=-10.0, se=1.0, looic=20.0, pareto_k_max=0.3))
        assert "elpd_loo" in repr(view)


# ---------------------------------------------------------------------------
# DiagnosticsView
# ---------------------------------------------------------------------------


class TestDiagnosticsView:
    def _view_with_mcmc(self, rhat=None, ess_bulk=None):
        rhat = rhat or {"alpha": 1.001, "beta": 1.0}
        ess_bulk = ess_bulk or {"alpha": 450.0, "beta": 500.0}
        mcmc_ds = xr.Dataset(
            {
                "rhat": xr.DataArray(
                    list(rhat.values()), dims=["param"], coords={"param": list(rhat.keys())}
                ),
                "ess_bulk": xr.DataArray(
                    list(ess_bulk.values()), dims=["param"], coords={"param": list(ess_bulk.keys())}
                ),
            }
        )
        tree = _diagnostics_tree(mcmc_ds=mcmc_ds)
        return DiagnosticsView(tree)

    def test_rhat_passthrough(self):
        view = self._view_with_mcmc(rhat={"mu": 1.002})
        assert "mu" in view.rhat

    def test_ess_bulk_passthrough(self):
        view = self._view_with_mcmc(ess_bulk={"mu": 600.0})
        assert view.ess_bulk["mu"] == pytest.approx(600.0)

    def test_mcmc_subview(self):
        view = self._view_with_mcmc()
        assert isinstance(view.mcmc, MCMCView)

    def test_ppc_subview_not_computed_when_absent(self):
        view = self._view_with_mcmc()
        assert not view.ppc.exists

    def test_loo_subview_not_computed_when_absent(self):
        view = self._view_with_mcmc()
        assert not view.loo.exists

    def test_runs_empty_when_no_runs(self):
        view = self._view_with_mcmc()
        assert view.runs == []

    def test_runs_lists_ppc(self):
        ppc_ds = xr.Dataset(
            {"p_value": xr.DataArray([0.5], dims=["test_fn"], coords={"test_fn": ["fn"]})}
        )
        tree = _diagnostics_tree(ppc_ds=ppc_ds)
        view = DiagnosticsView(tree)
        assert any(r.name == "ppc" for r in view.runs)

    def test_child_or_none_returns_none_when_child_lookup_fails(self):
        class _TreeLike:
            def __init__(self):
                self.children = {"runs": object()}

            def __getitem__(self, key):
                raise RuntimeError("cannot descend")

        assert DiagnosticsView(_TreeLike())._child_or_none("runs", "ppc") is None

    def test_summary_table_contains_params(self):
        view = self._view_with_mcmc()
        table = view.summary_table()
        assert "alpha" in table
        assert "R-hat" in table

    def test_summary_table_formats_not_computed(self):
        da = xr.DataArray(
            [float("nan")],
            dims=["param"],
            coords={"param": ["alpha"]},
            attrs={"not_computed_alpha": "single chain"},
        )
        view = DiagnosticsView(_diagnostics_tree(mcmc_ds=xr.Dataset({"rhat": da})))

        assert "N/A" in view.summary_table()

    def test_summary_table_no_diagnostics(self):
        view = DiagnosticsView(None)
        table = view.summary_table()
        assert "No MCMC" in table

    def test_to_dict_keys(self):
        view = self._view_with_mcmc()
        d = view.to_dict()
        assert set(d.keys()) >= {"mcmc", "ppc", "loo", "runs"}

    def test_to_dict_json_serialisable(self):
        view = self._view_with_mcmc()
        json.dumps(view.to_dict())

    def test_warnings_aggregates_mcmc_and_loo(self):
        view = self._view_with_mcmc()
        # No LOO, no bad R-hat — should be empty or a list
        assert isinstance(view.warnings, list)

    def test_repr_contains_params(self):
        view = self._view_with_mcmc(rhat={"mu": 1.0})
        assert "mu" in repr(view)

    def test_none_tree(self):
        view = DiagnosticsView(None)
        assert view.rhat == {}
        assert view.runs == []
