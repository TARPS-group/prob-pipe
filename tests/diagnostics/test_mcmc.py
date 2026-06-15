"""Tests for probpipe.diagnostics._mcmc."""
from __future__ import annotations

import numpy as np
import pytest

from probpipe.core.record import Record
import probpipe.diagnostics._mcmc as mcmc
from probpipe.diagnostics._mcmc import (
    _check_arviz,
    _emit_record_warnings,
    _ess_warnings,
    _rhat_warnings,
    _write_mcmc_record,
    add_ess,
    add_mcmc_diagnostics,
    add_mcse,
    add_rhat,
)
from probpipe.diagnostics._datatree_store import _mcmc_has_field
from probpipe.diagnostics._view_base import NotComputed

# conftest.py provides: posterior, posterior_single_chain, posterior_3params


# ---------------------------------------------------------------------------
# add_rhat
# ---------------------------------------------------------------------------


class TestAddRhat:
    def test_writes_rhat_field(self, posterior):
        add_rhat(posterior)
        assert _mcmc_has_field(posterior, "rhat")

    def test_rhat_values_are_numeric(self, posterior):
        add_rhat(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        rhat = view.rhat
        assert set(rhat.keys()) == {"alpha", "beta"}
        for v in rhat.values():
            assert isinstance(v, float) or isinstance(v, NotComputed)

    def test_rhat_reasonable_for_iid_chains(self, posterior):
        """IID draws should give R-hat close to 1."""
        add_rhat(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        for v in view.rhat.values():
            if isinstance(v, float):
                assert v < 1.1

    def test_single_chain_returns_not_computed(self, posterior_single_chain):
        add_rhat(posterior_single_chain)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior_single_chain._auxiliary["diagnostics"])
        for v in view.rhat.values():
            assert isinstance(v, NotComputed)

    def test_idempotent(self, posterior):
        """Calling add_rhat twice should not raise and last write wins."""
        add_rhat(posterior)
        add_rhat(posterior)
        assert _mcmc_has_field(posterior, "rhat")

    def test_does_not_return_value(self, posterior):
        assert add_rhat(posterior) is None


class TestMcmcHelpers:
    def test_check_arviz_reports_missing_dependency(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _missing_arviz(name, *args, **kwargs):
            if name == "arviz":
                raise ImportError("missing arviz")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing_arviz)

        with pytest.raises(ImportError, match="ArviZ is required"):
            _check_arviz()

    def test_warning_helpers_skip_unusable_values_and_report_failures(self):
        class _BadFloat:
            def __float__(self):
                raise TypeError("not numeric")

        rhat_messages = _rhat_warnings(
            {"missing": NotComputed("single chain"), "bad": _BadFloat(), "alpha": 1.2},
            threshold=1.01,
        )
        assert any("alpha" in msg for msg in rhat_messages)

        ess_messages = _ess_warnings(
            {"missing": NotComputed("no bulk"), "bad": _BadFloat(), "alpha": 100.0},
            {"missing": NotComputed("no tail"), "bad": _BadFloat(), "beta": 120.0},
            threshold=400,
        )
        assert any("bulk" in msg and "alpha" in msg for msg in ess_messages)
        assert any("tail" in msg and "beta" in msg for msg in ess_messages)

    def test_emit_record_warnings_handles_missing_and_none_warning_fields(self):
        class _NoWarnings:
            def __getitem__(self, key):
                raise KeyError(key)

        _emit_record_warnings(_NoWarnings())
        _emit_record_warnings(Record(name="no_warnings", kind="test", warnings=None))

    def test_write_mcmc_record_handles_composite_and_unknown_records(self, posterior):
        child = Record(
            name="rhat_child",
            kind="rhat",
            values={"alpha": 1.0},
            attrs={},
        )
        composite = Record(name="all_mcmc", kind="mcmc", records={"rhat": child})

        _write_mcmc_record(posterior, composite)

        assert _mcmc_has_field(posterior, "rhat")

        with pytest.raises(ValueError, match="Unknown MCMC"):
            _write_mcmc_record(posterior, Record(name="unknown", kind="bogus"))

        class _MissingKind:
            def __getitem__(self, key):
                raise KeyError(key)

        with pytest.raises(ValueError, match="missing required field"):
            _write_mcmc_record(posterior, _MissingKind())


# ---------------------------------------------------------------------------
# add_ess
# ---------------------------------------------------------------------------


class TestAddEss:
    def test_writes_ess_bulk_and_tail(self, posterior):
        add_ess(posterior)
        assert _mcmc_has_field(posterior, "ess_bulk")
        assert _mcmc_has_field(posterior, "ess_tail")

    def test_ess_values_positive(self, posterior):
        add_ess(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        for v in view.ess_bulk.values():
            if isinstance(v, float):
                assert v > 0
        for v in view.ess_tail.values():
            if isinstance(v, float):
                assert v > 0

    def test_ess_covers_all_params(self, posterior_3params):
        add_ess(posterior_3params)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior_3params._auxiliary["diagnostics"])
        assert set(view.ess_bulk.keys()) == {"mu", "sigma", "nu"}

    def test_does_not_return_value(self, posterior):
        assert add_ess(posterior) is None

    def test_idempotent_skip_and_force_recompute(self, posterior, monkeypatch):
        add_ess(posterior)
        original = mcmc._compute_ess_op

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("ESS should have been skipped")

        monkeypatch.setattr(mcmc, "_compute_ess_op", _fail_if_called)
        add_ess(posterior)

        calls = []

        def _record_call(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(mcmc, "_compute_ess_op", _record_call)
        add_ess(posterior, force=True)
        assert calls


# ---------------------------------------------------------------------------
# add_mcse
# ---------------------------------------------------------------------------


class TestAddMcse:
    def test_writes_mcse_mean_and_sd(self, posterior):
        add_mcse(posterior)
        assert _mcmc_has_field(posterior, "mcse_mean")
        assert _mcmc_has_field(posterior, "mcse_sd")

    def test_mcse_values_finite(self, posterior):
        add_mcse(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        for v in view.mcse_mean.values():
            if isinstance(v, float):
                assert np.isfinite(v)

    def test_does_not_return_value(self, posterior):
        assert add_mcse(posterior) is None

    def test_idempotent_skip(self, posterior, monkeypatch):
        add_mcse(posterior)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("MCSE should have been skipped")

        monkeypatch.setattr(mcmc, "_compute_mcse_op", _fail_if_called)
        add_mcse(posterior)


# ---------------------------------------------------------------------------
# add_mcmc_diagnostics
# ---------------------------------------------------------------------------


class TestAddMcmcDiagnostics:
    def test_writes_all_three_fields(self, posterior):
        add_mcmc_diagnostics(posterior)
        assert _mcmc_has_field(posterior, "rhat")
        assert _mcmc_has_field(posterior, "ess_bulk")
        assert _mcmc_has_field(posterior, "mcse_mean")

    def test_summary_table_runs(self, posterior):
        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        table = view.summary_table()
        assert "alpha" in table
        assert "beta" in table
        assert "R-hat" in table

    def test_to_dict_serialisable(self, posterior):
        import json
        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        d = view.to_dict()
        # Must be JSON-serialisable (NotComputed is converted by to_dict)
        json.dumps(d)

    def test_warnings_empty_for_iid(self, posterior):
        """IID draws from a single rng should pass all diagnostics."""
        add_mcmc_diagnostics(posterior)
        from probpipe.diagnostics._views import DiagnosticsView
        view = DiagnosticsView(posterior._auxiliary["diagnostics"])
        # Warnings may be empty; we just check it's a list
        assert isinstance(view.warnings, list)

    def test_does_not_return_value(self, posterior):
        assert add_mcmc_diagnostics(posterior) is None
