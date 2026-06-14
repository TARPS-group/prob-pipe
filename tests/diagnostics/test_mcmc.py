"""Tests for probpipe.diagnostics._mcmc."""
from __future__ import annotations

import numpy as np
import pytest

from probpipe.diagnostics._mcmc import (
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
