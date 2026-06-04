"""Diagnostic functions for ProbPipe.

MCMC diagnostics (mutate posterior in place, return None)::

    from probpipe.diagnostics import (
        mcmc_diagnostics,   # convenience: rhat + ess + mcse in one call
        compute_rhat,
        compute_ess,
        compute_mcse,
    )

    posterior = condition_on(model, data)
    mcmc_diagnostics(posterior)

    posterior.diagnostics.rhat      # {"intercept": 1.001}
    posterior.diagnostics.warnings  # []
    posterior.diagnostics.runs      # []

Predictive checks (mutate posterior in place, return None)::

    from probpipe.diagnostics import run_ppc, run_spc

LOO-PSIS (mutate posterior in place, return None)::

    from probpipe.diagnostics import loo, compare_loo

Sensitivity analysis::

    from probpipe.diagnostics import (
        prior_sensitivity,
        data_sensitivity,
        power_scale_sensitivity,
    )

Diagnostic accessors (returned by posterior.diagnostics)::

    from probpipe.diagnostics import DiagnosticsView, DiagnosticRunView, NotComputed

ArviZ plots — use posterior.inference_data directly::

    import arviz as az
    az.plot_trace(posterior.inference_data)    # always available
    az.plot_ppc(posterior.inference_data)      # after run_ppc
    az.plot_loo_pit(posterior.inference_data)  # after loo + run_ppc
"""
from __future__ import annotations

# ── MCMC diagnostics ──────────────────────────────────────────────────────
from ._mcmc import (
    mcmc_diagnostics,
    compute_rhat,
    compute_ess,
    compute_mcse,
)

# ── Predictive checks ────────────────────────────────────────────────────
from ._ppc_spc import run_ppc, run_spc

# ── LOO-PSIS ─────────────────────────────────────────────────────────────
from ._loo import loo, compare_loo

# ── Sensitivity analysis ─────────────────────────────────────────────────
from ._sensitivity import (
    prior_sensitivity,
    data_sensitivity,
    power_scale_sensitivity,
)

# ── Accessor classes ──────────────────────────────────────────────────────
from ._datatree import (
    DiagnosticsView,
    DiagnosticRunView,
    NotComputed,
)

__all__ = [
    # MCMC diagnostics
    "mcmc_diagnostics",
    "compute_rhat",
    "compute_ess",
    "compute_mcse",
    # Predictive checks
    "run_ppc",
    "run_spc",
    # LOO-PSIS
    "loo",
    "compare_loo",
    # Sensitivity analysis
    "prior_sensitivity",
    "data_sensitivity",
    "power_scale_sensitivity",
    # Accessor classes
    "DiagnosticsView",
    "DiagnosticRunView",
    "NotComputed",
]
