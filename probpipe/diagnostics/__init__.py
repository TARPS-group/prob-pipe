"""Diagnostic workflow functions for ProbPipe.

MCMC diagnostics::

    from probpipe.diagnostics import (
        compute_rhat,
        compute_ess,
        compute_mcse,
        mcmc_summary,
        plot_trace,
        plot_rank,
        plot_kde,
        fit_kde,
    )

Predictive checks::

    from probpipe.diagnostics import run_ppc, run_spc

LOO-PSIS::

    from probpipe.diagnostics import loo, compare_loo

Orchestrated module::

    from probpipe.diagnostics import DiagnosticsModule

    record = DiagnosticsModule.default().run(posterior)
    record["mcmc"]["warnings"]
"""
from __future__ import annotations

from .mcmc import (
    compute_rhat,
    compute_ess,
    compute_mcse,
    mcmc_summary,
    plot_trace,
    plot_rank,
    plot_kde,
    fit_kde,
)
from ._ppc_spc import run_ppc, run_spc
from ._loo import loo, compare_loo
from .diagnostics_workflow import DiagnosticsModule

__all__ = [
    # Numerical diagnostics
    "compute_rhat",
    "compute_ess",
    "compute_mcse",
    "mcmc_summary",
    # Visual diagnostics
    "plot_trace",
    "plot_rank",
    "plot_kde",
    "fit_kde",
    # Predictive checks
    "run_ppc",
    "run_spc",
    # LOO-PSIS
    "loo",
    "compare_loo",
    # Orchestrated module
    "DiagnosticsModule",
]