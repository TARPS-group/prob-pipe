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
        DiagnosticsModule,
    )
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
)
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
    # Orchestrated module
    "DiagnosticsModule",
]
