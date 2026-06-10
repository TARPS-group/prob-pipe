"""Diagnostic functions for ProbPipe.

MCMC diagnostics mutate posterior in place and return None::

    from probpipe.diagnostics import (
        mcmc_diagnostics,
        compute_rhat,
        compute_ess,
        compute_mcse,
    )

Predictive checks mutate posterior in place and return None::

    from probpipe.diagnostics import run_ppc, run_spc

LOO-PSIS mutates posterior in place and returns None::

    from probpipe.diagnostics import loo, run_loo

Diagnostic accessors::

    from probpipe.diagnostics import DiagnosticsView, DiagnosticRunView, NotComputed

ArviZ plots use posterior.inference_data directly::

    import arviz as az
    az.plot_trace(posterior.inference_data)
    az.plot_ppc(posterior.inference_data)
    az.plot_loo_pit(posterior.inference_data)
"""
from __future__ import annotations

from .diagnostics_workflow import DiagnosticsModule

__all__ = ["DiagnosticsModule"]

# ── MCMC diagnostics ──────────────────────────────────────────────────────
try:
    from ._mcmc import (
        mcmc_diagnostics,
        compute_rhat,
        compute_ess,
        compute_mcse,
    )

    __all__ += [
        "mcmc_diagnostics",
        "compute_rhat",
        "compute_ess",
        "compute_mcse",
    ]
except ImportError:
    pass


# ── Predictive checks ────────────────────────────────────────────────────
from ._ppc_spc import run_ppc, run_spc

__all__ += [
    "run_ppc",
    "run_spc",
]


# ── LOO-PSIS ─────────────────────────────────────────────────────────────
from ._loo import loo, run_loo

__all__ += [
    "loo",
    "run_loo",
]


# ── Sensitivity analysis ─────────────────────────────────────────────────
# Optional for now because _sensitivity may depend on helpers that are still
# being refactored during the diagnostics workflow work.
try:
    from ._sensitivity import (
        prior_sensitivity,
        data_sensitivity,
        power_scale_sensitivity,
    )

    __all__ += [
        "prior_sensitivity",
        "data_sensitivity",
        "power_scale_sensitivity",
    ]
except ImportError:
    pass


# ── Accessor classes ──────────────────────────────────────────────────────
from ._datatree import (
    DiagnosticsView,
    DiagnosticRunView,
    NotComputed,
)

__all__ += [
    "DiagnosticsView",
    "DiagnosticRunView",
    "NotComputed",
]
