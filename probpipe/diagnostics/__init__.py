"""Diagnostic functions for ProbPipe.

This package exposes diagnostic APIs for adding Bayesian diagnostics to
ProbPipe posterior objects.

Primary public API
------------------

In-place diagnostic functions mutate ``posterior._auxiliary`` and return
``None``::

    from probpipe.diagnostics import (
        add_rhat,
        add_ess,
        add_mcse,
        add_mcmc_diagnostics,
        add_ppc,
        add_loo,
        add_sensitivity,
    )

These functions write diagnostic summaries under::

    posterior._auxiliary["diagnostics"]

and, when needed, ArviZ-compatible data under::

    posterior._auxiliary["arviz"]

``posterior._auxiliary["arviz"]`` contains ArviZ-compatible xarray DataTree
data and raw diagnostic inputs. ``posterior._auxiliary["diagnostics"]`` contains
ProbPipe-computed summaries, results, warnings, and metadata exposed through
``posterior.diagnostics``.

Diagnostic accessors
--------------------

``posterior.diagnostics`` returns a structured view over the diagnostics
subtree. The accessor classes are available as::

    from probpipe.diagnostics import (
        DiagnosticsView,
        DiagnosticRunView,
        MCMCView,
        PPCView,
        LOOView,
        SensitivityView,
        NotComputed,
    )

ArviZ plotting
--------------

ArviZ-compatible data live under ``posterior._auxiliary["arviz"]`` and are
exposed as ``posterior.arviz_data``. For backward compatibility,
``posterior.inference_data`` is an alias for the same DataTree subtree.
Plotting support depends on which ArviZ groups have been written::

    import arviz as az

    az.plot_trace(posterior.arviz_data)
    az.plot_loo_pit(posterior.arviz_data)
"""

from __future__ import annotations

__all__: list[str] = []


# ── MCMC diagnostics ──────────────────────────────────────────────────────
from ._mcmc import (
    add_ess,
    add_mcmc_diagnostics,
    add_mcse,
    add_rhat,
)

__all__ += [
    "add_ess",
    "add_mcmc_diagnostics",
    "add_mcse",
    "add_rhat",
]


# ── Predictive checks ────────────────────────────────────────────────────
from ._ppc_spc import add_ppc

__all__ += [
    "add_ppc",
]


# ── LOO-PSIS ─────────────────────────────────────────────────────────────
from ._loo import add_loo

__all__ += [
    "add_loo",
]


# ── Power-scaling sensitivity ─────────────────────────────────────────────
from ._sensitivity import add_sensitivity

__all__ += [
    "add_sensitivity",
]


# ── Diagnostic view classes ───────────────────────────────────────────────
# Prefer the new public facade if present. Fall back to _datatree for
# compatibility during the refactor.
try:
    from .views import (
        DiagnosticRunView,
        DiagnosticsView,
        LOOView,
        MCMCView,
        NotComputed,
        PPCView,
        SensitivityView,
    )
except ImportError:
    from ._datatree import (
        DiagnosticRunView,
        DiagnosticsView,
        LOOView,
        MCMCView,
        NotComputed,
        PPCView,
        SensitivityView,
    )

__all__ += [
    "DiagnosticRunView",
    "DiagnosticsView",
    "LOOView",
    "MCMCView",
    "NotComputed",
    "PPCView",
    "SensitivityView",
]
