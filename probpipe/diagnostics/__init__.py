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
        add_spc,
        loo,
        run_loo,
    )

These functions write diagnostic summaries under::

    posterior._auxiliary["diagnostics"]

and, when needed, ArviZ-compatible data under::

    posterior._auxiliary["arviz"]

Unified diagnostic workflow
---------------------------

::

    from probpipe.diagnostics import DiagnosticsModule

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
        NotComputed,
    )

ArviZ plotting
--------------

ArviZ-compatible data live under ``posterior._auxiliary["arviz"]``.
For backward compatibility, ``posterior.inference_data`` may expose that
ArviZ-compatible DataTree subtree::

    import arviz as az

    az.plot_trace(posterior.inference_data)
    az.plot_ppc(posterior.inference_data)
    az.plot_loo_pit(posterior.inference_data)
"""
from __future__ import annotations

__all__: list[str] = []


# ── Unified diagnostics workflow ──────────────────────────────────────────
try:
    from .diagnostics_workflow import DiagnosticsModule

    __all__ += [
        "DiagnosticsModule",
    ]
except ImportError:
    pass


# ── MCMC diagnostics ──────────────────────────────────────────────────────
from ._mcmc import (
    add_rhat,
    add_ess,
    add_mcse,
    add_mcmc_diagnostics,
)

__all__ += [
    "add_rhat",
    "add_ess",
    "add_mcse",
    "add_mcmc_diagnostics",
]


# ── Predictive checks ────────────────────────────────────────────────────
from ._ppc_spc import (
    add_ppc,
    add_spc,
)

__all__ += [
    "add_ppc",
    "add_spc",
]


# # ── LOO-PSIS ─────────────────────────────────────────────────────────────
# # LOO may be mid-refactor, so support both:
# #   - loo_op + loo + run_loo
# #   - loo + run_loo only
# try:
#     from ._loo import (
#         loo_op,
#         loo,
#         run_loo,
#     )

#     __all__ += [
#         "loo_op",
#         "loo",
#         "run_loo",
#     ]
# except ImportError:
#     from ._loo import (
#         loo,
#         run_loo,
#     )

#     __all__ += [
#         "loo",
#         "run_loo",
#     ]


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


# ── Diagnostic view classes ───────────────────────────────────────────────
# Prefer the new public facade if present. Fall back to _datatree for
# compatibility during the refactor.
try:
    from .views import (
        DiagnosticsView,
        DiagnosticRunView,
        MCMCView,
        PPCView,
        LOOView,
        NotComputed,
    )
except ImportError:
    from ._datatree import (
        DiagnosticsView,
        DiagnosticRunView,
        MCMCView,
        PPCView,
        LOOView,
        NotComputed,
    )

__all__ += [
    "DiagnosticsView",
    "DiagnosticRunView",
    "MCMCView",
    "PPCView",
    "LOOView",
    "NotComputed",
]