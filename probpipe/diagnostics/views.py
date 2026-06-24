"""Public facade for diagnostic view classes.

This module re-exports the structured diagnostic accessor classes used by
``posterior.diagnostics``.

Import from this module, or from ``probpipe.diagnostics``, rather than from
the internal modules directly.

Examples
--------
::

    from probpipe.diagnostics.views import (
        DiagnosticsView,
        MCMCView,
        PPCView,
        LOOView,
        DiagnosticRunView,
        NotComputed,
    )

    posterior.diagnostics.mcmc.rhat
    posterior.diagnostics.ppc.result
    posterior.diagnostics.loo.elpd_loo
"""

from __future__ import annotations

# -- Generic view base ------------------------------------------------------
from ._view_base import (
    DatasetView,
    DataTreeView,
    DiagnosticRunView,
    NotComputed,
    read_indexed,
    read_json_attr,
    read_scalar,
)

# -- Concrete diagnostic views ---------------------------------------------
from ._views import (
    DiagnosticsView,
    LOOView,
    MCMCView,
    PPCView,
)

__all__ = [
    "DataTreeView",
    "DatasetView",
    "DiagnosticRunView",
    "DiagnosticsView",
    "LOOView",
    "MCMCView",
    "NotComputed",
    "PPCView",
    "read_indexed",
    "read_json_attr",
    "read_scalar",
]
