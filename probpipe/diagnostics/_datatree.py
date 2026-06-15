"""Backward-compatible facade for diagnostics DataTree helpers and views.

New code should prefer importing:

- storage helpers from ``probpipe.diagnostics._datatree_store``
- view classes from ``probpipe.diagnostics._views``

This module remains for existing internal imports.
"""
from __future__ import annotations

from ._datatree_store import (
    _add_group,
    _get_or_create_mcmc_ds,
    _write_mcmc_field,
    _mcmc_has_field,
    to_named_posterior_dataset,
)

from ._view_base import (
    NotComputed,
    read_scalar as _read_scalar,
    read_indexed,
    read_json_attr,
    DataTreeView,
    DatasetView,
    DiagnosticRunView,
)

from ._views import (
    DiagnosticsView,
    MCMCView,
    PPCView,
    LOOView,
)

__all__ = [
    # Storage helpers
    "_add_group",
    "_get_or_create_mcmc_ds",
    "_write_mcmc_field",
    "_mcmc_has_field",
    "to_named_posterior_dataset",
    # Generic view helpers/classes
    "NotComputed",
    "_read_scalar",
    "read_indexed",
    "read_json_attr",
    "DataTreeView",
    "DatasetView",
    "DiagnosticRunView",
    # Concrete views
    "DiagnosticsView",
    "MCMCView",
    "PPCView",
    "LOOView",
]