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
    _mcmc_has_field,
    _write_mcmc_field,
    to_named_posterior_dataset,
)
from ._view_base import (
    DatasetView,
    DataTreeView,
    DiagnosticRunView,
    NotComputed,
    read_indexed,
    read_json_attr,
)
from ._view_base import (
    read_scalar as _read_scalar,
)
from ._views import (
    DiagnosticsView,
    LOOView,
    MCMCView,
    PPCView,
    SensitivityView,
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
    "SensitivityView",
    "_add_group",
    "_get_or_create_mcmc_ds",
    "_mcmc_has_field",
    "_read_scalar",
    "_write_mcmc_field",
    "read_indexed",
    "read_json_attr",
    "to_named_posterior_dataset",
]
