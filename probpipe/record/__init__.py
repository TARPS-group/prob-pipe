"""Record-adjacent abstractions: parameter-sweep designs and related helpers.

The base ``Record`` / ``RecordArray`` family and their distribution
counterparts live in :mod:`probpipe.core` (they're foundational); this
subpackage collects higher-level record-based constructions that use
them. At present:

- :class:`~probpipe.record.design.FullFactorialDesign` and its
  :class:`Design` base — parameter-sweep ``RecordArray``s that plug
  directly into the ``WorkflowFunction`` sweep path.

Re-exported from :mod:`probpipe` for convenience.
"""

from __future__ import annotations

from .design import Design, FullFactorialDesign

__all__ = [
    "Design",
    "FullFactorialDesign",
]
