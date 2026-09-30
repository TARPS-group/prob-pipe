"""The program-defined families: backend models with explicit given and event contracts.

A program-defined model exposes what its backend provides: a ``Distribution``
for a joint law over modeled variables or a data-bound parameter target, and a
``ConditionalDistribution`` over its data inputs otherwise. Each adapter
declares its data inputs separately from its event variables, whose program
names determine the output components.

``StanModel`` is defined in :mod:`probpipe.modeling._stan`, and ``PyMCModel``
in :mod:`probpipe.modeling._pymc`.
"""

from __future__ import annotations

__all__: list[str] = []
