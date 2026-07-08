"""Discrepancy measures between pairs of distributions.

A *discrepancy* quantifies how far one distribution is from another —
divergences such as Kullback-Leibler and metrics such as total variation,
Wasserstein, or MMD.  Each family is a binary operation on
``(Distribution, Distribution)`` and is implemented as its own
:class:`~probpipe.core._registry.BinaryDispatchRegistry` so the best
feasible method (closed-form when available, Monte Carlo otherwise) is
selected by priority.  See ``docs/contributor/dispatch-conventions.md``.

This package is distinct from :mod:`probpipe.validation`, which is about
*model adequacy* (predictive checks and calibration); discrepancies are
general distribution-to-distribution comparison primitives that validation
and other layers may build on.

Importing this package constructs the discrepancy registries, which
self-register in the global :data:`~probpipe.registry_catalog` (e.g. under
the ``"kl"`` key) — the side effect that makes them discoverable after
``import probpipe``.

The public ops live in :mod:`probpipe.core.ops` (e.g.
:func:`probpipe.kl_divergence`).
"""

from ._kl_registry import kl_registry

__all__ = ["kl_registry"]
