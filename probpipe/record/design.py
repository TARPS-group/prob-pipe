"""Parameter-sweep Designs — materialised ``RecordArray``s with marginals.

A :class:`Design` is a :class:`~probpipe.RecordArray` whose rows are
materialised from per-field **marginals** — the candidate values for
each field — combined according to a subclass-specific rule. The
resulting ``RecordArray`` plugs into the ``WorkflowFunction`` sweep
path as a single array-valued input::

    result = fit(p=design)    # one inner call per row of the sweep

This module currently exports :class:`FullFactorialDesign` only;
additional subclasses (`RandomDesign`, `LatinHypercubeDesign`,
`SobolDesign`) are planned as follow-up PRs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from .._utils import prod
from ..core._record_array import RecordArray
from ..core.record import RecordTemplate

__all__ = ["Design", "FullFactorialDesign"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric_sequence(seq: Any) -> bool:
    """True if *seq* is a sequence (or array) of numeric scalars.

    Strings and byte sequences are rejected (they're iterable but
    categorical-valued). We probe via ``jnp.asarray`` and check the
    resulting dtype's kind.
    """
    if isinstance(seq, (str, bytes)):
        return False
    try:
        arr = jnp.asarray(seq)
    except (TypeError, ValueError):
        return False
    return arr.dtype.kind in "biufc"


def _seq_to_column(
    values: Sequence, *, indices,
) -> tuple[Any, tuple[int, ...] | None]:
    """Materialise ``values[indices]`` as a column array.

    Returns ``(column, leaf_shape)``. For numeric values the column is
    a ``jnp.ndarray`` and ``leaf_shape`` is ``()`` (scalar leaves) or
    the trailing shape of the first element. For non-numeric values
    (strings, Python objects) the column is a ``numpy.ndarray`` with
    ``dtype=object`` and ``leaf_shape`` is ``None`` (opaque leaf).
    """
    seq = list(values)
    if _is_numeric_sequence(seq):
        arr = jnp.asarray(seq)
        leaf_shape = tuple(arr.shape[1:])
        return arr[indices], leaf_shape
    obj = np.asarray(seq, dtype=object)
    return obj[np.asarray(indices)], None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Design(RecordArray):
    """``RecordArray`` that carries its per-field marginals.

    A ``Design`` is not meant to be instantiated directly — concrete
    subclasses (:class:`FullFactorialDesign`) assemble the underlying
    rows in ``__init__`` and stash the originating marginals for
    introspection.

    Usage inside a ``@workflow_function``: pass the design as a single
    ``Record`` / ``NumericRecord``-typed argument. The WF layer's
    sweep path iterates over its batch and returns a stacked output::

        @workflow_function
        def fit(p): ...                  # p arrives as one row (a Record)
        result = fit(p=design)           # one inner call per row

    ``select_all()`` returns the per-field columns as raw arrays — useful
    for inspection or for calls into JAX-vectorizable bodies that don't
    need the sweep machinery. It does **not** trigger the WF sweep (the
    raw arrays aren't recognised as array-valued inputs).

    Attributes
    ----------
    marginals : Mapping[str, Any]
        The per-field marginals this design was built from, in sorted
        field order. Kept for introspection; read-only.
    """

    __slots__ = ("_marginals",)

    @property
    def marginals(self) -> Mapping[str, Any]:
        """Per-field marginals this design was built from."""
        return dict(self._marginals)

    def select_all(self) -> dict[str, Any]:
        """Return ``{field: column_array}`` for splat-as-kwargs calls.

        The returned values are the underlying field columns, not
        single-field ``RecordArray`` wrappers. Splatting them as
        kwargs into a ``@workflow_function`` therefore **does not**
        trigger the WF sweep path — the WF runs once with full column
        arrays and relies on whatever broadcasting the inner body
        naturally provides.

        Use this as a faster alternative to the single-Record-arg
        sweep pattern (``f(p=design)``) when the body's ops broadcast
        over column arrays — typically pure JAX arithmetic. It is
        **not** a general substitute: anything that expects per-element
        values (``f'{x:.1f}'``, ``float(x)``, ``if x > 0``, scalar-only
        external library calls) will fail because those receive full
        arrays, not per-row scalars.
        """
        return {f: self._store[f] for f in self.fields}


# ---------------------------------------------------------------------------
# FullFactorialDesign
# ---------------------------------------------------------------------------


class FullFactorialDesign(Design):
    """Cartesian product over all marginals — one row per combination.

    Each marginal is a Python sequence (list, tuple, numpy / jax
    array). Numeric marginals become ``jnp.ndarray`` columns and
    categorical / string marginals become ``numpy.ndarray(dtype=object)``
    columns. Row order is lexicographic (row-major) over the sorted
    field names — matching the sort order :class:`~probpipe.Record`
    uses internally for field iteration.

    Parameters
    ----------
    **marginals : Sequence
        Candidate values for each field. Must pass at least one
        marginal; each must be non-empty.

    Examples
    --------
    Cartesian grid of two numeric fields:

    >>> ff = FullFactorialDesign(r=[1.5, 1.8], K=[60.0, 80.0])
    >>> ff.batch_shape
    (4,)
    >>> sorted(ff.fields)
    ['K', 'r']

    Mixed numeric / categorical marginals are supported — columns fall
    out as ``object``-dtype arrays for the categorical fields:

    >>> ff2 = FullFactorialDesign(method=['nutpie', 'pymc'], scale=[0.5, 1.0])
    >>> ff2.batch_shape
    (4,)
    """

    def __init__(self, **marginals: Sequence) -> None:
        if not marginals:
            raise ValueError(
                "FullFactorialDesign requires at least one marginal"
            )
        sorted_names = sorted(marginals)
        lists = [list(marginals[n]) for n in sorted_names]
        sizes = [len(v) for v in lists]
        if any(s == 0 for s in sizes):
            raise ValueError(
                "FullFactorialDesign marginals must each be non-empty; "
                f"got sizes {dict(zip(sorted_names, sizes))}"
            )
        n_total = prod(sizes)
        # ``meshgrid(..., indexing='ij')`` then flatten: each axis
        # iterates at its own stride; C-order flatten then yields a
        # lexicographic row-major traversal over the sorted axes.
        grids = np.meshgrid(
            *(np.arange(s) for s in sizes), indexing="ij",
        )
        flat_indices = {
            name: grid.reshape(-1) for name, grid in zip(sorted_names, grids)
        }

        fields: dict[str, Any] = {}
        template_spec: dict[str, Any] = {}
        for name, values in zip(sorted_names, lists):
            col, leaf_shape = _seq_to_column(
                values, indices=flat_indices[name],
            )
            fields[name] = col
            template_spec[name] = leaf_shape

        RecordArray.__init__(
            self, fields,
            batch_shape=(n_total,),
            template=RecordTemplate(template_spec),
            name=f"FullFactorialDesign({','.join(sorted_names)})",
        )
        object.__setattr__(self, "_marginals", dict(marginals))
