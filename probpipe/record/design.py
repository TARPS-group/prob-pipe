"""Parameter-sweep Designs — materialised batches of records with marginals.

A :class:`Design` is a :class:`~probpipe.RecordBatch` whose entries are
materialised from per-field **marginals** — the candidate values for
each field — combined according to a subclass-specific rule. The batch
carries a single ``design`` level, and plugs into the ``Function`` sweep
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

from ..core._array_backend import _is_numeric_dtype
from ..core._record_batch import RecordBatch
from ..core.event_template import EventTemplate

__all__ = ["Design", "FullFactorialDesign"]

# The level a design mints: one multiplicity over its own entries, which the
# sweep layer then zips sibling views on.
DESIGN_LEVEL = "design"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_numeric_sequence(seq: Any) -> bool:
    """True if *seq* is a sequence (or array) of numeric scalars.

    Strings and byte sequences are rejected (they're iterable but
    categorical-valued). We probe via ``jnp.asarray`` and check the
    resulting dtype with the shared numeric-dtype predicate.
    """
    if isinstance(seq, (str, bytes)):
        return False
    try:
        arr = jnp.asarray(seq)
    except (TypeError, ValueError):
        return False
    return _is_numeric_dtype(arr.dtype)


def _seq_to_column(
    values: Sequence,
    *,
    indices,
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


class Design(RecordBatch):
    """A batch of records that carries its per-field marginals.

    A ``Design`` is not meant to be instantiated directly — concrete
    subclasses (:class:`FullFactorialDesign`) assemble the underlying
    rows in ``__init__`` and stash the originating marginals for
    introspection.

    Two equivalent ways to drive a sweep through a ``@function``::

        @function
        def fit(p): ...
        result = fit(p=design)              # one row per call

        @function
        def fit(r, K): ...
        result = fit(**design.select_all()) # zip across sibling views

    ``select_all()`` returns one view per field; the views carry this Design's
    own ``design`` level, and the sweep layer zips arguments that share a level
    name (so the two shapes above produce identical outputs). For raw columns
    (no sweep — just JAX broadcasting), index with ``design["r"]``.

    Attributes
    ----------
    marginals : Mapping[str, Any]
        The per-field marginals this design was built from, in
        construction (insertion) order. Kept for introspection;
        read-only.
    """

    __slots__ = ("_marginals",)

    @property
    def marginals(self) -> Mapping[str, Any]:
        """Per-field marginals this design was built from."""
        return dict(self._marginals)

    @property
    def _view_type(self) -> type:
        """A view of a design is a plain batch, not a design.

        The marginals are a statement about the whole design; a view over one
        field holds none of them, so it takes the class that makes no such claim.
        It keeps the ``design`` level, which is what the sweep layer zips on.
        """
        return RecordBatch


# ---------------------------------------------------------------------------
# FullFactorialDesign
# ---------------------------------------------------------------------------


class FullFactorialDesign(Design):
    """Cartesian product over all marginals — one row per combination.

    Each marginal is a Python sequence (list, tuple, numpy / jax
    array). Numeric marginals become ``jnp.ndarray`` columns and
    categorical / string marginals become ``numpy.ndarray(dtype=object)``
    columns. Row order is row-major over the marginals in **insertion
    order** — i.e., the last-listed marginal varies fastest.

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
    >>> tuple(ff.event_template.keys())
    ('r', 'K')

    Mixed numeric / categorical marginals are supported — columns fall
    out as ``object``-dtype arrays for the categorical fields:

    >>> ff2 = FullFactorialDesign(method=['nutpie', 'pymc'], scale=[0.5, 1.0])
    >>> ff2.batch_shape
    (4,)
    """

    def __init__(self, **marginals: Sequence) -> None:
        if not marginals:
            raise ValueError("FullFactorialDesign requires at least one marginal")
        names = list(marginals)
        lists = [list(marginals[n]) for n in names]
        sizes = [len(v) for v in lists]
        if any(s == 0 for s in sizes):
            raise ValueError(
                "FullFactorialDesign marginals must each be non-empty; "
                f"got sizes {dict(zip(names, sizes))}"
            )
        # ``meshgrid(..., indexing='ij')`` then flatten: each axis
        # iterates at its own stride; C-order flatten then yields a
        # lexicographic row-major traversal over the marginals in
        # insertion order.
        grids = np.meshgrid(
            *(np.arange(s) for s in sizes),
            indexing="ij",
        )
        flat_indices = {name: grid.reshape(-1) for name, grid in zip(names, grids)}

        fields: dict[str, Any] = {}
        template_spec: dict[str, Any] = {}
        for name, values in zip(names, lists):
            col, leaf_shape = _seq_to_column(
                values,
                indices=flat_indices[name],
            )
            fields[name] = col
            template_spec[name] = leaf_shape

        RecordBatch.__init__(
            self,
            f"FullFactorialDesign({','.join(names)})",
            fields,
            DESIGN_LEVEL,
            element_spec=EventTemplate(template_spec),
            axes_per_level=(1,),
        )
        # The name is derived from the marginals, not user-typed.
        object.__setattr__(self, "_name_is_auto", True)
        object.__setattr__(self, "_marginals", dict(marginals))
