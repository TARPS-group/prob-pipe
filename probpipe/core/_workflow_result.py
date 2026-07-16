"""WorkflowFunction result-contract helpers.

This private module owns the boundary rule that raw workflow returns
become ``Record | RecordArray | Distribution`` values and receive
provenance when appropriate.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Literal

import jax.numpy as jnp

from ._broadcast_distributions import _make_stack
from ._distribution_base import Distribution
from ._numeric_record import _is_numeric_leaf
from .provenance import Provenance
from .record import Record

# Broadcast modes: how a value reached ``_coerce_output``. Named
# constants so callsites use the same spelling and typos fail loudly.
# ``BROADCAST_MARGINALISE`` is intentionally absent — the
# Distribution-only path goes through the distribution-broadcast layer
# and doesn't call ``_coerce_output`` at all; its marginal already
# carries provenance.
BroadcastMode = Literal["wrap", "stack", "nested"]
BROADCAST_WRAP: BroadcastMode = "wrap"
BROADCAST_STACK: BroadcastMode = "stack"
BROADCAST_NESTED: BroadcastMode = "nested"


def _wrap_as_record(value: Any, field_name: str) -> Any:
    """Coerce a raw return into the Record | RecordArray | Distribution contract.

    Uniform rule applied at the WorkflowFunction boundary:

    - Already-structured values (``Record`` / ``RecordArray`` /
      ``Distribution``) pass through unchanged — their domain field
      names are preserved.
    - ``dict`` (non-empty) → a ``Record`` keyed by the caller's keys; a
      nested ``dict`` value denotes tree structure and becomes a nested
      subtree (mappings are never leaves), not a single opaque field.
    - Non-empty ``list`` / ``tuple`` → ``_make_stack``: assembles a
      ``DistributionArray`` / ``RecordArray`` / ``NumericRecordArray``
      matching the inner element type.
    - Scalar numeric / ``jnp.ndarray`` → a single-field ``NumericRecord``
      named and keyed by the function's own name. Raw
      numeric access round-trips via the single-field shim:
      ``float(x)``, ``jnp.array(x)``, and (for callable-valued fields)
      ``x(args)`` call-forwarding.
    - Anything else (opaque Python object) → a single-field plain
      ``Record`` named and keyed by the function's own name.
    """
    if isinstance(value, (Distribution, Record)):
        return value
    if isinstance(value, dict) and value:
        return Record(field_name, value, name_is_auto=True)
    if isinstance(value, (list, tuple)) and value:
        try:
            return _make_stack(list(value), n=len(value), field_name=field_name)
        except (TypeError, ValueError):
            pass
    # Numeric scalar / array → NumericRecord with the function's
    # name as the single field; the wrap adds no batch_shape of its
    # own (batching comes from sweeps). ``_is_numeric_leaf`` excludes
    # opaque duck-typed objects (``unittest.mock.MagicMock`` etc.)
    # whose attribute probing would recurse inside ``jnp.asarray``.
    if _is_numeric_leaf(value):
        return Record(field_name, {field_name: jnp.asarray(value)}, name_is_auto=True)
    return Record(field_name, {field_name: value}, name_is_auto=True)


def _coerce_output(
    value: Any,
    *,
    broadcast_mode: BroadcastMode,
    provenance: Provenance | None,
    field_name: str,
) -> Any:
    """Enforce the Record | RecordArray | Distribution output contract.

    Parameters
    ----------
    value
        The raw output produced by the function body or a broadcast
        aggregator. For ``broadcast_mode != "wrap"`` this is always
        already one of the three contract types.
    broadcast_mode : {"wrap", "stack", "nested"}
        How the value was produced:

        * ``"wrap"`` — non-broadcast call; ``value`` is whatever the
          user's function returned. Scalars / arrays become
          a single-field record named after the function; dict / list / tuple
          promote via ``_wrap_as_record``; existing Record /
          RecordArray / Distribution values pass through.
        * ``"stack"`` — array-valued broadcast; ``value`` is a stacked
          aggregate from ``_make_stack`` (``NumericRecordArray`` /
          ``RecordArray`` / ``DistributionArray``).
        * ``"nested"`` — array + Distribution broadcast; ``value`` is
          a ``DistributionArray`` of per-row marginals.
    provenance : Provenance or None
        Provenance node to attach. ``None`` skips the attachment step.
    field_name : str
        Name used when wrapping bare scalar / array returns — always
        the WorkflowFunction's own name so the single-field record
        maps back to the op that produced it.

    Returns
    -------
    Record | RecordArray | Distribution
        The value, possibly wrapped, with ``.provenance`` attached when it
        was empty. An already-sourced value keeps its existing source
        (inner marginals produced by the broadcast layer carry their
        own provenance; ``_coerce_output`` doesn't overwrite).
    """
    if broadcast_mode == BROADCAST_WRAP:
        value = _wrap_as_record(value, field_name)
    if provenance is not None and hasattr(value, "with_provenance"):
        # Existing source, e.g. an inner marginal, keeps its provenance.
        with suppress(RuntimeError):
            value.with_provenance(provenance)
    return value
