"""The output boundary: what a Function call hands back.

This private module owns one rule. A raw return becomes the tracked term of the
kind it already is — a numeric value a ``NumericArray``, a mapping a ``Record``,
a callable a ``Function``, and anything else an ``Opaque`` — named for the
function that produced it. A term the body already produced is returned as it
is, every kind alike. Provenance is attached here.

See design V.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

from ._broadcast_distributions import _make_stack
from ._function_contract import _wrap_declared_function_output
from ._numeric_record import _is_numeric_leaf
from ._record_batch import RecordBatch
from .event_template import EventTemplate, _to_record_declaration
from .provenance import Provenance
from .record import Record
from .tracked import TrackedTerm

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


def _wrap_as_term(
    value: Any,
    field_name: str,
    output_template: EventTemplate | None = None,
) -> Any:
    """Wrap a raw return as the tracked term of its own kind.

    A tracked term is returned as it is, every kind alike (design V.0). A raw
    host takes the tracked class of its own kind.

    The kinds, most specific first, each named after the function that produced
    it:

    - ``dict`` (non-empty) → a ``Record`` keyed by the caller's keys, a nested
      ``dict`` becoming a subtree since a mapping is a tree rather than a leaf.
    - non-empty ``list`` / ``tuple`` → ``_make_stack``, which assembles the
      batch matching the inner element type.
    - numeric scalar or array → a ``NumericArray``.
    - a callable → a ``Function``.
    - anything else → an ``Opaque``.

    They are ordered rather than disjoint: a callable is also a non-mapping
    value, and ``Opaque`` is the fallback.
    """
    if output_template is not None:
        return _wrap_declared_function_output(
            value,
            function_name=field_name,
            output_template=output_template,
        )

    # -- already a term ----------------------------------------------------
    if isinstance(value, TrackedTerm):
        return value

    # -- a raw host, wrapped into its own kind -----------------------------
    # The kind follows the host's *type*, empty or not: a mapping is a tree and a
    # sequence a multiplicity, and having no entries does not change which.
    if isinstance(value, Mapping):
        # Any mapping, not only ``dict``: the value layer reads a mapping as a
        # tree, and an ``OrderedDict`` or a ``Mapping`` subclass is one. Falling
        # through would reach ``Opaque``, which refuses mappings, so the return
        # would raise rather than be wrapped.
        return Record(field_name, dict(value), name_is_auto=True)
    if isinstance(value, (list, tuple)):
        if not value:
            # No element to read a kind off, and every element spec holds
            # vacuously of none, so the batch makes the least specific claim it
            # can. Its own kind is still a batch, which is what the host says.
            from ._opaque_batch import OpaqueBatch

            return OpaqueBatch([], field_name, name=field_name, name_is_auto=True)
        # A returned sequence ranges over nothing the call named, so the level
        # takes the function's own name. Errors are not caught here: the stack
        # has a batch form for every element kind, so what reaches this and
        # raises is the rows disagreeing — which is the caller's to see, not
        # something to record as one opaque value.
        return _make_stack(
            list(value), n=len(value), level_names=(field_name,), field_name=field_name
        )
    # ``_is_numeric_leaf`` excludes duck-typed objects (``MagicMock`` and the
    # like) whose attribute probing recurses inside ``jnp.asarray``.
    if _is_numeric_leaf(value):
        from ._numeric_array import NumericArray

        return NumericArray(value, name=field_name, name_is_auto=True)
    if callable(value):
        from .node import Function

        return Function(func=value, name=field_name, name_is_auto=True)
    from ._opaque import Opaque

    return Opaque(field_name, value, name_is_auto=True)


def _coerce_output(
    value: Any,
    *,
    broadcast_mode: BroadcastMode,
    provenance: Provenance | None,
    field_name: str,
    output_template: EventTemplate | None = None,
) -> Any:
    """Enforce the record / batch / distribution output contract.

    Parameters
    ----------
    value
        The raw output produced by the function body or a broadcast
        aggregator. For ``broadcast_mode != "wrap"`` this is always
        already one of the three contract types.
    broadcast_mode : {"wrap", "stack", "nested"}
        How the value was produced:

        * ``"wrap"`` — non-broadcast call; ``value`` is whatever the
          user's function returned. ``_wrap_as_term`` gives it the tracked
          class of its own kind, and a term it already is becomes an
          independent shallow result copy.
        * ``"stack"`` — array-valued broadcast; ``value`` is a stacked
          aggregate from ``_make_stack`` (``NumericRecordBatch`` /
          ``RecordBatch`` / ``DistributionArray``).
        * ``"nested"`` — array + Distribution broadcast; ``value`` is
          a ``DistributionArray`` of per-row marginals.
    provenance : Provenance or None
        Provenance node to attach. ``None`` skips the attachment step.
    field_name : str
        Name used when wrapping bare scalar / array returns — always
        the Function's own name so the single-field record
        maps back to the op that produced it.

    Returns
    -------
    TrackedTerm
        The value as the tracked term of its kind, possibly wrapped or
        shallow-copied, with the current
        call's ``.provenance`` attached. A copied result does not retain the
        implementation-returned object's prior provenance.
    """
    if broadcast_mode == BROADCAST_WRAP:
        raw_value = value
        value = _wrap_as_term(value, field_name, output_template)
        # A term the body produced is returned as it is, so it arrives here
        # unchanged and becomes an independent result rather than the caller's
        # own object carrying this call's provenance.
        if value is raw_value and isinstance(value, TrackedTerm):
            value = _copy_result_term(value, output_template=output_template)
    elif isinstance(value, TrackedTerm) and value.provenance is not None:
        value = _copy_result_term(value)
    if provenance is not None and isinstance(value, TrackedTerm):
        value.with_provenance(provenance)
    return value


def _copy_result_term(
    value: TrackedTerm,
    *,
    output_template: EventTemplate | None = None,
) -> TrackedTerm:
    """Copy a retained tracked container into an independent result term."""
    clone = value._shallow_copy()
    if output_template is not None:
        if isinstance(clone, RecordBatch):
            element_spec = _to_record_declaration(output_template)
            object.__setattr__(clone, "_spec", replace(clone.spec, element_spec=element_spec))
            # The columns are reordered to match: a batch flattens its columns in
            # its spec's leaf order, so a declared template that orders the same
            # fields differently would otherwise pair every value with the wrong
            # key on the next unflatten.
            columns = clone._raw_columns()
            object.__setattr__(clone, "_columns", {p: columns[p] for p in output_template})
        elif isinstance(clone, Record):
            object.__setattr__(clone, "_spec", _to_record_declaration(output_template))
    object.__setattr__(clone, "_provenance", None)
    # The annotations container is already the clone's own: ``_shallow_copy``
    # decouples what the host declares in ``_decoupled_state``.
    return clone
