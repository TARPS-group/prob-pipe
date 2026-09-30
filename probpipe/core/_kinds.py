"""The kind table: which tracked class and which batch form each value spec has.

Every value spec has exactly one tracked class and one batch form (design II.2,
III.1). That correspondence was open-coded at six sites, each knowing a
different subset, so widening one of them left the others behind — a
``NumericArrayBatch`` that the record layer could present but the workflow
planner would not sweep.

The table is declared once here and each kind registers itself at the bottom of
its own module, next to the classes it names. Registration is import-order
sensitive by construction, which is why ``probpipe/__init__.py`` imports the
batch modules eagerly: a lookup before a kind's module has been imported would
see a table that does not yet mention it.

A spec with no registered batch form is not an oversight — a ``RecordSpec``
belongs to :class:`~probpipe.RecordBatch`, whose choice between the plain and
numeric class depends on the *template's* leaves rather than on the spec type
alone, so the record layer keeps its own resolver.

See design II.2, III.1.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "batch_class_for_spec",
    "register_kind",
    "term_class_for_spec",
]

#: spec type -> (tracked class, batch class). Either may be ``None`` where the
#: kind has no class of that side.
_KINDS: dict[type, tuple[type | None, type | None]] = {}


def register_kind(
    spec_type: type, *, term_class: type | None = None, batch_class: type | None = None
) -> None:
    """Declare the tracked class and batch form belonging to *spec_type*.

    Called at the bottom of the module defining the classes, so the table is
    populated by importing the kind rather than by a central list that a new
    kind has to be added to.

    Raises
    ------
    ValueError
        If *spec_type* is already registered with a different pair. A kind has
        one tracked class and one batch form; two registrations disagreeing is a
        contradiction rather than an override.
    """
    existing = _KINDS.get(spec_type)
    pair = (term_class, batch_class)
    if existing is not None and existing != pair:
        raise ValueError(
            f"{spec_type.__name__} is already registered as {existing}, so registering "
            f"{pair} would give one spec two kinds; a spec has one tracked class and one "
            f"batch form"
        )
    _KINDS[spec_type] = pair


def term_class_for_spec(spec: Any) -> type | None:
    """The tracked class a value satisfying *spec* is returned as, if any."""
    return _lookup(spec, 0)


def batch_class_for_spec(spec: Any) -> type | None:
    """The batch form a collection of values satisfying *spec* takes, if any."""
    return _lookup(spec, 1)


def _lookup(spec: Any, side: int) -> type | None:
    """Resolve *spec*'s registered class, honouring subclass registrations.

    The exact type first, then the MRO, so a spec subclass inherits its base's
    kind unless it registers its own.
    """
    for candidate in type(spec).__mro__:
        entry = _KINDS.get(candidate)
        if entry is not None and entry[side] is not None:
            return entry[side]
    return None
