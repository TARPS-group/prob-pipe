"""Opaque — the tracked class of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._immutable import Immutable
from .event_template import OpaqueSpec
from .provenance import Provenance
from .tracked import Annotated, TrackedTerm, auto_name

__all__ = ["Opaque"]


class Opaque(Immutable, TrackedTerm, Annotated):
    """One value of no exposed structure, with identity.

    The tracked class of the opaque kind, as :class:`~probpipe.Record` is of the
    record kind: what an operation returns when its declared kind is an
    :class:`~probpipe.OpaqueSpec`. Its batch form is
    :class:`~probpipe.OpaqueBatch`, which already existed — a batch is a tracked
    term whatever its elements are, which is why the collection had a class
    before the element did.

    **It adds identity and nothing else.** No attribute forwarding, no
    ``__call__``, no operators: the wrapped value is reached through
    :attr:`value`, explicitly. A box that forwards to be convenient is
    indistinguishable from the value it wraps, and then the wrapping is a
    surprise rather than a statement — so this one does not. What an opaque
    value affords is whatever its own type affords, once it is out.

    Parameters
    ----------
    value : Any
        The value this names. Held as given, never converted or copied — there
        is nothing to convert it to. Anything but a mapping, which the value
        layer reads as a subtree rather than a leaf.
    name : str, optional
        The value's name. Defaults to ``"opaque"``, marked auto-derived.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given. A value left
        unnamed is auto-named regardless.
    spec : OpaqueSpec, optional
        What this value satisfies, carrying any opaque ``meta``. Defaults to a
        bare :class:`~probpipe.OpaqueSpec`.
    provenance : Provenance, optional
        How this value was produced.

    Raises
    ------
    TypeError
        If *spec* is not an :class:`~probpipe.OpaqueSpec`, or if *value* is a
        mapping.

    Examples
    --------
    >>> fitted = Opaque(object(), name="sklearn_model")
    >>> fitted.name
    'sklearn_model'
    """

    __slots__ = (
        "_annotations",
        "_name",
        "_name_is_auto",
        "_provenance",
        "_spec",
        "_value",
    )

    def __init__(
        self,
        value: Any,
        *,
        name: str | None = None,
        name_is_auto: bool = False,
        spec: OpaqueSpec | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if spec is not None and not isinstance(spec, OpaqueSpec):
            raise TypeError(f"Opaque spec must be an OpaqueSpec, got {type(spec).__name__}")
        if isinstance(value, Mapping):
            raise TypeError(
                "Opaque holds one unstructured value, and the value layer reads a mapping as a "
                "subtree rather than a leaf; wrap it as a Record, or as a non-mapping value"
            )
        spec = OpaqueSpec() if spec is None else spec
        if name is None:
            name, name_is_auto = auto_name(name, "opaque")
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(name, name_is_auto=name_is_auto, provenance=provenance)

    @property
    def value(self) -> Any:
        """The wrapped value, untracked. The one way to reach it."""
        return self._value

    @property
    def spec(self) -> OpaqueSpec:
        """This value's own declaration."""
        return self._spec

    def __repr__(self) -> str:
        return f"Opaque({self._value!r}, name={self.name!r})"
