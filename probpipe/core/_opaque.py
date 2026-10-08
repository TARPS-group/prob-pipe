"""Opaque — the tracked class of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._array_backend import _read_only
from ._repr import term_repr
from ._spec_base import OpaqueSpec
from .provenance import Provenance
from .tracked import Annotated, TrackedTerm

__all__ = ["Opaque", "OpaqueSpec"]


class Opaque(TrackedTerm, Annotated):
    """One value of no exposed structure, with identity.

    The tracked class of the opaque kind, as :class:`~probpipe.Record` is of the
    record kind: what an operation returns when its declared kind is an
    :class:`~probpipe.OpaqueSpec`. Its batch form is
    :class:`~probpipe.OpaqueBatch`.

    Its interface is :attr:`value` plus the identity every tracked term carries.
    What the value affords is its own type's, once it is out.

    Parameters
    ----------
    label : str
        The value's label, required and first as a :class:`~probpipe.Record`
        takes it: the label is what says which opaque value this is.
    value : Any
        The value this term holds, stored as given. Any non-mapping value; the value
        layer reads a mapping as a subtree. A NumPy array is marked read-only in
        place.
    spec : OpaqueSpec, optional
        What this value satisfies, carrying any opaque ``meta``. Defaults to the
        :class:`~probpipe.OpaqueSpec` of the value's type.
    provenance : Provenance, optional
        How this value was produced.

    Raises
    ------
    TypeError
        If *spec* is not an :class:`~probpipe.OpaqueSpec`, or if *value* is a
        mapping.

    Examples
    --------
    >>> fitted = Opaque("sklearn_model", object())
    >>> fitted.label
    'sklearn_model'
    """

    __slots__ = (
        "_annotations",
        "_label",
        "_provenance",
        "_spec",
        "_value",
    )

    def __init__(
        self,
        label: str,
        value: Any,
        /,
        *,
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
        if spec is None:
            spec = OpaqueSpec(type=type(value))
        elif not spec.is_valid(value):
            raise TypeError(f"{spec!r} does not admit a {type(value).__name__}")
        object.__setattr__(self, "_value", _read_only(value))
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(label, provenance=provenance)

    @classmethod
    def _view(
        cls, label: str, value: Any, spec: OpaqueSpec, provenance: Provenance | None
    ) -> Opaque:
        """The opaque *value* under *spec*, as a container's view of it, without validation.

        The container validated the value against *spec* when it was built.
        """
        view = object.__new__(cls)
        object.__setattr__(view, "_value", value)
        object.__setattr__(view, "_spec", spec)
        view._init_tracked(label, provenance=provenance)
        return view

    @property
    def value(self) -> Any:
        """The wrapped value, untracked."""
        return self._value

    def raw(self) -> Any:
        """The wrapped value, as :attr:`value` holds it."""
        return self._value

    @property
    def spec(self) -> OpaqueSpec:
        """This value's own declaration."""
        return self._spec

    def __repr__(self) -> str:
        """The label, then the declared type and the metadata where the spec sets them."""
        return term_repr("Opaque", self.label, self._spec._repr_arguments())

    def __str__(self) -> str:
        """The wrapped value's string, as ``print`` and an f-string show the value."""
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        """The wrapped value formatted by *format_spec*."""
        return format(self._value, format_spec)
