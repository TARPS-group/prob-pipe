"""Opaque — the tracked class of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from .._messages import label_given_first
from ._array_backend import _read_only
from ._repr import term_repr, type_name
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
    value : Any
        The value this term holds, stored as given. Any non-mapping value; the value
        layer reads a mapping as a subtree. A NumPy array is marked read-only in
        place.
    label : str
        The required semantic description, passed by keyword: an opaque value
        has no fields or callable name from which to derive one.
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
    >>> fitted = Opaque(
    ...     object(),
    ...     label="sklearn_model",
    ... )
    >>> fitted.label
    'sklearn_model'
    """

    __slots__ = (
        "_annotations",
        "_expression",
        "_label",
        "_label_collapse",
        "_provenance",
        "_spec",
        "_value",
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # A string is a valid opaque value, so only a second positional argument
        # without a label keyword marks it as a label passed in the earlier
        # label-first form.
        if len(args) > 1 and isinstance(args[0], str) and "label" not in kwargs:
            raise TypeError(label_given_first(cls.__name__, "value", args[0]))
        return object.__new__(cls)

    def __init__(
        self,
        value: Any,
        /,
        *,
        label: str,
        spec: OpaqueSpec | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        if spec is not None and not isinstance(spec, OpaqueSpec):
            raise TypeError(f"Opaque spec must be an OpaqueSpec, got {type(spec).__name__}")
        if isinstance(value, Mapping):
            raise TypeError(
                f"Opaque cannot hold a mapping, got {type_name(value)}; use a Record for "
                f"structured values"
            )
        if spec is None:
            spec = OpaqueSpec(type=type(value))
        elif not spec.is_valid(value):
            raise TypeError(
                f"Opaque {label!r}: value of type {type_name(value)} does not match {spec!r}"
            )
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
        return term_repr("Opaque", self._displayed_label(), self._spec._repr_arguments())

    def __str__(self) -> str:
        """The wrapped value's string, as ``print`` and an f-string show the value."""
        return str(self._value)

    def __format__(self, format_spec: str) -> str:
        """The wrapped value formatted by *format_spec*."""
        return format(self._value, format_spec)
