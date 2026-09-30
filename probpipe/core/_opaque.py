"""Opaque — the tracked class of the opaque kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    name : str
        The value's name, required and first as a :class:`~probpipe.Record`
        takes it: the name is what says which opaque value this is.
    value : Any
        The value this names, held as given. Any non-mapping value; the value
        layer reads a mapping as a subtree.
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
    >>> fitted = Opaque("sklearn_model", object())
    >>> fitted.name
    'sklearn_model'
    """

    __slots__ = (
        "_annotations",
        "_name",
        "_provenance",
        "_spec",
        "_value",
    )

    def __init__(
        self,
        name: str,
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
        spec = OpaqueSpec() if spec is None else spec
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(name, provenance=provenance)

    @property
    def value(self) -> Any:
        """The wrapped value, untracked."""
        return self._value

    @property
    def spec(self) -> OpaqueSpec:
        """This value's own declaration."""
        return self._spec

    def __repr__(self) -> str:
        return f"Opaque({self.name!r}, {self._value!r})"
