"""Identity and metadata mixins: ``Tracked`` and ``Annotated``.

Every object a ProbPipe operation returns is a **tracked term**: it carries a
:attr:`~Tracked.name` (what the object is called) and, optionally, a
:attr:`~Tracked.provenance` (how it was produced). Some objects additionally
carry free-form :attr:`~Annotated.annotations` (auxiliary information supplied
by the user or an algorithm). These identity and metadata attributes are orthogonal
to what an object *is* mathematically, so they are defined once, here, as two
mixins:

- :class:`Tracked` — name + provenance. Every ProbPipe value, distribution,
  and batch is ``Tracked``.
- :class:`Annotated` — free-form annotations. Carried by the single value and
  distribution types (``Record``, ``Distribution``), not required of batches.

Classes mix these in alongside their mathematical base (e.g. ``class
Record(_NamedTree, Tracked, Annotated)``) and initialize the identity state in
their constructor via :meth:`Tracked._init_tracked`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping

# ``_ProtocolMeta`` is technically private (leading underscore in
# ``typing``), but it's the only way to compose a custom metaclass with
# ``@runtime_checkable`` protocols without a metaclass conflict.  The
# name has been stable since Python 3.7 and is widely used in the
# ecosystem (Pydantic, attrs, etc.). If a future Python release renames
# it, the metaclass would need to switch to whatever new base ``typing``
# exposes; the conflict-avoidance constraint itself doesn't change.
from typing import Any, Self, _ProtocolMeta

from .provenance import Provenance

__all__ = ["Annotated", "Tracked", "auto_name"]


def auto_name(name: str | None, default: str) -> tuple[str, bool]:
    """Resolve an optional user-supplied name against an auto-derived default.

    The standard idiom for a constructor whose ``name`` may be omitted:
    returns ``(name, False)`` when *name* was supplied (a user-given name)
    and ``(default, True)`` when it was ``None`` (an auto-derived name),
    ready to pass to ``__init__(name=..., name_is_auto=...)`` or
    :meth:`Tracked._init_tracked`.

    Parameters
    ----------
    name : str or None
        The caller-supplied name, or ``None`` to use *default*.
    default : str
        The auto-derived name to fall back on.

    Returns
    -------
    tuple of (str, bool)
        The resolved name and the matching ``name_is_auto`` flag.
    """
    if name is None:
        return default, True
    return name, False


class _TrackedMeta(_ProtocolMeta):
    """Metaclass enforcing that every ``Tracked`` instance has a
    non-empty ``name`` set by the time construction returns.

    The check runs after ``__init__`` so it covers every construction
    path: classes that call ``super().__init__(name=...)``, classes that
    call :meth:`Tracked._init_tracked` directly, and classes that assign
    ``self._name`` themselves. The only failure case is a class that
    finishes ``__init__`` without setting ``_name`` to a non-empty
    string — then construction raises ``TypeError``.

    Extends ``typing._ProtocolMeta`` (rather than the more obvious
    ``ABCMeta``) so ``Tracked`` hosts can mix in ``@runtime_checkable``
    protocols (``SupportsSampling``, ``SupportsLogProb``, …) without a
    metaclass conflict. ``_ProtocolMeta`` is itself an ``ABCMeta``
    subclass.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        name = getattr(instance, "_name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{cls.__name__}.__init__ must set a non-empty name "
                f"(via _init_tracked(name, ...) / super().__init__(name=...) "
                f"or by assigning self._name to a non-empty string) "
                f"before returning."
            )
        return instance


class Tracked(metaclass=_TrackedMeta):
    """Identity mixin: a :attr:`name` and a write-once :attr:`provenance`.

    A ``Tracked`` object carries, alongside its mathematical content, the two
    pieces of identity every ProbPipe object needs: a human-readable **name**
    and an optional **provenance** describing how it was produced. Any such
    object is a *tracked term* — the kind of object ProbPipe operations
    consume and produce.

    The name is either **user-given** or **auto-derived**, recorded by
    :attr:`name_is_auto`: a user constructing an object explicitly supplies
    its name, while an operation that produces an object derives a
    deterministic name from its inputs and marks it auto. The two behave
    differently downstream — an auto-derived name may be re-derived when the
    object is combined into a larger one, while a user-given name is
    preserved. :meth:`with_name` renames the object itself (returning a copy
    marked user-named); this is distinct from ``with_names`` on the named-tree
    types, which renames the *fields within* an object.

    Provenance is **write-once**: it is attached at most once via
    :meth:`with_provenance`, and a subsequent attempt raises. Transformations
    that build a new object attach fresh provenance to the result instead of
    rewriting the input's.

    Attributes
    ----------
    name : str
        Human-readable name of this object.
    name_is_auto : bool
        ``True`` when :attr:`name` was auto-derived by the operation that
        produced this object; ``False`` when it was supplied by the user
        (including via :meth:`with_name`).
    provenance : Provenance or None
        How this object was produced, or ``None`` if no provenance has been
        attached (an original user-constructed object, or provenance tracking
        disabled).

    Notes
    -----
    The mixin holds no per-instance storage of its own (``__slots__ = ()``);
    the state lives in the ``_name`` / ``_name_is_auto`` / ``_provenance``
    attributes, which a host class declares in its ``__slots__`` (when it uses
    slots) and initializes via :meth:`_init_tracked`. All writes go through
    ``object.__setattr__`` so the mixin also works on immutable hosts that
    block normal attribute assignment.

    The non-empty-name guarantee is enforced at construction by the mixin's
    metaclass (:class:`_TrackedMeta`): finishing ``__init__`` without a
    non-empty ``_name`` raises ``TypeError``. Host classes therefore never
    need their own name check.
    """

    __slots__ = ()

    def _init_tracked(
        self,
        name: str,
        *,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        """Initialize the identity state (constructor helper for host classes).

        Assigns ``_name``, ``_name_is_auto``, and ``_provenance`` via
        ``object.__setattr__`` so immutable hosts can call it from their
        constructor. Performs no validation — the host constructor owns its
        own ``name`` policy (required vs. auto-derived default).
        """
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_name_is_auto", bool(name_is_auto))
        object.__setattr__(self, "_provenance", provenance)

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable name of this object."""
        return self._name

    @property
    def name_is_auto(self) -> bool:
        """Whether :attr:`name` was auto-derived rather than user-given.

        ``True`` when the operation that produced this object derived the
        name from its inputs; ``False`` when the user supplied it — at
        construction or via :meth:`with_name`.
        """
        return getattr(self, "_name_is_auto", False)

    def with_name(self, name: str) -> Self:
        """Return a copy of this object under a new user-given name.

        The copy is shallow: it shares its data with the original but has
        ``name`` set to *name* and :attr:`name_is_auto` set to ``False`` (a
        rename is always a user choice). The copy's :attr:`provenance`
        records the rename, with the original as parent, so the lineage
        chain is preserved. On an ``Annotated`` host the annotations
        *container* is copied (its entries are shared), so annotations
        written after the rename land on one object without appearing on
        the other.

        This renames the object *itself*. To rename the named fields inside a
        structured object, use ``with_names`` on the named-tree types.

        Parameters
        ----------
        name : str
            The new name. Must be a non-empty string.

        Returns
        -------
        Self
            A shallow copy with the new name; the original is unchanged.

        Raises
        ------
        TypeError
            If *name* is not a non-empty string.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"{type(self).__name__}.with_name() requires a non-empty string name")
        clone = self._shallow_copy()
        object.__setattr__(clone, "_name", name)
        object.__setattr__(clone, "_name_is_auto", False)
        object.__setattr__(clone, "_provenance", None)
        # Decouple the annotations container: writers add entries in place
        # (the documented append-only channel), and a shared container would
        # let a post-rename write show through on the original. A shallow
        # container copy shares the entry values but not the container.
        annotations = getattr(clone, "_annotations", None)
        if annotations is not None:
            copied = annotations.copy() if hasattr(annotations, "copy") else dict(annotations)
            object.__setattr__(clone, "_annotations", copied)
        clone.with_provenance(
            Provenance.create(
                "with_name",
                parents=[self],
                metadata={"old_name": self.name, "new_name": name},
            )
        )
        return clone

    # -- provenance ----------------------------------------------------------

    @property
    def provenance(self) -> Provenance | None:
        """Provenance describing how this object was produced, or ``None``."""
        return getattr(self, "_provenance", None)

    def with_provenance(self, provenance: Provenance | None) -> Self:
        """Attach provenance to this object (write-once) and return it.

        Passing ``None`` (e.g. the result of :meth:`Provenance.create` when
        provenance tracking is off) is a no-op, so call sites can pass the
        result of ``Provenance.create(...)`` without a guard.

        Parameters
        ----------
        provenance : Provenance or None
            The provenance to attach, or ``None`` for a no-op.

        Returns
        -------
        Self
            This object (not a copy), for call chaining.

        Raises
        ------
        RuntimeError
            If provenance is already set (provenance is write-once).
        """
        if provenance is None:
            return self
        if getattr(self, "_provenance", None) is not None:
            raise RuntimeError(f"Provenance already set on {self!r}. Provenance is write-once.")
        object.__setattr__(self, "_provenance", provenance)
        return self

    def _restore_identity(self, *, name_is_auto: bool, provenance: Provenance | None) -> Self:
        """Restore identity state on a reconstructed object, returning it.

        The shared tail of every unpickle helper: after the constructor has
        rebuilt the object (setting ``_name`` from the stored name), this
        re-applies the stored :attr:`name_is_auto` flag and, when present,
        the stored provenance — bypassing the write-once guard, since the
        reconstruction is restoring recorded state rather than rewriting
        lineage.
        """
        object.__setattr__(self, "_name_is_auto", bool(name_is_auto))
        if provenance is not None:
            object.__setattr__(self, "_provenance", provenance)
        return self

    # -- copying -------------------------------------------------------------

    def _shallow_copy(self) -> Self:
        """Return a shallow copy sharing all internal state.

        Copies the instance ``__dict__`` (when present) and every assigned
        slot across the class hierarchy via ``object.__setattr__``, bypassing
        both ``__init__`` and any immutability guard on ``__setattr__``. Used
        by :meth:`with_name`; host classes with exotic storage may override.
        """
        cls = type(self)
        clone = cls.__new__(cls)
        instance_dict = getattr(self, "__dict__", None)
        if instance_dict is not None:
            clone.__dict__.update(instance_dict)
        seen: set[str] = set()
        for klass in cls.__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if slot in seen or slot in ("__dict__", "__weakref__"):
                    continue
                seen.add(slot)
                # A slot may be declared but never assigned; skip it.
                with contextlib.suppress(AttributeError):
                    object.__setattr__(clone, slot, getattr(self, slot))
        return clone


class Annotated:
    """Metadata mixin: free-form :attr:`annotations`.

    An ``Annotated`` object can carry auxiliary information beyond its
    mathematical content — diagnostic summaries, validation results, backend
    reports, or any other metadata supplied by the user or an algorithm. The
    store is a free-form string-keyed mapping (any ``Mapping[str, Any]``,
    including an ``xarray.DataTree``), or ``None`` when nothing has been
    attached.

    Attributes
    ----------
    annotations : Mapping[str, Any] or None
        The attached annotations, or ``None`` if there are none.

    Notes
    -----
    Annotations are the one documented exception to object immutability: the
    ``_annotations`` store is designed to be written *after* construction by
    inference backends, validators, and diagnostic operations, and mutated
    in place. Treat the channel as append-only — a writer should add its
    results under its own key and never overwrite mathematical state or
    another writer's entries.

    Like :class:`Tracked`, the mixin holds no storage of its own
    (``__slots__ = ()``); the state lives in the ``_annotations`` attribute,
    which a slotted host class declares in its ``__slots__``.
    """

    __slots__ = ()

    @property
    def annotations(self) -> Mapping[str, Any] | None:
        """Annotations attached to this object, or ``None``.

        A free-form string-keyed mapping of auxiliary information. The value
        may be any ``Mapping[str, Any]`` — in particular, an
        ``xarray.DataTree`` (the layout inference backends and the
        diagnostics subsystem use) is a valid value.
        """
        return getattr(self, "_annotations", None)
