"""Identity and metadata mixins: ``TrackedTerm`` and ``Annotated``.

Every object a ProbPipe operation returns is a **tracked term**: it carries a
:attr:`~TrackedTerm.name` (what the object is called) and, optionally, a
:attr:`~TrackedTerm.provenance` (how it was produced). Some objects additionally
carry free-form :attr:`~Annotated.annotations` (auxiliary information supplied
by the user or an algorithm). These identity and metadata attributes are orthogonal
to what an object *is* mathematically, so they are defined once, here, as two
mixins:

- :class:`TrackedTerm` — name + provenance. Every ProbPipe value, distribution,
  and batch is ``TrackedTerm``.
- :class:`Annotated` — free-form annotations. Carried by the single value and
  distribution types (``Record``, ``Distribution``), not required of batches.

Classes mix these in alongside their mathematical base (e.g. ``class
Record(NamedTree, TrackedTerm, Annotated)``) and initialize the identity state in
their constructor via :meth:`TrackedTerm._init_tracked`.
"""

from __future__ import annotations

from collections.abc import Mapping

# ``_ProtocolMeta`` is technically private (leading underscore in
# ``typing``), but it's the only way to compose a custom metaclass with
# ``@runtime_checkable`` protocols without a metaclass conflict.  The
# name has been stable since Python 3.7 and is widely used in the
# ecosystem (Pydantic, attrs, etc.). If a future Python release renames
# it, the metaclass would need to switch to whatever new base ``typing``
# exposes; the conflict-avoidance constraint itself doesn't change.
from typing import Any, Self, _ProtocolMeta

from ._immutable import Immutable, constructing, decoupled_container
from .provenance import Provenance

__all__ = ["Annotated", "TrackedTerm", "auto_name"]


def auto_name(name: str | None, default: str) -> tuple[str, bool]:
    """Resolve an optional user-supplied name against an auto-derived default.

    The standard idiom for a constructor whose ``name`` may be omitted:
    returns ``(name, False)`` when *name* was supplied (a user-given name)
    and ``(default, True)`` when it was ``None`` (an auto-derived name),
    ready to pass to ``__init__(name=..., name_is_auto=...)`` or
    :meth:`TrackedTerm._init_tracked`.

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


def _decoupled_annotations(annotations: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a shallow copy of an annotations container.

    Entries are shared, the container is not, so a write on one object does not
    show through on the object it was copied or rebuilt from — the annotations
    channel is written in place (see :class:`Annotated`), which is what makes a
    shared container observable. The rule itself lives with the state round-trip
    that also applies it (:func:`~probpipe.core._immutable.decoupled_container`),
    so a rename and a reconstruction decouple the same way.
    """
    return decoupled_container(annotations)


class _TrackedTermMeta(_ProtocolMeta):
    """Metaclass running construction in a window, and enforcing a non-empty name.

    A tracked term is immutable, so it can only be built by assigning to it
    before anyone holds it. This runs ``__init__`` inside
    :func:`~probpipe.core._immutable.constructing`, which is what lets a
    constructor assign normally while assignment anywhere else raises. The window
    is per instance and per thread, and closes even when ``__init__`` raises.

    The check runs after ``__init__`` so it covers every construction
    path: classes that call ``super().__init__(name=...)``, classes that
    call :meth:`TrackedTerm._init_tracked` directly, and classes that assign
    ``self._name`` themselves. The only failure case is a class that
    finishes ``__init__`` without setting ``_name`` to a non-empty
    string — then construction raises ``TypeError``.

    Extends ``typing._ProtocolMeta`` (rather than the more obvious
    ``ABCMeta``) so ``TrackedTerm`` hosts can mix in ``@runtime_checkable``
    protocols (``SupportsSampling``, ``SupportsLogProb``, …) without a
    metaclass conflict. ``_ProtocolMeta`` is itself an ``ABCMeta``
    subclass.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # ``type.__call__``, split so the window can be opened between the two
        # halves: allocation first, then ``__init__`` inside it. A host's
        # ``__new__`` may resolve to a subclass, and as ``type.__call__`` does,
        # ``__init__`` runs only when it returns an instance of *cls*.
        if cls.__new__ is object.__new__:
            instance = cls.__new__(cls)
        else:
            instance = cls.__new__(cls, *args, **kwargs)
        if isinstance(instance, cls):
            with constructing(instance):
                returned = instance.__init__(*args, **kwargs)
            if returned is not None:
                raise TypeError(f"__init__() should return None, not {type(returned).__name__!r}")
        name = getattr(instance, "_name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"{cls.__name__}.__init__ must set a non-empty name "
                f"(via _init_tracked(name, ...) / super().__init__(name=...) "
                f"or by assigning self._name to a non-empty string) "
                f"before returning."
            )
        return instance


class TrackedTerm(Immutable, metaclass=_TrackedTermMeta):
    """Identity mixin: a :attr:`name` and a write-once :attr:`provenance`.

    A ``TrackedTerm`` object carries, alongside its mathematical content, the two
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
    marked user-named); this is distinct from ``with_path_names`` on the named-tree
    types, which renames the *fields within* an object.

    Provenance is **write-once**: it is attached at most once via
    :meth:`with_provenance`, and a subsequent attempt raises. Transformations
    that build a new object attach fresh provenance to the result instead of
    rewriting the input's.

    A tracked term is **immutable**
    (:class:`~probpipe.core._immutable.Immutable`): assignment and deletion raise
    once its constructor has returned, and an operation that changes anything
    returns a new term. Construction assigns inside the window the metaclass
    opens, so a host's ``__init__`` is written normally. The distribution layer
    is exempt for now, for the reason its ``__setattr__`` gives.

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
    metaclass (:class:`_TrackedTermMeta`): finishing ``__init__`` without a
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
        *container* is its own (its entries are shared), so annotations
        written after the rename land on one object without appearing on the
        other — :meth:`_shallow_copy` does that, from the host's own
        ``_decoupled_state`` declaration.

        This renames the object *itself*. To rename the named fields inside a
        structured object, use ``with_path_names`` on the named-tree types.

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

    # -- copying -------------------------------------------------------------

    def _shallow_copy(self) -> Self:
        """Return a shallow copy sharing all internal state.

        Copies the instance ``__dict__`` (when present) and every assigned
        slot across the class hierarchy via ``object.__setattr__``, bypassing
        both ``__init__`` and any immutability guard on ``__setattr__``. Used
        by :meth:`with_name`; host classes with exotic storage may override.

        Allocation uses ``object.__new__`` directly: ``type(self)`` is
        already the resolved concrete class, so a host's own ``__new__`` —
        which exists to *select* a class from constructor arguments and may
        require them — must not run again here.

        The copy goes through the same state round-trip as ``copy`` and
        ``pickle``, so a rename honours what a class declares about its state: a
        memo is not carried into the copy, and a store written in place is not
        shared with it. All three copy paths therefore agree.
        """
        clone = object.__new__(type(self))
        clone.__setstate__(self.__getstate__())
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

    Like :class:`TrackedTerm`, the mixin holds no storage of its own
    (``__slots__ = ()``); the state lives in the ``_annotations`` attribute,
    which a slotted host class declares in its ``__slots__``.
    """

    __slots__ = ()

    #: The annotations store is written in place, so a copy or a reconstruction
    #: takes its own container (see ``Immutable._decoupled_state``).
    _decoupled_state = ("_annotations",)

    def _init_annotations(self, annotations: Mapping[str, Any] | None) -> None:
        """Attach a starting annotations store (constructor helper for hosts).

        The counterpart to :meth:`TrackedTerm._init_tracked`, for the one case a
        host is handed annotations up front rather than having them written into
        it later: reconstructing a term that already carried some. ``None``
        leaves the store unset, which is what an ordinary construction passes,
        so :attr:`annotations` stays ``None`` until a writer attaches something.

        The container is decoupled from the one passed in, for the reason
        :meth:`TrackedTerm.with_name` gives: writers add entries in place, so a
        shared container would let a write on this object show through on
        whatever it was built from. Writes go through ``object.__setattr__`` so
        an immutable host can call this from its constructor.
        """
        if annotations is not None:
            object.__setattr__(self, "_annotations", _decoupled_annotations(annotations))

    @property
    def annotations(self) -> Mapping[str, Any] | None:
        """Annotations attached to this object, or ``None``.

        A free-form string-keyed mapping of auxiliary information. The value
        may be any ``Mapping[str, Any]`` — in particular, an
        ``xarray.DataTree`` (the layout inference backends and the
        diagnostics subsystem use) is a valid value.
        """
        return getattr(self, "_annotations", None)
