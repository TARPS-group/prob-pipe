"""Object immutability: the assignment guard and the state round-trip.

Provides :class:`Immutable`, the mixin a class inherits to declare that its
instances cannot be modified after construction. It supplies the assignment
guard and, because an object that refuses assignment cannot be restored the way
``pickle`` and ``copy`` restore an ordinary one, the state round-trip those need.

:class:`~probpipe.core.tracked.TrackedTerm` inherits it, so a term is immutable by
being one (``C2 – Functional interface over immutable objects``);
:class:`~probpipe.core.event_template.EventTemplate` mixes it in directly, being
immutable without being a term.

One layer is exempt for now:
:class:`~probpipe.core._distribution_base.Distribution` permits assignment, since
the documented way to build an emulator is to subclass a random function and
train it in place, and fitting has no contract yet that returns a new term
instead. Deleting that one method turns the guard on for the layer.

A constructor still has to write, so construction runs inside
:func:`constructing`, a per-instance window in which assignment is allowed.
``TrackedTerm``'s metaclass opens it around every construction; code that
allocates with ``object.__new__`` and calls a constructor by hand opens it
itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

__all__ = ["Immutable", "constructing", "declared_state_names"]


def declared_state_names(cls: type, declaration: str) -> frozenset[str]:
    """The union of a state declaration over every class in *cls*'s MRO.

    Parameters
    ----------
    cls : type
        The class whose hierarchy is read.
    declaration : str
        ``"_transient_state"`` or ``"_decoupled_state"``.

    Returns
    -------
    frozenset of str
        Every attribute name any class in the hierarchy declares, so a subclass
        adds to its base's entries rather than restating them.

    Notes
    -----
    At module level rather than on :class:`Immutable` because the declarations
    describe how a class *copies*, which holds whether or not it also refuses
    assignment: :meth:`__getstate__` reads them, and so may any other copy path.
    """
    return frozenset(name for klass in cls.__mro__ for name in klass.__dict__.get(declaration, ()))


# Instances whose constructor is running on this thread, keyed by ``id``. The
# instance is the value, not just the key: holding it keeps the id from being
# reused by another object while the constructor is still on the stack.
_under_construction = threading.local()


def _constructing_now() -> dict[int, Any]:
    registry = getattr(_under_construction, "registry", None)
    if registry is None:
        registry = {}
        _under_construction.registry = registry
    return registry


@contextmanager
def constructing(instance: Any) -> Iterator[Any]:
    """Allow attribute assignment on *instance* for the duration of the block.

    The window construction runs in: an immutable object is built by assigning to
    it, and it belongs to nobody until its constructor returns. Opened per
    instance and per thread, and closed even if the constructor raises, so a
    half-built object left behind by a failure is as immutable as a finished one.

    :class:`~probpipe.core.tracked.TrackedTerm`'s metaclass opens this around
    every construction, so a term's ``__init__`` assigns normally. Code that
    allocates with ``object.__new__`` and then calls a constructor by hand has to
    open it itself.
    """
    registry = _constructing_now()
    key = id(instance)
    registry[key] = instance
    try:
        yield instance
    finally:
        registry.pop(key, None)


def decoupled_container(container: Any) -> Any:
    """Return a shallow copy of *container*: entries shared, container not.

    Parameters
    ----------
    container : Mapping
        The store to copy. Any mapping type is accepted — an ``xarray.DataTree``
        and the like copy themselves, anything else is rebuilt as a ``dict``.

    Returns
    -------
    Mapping
        A copy holding the same entries, which writes to either container do not
        share.
    """
    return container.copy() if hasattr(container, "copy") else dict(container)


class Immutable:
    """Mixin declaring that instances cannot be modified after construction.

    Assignment and deletion raise :exc:`AttributeError`, naming the class of the
    object that was touched — except inside :func:`constructing`, the window an
    object is built in, which a host's constructor may assign freely within.
    Writing through ``object.__setattr__`` also goes around the guard, and is
    what code outside a constructor uses when it has to.

    ``copy.copy``, ``copy.deepcopy``, and ``pickle`` return an object of the same
    class holding the same state: every attribute the original has assigned,
    whether it lives in a slot declared anywhere in the class hierarchy or in an
    instance dictionary. Reconstruction restores that state directly rather than
    calling the class's constructor, so nothing is re-derived from arguments the
    state no longer carries. Attributes named in :attr:`_transient_state` are
    left out and unset on the copy; those named in :attr:`_decoupled_state` are
    restored into a container of their own.

    ``pickle`` additionally requires the object's class to be **importable by
    name**, since that is what a pickle stores. A class built at runtime — the
    per-capability subclasses some distribution families generate — is not, and
    pickling an instance of one raises :exc:`pickle.PicklingError`. This is a
    property of the class, not of this mixin: the default protocol names the
    class too, so such an object was already unpicklable. ``copy`` and
    ``deepcopy`` hold the class object itself rather than its name and are
    unaffected; ``cloudpickle``, which serializes a class by value, also handles
    them.

    Attributes
    ----------
    _transient_state : tuple of str
        Names of memos to leave out of the round-trip, rebuilt on demand by the
        host. Read as the union over the class hierarchy, so a subclass adds to
        its base's entries rather than restating them.
    _decoupled_state : tuple of str
        Names of stores that are written in place, so a copy takes its own
        container. Same union rule.

    Raises
    ------
    AttributeError
        From any attribute assignment or deletion on an instance.

    Notes
    -----
    The round-trip is defined here rather than per class because the three ways
    to get it wrong are all silent. What to save comes from
    :meth:`object.__getstate__`, which reports every assigned slot and an
    instance dictionary if there is one: a walk over ``__slots__`` by hand misses
    a subclass's storage, and iterates a bare-string ``__slots__`` into
    characters rather than into the one name it declares. A hand-written list of
    state fields additionally misses whatever is assigned *after* construction,
    which no constructor argument names. A missing attribute is
    indistinguishable from an unassigned slot, so each mistake loses data without
    raising.
    """

    __slots__ = ()

    _transient_state: ClassVar[tuple[str, ...]] = ()
    _decoupled_state: ClassVar[tuple[str, ...]] = ()

    # -- The guard -----------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if id(self) in _constructing_now():
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    # -- The state round-trip ------------------------------------------------

    @classmethod
    def _declared_state_names(cls, declaration: str) -> frozenset[str]:
        """The union of the *declaration* tuple over every class in the MRO."""
        return frozenset(
            name for klass in cls.__mro__ for name in klass.__dict__.get(declaration, ())
        )

    def __getstate__(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return this object's state, minus the attributes declared transient.

        Returns
        -------
        tuple of (dict or None, dict or None)
            The instance dictionary and the assigned slots, in the pair form
            :meth:`object.__getstate__` uses; either half is ``None`` when empty.
        """
        state = object.__getstate__(self)
        instance_dict, slots = state if isinstance(state, tuple) else (state, None)
        transient = self._declared_state_names("_transient_state")
        if transient:
            instance_dict = {k: v for k, v in (instance_dict or {}).items() if k not in transient}
            slots = {k: v for k, v in (slots or {}).items() if k not in transient}
        return (instance_dict or None, slots or None)

    def __setstate__(self, state: Any) -> None:
        """Restore *state*, assigning through ``object.__setattr__``.

        Parameters
        ----------
        state : tuple of (dict or None, dict or None)
            The instance dictionary and slots, as :meth:`__getstate__` returns
            them. A bare dictionary is accepted as the instance-dictionary half.

        Notes
        -----
        A store named in :attr:`_decoupled_state` is restored into its own
        container. ``copy.copy`` passes the original's own attribute values as
        the state, so restoring such a store verbatim would leave both objects
        writing to one container.
        """
        instance_dict, slots = state if isinstance(state, tuple) else (state, None)
        decoupled = self._declared_state_names("_decoupled_state")
        for attribute, value in ((instance_dict or {}) | (slots or {})).items():
            if attribute in decoupled and value is not None:
                value = decoupled_container(value)
            object.__setattr__(self, attribute, value)

    def __reduce__(self) -> tuple[Any, ...]:
        """Return the reconstruction of this object: allocate its class, restore state.

        Returns
        -------
        tuple
            The three-element form ``pickle`` and ``copy`` consume:
            ``object.__new__``, the resolved class, and :meth:`__getstate__`.

        Notes
        -----
        ``object.__new__`` rather than the host's own ``__new__``, which selects
        a class from constructor arguments — ``Record`` promoting to
        ``NumericRecord``, the dynamic distribution subclasses. The class here is
        already the resolved one the state belongs to, so that choice must not be
        made again.
        """
        return (object.__new__, (type(self),), self.__getstate__())
