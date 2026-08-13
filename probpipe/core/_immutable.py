"""Immutability: the assignment guard and the state round-trip it forces.

An object that refuses attribute assignment cannot be restored the way
``pickle`` and ``copy`` restore an ordinary object, which is by assigning its
state back. :class:`Immutable` owns both halves of that bargain — the refusal
and the round-trip — so a class states once that it is immutable instead of
spelling out the consequences itself.

Every tracked term is immutable (``C2 – Functional interface over immutable
objects``, and the operation contract of ``design/05-operations.md`` §V.1, which
promises an implementer's object is never modified), so :class:`TrackedTerm`
mixes this in and a term inherits it by being one. ``EventTemplate`` mixes it in
directly, being immutable without being a term.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = ["Immutable"]


def decoupled_container(container: Any) -> Any:
    """Return a shallow copy of a mutable store: entries shared, container not.

    A store written in place — the annotations channel is the one ProbPipe has —
    must not be shared between an object and whatever it was copied or rebuilt
    from, or a write on one shows through on the other. Any mapping type is
    accepted: an ``xarray.DataTree`` and the like copy themselves, and anything
    else is rebuilt as a ``dict``.
    """
    return container.copy() if hasattr(container, "copy") else dict(container)


def _allocate(cls: type) -> Any:
    """Allocate *cls* without running its ``__new__`` or ``__init__``.

    The reconstruction entry point named by :meth:`Immutable.__reduce__`, at
    module level because a pickle has to be able to import it by name.

    ``object.__new__`` rather than ``cls.__new__``: a host's own ``__new__``
    exists to *select* a class from constructor arguments — ``Record`` promoting
    to ``NumericRecord``, the dynamic distribution subclasses — and *cls* is
    already the resolved class the state belongs to. Re-running that choice on
    load would let a reconstruction land on a different class than it was
    written from.
    """
    return object.__new__(cls)


class Immutable:
    """Refuses attribute assignment, and round-trips its state regardless.

    A host mixes this in to declare that its instances are immutable. Assignment
    and deletion raise :exc:`AttributeError` naming the class the caller
    actually touched; construction writes through ``object.__setattr__``, which
    goes around the guard because the object is not yet anyone's to hold.

    The round-trip
    --------------
    ``pickle`` and ``copy`` restore an object by assigning its state back, which
    the guard refuses, so an immutable class has to answer for its own
    reconstruction. Answering it here rather than per class matters because the
    answer is easy to get subtly wrong in a way that raises nothing:

    * what to save comes from :meth:`object.__getstate__`, which reports every
      assigned slot declared anywhere in the hierarchy **and** an instance
      dictionary if the host has one. A hand-walk over ``__slots__`` misses a
      subclass's storage, and iterates a bare-string ``__slots__`` into
      characters rather than into the one name it declares;
    * a hand-written list of state fields misses whatever was added to the class
      later — in particular anything assigned *after* construction, which no
      constructor argument names;
    * a missing attribute is indistinguishable from an unassigned slot, so both
      mistakes lose data in silence.

    Together with :func:`_allocate`, this makes a round-trip carry exactly the
    state the object holds, for any storage form, without the class restating
    what its state is.

    Transient state
    ---------------
    A host names memos in :attr:`_transient_state` to keep them out of the
    round-trip: a cache is derived from the state that *is* saved, so carrying it
    would bloat the pickle to no purpose and, for a cache of converted arrays,
    duplicate the payload. The attribute is left unset on the reconstruction and
    rebuilt on demand.
    """

    __slots__ = ()

    #: Attribute names excluded from the state round-trip — memos, rebuilt on
    #: demand rather than carried. Subclasses override; the union over the MRO
    #: applies, so a subclass need not repeat its base's entries.
    _transient_state: ClassVar[tuple[str, ...]] = ()

    #: Attribute names holding a store that is written in place, so a
    #: reconstruction takes its own container rather than sharing one. Same MRO
    #: union as :attr:`_transient_state`.
    _decoupled_state: ClassVar[tuple[str, ...]] = ()

    # -- the guard -----------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    # -- the round-trip ------------------------------------------------------

    @classmethod
    def _declared_state_names(cls, attribute: str) -> frozenset[str]:
        """The union of *attribute* over every class in this MRO."""
        return frozenset(
            name for klass in cls.__mro__ for name in klass.__dict__.get(attribute, ())
        )

    def __getstate__(self) -> Any:
        """This object's state, minus its declared memos."""
        state = object.__getstate__(self)
        instance_dict, slots = state if isinstance(state, tuple) else (state, None)
        transient = self._declared_state_names("_transient_state")
        if transient:
            instance_dict = {k: v for k, v in (instance_dict or {}).items() if k not in transient}
            slots = {k: v for k, v in (slots or {}).items() if k not in transient}
        return (instance_dict or None, slots or None)

    def __setstate__(self, state: Any) -> None:
        """Restore *state*, writing through ``object.__setattr__``.

        Both halves are restored — the instance dictionary, where the host has
        one, and the slots — going around the guard exactly as construction
        does. A store named in :attr:`_decoupled_state` is restored into its own
        container, so a write on this object does not reach the one it was
        rebuilt from; ``copy.copy`` is where that matters, its state being the
        original's own attribute values.
        """
        instance_dict, slots = state if isinstance(state, tuple) else (state, None)
        decoupled = self._declared_state_names("_decoupled_state")
        for attribute, value in ((instance_dict or {}) | (slots or {})).items():
            if attribute in decoupled and value is not None:
                value = decoupled_container(value)
            object.__setattr__(self, attribute, value)

    def __reduce__(self):
        """Reconstruct by allocating the resolved class and restoring state.

        Deliberately not the constructor: an immutable object's state is exactly
        what it holds, and rebuilding through ``__init__`` would re-derive it —
        re-inferring a schema an explicit declaration had pinned, or re-deciding
        a class from arguments the state no longer carries.
        """
        return (_allocate, (type(self),), self.__getstate__())
