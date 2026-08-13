"""Object immutability: the assignment guard and the state round-trip.

Provides :class:`Immutable`, the mixin a class inherits to declare that its
instances cannot be modified after construction. It supplies the assignment
guard and, because an object that refuses assignment cannot be restored the way
``pickle`` and ``copy`` restore an ordinary one, the state round-trip those need.

Every tracked term is immutable (``C2 – Functional interface over immutable
objects``), as is :class:`~probpipe.core.event_template.EventTemplate`, which is
immutable without being a term.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = ["Immutable"]


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


def _allocate(cls: type) -> Any:
    """Allocate an instance of *cls* without running ``__new__`` or ``__init__``.

    The reconstruction callable :meth:`Immutable.__reduce__` names, at module
    level so a pickle can import it.

    Notes
    -----
    ``object.__new__`` rather than ``cls.__new__``, because a host's own
    ``__new__`` selects a class from constructor arguments — ``Record``
    promoting to ``NumericRecord``, the dynamic distribution subclasses — and
    *cls* is already the resolved class the state belongs to.
    """
    return object.__new__(cls)


class Immutable:
    """Mixin declaring that instances cannot be modified after construction.

    Assignment and deletion raise :exc:`AttributeError`, naming the class of the
    object that was touched. A constructor writes through
    ``object.__setattr__``, which the guard does not intercept.

    ``copy.copy``, ``copy.deepcopy``, and ``pickle`` return an object of the same
    class holding the same state: every attribute the original has assigned,
    whether it lives in a slot declared anywhere in the class hierarchy or in an
    instance dictionary. Reconstruction restores that state directly rather than
    calling the class's constructor, so nothing is re-derived from arguments the
    state no longer carries. Attributes named in :attr:`_transient_state` are
    left out and unset on the copy; those named in :attr:`_decoupled_state` are
    restored into a container of their own.

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
            :func:`_allocate`, the resolved class, and :meth:`__getstate__`.
        """
        return (_allocate, (type(self),), self.__getstate__())
