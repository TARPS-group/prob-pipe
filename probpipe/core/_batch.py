"""Batch — the generic multiplicity axis.

A :class:`Batch` holds *how many* objects there are, separately from *what one
object contains*: an nd collection of elements of a common type, tracked like any
other term. ``len``, ``iter``, :attr:`Batch.batch_shape`, and
:attr:`Batch.batch_size` speak only about the batch axes, never about the
structure inside an element.

**Levels.** A batch's axes are partitioned into ordered *levels*:
:attr:`Batch.axis_groups` tiles ``batch_shape`` into contiguous groups, outermost
first, and :attr:`Batch.level_names` names them one for one.
:meth:`Batch.with_level_names` renames a level, and a name already in use raises.
`N` laws of `S` draws each are therefore ``(N,)`` of ``(S,)`` rather than one
anonymous ``(N, S)``, while anything stated over ``batch_shape`` — flat
vectorization above all — applies to a multi-level batch unchanged.

**Indexing.** ``[]`` dispatches on the key. A *position* — an integer, a slice,
or a tuple of those — addresses the batch axes; a *name*, or a tuple of names,
addresses a field within every element, which only a batch whose elements have
fields answers. :meth:`Batch.at_levels` is the by-name counterpart over levels,
taking one indexer per named level and keeping the levels not named. Either way
an integer drops its axis and a slice keeps it, a level whose axes are all
dropped is removed, and the result is an element once the selection reaches one
and a sub-batch view otherwise.

**A batch's type is its own.** :class:`BatchSpec` is the term spec at the
*family* kind: the element's specification together with that named
multiplicity. A batch stores it and nothing else about its type, so
:attr:`Batch.spec` names the collection just as any other term's spec names the
term, :attr:`Batch.element_spec` and the level accessors are views on it, and a
batch of values naming no kind is specified all the same.

**A view is named by what it selects.** A view's name is derived from the batch
it was taken from and the positions it selects, naming the level each selection
addresses: selecting chain 0 of ``posterior`` yields the name
``"posterior[chain=0]"``, and its draw 7 yields ``"posterior[chain=0, draw=7]"``.
Levels selected whole are left out, and the levels that appear are listed in the
batch's own order, so a derived name is a function of what the view selects: two
routes to one selection read alike, and two selections never do.

**Storage is the concrete class's business, and only storage.** This module
owns the level algebra: the shape invariants, the naming rules, index
normalization for :meth:`Batch.at_levels`, and the identity a view derives. A
concrete batch supplies two hooks, :meth:`Batch._element_at` and
:meth:`Batch._sub_batch_at`, which present an element or a sub-batch — a view
over the same storage, not a copy of it — at a normalized positional index. A
third, :meth:`Batch._at_fields`, addresses the *fields* of an element and is
supplied only by a batch whose elements have any: ``[]`` dispatches on the key
type, so a name reaches the elements and a position reaches the axes. Renaming a
level needs no hook at all: it touches no axes and no elements, so
:meth:`Batch._with_level_names` defaults to a shallow copy.

**A selection inherits, it does not record.** Reading one position out of a
collection computes nothing, so no provenance node claims it happened: a view
carries the lineage of the batch it came out of, and which position it was is
carried by its name. Element provenance is :meth:`Batch._element_at`'s own,
since only that hook knows whether it built the element or borrowed it.

See design II.5.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Self

from .event_template import TermSpec, ValueSpec
from .provenance import Provenance
from .tracked import TrackedTerm

__all__ = ["Batch", "BatchSpec"]

# One indexer per level: a single value addresses the level's first axis, a
# tuple addresses its axes in order. ``None`` means the whole axis — the form ``:``
# takes where a keyword cannot spell it, and the one place ``None`` says this.
type LevelIndexer = int | slice | tuple[int | slice | None, ...] | None


@dataclass(frozen=True, init=False)
class BatchSpec(TermSpec):
    """A term spec for a :class:`Batch`: an element spec plus a named multiplicity.

    ``element_spec`` is what every element satisfies; ``axis_groups`` and
    ``level_names`` are the multiplicity, the batch axes tiled into named levels
    as :class:`Batch` describes them. Level names are unique within a batch, and
    ``batch_shape`` / ``batch_size`` read off the tiling.

    This is the single stored source of a batch's type, so it specifies the
    *collection* rather than one element, and :class:`Batch` keeps no second copy
    of the multiplicity: its shape and level accessors read the stored spec.

    Parameters
    ----------
    element_spec : ValueSpec
        What every element of the batch satisfies. A raw-value spec is admitted
        as readily as a term spec.
    axis_groups : iterable of iterable of int
        The axis *sizes* each level holds, in order, outermost level first. Every
        level holds at least one axis, and there is at least one axis in all.
        Stored as a tuple of tuples, which is what makes the spec hashable.
    level_names : iterable of str
        One name per level, aligned with *axis_groups*. Each is a non-empty
        identifier, unique within the batch.

    Attributes
    ----------
    batch_shape : tuple of int
        The batch axes, flat: the concatenation of ``axis_groups``.
    batch_size : int
        The total element count, ``prod(batch_shape)``.

    Raises
    ------
    TypeError
        If ``element_spec`` is not a :class:`ValueSpec`, an axis size is not an
        integer, or a level name is not a string.
    ValueError
        If there are no batch axes, a level holds no axes, an axis size is
        negative, the number of names does not match the number of levels, or a
        level name is empty, not an identifier, or duplicated.

    Notes
    -----
    A batch whose elements name no kind is specified all the same: a raw-value
    ``element_spec`` is as well formed here as a term spec, which is what lets a
    batch of opaque values carry a term spec of its own.

    An axis size may be a **symbolic dimension name** instead of an integer, as
    an ``ArraySpec`` shape entry may, so that a declaration can fix the number of
    levels while deferring how many elements each holds — "returns a batch of
    ``S`` draws" before ``S`` is known. The names share one scope with the
    element's schema, so a batch of ``("n",)`` over arrays of shape ``("n",)`` is
    square by declaration. A *declaration* may be polymorphic; a live
    :class:`Batch` may not, since it holds elements at positions, so
    :meth:`Batch._init_batch` refuses a spec with free dimensions.

    A duplicate level name is an error rather than something this class resolves.
    An operation that mints a level takes the name to give it, so a name already
    in use means the caller must supply another, and
    :meth:`Batch.with_level_names` raises on a collision for the same reason.
    """

    element_spec: ValueSpec
    axis_groups: tuple[tuple[int | str, ...], ...]
    level_names: tuple[str, ...]

    def __init__(
        self,
        element_spec: ValueSpec,
        axis_groups: Iterable[Iterable[int]],
        level_names: Iterable[str],
    ) -> None:
        """Store the element spec and the multiplicity, validating the levels.

        The fields are the *stored* types; the iterables accepted here are
        normalized to tuples before assignment, so a stored spec is hashable.
        """
        if not isinstance(element_spec, ValueSpec):
            raise TypeError(
                f"BatchSpec.element_spec must be a ValueSpec, got {type(element_spec).__name__}"
            )
        if isinstance(level_names, str):
            raise TypeError(
                f"level_names holds one name per level, so the string {level_names!r} would be "
                f"read as one name per character; write a tuple, as ({level_names!r},) for a "
                f"single level"
            )
        groups: list[tuple[int, ...]] = []
        for group in axis_groups:
            if not isinstance(group, Iterable):
                # A flat batch_shape is the natural thing to reach for here, and
                # descending into it would fail without saying what was wrong.
                raise TypeError(
                    f"axis_groups holds one group of axis sizes per level, so {group!r} is not "
                    f"a group; a flat shape such as (4,) is one level of one axis, written "
                    f"((4,),)"
                )
            groups.append(tuple(_axis_size(size) for size in group))
        groups = tuple(groups)
        names = tuple(level_names)

        if not groups:
            raise ValueError("a Batch has at least one batch axis; axis_groups was empty")
        for group in groups:
            if not group:
                raise ValueError(f"every level holds at least one axis; got axis_groups={groups}")
            for size in group:
                if isinstance(size, int) and size < 0:
                    raise ValueError(f"axis sizes are non-negative; got axis_groups={groups}")
        if len(names) != len(groups):
            raise ValueError(
                f"level_names must name every level: {len(names)} names for {len(groups)} levels"
            )
        for level_name in names:
            if not isinstance(level_name, str):
                raise TypeError(
                    f"level names are strings, got {type(level_name).__name__}: {level_name!r}"
                )
            if not level_name:
                raise ValueError("level names must be non-empty")
            if not level_name.isidentifier():
                raise ValueError(
                    f"level name {level_name!r} must be an identifier, since at_levels "
                    f"addresses a level by keyword"
                )
        if len(set(names)) != len(names):
            raise ValueError(
                f"level names must be unique within a batch; got {names}. An operation "
                f"minting a level takes the name to use, so give the new level a name of "
                f"its own rather than reusing one already present"
            )

        object.__setattr__(self, "element_spec", element_spec)
        object.__setattr__(self, "axis_groups", groups)
        object.__setattr__(self, "level_names", names)

    @property
    def batch_shape(self) -> tuple[int | str, ...]:
        """The batch axes, flat: the concatenation of :attr:`axis_groups`."""
        return tuple(size for group in self.axis_groups for size in group)

    @property
    def batch_size(self) -> int:
        """The total element count, ``prod(batch_shape)``.

        Raises
        ------
        ValueError
            If the multiplicity is polymorphic. A count is a number, and a
            declaration that defers a size has none until it is bound — the same
            reason a polymorphic ``NumericEventTemplate`` has no flat layout.
        """
        if self.free_dims:
            dimensions = ", ".join(sorted(self.free_dims))
            raise ValueError(
                f"batch_size is undefined for a polymorphic BatchSpec; "
                f"unbound dimensions: {dimensions}"
            )
        return prod(size for size in self.batch_shape if isinstance(size, int))

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the element's schema and of the multiplicity.

        One scope, as everywhere: a name shared between an axis size and the
        element's schema is one dimension, so a batch declared as ``("n",)`` of
        arrays of shape ``("n",)`` states that it is square.
        """
        axes = frozenset(
            size for group in self.axis_groups for size in group if isinstance(size, str)
        )
        return self.element_spec.free_dims | axes

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a :class:`Batch` whose own spec equals this one.

        Anything that is not a ``Batch``, or a ``Batch`` whose spec cannot be
        read, does not satisfy the spec and returns ``False``. Mirrors
        :meth:`~probpipe.RecordSpec.is_valid`.
        """
        if not isinstance(value, Batch):
            return False
        try:
            spec = value.spec
        except (AttributeError, TypeError):
            return False
        return spec == self


class Batch[E](TrackedTerm, ABC):
    """A tracked nd collection of elements of a common type.

    The batch axes are grouped into named **levels**, and indexing them returns a
    *view* — an element once the selection reaches one, a sub-batch otherwise —
    whose name records what it selected. Index by position with ``[]``, by level
    name with :meth:`at_levels`, and, where the elements have fields, by field
    name with ``[]`` as well; ``len`` and ``iter`` walk the leading axis.

    A concrete subclass supplies the element storage, calling :meth:`_init_batch`
    from its constructor; this class stores the batch's :class:`BatchSpec` and the
    identity :class:`TrackedTerm` provides, and nothing else.

    Attributes
    ----------
    spec : BatchSpec
        This batch's own specification, at the family kind. The single stored
        source of its type: everything below is a view on it.
    element_spec : ValueSpec
        The specification every element satisfies.
    batch_shape : tuple of int
        The batch axes, the flat concatenation of :attr:`axis_groups`. Always
        non-empty: a batch has at least one batch axis.
    batch_size : int
        The total element count, ``prod(batch_shape)``.
    axis_groups : tuple of tuple of int
        ``batch_shape`` tiled into levels, outermost level first.
    level_names : tuple of str
        One name per level, aligned with :attr:`axis_groups`, unique within the
        batch.

    Notes
    -----
    ``batch_shape`` and ``batch_size`` are named rather than reusing numpy's
    ``shape`` / ``size`` because a bare name would ambiguously cover both the
    batch axes and the content of one element.

    A batch is immutable: assignment and deletion raise, and ``pickle`` / ``copy``
    restore the slots around that guard.
    """

    __slots__ = (
        "_name",
        "_name_is_auto",
        "_provenance",
        "_root_name",
        "_root_selection",
        "_root_spec",
        "_spec",
    )

    # The three ``_root_*`` slots are the machinery of view naming: the name and
    # spec of the batch a derivation starts from, and which of *that* batch's
    # positions this object selects — one entry per root axis, an integer where an
    # axis has been dropped and a range of positions where one is kept. ``_name``
    # is read off them, which is what makes two routes to one selection agree: the
    # reading is a function of the selection, not of the calls that reached it.
    #
    # They are never ``None``, not even on a batch nobody has indexed. Such a batch
    # is its own root and selects all of itself, so its derived name is the name it
    # was given, and composing a further selection needs no special case at the
    # head of the chain. Leaving them unset for a non-view would put a
    # "means everything" reading on ``None`` — the reading positional ``[]``
    # refuses — and every site that composes or renders a selection would carry a
    # branch for it. :meth:`with_name` re-roots a view: a user-given name replaces
    # the derivation and discards the selection accumulated before it.

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getstate__(self) -> Any:
        """This batch's whole state, for ``pickle`` and ``copy``.

        Delegates to :meth:`object.__getstate__`, which reports every assigned
        slot declared anywhere in the class hierarchy — a subclass's storage
        included, without it having to say so — together with an instance
        dictionary if the subclass has one.

        Notes
        -----
        Only :meth:`__setstate__` needs overriding here; ``__getstate__`` is
        defined alongside it so that the pair reads as one, and so that walking
        the hierarchy by hand is not reintroduced. That walk is easy to get
        subtly wrong: ``__slots__`` may be a bare string naming one slot, which
        iterates into characters rather than into that name, and a subclass that
        declares no ``__slots__`` keeps its attributes in a dictionary that no
        walk over ``__slots__`` would find. Either would drop state silently,
        since a missing attribute is indistinguishable from an unassigned slot.
        """
        return object.__getstate__(self)

    def __setstate__(self, state: Any) -> None:
        """Restore *state* through ``object.__setattr__``.

        ``pickle`` and ``copy`` restore state by assignment, which the
        immutability guard refuses, so the write has to go around it exactly as
        construction does. Both halves of the state are restored: the instance
        dictionary, where a subclass has one, and the slots.
        """
        instance_dict, slots = state if isinstance(state, tuple) else (state, None)
        for attribute, value in (instance_dict or {}).items():
            object.__setattr__(self, attribute, value)
        for slot, value in (slots or {}).items():
            object.__setattr__(self, slot, value)

    # -- construction -------------------------------------------------------

    def _init_batch(
        self,
        spec: BatchSpec,
        *,
        name: str,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        """Store the batch's *spec* and identity (constructor helper).

        Assigns the state via ``object.__setattr__`` so immutable hosts can call
        this from their constructor, and delegates identity to
        :meth:`TrackedTerm._init_tracked`. The level invariants are the spec's
        own and are checked when it is constructed. A subclass declares only its
        storage in ``__slots__``; this class declares the batch's own state and
        the identity slots it hosts.

        The batch becomes the **root** of the names its views derive: it selects
        all of itself, so its derived name is its own. A view built by
        :meth:`at_levels` or ``[]`` is re-pointed at its parent's root
        afterwards, which is what makes a derived name independent of the route
        taken to it.

        Raises
        ------
        TypeError
            If *spec* is not a :class:`BatchSpec`.
        """
        if not isinstance(spec, BatchSpec):
            raise TypeError(f"a Batch is specified by a BatchSpec, got {type(spec).__name__}")
        if spec.free_dims:
            dimensions = ", ".join(sorted(spec.free_dims))
            raise ValueError(
                f"a Batch holds elements at positions, so its multiplicity is concrete; "
                f"this spec leaves {dimensions} unbound. A polymorphic BatchSpec is a "
                f"declaration — bind it before building the batch it describes"
            )
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_root_name", name)
        object.__setattr__(self, "_root_spec", spec)
        object.__setattr__(self, "_root_selection", _whole_of(spec))
        self._init_tracked(name, name_is_auto=name_is_auto, provenance=provenance)

    # -- the specification --------------------------------------------------

    @property
    def spec(self) -> BatchSpec:
        """This batch's own specification, at the family kind."""
        return self._spec

    @property
    def element_spec(self) -> ValueSpec:
        """The specification every element satisfies — a view on :attr:`spec`."""
        return self._spec.element_spec

    # -- shape and levels ---------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch axes, flat: the concatenation of :attr:`axis_groups`."""
        return self._spec.batch_shape

    @property
    def batch_size(self) -> int:
        """The total element count, ``prod(batch_shape)``."""
        return self._spec.batch_size

    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]:
        """``batch_shape`` tiled into levels, outermost level first."""
        return self._spec.axis_groups

    @property
    def level_names(self) -> tuple[str, ...]:
        """One name per level, aligned with :attr:`axis_groups`."""
        return self._spec.level_names

    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self:
        """Rename levels ``old -> new``; shapes and elements are unchanged.

        Accepts a positional mapping, keyword pairs, or both. Every name given must
        be a level of this batch, and the result must still name each level once.

        Parameters
        ----------
        mapping : Mapping of str to str, optional
            Renames as ``{old: new}``, positional so that any level name is
            addressable even where it is not a valid keyword.
        **kwargs : str
            Renames as ``old="new"``, for the common case. A level named in both
            must be given the same new name in each.

        Returns
        -------
        Self
            A shallow copy over the same axes and elements, specified over the new
            level names, with its own derived name re-read under them.

        Raises
        ------
        KeyError
            If a name to rename is not a level of this batch.
        ValueError
            If a level is renamed twice with different names, or a new name is
            empty, not an identifier, collides with a level that is being kept, is
            the target of two renames, or — renaming a view — belongs to a level
            the view derives its name from but no longer carries.
        TypeError
            If a new name is not a string.

        Notes
        -----
        The level counterpart of ``with_path_names``, which renames the *fields
        within* an element. The two namespaces are independent: renaming a level
        never touches a field name, or the reverse.
        """
        positional = dict(mapping or {})
        conflicting = sorted(
            old for old, new in kwargs.items() if old in positional and positional[old] != new
        )
        if conflicting:
            raise ValueError(
                f"level {conflicting[0]!r} is renamed twice, by the positional mapping and "
                f"by a keyword; give it one new name"
            )
        renames: dict[str, str] = {**positional, **kwargs}
        unknown = set(renames) - set(self.level_names)
        if unknown:
            raise KeyError(
                f"not levels of this batch: {sorted(unknown)}; have {list(self.level_names)}"
            )
        for new in renames.values():
            # Type before emptiness: None, 0 and [] are all falsy, and reporting
            # them as empty names would describe the wrong problem.
            if not isinstance(new, str):
                raise TypeError(f"level names are strings, got {type(new).__name__}: {new!r}")
            if not new:
                raise ValueError("level names must be non-empty")

        renamed = tuple(renames.get(old, old) for old in self.level_names)
        if len(set(renamed)) != len(renamed):
            raise ValueError(
                f"renaming would duplicate a level name: {self.level_names} -> {renamed}"
            )
        return self._with_level_names(renamed)

    def with_name(self, name: str) -> Self:
        """Rename the batch, which becomes the root its view names derive from.

        A user-given name overrides derivation, so the copy selects all of
        itself: its own name is *name*, and a view of it reads
        ``name[level=...]`` rather than carrying any selection the original had
        accumulated. This is the way to rename a level a view derives its name
        from but no longer carries, which :meth:`with_level_names` refuses.

        Parameters
        ----------
        name : str
            The new name, taken as user-given: the copy's ``name_is_auto`` is
            ``False``, so no later transform re-derives it.

        Returns
        -------
        Self
            A shallow copy over the same axes, elements, and spec, named *name*
            and rooted at itself.

        Raises
        ------
        TypeError
            If *name* is not a non-empty string.
        """
        renamed = super().with_name(name)
        object.__setattr__(renamed, "_root_name", name)
        object.__setattr__(renamed, "_root_spec", renamed._spec)
        object.__setattr__(renamed, "_root_selection", _whole_of(renamed._spec))
        return renamed

    # -- reading ------------------------------------------------------------

    def __repr__(self) -> str:
        """The class, the batch's name, and each level with its sizes.

        A level of one axis reports that size, and a level of several reports them
        as a tuple, so a two-level batch of chains and draws reads
        ``<class>(name='posterior', chain=4, draw=1000)``.

        Notes
        -----
        No element is read, so the cost is the same on a batch of any size and
        nothing here can raise from storage. That matters beyond convenience:
        :meth:`TrackedTerm.with_provenance` interpolates the batch into its
        write-once error, so a ``repr`` that could fail would fail there.
        """
        levels = ", ".join(
            f"{level_name}={group[0] if len(group) == 1 else group}"
            for level_name, group in zip(self.level_names, self.axis_groups, strict=True)
        )
        return f"{type(self).__name__}(name={self.name!r}, {levels})"

    # -- indexing -----------------------------------------------------------

    def __len__(self) -> int:
        """The leading batch axis, ``batch_shape[0]``."""
        return self.batch_shape[0]

    def __iter__(self) -> Iterator[E | Self]:
        """Iterate the leading batch axis, yielding views."""
        return (self[index] for index in range(len(self)))

    def __getitem__(self, key: Any) -> Any:
        """Index the batch axes by position, or an element's fields by name.

        By **position**, a single indexer addresses the leading axis and a tuple
        the leading axes in order, as ``batch[i, j]`` does for an array. Each
        indexer is an integer or a slice; an integer drops its axis and a slice
        keeps it, a whole axis being written ``:``. The result is an element once
        every axis is dropped and a sub-batch view otherwise, exactly as for
        :meth:`at_levels`, which is the by-name counterpart over *levels*.

        By **name**, ``batch["x"]`` addresses a field within every element and
        ``batch["outer", "a"]`` a path of fields, which a batch whose elements have
        fields answers and others refuse.

        Returns
        -------
        E or Self or Any
            An element or a sub-batch view for a position; whatever the elements'
            fields yield for a name.

        Raises
        ------
        IndexError
            If more indexers are given than there are batch axes, or an integer
            is out of range for its axis.
        TypeError
            If an indexer is not an integer or a slice, if a tuple mixes field
            names with axis indexers, or if a name is given to a batch whose
            elements have no fields.
        ValueError
            If a slice has a step of zero.

        Notes
        -----
        The two readings never collide, an axis having no name and a field no
        position, which is what lets one operator serve both. The axis side is
        stated once here for every batch; the field side is left to
        :meth:`_at_fields`. ``None`` is not a position: it spells a whole axis in
        :meth:`at_levels` alone, where a keyword cannot take a ``:`` literal.
        """
        # A lone key is decided by its type: a string is a one-element field path,
        # and anything else is an indexer for the leading axis. A non-position is
        # not rejected here — _at_axes reports it against the axis it was given
        # for, which is more than this method knows.
        if isinstance(key, str):
            return self._at_fields((key,))
        if not isinstance(key, tuple):
            return self._at_axes((key,))
        # A tuple is the one key legal on both sides — a path of field names, or
        # one indexer per leading axis — so what is inside it decides rather than
        # its type. All names is a path; no names is an index, which is also what
        # gives ``batch[()]`` the whole batch instead of an empty path; and a
        # mixture is neither, so it is refused as the mixture it is rather than
        # left for the axis side to complain about the count.
        named = [entry for entry in key if isinstance(entry, str)]
        if not named:
            return self._at_axes(key)
        if len(named) == len(key):
            return self._at_fields(key)
        raise TypeError(
            f"{key!r} mixes field names with axis indexers: a tuple key addresses either a "
            f"path of fields within an element or the batch axes in order, not both. Index "
            f"the axes and the fields in separate steps"
        )

    def at_levels(self, /, **levels: LevelIndexer) -> E | Self:
        """Index by named level, returning an element or a sub-batch view.

        Each keyword names a level of this batch and gives it an indexer: an
        integer, a slice, ``None``, or a tuple of those addressing the level's axes
        in order. An integer drops its axis; a slice or ``None`` keeps it, ``None``
        standing for the whole axis as ``:`` does. A shorter tuple fills the
        level's leading axes and leaves the rest whole, so a scalar ``draw=i`` on a
        two-axis ``draw`` level means ``draw=(i, None)``. A level not named is kept
        whole, and a level whose axes are all dropped is removed, yielding the
        inner batch or element just as positional indexing does.

        The receiver is positional-only, so every level name is addressable as a
        keyword — including one that happens to spell a parameter of this method.

        Returns
        -------
        E or Self
            The element the selection reaches, or a sub-batch view over the levels
            that remain.

        Raises
        ------
        KeyError
            If a keyword is not a level of this batch.
        ValueError
            If a level is given more indexers than it has axes, or a slice has a
            step of zero.
        IndexError
            If an integer is out of range for its axis.
        TypeError
            If an indexer is not an integer, a slice, ``None``, or a tuple of
            those.

        Notes
        -----
        The by-name counterpart of positional ``[]``, and the level analogue of
        ``NamedTree.at_path``: a path addresses a position in a tree and returns a
        leaf or a subtree, while named level indexers address positions here and
        return an element or a sub-batch. ``None`` spells a whole axis in this form
        alone, a keyword being unable to take a ``:`` literal; positional ``[]``
        writes ``:`` and refuses ``None``.
        """
        unknown = set(levels) - set(self.level_names)
        if unknown:
            raise KeyError(
                f"not levels of this batch: {sorted(unknown)}; have {list(self.level_names)}"
            )

        axis_index: list[int | slice] = [slice(None)] * len(self.batch_shape)
        # Where each addressed axis came from, so a complaint about an indexer
        # names the level it was given for rather than the flat axis it landed on.
        # Only addressed axes carry one; the rest cannot fail, being whole.
        where: list[str | None] = [None] * len(self.batch_shape)
        start = 0
        for level_name, group in zip(self.level_names, self.axis_groups, strict=True):
            if level_name in levels:
                given = levels[level_name]
                indexers = given if isinstance(given, tuple) else (given,)
                if len(indexers) > len(group):
                    raise ValueError(
                        f"level {level_name!r} has {len(group)} axes but got "
                        f"{len(indexers)} indexers"
                    )
                for offset, indexer in enumerate(indexers):
                    axis_index[start + offset] = slice(None) if indexer is None else indexer
                    where[start + offset] = (
                        f"level {level_name!r} of size {group[0]}"
                        if len(group) == 1
                        else f"axis {offset} of level {level_name!r}, axes {group}"
                    )
            start += len(group)
        return self._at_axes(tuple(axis_index), where=tuple(where))

    # -- the concrete-storage seam ------------------------------------------

    @abstractmethod
    def _element_at(self, index: tuple[int, ...], *, name: str) -> E:
        """The single element at a fully-integer positional *index*.

        *name* is the identity this class derived for the element view, and the
        same split governs it as governs provenance below. A batch that
        *materializes* an element gives it that name, marked auto-derived. A batch
        that *stores* its elements hands back the stored object under the name it
        already carries: renaming it would mean returning a copy, and an object
        placed in a batch by name already means something. An element that is a
        bare value has no identity to carry either way.

        **Provenance is this hook's own**, because only it knows whether the
        element was built or borrowed. A batch that *materializes* an element —
        a row of columnar storage does not exist until it is built — calls
        :meth:`_inherit_provenance` on what it built. A batch that *stores* its
        elements returns the stored object untouched: it did not produce that
        object, so it cannot truthfully claim its lineage, and writing to it
        would reach into something the caller still holds.
        """

    @abstractmethod
    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the sub-batch at a partial positional *index*.

        *index* is one entry per axis of this batch: a resolved position for an
        axis being dropped, or a slice selecting the positions a kept axis spans,
        **in the order the view presents them** — a descending slice for a
        reversed selection, which storage must honor rather than re-sort, since
        the view's derived names are stated in that order.

        *spec* is the view's own specification: the same ``element_spec`` over
        the surviving levels, with every integer-indexed axis already removed.
        *name* is the derived identity, taken marked auto-derived. A subclass
        stores both as given rather than recomputing either; the names a further
        view derives from are re-pointed at this view's own root afterwards.

        **A view, not a copy.** The result shares this batch's storage wherever
        the storage affords it — a numpy or JAX slice, or a list re-indexed over
        the same objects — so that a batch stays the single source of its elements
        and a selection does not pay for what it selects. Selecting the whole batch
        reaches this hook like any other selection, which costs nothing once the
        result is a view; it is deliberately not short-circuited to ``self``, since
        the view carries a name and provenance of its own and ``self`` already has
        both.
        """

    def _inherit_provenance[T](self, produced: T) -> T:
        """Give something this batch *produced* the batch's own provenance.

        Selecting is not a step in a computation. Nothing is computed by reading
        one position out of a collection, so no node records the reading, and
        what a selection carries is the lineage of the batch it came out of.
        *Which* position was selected is carried by the name, which states it
        exactly — ``posterior[chain=0, draw=7]`` — so nothing is lost by not
        recording it twice.

        This class applies it to every sub-batch view, which it manufactures. An
        element is :meth:`_element_at`'s to attribute, since only that hook knows
        whether it built the element or borrowed it from storage.

        A *produced* object that already carries provenance keeps it, and one
        that is not a tracked term has nowhere to carry it, so neither pays for a
        record that would be discarded.

        Notes
        -----
        The consequence worth knowing: a view's lineage is indistinguishable from
        its batch's, and ``provenance.parents`` does not point at the batch. That
        is the intended reading of "selection is not an event" — there is no edge
        because there was no computation — but it does mean a lineage walk shows
        no selection step, and only the name says a view is one.
        """
        if self._provenance is None or not isinstance(produced, TrackedTerm):
            return produced
        if produced.provenance is not None:
            return produced
        return produced.with_provenance(self._provenance)

    def _at_fields(self, path: tuple[str, ...]) -> Any:
        """The batched field at *path* within every element, for a named ``[]`` key.

        A batch whose elements have named fields answers this; the default
        reports that these elements have none. Only the *field* side of ``[]``
        lands here — the axis side is this class's own, so a subclass gains
        indexing by name without restating what indexing by position means.
        """
        addressed = path[0] if len(path) == 1 else path
        raise TypeError(
            f"the elements of this {type(self).__name__} have no fields to address by name, "
            f"so {addressed!r} indexes nothing. [] addresses the batch axes by position, and "
            f"at_levels addresses them by level name"
        )

    # -- internals ----------------------------------------------------------

    def _at_axes(
        self, index: tuple[int | slice | None, ...], *, where: tuple[str | None, ...] = ()
    ) -> E | Self:
        """Resolve a positional index over the batch axes to an element or view.

        *where* optionally says where each indexer came from, for the error
        messages: :meth:`at_levels` names the level it was given for, and a
        positional index falls back to the flat axis it addressed.
        """
        shape = self.batch_shape
        if len(index) > len(shape):
            raise IndexError(f"too many indices for batch_shape {shape}: got {len(index)}")

        normalized: list[int | range] = []
        for axis, size in enumerate(shape):
            indexer = index[axis] if axis < len(index) else slice(None)
            normalized.append(
                _normalize_indexer(
                    indexer, size, axis, shape, where[axis] if axis < len(where) else None
                )
            )

        selection = self._compose_selection(normalized)
        label = _render_index(self._root_spec, selection)
        name = f"{self._root_name}[{label}]" if label else self._root_name

        dropped = tuple(i for i in normalized if isinstance(i, int))
        if len(dropped) == len(shape):
            return self._element_at(dropped, name=name)

        groups, names = self._surviving_levels(normalized)
        spec = replace(self._spec, axis_groups=groups, level_names=names)
        view = self._sub_batch_at(
            tuple(_as_storage_slice(i) for i in normalized), spec=spec, name=name
        )
        object.__setattr__(view, "_root_name", self._root_name)
        object.__setattr__(view, "_root_spec", self._root_spec)
        object.__setattr__(view, "_root_selection", selection)
        if not label:
            # Selecting the whole batch derives nothing, so the view keeps the
            # name it came with: a user-given name stays user-given and is not
            # re-derived by a later transform.
            object.__setattr__(view, "_name_is_auto", self._name_is_auto)
        return self._inherit_provenance(view)

    def _compose_selection(self, normalized: list[int | range]) -> tuple[int | range, ...]:
        """This view's selection composed with *normalized*, in root coordinates.

        Each entry of :attr:`_root_selection` is one root axis: an integer for an
        axis already dropped, or the ``range`` of root positions a kept axis
        still spans, in the order the axis presents them. Composing in root
        coordinates is what makes a derived name a function of the object rather
        than of the route to it — two routes to the same selection compose to
        the same tuple.

        A ``range`` is the resolved form throughout: it carries the selected
        positions and their order without a ``slice``'s from-the-end bounds, so
        composing and sizing it never re-interpret a bound.
        """
        composed: list[int | range] = []
        axis = 0
        for entry in self._root_selection:
            if isinstance(entry, int):
                composed.append(entry)
                continue
            indexer = normalized[axis]
            axis += 1
            if isinstance(indexer, int):
                composed.append(entry[indexer])
            else:
                composed.append(
                    range(
                        entry.start + indexer.start * entry.step,
                        entry.start + indexer.stop * entry.step,
                        entry.step * indexer.step,
                    )
                )
        return tuple(composed)

    def _surviving_levels(
        self, normalized: list[int | range]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        """The levels left after dropping every integer-indexed axis.

        A kept axis is sized by the number of positions its selection spans; an
        integer-indexed axis is gone, and a level all of whose axes are gone goes
        with them.
        """
        groups: list[tuple[int, ...]] = []
        names: list[str] = []
        start = 0
        for level_name, group in zip(self.level_names, self.axis_groups, strict=True):
            surviving = tuple(
                len(entry)
                for entry in normalized[start : start + len(group)]
                if isinstance(entry, range)
            )
            if surviving:
                groups.append(surviving)
                names.append(level_name)
            start += len(group)
        return tuple(groups), tuple(names)

    def _with_level_names(self, level_names: tuple[str, ...]) -> Self:
        """A shallow copy specified over *level_names*, sharing shape and elements.

        Renaming touches no axes and no elements, so the default is a shallow
        copy carrying a renamed spec. :meth:`TrackedTerm._shallow_copy` assigns
        through ``object.__setattr__``, which is what makes this safe to define
        here: it runs no ``__init__`` and survives this class's immutability
        guard, so nothing is assumed about a subclass's constructor.

        The names the copy's own name derives from are renamed with it, and its
        name is re-derived, so a renamed view and any view taken from it read the
        level the same way. The copy carries no provenance of its own beyond the
        rename: the record of how the batch it was renamed from arose belongs to
        that batch.

        Override only when a subclass caches something derived from the level
        names, since a shallow copy would carry the stale cache; call ``super``
        for the renamed copy rather than reassigning ``_spec`` by hand.
        """
        repinned = dict(zip(self.level_names, level_names, strict=True))
        root_names = tuple(repinned.get(name, name) for name in self._root_spec.level_names)
        if len(set(root_names)) != len(root_names):
            taken = sorted({name for name in root_names if root_names.count(name) > 1})
            raise ValueError(
                f"level name {taken[0]!r} is already used by a level this view derives its "
                f"name from but no longer carries; renaming onto it would make the derived "
                f"name ambiguous. Rename it on the batch this view came from, or give the "
                f"level another name"
            )

        renamed = self._shallow_copy()
        object.__setattr__(renamed, "_spec", replace(self._spec, level_names=level_names))
        object.__setattr__(renamed, "_root_spec", replace(self._root_spec, level_names=root_names))
        label = _render_index(renamed._root_spec, self._root_selection)
        object.__setattr__(
            renamed, "_name", f"{self._root_name}[{label}]" if label else self._root_name
        )
        object.__setattr__(renamed, "_provenance", None)
        renamed.with_provenance(Provenance.create("with_level_names", parents=[self]))
        return renamed


def _axis_size(size: Any) -> int | str:
    """An axis size as an ``int``, or a symbolic dimension name as a ``str``.

    The two spellings ``ArraySpec.shape`` accepts, for the same reason: a
    declaration may defer a size while fixing the rank. A name must be a
    non-empty identifier, as a level name must be.
    """
    if isinstance(size, str):
        if not size.isidentifier():
            raise ValueError(
                f"a symbolic axis size must be an identifier, so that with_dims can "
                f"bind it by keyword; got {size!r}"
            )
        return size
    try:
        return operator.index(size)
    except TypeError:
        raise TypeError(
            f"axis sizes are integers or symbolic dimension names, "
            f"got {type(size).__name__}: {size!r}"
        ) from None


def _normalize_indexer(
    indexer: Any, size: int, axis: int, shape: tuple[int, ...], where: str | None = None
) -> int | range:
    """One axis indexer as a resolved integer or the ``range`` of positions it selects.

    An omitted axis and ``:`` both mean the whole axis. An integer is resolved
    against *size* (so a negative index names the same position as its
    non-negative twin, and both derive the same name); a slice is resolved to the
    positions it selects, in order. A ``bool`` is rejected rather than read as
    ``0`` / ``1``, since a batch axis has no mask indexing for it to mean, and a
    bare ``None`` is rejected here because :meth:`Batch.at_levels`, the one place
    it spells a whole axis, has already turned it into ``:``: silently reading a
    ``None`` left by an unset argument as *all of it* would answer a question the
    caller never asked.

    Resolving to a ``range`` rather than a bounded slice is what keeps a
    descending selection intact: ``slice.indices`` reports the stop of a reverse
    slice as ``-1``, a bound only meaningful once, so anything that resolved it a
    second time would read that as the last position and select nothing.
    """
    if isinstance(indexer, slice):
        if indexer.step == 0:
            # ``slice.indices`` would raise this itself, but naming neither the
            # batch nor the axis the step was given for.
            raise ValueError(
                f"a batch axis is not selected with a step of zero "
                f"({_location(axis, shape, where)}); a step is how far apart the selected "
                f"positions are, so zero selects nothing and no position twice"
            )
        try:
            return range(*indexer.indices(size))
        except TypeError:
            # ``slice.indices`` would raise this itself, naming neither the batch
            # nor the axis -- and a bound computed with ``/`` is a float, which is
            # the ordinary way to arrive here.
            raise TypeError(
                f"a batch axis is sliced by integers, and {indexer!r} is not "
                f"({_location(axis, shape, where)})"
            ) from None
    if indexer is None:
        raise TypeError(
            f"a batch axis is not indexed by None ({_location(axis, shape, where)}); write "
            f"':' for the whole axis. None spells it in at_levels alone, where a keyword "
            f"cannot take a ':' literal"
        )
    if isinstance(indexer, bool):
        raise TypeError(
            f"a batch axis is not indexed by a bool ({_location(axis, shape, where)}); use "
            f"an integer or a slice"
        )
    try:
        position = operator.index(indexer)
    except TypeError:
        raise TypeError(
            f"a batch axis is indexed by an integer or a slice, not "
            f"{type(indexer).__name__} ({_location(axis, shape, where)})"
        ) from None
    resolved = position + size if position < 0 else position
    if not 0 <= resolved < size:
        raise IndexError(f"index {position} is out of range for {_location(axis, shape, where)}")
    return resolved


def _location(axis: int, shape: tuple[int, ...], where: str | None) -> str:
    """Where an indexer was given, as an error names it.

    *where* is the caller's own account of the position — the level a keyword
    addressed, say — and the flat axis is the fallback for an index given
    positionally, which is the only reading available there. Either way the phrase
    carries the sizes it is judged against. Formatted on the error path alone, so
    naming a position costs nothing when nothing is wrong.
    """
    return where if where is not None else f"axis {axis} of batch_shape {shape}"


def _bounds(selected: range) -> tuple[int, int | None, int]:
    """The first position, the bound after the last, and the step of *selected*.

    One form per set of positions, so a name stays a function of the selection:
    the bounds are pinned to the first and last position actually selected, and a
    single position takes the ascending unit-step form whatever step reached it,
    since a step spans nothing there. The bound is ``None`` where a descending
    selection runs down to position 0, since no integer sits before it: that is
    the one case a stop cannot state, and it is why both the rendered name and
    the storage slice omit it there.
    """
    start = selected[0]
    if len(selected) == 1:
        return start, start + 1, 1
    stop = selected[-1] + (1 if selected.step > 0 else -1)
    return start, None if stop < 0 else stop, selected.step


def _as_storage_slice(indexer: int | range) -> int | slice:
    """One axis indexer as the storage seam takes it: a position or a slice.

    A ``range`` becomes the slice selecting the same positions in the same order,
    so applying it to a list, a numpy array, or a JAX array reproduces the
    selection exactly — including a descending one, which is spelled with an
    omitted stop rather than a from-the-end bound.
    """
    if isinstance(indexer, int):
        return indexer
    if not indexer:
        return slice(0, 0, 1)
    start, stop, step = _bounds(indexer)
    return slice(start, stop, step)


def _whole_of(spec: BatchSpec) -> tuple[range, ...]:
    """The selection of a batch that selects all of itself: every axis whole."""
    return tuple(range(size) for size in spec.batch_shape)


def _render_axis(entry: int | range) -> str:
    """One axis of a selection: its position, or the positions it spans.

    A span is rendered so that reading it back selects the same positions in the
    same order, which is what lets a derived name be read as an index.
    """
    if isinstance(entry, int):
        return str(entry)
    if not entry:
        return "0:0"
    start, stop, step = _bounds(entry)
    bound = "" if stop is None else str(stop)
    if step == 1:
        return f"{start}:{bound}"
    return f"{start}:{bound}:{step}"


def _render_index(root_spec: BatchSpec, selection: tuple[int | range, ...]) -> str:
    """A selection as ``level=positions`` per touched level, in level order.

    Levels selected whole are left out, and the levels that remain appear in the
    batch's own order rather than the order they were indexed in, so the reading
    identifies the object and not the route to it. A level of several axes
    renders its axes as a tuple. The result is empty when the selection is the
    whole batch.
    """
    parts: list[str] = []
    start = 0
    for level_name, group in zip(root_spec.level_names, root_spec.axis_groups, strict=True):
        entries = selection[start : start + len(group)]
        start += len(group)
        # A level selected whole is left out, an axis counting as whole only when
        # it spans every position *in order* — a reversal is a selection, not a
        # no-op, so ``range(size)`` is compared rather than the count of positions.
        if all(entry == range(size) for entry, size in zip(entries, group, strict=True)):
            continue
        rendered = tuple(_render_axis(entry) for entry in entries)
        positions = rendered[0] if len(rendered) == 1 else f"({', '.join(rendered)})"
        parts.append(f"{level_name}={positions}")
    return ", ".join(parts)
