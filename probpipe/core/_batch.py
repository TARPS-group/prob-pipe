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

**A view is labeled by what it selects.** A view's label is derived from the batch
it was taken from and the positions it selects, naming the level each selection
addresses: selecting chain 0 of ``posterior`` yields the label
``"posterior[chain=0]"``, and its draw 7 yields ``"posterior[chain=0, draw=7]"``.
Levels selected whole are left out, and the levels that appear are listed in the
batch's own order, so a derived label is a function of what the view selects: two
ways of indexing one selection read alike, and two selections never do. The
batch's label is grouped before the selection as design II.4 states, so a
product reads as one label, as in ``"(x·y)[sample=0:2]"``.

A view receives its label when selected. Renaming its levels preserves that
label, and the new level names appear in the labels of later selections.

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
carried by its label. Element provenance is :meth:`Batch._element_at`'s own,
since only that hook knows whether it built the element or borrowed it.

See design II.5.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from math import prod
from types import MappingProxyType
from typing import Any, Self, cast

from .._messages import count, unknown_names
from ._expression import (
    Collapse,
    Expression,
    Indexed,
)
from ._record_spec import RecordSpec, _check_kind_of
from ._repr import (
    call_repr,
    format_levels,
    public_class_name,
    term_repr,
    type_name,
)
from ._shapes import LevelsLike, ShapeLike, _as_dim, _as_levels
from ._spec_base import OpaqueSpec, _agree, _unify_array_shape, _unify_specs
from ._specs import TermSpec, _check_component_name
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

    ``BatchSpec(element_spec, **levels)`` declares a batch whose elements satisfy
    *element_spec*, with one keyword per level mapping the level's name to the
    shape of its axes, outermost level first::

        BatchSpec(NumericArraySpec(()), chain=4, draw=1000)
        BatchSpec(NumericArraySpec(()), grid=(3, 4))       # one level of two axes
        BatchSpec(NumericArraySpec(("n",)), draw="S")      # S draws, S unbound
        BatchSpec(OpaqueSpec(), {"my level": 2})           # a name no keyword spells

    Each level's shape is read as :class:`NumericArraySpec` reads ``shape``, so a
    single int or str is one axis: ``draw=4`` is ``draw=(4,)``. A level name that
    cannot be written as a keyword argument, such as ``"my level"`` or
    ``"class"``, is given in a mapping passed positionally instead, and the two
    forms are not combined. :attr:`levels`
    returns the mapping, so ``BatchSpec(other_spec, batch.spec.levels)`` declares
    another element type over the same levels.

    The stored form is :attr:`axis_groups` and :attr:`level_names`: the batch
    axes tiled into named levels as :class:`Batch` describes them. Level names
    are unique within a batch, and ``batch_shape`` / ``batch_size`` read off the
    tiling.

    This is the single stored source of a batch's type, so it specifies the
    *collection* rather than one element, and :class:`Batch` keeps no second copy
    of the multiplicity: its shape and level accessors read the stored spec.

    Parameters
    ----------
    element_spec : TermSpec
        What every element of the batch satisfies, including numeric-array
        and opaque kinds. Positional-only.
    levels : Mapping of str to shape, optional
        The levels as a mapping from level name to shape, outermost first.
        Positional-only, and given only when no level is given as a keyword.
    **level_shapes : int, str, or sequence of int or str
        The levels as keywords, outermost first, each mapping a level name to the
        shape of its axes. A level name follows the rule for component names, so
        it is any non-empty string without ``/``. Every level holds at least one
        axis, and each axis size is a non-negative integer or a symbolic
        dimension name, which is a Python identifier.

    Attributes
    ----------
    axis_groups : tuple of tuple of int or str
        The axis sizes each level holds, in order, outermost level first.
    level_names : tuple of str
        One name per level, aligned with ``axis_groups``.
    batch_shape : tuple of int or str
        The batch axes, flat: the concatenation of ``axis_groups``.
    batch_size : int
        The total element count, ``prod(batch_shape)``.

    Raises
    ------
    TypeError
        If ``element_spec`` is not a :class:`TermSpec`; both a mapping and
        keywords are given; ``levels`` is not a mapping, or is given as a keyword
        holding a mapping; a key of the mapping is not a str; or a level's shape is
        not an int, a str, or a sequence of them.
    ValueError
        If no level is given, a level holds no axes, an axis size is negative or
        is a name that is not a Python identifier, or a level name is empty or
        contains ``/``.

    Notes
    -----
    A batch whose elements name no kind is specified all the same: a raw-value
    ``element_spec`` is as well formed here as a term spec, which is what lets a
    batch of opaque values carry a term spec of its own.

    An axis size may be a **symbolic dimension name** instead of an integer, as
    a ``NumericArraySpec`` shape entry may, so that a declaration can fix the number of
    levels while deferring how many elements each holds — "returns a batch of
    ``S`` draws" before ``S`` is known. The names share one scope with the
    element's schema, so a batch of ``("n",)`` over arrays of shape ``("n",)`` is
    square by declaration. Only the *multiplicity* must be concrete for a live
    :class:`Batch`, which holds elements at positions; an element's own schema may
    stay polymorphic, since how many elements there are is a different question
    from what one of them looks like.

    A duplicate level name cannot be written in either form, and
    :meth:`Batch.with_level_names` raises on a collision, since an operation that
    mints a level takes the name to give it.
    """

    # ``init=False``: the constructor takes levels by name rather than these
    # fields, so ``dataclasses.replace`` refuses them; ``copy.replace`` and
    # ``_replace`` rebuild a spec from them instead.
    element_spec: TermSpec = field(init=False)
    axis_groups: tuple[tuple[int | str, ...], ...] = field(init=False)
    level_names: tuple[str, ...] = field(init=False)

    def __init__(
        self,
        element_spec: TermSpec,
        levels: LevelsLike | None = None,
        /,
        **level_shapes: ShapeLike,
    ) -> None:
        if isinstance(level_shapes.get("levels"), Mapping) and levels is None:
            raise TypeError(
                "BatchSpec takes its levels mapping as the second positional argument, "
                "got levels= as a keyword"
            )
        names, groups = _as_levels(levels, level_shapes, what="BatchSpec")
        self._init_fields(element_spec, groups, names)

    @classmethod
    def _from_groups(
        cls,
        element_spec: TermSpec,
        axis_groups: Iterable[Iterable[int | str]],
        level_names: Iterable[str],
    ) -> BatchSpec:
        """The spec over aligned axis groups and level names, as the library holds them.

        The constructor takes levels by name; an operation that already holds a
        tiling, such as a batch's own ``axis_groups`` and ``level_names``, builds
        its spec here instead.
        """
        spec = object.__new__(cls)
        spec._init_fields(
            element_spec, tuple(tuple(group) for group in axis_groups), tuple(level_names)
        )
        return spec

    def _replace(
        self,
        *,
        element_spec: TermSpec | None = None,
        axis_groups: Iterable[Iterable[int | str]] | None = None,
        level_names: Iterable[str] | None = None,
    ) -> BatchSpec:
        """This spec with the given parts replaced and the rest kept."""
        return BatchSpec._from_groups(
            self.element_spec if element_spec is None else element_spec,
            self.axis_groups if axis_groups is None else axis_groups,
            self.level_names if level_names is None else level_names,
        )

    def __replace__(self, **changes: Any) -> BatchSpec:
        """This spec with the given fields replaced, for ``copy.replace``.

        Parameters
        ----------
        **changes : Any
            New values of ``element_spec``, ``axis_groups``, or ``level_names``.

        Returns
        -------
        BatchSpec
            The rebuilt spec, checked as a constructed one is.

        Raises
        ------
        TypeError
            If a change names any other field.
        """
        unknown = sorted(set(changes) - {"element_spec", "axis_groups", "level_names"})
        if unknown:
            raise TypeError(
                f"copy.replace() can change element_spec, axis_groups, and level_names of a "
                f"BatchSpec, got {', '.join(unknown)}"
            )
        return self._replace(**changes)

    def _init_fields(
        self,
        element_spec: TermSpec,
        axis_groups: tuple[tuple[Any, ...], ...],
        level_names: tuple[str, ...],
    ) -> None:
        """Check the parts and store them; the one place a spec's fields are set."""
        if not isinstance(element_spec, TermSpec):
            raise TypeError(
                f"BatchSpec element_spec must be a TermSpec, got {type_name(element_spec)}"
            )
        if not axis_groups:
            raise ValueError("BatchSpec must have at least one level, such as draw=4")
        if len(level_names) != len(axis_groups):
            raise ValueError(
                f"BatchSpec has {count(len(level_names), 'level name')} but "
                f"{count(len(axis_groups), 'axis group')}"
            )
        for level_name, group in zip(level_names, axis_groups, strict=True):
            if not isinstance(level_name, str):
                raise TypeError(
                    f"BatchSpec level names must be str, got {type_name(level_name)} {level_name!r}"
                )
            _check_component_name(level_name, context="BatchSpec level names")
            if not group:
                raise ValueError(
                    f"BatchSpec level {level_name!r} must have at least one axis, got ()"
                )
        tiled = tuple(
            tuple(_as_dim(size, what=f"BatchSpec level {name!r} entry") for size in group)
            for name, group in zip(level_names, axis_groups, strict=True)
        )
        if len(set(level_names)) != len(level_names):
            raise ValueError(f"BatchSpec level names must be unique, got {level_names}")

        object.__setattr__(self, "element_spec", element_spec)
        object.__setattr__(self, "axis_groups", tiled)
        object.__setattr__(self, "level_names", level_names)

    @property
    def levels(self) -> Mapping[str, tuple[int | str, ...]]:
        """The levels as a read-only mapping from level name to its axis sizes, outermost first.

        ``BatchSpec(element_spec, spec.levels)`` rebuilds a spec with the same
        levels, so another element type is declared over them.
        """
        return MappingProxyType(dict(zip(self.level_names, self.axis_groups, strict=True)))

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
            reason a polymorphic ``NumericRecordSpec`` has no flat layout.
        """
        if self.free_axis_dims:
            raise ValueError(_unbound_axis_sizes("batch_size is undefined", self.free_axis_dims))
        return prod(cast(tuple[int, ...], self.batch_shape))

    @property
    def free_dims(self) -> frozenset[str]:
        """The unbound dimensions of the element's schema and of the multiplicity.

        One scope, as everywhere: a name shared between an axis size and the
        element's schema is one dimension, so a batch declared as ``("n",)`` of
        arrays of shape ``("n",)`` states that it is square.
        """
        return self.element_spec.free_dims | self.free_axis_dims

    @property
    def free_axis_dims(self) -> frozenset[str]:
        """The unbound dimensions of the *multiplicity* alone.

        What a live batch requires bound, since it holds elements at positions.
        An element's own schema may stay polymorphic: how many elements there are
        is a different question from what one of them looks like.
        """
        return frozenset(size for size in self.batch_shape if isinstance(size, str))

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> BatchSpec:
        """This spec with both its element schema and its axis sizes substituted."""
        return BatchSpec._from_groups(
            self.element_spec._substitute_dims(bindings),
            tuple(
                tuple(bindings.get(size, size) if isinstance(size, str) else size for size in group)
                for group in self.axis_groups
            ),
            self.level_names,
        )

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        """Bind the declared multiplicity and element schema from a live *value*.

        A live :class:`Batch` carries a concrete spec of its own, so it binds
        through that spec.
        """
        actual = getattr(value, "spec", None)
        if not isinstance(actual, BatchSpec):
            raise ValueError(f"{path} must be a batch matching {self!r}, got {type_name(value)}")
        _check_kind_of(actual, value, self, path)
        self._bind_dims_from_spec(actual, bindings, path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        """Bind the declared axis sizes and element schema against *actual*'s own.

        The multiplicity binds like an array shape: a symbolic axis size takes the
        actual size, and a name already bound must agree. The element spec binds
        by the same rule one level in. Both use the caller's *bindings*, so an
        axis size and an element dimension sharing a name are one dimension — a
        batch of ``("n",)`` over arrays of shape ``("n",)`` binds ``n`` once and
        refuses a batch that is not square.

        The level tiling is structure rather than size, so how many levels there
        are, how many axes each holds, and what they are called are checked
        rather than bound.
        """
        if not isinstance(actual, BatchSpec):
            return False
        if self.level_names != actual.level_names:
            raise ValueError(
                f"{path} has levels {list(actual.level_names)}, expected {list(self.level_names)}"
            )
        declared_arity = [len(group) for group in self.axis_groups]
        actual_arity = [len(group) for group in actual.axis_groups]
        if declared_arity != actual_arity:
            raise ValueError(
                f"{path} has {actual_arity} axes per level, expected {declared_arity} "
                f"from axis_groups={self.axis_groups!r}"
            )
        _unify_array_shape(self.batch_shape, actual.batch_shape, bindings, path)
        _unify_specs(self.element_spec, actual.element_spec, bindings, path)
        return True

    def __repr__(self) -> str:
        """The constructor call: the element spec, then one keyword per level.

        A level of one axis shows its size alone, as ``draw=4``, and a level name
        that no keyword spells puts the levels in one ``**{...}`` argument.
        """
        return call_repr(
            "BatchSpec",
            [repr(self.element_spec)],
            [
                (name, repr(group[0] if len(group) == 1 else group))
                for name, group in zip(self.level_names, self.axis_groups, strict=True)
            ],
        )

    def is_valid(self, value: Any) -> bool:
        """Whether *value* is a :class:`Batch` whose own spec equals this one.

        An opaque type this spec leaves open admits the type a batch inferred
        from its values, in the elements and in any record field of them.
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
        return _admits(self, spec)


def _admits(declared: TermSpec, actual: TermSpec) -> bool:
    """Whether *actual* is *declared*, with an opaque type filled in where *declared* leaves it open."""
    if declared == actual:
        return True
    if isinstance(declared, OpaqueSpec) and isinstance(actual, OpaqueSpec):
        return declared.type in (None, actual.type) and _agree(declared.meta, actual.meta)
    if isinstance(declared, RecordSpec) and isinstance(actual, RecordSpec):
        return list(declared.keys()) == list(actual.keys()) and all(
            _admits(declared[key], actual[key]) for key in declared
        )
    if isinstance(declared, BatchSpec) and isinstance(actual, BatchSpec):
        return (declared.axis_groups, declared.level_names) == (
            actual.axis_groups,
            actual.level_names,
        ) and _admits(declared.element_spec, actual.element_spec)
    return False


class Batch[E](TrackedTerm, ABC):
    """A tracked nd collection of elements of a common type.

    The batch axes are grouped into named **levels**, and indexing them returns a
    *view* — an element once the selection reaches one, a sub-batch otherwise —
    whose label records what it selected. Index by position with ``[]``, by level
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
    element_spec : TermSpec
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

    A batch is immutable, by :class:`~probpipe.core._immutable.Immutable`:
    assignment and deletion raise, and ``pickle`` / ``copy`` restore its state
    around that guard.
    """

    __slots__ = (
        "_expression",
        "_label",
        "_label_collapse",
        "_provenance",
        "_root_expression",
        "_root_selection",
        "_root_spec",
        "_spec",
    )

    # The three ``_root_*`` slots derive a view's label: they hold the expression
    # and spec of the batch a derivation starts from, and which of *that* batch's
    # positions this object selects — one entry per root axis, an integer where an
    # axis has been dropped and a range of positions where one is kept. ``_label``
    # and the expression are read off them, which is what makes two routes to one
    # selection agree: the
    # reading is a function of the selection, not of the calls that reached it.
    #
    # They are never ``None``, not even on a batch nobody has indexed. Such a batch
    # is its own root and selects all of itself, so its derived label is the label it
    # was given, and composing a further selection needs no special case at the
    # head of the chain. Leaving them unset for a non-view would put a
    # "means everything" reading on ``None`` — the reading positional ``[]``
    # refuses — and every site that composes or renders a selection would carry a
    # branch for it. :meth:`with_label` re-roots a view: a user-given label replaces
    # the derivation and discards the selection accumulated before it.

    # -- construction -------------------------------------------------------

    def _init_batch(
        self,
        spec: BatchSpec,
        *,
        label: str,
        provenance: Provenance | None = None,
    ) -> None:
        """Store the batch's *spec* and identity (constructor helper).

        Assigns the state via ``object.__setattr__`` so immutable hosts can call
        this from their constructor, and delegates identity to
        :meth:`TrackedTerm._init_tracked`. The level invariants are the spec's
        own and are checked when it is constructed. A subclass declares only its
        storage in ``__slots__``; this class declares the batch's own state and
        the identity slots it hosts.

        The batch becomes the **root** of the labels its views derive: it selects
        all of itself, so its derived label is its own. A view built by
        :meth:`at_levels` or ``[]`` is re-pointed at its parent's root
        afterwards, so a derived label depends only on what the view selects.

        Parameters
        ----------
        spec : BatchSpec
            The batch's type, whose axis sizes must all be integers.
        label : str
            The batch's label, from which its views derive theirs.
        provenance : Provenance, optional
            How this batch was produced.

        Raises
        ------
        TypeError
            If *spec* is not a :class:`BatchSpec`.
        ValueError
            If an axis size of *spec* is an unbound symbolic dimension.
        """
        if not isinstance(spec, BatchSpec):
            raise TypeError(f"spec must be a BatchSpec, got {type(spec).__name__}")
        if spec.free_axis_dims:
            raise ValueError(_unbound_axis_sizes("cannot build a batch", spec.free_axis_dims))
        object.__setattr__(self, "_spec", spec)
        self._init_tracked(label, provenance=provenance)
        object.__setattr__(self, "_root_expression", self._expression)
        object.__setattr__(self, "_root_spec", spec)
        object.__setattr__(self, "_root_selection", _whole_of(spec))

    # -- the specification --------------------------------------------------

    @property
    def spec(self) -> BatchSpec:
        """This batch's own specification, at the family kind."""
        return self._spec

    @property
    def element_spec(self) -> TermSpec:
        """The specification every element satisfies — a view on :attr:`spec`."""
        return self._spec.element_spec

    @property
    def _view_type(self) -> type:
        """The class a view over this batch's own storage takes.

        This class, ordinarily: a view is the same kind of batch over the same
        elements, which is what lets a batch be its own view type. A subclass
        holding state beyond the batch's — a :class:`~probpipe.record.Design` and
        its marginals — overrides this with the class that state belongs to,
        since a view carries none of it and would otherwise claim to answer for
        it.
        """
        return type(self)

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
            level names, preserving its own label.

        Raises
        ------
        KeyError
            If a name to rename is not a level of this batch.
        ValueError
            If a level is renamed twice with different names, or a new name is
            empty, contains ``/``, collides with a level that is being kept, is
            the target of two renames, or belongs to a dropped root level still
            used to label subsequent selections from a view.
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
        unknown = [old for old in renames if old not in self.level_names]
        if unknown:
            raise KeyError(unknown_names("level", unknown, self.level_names))
        for new in renames.values():
            # Type before emptiness: None, 0 and [] are all falsy, and reporting
            # them as empty names would describe the wrong problem.
            if not isinstance(new, str):
                raise TypeError(f"level names must be strings, got {type(new).__name__}: {new!r}")
            _check_component_name(new, context="level names")

        renamed = tuple(renames.get(old, old) for old in self.level_names)
        if len(set(renamed)) != len(renamed):
            raise ValueError(
                f"renaming would duplicate a level name: {self.level_names} -> {renamed}"
            )
        return self._with_level_names(renamed)

    def _store_expression(
        self, expression: Expression, rendering: tuple[str, Collapse | None] | None = None
    ) -> None:
        """Store *expression* and its label, and make the batch the root its view labels derive from.

        A new expression starts a new view root, so the batch selects all of
        itself: its own label is the expression's, and a view of it reads
        ``label[level=...]`` rather than carrying any selection the original had
        accumulated. Relabeling is therefore the way to rename a level a view
        derives its label from but no longer carries, which
        :meth:`with_level_names` refuses.

        Parameters
        ----------
        expression : Expression
            The batch's new expression, preserved by later transforms.
        rendering : tuple of (str, Collapse or None), optional
            The label and what it left out, for a caller that rendered it already.
        """
        super()._store_expression(expression, rendering)
        object.__setattr__(self, "_root_expression", expression)
        object.__setattr__(self, "_root_spec", self._spec)
        object.__setattr__(self, "_root_selection", _whole_of(self._spec))

    # -- reading ------------------------------------------------------------

    def __repr__(self) -> str:
        """The public class, the label, the levels as a mapping, and the elements' structure.

        A level of one axis reports its size, and a level of several the tuple of
        its sizes, so a two-level batch of chains and draws reads
        ``levels={'chain': 4, 'draw': 1000}``. The elements' structure is their
        spec, or their field paths for a batch of records.

        Notes
        -----
        No element is read, so the cost is the same on a batch of any size and
        nothing here can raise from storage. That matters beyond convenience:
        :meth:`TrackedTerm.with_provenance` interpolates the batch into its
        write-once error, so a ``repr`` that could fail would fail there.
        """
        levels = ("levels", format_levels(self.level_names, self.axis_groups))
        return term_repr(
            public_class_name(type(self)),
            self._displayed_label(),
            [levels, *self._element_repr_arguments()],
        )

    def _element_repr_arguments(self) -> list[tuple[str, str]]:
        """The elements' structure as the repr shows it: their spec, by default."""
        return [("element_spec", repr(self.element_spec))]

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

        Parameters
        ----------
        key : int, slice, str, or tuple
            An integer, a slice, or a tuple of them addresses the batch axes, and a
            string or a tuple of strings addresses a field path.

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
            f"cannot mix field names and axis indices in one key, got {key!r}; index the "
            f"axes and the fields in separate steps"
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

        A level name that is no Python identifier is passed in a mapping, as
        ``at_levels(**{"my level": 0})``. The receiver is positional-only, so every
        level name is addressable as a keyword — including one that happens to
        spell a parameter of this method.

        Parameters
        ----------
        **levels : int, slice, None, or tuple
            One indexer per named level, keyed by the level's name.

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
        unknown = [name for name in levels if name not in self.level_names]
        if unknown:
            raise KeyError(unknown_names("level", unknown, self.level_names))

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
                        f"level {level_name!r} has {count(len(group), 'axis', 'axes')} but "
                        f"got {count(len(indexers), 'indexer')}"
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

    #: A subclass returning borrowed elements sets this so selection never writes to them.
    _borrows_elements = False

    @abstractmethod
    def _element_at(self, index: tuple[int, ...], *, label: str) -> E:
        """The single element at a fully-integer positional *index*, as a view labeled *label*.

        *label* is the identity this class derived for the element view. A batch
        that *materializes* an element, as columnar storage builds a row, builds
        a term of the element kind under *label* and gives it this batch's
        provenance through :meth:`_inherit_provenance`. A batch that *stores*
        its elements returns a view of the stored object under *label*: a copy of
        a stored tracked term that shares its representation, or the stored
        value wrapped as a term of the element kind. That view's provenance
        records this batch and the stored term, and the stored object keeps its
        own label and provenance, since the caller may still hold it.

        Provenance is this hook's own, because only it knows whether the element
        was built or borrowed. An element built under *label* alone then
        carries the expression of the selection, as ``(mu ~ prior)[sample=0]``,
        and a stored object returned as it is keeps its own.
        """

    def _element_call(self, index: tuple[int, ...]) -> Expression | None:
        """The call of the row at *index* of a batch of lifted laws, which the element's notation reads.

        A batch that stores the laws a function lifted over laws gives returns
        the stored law's call, as ``f(mu ~ prior, 4.0)``; this default, and any
        other batch, returns ``None``.
        """
        return None

    @abstractmethod
    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, label: str) -> Self:
        """A view over the sub-batch at a partial positional *index*.

        *index* is one entry per axis of this batch: a resolved position for an
        axis being dropped, or a slice selecting the positions a kept axis spans,
        **in the order the view presents them** — a descending slice for a
        reversed selection, which storage must honor rather than re-sort, since
        the view's derived labels are stated in that order.

        *spec* is the view's own specification: the same ``element_spec`` over
        the surviving levels, with every integer-indexed axis already removed.
        *label* is the derived identity. A subclass
        stores both as given rather than recomputing either; the root slots a
        further view derives its label from are re-pointed at this view's own
        root afterwards.

        **A view, not a copy.** The result shares this batch's storage wherever
        the storage affords it — a numpy or JAX slice, or a list re-indexed over
        the same objects — so that a batch stays the single source of its elements
        and a selection does not pay for what it selects. Selecting the whole batch
        reaches this hook like any other selection, which costs nothing once the
        result is a view; it is not short-circuited to ``self``, since the view
        carries a label and provenance of its own and ``self`` already has both.
        """

    def _inherit_provenance[T](self, produced: T) -> T:
        """Give something this batch *produced* the batch's own provenance.

        Selecting is not a step in a computation. Nothing is computed by reading
        one position out of a collection, so no node records the reading, and
        what a selection carries is the lineage of the batch it came out of.
        *Which* position was selected is carried by the label, which states it
        in full, as ``posterior[chain=0, draw=7]`` does. Nothing is lost by not
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
        no selection step, and only the label says a view is one.
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
            f"cannot index {public_class_name(type(self))} by {addressed!r}: its elements have "
            f"no named fields. Use integers or slices for the batch axes, or at_levels() to "
            f"index by level name"
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
        rendered = _render_index(self._root_spec, selection)
        if selection == self._root_selection:
            expression = self._expression
        elif rendered:
            expression = Indexed(self._root_expression, rendered)
        else:
            expression = self._root_expression
        label, collapse = expression.label_rendering()

        dropped = tuple(i for i in normalized if isinstance(i, int))
        if len(dropped) == len(shape):
            call = self._element_call(dropped)
            if call is not None and isinstance(expression, Indexed):
                expression = replace(expression, element=call)
            element = self._element_at(dropped, label=label)
            if isinstance(element, TrackedTerm) and not self._borrows_elements:
                # An element built under the derived label carries the selection,
                # and a stored law keeps the paths it holds fixed, after the batch's.
                held = expression.with_fixed(element._expression.fixed_paths())
                _assign_expression(element, held, (label, collapse))
            return element

        groups, names = self._surviving_levels(normalized)
        spec = self._spec._replace(axis_groups=groups, level_names=names)
        view = self._sub_batch_at(
            tuple(_as_storage_slice(i) for i in normalized), spec=spec, label=label
        )
        _assign_expression(view, expression, (label, collapse))
        object.__setattr__(view, "_root_expression", self._root_expression)
        object.__setattr__(view, "_root_spec", self._root_spec)
        object.__setattr__(view, "_root_selection", selection)
        return self._inherit_provenance(view)

    def _compose_selection(self, normalized: list[int | range]) -> tuple[int | range, ...]:
        """This view's selection composed with *normalized*, in root coordinates.

        Each entry of :attr:`_root_selection` is one root axis: an integer for an
        axis already dropped, or the ``range`` of root positions a kept axis
        still spans, in the order the axis presents them. Composing in root
        coordinates makes a derived label a function of the object rather than of
        the indexing that produced it, so two ways of indexing one selection
        compose to the same tuple.

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
        groups: list[tuple[int | str, ...]] = []
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
        return cast(tuple[tuple[int, ...], ...], tuple(groups)), tuple(names)

    def _with_level_names(self, level_names: tuple[str, ...]) -> Self:
        """A shallow copy specified over *level_names*, sharing shape and elements.

        Renaming touches no axes and no elements, so the default is a shallow
        copy carrying a renamed spec. :meth:`TrackedTerm._shallow_copy` assigns
        through ``object.__setattr__``, which is what makes this safe to define
        here: it runs no ``__init__`` and survives this class's immutability
        guard, so nothing is assumed about a subclass's constructor.

        The root's level names are updated for subsequent indexing, while the
        copy keeps its current label. The copy carries no provenance beyond the
        rename: the record of how the batch it was renamed from arose belongs to
        that batch.

        Override only when a subclass caches something derived from the level
        names, since a shallow copy would carry the stale cache; call ``super``
        for the renamed copy rather than reassigning ``_spec`` by hand.
        """
        repinned = dict(zip(self.level_names, level_names, strict=True))
        root_names = tuple(repinned.get(name, name) for name in self._root_spec.level_names)
        if len(set(root_names)) != len(root_names):
            taken = sorted({name for name in root_names if root_names.count(name) > 1})[0]
            old = next((o for o, n in repinned.items() if n == taken and o != n), taken)
            raise ValueError(
                f"cannot rename level {old!r} to {taken!r}: this view {self.label} was indexed "
                f"from a level named {taken!r}. Choose another name, or relabel the view with "
                f"with_label() first"
            )

        renamed = self._shallow_copy()
        object.__setattr__(renamed, "_spec", self._spec._replace(level_names=level_names))
        object.__setattr__(renamed, "_root_spec", self._root_spec._replace(level_names=root_names))
        object.__setattr__(renamed, "_provenance", None)
        renamed.with_provenance(Provenance.create("with_level_names", parents=[self]))
        return renamed


def _cannot_rebuild(kind: str) -> str:
    """The opening of the message for a batch a pytree transform left unrebuildable.

    Each refusal appends its reason after a colon, so every one of them reads
    alike: ``"cannot rebuild RecordBatch after a pytree transform: ..."``.
    """
    return f"cannot rebuild {kind} after a pytree transform"


def _changed_batch_shape(refused: str, before: tuple[int, ...], after: tuple[int, ...]) -> str:
    """The message for a pytree transform that changed some batch axes but not all.

    *refused* is the opening :func:`_cannot_rebuild` gives.
    """
    return (
        f"{refused}: the batch shape changed from {before} to {after}. A transform must keep "
        f"every batch axis or remove all of them; to select elements, index the batch instead"
    )


def _uncastable_dtype(subject: str, dtype: Any, declared: Any) -> str:
    """The message for stored values whose dtype the declared dtype does not admit.

    *subject* names the values, such as ``"RecordBatch: field 'x'"``.
    """
    return (
        f"{subject} has dtype {dtype}, which cannot be cast to the declared {declared} "
        f"(only same-kind casts are allowed)"
    )


def _unbound_axis_sizes(failed: str, dims: Iterable[str]) -> str:
    """The message for a batch whose axis sizes include unbound symbolic dimensions.

    *failed* names what could not happen, such as ``"cannot build a batch"``.
    """
    names = sorted(dims)
    which = f"axis size {names[0]} is" if len(names) == 1 else f"axis sizes {', '.join(names)} are"
    sizes = ", ".join(f"{name}=..." for name in names)
    it = "it" if len(names) == 1 else "them"
    return f"{failed}: {which} symbolic; bind {it} first with with_dim_sizes({sizes})"


def _axis_groups_for(
    shape: tuple[int, ...],
    names: tuple[str, ...],
    axes_per_level: tuple[int, ...] | None,
    *,
    kind: str,
) -> tuple[tuple[int, ...], ...]:
    """The axis groups for *shape*, from how many axes each level holds.

    *axes_per_level* gives one count per level, outermost first, and they must
    account for every axis the elements are stored in. The sizes are then read off
    *shape* rather than restated: the elements are present, so the shape is
    already known, and the only thing a caller can tell this function is where the
    boundaries between levels fall. ``None`` puts one axis on each level.

    A :class:`BatchSpec` states the sizes instead, and is right to — a
    *declaration* may leave them symbolic, fixing the number of levels before the
    counts are known. A live batch holds elements at positions, so it cannot.
    """
    if axes_per_level is None:
        if len(names) != len(shape):
            raise ValueError(
                f"{kind} got batch shape {shape} but {count(len(names), 'level name')} "
                f"{list(names)}; give one name per axis, or pass axes_per_level to group "
                f"axes into levels"
            )
        return tuple((size,) for size in shape)

    counts = axes_per_level
    if len(counts) != len(names):
        raise ValueError(
            f"axes_per_level must give one count per level, got {counts} for level names "
            f"{list(names)}"
        )
    if sum(counts) != len(shape):
        raise ValueError(
            f"axes_per_level {counts} covers {count(sum(counts), 'axis', 'axes')}, but "
            f"{kind} got batch shape {shape} with {count(len(shape), 'axis', 'axes')}"
        )
    groups, at = [], 0
    for size in counts:
        groups.append(shape[at : at + size])
        at += size
    return tuple(groups)


def _batch_axis_count(names: tuple[str, ...], axes_per_level: tuple[int, ...] | None) -> int:
    """How many batch axes the levels hold: the sum of *axes_per_level*, or one per name.

    A constructor that infers its element spec reads the event axes as the axes
    past these, so this count fixes where the batch axes end.

    Parameters
    ----------
    names : tuple of str
        The level names, of which only the count is read.
    axes_per_level : tuple of int or None
        The axis count of each level, outermost first, as ``_as_axis_counts``
        returns it; ``None`` gives each level one axis.

    Returns
    -------
    int
        The number of batch axes, which lead each stored array.
    """
    if axes_per_level is None:
        return len(names)
    return sum(axes_per_level)


def _ranks_of(groups: Iterable[Iterable[Any]]) -> tuple[int, ...]:
    """*groups* as the axis counts a constructor takes.

    The bridge for the operations that already hold grouped sizes — an
    aggregation composing a sweep's levels with a row's — and need to state the
    same partition to a constructor, which reads the sizes from the elements.
    """
    return tuple(len(tuple(group)) for group in groups)


def _normalize_indexer(
    indexer: Any, size: int, axis: int, shape: tuple[int, ...], where: str | None = None
) -> int | range:
    """One axis indexer as a resolved integer or the ``range`` of positions it selects.

    An omitted axis and ``:`` both mean the whole axis. An integer is resolved
    against *size* (so a negative index names the same position as its
    non-negative twin, and both derive the same label); a slice is resolved to the
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
            raise ValueError(f"slice step cannot be zero ({_location(axis, shape, where)})")
        try:
            return range(*indexer.indices(size))
        except TypeError:
            # ``slice.indices`` would raise this itself, naming neither the batch
            # nor the axis -- and a bound computed with ``/`` is a float, which is
            # the ordinary way to arrive here.
            raise TypeError(
                f"slice bounds must be integers, got {indexer!r} ({_location(axis, shape, where)})"
            ) from None
    if indexer is None:
        raise TypeError(
            f"cannot index a batch axis with None ({_location(axis, shape, where)}); "
            f"use ':' for the whole axis"
        )
    if isinstance(indexer, bool):
        raise TypeError(
            f"cannot index a batch axis with a bool ({_location(axis, shape, where)}); use "
            f"an integer or a slice"
        )
    try:
        position = operator.index(indexer)
    except TypeError:
        raise TypeError(
            f"batch axes must be indexed by integers or slices, got "
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

    One form per set of positions, so a label stays a function of the selection:
    the bounds are set at the first and last position actually selected, and a
    single position takes the ascending unit-step form whatever step selected it,
    since a step spans nothing there. The bound is ``None`` where a descending
    selection runs down to position 0, since no integer precedes it: that is
    the one case a stop cannot state, and it is why both the rendered label and
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
    """The selection of a batch that selects all of itself: every axis whole.

    Reads a *live* batch's spec, whose axes are concrete — a symbolic
    multiplicity is refused at construction.
    """
    return tuple(range(size) for size in cast(tuple[int, ...], spec.batch_shape))


def _render_axis(entry: int | range) -> str:
    """One axis of a selection: its position, or the positions it spans.

    A span is rendered so that reading it back selects the same positions in the
    same order, so a derived label reads back as an index.
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
        if all(entry == range(cast(int, size)) for entry, size in zip(entries, group, strict=True)):
            continue
        rendered = tuple(_render_axis(entry) for entry in entries)
        positions = rendered[0] if len(rendered) == 1 else f"({', '.join(rendered)})"
        parts.append(f"{level_name}={positions}")
    return ", ".join(parts)


def _assign_expression(
    term: Any, expression: Expression, rendering: tuple[str, Collapse | None]
) -> None:
    """Give *term*, a view just selected, *expression* and its rendered label.

    *rendering* is the label and what it left out, as
    :meth:`~probpipe.core._expression.Expression.label_rendering` gives them. A
    view keeps the root it was selected from, so the assignment stores the
    expression without re-rooting it, as :meth:`Batch._store_expression` would.
    """
    label, collapse = rendering
    object.__setattr__(term, "_expression", expression)
    object.__setattr__(term, "_label", label)
    object.__setattr__(term, "_label_collapse", collapse)
