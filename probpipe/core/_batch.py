"""Batch — the generic multiplicity axis.

A :class:`Batch` holds *how many* objects there are, separately from *what one
object contains*. It is an nd collection of a common element type, tracked like
any other term, and it is deliberately **not** a named tree: ``len`` / ``iter``
/ ``batch_shape`` speak only about the batch axes, never about the structure
inside an element.

**Levels.** A batch's axes are partitioned into ordered *levels*, reported by
:attr:`axis_groups` — contiguous groups of axes, outermost first — with
:attr:`batch_shape` their flat concatenation. `N` laws of `S` draws each are
therefore ``(N,)`` of ``(S,)`` rather than one anonymous ``(N, S)``, and
anything stated over ``batch_shape`` (flat vectorization above all) applies to a
multi-level batch unchanged. Each level carries a name, so operations can align
batched operands by meaning rather than by position.

**Storage is the concrete class's business.** This module owns the level
algebra: the shape invariants, the naming rules, index normalisation for
:meth:`Batch.at_levels`, and the identity a view derives. A concrete batch
supplies the two hooks :meth:`Batch._element_at` and
:meth:`Batch._sub_batch_at`, which materialise an element or a sub-batch view
from a normalised positional index.

See design II.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from math import prod
from typing import Any, Self

from .provenance import Provenance
from .tracked import TrackedTerm

__all__ = ["Batch"]

# One indexer per level: a single value addresses the level's first axis, a
# tuple addresses its axes in order. ``None`` means the whole axis, as ``:``.
type LevelIndexer = int | slice | None | tuple[int | slice | None, ...]


class Batch[E](TrackedTerm, ABC):
    """A tracked nd collection of elements of a common type.

    Parameters are supplied by the concrete subclass through
    :meth:`_init_batch`; this class stores only the level metadata (the axis
    groups and their names) and the identity :class:`TrackedTerm` provides.

    Attributes
    ----------
    batch_shape : tuple of int
        The batch axes, the flat concatenation of :attr:`axis_groups`. Always
        non-empty: a batch has at least one batch axis.
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
    """

    __slots__ = ()

    # -- construction -------------------------------------------------------

    def _init_batch(
        self,
        axis_groups: Iterable[Iterable[int]],
        level_names: Iterable[str],
        *,
        name: str,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        """Initialize the level metadata and identity (constructor helper).

        Validates the level invariants, then assigns ``_axis_groups`` and
        ``_level_names`` via ``object.__setattr__`` so immutable hosts can call
        this from their constructor, and delegates identity to
        :meth:`TrackedTerm._init_tracked`.

        Raises
        ------
        ValueError
            If there are no batch axes, a level holds no axes, an axis size is
            negative, the number of names does not match the number of levels,
            or a level name is empty or duplicated.
        """
        groups = tuple(tuple(int(size) for size in group) for group in axis_groups)
        names = tuple(level_names)

        if not groups:
            raise ValueError("a Batch has at least one batch axis; axis_groups was empty")
        for group in groups:
            if not group:
                raise ValueError(f"every level holds at least one axis; got axis_groups={groups}")
            for size in group:
                if size < 0:
                    raise ValueError(f"axis sizes are non-negative; got axis_groups={groups}")
        if len(names) != len(groups):
            raise ValueError(
                f"level_names must name every level: {len(names)} names for {len(groups)} levels"
            )
        for level_name in names:
            if not level_name:
                raise ValueError("level names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"level names must be unique within a batch; got {names}")

        object.__setattr__(self, "_axis_groups", groups)
        object.__setattr__(self, "_level_names", names)
        self._init_tracked(name, name_is_auto=name_is_auto, provenance=provenance)

    # -- shape and levels ---------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch axes, flat: the concatenation of :attr:`axis_groups`."""
        return tuple(size for group in self._axis_groups for size in group)

    @property
    def batch_size(self) -> int:
        """The total element count, ``prod(batch_shape)``."""
        return prod(self.batch_shape)

    @property
    def axis_groups(self) -> tuple[tuple[int, ...], ...]:
        """``batch_shape`` tiled into levels, outermost level first."""
        return self._axis_groups

    @property
    def level_names(self) -> tuple[str, ...]:
        """One name per level, aligned with :attr:`axis_groups`."""
        return self._level_names

    def with_level_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self:
        """Rename levels ``old -> new``; shapes and elements are unchanged.

        The level counterpart of ``with_path_names``, which renames the *fields
        within* an element. Accepts a positional mapping, keyword pairs, or
        both.

        Raises
        ------
        KeyError
            If a name to rename is not a level of this batch.
        ValueError
            If a new name is empty, collides with a level that is being kept,
            or two renames target the same name.
        """
        renames: dict[str, str] = {**(mapping or {}), **kwargs}
        unknown = set(renames) - set(self._level_names)
        if unknown:
            raise KeyError(
                f"not levels of this batch: {sorted(unknown)}; have {list(self._level_names)}"
            )
        for new in renames.values():
            if not new:
                raise ValueError("level names must be non-empty")

        renamed = tuple(renames.get(old, old) for old in self._level_names)
        if len(set(renamed)) != len(renamed):
            raise ValueError(
                f"renaming would duplicate a level name: {self._level_names} -> {renamed}"
            )
        return self._with_level_names(renamed)

    @staticmethod
    def disambiguate_level_name(existing: Iterable[str], candidate: str) -> str:
        """A free level name for *candidate*, given the *existing* names.

        The rule an operation follows when it mints a level whose natural name
        is taken: append the smallest free integer suffix, so repeated sampling
        yields ``draw``, then ``draw2``, then ``draw3``. Renaming, by contrast,
        raises on a collision rather than suffixing — see
        :meth:`with_level_names`.
        """
        taken = set(existing)
        if candidate not in taken:
            return candidate
        suffix = 2
        while f"{candidate}{suffix}" in taken:
            suffix += 1
        return f"{candidate}{suffix}"

    # -- indexing -----------------------------------------------------------

    def __len__(self) -> int:
        """The leading batch axis, ``batch_shape[0]``."""
        return self.batch_shape[0]

    def __iter__(self) -> Iterator[E | Self]:
        """Iterate the leading batch axis, yielding views."""
        return (self[index] for index in range(len(self)))

    def __getitem__(self, index: Any) -> E | Self:
        """Index the leading batch axis, returning an element or a sub-batch view."""
        return self._at_axes((index,))

    def at_levels(self, **levels: LevelIndexer) -> E | Self:
        """Index by named level, returning an element or a sub-batch view.

        The by-name counterpart of positional ``[]``, and the level analogue of
        ``NamedTree.at_path``: a path addresses a position in a tree and returns
        a leaf or a subtree, while named level indexers address positions here
        and return an element or a sub-batch.

        Each indexer is an integer, a slice, ``None``, or a tuple of these
        addressing the level's axes in order. An integer drops its axis; a slice
        or ``None`` keeps it, ``None`` meaning the whole axis as ``:`` does. A
        shorter tuple fills the level's leading axes and leaves the rest whole,
        so a scalar ``draw=i`` on a two-axis ``draw`` level means
        ``draw=(i, None)``. A level not named is kept whole, and a level whose
        every axis is dropped is removed, yielding the inner batch or element
        just as positional indexing does.

        Raises
        ------
        KeyError
            If a keyword is not a level of this batch.
        ValueError
            If a level is given more indexers than it has axes.
        """
        unknown = set(levels) - set(self._level_names)
        if unknown:
            raise KeyError(
                f"not levels of this batch: {sorted(unknown)}; have {list(self._level_names)}"
            )

        axis_index: list[int | slice] = [slice(None)] * len(self.batch_shape)
        start = 0
        for level_name, group in zip(self._level_names, self._axis_groups, strict=True):
            if level_name in levels:
                indexers = _as_axis_indexers(levels[level_name])
                if len(indexers) > len(group):
                    raise ValueError(
                        f"level {level_name!r} has {len(group)} axes but got "
                        f"{len(indexers)} indexers"
                    )
                for offset, indexer in enumerate(indexers):
                    axis_index[start + offset] = indexer
            start += len(group)
        return self._at_axes(tuple(axis_index))

    # -- the concrete-storage seam ------------------------------------------

    @abstractmethod
    def _element_at(self, index: tuple[int, ...], *, name: str) -> E:
        """The single element at a fully-integer positional *index*.

        *name* is the identity this class derived for the element view. A
        tracked element takes it, marked auto-derived; an element that is a bare
        value has no identity to carry and ignores it.
        """

    @abstractmethod
    def _sub_batch_at(
        self,
        index: tuple[int | slice, ...],
        *,
        axis_groups: tuple[tuple[int, ...], ...],
        level_names: tuple[str, ...],
        name: str,
    ) -> Self:
        """A view over the sub-batch at a partial positional *index*.

        *axis_groups* and *level_names* are the surviving levels, with every
        integer-indexed axis already removed, and *name* is the derived identity.
        """

    # -- internals ----------------------------------------------------------

    def _at_axes(self, index: tuple[int | slice | None, ...]) -> E | Self:
        """Resolve a positional index over the batch axes to an element or view."""
        shape = self.batch_shape
        if len(index) > len(shape):
            raise IndexError(f"too many indices for batch_shape {shape}: got {len(index)}")

        normalised: list[int | slice] = []
        for axis, size in enumerate(shape):
            indexer = index[axis] if axis < len(index) else slice(None)
            if indexer is None:
                indexer = slice(None)
            if isinstance(indexer, int):
                normalised.append(_check_in_range(indexer, size, axis, shape))
            else:
                normalised.append(indexer)

        dropped = tuple(i for i in normalised if isinstance(i, int))
        name = self._derived_name(dropped)
        if len(dropped) == len(shape):
            return self._element_at(dropped, name=name)

        groups, names = self._surviving_levels(normalised)
        return self._sub_batch_at(
            tuple(normalised), axis_groups=groups, level_names=names, name=name
        )

    def _surviving_levels(
        self, normalised: list[int | slice]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
        """The levels left after dropping every integer-indexed axis."""
        groups: list[tuple[int, ...]] = []
        names: list[str] = []
        start = 0
        for level_name, group in zip(self._level_names, self._axis_groups, strict=True):
            kept = tuple(
                _sliced_size(normalised[start + offset], size) for offset, size in enumerate(group)
            )
            surviving = tuple(size for size in kept if size is not None)
            if surviving:
                groups.append(surviving)
                names.append(level_name)
            start += len(group)
        return tuple(groups), tuple(names)

    def _derived_name(self, dropped: tuple[int, ...]) -> str:
        """``name[i]`` per dropped axis, composing as ``name[i][j]`` across levels."""
        return self._name + "".join(f"[{index}]" for index in dropped)

    @abstractmethod
    def _with_level_names(self, level_names: tuple[str, ...]) -> Self:
        """A shallow copy carrying *level_names*, sharing shape and elements."""


def _as_axis_indexers(indexer: LevelIndexer) -> tuple[int | slice | None, ...]:
    """A level's indexer as a tuple, one entry per addressed axis."""
    return indexer if isinstance(indexer, tuple) else (indexer,)


def _check_in_range(index: int, size: int, axis: int, shape: tuple[int, ...]) -> int:
    """Resolve a possibly-negative integer index, or raise ``IndexError``."""
    resolved = index + size if index < 0 else index
    if not 0 <= resolved < size:
        raise IndexError(f"index {index} is out of range for axis {axis} of batch_shape {shape}")
    return resolved


def _sliced_size(indexer: int | slice, size: int) -> int | None:
    """The axis size after *indexer*, or ``None`` when the axis is dropped."""
    if isinstance(indexer, int):
        return None
    return len(range(*indexer.indices(size)))
