"""The reading of shape, level-name, and axis-count arguments.

Every public argument that takes a shape, a sequence of level names, or one
axis count per level is read by a function of this module, so each kind of
argument has one reading and one set of error messages wherever it is taken:

- a **shape** (:func:`_as_shape`) is a tuple of dimensions, each a
  non-negative ``int`` size or a symbolic dimension name, and a bare ``int``
  or ``str`` is a shape of one dimension, so ``3`` is ``(3,)`` and ``"n"`` is
  ``("n",)``;
- **level names** (:func:`_as_level_names`) are a tuple of strings, and a bare
  ``str`` is one name, so ``"draw"`` is ``("draw",)``;
- **axis counts** (:func:`_as_axis_counts`) are a tuple of positive integers,
  one per level, and a bare ``int`` is one count;
- **levels** (:func:`_as_levels`) are a mapping from level name to that level's
  shape, given as a mapping or as keyword arguments.

A dimension name is a Python identifier, such as ``n`` or ``n_obs``. An integer
is anything ``operator.index`` accepts other than a ``bool``, and it is stored as
a Python ``int``. A ``bytes`` or a mapping is refused where a sequence is taken,
since iterating it yields byte values or keys.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

from ._repr import type_name

__all__ = [
    "AxisCountsLike",
    "DimLike",
    "LevelNamesLike",
    "LevelsLike",
    "ShapeLike",
]

#: One dimension: a non-negative size or a symbolic dimension name.
type DimLike = int | str
#: A shape: one dimension, or an iterable of them.
type ShapeLike = DimLike | Iterable[DimLike]
#: Level names: one name, or an iterable of them.
type LevelNamesLike = str | Iterable[str]
#: Axis counts: one count, or an iterable of one count per level.
type AxisCountsLike = int | Iterable[int]
#: Levels: each level's name mapped to the shape of its axes, outermost level first.
type LevelsLike = Mapping[str, ShapeLike]

#: The types whose iteration yields something other than their items.
_NOT_SEQUENCES = (bytes, bytearray, Mapping)


def _is_scalar(value: Any) -> bool:
    """Whether *value* is one item rather than an iterable of items.

    A zero-dimensional array is one item, although its type defines iteration.
    """
    return getattr(value, "ndim", None) == 0 or not isinstance(value, Iterable)


def _as_int(value: Any) -> int | None:
    """*value* as a Python ``int``, or ``None`` when it is not an integer or is a ``bool``."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _as_dim(entry: Any, *, what: str, symbolic: bool = True) -> int | str:
    """One dimension of a shape: a non-negative ``int``, or a dimension name.

    Parameters
    ----------
    entry : Any
        The dimension as the caller gave it.
    what : str
        The caller's function and argument, such as ``"NumericArraySpec shape"``,
        which each error message names.
    symbolic : bool
        Whether a dimension name is accepted. A shape that counts draws or sizes
        an array takes integers only.

    Returns
    -------
    int or str
        The size as a Python ``int``, or the name as a Python ``str``.

    Raises
    ------
    TypeError
        If *entry* is neither an integer nor a string, is a ``bool``, or is a
        string where *symbolic* is false.
    ValueError
        If *entry* is a negative integer or a name that is not a Python identifier.
    """
    if isinstance(entry, str):
        if not symbolic:
            raise TypeError(f"{what} entries must be ints, got str {entry!r}")
        if not entry.isidentifier():
            raise ValueError(
                f"{what} dimension names must be Python identifiers such as 'n_obs', got {entry!r}"
            )
        return str(entry)
    size = _as_int(entry)
    if size is None:
        kinds = "non-negative ints or dimension names" if symbolic else "non-negative ints"
        raise TypeError(f"{what} entries must be {kinds}, got {type_name(entry)} {entry!r}")
    if size < 0:
        raise ValueError(f"{what} entries must be non-negative, got {size}")
    return size


@overload
def _as_shape(arg: Any, *, what: str, symbolic: Literal[False]) -> tuple[int, ...]: ...
@overload
def _as_shape(arg: Any, *, what: str, symbolic: bool = True) -> tuple[int | str, ...]: ...
def _as_shape(arg: Any, *, what: str, symbolic: bool = True) -> tuple[int | str, ...]:
    """A shape argument as a tuple of dimensions; a bare ``int`` or ``str`` is one dimension.

    ``3`` reads as ``(3,)`` and ``"n"`` as ``("n",)``. Any other iterable reads
    as one dimension per item, and an empty one is the shape ``()`` of rank 0.

    Parameters
    ----------
    arg : Any
        The shape as the caller gave it.
    what : str
        The caller's function and argument, which each error message names.
    symbolic : bool
        Whether a dimension name is accepted.

    Returns
    -------
    tuple of int or str
        One entry per dimension, each a Python ``int`` or a name.

    Raises
    ------
    TypeError
        If *arg* is not an integer, a string, or an iterable, is a ``bool``,
        ``bytes``, or a mapping, or holds an entry :func:`_as_dim` refuses.
    ValueError
        If an entry is negative or is a name that is not a Python identifier.
    """
    if isinstance(arg, str):
        return (_as_dim(arg, what=what, symbolic=symbolic),)
    if (_is_scalar(arg) and _as_int(arg) is None) or isinstance(arg, _NOT_SEQUENCES):
        raise TypeError(f"{what} must be {_shape_forms(symbolic)}, got {type_name(arg)} {arg!r}")
    if _is_scalar(arg):
        return (_as_dim(arg, what=what, symbolic=symbolic),)
    return tuple(_as_dim(entry, what=what, symbolic=symbolic) for entry in arg)


def _shape_forms(symbolic: bool) -> str:
    """The accepted forms of a shape argument, as an error message lists them."""
    return "an int, a str, or an iterable of them" if symbolic else "an int or an iterable of ints"


def _as_level_names(arg: Any, *, what: str) -> tuple[str, ...]:
    """A level-names argument as a tuple of strings; a bare ``str`` is one name.

    The names are not checked against the rule for level names here: every
    batch is built through a :class:`~probpipe.BatchSpec`, which checks them.

    Parameters
    ----------
    arg : Any
        The level names as the caller gave them.
    what : str
        The caller's function and argument, which each error message names.

    Returns
    -------
    tuple of str
        One name per level, outermost first.

    Raises
    ------
    TypeError
        If *arg* is neither a string nor an iterable, is ``bytes`` or a mapping,
        or holds an entry that is not a string.
    """
    if isinstance(arg, str):
        return (arg,)
    if _is_scalar(arg) or isinstance(arg, _NOT_SEQUENCES):
        raise TypeError(f"{what} must be a str or an iterable of str, got {type_name(arg)} {arg!r}")
    names = tuple(arg)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{what} entries must be str, got {type_name(name)} {name!r}")
    return names


def _as_axis_count(entry: Any, *, what: str) -> int:
    """One per-level axis count: a positive ``int``.

    Parameters
    ----------
    entry : Any
        The count as the caller gave it.
    what : str
        The caller's function and argument, which each error message names.

    Returns
    -------
    int
        The count as a Python ``int``.

    Raises
    ------
    TypeError
        If *entry* is not an integer or is a ``bool``.
    ValueError
        If *entry* is less than 1.
    """
    count = _as_int(entry)
    if count is None:
        raise TypeError(f"{what} entries must be ints, got {type_name(entry)} {entry!r}")
    if count < 1:
        raise ValueError(f"{what} entries must be at least 1, got {count}")
    return count


def _as_axis_counts(arg: Any, *, what: str) -> tuple[int, ...]:
    """An axis-counts argument as a tuple of positive ints; a bare ``int`` is one count.

    Parameters
    ----------
    arg : Any
        The counts as the caller gave them, one per level.
    what : str
        The caller's function and argument, which each error message names.

    Returns
    -------
    tuple of int
        One count per level, outermost first.

    Raises
    ------
    TypeError
        If *arg* is neither an integer nor an iterable, is a string, ``bytes``,
        or a mapping, or holds an entry that is not an integer.
    ValueError
        If a count is less than 1.
    """
    if isinstance(arg, (str, *_NOT_SEQUENCES)) or (_is_scalar(arg) and _as_int(arg) is None):
        raise TypeError(
            f"{what} must be an int or an iterable of ints, got {type_name(arg)} {arg!r}"
        )
    if _is_scalar(arg):
        return (_as_axis_count(arg, what=what),)
    return tuple(_as_axis_count(entry, what=what) for entry in arg)


def _as_levels(
    levels: Any, level_shapes: Mapping[str, Any], *, what: str
) -> tuple[tuple[str, ...], tuple[tuple[int | str, ...], ...]]:
    """Levels given as a mapping or as keywords, as aligned level names and axis groups.

    Each value is a shape read by :func:`_as_shape`, so ``draw=4`` is one level
    of one axis and ``grid=(3, 4)`` is one level of two axes. The levels are
    ordered outermost first, as written.

    Parameters
    ----------
    levels : Mapping of str to shape, or None
        The levels as a mapping, for a level name that cannot be written as a
        keyword.
    level_shapes : Mapping of str to shape
        The levels given as keyword arguments.
    what : str
        The caller's name, which each error message names.

    Returns
    -------
    tuple of (tuple of str, tuple of tuple of int or str)
        The level names and, aligned with them, each level's axis sizes.

    Raises
    ------
    TypeError
        If both forms are given, *levels* is not a mapping, a level name is not
        a string, or a level's shape is malformed.
    ValueError
        If no level is given, a level has no axes, or an axis size is negative
        or is a name that is not a Python identifier.
    """
    if levels is not None and level_shapes:
        raise TypeError(f"{what} takes its levels as a mapping or as keywords, but got both")
    if levels is None:
        levels = level_shapes
    elif not isinstance(levels, Mapping):
        raise TypeError(
            f"{what} levels must be a mapping from level name to shape, "
            f"got {type_name(levels)} {levels!r}"
        )
    if not levels:
        raise ValueError(f"{what} must have at least one level, such as draw=4")
    names: list[str] = []
    groups: list[tuple[int | str, ...]] = []
    for name, shape in levels.items():
        if not isinstance(name, str):
            raise TypeError(f"{what} level names must be str, got {type_name(name)} {name!r}")
        group = _as_shape(shape, what=f"{what} level {name!r}")
        if not group:
            raise ValueError(f"{what} level {name!r} must have at least one axis, got ()")
        names.append(name)
        groups.append(group)
    return tuple(names), tuple(groups)
