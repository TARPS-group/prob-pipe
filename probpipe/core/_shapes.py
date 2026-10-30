"""The reading of shape, name-list, axis-count, and path arguments.

Every public argument that takes a shape, a sequence of names, one axis count
per level, or paths is read by a function of this module, so each kind of
argument has one reading and one set of error messages wherever it is taken:

- a **shape** (:func:`_as_shape`) is a tuple of dimensions, each a
  non-negative ``int`` size or a symbolic dimension name, and a bare ``int``
  or ``str`` is a shape of one dimension, so ``3`` is ``(3,)`` and ``"n"`` is
  ``("n",)``;
- **names** (:func:`_as_names`), such as level names or metric names, are a
  tuple of strings, and a bare ``str`` is one name, so ``"draw"`` is
  ``("draw",)``;
- **axis counts** (:func:`_as_axis_counts`) are a tuple of positive integers,
  one per level, and a bare ``int`` is read as one count;
- **levels** (:func:`_as_levels`) are a mapping from level name to that level's
  shape, given as a mapping or as keyword arguments;
- **paths** (:func:`_as_paths`) are what ``[]`` takes on a term indexed by
  path: a ``str`` is read as one path, and a list or a tuple of ``str`` as a
  list of paths. On a batch, an integer, a slice, or a tuple of them is read as
  a position (:func:`_as_positions`). An argument that takes one path per argument, such
  as ``without(*paths)``, is read by :func:`_as_path_arguments`.

A **path** is the ``/``-joined names from the top of a term down to a node,
such as ``"g/c"``. A name is a non-empty string without ``/``. A list of paths
must name at least one path, name no path twice, and hold no path inside
another, since a list that held both ``"g"`` and ``"g/c"`` would name the
variable at ``g/c`` twice.

A sequence is read as NumPy reads a shape: a tuple, a list, a ``range``, or a
1-D array is one item per entry, stored as a tuple. An iterator such as a
generator, a set, ``bytes``, a ``memoryview``, and a mapping are refused, since
an iterator is used up by one reading, a set has no order, and the others
iterate into byte values or keys.

A dimension name is a Python identifier, such as ``n`` or ``n_obs``. An integer
is anything ``operator.index`` accepts other than a ``bool``, and it is stored as
a Python ``int``. A name is stored as a Python ``str``.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Mapping, Set
from typing import Any, Literal, overload

import numpy as np

from ._repr import type_name

__all__ = [
    "AxisCountsLike",
    "DimLike",
    "LevelsLike",
    "NamesLike",
    "PathsLike",
    "ShapeLike",
    "SizesLike",
]

#: One dimension: a non-negative size or a symbolic dimension name.
type DimLike = int | str
#: A shape: one dimension, or a sequence of them.
type ShapeLike = DimLike | Iterable[DimLike]
#: A shape of sizes only, such as a ``sample_shape``: one size, or a sequence of them.
type SizesLike = int | Iterable[int]
#: Names, such as level names or metric names: one name, or a sequence of them.
type NamesLike = str | Iterable[str]
#: Axis counts: one count, or a sequence of one count per level.
type AxisCountsLike = int | Iterable[int]
#: Levels: each level's name mapped to the shape of its axes, outermost level first.
type LevelsLike = Mapping[str, ShapeLike]
#: Paths: one path, such as ``"g/c"``, or a list or tuple of paths.
type PathsLike = str | list[str] | tuple[str, ...]

#: The iterables refused where a sequence is taken.
_NOT_SEQUENCES = (bytes, bytearray, memoryview, Mapping, Set, Iterator)


def _is_scalar(value: Any) -> bool:
    """Whether *value* is one item rather than an iterable of items.

    A zero-dimensional array is one item, although its type defines iteration.
    """
    return getattr(value, "ndim", None) == 0 or not isinstance(value, Iterable)


def _as_int(value: Any) -> int | None:
    """*value* as a Python ``int``, or ``None`` when it is not an integer or is a ``bool``."""
    if isinstance(value, bool | np.bool_):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _as_dim_name(name: str, *, what: str) -> str:
    """*name* as a dimension name: a Python ``str`` that is a Python identifier.

    Parameters
    ----------
    name : str
        The name as the caller gave it.
    what : str
        The item the error message names, such as ``"NumericArraySpec shape entry"``.

    Returns
    -------
    str
        The name as a Python ``str``.

    Raises
    ------
    ValueError
        If *name* is not a Python identifier.
    """
    if not name.isidentifier():
        raise ValueError(f"{what} must be a Python identifier such as 'n_obs', got {name!r}")
    return str(name)


def _as_dim(entry: Any, *, what: str, symbolic: bool = True) -> int | str:
    """One dimension of a shape: a non-negative ``int``, or a dimension name.

    Parameters
    ----------
    entry : Any
        The dimension as the caller gave it.
    what : str
        The item the error message names, such as ``"NumericArraySpec shape entry"``.
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
    if isinstance(entry, str) and symbolic:
        return _as_dim_name(entry, what=what)
    size = _as_int(entry)
    if size is None:
        kinds = "a non-negative int or a dimension name" if symbolic else "a non-negative int"
        raise TypeError(f"{what} must be {kinds}, got {type_name(entry)} {entry!r}")
    if size < 0:
        raise ValueError(f"{what} must be non-negative, got {size}")
    return size


@overload
def _as_shape(arg: Any, *, what: str, symbolic: Literal[False]) -> tuple[int, ...]: ...
@overload
def _as_shape(arg: Any, *, what: str, symbolic: bool = True) -> tuple[int | str, ...]: ...
def _as_shape(arg: Any, *, what: str, symbolic: bool = True) -> tuple[int | str, ...]:
    """A shape argument as a tuple of dimensions; a bare ``int`` or ``str`` is one dimension.

    ``3`` reads as ``(3,)`` and ``"n"`` as ``("n",)``. Any other sequence reads
    as one dimension per item, and an empty one is the shape ``()`` of rank 0.

    Parameters
    ----------
    arg : Any
        The shape as the caller gave it.
    what : str
        The caller's function and argument, such as ``"NumericArraySpec shape"``,
        which each error message names.
    symbolic : bool
        Whether a dimension name is accepted.

    Returns
    -------
    tuple of int or str
        One entry per dimension, each a Python ``int`` or a Python ``str``.

    Raises
    ------
    TypeError
        If *arg* is not an integer, a string, or a sequence, is a ``bool``, is a
        string where *symbolic* is false, is one of the iterables the module
        refuses, or holds an entry :func:`_as_dim` refuses.
    ValueError
        If an entry is negative or is a name that is not a Python identifier.
    """
    forms = "an int, a str, or a sequence of them" if symbolic else "an int or a sequence of ints"
    if isinstance(arg, str):
        if not symbolic:
            raise TypeError(f"{what} must be {forms}, got str {arg!r}")
        return (_as_dim_name(arg, what=f"{what} entry"),)
    if isinstance(arg, _NOT_SEQUENCES) or (_is_scalar(arg) and _as_int(arg) is None):
        raise TypeError(f"{what} must be {forms}, got {type_name(arg)} {arg!r}")
    entries = (arg,) if _is_scalar(arg) else arg
    return tuple(_as_dim(entry, what=f"{what} entry", symbolic=symbolic) for entry in entries)


def _as_names(arg: Any, *, what: str) -> tuple[str, ...]:
    """A name-list argument as a tuple of strings; a bare ``str`` is one name.

    Each name is kept as given, and the caller checks it against its own rule:
    a :class:`~probpipe.BatchSpec` checks level names, and a function that
    takes metric or variable names checks that each one exists. An empty
    sequence reads as ``()``.

    Parameters
    ----------
    arg : Any
        The names as the caller gave them.
    what : str
        The caller's function and argument, which each error message names.

    Returns
    -------
    tuple of str
        One entry per name, in the order given, each a Python ``str``.

    Raises
    ------
    TypeError
        If *arg* is neither a string nor a sequence, is one of the iterables the
        module refuses, or holds an entry that is not a string.
    """
    if isinstance(arg, str):
        return (str(arg),)
    if _is_scalar(arg) or isinstance(arg, _NOT_SEQUENCES):
        raise TypeError(f"{what} must be a str or a sequence of str, got {type_name(arg)} {arg!r}")
    names = tuple(arg)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{what} entry must be a str, got {type_name(name)} {name!r}")
    return tuple(str(name) for name in names)


def _as_axis_count(entry: Any, *, what: str) -> int:
    """One per-level axis count: a positive ``int``.

    Parameters
    ----------
    entry : Any
        The count as the caller gave it.
    what : str
        The item the error message names, such as ``"axes_per_level entry"``.

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
        raise TypeError(f"{what} must be an int, got {type_name(entry)} {entry!r}")
    if count < 1:
        raise ValueError(f"{what} must be at least 1, got {count}")
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
        If *arg* is neither an integer nor a sequence, is a string or one of the
        iterables the module refuses, or holds an entry that is not an integer.
    ValueError
        If a count is less than 1.
    """
    if isinstance(arg, (str, *_NOT_SEQUENCES)) or (_is_scalar(arg) and _as_int(arg) is None):
        raise TypeError(
            f"{what} must be an int or a sequence of ints, got {type_name(arg)} {arg!r}"
        )
    if _is_scalar(arg):
        return (_as_axis_count(arg, what=what),)
    return tuple(_as_axis_count(entry, what=f"{what} entry") for entry in arg)


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
        names.append(str(name))
        groups.append(group)
    return tuple(names), tuple(groups)


def _as_paths(arg: Any, *, action: str, positions: bool = False) -> str | tuple[str, ...]:
    """What ``[]`` takes on a term indexed by path: one path, or a list of paths.

    A ``str`` is read as one path, returned as a ``str``. A list or a tuple of
    ``str`` is read as a list of paths, returned as a tuple in the order given. A subclass of
    ``str``, such as ``numpy.str_``, is returned as a Python ``str``.

    Parameters
    ----------
    arg : Any
        The paths as the caller gave them.
    action : str
        What the caller was asked to do, which each error message completes
        with *arg*, such as ``"index Record by"``.
    positions : bool
        Whether the caller also takes positions, as a batch does. A caller that
        takes them reads them first with :func:`_as_positions`, and the errors
        then name what is wrong with the positions *arg* holds.

    Returns
    -------
    str or tuple of str
        The one path, or the list of paths as a tuple.

    Raises
    ------
    TypeError
        If *arg* is neither a string nor a list or a tuple, such as a set, an
        iterator, ``None``, a ``bool``, or ``...``; if an entry of the list is
        not a string, or a path is mixed with a position; or, for a caller that
        takes positions, if *arg* is a list of positions.
    ValueError
        If a path is empty or has an empty name, as ``"g//c"`` and ``"/a"``
        do, the list is empty, or it names a path twice or a path inside
        another.
    """
    if isinstance(arg, str):
        return _checked_path(str(arg), action=action, given=arg)
    if isinstance(arg, (list, tuple)):
        if not arg:
            raise ValueError(
                f"cannot {action} {arg!r}, since it names no paths. Give at least one path"
            )
        holds_position = any(_is_position(entry) for entry in arg)
        if holds_position and not positions:
            position = next(entry for entry in arg if _is_position(entry))
            raise TypeError(
                f"cannot {action} {arg!r}, since {position!r} is a position and only a batch "
                f"takes positions. Give only paths"
            )
        if holds_position and any(isinstance(entry, str) for entry in arg):
            raise TypeError(
                f"cannot mix a path and a position in one [], got {arg!r}. Index the position "
                f"and the path separately, as b[0]['a']"
            )
        if positions and holds_position and all(_is_position(entry) for entry in arg):
            raise TypeError(
                f"cannot index a batch by a list of positions, got {arg!r}. Index one position "
                f"or a slice"
            )
        for entry in arg:
            if not isinstance(entry, str):
                raise TypeError(
                    f"cannot {action} {arg!r}, since each path must be a str, got "
                    f"{type_name(entry)} {entry!r}"
                )
        paths = tuple(_checked_path(str(entry), action=action, given=arg) for entry in arg)
        _check_distinct_paths(paths, action="select", given=arg)
        return paths
    forms = "a path, a list of paths, or a position" if positions else "a path or a list of paths"
    raise TypeError(f"cannot {action} {arg!r}, since it must be {forms}, got {type_name(arg)}")


def _as_positions(arg: Any) -> tuple[Any, ...] | None:
    """*arg* as a position of a batch, one indexer per leading axis, or None when it is paths.

    A ``str``, a list, an empty tuple, a tuple that holds a ``str``, and an
    iterable this module refuses, such as a set, give None, and the caller
    reads them with :func:`_as_paths`, whose errors name what is wrong with
    them. Any other tuple gives one indexer per leading axis, and any other
    value is the indexer of the leading axis. The batch checks each indexer
    against its axis, so that ``None``, a ``bool``, a float, or an array is
    refused with the axis it was given for.

    Parameters
    ----------
    arg : Any
        The argument of ``[]`` as the caller gave it.

    Returns
    -------
    tuple or None
        The indexers as given, one per leading axis, or None.
    """
    if isinstance(arg, (str, list, *_NOT_SEQUENCES)):
        return None
    if isinstance(arg, tuple):
        if not arg or any(isinstance(entry, str) for entry in arg):
            return None
        return arg
    return (arg,)


def _as_path(arg: Any, *, action: str) -> str:
    """One path argument, a ``str``, such as a key of a rename.

    Parameters
    ----------
    arg : Any
        The path as the caller gave it.
    action : str
        What the caller was asked to do, which each error message completes
        with *arg*, such as ``"rename"``.

    Returns
    -------
    str
        The path, as a Python ``str``.

    Raises
    ------
    TypeError
        If *arg* is not a string, such as a tuple of names.
    ValueError
        If *arg* is empty or has an empty name.
    """
    if not isinstance(arg, str):
        raise TypeError(
            f"cannot {action} {arg!r}, since a path must be a str, got {type_name(arg)}"
        )
    return _checked_path(str(arg), action=action, given=arg)


def _as_path_arguments(arguments: tuple[Any, ...], *, action: str) -> tuple[str, ...]:
    """Paths given one per argument, as ``without(*paths)`` takes them.

    Each argument is one path, a ``str``, and together they follow the rules
    of a list of paths, apart from being empty: no path twice, and no path
    inside another.

    Parameters
    ----------
    arguments : tuple
        The arguments as the caller gave them.
    action : str
        What the caller was asked to do, which each error message completes
        with *arguments*, such as ``"call without() with"``.

    Returns
    -------
    tuple of str
        The paths, in the order given.

    Raises
    ------
    TypeError
        If an argument is not a string, such as a tuple of paths.
    ValueError
        If a path is empty or has an empty name, or the arguments name a path
        twice or a path inside another.
    """
    shown_arguments = arguments[0] if len(arguments) == 1 else arguments
    for argument in arguments:
        if not isinstance(argument, str):
            fix = ". Give each path as a separate argument" if isinstance(argument, tuple) else ""
            raise TypeError(
                f"cannot {action} {shown_arguments!r}, since a path must be a str, got "
                f"{type_name(argument)} {argument!r}{fix}"
            )
    paths = tuple(
        _checked_path(str(argument), action=action, given=shown_arguments) for argument in arguments
    )
    _check_distinct_paths(paths, action=action, given=shown_arguments)
    return paths


def _as_joined_path(segments: tuple[Any, ...], *, function: str) -> str:
    """The one path that *segments* spell, each a path or a name, joined with ``/``.

    ``("g", "c")`` and ``("g/c",)`` both spell the path ``"g/c"``.

    Parameters
    ----------
    segments : tuple
        The segments as the caller gave them, one per argument.
    function : str
        The name of the method that takes *segments*, such as ``"at_path"``,
        which each error message names.

    Returns
    -------
    str
        The path.

    Raises
    ------
    TypeError
        If a segment is not a string, such as a tuple of segments.
    ValueError
        If no segment is given, or the path has an empty name.
    """
    from .named_tree import _PATH_SEP

    action = f"call {function}() with"
    shown_segments = segments[0] if len(segments) == 1 else segments
    for segment in segments:
        if isinstance(segment, tuple):
            raise TypeError(
                f"{function} takes each segment as a separate argument, got a tuple. Write "
                f"{function}(*segments)"
            )
        if not isinstance(segment, str):
            raise TypeError(
                f"cannot {action} {shown_segments!r}, since a path segment must be a str, got "
                f"{type_name(segment)} {segment!r}"
            )
    if not segments:
        raise ValueError(f"cannot {action} no segments. Give at least one path segment")
    return _checked_path(
        _PATH_SEP.join(str(segment) for segment in segments),
        action=action,
        given=shown_segments,
    )


def _missing_path(
    path: str, *, owner: str, available: Iterable[str], requested: tuple[str, ...] = ()
) -> str:
    """The message for *path*, which is not a path of *owner*, listing the paths it has.

    When *path* comes from a list of paths whose ``/``-joined form is a path of
    *owner*, as ``site["habitat", "forest"]`` on a record with the path
    ``habitat/forest``, the message adds how to write the nested path.

    Parameters
    ----------
    path : str
        The path that was not found.
    owner : str
        The term the message names, such as ``"'site'"``.
    available : iterable of str
        The paths *owner* has, in its canonical order.
    requested : tuple of str
        The list of paths that held *path*, or ``()`` for one path.

    Returns
    -------
    str
        The message.
    """
    from .named_tree import _PATH_SEP

    paths = list(available)
    have = f"available paths: {paths}" if paths else "it has no paths"
    message = f"unknown path {path!r} in {owner}; {have}"
    joined = _PATH_SEP.join(requested)
    if len(requested) > 1 and joined in paths:
        message += f". To select the nested path, write {joined!r}"
    return message


def _is_position(entry: Any) -> bool:
    """Whether *entry* indexes one batch axis: an integer that is not a ``bool``, or a slice."""
    return isinstance(entry, slice) or (not isinstance(entry, str) and _as_int(entry) is not None)


def _checked_path(path: str, *, action: str, given: Any) -> str:
    """*path*, checked to be non-empty with no empty name, as in ``"g//c"`` or ``"/a"``.

    Parameters
    ----------
    path : str
        The path.
    action : str
        What the caller was asked to do, which the message completes.
    given : Any
        The argument that held *path*, which the message shows.

    Returns
    -------
    str
        *path* itself.

    Raises
    ------
    ValueError
        If *path* is empty or has an empty name.
    """
    from .named_tree import _PATH_SEP

    if not path:
        raise ValueError(f"cannot {action} {given!r}, since the path is empty")
    if "" in path.split(_PATH_SEP):
        raise ValueError(
            f"cannot {action} {given!r}, since the path {path!r} has an empty name. Join the "
            f"names with a single {_PATH_SEP!r}, as 'g/c'"
        )
    return path


def _check_distinct_paths(paths: tuple[str, ...], *, action: str, given: Any) -> None:
    """Check that *paths* name no path twice and hold no path inside another.

    Parameters
    ----------
    paths : tuple of str
        The paths.
    action : str
        What the caller does with *paths*, which each message leads with, such
        as ``"select"`` or ``"call without() with"``.
    given : Any
        The argument that held *paths*, which the message shows.

    Raises
    ------
    ValueError
        If a path appears twice, or one path is inside another.
    """
    from .named_tree import _PATH_SEP

    seen: set[str] = set()
    for path in paths:
        if path in seen:
            raise ValueError(f"cannot {action} {path!r} twice, got {given!r}. Give each path once")
        seen.add(path)
    for outer in paths:
        for inner in paths:
            if inner.startswith(outer + _PATH_SEP):
                raise ValueError(
                    f"cannot {action} both {outer!r} and {inner!r}, since {inner!r} is inside "
                    f"{outer!r}. Keep one of them"
                )
