"""The parts every public kind assembles its repr from, under the convention of design II.4.

A repr reads as a constructor call of the term's public class, with the label
first and positionally, and it shows one argument per line once it is longer
than :data:`WIDTH` characters. Each helper formats one part, such as the levels
of a batch, the field paths of a record, or a family parameter.

A repr built here is a :class:`_Layout`, a string that keeps its arguments, so
an enclosing repr lays it out again where its line starts. A nested repr
therefore takes several lines when its own line would pass :data:`WIDTH`.

A helper reads declarations and array metadata, and it reads a parameter's
entries only when it has at most eight, so a repr raises no storage error.
:meth:`~probpipe.core.tracked.TrackedTerm.with_provenance` interpolates a term
into its error, which depends on that.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable, Mapping
from math import prod
from typing import Any

import numpy as np

__all__ = [
    "WIDTH",
    "call_repr",
    "format_dtype",
    "format_levels",
    "format_names",
    "format_value",
    "mapping_repr",
    "public_class_name",
    "sequence_repr",
    "term_repr",
]

#: The length past which a repr shows one argument per line.
WIDTH = 100

#: How many entries a parameter may hold before its repr gives its shape instead.
_MAX_SHOWN_ENTRIES = 8

_INDENT = "    "

#: One argument of a layout: the text before its value, such as ``"shape="``, and the value.
type _Part = tuple[str, str]


class _Layout(str):
    """A repr laid out at the left margin, which keeps its arguments to lay itself out again.

    The string is the layout of a repr that starts a line. An enclosing layout
    lays each argument out again where its line starts, so a nested repr breaks
    its lines where its own line would pass :data:`WIDTH`, and indents them
    under the enclosing argument.
    """

    _head: str
    _parts: tuple[_Part, ...]
    _brackets: tuple[str, str]

    def __new__(cls, head: str, parts: Iterable[_Part], brackets: tuple[str, str]) -> _Layout:
        parts = tuple(parts)
        layout = super().__new__(cls, _laid_out(head, parts, brackets, 0, 0, 0))
        layout._head, layout._parts, layout._brackets = head, parts, brackets
        return layout

    def __reduce_ex__(self, protocol: Any) -> tuple[type[str], tuple[str]]:
        """Copy and pickle as the plain string, since the parts serve layout alone."""
        return (str, (str(self),))

    def laid_out(self, indent: int, column: int, trailing: int) -> str:
        """This repr laid out to start at *column* on a line indented by *indent*.

        *trailing* counts the characters that follow it on its last line. The
        lines after the first are indented relative to the first line's indent.
        """
        return _laid_out(self._head, self._parts, self._brackets, indent, column, trailing)

    def one_line(self) -> str:
        """This repr on one line, whatever its length."""
        opener, closer = self._brackets
        parts = ", ".join(prefix + _one_line(value) for prefix, value in self._parts)
        return f"{self._head}{opener}{parts}{closer}"


def _one_line(value: str) -> str:
    """*value* on one line: a layout's one-line form, and any other string as it is."""
    return value.one_line() if isinstance(value, _Layout) else value


def _laid_out(
    head: str,
    parts: tuple[_Part, ...],
    brackets: tuple[str, str],
    indent: int,
    column: int,
    trailing: int,
) -> str:
    """*parts* between the *brackets* after *head*: on one line when it fits, else one per line.

    The one-line form fits when it ends within :data:`WIDTH` starting at
    *column*, with *trailing* characters after it. Otherwise each part takes a
    line of its own, indented one level past *indent*, and is itself laid out
    where that line places it.
    """
    opener, closer = brackets
    one_line = (
        f"{head}{opener}{', '.join(prefix + _one_line(value) for prefix, value in parts)}{closer}"
    )
    if column + len(one_line) + trailing <= WIDTH and "\n" not in one_line:
        return one_line
    inner = indent + len(_INDENT)
    lines = []
    for prefix, value in parts:
        text = (
            value.laid_out(inner, inner + len(prefix), 1) if isinstance(value, _Layout) else value
        )
        lines.append(_INDENT + (prefix + text).replace("\n", "\n" + _INDENT) + ",\n")
    return f"{head}{opener}\n{''.join(lines)}{closer.lstrip(',')}"


def call_repr(
    class_name: str, positional: Iterable[str] = (), keywords: Iterable[tuple[str, str]] = ()
) -> str:
    """``class_name(positional, ..., name=value, ...)`` over formatted arguments.

    One argument goes on each line past :data:`WIDTH` characters, and a value
    that is itself such a repr is laid out at the width its line leaves.
    """
    parts = [("", value) for value in positional]
    parts.extend((f"{name}=", value) for name, value in keywords)
    return _Layout(class_name, parts, ("(", ")"))


def term_repr(
    class_name: str, label: str | None = None, keywords: Iterable[tuple[str, str]] = ()
) -> str:
    """``class_name('label', name=value, ...)``: the label first and positionally, then keywords.

    Parameters
    ----------
    class_name : str
        The public class or kind the repr names.
    label : str, optional
        The term's label, omitted for an object without one, such as a spec.
    keywords : iterable of (str, str)
        Each keyword argument's name and its formatted value, in order.
    """
    return call_repr(class_name, [] if label is None else [repr(label)], keywords)


def sequence_repr(items: Iterable[str]) -> str:
    """The formatted *items* as a tuple, keeping the trailing comma of a lone item on one line."""
    items = list(items)
    closer = ",)" if len(items) == 1 else ")"
    return _Layout("", [("", item) for item in items], ("(", closer))


def mapping_repr(items: Mapping[str, str]) -> str:
    """The formatted *items* as the dict ``{'name': value, ...}``."""
    return _Layout("", [(f"{name!r}: ", value) for name, value in items.items()], ("{", "}"))


def format_levels(level_names: Iterable[str], axis_groups: Iterable[Iterable[Any]]) -> str:
    """A batch's levels as ``{'chain': 4, 'draw': 500}``; a level of several axes gives a tuple."""
    sizes: dict[str, str] = {}
    for level_name, group in zip(level_names, axis_groups, strict=True):
        group = tuple(group)
        sizes[level_name] = repr(group[0] if len(group) == 1 else group)
    return mapping_repr(sizes)


def format_names(names: Iterable[str]) -> str:
    """Names or paths as a tuple of strings, such as ``('data/effect', 'data/se', 'label')``."""
    return sequence_repr(repr(name) for name in names)


def format_dtype(dtype: Any) -> str:
    """A dtype by its name, as ``float32``, and ``None`` when there is none."""
    return "None" if dtype is None else str(np.dtype(dtype))


def format_value(value: Any) -> str:
    """A parameter's value: its entries when it has few, and its shape and dtype otherwise.

    A scalar reads as a number and an array of at most eight entries as a list
    of them, while a larger array, or a traced one, whose entries cannot be
    read, gives ``array(shape=..., dtype=...)``. A function or a class reads as
    its name, and any other value, a tracked term or an operator among them,
    keeps its own repr.
    """
    if isinstance(value, bool | int | float | complex | str | type(None)) or hasattr(value, "raw"):
        return repr(value)
    if isinstance(value, type):
        return value.__name__
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return value.__qualname__
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return repr(value)
    shape = tuple(shape)
    if prod(shape) <= _MAX_SHOWN_ENTRIES:
        try:
            entries = np.asarray(value)
        except Exception:  # a traced array has no entries to read
            entries = None
        if entries is not None and entries.dtype != object:
            if entries.ndim == 0:
                return str(entries[()])
            text = np.array2string(
                entries, separator=", ", formatter={"all": str}, max_line_width=sys.maxsize
            )
            return text.replace("\n", "")
    return f"array(shape={shape}, dtype={format_dtype(dtype)})"


def public_class_name(cls: type) -> str:
    """The name of the first public class in *cls*'s method-resolution order.

    A private class, whose name starts with an underscore, presents the public
    class it refines, so a repr never names a class a user cannot import.
    """
    for klass in cls.__mro__:
        if not klass.__name__.startswith("_"):
            return klass.__name__
    return cls.__name__
