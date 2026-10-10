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

The module also groups a label where a derived label is built from it, and
formats the notation ``label(signature)`` that ``str()`` of a law, a kernel, or
a function shows.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable, Mapping
from keyword import iskeyword
from math import prod
from typing import Any

import numpy as np

__all__ = [
    "WIDTH",
    "call_repr",
    "format_components",
    "format_default",
    "format_dtype",
    "format_levels",
    "format_names",
    "format_notation",
    "format_signature",
    "format_value",
    "mapping_repr",
    "public_class_name",
    "sequence_repr",
    "term_repr",
    "type_name",
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
    that is itself such a repr is laid out at the width its line leaves. When a
    name is no Python identifier, as a component ``mean(mu)`` is, the keywords
    are written in order as one ``**{'name': value, ...}`` argument, which the
    constructor takes alike.
    """
    parts = [("", value) for value in positional]
    keywords = list(keywords)
    if all(name.isidentifier() and not iskeyword(name) for name, _ in keywords):
        parts.extend((f"{name}=", value) for name, value in keywords)
    else:
        parts.append(("**", mapping_repr(dict(keywords))))
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

    Returns
    -------
    str
        A ``_Layout``, which an enclosing repr lays out again where its line starts.
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


def type_name(value: Any) -> str:
    """The name of *value*'s type as an error message shows it.

    A JAX array shows as ``jax.Array`` rather than as its private concrete
    class, and a value of a ProbPipe class shows as its first public class. Any
    other value shows as its own class, which is the caller's.
    """
    cls = type(value)
    package = cls.__module__.split(".")[0]
    if package in {"jax", "jaxlib"} and hasattr(value, "shape"):
        return "jax.Array"
    return public_class_name(cls) if package == "probpipe" else cls.__name__


# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------

#: The symbol each binary operator writes in the label its result derives.
BINARY_SYMBOLS = {
    "add": "+", "sub": "-", "mul": "*", "matmul": "@", "truediv": "/", "floordiv": "//",
    "mod": "%", "pow": "**", "lshift": "<<", "rshift": ">>", "and": "&", "xor": "^", "or": "|",
    "lt": "<", "le": "<=", "eq": "==", "ne": "!=", "gt": ">", "ge": ">=",
}  # fmt: skip

#: The symbol that joins the labels of a product's factors, as in ``lik·prior``.
PRODUCT_SYMBOL = "·"

#: The symbols that make a label compound as a top-level word: the binary
#: operators', of which ``|`` also reads as conditioning, and the ``~`` of a
#: draw, as in ``mu ~ prior``.
_OPERATOR_SYMBOLS = frozenset(BINARY_SYMBOLS.values()) | {"~"}

#: The prefixes of the unary operators' forms that apply an operator to what follows.
_UNARY_PREFIXES = ("-", "+", "~")

#: The prefix of a score's label, as in ``log prior(mu)``.
_SCORE_PREFIX = "log "


def _top_level_words(label: str) -> list[str]:
    """*label* split at the spaces outside its parentheses and brackets."""
    words, current, depth = [], [], 0
    for char in label:
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        if char == " " and depth == 0:
            words.append("".join(current))
            current = []
        else:
            current.append(char)
    words.append("".join(current))
    return words


def _top_level_text(label: str) -> str:
    """The characters of *label* outside its parentheses and brackets."""
    kept, depth = [], 0
    for char in label:
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        elif depth == 0:
            kept.append(char)
    return "".join(kept)


def is_compound(label: str) -> bool:
    """Whether *label* is compound, which a derived label parenthesizes.

    A label is compound when one of these holds:

    1. a top-level word is an operator's symbol, as in ``effect + 1.0``,
       ``model | y``, or the draw ``mu ~ prior``;
    2. it joins labels with ``·`` outside its parentheses and brackets, as the
       product ``lik·prior`` does;
    3. it opens with a unary operator, as ``-effect`` does, or with ``log``, as
       the score ``log prior(mu)`` does.

    A call such as ``prior(mu)`` is one word with no top-level symbol, so it is
    not compound.
    """
    return (
        label.startswith((*_UNARY_PREFIXES, _SCORE_PREFIX))
        or any(word in _OPERATOR_SYMBOLS for word in _top_level_words(label))
        or PRODUCT_SYMBOL in _top_level_text(label)
    )


def is_product(label: str) -> bool:
    """Whether *label* is a product of labels: one word that joins labels with ``·``.

    A product, such as ``lik·prior`` or ``lik·(model | y)``, joins a further
    factor's label as it is, so labels join associatively.
    """
    return (
        len(_top_level_words(label)) == 1
        and PRODUCT_SYMBOL in _top_level_text(label)
        and not label.startswith(_UNARY_PREFIXES)
    )


def grouped_label(label: str) -> str:
    """*label* as it reads inside a derived label, grouped so it reads as one operand.

    A compound label (:func:`is_compound`) is parenthesized, so the derived
    label states the order of evaluation, as in ``(x·y)[sample=0:2]``. Any
    other label with a top-level space, such as a user's label ``other
    effect``, is bracketed, so it reads as one label. A label of one word, a
    call such as ``prior(mu)`` included, is used as it is.
    """
    if is_compound(label):
        return f"({label})"
    return f"[{label}]" if len(_top_level_words(label)) > 1 else label


def format_signature(
    components: Iterable[str],
    given: Iterable[str] = (),
    fixed: Iterable[str] = (),
    defaults: Mapping[str, str] | None = None,
) -> str:
    """The signature of a law, a kernel, or a function, which states what it is over.

    Parameters
    ----------
    components : iterable of str
        A law's or a kernel's event components, or a function's parameters, in
        declaration order.
    given : iterable of str, optional
        A kernel's given slots, in declaration order.
    fixed : iterable of str, optional
        The paths fixed at given values.
    defaults : Mapping[str, str], optional
        The formatted default of each given slot or parameter that has one,
        as :func:`format_default` gives it.

    Returns
    -------
    str
        The components joined by ``", "``, then `` | `` and the given slots when
        there are any, then ``; `` and the fixed paths when there are any, as
        ``y, mu``, ``y | beta``, or ``y | sigma; beta``. A name with a default
        reads ``name=value``, as ``y | K, n0=50.0``.
    """
    defaults = defaults or {}

    def entry(name: str) -> str:
        return f"{name}={defaults[name]}" if name in defaults else name

    text = ", ".join(entry(name) for name in components)
    given, fixed = list(given), list(fixed)
    if given:
        text = f"{text} | {', '.join(entry(name) for name in given)}"
    if fixed:
        text = f"{text}; {', '.join(fixed)}"
    return text


def format_default(value: Any) -> str:
    """The default of a given slot or a parameter as a signature shows it.

    A number, a string, ``None``, or an array of no axes reads as its value,
    as ``50.0``, and any other value as ``…``, since its entries would not
    read as one name's value.
    """
    if isinstance(value, bool | int | float | complex | str | type(None)):
        return repr(value)
    if getattr(value, "shape", None) == () and getattr(value, "dtype", None) is not None:
        return format_value(value)
    return "…"


def format_components(components: Iterable[str]) -> str:
    """Several components as one label: one as it is, as ``mu``, and several in parentheses, as ``(y, mu)``.

    A draw's components read this way before its ``~``, and the atoms of a
    posterior are labeled this way by its components.

    Parameters
    ----------
    components : iterable of str
        A law's event components, in declaration order.

    Returns
    -------
    str
        The one component, or the components joined by ``", "`` in parentheses.
    """
    names = list(components)
    return names[0] if len(names) == 1 else f"({', '.join(names)})"


def format_notation(label: str, signature: str) -> str:
    """The notation ``label(signature)``, with *label* grouped as :func:`grouped_label` groups it.

    A label of one word is used as it is, as in ``prior(mu)``, and a product
    is parenthesized, as in ``(lik·prior)(y)``, so the call applies to all of it.
    """
    return f"{grouped_label(label)}({signature})"
