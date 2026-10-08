"""Shared wording for error and warning messages.

``STYLE_GUIDE.md`` §9.3 owns the rules for messages. These helpers give one
wording to the checks that recur across the package, so the same mistake reads
the same way wherever it is raised.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["count", "unknown_names"]


def count(n: int, noun: str, plural: str | None = None) -> str:
    """*n* followed by *noun*, pluralized unless *n* is one.

    Parameters
    ----------
    n : int
        The number to show.
    noun : str
        The singular noun, such as ``"axis"``.
    plural : str or None
        The plural noun, such as ``"axes"``; ``noun + "s"`` when omitted.

    Returns
    -------
    str
        Such as ``"1 axis"`` or ``"2 axes"``.
    """
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def unknown_names(
    noun: str, unknown: Iterable[str], available: Iterable[str], plural: str | None = None
) -> str:
    """The message for a lookup that names something that does not exist.

    Parameters
    ----------
    noun : str
        What the names name, singular, such as ``"level"`` or ``"field"``.
    unknown : iterable of str
        The names that were asked for and do not exist, in the order to show.
    available : iterable of str
        The names that do exist, in the order to show.
    plural : str or None
        The plural of *noun*; ``noun + "s"`` when omitted.

    Returns
    -------
    str
        Such as ``"unknown level 'test'; available levels: ['quantile']"``.
    """
    plural = plural or noun + "s"
    names, have = list(unknown), list(available)
    head = f"unknown {noun} {names[0]!r}" if len(names) == 1 else f"unknown {plural} {names}"
    tail = f"available {plural}: {have}" if have else f"there are no {plural}"
    return f"{head}; {tail}"
