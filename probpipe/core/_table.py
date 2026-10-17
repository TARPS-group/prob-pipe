"""The table display of a record of summaries.

Design II.4 states the rule. A numeric record displays as a table when its
top-level fields are records over the same paths, with the same shape at each
path, and at least one of them records how it was computed. The record
:func:`~probpipe.summarize` returns is one, and so is a record built by hand
from the summaries operations return. A record of plain numbers, or the draw or
mean of a law whose groups share their field names, records no computation and
shows one line for each path.

A row is a path whose value is a scalar, or one element of a path whose value
is an array, as ``y[0]``. NumPy's print options shorten the rows of an array of
more than ``threshold`` elements to ``edgeitems`` at each end, and format the
entries of each column, so a column's entries share one precision. Only the
entries shown are read from the device.

The line below the table states how each column was computed: ``exact`` for an
exact route, ``Monte Carlo`` with the number of draws for a route that records
its draws, and ``approximate`` for any other approximate route. A record whose
provenance records a description of each column under :data:`COMPUTATIONS`, as
the record ``summarize`` returns does, takes those descriptions, and any other
record reads each column's own provenance.
"""

from __future__ import annotations

import html
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._array_backend import _event_shape_of, _to_numpy_array
from ._repr import VALUE_INDENT
from .tracked import TrackedTerm

__all__ = [
    "COMPUTATIONS",
    "RecordTable",
    "batch_tables_html",
    "batch_tables_text",
    "column_provenance",
    "computation",
    "path_shapes",
    "record_table",
]

#: The provenance metadata entry that describes how each column of a record of
#: summaries was computed: a mapping from each column's name to its description,
#: or, for a batch, to the list of its elements' descriptions in C order.
COMPUTATIONS = "computations"

#: The space between two columns of a table.
_GAP = "  "

#: The text of a row that stands for the elements a shortened path leaves out.
_SKIPPED = "..."

#: A separator NumPy cannot print inside an entry, which splits a printed column.
_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class RecordTable:
    """The rows, the columns, and the footer of a record that displays as a table.

    Attributes
    ----------
    rows : tuple of str
        Each row's label, as ``mu`` or ``y[0]``, with ``...`` where a shortened
        path leaves elements out.
    columns : tuple of str
        The top-level fields, in order.
    cells : tuple of tuple of str
        Each column's formatted entries, one for each row.
    footer : str
        How each column was computed, as ``mean, sd: exact.``.
    """

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    cells: tuple[tuple[str, ...], ...]
    footer: str

    def lines(self, indent: str = "") -> list[str]:
        """The header line of column names and one line for each row, aligned and indented.

        Parameters
        ----------
        indent : str, optional
            The text each line starts with.

        Returns
        -------
        list of str
            The lines, without the footer.
        """
        label_width = max((len(row) for row in self.rows), default=0)
        widths = [
            max(len(name), *(len(cell) for cell in column))
            for name, column in zip(self.columns, self.cells, strict=True)
        ]
        header = " " * label_width + "".join(
            f"{_GAP}{name:>{width}}" for name, width in zip(self.columns, widths, strict=True)
        )
        lines = [indent + header.rstrip()]
        for index, row in enumerate(self.rows):
            entries = "".join(
                f"{_GAP}{column[index]:>{width}}"
                for column, width in zip(self.cells, widths, strict=True)
            )
            lines.append(f"{indent}{row:<{label_width}}{entries}".rstrip())
        return lines

    def html(self) -> str:
        """The table as an HTML table, with the footer below it, every cell escaped."""
        head = "".join(f"<th>{html.escape(name)}</th>" for name in self.columns)
        body = "".join(
            f'<tr><th style="text-align: left">{html.escape(row)}</th>'
            + "".join(f"<td>{html.escape(column[index])}</td>" for column in self.cells)
            + "</tr>"
            for index, row in enumerate(self.rows)
        )
        return (
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"
            f"<p>{html.escape(self.footer)}</p>"
        )


def record_table(record: Any) -> RecordTable | None:
    """The table of *record*, or ``None`` when the record does not display as one.

    A record displays as a table when it has at least one top-level field, every
    top-level field is a record, those records have the same field paths with
    the same shape at each path, and at least one of them records how it was
    computed. A record whose entries cannot be read, as inside a JAX trace,
    gives ``None`` too.

    Parameters
    ----------
    record : NumericRecord
        The record to display.

    Returns
    -------
    RecordTable or None
        The rows, the columns, the formatted cells, and the footer.
    """
    children = _columns(record)
    if children is None:
        return None
    recorded = _recorded_computations(record.provenance)
    if recorded is None:
        computations = {
            name: computation(column_provenance(child)) for name, child in children.items()
        }
    else:
        computations = {name: recorded.get(name) for name in children}
    return _table(children, computations)


def column_provenance(column: Any) -> Any:
    """The provenance of a column: the column's own, or that of the one term it holds.

    Parameters
    ----------
    column : Record
        A top-level field of a record of summaries.

    Returns
    -------
    Provenance or None
        The provenance, or ``None`` when the column records none.
    """
    if column.provenance is not None:
        return column.provenance
    leaves = [leaf for _, leaf in column._walk_leaves()]
    if len(leaves) == 1 and isinstance(leaves[0], TrackedTerm):
        return leaves[0].provenance
    return None


def computation(provenance: Any) -> str | None:
    """How a summary was computed, as its provenance records it.

    An exact route reads ``exact``. An approximate route reads ``Monte Carlo,
    N draws`` when it records one count of Monte Carlo draws, ``Monte Carlo``
    when it records several, and ``approximate`` when it records none, as a
    route that takes its draws in a nested call does.

    Parameters
    ----------
    provenance : Provenance or None
        The provenance of the summary.

    Returns
    -------
    str or None
        The description, or ``None`` when the provenance records no exactness.
    """
    if provenance is None:
        return None
    exact = (provenance.metadata or {}).get("exact")
    if exact is True:
        return "exact"
    if exact is not False:
        return None
    # The replay anchor's format belongs to functions/, which core/ imports
    # lazily (STYLE_GUIDE.md §6).
    from ..functions._recipe import monte_carlo_draws

    counts = set(monte_carlo_draws(provenance).values())
    if len(counts) == 1:
        return f"Monte Carlo, {counts.pop():,} draws"
    return "Monte Carlo" if counts else "approximate"


def path_shapes(record: Any) -> dict[str, tuple[int, ...]]:
    """Each field path of *record* and the shape of its value, in canonical order.

    Parameters
    ----------
    record : Record
        The record.

    Returns
    -------
    dict of str to tuple of int
        The shape of each path's value.
    """
    from .record import _raw_leaf

    return {path: tuple(_event_shape_of(_raw_leaf(leaf))) for path, leaf in record._walk_leaves()}


def batch_tables_text(batch: Any, head: str | None) -> str | None:
    """A batch of records of summaries as one table for each element, or ``None``.

    Each element's table is titled by its levels, as ``law=0``. Under *head*
    each title is indented by two spaces and each table by four, and the line
    that states how the columns were computed follows each table, or follows
    the last when every element's line is the same. Without a head the titles
    and tables start at the margin and the lines are left out, as ``str()`` of
    a record leaves its line out.

    Parameters
    ----------
    batch : NumericRecordBatch
        The batch to display.
    head : str or None
        The line above the tables, or ``None`` for none.

    Returns
    -------
    str or None
        The lines of the display, or ``None`` when the elements do not display
        as tables, as :func:`record_table` states, or their entries cannot be
        read.
    """
    shown = _batch_tables(batch)
    if shown is None:
        return None
    footers = {table.footer for _, table in shown if table is not None}
    title_indent = "" if head is None else VALUE_INDENT
    table_indent = "" if head is None else VALUE_INDENT * 2
    lines = [] if head is None else [head]
    for title, table in shown:
        lines.append(title_indent + title)
        if table is not None:
            lines.extend(table.lines(table_indent))
            if head is not None and len(footers) > 1:
                lines.append(table_indent + table.footer)
    if head is not None and len(footers) == 1:
        lines.append(footers.pop())
    return "\n".join(lines)


def batch_tables_html(batch: Any, head: str) -> str | None:
    """A batch of records of summaries as one HTML table for each element, or ``None``.

    Each table is titled by its element's levels and followed by the line that
    states how its columns were computed.

    Parameters
    ----------
    batch : NumericRecordBatch
        The batch to display.
    head : str
        The text above the tables.

    Returns
    -------
    str or None
        The HTML of the display, or ``None`` when the text display is ``None``.
    """
    shown = _batch_tables(batch)
    if shown is None:
        return None
    parts = [f'<code style="white-space: pre">{html.escape(head)}</code>']
    for title, table in shown:
        if table is None:
            parts.append(f"<p>{_SKIPPED}</p>")
        else:
            parts.append(f"<p><b>{html.escape(title)}</b></p>{table.html()}")
    return f"<div>{''.join(parts)}</div>"


# ---------------------------------------------------------------------------
# Building a table
# ---------------------------------------------------------------------------


def _columns(record: Any) -> dict[str, Any] | None:
    """The top-level fields of *record*, or ``None`` unless each is a record."""
    from .record import Record

    children = dict(record._tree)
    if not children or not all(isinstance(child, Record) for child in children.values()):
        return None
    return children


def _recorded_computations(provenance: Any) -> Mapping[str, Any] | None:
    """The description of each column that *provenance* records, or ``None``."""
    recorded = None if provenance is None else (provenance.metadata or {}).get(COMPUTATIONS)
    return recorded if isinstance(recorded, Mapping) else None


def _table(
    children: Mapping[str, Any], computations: Mapping[str, str | None]
) -> RecordTable | None:
    """The table of the columns *children*, each described by *computations*.

    ``None`` when no column has a description, the columns differ in their
    paths or shapes, or their entries cannot be read.
    """
    from .record import _raw_leaf

    described = {name: text for name, text in computations.items() if text is not None}
    if not described:
        return None
    columns = {
        name: {path: _raw_leaf(leaf) for path, leaf in child._walk_leaves()}
        for name, child in children.items()
    }
    shapes = {
        path: tuple(_event_shape_of(value)) for path, value in next(iter(columns.values())).items()
    }
    for column in columns.values():
        if {path: tuple(_event_shape_of(value)) for path, value in column.items()} != shapes:
            return None
    values = [value for column in columns.values() for value in column.values()]
    if any(isinstance(value, jax.core.Tracer) for value in values):
        return None
    rows = _rows(shapes)
    try:
        cells = tuple(_column_cells(column, rows) for column in columns.values())
    except Exception:  # a display shows the summary alone rather than raise, as _repr.py does
        return None
    return RecordTable(tuple(row.label for row in rows), tuple(children), cells, _footer(described))


@dataclass(frozen=True)
class _Row:
    """One row of a table: its label, and the path and flat index of the entry it shows.

    A row of ``...`` has no path, since it stands for the elements a shortened
    path leaves out.
    """

    label: str
    path: str | None = None
    index: int = 0


def _shown_positions(size: int, rows_each: int = 1) -> list[int | None]:
    """The flat positions shown of *size* items of *rows_each* rows, ``None`` for ``...``.

    As NumPy shortens an array, the items are shortened to ``edgeitems`` at each
    end when their rows number more than ``threshold``.
    """
    options = np.get_printoptions()
    edge = options["edgeitems"]
    if size * rows_each > options["threshold"] and size > 2 * edge:
        return [*range(edge), None, *range(size - edge, size)]
    return list(range(size))


def _rows(shapes: Mapping[str, tuple[int, ...]]) -> list[_Row]:
    """The rows of a table over paths of the given *shapes*, in order.

    A scalar path takes one row. An array path takes one row for each element, in
    C order, shortened past ``threshold`` elements to ``edgeitems`` at each end.
    """
    rows: list[_Row] = []
    for path, shape in shapes.items():
        if shape == ():
            rows.append(_Row(path, path))
            continue
        for index in _shown_positions(prod(shape)):
            if index is None:
                rows.append(_Row(_SKIPPED))
                continue
            element = ", ".join(str(int(i)) for i in np.unravel_index(index, shape))
            rows.append(_Row(f"{path}[{element}]", path, index))
    return rows


def _entries(value: Any, flat: Sequence[int]) -> np.ndarray:
    """The entries of *value* at the flat indices *flat*, read to the host."""
    if isinstance(value, jax.Array):
        return np.asarray(jnp.ravel(value)[np.asarray(flat)])
    return np.asarray(_to_numpy_array(value)).reshape(-1)[np.asarray(flat)]


def _column_cells(column: Mapping[str, Any], rows: Sequence[_Row]) -> tuple[str, ...]:
    """One column's cells, one for each row, its entries formatted together by NumPy.

    Each path's shown entries are read at once, and a row of ``...`` shows ``...``.
    """
    wanted: dict[str, list[int]] = {}
    for row in rows:
        if row.path is not None:
            wanted.setdefault(row.path, []).append(row.index)
    read = {path: iter(_entries(column[path], flat)) for path, flat in wanted.items()}
    entries = np.asarray([next(read[row.path]) for row in rows if row.path is not None])
    text = np.array2string(
        entries, separator=_SEPARATOR, threshold=sys.maxsize, max_line_width=sys.maxsize
    )
    formatted = iter(text[1:-1].split(_SEPARATOR))
    return tuple(_SKIPPED if row.path is None else next(formatted) for row in rows)


def _footer(computations: Mapping[str, str]) -> str:
    """The columns grouped by how each was computed, as ``mean, sd: exact.``, in order."""
    groups: dict[str, list[str]] = {}
    for name, text in computations.items():
        groups.setdefault(text, []).append(name)
    return " ".join(f"{', '.join(names)}: {text}." for text, names in groups.items())


# ---------------------------------------------------------------------------
# A batch of tables
# ---------------------------------------------------------------------------


def _batch_tables(batch: Any) -> list[tuple[str, RecordTable | None]] | None:
    """Each shown element's title and table, with ``("...", None)`` where elements are left out.

    The elements display as tables when the batch's provenance describes each
    column of each element under :data:`COMPUTATIONS`, as the batch that
    ``summarize`` returns for a batch of laws does. The elements are shortened
    when their rows number more than ``threshold``.
    """
    recorded = _recorded_computations(batch.provenance)
    shape = tuple(batch.batch_shape)
    size = prod(shape)
    if recorded is None or size == 0:
        return None

    def table_at(position: int) -> RecordTable | None:
        children = _columns(batch[_index(shape, position)])
        if children is None:
            return None
        return _table(children, {name: texts[position] for name, texts in recorded.items()})

    first = table_at(0)
    if first is None:
        return None
    shown: list[tuple[str, RecordTable | None]] = []
    for position in _shown_positions(size, len(first.rows)):
        if position is None:
            shown.append((_SKIPPED, None))
            continue
        table = first if position == 0 else table_at(position)
        if table is None:
            return None
        shown.append((_title(batch, _index(shape, position)), table))
    return shown


def _index(shape: tuple[int, ...], position: int) -> tuple[int, ...]:
    """The batch index of the flat *position* in a batch of *shape*."""
    return tuple(int(i) for i in np.unravel_index(position, shape))


def _title(batch: Any, index: tuple[int, ...]) -> str:
    """The element at *index* titled by its levels, as ``law=0`` or ``chain=0, draw=1``."""
    parts, start = [], 0
    for name, group in zip(batch.level_names, batch.axis_groups, strict=True):
        level = index[start : start + len(group)]
        start += len(group)
        parts.append(f"{name}={level[0] if len(level) == 1 else level}")
    return ", ".join(parts)
