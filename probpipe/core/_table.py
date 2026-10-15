"""The table display of a record whose top-level fields are records over the same paths.

Design II.4 states the rule: such a record displays as a table with one row for
each path and one column for each top-level field, in its repr and in a
notebook. The record :func:`~probpipe.summarize` returns is one, and so is a
record built by hand from summaries of one law.

A row is a path whose value is a scalar, or one element of a path whose value
is an array, as ``y[0]``. NumPy's print options shorten the rows of an array of
more than ``threshold`` elements to ``edgeitems`` at each end, and format the
entries of each column, so a column's entries share one precision. Only the
entries shown are read from the device.

The line below the table states how each column was computed, read from the
column's provenance: ``exact`` for an exact route, ``Monte Carlo`` with the
number of draws for a route that records Monte Carlo draws, and
``approximate`` for any other approximate route.
"""

from __future__ import annotations

import html
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._array_backend import _event_shape_of, _to_numpy_array
from .tracked import TrackedTerm

__all__ = [
    "RecordTable",
    "batch_tables_html",
    "batch_tables_text",
    "column_provenance",
    "monte_carlo_draws",
    "record_table",
    "record_table_shape",
]

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
        How each column was computed, or ``""`` when no column records it.
    """

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    cells: tuple[tuple[str, ...], ...]
    footer: str

    def lines(self) -> list[str]:
        """The header line of column names and one line for each row, aligned."""
        label_width = max((len(row) for row in self.rows), default=0)
        widths = [
            max(len(name), *(len(cell) for cell in column))
            for name, column in zip(self.columns, self.cells, strict=True)
        ]
        header = " " * label_width + "".join(
            f"{_GAP}{name:>{width}}" for name, width in zip(self.columns, widths, strict=True)
        )
        lines = [header.rstrip()]
        for index, row in enumerate(self.rows):
            entries = "".join(
                f"{_GAP}{column[index]:>{width}}"
                for column, width in zip(self.cells, widths, strict=True)
            )
            lines.append(f"{row:<{label_width}}{entries}".rstrip())
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
        footer = f"<p>{html.escape(self.footer)}</p>" if self.footer else ""
        return (
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>{footer}"
        )


def record_table(record: Any) -> RecordTable | None:
    """The table of *record*, or ``None`` when the record does not display as one.

    A record displays as a table when it has at least one top-level field, every
    top-level field is a record, and those records have the same field paths
    with the same shape at each path. A record whose entries cannot be read, as
    inside a JAX trace, gives ``None`` too.

    Parameters
    ----------
    record : NumericRecord
        The record to display.

    Returns
    -------
    RecordTable or None
        The rows, the columns, the formatted cells, and the footer.
    """
    from .record import Record, _raw_leaf

    children = dict(record._tree)
    if not children or not all(isinstance(child, Record) for child in children.values()):
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
    except Exception:  # a display shows the summary alone rather than raise
        return None
    footer = _footer({name: column_provenance(child) for name, child in children.items()})
    return RecordTable(tuple(row.label for row in rows), tuple(children), cells, footer)


def record_table_shape(spec: Any) -> bool:
    """Whether a record of *spec* displays as a table, read from the spec alone.

    Parameters
    ----------
    spec : RecordSpec
        The record's spec, such as a batch's element spec.

    Returns
    -------
    bool
        Whether the spec has at least one top-level field, every top-level field
        is a record, and those records have the same field paths with the same
        shape at each path.
    """
    from ._record_spec import RecordSpec

    children = dict(getattr(spec, "children", {}))
    if not children or not all(isinstance(child, RecordSpec) for child in children.values()):
        return False
    shapes = [
        {path: getattr(leaf, "shape", None) for path, leaf in child.items()}
        for child in children.values()
    ]
    return all(shape == shapes[0] for shape in shapes)


@dataclass(frozen=True)
class _Row:
    """One row of a table: its label, and the path and flat index of the entry it shows.

    A row of ``...`` has no path, since it stands for the elements a shortened
    path leaves out.
    """

    label: str
    path: str | None = None
    index: int = 0


def _rows(shapes: Mapping[str, tuple[int, ...]]) -> list[_Row]:
    """The rows of a table over paths of the given *shapes*, in order.

    A scalar path takes one row. An array path takes one row for each element, in
    C order, shortened past ``threshold`` elements to ``edgeitems`` at each end.
    """
    options = np.get_printoptions()
    edge = options["edgeitems"]
    rows: list[_Row] = []
    for path, shape in shapes.items():
        if shape == ():
            rows.append(_Row(path, path))
            continue
        size = prod(shape)
        flat: list[int | None] = list(range(size))
        if size > options["threshold"] and size > 2 * edge:
            flat = [*range(edge), None, *range(size - edge, size)]
        for index in flat:
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


def column_provenance(column: Any) -> Any:
    """The provenance of a column: the column's own, or that of the one term it holds."""
    if column.provenance is not None:
        return column.provenance
    leaves = [leaf for _, leaf in column._walk_leaves()]
    if len(leaves) == 1 and isinstance(leaves[0], TrackedTerm):
        return leaves[0].provenance
    return None


def _fidelity(provenance: Any) -> str | None:
    """How a column was computed, as its provenance records it, or ``None`` when it does not.

    An exact route reads ``exact``. An approximate route reads ``Monte Carlo,
    N draws`` when its provenance records one count of Monte Carlo draws, and
    ``approximate`` otherwise.
    """
    if provenance is None:
        return None
    exact = (provenance.metadata or {}).get("exact")
    if exact is True:
        return "exact"
    if exact is not False:
        return None
    counts = {prod(effect.get("sample_shape") or ()) for effect in _monte_carlo_effects(provenance)}
    if len(counts) == 1:
        return f"Monte Carlo, {counts.pop():,} draws"
    return "Monte Carlo" if counts else "approximate"


def _monte_carlo_effects(provenance: Any) -> list[Mapping[str, Any]]:
    """The Monte Carlo draws a call's provenance records, each with its sample shape and path."""
    replay = (provenance.controls or {}).get("replay")
    plan = replay.get("plan") if isinstance(replay, Mapping) else None
    effects = plan.get("expected_effects") if isinstance(plan, Mapping) else None
    return [
        effect
        for effect in effects or ()
        if isinstance(effect, Mapping) and effect.get("execution_mode") == "monte_carlo"
    ]


def monte_carlo_draws(provenance: Any) -> dict[tuple[str, ...], int]:
    """The Monte Carlo draws a call's provenance records, by the path of the part each drew.

    Parameters
    ----------
    provenance : Provenance or None
        The provenance of a summary.

    Returns
    -------
    dict of tuple of str to int
        The number of draws of each part, keyed by the part's path as a tuple of
        names, with ``()`` for the whole law.
    """
    if provenance is None:
        return {}
    return {
        tuple(effect.get("record_path") or ()): prod(effect.get("sample_shape") or ())
        for effect in _monte_carlo_effects(provenance)
    }


def _footer(provenances: Mapping[str, Any]) -> str:
    """The columns grouped by how each was computed, as ``mean, sd: exact.``, in order."""
    groups: dict[str, list[str]] = {}
    for name, provenance in provenances.items():
        fidelity = _fidelity(provenance)
        if fidelity is not None:
            groups.setdefault(fidelity, []).append(name)
    return " ".join(f"{', '.join(names)}: {fidelity}." for fidelity, names in groups.items())


def _element_titles(batch: Any) -> Iterable[tuple[tuple[int, ...] | None, str]]:
    """Each element's index and its levels as a title, as ``law=0``, ``None`` for ``...``.

    The elements are shortened as NumPy shortens an array: past ``threshold``
    elements, ``edgeitems`` at each end.
    """
    options = np.get_printoptions()
    edge = options["edgeitems"]
    shape = tuple(batch.batch_shape)
    size = prod(shape)
    flat: list[int | None] = list(range(size))
    if size > options["threshold"] and size > 2 * edge:
        flat = [*range(edge), None, *range(size - edge, size)]
    for position in flat:
        if position is None:
            yield None, _SKIPPED
            continue
        index = tuple(int(i) for i in np.unravel_index(position, shape))
        parts, start = [], 0
        for name, group in zip(batch.level_names, batch.axis_groups, strict=True):
            level = index[start : start + len(group)]
            start += len(group)
            parts.append(f"{name}={level[0] if len(level) == 1 else level}")
        yield index, ", ".join(parts)


def batch_tables_text(batch: Any, head: str | None) -> str | None:
    """A batch of records as one table for each element, under *head*, or ``None``.

    Each element's table is titled by its levels. The result is ``None`` when the
    elements do not display as tables or their entries cannot be read.

    Parameters
    ----------
    batch : NumericRecordBatch
        The batch to display.
    head : str or None
        The line above the tables, or ``None`` for none.

    Returns
    -------
    str or None
        The lines of the display.
    """
    lines = [] if head is None else [head]
    for index, title in _element_titles(batch):
        if index is None:
            lines.append(title)
            continue
        table = record_table(batch[index])
        if table is None:
            return None
        lines.append(title)
        lines.extend(table.lines())
    return "\n".join(lines)


def batch_tables_html(batch: Any, head: str) -> str | None:
    """A batch of records as one HTML table for each element, or ``None``, as the text form.

    Parameters
    ----------
    batch : NumericRecordBatch
        The batch to display.
    head : str
        The text above the tables.

    Returns
    -------
    str or None
        The HTML of the display.
    """
    parts = [f'<code style="white-space: pre">{html.escape(head)}</code>']
    for index, title in _element_titles(batch):
        if index is None:
            parts.append(f"<p>{_SKIPPED}</p>")
            continue
        table = record_table(batch[index])
        if table is None:
            return None
        parts.append(f"<p><b>{html.escape(title)}</b></p>{table.html()}")
    return f"<div>{''.join(parts)}</div>"
