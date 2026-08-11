"""RecordBatch — the batch form of the record kind, stored columnar.

A :class:`RecordBatch` is a batch of :class:`~probpipe.Record`\\ s that all
conform to one shared ``EventTemplate``: the batched value a ``Function``
produces and consumes, such as the many draws a ``sample`` yields.

**Storage is columnar, keyed by leaf path.** One column per *field* — not per
top-level child — each holding that field's values for every element, shaped
``(*batch_shape, *event_shape)``. A column is therefore what a field access
hands back directly, and an element is assembled from the columns on demand
rather than stored a second time. Keying by leaf path is what lets a nested
field be reached at all: ``batch["outer/a"]`` is a column like any other, and
``batch["outer"]`` is the sub-batch over the columns beneath it.

**A batch is a collection, not a named tree.** Unlike a ``Record``, a
``RecordBatch`` does not implement the field-keyed ``Mapping`` protocol: it has
no ``keys()`` / ``values()`` / ``items()`` / ``children`` / ``at_path``, so
``len`` and ``iter`` speak unambiguously about the *batch*. The field structure
is read from :attr:`RecordBatch.event_template`, which is where it belongs. What
``[]`` does depends on the key: a position addresses the batch axes, a name
addresses a field within every element.

What does carry over is the **structure-preserving transforms** —
``with_path_names``, ``without``, ``replace``, ``merge``, ``map`` — since those
act on the elements rather than navigating the batch as a tree. Each is its
record counterpart applied to every element at once, which for field-wise
storage means applying it to the fields.

See design III.3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from itertools import combinations
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np

from ._array_backend import _is_numeric_dtype, _to_jax_array
from ._batch import Batch, BatchSpec, _axis_groups_for
from ._function_batch import FunctionBatch
from ._object_batch import _from_iterable, _frozen_object_column, _is_object_array
from ._opaque_batch import OpaqueBatch
from .event_template import (
    ArraySpec,
    EventTemplate,
    FunctionSpec,
    NumericEventTemplate,
    OpaqueSpec,
    RecordSpec,
    ValueSpec,
    _record_declaration_template,
    _to_record_declaration,
)
from .named_tree import _PATH_SEP, _unflatten_paths
from .provenance import Provenance
from .record import Record

__all__ = ["RecordBatch"]


class RecordBatch(Batch[Record]):
    """A batch of records sharing one ``EventTemplate``, stored as columns.

    Parameters
    ----------
    fields : Mapping of str to array
        The field columns, keyed by **leaf path** (``"outer/a"``) or given as a
        nested mapping, which is flattened to leaf paths. Each column holds one
        field's values across the batch, shaped ``(*batch_shape, *event_shape)``
        where the event shape is the field spec's for an ``ArraySpec`` and empty
        otherwise — so a field that is not an array takes an object array, one
        entry per element. The keys must be exactly the fields of *element_spec*.
    level_names : str or iterable of str
        One name per level, outermost first; a single string names a single
        level. There is no default, for the reason
        :class:`~probpipe.core._batch.Batch` gives: a level is named so that
        operations can align operands by meaning.
    element_spec : RecordSpec or EventTemplate
        What every element satisfies, as the spec or as the bare template it
        wraps — the two denote the same space. Required: a batch cannot recover
        an element's event shape from a column without it, since the column
        carries the batch axes and the event axes together.
    axis_groups : iterable of iterable of int, optional
        The axis *sizes* each level holds, in order, tiling ``batch_shape``.
        Defaults to one axis per level, which requires as many names as there
        are batch axes.
    name : str, optional
        The batch's name. Defaults to the class name lowercased, marked
        auto-derived.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given. A batch left
        unnamed is auto-named regardless.
    provenance : Provenance, optional
        How this batch was produced.

    Attributes
    ----------
    spec : BatchSpec
        This batch's own type, at the family kind over its elements'.
    element_spec : RecordSpec
        What every element satisfies — a view on :attr:`spec`.
    event_template : EventTemplate
        The structure of one element — a view on :attr:`element_spec`.

    Raises
    ------
    TypeError
        If *element_spec* is not a ``RecordSpec`` or ``EventTemplate``; if
        *fields* is not a mapping; if a column reports no ``shape``, which is
        what a batch axis is read from; or if an entry of a non-array column is
        not admitted by its field's spec.
    ValueError
        If *fields* is empty; if its keys are not exactly the fields of
        *element_spec*; if a column's trailing axes are not the event shape its
        field declares; if two columns disagree on the batch axes; if a field
        declares a symbolic dimension, which gives its event shape no size to
        split by; if a column leaves no batch axis; or if *axis_groups* does not
        tile ``batch_shape``.

    Notes
    -----
    Construction requires at least one element, while *selecting* none is
    allowed: ``batch[0:0]`` is a batch of nothing, as the level algebra intends.
    The asymmetry is the one the object batches state — an empty literal is
    almost always a mistake, and a shape cannot be inferred from it.

    An **object** column is copied and frozen, so a caller keeping a handle on
    what they passed cannot afterwards write a value the batch's spec refuses.
    Only the pointer array is copied, so the elements stay shared. An array
    column is stored as given: a JAX array is already immutable, and a numpy one
    follows the aliasing convention the single-record types set.

    Reading one element materializes it from the columns, which costs one gather
    per field, so iterating a batch is proportional to elements × fields. Reading
    a *column* costs nothing of the sort — prefer it where a whole field will do.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from probpipe import EventTemplate
    >>> batch = RecordBatch({"x": jnp.zeros((3, 2))}, "draw",
    ...                     element_spec=EventTemplate(x=(2,)), name="draws")
    >>> batch.batch_shape
    (3,)
    >>> batch["x"].shape
    (3, 2)
    >>> batch[0].name
    'draws[draw=0]'
    """

    _columns: dict[str, Any]

    __slots__ = ("_columns",)

    #: Completes "every entry of the column at ... must ..." in the refusal a bad
    #: entry earns. A subclass that narrows what a column may hold supplies its own.
    _entry_rule = "satisfy its field's specification"

    def __init__(
        self,
        fields: Mapping[str, Any],
        level_names: str | Iterable[str],
        *,
        element_spec: RecordSpec | EventTemplate,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        kind = type(self).__name__
        spec = _record_element_spec(element_spec, kind=kind)
        template = spec.event_template
        store = _leaf_keyed_columns(fields, template, kind=kind)
        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)

        batch_shape = _batch_shape_of(store, template, kind=kind)
        groups = _axis_groups_for(batch_shape, names, axis_groups, kind=kind)
        type(self)._check_columns(store, template, kind=kind)
        store = {
            path: _frozen_object_column(column) if _is_object_array(column) else column
            for path, column in store.items()
        }

        object.__setattr__(self, "_columns", store)
        self._init_batch(
            BatchSpec(spec, groups, names),
            name=name if name is not None else kind.lower(),
            name_is_auto=name is None or name_is_auto,
            provenance=provenance,
        )

    # -- the specification --------------------------------------------------

    @property
    def element_spec(self) -> RecordSpec:
        """The :class:`RecordSpec` every element satisfies — a view on ``spec``."""
        spec = self._spec.element_spec
        assert isinstance(spec, RecordSpec)  # narrowed at construction
        return spec

    @property
    def event_template(self) -> EventTemplate:
        """The structure of one element — a view on :attr:`element_spec`."""
        return self.element_spec.event_template

    # -- validation ---------------------------------------------------------

    @classmethod
    def _check_columns(cls, store: dict[str, Any], template: EventTemplate, *, kind: str) -> None:
        """Fail on the first column its field's spec does not admit.

        Checked at construction rather than left to ``is_valid`` because a batch
        asserts its ``element_spec`` of *every* element: a column the field's spec
        refuses makes the batch's own spec a false statement, and where it sits is
        what a caller needs to hear. The shapes are already settled by then, by
        the batch-axis derivation, so what is left is the dtype and the entries.

        An **array** field declares a dtype, so its column is checked against
        it by the rule :meth:`~probpipe.ArraySpec.is_valid` applies to a single
        value: same-kind castable, so a widening or a within-kind narrowing passes
        and a cross-kind conversion does not. Any **other** field holds one entry
        per element, so its column is walked entry by entry, as ``_ObjectBatch``
        walks the elements of a batch that stores them.

        Raises
        ------
        TypeError
            If an array field's column is not a numeric array, or carries a dtype
            its declaration does not admit; if a field that is not an array is
            given a column that is not an object array, since its entries would
            then be array elements rather than the values themselves; or if such
            an entry does not satisfy its field's spec, naming the position.
        """
        for path, column in store.items():
            spec = template[path]
            if isinstance(spec, ArraySpec):
                _check_array_column(column, spec, path=path, kind=kind)
                continue
            if not _is_object_array(column):
                raise TypeError(
                    f"{kind}: the field {path!r} is declared {type(spec).__name__}, which has no "
                    f"stacked form, so its column holds one entry per element as an object "
                    f"array; got a {type(column).__name__}"
                    + (f" of {column.dtype}" if hasattr(column, "dtype") else "")
                )
            for index, entry in np.ndenumerate(column):
                if not spec.is_valid(entry):
                    position = index[0] if len(index) == 1 else index
                    raise TypeError(
                        f"{kind}: every entry of the column at {path!r} must "
                        f"{cls._entry_rule}; the entry at {position} is a "
                        f"{type(entry).__name__}"
                    )

    # -- the storage seam ---------------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> Record:
        """The record at a fully-integer positional *index*, built from the columns.

        A row of columnar storage does not exist until it is built, so this is
        the *materializing* side of both rules
        :meth:`~probpipe.core._batch.Batch._element_at` states: the element takes
        the derived *name*, marked auto, and inherits this batch's provenance.

        The element is constructed against :attr:`element_spec` itself, so it
        stores the very object this batch stores rather than an equal copy: a
        record keeps a supplied declaration verbatim, which makes batch and
        element agree on their schema structurally and costs no allocation per
        row.
        """
        row = {path: column[index] for path, column in self._columns.items()}
        return self._inherit_provenance(
            Record(
                name,
                row,
                event_template=self.element_spec,
                name_is_auto=True,
                # The columns were checked against the element spec at
                # construction, so re-checking each row's leaves repeats work that
                # iteration pays per element.
                _validate_leaves=False,
            )
        )

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the same columns, indexed on the batch axes as given.

        *index* addresses the leading (batch) axes only, so each column keeps its
        event axes and array indexing yields a view rather than a copy. Built
        without ``__init__``, since the spec and the name are already decided and
        re-deriving them from the view's own shape would lose the levels a
        dropped axis came from.
        """
        # ``object.__new__`` for the reason ``TrackedTerm._shallow_copy`` gives: a
        # host's own ``__new__`` may select a class from constructor arguments and
        # must not run again where there are none.
        view = object.__new__(type(self))
        object.__setattr__(
            view, "_columns", {path: column[index] for path, column in self._columns.items()}
        )
        view._init_batch(spec, name=name, name_is_auto=True)
        return view

    def _at_fields(self, path: tuple[str, ...]) -> Any:
        """The batched field at *path*, for a named ``[]`` key.

        A **key** — a path reaching a field — yields that field's column in its
        native batch form: an array for an array field, and the batch form of the
        element kind otherwise, a :class:`FunctionBatch` or an
        :class:`OpaqueBatch`. A path reaching an **interior node** yields the
        sub-batch over the columns beneath it, a view over the same storage with
        the same axis levels.

        Raises
        ------
        KeyError
            If *path* addresses neither a field nor an interior node, naming the
            fields this batch does have.
        """
        joined = _PATH_SEP.join(path)
        template = self.event_template
        if joined in self._columns:
            return self._column_as_batch(joined)
        prefix = joined + _PATH_SEP
        if any(key.startswith(prefix) for key in self._columns):
            return self._field_view(joined, template.at_path(joined))
        raise KeyError(
            f"{joined!r} is neither a field nor an interior node of this "
            f"{type(self).__name__}; its fields are {list(template.keys())}"
        )

    # -- field access -------------------------------------------------------

    def _column_as_batch(self, key: str) -> Any:
        """One field's column, in the batch form its spec calls for.

        An ``ArraySpec`` batches natively — the column *is* the array, with the
        batch axes leading — so it is returned as stored. A callable or an opaque
        value has no such form, so the column is presented as the matching object
        batch over the same elements, carrying this batch's own levels.

        Either way the result is a **view**: the object batch shares this batch's
        column rather than copying it, so reading one field costs nothing per
        element. That is what makes the object column safe to share — this batch
        froze it and checked its entries against the same spec when it was built,
        which is the whole of what the object batch's own constructor would
        redo.
        """
        column = self._columns[key]
        spec = self.event_template[key]
        if isinstance(spec, ArraySpec):
            return column
        name = f"{self.name}[{key!r}]"
        column_spec = BatchSpec(spec, self.axis_groups, self.level_names)
        if isinstance(spec, FunctionSpec):
            return self._inherit_provenance(
                FunctionBatch._over_store(column, spec=column_spec, name=name)
            )
        if isinstance(spec, OpaqueSpec):
            return self._inherit_provenance(
                OpaqueBatch._over_store(column, spec=column_spec, name=name)
            )
        raise TypeError(
            f"the field {key!r} is declared {type(spec).__name__}, which has no batch form yet; "
            f"an array field, a callable field, and an opaque field are the forms a column takes"
        )

    def _field_view(self, path: str, template: EventTemplate) -> Self:
        """A sub-batch over the columns under *path*, as a view.

        *template* is the element structure the view presents, which is this
        batch's own template at *path*. The columns are re-keyed relative to
        *path*, so the view reads as a batch of the sub-records rather than of
        their parents.
        """
        prefix = path + _PATH_SEP
        columns = {
            key[len(prefix) :]: column
            for key, column in self._columns.items()
            if key.startswith(prefix)
        }
        view = object.__new__(type(self))
        object.__setattr__(view, "_columns", columns)
        view._init_batch(
            BatchSpec(_to_record_declaration(template), self.axis_groups, self.level_names),
            name=f"{self.name}[{path!r}]",
            name_is_auto=True,
        )
        return self._inherit_provenance(view)

    def _single_field_view(self, key: str) -> Self:
        """A one-column sub-batch presenting *key* alone, as a view.

        What :meth:`select` hands back. Unlike ``self[key]``, which yields the
        column, this stays a ``RecordBatch``: it carries this batch's levels, so
        an operation aligning operands by level name lines it up with its
        siblings and with the batch it came from.
        """
        view = object.__new__(type(self))
        object.__setattr__(view, "_columns", {key: self._columns[key]})
        view._init_batch(
            BatchSpec(
                _to_record_declaration(EventTemplate({key: self.event_template[key]})),
                self.axis_groups,
                self.level_names,
            ),
            name=f"{self.name}[{key!r}]",
            name_is_auto=True,
        )
        return self._inherit_provenance(view)

    def select(self, *paths: str, **mapping: str) -> dict[str, Self]:
        """Batch views of the named parts, ready to splat into a call.

        The batch counterpart of :meth:`~probpipe.Record.select`, and it resolves
        a path the same way: a **key** reaching a field gives a single-field view,
        and a **partial path** gives the sub-batch under it. Each entry is a
        ``RecordBatch`` view rather than a bare array, so splatting the result
        into a ``Function`` call carries the level names an operation aligns
        operands by. Keywords remap, as on a record: ``select(x="r")`` keys the
        view of ``r`` under ``"x"``.

        Raises
        ------
        KeyError
            If a path addresses neither a field nor an interior node.
        """
        selected = {path: path for path in paths}
        selected.update(mapping)
        return {argument: self._view_at(path) for argument, path in selected.items()}

    def select_all(self) -> dict[str, Self]:
        """Every **top-level** part as a batch view, keyed by its own name.

        Keyed by top-level name rather than by leaf path, as
        :meth:`~probpipe.Record.select_all` is, so the result splats into a call:
        a name binds to a parameter where a ``/``-path could not. A top-level
        name that is an interior node gives the sub-batch beneath it, so the parts
        cover the batch between them.
        """
        return self.select(*self.event_template.children)

    def _view_at(self, path: str) -> Self:
        """A batch view of the field or subtree at *path*.

        The single-field case is not ``self[path]``, which yields the values: a
        view keeps the levels, which is what an operation aligns by.
        """
        if path in self._columns:
            return self._single_field_view(path)
        template = self.event_template
        prefix = path + _PATH_SEP
        if any(key.startswith(prefix) for key in self._columns):
            node = template.at_path(path)
            assert isinstance(node, EventTemplate)  # a prefix of a key is an interior node
            return self._field_view(path, node)
        raise KeyError(
            f"{path!r} is neither a field nor an interior node of this "
            f"{type(self).__name__}; its fields are {list(template.keys())}"
        )

    # -- structural transforms ----------------------------------------------
    #
    # A batch is a collection, so it carries no tree *navigation* — but the
    # structure-preserving transforms do apply, elementwise. Each is the record
    # transform of the same name applied to every element at once, which for
    # field-wise storage means applying it to the fields: the batch axes are
    # untouched throughout, and the levels come through unchanged.

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self:
        """Rename fields ``old -> new`` within every element.

        The element counterpart of :meth:`~probpipe.core._batch.Batch.with_level_names`,
        which renames the *levels*: the two namespaces are independent, and
        renaming one never touches the other. Paths resolve as they do on a
        record — a full path, or a bare name where it is unambiguous — and no
        stored value moves.

        Returns
        -------
        Self
            A batch over the same values and levels, its elements' fields renamed.

        Raises
        ------
        KeyError
            If a name to rename is not a field of the elements.
        ValueError
            If a bare name is ambiguous, a new name is empty or contains ``/``,
            two renames target the same field, or a rename collides with a
            sibling.
        """
        renamed = self.event_template.with_path_names(mapping, **kwargs)
        # ``with_path_names`` leaves field order untouched, so the old and new
        # key sequences correspond position by position.
        moved = dict(zip(self.event_template.keys(), renamed.keys(), strict=True))
        return self._rebuilt(
            {moved[path]: column for path, column in self._columns.items()}, renamed
        )

    def without(self, *paths: str) -> Self:
        """Drop the fields or subtrees at *paths* from every element.

        Each path is a key, dropping one field, or a partial path, dropping the
        subtree beneath it. The surviving fields keep their order and their specs,
        and the batch axes are unchanged.

        Raises
        ------
        KeyError
            If a path is not a field or an interior node of the elements.
        ValueError
            If every field would be dropped — a batch of records with no fields
            is not a value.
        """
        kept = self.event_template.without(*paths)
        return self._rebuilt({path: self._columns[path] for path in kept}, kept)

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Self:
        """Replace the values at the given paths, across the whole batch.

        Each replacement is that field's values for *every* element, shaped
        ``(*batch_shape, *event_shape)`` — the same form the field is read back
        in, not one element's value. Every path must already exist: this edits,
        it does not add. An untouched field keeps its spec; a replaced one takes
        the spec its new values imply.

        Raises
        ------
        KeyError
            If a path is not a field of the elements.
        ValueError
            If no updates are given, or a replacement does not carry the batch
            axes.
        """
        if _updates is not None and updates:
            raise ValueError(
                "replace() takes a path-keyed mapping or keyword updates, not both: a path "
                "given in each would have no unambiguous value"
            )
        edits = {path: _unwrapped_field(value) for path, value in (_updates or updates).items()}
        if not edits:
            raise ValueError(f"{type(self).__name__}.replace() needs at least one update")
        unknown = [path for path in edits if path not in self._columns]
        if unknown:
            raise KeyError(
                f"not fields of this {type(self).__name__}: {sorted(unknown)}; "
                f"replace edits, it does not add"
            )
        columns = {**self._columns, **edits}
        return self._rebuilt(columns, _element_template_for(columns, self, edited=set(edits)))

    def merge(self, other: Self) -> Self:
        """Union this batch's elements with *other*'s, field by field.

        Both batches must span the same axes under the same level names, since
        the result's elements pair up one for one, and their field sets must not
        overlap. Each side keeps its own fields' specs.

        Raises
        ------
        TypeError
            If *other* is not a batch of records.
        ValueError
            If the two disagree on their levels or axes, or their fields overlap.
        """
        if not isinstance(other, RecordBatch):
            raise TypeError(
                f"{type(self).__name__}.merge() takes another batch of records, "
                f"got {type(other).__name__}"
            )
        if (self.axis_groups, self.level_names) != (other.axis_groups, other.level_names):
            raise ValueError(
                f"merge() pairs elements one for one, so both batches span the same axes under "
                f"the same names: {self.level_names}={self.axis_groups} vs "
                f"{other.level_names}={other.axis_groups}"
            )
        overlap = sorted(set(self._columns) & set(other._columns))
        if overlap:
            raise ValueError(f"merge() overlapping field keys: {overlap}")
        merged = self.event_template.merge(other.event_template)
        return self._rebuilt({**self._columns, **other._columns}, merged)

    def map(self, f: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
        """Apply *f* to every field's values, keeping the structure.

        What a batch presents at each field is that field's values across the
        batch, so that is what *f* receives — one call per field, vectorized,
        rather than one per element. A function that preserves the batch axes
        therefore yields a batch of the same shape; each field takes the spec its
        result implies.

        Raises
        ------
        ValueError
            If a result does not carry the batch axes.
        """
        columns = {path: f(column, *args, **kwargs) for path, column in self._columns.items()}
        return self._rebuilt(columns, _element_template_for(columns, self, edited=set(columns)))

    def map_with_keys(self, f: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Self:
        """:meth:`map`, with each field's leaf path passed as the first argument."""
        columns = {path: f(path, column, *args, **kwargs) for path, column in self._columns.items()}
        return self._rebuilt(columns, _element_template_for(columns, self, edited=set(columns)))

    def _rebuilt(self, columns: Mapping[str, Any], template: EventTemplate) -> RecordBatch:
        """This batch over *columns* and *template*, at the same levels.

        Two rules the record transforms apply, both here. The **class** follows the
        edited fields rather than the object's history: a batch whose fields are all
        numeric is a ``NumericRecordBatch``, so an edit that removes the last
        non-numeric field promotes and one that introduces a non-numeric field
        demotes — which also makes a mixed ``merge`` give the same answer whichever
        way round it is written. The **name** is preserved when the caller gave it
        and left to the constructor to re-derive when it was auto, since the old one
        described the pre-edit fields.
        """
        # Lazy: the numeric module builds on this one, so the edge points that way.
        from ._numeric_record_batch import NumericRecordBatch

        cls = NumericRecordBatch if isinstance(template, NumericEventTemplate) else RecordBatch
        return cls(
            dict(columns),
            self.level_names,
            element_spec=template,
            axis_groups=self.axis_groups,
            name=None if self.name_is_auto else self.name,
        )

    # -- construction from elements -----------------------------------------

    @classmethod
    def stack(
        cls,
        records: list[Record],
        *,
        level_name: str,
        element_spec: RecordSpec | EventTemplate | None = None,
    ) -> Self:
        """Stack records into a batch with one level of ``(len(records),)``.

        Parameters
        ----------
        records : list of Record
            The elements, all conforming to one template. At least one.
        level_name : str
            Names the level the stacking mints. Required, since the operation
            doing the stacking is what knows what the axis means.
        element_spec : RecordSpec or EventTemplate, optional
            What every element satisfies. Taken from the first record when
            omitted, which is exact whenever the records were built against a
            shared declaration.

        Returns
        -------
        RecordBatch
            The batch, with ``batch_shape == (len(records),)``.

        Raises
        ------
        ValueError
            If *records* is empty, or a record's fields are not exactly the
            spec's, naming the record's position and what differs.

        Notes
        -----
        Nested templates stack like any other: a column is keyed by leaf path, so
        a nested field is one column and nesting costs the stacking nothing.
        """
        if not records:
            raise ValueError(f"{cls.__name__}.stack needs at least one record")
        kind = f"{cls.__name__}.stack"
        spec = _record_element_spec(
            element_spec if element_spec is not None else records[0].event_template, kind=kind
        )
        template = spec.event_template
        fields = template.keys()
        for position, record in enumerate(records):
            # Checked rather than left to a KeyError from the column loop: a
            # record with *extra* fields raises nothing at all there, and the
            # batch's spec would silently become a false statement about it.
            if tuple(record.keys()) != tuple(fields):
                missing = [key for key in fields if key not in record]
                extra = [key for key in record if key not in fields]
                parts = []
                if missing:
                    parts.append(f"missing {missing}")
                if extra:
                    parts.append(f"unexpected {extra}")
                if not parts:
                    parts.append(f"ordered {list(record.keys())}")
                raise ValueError(
                    f"{kind}: the record at {position} must have exactly the fields "
                    f"{list(fields)} — {'; '.join(parts)}"
                )
        columns = {
            key: _stack_column([record[key] for record in records], template[key], kind=kind)
            for key in fields
        }
        return cls(columns, (level_name,), element_spec=spec)

    # -- equality -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Whether *other* is the same class, equally specified, with equal columns.

        Equality is structural over the stored columns, compared field by field.
        A batch is unhashable for the reason an array-carrying container usually
        is — see :attr:`__hash__`.
        """
        # Identity first, so reflexivity holds for a batch carrying NaN, which
        # elementwise comparison would deny.
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        if self._spec != other._spec:
            return False
        if self._columns.keys() != other._columns.keys():
            # Equal specs with unequal stored columns means one of the two is
            # malformed. Comparing only the columns *self* has would call it
            # equal, and the mirror comparison would raise.
            return False
        for path, column in self._columns.items():
            if not _columns_equal(column, other._columns[path]):
                return False
        return True

    #: A batch is unhashable: ``__eq__`` compares columns elementwise, and a
    #: value-based hash would have to materialise every byte — prohibitive for a
    #: posterior batch, and impossible inside a JAX trace, where a traced array
    #: is not hashable by content. This follows the numpy precedent rather than
    #: offering a hash that is silently O(n). For a structural key, use
    #: ``(type(batch), batch.spec)``.
    __hash__ = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _record_element_spec(decl: RecordSpec | EventTemplate, *, kind: str) -> RecordSpec:
    """*decl* as the ``RecordSpec`` a batch of records is specified over."""
    template = _record_declaration_template(decl)
    if not isinstance(template, EventTemplate):
        raise TypeError(
            f"{kind}: element_spec must be a RecordSpec or an EventTemplate, "
            f"got {type(decl).__name__}"
        )
    return _to_record_declaration(decl)


def _leaf_keyed_columns(
    fields: Mapping[str, Any], template: EventTemplate, *, kind: str
) -> dict[str, Any]:
    """*fields* as a flat leaf-path dict in the template's canonical order.

    A nested mapping is flattened, so the two ways of writing the same columns
    agree. The keys must be exactly the template's fields: a batch asserts its
    element spec of every element, which a missing or unknown column would make
    a false statement.
    """
    if not isinstance(fields, Mapping):
        raise TypeError(
            f"{kind} stores one array per field, keyed by leaf path, so fields must be a "
            f"mapping; got {type(fields).__name__}"
        )
    if not fields:
        raise ValueError(f"{kind} requires at least one field")
    nested = _unflatten_paths(dict(fields))
    flat: dict[str, Any] = {}

    def _walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, Mapping):
                _walk(value, f"{path}{_PATH_SEP}")
            else:
                flat[path] = value

    _walk(nested, "")
    names = template.keys()
    missing = [key for key in names if key not in flat]
    unknown = [key for key in flat if key not in names]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unexpected {unknown}")
        raise ValueError(
            f"{kind}: the given fields must be exactly those of element_spec "
            f"{list(names)} — {'; '.join(parts)}"
        )
    return {key: flat[key] for key in names}


def _element_template_for(
    columns: Mapping[str, Any], batch: RecordBatch, *, edited: set[str]
) -> EventTemplate:
    """The element structure *columns* describe, given *batch*'s axes.

    An untouched field keeps the spec it already carried; an *edited* one takes
    the spec its new values imply, read by removing the batch axes from the
    front of their shape. This is the record transforms' own policy — an
    untouched child is identity-preserved, a replaced one is re-inferred — with
    the batch axes discounted first.
    """
    rank = len(batch.batch_shape)
    kind = f"{type(batch).__name__} transform"
    specs: dict[str, Any] = {}
    for path, column in columns.items():
        if path not in edited:
            specs[path] = batch.event_template[path]
            continue
        shape = _column_shape(column)
        if shape is None:
            specs[path] = None  # an opaque value: no shape to describe
            continue
        if len(shape) < rank or tuple(shape[:rank]) != batch.batch_shape:
            raise ValueError(
                f"{kind}: the values given for {path!r} have shape {tuple(shape)}, which does "
                f"not carry this batch's axes {batch.batch_shape}; a field's values span the "
                f"batch, so a replacement does too"
            )
        specs[path] = _inferred_field_spec(column, tuple(shape[rank:]))
    return EventTemplate(specs)


def _unwrapped_field(value: Any) -> Any:
    """*value* as a field's stored values, unwrapping a batch of them.

    Reading a field that is not an array gives the matching object batch, so that
    is what a caller has to hand when they want to put one back. Unwrapping it
    here keeps the two directions symmetric: what ``[]`` yields is accepted
    wherever a field's values are taken.
    """
    if isinstance(value, FunctionBatch | OpaqueBatch):
        return value._store
    return value


def _inferred_field_spec(column: Any, event_shape: tuple[int, ...]) -> Any:
    """The spec an edited field's values imply, in the template's own terms.

    A numeric column is an array field of *event_shape*. Anything else holds one
    value per element, so the values decide: all callable makes it a function
    field, and otherwise it is opaque. This is what template inference concludes
    for a single value, applied across the column — a unicode array is not
    numeric, and a column of callables does not become opaque.
    """
    dtype = getattr(column, "dtype", None)
    if dtype is not None and _is_numeric_dtype(dtype):
        return event_shape
    entries = list(np.asarray(column, dtype=object).flat) if _is_object_array(column) else []
    if entries and all(callable(entry) for entry in entries):
        return FunctionSpec()
    return None


def _event_shape(spec: ValueSpec, *, path: str, kind: str) -> tuple[int, ...]:
    """The event shape a column carries beyond the batch axes.

    An ``ArraySpec`` declares one; every other leaf kind exposes no shape, so its
    column is one entry per element and the batch axes are all of it.

    Raises
    ------
    ValueError
        If the field's shape carries a symbolic dimension. Splitting a column into
        batch and event axes needs sizes, and an unbound name is not one, so it is
        refused where it can be named rather than compared against an integer
        further on.
    """
    if not isinstance(spec, ArraySpec):
        return ()
    shape = tuple(spec.shape)
    symbolic = [entry for entry in shape if not isinstance(entry, int)]
    if symbolic:
        raise ValueError(
            f"{kind}: the field {path!r} declares the symbolic dimension "
            f"{symbolic[0]!r}, so its event shape has no size to split a column by; bind the "
            f"template's dimensions before batching against it"
        )
    return shape


def _column_shape(column: Any) -> tuple[int, ...] | None:
    """*column*'s shape, or ``None`` for something that reports none."""
    shape = getattr(column, "shape", None)
    return None if shape is None else tuple(shape)


def _batch_shape_of(
    store: dict[str, Any], template: EventTemplate, *, kind: str
) -> tuple[int, ...]:
    """The batch axes every column agrees on, checking each against the first.

    A column is ``(*batch_shape, *event_shape)``, so its batch axes are whatever
    its shape carries beyond the event shape its spec declares. Every column is
    read, not just the first: a shape is what a batch axis is derived *from*, so
    a column that reports none, or one whose batch axes disagree, is refused here
    rather than passed on to fail somewhere that cannot say what went wrong. A
    disagreement names both fields, since which of the two is wrong is the
    caller's to know.

    Raises
    ------
    TypeError
        If a column reports no shape.
    ValueError
        If a column is too short to carry its event shape, if its event shape
        accounts for the whole column so no batch axis is left, or if two
        columns disagree on the batch axes.
    """
    batch_shape: tuple[int, ...] | None = None
    first: str = ""
    for path, column in store.items():
        shape = _column_shape(column)
        if shape is None:
            raise TypeError(
                f"{kind}: the column at {path!r} is a {type(column).__name__}, which reports no "
                f"shape, so its batch axes cannot be read; a column is an array, or an object "
                f"array for a field that is not an array"
            )
        event_shape = _event_shape(template[path], path=path, kind=kind)
        if len(event_shape) > len(shape):
            raise ValueError(
                f"{kind}: the column at {path!r} has shape {shape}, which is too short to carry "
                f"its event shape {event_shape} after any batch axis"
            )
        split = len(shape) - len(event_shape)
        if tuple(shape[split:]) != tuple(event_shape):
            raise ValueError(
                f"{kind}: the column at {path!r} has shape {shape}, whose trailing axes "
                f"{tuple(shape[split:])} are not the event shape {event_shape} its field "
                f"declares; a column is (*batch_shape, *event_shape)"
            )
        candidate = shape[:split] if event_shape else shape
        if not candidate:
            raise ValueError(
                f"{kind}: a batch has at least one batch axis, but the column at {path!r} has "
                f"shape {shape}, which its event shape {event_shape} accounts for entirely"
            )
        if batch_shape is None:
            batch_shape, first = candidate, path
        elif candidate != batch_shape:
            raise ValueError(
                f"{kind}: the columns disagree on the batch axes — {first!r} carries "
                f"{batch_shape} and {path!r} carries {candidate}. Every column holds one value "
                f"per element, so all of them span the same batch axes"
            )
    if batch_shape is None:
        raise AssertionError("unreachable: an empty store is refused before this")
    return batch_shape


def _stack_column(values: list[Any], spec: ValueSpec, *, kind: str) -> Any:
    """One field's values across the elements, stacked into a column.

    The field's *spec* decides the form, not the values: an ``ArraySpec`` field
    batches natively into an array with the batch axis leading, and every other
    leaf kind goes into an object array, one entry per element, which is the form
    the object batches present. Reading the spec rather than the values is what
    keeps an opaque field opaque when its values happen to be numeric — a case
    ``OpaqueSpec`` admits — so the column comes back as the batch form its spec
    calls for and an element comes back as the object that was put in.
    """
    if isinstance(spec, ArraySpec):
        # Through ``_to_jax_array``, the one conversion every compute boundary
        # routes through, so a leaf whose type is registered with an
        # ``ArrayBackend`` converts by its backend's rule rather than by whatever
        # the numpy protocol happens to make of it.
        return jnp.stack([_to_jax_array(value) for value in values])
    store = _from_iterable(values, kind=kind)
    store.setflags(write=False)
    return store


def _columns_equal(left: Any, right: Any) -> bool:
    """Whether two columns hold equal values, elementwise.

    An array column compares with ``jnp.array_equal``. An **object** column
    cannot: its entries may be arrays, so a vectorized ``==`` yields an array of
    arrays that reduces to no single truth value. Those compare entry by entry
    instead, each entry by value where it has one and by identity where it does
    not, which is what lets a batch of arbitrary objects be compared at all.

    Notes
    -----
    NaN is not equal to itself here, as it is not for ``numpy``: two
    independently built batches carrying NaN in the same place compare unequal.
    ``RecordBatch.__eq__`` short-circuits on identity, so a batch still equals
    itself.
    """
    if left is right:
        return True
    if _is_object_array(left) or _is_object_array(right):
        if not (_is_object_array(left) and _is_object_array(right)):
            return False
        if left.shape != right.shape:
            return False
        return all(_one_entry_equal(a, b) for a, b in zip(left.flat, right.flat, strict=True))
    return bool(jnp.array_equal(left, right))


def _one_entry_equal(left: Any, right: Any) -> bool:
    """Whether two entries of an object column are equal, by value where they have one."""
    if left is right:
        return True
    try:
        return bool(np.array_equal(left, right))
    except (TypeError, ValueError):
        return False


def _check_array_column(column: Any, spec: ArraySpec, *, path: str, kind: str) -> None:
    """Fail unless *column* is a numeric array whose dtype *spec* admits.

    The dtype rule is :meth:`~probpipe.ArraySpec.is_valid`'s, applied to the
    column rather than to one value: same-kind castable, so a widening or a
    within-kind narrowing passes and a cross-kind conversion does not. The shape
    is already settled by the batch-axis derivation.
    """
    dtype = getattr(column, "dtype", None)
    if dtype is None or not _is_numeric_dtype(dtype):
        raise TypeError(
            f"{kind}: the field {path!r} is declared an array, so its column is a numeric "
            f"array; got a {type(column).__name__}" + (f" of {dtype}" if dtype is not None else "")
        )
    if spec.dtype is not None and not np.can_cast(dtype, spec.dtype, casting="same_kind"):
        raise TypeError(
            f"{kind}: the column at {path!r} has dtype {dtype}, which its declared "
            f"{np.dtype(spec.dtype)} does not admit; a widening or a within-kind narrowing "
            f"passes, a cross-kind conversion does not"
        )


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _record_batch_flatten(batch: RecordBatch) -> tuple[list, tuple[BatchSpec, str, bool]]:
    """Flatten for JAX pytree traversal: the columns, keyed by the aux spec.

    Children are the columns in the template's canonical order, so they realign
    with the aux spec on unflatten. The static aux is the
    ``(spec, name, name_is_auto)`` triple, matching ``Record``: the batch's own
    type and its name survive a round-trip, while provenance does not cross a
    JAX transform boundary.
    """
    # ``_columns`` is already in the template's canonical order at every
    # construction site, so the order the aux spec expects needs no second walk —
    # this runs at every jit / vmap / grad boundary and every ``tree_map``.
    return list(batch._columns.values()), (batch._spec, batch._name, batch._name_is_auto)


def _unflatten_with(cls: type[RecordBatch]):
    """Build the unflatten hook for *cls*, rebuilding at the multiplicity that arrived.

    A transform may hand back children with fewer axes than the batch was
    flattened with: ``vmap`` strips the mapped axis, so what its body receives is
    one *element*, not the batch. The stored spec describes the batch, so
    rebuilding against it verbatim would produce an object whose ``batch_shape``
    its own columns contradict — and every method that reads the shape,
    ``to_vector`` among them, would then be wrong.

    So the multiplicity is re-derived from the children. Axes are consumed from
    the outermost level in, since that is the end a transform strips; when none
    remain the value *is* one element, and a ``Record`` is returned rather than a
    batch of nothing. Validation is skipped either way: a traced column's dtype
    and shape are the transform's business, and the batch was validated when
    first built.
    """

    def _unflatten(aux: tuple[BatchSpec, str, bool], children: list) -> RecordBatch | Record:
        spec, name, name_is_auto = aux
        element_spec = spec.element_spec
        assert isinstance(element_spec, RecordSpec)
        template = element_spec.event_template
        # ``strict``: a child count that disagrees with the spec's fields would
        # otherwise truncate the columns silently, leaving a value whose own spec
        # is a false statement about it.
        columns = dict(zip(template.keys(), children, strict=True))

        rank = _surviving_batch_rank(columns, template, spec)
        if rank is not None:
            _refuse_resized_event_axes(columns, template, rank)
        if rank is None:
            # The children's shapes are unreadable, so take the spec as given —
            # the round trip out of an untransformed batch, where it is right.
            batch = object.__new__(cls)
            object.__setattr__(batch, "_columns", columns)
            batch._init_batch(spec, name=name, name_is_auto=name_is_auto)
            return batch
        if rank > len(spec.batch_shape):
            raise ValueError(
                f"a transform left this {cls.__name__}'s fields with {rank} batch axes where "
                f"its levels account for {len(spec.batch_shape)}. An added axis belongs to no "
                f"level, and unflattening has no name to give one — the operation that adds an "
                f"axis is what names its level. Build the batch where the axis is added, or map "
                f"over its columns rather than over the batch"
            )
        if rank == 0:
            return Record(
                name,
                columns,
                event_template=element_spec,
                name_is_auto=name_is_auto,
                _validate_leaves=False,
            )
        if rank == len(spec.batch_shape):
            surviving = _surviving_batch_shape(columns, rank)
            if surviving == spec.batch_shape:
                # Reuse the stored spec where the multiplicity is unchanged, which
                # is the ordinary round trip: rebuilding an equal one would
                # allocate at every transform boundary and cost the identity a
                # caller relies on.
                rebuilt = spec
            elif sorted(surviving) == sorted(spec.batch_shape):
                # The sizes moved rather than changed: a transpose reads like a
                # per-axis resize by shape alone, and retiling would keep every
                # level name on an axis that now holds another level's positions.
                raise ValueError(
                    f"a transform left this batch's axes at {surviving} where its levels "
                    f"{spec.level_names} held {spec.batch_shape} — a permutation of the same "
                    f"sizes, which shape alone cannot tell from a per-axis resize. Rebuild "
                    f"the result naming its levels"
                )
            else:
                # Same levels, new sizes: a transform that keeps every axis but
                # resizes some — a per-column slice, say — changes what each
                # level holds, not which levels there are, so the names carry
                # over onto the sizes the columns actually have.
                groups: list[tuple[int, ...]] = []
                consumed = 0
                for group in spec.axis_groups:
                    groups.append(tuple(surviving[consumed : consumed + len(group)]))
                    consumed += len(group)
                rebuilt = replace(spec, axis_groups=tuple(groups))
        else:
            surviving = _surviving_batch_shape(columns, rank)
            levels = _levels_for_shape(spec, surviving)
            if levels is None:
                raise ValueError(
                    f"a transform left this {cls.__name__}'s fields with batch axes "
                    f"{surviving}, which no selection of {spec.batch_shape} explains, so its "
                    f"levels cannot be named"
                )
            rebuilt = replace(spec, **levels)
        batch = object.__new__(cls)
        object.__setattr__(batch, "_columns", columns)
        batch._init_batch(rebuilt, name=name, name_is_auto=name_is_auto)
        return batch

    return _unflatten


def _refuse_resized_event_axes(columns: dict[str, Any], template: EventTemplate, rank: int) -> None:
    """Raise if a column's trailing axes differ from its field's event shape.

    The batch axes are a transform's to drop or resize; the element's own axes
    are the element type's, and a column whose event part changed would leave the
    rebuilt batch declaring an element it does not hold — a per-column slice
    passing the rank check while shrinking the event, or a transpose reading an
    event axis as a batch axis. Only fields declaring a concrete shape can be
    checked; the rest are declared by rank alone, which the rank derivation
    already enforced.
    """
    for path, column in columns.items():
        shape = _column_shape(column)
        if shape is None:
            continue
        field = template[path]
        if not isinstance(field, ArraySpec) or not all(isinstance(s, int) for s in field.shape):
            continue
        if tuple(shape[rank:]) != tuple(field.shape):
            raise ValueError(
                f"a transform left the column {path!r} with event axes {tuple(shape[rank:])} "
                f"where its field declares {tuple(field.shape)}; a transform may drop or "
                f"resize batch axes, never the element's own. Build a new batch under the "
                f"element type it now holds"
            )
        # A pinned dtype is the element's own as much as its axes are, held to
        # the constructor's rule: same-kind casting admits a widening, a change
        # of kind does not.
        dtype = getattr(column, "dtype", None)
        if (
            field.dtype is not None
            and dtype is not None
            and not np.can_cast(dtype, field.dtype, casting="same_kind")
        ):
            raise ValueError(
                f"a transform left the column {path!r} with dtype {dtype} where its field "
                f"declares {field.dtype}; a transform may drop or resize batch axes, never "
                f"retype the element. Build a new batch under the element type it now holds"
            )


def _surviving_batch_rank(
    columns: dict[str, Any], template: EventTemplate, spec: BatchSpec
) -> int | None:
    """How many batch axes the *columns* still carry, or ``None`` when unreadable.

    Each column is ``(*batch_axes, *event_shape)``, so what its rank has beyond
    its field's event rank is the batch part. A column reporting no shape, or
    columns that disagree, leaves the question unanswerable.
    """
    ranks = set()
    for path, column in columns.items():
        shape = _column_shape(column)
        if shape is None:
            return None
        field = template[path]
        event_rank = len(field.shape) if isinstance(field, ArraySpec) else 0
        if len(shape) < event_rank:
            raise ValueError(
                f"a transform left the column {path!r} with shape {tuple(shape)}, fewer axes "
                f"than the {event_rank} its field's event shape declares; a transform may "
                f"drop or resize batch axes, never the element's own"
            )
        ranks.add(len(shape) - event_rank)
    if len(ranks) != 1:
        # Every column was readable and they disagree: whatever shape the batch
        # stated, it would be false for some column, so there is no honest spec
        # to rebuild under. Distinct from the unreadable case above, where the
        # spec is taken as given because nothing contradicts it.
        raise ValueError(
            f"a transform left this batch's fields with disagreeing batch axes "
            f"({sorted(ranks)} axes beyond their event shapes); a batch states one "
            f"multiplicity for all its fields, so there is none to rebuild under"
        )
    rank = ranks.pop()
    # A rank *above* the spec's is returned rather than reported unreadable: the
    # caller refuses it, and conflating the two would treat an added axis as a
    # column whose shape could not be read and take the spec as given.
    return rank


def _surviving_batch_shape(columns: dict[str, Any], rank: int) -> tuple[int, ...]:
    """The batch axes the *columns* now carry, checked for agreement.

    Rank agreement alone is not enough: a leaf-dependent transform can leave two
    columns the same number of batch axes at different sizes, and a batch states
    one multiplicity for all its fields.
    """
    shapes = {
        tuple(shape[:rank]) for shape in map(_column_shape, columns.values()) if shape is not None
    }
    if len(shapes) > 1:
        raise ValueError(
            f"a transform left this batch's fields with disagreeing batch axes "
            f"{sorted(shapes)}; a batch states one multiplicity for all its fields, "
            f"so there is none to rebuild under"
        )
    return next(iter(shapes), ())


def _levels_for_shape(spec: BatchSpec, surviving: tuple[int, ...]) -> dict[str, tuple] | None:
    """The level tiling for *surviving*, given the axes *spec* started with.

    A transform removes whole axes, so *which* of the original axes are left is
    what names the levels — and that cannot be read off the count alone. ``vmap``
    maps the leading axis by default but takes ``in_axes`` for any other, so the
    removed axes are found by matching *surviving* against every candidate
    selection. A level whose axes are all gone goes with them, and one partly
    consumed keeps what is left.

    Candidates that disagree are refused rather than chosen among: two levels of
    equal size explain the same removal, and shape alone cannot say which axis
    the transform actually took — picking the leftmost would hand the body a
    batch named for the level that was removed.

    Returns ``None`` when no selection of the original axes yields *surviving*:
    the transform did not simply drop axes, and the levels cannot be named
    honestly.
    """
    original = spec.batch_shape
    dropped_count = len(original) - len(surviving)
    if dropped_count < 0:
        return None
    candidates: list[dict[str, tuple]] = []
    for dropped in combinations(range(len(original)), dropped_count):
        if tuple(size for i, size in enumerate(original) if i not in dropped) != surviving:
            continue
        groups: list[tuple[int, ...]] = []
        names: list[str] = []
        axis = 0
        for level_name, group in zip(spec.level_names, spec.axis_groups, strict=True):
            kept = tuple(size for offset, size in enumerate(group) if axis + offset not in dropped)
            axis += len(group)
            if kept:
                groups.append(kept)
                names.append(level_name)
        candidate = {"axis_groups": tuple(groups), "level_names": tuple(names)}
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            f"a transform removed axes from batch shape {original} whose sizes match more "
            f"than one of the levels {spec.level_names}, so shape alone cannot say which "
            f"level survived: the axes left could be "
            f"{' or '.join(str(c['level_names']) for c in candidates)}. Map over an axis "
            f"whose size is unique to its level, or rebuild the result naming its levels"
        )
    return candidates[0]


jax.tree_util.register_pytree_node(RecordBatch, _record_batch_flatten, _unflatten_with(RecordBatch))
