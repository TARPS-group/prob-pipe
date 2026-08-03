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

See design III.3.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array
from ._array_backend import _is_numeric_dtype
from ._batch import Batch, BatchSpec, _axis_groups_for
from ._function_batch import FunctionBatch
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

__all__ = ["NumericRecordBatch", "RecordBatch"]


class RecordBatch(Batch[Record]):
    """A batch of records sharing one ``EventTemplate``, stored as columns.

    Parameters
    ----------
    columns : Mapping
        The field columns, keyed by **leaf path** (``"outer/a"``) or given as a
        nested mapping, which is flattened to leaf paths. Each column holds one
        field's values across the batch, shaped ``(*batch_shape, *event_shape)``
        where the event shape is the field spec's for an ``ArraySpec`` and empty
        otherwise. The keys must be exactly the fields of *element_spec*.
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
        If *element_spec* is not a ``RecordSpec`` or ``EventTemplate``, if
        *columns* is not a mapping, or if a column is missing the ``shape`` a
        batch axis is read from.
    ValueError
        If *columns* is empty, if its keys are not exactly the fields of
        *element_spec*, if the columns disagree on ``batch_shape``, if a column
        is not shaped ``(*batch_shape, *event_shape)``, if there are no batch
        axes, or if *axis_groups* does not tile ``batch_shape``.

    Notes
    -----
    Construction requires at least one element, while *selecting* none is
    allowed: ``batch[0:0]`` is a batch of nothing, as the level algebra intends.
    The asymmetry is the one the object batches state — an empty literal is
    almost always a mistake, and a shape cannot be inferred from it.

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

    #: What this batch's columns must be, phrased for the refusal a bad one earns.
    _column_rule = "conform to its field spec"

    def __init__(
        self,
        columns: Mapping[str, Any],
        level_names: str | Iterable[str],
        *,
        element_spec: RecordSpec | EventTemplate,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
        _validate: bool = True,
    ) -> None:
        spec = _record_element_spec(element_spec, kind=type(self).__name__)
        template = spec.event_template
        store = _leaf_keyed_columns(columns, template, kind=type(self).__name__)
        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)

        batch_shape = _batch_shape_of(store, template, kind=type(self).__name__)
        groups = _axis_groups_for(batch_shape, names, axis_groups, kind=type(self).__name__)

        object.__setattr__(self, "_columns", store)
        if _validate:
            type(self)._check_columns(store, template, batch_shape, kind=type(self).__name__)
        self._init_batch(
            BatchSpec(spec, groups, names),
            name=name if name is not None else type(self).__name__.lower(),
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
    def _check_columns(
        cls,
        store: dict[str, Any],
        template: EventTemplate,
        batch_shape: tuple[int, ...],
        *,
        kind: str,
    ) -> None:
        """Fail on the first column that does not match its field spec.

        Checked at construction rather than left to ``is_valid`` because a batch
        asserts its ``element_spec`` of *every* element: a column that does not
        carry the declared event shape makes the batch's own spec a false
        statement, and which field it was is what a caller needs to hear.
        The base checks shape alone; ``NumericRecordBatch`` adds dtype.
        """
        for path, column in store.items():
            expected = (*batch_shape, *_event_shape(template[path]))
            actual = _column_shape(column)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"{kind}: the column at {path!r} has shape {actual}, expected {expected} "
                    f"(batch_shape={batch_shape}, "
                    f"event_shape={_event_shape(template[path])})"
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
            Record(name, row, event_template=self.element_spec, name_is_auto=True)
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
        """
        column = self._columns[key]
        spec = self.event_template[key]
        if isinstance(spec, ArraySpec):
            return column
        name = f"{self.name}[{key!r}]"
        shared = {
            "level_names": self.level_names,
            "axis_groups": self.axis_groups,
            "name": name,
            "name_is_auto": True,
        }
        if isinstance(spec, FunctionSpec):
            return self._inherit_provenance(FunctionBatch(column, element_spec=spec, **shared))
        if isinstance(spec, OpaqueSpec):
            return self._inherit_provenance(OpaqueBatch(column, element_spec=spec, **shared))
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

    def select(self, *fields: str, **mapping: str) -> dict[str, Self]:
        """Fields as single-field batch views, ready to splat into a call.

        The batch counterpart of :meth:`~probpipe.Record.select`: each entry is a
        one-column ``RecordBatch`` view rather than a bare column, so splatting
        the result into a ``Function`` call carries the level names an operation
        aligns operands by. Keywords remap, as on a record:
        ``select(x="r") == {"x": self["r"]}`` in structure, with the view in
        place of the column.

        Raises
        ------
        KeyError
            If a name is not a field of this batch.
        """
        selected = {field: field for field in fields}
        selected.update(mapping)
        for key in selected.values():
            if key not in self._columns:
                raise KeyError(
                    f"no field {key!r} in this {type(self).__name__}; "
                    f"its fields are {list(self.event_template.keys())}"
                )
        return {argument: self._single_field_view(key) for argument, key in selected.items()}

    def select_all(self) -> dict[str, Self]:
        """Every field as a single-field batch view, keyed by its own path."""
        return self.select(*self.event_template.keys())

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
            If *records* is empty, or a record's fields do not match the spec.

        Notes
        -----
        Nested templates stack like any other: a column is keyed by leaf path, so
        a nested field is one column and nesting costs the stacking nothing.
        """
        if not records:
            raise ValueError(f"{cls.__name__}.stack needs at least one record")
        spec = _record_element_spec(
            element_spec if element_spec is not None else records[0].event_template,
            kind=f"{cls.__name__}.stack",
        )
        columns: dict[str, Any] = {}
        for key in spec.event_template:
            values = [record[key] for record in records]
            columns[key] = _stack_column(values)
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


class NumericRecordBatch(RecordBatch):
    """A :class:`RecordBatch` whose every column is a numeric array.

    The all-numeric specialization, carrying a ``NumericEventTemplate``: a bare
    pytree of arrays whose leading axes are the ``batch_shape``, so it passes
    through ``jit`` / ``vmap`` / ``grad`` unchanged. It adds the batched flat
    layout, :meth:`to_vector` and :meth:`from_vector`.

    Construction is that of :class:`RecordBatch`, narrowed: *element_spec* must
    describe an all-numeric element, and every column must carry a numeric dtype.

    Raises
    ------
    TypeError
        If *element_spec* does not describe an all-numeric element, or a column
        is not a numeric array.
    """

    __slots__ = ()

    @property
    def element_spec(self) -> RecordSpec:
        """The :class:`RecordSpec` every element satisfies, over a numeric template."""
        spec = self._spec.element_spec
        assert isinstance(spec, RecordSpec)  # narrowed at construction
        return spec

    @property
    def event_template(self) -> NumericEventTemplate:
        """The numeric structure of one element — a view on :attr:`element_spec`."""
        template = self.element_spec.event_template
        assert isinstance(template, NumericEventTemplate)  # narrowed at construction
        return template

    def __init__(
        self,
        columns: Mapping[str, Any],
        level_names: str | Iterable[str],
        *,
        element_spec: RecordSpec | EventTemplate,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
        _validate: bool = True,
    ) -> None:
        template = _record_element_spec(element_spec, kind=type(self).__name__).event_template
        if not isinstance(template, NumericEventTemplate):
            raise TypeError(
                f"{type(self).__name__} describes an all-numeric element, so its element_spec "
                f"carries a NumericEventTemplate; got one over {type(template).__name__} with "
                f"fields {list(template.keys())}"
            )
        super().__init__(
            columns,
            level_names,
            element_spec=element_spec,
            axis_groups=axis_groups,
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
            _validate=_validate,
        )

    @classmethod
    def _check_columns(
        cls,
        store: dict[str, Any],
        template: EventTemplate,
        batch_shape: tuple[int, ...],
        *,
        kind: str,
    ) -> None:
        """Require a numeric dtype on every column, then check shapes as the base does."""
        for path, column in store.items():
            if not hasattr(column, "dtype") or not hasattr(column, "shape"):
                raise TypeError(
                    f"{kind}: the column at {path!r} must be a numeric array, "
                    f"got {type(column).__name__}"
                )
            if not _is_numeric_dtype(column.dtype):
                raise TypeError(
                    f"{kind}: the column at {path!r} has the non-numeric dtype {column.dtype!r}"
                )
        super()._check_columns(store, template, batch_shape, kind=kind)

    # -- flat layout --------------------------------------------------------

    def to_vector(self) -> Array:
        """Every element's flat vector, stacked.

        Returns
        -------
        Array
            Shape ``(*batch_shape, vector_size)``: one raveled vector per
            element, fields visited in the template's canonical order, each
            field's event axes raveled and the fields concatenated. The inverse
            is :meth:`from_vector`.

        Notes
        -----
        Distinct from reading the columns, which keeps each field whole and its
        event axes intact. Because ``batch_shape`` is the flat concatenation of
        the axis levels, a multi-level batch vectorizes exactly as a single-level
        one does, its levels flattened outermost-first.
        """
        batch_shape = self.batch_shape
        return jnp.concatenate(
            [
                jnp.reshape(jnp.asarray(self._columns[key]), (*batch_shape, -1))
                for key in self.event_template
            ],
            axis=-1,
        )

    @classmethod
    def from_vector(
        cls,
        name: str,
        template: NumericEventTemplate,
        vec: Array,
        *,
        level_name: str,
    ) -> Self:
        """Rebuild a batch from its elements' flat vectors.

        Parameters
        ----------
        name : str
            The reconstructed batch's name (user-given).
        template : NumericEventTemplate
            The flat layout: field names, event shapes, and canonical order.
        vec : Array
            Shape ``(*batch_shape, vector_size)`` — the trailing axis is the flat
            dimension, and the leading axes are the batch.
        level_name : str
            Names the level the reconstruction mints. Required for the reason
            :meth:`RecordBatch.stack` states.

        Returns
        -------
        NumericRecordBatch
            The batch, satisfying ``batch.to_vector() == vec``.

        Raises
        ------
        TypeError
            If *vec* has no batch axis — reconstruct a single value with
            ``NumericRecord.from_vector``.
        ValueError
            If the trailing axis is not ``template.vector_size``.
        """
        vec = jnp.asarray(vec)
        if vec.ndim < 2:
            raise TypeError(
                f"{cls.__name__}.from_vector takes a batched matrix, shaped "
                f"(*batch_shape, vector_size); got shape {tuple(vec.shape)}. Reconstruct a "
                f"single value with NumericRecord.from_vector"
            )
        if vec.shape[-1] != template.vector_size:
            raise ValueError(
                f"{cls.__name__}.from_vector: the trailing axis is {vec.shape[-1]}, expected "
                f"{template.vector_size} for this template"
            )
        batch_shape = tuple(vec.shape[:-1])
        columns: dict[str, Any] = {}
        offset = 0
        for key, event_shape in template.leaf_shapes.items():
            size = int(np.prod(event_shape, dtype=int))
            block = vec[..., offset : offset + size]
            columns[key] = jnp.reshape(block, (*batch_shape, *event_shape))
            offset += size
        return cls(columns, (level_name,), element_spec=template, name=name)


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
    columns: Mapping[str, Any], template: EventTemplate, *, kind: str
) -> dict[str, Any]:
    """*columns* as a flat leaf-path dict in the template's canonical order.

    A nested mapping is flattened, so the two ways of writing the same columns
    agree. The keys must be exactly the template's fields: a batch asserts its
    element spec of every element, which a missing or unknown column would make
    a false statement.
    """
    if not isinstance(columns, Mapping):
        raise TypeError(
            f"{kind} stores one column per field, keyed by leaf path, so columns must be a "
            f"mapping; got {type(columns).__name__}"
        )
    if not columns:
        raise ValueError(f"{kind} requires at least one column")
    flat = _flatten_columns(columns)
    fields = template.keys()
    missing = [key for key in fields if key not in flat]
    unknown = [key for key in flat if key not in fields]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unexpected {unknown}")
        raise ValueError(
            f"{kind}: the columns must be exactly the fields of element_spec "
            f"{list(fields)} — {'; '.join(parts)}"
        )
    return {key: flat[key] for key in fields}


def _flatten_columns(columns: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a possibly-nested column mapping to ``/``-joined leaf paths."""
    nested = _unflatten_paths(dict(columns))
    flat: dict[str, Any] = {}

    def _walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            if isinstance(value, Mapping):
                _walk(value, f"{path}{_PATH_SEP}")
            else:
                flat[path] = value

    _walk(nested, "")
    return flat


def _event_shape(spec: ValueSpec) -> tuple[int, ...]:
    """The event shape a column carries beyond the batch axes.

    An ``ArraySpec`` declares one; every other leaf kind exposes no shape, so its
    column is one entry per element and the batch axes are all of it.
    """
    return tuple(spec.shape) if isinstance(spec, ArraySpec) else ()


def _column_shape(column: Any) -> tuple[int, ...] | None:
    """*column*'s shape, or ``None`` for something that reports none."""
    shape = getattr(column, "shape", None)
    return None if shape is None else tuple(shape)


def _batch_shape_of(
    store: dict[str, Any], template: EventTemplate, *, kind: str
) -> tuple[int, ...]:
    """The batch axes the columns agree on, read off the first one.

    A column is ``(*batch_shape, *event_shape)``, so its batch axes are whatever
    its shape has beyond the event shape its spec declares. The first column
    fixes them and the rest are checked against it at validation, which is where
    a disagreement is reported against the field that disagreed.
    """
    for path, column in store.items():
        shape = _column_shape(column)
        if shape is None:
            raise TypeError(
                f"{kind}: the column at {path!r} reports no shape, so its batch axes cannot be "
                f"read; a column is an array, or an object array for a non-array field"
            )
        event_rank = len(_event_shape(template[path]))
        if event_rank > len(shape):
            raise ValueError(
                f"{kind}: the column at {path!r} has shape {shape}, which is too short to carry "
                f"its event shape {_event_shape(template[path])} after any batch axis"
            )
        batch_shape = shape[: len(shape) - event_rank] if event_rank else shape
        if not batch_shape:
            raise ValueError(
                f"{kind}: a batch has at least one batch axis, but the column at {path!r} has "
                f"shape {shape}, which its event shape "
                f"{_event_shape(template[path])} accounts for entirely"
            )
        return batch_shape
    raise AssertionError("unreachable: an empty store is refused before this")


def _stack_column(values: list[Any]) -> Any:
    """One field's values across the elements, stacked into a column.

    Numeric values stack natively into an array. Anything else has no stacked
    form, so it goes into an object array, one entry per element, which is what
    the object batches present a non-array column as.
    """
    if all(
        hasattr(value, "shape") or isinstance(value, (int, float, complex, bool))
        for value in values
    ):
        try:
            return jnp.stack([jnp.asarray(value) for value in values])
        except (TypeError, ValueError):
            pass
    store = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        store[position] = value
    store.setflags(write=False)
    return store


def _columns_equal(left: Any, right: Any) -> bool:
    """Whether two columns hold equal values, elementwise where they are arrays."""
    if left is right:
        return True
    try:
        return bool(jnp.array_equal(left, right))
    except Exception:
        left_array, right_array = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
        return left_array.shape == right_array.shape and bool((left_array == right_array).all())


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
    keys = batch.event_template.keys()
    return [batch._columns[key] for key in keys], (batch._spec, batch._name, batch._name_is_auto)


def _unflatten_with(cls: type[RecordBatch]):
    """Build the unflatten hook for *cls*, which rebuilds without validating.

    Inside a transform a column's shape is relative to it — ``vmap`` strips the
    mapped axis — so the stored ``axis_groups`` need not describe what arrives.
    The batch was validated when first built, so validation is skipped rather
    than made to pass.
    """

    def _unflatten(aux: tuple[BatchSpec, str, bool], children: list) -> RecordBatch:
        spec, name, name_is_auto = aux
        assert isinstance(spec.element_spec, RecordSpec)
        keys = spec.element_spec.event_template.keys()
        batch = object.__new__(cls)
        object.__setattr__(batch, "_columns", dict(zip(keys, children, strict=False)))
        batch._init_batch(spec, name=name, name_is_auto=name_is_auto)
        return batch

    return _unflatten


jax.tree_util.register_pytree_node(RecordBatch, _record_batch_flatten, _unflatten_with(RecordBatch))
jax.tree_util.register_pytree_node(
    NumericRecordBatch, _record_batch_flatten, _unflatten_with(NumericRecordBatch)
)
