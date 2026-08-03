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
from ._object_batch import _from_iterable
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
    columns : Mapping of str to array
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
        *columns* is not a mapping; if a column reports no ``shape``, which is
        what a batch axis is read from; or if an entry of a non-array column is
        not admitted by its field's spec.
    ValueError
        If *columns* is empty; if its keys are not exactly the fields of
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

    #: What every entry of a non-array column must be, phrased for the refusal a
    #: bad one earns. A subclass narrowing the admissible columns supplies its own.
    _entry_rule = "satisfy its field's specification"

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
    ) -> None:
        kind = type(self).__name__
        spec = _record_element_spec(element_spec, kind=kind)
        template = spec.event_template
        store = _leaf_keyed_columns(columns, template, kind=kind)
        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)

        batch_shape = _batch_shape_of(store, template, kind=kind)
        groups = _axis_groups_for(batch_shape, names, axis_groups, kind=kind)
        type(self)._check_columns(store, template, kind=kind)
        # An object column is frozen against later writes; see the helper.
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
        """Fail on the first column entry its field's spec does not admit.

        Checked at construction rather than left to ``is_valid`` because a batch
        asserts its ``element_spec`` of *every* element: an entry the field's spec
        refuses makes the batch's own spec a false statement, and where it sits is
        what a caller needs to hear. The shapes are already settled by then, by
        the batch-axis derivation.

        An **array** column carries no entries to walk — its spec's shape and the
        column's are the whole of what it declares, and the derivation has checked
        that. A **non-array** column is walked entry by entry, the counterpart of
        what ``_ObjectBatch`` does for a batch that stores its elements.
        ``NumericRecordBatch`` adds the dtype check its columns are narrowed by.

        Raises
        ------
        TypeError
            If an entry of a non-array column does not satisfy its field's spec,
            naming the field and the position.
        """
        for path, column in store.items():
            spec = template[path]
            if isinstance(spec, ArraySpec) or not _is_object_array(column):
                continue
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

    def select(self, *paths: str, **mapping: str) -> dict[str, Self]:
        """Batch views of the named parts, ready to splat into a call.

        The batch counterpart of :meth:`~probpipe.Record.select`, and it resolves
        a path the same way: a **key** reaching a field gives a one-column view,
        and a **partial path** gives the sub-batch under it. Each entry is a
        ``RecordBatch`` view rather than a bare column, so splatting the result
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

        The single-field case is not ``self[path]``, which yields the column: a
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
        )

    # ``element_spec`` is not overridden: it already reports the ``RecordSpec``
    # this class stores, and there is nothing narrower to say. Only
    # ``event_template`` narrows, to the numeric template construction required.

    @property
    def event_template(self) -> NumericEventTemplate:
        """The numeric structure of one element — a view on :attr:`element_spec`."""
        template = self.element_spec.event_template
        assert isinstance(template, NumericEventTemplate)  # narrowed at construction
        return template

    @classmethod
    def _check_columns(cls, store: dict[str, Any], template: EventTemplate, *, kind: str) -> None:
        """Require a numeric dtype on every column, then check entries as the base does."""
        for path, column in store.items():
            if not hasattr(column, "dtype"):
                raise TypeError(
                    f"{kind}: the column at {path!r} must be a numeric array, "
                    f"got {type(column).__name__}"
                )
            if not _is_numeric_dtype(column.dtype):
                raise TypeError(
                    f"{kind}: the column at {path!r} has the non-numeric dtype {column.dtype!r}"
                )
        super()._check_columns(store, template, kind=kind)

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
        event axes intact. The batch axes are left as they are: a multi-level
        batch keeps its levels as the leading axes of the result, so the flat
        dimension is always the last one and the levels read outermost-first.
        """
        batch_shape = self.batch_shape
        template = self.event_template
        leaf_shapes = template.leaf_shapes
        return jnp.concatenate(
            [
                # The field's own flat width rather than ``-1``: an empty batch
                # axis leaves nothing for ``-1`` to be inferred from.
                jnp.reshape(
                    jnp.asarray(self._columns[key]),
                    (*batch_shape, int(np.prod(leaf_shapes[key], dtype=int))),
                )
                for key in template
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
        level_names: str | Iterable[str],
        axis_groups: Iterable[Iterable[int]] | None = None,
    ) -> Self:
        """Rebuild a batch from its elements' flat vectors — the inverse of
        :meth:`to_vector`.

        Parameters
        ----------
        name : str
            The reconstructed batch's name (user-given).
        template : NumericEventTemplate
            The flat layout: field names, event shapes, and canonical order.
        vec : Array
            Shape ``(*batch_shape, vector_size)`` — the trailing axis is the flat
            dimension, and every leading axis is a batch axis.
        level_names : str or iterable of str
            One name per level of the reconstructed batch, outermost first; a
            single string names a single level. Required for the reason
            :meth:`RecordBatch.stack` states, and plural because *vec* may carry
            several batch axes: naming them is how a multi-level batch round-trips.
        axis_groups : iterable of iterable of int, optional
            The axis sizes each level holds, as for the constructor. Defaults to
            one axis per level.

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
            If the trailing axis is not ``template.vector_size``, or if the level
            names do not account for *vec*'s leading axes.

        Examples
        --------
        A two-level batch round-trips when both levels are named:

        >>> import jax.numpy as jnp
        >>> from probpipe import EventTemplate
        >>> template = EventTemplate(x=(2,))
        >>> batch = NumericRecordBatch({"x": jnp.zeros((4, 5, 2))}, ("chain", "draw"),
        ...                            element_spec=template)
        >>> rebuilt = NumericRecordBatch.from_vector(
        ...     "post", template, batch.to_vector(), level_names=("chain", "draw"))
        >>> rebuilt.batch_shape
        (4, 5)
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
        return cls(
            columns,
            level_names,
            element_spec=template,
            axis_groups=axis_groups,
            name=name,
        )


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
        return jnp.stack([jnp.asarray(value) for value in values])
    # The same object-array construction the object batches use, frozen for the
    # reason ``_frozen_object_column`` gives.
    store = _from_iterable(values, kind=kind)
    store.setflags(write=False)
    return store


def _frozen_object_column(column: np.ndarray) -> np.ndarray:
    """*column* as an object array nobody can write through.

    A batch holds the columns it validated. An object column is the one kind a
    caller can still mutate after construction — a JAX array is already immutable
    and a numpy numeric column follows the aliasing convention the single-record
    types already set — so it is copied and frozen, for the reason
    ``_ObjectBatch`` states: a caller keeping a handle on what they passed cannot
    write a value into the batch that its spec does not admit. Only the pointer
    array is copied, so the elements stay shared.
    """
    frozen = np.array(column, dtype=object, subok=False)
    frozen.setflags(write=False)
    return frozen


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
        return all(_entries_equal(a, b) for a, b in zip(left.flat, right.flat, strict=True))
    return bool(jnp.array_equal(left, right))


def _is_object_array(column: Any) -> bool:
    """Whether *column* is a numpy array of objects, the non-array column form."""
    return isinstance(column, np.ndarray) and column.dtype == object


def _entries_equal(left: Any, right: Any) -> bool:
    """Whether two object-column entries are equal, by value where they have one."""
    if left is right:
        return True
    try:
        return bool(np.array_equal(left, right))
    except (TypeError, ValueError):
        return False


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
        # ``strict``: a child count that disagrees with the spec's fields would
        # otherwise truncate the columns silently, leaving a batch whose own spec
        # is a false statement about it.
        object.__setattr__(batch, "_columns", dict(zip(keys, children, strict=True)))
        batch._init_batch(spec, name=name, name_is_auto=name_is_auto)
        return batch

    return _unflatten


jax.tree_util.register_pytree_node(RecordBatch, _record_batch_flatten, _unflatten_with(RecordBatch))
jax.tree_util.register_pytree_node(
    NumericRecordBatch, _record_batch_flatten, _unflatten_with(NumericRecordBatch)
)
