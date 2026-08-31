"""NumericArrayBatch — the batch form of the numeric-array kind.

See design III.1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

import jax
import numpy as np

from ._array_backend import (
    _event_shape_of,
    _is_numeric_leaf,
    _numpy_dtype_of,
    _take_at,
    _to_jax_array,
    _to_numpy_array,
)
from ._batch import Batch, BatchSpec, _axis_groups_for
from ._kinds import register_kind
from ._numeric_array import NumericArray
from .event_template import NumericArraySpec
from .provenance import Provenance

__all__ = ["NumericArrayBatch"]


class NumericArrayBatch(Batch[NumericArray]):
    """A batch of numeric arrays sharing one :class:`NumericArraySpec`.

    Storage is a single array with the batch axes leading — the split
    :class:`~probpipe.RecordBatch` uses, with one column instead of many. This
    is where a `draw` level lives for an array-valued law.

    Parameters
    ----------
    name : str
        The batch's name, **required**, as a :class:`~probpipe.Record`'s and an
        :class:`~probpipe.Opaque`'s are. A batch is what an operation hands back,
        and the name is what says which one it is; a class-name default would name
        every batch in a pipeline alike. A caller that derives one says so with
        *name_is_auto*.
    values : array-like
        One array holding every element, shaped ``(*batch_shape, *event_shape)``.
        Stored verbatim in its native form, as a :class:`NumericArray`'s value
        is, so a lazy or disk-backed column is not materialised to be batched.
    level_names : str or iterable of str
        One name per level, outermost first; a single string names one level.
    element_spec : NumericArraySpec
        What every element satisfies. Its ``shape`` is the event shape, so it is
        what splits the stored array's axes into batch and event.
    axes_per_level : iterable of int, optional
        How many axes each level holds, outermost first; they must account for
        every batch axis. Defaults to one axis per level, which requires as many
        names as there are batch axes. The *sizes* are read off the elements
        rather than restated here — they are already fixed by the data, so the
        only thing left to say is where one level ends and the next begins.
    name_is_auto : bool, default False
        Whether *name* is auto-derived rather than user-given — set by an
        operation that derives one, as a view does for a selected sub-batch.
    provenance : Provenance, optional
        How this batch was produced.

    Raises
    ------
    TypeError
        If *element_spec* is not a :class:`NumericArraySpec`, or *values* does
        not convert to an array.
    TypeError
        Also if the stored array's dtype is one the declared element does not
        admit — a cross-kind conversion — or if the store reports no single
        dtype while the element declares one, which leaves the claim
        unsubstantiated.
    ValueError
        If *values* has fewer axes than the event shape it must end with, which
        would leave no batch axis; if its trailing axes are not that event
        shape; if a declared dimension is symbolic, which gives the event shape
        no size to split by; or if *axes_per_level* does not account for every
        batch axis, or gives a count that is not one per level.

    Notes
    -----
    Selection yields the element kind, as it does for every batch:
    ``batch[i]`` is a :class:`NumericArray`. It materializes, so each element
    takes the derived name and inherits this batch's lineage.
    """

    _values: Any

    __slots__ = ("_jax_cache", "_values")

    #: Derived from the store rather than transported, as for ``NumericArray``.
    _transient_state = ("_jax_cache",)

    def __init__(
        self,
        name: str,
        values: Any,
        /,
        level_names: str | Iterable[str],
        *,
        element_spec: NumericArraySpec,
        axes_per_level: Iterable[int] | None = None,
        name_is_auto: bool = False,
        provenance: Provenance | None = None,
    ) -> None:
        if not isinstance(element_spec, NumericArraySpec):
            raise TypeError(
                f"NumericArrayBatch element_spec must be a NumericArraySpec, "
                f"got {type(element_spec).__name__}"
            )
        event_shape = tuple(element_spec.shape)
        if any(not isinstance(axis, int) for axis in event_shape):
            raise ValueError(
                f"a symbolic dimension gives the event shape no size to split the stored "
                f"axes by; bind {element_spec.shape} with with_dims before batching"
            )
        if not _is_numeric_leaf(values):
            raise TypeError(
                f"NumericArrayBatch stores one array; {type(values).__name__} is not a numeric leaf"
            )
        stored = _event_shape_of(values)
        n_event = len(event_shape)
        if n_event and stored[len(stored) - n_event :] != event_shape:
            raise ValueError(
                f"the stored array ends with {stored[len(stored) - n_event :]} where its "
                f"elements declare the event shape {event_shape}"
            )
        # The batch asserts *element_spec* of every element, so the store's own
        # dtype has to satisfy it.
        dtype = _numpy_dtype_of(values)
        if element_spec.dtype is not None:
            if dtype is None:
                # A heterogeneous store has no single dtype to check a pinned
                # one against, so the claim is unsupportable either way.
                raise TypeError(
                    f"the stored array reports no single dtype, so the declared element "
                    f"{np.dtype(element_spec.dtype)} cannot be shown to hold of it; declare "
                    f"no dtype, or store a container that has one"
                )
            if not np.can_cast(dtype, element_spec.dtype, casting="same_kind"):
                raise TypeError(
                    f"the stored array has dtype {dtype}, which the declared element "
                    f"{np.dtype(element_spec.dtype)} does not admit; a widening or a "
                    f"within-kind narrowing passes, a cross-kind conversion does not"
                )
        batch_shape = stored[: len(stored) - n_event] if n_event else stored
        if not batch_shape:
            raise ValueError(
                f"a batch has at least one batch axis, and an array of shape {stored} over "
                f"elements of event shape {event_shape} leaves none; a single value is a "
                f"NumericArray"
            )

        names = (level_names,) if isinstance(level_names, str) else tuple(level_names)
        groups = _axis_groups_for(batch_shape, names, axes_per_level, kind="NumericArrayBatch")
        object.__setattr__(self, "_values", values)
        self._init_batch(
            BatchSpec(element_spec, groups, names),
            name=name,
            name_is_auto=name_is_auto,
            provenance=provenance,
        )

    # -- what it holds ------------------------------------------------------

    @property
    def element_spec(self) -> NumericArraySpec:
        """What every element satisfies — a view on :attr:`spec`."""
        spec = self.spec.element_spec
        assert isinstance(spec, NumericArraySpec)
        return spec

    @property
    def values(self) -> Any:
        """The stored array, batch axes leading, in native form. Untracked."""
        return self._values

    # -- the array shim, as ``NumericRecordBatch`` carries from its sole field.
    # ``batch_shape`` / ``batch_size`` stay the names for the batch axes alone.

    @property
    def shape(self) -> tuple[int, ...]:
        """The store's full shape, ``(*batch_shape, *event_shape)``."""
        return tuple(_event_shape_of(self._values))

    @property
    def dtype(self) -> Any:
        """The store's dtype — the value's, as a :class:`NumericArray`'s is."""
        return _numpy_dtype_of(self._values)

    @property
    def ndim(self) -> int:
        """The rank of the store, batch axes included."""
        return len(self.shape)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = _to_numpy_array(self._values)
        arr = np.asarray(arr, dtype=dtype) if dtype is not None else arr
        return arr.copy() if copy else arr

    def as_jax(self) -> Any:
        """The store as a ``jax.Array`` — the single conversion point.

        A store already held as one passes through, tracers included; a native
        container converts through its registered backend once and is memoised
        for this instance, as a :class:`NumericArray`'s value is.
        """
        if isinstance(self._values, jax.Array):
            return self._values
        cached = getattr(self, "_jax_cache", None)
        if cached is None:
            cached = _to_jax_array(self._values)
            object.__setattr__(self, "_jax_cache", cached)
        return cached

    def __jax_array__(self) -> Any:
        return self.as_jax()

    def __repr__(self) -> str:
        return (
            f"NumericArrayBatch(batch_shape={self.batch_shape}, "
            f"levels={self.level_names}, event_shape={tuple(self.element_spec.shape)})"
        )

    # -- the concrete-storage seam ------------------------------------------

    def _element_at(self, index: tuple[int, ...], *, name: str) -> NumericArray:
        """The array at a fully-integer positional *index*, as a `NumericArray`.

        The materializing side of both rules
        :meth:`~probpipe.core._batch.Batch._element_at` states: the element
        takes the derived *name*, marked auto, and inherits this batch's
        provenance. Selection goes through the backend, since ``[]`` is
        positional only on a numpy-protocol container.
        """
        return self._inherit_provenance(
            NumericArray(
                name,
                _take_at(self._values, index),
                name_is_auto=True,
                spec=self.element_spec,
            )
        )

    def _sub_batch_at(self, index: tuple[int | slice, ...], *, spec: BatchSpec, name: str) -> Self:
        """A view over the same array, indexed on the batch axes as given.

        *index* addresses the leading axes only, so the event axes are untouched
        and selection yields a view rather than a copy. It goes through the
        value's backend, since ``[]`` is not positional on every container. Built without
        ``__init__`` for the reason ``RecordBatch`` gives: the spec and the name
        are already decided, and re-deriving them from the view's own shape
        would lose the levels a dropped axis came from.
        """
        view = object.__new__(self._view_type)
        object.__setattr__(view, "_values", _take_at(self._values, index))
        view._init_batch(spec, name=name, name_is_auto=True)
        return view


# ---------------------------------------------------------------------------
# JAX PyTree registration
# ---------------------------------------------------------------------------


def _numeric_array_batch_flatten(batch: NumericArrayBatch):
    """Flatten for JAX traversal: the store, keyed by the aux spec.

    The boundary presents a bare array, as a ``NumericArray``'s flatten does.
    Handing the native container to JAX instead fails abstractification before
    ``__jax_array__`` is ever consulted, so a pandas- or xarray-backed batch
    could not enter a trace at all.
    """
    return [batch.as_jax()], (batch._spec, batch._name, batch._name_is_auto)


def _numeric_array_batch_unflatten(aux, children):
    """Rebuild over the transformations a batch can state honestly.

    The contract ``RecordBatch`` states, over one column instead of many. A
    treedef carries neither ``in_axes`` nor ``out_axes``, so which level a
    transform consumed is unrecoverable — *a shape is not a provenance* — and
    two transformations are supported:

    - **Every batch axis preserved**, the ordinary round trip, reusing the spec.
    - **Every batch axis removed**: the value is one element, so a
      :class:`NumericArray` is returned.
    """
    spec, name, name_is_auto = aux
    (values,) = children
    element_spec = spec.element_spec
    event_rank = len(element_spec.shape)
    shape = getattr(values, "shape", None)
    if shape is None:
        # A skeleton or sentinel has no shape to measure; rebuilt verbatim.
        view = object.__new__(NumericArrayBatch)
        object.__setattr__(view, "_values", values)
        view._init_batch(spec, name=name, name_is_auto=name_is_auto)
        return view
    # The element's own axes are not the transform's to change. A rank check
    # alone would admit a store whose trailing axes no longer match what the
    # element declares, and the batch would build with a spec that is a false
    # statement about its own store — surfacing only at the first selection.
    event_shape = tuple(element_spec.shape)
    if event_rank and tuple(shape)[len(shape) - event_rank :] != event_shape:
        raise ValueError(
            f"a transform left this NumericArrayBatch over a store of {tuple(shape)}, whose "
            f"trailing axes are not the event shape {event_shape} its elements declare. A "
            f"transform maps the elements; changing what one *is* rebuilds the batch where "
            f"the new element spec is known"
        )
    surviving = tuple(shape)[: len(shape) - event_rank] if event_rank else tuple(shape)
    if surviving == tuple(spec.batch_shape):
        view = object.__new__(NumericArrayBatch)
        object.__setattr__(view, "_values", values)
        view._init_batch(spec, name=name, name_is_auto=name_is_auto)
        return view
    if not surviving:
        return NumericArray(name, values, name_is_auto=name_is_auto, spec=element_spec)
    raise ValueError(
        f"a transform left this NumericArrayBatch over {surviving} where its levels account "
        f"for {tuple(spec.batch_shape)}. A batch keeps every batch axis or removes all of "
        f"them, since an added or resized axis belongs to no level and unflattening has no "
        f"name to give one; build the batch where the axis is added, or map over its store"
    )


jax.tree_util.register_pytree_node(
    NumericArrayBatch, _numeric_array_batch_flatten, _numeric_array_batch_unflatten
)

# The numeric-array kind: its spec, its tracked class, and this batch form.
register_kind(NumericArraySpec, term_class=NumericArray, batch_class=NumericArrayBatch)


class _MappedBatchStore:
    """A single-store batch taken apart, to cross a mapping transform.

    The counterpart of :class:`~probpipe.core._record_batch._MappedBatchColumns`
    for a batch that holds one store rather than a column per field. The reason
    is the same: unflattening refuses an added axis because it has no name to
    give the level, which is right for a raw transform and wrong for an executor
    that knows both. So the executor hands the transform this inert carrier —
    its unflatten rebuilds it verbatim, checking nothing — and rebuilds the batch
    on the far side, where the level name is in hand.

    Private and short-lived: wrapped and unwrapped within one call.
    """

    __slots__ = ("axis_groups", "element_spec", "level_names", "name", "name_is_auto", "store")

    def __init__(
        self,
        name: str,
        store: Any,
        /,
        *,
        element_spec: NumericArraySpec,
        level_names: tuple[str, ...],
        axis_groups: tuple[tuple[int, ...], ...],
        name_is_auto: bool,
    ):
        self.store = store
        self.element_spec = element_spec
        self.level_names = level_names
        self.axis_groups = axis_groups
        self.name = name
        self.name_is_auto = name_is_auto

    @classmethod
    def of(cls, batch: NumericArrayBatch) -> _MappedBatchStore:
        """Take *batch* apart, keeping what unflattening could not have inferred."""
        return cls(
            batch._name,
            batch.values,
            element_spec=batch.element_spec,
            level_names=tuple(batch.level_names),
            axis_groups=tuple(batch.axis_groups),
            name_is_auto=batch._name_is_auto,
        )


def _mapped_batch_store_flatten(carried: _MappedBatchStore):
    return [carried.store], (
        carried.element_spec,
        carried.level_names,
        carried.axis_groups,
        carried.name,
        carried.name_is_auto,
    )


def _mapped_batch_store_unflatten(aux, children) -> _MappedBatchStore:
    element_spec, level_names, axis_groups, name, name_is_auto = aux
    (store,) = children
    # No rank check, deliberately: the added axis is the point, and the caller
    # that added it is the one that can name it.
    return _MappedBatchStore(
        name,
        store,
        element_spec=element_spec,
        level_names=level_names,
        axis_groups=axis_groups,
        name_is_auto=name_is_auto,
    )


jax.tree_util.register_pytree_node(
    _MappedBatchStore, _mapped_batch_store_flatten, _mapped_batch_store_unflatten
)
