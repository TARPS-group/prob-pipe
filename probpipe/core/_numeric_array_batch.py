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
    values : array-like
        One array holding every element, shaped ``(*batch_shape, *event_shape)``.
        Stored verbatim in its native form, as a :class:`NumericArray`'s value
        is, so a lazy or disk-backed column is not materialised to be batched.
    level_names : str or iterable of str
        One name per level, outermost first; a single string names one level.
    element_spec : NumericArraySpec
        What every element satisfies. Its ``shape`` is the event shape, so it is
        what splits the stored array's axes into batch and event.
    axis_groups : iterable of iterable of int, optional
        The axis sizes each level holds, in order, tiling ``batch_shape``.
        Defaults to one axis per level.
    name : str
        The batch's name, **required**, as a :class:`~probpipe.Record`'s and an
        :class:`~probpipe.Opaque`'s are. A batch is what an operation hands back,
        and the name is what says which one it is; a class-name default would name
        every batch in a pipeline alike. A caller that derives one says so with
        *name_is_auto*.
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
        no size to split by; or if *axis_groups* does not tile ``batch_shape``.

    Notes
    -----
    Selection yields the element kind, as it does for every batch:
    ``batch[i]`` is a :class:`NumericArray`. It materializes, so each element
    takes the derived name and inherits this batch's lineage.
    """

    _values: Any

    __slots__ = ("_values",)

    def __init__(
        self,
        values: Any,
        level_names: str | Iterable[str],
        *,
        element_spec: NumericArraySpec,
        axis_groups: Iterable[Iterable[int]] | None = None,
        name: str,
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
        groups = _axis_groups_for(batch_shape, names, axis_groups, kind="NumericArrayBatch")
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

    def __jax_array__(self) -> Any:
        return _to_jax_array(self._values)

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
                _take_at(self._values, index),
                name=name,
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
    """Flatten for JAX traversal: the store, keyed by the aux spec."""
    return [batch._values], (batch._spec, batch._name, batch._name_is_auto)


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
    surviving = tuple(shape)[: len(shape) - event_rank] if event_rank else tuple(shape)
    if surviving == tuple(spec.batch_shape):
        view = object.__new__(NumericArrayBatch)
        object.__setattr__(view, "_values", values)
        view._init_batch(spec, name=name, name_is_auto=name_is_auto)
        return view
    if not surviving:
        return NumericArray(values, name=name, name_is_auto=name_is_auto, spec=element_spec)
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
