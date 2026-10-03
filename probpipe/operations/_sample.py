"""The sample operation: one draw, or a batch of draws, of a distribution.

``sample(d)`` returns one draw at the kind the law's event declaration names,
and a non-empty ``sample_shape`` prepends batch axes on a level named
``sample``, returning the batch form of that kind. The draw's key is a
workflow-owned random event, so the operation takes no key.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np

from ..core._batch import BatchSpec, _ranks_of
from ..core._record_batch import _batch_class_for
from ..core._record_spec import RecordSpec
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsSampling
from ..distributions._distribution import Distribution, DistributionSpec
from ..distributions._factored import _raw_record
from ..functions._call import ApplicabilityError
from ..functions._result import SAMPLE_LEVEL
from ..values import FunctionSpec
from ._operation import BoundCall, _call_label, _workflow_draws, operation

__all__ = ["sample"]


def _record_batch(value: Any, call: BoundCall, result: OutputSpec | None) -> Any:
    """*value*, a nested mapping of raw columns, as the batch of records *result* declares.

    A record-valued raw result is a nested mapping of raw leaves, stacked with
    the batch axes leading. The engine's return step assembles a declared batch
    from arrays, records of columns, and object arrays, so a mapping is
    assembled here: as the ``RecordBatch`` or ``NumericRecordBatch`` the
    declared element calls for, on the declared levels and under the
    call's result label. A record held inside the mapping is read as its
    own nested mapping. Under ``raw`` the mapping is the result. Any other value
    is returned as it is, and so is a value whose declared result is not a
    batch of records.
    """
    if call.controls["raw"] or not isinstance(value, Mapping) or result is None:
        return value
    spec = result.spec
    if not isinstance(spec, BatchSpec) or not isinstance(spec.element_spec, RecordSpec):
        return value
    return _batch_class_for(spec.element_spec)(
        _call_label(call),
        _raw_record(value),
        tuple(spec.level_names),
        element_spec=spec.element_spec,
        axes_per_level=_ranks_of(spec.axis_groups),
    )


def _sample_shape(sample_shape: Any) -> tuple[int, ...]:
    """*sample_shape* as a tuple of sizes, a bare integer being one axis.

    Raises
    ------
    ApplicabilityError
        If a size is not a non-negative integer, a bool included.
    """
    axes = sample_shape if isinstance(sample_shape, tuple) else (sample_shape,)
    if any(isinstance(axis, bool) for axis in axes):
        raise ApplicabilityError("sample_shape must be an integer or a tuple of integers, not bool")
    try:
        shape = tuple(operator.index(axis) for axis in axes)
    except TypeError:
        raise ApplicabilityError("sample_shape must be an integer or a tuple of integers") from None
    if any(axis < 0 for axis in shape):
        raise ApplicabilityError(f"sample_shape sizes must be non-negative; got {shape!r}")
    return shape


def _sample_result(d: DistributionSpec, sample_shape: Any) -> OutputSpec:
    """A draw and a batch of draws carry the law's event components, the batch on a level named sample.

    The returned kind is read from ``event_spec.spec``, so an array and a
    one-field record stay distinct under every sample shape.

    Raises
    ------
    ApplicabilityError
        If *sample_shape* is malformed, or the declaration has free dimensions,
        which a draw has no value to bind.
    """
    shape = _sample_shape(sample_shape)
    free = d.free_dims
    if free:
        raise ApplicabilityError(
            f"sample requires a concrete declaration; the free dimensions are {sorted(free)}"
        )
    if not shape:
        return d.event_spec
    return d.event_spec._with_spec(BatchSpec(d.event_spec.spec, (shape,), (SAMPLE_LEVEL,)))


@operation(result=_sample_result)
def sample(d: Distribution, sample_shape: tuple[int, ...] = ()):
    """Draw from a distribution.

    Parameters
    ----------
    d : Distribution
        The law to draw from.
    sample_shape : int or tuple of int
        The batch axes to prepend; ``()`` draws once, and a bare integer is one
        axis.

    Returns
    -------
    TrackedTerm
        One draw at the kind the event declaration names, such as a
        ``NumericArray`` or a ``Record``; or, for a non-empty *sample_shape*, the
        batch form of that kind with the leading axes on a level named
        ``sample``. Under ``with_options(raw=True)`` the draws are returned as
        ``d._sample`` gives them, so a record-valued law's batch of draws is the
        nested mapping of its raw columns.

    Raises
    ------
    ApplicabilityError
        If *sample_shape* is malformed, or the declaration has free dimensions.
    ResolutionError
        If *d* does not sample.
    """


def _draw(call: BoundCall, result: OutputSpec | None) -> Any:
    """``d._sample`` under the sample shape, with a workflow-owned key.

    Under a non-empty shape the key splits by draw index inside ``_sample``, so
    the draws are jointly independent and reproducible together. Draws that are
    a nested mapping of raw columns become the declared batch of records.
    """
    d, shape = call.operands["d"], _sample_shape(call.operands["sample_shape"])
    draws = _workflow_draws(d, shape, operation_kind="sample", execution_mode="sampled")
    if isinstance(d.event_spec.spec, FunctionSpec):
        return _function_draws(draws, shape)
    return _record_batch(draws, call, result)


class _DrawAt:
    """One function draw of several evaluated together: their values at one index of the sample axes."""

    def __init__(self, draws: Callable[..., Any], index: tuple[int, ...]) -> None:
        self._draws, self._index = draws, index

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return jax.tree.map(lambda values: values[self._index], self._draws(*args, **kwargs))


def _function_draws(draws: Any, shape: tuple[int, ...]) -> Any:
    """Function draws under *shape* as the object array of the drawn callables.

    A random function's ``_sample`` may return one callable that evaluates every
    draw, its values led by the sample axes, and the batch form holds each draw
    as a callable of its own.
    """
    if not shape or isinstance(draws, np.ndarray) or not callable(draws):
        return draws
    elements = np.empty(shape, dtype=object)
    for index in np.ndindex(*shape):
        elements[index] = _DrawAt(draws, index)
    return elements


sample.capability_route(
    "exact", operand="d", protocol=SupportsSampling, method="_sample", exact=True, execute=_draw
)
