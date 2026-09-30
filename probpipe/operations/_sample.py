"""The sample operation: one draw, or a batch of draws, of a distribution.

``sample(d)`` returns one draw at the kind the law's event declaration names,
and a non-empty ``sample_shape`` prepends batch axes on a level named
``sample``, returning the batch form of that kind. The draw's key is a
workflow-owned random event, so the operation takes no key.
"""

from __future__ import annotations

import operator
from typing import Any

from ..core._batch import BatchSpec
from ..core._broadcast_distributions import SAMPLE_LEVEL
from ..core._specs import OutputSpec
from ..distributions._capabilities import SupportsSampling
from ..distributions._distribution import Distribution, DistributionSpec
from ._operation import ApplicabilityError, BoundCall, _workflow_draws, operation

__all__ = ["sample"]


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
    """A draw carries the law's event declaration, and a batch of draws is on a level named sample.

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
    return OutputSpec(sample=BatchSpec(d.event_spec.spec, (shape,), (SAMPLE_LEVEL,)))


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
        ``d._sample`` gives them.

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
    the draws are jointly independent and reproducible together.
    """
    return _workflow_draws(
        call.operands["d"],
        _sample_shape(call.operands["sample_shape"]),
        operation_kind="sample",
        execution_mode="sampled",
    )


sample.capability_route(
    "exact", operand="d", protocol=SupportsSampling, method="_sample", exact=True, execute=_draw
)
