"""Every public argument of a shape, level names, or axis counts reads a single item alike.

STYLE_GUIDE §9.4 states the rule: a single int or str where a sequence is
expected is a sequence of one. Each case below builds the same object from the
single item and from its tuple of one, and refuses the same malformed inputs, so
an entry point that reads its argument by hand rather than through
``core/_shapes.py`` fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probpipe import (
    BatchSpec,
    DistributionBatch,
    Normal,
    NumericArrayBatch,
    NumericArraySpec,
    NumericRecordBatch,
    OpaqueBatch,
    OpaqueSpec,
    RecordBatch,
    Weights,
    sample,
)
from probpipe.functions._call import ApplicabilityError


@dataclass(frozen=True)
class _Case:
    """One public argument: how to build from it, the single item, and how to compare results."""

    build: Callable[[Any], Any]
    item: Any
    read: Callable[[Any], Any] = lambda result: result
    #: The error each malformed input raises: ``""``, ``True``, ``b"ab"``, ``{"a": 1}``.
    errors: tuple[type[Exception], ...] = (ValueError, TypeError, TypeError, TypeError)


def _objects(n: int) -> np.ndarray:
    store = np.empty(n, dtype=object)
    store[...] = "x"
    return store


_CASES = {
    "NumericArraySpec shape, a name": _Case(NumericArraySpec, "loc"),
    "NumericArraySpec shape, a size": _Case(NumericArraySpec, 3),
    "BatchSpec level, a name": _Case(lambda shape: BatchSpec(OpaqueSpec(), draw=shape), "S"),
    "BatchSpec level, a size": _Case(lambda shape: BatchSpec(OpaqueSpec(), draw=shape), 4),
    "NumericArrayBatch level_names": _Case(
        lambda names: NumericArrayBatch("b", jnp.zeros(2), names), "draw", lambda b: b.spec
    ),
    "RecordBatch level_names": _Case(
        lambda names: RecordBatch("b", {"x": _objects(2)}, names),
        "draw",
        lambda b: b.spec,
    ),
    "NumericRecordBatch level_names": _Case(
        lambda names: NumericRecordBatch("b", {"x": jnp.zeros(2)}, names), "draw", lambda b: b.spec
    ),
    "OpaqueBatch level_names": _Case(
        lambda names: OpaqueBatch("b", _objects(2), names), "draw", lambda b: b.spec
    ),
    "DistributionBatch level_names": _Case(
        lambda names: DistributionBatch("b", [Normal("n", 0.0, 1.0), Normal("n", 1.0, 1.0)], names),
        "draw",
        lambda b: b.spec,
    ),
    "NumericArrayBatch axes_per_level": _Case(
        lambda axes: NumericArrayBatch("b", jnp.zeros((2, 3)), "draw", axes_per_level=axes),
        2,
        lambda b: b.spec,
        (TypeError, TypeError, TypeError, TypeError),
    ),
    "sample sample_shape": _Case(
        lambda shape: sample(Normal("n", 0.0, 1.0), sample_shape=shape),
        5,
        lambda draws: draws.batch_shape,
        (ApplicabilityError,) * 4,
    ),
    "Weights.choice shape": _Case(
        lambda shape: Weights(n=10).choice(jax.random.PRNGKey(0), shape=shape),
        5,
        lambda indices: indices.shape,
        (TypeError,) * 4,
    ),
}


@pytest.fixture(params=list(_CASES), ids=list(_CASES))
def case(request) -> _Case:
    return _CASES[request.param]


def test_a_single_item_and_its_tuple_of_one_build_alike(case):
    single, tupled = case.build(case.item), case.build((case.item,))

    assert case.read(single) == case.read(tupled)
    assert repr(case.read(single)) == repr(case.read(tupled))


@pytest.mark.parametrize(
    ("index", "malformed"),
    [(0, ""), (1, True), (2, b"ab"), (3, {"a": 1})],
    ids=["empty-str", "bool", "bytes", "mapping"],
)
def test_a_malformed_argument_is_refused_alike(case, index, malformed):
    with pytest.raises(case.errors[index]):
        case.build(malformed)
