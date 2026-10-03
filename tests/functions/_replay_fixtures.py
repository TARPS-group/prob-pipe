"""Importable callable fixtures for workflow replay ABI tests."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from enum import Enum

import numpy as np

from probpipe import Normal, positive, sample

ENABLE_EXTRA_AUTOMATIC = False

_CYCLIC_DEFAULT: list[object] = []
_CYCLIC_DEFAULT.append(_CYCLIC_DEFAULT)
_NUMERIC_ARRAY_DEFAULT = np.asarray([1.0, 2.0], dtype=np.float64)
_OBJECT_ARRAY_DEFAULT = np.asarray([object()], dtype=object)
_STRUCTURED_ARRAY_DEFAULT = np.asarray([(1,)], dtype=[("value", "<i4")])
_BIG_ENDIAN_ARRAY_DEFAULT = np.asarray([1, 2], dtype=">i2")
_SET_DEFAULT = {3, 1}
_FROZENSET_DEFAULT = frozenset({4, 2})
_MAPPING_DEFAULT = {"b": 2, "a": 1}
_DTYPE_DEFAULT = np.dtype(">i4")
_SCALAR_DEFAULT = np.int16(7)
_TYPING_LIST_DEFAULT = typing.List[int]  # noqa: UP006 - exercises the typing ABI branch


@dataclass(frozen=True)
class ReplayableDefaultState:
    """Portable dataclass state used by callable-definition ABI tests."""

    count: int
    labels: tuple[str, ...]


class ReplayableDefaultMode(Enum):
    """Portable enum state used by callable-definition ABI tests."""

    FAST = "fast"


_DATACLASS_DEFAULT = ReplayableDefaultState(2, ("left", "right"))


def replayable_affine(value: float, offset: float = 1.25) -> float:
    """Return a small expression with a default and local bytecode state."""
    shifted = value + offset
    return shifted * 2.0


def replayable_identity(value):
    """Return one value unchanged."""
    return value


def replayable_difference(left, right):
    """Return a simple two-source difference."""
    return right - left


def replayable_optional_nested(value):
    """Use a mutable global only to exercise unexpected-event validation."""
    if ENABLE_EXTRA_AUTOMATIC:
        sample(Normal(loc=value, scale=1.0, name="extra"))
    return value


def replayable_cyclic_default(value, state=_CYCLIC_DEFAULT):
    """Return one value while carrying cyclic default state."""
    del state
    return value


def replayable_numeric_array_default(value, state=_NUMERIC_ARRAY_DEFAULT):
    """Return one value while carrying a portable numeric array default."""
    del state
    return value


def replayable_object_array_default(value, state=_OBJECT_ARRAY_DEFAULT):
    """Return one value while carrying process-local object-array state."""
    del state
    return value


def replayable_structured_array_default(value, state=_STRUCTURED_ARRAY_DEFAULT):
    """Return one value while carrying structured array state."""
    del state
    return value


def replayable_canonical_defaults(
    value,
    ellipsis_value=...,
    complex_value=1 + 2j,
    set_value=_SET_DEFAULT,
    frozenset_value=_FROZENSET_DEFAULT,
    mapping_value=_MAPPING_DEFAULT,
    dtype_value=_DTYPE_DEFAULT,
    scalar_value=_SCALAR_DEFAULT,
    array_value=_BIG_ENDIAN_ARRAY_DEFAULT,
    dataclass_value=_DATACLASS_DEFAULT,
    constraint_value=positive,
    enum_value=ReplayableDefaultMode.FAST,
    generic_value=list[int],
    typing_value=_TYPING_LIST_DEFAULT,
    type_value=int,
):
    """Return one value while carrying every approved structured default family."""
    del (
        ellipsis_value,
        complex_value,
        set_value,
        frozenset_value,
        mapping_value,
        dtype_value,
        scalar_value,
        array_value,
        dataclass_value,
        constraint_value,
        enum_value,
        generic_value,
        typing_value,
        type_value,
    )
    return value
