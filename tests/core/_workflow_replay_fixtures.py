"""Importable callable fixtures for workflow replay ABI tests."""

from __future__ import annotations

import numpy as np

from probpipe import Normal, sample

ENABLE_EXTRA_AUTOMATIC = False

_CYCLIC_DEFAULT: list[object] = []
_CYCLIC_DEFAULT.append(_CYCLIC_DEFAULT)
_NUMERIC_ARRAY_DEFAULT = np.asarray([1.0, 2.0], dtype=np.float64)
_OBJECT_ARRAY_DEFAULT = np.asarray([object()], dtype=object)
_STRUCTURED_ARRAY_DEFAULT = np.asarray([(1,)], dtype=[("value", "<i4")])


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
