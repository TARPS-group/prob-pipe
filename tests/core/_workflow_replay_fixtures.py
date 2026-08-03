"""Importable callable fixtures for workflow replay ABI tests."""

from __future__ import annotations

from probpipe import Normal, sample

ENABLE_EXTRA_AUTOMATIC = False


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
