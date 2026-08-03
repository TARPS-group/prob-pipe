"""Importable callable fixtures for workflow replay ABI tests."""

from __future__ import annotations


def replayable_affine(value: float, offset: float = 1.25) -> float:
    """Return a small expression with a default and local bytecode state."""
    shifted = value + offset
    return shifted * 2.0


def replayable_identity(value):
    """Return one value unchanged."""
    return value
