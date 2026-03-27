"""Shared utility functions for probpipe."""

from __future__ import annotations

import math


def prod(shape: tuple[int, ...]) -> int:
    """Product of a shape tuple, returning 1 for an empty tuple.

    This is a thin wrapper around :func:`math.prod` that treats the empty
    tuple as having product 1, which is the correct convention for scalar
    (0-dimensional) shapes throughout probpipe.
    """
    return math.prod(shape) if shape else 1
