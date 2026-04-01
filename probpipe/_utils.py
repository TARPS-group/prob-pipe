"""Shared utility functions for probpipe."""

from __future__ import annotations

import math

import jax

from .custom_types import PRNGKey

# ---------------------------------------------------------------------------
# Auto-key helper (for convenience when key is omitted)
# ---------------------------------------------------------------------------

_AUTO_KEY_COUNTER: int = 0


def _auto_key() -> PRNGKey:
    """Generate a JAX PRNGKey from a global counter.

    Convenient for interactive / exploratory use.  Not reproducible
    across runs — pass an explicit key when reproducibility matters.
    """
    global _AUTO_KEY_COUNTER
    key = jax.random.PRNGKey(_AUTO_KEY_COUNTER)
    _AUTO_KEY_COUNTER += 1
    return key


def prod(shape: tuple[int, ...]) -> int:
    """Product of a shape tuple, returning 1 for an empty tuple.

    This is a thin wrapper around :func:`math.prod` that treats the empty
    tuple as having product 1, which is the correct convention for scalar
    (0-dimensional) shapes throughout probpipe.
    """
    return math.prod(shape) if shape else 1
