"""Shared utility functions for probpipe."""

from __future__ import annotations

import math

import numpy as np

import jax
import jax.numpy as jnp

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


def _is_numeric_array(x: object) -> bool:
    """Return ``True`` if *x* is a JAX or numpy array with a numeric dtype.

    Numpy object arrays (used for generic non-array samples in
    ``EmpiricalDistribution``) return ``False``.
    """
    if isinstance(x, jax.Array):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype != object
    return False


def prod(shape: tuple[int, ...]) -> int:
    """Product of a shape tuple, returning 1 for an empty tuple.

    This is a thin wrapper around :func:`math.prod` that treats the empty
    tuple as having product 1, which is the correct convention for scalar
    (0-dimensional) shapes throughout probpipe.
    """
    return math.prod(shape) if shape else 1
