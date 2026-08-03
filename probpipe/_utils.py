"""Shared utility functions for ProbPipe."""

from __future__ import annotations

import jax
import numpy as np


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
