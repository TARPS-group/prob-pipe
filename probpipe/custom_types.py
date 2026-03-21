"""
Type aliases for the array backend.

ProbPipe uses JAX as its primary array backend. These aliases provide a
single place to change if the backend ever needs to swap.
"""

from __future__ import annotations
from typing import TypeAlias

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------
Array: TypeAlias = jnp.ndarray
ArrayLike: TypeAlias = jnp.ndarray | list | tuple | float | int

# ---------------------------------------------------------------------------
# Scalar numeric types
# ---------------------------------------------------------------------------
Float: TypeAlias = jnp.floating
Number: TypeAlias = jnp.floating | jnp.integer

# ---------------------------------------------------------------------------
# Random state
# ---------------------------------------------------------------------------
PRNGKey: TypeAlias = jax.Array  # JAX PRNG key
