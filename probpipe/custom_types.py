"""
Type aliases for the array backend.

ProbPipe uses JAX as its primary array backend. These aliases provide a
single place to change if the backend ever needs to swap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------
type Array = jnp.ndarray
type ArrayLike = jnp.ndarray | list | tuple | float | int

# ---------------------------------------------------------------------------
# Scalar numeric types
# ---------------------------------------------------------------------------
type Float = jnp.floating
type Number = jnp.floating | jnp.integer

# ---------------------------------------------------------------------------
# Random state
# ---------------------------------------------------------------------------
type PRNGKey = jax.Array  # JAX PRNG key
