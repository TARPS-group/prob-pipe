"""Shared construction for the contract tests of the call stack (design Part V)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import EmpiricalDistribution, Normal, Record


def error_of(call: Callable[[], Any]) -> BaseException | None:
    """The exception *call* raises, or None when it returns.

    A contract test asserts the error's type with ``isinstance``, so a call that
    raises another error, or none, fails the assertion rather than escaping it.
    """
    try:
        call()
    except Exception as error:
        return error
    return None


def standard_normal(name: str = "z") -> Normal:
    """A scalar law that samples."""
    return Normal(loc=0.0, scale=1.0, name=name)


def record_law(name: str = "joint", *, n: int = 12) -> EmpiricalDistribution:
    """An empirical law over records ``{a, b}`` whose every atom has ``b == 2 * a``."""
    a = jnp.arange(float(n))
    return EmpiricalDistribution(name, Record("atoms", {"a": a, "b": 2.0 * a}))


def one_field_law(name: str = "posterior", *, n: int = 12) -> EmpiricalDistribution:
    """An empirical law whose event is a record with the single field ``beta``."""
    return EmpiricalDistribution(name, Record("atoms", {"beta": jnp.arange(2.0 * n).reshape(n, 2)}))


def atom_leaves(law: Any) -> list[np.ndarray]:
    """The stored atoms of an empirical law, one array per leaf, whatever its event kind."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(law.samples)]
