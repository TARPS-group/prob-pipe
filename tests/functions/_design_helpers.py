"""Shared construction for the contract tests of the call stack (design Part V)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from probpipe import (
    EmpiricalDistribution,
    Normal,
    NumericArraySpec,
    NumericRecordBatch,
    NumericRecordSpec,
)


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


def standard_normal(label: str = "z") -> Normal:
    """A scalar law that samples."""
    return Normal(loc=0.0, scale=1.0, label=label)


def record_law(label: str = "joint", *, n: int = 12) -> EmpiricalDistribution:
    """An empirical law over records ``{a, b}`` whose every atom has ``b == 2 * a``."""
    a = jnp.arange(float(n))
    spec = NumericRecordSpec(a=NumericArraySpec((), a.dtype), b=NumericArraySpec((), a.dtype))
    atoms = NumericRecordBatch("atoms", {"a": a, "b": 2.0 * a}, "atom", element_spec=spec)
    return EmpiricalDistribution(label, atoms)


def one_field_law(label: str = "posterior", *, n: int = 12) -> EmpiricalDistribution:
    """An empirical law whose event is a record with the single field ``beta``."""
    beta = jnp.arange(2.0 * n).reshape(n, 2)
    spec = NumericRecordSpec(beta=NumericArraySpec((2,), beta.dtype))
    atoms = NumericRecordBatch("atoms", {"beta": beta}, "atom", element_spec=spec)
    return EmpiricalDistribution(label, atoms)


def atom_leaves(law: Any) -> list[np.ndarray]:
    """The stored atoms of an empirical law, one array per leaf, whatever its event kind."""
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(law.atoms)]
