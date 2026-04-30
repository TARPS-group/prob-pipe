"""Support constraints for distributions (real, positive, simplex, ...)."""

from __future__ import annotations

import jax.numpy as jnp

from ..custom_types import Array, ArrayLike

__all__ = [
    "Constraint",
    "real",
    "positive",
    "non_negative",
    "non_negative_integer",
    "boolean",
    "unit_interval",
    "simplex",
    "positive_definite",
    "sphere",
    "interval",
    "greater_than",
    "integer_interval",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Constraint:
    """Describes the support of a distribution (the set of valid values)."""

    def check(self, value: ArrayLike) -> Array:
        """Return a boolean array indicating which elements satisfy the constraint."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))


# ---------------------------------------------------------------------------
# Concrete constraints
# ---------------------------------------------------------------------------

class _Real(Constraint):
    """All real numbers."""
    def check(self, value: ArrayLike) -> Array:
        return jnp.isfinite(jnp.asarray(value))
    def __repr__(self) -> str:
        return "real"

class _Positive(Constraint):
    """Strictly positive reals (0, inf)."""
    def check(self, value: ArrayLike) -> Array:
        return jnp.asarray(value) > 0
    def __repr__(self) -> str:
        return "positive"

class _NonNegative(Constraint):
    """Non-negative reals [0, inf)."""
    def check(self, value: ArrayLike) -> Array:
        return jnp.asarray(value) >= 0
    def __repr__(self) -> str:
        return "non_negative"

class _NonNegativeInteger(Constraint):
    """Non-negative integers {0, 1, 2, ...}."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= 0) & (v == jnp.floor(v))
    def __repr__(self) -> str:
        return "non_negative_integer"

class _Boolean(Constraint):
    """Binary values {0, 1}."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v == 0) | (v == 1)
    def __repr__(self) -> str:
        return "boolean"

class _UnitInterval(Constraint):
    """Closed unit interval [0, 1]."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= 0) & (v <= 1)
    def __repr__(self) -> str:
        return "unit_interval"

class _Simplex(Constraint):
    """Probability simplex (non-negative, sums to 1 along last axis)."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (jnp.all(v >= 0, axis=-1)) & (jnp.abs(jnp.sum(v, axis=-1) - 1.0) < 1e-5)
    def __repr__(self) -> str:
        return "simplex"

class _PositiveDefinite(Constraint):
    """Positive-definite matrices."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        eigvals = jnp.linalg.eigvalsh(v)
        return jnp.all(eigvals > 0, axis=-1)
    def __repr__(self) -> str:
        return "positive_definite"

class _Sphere(Constraint):
    """Unit sphere (vectors with unit L2 norm)."""
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return jnp.abs(jnp.linalg.norm(v, axis=-1) - 1.0) < 1e-5
    def __repr__(self) -> str:
        return "sphere"

class _Interval(Constraint):
    """Half-open or closed interval [low, high]."""
    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = low
        self.high = high
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= self.low) & (v <= self.high)
    def __repr__(self) -> str:
        return f"interval({self.low}, {self.high})"
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return bool(jnp.array_equal(self.low, other.low)) and bool(
            jnp.array_equal(self.high, other.high)
        )
    def __hash__(self) -> int:
        # Coarse but valid: equal instances hash equal. Parameterized
        # constraints rarely end up in sets/dicts, and array-valued
        # bounds aren't directly hashable, so we hash on type only.
        return hash(type(self))

class _GreaterThan(Constraint):
    """Record strictly greater than a lower bound."""
    def __init__(self, lower_bound: ArrayLike):
        self.lower_bound = lower_bound
    def check(self, value: ArrayLike) -> Array:
        return jnp.asarray(value) > self.lower_bound
    def __repr__(self) -> str:
        return f"greater_than({self.lower_bound})"
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return bool(jnp.array_equal(self.lower_bound, other.lower_bound))
    def __hash__(self) -> int:
        return hash(type(self))

class _IntegerInterval(Constraint):
    """Integer values in [low, high]."""
    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = low
        self.high = high
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= self.low) & (v <= self.high) & (v == jnp.floor(v))
    def __repr__(self) -> str:
        return f"integer_interval({self.low}, {self.high})"
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return bool(jnp.array_equal(self.low, other.low)) and bool(
            jnp.array_equal(self.high, other.high)
        )
    def __hash__(self) -> int:
        return hash(type(self))


# ---------------------------------------------------------------------------
# Singleton instances for common constraints
# ---------------------------------------------------------------------------

real = _Real()
positive = _Positive()
non_negative = _NonNegative()
non_negative_integer = _NonNegativeInteger()
boolean = _Boolean()
unit_interval = _UnitInterval()
simplex = _Simplex()
positive_definite = _PositiveDefinite()
sphere = _Sphere()


# ---------------------------------------------------------------------------
# Factory functions for parameterized constraints
# ---------------------------------------------------------------------------

def interval(low: ArrayLike, high: ArrayLike) -> _Interval:
    return _Interval(low, high)

def greater_than(lower_bound: ArrayLike) -> _GreaterThan:
    return _GreaterThan(lower_bound)

def integer_interval(low: ArrayLike, high: ArrayLike) -> _IntegerInterval:
    return _IntegerInterval(low, high)


# ---------------------------------------------------------------------------
# Constraint lattice (subset relationships)
# ---------------------------------------------------------------------------

# Immediate supersets in the constraint partial order.
# Each constraint type maps to the types that are its direct (one-step)
# supersets.  The transitive closure is computed once by ``_all_supersets``.
_IMMEDIATE_SUPERSETS: dict[type, tuple[type, ...]] = {
    _Boolean: (_NonNegativeInteger, _UnitInterval),
    _UnitInterval: (_NonNegative,),
    _Positive: (_NonNegative,),
    _NonNegative: (_Real,),
    _NonNegativeInteger: (_NonNegative,),
    _Simplex: (_UnitInterval,),
    _Sphere: (_Real,),
    _PositiveDefinite: (_Real,),
}

_ALL_SUPERSETS: dict[type, set[type]] | None = None


def _all_supersets() -> dict[type, set[type]]:
    """Return the transitive closure of ``_IMMEDIATE_SUPERSETS``.

    Computed once and cached at module level.
    """
    global _ALL_SUPERSETS
    if _ALL_SUPERSETS is not None:
        return _ALL_SUPERSETS

    result: dict[type, set[type]] = {}

    def _collect(t: type) -> set[type]:
        if t in result:
            return result[t]
        immediate = _IMMEDIATE_SUPERSETS.get(t, ())
        sups: set[type] = set(immediate)
        for parent in immediate:
            sups |= _collect(parent)
        result[t] = sups
        return sups

    for t in _IMMEDIATE_SUPERSETS:
        _collect(t)

    _ALL_SUPERSETS = result
    return result


def _supports_compatible(source: Constraint, target: Constraint) -> bool:
    """Check whether *source* support is a subset of *target* support.

    Conservative: returns ``True`` when in doubt (e.g. for parameterized
    constraints that can't be compared structurally).
    """
    # Fast path for identical instances; parameterized branches below
    # handle the equal-but-distinct case for array-valued bounds.
    if source is target:
        return True

    supersets = _all_supersets()
    source_type = type(source)
    target_type = type(target)

    # source ⊆ target?
    if target_type in supersets.get(source_type, set()):
        return True

    # target ⊆ source means target is *narrower* — incompatible
    if source_type in supersets.get(target_type, set()):
        return False

    # Parameterized constraints: interval/greater_than. Bounds may be
    # array-valued (per-dim Uniform, TruncatedNormal, ...), so reduce
    # element-wise comparisons with ``jnp.all``.
    if isinstance(source, _Interval) and isinstance(target, _Interval):
        return bool(jnp.all(source.low >= target.low)) and bool(
            jnp.all(source.high <= target.high)
        )
    if isinstance(source, _Interval) and isinstance(target, _Real):
        return True
    if isinstance(source, _GreaterThan) and isinstance(target, _GreaterThan):
        return bool(jnp.all(source.lower_bound >= target.lower_bound))
    if isinstance(source, _GreaterThan) and isinstance(target, _Real):
        return True
    if isinstance(source, (_Positive, _NonNegative)) and isinstance(target, _GreaterThan):
        return True  # (0, inf) or [0, inf) ⊂ (lb, inf) when lb <= 0
    if isinstance(source, _IntegerInterval) and isinstance(target, _IntegerInterval):
        return bool(jnp.all(source.low >= target.low)) and bool(
            jnp.all(source.high <= target.high)
        )
    if isinstance(source, _IntegerInterval) and isinstance(target, (_NonNegativeInteger, _Real)):
        return True

    # Conservative: allow when we can't determine
    return True
