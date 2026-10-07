"""Support constraints for distributions (real, positive, simplex, ...)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike

__all__ = [
    "Constraint",
    "boolean",
    "greater_than",
    "integer_interval",
    "interval",
    "non_negative",
    "non_negative_integer",
    "positive",
    "positive_definite",
    "real",
    "simplex",
    "sphere",
    "unit_interval",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Constraint:
    """Describes the support of a distribution (the set of valid values)."""

    def check(self, value: ArrayLike) -> Array:
        """Return a boolean array indicating which elements satisfy the constraint.

        A support that is a subset of the reals contains a complex entry only
        where its imaginary part is zero and its real part is in the support.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.__dict__.items()))))


def _known_equal(a: Constraint | None, b: Constraint | None) -> bool:
    """Whether two supports are known to be equal without reading a traced value.

    Supports whose comparison needs a traced parameter's value, as under ``jit``,
    count as different unless they are the same object. ``None``, an unset
    support, equals only ``None``.
    """
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        return bool(a == b)
    except jax.errors.ConcretizationTypeError:
        return False


class _RealConstraint(Constraint):
    """A constraint whose support is a subset of the reals.

    A subclass states the membership of a real array in :meth:`_check_real`,
    and :meth:`check` applies the rule of :meth:`Constraint.check` for complex
    values around it. The rule needs an explicit check because JAX orders
    complex values lexicographically, so ``1j > 0`` is true. ``_event_ndim``
    is the number of trailing axes that one membership result covers.
    """

    _event_ndim: int = 0

    def check(self, value: ArrayLike) -> Array:
        """Return a boolean array indicating which elements satisfy the constraint."""
        v = jnp.asarray(value)
        if not jnp.iscomplexobj(v):
            return self._check_real(v)
        is_real = jnp.isreal(v)
        if self._event_ndim:
            is_real = jnp.all(is_real, axis=tuple(range(-self._event_ndim, 0)))
        return is_real & self._check_real(v.real)

    def _check_real(self, v: Array) -> Array:
        """Return the membership of each entry of the real array *v*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete constraints
# ---------------------------------------------------------------------------


class _Real(_RealConstraint):
    """All real numbers."""

    def _check_real(self, v: Array) -> Array:
        return jnp.isfinite(v)

    def __repr__(self) -> str:
        return "real"


class _Positive(_RealConstraint):
    """Strictly positive reals (0, inf)."""

    def _check_real(self, v: Array) -> Array:
        return v > 0

    def __repr__(self) -> str:
        return "positive"


class _NonNegative(_RealConstraint):
    """Non-negative reals [0, inf)."""

    def _check_real(self, v: Array) -> Array:
        return v >= 0

    def __repr__(self) -> str:
        return "non_negative"


class _NonNegativeInteger(_RealConstraint):
    """Non-negative integers {0, 1, 2, ...}."""

    def _check_real(self, v: Array) -> Array:
        return (v >= 0) & (v == jnp.floor(v))

    def __repr__(self) -> str:
        return "non_negative_integer"


class _Boolean(_RealConstraint):
    """Binary values {0, 1}."""

    def _check_real(self, v: Array) -> Array:
        return (v == 0) | (v == 1)

    def __repr__(self) -> str:
        return "boolean"


class _UnitInterval(_RealConstraint):
    """Closed unit interval [0, 1]."""

    def _check_real(self, v: Array) -> Array:
        return (v >= 0) & (v <= 1)

    def __repr__(self) -> str:
        return "unit_interval"


class _Simplex(_RealConstraint):
    """Probability simplex (non-negative, sums to 1 along last axis)."""

    _event_ndim = 1

    def _check_real(self, v: Array) -> Array:
        return (jnp.all(v >= 0, axis=-1)) & (jnp.abs(jnp.sum(v, axis=-1) - 1.0) < 1e-5)

    def __repr__(self) -> str:
        return "simplex"


class _PositiveDefinite(_RealConstraint):
    """Positive-definite matrices."""

    _event_ndim = 2

    def _check_real(self, v: Array) -> Array:
        eigvals = jnp.linalg.eigvalsh(v)
        return jnp.all(eigvals > 0, axis=-1)

    def __repr__(self) -> str:
        return "positive_definite"


class _Sphere(_RealConstraint):
    """Unit sphere (vectors with unit L2 norm)."""

    _event_ndim = 1

    def _check_real(self, v: Array) -> Array:
        return jnp.abs(jnp.linalg.norm(v, axis=-1) - 1.0) < 1e-5

    def __repr__(self) -> str:
        return "sphere"


class _Interval(_RealConstraint):
    """Half-open or closed interval [low, high]."""

    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = low
        self.high = high

    def _check_real(self, v: Array) -> Array:
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


class _GreaterThan(_RealConstraint):
    """Record strictly greater than a lower bound."""

    def __init__(self, lower_bound: ArrayLike):
        self.lower_bound = lower_bound

    def _check_real(self, v: Array) -> Array:
        return v > self.lower_bound

    def __repr__(self) -> str:
        return f"greater_than({self.lower_bound})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return bool(jnp.array_equal(self.lower_bound, other.lower_bound))

    def __hash__(self) -> int:
        return hash(type(self))


class _IntegerInterval(_RealConstraint):
    """Integer values in [low, high]."""

    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = low
        self.high = high

    def _check_real(self, v: Array) -> Array:
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
"""The support of all finite real numbers."""
positive = _Positive()
"""The support of the strictly positive reals, ``(0, inf)``."""
non_negative = _NonNegative()
"""The support of the non-negative reals, ``[0, inf)``."""
non_negative_integer = _NonNegativeInteger()
"""The support of the non-negative integers, ``{0, 1, 2, ...}``."""
boolean = _Boolean()
"""The support of the binary values ``{0, 1}``."""
unit_interval = _UnitInterval()
"""The support of the closed unit interval, ``[0, 1]``."""
simplex = _Simplex()
"""The probability simplex: non-negative vectors that sum to one along the last axis."""
positive_definite = _PositiveDefinite()
"""The support of the positive-definite matrices."""
sphere = _Sphere()
"""The unit sphere: vectors of unit Euclidean norm along the last axis."""


# ---------------------------------------------------------------------------
# Factory functions for parameterized constraints
# ---------------------------------------------------------------------------


def interval(low: ArrayLike, high: ArrayLike) -> _Interval:
    """The support of the closed interval ``[low, high]``."""
    return _Interval(low, high)


def greater_than(lower_bound: ArrayLike) -> _GreaterThan:
    """The support of the values strictly greater than *lower_bound*."""
    return _GreaterThan(lower_bound)


def integer_interval(low: ArrayLike, high: ArrayLike) -> _IntegerInterval:
    """The support of the integers in the closed interval ``[low, high]``."""
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
        return bool(jnp.all(source.low >= target.low)) and bool(jnp.all(source.high <= target.high))
    if isinstance(source, _Interval) and isinstance(target, _Real):
        return True
    if isinstance(source, _GreaterThan) and isinstance(target, _GreaterThan):
        return bool(jnp.all(source.lower_bound >= target.lower_bound))
    if isinstance(source, _GreaterThan) and isinstance(target, _Real):
        return True
    if isinstance(source, (_Positive, _NonNegative)) and isinstance(target, _GreaterThan):
        return True  # (0, inf) or [0, inf) ⊂ (lb, inf) when lb <= 0
    if isinstance(source, _IntegerInterval) and isinstance(target, _IntegerInterval):
        return bool(jnp.all(source.low >= target.low)) and bool(jnp.all(source.high <= target.high))
    if isinstance(source, _IntegerInterval) and isinstance(target, (_NonNegativeInteger, _Real)):
        return True

    # Conservative: allow when we can't determine
    return True
