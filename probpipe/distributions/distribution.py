"""
Core distribution abstractions for ProbPipe.

Provides:
  - ``Distribution``          – Abstract base class following TFP shape semantics.
  - ``TFPDistribution``       – Mixin that delegates to an internal ``tfd.*`` instance.
  - ``EmpiricalDistribution`` – Weighted set of samples.
  - ``Provenance``            – Lightweight lineage tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import math

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..custom_types import Array, ArrayLike, PRNGKey

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


# ---------------------------------------------------------------------------
# Constraints
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
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= self.low) & (v <= self.high)
    def __repr__(self) -> str:
        return f"interval({self.low}, {self.high})"

class _GreaterThan(Constraint):
    """Values strictly greater than a lower bound."""
    def __init__(self, lower_bound: float):
        self.lower_bound = lower_bound
    def check(self, value: ArrayLike) -> Array:
        return jnp.asarray(value) > self.lower_bound
    def __repr__(self) -> str:
        return f"greater_than({self.lower_bound})"

class _IntegerInterval(Constraint):
    """Integer values in [low, high]."""
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
    def check(self, value: ArrayLike) -> Array:
        v = jnp.asarray(value)
        return (v >= self.low) & (v <= self.high) & (v == jnp.floor(v))
    def __repr__(self) -> str:
        return f"integer_interval({self.low}, {self.high})"


# Singleton instances for common constraints
real = _Real()
positive = _Positive()
non_negative = _NonNegative()
non_negative_integer = _NonNegativeInteger()
boolean = _Boolean()
unit_interval = _UnitInterval()
simplex = _Simplex()
positive_definite = _PositiveDefinite()
sphere = _Sphere()

# Factory functions for parameterized constraints
def interval(low: float, high: float) -> _Interval:
    return _Interval(low, high)

def greater_than(lower_bound: float) -> _GreaterThan:
    return _GreaterThan(lower_bound)

def integer_interval(low: int, high: int) -> _IntegerInterval:
    return _IntegerInterval(low, high)


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
    # Identical constraints are always compatible
    if source == target:
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

    # Parameterized constraints: interval/greater_than
    if isinstance(source, _Interval) and isinstance(target, _Interval):
        return source.low >= target.low and source.high <= target.high
    if isinstance(source, _Interval) and isinstance(target, _Real):
        return True
    if isinstance(source, _GreaterThan) and isinstance(target, _GreaterThan):
        return source.lower_bound >= target.lower_bound
    if isinstance(source, _GreaterThan) and isinstance(target, _Real):
        return True
    if isinstance(source, (_Positive, _NonNegative)) and isinstance(target, _GreaterThan):
        return True  # (0, inf) or [0, inf) ⊂ (lb, inf) when lb <= 0
    if isinstance(source, _IntegerInterval) and isinstance(target, _IntegerInterval):
        return source.low >= target.low and source.high <= target.high
    if isinstance(source, _IntegerInterval) and isinstance(target, (_NonNegativeInteger, _Real)):
        return True

    # Conservative: allow when we can't determine
    return True


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Tracks how a distribution was created."""

    operation: str
    parents: tuple[Distribution, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parent_names = ", ".join(
            p.name or type(p).__name__ for p in self.parents
        )
        return f"Provenance({self.operation!r}, parents=[{parent_names}])"

    # -- Serialization -----------------------------------------------------

    def to_dict(self, *, recurse: bool = True) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Parameters
        ----------
        recurse : bool
            If True, recursively serialize parent provenance chains.
            If False, only include parent type/name references.
        """
        parent_dicts = []
        for p in self.parents:
            entry: dict[str, Any] = {
                "type": type(p).__name__,
                "name": p.name,
            }
            if recurse and p.source is not None:
                entry["source"] = p.source.to_dict(recurse=True)
            parent_dicts.append(entry)

        # Filter metadata to JSON-serializable values
        safe_metadata = {}
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                safe_metadata[k] = v
            else:
                safe_metadata[k] = str(v)

        return {
            "operation": self.operation,
            "parents": parent_dicts,
            "metadata": safe_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Parent distributions are not available at deserialization time, so
        ``parents`` will be an empty tuple.  The parent information is
        preserved in the dict under ``"parents"`` for inspection.
        """
        return cls(
            operation=d["operation"],
            parents=(),
            metadata={**d.get("metadata", {}), "_parents_info": d.get("parents", [])},
        )


# ---------------------------------------------------------------------------
# Distribution ABC
# ---------------------------------------------------------------------------

class Distribution(ABC):
    """
    Abstract base for all ProbPipe distributions.

    Shape semantics follow TFP conventions:

    * ``event_shape``  – shape of a single draw (e.g. ``(d,)`` for a
      *d*-dimensional vector distribution).
    * ``batch_shape``  – shape of independent-but-not-identically-distributed
      parameter batches.
    * ``sample(key, sample_shape)`` returns an array of shape
      ``sample_shape + batch_shape + event_shape``.
    """

    # -- core abstract interface ---------------------------------------------

    @property
    @abstractmethod
    def event_shape(self) -> tuple[int, ...]:
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.float32

    @abstractmethod
    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        """Subclass implementation of sampling (called by :meth:`sample`)."""
        ...

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples from this distribution.

        Parameters
        ----------
        key : PRNGKey, optional
            JAX PRNG key.  If ``None``, a key is generated automatically
            from a global counter (convenient for interactive use but not
            reproducible across runs).
        sample_shape : tuple of int
            Leading dimensions for the returned array.
        """
        if key is None:
            key = _auto_key()
        return self._sample(key, sample_shape)

    @abstractmethod
    def log_prob(self, x: ArrayLike) -> Array:
        ...

    def unnormalized_log_prob(self, x: ArrayLike) -> Array:
        """Evaluate the unnormalized log-density at *x*.

        By default this returns ``log_prob(x)``.  Subclasses that only
        know the density up to a normalizing constant (e.g., conditioned
        distributions) should override this method and may raise
        ``NotImplementedError`` from ``log_prob`` instead.
        """
        return self.log_prob(x)

    # -- optional concrete methods ------------------------------------------

    def prob(self, x: ArrayLike) -> Array:
        return jnp.exp(self.log_prob(x))

    def mean(self) -> Array:
        raise NotImplementedError(f"{type(self).__name__}.mean()")

    def variance(self) -> Array:
        raise NotImplementedError(f"{type(self).__name__}.variance()")

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        """The support of this distribution (set of values with non-zero density)."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    # -- naming & provenance ------------------------------------------------

    @property
    def name(self) -> str | None:
        return getattr(self, "_name", None)

    @property
    def source(self) -> Provenance | None:
        return getattr(self, "_source", None)

    def with_source(self, source: Provenance) -> Distribution:
        """Attach provenance to this distribution (write-once)."""
        if getattr(self, "_source", None) is not None:
            raise RuntimeError(
                f"Source already set on {self!r}. "
                "Provenance is write-once; create a new distribution instead."
            )
        self._source = source
        return self

    # -- conversion ---------------------------------------------------------

    @classmethod
    def from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        **kwargs: Any,
    ) -> Distribution:
        """Convert *other* into an instance of *cls*.

        Parameters
        ----------
        other : Distribution
            Source distribution to convert.
        key : PRNGKey, optional
            JAX PRNG key for sampling-based conversion.  If ``None``,
            a key is generated automatically via ``_auto_key()``.
        check_support : bool
            If ``True`` (default), verify the supports are compatible
            before converting. Raises ``ValueError`` on mismatch.
        **kwargs
            Subclass-specific keyword arguments. Common options:

            * ``num_samples`` (int, default 1024) — number of samples
              drawn from *other* for moment-matching or empirical
              estimation. Ignored when *other* is the same class
              (parameters are copied directly).
            * ``total_count`` (int) — required by ``Binomial``,
              ``NegativeBinomial``, and ``Multinomial`` when *other*
              is not the same class.
            * ``name`` (str | None) — name for the resulting
              distribution; defaults to ``other.name``.
        """
        if key is None:
            key = _auto_key()
        if check_support:
            cls._check_support_compatible(other)
        return cls._from_distribution(other, key=key, **kwargs)

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        **kwargs: Any,
    ) -> Distribution:
        """Subclass hook for conversion logic.

        Called by :meth:`from_distribution` after key generation and
        support checking.  *key* is guaranteed to be a valid PRNGKey.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_distribution() is not implemented."
        )

    @classmethod
    def _check_support_compatible(cls, other: Distribution) -> None:
        """Raise ValueError if *other*'s support is incompatible with *cls*."""
        try:
            target_support = cls._default_support()
        except NotImplementedError:
            return  # can't check if target doesn't declare support
        try:
            source_support = other.support
        except NotImplementedError:
            return  # can't check if source doesn't declare support

        if not _supports_compatible(source_support, target_support):
            raise ValueError(
                f"Cannot convert {type(other).__name__} (support={source_support}) "
                f"to {cls.__name__} (support={target_support}). "
                f"Pass check_support=False to override."
            )

    @classmethod
    def _default_support(cls) -> Constraint:
        """Return the default support for this distribution class.

        Override in subclasses. Used by ``_check_support_compatible``
        when no instance is available yet.
        """
        raise NotImplementedError

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        parts.append(f"event_shape={self.event_shape}")
        if self.batch_shape:
            parts.append(f"batch_shape={self.batch_shape}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# TFPDistribution mixin
# ---------------------------------------------------------------------------

class TFPDistribution(Distribution):
    """
    Base class for distributions backed by a ``tfd.Distribution`` instance.

    Subclasses set ``self._tfp_dist`` in ``__init__``.  Sampling,
    ``log_prob``, ``mean``, and ``variance`` delegate to TFP.
    """

    _tfp_dist: tfd.Distribution

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.batch_shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self._tfp_dist.dtype

    # -- sampling & density -------------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.log_prob(jnp.asarray(x))

    def unnormalized_log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.unnormalized_log_prob(jnp.asarray(x))

    def prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.prob(jnp.asarray(x))

    def mean(self) -> Array:
        return self._tfp_dist.mean()

    def variance(self) -> Array:
        return self._tfp_dist.variance()


# ---------------------------------------------------------------------------
# EmpiricalDistribution
# ---------------------------------------------------------------------------

class EmpiricalDistribution(Distribution):
    """
    Weighted empirical distribution over a finite set of samples.

    Parameters
    ----------
    samples : array-like, shape ``(n, *event_shape)``
        The support points.  The leading axis is the sample axis.
    weights : array-like, shape ``(n,)``, optional
        Non-negative weights (normalised internally).  Mutually exclusive
        with *log_weights*.  When neither is given the distribution is
        uniform.
    log_weights : array-like, shape ``(n,)``, optional
        Log-unnormalised weights.  Preferred when weights span many orders
        of magnitude (e.g. importance sampling).  Normalised internally via
        ``jax.nn.softmax``.  Mutually exclusive with *weights*.
    name : str, optional
        An optional name for provenance / JointDistribution integration.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        samples = jnp.asarray(samples, dtype=jnp.float32)
        if samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension (the sample axis).")

        n = samples.shape[0]

        if weights is not None and log_weights is not None:
            raise ValueError(
                "Provide either weights or log_weights, not both."
            )

        if weights is not None:
            weights = jnp.asarray(weights, dtype=jnp.float32)
            if weights.shape != (n,):
                raise ValueError(
                    f"weights shape {weights.shape} does not match "
                    f"number of samples {n}."
                )
            if jnp.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            total = jnp.sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            self._log_weights = jnp.log(weights)
            self._is_uniform = False
        elif log_weights is not None:
            log_weights = jnp.asarray(log_weights, dtype=jnp.float32)
            if log_weights.shape != (n,):
                raise ValueError(
                    f"log_weights shape {log_weights.shape} does not match "
                    f"number of samples {n}."
                )
            self._log_weights = log_weights
            self._is_uniform = False
        else:
            self._log_weights = None
            self._is_uniform = True

        self._samples = samples
        self._weights_cache: Array | None = None
        self._name = name

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        return self._samples.shape[0]

    @property
    def dim(self) -> int:
        """Flat dimensionality of each sample (product of event_shape, or 1 for scalars)."""
        return max(1, int(math.prod(self._samples.shape[1:])))

    @property
    def samples(self) -> Array:
        return self._samples

    @property
    def is_uniform(self) -> bool:
        """True when all samples have equal weight."""
        return self._is_uniform

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        if self._is_uniform:
            return jnp.ones(self.n, dtype=jnp.float32) / self.n
        if self._weights_cache is None:
            self._weights_cache = jax.nn.softmax(self._log_weights)
        return self._weights_cache

    @property
    def log_weights(self) -> Array | None:
        """Normalised log-weights, shape ``(n,)``.  ``None`` when uniform."""
        if self._is_uniform:
            return None
        return self._log_weights - jax.scipy.special.logsumexp(self._log_weights)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._samples.shape[1:]

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    @property
    def support(self) -> Constraint:
        return real  # empirical samples can be any real values

    # -- sampling -----------------------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        n_draws = int(math.prod(sample_shape)) if sample_shape else 1
        if self._is_uniform:
            indices = jax.random.randint(key, shape=(n_draws,), minval=0, maxval=self.n)
        else:
            indices = jax.random.choice(
                key, self.n, shape=(n_draws,), p=self.weights, replace=True,
            )
        draws = self._samples[indices]
        if sample_shape:
            return draws.reshape(sample_shape + self.event_shape)
        return draws.squeeze(axis=0)

    # -- density (Gaussian approximation) -----------------------------------

    def log_prob(self, x: ArrayLike) -> Array:
        """Gaussian-approximation log-density."""
        x = jnp.asarray(x)
        mu = self.mean()
        var = self.variance()
        # Diagonal Gaussian approx; clamp variance to avoid log(0)
        var = jnp.maximum(var, 1e-12)
        log_norm = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
        diff = x - mu
        return log_norm - 0.5 * jnp.sum(diff**2 / var, axis=-1)

    # -- moments ------------------------------------------------------------

    def mean(self) -> Array:
        if self._is_uniform:
            return jnp.mean(self._samples, axis=0)
        return jnp.einsum("n,n...->...", self.weights, self._samples)

    def variance(self) -> Array:
        mu = self.mean()
        diff = self._samples - mu
        if self._is_uniform:
            return jnp.mean(diff**2, axis=0)
        return jnp.einsum("n,n...->...", self.weights, diff**2)

    def cov(self) -> Array:
        """Weighted sample covariance matrix, shape ``(d, d)``."""
        mu = self.mean()
        # Flatten to 2D: (n, d)
        flat_samples = self._samples.reshape(self.n, -1)
        diff = flat_samples - mu.reshape(-1)
        if self._is_uniform:
            return jnp.einsum("ni,nj->ij", diff, diff) / self.n
        return jnp.einsum("ni,nj,n->ij", diff, diff, self.weights)

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: Distribution,
        *,
        key: PRNGKey,
        name: str | None = None,
        **kwargs: Any,
    ) -> EmpiricalDistribution:
        """Create an empirical approximation by sampling from *other*.

        Keyword Args
        -------------
        num_samples : int
            Number of samples to draw (default 1024).
        name : str | None
            Name for the result; defaults to ``other.name``.
        """
        num_samples = kwargs.pop("num_samples", 1024)
        samples = other.sample(key, sample_shape=(num_samples,))
        ed = cls(samples, name=name or other.name)
        ed.with_source(Provenance("from_distribution", parents=(other,)))
        return ed
