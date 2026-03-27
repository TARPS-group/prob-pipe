"""
Core distribution abstractions for ProbPipe.

Provides:
  - ``Distribution``                – Generic base class parameterized by value type T.
  - ``PyTreeArrayDistribution``     – Pytree-of-arrays layer with batch/event shape
                                      semantics, flatten/unflatten, and flat-view interop.
  - ``ArrayDistribution``           – Single-array specialization (T = Array) with TFP
                                      shape conventions. All standard distributions
                                      (Normal, Gamma, MVN, ...) inherit from this.
  - ``FlattenedView``               – Wraps any ``PyTreeArrayDistribution`` as a flat
                                      ``ArrayDistribution`` for algorithm interoperability.
  - ``TFPDistribution``             – Mixin that delegates to an internal ``tfd.*`` instance.
  - ``EmpiricalDistribution``       – Weighted set of samples.
  - ``BootstrapDistribution``       – MC error tracking via bootstrap resampling.
  - ``Provenance``                  – Lightweight lineage tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
import functools

from .._utils import prod

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..custom_types import Array, ArrayLike, PRNGKey

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

T = TypeVar('T')

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_EVALUATIONS: int = 1024
"""Default number of function evaluations for sample-based expectations."""

RETURN_APPROX_DIST: bool = True
"""When True, approximate expectations return a BootstrapDistribution
capturing MC error instead of a plain array."""


def set_default_num_evaluations(n: int) -> None:
    """Set the global default for ``expectation()`` on infinite-support distributions."""
    global DEFAULT_NUM_EVALUATIONS
    if n < 1:
        raise ValueError("num_evaluations must be at least 1")
    DEFAULT_NUM_EVALUATIONS = n


def set_return_approx_dist(value: bool) -> None:
    """Set whether approximate expectations return error-tracking distributions."""
    global RETURN_APPROX_DIST
    RETURN_APPROX_DIST = bool(value)


# ---------------------------------------------------------------------------
# @monte_carlo decorator
# ---------------------------------------------------------------------------


def monte_carlo(method):
    """Decorator for methods that compute expectations via Monte Carlo.

    The decorated method should return a callable ``f`` such that the
    desired quantity is ``E[f(X)]``.  The decorator adds
    ``num_evaluations``, ``return_dist``, and ``key`` keyword arguments,
    delegates to ``self.expectation(f, ...)``, and returns either an
    ``Array`` or a ``BootstrapDistribution`` depending on settings.

    Example::

        class MyDist(ArrayDistribution):
            @monte_carlo
            def mean(self):
                return lambda x: x
    """

    @functools.wraps(method)
    def wrapper(
        self,
        *args,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
        **kwargs,
    ):
        f = method(self, *args, **kwargs)
        return self.expectation(
            f, key=key, num_evaluations=num_evaluations, return_dist=return_dist,
        )

    return wrapper


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
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
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
# Distribution[T] — generic base class
# ---------------------------------------------------------------------------

class Distribution(Generic[T], ABC):
    """
    Abstract base for all ProbPipe distributions, parameterized by
    value type ``T``.

    Commits to a sampling contract and optional log-density.  Shape
    semantics, batch/event conventions, and domain-specific methods
    live in specialized subclasses (e.g., ``ArrayDistribution``).

    The abstract primitive is ``_sample(key)``, which draws a single
    sample of type ``T``.  The public ``sample(key, sample_shape)``
    method handles batching via ``jax.vmap`` by default; subclasses
    may override for efficiency.
    """

    # -- core abstract interface ---------------------------------------------

    @abstractmethod
    def _sample(self, key: PRNGKey) -> T:
        """Draw a single sample. Subclasses implement this."""
        ...

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> T:
        """Draw sample(s) from this distribution.

        Parameters
        ----------
        key : PRNGKey, optional
            JAX PRNG key.  If ``None``, a key is generated automatically
            from a global counter (convenient for interactive use but not
            reproducible across runs).
        sample_shape : tuple of int
            Number of independent draws.  The meaning depends on ``T``:
            for arrays, prepends dimensions; for pytrees, prepends
            dimensions to every leaf; for distributions, enriches
            batch shape.

        Returns
        -------
        T
            A single sample when ``sample_shape == ()``, or a batched
            representation when ``sample_shape`` is non-empty.
        """
        if key is None:
            key = _auto_key()
        if sample_shape == ():
            return self._sample(key)
        n = prod(sample_shape)
        keys = jax.random.split(key, n)
        flat_samples = jax.vmap(self._sample)(keys)
        return jax.tree.map(
            lambda x: x.reshape(*sample_shape, *x.shape[1:]),
            flat_samples,
        )

    # -- orchestration hints (protocol defaults) -----------------------------

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- log-density (optional) ----------------------------------------------

    def log_prob(self, value: T) -> Array:
        """Log-density of *value*.  Optional; raises by default.

        Subclasses that define a density should override this method.
        Takes one value of type T in, returns a scalar (or batch of
        scalars) out.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support log_prob"
        )

    # Protocol alias — will become the canonical name in a future PR.
    def _log_prob(self, value: T) -> Array:
        return self.log_prob(value)

    def unnormalized_log_prob(self, value: T) -> Array:
        """Evaluate the unnormalized log-density at *value*.

        By default this returns ``log_prob(value)``.  Subclasses that only
        know the density up to a normalizing constant (e.g., conditioned
        distributions) should override this method and may raise
        ``NotImplementedError`` from ``log_prob`` instead.
        """
        return self.log_prob(value)

    # Protocol alias
    def _unnormalized_log_prob(self, value: T) -> Array:
        return self.unnormalized_log_prob(value)

    # Protocol method — exp of the log form
    def _unnormalized_prob(self, value: T) -> Array:
        return jnp.exp(self._unnormalized_log_prob(value))

    # Protocol method — exp of the log form
    def _prob(self, value: T) -> Array:
        return jnp.exp(self._log_prob(value))

    # -- expectations ---------------------------------------------------------

    def expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        """Estimate ``E[f(X)]`` where ``X ~ self``.

        Parameters
        ----------
        f : callable
            Function mapping a single sample to an array (or pytree of
            arrays).
        key : PRNGKey, optional
            JAX PRNG key for sampling.  Auto-generated if ``None``.
        num_evaluations : int, optional
            Number of samples to draw.  If ``None``, uses
            ``DEFAULT_NUM_EVALUATIONS``.  Subclasses with finite support
            override this to compute exactly when ``None``.
        return_dist : bool, optional
            If ``True``, return a ``BootstrapDistribution`` capturing
            estimation uncertainty.  If ``False``, return a plain array.
            If ``None``, use the global ``RETURN_APPROX_DIST`` setting.
        """
        n = num_evaluations if num_evaluations is not None else DEFAULT_NUM_EVALUATIONS
        if key is None:
            key = _auto_key()
        samples = self.sample(key, sample_shape=(n,))
        evals = jax.vmap(f)(samples)

        rd = return_dist if return_dist is not None else RETURN_APPROX_DIST
        if rd:
            return BootstrapDistribution(evals, name=f"E[f(X)]")
        return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)

    # -- approximation tracking ---------------------------------------------

    @property
    def is_approximate(self) -> bool:
        """Whether this distribution is an approximation (e.g., from sampling or MCMC)."""
        return getattr(self, "_approximate", False)

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
        other: "Distribution",
        *,
        key: PRNGKey | None = None,
        check_support: bool = True,
        **kwargs: Any,
    ) -> "Distribution":
        """Convert *other* into an instance of *cls*.

        Delegates to the global converter registry.  See
        :mod:`probpipe.converters` for details on how conversions are
        resolved.

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
            Subclass-specific keyword arguments.  Common options for
            ``ArrayDistribution`` targets:

            * ``num_samples`` (int, default 1024) -- number of samples
              drawn from *other* for moment-matching or empirical
              estimation. Ignored when *other* is the same class
              (parameters are copied directly).
            * ``total_count`` (int) -- required by ``Binomial``,
              ``NegativeBinomial``, and ``Multinomial`` when *other*
              is not the same class.
            * ``name`` (str | None) -- name for the resulting
              distribution; defaults to ``other.name``.
        """
        from ..converters import converter_registry
        return converter_registry.convert(
            other, cls, key=key, check_support=check_support, **kwargs
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# PyTreeArrayDistribution[T] — pytree of arrays with shape semantics
# ---------------------------------------------------------------------------

class PyTreeArrayDistribution(Distribution[T]):
    """
    Distribution over a pytree of arrays with TFP-style shape semantics.

    All leaves are JAX arrays. ``batch_shape`` is shared across all leaves.
    Each leaf has its own ``event_shape``. The full shape identity is:
    ``treedef + batch_shape + per-leaf event_shapes``.

    This is the workhorse layer for distributions over structured values
    (e.g., dicts of arrays for joint parameter distributions). Single-array
    distributions use the ``ArrayDistribution`` subclass.
    """

    # -- pytree structure ---------------------------------------------------

    @property
    @abstractmethod
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """The pytree structure of a single sample."""
        ...

    # -- shape properties ---------------------------------------------------

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape, shared across all leaves."""
        ...

    @property
    @abstractmethod
    def event_shapes(self) -> T:
        """Per-leaf event shapes.

        Returns a pytree with the same structure as T, with
        ``tuple[int, ...]`` at each leaf position.
        """
        ...

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Event shapes as a flat list in canonical leaf order."""
        return self.treedef.flatten_up_to(self.event_shapes)

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one event (analogous to ``numpy.ndarray.size``).

        Equal to the sum of ``prod(s)`` over all leaf event shapes.
        """
        return sum(prod(s) for s in self.flat_event_shapes)

    # -- flatten / unflatten ------------------------------------------------

    def flatten_value(self, value: T) -> Array:
        """Flatten the event dimensions of a pytree value into a single trailing axis.

        Each leaf's event dimensions are raveled and the results are
        concatenated in canonical leaf order.  All leading dimensions
        (``sample_shape``, ``batch_shape``, or both) are preserved.

        Shape contract for each leaf::

            (*sample_shape, *batch_shape, *event_shape)
            → (*sample_shape, *batch_shape, prod(event_shape))

        After concatenating across leaves the result has shape
        ``(*sample_shape, *batch_shape, event_size)``.
        """
        leaves = jax.tree.leaves(value)
        event_shapes = self.flat_event_shapes
        flat_leaves = []
        for leaf, es in zip(leaves, event_shapes):
            leaf = jnp.asarray(leaf)
            n_event = prod(es)
            # Reshape: (*batch_dims, *event_shape) -> (*batch_dims, n_event)
            n_event_dims = len(es)
            batch_dims = leaf.shape[:leaf.ndim - n_event_dims] if n_event_dims else leaf.shape
            flat_leaves.append(leaf.reshape(*batch_dims, n_event))
        return jnp.concatenate(flat_leaves, axis=-1)

    def unflatten_value(self, flat: Array) -> T:
        """Unflatten a flat array back to the pytree structure.

        Reverses :meth:`flatten_value`.  All dimensions preceding the
        final ``event_size`` axis are preserved (whether they represent
        ``sample_shape``, ``batch_shape``, or both).

        Parameters
        ----------
        flat : Array
            Array of shape ``(*sample_shape, *batch_shape, event_size)``.

        Returns
        -------
        T
            Pytree with each leaf reshaped to
            ``(*sample_shape, *batch_shape, *event_shape)``.
        """
        event_shapes = self.flat_event_shapes
        batch_dims = flat.shape[:-1]
        leaves = []
        offset = 0
        for es in event_shapes:
            n = prod(es)
            chunk = flat[..., offset:offset + n]
            leaves.append(chunk.reshape(*batch_dims, *es))
            offset += n
        return jax.tree.unflatten(self.treedef, leaves)

    def as_flat_distribution(self) -> ArrayDistribution:
        """View this distribution as a flat ``ArrayDistribution``.

        Returns an ``ArrayDistribution`` with ``event_shape=(event_size,)``.
        Enables interoperability with algorithms that expect flat vectors
        (MCMC, optimizers, VI methods).
        """
        return FlattenedView(self)

    # -- moments ------------------------------------------------------------

    @monte_carlo
    def mean(self):
        """Mean of this distribution.

        Subclasses with exact means (e.g., ``TFPDistribution``) override
        this.  The default implementation uses ``expectation(lambda x: x)``.
        """
        return lambda x: x

    @monte_carlo
    def variance(self):
        """Variance of this distribution.

        Subclasses with exact variance (e.g., ``TFPDistribution``)
        override this.  The default implementation uses ``expectation``.
        """
        mu = self.mean(return_dist=False)
        return lambda x: jax.tree.map(lambda xi, mi: (xi - mi) ** 2, x, mu)

    # -- supports (pytree of per-leaf constraints) ---------------------------

    @property
    def supports(self) -> T:
        """Per-leaf constraints. Default returns ``real`` for every leaf."""
        return jax.tree.map(lambda _: real, self.event_shapes)


# ---------------------------------------------------------------------------
# ArrayDistribution — distribution over arrays (TFP shape semantics)
# ---------------------------------------------------------------------------

class ArrayDistribution(PyTreeArrayDistribution[Array]):
    """
    Distribution over a single array with TFP-style shape semantics.

    Shape semantics follow TFP conventions:

    * ``event_shape``  -- shape of a single draw (e.g. ``(d,)`` for a
      *d*-dimensional vector distribution).
    * ``batch_shape``  -- shape of independent-but-not-identically-distributed
      parameter batches.
    * ``sample(key, sample_shape)`` returns an array of shape
      ``sample_shape + batch_shape + event_shape``.

    Standard distributions (Normal, Gamma, Poisson, etc.) inherit from this class.
    """

    # -- shape properties ---------------------------------------------------

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

    # -- PyTreeArrayDistribution interface (trivial single-leaf) -------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Treedef for a single array (one-leaf pytree)."""
        return jax.tree.structure(None)

    @property
    def event_shapes(self) -> Array:
        """Single event shape, wrapped to satisfy the pytree interface."""
        return self.event_shape

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """Single-leaf: just the one event_shape."""
        return [self.event_shape]

    @property
    def event_size(self) -> int:
        """Total flat dimensionality."""
        es = self.event_shape
        return prod(es)

    def flatten_value(self, value: ArrayLike) -> Array:
        """Flatten event dimensions into a single trailing axis.

        Leading dimensions (``sample_shape``, ``batch_shape``) are
        preserved::

            (*sample_shape, *batch_shape, *event_shape)
            → (*sample_shape, *batch_shape, prod(event_shape))
        """
        value = jnp.asarray(value)
        es = self.event_shape
        n_event = prod(es)
        if not es:
            # scalar event: just add a trailing dim
            return value[..., None]
        n_batch = value.ndim - len(es)
        batch_dims = value.shape[:n_batch]
        return value.reshape(*batch_dims, n_event)

    def unflatten_value(self, flat: ArrayLike) -> Array:
        """Unflatten a flat trailing axis back to event dimensions.

        Leading dimensions (``sample_shape``, ``batch_shape``) are
        preserved::

            (*sample_shape, *batch_shape, event_size)
            → (*sample_shape, *batch_shape, *event_shape)
        """
        flat = jnp.asarray(flat)
        es = self.event_shape
        if not es:
            return flat[..., 0]
        batch_dims = flat.shape[:-1]
        return flat.reshape(*batch_dims, *es)

    # -- array-specific convenience methods ----------------------------------

    def prob(self, x: ArrayLike) -> Array:
        return jnp.exp(self.log_prob(x))

    # Protocol alias
    def _prob(self, x: ArrayLike) -> Array:
        return self.prob(x)

    @monte_carlo
    def cov(self):
        """Covariance matrix of this distribution.

        The base implementation falls back to Monte Carlo.
        """
        mu = self.mean(return_dist=False)

        def _outer_diff(x):
            d = x.ravel() - mu.ravel()
            return jnp.outer(d, d)

        return _outer_diff

    # -- support ------------------------------------------------------------

    @property
    def support(self) -> Constraint:
        """The support of this distribution (set of values with non-zero density)."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    @property
    def supports(self) -> Constraint:
        """For ArrayDistribution, supports is just the singular support."""
        return self.support

    # -- support compatibility (for conversion) ------------------------------

    @classmethod
    def _check_support_compatible(cls, other: ArrayDistribution) -> None:
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
# FlattenedView — wrap a PyTreeArrayDistribution as an ArrayDistribution
# ---------------------------------------------------------------------------

class FlattenedView(ArrayDistribution):
    """Wraps a ``PyTreeArrayDistribution`` as a flat ``ArrayDistribution``.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    for ``ArrayDistribution`` works with ``PyTreeArrayDistribution`` via
    ``dist.as_flat_distribution()``.
    """

    def __init__(self, base: PyTreeArrayDistribution):
        self._base = base

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._base.event_size,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._base.batch_shape

    def _sample(self, key: PRNGKey) -> Array:
        pytree_sample = self._base._sample(key)
        return self._base.flatten_value(pytree_sample)

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if key is None:
            key = _auto_key()
        pytree_samples = self._base.sample(key, sample_shape)
        return self._base.flatten_value(pytree_samples)

    def log_prob(self, x: ArrayLike) -> Array:
        x = jnp.asarray(x)
        value = self._base.unflatten_value(x)
        return self._base.log_prob(value)

    @property
    def support(self) -> Constraint:
        return real

    @property
    def base_distribution(self) -> PyTreeArrayDistribution:
        """The underlying pytree distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return self._base.unflatten_value(jnp.asarray(flat_sample))

    def __repr__(self) -> str:
        return (
            f"FlattenedView(base={type(self._base).__name__}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# TFPDistribution mixin
# ---------------------------------------------------------------------------

class TFPDistribution(ArrayDistribution):
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

    def _sample(self, key: PRNGKey) -> Array:
        """Draw a single sample from the TFP distribution."""
        return self._tfp_dist.sample(seed=key)

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples using TFP's efficient batched sampling."""
        if key is None:
            key = _auto_key()
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.log_prob(jnp.asarray(x))

    def unnormalized_log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.unnormalized_log_prob(jnp.asarray(x))

    def prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.prob(jnp.asarray(x))

    def mean(self, **kwargs) -> Array:
        return self._tfp_dist.mean()

    # Protocol aliases for exact moments
    def _mean(self) -> Array:
        return self._tfp_dist.mean()

    def variance(self, **kwargs) -> Array:
        return self._tfp_dist.variance()

    def _variance(self) -> Array:
        return self._tfp_dist.variance()


# ---------------------------------------------------------------------------
# EmpiricalDistribution
# ---------------------------------------------------------------------------

class EmpiricalDistribution(ArrayDistribution):
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
        self._approximate = True

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        return self._samples.shape[0]

    @property
    def dim(self) -> int:
        """Flat dimensionality of each sample (product of event_shape, or 1 for scalars)."""
        return max(1, prod(self._samples.shape[1:]))

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

    def _sample(self, key: PRNGKey) -> Array:
        """Draw a single sample (with replacement according to weights)."""
        if self._is_uniform:
            idx = jax.random.randint(key, shape=(), minval=0, maxval=self.n)
        else:
            idx = jax.random.choice(key, self.n, p=self.weights)
        return self._samples[idx]

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples using efficient batched resampling."""
        if key is None:
            key = _auto_key()
        if sample_shape == ():
            return self._sample(key)
        n_draws = prod(sample_shape)
        if self._is_uniform:
            indices = jax.random.randint(key, shape=(n_draws,), minval=0, maxval=self.n)
        else:
            indices = jax.random.choice(
                key, self.n, shape=(n_draws,), p=self.weights, replace=True,
            )
        draws = self._samples[indices]
        return draws.reshape(sample_shape + self.event_shape)

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

    def mean(self, **kwargs) -> Array:
        if self._is_uniform:
            return jnp.mean(self._samples, axis=0)
        return jnp.einsum("n,n...->...", self.weights, self._samples)

    # Protocol aliases for exact moments
    def _mean(self) -> Array:
        return self.mean()

    def variance(self, **kwargs) -> Array:
        mu = self.mean()
        diff = self._samples - mu
        if self._is_uniform:
            return jnp.mean(diff**2, axis=0)
        return jnp.einsum("n,n...->...", self.weights, diff**2)

    def _variance(self) -> Array:
        return self.variance()

    def cov(self, **kwargs) -> Array:
        """Weighted sample covariance matrix, shape ``(d, d)``."""
        mu = self.mean()
        # Flatten to 2D: (n, d)
        flat_samples = self._samples.reshape(self.n, -1)
        diff = flat_samples - mu.reshape(-1)
        if self._is_uniform:
            return jnp.einsum("ni,nj->ij", diff, diff) / self.n
        return jnp.einsum("ni,nj,n->ij", diff, diff, self.weights)

    def _cov(self) -> Array:
        return self.cov()

    def expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array | ArrayDistribution:
        """Compute ``E[f(X)]`` exactly over the empirical support.

        When ``num_evaluations`` is ``None``, the expectation is computed
        exactly as a weighted sum over all support points (always returns
        ``Array``).  When ``num_evaluations`` is specified and smaller
        than ``self.n``, a random subsample is used and ``return_dist``
        controls whether a ``BootstrapDistribution`` is returned.
        """
        if num_evaluations is not None and num_evaluations < self.n:
            # Subsample — this is approximate
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(key, self.n, shape=(num_evaluations,), replace=False)
            sub_samples = self._samples[idx]
            f_vals = jax.vmap(f)(sub_samples)

            rd = return_dist if return_dist is not None else RETURN_APPROX_DIST
            if rd:
                sub_w = None
                if not self._is_uniform:
                    sub_w = self.weights[idx]
                    sub_w = sub_w / jnp.sum(sub_w)
                return BootstrapDistribution(f_vals, weights=sub_w)

            if self._is_uniform:
                return jnp.mean(f_vals, axis=0)
            sub_w = self.weights[idx]
            sub_w = sub_w / jnp.sum(sub_w)
            return jnp.einsum("n,n...->...", sub_w, f_vals)

        # Exact: evaluate f on all support points — always returns Array
        f_vals = jax.vmap(f)(self._samples)
        if self._is_uniform:
            return jnp.mean(f_vals, axis=0)
        return jnp.einsum("n,n...->...", self.weights, f_vals)

    # -- conversion ---------------------------------------------------------

    @classmethod
    def _from_distribution(
        cls,
        other: ArrayDistribution,
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


# ---------------------------------------------------------------------------
# BootstrapDistribution
# ---------------------------------------------------------------------------

class BootstrapDistribution(ArrayDistribution):
    """Distribution over bootstrap-resampled means of a statistic.

    Given *n* evaluations ``f(x_1), ..., f(x_n)`` where ``x_i ~ P``,
    this represents the sampling distribution of the sample mean
    ``(1/n) sum f(x_i)``, capturing Monte Carlo error.

    Parameters
    ----------
    evaluations : array-like, shape ``(n, *stat_shape)``
        The individual ``f(x_i)`` values.
    weights : array-like, shape ``(n,)``, optional
        Non-negative weights (normalised internally).  When ``None``,
        uniform weights are used.
    name : str, optional
        Distribution name.
    """

    def __init__(
        self,
        evaluations: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        name: str | None = None,
    ):
        self._evaluations = jnp.asarray(evaluations, dtype=jnp.float32)
        if self._evaluations.ndim == 0:
            raise ValueError("evaluations must have at least 1 dimension.")
        self._n = self._evaluations.shape[0]

        if weights is not None:
            w = jnp.asarray(weights, dtype=jnp.float32)
            self._weights = w / jnp.sum(w)
        else:
            self._weights = None

        self._name = name
        self._approximate = True

    @property
    def n(self) -> int:
        """Number of function evaluations."""
        return self._n

    @property
    def evaluations(self) -> Array:
        return self._evaluations

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._evaluations.shape[1:]

    def mean(self) -> Array:
        """Point estimate: (weighted) mean of evaluations."""
        if self._weights is None:
            return jnp.mean(self._evaluations, axis=0)
        return jnp.einsum("n,n...->...", self._weights, self._evaluations)

    # Protocol alias
    _mean = mean

    def variance(self) -> Array:
        """Variance of the sampling distribution (approx Var[f(X)] / n_eff)."""
        mu = self.mean()
        diff = self._evaluations - mu
        if self._weights is None:
            sample_var = jnp.mean(diff ** 2, axis=0)
            return sample_var / self._n
        # Weighted variance / effective sample size
        sample_var = jnp.einsum("n,n...->...", self._weights, diff ** 2)
        n_eff = 1.0 / jnp.sum(self._weights ** 2)
        return sample_var / n_eff

    # Protocol alias
    _variance = variance

    def _sample(self, key: PRNGKey) -> Array:
        """Draw a single bootstrap resample of the mean."""
        if self._weights is None:
            idx = jax.random.choice(key, self._n, shape=(self._n,), replace=True)
            return jnp.mean(self._evaluations[idx], axis=0)
        else:
            idx = jax.random.choice(
                key, self._n, shape=(self._n,), replace=True, p=self._weights
            )
            return jnp.mean(self._evaluations[idx], axis=0)

    def sample(
        self,
        key: PRNGKey | None = None,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw bootstrap resamples of the mean."""
        if key is None:
            key = _auto_key()
        if sample_shape == ():
            return self._sample(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)

        def _one_resample(k):
            if self._weights is None:
                idx = jax.random.choice(k, self._n, shape=(self._n,), replace=True)
                return jnp.mean(self._evaluations[idx], axis=0)
            else:
                idx = jax.random.choice(
                    k, self._n, shape=(self._n,), replace=True, p=self._weights
                )
                return jnp.mean(self._evaluations[idx], axis=0)

        results = jax.vmap(_one_resample)(keys)
        return results.reshape(sample_shape + self.event_shape)

    def log_prob(self, x: ArrayLike) -> Array:
        """Log-density via Gaussian approximation (mean +/- SE)."""
        x = jnp.asarray(x)
        mu = self.mean()
        var = jnp.maximum(self.variance(), 1e-12)
        # Diagonal Gaussian
        return -0.5 * jnp.sum(((x - mu) ** 2) / var + jnp.log(2 * jnp.pi * var))

    @property
    def support(self) -> Constraint:
        return real

    def __repr__(self) -> str:
        return f"BootstrapDistribution(n={self._n}, event_shape={self.event_shape})"
