"""Centralized weight utilities for ProbPipe.

Provides:
  - ``Weights``               – Normalized probability weights (JAX-array-compatible).
  - ``weighted_mean()``       – Weighted mean over leading axis.
  - ``weighted_variance()``   – Weighted variance over leading axis.
  - ``weighted_covariance()`` – Weighted covariance matrix.
  - ``weighted_choice()``     – Draw weighted random indices.
  - ``validate_and_normalize_log_weights()`` – Validate raw user input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .custom_types import Array, ArrayLike, PRNGKey


# ---------------------------------------------------------------------------
# Pure utility functions (internal workhorses)
# ---------------------------------------------------------------------------

def validate_and_normalize_log_weights(
    n: int,
    weights: ArrayLike | None = None,
    *,
    log_weights: ArrayLike | None = None,
) -> tuple[Array | None, bool]:
    """Validate and convert user-supplied weights to log-unnormalized form.

    Parameters
    ----------
    n : int
        Expected number of weight entries.
    weights : array-like, optional
        Non-negative weights.  Mutually exclusive with *log_weights*.
    log_weights : array-like, optional
        Log-unnormalized weights.  Mutually exclusive with *weights*.

    Returns
    -------
    log_w : Array or None
        Log-unnormalized weights, or ``None`` when uniform.
    is_uniform : bool
        ``True`` when neither *weights* nor *log_weights* was provided.
    """
    if weights is not None and log_weights is not None:
        raise ValueError("Provide either weights or log_weights, not both.")

    if weights is not None:
        weights = jnp.asarray(weights, dtype=jnp.float32)
        if weights.shape != (n,):
            raise ValueError(
                f"weights shape {weights.shape} does not match "
                f"number of items {n}."
            )
        if jnp.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        total = jnp.sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        return jnp.log(weights), False

    if log_weights is not None:
        log_weights = jnp.asarray(log_weights, dtype=jnp.float32)
        if log_weights.shape != (n,):
            raise ValueError(
                f"log_weights shape {log_weights.shape} does not match "
                f"number of items {n}."
            )
        return log_weights, False

    return None, True


def normalize_weights(log_weights: Array) -> Array:
    """Convert log-unnormalized weights to normalized probabilities.

    Uses ``jax.nn.softmax`` for numerical stability.
    """
    return jax.nn.softmax(log_weights)


def normalized_log_weights(log_weights: Array) -> Array:
    """Normalize log-weights in log-space (subtract logsumexp)."""
    return log_weights - jax.scipy.special.logsumexp(log_weights)


def uniform_weights(n: int) -> Array:
    """Return uniform weight vector of length *n*."""
    return jnp.ones(n, dtype=jnp.float32) / n


def weighted_mean(weights: Array | None, values: Array) -> Array:
    """Weighted mean over the leading axis of *values*.

    Parameters
    ----------
    weights : Array or None
        Normalized weights of shape ``(n,)``.  ``None`` for uniform.
    values : Array
        Values of shape ``(n, ...)``.
    """
    if weights is None:
        return jnp.mean(values, axis=0)
    return jnp.einsum("n,n...->...", weights, values)


def weighted_variance(
    weights: Array | None,
    values: Array,
    mean: Array | None = None,
) -> Array:
    """Weighted variance over the leading axis of *values*.

    Parameters
    ----------
    weights : Array or None
        Normalized weights of shape ``(n,)``.  ``None`` for uniform.
    values : Array
        Values of shape ``(n, ...)``.
    mean : Array, optional
        Pre-computed weighted mean.  Computed if ``None``.
    """
    if mean is None:
        mean = weighted_mean(weights, values)
    diff = values - mean
    return weighted_mean(weights, diff ** 2)


def weighted_covariance(
    weights: Array | None,
    values: Array,
    mean: Array | None = None,
) -> Array:
    """Weighted covariance matrix over the leading axis of *values*.

    Parameters
    ----------
    weights : Array or None
        Normalized weights of shape ``(n,)``.  ``None`` for uniform.
    values : Array
        Values of shape ``(n, ...)``.  Flattened to ``(n, d)`` internally.
    mean : Array, optional
        Pre-computed weighted mean.  Computed if ``None``.

    Returns
    -------
    Array, shape ``(d, d)``
        Weighted covariance matrix where ``d = prod(values.shape[1:])``.
    """
    n = values.shape[0]
    flat = values.reshape(n, -1)
    if mean is None:
        mean = weighted_mean(weights, values)
    diff = flat - mean.reshape(-1)
    if weights is None:
        return jnp.einsum("ni,nj->ij", diff, diff) / n
    return jnp.einsum("ni,nj,n->ij", diff, diff, weights)


def weighted_choice(
    key: PRNGKey,
    n: int,
    *,
    weights: Array | None = None,
    shape: tuple[int, ...] = (),
) -> Array:
    """Draw random indices with optional weighting.

    Parameters
    ----------
    key : PRNGKey
        JAX PRNG key.
    n : int
        Number of items to choose from (indices ``0..n-1``).
    weights : Array or None
        Normalized weights of shape ``(n,)``.  ``None`` for uniform.
    shape : tuple of int
        Output shape of index array.

    Returns
    -------
    Array
        Integer index array of the given *shape*.
    """
    if not shape:
        shape = (1,)
        squeeze = True
    else:
        squeeze = False

    if weights is None:
        indices = jax.random.randint(key, shape=shape, minval=0, maxval=n)
    else:
        indices = jax.random.choice(
            key, n, shape=shape, p=weights, replace=True,
        )

    if squeeze:
        return indices[0]
    return indices


# ---------------------------------------------------------------------------
# Weights class
# ---------------------------------------------------------------------------

class Weights:
    """Normalized probability weights over *n* items.

    ``Weights`` stores log-unnormalized weights internally for numerical
    stability and provides lazy-cached access to normalized weights and
    log-weights.

    It implements the JAX array protocol (``__jax_array__``), so a
    ``Weights`` object can be passed directly to any JAX operation that
    expects an array — it will automatically convert to its **normalized**
    weight vector.

    Parameters
    ----------
    n : int
        Number of items.
    weights : ArrayLike, optional
        Non-negative weights (normalized internally).  Mutually exclusive
        with *log_weights*.
    log_weights : ArrayLike, optional
        Log-unnormalized weights.  Preferred when weights span many orders
        of magnitude (e.g. importance sampling).  Mutually exclusive with
        *weights*.

    Examples
    --------
    Create from raw weights or log-weights:

    >>> w = Weights(3, weights=jnp.array([1.0, 2.0, 1.0]))
    >>> w = Weights(3, log_weights=jnp.array([-1.0, 0.0, -1.0]))

    Create uniform weights:

    >>> w = Weights.uniform(5)

    **Use as a JAX array** — returns normalized weights automatically:

    >>> jnp.sum(w)                          # -> ~1.0
    >>> jnp.einsum("n,n...->...", w, vals)  # weighted sum
    >>> w * values                          # element-wise product

    This means ``Weights`` can be passed anywhere a weight array is
    expected, including ``jax.random.choice(..., p=w)``.

    **Access different representations** explicitly when needed:

    >>> w.normalized         # Array, shape (n,) — probabilities summing to 1
    >>> w.log_normalized     # Array | None — log-probabilities (None if uniform)
    >>> w.log_unnormalized   # Array | None — raw stored log-weights (None if uniform)
    >>> w.is_uniform         # bool — True when all items are equally weighted

    **Compute weighted statistics** directly:

    >>> w.mean(samples)                 # weighted mean along leading axis
    >>> w.variance(samples)             # weighted variance
    >>> w.covariance(samples)           # weighted covariance matrix
    >>> w.choice(key, shape=(10,))      # draw 10 weighted random indices

    **JAX compatibility** — ``Weights`` is registered as a JAX pytree,
    so it works transparently inside ``jax.jit``, ``jax.vmap``, and
    ``jax.grad``.  The log-weights array is the traceable leaf;
    ``is_uniform`` and ``n`` are static auxiliary data.  The
    normalization cache is a Python-side optimization that is recomputed
    inside traced contexts.
    """

    __slots__ = ("_n", "_log_weights", "_is_uniform", "_cache")

    def __init__(
        self,
        n: int,
        weights: ArrayLike | None = None,
        *,
        log_weights: ArrayLike | None = None,
    ):
        self._n = n
        self._log_weights, self._is_uniform = validate_and_normalize_log_weights(
            n, weights, log_weights=log_weights,
        )
        self._cache: Array | None = None

    @staticmethod
    def uniform(n: int) -> Weights:
        """Create uniform weights over *n* items."""
        w = object.__new__(Weights)
        w._n = n
        w._log_weights = None
        w._is_uniform = True
        w._cache = None
        return w

    @staticmethod
    def _coerce(
        n: int,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | None = None,
    ) -> Weights:
        """Build a ``Weights`` from flexible input.

        Accepts either an existing ``Weights`` instance (returned as-is),
        or raw ``weights`` / ``log_weights`` arrays (validated and
        wrapped).  If neither is provided, returns uniform weights.

        This is the canonical way to initialize ``Weights`` inside
        distribution constructors, ensuring a consistent API: callers
        can pass unnormalized weights, log-weights, or a pre-built
        ``Weights`` object.
        """
        if isinstance(weights, Weights):
            return weights
        if weights is None and log_weights is None:
            return Weights.uniform(n)
        return Weights(n, weights, log_weights=log_weights)

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of items."""
        return self._n

    @property
    def is_uniform(self) -> bool:
        """``True`` when all items are equally weighted."""
        return self._is_uniform

    @property
    def normalized(self) -> Array:
        """Normalized weights, shape ``(n,)``.  Cached after first access."""
        if self._is_uniform:
            return uniform_weights(self._n)
        if self._cache is None:
            self._cache = normalize_weights(self._log_weights)
        return self._cache

    @property
    def log_normalized(self) -> Array | None:
        """Normalized log-weights, shape ``(n,)``.  ``None`` when uniform."""
        if self._is_uniform:
            return None
        return normalized_log_weights(self._log_weights)

    @property
    def log_unnormalized(self) -> Array | None:
        """Raw log-unnormalized weights as stored.  ``None`` when uniform."""
        return self._log_weights

    # -- JAX array protocol -------------------------------------------------

    def __jax_array__(self) -> Array:
        """Return normalized weights as a JAX array.

        This allows ``Weights`` to be used directly in JAX operations
        (``jnp.sum(w)``, ``jnp.einsum(..., w, ...)``, ``w * arr``, etc.).
        """
        return self.normalized

    # -- array duck-typing --------------------------------------------------

    @property
    def shape(self) -> tuple[int]:
        """Shape of the weight vector: ``(n,)``."""
        return (self._n,)

    @property
    def dtype(self) -> jnp.dtype:
        """Data type: always ``float32``."""
        return jnp.float32

    def __len__(self) -> int:
        return self._n

    # -- weighted operations ------------------------------------------------

    def mean(self, values: Array) -> Array:
        """Compute weighted mean: ``sum_i w_i * values[i]``."""
        return weighted_mean(
            None if self._is_uniform else self.normalized, values,
        )

    def variance(self, values: Array, mean: Array | None = None) -> Array:
        """Compute weighted variance over the leading axis."""
        return weighted_variance(
            None if self._is_uniform else self.normalized, values, mean=mean,
        )

    def covariance(self, values: Array, mean: Array | None = None) -> Array:
        """Compute weighted covariance matrix over the leading axis."""
        return weighted_covariance(
            None if self._is_uniform else self.normalized, values, mean=mean,
        )

    def choice(self, key: PRNGKey, *, shape: tuple[int, ...] = ()) -> Array:
        """Draw weighted random indices from ``0..n-1``."""
        return weighted_choice(
            key, self._n,
            weights=None if self._is_uniform else self.normalized,
            shape=shape,
        )

    def subsample(self, indices: Array) -> Weights:
        """Return a new ``Weights`` for a subset, re-normalized.

        Parameters
        ----------
        indices : Array
            Integer indices selecting a subset of items.

        Returns
        -------
        Weights
            New ``Weights`` over ``len(indices)`` items with weights
            proportional to the original weights at *indices*.
        """
        if self._is_uniform:
            return Weights.uniform(len(indices))
        sub_log = self._log_weights[indices]
        return Weights(len(indices), log_weights=sub_log)

    # -- JAX pytree registration --------------------------------------------

    def tree_flatten(self):
        """Flatten for JAX pytree: leaves are the traceable arrays."""
        children = (self._log_weights,)
        aux = (self._n, self._is_uniform)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Unflatten from JAX pytree."""
        (log_weights,) = children
        n, is_uniform = aux
        w = object.__new__(cls)
        w._n = n
        w._log_weights = log_weights
        w._is_uniform = is_uniform
        w._cache = None
        return w

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        if self._is_uniform:
            return f"Weights(n={self._n}, uniform)"
        return f"Weights(n={self._n})"


jax.tree_util.register_pytree_node(
    Weights,
    lambda w: w.tree_flatten(),
    lambda aux, children: Weights.tree_unflatten(aux, children),
)
