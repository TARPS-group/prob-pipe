"""Centralized weight utilities for ProbPipe.

Provides:
  - ``Weights``               – Normalized probability weights (JAX-array-compatible).
  - ``weighted_mean()``       – Weighted mean over leading axis.
  - ``weighted_variance()``   – Weighted variance over leading axis.
  - ``weighted_covariance()`` – Weighted covariance matrix.
  - ``weighted_choice()``     – Draw weighted random indices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .custom_types import Array, ArrayLike, PRNGKey


# ---------------------------------------------------------------------------
# Pure utility functions (internal workhorses)
# ---------------------------------------------------------------------------

def _validate_to_log_weights(
    n: int,
    weights: ArrayLike | None = None,
    *,
    log_weights: ArrayLike | None = None,
) -> tuple[Array | None, bool]:
    """Validate user-supplied weights and convert to log-unnormalized form.

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
    n : int, optional
        Number of items.  Required (and only allowed) for uniform
        weights — i.e. when neither *weights* nor *log_weights* is
        provided.
    weights : ArrayLike, optional
        Non-negative weights (normalized internally).  ``n`` is inferred
        from ``len(weights)``.  Mutually exclusive with *log_weights*
        and *n*.
    log_weights : ArrayLike, optional
        Log-unnormalized weights.  Preferred when weights span many
        orders of magnitude (e.g. importance sampling).  ``n`` is
        inferred from ``len(log_weights)``.  Mutually exclusive with
        *weights* and *n*.

    Exactly one of *n*, *weights*, or *log_weights* must be provided.

    Examples
    --------
    Create from raw weights or log-weights:

    >>> w = Weights(weights=jnp.array([1.0, 2.0, 1.0]))
    >>> w = Weights(log_weights=jnp.array([-1.0, 0.0, -1.0]))

    Create uniform weights:

    >>> w = Weights(n=5)

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

    **Passing to distribution constructors** — all ProbPipe distribution
    constructors that accept ``weights`` or ``log_weights`` also accept
    a pre-built ``Weights`` object for either parameter.  When a
    ``Weights`` object is passed, it is used as-is (no re-validation).
    The behavior is the same regardless of which parameter it is passed
    to, since the ``Weights`` object already encapsulates its
    representation::

        w = Weights(log_weights=log_w)
        EmpiricalDistribution(samples, weights=w)       # OK
        EmpiricalDistribution(samples, log_weights=w)   # also OK, same result

    **JAX compatibility** — ``Weights`` is registered as a JAX pytree,
    so it works transparently inside ``jax.jit``, ``jax.vmap``, and
    ``jax.grad``.  The log-weights array is the traceable leaf;
    ``is_uniform`` and ``n`` are static auxiliary data.  The
    normalization cache is a Python-side optimization that is recomputed
    inside traced contexts.

    Notes
    -----
    **Zero weights and ``-inf`` in log-space.** When a weight array
    contains zeros (e.g. ``[0.0, 1.0, 0.0]``), the internal
    log-representation stores ``-inf`` for those entries.  All
    ``Weights`` operations handle this correctly:

    - ``normalized`` produces ``0.0`` for those items (via softmax).
    - ``log_normalized`` contains ``-inf`` entries (mathematically
      correct: ``log(0) = -inf``).
    - ``effective_sample_size`` is unaffected (``logsumexp`` handles
      ``-inf`` inputs).
    - ``choice`` never selects zero-weight items.

    Code that consumes ``log_normalized`` directly should be aware
    that ``-inf`` values may be present.
    """

    __slots__ = ("_n", "_log_weights", "_is_uniform", "_cache")

    def __init__(
        self,
        *,
        n: int | None = None,
        weights: ArrayLike | None = None,
        log_weights: ArrayLike | None = None,
    ):
        provided = (n is not None) + (weights is not None) + (log_weights is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of n, weights, or log_weights must be provided."
            )

        if n is not None:
            # Uniform weights
            self._n = n
            self._log_weights = None
            self._is_uniform = True
        elif weights is not None:
            weights = jnp.asarray(weights, dtype=jnp.float32)
            if weights.ndim != 1 or weights.shape[0] == 0:
                raise ValueError("weights must be a non-empty 1-D array.")
            if jnp.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            total = jnp.sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            self._n = weights.shape[0]
            self._log_weights = jnp.log(weights)
            self._is_uniform = False
        else:  # log_weights is not None
            log_weights = jnp.asarray(log_weights, dtype=jnp.float32)
            if log_weights.ndim != 1 or log_weights.shape[0] == 0:
                raise ValueError("log_weights must be a non-empty 1-D array.")
            self._n = log_weights.shape[0]
            self._log_weights = log_weights
            self._is_uniform = False

        self._cache: Array | None = None

    @staticmethod
    def uniform(n: int) -> Weights:
        """Create uniform weights over *n* items.

        Equivalent to ``Weights(n=n)`` but avoids keyword overhead in
        hot internal paths.
        """
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
        log_weights: ArrayLike | Weights | None = None,
    ) -> Weights:
        """Build a ``Weights`` from flexible input.

        This is the canonical entry point used by distribution
        constructors to normalize their ``weights`` / ``log_weights``
        arguments into a ``Weights`` object in a single line.

        Parameters
        ----------
        n : int
            Number of items (used only when both *weights* and
            *log_weights* are ``None``, to create uniform weights).
        weights : ArrayLike, Weights, or None
            Unnormalized weight array, a pre-built ``Weights`` object,
            or ``None``.
        log_weights : ArrayLike, Weights, or None
            Log-unnormalized weight array, a pre-built ``Weights``
            object, or ``None``.

        Returns
        -------
        Weights

        Notes
        -----
        When a ``Weights`` object is passed (to either parameter), it
        is returned as-is with no re-validation.  The behavior is
        identical regardless of which parameter receives the object,
        since a ``Weights`` instance already encapsulates its internal
        representation.

        Raises ``ValueError`` if both *weights* and *log_weights* are
        non-``None``.
        """
        # Fast path: already a Weights object
        if isinstance(weights, Weights):
            if log_weights is not None:
                raise ValueError(
                    "Provide either weights or log_weights, not both."
                )
            return weights
        if isinstance(log_weights, Weights):
            if weights is not None:
                raise ValueError(
                    "Provide either weights or log_weights, not both."
                )
            return log_weights

        # Raw arrays or None
        if weights is not None and log_weights is not None:
            raise ValueError(
                "Provide either weights or log_weights, not both."
            )
        if weights is not None:
            w = Weights(weights=weights)
            if w.n != n:
                raise ValueError(
                    f"weights length {w.n} does not match "
                    f"number of items {n}."
                )
            return w
        if log_weights is not None:
            w = Weights(log_weights=log_weights)
            if w.n != n:
                raise ValueError(
                    f"log_weights length {w.n} does not match "
                    f"number of items {n}."
                )
            return w
        return Weights.uniform(n)

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

    @property
    def effective_sample_size(self) -> Array:
        r"""Kish's effective sample size (ESS).

        .. math::

            n_{\mathrm{eff}} = \frac{1}{\sum_i w_i^2}

        where :math:`w_i` are the **normalized** weights.  For uniform
        weights this equals *n* exactly.

        Computed in log-space for numerical stability:

        .. math::

            n_{\mathrm{eff}}
            = \exp\!\bigl(-\log \sum_i \exp(2 \log w_i)\bigr)
            = \exp\!\bigl(-\mathrm{logsumexp}(2\,\log\mathbf{w})\bigr)

        Returns
        -------
        Array
            Scalar effective sample size (``1 <= n_eff <= n``).
        """
        if self._is_uniform:
            return jnp.array(self._n, dtype=jnp.float32)
        log_norm = normalized_log_weights(self._log_weights)
        return jnp.exp(-jax.scipy.special.logsumexp(2.0 * log_norm))

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

    # -- equality & hashing -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Weights):
            return NotImplemented
        if self._n != other._n or self._is_uniform != other._is_uniform:
            return False
        if self._is_uniform:
            return True
        return bool(jnp.array_equal(self._log_weights, other._log_weights))

    def __hash__(self) -> int:
        if self._is_uniform:
            return hash((self._n, True))
        # JAX arrays are immutable, so hashing the bytes is safe.
        return hash((self._n, False, self._log_weights.tobytes()))

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
        return Weights(log_weights=sub_log)

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
