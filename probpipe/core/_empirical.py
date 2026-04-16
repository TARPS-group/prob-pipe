"""Empirical and bootstrap distribution classes.

Provides:
  - ``EmpiricalDistribution[T]``        – Generic weighted empirical distribution.
  - ``NumericEmpiricalDistribution``      – Numeric array specialization with moments.
  - ``_RecordEmpiricalDistribution``     – Record specialization with per-field moments.
  - ``BootstrapReplicateDistribution[T]``    – Bootstrap resampling over datasets.
  - ``ArrayBootstrapReplicateDistribution``  – Array specialization.
  - ``_RecordBootstrapReplicateDistribution`` – Record specialization with joint row resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .._utils import prod, _is_numeric_array
from .protocols import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

import jax
import jax.numpy as jnp

from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights, weighted_mean
from .constraints import Constraint, real
from . import _distribution_base as _base
from .._utils import _auto_key
from ._distribution_base import Distribution
from ._array_distributions import (
    NumericRecordDistribution,
    BootstrapDistribution,
)
from ._record_distribution import RecordDistribution
from .record import Record, RecordTemplate


# ---------------------------------------------------------------------------
# EmpiricalDistribution (generic base)
# ---------------------------------------------------------------------------

class EmpiricalDistribution[T](
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
):
    """
    Weighted empirical distribution over a finite set of samples.

    This is the generic base class.  It stores samples in a numpy object
    array, supporting arbitrary sample types ``T`` (arrays, pytrees,
    distributions, callables, etc.).

    **Automatic array dispatch:** When *samples* is a numeric JAX or
    numpy array, ``EmpiricalDistribution(samples, ...)`` automatically
    returns a :class:`NumericEmpiricalDistribution` instance, which
    provides TFP-style shape semantics (``batch_shape``, ``event_shape``,
    ``flatten_value``, ``support``, etc.) and exact weighted moments.
    Pass a non-numeric sequence (e.g. a list of objects) to get the
    generic base class.

    Parameters
    ----------
    samples : sequence of T
        The support points.  Must be a non-empty sequence (list, tuple,
        or array).
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted and used
        as-is.  Mutually exclusive with *log_weights*.  When neither is
        given the distribution is uniform.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  Preferred when weights span many
        orders of magnitude (e.g. importance sampling).  A pre-built
        :class:`~probpipe.Weights` object is also accepted and used
        as-is.  Mutually exclusive with *weights*.
    name : str, optional
        An optional name for provenance / JointDistribution integration.
    """

    def __new__(cls, samples=None, *args, **kwargs):
        if cls is EmpiricalDistribution and samples is not None:
            if isinstance(samples, Record):
                return object.__new__(_RecordEmpiricalDistribution)
            if _is_numeric_array(samples):
                return object.__new__(NumericEmpiricalDistribution)
        return object.__new__(cls)

    def __init__(
        self,
        samples: Sequence[T] | ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
    ):
        # If samples are already a JAX or numpy array, store as-is.
        # Otherwise store as a numpy object array for generic indexing.
        if isinstance(samples, (jnp.ndarray, np.ndarray)):
            self._samples = samples
        else:
            self._samples = np.asarray(samples, dtype=object)
        n = len(self._samples)
        if n == 0:
            raise ValueError("samples must be a non-empty sequence.")
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        if name is None:
            name = "empirical"
        super().__init__(name=name)
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of samples."""
        return len(self._samples)

    @property
    def samples(self) -> np.ndarray:
        """The stored samples as a numpy object array."""
        return self._samples

    @property
    def is_uniform(self) -> bool:
        """True when all samples have equal weight."""
        return self._w.is_uniform

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        return self._w.normalized

    @property
    def log_weights(self) -> Array:
        """Normalised log-weights, shape ``(n,)``."""
        return self._w.log_normalized

    @property
    def effective_sample_size(self) -> Array:
        """Kish's effective sample size (ESS).

        Returns ``n`` for uniformly weighted samples; less than ``n``
        when weights are non-uniform.  Computed in log-space for
        numerical stability.
        """
        return self._w.effective_sample_size

    # -- sampling -----------------------------------------------------------

    def _sample_one(self, key: PRNGKey) -> T:
        """Draw a single sample (with replacement according to weights)."""
        idx = self._w.choice(key)
        return self._samples[idx]

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw samples with replacement according to weights.

        When samples are stored as an array (JAX or numpy), returns an
        array of shape ``(*sample_shape, *event_shape)`` via fancy
        indexing.  When samples are stored as a numpy object array
        (arbitrary types), returns an object array of shape
        ``sample_shape``.
        """
        if sample_shape == ():
            return self._sample_one(key)
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        draws = self._samples[indices]
        return draws.reshape(sample_shape + draws.shape[1:])

    @property
    def _is_object_array(self) -> bool:
        """True when samples are stored as a numpy object array."""
        return isinstance(self._samples, np.ndarray) and self._samples.dtype == object

    def _eval_f(self, f: Callable, samples: Any) -> Array:
        """Evaluate *f* over *samples*, using vmap when possible."""
        if self._is_object_array:
            return jnp.stack([f(x) for x in samples])
        return jax.vmap(f)(samples)

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        """Compute ``E[f(X)]`` over the empirical support.

        When ``num_evaluations`` is ``None``, the expectation is computed
        exactly as a weighted sum over all support points.  When
        ``num_evaluations`` is specified and smaller than ``self.n``, a
        random subsample is used.
        """
        if num_evaluations is not None and num_evaluations < self.n:
            # Subsample — this is approximate
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(key, self.n, shape=(num_evaluations,), replace=False)
            f_vals = self._eval_f(f, self._samples[idx])
            sub_w = self._w.subsample(idx)

            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                return BootstrapDistribution(f_vals, weights=sub_w)

            return sub_w.mean(f_vals)

        # Exact: evaluate f on all support points
        f_vals = self._eval_f(f, self._samples)
        return self._w.mean(f_vals)


# ---------------------------------------------------------------------------
# NumericEmpiricalDistribution
# ---------------------------------------------------------------------------

class NumericEmpiricalDistribution(
    EmpiricalDistribution[Array],
    NumericRecordDistribution,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """Empirical distribution with full numeric shape semantics.

    Stores samples as a stacked JAX array for efficient vectorised
    operations (``jax.vmap``-based sampling and expectations).

    Inherits weight management from :class:`EmpiricalDistribution` and
    adds TFP-style shape properties (``batch_shape``, ``flatten_value``,
    ``unflatten_value``, ``support``, etc.) via :class:`NumericRecordDistribution`,
    plus exact weighted moments (mean, variance, covariance).

    Parameters
    ----------
    samples : array-like
        Sample array.  The leading axis (or axes, if *sample_shape* is
        given) indexes the support points; trailing axes form the event
        shape.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    sample_shape : tuple of int, optional
        Shape of the leading sample dimensions.  When ``None`` (default),
        the first axis is the single sample axis (``n = samples.shape[0]``)
        and ``event_shape = samples.shape[1:]``.  When provided,
        ``n = prod(sample_shape)``, the leading dimensions must match
        *sample_shape*, and ``event_shape = samples.shape[len(sample_shape):]``.
        The samples array is reshaped internally to ``(n, *event_shape)``.
    name : str, optional
        Distribution name.
    """

    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        sample_shape: tuple[int, ...] | None = None,
        name: str | None = None,
    ):
        # Store only the JAX array — bypass the generic base's storage.
        self._samples = jnp.asarray(samples, dtype=jnp.float32)
        if self._samples.ndim == 0:
            raise ValueError("samples must have at least 1 dimension (the sample axis).")

        if sample_shape is not None:
            n_sample_dims = len(sample_shape)
            if self._samples.shape[:n_sample_dims] != sample_shape:
                raise ValueError(
                    f"Leading dimensions {self._samples.shape[:n_sample_dims]} "
                    f"do not match sample_shape {sample_shape}."
                )
            n = prod(sample_shape)
            event_shape = self._samples.shape[n_sample_dims:]
            self._samples = self._samples.reshape(n, *event_shape)

        self._w = Weights(
            n=self._samples.shape[0], weights=weights, log_weights=log_weights,
        )
        if name is None:
            name = "empirical"
        # Skip EmpiricalDistribution.__init__ (different init pattern),
        # go directly to Distribution.__init__ for name registration.
        Distribution.__init__(self, name=name)
        self._approximate = True

    # -- array-specific properties ------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (excluding the sample axis)."""
        return self._samples.shape[1:]

    @property
    def dim(self) -> int:
        """Flat dimensionality of each sample (product of event_shape, or 1 for scalars)."""
        return max(1, prod(self._samples.shape[1:]))

    @property
    def dtype(self) -> jnp.dtype:
        return self._samples.dtype

    @property
    def support(self) -> Constraint:
        return real

    # -- moments ------------------------------------------------------------

    def _mean(self) -> Array:
        return self._w.mean(self._samples)

    def _variance(self) -> Array:
        return self._w.variance(self._samples)

    def _cov(self) -> Array:
        """Weighted sample covariance matrix, shape ``(d, d)``."""
        return self._w.covariance(self._samples)


# ---------------------------------------------------------------------------
# Shared helpers for Record-based distribution classes
# ---------------------------------------------------------------------------


def _record_template_from_data(
    values_data: Record,
    leading_shape: tuple[int, ...] = (),
) -> RecordTemplate:
    """Build a RecordTemplate from stored Record data.

    Strips the first dimension (sample axis) from each field to get
    event shapes, optionally prepending ``leading_shape``.
    """
    specs: dict[str, tuple[int, ...]] = {}
    for fname in values_data.fields:
        arr = jnp.asarray(values_data[fname])
        specs[fname] = (*leading_shape, *arr.shape[1:])
    return RecordTemplate(specs)


def _index_record(values_data: Record, idx) -> Record:
    """Index all fields of a Record object with the same indices."""
    return Record({
        f: jnp.asarray(values_data[f])[idx]
        for f in values_data.fields
    })


def _fieldwise_op(values_data: Record, op: Callable) -> Record:
    """Apply an operation to each field of a Record object."""
    return Record({
        f: op(jnp.asarray(values_data[f]))
        for f in values_data.fields
    })


# ---------------------------------------------------------------------------
# _RecordEmpiricalDistribution (Record specialization)
# ---------------------------------------------------------------------------


class _RecordEmpiricalDistribution(
    EmpiricalDistribution[Record],
    RecordDistribution,
    SupportsMean,
    SupportsVariance,
):
    """Empirical distribution over Record-structured data.

    Each sample is a *row* of the stored Record: if the Record has fields
    ``X`` of shape ``(n, p)`` and ``y`` of shape ``(n,)``, then ``n`` is
    the number of samples and each draw is a ``Record(X=array(p,), y=scalar)``.

    Joint row indexing ensures that all fields are resampled with the same
    indices, preserving per-observation relationships.

    The ``record_template`` is set automatically from the field shapes
    (leading sample dimension removed), enabling ``__getitem__``,
    ``select()``, and ``component_names`` via
    :class:`~probpipe.core._record_distribution.RecordDistribution`.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        samples: Record,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
    ):
        self._record_data = samples
        first_field = samples[samples.fields[0]]
        n = jnp.asarray(first_field).shape[0]
        self._n_samples = n
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        if name is None:
            name = "empirical(" + ",".join(samples.fields) + ")"
        # Skip EmpiricalDistribution.__init__ (different init pattern),
        # go directly to Distribution.__init__ for name registration.
        Distribution.__init__(self, name=name)
        self._approximate = True
        self._record_template = _record_template_from_data(samples)

    @property
    def n(self) -> int:
        return self._n_samples

    @property
    def samples(self) -> Record:
        """The stored Record data."""
        return self._record_data

    def _sample_one(self, key: PRNGKey) -> Record:
        idx = self._w.choice(key)
        return _index_record(self._record_data, idx)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Record:
        if sample_shape == ():
            return self._sample_one(key)
        indices = self._w.choice(key, shape=(prod(sample_shape),))
        fields: dict[str, jnp.ndarray] = {}
        for f in self._record_data.fields:
            arr = jnp.asarray(self._record_data[f])
            fields[f] = arr[indices].reshape(*sample_shape, *arr.shape[1:])
        return Record(fields)

    def _mean(self) -> Record:
        return _fieldwise_op(self._record_data, self._w.mean)

    def _variance(self) -> Record:
        return _fieldwise_op(self._record_data, self._w.variance)

    def __repr__(self) -> str:
        fields = ", ".join(self._record_data.fields)
        return (
            f"_RecordEmpiricalDistribution(n={self._n_samples}, "
            f"fields=({fields}))"
        )


# ---------------------------------------------------------------------------
# BootstrapReplicateDistribution (generic base)
# ---------------------------------------------------------------------------

class BootstrapReplicateDistribution[T](
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
):
    """N-fold product of an empirical distribution (bootstrap resampling).

    This is the generic base class.  It stores observations in a numpy
    object array, supporting arbitrary observation types ``T``.

    **Automatic array dispatch:** When *source* is a numeric JAX or
    numpy array (or an ``EmpiricalDistribution`` backed by numeric
    arrays), ``BootstrapReplicateDistribution(source, ...)``
    automatically returns an :class:`ArrayBootstrapReplicateDistribution`
    instance with TFP-style shape semantics.

    Each sample from this distribution is a bootstrapped dataset — ``n``
    observations drawn i.i.d. with replacement from the source data.
    This provides the sampling distribution over datasets needed for
    BayesBag (bagged posteriors).

    When the source is an :class:`EmpiricalDistribution`, ``n`` defaults
    to the number of samples in the empirical distribution.  Otherwise
    ``n`` must be specified explicitly.

    Parameters
    ----------
    source : EmpiricalDistribution or sequence
        The data to bootstrap from.  If an ``EmpiricalDistribution``,
        its samples and weights are used directly.  If a sequence,
        it is treated as an equally-weighted dataset.
    n : int or None
        Number of observations per bootstrap dataset.  Defaults to the
        number of samples in ``source``.
    name : str or None
        Distribution name for provenance.

    Examples
    --------
    >>> data = EmpiricalDistribution(observed_data)
    >>> bootstrap = BootstrapReplicateDistribution(data)
    >>> # Each sample is a bootstrapped dataset of the same size
    >>> boot_dataset = sample(bootstrap, key=jax.random.PRNGKey(0))
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(cls, source=None, *args, **kwargs):
        if cls is BootstrapReplicateDistribution and source is not None:
            if isinstance(source, _RecordEmpiricalDistribution):
                return object.__new__(_RecordBootstrapReplicateDistribution)
            if isinstance(source, Record):
                return object.__new__(_RecordBootstrapReplicateDistribution)
            if _is_numeric_array(source):
                return object.__new__(ArrayBootstrapReplicateDistribution)
            if isinstance(source, EmpiricalDistribution) and _is_numeric_array(source.samples):
                return object.__new__(ArrayBootstrapReplicateDistribution)
        return object.__new__(cls)

    def __init__(
        self,
        source: EmpiricalDistribution | Sequence | ArrayLike,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        if isinstance(source, EmpiricalDistribution):
            self._data = source.samples
            self._w = source._w
            default_n = source.n
        elif isinstance(source, (jnp.ndarray, np.ndarray)):
            self._data = source
            if self._data.ndim == 0:
                raise ValueError("source must have at least 1 dimension (the observation axis).")
            if len(self._data) == 0:
                raise ValueError("source must be a non-empty sequence.")
            self._w = Weights.uniform(len(self._data))
            default_n = len(self._data)
        else:
            self._data = np.asarray(source, dtype=object)
            if len(self._data) == 0:
                raise ValueError("source must be a non-empty sequence.")
            self._w = Weights.uniform(len(self._data))
            default_n = len(self._data)
        self._init_bootstrap_state(default_n, n=n, name=name)

    def _init_bootstrap_state(
        self,
        default_n: int,
        *,
        n: int | None,
        name: str | None,
    ) -> None:
        """Set n, source_n, name, and flags. Shared with Array subclass."""
        if n is None:
            self._n = default_n
        else:
            if n < 1:
                raise ValueError(f"n must be positive, got {n}")
            self._n = n

        if name is None:
            name = "bootstrap"
        super().__init__(name=name)
        self._source_n = len(self._data)
        self._approximate = True

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of observations per bootstrap dataset."""
        return self._n

    @property
    def source_n(self) -> int:
        """Number of observations in the source data."""
        return self._source_n

    @property
    def data(self) -> np.ndarray:
        """The source data as a numpy object array."""
        return self._data

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(source_n,)``."""
        return self._w.normalized

    @property
    def is_uniform(self) -> bool:
        """True when all source observations have equal weight."""
        return self._w.is_uniform

    def _sample_one(self, key: PRNGKey) -> Any:
        """Draw a single bootstrapped dataset."""
        idx = self._w.choice(key, shape=(self._n,))
        return self._data[idx]

    @property
    def _is_object_data(self) -> bool:
        """True when source data is stored as a numpy object array."""
        return isinstance(self._data, np.ndarray) and self._data.dtype == object

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        """Draw bootstrap datasets."""
        if sample_shape == ():
            return self._sample_one(key)

        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        if self._is_object_data:
            results = np.empty(total, dtype=object)
            for i in range(total):
                results[i] = self._sample_one(keys[i])
            return results.reshape(sample_shape)
        results = jax.vmap(self._sample_one)(keys)
        return results.reshape(*sample_shape, *results.shape[1:])

    # -- expectation --------------------------------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Array | BootstrapDistribution:
        """Compute ``E[f(dataset)]`` via Monte Carlo over bootstrap datasets.

        Parameters
        ----------
        f : callable
            Function mapping a bootstrap dataset to an array.
        key : PRNGKey, optional
            Random key.  Auto-generated if ``None``.
        num_evaluations : int, optional
            Number of bootstrap datasets to draw.  Defaults to
            :data:`DEFAULT_NUM_EVALUATIONS`.
        return_dist : bool, optional
            If ``True``, return a :class:`BootstrapDistribution` over
            the evaluations.  Defaults to :data:`RETURN_APPROX_DIST`.
        """
        if key is None:
            key = _auto_key()
        if num_evaluations is None:
            num_evaluations = _base.DEFAULT_NUM_EVALUATIONS

        datasets = self._sample(key, sample_shape=(num_evaluations,))
        if self._is_object_data:
            f_vals = jnp.stack([f(datasets[i]) for i in range(num_evaluations)])
        else:
            f_vals = jax.vmap(f)(datasets)

        rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
        if rd:
            return BootstrapDistribution(f_vals)
        return jnp.mean(f_vals, axis=0)

    def __repr__(self) -> str:
        return (
            f"BootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n})"
        )



class ArrayBootstrapReplicateDistribution(BootstrapReplicateDistribution[Array], NumericRecordDistribution):
    """Joint bootstrap distribution with full :class:`NumericRecordDistribution` shape semantics.

    Inherits all functionality from :class:`BootstrapReplicateDistribution` and adds
    TFP-style shape properties (``batch_shape``, ``event_shape``, ``support``,
    etc.) via :class:`NumericRecordDistribution`.

    Use this instead of :class:`BootstrapReplicateDistribution` when the distribution
    must interoperate with code that requires :class:`NumericRecordDistribution` instances.
    """

    def __init__(
        self,
        source: EmpiricalDistribution | ArrayLike,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        # Coerce to JAX array, then let the generic base store it.
        if isinstance(source, EmpiricalDistribution):
            jax_data = jnp.asarray(list(source.samples), dtype=jnp.float32)
            self._data = jax_data
            self._w = source._w
            default_n = source.n
        else:
            jax_data = jnp.asarray(source, dtype=jnp.float32)
            if jax_data.ndim == 0:
                raise ValueError("source must have at least 1 dimension (the observation axis).")
            self._data = jax_data
            self._w = Weights.uniform(jax_data.shape[0])
            default_n = jax_data.shape[0]

        self._event_shape_per_obs = self._data.shape[1:]
        self._init_bootstrap_state(default_n, n=n, name=name)

    # -- array-specific properties ------------------------------------------

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of a single observation (excluding the observation axis)."""
        return self._event_shape_per_obs

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single bootstrap dataset: ``(n, *obs_shape)``."""
        return (self._n, *self._event_shape_per_obs)

    @property
    def dim(self) -> int:
        """Flat dimensionality of each bootstrap dataset."""
        return self._n * max(1, prod(self._event_shape_per_obs))

    @property
    def dtype(self) -> jnp.dtype:
        return self._data.dtype

    @property
    def support(self) -> Constraint:
        return real

    def __repr__(self) -> str:
        return (
            f"ArrayBootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n}, "
            f"obs_shape={self._event_shape_per_obs})"
        )


# ---------------------------------------------------------------------------
# _RecordBootstrapReplicateDistribution (Record specialization)
# ---------------------------------------------------------------------------


class _RecordBootstrapReplicateDistribution(
    BootstrapReplicateDistribution[Record],
    RecordDistribution,
):
    """Bootstrap replicate distribution over Record-structured data.

    Each sample is a full bootstrapped dataset: ``n`` rows drawn i.i.d.
    with replacement from the source data, with the *same* row indices
    applied to all fields jointly.

    Supports named field access (``bootstrap["X"]``, ``bootstrap["y"]``)
    via :class:`~probpipe.core._record_distribution.RecordDistribution`,
    returning ``_RecordDistributionView`` instances that preserve
    correlation when used together in workflow function broadcasting.

    Parameters
    ----------
    source : _RecordEmpiricalDistribution or Record
        The data to bootstrap from.
    n : int or None
        Number of observations per bootstrap dataset.  Defaults to the
        number of rows in ``source``.
    name : str or None
        Distribution name for provenance.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        source: _RecordEmpiricalDistribution | Record,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        if isinstance(source, _RecordEmpiricalDistribution):
            self._record_data = source._record_data
            self._w = source._w
            default_n = source.n
        elif isinstance(source, Record):
            self._record_data = source
            first = jnp.asarray(source[source.fields[0]])
            default_n = first.shape[0]
            self._w = Weights.uniform(default_n)
        else:
            raise TypeError(
                f"Expected Record or _RecordEmpiricalDistribution, "
                f"got {type(source).__name__}"
            )
        # self._data required by base class .data property
        self._data = self._record_data
        self._init_bootstrap_state(default_n, n=n, name=name)
        self._record_template = _record_template_from_data(
            self._record_data, leading_shape=(self._n,),
        )

    def _sample_one(self, key: PRNGKey) -> Record:
        idx = self._w.choice(key, shape=(self._n,))
        return _index_record(self._record_data, idx)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Record:
        if sample_shape == ():
            return self._sample_one(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        results = [self._sample_one(k) for k in keys]
        # Stack: each result is Record(X=array(n,p), y=array(n,))
        # → Record(X=array(*sample_shape, n, p), y=array(*sample_shape, n))
        stacked: dict[str, jnp.ndarray] = {}
        for f in self._record_data.fields:
            arrs = jnp.stack([jnp.asarray(r[f]) for r in results])
            stacked[f] = arrs.reshape(*sample_shape, *arrs.shape[1:])
        return Record(stacked)

    def __repr__(self) -> str:
        fields = ", ".join(self._record_data.fields)
        return (
            f"_RecordBootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n}, fields=({fields}))"
        )
