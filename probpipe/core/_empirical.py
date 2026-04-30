"""Empirical and bootstrap distribution classes.

The hierarchy follows the *two implementations per concept* rule (see
``CONTRIBUTING.md`` "Framework abstraction hierarchy"): a generic
:class:`EmpiricalDistribution` / :class:`BootstrapReplicateDistribution`
parameterised over the value type ``T``, plus a Record-based
specialisation that adds :class:`NumericRecordDistribution` shape
semantics. There is no third numeric-array variant — a bare numeric
array is wrapped as a single-field :class:`Record` at the constructor
boundary and dispatches to the Record-based class.

Construction-time dispatch via ``__new__``: calling the generic base
``EmpiricalDistribution(samples, ...)`` returns a
:class:`RecordEmpiricalDistribution` when ``samples`` is a ``Record``
or a numeric array (the latter requires ``name=`` so the auto-wrapped
Record has a meaningful field key). Likewise
``BootstrapReplicateDistribution(source, ...)`` returns a
:class:`RecordBootstrapReplicateDistribution` for ``Record`` /
numeric-array / numeric-array-backed ``EmpiricalDistribution``
sources, and stays in the generic base for non-array
``SupportsSampling`` sources or opaque-object sequences.

Provides:

- :class:`EmpiricalDistribution[T]` — generic weighted empirical
  distribution.
- :class:`RecordEmpiricalDistribution` — Record-valued empirical
  distribution with per-field weighted moments and TFP-style shape
  semantics. Accepts a ``Record`` or (with ``name=...``) a numeric
  array which is auto-wrapped as a single-field Record.
- :class:`BootstrapReplicateDistribution[T]` — generic bootstrap
  replicate distribution. Accepts arbitrary samples, an
  ``EmpiricalDistribution``, or any ``SupportsSampling`` source.
- :class:`RecordBootstrapReplicateDistribution` — Record-valued
  bootstrap replicate with joint row resampling.
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
from .._dtype import _as_float_array
from .._weights import Weights
from .constraints import Constraint, real
from . import _distribution_base as _base
from .._utils import _auto_key
from ._distribution_base import Distribution
from ._numeric_record_distribution import (
    NumericRecordDistribution,
    BootstrapDistribution,
)
from ._numeric_record import NumericRecord
from .record import Record, RecordTemplate


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _record_template_from_data(
    record_data: Record,
    leading_shape: tuple[int, ...] = (),
) -> RecordTemplate:
    """Build a ``RecordTemplate`` from stored Record data.

    Strips the first dimension (sample axis) from each field to get
    event shapes, optionally prepending ``leading_shape``.
    """
    specs: dict[str, tuple[int, ...]] = {}
    for fname in record_data.fields:
        arr = jnp.asarray(record_data[fname])
        specs[fname] = (*leading_shape, *arr.shape[1:])
    return RecordTemplate(specs)


def _index_record(record_data: Record, idx) -> NumericRecord:
    """Index every field of a Record with the same indices.

    Returns a :class:`NumericRecord` so single-field results expose the
    ``__jax_array__`` / ``__float__`` shims at downstream call sites.
    """
    return NumericRecord({
        f: jnp.asarray(record_data[f])[idx]
        for f in record_data.fields
    })


def _fieldwise_op(record_data: Record, op: Callable) -> NumericRecord:
    """Apply *op* to each field of a Record, returning a :class:`NumericRecord`.

    All-numeric outputs by construction; returning a ``NumericRecord``
    lets single-field consumers use ``jnp.asarray(result)`` /
    ``float(result)`` directly via the existing single-field shims.
    """
    return NumericRecord({
        f: op(jnp.asarray(record_data[f]))
        for f in record_data.fields
    })


def _wrap_numeric_array_as_record(
    arr: ArrayLike,
    *,
    name: str | None,
    sample_shape: tuple[int, ...] | None = None,
    role: str = "EmpiricalDistribution",
) -> tuple[Record, str]:
    """Auto-wrap a numeric array as a single-field ``Record``.

    The mandatory ``name`` becomes the field name; without it the
    auto-wrap is ambiguous (the field's identity is lost downstream).

    Returns ``(record, name)`` where ``name`` is the validated field /
    distribution name.

    Parameters
    ----------
    arr : array-like
        Numeric array. Leading axis is the sample axis (or, with
        ``sample_shape``, the sample axes).
    name : str
        Field name for the wrapped Record. Required.
    sample_shape : tuple of int, optional
        Shape of the leading sample dimensions; rest is event shape.
        Defaults to ``(arr.shape[0],)``.
    role : str
        Role string used in error messages (e.g. ``"EmpiricalDistribution"``).
    """
    if not name:
        raise ValueError(
            f"{role} from a numeric array requires a non-empty name=, "
            f"so the auto-wrapped Record has a meaningful field name. "
            f"Pass name='theta' (or similar), or wrap the array yourself: "
            f"Record(theta=arr)."
        )
    arr = _as_float_array(arr)
    if arr.ndim == 0:
        raise ValueError(
            f"{role} samples must have at least 1 dimension (the sample axis)."
        )
    if sample_shape is not None:
        n_sample_dims = len(sample_shape)
        if arr.shape[:n_sample_dims] != sample_shape:
            raise ValueError(
                f"Leading dimensions {arr.shape[:n_sample_dims]} do not match "
                f"sample_shape {sample_shape}."
            )
        n = prod(sample_shape)
        event_shape = arr.shape[n_sample_dims:]
        arr = arr.reshape(n, *event_shape)
    return Record({name: arr}), name


def _validate_record_samples(record_data: Record) -> int:
    """Validate that every field shares the same sample-axis length.

    Returns the common ``n``.
    """
    if not record_data.fields:
        raise ValueError("Record samples must have at least one field.")
    first = jnp.asarray(record_data[record_data.fields[0]])
    if first.ndim == 0:
        raise ValueError(
            "Record empirical samples need a leading sample axis "
            "(every field must have shape (n, *event_shape))."
        )
    n = int(first.shape[0])
    for f in record_data.fields[1:]:
        arr = jnp.asarray(record_data[f])
        if arr.ndim == 0 or int(arr.shape[0]) != n:
            raise ValueError(
                f"Field {f!r} has sample-axis length "
                f"{None if arr.ndim == 0 else arr.shape[0]}, expected {n}."
            )
    return n


# ---------------------------------------------------------------------------
# EmpiricalDistribution (generic base)
# ---------------------------------------------------------------------------


class EmpiricalDistribution[T](
    Distribution[T],
    SupportsSampling,
    SupportsExpectation,
):
    """Weighted empirical distribution over a finite set of samples.

    This is the generic base. Concrete sample types ``T`` (objects,
    callables, opaque user values, ...) are stored in a numpy object
    array.

    **Automatic Record dispatch:** ``EmpiricalDistribution(samples,
    ...)`` returns a :class:`RecordEmpiricalDistribution` when

    - ``samples`` is a :class:`Record` (each field stacked along axis 0),
    - or ``samples`` is a numeric JAX/numpy array and ``name=...`` is
      passed (the array auto-wraps as a single-field ``Record({name:
      arr})``).

    Otherwise, the generic base is returned and stores ``samples`` as a
    numpy object array.

    Parameters
    ----------
    samples : Record | sequence of T | array-like
        The support points. Numeric-array inputs require ``name=`` so
        the auto-wrapped Record has a field name; without it construction
        raises ``ValueError``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalised internally). Mutually
        exclusive with *log_weights*. Uniform when neither is given.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalised weights. Mutually exclusive with *weights*.
    name : str, optional
        Distribution name. Mandatory when *samples* is a bare numeric
        array.
    """

    def __new__(cls, samples=None, *args, **kwargs):
        if cls is EmpiricalDistribution and samples is not None:
            if isinstance(samples, Record):
                return object.__new__(RecordEmpiricalDistribution)
            if _is_numeric_array(samples):
                return object.__new__(RecordEmpiricalDistribution)
        return object.__new__(cls)

    def __init__(
        self,
        samples: Sequence[T] | ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        name: str | None = None,
    ):
        # Generic-T storage: a numpy object array.
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
        """Stored samples."""
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
        """Kish's effective sample size (ESS)."""
        return self._w.effective_sample_size

    # -- sampling -----------------------------------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        if sample_shape == ():
            idx = self._w.choice(key)
            return self._samples[idx]
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        draws = self._samples[indices]
        return draws.reshape(sample_shape + draws.shape[1:])

    @property
    def _is_object_array(self) -> bool:
        return isinstance(self._samples, np.ndarray) and self._samples.dtype == object

    def _eval_f(self, f: Callable, samples: Any) -> Array:
        """Evaluate *f* over *samples*, vmap when possible."""
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
        """Compute ``E[f(X)]`` over the empirical support."""
        if num_evaluations is not None and num_evaluations < self.n:
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(key, self.n, shape=(num_evaluations,), replace=False)
            f_vals = self._eval_f(f, self._samples[idx])
            sub_w = self._w.subsample(idx)

            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                return BootstrapDistribution(f_vals, weights=sub_w)
            return sub_w.mean(f_vals)

        f_vals = self._eval_f(f, self._samples)
        return self._w.mean(f_vals)


# ---------------------------------------------------------------------------
# RecordEmpiricalDistribution (Record specialisation)
# ---------------------------------------------------------------------------


class RecordEmpiricalDistribution(
    EmpiricalDistribution[Record],
    NumericRecordDistribution,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """Empirical distribution over Record-structured numeric samples.

    Each *sample* is a row of the stored Record: if the data has fields
    ``X`` of shape ``(n, p)`` and ``y`` of shape ``(n,)``, then a single
    draw is ``Record(X=array(p,), y=scalar)``. Joint row indexing
    preserves per-observation correlation across fields during sampling
    and resampling.

    A bare numeric array auto-wraps as a single-field ``Record`` keyed
    by ``name`` — that is the migration path for the previous
    ``NumericEmpiricalDistribution(arr)`` form. The auto-wrap requires
    ``name=`` so the field's identity is unambiguous downstream.

    Inherits :class:`NumericRecordDistribution` shape semantics
    (``record_template``, ``event_shapes``, ``event_size``,
    ``batch_shape``) plus exact weighted moments
    (``mean``, ``variance``, ``cov``) over each field.

    Parameters
    ----------
    samples : Record | array-like
        Sample data. A Record's fields each stack along axis 0; a
        numeric array auto-wraps as ``Record({name: arr})``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalised internally). Mutually
        exclusive with *log_weights*.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalised weights. Mutually exclusive with *weights*.
    sample_shape : tuple of int, optional
        Only valid for numeric-array auto-wrap: leading-axis sample
        shape; trailing axes form the field's event shape.
    name : str, optional
        Distribution name. Required when *samples* is a numeric array
        (used as the auto-wrapped field name).
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        samples: Record | ArrayLike,
        weights: ArrayLike | Weights | None = None,
        *,
        log_weights: ArrayLike | Weights | None = None,
        sample_shape: tuple[int, ...] | None = None,
        name: str | None = None,
    ):
        if not isinstance(samples, Record):
            if not _is_numeric_array(samples):
                raise TypeError(
                    f"RecordEmpiricalDistribution: samples must be a "
                    f"Record or a numeric array, got "
                    f"{type(samples).__name__}"
                )
            samples, name = _wrap_numeric_array_as_record(
                samples, name=name, sample_shape=sample_shape,
                role="RecordEmpiricalDistribution",
            )
        elif sample_shape is not None:
            raise TypeError(
                "sample_shape is only valid when constructing from a "
                "bare numeric array (single-field auto-wrap path)."
            )
        n = _validate_record_samples(samples)
        self._record_data = samples
        self._n_samples = n
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        if name is None:
            name = "empirical(" + ",".join(samples.fields) + ")"
        # Skip EmpiricalDistribution.__init__ (different storage shape);
        # call Distribution.__init__ directly for name registration.
        Distribution.__init__(self, name=name)
        self._approximate = True
        self._record_template = _record_template_from_data(samples)

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        return self._n_samples

    @property
    def samples(self) -> NumericRecord:
        """Stored stacked-sample data as a structured :class:`NumericRecord`.

        Use ``self.samples[field_name]`` for per-field array access. For
        a flat ``(n, dim)`` matrix view across all fields, use
        :attr:`flat_samples` instead.
        """
        # Cache so repeated access doesn't re-validate. ``object.__setattr__``
        # bypasses any subclass that adds ``__slots__`` or a custom
        # ``__setattr__`` guard (e.g. ``Record`` itself raises on direct
        # attribute assignment to enforce immutability of user-visible
        # state); the cache is internal-only and a controlled exception
        # to that rule.
        cached = getattr(self, "_samples_record", None)
        if cached is None:
            cached = NumericRecord({
                f: jnp.asarray(self._record_data[f])
                for f in self._record_data.fields
            })
            object.__setattr__(self, "_samples_record", cached)
        return cached

    @property
    def flat_samples(self) -> jnp.ndarray:
        """Flat ``(n, dim)`` view across all fields, in insertion order.

        ``dim = sum_over_fields(prod(event_shape_f))``. Multi-dim event
        shapes are flattened row-major; field order matches
        :attr:`fields`. Use :attr:`samples` for the structured per-field
        view.

        Examples
        --------
        Single-field auto-wrap with a 1-D event::

            EmpiricalDistribution(jnp.zeros((100, 5)), name="theta").flat_samples.shape
            # (100, 5)

        Multi-field posterior::

            posterior = ApproximateDistribution(...)  # mu, log_sigma fields
            posterior.flat_samples.shape  # (n, 2)
            posterior.flat_samples.mean(axis=0)  # per-parameter posterior mean
        """
        # Cache so repeated access (notebook diagnostics, posterior
        # summaries) doesn't re-materialise the (n, dim) matrix —
        # ~8MB per call for a 100k-sample / 10-parameter posterior.
        cached = getattr(self, "_flat_samples_cache", None)
        if cached is None:
            parts = [
                jnp.asarray(self._record_data[f]).reshape(self._n_samples, -1)
                for f in self._record_data.fields
            ]
            cached = jnp.concatenate(parts, axis=-1)
            object.__setattr__(self, "_flat_samples_cache", cached)
        return cached

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Single-field shortcut. Multi-field records use ``event_shapes``."""
        if len(self._record_data.fields) == 1:
            f = self._record_data.fields[0]
            return tuple(jnp.asarray(self._record_data[f]).shape[1:])
        return ()

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes (sample axis stripped)."""
        return {
            f: tuple(jnp.asarray(self._record_data[f]).shape[1:])
            for f in self._record_data.fields
        }

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        return {
            f: jnp.asarray(self._record_data[f]).dtype
            for f in self._record_data.fields
        }

    @property
    def dim(self) -> int:
        """Flat dimensionality of a single Record draw."""
        return sum(
            max(1, prod(shape))
            for shape in self.event_shapes.values()
        )

    @property
    def support(self) -> Constraint:
        return real

    @property
    def supports(self) -> dict[str, Constraint]:
        return {f: real for f in self._record_data.fields}

    # -- sampling -----------------------------------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> NumericRecord:
        if sample_shape == ():
            idx = self._w.choice(key)
            return _index_record(self._record_data, idx)
        indices = self._w.choice(key, shape=(prod(sample_shape),))
        fields: dict[str, jnp.ndarray] = {}
        for f in self._record_data.fields:
            arr = jnp.asarray(self._record_data[f])
            fields[f] = arr[indices].reshape(*sample_shape, *arr.shape[1:])
        # Return a ``NumericRecord`` rather than a ``NumericRecordArray``
        # for the batched case. ``NumericRecordArray`` would carry a
        # ``batch_shape`` that ``jax.vmap``'s pytree validation rejects
        # when the empirical's ``_sample`` is wrapped inside a vmap'd
        # callable (the array variant treats its leading axes as
        # structural batch dims and refuses to be flattened/unflattened
        # across that axis). Returning the looser ``NumericRecord``
        # form is a deliberate deviation from the WF "uniform output
        # wrap" contract; downstream consumers that need a batched
        # container instead of per-field arrays don't currently exist
        # — every callsite either iterates fields, indexes by name,
        # or calls ``flatten_value`` (which handles both types).
        # Single-field consumers still get the shape shim via the
        # NumericRecord coercion path.
        return NumericRecord(fields)

    # -- moments ------------------------------------------------------------

    def _mean(self) -> NumericRecord:
        return _fieldwise_op(self._record_data, self._w.mean)

    def _variance(self) -> NumericRecord:
        return _fieldwise_op(self._record_data, self._w.variance)

    def _cov(self) -> NumericRecord:
        """Per-field weighted covariance.

        For a 1-D-per-row field the result is a covariance matrix; for
        a scalar field it collapses to a 0-D ``Record`` entry equal to
        the variance.
        """
        return _fieldwise_op(self._record_data, self._w.covariance)

    # -- expectation --------------------------------------------------------

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        """Compute ``E[f(record)]`` over the empirical rows.

        ``f`` is called per row; for single-field auto-wrap empiricals,
        the row is unwrapped to the bare ``jax.Array`` (the user
        constructed from an array, so they expect to operate on
        arrays). Multi-field records pass the row Record as-is.
        """
        single_field = len(self._record_data.fields) == 1
        only_field = self._record_data.fields[0] if single_field else None

        def _row(i):
            r = _index_record(self._record_data, i)
            return r[only_field] if single_field else r

        if num_evaluations is not None and num_evaluations < self._n_samples:
            if key is None:
                key = _auto_key()
            idx = jax.random.choice(
                key, self._n_samples,
                shape=(num_evaluations,), replace=False,
            )
            f_vals = jnp.stack([f(_row(int(i))) for i in idx])
            sub_w = self._w.subsample(idx)
            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                return BootstrapDistribution(f_vals, weights=sub_w)
            return sub_w.mean(f_vals)
        # Exact: evaluate f on every row.
        f_vals = jnp.stack([f(_row(i)) for i in range(self._n_samples)])
        return self._w.mean(f_vals)

    def __repr__(self) -> str:
        return (
            f"RecordEmpiricalDistribution(n={self._n_samples}, "
            f"fields=({', '.join(self._record_data.fields)}))"
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

    Each draw from this distribution is a *bootstrapped dataset* — ``n``
    observations drawn i.i.d. (with replacement) from the source.

    **Source dispatch:**

    - ``Record`` / ``RecordEmpiricalDistribution`` / numeric array /
      numeric-array-backed ``EmpiricalDistribution`` → returns a
      :class:`RecordBootstrapReplicateDistribution`. The numeric array
      path requires ``name=`` (single-field auto-wrap).
    - Any :class:`SupportsSampling` source (e.g. ``Normal``, a custom
      ``Distribution``) → stays in the generic base. ``n`` is mandatory
      because no canonical observation count exists; each replicate is
      ``n`` i.i.d. draws from ``source._sample``.
    - Any other sequence → generic base, equally weighted, with
      object-array storage.

    Parameters
    ----------
    source : Record | EmpiricalDistribution | SupportsSampling | sequence
        Data to bootstrap from.
    n : int or None
        Number of observations per bootstrap dataset. Required when
        ``source`` is a non-array ``SupportsSampling`` (no canonical
        count); defaults to the source's observation count otherwise.
    name : str or None
        Distribution name. Mandatory when ``source`` is a numeric array
        (used as the single-field auto-wrap field name).
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(cls, source=None, *args, **kwargs):
        if cls is BootstrapReplicateDistribution and source is not None:
            if isinstance(source, RecordEmpiricalDistribution):
                return object.__new__(RecordBootstrapReplicateDistribution)
            if isinstance(source, Record):
                return object.__new__(RecordBootstrapReplicateDistribution)
            if _is_numeric_array(source):
                return object.__new__(RecordBootstrapReplicateDistribution)
            # Otherwise (SupportsSampling non-array sources or generic
            # opaque-object sequences) stay in the generic base. Note: a
            # generic ``EmpiricalDistribution(numeric_array)`` can never
            # arrive here as a generic instance — the generic base's own
            # ``__new__`` already routes numeric-array samples to
            # ``RecordEmpiricalDistribution``, which the first branch
            # above catches.
        return object.__new__(cls)

    def __init__(
        self,
        source: Any,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        # SupportsSampling source: each replicate is n i.i.d. draws from
        # source._sample. n is mandatory (no canonical observation count
        # for a generic sampleable source).
        if (
            isinstance(source, SupportsSampling)
            and not isinstance(source, EmpiricalDistribution)
            and not _is_numeric_array(source)
        ):
            if n is None or n < 1:
                raise ValueError(
                    f"BootstrapReplicateDistribution: when source is a "
                    f"SupportsSampling distribution (got "
                    f"{type(source).__name__}), n must be a positive int "
                    f"giving the number of observations per replicate."
                )
            self._source_kind = "sampleable"
            self._source = source
            self._data = None
            self._w = None
            default_n = n
            self._init_bootstrap_state(default_n, n=n, name=name)
            return

        self._source_kind = "data"
        self._source = None
        if isinstance(source, EmpiricalDistribution):
            self._data = source.samples
            self._w = source._w
            default_n = source.n
        elif isinstance(source, (jnp.ndarray, np.ndarray)):
            self._data = source
            if self._data.ndim == 0:
                raise ValueError(
                    "source must have at least 1 dimension (the observation axis)."
                )
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
        source_n: int | None = None,
    ) -> None:
        if n is None:
            self._n = default_n
        else:
            if n < 1:
                raise ValueError(f"n must be positive, got {n}")
            self._n = n
        if name is None:
            name = "bootstrap"
        super().__init__(name=name)
        if self._source_kind == "sampleable":
            self._source_n = None
        else:
            # ``source_n`` is the actual source observation count.
            # Fall back to ``default_n`` when the caller didn't pass it
            # (matches the old behaviour where source_n == default_n).
            self._source_n = source_n if source_n is not None else default_n
        self._approximate = True

    # -- properties ---------------------------------------------------------

    @property
    def n(self) -> int:
        """Observations per bootstrap dataset."""
        return self._n

    @property
    def source_n(self) -> int | None:
        """Number of source observations, or ``None`` for a sampleable source."""
        return self._source_n

    @property
    def data(self) -> Any:
        """Source data (``None`` for a sampleable source)."""
        return self._data

    @property
    def weights(self) -> Array:
        """Source weights (``None`` for a sampleable source)."""
        return None if self._w is None else self._w.normalized

    @property
    def is_uniform(self) -> bool:
        """True when source observations are equally weighted."""
        return True if self._w is None else self._w.is_uniform

    @property
    def _is_object_data(self) -> bool:
        return (
            self._data is not None
            and isinstance(self._data, np.ndarray)
            and self._data.dtype == object
        )

    # -- sampling -----------------------------------------------------------

    def _one_bootstrap(self, key: PRNGKey) -> Any:
        if self._source_kind == "sampleable":
            return self._source._sample(key, sample_shape=(self._n,))
        idx = self._w.choice(key, shape=(self._n,))
        return self._data[idx]

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Any:
        if sample_shape == ():
            return self._one_bootstrap(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        if self._is_object_data:
            results = np.empty(total, dtype=object)
            for i in range(total):
                results[i] = self._one_bootstrap(keys[i])
            return results.reshape(sample_shape)
        # Non-object data: vmap is safe for both ``data`` and
        # ``sampleable`` source kinds (the latter calls source._sample
        # under vmap, the former does weighted-choice + slicing).
        results = jax.vmap(self._one_bootstrap)(keys)
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

        ``f`` receives one bootstrapped dataset per call. For
        Record-shaped sources with a single field (the auto-wrap case),
        the dataset is unwrapped to the bare stacked ``jax.Array`` so
        ``f`` operates on arrays directly.
        """
        if key is None:
            key = _auto_key()
        if num_evaluations is None:
            num_evaluations = _base.DEFAULT_NUM_EVALUATIONS
        keys = jax.random.split(key, num_evaluations)

        record_data = getattr(self, "_record_data", None)
        single_field = (
            record_data is not None
            and len(record_data.fields) == 1
        )
        only_field = record_data.fields[0] if single_field else None

        def _ds(k):
            d = self._one_bootstrap(k)
            if single_field and isinstance(d, Record):
                return d[only_field]
            return d

        f_vals = jnp.stack([f(_ds(k)) for k in keys])
        rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
        if rd:
            return BootstrapDistribution(f_vals)
        return jnp.mean(f_vals, axis=0)

    def __repr__(self) -> str:
        if self._source_kind == "sampleable":
            return (
                f"BootstrapReplicateDistribution(n={self._n}, "
                f"source={type(self._source).__name__})"
            )
        return (
            f"BootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n})"
        )


# ---------------------------------------------------------------------------
# RecordBootstrapReplicateDistribution (Record specialisation)
# ---------------------------------------------------------------------------


class RecordBootstrapReplicateDistribution(
    BootstrapReplicateDistribution[Record],
    NumericRecordDistribution,
):
    """Bootstrap replicate distribution over Record-structured data.

    Each sample is a full bootstrapped dataset: ``n`` rows drawn i.i.d.
    with replacement from the source data, with the same row indices
    applied jointly across fields.

    Inherits :class:`NumericRecordDistribution` shape semantics
    (``record_template``, ``event_shapes``, ...). A bare numeric array
    source auto-wraps as a single-field Record keyed by ``name`` —
    matching the migration path for the previous
    ``ArrayBootstrapReplicateDistribution(arr)`` form.

    Parameters
    ----------
    source : Record | RecordEmpiricalDistribution | EmpiricalDistribution | array-like
        Data to bootstrap from.
    n : int or None
        Observations per bootstrap dataset. Defaults to the source's
        observation count.
    name : str or None
        Distribution name. Mandatory when *source* is a bare numeric
        array (used as the single-field auto-wrap field name).
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        source: Any,
        *,
        n: int | None = None,
        name: str | None = None,
    ):
        if isinstance(source, RecordEmpiricalDistribution):
            self._record_data = source._record_data
            self._w = source._w
            default_n = source.n
        elif isinstance(source, Record):
            n_rows = _validate_record_samples(source)
            self._record_data = source
            default_n = n_rows
            self._w = Weights.uniform(n_rows)
        elif isinstance(source, EmpiricalDistribution):
            # Numeric-array-backed EmpiricalDistribution: dispatch wrapped
            # us here. Auto-wrap the stacked array as a single-field
            # Record using the source's name.
            arr = source.samples
            field_name = name or source.name or "data"
            wrapped, field_name = _wrap_numeric_array_as_record(
                arr, name=field_name,
                role="RecordBootstrapReplicateDistribution",
            )
            self._record_data = wrapped
            self._w = source._w
            default_n = source.n
            if name is None:
                name = field_name
        elif _is_numeric_array(source):
            wrapped, field_name = _wrap_numeric_array_as_record(
                source, name=name,
                role="RecordBootstrapReplicateDistribution",
            )
            self._record_data = wrapped
            n_rows = _validate_record_samples(wrapped)
            default_n = n_rows
            self._w = Weights.uniform(n_rows)
            name = field_name
        else:
            raise TypeError(
                f"RecordBootstrapReplicateDistribution: source must be a "
                f"Record, RecordEmpiricalDistribution, numeric array, or "
                f"numeric-array-backed EmpiricalDistribution, got "
                f"{type(source).__name__}"
            )
        # Bootstrap-base bookkeeping. Set self._data so the base's
        # `.data` property returns the Record (matches old behaviour).
        self._source_kind = "data"
        self._source = None
        self._data = self._record_data
        self._init_bootstrap_state(
            default_n, n=n, name=name, source_n=default_n,
        )
        # Replicate produces (n, *event_shape) per field; advertise that
        # via the record_template.
        self._record_template = _record_template_from_data(
            self._record_data, leading_shape=(self._n,),
        )

    # -- shape ---------------------------------------------------------------

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field replicate event shapes ``(n, *obs_event_shape)``."""
        return {
            f: (self._n, *jnp.asarray(self._record_data[f]).shape[1:])
            for f in self._record_data.fields
        }

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        return {
            f: jnp.asarray(self._record_data[f]).dtype
            for f in self._record_data.fields
        }

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Single-field shortcut. Multi-field replicates use ``event_shapes``."""
        if len(self._record_data.fields) == 1:
            f = self._record_data.fields[0]
            return (self._n, *jnp.asarray(self._record_data[f]).shape[1:])
        return ()

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Per-observation event shape for a single-field replicate.

        Multi-field replicates use ``obs_shapes`` (per-field).
        """
        if len(self._record_data.fields) == 1:
            f = self._record_data.fields[0]
            return tuple(jnp.asarray(self._record_data[f]).shape[1:])
        return ()

    @property
    def obs_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field observation event shapes (replicate axis stripped)."""
        return {
            f: tuple(jnp.asarray(self._record_data[f]).shape[1:])
            for f in self._record_data.fields
        }

    @property
    def dim(self) -> int:
        """Flat dimensionality of a single bootstrap dataset.

        Sum across fields of ``n * max(1, prod(obs_event_shape))``.
        """
        return sum(
            self._n * max(1, prod(shape))
            for shape in self.obs_shapes.values()
        )

    @property
    def support(self) -> Constraint:
        return real

    @property
    def supports(self) -> dict[str, Constraint]:
        return {f: real for f in self._record_data.fields}

    # -- sampling -----------------------------------------------------------

    def _one_bootstrap(self, key: PRNGKey) -> NumericRecord:
        idx = self._w.choice(key, shape=(self._n,))
        return _index_record(self._record_data, idx)

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> NumericRecord:
        if sample_shape == ():
            return self._one_bootstrap(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)
        results = [self._one_bootstrap(k) for k in keys]
        stacked: dict[str, jnp.ndarray] = {}
        for f in self._record_data.fields:
            arrs = jnp.stack([jnp.asarray(r[f]) for r in results])
            stacked[f] = arrs.reshape(*sample_shape, *arrs.shape[1:])
        return NumericRecord(stacked)

    def __repr__(self) -> str:
        return (
            f"RecordBootstrapReplicateDistribution(n={self._n}, "
            f"source_n={self._source_n}, "
            f"fields=({', '.join(self._record_data.fields)}))"
        )
