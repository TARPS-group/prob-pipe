"""Broadcast distribution and marginal types.

Provides:
  - ``_ArrayMarginal``                     – Array output marginal.
  - ``_MixtureMarginal[T]``                – Distribution output marginal (mixture).
  - ``_ListMarginal[T]``                   – Non-stackable output marginal.
  - ``MarginalizedBroadcastDistribution``  – Union type alias.
  - ``_make_marginal()``                   – Factory for marginal construction.
  - ``BroadcastDistribution``              – Joint over broadcast inputs and output.
"""

from __future__ import annotations

from typing import Any

from .._utils import prod
from .protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array
from .._weights import Weights
from ._distribution_base import Distribution
from ._record_distribution import RecordDistribution
from ._empirical import (
    NumericEmpiricalDistribution,
    EmpiricalDistribution,
    _RecordEmpiricalDistribution,
)
from ._record_array import RecordArray
from .record import Record, RecordTemplate


# ---------------------------------------------------------------------------
# MarginalizedBroadcastDistribution — output marginal of a broadcast
# ---------------------------------------------------------------------------
#
# Protocol support is determined dynamically via a factory function that
# picks the right concrete subclass, so ``isinstance`` checks are truthful.
# ---------------------------------------------------------------------------


class _ArrayMarginal(NumericEmpiricalDistribution):
    """Output marginal when broadcast outputs are stackable arrays.

    Inherits from :class:`NumericEmpiricalDistribution` for weighted
    resampling and exact weighted moments.
    """

    def __init__(
        self,
        samples: Array,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        super().__init__(samples, weights=weights, log_weights=log_weights, name=name)

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(n={self.n}, event_shape={self.event_shape})"


class _MixtureMarginal[T](Distribution[T]):
    """Output marginal when broadcast outputs are Distribution objects.

    Acts as a finite mixture: ``p(y) = Σ_i w_i p_i(y)``.  Protocol support
    depends on what the component distributions support.

    This base class provides no protocol methods.  The factory
    :func:`_make_mixture_marginal` dynamically constructs a subclass that
    inherits the appropriate protocol mixins.
    """

    def __init__(
        self,
        components: list,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        n = len(components)
        self._components = components
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        if name is None:
            name = "mixture_marginal"
        super().__init__(name=name)
        self._approximate = True

    @property
    def n(self) -> int:
        return len(self._components)

    @property
    def components(self) -> list:
        return self._components

    @property
    def weights(self) -> Array:
        return self._w.normalized

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(mixture, n={self.n})"


# -- Mixture protocol mixins (combined dynamically) -------------------------

class _MixtureSampling:
    """SupportsSampling mixin for mixture marginals.

    Returns a raw ``Array`` when component samples are arrays (the
    common case — broadcasting a numeric function over a distribution
    of inputs), and a ``RecordArray`` when component samples are
    ``Record``-valued (e.g., broadcasting a ``Record``-returning
    ``WorkflowFunction``). Opaque / non-stackable component outputs
    raise a ``TypeError`` with the component types listed.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        from ._record_array import RecordArray
        from .record import Record

        n_draws = prod(sample_shape) if sample_shape else 1
        key1, key2 = jax.random.split(key)
        indices = self._w.choice(key1, shape=(n_draws,))
        keys = jax.random.split(key2, n_draws)

        results = [
            self._components[int(indices[i])]._sample(keys[i], ())
            for i in range(n_draws)
        ]

        # Dispatch on result type so a mixture of Record-returning
        # distributions produces a RecordArray rather than crashing in
        # jnp.stack. Scalars / arrays stay on the numeric path. Exclude
        # RecordArray leaves here — ``RecordArray.stack`` expects
        # scalar Records; a mixture over already-batched Record
        # samples isn't supported on this path.
        if all(
            isinstance(r, Record) and not isinstance(r, RecordArray)
            for r in results
        ):
            stacked_ra = RecordArray.stack(results)
            if sample_shape == ():
                return stacked_ra[0]
            # Reshape the leading batch axis to sample_shape.
            fields = {
                name: stacked_ra[name].reshape(sample_shape + stacked_ra[name].shape[1:])
                for name in stacked_ra.fields
            }
            return type(stacked_ra)(
                fields, batch_shape=sample_shape, template=stacked_ra.template,
            )

        try:
            stacked = jnp.stack(results, axis=0)
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(r).__name__ for r in results})
            raise TypeError(
                f"_MixtureSampling cannot stack component samples of "
                f"types {types_seen}; mixture marginals support numeric "
                f"arrays and Record values only."
            ) from exc
        if sample_shape == ():
            return stacked[0]
        return stacked.reshape(sample_shape + stacked.shape[1:])


class _MixtureMean:
    """SupportsMean mixin for mixture marginals."""

    def _mean(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        return self._w.mean(means)


class _MixtureVariance:
    """SupportsVariance mixin for mixture marginals (law of total variance)."""

    def _variance(self):
        means = jnp.stack([c._mean() for c in self._components], axis=0)
        variances = jnp.stack([c._variance() for c in self._components], axis=0)
        overall_mean = self._w.mean(means)
        # Law of total variance: E[Var(Y|X)] + Var(E[Y|X])
        e_var = self._w.mean(variances)
        diff = means - overall_mean
        var_e = self._w.mean(diff ** 2)
        return e_var + var_e


class _MixtureLogProb:
    """SupportsLogProb mixin for mixture marginals."""

    def _log_prob(self, value):
        log_w = self._w.log_normalized
        component_lps = jnp.stack(
            [c._log_prob(value) for c in self._components], axis=0
        )
        return jax.scipy.special.logsumexp(log_w + component_lps)


# Map protocol → (mixin class, required component protocols)
_MIXTURE_PROTOCOL_MAP: list[tuple[type, type, tuple[type, ...]]] = [
    (SupportsSampling, _MixtureSampling, (SupportsSampling,)),
    (SupportsMean, _MixtureMean, (SupportsMean,)),
    (SupportsVariance, _MixtureVariance, (SupportsMean, SupportsVariance)),
    (SupportsLogProb, _MixtureLogProb, (SupportsLogProb,)),
]

# Cache dynamically created classes to avoid repeated type() calls
_mixture_class_cache: dict[tuple[type, ...], type] = {}


def _make_mixture_marginal(
    components: list,
    weights: Array | Weights | None = None,
    *,
    name: str | None = None,
) -> _MixtureMarginal:
    """Factory that builds a mixture marginal with dynamic protocol support.

    Inspects the component distributions to determine which protocols they
    all support, then creates (and caches) a concrete subclass that inherits
    the corresponding mixin classes.
    """
    # Determine which protocols all components support
    active_protocols: list[type] = []
    active_mixins: list[type] = []
    for protocol, mixin, required in _MIXTURE_PROTOCOL_MAP:
        if all(isinstance(c, req) for c in components for req in required):
            active_protocols.append(protocol)
            active_mixins.append(mixin)

    cache_key = tuple(active_protocols)
    if cache_key not in _mixture_class_cache:
        bases = tuple(active_mixins) + (_MixtureMarginal,) + tuple(active_protocols)
        cls_name = "_DynMixtureMarginal"
        _mixture_class_cache[cache_key] = type(cls_name, bases, {})

    cls = _mixture_class_cache[cache_key]
    obj = object.__new__(cls)
    _MixtureMarginal.__init__(obj, components, weights, name=name)
    return obj


class _RecordArrayMarginal(_RecordEmpiricalDistribution):
    """Output marginal when broadcast outputs are Records.

    Wraps a :class:`RecordArray` as a Record-valued empirical distribution
    with per-field weighted mean, variance, and resampling.
    """

    def __init__(
        self,
        record_array: RecordArray,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        if not isinstance(record_array, RecordArray):
            raise TypeError(
                f"Expected RecordArray, got {type(record_array).__name__}"
            )
        # RecordArray is structurally a Record with batched fields; wrap it
        # directly so _RecordEmpiricalDistribution sees one row per batch index.
        samples = Record({f: record_array[f] for f in record_array.fields})
        super().__init__(samples, weights=weights, log_weights=log_weights, name=name)
        # Preserve the original template so event_shapes stay accurate
        self._record_template = record_array.template

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(n={self.n}, fields={self._record_data.fields})"


class _ListMarginal[T](Distribution[T]):
    """Output marginal when broadcast outputs are non-stackable (e.g., strings).

    No protocol support — outputs cannot be resampled or summarised.
    """

    def __init__(
        self,
        items: list,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        self._items = items
        self._w = Weights(n=len(items), weights=weights, log_weights=log_weights)
        if name is None:
            name = "list_marginal"
        super().__init__(name=name)

    @property
    def n(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list:
        return self._items

    @property
    def weights(self) -> Array:
        return self._w.normalized

    def __repr__(self):
        return f"MarginalizedBroadcastDistribution(list, n={self.n})"


# Public alias for type checking / isinstance
MarginalizedBroadcastDistribution = _ArrayMarginal | _RecordArrayMarginal | _MixtureMarginal | _ListMarginal
"""Union type for the output marginal of a :class:`BroadcastDistribution`.

Concrete subtype depends on output kind:

- :class:`_ArrayMarginal` — stackable array outputs
- :class:`_MixtureMarginal` — distribution outputs (mixture)
- :class:`_ListMarginal` — non-stackable outputs
"""


def _make_marginal(
    output_samples: Any,
    weights: Array | Weights | None = None,
    *,
    output_distributions: list | None = None,
    name: str | None = None,
) -> MarginalizedBroadcastDistribution:
    """Factory to construct the appropriate marginal subtype."""
    if output_distributions is not None:
        return _make_mixture_marginal(output_distributions, weights, name=name)

    if isinstance(output_samples, RecordArray):
        return _RecordArrayMarginal(output_samples, weights, name=name)

    # Record with batched leaves (e.g., from jax.vmap over a Record-returning fn).
    # All fields must be arrays with a consistent leading batch dimension.
    # Exclude RecordArray — the earlier branch handles it already, but
    # since ``RecordArray`` is a ``Record`` subclass we'd otherwise
    # match here for a RecordArray whose fields have extra axes.
    if (
        isinstance(output_samples, Record)
        and not isinstance(output_samples, RecordArray)
        and output_samples.fields
    ):
        resolved = [output_samples[f] for f in output_samples.fields]
        if all(hasattr(v, "ndim") and v.ndim > 0 for v in resolved):
            n = resolved[0].shape[0]
            if all(v.shape[0] == n for v in resolved):
                tpl = RecordTemplate.from_record(output_samples, batch_shape=(n,))
                fields = dict(zip(output_samples.fields, resolved))
                ra = RecordArray(fields, batch_shape=(n,), template=tpl)
                return _RecordArrayMarginal(ra, weights, name=name)

    if isinstance(output_samples, jnp.ndarray):
        return _ArrayMarginal(output_samples, weights, name=name)

    if isinstance(output_samples, list):
        if output_samples and all(
            isinstance(r, Record) and not isinstance(r, RecordArray)
            for r in output_samples
        ):
            try:
                ra = RecordArray.stack(output_samples)
                return _RecordArrayMarginal(ra, weights, name=name)
            except (ValueError, TypeError):
                pass
        try:
            stacked = jnp.stack(
                [jnp.asarray(r, dtype=jnp.float32) for r in output_samples], axis=0
            )
            return _ArrayMarginal(stacked, weights, name=name)
        except (ValueError, TypeError):
            pass
        # Check if all results are distributions
        if output_samples and all(isinstance(r, Distribution) for r in output_samples):
            return _make_mixture_marginal(output_samples, weights, name=name)
        return _ListMarginal(output_samples, weights, name=name)

    # Single array result (e.g., from vmap); ensure at least 1D for the sample axis
    arr = jnp.atleast_1d(jnp.asarray(output_samples, dtype=jnp.float32))
    return _ArrayMarginal(arr, weights, name=name)


# ---------------------------------------------------------------------------
# _make_stack — stacked sibling of _make_marginal for RecordArray broadcasts
# ---------------------------------------------------------------------------
#
# When a WorkflowFunction broadcasts over a RecordArray (parameter
# sweep), the n inner outputs are independent scenarios indexed by
# input row — *not* MC draws. The wrapper must preserve row identity:
#
#   numeric → NumericRecordArray(result=..., batch_shape=(n,))
#   Record → RecordArray.stack (NumericRecordArray when all leaves numeric)
#   Distribution → DistributionArray
#   RecordArray (per row batch_shape=(m,)) → RecordArray(batch_shape=(n, m))
#
# Opaque Python values (e.g. strings) that can't be stacked fall
# through to a plain-list wrapping with a clear error if even that
# fails.
#
# Caller attaches ``.with_source(...)`` externally via ``_coerce_output``.
# ---------------------------------------------------------------------------


def _make_stack(
    inner_outputs: Any,
    *,
    batch_shape: tuple[int, ...] | None = None,
    n: int | None = None,
    name: str | None = None,
    field_name: str,
) -> Any:
    """Wrap inner workflow-function outputs as a shape-``batch_shape``
    aggregate.

    Internally the outputs are aggregated along a single leading axis
    of length ``prod(batch_shape)``; the final aggregate reshapes that
    axis to ``batch_shape`` so multi-d sweeps produce multi-d output
    shapes.

    Dispatch on ``inner_outputs`` — either a Python ``list`` of length
    ``prod(batch_shape)`` (Python-loop execution path) or a pytree with
    a leading axis of length ``prod(batch_shape)`` (``jax.vmap``
    execution path).

    Parameters
    ----------
    inner_outputs : list or pytree
        Either a list of inner-function results, or a single stackable
        pytree with a leading axis equal to ``prod(batch_shape)``.
    batch_shape : tuple of int, optional
        Shape of the output aggregate's leading axes. Pass either
        ``batch_shape`` or ``n`` (the 1-D shortcut); exactly one.
    n : int, optional
        Shortcut for ``batch_shape=(n,)``.
    name : str, optional
        Name for the resulting aggregate.

    Returns
    -------
    NumericRecordArray | RecordArray | DistributionArray
        Output type depends on the inner-return type; see module
        docstring for the dispatch table.

    Raises
    ------
    TypeError
        If the inner outputs can't be coerced into any of the three
        aggregate types. The error lists the observed types.
    """
    from ._distribution_array import _make_distribution_array
    from .record import Record, RecordTemplate

    # Resolve batch_shape vs. n. Exactly one must be provided.
    if batch_shape is None and n is None:
        raise TypeError("_make_stack requires either batch_shape or n")
    if batch_shape is not None and n is not None:
        raise TypeError("_make_stack: pass batch_shape OR n, not both")
    if batch_shape is None:
        batch_shape = (n,)
    batch_shape = tuple(batch_shape)
    n_total = int(prod(batch_shape)) if batch_shape else 1

    # --- List-of-X path (Python-loop execution) -------------------------
    if isinstance(inner_outputs, list):
        if len(inner_outputs) != n_total:
            raise ValueError(
                f"_make_stack got {len(inner_outputs)} outputs but "
                f"expected prod(batch_shape)={n_total} "
                f"(batch_shape={batch_shape})."
            )
        outs = inner_outputs

        # Check the more-specific subclass first: all RecordArrays
        # (since ``RecordArray`` is itself a ``Record`` subclass, the
        # generic Record branch below would otherwise claim them and
        # collapse the inner batch axis).
        if outs and all(isinstance(o, RecordArray) for o in outs):
            first = outs[0]
            if all(ra.batch_shape == first.batch_shape for ra in outs):
                fields = {
                    fname: jnp.stack([ra[fname] for ra in outs], axis=0)
                    for fname in first.fields
                }
                # Reshape the leading (n_total,) axis to batch_shape.
                reshaped = {
                    fname: arr.reshape(batch_shape + arr.shape[1:])
                    for fname, arr in fields.items()
                }
                return type(first)(
                    reshaped,
                    batch_shape=batch_shape + first.batch_shape,
                    template=first.template,
                )
            # Mismatched inner shapes fall through to the generic
            # Record / list handlers.

        # All (scalar) Records → stack as a RecordArray. NumericRecordArray
        # if every leaf is numeric; otherwise fall back to the permissive
        # RecordArray class, building the fields manually so non-numeric
        # leaves (strings, xarray objects, ...) survive.
        if outs and all(
            isinstance(o, Record) and not isinstance(o, RecordArray)
            for o in outs
        ):
            # Stack flat, then reshape to batch_shape.
            try:
                from ._record_array import NumericRecordArray
                flat = NumericRecordArray.stack(list(outs))
            except (TypeError, ValueError):
                flat = None
            if flat is not None:
                if batch_shape == (n_total,):
                    return flat
                # Reshape each field's leading axis.
                n_cur = len(flat.batch_shape)
                new_fields = {
                    fname: flat[fname].reshape(
                        batch_shape + flat[fname].shape[n_cur:]
                    )
                    for fname in flat.fields
                }
                return type(flat)(
                    new_fields,
                    batch_shape=batch_shape,
                    template=flat.template,
                )
            # Manual per-field assembly: numpy-array-like leaves stack
            # numerically, object-dtype leaves use np.asarray(..., dtype=object).
            first = outs[0]
            if any(o.fields != first.fields for o in outs):
                raise TypeError(
                    "_make_stack: Records in list have inconsistent fields."
                )
            fields: dict[str, Any] = {}
            for fname in first.fields:
                values = [o[fname] for o in outs]
                try:
                    stacked = jnp.stack(values, axis=0)
                    fields[fname] = stacked.reshape(batch_shape + stacked.shape[1:])
                except (TypeError, ValueError):
                    arr = np.asarray(values, dtype=object)
                    fields[fname] = arr.reshape(batch_shape + arr.shape[1:])
            tpl_spec: dict[str, Any] = {}
            for fname, v in fields.items():
                if hasattr(v, "dtype") and getattr(v.dtype, "kind", None) in "biufc":
                    tpl_spec[fname] = tuple(v.shape[len(batch_shape):])
                else:
                    tpl_spec[fname] = None
            return RecordArray(
                fields,
                batch_shape=batch_shape,
                template=RecordTemplate(tpl_spec),
            )

        # All Distributions → stacked DistributionArray, shaped to
        # batch_shape.
        if outs and all(isinstance(o, Distribution) for o in outs):
            return _make_distribution_array(
                outs, batch_shape=batch_shape, name=name,
            )

        # Numeric scalars / arrays → wrap in NumericRecordArray with
        # the single "result" field carrying the stacked values,
        # reshape leading axis to batch_shape.
        try:
            stacked = jnp.stack(
                [jnp.asarray(o) for o in outs], axis=0,
            )
        except (TypeError, ValueError):
            stacked = None

        if stacked is not None:
            from ._record_array import NumericRecordArray
            event_shape = tuple(stacked.shape[1:])
            reshaped = stacked.reshape(batch_shape + event_shape)
            tpl = RecordTemplate(**{field_name: event_shape})
            return NumericRecordArray(
                {field_name: reshaped},
                batch_shape=batch_shape,
                template=tpl,
            )

        # Last-ditch: wrap as a RecordArray whose single field holds a
        # numpy object-dtype array of the opaque outputs.
        try:
            object_array = np.asarray(outs, dtype=object).reshape(batch_shape)
            tpl = RecordTemplate(**{field_name: None})
            return RecordArray(
                {field_name: object_array},
                batch_shape=batch_shape,
                template=tpl,
            )
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(o).__name__ for o in outs})
            raise TypeError(
                f"_make_stack cannot aggregate outputs of types "
                f"{types_seen}; supported: numeric arrays, Record, "
                f"RecordArray, Distribution."
            ) from exc

    # --- Single-pytree path (jax.vmap execution) ------------------------

    # vmap of a numeric-returning function produces a jnp.ndarray with
    # leading axis of length n_total. Reshape to batch_shape.
    if isinstance(inner_outputs, jnp.ndarray):
        if inner_outputs.shape[:1] != (n_total,):
            raise ValueError(
                f"_make_stack got array of shape {inner_outputs.shape} but "
                f"expected leading axis of length {n_total} "
                f"(batch_shape={batch_shape})."
            )
        from ._record_array import NumericRecordArray
        event_shape = tuple(inner_outputs.shape[1:])
        reshaped = inner_outputs.reshape(batch_shape + event_shape)
        tpl = RecordTemplate(**{field_name: event_shape})
        return NumericRecordArray(
            {field_name: reshaped},
            batch_shape=batch_shape,
            template=tpl,
        )

    # vmap of a Record-returning function produces a Record with
    # batched leaves (each leaf has leading axis n_total). Promote to a
    # RecordArray — NumericRecordArray when all leaves numeric — with
    # the leading axis reshaped to batch_shape.
    if (
        isinstance(inner_outputs, Record)
        and not isinstance(inner_outputs, RecordArray)
        and inner_outputs.fields
    ):
        resolved = [inner_outputs[f] for f in inner_outputs.fields]
        if all(hasattr(v, "shape") and v.shape[:1] == (n_total,) for v in resolved):
            event_shapes = tuple(v.shape[1:] for v in resolved)
            tpl = RecordTemplate(**dict(zip(inner_outputs.fields, event_shapes)))
            reshaped_fields = {
                fname: v.reshape(batch_shape + v.shape[1:])
                for fname, v in zip(inner_outputs.fields, resolved)
            }
            try:
                from ._record_array import NumericRecordArray
                return NumericRecordArray(
                    reshaped_fields, batch_shape=batch_shape, template=tpl,
                )
            except (TypeError, ValueError):
                return RecordArray(
                    reshaped_fields, batch_shape=batch_shape, template=tpl,
                )

    # Fallback — shouldn't reach here with well-formed vmap output; if
    # we do, raise with the type info.
    raise TypeError(
        f"_make_stack cannot aggregate output of type "
        f"{type(inner_outputs).__name__}; expected a list, jnp.ndarray, "
        f"or batched Record."
    )


# ---------------------------------------------------------------------------
# BroadcastDistribution — joint over broadcast inputs and function output
# ---------------------------------------------------------------------------


class BroadcastDistribution(Distribution[dict], SupportsSampling):
    """Joint distribution over broadcast inputs and function output.

    Stores the paired input–output samples from a
    :class:`~probpipe.core.node.WorkflowFunction` broadcast.  Supports
    joint sampling (resampling paired input–output tuples) and named
    component access.

    Call :meth:`marginalize` to obtain the output-only marginal, which
    supports moment protocols (mean, variance, etc.) when the output
    data permits.

    .. note::

       ``BroadcastDistribution`` does **not** inherit from
       :class:`~probpipe.distributions.joint.JointDistribution`.
       ``JointDistribution`` requires all leaves to be
       ``NumericRecordDistribution`` instances with TFP shape semantics
       (``batch_shape``, ``event_shape``), but a broadcast output can be
       any type — arrays, distributions, strings, etc. — and input
       samples are plain arrays without distribution metadata.  The two
       hierarchies serve different roles: ``JointDistribution`` models
       structured probabilistic variables; ``BroadcastDistribution``
       records the empirical input–output mapping of a function
       evaluation.

    Parameters
    ----------
    input_samples : dict[str, Array]
        ``{arg_name: (n, *event_shape)}`` for each broadcast argument.
    output_samples : Array or list
        ``(n, *event_shape)`` for array outputs, or a list of length *n*.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.  ``None`` for uniform.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    output_distributions : list of Distribution or None
        When each function evaluation returns a ``Distribution``, these
        are the *n* component distributions for the mixture marginal.
    broadcast_args : list of str
        Ordered names of the broadcast arguments.
    name : str or None
        Distribution name for provenance.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        input_samples: dict[str, Any],
        output_samples: Any,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        output_distributions: list | None = None,
        broadcast_args: list[str],
        name: str | None = None,
    ):
        self._input_samples = input_samples
        self._output_samples = output_samples
        self._output_distributions = output_distributions

        # Determine n from first broadcast arg
        first_key = list(broadcast_args)[0]
        first_arr = input_samples[first_key]
        n = first_arr.shape[0] if hasattr(first_arr, 'shape') else len(first_arr)
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        self._broadcast_args = list(broadcast_args)
        if name is None:
            name = "broadcast"
        super().__init__(name=name)
        self._approximate = True
        self._marginal_cache: MarginalizedBroadcastDistribution | None = None

    # -- basic properties ---------------------------------------------------

    @property
    def n(self) -> int:
        """Number of input–output pairs."""
        return self._w.n

    @property
    def weights(self) -> Array:
        """Normalised weights, shape ``(n,)``."""
        return self._w.normalized

    @property
    def input_samples(self) -> dict[str, Any]:
        """Broadcast input samples: ``{arg_name: (n, *event_shape)}``."""
        return self._input_samples

    @property
    def samples(self) -> Any:
        """Output samples (forwarded to output marginal for backward compat)."""
        m = self.marginalize()
        return m.samples if hasattr(m, 'samples') else m.items

    # -- Named components ----------------------------------------------------

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self._broadcast_args) + ("_output",)

    def __getitem__(self, key: str):
        if key == "_output":
            return self.marginalize()
        if key in self._input_samples:
            arr = self._input_samples[key]
            return EmpiricalDistribution(arr, weights=self._w)
        raise KeyError(f"Unknown component {key!r}; available: {self.component_names}")

    # -- joint sampling -----------------------------------------------------

    def _sample(self, key, sample_shape=()):
        """Resample paired input–output tuples."""
        n_draws = prod(sample_shape) if sample_shape else 1
        indices = self._w.choice(key, shape=(n_draws,))

        result = {}
        for arg_name in self._broadcast_args:
            arr = self._input_samples[arg_name]
            result[arg_name] = arr[indices]

        # Output
        if isinstance(self._output_samples, jnp.ndarray):
            result["_output"] = self._output_samples[indices]
        elif isinstance(self._output_samples, list):
            result["_output"] = [self._output_samples[int(i)] for i in indices]
        else:
            result["_output"] = self._output_samples[indices]

        if sample_shape == ():
            return jax.tree.map(lambda x: x[0] if hasattr(x, '__getitem__') else x, result)
        return result

    # -- marginalization ----------------------------------------------------

    def marginalize(self) -> MarginalizedBroadcastDistribution:
        """Return the output marginal distribution.

        Lazy — the marginal is constructed on first call and cached.
        The marginal inherits this distribution's provenance (if any)
        so the lineage is preserved without a direct reference to the
        ``BroadcastDistribution``.
        """
        if self._marginal_cache is None:
            self._marginal_cache = _make_marginal(
                self._output_samples,
                self._w,
                output_distributions=self._output_distributions,
            )
            if self.source is not None and isinstance(self._marginal_cache, Distribution):
                self._marginal_cache.with_source(self.source)
        return self._marginal_cache

    @property
    def output(self) -> MarginalizedBroadcastDistribution:
        """Alias for :meth:`marginalize`."""
        return self.marginalize()

    def __repr__(self):
        return (
            f"BroadcastDistribution(n={self.n}, "
            f"broadcast_args={self._broadcast_args})"
        )
