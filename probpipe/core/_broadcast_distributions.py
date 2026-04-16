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

from ..custom_types import Array
from .._weights import Weights
from ._distribution_base import Distribution
from ._record_distribution import RecordDistribution
from ._empirical import (
    ArrayEmpiricalDistribution,
    EmpiricalDistribution,
)


# ---------------------------------------------------------------------------
# MarginalizedBroadcastDistribution — output marginal of a broadcast
# ---------------------------------------------------------------------------
#
# Protocol support is determined dynamically via a factory function that
# picks the right concrete subclass, so ``isinstance`` checks are truthful.
# ---------------------------------------------------------------------------


class _ArrayMarginal(ArrayEmpiricalDistribution):
    """Output marginal when broadcast outputs are stackable arrays.

    Inherits from :class:`ArrayEmpiricalDistribution` for weighted
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
    """SupportsSampling mixin for mixture marginals."""

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        n_draws = prod(sample_shape) if sample_shape else 1
        key1, key2 = jax.random.split(key)
        indices = self._w.choice(key1, shape=(n_draws,))
        keys = jax.random.split(key2, n_draws)

        results = []
        for i in range(n_draws):
            comp = self._components[int(indices[i])]
            results.append(comp._sample(keys[i], ()))
        stacked = jnp.stack(results, axis=0)
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


class _RecordArrayMarginal(
    RecordDistribution, SupportsSampling, SupportsMean, SupportsVariance,
):
    """Output marginal when broadcast outputs are Records.

    Wraps a :class:`RecordArray` and provides per-field weighted
    mean, variance, and resampling.
    """

    def __init__(
        self,
        record_array,
        weights: Array | Weights | None = None,
        *,
        log_weights: Array | Weights | None = None,
        name: str | None = None,
    ):
        from ._record_array import RecordArray
        if not isinstance(record_array, RecordArray):
            raise TypeError(
                f"Expected RecordArray, got {type(record_array).__name__}"
            )
        self._record_array = record_array
        n = len(record_array)
        self._w = Weights(n=n, weights=weights, log_weights=log_weights)
        if name is None:
            name = "record_array_marginal"
        super().__init__(name=name)
        self._record_template = record_array.template
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    @property
    def n(self) -> int:
        return len(self._record_array)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self._record_array.template.flat_size,)

    def _mean(self):
        from .record import Record
        fields = {}
        for f in self._record_array.fields:
            fields[f] = self._w.mean(self._record_array[f])
        return Record(fields)

    def _variance(self):
        from .record import Record
        fields = {}
        for f in self._record_array.fields:
            fields[f] = self._w.variance(self._record_array[f])
        return Record(fields)

    def _sample_one(self, key):
        idx = self._w.choice(key)
        return self._record_array[idx]

    def _sample(self, key, sample_shape=()):
        from ._record_array import RecordArray
        if sample_shape == ():
            return self._sample_one(key)
        n_draws = prod(sample_shape)
        indices = self._w.choice(key, shape=(n_draws,))
        fields = {}
        for f in self._record_array.fields:
            arr = self._record_array[f]
            drawn = arr[indices]
            if sample_shape != (n_draws,):
                drawn = drawn.reshape(sample_shape + arr.shape[1:])
            fields[f] = drawn
        return RecordArray(
            fields,
            batch_shape=sample_shape,
            template=self._record_array.template,
        )

    def __repr__(self):
        return (
            f"MarginalizedBroadcastDistribution(record_array, "
            f"n={self.n}, fields={self._record_array.fields})"
        )


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

    # Already a RecordArray (e.g., from a distribution's _sample)
    from ._record_array import RecordArray
    if isinstance(output_samples, RecordArray):
        return _RecordArrayMarginal(output_samples, weights, name=name)

    # Record with batched leaves (e.g., from jax.vmap over a Record-returning fn)
    from .record import Record, RecordTemplate
    if isinstance(output_samples, Record):
        first_field = output_samples[output_samples.fields[0]]
        if hasattr(first_field, 'ndim') and first_field.ndim > 0:
            n = first_field.shape[0]
            tpl = RecordTemplate.from_record(output_samples, batch_shape=(n,))
            fields = {f: output_samples[f] for f in output_samples.fields}
            ra = RecordArray(fields, batch_shape=(n,), template=tpl)
            return _RecordArrayMarginal(ra, weights, name=name)

    # Try stacking into an array
    if isinstance(output_samples, jnp.ndarray):
        return _ArrayMarginal(output_samples, weights, name=name)

    if isinstance(output_samples, list):
        # Try stacking Records into a RecordArray
        from .record import Record
        if output_samples and all(isinstance(r, Record) for r in output_samples):
            from ._record_array import RecordArray
            try:
                ra = RecordArray.stack(output_samples)
                return _RecordArrayMarginal(ra, weights, name=name)
            except (ValueError, TypeError):
                pass
        # Try stacking into a flat array
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
       ``ArrayDistribution`` instances with TFP shape semantics
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
