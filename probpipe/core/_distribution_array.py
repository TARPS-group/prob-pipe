"""``DistributionArray`` — shape-indexed collection of independent distributions.

A ``DistributionArray`` wraps ``n`` independent distributions as a single
``Distribution`` with leading ``batch_shape=(n,)``. It is **not** a
mixture: indexing returns one component, sampling draws one per
component and stacks, mean/variance return per-component stacks.

Use case: the output type of a :class:`~probpipe.WorkflowFunction`
parameter sweep whose inner call returns a ``Distribution``. Each row of
the sweep produces its own posterior; the stacked result must preserve
row identity rather than marginalise, which is why we cannot reuse
:class:`_MixtureMarginal`.

Protocol support is dynamic (Pattern B): a ``DistributionArray``
satisfies ``SupportsX`` iff every component does. The factory
:func:`_make_distribution_array` constructs a cached subclass with the
appropriate protocol mixins based on component capabilities.

``DistributionArray`` vs. ``ProductDistribution``
-------------------------------------------------

Both express a set of independent components, but the access pattern
differs:

- :class:`~probpipe.ProductDistribution` bundles **heterogeneous
  independent components** addressed by name — e.g.
  ``ProductDistribution(theta=Normal(0, 1), sigma=Gamma(2, 1))``.
  ``sample`` returns a ``Record`` keyed by component name; ``log_prob``
  takes a same-shaped Record and sums the marginals. Different
  components are typically different families (Normal + Gamma + ...).

- :class:`DistributionArray` bundles **positionally-indexed components**
  along a batch axis — e.g.
  ``DistributionArray([Normal(loc=i, scale=1.0, name=f"n{i}") for i in
  range(5)])``. ``sample`` returns a stacked array of shape
  ``(5,) + event_shape``; ``log_prob`` takes a same-shaped array and
  returns a per-row vector. Components can still be heterogeneous
  as long as they share ``event_shape``, but the common use case is
  n instances of the same family parameterised differently (the output
  of a parameter sweep).

Rule of thumb: if you'd write ``d["sigma"]`` to pull out a specific
**named** quantity → ``ProductDistribution``. If you'd write ``d[i]``
to pull out the i-th element of a **batch** → ``DistributionArray``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..custom_types import Array
from ._distribution_base import Distribution
from ._record_array import RecordArray
from .protocols import (
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from .record import Record

__all__ = ["DistributionArray"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DistributionArray[T](Distribution[T]):
    """Ordered collection of ``n`` independent distributions.

    Leading ``batch_shape=(n,)`` prepended to the shared inner
    ``batch_shape`` of the components (all components must have the
    same inner batch_shape and event_shape).

    Use the :func:`_make_distribution_array` factory to construct
    instances — it picks the right subclass based on which protocols
    all components support. The base class itself exposes only
    indexing, iteration, and the ``components`` accessor.

    Parameters
    ----------
    components : sequence of Distribution
        The n component distributions. Must be non-empty and share
        ``event_shape`` (and inner ``batch_shape`` if any).
    name : str, optional
        Name for provenance / introspection. Defaults to
        ``"distribution_array"``.

    Notes
    -----
    Unlike :class:`~probpipe.core._broadcast_distributions._MixtureMarginal`,
    which marginalises over the components, ``DistributionArray`` is the
    "stack" counterpart used when the component identity matters — each
    row is a distinct scenario, not a mixture component. Sampling
    produces one sample per component stacked along the leading axis.
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def __init__(
        self,
        components,
        *,
        batch_shape: tuple[int, ...] | None = None,
        name: str | None = None,
    ):
        components = tuple(components)
        if not components:
            raise ValueError("DistributionArray requires at least one component")
        # All components must share event_shape. Additionally, each
        # component must be scalar (batch_shape == ()) — batching is
        # DistributionArray's job, not an orthogonal axis living on
        # individual components. Rejecting batched-param components
        # enforces the "one Distribution = one random variable" rule
        # the wider codebase is moving toward (see issue #134).
        es0 = getattr(components[0], "event_shape", ())
        for i, c in enumerate(components):
            es = getattr(c, "event_shape", ())
            bs = getattr(c, "batch_shape", ())
            if es != es0:
                raise ValueError(
                    f"DistributionArray requires matching event_shape "
                    f"across components; components[0].event_shape={es0} "
                    f"but components[{i}].event_shape={es}."
                )
            if bs != ():
                raise ValueError(
                    f"DistributionArray components must be scalar "
                    f"(batch_shape == ()); components[{i}] has "
                    f"batch_shape={bs}. Batching is expressed by "
                    f"DistributionArray itself — pass scalar components "
                    f"and set batch_shape=... on the DistributionArray."
                )
        # ``batch_shape`` defaults to (n,) for backward compatibility
        # with the 1-D-only form used until now. Multi-d broadcasting
        # passes the full sweep shape explicitly.
        if batch_shape is None:
            batch_shape = (len(components),)
        else:
            batch_shape = tuple(batch_shape)
            from .._utils import prod as _prod
            if _prod(batch_shape) != len(components):
                raise ValueError(
                    f"DistributionArray batch_shape={batch_shape} implies "
                    f"{_prod(batch_shape)} components but got "
                    f"{len(components)}."
                )
        self._components = components
        self._batch_shape_leading = batch_shape
        if name is None:
            name = "distribution_array"
        super().__init__(name=name)
        # A DistributionArray holding MC-marginal components inherits
        # their approximation status; if any component is approximate
        # (a _MixtureMarginal or NumericEmpiricalDistribution), so is
        # the stack.
        self._approximate = any(
            getattr(c, "is_approximate", False) for c in components
        )

    # -- structure -----------------------------------------------------------

    @property
    def n(self) -> int:
        """Total number of components (``prod(batch_shape_leading)``)."""
        return len(self._components)

    @property
    def components(self) -> tuple[Distribution, ...]:
        """Flat tuple of component distributions, in row-major order
        across the leading ``batch_shape``."""
        return self._components

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch axes of this DistributionArray.

        The components themselves are required to be scalar
        (``batch_shape == ()``), so this is just
        ``self._batch_shape_leading`` — no inherit-from-component
        composition. Multi-d broadcasting outputs pass the full
        sweep shape; the default 1-D form is ``(n,)``.
        """
        return tuple(self._batch_shape_leading)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shared ``event_shape`` across components."""
        return getattr(self._components[0], "event_shape", ())

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return self._batch_shape_leading[0]

    def __getitem__(self, key):
        """Index a single component, slice, or multi-d tuple.

        Supported key forms:

        * ``int`` → index along the leading batch axis.
        * ``slice`` → slice along the leading batch axis; returns a
          new ``DistributionArray`` containing the sliced subset.
        * ``tuple`` → multi-axis index (int or slice per axis);
          collapses along int axes and slices along slice axes.

        Indexing uses row-major order across ``batch_shape_leading``.
        """
        import numpy as _np
        bshape = self._batch_shape_leading
        # Normalise the key to a tuple, one entry per leading axis.
        if isinstance(key, tuple):
            if len(key) > len(bshape):
                raise IndexError(
                    f"DistributionArray has {len(bshape)} leading batch "
                    f"axes; got {len(key)}-tuple key."
                )
            key_tuple = key + (slice(None),) * (len(bshape) - len(key))
        else:
            key_tuple = (key,) + (slice(None),) * (len(bshape) - 1)

        # Reshape the flat component tuple to match bshape, apply the
        # key, and decide whether the result is a single distribution
        # or a new DistributionArray.
        components_nd = _np.empty(bshape, dtype=object)
        for flat_idx, comp in enumerate(self._components):
            components_nd[_np.unravel_index(flat_idx, bshape)] = comp
        sliced = components_nd[key_tuple]

        if isinstance(sliced, Distribution):
            return sliced  # single component
        # np.ndarray (possibly 0-d) of distributions.
        if sliced.ndim == 0:
            return sliced.item()
        new_components = list(sliced.ravel())
        if not new_components:
            raise ValueError(
                "DistributionArray index produced an empty sequence; "
                "at least one component is required."
            )
        return _make_distribution_array(
            new_components,
            batch_shape=sliced.shape,
            name=self._name,
        )

    def __iter__(self):
        return iter(self._components)

    def __repr__(self) -> str:
        return (
            f"DistributionArray(n={self.n}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# Protocol mixins (combined dynamically via factory below)
# ---------------------------------------------------------------------------


class _DistArraySampling:
    """SupportsSampling mixin for DistributionArray.

    Per-component sampling — one draw from each component, stacked
    along the leading axis. **Not** a mixture draw: the i-th sample
    comes deterministically from ``components[i]`` (modulo the key
    derived from ``jax.random.split``).
    """

    _sampling_cost: str = "medium"
    _preferred_orchestration: str | None = None

    def _sample(self, key, sample_shape=()):
        keys = jax.random.split(key, self.n)
        # Draw from each component at the requested sample_shape.
        per_component = [
            self._components[i]._sample(keys[i], sample_shape)
            for i in range(self.n)
        ]
        bshape = self._batch_shape_leading

        # Record-valued components — two sample-shape regimes:
        #
        #  1. sample_shape == ()  → each per_component[i] is a Record.
        #     Stack n of them into a RecordArray, then reshape its
        #     leading axis to bshape.
        #  2. sample_shape != ()  → each per_component[i] is a
        #     RecordArray with batch_shape=sample_shape. Stack fields
        #     along a new trailing axis so the leading shape is
        #     ``sample_shape + bshape``.
        if sample_shape == () and all(
            isinstance(s, Record) and not isinstance(s, RecordArray)
            for s in per_component
        ):
            flat_ra = RecordArray.stack(list(per_component))
            return _reshape_record_array_batch(flat_ra, bshape)
        if sample_shape != () and all(
            isinstance(s, RecordArray) for s in per_component
        ):
            fields = {
                name: jnp.stack(
                    [ra[name] for ra in per_component], axis=len(sample_shape)
                )
                for name in per_component[0].fields
            }
            # Reshape the flat n-axis to bshape.
            reshaped = {
                name: arr.reshape(sample_shape + bshape + arr.shape[len(sample_shape) + 1:])
                for name, arr in fields.items()
            }
            return type(per_component[0])(
                reshaped,
                batch_shape=sample_shape + bshape,
                template=per_component[0].template,
            )

        # Numeric-array case. Stack along the axis right after
        # sample_shape, then reshape the leading axis from (n,) to
        # bshape so the final shape is ``sample_shape + bshape +
        # event_shape``.
        try:
            stacked = jnp.stack(per_component, axis=len(sample_shape))
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(s).__name__ for s in per_component})
            raise TypeError(
                f"_DistArraySampling cannot stack component samples of "
                f"types {types_seen}; DistributionArray supports numeric "
                f"arrays and Record values only."
            ) from exc
        if bshape == (self.n,):
            return stacked
        new_shape = (
            stacked.shape[:len(sample_shape)]
            + bshape
            + stacked.shape[len(sample_shape) + 1:]
        )
        return stacked.reshape(new_shape)


def _reshape_record_array_batch(
    ra: RecordArray, new_batch_shape: tuple[int, ...]
) -> RecordArray:
    """Reshape a RecordArray's leading batch axis to ``new_batch_shape``.

    Assumes the RecordArray's current ``batch_shape`` has the same
    flat size as ``new_batch_shape`` (``prod(current) == prod(new)``).
    Each field array is reshaped so its leading axes match.
    """
    from .._utils import prod as _prod

    if _prod(ra.batch_shape) != _prod(new_batch_shape):
        raise ValueError(
            f"Cannot reshape RecordArray batch_shape={ra.batch_shape} to "
            f"{new_batch_shape}: different total size."
        )
    if ra.batch_shape == new_batch_shape:
        return ra
    n_cur = len(ra.batch_shape)
    new_fields = {}
    for name in ra.fields:
        val = ra[name]
        event_tail = val.shape[n_cur:]
        new_fields[name] = val.reshape(new_batch_shape + event_tail)
    return type(ra)(
        new_fields,
        batch_shape=new_batch_shape,
        template=ra.template,
    )


def _stack_leading(values: list) -> Any:
    """Stack a list of per-component values along a new leading axis.

    Handles the three value-type regimes that arise from distribution
    ops (``_mean`` / ``_variance`` / ``_sample`` / ``_log_prob``):

    - Array leaves → ``jnp.stack(axis=0)``.
    - ``Record`` leaves → ``RecordArray.stack``.
    - ``RecordArray`` leaves (each with matching batch_shape) →
      nested ``RecordArray`` with ``batch_shape=(n,) + inner_batch``.

    Raises :class:`TypeError` with the observed type list if the
    values can't be stacked (no common regime).
    """
    if not values:
        raise ValueError("cannot stack an empty list of values")
    # Check the more-specific RecordArray subclass first — else the
    # Record branch below would claim batched values and collapse their
    # inner batch axis (``RecordArray`` is now a ``Record`` subclass).
    if all(isinstance(v, RecordArray) for v in values):
        # Stack each field along a new leading axis; shapes must match.
        first = values[0]
        fields = {
            name: jnp.stack([ra[name] for ra in values], axis=0)
            for name in first.fields
        }
        return type(first)(
            fields,
            batch_shape=(len(values),) + first.batch_shape,
            template=first.template,
        )
    if all(
        isinstance(v, Record) and not isinstance(v, RecordArray)
        for v in values
    ):
        return RecordArray.stack(list(values))
    try:
        return jnp.stack(values, axis=0)
    except (TypeError, ValueError) as exc:
        types_seen = sorted({type(v).__name__ for v in values})
        raise TypeError(
            f"DistributionArray cannot stack component values of types "
            f"{types_seen}; supported: numeric arrays, Record, RecordArray."
        ) from exc


def _reshape_flat_to_bshape(flat: Any, bshape: tuple[int, ...]) -> Any:
    """Reshape the leading axis of ``flat`` from ``(n,)`` to ``bshape``.

    Works for arrays, Records (no-op), and RecordArrays. Called by the
    DistArray reduction mixins to present multi-d batch outputs.
    """
    from .._utils import prod as _prod
    if bshape == (_prod(bshape),):
        return flat
    if isinstance(flat, jnp.ndarray):
        return flat.reshape(bshape + flat.shape[1:])
    if isinstance(flat, RecordArray):
        return _reshape_record_array_batch(flat, bshape)
    # Record (scalar, stack_leading returned one) — bshape must be ()
    # which the no-op fast-path above caught. Otherwise it's a mismatch.
    return flat


class _DistArrayMean:
    """SupportsMean mixin for DistributionArray.

    Per-component mean stacked along the leading batch axes of the
    DistributionArray. **Not** a mixture mean — each row is a distinct
    distribution, so its mean is reported independently. The stack
    adapts to the value type so Record-valued means become a
    ``RecordArray`` with the batch axes prepended.
    """

    def _mean(self):
        stacked = _stack_leading([c._mean() for c in self._components])
        return _reshape_flat_to_bshape(stacked, self._batch_shape_leading)


class _DistArrayVariance:
    """SupportsVariance mixin for DistributionArray.

    Per-component variance stacked along the leading batch axes.
    **Not** the law-of-total-variance mixture formula — each row is a
    distinct distribution, so its variance is reported independently.
    """

    def _variance(self):
        stacked = _stack_leading([c._variance() for c in self._components])
        return _reshape_flat_to_bshape(stacked, self._batch_shape_leading)


class _DistArrayLogProb:
    """SupportsLogProb mixin for DistributionArray.

    Per-component log-density. ``value`` must have leading axes
    ``self.batch_shape`` (after stripping any outer sample axes the
    caller prepended); returns an array of shape ``self.batch_shape``
    of per-component log-probs.
    """

    def _log_prob(self, value):
        bshape = self._batch_shape_leading
        # Flatten value's leading batch axes so we can address
        # components in row-major order.
        if isinstance(value, jnp.ndarray):
            flat_leading = int(jnp.prod(jnp.array(bshape))) if bshape else 1
            flat_value = value.reshape((flat_leading,) + value.shape[len(bshape):])
            lps = jnp.stack(
                [
                    self._components[i]._log_prob(flat_value[i])
                    for i in range(self.n)
                ],
                axis=0,
            )
            return lps.reshape(bshape)
        # Non-array value: evaluate component-by-component (slow path).
        lps = jnp.stack(
            [
                self._components[i]._log_prob(value[i])
                for i in range(self.n)
            ],
            axis=0,
        )
        return lps.reshape(bshape)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


# Map protocol → (mixin class, required component protocols).
# Order matters only for the order mixins appear in the MRO; Python
# resolves method lookup by class order, so sampling sits first.
_DISTARRAY_PROTOCOL_MAP: list[tuple[type, type, tuple[type, ...]]] = [
    (SupportsSampling, _DistArraySampling, (SupportsSampling,)),
    (SupportsMean, _DistArrayMean, (SupportsMean,)),
    (SupportsVariance, _DistArrayVariance, (SupportsVariance,)),
    (SupportsLogProb, _DistArrayLogProb, (SupportsLogProb,)),
]


# Cache dynamically created classes so repeat constructions share a
# single type object (cheap ``isinstance``, JIT-cache friendliness).
_distarray_class_cache: dict[tuple[type, ...], type] = {}


def _make_distribution_array(
    components,
    *,
    batch_shape: tuple[int, ...] | None = None,
    name: str | None = None,
) -> DistributionArray:
    """Factory: build a ``DistributionArray`` with dynamic protocol support.

    Inspects the components to determine which protocols *all* of them
    support, creates (and caches) a concrete subclass that inherits
    those protocol mixins, and returns an instance of it.

    Parameters
    ----------
    components : sequence of Distribution
        Component distributions (must each be scalar,
        ``batch_shape == ()``).
    batch_shape : tuple of int, optional
        Leading batch shape for the DistributionArray. Defaults to
        ``(len(components),)`` for the 1-D form. Multi-d broadcasting
        passes the full sweep shape; ``prod(batch_shape)`` must equal
        ``len(components)``.
    name : str, optional
        Name for provenance.

    Returns
    -------
    DistributionArray
        Concrete subclass carrying the minimal protocol set satisfied
        by all components.
    """
    components = tuple(components)
    if not components:
        raise ValueError("DistributionArray requires at least one component")

    active_protocols: list[type] = []
    active_mixins: list[type] = []
    for protocol, mixin, required in _DISTARRAY_PROTOCOL_MAP:
        if all(isinstance(c, req) for c in components for req in required):
            active_protocols.append(protocol)
            active_mixins.append(mixin)

    cache_key = tuple(active_protocols)
    if cache_key not in _distarray_class_cache:
        bases = (
            tuple(active_mixins) + (DistributionArray,) + tuple(active_protocols)
        )
        cls_name = "_DynDistributionArray"
        _distarray_class_cache[cache_key] = type(cls_name, bases, {})

    cls = _distarray_class_cache[cache_key]
    obj = object.__new__(cls)
    DistributionArray.__init__(obj, components, batch_shape=batch_shape, name=name)
    return obj
