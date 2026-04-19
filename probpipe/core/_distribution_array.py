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
        name: str | None = None,
    ):
        components = tuple(components)
        if not components:
            raise ValueError("DistributionArray requires at least one component")
        # All components must share event_shape. Inner batch_shape
        # mismatches would break the (n,) + inner batch semantics, so
        # enforce uniformity eagerly.
        es0 = getattr(components[0], "event_shape", ())
        bs0 = getattr(components[0], "batch_shape", ())
        for i, c in enumerate(components[1:], start=1):
            es = getattr(c, "event_shape", ())
            bs = getattr(c, "batch_shape", ())
            if es != es0:
                raise ValueError(
                    f"DistributionArray requires matching event_shape across "
                    f"components; components[0].event_shape={es0} but "
                    f"components[{i}].event_shape={es}."
                )
            if bs != bs0:
                raise ValueError(
                    f"DistributionArray requires matching inner batch_shape "
                    f"across components; components[0].batch_shape={bs0} but "
                    f"components[{i}].batch_shape={bs}."
                )
        self._components = components
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
        return len(self._components)

    @property
    def components(self) -> tuple[Distribution, ...]:
        """The n component distributions in sweep-row order."""
        return self._components

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """(n,) prepended to the shared inner batch_shape of components."""
        inner = getattr(self._components[0], "batch_shape", ())
        return (self.n,) + tuple(inner)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shared ``event_shape`` across components."""
        return getattr(self._components[0], "event_shape", ())

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key):
        """Index a single component or slice a sub-range.

        * ``int`` → returns the single component distribution.
        * ``slice`` → returns a new ``DistributionArray`` containing the
          sliced subset (uses the factory so protocol support is
          re-resolved for the shrunk component list).
        """
        if isinstance(key, slice):
            sliced = list(self._components)[key]
            if not sliced:
                raise ValueError(
                    "DistributionArray slice produced an empty sequence; "
                    "at least one component is required."
                )
            return _make_distribution_array(sliced, name=self._name)
        if isinstance(key, (int, jnp.integer)) or hasattr(key, "__index__"):
            return self._components[int(key)]
        raise TypeError(
            f"DistributionArray index must be int or slice, got "
            f"{type(key).__name__}"
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

        # Record-valued components — two sample-shape regimes:
        #
        #  1. sample_shape == ()  → each per_component[i] is a Record.
        #     Stack n of them into a RecordArray (batch_shape=(n,)).
        #
        #  2. sample_shape != ()  → each per_component[i] is already a
        #     RecordArray with batch_shape=sample_shape. Stack fields
        #     along a new trailing batch axis so the final leading
        #     shape is ``sample_shape + (n,)``.
        if sample_shape == () and all(
            isinstance(s, Record) and not isinstance(s, RecordArray)
            for s in per_component
        ):
            return RecordArray.stack(list(per_component))
        if sample_shape != () and all(
            isinstance(s, RecordArray) for s in per_component
        ):
            fields = {
                name: jnp.stack(
                    [ra[name] for ra in per_component], axis=len(sample_shape)
                )
                for name in per_component[0].fields
            }
            return type(per_component[0])(
                fields,
                batch_shape=sample_shape + (self.n,),
                template=per_component[0].template,
            )

        # Numeric-array case. Stack along the axis right after
        # sample_shape — leading axes are sample_shape, then the (n,)
        # DistributionArray axis, then any inner batch + event axes.
        try:
            return jnp.stack(per_component, axis=len(sample_shape))
        except (TypeError, ValueError) as exc:
            types_seen = sorted({type(s).__name__ for s in per_component})
            raise TypeError(
                f"_DistArraySampling cannot stack component samples of "
                f"types {types_seen}; DistributionArray supports numeric "
                f"arrays and Record values only."
            ) from exc


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


class _DistArrayMean:
    """SupportsMean mixin for DistributionArray.

    Per-component mean — stacks ``components[i]._mean()`` along the
    leading (n,) axis. This is **not** a mixture mean; there is no
    weighted average across components. The stack adapts to the value
    type so Record-valued means become a ``RecordArray`` with the
    (n,) axis prepended.
    """

    def _mean(self):
        return _stack_leading([c._mean() for c in self._components])


class _DistArrayVariance:
    """SupportsVariance mixin for DistributionArray.

    Per-component variance stacked along the leading (n,) axis.
    **Not** the law-of-total-variance mixture formula — each row is a
    distinct distribution, so its variance is reported independently.
    """

    def _variance(self):
        return _stack_leading([c._variance() for c in self._components])


class _DistArrayLogProb:
    """SupportsLogProb mixin for DistributionArray.

    Per-component log-density. ``value`` must have leading axis of
    length n (after stripping any sample_shape the caller prepended);
    returns a shape-(n,) array of per-component log-probs.
    """

    def _log_prob(self, value):
        # value shape is (n,) + event_shape for array-valued components,
        # or a RecordArray/batched Record for structured components —
        # slice along axis 0 and evaluate each component on its slice.
        return jnp.stack(
            [
                self._components[i]._log_prob(value[i])
                for i in range(self.n)
            ],
            axis=0,
        )


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
    name: str | None = None,
) -> DistributionArray:
    """Factory: build a ``DistributionArray`` with dynamic protocol support.

    Inspects the components to determine which protocols *all* of them
    support, creates (and caches) a concrete subclass that inherits
    those protocol mixins, and returns an instance of it.

    Parameters
    ----------
    components : sequence of Distribution
        Component distributions.
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
    DistributionArray.__init__(obj, components, name=name)
    return obj
