"""``DistributionArray`` — shape-indexed collection of independent distributions.

A ``DistributionArray`` is ``Array[Distribution]``: ``n`` independent
scalar distributions addressed by a (multi-d) ``batch_shape``. It is
**not** a mixture.

Vectorized ops are delivered by the :class:`~probpipe.WorkflowFunction`
sweep layer — when a ``DistributionArray`` is passed to an op like
``sample`` / ``mean`` / ``log_prob`` whose signature expects a scalar
``Distribution``, the WF dispatches cell-by-cell and stacks the results
into a ``NumericRecordArray`` / ``RecordArray`` (or nested
``DistributionArray`` when the op returns a distribution per cell).
The class itself only carries the container surface.

Use case: the natural output type of a ``WorkflowFunction`` parameter
sweep whose inner call returns a ``Distribution``. Each cell's
posterior stays identifiable rather than being marginalised.

``DistributionArray`` vs. ``ProductDistribution``
-------------------------------------------------

Both express a set of independent components, but the access pattern
differs:

- :class:`~probpipe.ProductDistribution` bundles **heterogeneous
  independent components** addressed by name — e.g.
  ``ProductDistribution(theta=Normal(0, 1), sigma=Gamma(2, 1))``.
  ``sample`` returns a ``Record`` keyed by component name.

- :class:`DistributionArray` bundles **positionally-indexed components**
  along a batch axis — e.g.
  ``DistributionArray([Normal(loc=i, scale=1.0, name=f"n{i}") for i in
  range(5)])``. ``sample(da)`` vectorizes over cells and returns a
  ``NumericRecordArray`` at ``batch_shape=da.batch_shape``.

Rule of thumb: if you'd write ``d["sigma"]`` to pull out a specific
**named** quantity → ``ProductDistribution``. If you'd write ``d[i]``
to pull out the i-th element of a **batch** → ``DistributionArray``.
"""

from __future__ import annotations

import numpy as np

from .._utils import prod
from ._distribution_base import Distribution

__all__ = ["DistributionArray"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DistributionArray[T](Distribution[T]):
    """Ordered collection of ``n`` independent scalar distributions.

    Exposes only the container surface (indexing, iteration,
    ``components``, ``batch_shape``, ``event_shape``, ``n``). Vectorized
    ops (``sample``, ``mean``, ``variance``, ``log_prob``, …) are
    delivered by the :class:`~probpipe.core.node.WorkflowFunction`
    sweep layer, which treats the array as ``Array[Distribution]`` and
    dispatches cell-by-cell.

    Parameters
    ----------
    components : sequence of Distribution
        The n component distributions. Must be non-empty, share
        ``event_shape``, and each have ``batch_shape == ()``.
    batch_shape : tuple of int, optional
        Leading batch shape. Defaults to ``(len(components),)`` for the
        1-D form; ``prod(batch_shape)`` must equal ``len(components)``.
    name : str, optional
        Name for provenance / introspection. Defaults to
        ``"distribution_array"``.
    """

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
        # Components must share event_shape and each be scalar
        # (``batch_shape == ()``); batching lives on the
        # DistributionArray itself, not on its elements.
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
            if prod(batch_shape) != len(components):
                raise ValueError(
                    f"DistributionArray batch_shape={batch_shape} implies "
                    f"{prod(batch_shape)} components but got "
                    f"{len(components)}."
                )
        self._components = components
        self._batch_shape = batch_shape
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
        """Total number of components (``prod(batch_shape)``)."""
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
        ``self._batch_shape`` — no inherit-from-component
        composition. Multi-d broadcasting outputs pass the full
        sweep shape; the default 1-D form is ``(n,)``.
        """
        return tuple(self._batch_shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shared ``event_shape`` across components."""
        return getattr(self._components[0], "event_shape", ())

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return self._batch_shape[0]

    def __getitem__(self, key):
        """Index a single component, slice, or multi-d tuple.

        Supported key forms:

        * ``int`` → index along the leading batch axis.
        * ``slice`` → slice along the leading batch axis; returns a
          new ``DistributionArray`` containing the sliced subset.
        * ``tuple`` → multi-axis index (int or slice per axis);
          collapses along int axes and slices along slice axes.

        Indexing uses row-major order across ``batch_shape``.
        The pure-int path translates the key directly to a flat index
        via ``np.ravel_multi_index`` without materialising a
        ``shape=batch_shape`` object array.
        """
        bshape = self._batch_shape
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

        # Fast path: all axes addressed by int (or int-like). Compute
        # the flat index directly; no object-array materialisation.
        # ``np.ravel_multi_index`` rejects negative indices, so wrap
        # them into the positive range first (``dists[-1]`` is a
        # common pattern — e.g. "last posterior in an iterate output").
        if all(
            isinstance(k, (int, np.integer)) or hasattr(k, "__index__")
            for k in key_tuple
        ):
            indices = tuple(
                int(k) % dim for k, dim in zip(key_tuple, bshape)
            )
            flat = int(np.ravel_multi_index(indices, bshape))
            return self._components[flat]

        # General path: object-array view for slice / mixed-key
        # support. Only materialised when slices are actually used,
        # and only for the axes that involve them.
        components_nd = np.asarray(self._components, dtype=object).reshape(bshape)
        sliced = components_nd[key_tuple]
        if isinstance(sliced, Distribution):
            return sliced
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

    def _flat_component(self, i: int) -> Distribution:
        """Return the i-th component in row-major order over ``batch_shape``.

        ``__getitem__`` is a leading-axis indexer (partial slicing); this
        method bypasses it for the sweep layer, which unravels its own
        flat index across multiple array inputs.
        """
        return self._components[i]

    def __repr__(self) -> str:
        return (
            f"DistributionArray(n={self.n}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_distribution_array(
    components,
    *,
    batch_shape: tuple[int, ...] | None = None,
    name: str | None = None,
) -> DistributionArray:
    """Factory: build a ``DistributionArray``.

    Vectorized ops (``sample`` / ``mean`` / ``variance`` / ``log_prob``
    / ...) are delivered uniformly by the ``WorkflowFunction`` sweep
    layer, which treats a ``DistributionArray`` as ``Array[Distribution]``
    and dispatches cell-by-cell. This factory therefore doesn't build
    protocol mixins — the class is one fixed type.

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
    """
    return DistributionArray(components, batch_shape=batch_shape, name=name)
