"""``DistributionArray`` — shape-indexed collection of independent distributions.

A ``DistributionArray`` is ``Array[Distribution]``: an ordered
collection of independent scalar distributions addressed by a
(multi-d) ``batch_shape``.

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

from math import prod
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .._array_utils import _slice_leading_axes
from ._distribution_base import Distribution
from .protocols import SupportsArrayBackend

if TYPE_CHECKING:
    from .protocols import _DistributionArrayBackend

__all__ = ["DistributionArray"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DistributionArray[T](Distribution[T]):
    """Ordered collection of independent scalar distributions
    addressed by a (multi-d) ``batch_shape``.

    Exposes only the container surface (indexing, iteration,
    ``components``, ``batch_shape``, ``event_shape``). Vectorized
    ops (``sample``, ``mean``, ``variance``, ``log_prob``, …) are
    delivered by the :class:`~probpipe.core.node.WorkflowFunction`
    sweep layer, which treats the array as ``Array[Distribution]`` and
    dispatches cell-by-cell.

    Parameters
    ----------
    components : sequence of Distribution
        The n component distributions. Must be non-empty and share
        ``event_shape``.
    batch_shape : tuple of int, optional
        Leading batch shape. Defaults to ``(len(components),)`` for the
        1-D form; ``prod(batch_shape)`` must equal ``len(components)``.
    name : str, optional
        Name for provenance / introspection. Defaults to
        ``"distribution_array"``.

    Notes
    -----
    **How ops work on a DistributionArray.** The class deliberately
    does *not* implement ``_sample`` / ``_mean`` / ``_log_prob`` /
    etc. — those would couple the array to specific component
    capabilities. Vectorization is handled at a different layer:

    1. ``sample(da, ...)`` calls the :class:`~probpipe.sample`
       :class:`~probpipe.core.node.WorkflowFunction`, whose dispatch
       sees a ``DistributionArray`` argument where the op's annotation
       expects a scalar ``SupportsSampling``.
    2. WF dispatches cell-by-cell: each ``da[i]`` is sampled, results
       are stacked along ``batch_shape`` and returned as a
       :class:`~probpipe.NumericRecordArray` (or
       :class:`~probpipe.RecordArray` for non-numeric components).
       For ops whose inner return is itself a ``Distribution`` (e.g.
       posterior-predictive sweeps), the result is a nested
       ``DistributionArray``.
    3. Multiple swept arguments combine by the **product rule**:
       passing two ``DistributionArray`` args of shapes ``(m,)`` and
       ``(n,)`` produces an output of shape ``(m, n)``.

    Consequences of this design:

    * Calling ``da._sample(key)`` directly raises ``AttributeError`` —
      ``DistributionArray`` doesn't have ``_sample``. Always use the
      public op (``sample(da, key=...)``).
    * ``isinstance(da, SupportsSampling)`` is ``False`` even when
      every component supports sampling. The protocol attaches to
      individual cells, not to the array.
    * Component capabilities don't have to be uniform: an array
      where some cells are ``SupportsLogProb`` and some are not will
      fail at op-dispatch on the first non-supporting cell, rather
      than rejecting at construction.
    """

    # Storage slots. ``_components`` is ``None`` for backend-delegated
    # arrays until :attr:`components` materialises the eager tuple
    # lazily; the literal-array constructor sets it directly.
    # ``_backend`` is ``None`` for the literal path and set by
    # :meth:`_from_backend`.
    _components: "tuple[Distribution, ...] | None"
    _backend: "_DistributionArrayBackend | None"

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
        # Components must share event_shape. Batching lives on the
        # DistributionArray itself; per the "one random variable per
        # Distribution" rule, components have no batch_shape.
        es0 = getattr(components[0], "event_shape", ())
        for i, c in enumerate(components):
            es = getattr(c, "event_shape", ())
            if es != es0:
                raise ValueError(
                    f"DistributionArray requires matching event_shape "
                    f"across components; components[0].event_shape={es0} "
                    f"but components[{i}].event_shape={es}."
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
        # Set only by :meth:`_from_backend`. The literal-array path
        # leaves it ``None`` and uses ``_components`` as the
        # storage-of-truth.
        self._backend = None
        if name is None:
            name = "distribution_array"
        super().__init__(name=name)
        # A DistributionArray holding MC-marginal components inherits
        # their approximation status; if any component is approximate
        # (a _MixtureMarginal or RecordEmpiricalDistribution), so is
        # the stack.
        self._approximate = any(
            getattr(c, "is_approximate", False) for c in components
        )

    # -- public batched-construction factory --------------------------------

    @classmethod
    def from_batched_params(
        cls,
        dist_cls: type,
        *,
        name: str,
        batch_shape: tuple[int, ...] | None = None,
        **batched_params,
    ) -> "DistributionArray":
        """Construct a ``DistributionArray`` of homogeneous components.

        The recommended way to build a ``DistributionArray`` whose
        cells are all instances of the same class — most often a
        TFP-backed family like ``Normal`` — without manually
        constructing each cell::

            DistributionArray.from_batched_params(
                Normal, loc=jnp.zeros(5), scale=1.0, name="x",
            )

        When ``dist_cls`` implements
        :class:`~probpipe.core.protocols.SupportsArrayBackend` (every
        TFP-backed concrete class does — ``Normal``, ``Beta``,
        ``Gamma``, ``MultivariateNormal``, …), the factory dispatches
        onto the backend's fused-storage path: a single batched
        TFP backend owns the parameters; cells are materialised lazily
        on demand. Otherwise the factory falls back to the
        literal-array path: one ``dist_cls`` instance per cell with
        per-cell parameters auto-sliced and names auto-suffixed
        ``f"{name}_{flat_index}"``.

        Parameters
        ----------
        dist_cls : type
            A ``Distribution`` subclass. The factory does not
            instantiate ``dist_cls`` directly when the protocol path
            is taken; per-cell scalars are produced by the backend.
        name : str
            Base name; per-cell scalars are named
            ``f"{name}_{flat_index}"`` (row-major over
            ``batch_shape``).
        batch_shape : tuple of int, optional
            Leading shape of the batched parameters. Inferred from
            ``batched_params`` (broadcast shape of array-valued
            entries) when omitted.
        **batched_params
            Constructor kwargs for ``dist_cls`` with leading
            ``batch_shape`` already applied. Scalars are broadcast
            across every cell.

        Returns
        -------
        DistributionArray
            Backend-delegated when ``dist_cls`` implements
            ``SupportsArrayBackend``; literal-array fallback otherwise.

        Raises
        ------
        ValueError
            If ``batch_shape`` cannot be inferred (no array-valued
            params) and the caller did not pass it explicitly.

        Examples
        --------
        Backend-delegated TFP path::

            da = DistributionArray.from_batched_params(
                Normal, loc=jnp.zeros(5), scale=1.0, name="x",
            )
            da.batch_shape       # (5,)
            da[0].name           # "x_0"
            da[0].loc            # 0.0
            da._backend          # _TFPArrayBackend(...)

        Literal-array fallback (any class without the protocol)::

            da = DistributionArray.from_batched_params(
                MyCustomDist, param=jnp.arange(4), name="z",
            )
            da._backend          # None — fallback path
            da[0].name           # "z_0"
        """
        inferred_shape = _infer_batch_shape(batched_params, batch_shape)
        if isinstance(dist_cls, SupportsArrayBackend):
            backend = dist_cls._make_array_backend(
                name=name,
                batch_shape=inferred_shape,
                **batched_params,
            )
            return cls._from_backend(backend, name=name)
        return cls._from_literal_components(
            dist_cls,
            name=name,
            batch_shape=inferred_shape,
            batched_params=batched_params,
        )

    @classmethod
    def _from_literal_components(
        cls,
        dist_cls: type,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        batched_params: dict,
    ) -> "DistributionArray":
        """Build by constructing one ``dist_cls`` instance per cell.

        Used by :meth:`from_batched_params` for any ``dist_cls`` that
        does not implement
        :class:`~probpipe.core.protocols.SupportsArrayBackend`.
        Per-cell parameters are sliced from ``batched_params`` at the
        flat row-major index; cell names auto-suffix as
        ``f"{name}_{flat}"``.
        """
        components = []
        n = prod(batch_shape)
        for flat in range(n):
            multi = (
                np.unravel_index(flat, batch_shape) if batch_shape else ()
            )
            multi_t = tuple(int(x) for x in multi)
            cell_params = {
                k: _slice_leading_axes(v, multi_t)
                for k, v in batched_params.items()
            }
            components.append(
                dist_cls(name=f"{name}_{flat}", **cell_params)
            )
        return cls(components, batch_shape=batch_shape, name=name)

    # -- backend-delegated constructor --------------------------------------

    @classmethod
    def _from_backend(
        cls,
        backend: "_DistributionArrayBackend",
        *,
        name: str | None = None,
    ) -> "DistributionArray":
        """Construct a backend-delegated ``DistributionArray``.

        Storage refactor entry point: when a homogeneous batched form
        is available (e.g., via
        :meth:`~probpipe.core.protocols.SupportsArrayBackend._make_array_backend`),
        the array stores the backend rather than a tuple of eagerly
        materialised components. Per-cell access (``__getitem__`` /
        ``_flat_component`` / iteration) goes through
        ``backend.cell(...)``; the literal ``_components`` tuple is
        materialised lazily on first access via :attr:`components`.

        Private — public construction goes through
        :meth:`from_batched_params`.

        Parameters
        ----------
        backend : _DistributionArrayBackend
            Storage backend produced by a ``SupportsArrayBackend``
            distribution class. Carries ``batch_shape``,
            ``event_shape``, and ``cell(index)`` plus whichever ops
            the underlying distribution class supports.
        name : str, optional
            Name for provenance / introspection. Defaults to
            ``"distribution_array"``.
        """
        instance = cls.__new__(cls)
        # Bypass __init__ entirely — backend-delegated arrays have no
        # eager component list. Initialise fields the same way
        # __init__ would, but without per-component validation (the
        # backend already vouched for shape consistency).
        instance._components = None
        instance._batch_shape = tuple(backend.batch_shape)
        instance._backend = backend
        if name is None:
            name = "distribution_array"
        Distribution.__init__(instance, name=name)
        # Approximation status flows from the backend. TFP-backed
        # arrays are exact; a future Record-backend (over a
        # ``RecordEmpiricalDistribution``) will report
        # ``is_approximate=True`` on its own samples and the array
        # picks that up here.
        instance._approximate = bool(getattr(backend, "is_approximate", False))
        return instance

    # -- structure -----------------------------------------------------------

    @property
    def components(self) -> tuple[Distribution, ...]:
        """Flat tuple of component distributions, in row-major order
        across the leading ``batch_shape``.

        For backend-delegated arrays the tuple is materialised lazily
        on first access via ``backend.cell(i)`` for each flat index
        and cached. Cells are still freshly constructed inside
        ``cell()`` (no de-duplication), so successive ``components``
        accesses return the same cached tuple but indexing via
        :meth:`__getitem__` / :meth:`_flat_component` always returns a
        fresh scalar.
        """
        if self._components is None:
            assert self._backend is not None  # invariant
            n = prod(self._batch_shape)
            self._components = tuple(
                self._backend.cell(i) for i in range(n)
            )
        return self._components

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch axes of this DistributionArray.

        Components are scalar (one random variable per
        ``Distribution``), so this is simply ``self._batch_shape``
        — there is no inherit-from-component composition. Multi-d
        broadcasting outputs pass the full sweep shape; the default
        1-D form is ``(n,)``.
        """
        return tuple(self._batch_shape)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shared ``event_shape`` across components."""
        if self._backend is not None:
            return tuple(self._backend.event_shape)
        return getattr(self._components[0], "event_shape", ())

    @property
    def dtype(self):
        """Per-cell dtype.

        Cells share an event shape and (in practice) a dtype because
        homogeneous backends produce uniformly-typed cells and
        literal-array constructions inherit from the source.
        Backend-delegated arrays read it from the backend; literal
        arrays read it from the first component.
        """
        if self._backend is not None:
            return getattr(self._backend, "dtype", None)
        return getattr(self._components[0], "dtype", None)

    @property
    def size(self) -> int:
        """Total number of cells (``prod(batch_shape)``).

        Mirrors ``np.ndarray.size`` / ``jax.Array.size``: ``len(da)``
        is the leading-axis dim, ``da.size`` is the total cell count.
        """
        return prod(self._batch_shape)

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        if not self._batch_shape:
            raise TypeError(
                "len() of unsized 0-d DistributionArray "
                "(batch_shape=()). Use da.size for the cell count."
            )
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
            if self._backend is not None:
                return self._backend.cell(flat)
            return self._components[flat]

        # General path: object-array view for slice / mixed-key
        # support. Only materialised when slices are actually used,
        # and only for the axes that involve them. Backend-delegated
        # arrays materialise their components on this path too — slice
        # indexing is rare and a one-time materialisation cost is
        # acceptable.
        components_nd = np.asarray(self.components, dtype=object).reshape(bshape)
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
        """Iterate the leading axis (numpy / jax convention).

        ``len(self)`` items are yielded:

        * ``ndim == 1`` (the common case): each item is a scalar
          :class:`~probpipe.Distribution` cell.
        * ``ndim >= 2``: each item is a ``DistributionArray`` of
          shape ``batch_shape[1:]`` — a leading-axis slice, mirroring
          ``iter(np.zeros((2, 3)))`` yielding two ``(3,)``-shaped
          views.
        * ``ndim == 0`` (``batch_shape == ()``): raises
          ``TypeError`` to match ``iter(np.zeros(()))``. Reach for
          :meth:`_flat_component` (or :attr:`components`) to access
          the single cell — those work uniformly across every
          ``batch_shape`` including ``()``.

        For flat row-major access over every cell (the pre-#178
        behaviour), use :attr:`components` or
        ``range(self.size)`` with :meth:`_flat_component`.
        """
        bshape = self._batch_shape
        if not bshape:
            raise TypeError(
                "iteration over a 0-d DistributionArray "
                f"(batch_shape={bshape}). Reach for "
                "da.components or da._flat_component(0) for the "
                "single cell."
            )
        n_lead = bshape[0]
        if len(bshape) == 1:
            if self._backend is not None:
                return (self._backend.cell(i) for i in range(n_lead))
            return iter(self._components)
        # Multi-d: __getitem__(int) on a multi-d DA returns a
        # sub-DistributionArray of shape batch_shape[1:].
        return (self[i] for i in range(n_lead))

    def _flat_component(self, i: int) -> Distribution:
        """Return the i-th component in row-major order over ``batch_shape``.

        ``__getitem__`` is a leading-axis indexer (partial slicing); this
        method bypasses it for the sweep layer, which unravels its own
        flat index across multiple array inputs and always passes
        non-negative integers. Negatives are rejected here — both
        backend and literal paths behave identically. User-facing
        ``da[-1]`` wraps via ``__getitem__`` before this method is
        called.
        """
        i_int = int(i)
        n_cells = prod(self._batch_shape)
        if not 0 <= i_int < n_cells:
            raise IndexError(
                f"_flat_component: index {i_int} out of range for "
                f"DistributionArray with batch_shape={self._batch_shape}."
            )
        if self._backend is not None:
            return self._backend.cell(i_int)
        return self._components[i_int]

    def __repr__(self) -> str:
        backed = " backend=True" if self._backend is not None else ""
        return (
            f"DistributionArray(batch_shape={self._batch_shape}, "
            f"event_shape={self.event_shape}{backed})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _infer_batch_shape(
    batched_params: dict, declared: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return the broadcast shape of array-valued entries in
    ``batched_params``, or ``declared`` if explicitly supplied.

    Used by :meth:`DistributionArray.from_batched_params` when the
    caller doesn't pass ``batch_shape`` explicitly. Walks the values,
    treating scalars / 0-D arrays as broadcast-across-batch and
    everything else by its leading shape (since trailing axes may be
    event-shape, e.g. MVN's ``loc`` is ``(*batch, d)`` — but for
    inference we use the full broadcast shape across non-scalar
    params, which equals the batch leading prefix when scalars are
    excluded).
    """
    if declared is not None:
        return tuple(int(x) for x in declared)
    shapes = []
    for value in batched_params.values():
        try:
            arr = jnp.asarray(value)
        except (TypeError, ValueError):
            continue
        if arr.ndim > 0:
            shapes.append(arr.shape)
    if not shapes:
        raise ValueError(
            "from_batched_params: cannot infer batch_shape — no "
            "array-valued parameters were passed (all are scalar). "
            "Pass batch_shape=... explicitly, or use the regular "
            "Distribution constructor for a single instance."
        )
    # The broadcast shape across non-scalar params is the batch prefix
    # only when all params share the same event_shape. For families
    # with differing per-param event ranks (e.g. ``MultivariateNormal``
    # with ``loc.shape=(*batch, d)`` and ``scale_tril.shape=(*batch,
    # d, d)``), the broadcast attempt fails — punt to the caller.
    try:
        bs = jnp.broadcast_shapes(*shapes)
    except ValueError as err:
        raise ValueError(
            "from_batched_params: cannot infer batch_shape from the "
            f"given parameter shapes {shapes}. This typically happens "
            "for distributions with non-trivial event_shape (e.g. "
            "MultivariateNormal, Dirichlet) where each parameter has "
            "a different number of event axes. Pass batch_shape=... "
            "explicitly."
        ) from err
    return tuple(int(x) for x in bs)


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
        Component distributions. Each must be a scalar
        ``Distribution`` (one random variable per cell).
    batch_shape : tuple of int, optional
        Leading batch shape for the DistributionArray. Defaults to
        ``(len(components),)`` for the 1-D form. Multi-d broadcasting
        passes the full sweep shape; ``prod(batch_shape)`` must equal
        ``len(components)``.
    name : str, optional
        Name for provenance.
    """
    return DistributionArray(components, batch_shape=batch_shape, name=name)
