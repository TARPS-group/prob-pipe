"""The backend adapter of the parametric families.

``TFPDistribution`` implements the capability set on raw arrays over a wrapped
backend distribution, and every parametric family is a thin constructor over
it. It is the only class that knows the backend exists, and its ``raw()`` is
the wrapped backend distribution.

Every family samples and has a normalized density. Beyond those, a family
claims exactly the capabilities its backend computes: it lists them in its
class-level table ``_backend_capabilities``, and the adapter gives the family
the method that realizes each one unless the family defines its own. A moment
is the backend's, the covariance is a linear operator over the flattened draw,
and the quantiles are per coordinate with the level axes leading.

A scalar family given parameters with axes draws one array of independent
coordinates, one per entry of the broadcast parameters, so the backend's batch
axes become the event's axes. A family whose draws are themselves arrays takes
parameters for one law; a batch of separate laws is a ``DistributionBatch``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .._array_utils import _slice_leading_axes
from ..core._specs import NumericArraySpec, OutputSpec
from ..core.constraints import Constraint
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsQuantile,
    SupportsSampling,
    SupportsVariance,
)
from ..distributions._distribution import Distribution, NumericDistribution
from ..linalg import DenseLinOp, DiagonalLinOp, LinOp

if TYPE_CHECKING:
    from ..core._spec_base import TermSpec

__all__ = ["TFPDistribution"]

# ---------------------------------------------------------------------------
# The separate-laws form for the batched storage
# ---------------------------------------------------------------------------
#
# Inside ``_allow_batched_tfp_init`` a family given parameters with axes keeps
# the backend's batch axes as the axes of separate laws: one draw is an array
# of draws of separate laws, and the density is per law. The fused storage of a
# ``DistributionArray`` reads that form, as do the moment-matching converters
# and a sequential joint's components given batched parents. Outside it, a
# scalar family's batch axes are the axes of one event.

_BATCHED_INIT_BYPASS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_BATCHED_INIT_BYPASS",
    default=False,
)
"""Whether a family built in the current context keeps its backend's batch axes as separate laws.

A ``ContextVar`` scopes the form to the dynamic extent of the ``with`` block,
across threads and ``asyncio`` tasks."""


@contextlib.contextmanager
def _allow_batched_tfp_init() -> Generator[None, None, None]:
    """Build families in the separate-laws form within the block.

    A family given parameters with axes then keeps the backend's batch axes as
    the axes of separate laws, with one draw per law and a density per law,
    rather than as the axes of one event.
    """
    token = _BATCHED_INIT_BYPASS.set(True)
    try:
        yield
    finally:
        _BATCHED_INIT_BYPASS.reset(token)


# ---------------------------------------------------------------------------
# The capabilities a backend computes
# ---------------------------------------------------------------------------


def _coordinates(backend: tfd.Distribution) -> tfd.Distribution:
    """The backend of one coordinate, whose batch axes the event of independent coordinates reinterprets."""
    return backend.distribution if isinstance(backend, tfd.Independent) else backend


def _backend_mean(self: TFPDistribution) -> Array:
    """The mean, shaped like one draw."""
    return self._tfp_dist.mean()


def _backend_variance(self: TFPDistribution) -> Array:
    """The variance of each coordinate, shaped like one draw."""
    return self._tfp_dist.variance()


def _backend_cov(self: TFPDistribution) -> LinOp:
    """The covariance of the flattened draw, a ``(d, d)`` operator.

    Independent coordinates have the diagonal operator of their variances, and
    a law over a vector has the backend's dense covariance.
    """
    backend = self._tfp_dist
    if isinstance(backend, tfd.Independent) or tuple(backend.event_shape) == ():
        return DiagonalLinOp(jnp.reshape(backend.variance(), (-1,)))
    return DenseLinOp(backend.covariance())


def _backend_quantile(self: TFPDistribution, q: ArrayLike) -> Array:
    """The quantile of each coordinate at the levels *q*, of shape ``(*q.shape, *event_shape)``."""
    coordinates = _coordinates(self._tfp_dist)
    levels = jnp.asarray(q, dtype=coordinates.dtype)
    event_rank = len(self.event_spec.spec.shape)
    return coordinates.quantile(jnp.reshape(levels, levels.shape + (1,) * event_rank))


#: The method that realizes each capability a family's backend may compute, by name.
_BACKEND_METHODS: dict[type, dict[str, Callable[..., Any]]] = {
    SupportsMean: {"_mean": _backend_mean},
    SupportsVariance: {"_variance": _backend_variance},
    SupportsCovariance: {"_cov": _backend_cov},
    SupportsQuantile: {"_quantile": _backend_quantile},
}


# ---------------------------------------------------------------------------
# Rebuilding a family from its constructor arguments
# ---------------------------------------------------------------------------


def _recording_arguments(init: Callable[..., None]) -> Callable[..., None]:
    """*init*, recording on the instance the arguments of the outermost constructor call.

    A family's constructor calls the adapter's, so the arguments recorded are
    those the family was called with, together with whether it was built in the
    separate-laws form.
    """

    @functools.wraps(init)
    def __init__(self: TFPDistribution, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_constructor_arguments", None) is None:
            recorded = (args, kwargs, _BATCHED_INIT_BYPASS.get())
            object.__setattr__(self, "_constructor_arguments", recorded)
        init(self, *args, **kwargs)

    return __init__


def _rebuilt_family(
    cls: type[TFPDistribution],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    separate_laws: bool,
) -> TFPDistribution:
    """The family *cls* constructed from *args* and *kwargs* in the form it was built in."""
    form = _allow_batched_tfp_init() if separate_laws else contextlib.nullcontext()
    with form:
        return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class TFPDistribution(NumericDistribution, SupportsSampling, SupportsLogProb):
    """The law of one array, realized by a wrapped backend distribution.

    The adapter samples through the backend and scores with its normalized
    log-density. A family claims the further capabilities its backend
    computes by listing them in ``_backend_capabilities``: the mean and the
    variance, shaped like one draw; the covariance of the flattened draw as a
    linear operator; and the quantiles of each coordinate. The adapter gives
    the family the method realizing each, unless the family defines its own.

    One draw is the backend event's array, with its shape and dtype and the
    family's support, declared as a whole term, so every instance is a
    :class:`~probpipe.NumericDistribution`. Its component defaults to the law's
    name, and an ``event_spec`` declaration names another. A backend whose
    draws are scalars and whose parameters have axes is reinterpreted as one
    array of independent coordinates along those axes.

    Parameters
    ----------
    name : str
        The law's label, and the component of its event unless *event_spec*
        names another.
    backend_dist : tfd.Distribution
        The wrapped backend distribution, which a family's constructor builds
        from its parameters.
    event_spec : OutputSpec, optional
        The declaration of one draw, which names its component. The adapter
        fills a pending type, as in ``OutputSpec(theta=None)``, with the
        backend event's array.

    Raises
    ------
    TypeError
        If *backend_dist* is not a backend distribution, or *event_spec* is
        not an :class:`~probpipe.OutputSpec` or exposes a record.
    ValueError
        If the backend's draws are arrays and its parameters have axes, since
        a batch of separate laws is a ``DistributionBatch``, or *event_spec*
        declares a type that one draw does not conform to.

    Notes
    -----
    A family pickles and copies by rebuilding from the arguments its
    constructor was called with, in the form it was built in, and then
    restoring the state assigned since, such as a new label or provenance. The
    backend is rebuilt rather than copied, since some backends, such as the
    one reinterpreting independent coordinates, neither pickle nor deep-copy.
    """

    #: The capabilities the family's backend computes beyond sampling and the density.
    _backend_capabilities: ClassVar[frozenset[type]] = frozenset()

    _tfp_dist: tfd.Distribution

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Give a family the method realizing each capability its table lists.

        A family's own constructor is wrapped to record the arguments it is
        called with, which :meth:`__reduce__` rebuilds the family from.
        """
        super().__init_subclass__(**kwargs)
        for protocol in vars(cls).get("_backend_capabilities", ()):
            for method, implementation in _BACKEND_METHODS[protocol].items():
                if method not in vars(cls):
                    setattr(cls, method, implementation)
        if "__init__" in vars(cls):
            cls.__init__ = _recording_arguments(vars(cls)["__init__"])

    @_recording_arguments
    def __init__(
        self,
        name: str,
        backend_dist: tfd.Distribution,
        *,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not isinstance(backend_dist, tfd.Distribution):
            raise TypeError(
                f"backend_dist must be a backend distribution, got {type(backend_dist).__name__}"
            )
        backend_dist = self._reinterpreted(backend_dist)
        self._tfp_dist = backend_dist
        produced = NumericArraySpec(
            tuple(backend_dist.event_shape), backend_dist.dtype, self._event_support()
        )
        if event_spec is None:
            declaration: OutputSpec | TermSpec = produced
        elif isinstance(event_spec, OutputSpec):
            declaration = event_spec.with_spec(produced)
        else:
            raise TypeError(f"event_spec must be an OutputSpec, got {type(event_spec).__name__}")
        super().__init__(name, declaration)

    def _reinterpreted(self, backend: tfd.Distribution) -> tfd.Distribution:
        """*backend* with its batch axes as the event's, for draws of independent coordinates.

        Raises
        ------
        ValueError
            If the backend's draws are arrays and its parameters have axes.
        """
        batch = tuple(backend.batch_shape)
        if not batch or _BATCHED_INIT_BYPASS.get():
            return backend
        if tuple(backend.event_shape) != ():
            raise ValueError(
                f"{type(self).__name__} parameters imply {batch} separate laws over arrays "
                f"of shape {tuple(backend.event_shape)}; a batch of separate laws is a "
                f"DistributionBatch"
            )
        return tfd.Independent(backend, reinterpreted_batch_ndims=len(batch))

    # -- the event declaration ----------------------------------------------

    def _event_support(self) -> Constraint | None:
        """The support of one draw, which ``__init__`` declares.

        Each family states its support, and a backend wrapped without a family
        leaves it undeclared.
        """
        return None

    def raw(self) -> tfd.Distribution:
        """The wrapped backend distribution."""
        return self._tfp_dist

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from the recorded constructor arguments, then restore the other state.

        Every attribute but the backend is restored, so a label or provenance
        assigned after construction is kept.
        """
        arguments = getattr(self, "_constructor_arguments", None)
        if arguments is None:
            return super().__reduce__()
        instance_dict, slots = self.__getstate__()
        kept = {key: value for key, value in (instance_dict or {}).items() if key != "_tfp_dist"}
        return (_rebuilt_family, (type(self), *arguments), (kept or None, slots))

    # -- sampling and the density ---------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draws of the backend, with *sample_shape* leading."""
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def _log_prob(self, value: ArrayLike) -> Array:
        """The backend's normalized log-density, keeping the leading axes of *value*."""
        return self._tfp_dist.log_prob(jnp.asarray(value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, event_shape={self.event_shape})"

    # -- the fused storage of a DistributionArray ------------------------------

    @classmethod
    def _make_array_backend(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> _TFPArrayBackend:
        """Construct the fused storage of a DistributionArray of this family.

        The storage holds one backend over the batched parameters in the
        separate-laws form, and builds each cell with the family's constructor
        at that cell's parameters.
        """
        return _TFPArrayBackend(
            dist_cls=cls,
            name=name,
            batch_shape=tuple(batch_shape),
            batched_params=dict(batched_params),
        )


# ---------------------------------------------------------------------------
# Fused storage backend for DistributionArray
# ---------------------------------------------------------------------------


_ARRAY_BACKEND_NAME_SUFFIX = "__array_backend"
"""Suffix appended to a backend's base ``name`` when constructing the
wrapped batched ``TFPDistribution``. Centralised so
``_TFPArrayBackend.__init__`` and ``tree_unflatten`` can't drift."""


def _construct_batched_dist(
    dist_cls: type[TFPDistribution],
    *,
    name: str,
    batched_params: dict[str, Any],
) -> TFPDistribution:
    """Construct the fused batched distribution in the separate-laws form,
    with the centralised ``__array_backend``-suffixed name.

    Used by both :meth:`_TFPArrayBackend.__init__` and
    :meth:`_TFPArrayBackend.tree_unflatten` so the suffix and the form
    are fixed in one place.
    """
    with _allow_batched_tfp_init():
        return dist_cls(
            **batched_params,
            name=f"{name}{_ARRAY_BACKEND_NAME_SUFFIX}",
        )


class _TFPArrayBackend:
    """Fused TFP-batched backend for ``DistributionArray``.

    Owns one ``tfd.Distribution`` instance with TFP's native
    ``batch_shape != ()`` plus the constructor params used to make it,
    so per-cell materialisation (``cell(i)``) can construct a fresh
    *scalar* :class:`Distribution` with the row-``i`` slice of each
    param.

    Implementation strategy: the backend wraps a *single* ProbPipe
    ``Distribution`` instance constructed with the batched params.
    Vectorised ops forward to that wrapped instance's TFP backend;
    ``cell(i)`` slices the params and runs the ordinary scalar
    constructor with a suffixed name.

    Not a :class:`Distribution` itself — the backend exists only as
    the contract between :meth:`TFPDistribution._make_array_backend`
    and :class:`~probpipe.DistributionArray`. See
    :class:`probpipe.core.protocols._DistributionArrayBackend`.

    Parameters
    ----------
    dist_cls : type[TFPDistribution]
        The concrete ``TFPDistribution`` subclass (e.g., ``Normal``).
        Used to materialise per-cell scalars.
    name : str
        Base name. Per-cell scalars auto-suffix as ``f"{name}_{flat}"``
        where ``flat`` is the row-major flat index over ``batch_shape``.
    batch_shape : tuple of int
        Leading shape of the batched parameters.
    batched_params : dict[str, Any]
        Constructor kwargs for ``dist_cls`` with leading ``batch_shape``
        already applied. Scalars are passed through unchanged in
        ``cell(i)`` (broadcast across all cells).
    """

    def __init__(
        self,
        *,
        dist_cls: type[TFPDistribution],
        name: str,
        batch_shape: tuple[int, ...],
        batched_params: dict[str, Any],
    ) -> None:
        self._dist_cls = dist_cls
        self._name = name
        self._batch_shape = tuple(batch_shape)
        # Single pass: validate every higher-rank param's leading
        # axes against the declared ``batch_shape``, broadcasting
        # 0-D scalars up to ``batch_shape`` so callers can mix
        # scalars with arrays —
        # ``from_batched_params(Normal, loc=0.0, scale=1.0,
        # batch_shape=(5,))`` constructs five identical Normals. The
        # leading-axes check raises with a per-parameter message
        # before TFP gets to raise its generic "Arguments ... must
        # have compatible shapes".
        if self._batch_shape:
            normalised: dict[str, Any] = {}
            for key, value in batched_params.items():
                arr = jnp.asarray(value)
                if arr.ndim == 0:
                    arr = jnp.broadcast_to(arr, self._batch_shape)
                elif arr.ndim >= len(self._batch_shape):
                    leading = arr.shape[: len(self._batch_shape)]
                    if leading != self._batch_shape:
                        raise ValueError(
                            f"_TFPArrayBackend: declared "
                            f"batch_shape={self._batch_shape} but "
                            f"parameter {key!r} has leading shape "
                            f"{leading}; the two must match. Check "
                            f"that every batched parameter broadcasts "
                            f"to batch_shape."
                        )
                normalised[key] = arr
            batched_params = normalised
        self._batched_params = batched_params
        self._batched_dist: TFPDistribution = _construct_batched_dist(
            dist_cls,
            name=name,
            batched_params=batched_params,
        )
        # Final sanity check: TFP's inferred batch_shape must match
        # the caller's declaration. Catches the rare case where a
        # higher-rank param's *trailing* axes don't agree but the
        # leading-axes check above passed (e.g., MVN where ``loc`` /
        # ``scale_tril`` event ranks differ). Reads the underlying
        # ``tfd.Distribution.batch_shape`` directly because
        # ProbPipe-side ``Distribution`` has no ``batch_shape``.
        actual = tuple(self._batched_dist._tfp_dist.batch_shape)
        if actual != self._batch_shape:
            raise ValueError(
                f"_TFPArrayBackend: declared batch_shape={self._batch_shape} "
                f"but {dist_cls.__name__} with the given batched_params "
                f"produced TFP batch_shape={actual}. Check that every "
                f"batched parameter broadcasts to batch_shape."
            )

    # -- shape ---------------------------------------------------------------

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self._batched_dist.event_shape

    @property
    def dtype(self) -> jnp.dtype:
        return self._batched_dist.dtype

    @property
    def cell_spec(self) -> NumericArraySpec:
        """The term every cell draws, read without materialising a cell.

        The batched law declares the TFP event's array with its dtype, and its
        support is the family's at the batched parameters. That support holds for
        every cell only when no parameter is batched into it, so a support that
        holds a batched parameter is left unset.
        """
        spec = self._batched_dist.event_spec.spec
        support = spec.support
        if support is not None and any(jnp.ndim(value) > 0 for value in vars(support).values()):
            support = None
        return NumericArraySpec(spec.shape, spec.dtype, support)

    # -- per-cell materialisation -------------------------------------------

    def cell(self, index: int | tuple[int, ...]) -> Distribution:
        """Fabricate a fresh scalar :class:`Distribution` for cell ``index``.

        ``index`` may be a flat ``int`` (interpreted row-major over
        ``batch_shape``) or a ``tuple[int, ...]`` of axis-aligned
        indices. The returned distribution is fully scalar
        (``batch_shape == ()``) — no caching; each call re-runs the
        ordinary ``dist_cls(**scalar_params, name=...)`` constructor.

        ``batch_shape`` is non-empty by construction (
        :func:`DistributionArray._infer_batch_shape` rejects scalar-
        only param sets), so we never have to handle a degenerate
        zero-axis backend here.
        """
        multi, flat = self._normalize_index(index)
        scalar_params = {
            key: _slice_leading_axes(value, multi) for key, value in self._batched_params.items()
        }
        cell = self._dist_cls(
            **scalar_params,
            name=f"{self._name}_{flat}",
        )
        # The per-cell suffix is derived by the backend, not user-typed.
        return cell

    def _normalize_index(self, index: int | tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        """Return ``(multi_index, flat_index)`` for the given input.

        Lets :meth:`cell` slice with the multi-d index *and* name the
        result with the flat index in one pass, without round-tripping
        through ``np.ravel_multi_index`` / ``np.unravel_index`` for
        the common 1-D case. Out-of-range indices raise ``IndexError``
        via NumPy; rank mismatches are caught here with a clearer
        message than NumPy's default.
        """
        bshape = self._batch_shape
        if isinstance(index, (int, np.integer)) or hasattr(index, "__index__"):
            i = int(index)
            if len(bshape) == 1:
                if not 0 <= i < bshape[0]:
                    raise IndexError(
                        f"_TFPArrayBackend.cell: index {i} out of range for batch_shape={bshape}."
                    )
                return (i,), i
            multi = tuple(int(x) for x in np.unravel_index(i, bshape))
            return multi, i
        idx = tuple(int(x) for x in index)
        if len(idx) != len(bshape):
            raise IndexError(
                f"_TFPArrayBackend.cell: index {idx} has rank "
                f"{len(idx)} but batch_shape={bshape} has rank "
                f"{len(bshape)}."
            )
        flat = int(np.ravel_multi_index(idx, bshape))
        return idx, flat

    # -- vectorised ops (forward to the wrapped batched distribution) -------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return self._batched_dist._sample(key, sample_shape)

    def _log_prob(self, value: ArrayLike) -> Array:
        return self._batched_dist._log_prob(value)

    def _mean(self) -> Array:
        return self._batched_dist._mean()

    def _variance(self) -> Array:
        return self._batched_dist._variance()

    def _cov(self) -> Array:
        return self._batched_dist._cov()

    def __repr__(self) -> str:
        return (
            f"_TFPArrayBackend({self._dist_cls.__name__}, "
            f"batch_shape={self._batch_shape}, name={self._name!r})"
        )

    # -- JAX pytree registration --------------------------------------------

    def tree_flatten(self):
        """Split the backend into JAX-traceable children + static aux.

        Children are the batched parameter values (the JAX-array
        leaves the user passed); aux carries everything needed to
        reconstruct the backend (the distribution class, the cell
        name, the declared ``batch_shape``, and the parameter keys
        in iteration order). The wrapped ``_batched_dist`` is
        reconstructed inside ``tree_unflatten`` from the params, so
        successive ``jit`` / ``vmap`` traces stay consistent.
        """
        keys = tuple(self._batched_params.keys())
        children = tuple(self._batched_params[k] for k in keys)
        aux = (self._dist_cls, self._name, self._batch_shape, keys)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children) -> _TFPArrayBackend:
        """Reconstruct the backend without re-running the
        ``__init__`` shape sanity check.

        ``tree_map`` and ``vmap`` both invoke ``tree_unflatten`` with
        leaf shapes that may not match the originally-declared
        ``batch_shape`` (e.g., a fresh leading axis stacked by
        ``tree_map``, an abstract per-cell shape inside a ``vmap``
        trace). The aux is informational and preserved for the
        round-trip; the wrapped ``_batched_dist`` is rebuilt directly
        from the leaves.
        """
        dist_cls, name, batch_shape, keys = aux
        instance = cls.__new__(cls)
        instance._dist_cls = dist_cls
        instance._name = name
        instance._batch_shape = tuple(batch_shape)
        instance._batched_params = dict(zip(keys, children))
        instance._batched_dist = _construct_batched_dist(
            dist_cls,
            name=name,
            batched_params=instance._batched_params,
        )
        return instance


jax.tree_util.register_pytree_node_class(_TFPArrayBackend)
