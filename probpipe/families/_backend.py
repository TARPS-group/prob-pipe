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

Parameters with more axes than one law needs give one law whose extra leading
axes are event axes of independent coordinates, so the backend's batch axes
become the event's leading axes: a scalar family draws one coordinate per entry
of the broadcast parameters, and a family whose draws are arrays draws one
independent row per entry. Separate laws form a ``DistributionBatch``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
from collections.abc import Callable, Generator
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from ..core._array_backend import _read_only
from ..core._repr import format_value
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
    _implements,
)
from ..distributions._distribution import (
    NumericDistribution,
    _class_label,
    _constructor_label,
    _whole_term_event,
)
from ..linalg import DenseLinOp, DiagonalLinOp, LinOp

__all__ = ["TFPDistribution"]

# ---------------------------------------------------------------------------
# The separate-laws form for the batched storage
# ---------------------------------------------------------------------------
#
# Inside ``_allow_batched_tfp_init`` a family given parameters with axes keeps
# the backend's batch axes as the axes of separate laws: one draw is an array
# of draws of separate laws, and the density is per law. The fused storage of
# laws at batched parameters below reads that form. Outside it, a scalar
# family's batch axes are the axes of one event.

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
    """The backend of one coordinate or row, whose batch axes the event's leading axes reinterpret."""
    return backend.distribution if isinstance(backend, tfd.Independent) else backend


def _block_diagonal(blocks: Array) -> Array:
    """The block-diagonal matrix of *blocks* ``(*rows, k, k)`` over the rows' row-major order."""
    k = blocks.shape[-1]
    flat = jnp.reshape(blocks, (-1, k, k))
    n = flat.shape[0]
    joint = jnp.einsum("ij,iab->iajb", jnp.eye(n, dtype=flat.dtype), flat)
    return jnp.reshape(joint, (n * k, n * k))


def _backend_mean(self: TFPDistribution) -> Array:
    """The mean, shaped like one draw."""
    return self._tfp_dist.mean()


def _backend_variance(self: TFPDistribution) -> Array:
    """The variance of each coordinate, shaped like one draw."""
    return self._tfp_dist.variance()


def _backend_cov(self: TFPDistribution) -> LinOp:
    """The covariance of the flattened draw, a ``(d, d)`` operator.

    Independent coordinates have the diagonal operator of their variances, a
    law over a vector has the backend's dense covariance, and independent rows
    have the block-diagonal matrix of the rows' covariances.
    """
    backend = self._tfp_dist
    row = _coordinates(backend)
    if tuple(row.event_shape) == ():
        return DiagonalLinOp(jnp.reshape(backend.variance(), (-1,)))
    if row is backend:
        return DenseLinOp(backend.covariance())
    return DenseLinOp(_block_diagonal(row.covariance()))


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
    separate-laws form. A pickle rebuilds the law from them, so a NumPy array
    among them is marked read-only.
    """

    @functools.wraps(init)
    def __init__(self: TFPDistribution, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_constructor_arguments", None) is None:
            args = tuple(_read_only(value) for value in args)
            kwargs = {name: _read_only(value) for name, value in kwargs.items()}
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
    layout = _allow_batched_tfp_init() if separate_laws else contextlib.nullcontext()
    with layout:
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
    family's support, declared as a whole term under *component*, so every
    instance is a :class:`~probpipe.NumericDistribution`. A family's label
    defaults to its class name, as ``Normal``, and the adapter's to the name of
    the backend's class. A backend whose parameters have axes beyond one law's
    is reinterpreted as one law whose leading event axes are those axes, over
    independent coordinates or rows.

    Parameters
    ----------
    component : str
        The component of the law's event.
    backend_dist : tfd.Distribution
        The wrapped backend distribution, which a family's constructor builds
        from its parameters.
    label : str, optional
        The law's label. It defaults to the family's class name, and for the
        adapter itself to the name of the backend's class, as ``Normal`` for
        a ``tfd.Normal``.
    event_spec : OutputSpec, optional
        A declaration of *component* that declares the type of one draw, which
        the adapter completes with the backend event's array.

    Raises
    ------
    TypeError
        If *component* is not a string, *label* is not a non-empty string,
        *backend_dist* is not a backend distribution, or *event_spec* is not an
        :class:`~probpipe.OutputSpec` or exposes a record.
    ValueError
        If *component* is not a valid component name, or *event_spec* names
        another component or declares a type that one draw does not conform to.

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

        A method the family implements, in its body or through a base it
        inherits, is kept. A family's own constructor is wrapped to record the
        arguments it is called with, which :meth:`__reduce__` rebuilds the
        family from.
        """
        super().__init_subclass__(**kwargs)
        for protocol in vars(cls).get("_backend_capabilities", ()):
            for method, implementation in _BACKEND_METHODS[protocol].items():
                if not _implements(cls, method):
                    setattr(cls, method, implementation)
        if "__init__" in vars(cls):
            cls.__init__ = _recording_arguments(vars(cls)["__init__"])

    @_recording_arguments
    def __init__(
        self,
        component: str,
        backend_dist: tfd.Distribution,
        *,
        label: str | None = None,
        event_spec: OutputSpec | None = None,
    ) -> None:
        if not isinstance(backend_dist, tfd.Distribution):
            raise TypeError(
                f"backend_dist must be a backend distribution, got {type(backend_dist).__name__}"
            )
        owner = _class_label(self)
        default = type(backend_dist).__name__ if owner == "TFPDistribution" else owner
        backend_dist = self._reinterpreted(backend_dist)
        self._tfp_dist = backend_dist
        produced = NumericArraySpec(
            tuple(backend_dist.event_shape), backend_dist.dtype, self._event_support()
        )
        declaration = _whole_term_event(component, produced, event_spec, owner)
        super().__init__(_constructor_label(self, label, default), declaration)

    def _reinterpreted(self, backend: tfd.Distribution) -> tfd.Distribution:
        """*backend* with its batch axes leading the event's, over independent coordinates or rows."""
        batch = tuple(backend.batch_shape)
        if not batch or _BATCHED_INIT_BYPASS.get():
            return backend
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

        Every attribute but the backend and the recorded arguments is restored,
        so a label or provenance assigned after construction is kept.
        """
        arguments = getattr(self, "_constructor_arguments", None)
        if arguments is None:
            return super().__reduce__()
        instance_dict, slots = self.__getstate__()
        rebuilt = ("_tfp_dist", "_constructor_arguments")
        kept = {key: value for key, value in (instance_dict or {}).items() if key not in rebuilt}
        return (
            _rebuilt_family,
            (type(self), *arguments),
            (kept or None, slots),
        )

    # -- sampling and the density ---------------------------------------------

    def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
        """Draws of the backend, with *sample_shape* leading."""
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def _log_prob(self, value: ArrayLike) -> Array:
        """The backend's normalized log-density, keeping the leading axes of *value*."""
        return self._tfp_dist.log_prob(jnp.asarray(value))

    def _repr_arguments(self) -> list[tuple[str, str]]:
        """The arguments the family's constructor was called with, other than the component, the label, and the declaration.

        The repr shows the arguments the call passed, so it reads as the call
        that built the law.
        """
        recorded = getattr(self, "_constructor_arguments", None)
        if recorded is None:
            return []
        args, kwargs, _ = recorded
        try:
            bound = inspect.signature(type(self).__init__).bind(self, *args, **kwargs)
        except TypeError:
            return []
        parameters = bound.signature.parameters
        fields: list[tuple[str, str]] = []
        for parameter, value in list(bound.arguments.items())[1:]:
            if parameter in ("component", "label", "event_spec"):
                continue
            if parameters[parameter].kind is inspect.Parameter.VAR_KEYWORD:
                fields.extend((key, format_value(entry)) for key, entry in value.items())
            else:
                fields.append((parameter, format_value(value)))
        return fields

    # -- the fused storage of laws at batched parameters ------------------------

    @classmethod
    def _make_array_backend(
        cls,
        *,
        component: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> _TFPArrayBackend:
        """Construct the fused storage of this family's laws at batched parameters.

        The storage holds one backend over the batched parameters in the
        separate-laws form.
        """
        return _TFPArrayBackend(
            dist_cls=cls,
            component=component,
            batch_shape=tuple(batch_shape),
            batched_params=dict(batched_params),
        )


# ---------------------------------------------------------------------------
# The fused storage of a family's laws at batched parameters
# ---------------------------------------------------------------------------


_ARRAY_BACKEND_LABEL_SUFFIX = "__array_backend"
"""Suffix appended to the component of a backend's laws to label the
wrapped batched ``TFPDistribution``. Centralised so
``_TFPArrayBackend.__init__`` and ``tree_unflatten`` can't drift."""


def _construct_batched_dist(
    dist_cls: type[TFPDistribution],
    *,
    component: str,
    batched_params: dict[str, Any],
) -> TFPDistribution:
    """Construct the fused batched distribution in the separate-laws form,
    over *component*, with the centralised ``__array_backend``-suffixed label.

    Used by both :meth:`_TFPArrayBackend.__init__` and
    :meth:`_TFPArrayBackend.tree_unflatten` so the suffix and the form
    are fixed in one place.
    """
    with _allow_batched_tfp_init():
        return dist_cls(
            component, **batched_params, label=f"{component}{_ARRAY_BACKEND_LABEL_SUFFIX}"
        )


class _TFPArrayBackend:
    """The fused storage of a family's laws at batched parameters, over one TFP batch.

    Owns one ``tfd.Distribution`` instance with TFP's native
    ``batch_shape != ()`` plus the constructor params used to make it.

    Implementation strategy: the backend wraps a *single* ProbPipe
    ``Distribution`` instance constructed with the batched params, and the
    vectorised ops forward to that wrapped instance's TFP backend.

    Not a :class:`Distribution` itself — the backend exists only as
    the contract of :meth:`TFPDistribution._make_array_backend`. See
    :class:`probpipe.core.protocols._DistributionArrayBackend`.

    Parameters
    ----------
    dist_cls : type[TFPDistribution]
        The concrete ``TFPDistribution`` subclass (e.g., ``Normal``).
    component : str
        The component of the wrapped law, whose label is *component* with the
        suffix ``__array_backend``.
    batch_shape : tuple of int
        Leading shape of the batched parameters.
    batched_params : dict[str, Any]
        Constructor kwargs for ``dist_cls`` with leading ``batch_shape``
        already applied.
    """

    def __init__(
        self,
        *,
        dist_cls: type[TFPDistribution],
        component: str,
        batch_shape: tuple[int, ...],
        batched_params: dict[str, Any],
    ) -> None:
        self._dist_cls = dist_cls
        self._component = component
        self._batch_shape = tuple(batch_shape)
        # Single pass: validate every higher-rank param's leading
        # axes against the declared ``batch_shape``, broadcasting
        # 0-D scalars up to ``batch_shape`` so callers can mix
        # scalars with arrays — ``loc=0.0, scale=1.0`` at
        # ``batch_shape=(5,)`` stores five identical Normals. The
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
                            f"batch_shape={self._batch_shape} does not match parameter "
                            f"{key!r} with leading shape {leading}; every batched parameter "
                            f"must broadcast to batch_shape"
                        )
                normalised[key] = arr
            batched_params = normalised
        self._batched_params = batched_params
        self._batched_dist: TFPDistribution = _construct_batched_dist(
            dist_cls,
            component=component,
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
                f"batch_shape={self._batch_shape} does not match the batch shape {actual} that "
                f"{dist_cls.__name__} gets from its parameters; every batched parameter must "
                f"broadcast to batch_shape"
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
            f"batch_shape={self._batch_shape}, component={self._component!r})"
        )

    # -- JAX pytree registration --------------------------------------------

    def tree_flatten(self):
        """Split the backend into JAX-traceable children + static aux.

        Children are the batched parameter values (the JAX-array
        leaves the user passed); aux carries everything needed to
        reconstruct the backend (the distribution class, the
        component of its laws, the declared ``batch_shape``, and the parameter keys
        in iteration order). The wrapped ``_batched_dist`` is
        reconstructed inside ``tree_unflatten`` from the params, so
        successive ``jit`` / ``vmap`` traces stay consistent.
        """
        keys = tuple(self._batched_params.keys())
        children = tuple(self._batched_params[k] for k in keys)
        aux = (self._dist_cls, self._component, self._batch_shape, keys)
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
        dist_cls, component, batch_shape, keys = aux
        instance = cls.__new__(cls)
        instance._dist_cls = dist_cls
        instance._component = component
        instance._batch_shape = tuple(batch_shape)
        instance._batched_params = dict(zip(keys, children))
        instance._batched_dist = _construct_batched_dist(
            dist_cls,
            component=component,
            batched_params=instance._batched_params,
        )
        return instance


jax.tree_util.register_pytree_node_class(_TFPArrayBackend)
