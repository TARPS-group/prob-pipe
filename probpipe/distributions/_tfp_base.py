"""TFPDistribution base class for distributions backed by TFP instances.

Factored out of ``core/distribution.py`` because no ``core/`` module
imports ``TFPDistribution`` – it is only used by the concrete
distribution modules in ``distributions/``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

from .._array_utils import _slice_leading_axes
from ..core._distribution_base import Distribution
from ..core.distribution import (
    NumericRecordDistribution,
    _mc_expectation,
)
from ..core.protocols import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..core.record import Record, RecordTemplate
from ..custom_types import Array, ArrayLike, PRNGKey


# ---------------------------------------------------------------------------
# Internal bypass for the upcoming batched-parameters rejection
# ---------------------------------------------------------------------------
# The next commit in this PR (PR-C.2) adds a runtime check inside
# ``TFPDistribution.__init__`` that raises ``ValueError`` whenever a
# concrete subclass (``Normal``, ``Beta``, ``Gamma``, …) is constructed
# with parameters whose ``tfd.Distribution.batch_shape`` is non-empty.
# The framework hierarchy rule (CONTRIBUTING.md) is "one random
# variable per ``Distribution``"; collections live in
# ``DistributionArray``.
#
# Some library-internal call sites legitimately need the legacy
# batched form — the fused-storage ``_TFPArrayBackend`` (PR-C.1)
# wraps a TFP-batched dist; converters, sequential joints, and
# random functions construct ``Normal(loc=batch_array, ...)`` from
# arrays produced inside the WF sweep. These sites bypass the
# upcoming rejection via the ``_allow_batched_tfp_init`` context
# manager added here. User-facing callers never use the bypass.
#
# This commit ships only the scaffolding (flag + context manager).
# The rejection itself lands later in the PR after every legitimate
# internal site has been wrapped and every test that exercises the
# legacy user-facing form has been migrated to
# ``DistributionArray.from_batched_params``.

_ALLOW_BATCHED_INIT: bool = False
"""Module-level toggle consulted by the upcoming
``TFPDistribution.__init__`` rejection. Default ``False`` enforces the
"one RV per Distribution" rule for all user-facing construction;
internal infra opts in with :func:`_allow_batched_tfp_init`."""


@contextlib.contextmanager
def _allow_batched_tfp_init() -> Iterator[None]:
    """Context manager: allow TFP-backed constructors to accept
    parameters whose implied ``batch_shape`` is non-empty.

    Used by library-internal infrastructure that legitimately needs
    to construct a TFP-batched form (e.g. :class:`_TFPArrayBackend`,
    converters that produce batched-Normal forms during a WF sweep,
    sequential-joint sampling whose lambda receives a batched
    sample). User code never uses this.

    The rejection it bypasses is added in a subsequent commit in this
    PR; until then this context manager is a no-op functionally but
    is referenced by the internal call sites that will need it.
    """
    global _ALLOW_BATCHED_INIT
    previous = _ALLOW_BATCHED_INIT
    _ALLOW_BATCHED_INIT = True
    try:
        yield
    finally:
        _ALLOW_BATCHED_INIT = previous


class TFPDistribution(
    NumericRecordDistribution,
    SupportsSampling,
    SupportsLogProb,
    SupportsMean,
    SupportsVariance,
    SupportsCovariance,
):
    """
    Base class for distributions backed by a ``tfd.Distribution`` instance.

    Subclasses set ``self._tfp_dist`` in ``__init__``.  The private
    protocol methods ``_sample``, ``_expectation``, ``_log_prob``,
    ``_mean``, and ``_variance`` all delegate to TFP (or use MC
    fallback for expectations).

    Inherits from :class:`SupportsSampling`, :class:`SupportsExpectation`,
    :class:`SupportsLogProb` (provides ``_prob``,
    ``_unnormalized_log_prob``, ``_unnormalized_prob`` defaults),
    :class:`SupportsMean`, and :class:`SupportsVariance`.

    Rejects batched parameters
    --------------------------
    Per the framework hierarchy "one random variable per
    ``Distribution``" rule (CONTRIBUTING.md), the constructor raises
    :class:`ValueError` when the underlying ``tfd.Distribution`` has
    a non-empty ``batch_shape``. Wrap multiple distributions in a
    :class:`~probpipe.DistributionArray` instead — the migration
    factory is :meth:`~probpipe.DistributionArray.from_batched_params`
    (or the per-class alias :meth:`Distribution.from_batched_params`).

    The check fires in ``__init__`` after ``super().__init__(name=name)``
    completes, so concrete subclasses that set ``self._tfp_dist``
    *before* calling ``super().__init__`` (the standard pattern used
    by ``Normal``, ``Beta``, ``Gamma``, …) are validated. Subclasses
    that set ``_tfp_dist`` *after* ``super().__init__`` (e.g.
    :class:`~probpipe.distributions.kde.KDEDistribution`) are skipped
    via the ``hasattr`` guard — those classes are responsible for
    their own shape invariants and don't go through TFP's batched
    parameter convention.

    Internal infrastructure that legitimately needs the batched form
    (the ``_TFPArrayBackend`` fused storage, converters, sequential
    joints, GRF predictions) opts into the bypass via
    :func:`_allow_batched_tfp_init`.
    """

    _tfp_dist: tfd.Distribution
    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __init__(self, *, name: str) -> None:
        """Final-stage initializer for TFP-backed distributions.

        Concrete subclasses (``Normal``, ``Beta``, …) set
        ``self._tfp_dist`` in their own ``__init__`` *before* calling
        ``super().__init__(name=name)``, so by the time we get here
        the TFP backend is fully constructed and we can validate its
        ``batch_shape``.
        """
        super().__init__(name=name)
        if _ALLOW_BATCHED_INIT:
            return
        # KDE-style subclasses set ``_tfp_dist`` *after* this call;
        # skip the check rather than crash on a missing attribute.
        # Such classes are responsible for their own shape invariants.
        tfp_dist = getattr(self, "_tfp_dist", None)
        if tfp_dist is None:
            return
        actual = tuple(tfp_dist.batch_shape)
        if actual != ():
            cls_name = type(self).__name__
            raise ValueError(
                f"{cls_name} parameters imply batch_shape={actual}; "
                f"wrap multiple distributions in a DistributionArray "
                f"instead. See "
                f"DistributionArray.from_batched_params({cls_name}, ...) "
                f"(or the alias {cls_name}.from_batched_params(...)) "
                f"for the factory."
            )

    # -- record_template auto-generation ------------------------------------

    @property
    def record_template(self):
        """Auto-build record_template from name + event_shape when named."""
        tpl = getattr(self, "_record_template", None)
        if tpl is not None:
            return tpl
        name = getattr(self, "_name", None)
        if name is not None:
            tpl = RecordTemplate(**{name: self.event_shape})
            object.__setattr__(self, "_record_template", tpl)
            return tpl
        return None

    # -- shape delegation ---------------------------------------------------

    @property
    def event_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.event_shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self._tfp_dist.batch_shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self._tfp_dist.dtype

    @property
    def dtypes(self) -> dict[str, jnp.dtype]:
        """Per-field dtypes (single field for TFPDistribution)."""
        tpl = self.record_template
        if tpl is not None:
            return {name: self.dtype for name in tpl.fields}
        return {}

    @property
    def support(self):
        """The support of this distribution.  Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}.support")

    @property
    def supports(self) -> dict[str, any]:
        """Per-field support constraints (single field for TFPDistribution)."""
        tpl = self.record_template
        if tpl is not None:
            return {name: self.support for name in tpl.fields}
        return {}

    # -- sampling & density -------------------------------------------------

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw samples using TFP's efficient batched sampling."""
        return self._tfp_dist.sample(seed=key, sample_shape=sample_shape)

    def _log_prob(self, x: ArrayLike) -> Array:
        return self._tfp_dist.log_prob(jnp.asarray(x))

    def _mean(self) -> Array:
        return self._tfp_dist.mean()

    def _variance(self) -> Array:
        return self._tfp_dist.variance()

    def _cov(self) -> Array:
        if self.event_shape == () or self.event_shape == (1,):
            return self._tfp_dist.variance()
        return self._tfp_dist.covariance()

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        return _mc_expectation(
            self, f, key=key, num_evaluations=num_evaluations,
            return_dist=return_dist,
        )

    # -- SupportsArrayBackend (fused storage for DistributionArray) ----------

    @classmethod
    def _make_array_backend(
        cls,
        *,
        name: str,
        batch_shape: tuple[int, ...],
        **batched_params: Any,
    ) -> "_TFPArrayBackend":
        """Construct a fused TFP-batched backend for ``DistributionArray``.

        Inherited automatically by every concrete TFP-backed distribution
        (``Normal``, ``Beta``, ``Gamma``, ``MultivariateNormal``, …); the
        same code path covers the whole family because the wrapped TFP
        constructor handles the per-class param-name mapping.

        See :class:`probpipe.core.protocols.SupportsArrayBackend` for the
        protocol contract. Additive — ``DistributionArray.from_batched_params``
        is the only consumer; user code never calls this directly.
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
    dist_cls : type
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
        dist_cls: type,
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
        # Construct the fused ProbPipe Distribution with the batched
        # params. ``TFPDistribution.__init__`` rejects batched
        # parameters for user code; the backend is internal infra that
        # exists *to* hold a batched form, so it opts into the bypass
        # via ``_allow_batched_tfp_init``.
        with _allow_batched_tfp_init():
            self._batched_dist: TFPDistribution = dist_cls(
                **batched_params,
                name=f"{name}{_ARRAY_BACKEND_NAME_SUFFIX}",
            )
        # Final sanity check: TFP's inferred batch_shape must match
        # the caller's declaration. Catches the rare case where a
        # higher-rank param's *trailing* axes don't agree but the
        # leading-axes check above passed (e.g., MVN where ``loc`` /
        # ``scale_tril`` event ranks differ).
        actual = self._batched_dist.batch_shape
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
            key: _slice_leading_axes(value, multi)
            for key, value in self._batched_params.items()
        }
        return self._dist_cls(
            **scalar_params,
            name=f"{self._name}_{flat}",
        )

    def _normalize_index(
        self, index: int | tuple[int, ...]
    ) -> tuple[tuple[int, ...], int]:
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
                        f"_TFPArrayBackend.cell: index {i} out of range "
                        f"for batch_shape={bshape}."
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
    def tree_unflatten(cls, aux, children) -> "_TFPArrayBackend":
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
        instance._batched_dist = dist_cls(
            **instance._batched_params,
            name=f"{name}{_ARRAY_BACKEND_NAME_SUFFIX}",
        )
        return instance


jax.tree_util.register_pytree_node_class(_TFPArrayBackend)
