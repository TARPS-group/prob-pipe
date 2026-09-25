"""``NumericRecordDistribution`` and its closely-related helpers.

The primary class is :class:`NumericRecordDistribution` — a
:class:`~probpipe.core._record_distribution.RecordDistribution` whose
draws are numeric, adding the flat-vector interface (``event_size``,
``flatten_value`` / ``unflatten_value``, ``as_flat_distribution``) to the
schema views every distribution reads off its declaration. It is the base
class for every numeric ProbPipe distribution (``Normal``, ``Beta``,
``ProductDistribution``, ...).

Provides:

  - :class:`NumericRecordDistribution` — the base class.
  - :class:`FlatNumericRecordDistribution` — the flat-shaped subset
    (single field, ``event_shape=(N,)``), used as the input type for
    algorithms that consume a flat parameter vector and as the source
    of :meth:`~FlatNumericRecordDistribution.as_record_distribution`.
  - :class:`BootstrapDistribution` — MC error tracking via bootstrap
    resampling.
  - :class:`FlattenedDistributionView` — flat view of any distribution
    (always a ``FlatNumericRecordDistribution`` by construction).
  - :class:`NumericRecordDistributionView` — inverse, lifting a flat
    source to a Record-keyed view under a user-supplied template.
  - Private helpers ``_vmap_sample`` / ``_mc_expectation``.

Distinct from :class:`~probpipe.DistributionArray` (housed in
:mod:`_distribution_array`), which represents *n independent
distributions stacked along a batch axis* — many random variables
indexed by position, e.g. ``Normal(loc=jnp.zeros(5), scale=1.0)``
stored as a length-5 array of independent ``Normal`` instances. A
``NumericRecordDistribution`` represents *one* random variable
whose draw can itself have a numeric-valued event structure (a
scalar, a vector, or a multi-field record), and ``DistributionArray``
holds many such variables. The two compose: a ``DistributionArray``
of ``NumericRecordDistribution`` instances is the canonical way to
express a vectorized batch of structured random variables.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._specs import NumericRecordSpec

import jax
import jax.numpy as jnp

from .._dtype import _as_float_array
from .._weights import Weights
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions import _distribution as _base
from ..distributions._distribution import Distribution, NumericDistribution
from . import _workflow_broker, _workflow_descendants
from ._record_distribution import (
    RecordDistribution,
    _field_event_shape,
    _record_with_leaves,
)
from ._specs import NumericArraySpec, OutputSpec
from .constraints import (
    _supports_compatible,
    real,
)
from .protocols import (
    SupportsCovariance,
    SupportsExpectation,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)

# ---------------------------------------------------------------------------
# Sampling & expectation helpers
# ---------------------------------------------------------------------------


def _vmap_sample(
    dist: NumericRecordDistribution,
    key: PRNGKey,
    sample_shape: tuple[int, ...] = (),
) -> Any:
    """Draw samples via ``jax.vmap`` over ``dist._sample(key, ())``.

    Convenience for distributions whose ``_sample`` implementation is
    naturally a single-draw function: call this helper from ``_sample``
    and it will handle the ``sample_shape`` prefix by splitting keys
    and vmap-ing over the single-draw path.

    Parameters
    ----------
    dist : NumericRecordDistribution
        Distribution whose ``_sample(key, ())`` draws one unbatched
        sample (array or pytree of arrays).
    key : PRNGKey
        JAX PRNG key.
    sample_shape : tuple of int
        Shape prefix for independent draws.
    """

    def _one(k: PRNGKey) -> Any:
        return dist._sample(k, ())

    if sample_shape == ():
        return _one(key)
    n = prod(sample_shape)
    keys = jax.random.split(key, n)
    flat_samples = jax.vmap(_one)(keys)
    return jax.tree.map(
        lambda x: x.reshape(*sample_shape, *x.shape[1:]),
        flat_samples,
    )


def _mc_expectation(
    dist: NumericRecordDistribution,
    f: Callable[[Any], Any],
    *,
    key: PRNGKey | None = None,
    num_evaluations: int | None = None,
    return_dist: bool | None = None,
) -> Any:
    """Estimate ``E[f(X)]`` where ``X ~ dist`` via Monte Carlo.

    Parameters
    ----------
    dist
        Distribution with a ``_sample(key, sample_shape)`` method.
    f : callable
        Function mapping a single sample to an array (or pytree of arrays).
    key : PRNGKey, optional
        JAX PRNG key for sampling.  Auto-generated if ``None``.
    num_evaluations : int, optional
        Number of samples to draw.  If ``None``, uses
        ``DEFAULT_NUM_EVALUATIONS``.
    return_dist : bool, optional
        If ``True``, return a ``BootstrapDistribution`` capturing
        estimation uncertainty.  If ``False``, return a plain array.
        If ``None``, use the global ``RETURN_APPROX_DIST`` setting.
    """
    n = num_evaluations if num_evaluations is not None else _base.DEFAULT_NUM_EVALUATIONS
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"num_evaluations must be an integer; got {n!r}")
    if n <= 0:
        raise ValueError(f"num_evaluations must be positive; got {n!r}")
    if key is None:
        captured = _workflow_descendants.capture_stochastic_consumer(dist)
        key = _workflow_broker._resolve_automatic_key(
            None,
            _workflow_broker._singleton_effect_plan(
                operation_kind="expectation",
                execution_mode="monte_carlo",
                sample_shape=(n,),
                record_path=captured.record_path,
                descendant_descriptor=captured.descendant_descriptor,
            ),
        )
        samples = _workflow_descendants.sample_captured_consumer(captured, key, (n,))
    else:
        samples = dist._sample(key, sample_shape=(n,))
    evals = jax.vmap(f)(samples)

    rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
    if rd:
        return BootstrapDistribution("expectation", evals)
    return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)


def _raw_event_shape(law: Distribution) -> tuple[int, ...]:
    """The shape of a raw array draw of *law*, which ``flatten_value`` needs, else ``()``.

    A record draw carries its own structure, so flattening one reads no shape.
    """
    if not isinstance(law.event_spec.spec, NumericArraySpec):
        return ()
    return law.event_shape


# ---------------------------------------------------------------------------
# NumericRecordDistribution — RecordDistribution + numeric shape semantics
# ---------------------------------------------------------------------------


class NumericRecordDistribution(RecordDistribution, NumericDistribution):
    """Distribution over numeric arrays with Record support.

    Extends :class:`RecordDistribution` with numeric-specific metadata
    (per-field shape, dtype, and support). The class is the most
    general numeric random variable in ProbPipe: one draw is a pytree
    of ``jax.Array`` leaves named via a :class:`RecordSpec`.
    Single-leaf distributions (``Normal``, ``Beta``,
    ``MultivariateNormal``, ...) are the trivial case; joint
    distributions (``ProductDistribution``, ``SequentialJointDistribution``,
    ``JointGaussian``, ...) reuse the same machinery with a multi-leaf
    template.

    A ``Distribution`` represents one random variable; collections of
    independent distributions live in
    :class:`~probpipe.DistributionArray`.

    Schema views
    ------------

    ``event_shape`` is the base's view on the event declaration, and
    ``dtypes``, ``supports``, ``dtype``, and ``support`` are those of
    :class:`~probpipe.NumericDistribution`, whose marker this class claims,
    since every instance draws a numeric value. The per-field
    ``event_shapes``, the flat ``event_size``, and :attr:`treedef` read the
    record the declaration presents, an interim implementation detail.

    ``_sample`` contract
    --------------------

    Subclasses implement ``_sample(key, sample_shape) -> draw``; the
    public :func:`~probpipe.sample` op (in
    :mod:`probpipe.core.ops`) handles key auto-generation,
    protocol dispatch, and source/provenance tracking before delegating
    to ``dist._sample(...)``. Subclass code should never call the public
    ``sample`` op on ``self`` — call ``self._sample(key, sample_shape)``
    directly to avoid the ops layer.

    The shape of one draw is fully determined by ``event_template``:

    - **Single-leaf** template → ``_sample(key, sample_shape)`` returns
      a raw ``jax.Array`` of shape ``sample_shape + event_shape``.
    - **Multi-leaf** template → ``_sample(key, sample_shape)`` returns a
      :class:`~probpipe.NumericRecord` (or a
      :class:`~probpipe.NumericRecordBatch` over one ``draw`` level for a
      non-empty ``sample_shape``) keyed by ``event_template.fields``.

    The :attr:`treedef` property locks this invariant by deriving from
    ``event_template``.

    Standard distributions (``Normal``, ``Gamma``, ``Poisson``, ...)
    inherit from this class via :class:`TFPDistribution`.
    """

    def _check_support_compatible(
        self,
        source: NumericRecordDistribution,
    ) -> None:
        """Raise ``ValueError`` if *source*'s per-field supports are
        incompatible with *self*'s (the target's) per-field supports.

        Called post-construction by the converter, so both sides expose
        instance-level ``supports`` — no class-level default-support
        approximation. For a single-field target (the common case),
        every source field's support is compared against the lone
        target support. For a multi-field target, supports pair up
        field-by-field in insertion order; field-count mismatches raise
        ``ValueError`` rather than silently truncating via ``zip``.

        Sources that don't expose per-field supports (non-NRD endpoints
        like ``EmpiricalDistribution`` with object-dtype data) are
        treated as "unknown" and the check returns without complaint.
        """
        try:
            target_per_field = self.supports
            source_per_field = source.supports
        except (NotImplementedError, AttributeError):
            return

        multi_leaf_source = len(source_per_field) > 1

        if len(target_per_field) == 1:
            target_support = next(iter(target_per_field.values()))
            for field_name, source_support in source_per_field.items():
                if _supports_compatible(source_support, target_support):
                    continue
                field_part = f" field {field_name!r}" if multi_leaf_source else ""
                raise ValueError(
                    f"Cannot convert {type(source).__name__}{field_part} "
                    f"(support={source_support}) to {type(self).__name__} "
                    f"(support={target_support}). "
                    f"Pass check_support=False to override."
                )
            return

        # Multi-field target — field counts must match to pair
        # positionally; ``zip`` would silently truncate, hiding bugs
        # where the converter produced a target with the wrong arity.
        if len(source_per_field) != len(target_per_field):
            raise ValueError(
                f"Cannot convert {type(source).__name__} "
                f"({len(source_per_field)} fields: "
                f"{tuple(source_per_field)}) to {type(self).__name__} "
                f"({len(target_per_field)} fields: "
                f"{tuple(target_per_field)}): field-count mismatch. "
                f"Pass check_support=False to override."
            )
        for (s_name, s_sup), (t_name, t_sup) in zip(
            source_per_field.items(),
            target_per_field.items(),
        ):
            if _supports_compatible(s_sup, t_sup):
                continue
            raise ValueError(
                f"Cannot convert {type(source).__name__} field "
                f"{s_name!r} (support={s_sup}) to "
                f"{type(self).__name__} field {t_name!r} "
                f"(support={t_sup}). "
                f"Pass check_support=False to override."
            )

    # -- Single-leaf pytree interface -----------------------------------------

    @property
    def treedef(self) -> jax.tree_util.PyTreeDef:
        """Treedef of one sample, derived from the declaration.

        Locks the relationship between the declared kind and the sample's
        pytree structure:

        - A law that draws one array → a leaf treedef
          (``jax.tree.structure(None)``), since ``_sample`` returns a raw
          ``jax.Array``.
        - A law that draws a record → the treedef of a ``NumericRecord``
          skeleton with the same field names, since ``_sample`` returns a
          ``NumericRecord``.

        Cached on first read; the declaration is immutable after
        construction, so the cache is always valid.
        """
        cached = getattr(self, "_treedef", None)
        if cached is not None:
            return cached
        spec = self.event_spec.spec
        if isinstance(spec, NumericArraySpec):
            td = jax.tree.structure(None)
        else:
            from .record import Record

            placeholder = Record(
                self.name,
                {name: jnp.zeros(_field_event_shape(spec, name)) for name in spec.fields},
            )
            td = jax.tree.structure(placeholder)
        object.__setattr__(self, "_treedef", td)
        return td

    @property
    def flat_event_shapes(self) -> list[tuple[int, ...]]:
        """List of per-field event shapes in template field order.

        Tree-walk over :attr:`event_shapes`: ``list(event_shapes.values())``.
        For a single-field distribution this is ``[event_shape]``;
        for a multi-leaf distribution it's one entry per leaf.
        """
        return list(self.event_shapes.values())

    @property
    def event_size(self) -> int:
        """Total number of scalar elements in one sample, the declaration's ``vector_size``.

        Raises
        ------
        ValueError
            If the declaration has unbound dimensions.
        """
        return self.event_spec.spec.vector_size

    @staticmethod
    def flatten_value(value, *, event_shape: tuple[int, ...] = ()) -> Array:
        """Flatten a sample to a flat trailing axis.

        Accepts a ``Record``, a ``NumericRecord``, or a batch of either
        (which already carry their template) or a raw array. Raw-array
        inputs need ``event_shape`` to disambiguate batch axes from
        event axes; without it, the input gets a trailing singleton
        axis (matching the scalar-event default).
        """
        from ._numeric_record import NumericRecord
        from ._numeric_record_batch import NumericRecordBatch
        from .record import Record

        if isinstance(value, (NumericRecordBatch, NumericRecord)):
            return value.to_vector()
        if isinstance(value, Record):
            return value.to_numeric().to_vector()
        value = jnp.asarray(value)
        if not event_shape:
            return value[..., None]
        n_event = prod(event_shape)
        n_batch = value.ndim - len(event_shape)
        return value.reshape(*value.shape[:n_batch], n_event)

    @staticmethod
    def unflatten_value(flat, *, template):
        """Unflatten a flat trailing axis back to what *template* declares.

        *template* is a law's declared term. An array spec rebuilds a raw
        array of shape ``(*batch, *shape)``, as a law drawing one array
        samples and ``_log_prob`` reads. A record spec rebuilds a
        ``NumericRecord`` from a 1-D *flat* and a ``NumericRecordBatch`` from
        a batched one, whatever its number of fields.
        """
        flat = jnp.asarray(flat)
        if isinstance(template, NumericArraySpec):
            return flat.reshape(*flat.shape[:-1], *template.shape)
        from ._numeric_record import _reconstruct_from_vector

        # ``_reconstruct_from_vector`` selects single (NumericRecord) vs
        # batched (NumericRecordBatch) from the rank of ``flat``.
        return _reconstruct_from_vector("value", template, flat)

    def as_flat_distribution(self) -> FlatNumericRecordDistribution:
        """View this distribution as a flat distribution.

        Returns a :class:`FlattenedDistributionView` wrapping this
        distribution. The view satisfies the
        :class:`FlatNumericRecordDistribution` contract regardless of
        ``self``'s structure (multi-field, multi-dim event, …) — its
        ``event_shape`` is always ``(self.event_size,)``.

        Inverse: :meth:`FlatNumericRecordDistribution.as_record_distribution`.
        """
        return FlattenedDistributionView(self)

    def as_record_distribution(
        self,
        *,
        template: NumericRecordSpec,
        name: str | None = None,
    ) -> NumericRecordDistribution:
        """Lift this distribution to a Record-keyed view under *template*.

        **Only available on :class:`FlatNumericRecordDistribution` subclasses.**
        Calling this on a non-flat :class:`NumericRecordDistribution`
        raises :class:`TypeError` with a hint to call
        :meth:`as_flat_distribution` first.

        See :meth:`FlatNumericRecordDistribution.as_record_distribution`
        for the actual implementation and parameters.
        """
        raise TypeError(
            f"as_record_distribution is only available on "
            f"FlatNumericRecordDistribution subclasses. "
            f"{type(self).__name__} is not flat. Chain: "
            f"source.as_flat_distribution().as_record_distribution(template=...)."
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = [type(self).__name__]
        if self.name:
            parts.append(f"name={self.name!r}")
        # Multi-field NRDs (joints) can't summarise the event with a
        # single ``event_shape`` — ``_single_field_name`` raises
        # ``TypeError`` there, and the base default raises
        # ``NotImplementedError`` for subclasses that haven't overridden
        # ``event_shape``. Either way, fall back to the per-field dict.
        try:
            parts.append(f"event_shape={self.event_shape}")
        except (TypeError, NotImplementedError):
            parts.append(f"event_shapes={self.event_shapes}")
        return f"{parts[0]}({', '.join(parts[1:])})"


# ---------------------------------------------------------------------------
# BootstrapDistribution
# ---------------------------------------------------------------------------


class BootstrapDistribution(
    NumericRecordDistribution, SupportsSampling, SupportsMean, SupportsVariance
):
    """Distribution over bootstrap-resampled means of a statistic.

    Given *n* evaluations ``f(x_1), ..., f(x_n)`` where ``x_i ~ P``,
    this represents the sampling distribution of the sample mean
    ``(1/n) sum f(x_i)``, capturing Monte Carlo error.

    Parameters
    ----------
    name : str
        Distribution name.
    evaluations : array-like, shape ``(n, *stat_shape)``
        The individual ``f(x_i)`` values.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Non-negative weights (normalized internally).  A pre-built
        :class:`~probpipe.Weights` object is also accepted.  Mutually
        exclusive with *log_weights*.  When neither is given, uniform
        weights are used.
    log_weights : array-like, :class:`~probpipe.Weights`, or None
        Log-unnormalized weights.  A pre-built :class:`~probpipe.Weights`
        object is also accepted.  Mutually exclusive with *weights*.
    """

    def __init__(
        self,
        name: str,
        evaluations: ArrayLike,
        *,
        weights: ArrayLike | Weights | None = None,
        log_weights: ArrayLike | Weights | None = None,
    ):
        self._evaluations = _as_float_array(evaluations)
        if self._evaluations.ndim == 0:
            raise ValueError("evaluations must have at least 1 dimension.")
        self._num_atoms = self._evaluations.shape[0]
        self._w = Weights(
            n=self._num_atoms,
            weights=weights,
            log_weights=log_weights,
        )
        # A draw is one resampled mean, an array of the statistic's shape.
        super().__init__(
            name,
            NumericArraySpec(self._evaluations.shape[1:], self._evaluations.dtype, real),
        )
        self._approximate = True

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    @property
    def num_atoms(self) -> int:
        """Number of stored atoms (function evaluations) backing this distribution."""
        return self._num_atoms

    @property
    def evaluations(self) -> Array:
        return self._evaluations

    def _mean(self) -> Array:
        """Point estimate: (weighted) mean of evaluations."""
        return self._w.mean(self._evaluations)

    def _variance(self) -> Array:
        """Variance of the sampling distribution (approx Var[f(X)] / n_eff)."""
        sample_var = self._w.variance(self._evaluations)
        return sample_var / self._w.effective_sample_size

    def _sample(
        self,
        key: PRNGKey,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        """Draw bootstrap resamples of the mean."""

        def _one_resample(k):
            idx = self._w.choice(k, shape=(self._num_atoms,))
            return jnp.mean(self._evaluations[idx], axis=0)

        if sample_shape == ():
            return _one_resample(key)
        total = prod(sample_shape)
        keys = jax.random.split(key, total)

        results = jax.vmap(_one_resample)(keys)
        return results.reshape(sample_shape + self.event_shape)

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        return _mc_expectation(
            self,
            f,
            key=key,
            num_evaluations=num_evaluations,
            return_dist=return_dist,
        )

    def __repr__(self) -> str:
        return f"BootstrapDistribution(num_atoms={self._num_atoms}, event_shape={self.event_shape})"


# ---------------------------------------------------------------------------
# FlatNumericRecordDistribution — the flat-shaped subset of NRD
# ---------------------------------------------------------------------------


class FlatNumericRecordDistribution(NumericRecordDistribution):
    """A :class:`NumericRecordDistribution` whose samples are flat 1-D vectors.

    The flat contract:

    * exactly one field (``len(fields) == 1``)
    * ``event_shape == (N,)`` for some ``N``
    * samples shaped ``sample_shape + (N,)``

    Algorithms that operate on a flat parameter vector — MCMC kernels,
    optimisers, Hessian / curvature builders, variational families,
    Pathfinder / Laplace surrogates — should declare their input as
    :class:`FlatNumericRecordDistribution`. The natively-multivariate
    parametrics (:class:`~probpipe.MultivariateNormal`,
    :class:`~probpipe.Dirichlet`, :class:`~probpipe.Multinomial`,
    :class:`~probpipe.VonMisesFisher`) and
    :class:`FlattenedDistributionView` all satisfy this contract.

    Scalar parametrics (``Normal``, ``Beta``, …) have
    ``event_shape == ()`` and do **not** satisfy the contract directly;
    call :meth:`~NumericRecordDistribution.as_flat_distribution` to get
    a :class:`FlattenedDistributionView` (whose event_shape is ``(1,)``).

    This class is also the home of
    :meth:`as_record_distribution` — the inverse of
    :meth:`~NumericRecordDistribution.as_flat_distribution`. Receiver
    typing means non-flat callers fail at the type level rather than at
    a runtime shape check.
    """

    @property
    def vector_size(self) -> int:
        """Length of the per-element 1-D vector — equal to ``event_shape[0]``.

        Validates the flat contract on access: subclasses with
        non-1-D ``event_shape`` raise ``TypeError`` here rather than
        silently truncating to the first dimension.
        """
        es = self.event_shape
        if len(es) != 1:
            raise TypeError(
                f"{type(self).__name__} declares FlatNumericRecordDistribution "
                f"but has event_shape={es}; expected 1-D (N,)."
            )
        return es[0]

    def as_record_distribution(
        self,
        *,
        template: NumericRecordSpec,
        name: str | None = None,
    ) -> NumericRecordDistribution:
        """Lift this flat distribution to a Record-keyed view under *template*.

        Inverse of :meth:`~NumericRecordDistribution.as_flat_distribution`.
        Samples come back as :class:`NumericRecord` /
        :class:`NumericRecordBatch` keyed by ``template.fields``.

        Parameters
        ----------
        template : NumericRecordSpec
            Target structural skeleton. Must be a
            :class:`NumericRecordSpec` — opaque (``None``) leaves
            cannot be reconstructed from a flat numeric array.
        name : str, optional
            Name for the lifted distribution. Defaults to ``self.name``.

        Returns
        -------
        NumericRecordDistribution
            A thin view over ``self``. Sampling, log-prob, moments, and
            ``expectation`` delegate to the source and reshape via the
            template. Capability protocols match the source.

        Raises
        ------
        TypeError
            If ``template`` is not a ``NumericRecordSpec``.
        ValueError
            If ``self.vector_size`` does not match ``template.vector_size``.
        """
        from ._specs import NumericRecordSpec

        if not isinstance(template, NumericRecordSpec):
            raise TypeError(
                f"as_record_distribution requires a NumericRecordSpec, "
                f"got {type(template).__name__}. Opaque (None) leaves "
                f"cannot be reconstructed from a flat numeric array."
            )
        if self.vector_size != template.vector_size:
            raise ValueError(
                f"vector_size mismatch: source vector_size={self.vector_size}, "
                f"template.vector_size={template.vector_size}."
            )
        cls = _numeric_record_distribution_view_class_for_base(self)
        return cls(self, template, name=name)


# ---------------------------------------------------------------------------
# FlattenedDistributionView — wrap any distribution as a flat NRD
# ---------------------------------------------------------------------------

_FLATTENED_VIEW_CLASS_CACHE: dict[frozenset[str], type] = {}


def _flattened_distribution_view_class_for_base(base: Distribution) -> type:
    """Return a ``FlattenedDistributionView`` subclass whose protocol bases
    match the capabilities of *base*.

    The view only delegates sampling and log-prob; those are the only
    protocols that make sense to inherit. A view over a log-prob-only
    base should not advertise ``SupportsSampling``, and vice versa.
    """
    protocols: set[str] = set()
    if isinstance(base, SupportsSampling):
        protocols.add("sample")
    if isinstance(base, SupportsLogProb):
        protocols.add("log_prob")

    key = frozenset(protocols)
    if key in _FLATTENED_VIEW_CLASS_CACHE:
        return _FLATTENED_VIEW_CLASS_CACHE[key]

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "sample" in protocols:
        extra_bases.append(SupportsSampling)

        def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()) -> Array:
            pytree_samples = self._base._sample(key, sample_shape)
            return self._base.flatten_value(
                pytree_samples,
                event_shape=_raw_event_shape(self._base),
            )

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x: ArrayLike) -> Array:
            x = jnp.asarray(x)
            value = self._base.unflatten_value(x, template=self._base.event_spec.spec)
            return self._base._log_prob(value)

        extra_methods["_log_prob"] = _log_prob

    if not extra_bases:
        _FLATTENED_VIEW_CLASS_CACHE[key] = FlattenedDistributionView
        return FlattenedDistributionView

    new_cls = type(
        "FlattenedDistributionView",
        (FlattenedDistributionView, *extra_bases),
        extra_methods,
    )
    _workflow_descendants._register_unsupported_descendant_type(
        new_cls,
        "FlattenedDistributionView",
    )
    _FLATTENED_VIEW_CLASS_CACHE[key] = new_cls
    return new_cls


class FlattenedDistributionView(FlatNumericRecordDistribution):
    """Wraps a distribution as a flat :class:`FlatNumericRecordDistribution`.

    Sampling produces flat vectors of shape ``(event_size,)``, and
    ``_log_prob`` accepts flat vectors and delegates to the wrapped
    distribution after unflattening.

    This is the primary interoperability mechanism: any algorithm written
    against :class:`FlatNumericRecordDistribution` works with an
    arbitrary :class:`RecordDistribution` /
    :class:`NumericRecordDistribution` via
    ``dist.as_flat_distribution()``.

    **Dynamic protocol support:** the view's ``isinstance`` compliance
    matches the base's capabilities — a log-prob-only base produces a
    view that is not ``SupportsSampling``, and a sampling-only base
    produces one that is not ``SupportsLogProb``.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(cls, base: Distribution):
        actual_cls = _flattened_distribution_view_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(self, base: Distribution):
        self._base = base
        # The view preserves the base's construction-time name. A draw is one
        # real vector, whose component is named for the flat map, as a base's
        # name need not be a component name.
        self._init_tracked(base.name)
        self._init_declaration(
            OutputSpec(to_vector=NumericArraySpec((base.event_size,), base.dtype, real))
        )

    def _expectation(
        self,
        f: Callable,
        *,
        key: PRNGKey | None = None,
        num_evaluations: int | None = None,
        return_dist: bool | None = None,
    ) -> Any:
        return _mc_expectation(
            self,
            f,
            key=key,
            num_evaluations=num_evaluations,
            return_dist=return_dist,
        )

    @property
    def base_distribution(self) -> Distribution:
        """The underlying distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return self._base.unflatten_value(
            jnp.asarray(flat_sample),
            template=self._base.event_spec.spec,
        )

    def __repr__(self) -> str:
        return (
            f"FlattenedDistributionView(base={type(self._base).__name__}, "
            f"event_shape={self.event_shape})"
        )


# ---------------------------------------------------------------------------
# NumericRecordDistributionView — lift a flat distribution to a Record view
# ---------------------------------------------------------------------------

_LIFTED_VIEW_CLASS_CACHE: dict[type, type] = {}


def _numeric_record_distribution_view_class_for_base(base: Distribution) -> type:
    """Return a ``NumericRecordDistributionView`` subclass advertising the
    same capability protocols as *base*.

    Mirrors :func:`_flattened_distribution_view_class_for_base` for the
    inverse direction. The protocol-bearing methods (``_sample``,
    ``_log_prob``, ``_mean``, ``_variance``, ``_cov``, ``_expectation``)
    are attached dynamically by this factory rather than living on
    :class:`NumericRecordDistributionView` itself — otherwise every
    view would appear to satisfy every protocol by virtue of method
    presence (``@runtime_checkable`` semantics).
    """
    # Cache on the source's concrete type: any two instances of the same
    # Distribution subclass advertise the same protocol set, so the
    # frozenset key would collide anyway. Type-based caching avoids
    # six runtime_checkable isinstance scans on every construction.
    cached = _LIFTED_VIEW_CLASS_CACHE.get(type(base))
    if cached is not None:
        return cached

    # Imports hoisted once for all closures below (and to avoid
    # circular-import risk at module load time).
    from ._numeric_record import NumericRecord
    from ._numeric_record_batch import NumericRecordBatch

    protocols: set[str] = set()
    if isinstance(base, SupportsSampling):
        protocols.add("sample")
    if isinstance(base, SupportsLogProb):
        protocols.add("log_prob")
    if isinstance(base, SupportsMean):
        protocols.add("mean")
    if isinstance(base, SupportsVariance):
        protocols.add("variance")
    if isinstance(base, SupportsCovariance):
        protocols.add("cov")
    if isinstance(base, SupportsExpectation):
        protocols.add("expectation")

    extra_bases: list[type] = []
    extra_methods: dict[str, object] = {}

    if "sample" in protocols:
        extra_bases.append(SupportsSampling)

        def _sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()):
            base_sample = self._base._sample(key, sample_shape)
            flat = self._base.flatten_value(
                base_sample,
                event_shape=self._base.event_shape,
            )
            from ._numeric_record import _reconstruct_from_vector

            # ``_reconstruct_from_vector`` selects single (NumericRecord, flat
            # is 1-D) vs batched (NumericRecordBatch, batch_shape ==
            # sample_shape) from the rank of ``flat``.
            return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x) -> Array:
            if isinstance(x, (NumericRecord, NumericRecordBatch)):
                flat = x.to_vector()
            else:
                flat = jnp.asarray(x)
            value = self._base.unflatten_value(flat, template=self._base.event_spec.spec)
            return self._base._log_prob(value)

        extra_methods["_log_prob"] = _log_prob

    if "mean" in protocols:
        extra_bases.append(SupportsMean)

        def _mean(self):
            from ._numeric_record import _reconstruct_from_vector

            flat = self._base.flatten_value(
                self._base._mean(),
                event_shape=self._base.event_shape,
            )
            return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)

        extra_methods["_mean"] = _mean

    if "variance" in protocols:
        extra_bases.append(SupportsVariance)

        def _variance(self):
            from ._numeric_record import _reconstruct_from_vector

            flat = self._base.flatten_value(
                self._base._variance(),
                event_shape=self._base.event_shape,
            )
            return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)

        extra_methods["_variance"] = _variance

    if "cov" in protocols:
        extra_bases.append(SupportsCovariance)

        def _cov(self):
            # Covariance stays flat (event_size × event_size matrix).
            # The Record / field-block structure is implicit in the
            # template's flat ordering.
            return self._base._cov()

        extra_methods["_cov"] = _cov

    if "expectation" in protocols:
        extra_bases.append(SupportsExpectation)

        def _expectation(
            self,
            f: Callable,
            *,
            key: PRNGKey | None = None,
            num_evaluations: int | None = None,
            return_dist: bool | None = None,
        ) -> Any:
            # ``f`` operates on a Record-shaped sample. We can't pass the
            # batched ``NumericRecordBatch`` returned by ``self._sample``
            # through ``jax.vmap(f)`` directly — vmap strips the leading
            # axis from each leaf while preserving ``batch_shape`` aux,
            # producing an invariant violation. Instead, sample the base
            # in flat form (no aux-shape invariants) and run vmap over a
            # closure that unflattens to a Record inside the loop body.
            n = num_evaluations if num_evaluations is not None else _base.DEFAULT_NUM_EVALUATIONS
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError(f"num_evaluations must be an integer; got {n!r}")
            if n <= 0:
                raise ValueError(f"num_evaluations must be positive; got {n!r}")
            sample_key = key
            if sample_key is None:
                captured = _workflow_descendants.capture_stochastic_consumer(self)
                sample_key = _workflow_broker._resolve_automatic_key(
                    None,
                    _workflow_broker._singleton_effect_plan(
                        operation_kind="expectation",
                        execution_mode="monte_carlo",
                        sample_shape=(n,),
                        record_path=captured.record_path,
                        descendant_descriptor=captured.descendant_descriptor,
                    ),
                )
            base_samples = self._base._sample(sample_key, sample_shape=(n,))
            flat_samples = self._base.flatten_value(
                base_samples,
                event_shape=self._base.event_shape,
            )
            template = self.event_spec.spec
            dist_name = self.name

            def _f_on_flat(flat_row):
                from ._numeric_record import _reconstruct_from_vector

                return f(_reconstruct_from_vector(dist_name, template, flat_row))

            evals = jax.vmap(_f_on_flat)(flat_samples)
            rd = return_dist if return_dist is not None else _base.RETURN_APPROX_DIST
            if rd:
                return BootstrapDistribution("expectation", evals)
            return jax.tree.map(lambda v: jnp.mean(v, axis=0), evals)

        extra_methods["_expectation"] = _expectation

    if not extra_bases:
        _LIFTED_VIEW_CLASS_CACHE[type(base)] = NumericRecordDistributionView
        return NumericRecordDistributionView

    new_cls = type(
        "NumericRecordDistributionView",
        (NumericRecordDistributionView, *extra_bases),
        extra_methods,
    )
    _workflow_descendants._register_unsupported_descendant_type(
        new_cls,
        "NumericRecordDistributionView",
    )
    _LIFTED_VIEW_CLASS_CACHE[type(base)] = new_cls
    return new_cls


class NumericRecordDistributionView(NumericRecordDistribution):
    """View that lifts a flat distribution to a Record-keyed structure.

    Inverse of :class:`FlattenedDistributionView`. ``self._base`` is a
    :class:`FlatNumericRecordDistribution` (single-field, ``event_shape
    == (N,)``); ``self.event_template`` is the user-supplied
    :class:`NumericRecordSpec`, not the source's.

    Sampling, log-prob, and moments delegate to ``self._base`` and
    reshape via the template's flatten / unflatten machinery.
    Capability protocols match the source via
    :func:`_numeric_record_distribution_view_class_for_base`.

    Constructed via
    :meth:`FlatNumericRecordDistribution.as_record_distribution`.
    """

    _sampling_cost: str = "low"
    _preferred_orchestration: str | None = None

    def __new__(
        cls,
        base: Distribution,
        template: NumericRecordSpec,
        *,
        name: str | None = None,
    ):
        actual_cls = _numeric_record_distribution_view_class_for_base(base)
        return object.__new__(actual_cls)

    def __init__(
        self,
        base: Distribution,
        template: NumericRecordSpec,
        *,
        name: str | None = None,
    ):
        # Skip ``Distribution.__init__`` to avoid double-validation;
        # the TrackedTerm metaclass check still enforces a non-empty
        # ``_name``. ``base.name`` is guaranteed non-empty by that same
        # check, so the fallback is always valid.
        self._base = base
        if name is not None:
            self._init_tracked(name)
        else:
            # Fall back to the base's name.
            self._init_tracked(base.name)
        # A draw is the user-supplied record, every leaf taking the source's
        # dtype and support.
        object.__setattr__(self, "_event_template", template)
        self._init_declaration(_record_with_leaves(template, base.dtype, base.support))

    # ---- structural ---------------------------------------------------------

    @property
    def base_distribution(self) -> Distribution:
        """The underlying single-field flat distribution."""
        return self._base

    def __repr__(self) -> str:
        return (
            f"NumericRecordDistributionView(base={type(self._base).__name__}, "
            f"event_spec={self.event_spec.spec!r})"
        )


_workflow_descendants._register_unsupported_descendant_type(
    FlattenedDistributionView,
    "FlattenedDistributionView",
)
_workflow_descendants._register_unsupported_descendant_type(
    NumericRecordDistributionView,
    "NumericRecordDistributionView",
)
