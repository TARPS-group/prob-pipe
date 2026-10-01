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
indexed by position, e.g. ``Normal.from_batched_params(name="x",
loc=jnp.zeros(5), scale=1.0)``, a length-5 array of independent ``Normal``
instances. A
``NumericRecordDistribution`` represents *one* random variable
whose draw can itself have a numeric-valued event structure (a
scalar, a vector, or a multi-field record), and ``DistributionArray``
holds many such variables. The two compose: a ``DistributionArray``
of ``NumericRecordDistribution`` instances is the canonical way to
express a vectorized batch of structured random variables.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._specs import NumericRecordSpec

import jax
import jax.numpy as jnp

from .._dtype import _as_float_array
from .._weights import Weights
from ..custom_types import Array, ArrayLike, PRNGKey
from ..distributions._capabilities import (
    SupportsCovariance,
    SupportsLogProb,
    SupportsMean,
    SupportsSampling,
    SupportsVariance,
)
from ..distributions._distribution import Distribution, NumericDistribution
from ..functions import _descendants
from ._record_distribution import (
    RecordDistribution,
    _field_event_shape,
    _record_with_leaves,
)
from ._specs import NumericArraySpec, OutputSpec
from .constraints import (
    Constraint,
    _PositiveDefinite,
    _Simplex,
    _Sphere,
    _supports_compatible,
    real,
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


def _pairs_by_path(
    source: dict[str, Any], target: dict[str, Any]
) -> list[tuple[tuple[str, Any], tuple[str, Any]]] | None:
    """Pair each target leaf with the source leaf whose path holds it, or ``None``.

    A source leaf holds the target leaf of its own path and the target leaves
    under it, as a posterior's flat chunk holds a nested component's leaves. The
    pairing exists when the source leaves, in order, hold consecutive runs of
    the target leaves that together cover them all.
    """
    targets = list(target.items())
    pairs: list[tuple[tuple[str, Any], tuple[str, Any]]] = []
    i = 0
    for source_item in source.items():
        path = source_item[0]
        start = i
        while i < len(targets) and (targets[i][0] == path or targets[i][0].startswith(path + "/")):
            pairs.append((source_item, targets[i]))
            i += 1
        if i == start:
            return None
    return pairs if i == len(targets) else None


class NumericRecordDistribution(RecordDistribution, NumericDistribution):
    """Distribution over numeric arrays with Record support.

    Extends :class:`RecordDistribution` to laws whose event declaration is
    numeric, each array leaf declaring its shape, dtype, and support. The class is the most
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

    The declaration determines the shape of one draw:

    - **An array** → ``_sample(key, sample_shape)`` returns a raw
      ``jax.Array`` of shape ``sample_shape + event_shape``.
    - **A record** → ``_sample(key, sample_shape)`` returns a
      :class:`~probpipe.NumericRecord` (or a
      :class:`~probpipe.NumericRecordBatch` over one ``draw`` level for a
      non-empty ``sample_shape``) keyed by the record's fields.

    The :attr:`treedef` property locks this invariant by deriving from the
    declaration.

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
        field-by-field in insertion order, or else a source field pairs
        with each target leaf under its path. Any other field-count
        mismatch raises ``ValueError`` rather than silently truncating
        via ``zip``.

        Sources that don't expose per-field supports (non-NRD endpoints
        like ``EmpiricalDistribution`` with object-dtype data) are
        treated as "unknown" and the check returns without complaint.
        """
        try:
            target_per_field = self.supports
            source_per_field = source.supports
        except AttributeError:
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

        # Multi-field target. Equal field counts pair positionally. Otherwise
        # a source leaf that holds a flattened group, as a posterior holds a
        # nested component, pairs with each target leaf under its path, and any
        # other mismatch raises, since ``zip`` would silently truncate.
        if len(source_per_field) == len(target_per_field):
            pairs = list(zip(source_per_field.items(), target_per_field.items()))
        else:
            pairs = _pairs_by_path(source_per_field, target_per_field)
        if pairs is None:
            raise ValueError(
                f"Cannot convert {type(source).__name__} "
                f"({len(source_per_field)} fields: "
                f"{tuple(source_per_field)}) to {type(self).__name__} "
                f"({len(target_per_field)} fields: "
                f"{tuple(target_per_field)}): field-count mismatch. "
                f"Pass check_support=False to override."
            )
        for (s_name, s_sup), (t_name, t_sup) in pairs:
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
        # A law that draws no single array, or whose shape keeps unbound
        # dimensions, has no event_shape, so the per-leaf shapes stand in.
        try:
            parts.append(f"event_shape={self.event_shape}")
        except (AttributeError, ValueError):
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
            return NumericRecordDistribution.flatten_value(
                pytree_samples,
                event_shape=_raw_event_shape(self._base),
            )

        extra_methods["_sample"] = _sample

    if "log_prob" in protocols:
        extra_bases.append(SupportsLogProb)

        def _log_prob(self, x: ArrayLike) -> Array:
            x = jnp.asarray(x)
            value = NumericRecordDistribution.unflatten_value(
                x, template=self._base.event_spec.spec
            )
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
    _descendants._register_unsupported_descendant_type(
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
            OutputSpec(
                to_vector=NumericArraySpec((base.event_spec.spec.vector_size,), base.dtype, real)
            )
        )

    @property
    def base_distribution(self) -> Distribution:
        """The underlying distribution."""
        return self._base

    def unflatten_sample(self, flat_sample: ArrayLike):
        """Convenience: unflatten a flat sample back to the pytree structure."""
        return NumericRecordDistribution.unflatten_value(
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


def _nrdvfactory_sample(self, key: PRNGKey, sample_shape: tuple[int, ...] = ()):
    from ._numeric_record import _reconstruct_from_vector

    base_sample = self._base._sample(key, sample_shape)
    flat = NumericRecordDistribution.flatten_value(
        base_sample,
        event_shape=self._base.event_shape,
    )
    # ``_reconstruct_from_vector`` selects single (NumericRecord, flat
    # is 1-D) vs batched (NumericRecordBatch, batch_shape ==
    # sample_shape) from the rank of ``flat``.
    return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)


def _nrdvfactory_log_prob(self, x) -> Array:
    from ._numeric_record import NumericRecord
    from ._numeric_record_batch import NumericRecordBatch

    flat = x.to_vector() if isinstance(x, (NumericRecord, NumericRecordBatch)) else jnp.asarray(x)
    value = NumericRecordDistribution.unflatten_value(flat, template=self._base.event_spec.spec)
    return self._base._log_prob(value)


def _nrdvfactory_mean(self):
    from ._numeric_record import _reconstruct_from_vector

    value = self._base._mean()
    flat = NumericRecordDistribution.flatten_value(
        value,
        event_shape=self._base.event_shape,
    )
    return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)


def _nrdvfactory_variance(self):
    from ._numeric_record import _reconstruct_from_vector

    value = self._base._variance()
    flat = NumericRecordDistribution.flatten_value(
        value,
        event_shape=self._base.event_shape,
    )
    return _reconstruct_from_vector(self.name, self.event_spec.spec, flat)


def _nrdvfactory_cov(self):
    # Covariance stays flat (event_size × event_size matrix).
    # The Record / field-block structure is implicit in the
    # template's flat ordering.
    return self._base._cov()


_CAPABILITIES = (
    (SupportsSampling, "_sample", _nrdvfactory_sample),
    (SupportsLogProb, "_log_prob", _nrdvfactory_log_prob),
    (SupportsMean, "_mean", _nrdvfactory_mean),
    (SupportsVariance, "_variance", _nrdvfactory_variance),
    (SupportsCovariance, "_cov", _nrdvfactory_cov),
)


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
    base_type = type(base)
    cached = _LIFTED_VIEW_CLASS_CACHE.get(base_type)
    if cached is not None:
        return cached

    bases = [NumericRecordDistributionView]
    methods = {}
    for protocol, method_name, method in _CAPABILITIES:
        if isinstance(base, protocol):
            bases.append(protocol)
            methods[method_name] = method

    if methods:
        cls = type(
            "NumericRecordDistributionView",
            tuple(bases),
            methods,
        )
        _descendants._register_unsupported_descendant_type(
            cls,
            "NumericRecordDistributionView",
        )
    else:
        cls = NumericRecordDistributionView

    _LIFTED_VIEW_CLASS_CACHE[base_type] = cls
    return cls


def _piecewise_support(support: Constraint | None) -> Constraint | None:
    """*support* when every piece of a draw satisfies it too, else None.

    A joint constraint such as ``simplex`` holds for the whole vector only, and a
    bound that varies by element does not carry over to a piece of another shape.
    """
    if support is None or isinstance(support, (_Simplex, _Sphere, _PositiveDefinite)):
        return None
    if any(jnp.ndim(value) > 0 for value in vars(support).values()):
        return None
    return support


class NumericRecordDistributionView(NumericRecordDistribution):
    """View that lifts a flat distribution to a Record-keyed structure.

    Inverse of :class:`FlattenedDistributionView`. ``self._base`` is a
    :class:`FlatNumericRecordDistribution` (single-field, ``event_shape
    == (N,)``); the view declares the user-supplied
    :class:`NumericRecordSpec`.

    Sampling, log-prob, and moments delegate to ``self._base`` and
    reshape via the template's flatten / unflatten machinery.
    Capability protocols match the source via
    :func:`_numeric_record_distribution_view_class_for_base`.

    Constructed via
    :meth:`FlatNumericRecordDistribution.as_record_distribution`.
    """

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
        # dtype, and the source's support where it holds piecewise.
        self._init_declaration(
            _record_with_leaves(template, base.dtype, _piecewise_support(base.support))
        )

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


_descendants._register_unsupported_descendant_type(
    FlattenedDistributionView,
    "FlattenedDistributionView",
)
_descendants._register_unsupported_descendant_type(
    NumericRecordDistributionView,
    "NumericRecordDistributionView",
)
