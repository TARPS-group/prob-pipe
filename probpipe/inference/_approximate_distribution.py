"""Approximate empirical distribution with chain structure and annotations DataTree."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from xarray import DataTree

    from ..core._spec_base import TermSpec

import jax.numpy as jnp

from .._weights import Weights
from ..core._empirical import RecordEmpiricalDistribution
from ..core._immutable import transient_memo
from ..core._opaque import OpaqueSpec
from ..core._specs import (
    NumericArraySpec,
    NumericRecordSpec,
    OutputSpec,
    RecordSpec,
    _components_record,
)
from ..core.provenance import Provenance
from ..core.record import Record
from ..custom_types import Array, ArrayLike
from ..distributions._distribution import Distribution, _complete_event_spec

__all__ = ["ApproximateDistribution", "make_posterior"]


def _spec_size(spec: NumericArraySpec | RecordSpec) -> int:
    """Number of scalar elements one field contributes to a flat vector.

    Given the spec of a single field of a :class:`RecordSpec`, return how
    many scalars that field occupies in the dense 1-D vector layout (see
    :meth:`~probpipe.NumericRecord.to_vector`): ``prod(shape)`` for an
    :class:`NumericArraySpec`, or :attr:`~NumericRecordSpec.vector_size` for a
    nested :class:`NumericRecordSpec`. Summing this over a template's fields
    gives the template's own ``vector_size``; it is used here to size each
    field's contiguous column block when splitting a flat chain.

    Parameters
    ----------
    spec
        One field's spec, as returned by :meth:`RecordSpec.__getitem__`.

    Raises
    ------
    TypeError
        If the field has no flat size — a non-numeric leaf
        (:class:`~probpipe.OpaqueSpec` / :class:`~probpipe.DistributionSpec` /
        :class:`~probpipe.FunctionSpec`) or a mixed (non-all-numeric) nested
        :class:`RecordSpec`.
    """
    if isinstance(spec, NumericRecordSpec):
        return spec.vector_size
    if isinstance(spec, RecordSpec):
        raise TypeError(
            f"nested {type(spec).__name__} contains non-numeric leaves; "
            f"a flat size requires a NumericRecordSpec."
        )
    if isinstance(spec, NumericArraySpec):
        return prod(spec.shape) if spec.shape else 1
    raise TypeError(
        f"template field ({type(spec).__name__}) has no flat size; only numeric "
        f"(NumericArraySpec) fields and nested NumericRecordSpec fields do."
    )


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------


def _column_permutation(
    record: RecordSpec,
    field_order: list[str],
) -> list[int]:
    """Column-index permutation mapping a *field_order*-laid-out flat chain
    into ``record.fields`` order.

    *field_order* names the field each contiguous column-block of the flat
    chain occupies. The returned ``perm`` satisfies: ``flat[..., perm]``
    lays the columns out in template-field order, so the positional split
    in :class:`ApproximateDistribution` maps each column to the right
    field by name. See issue #233.

    Raises
    ------
    ValueError
        If *field_order* is not a permutation of the record's fields, or
        a field has an opaque (``spec=None``) leaf with no flat size.
    """
    if sorted(field_order) != sorted(record.fields):
        raise ValueError(
            f"field_order {list(field_order)} is not a permutation of "
            f"template fields {list(record.fields)}."
        )
    sizes: dict[str, int] = {}
    for field_name in record.fields:
        spec = record.children[field_name]
        if isinstance(spec, OpaqueSpec):
            raise ValueError(
                f"ApproximateDistribution requires a numeric template; "
                f"field {field_name!r} has an opaque spec. Opaque "
                f"leaves don't have a flat size."
            )
        sizes[field_name] = _spec_size(spec)
    bounds: dict[str, tuple[int, int]] = {}
    offset = 0
    for field_name in field_order:
        bounds[field_name] = (offset, offset + sizes[field_name])
        offset += sizes[field_name]
    perm: list[int] = []
    for field_name in record.fields:
        lo, hi = bounds[field_name]
        perm.extend(range(lo, hi))
    return perm


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class ApproximateDistribution(RecordEmpiricalDistribution):
    """Empirical distribution with chain structure.

    Stores per-chain sample arrays for chain-structured access via
    :meth:`draws`.  Algorithm metadata, sample statistics, warmup
    samples, and the ArviZ ``DataTree`` live in ``dist.annotations``
    (on the Distribution base class), not as attributes of this class.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each of shape ``(num_draws, *event_shape)``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-sample importance weights (across all chains).
    name : str or None
        Distribution name for provenance.
        Keyword-only, as an interim detail (see :class:`~probpipe.Distribution`).
    event_spec : OutputSpec, TermSpec, or None
        The target's declaration, usually the prior's ``event_spec``. Its
        components name the posterior's fields: the concatenated chain is
        split into one array per component, so :meth:`draws`,
        :meth:`_mean` / :meth:`_variance`, etc. return Records keyed by
        them. A bare ``RecordSpec`` exposes its fields, as
        :class:`~probpipe.Distribution` completes one. ``None`` leaves the
        posterior a single unnamed numeric block.
    field_order : list of str or None
        Names the field each contiguous column-block of *chains* belongs
        to, in the order they appear. Default (``None``) assumes the
        columns are already in the order of the target's components. Pass
        this when the chain's column order may differ, as for a backend that
        sorts variable names, so columns are aligned to fields by name
        rather than position. Requires *event_spec*, and must be a
        permutation of its components.

    Notes
    -----
    When the target has several components, ``__init__`` slices the
    concatenated chain into one array per component so
    :attr:`fields`, :attr:`event_shapes`, :attr:`dtypes`,
    :meth:`_mean` / :meth:`_variance`, and the public ops
    (``mean(post)`` / ``variance(post)``) all return Records whose
    keys match :attr:`fields`. A nested record component is stored as a
    flat ``(n, nested_vector_size)`` array under its name; :meth:`draws`
    recovers the nesting from the target's declaration.

    A whole-term target gives a posterior that draws a one-field record
    under the target's component, an interim implementation detail.
    """

    #: The memo is not state: a copy recomputes rather than inheriting one. It
    #: matters for more than size here, since a memoised value can carry the
    #: provenance of the term that computed it.
    _transient_state = ("_memo",)

    def __init__(
        self,
        chains: list[Array],
        *,
        weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        event_spec: OutputSpec | TermSpec | None = None,
        field_order: list[str] | None = None,
    ):
        if not chains:
            raise ValueError("Must provide at least one chain")
        # The record the target's components form, which names the fields.
        record = (
            None
            if event_spec is None
            else _components_record(_complete_event_spec(event_spec, name or "posterior"))
        )

        self._chains = [jnp.asarray(c) for c in chains]
        # A memo, filled on first read. Reading fills it in place, which leaves
        # the term's own attributes as construction set them — what the
        # immutability guard sees, and what a copy drops rather than inherits.
        self._memo: dict[str, Array] = {}

        # When the caller's chain columns are laid out in a different
        # field order than the record — e.g. a backend whose trace
        # sorts variable names — permute them into the record's order.
        # The positional split below (and ``draws()`` unflatten) then map
        # each column to the right field by name rather than by position,
        # so callers don't have to pre-sort.
        if field_order is not None:
            if record is None:
                raise ValueError(
                    "field_order requires an event_spec; it names the "
                    "target's components and is meaningless without one."
                )
            # Validates field_order is a permutation of the template
            # fields (raises otherwise) for any field count, so a
            # single-field typo or wrong name is also caught.
            perm = _column_permutation(record, field_order)
            # Validate width for any field count, *before* the gather:
            # ``c[..., perm]`` would otherwise silently drop extra columns
            # (too-wide chain) or clamp out-of-bounds indices (too-narrow),
            # then pass the post-gather total-size check.
            for c in self._chains:
                if c.shape[-1] != len(perm):
                    raise ValueError(
                        f"chain last dim ({c.shape[-1]}) doesn't match "
                        f"the template total flat size ({len(perm)})."
                    )
            if len(record.fields) > 1:
                self._chains = [c[..., perm] for c in self._chains]
                transient_memo(self).pop("concatenated", None)

        flat = self._concat_chains()
        # ``draws()`` rebuilds Records from the target's record when there is
        # one, and returns the raw concatenated array otherwise.
        self._target_record = record
        # Several components → split the flat chain by top-level field. A
        # nested record component is stored as a 2-D
        # ``(n, nested_vector_size)`` slice under its name; ``draws()``
        # recovers the nesting. Slice sizes use ``_spec_size``, which handles
        # both flat and nested specs.
        if record is not None and len(record.fields) > 1:
            # Compute per-field sizes upfront so we can sanity-check the
            # chain's last dim against the template's total flat size
            # (catching template/data mismatch before silent slicing past
            # the end produces zero-sized chunks). ``_spec_size`` raises
            # on opaque (``spec=None``) leaves; pre-validate here so the
            # error names the offending field rather than the generic
            # ``_spec_size`` message.
            sizes: list[int] = []
            for field_name in record.fields:
                spec = record.children[field_name]
                if isinstance(spec, OpaqueSpec):
                    raise ValueError(
                        f"ApproximateDistribution requires a numeric "
                        f"template; field {field_name!r} has an opaque "
                        f"spec. Opaque leaves don't have a flat size."
                    )
                sizes.append(_spec_size(spec))
            total = sum(sizes)
            if flat.shape[-1] != total:
                raise ValueError(
                    f"chain last dim ({flat.shape[-1]}) doesn't match "
                    f"template total flat size ({total}); template "
                    f"fields={record.fields}, sizes={sizes}."
                )
            offset = 0
            fields: dict[str, Array] = {}
            for field_name, size in zip(record.fields, sizes):
                spec = record.children[field_name]
                chunk = flat[..., offset : offset + size]
                if isinstance(spec, RecordSpec):
                    # Nested: keep flat-per-top-level-field. Shape is
                    # ``(*sample_shape, nested_vector_size)``.
                    fields[field_name] = chunk
                else:
                    # NumericArraySpec leaf (opaque rejected above, nested handled).
                    shape = cast(NumericArraySpec, spec).shape
                    fields[field_name] = chunk.reshape(*flat.shape[:-1], *shape)
                offset += size
            super().__init__(
                name or "posterior",
                Record(name or "posterior", fields),
                weights=weights,
            )
        else:
            # One component or none: the component (default ``name``, then
            # ``"posterior"``) becomes the auto-wrapped field name.
            field_name = name or "posterior"
            if record is not None and len(record.fields) == 1:
                field_name = record.fields[0]
            super().__init__(field_name, flat, weights=weights)

    def _concat_chains(self) -> Array:
        """Lazily concatenated view of all chains."""
        concatenated = transient_memo(self).get("concatenated")
        if concatenated is None:
            concatenated = jnp.concatenate(self._chains, axis=0)
            transient_memo(self)["concatenated"] = concatenated
        return concatenated

    # -- Chain access ---------------------------------------------------------

    @property
    def chains(self) -> list[Array]:
        """Per-chain sample arrays."""
        return self._chains

    @property
    def num_chains(self) -> int:
        """Number of chains."""
        return len(self._chains)

    @property
    def num_draws(self) -> int:
        """Number of draws *per chain* (assumes equal-length chains).

        Distinct from ``num_atoms`` (inherited from
        :class:`~probpipe.core._empirical.RecordEmpiricalDistribution`),
        which counts the total atoms across all chains —
        ``num_atoms == num_chains * num_draws``.
        """
        return self._chains[0].shape[0]

    @property
    def algorithm(self) -> str:
        """Name of the inference algorithm (read from provenance)."""
        src = self.provenance
        if src is not None:
            return src.metadata.get("algorithm", src.operation)
        return "unknown"

    @property
    def arviz_data(self) -> DataTree | None:
        """The ArviZ-compatible xarray DataTree stored under ``_annotations["arviz"]``.

        Use ArviZ, arviz-stats, and arviz-plots functions for diagnostics and
        plots::

            import arviz_stats
            arviz_stats.summary(posterior.arviz_data)

        Plotting utilities may also consume this tree where supported.

        Returns ``None`` if no annotations have been attached. Falls
        back to ``_annotations`` directly if set before the ``/arviz/``
        subtree convention was adopted.
        """
        aux = self.annotations
        if aux is None:
            return None
        if hasattr(aux, "children") and "arviz" in aux.children:
            return aux["arviz"]
        return aux

    @property
    def inference_data(self) -> DataTree | None:
        """Backward-compatible alias for :attr:`arviz_data`.

        Modern ArviZ uses ``xarray.DataTree`` via ``arviz-base`` rather than
        the legacy ``InferenceData`` object. New ProbPipe code should prefer
        ``posterior.arviz_data``.
        """
        return self.arviz_data

    @property
    def warmup_samples(self) -> list[Array] | None:
        """Per-chain warmup samples extracted from the annotations."""
        arviz_data = self.arviz_data
        if arviz_data is None:
            return None

        # ``arviz_data`` is expected to be an ArviZ-compatible xarray DataTree.
        # Warmup samples, when present, live under the ``warmup`` group.
        children = arviz_data.children if hasattr(arviz_data, "children") else {}
        if "warmup" not in children:
            return None

        warmup = arviz_data["warmup"]["params"]
        n_chains = warmup.sizes.get("chain", 1)
        return [jnp.asarray(warmup.sel(chain=i).values) for i in range(n_chains)]

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array | Record:
        """Access draws, named by the target's components when there is a target.

        Parameters
        ----------
        chain : int or None
            Chain index.  If ``None``, concatenates all chains.
        include_warmup : bool
            If ``True`` and warmup samples are in the annotations DataTree,
            prepend them.

        Returns
        -------
        Array or Record
            With a target declaration, a :class:`~probpipe.Record` whose fields
            are its components. Otherwise a raw array.
        """
        if chain is not None:
            samples = self._chains[chain]
            if include_warmup:
                warmup = self.warmup_samples
                if warmup is not None:
                    samples = jnp.concatenate([warmup[chain], samples], axis=0)
        else:
            parts = list(self._chains)
            if include_warmup:
                warmup = self.warmup_samples
                if warmup is not None:
                    parts = [jnp.concatenate([w, c], axis=0) for w, c in zip(warmup, parts)]
            samples = jnp.concatenate(parts, axis=0)

        record = getattr(self, "_target_record", None)
        if record is not None:
            # Reconstruct: batch_shape is inferred from the leading axes of
            # the concatenated draws (a matrix ``(n, vector_size)``).
            from ..core._numeric_record import _reconstruct_from_vector

            return _reconstruct_from_vector(self.name, record, samples)
        return samples

    def __repr__(self) -> str:
        # Use ``event_shapes`` (plural) for multi-field posteriors so
        # the repr stays valid; ``event_shape`` (singular) raises on
        # multi-field by design.
        if len(self._record_data.fields) == 1:
            shape_part = f"event_shape={self.event_shape}"
        else:
            shape_part = f"event_shapes={self.event_shapes}"
        return (
            f"ApproximateDistribution("
            f"algorithm={self.algorithm!r}, "
            f"num_chains={self.num_chains}, "
            f"num_draws={self.num_draws}, "
            f"{shape_part})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_posterior(
    chains: list[Array],
    parents: tuple[Distribution, ...],
    algorithm: str,
    *,
    annotations: DataTree | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
    field_order: list[str] | None = None,
    weights: ArrayLike | Weights | None = None,
    **meta: Any,
) -> ApproximateDistribution:
    """Build an ApproximateDistribution with provenance.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each shaped ``(num_draws, *event_shape)``.
    parents : tuple of Distribution
        Parent distributions for provenance tracking.
    algorithm : str
        Inference algorithm name (e.g. ``"tfp_nuts"``, ``"blackjax_rwmh"``).
    annotations : DataTree or None
        Pre-built annotations DataTree (diagnostics, sample stats, warmup).
        Inference methods are responsible for building this.
    event_spec : OutputSpec, TermSpec, or None
        The target's declaration, usually the prior's ``event_spec``. If
        provided, ``draws()`` returns a named ``Record``.
    field_order : list of str or None
        Names the field each contiguous column-block of ``chains`` belongs
        to. Default (``None``) assumes the columns are laid out in the
        order of the target's components. Pass this when the chain's
        column order may differ, as for a backend that sorts variable names,
        so columns are aligned to fields by name rather than position.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-sample importance weights (across all chains),
        forwarded to :class:`ApproximateDistribution`. Lets weighted
        backends — e.g. SMC-ABC, which returns importance-weighted
        particles — preserve their weights instead of resampling to an
        equal-weight chain.
    **meta
        Additional metadata stored in provenance.

    Returns
    -------
    ApproximateDistribution
        Posterior with chain structure, annotations DataTree, and provenance.
    """
    import xarray as xr

    result = ApproximateDistribution(
        chains,
        name="posterior",
        event_spec=event_spec,
        field_order=field_order,
        weights=weights,
    )

    if annotations is not None:
        # Nest ArviZ-compatible DataTree groups under /arviz/ so _annotations
        # can hold other subtrees (e.g. /diagnostics/) alongside it.
        dicto: dict = {}
        for group_path, node in annotations.items():
            if group_path == "/":
                continue
            clean = group_path.lstrip("/")
            ds = node.to_dataset() if isinstance(node, xr.DataTree) else node
            dicto[f"arviz/{clean}"] = ds
        result._init_annotations(xr.DataTree.from_dict(dicto))

    result.with_provenance(
        Provenance.create(
            algorithm, parents=list(parents), metadata={"algorithm": algorithm, **meta}
        )
    )
    return result
