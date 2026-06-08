"""Approximate empirical distribution with chain structure and auxiliary DataTree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

import jax.numpy as jnp

from ..core.distribution import RecordEmpiricalDistribution, Distribution
from ..core._record_distribution import _RecordDistributionView
from ..core.provenance import Provenance
from ..core.record import Record, RecordTemplate, _spec_size
from ..custom_types import Array, ArrayLike, PRNGKey
from .._weights import Weights

__all__ = ["ApproximateDistribution", "make_posterior"]


# ---------------------------------------------------------------------------
# Column ordering
# ---------------------------------------------------------------------------


def _column_permutation(
    record_template: RecordTemplate, field_order: list[str],
) -> list[int]:
    """Column-index permutation mapping a *field_order*-laid-out flat chain
    into ``record_template.fields`` order.

    *field_order* names the field each contiguous column-block of the flat
    chain occupies. The returned ``perm`` satisfies: ``flat[..., perm]``
    lays the columns out in template-field order, so the positional split
    in :class:`ApproximateDistribution` maps each column to the right
    field by name. See issue #233.

    Raises
    ------
    ValueError
        If *field_order* is not a permutation of the template's fields, or
        a field has an opaque (``spec=None``) leaf with no flat size.
    """
    if sorted(field_order) != sorted(record_template.fields):
        raise ValueError(
            f"field_order {list(field_order)} is not a permutation of "
            f"template fields {list(record_template.fields)}."
        )
    sizes: dict[str, int] = {}
    for field_name in record_template.fields:
        spec = record_template[field_name]
        if spec is None:
            raise ValueError(
                f"ApproximateDistribution requires a numeric template; "
                f"field {field_name!r} has spec=None (opaque). Opaque "
                f"leaves don't have a flat size."
            )
        sizes[field_name] = _spec_size(spec)
    bounds: dict[str, tuple[int, int]] = {}
    offset = 0
    for field_name in field_order:
        bounds[field_name] = (offset, offset + sizes[field_name])
        offset += sizes[field_name]
    perm: list[int] = []
    for field_name in record_template.fields:
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
    samples, and the ArviZ ``InferenceData`` object live in
    ``dist.auxiliary`` (a DataTree on the Distribution base class),
    not as attributes of this class.

    Parameters
    ----------
    chains : list of Array
        Per-chain sample arrays, each of shape ``(num_draws, *event_shape)``.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-sample importance weights (across all chains).
    name : str or None
        Distribution name for provenance.
    record_template : RecordTemplate or None
        If given, names the posterior's fields: the concatenated chain is
        split into per-field arrays (multi-field) so :meth:`draws`,
        :meth:`_mean` / :meth:`_variance`, etc. return Records keyed by
        the template fields. ``None`` leaves the posterior a single
        unnamed numeric block.
    field_order : list of str or None
        Names the field each contiguous column-block of *chains* belongs
        to, in the order they appear. Default (``None``) assumes the
        columns are already in ``record_template.fields`` order. Pass this
        when the chain's column order may differ (e.g. a backend that
        sorts variable names) so columns are aligned to fields by name
        rather than position. Requires *record_template*, and must be a
        permutation of its fields.

    Notes
    -----
    When ``record_template`` is multi-field, ``__init__`` slices the
    concatenated chain into per-top-level-field arrays so
    :attr:`fields`, :attr:`event_shapes`, :attr:`dtypes`,
    :meth:`_mean` / :meth:`_variance`, and the public ops
    (``mean(post)`` / ``variance(post)``) all return Records whose
    keys match :attr:`fields`. Nested ``RecordTemplate`` fields are
    stored as a flat ``(n, nested_flat_size)`` array under the
    top-level field name; the nested structure is recoverable via
    ``record_template[field]`` and via :meth:`draws`, which walks
    the full template (including nesting) using the original
    per-chain samples.
    """

    def __init__(
        self,
        chains: list[Array],
        *,
        weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        record_template: RecordTemplate | None = None,
        field_order: list[str] | None = None,
    ):
        if not chains:
            raise ValueError("Must provide at least one chain")

        self._chains = [jnp.asarray(c) for c in chains]
        self._concatenated: Array | None = None

        # When the caller's chain columns are laid out in a different
        # field order than the template — e.g. a backend whose trace
        # sorts variable names — permute them into ``record_template``
        # order. The positional split below (and ``draws()`` unflatten)
        # then map each column to the right field by name rather than by
        # position, so callers don't have to pre-sort. See issue #233.
        if field_order is not None:
            if record_template is None:
                raise ValueError(
                    "field_order requires a record_template; it names "
                    "template fields and is meaningless without one."
                )
            # Validates field_order is a permutation of the template
            # fields (raises otherwise) for any field count, so a
            # single-field typo or wrong name is also caught.
            perm = _column_permutation(record_template, field_order)
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
            if len(record_template.fields) > 1:
                self._chains = [c[..., perm] for c in self._chains]
                self._concatenated = None

        flat = self._concat_chains()
        # Track whether the user explicitly supplied a template; we use
        # this in ``draws()`` to decide whether to wrap the output.
        self._user_template = record_template is not None
        # Multi-field template → split the flat chain by top-level
        # field. Nested ``RecordTemplate`` fields are stored as a
        # 2-D ``(n, nested_flat_size)`` slice under the top-level
        # field name; the nested structure is recovered via
        # ``record_template[field]`` and ``draws()``. Slice sizes use
        # ``_spec_size``, which already handles both flat and nested
        # specs.
        if record_template is not None and len(record_template.fields) > 1:
            # Compute per-field sizes upfront so we can sanity-check the
            # chain's last dim against the template's total flat size
            # (catching template/data mismatch before silent slicing past
            # the end produces zero-sized chunks). ``_spec_size`` raises
            # on opaque (``spec=None``) leaves; pre-validate here so the
            # error names the offending field rather than the generic
            # ``_spec_size`` message.
            sizes: list[int] = []
            for field_name in record_template.fields:
                spec = record_template[field_name]
                if spec is None:
                    raise ValueError(
                        f"ApproximateDistribution requires a numeric "
                        f"template; field {field_name!r} has spec=None "
                        f"(opaque). Opaque leaves don't have a flat size."
                    )
                sizes.append(_spec_size(spec))
            total = sum(sizes)
            if flat.shape[-1] != total:
                raise ValueError(
                    f"chain last dim ({flat.shape[-1]}) doesn't match "
                    f"template total flat size ({total}); template "
                    f"fields={record_template.fields}, sizes={sizes}."
                )
            offset = 0
            fields: dict[str, Array] = {}
            for field_name, size in zip(record_template.fields, sizes):
                spec = record_template[field_name]
                chunk = flat[..., offset : offset + size]
                if isinstance(spec, RecordTemplate):
                    # Nested: keep flat-per-top-level-field. Shape is
                    # ``(*sample_shape, nested_flat_size)``.
                    fields[field_name] = chunk
                else:
                    shape = spec if spec is not None else ()
                    fields[field_name] = chunk.reshape(*flat.shape[:-1], *shape)
                offset += size
            super().__init__(Record(fields), weights=weights, name=name or "posterior")
            self._record_template = record_template
        else:
            # Single-field path: ``name`` (default ``"posterior"``)
            # becomes the auto-wrapped field name. If the user passed a
            # single-field template, rename to honor it.
            field_name = name or "posterior"
            if record_template is not None and len(record_template.fields) == 1:
                field_name = record_template.fields[0]
            super().__init__(flat, weights=weights, name=field_name)
            if record_template is not None:
                self._record_template = record_template

    def _concat_chains(self) -> Array:
        """Lazily concatenated view of all chains."""
        if self._concatenated is None:
            self._concatenated = jnp.concatenate(self._chains, axis=0)
        return self._concatenated

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
        src = self.source
        if src is not None:
            return src.metadata.get("algorithm", src.operation)
        return "unknown"

    @property
    def inference_data(self) -> DataTree | None:
        """The auxiliary DataTree, for ArviZ compatibility.

        Alias for ``self.auxiliary``.  Use ArviZ functions for diagnostics::

            import arviz as az
            az.summary(posterior.inference_data)
        """
        return self.auxiliary

    @property
    def warmup_samples(self) -> list[Array] | None:
        """Per-chain warmup samples extracted from auxiliary data."""
        aux = self.auxiliary
        if aux is None:
            return None
        # arviz 1.x DataTree uses .children; arviz 0.x InferenceData uses .groups()
        has_warmup = (
            ("warmup" in aux.children) if hasattr(aux, "children")
            else hasattr(aux, "warmup")
        )
        if not has_warmup:
            return None
        warmup = aux["warmup"]["params"]
        n_chains = warmup.sizes.get("chain", 1)
        return [jnp.asarray(warmup.sel(chain=i).values) for i in range(n_chains)]

    def draws(
        self,
        chain: int | None = None,
        *,
        include_warmup: bool = False,
    ) -> Array | Record:
        """Access draws, optionally named via record_template.

        Parameters
        ----------
        chain : int or None
            Chain index.  If ``None``, concatenates all chains.
        include_warmup : bool
            If ``True`` and warmup samples are in the auxiliary DataTree,
            prepend them.

        Returns
        -------
        Array or Record
            If ``record_template`` is set, returns a :class:`~probpipe.Record`
            with named fields.  Otherwise returns a raw array.
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
                    parts = [jnp.concatenate([w, c], axis=0)
                             for w, c in zip(warmup, parts)]
            samples = jnp.concatenate(parts, axis=0)

        # Honor any user-supplied template (single-field or multi-field).
        # Without one, return the raw concatenated array — matches the
        # historical behaviour of single-field auto-wrap empiricals
        # under the previous numeric-array hierarchy.
        if getattr(self, "_user_template", False):
            from ..core._record_array import NumericRecordArray
            return NumericRecordArray.unflatten(samples, template=self.record_template)
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
    auxiliary: DataTree | None = None,
    record_template: RecordTemplate | None = None,
    field_order: list[str] | None = None,
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
    auxiliary : DataTree or None
        Pre-built auxiliary DataTree (diagnostics, sample stats, warmup).
        Inference methods are responsible for building this.
    record_template : RecordTemplate or None
        If provided, ``draws()`` returns named ``Record``.
    field_order : list of str or None
        Names the field each contiguous column-block of ``chains`` belongs
        to. Default (``None``) assumes the columns are laid out in
        ``record_template.fields`` order. Pass this when the chain's column
        order may differ from the template's (e.g. a backend that sorts
        variable names) so columns are aligned to fields by name rather
        than position (see issue #233).
    **meta
        Additional metadata stored in provenance.

    Returns
    -------
    ApproximateDistribution
        Posterior with chain structure, auxiliary DataTree, and provenance.
    """
    result = ApproximateDistribution(
        chains, name="posterior", record_template=record_template,
        field_order=field_order,
    )

    if auxiliary is not None:
        result._auxiliary = auxiliary

    result.with_source(
        Provenance(algorithm, parents=parents, metadata={"algorithm": algorithm, **meta})
    )
    return result
