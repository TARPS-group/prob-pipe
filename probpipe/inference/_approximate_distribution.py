"""Inference results: the empirical law of a run's draws, with its chains and annotations."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

    from ..core._spec_base import TermSpec

import jax.numpy as jnp

from .._weights import Weights
from ..core._immutable import transient_memo
from ..core._numeric_array_batch import NumericArrayBatch
from ..core._numeric_record import _reconstruct_from_vector
from ..core._numeric_record_batch import NumericRecordBatch
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
from ..distributions._capabilities import _capability_subclass
from ..distributions._distribution import Distribution, _complete_event_spec
from ..distributions._empirical import _NUMERIC_MOMENTS, EmpiricalDistribution

__all__ = ["ApproximateDistribution", "make_posterior"]

#: The levels of an inference result's atoms, outermost first.
_CHAIN_LEVELS = ("chain", "draw")


def _spec_size(spec: NumericArraySpec | RecordSpec) -> int:
    """Number of scalar elements one field contributes to a flat vector.

    Given the spec of a single field of a :class:`RecordSpec`, return how
    many scalars that field occupies in the dense 1-D vector layout (see
    :meth:`~probpipe.NumericRecord.to_vector`): ``prod(shape)`` for an
    :class:`NumericArraySpec`, or :attr:`~NumericRecordSpec.vector_size` for a
    nested :class:`NumericRecordSpec`. It sizes each field's contiguous column
    block when the columns of a flat chain are permuted.

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
    lays the columns out in template-field order, so the columns of each field
    are read by name rather than by position.

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
# The atoms
# ---------------------------------------------------------------------------


def _array_atoms(name: str, stacked: Array, term: NumericArraySpec) -> NumericArrayBatch:
    """The draws *stacked* ``(chains, draws, *flat)`` as atoms of the array term *term*.

    Each draw takes the shape the term declares, so a scalar term's draws are
    scalars, and the dtype the term declares.

    Raises
    ------
    ValueError
        If a draw's flat width is not the term's size.
    """
    chains, draws = stacked.shape[:2]
    if all(isinstance(size, int) for size in term.shape):
        width = prod(stacked.shape[2:])
        if width != prod(term.shape):
            raise ValueError(
                f"chain last dim ({width}) doesn't match the target's size {prod(term.shape)} "
                f"for its shape {term.shape}."
            )
        values = jnp.reshape(stacked, (chains, draws, *term.shape))
    else:
        values = stacked
        term = NumericArraySpec(tuple(stacked.shape[2:]), term.dtype, term.support)
    if term.dtype is not None:
        values = values.astype(term.dtype)
    return NumericArrayBatch(name, values, _CHAIN_LEVELS, element_spec=term, axes_per_level=(1, 1))


def _record_atoms(name: str, stacked: Array, record: RecordSpec) -> NumericRecordBatch:
    """The draws *stacked* ``(chains, draws, d)`` as atoms of *record*, in its flat layout.

    The columns follow the record's canonical leaf order, nested groups
    included, and each leaf takes the shape and dtype the record declares.

    Raises
    ------
    TypeError
        If *record* has a leaf that is neither numeric nor opaque.
    ValueError
        If *record* has an opaque leaf, or a draw's flat width is not the
        record's flat size.
    """
    for path, spec in record.items():
        if isinstance(spec, OpaqueSpec):
            raise ValueError(
                f"ApproximateDistribution requires a numeric template; field {path!r} has an "
                f"opaque spec. Opaque leaves don't have a flat size."
            )
    if not isinstance(record, NumericRecordSpec):
        raise TypeError(
            f"ApproximateDistribution requires a numeric target; {record!r} has a leaf "
            f"without a flat size."
        )
    chains, draws = stacked.shape[:2]
    flat = jnp.reshape(stacked, (chains, draws, -1))
    if flat.shape[-1] != record.vector_size:
        raise ValueError(
            f"chain last dim ({flat.shape[-1]}) doesn't match the target's flat size "
            f"({record.vector_size}); target fields={record.fields}."
        )
    columns, offset = {}, 0
    for path, spec in record.items():
        width = prod(spec.shape)
        column = jnp.reshape(flat[..., offset : offset + width], (chains, draws, *spec.shape))
        columns[path] = column if spec.dtype is None else column.astype(spec.dtype)
        offset += width
    return NumericRecordBatch(
        name, columns, _CHAIN_LEVELS, element_spec=record, axes_per_level=(1, 1)
    )


def _chain_atoms(name: str, stacked: Array, declaration: OutputSpec | None) -> Any:
    """The draws *stacked* ``(chains, draws, *flat)`` as atoms of the target's event term.

    Without a target, each draw is one array.
    """
    if declaration is None:
        element = NumericArraySpec(tuple(stacked.shape[2:]), stacked.dtype)
        return NumericArrayBatch(
            name, stacked, _CHAIN_LEVELS, element_spec=element, axes_per_level=(1, 1)
        )
    term = declaration.spec
    if isinstance(term, RecordSpec):
        return _record_atoms(name, stacked, term)
    if not isinstance(term, NumericArraySpec):
        raise TypeError(
            f"ApproximateDistribution requires a numeric target; {name!r} declares {term!r}"
        )
    return _array_atoms(name, stacked, term)


# ---------------------------------------------------------------------------
# ApproximateDistribution
# ---------------------------------------------------------------------------


class ApproximateDistribution(EmpiricalDistribution):
    """The empirical law of an inference run's draws, with the chains that produced them.

    An MCMC or ABC result is an :class:`~probpipe.EmpiricalDistribution` whose
    atoms are its draws on the levels ``chain`` and ``draw``, so its sampling,
    expectation, moments, quantiles, and marginals are the empirical law's. Its
    event declaration is its target's: a whole-term target stays whole, and a
    record target keeps its fields' supports, scalar shapes, and nested groups.
    The method's chains are kept as it produced them, in the target's flat
    layout, for :attr:`chains` and :meth:`draws`.

    What the result shares with every inference result is its record:
    :func:`make_posterior` gives it ``provenance`` naming the method and the
    target, and stores the method's diagnostics, sample statistics, and warmup
    draws in :attr:`~probpipe.Distribution.annotations`, an ArviZ-compatible
    ``DataTree`` under ``arviz/``. Whether the result is exact or approximate,
    and relative to what, is read from that record.

    Parameters
    ----------
    chains : list of Array
        Per-chain draws, each of shape ``(num_draws, *flat)`` in the target's
        flat layout; the chains have equal lengths.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Optional per-draw importance weights, across all chains in chain order.
    name : str or None
        The result's label. Keyword-only; defaults to ``"posterior"``.
    event_spec : OutputSpec, TermSpec, or None
        The target's declaration, usually the prior's ``event_spec``, which the
        result declares as its event. A bare ``RecordSpec`` exposes its fields,
        as :class:`~probpipe.Distribution` completes one, and any other term is
        a whole term under *name*. ``None`` makes each draw one array, a whole
        term under *name*.
    field_order : list of str or None
        Names the field each contiguous column-block of *chains* belongs
        to, in the order they appear. Default (``None``) assumes the
        columns are already in the order of the target's components. Pass
        this when the chain's column order may differ, as for a backend that
        sorts variable names, so columns are aligned to fields by name
        rather than position. Requires *event_spec*, and must be a
        permutation of its components.

    Raises
    ------
    ValueError
        If *chains* is empty or the chains differ in length, *field_order* is
        given without *event_spec* or is not a permutation of its components, or
        a draw's flat width is not the target's flat size.
    TypeError
        If the target has a leaf without a flat size.
    """

    #: The memo is not state: a copy recomputes rather than inheriting one. It
    #: matters for more than size here, since a memoised value can carry the
    #: provenance of the term that computed it.
    _transient_state = (*EmpiricalDistribution._transient_state, "_memo")

    def __new__(
        cls,
        chains: list[Array],
        *,
        weights: ArrayLike | Weights | None = None,
        name: str | None = None,
        event_spec: OutputSpec | TermSpec | None = None,
        field_order: list[str] | None = None,
    ) -> ApproximateDistribution:
        base = vars(cls).get("_capability_base", cls)
        return object.__new__(_capability_subclass(base, _NUMERIC_MOMENTS))

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
        label = name or "posterior"
        declaration = None if event_spec is None else _complete_event_spec(event_spec, label)
        # The record the target's components form, which names the fields of draws().
        record = None if declaration is None else _components_record(declaration)
        flat_chains = [jnp.asarray(chain) for chain in chains]

        # When the chain columns are laid out in a different field order than the
        # record, as for a backend whose trace sorts variable names, permute them
        # into the record's order, so each column is read by name.
        if field_order is not None:
            if record is None:
                raise ValueError(
                    "field_order requires an event_spec; it names the "
                    "target's components and is meaningless without one."
                )
            perm = _column_permutation(record, field_order)
            # The width is checked before the gather, which would otherwise drop
            # extra columns or clamp out-of-bounds indices.
            for chain in flat_chains:
                if chain.shape[-1] != len(perm):
                    raise ValueError(
                        f"chain last dim ({chain.shape[-1]}) doesn't match "
                        f"the template total flat size ({len(perm)})."
                    )
            if len(record.fields) > 1:
                flat_chains = [chain[..., perm] for chain in flat_chains]

        lengths = sorted({int(chain.shape[0]) for chain in flat_chains})
        if len(lengths) > 1:
            raise ValueError(f"the chains of an inference result have equal lengths, got {lengths}")
        atoms = _chain_atoms(label, jnp.stack(flat_chains), declaration)
        super().__init__(label, atoms, weights, event_spec=declaration)
        object.__setattr__(self, "_chains", flat_chains)
        object.__setattr__(self, "_target_record", record)
        # A memo, filled on first read. Reading fills it in place, which leaves
        # the term's own attributes as construction set them — what the
        # immutability guard sees, and what a copy drops rather than inherits.
        object.__setattr__(self, "_memo", {})

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
        """Per-chain draws, in the target's flat layout."""
        return self._chains

    @property
    def num_chains(self) -> int:
        """Number of chains."""
        return len(self._chains)

    @property
    def num_draws(self) -> int:
        """Number of draws *per chain*.

        Distinct from ``num_atoms``, which counts the draws across all chains,
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
            With a target declaration, a batch of records whose fields are its
            components. Otherwise a raw array in the flat layout.
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

        record = self._target_record
        if record is not None:
            # Reconstruct: batch_shape is inferred from the leading axes of
            # the concatenated draws (a matrix ``(n, vector_size)``).
            return _reconstruct_from_vector(self.name, record, samples)
        return samples

    def __repr__(self) -> str:
        return (
            f"ApproximateDistribution("
            f"algorithm={self.algorithm!r}, "
            f"num_chains={self.num_chains}, "
            f"num_draws={self.num_draws}, "
            f"components={list(self.event_spec.components)})"
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
        Per-chain draws, each shaped ``(num_draws, *flat)`` in the target's
        flat layout.
    parents : tuple of Distribution
        Parent distributions for provenance tracking.
    algorithm : str
        Inference algorithm name (e.g. ``"tfp_nuts"``, ``"blackjax_rwmh"``).
    annotations : DataTree or None
        Pre-built annotations DataTree (diagnostics, sample stats, warmup).
        Inference methods are responsible for building this.
    event_spec : OutputSpec, TermSpec, or None
        The target's declaration, usually the prior's ``event_spec``, which the
        result declares as its event.
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
