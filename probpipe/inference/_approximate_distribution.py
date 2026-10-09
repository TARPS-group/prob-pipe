"""Inference results: the empirical law of a run's draws on the levels ``chain`` and ``draw``."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xarray import DataTree

    from ..core._spec_base import TermSpec

import jax.numpy as jnp
import numpy as np

from .._weights import Weights
from ..core._numeric_array_batch import NumericArrayBatch
from ..core._numeric_record_batch import NumericRecordBatch
from ..core._opaque import OpaqueSpec
from ..core._repr import format_components
from ..core._specs import (
    NumericArraySpec,
    NumericRecordSpec,
    OutputSpec,
    RecordSpec,
    _components_record,
)
from ..core.named_tree import _PATH_SEP
from ..core.provenance import Provenance
from ..custom_types import Array, ArrayLike
from ..distributions._distribution import Distribution, _complete_event_spec
from ..distributions._empirical import EmpiricalDistribution

__all__ = ["make_posterior"]

#: The levels of an inference result's atoms, outermost first.
_CHAIN_LEVELS = ("chain", "draw")

#: The variable under which ``build_mcmc_datatree`` stores a run's flat draws.
_FLAT_DRAWS = "params"

#: The ArviZ groups whose flat draws an inference result names by leaf.
_DRAW_GROUPS = ("posterior", "warmup")


def _non_numeric(path: str, spec: Any) -> str:
    """The message for the parameter *path*, whose *spec* a numeric draw cannot fill."""
    return (
        f"cannot store the posterior draws of {path!r}: every parameter must be a numeric "
        f"array, but its spec is {type(spec).__name__}"
    )


def _first_non_numeric(record: RecordSpec) -> tuple[str, Any]:
    """The path and spec of the first leaf of *record* that is not a numeric array."""
    return next(
        (path, spec) for path, spec in record.items() if not isinstance(spec, NumericArraySpec)
    )


def _width_mismatch(width: int, needed: int, names: Any) -> str:
    """The message for draws of *width* values where the parameters *names* need *needed*."""
    return f"each draw has {width} values, but the parameters {list(names)} need {needed}"


def _spec_size(spec: NumericArraySpec | RecordSpec, path: str) -> int:
    """Number of scalar elements one field contributes to a flat vector.

    Given the spec of a single field of a :class:`RecordSpec`, return how
    many scalars that field occupies in the dense 1-D vector layout (see
    :meth:`~probpipe.NumericRecord.to_vector`): ``prod(shape)`` for an
    :class:`NumericArraySpec`, or :attr:`~NumericRecordSpec.vector_size` for a
    nested :class:`NumericRecordSpec`. It sizes each field's contiguous column
    block when the columns of a flat chain are permuted.

    Parameters
    ----------
    spec : NumericArraySpec or RecordSpec
        One field's spec, as returned by :meth:`RecordSpec.__getitem__`.
    path : str
        The field's name, which a message names.

    Returns
    -------
    int
        The width of the field's column block.

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
        leaf, leaf_spec = _first_non_numeric(spec)
        raise TypeError(_non_numeric(f"{path}{_PATH_SEP}{leaf}", leaf_spec))
    if isinstance(spec, NumericArraySpec):
        return prod(spec.shape) if spec.shape else 1
    raise TypeError(_non_numeric(path, spec))


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

    Parameters
    ----------
    record : RecordSpec
        The target's record, whose fields' flat sizes give the width of each
        column block.
    field_order : list of str
        The order of the flat chain's column blocks, which permutes
        ``record.fields``.

    Returns
    -------
    list of int
        The permutation ``perm``, which holds the source column of each column of
        the result.

    Raises
    ------
    ValueError
        If *field_order* is not a permutation of the record's fields, or
        a field has an opaque (``spec=None``) leaf with no flat size.
    """
    if sorted(field_order) != sorted(record.fields):
        raise ValueError(
            f"field_order {list(field_order)} is not a permutation of the fields "
            f"{list(record.fields)}"
        )
    sizes: dict[str, int] = {}
    for field_name in record.fields:
        spec = record.children[field_name]
        if isinstance(spec, OpaqueSpec):
            raise ValueError(_non_numeric(field_name, spec))
        sizes[field_name] = _spec_size(spec, field_name)
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


def _array_atoms(label: str, stacked: Array, term: NumericArraySpec) -> NumericArrayBatch:
    """The draws *stacked* ``(chains, draws, *flat)`` as atoms of the array term *term*.

    Each draw takes the shape the term declares, so a scalar term's draws are
    scalars, and the dtype the term declares.

    Parameters
    ----------
    label : str
        The batch's label.
    stacked : Array
        The draws of every chain, stacked along the leading axis.
    term : NumericArraySpec
        The target's array term; a term of symbolic shape takes the draws'
        trailing shape.

    Returns
    -------
    NumericArrayBatch
        The atoms on the levels ``chain`` and ``draw``, which take one axis each.

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
                f"each draw has {width} values, but {label!r} of shape {term.shape} needs "
                f"{prod(term.shape)}"
            )
        values = jnp.reshape(stacked, (chains, draws, *term.shape))
    else:
        values = stacked
        term = NumericArraySpec(tuple(stacked.shape[2:]), term.dtype, term.support)
    if term.dtype is not None:
        values = values.astype(term.dtype)
    return NumericArrayBatch(label, values, _CHAIN_LEVELS, element_spec=term, axes_per_level=(1, 1))


def _record_atoms(label: str, stacked: Array, record: RecordSpec) -> NumericRecordBatch:
    """The draws *stacked* ``(chains, draws, d)`` as atoms of *record*, in its flat layout.

    The columns follow the record's canonical leaf order, nested groups
    included, and each leaf takes the shape and dtype the record declares.

    Parameters
    ----------
    label : str
        The batch's label.
    stacked : Array
        The draws of every chain, stacked along the leading axis.
    record : RecordSpec
        The target's record declaration.

    Returns
    -------
    NumericRecordBatch
        The atoms on the levels ``chain`` and ``draw``, with one column per leaf
        path.

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
            raise ValueError(_non_numeric(path, spec))
    if not isinstance(record, NumericRecordSpec):
        raise TypeError(_non_numeric(*_first_non_numeric(record)))
    chains, draws = stacked.shape[:2]
    flat = jnp.reshape(stacked, (chains, draws, -1))
    if flat.shape[-1] != record.vector_size:
        raise ValueError(_width_mismatch(flat.shape[-1], record.vector_size, record.fields))
    columns, offset = {}, 0
    for path, spec in record.items():
        width = prod(spec.shape)
        column = jnp.reshape(flat[..., offset : offset + width], (chains, draws, *spec.shape))
        columns[path] = column if spec.dtype is None else column.astype(spec.dtype)
        offset += width
    return NumericRecordBatch(
        label, columns, _CHAIN_LEVELS, element_spec=record, axes_per_level=(1, 1)
    )


def _chain_atoms(label: str, stacked: Array, declaration: OutputSpec | None) -> Any:
    """The draws *stacked* ``(chains, draws, *flat)`` as atoms of the target's event term.

    Without a target, each draw is one array.
    """
    if declaration is None:
        element = NumericArraySpec(tuple(stacked.shape[2:]), stacked.dtype)
        return NumericArrayBatch(
            label, stacked, _CHAIN_LEVELS, element_spec=element, axes_per_level=(1, 1)
        )
    term = declaration.spec
    if isinstance(term, RecordSpec):
        return _record_atoms(label, stacked, term)
    if not isinstance(term, NumericArraySpec):
        raise TypeError(_non_numeric(label, term))
    return _array_atoms(label, stacked, term)


# ---------------------------------------------------------------------------
# Reading a result's chains
# ---------------------------------------------------------------------------


def _has_chains(law: Any) -> bool:
    """Whether *law* is an empirical law whose atoms lie on the levels ``chain`` and ``draw``."""
    return isinstance(law, EmpiricalDistribution) and tuple(law.atoms.level_names) == _CHAIN_LEVELS


def _num_chains(law: Any) -> int:
    """The number of chains of an inference result, and 1 for any other law."""
    return int(law.atoms.batch_shape[0]) if _has_chains(law) else 1


def _chain_columns(law: EmpiricalDistribution) -> dict[str, Array]:
    """The draws of *law* by leaf path, each an array ``(chains, draws, *shape)``.

    A whole-term result has one entry, under its component.

    Parameters
    ----------
    law : EmpiricalDistribution
        The inference result.

    Returns
    -------
    dict of str to Array
        One entry per leaf path, in the order the law stores its columns.

    Raises
    ------
    ValueError
        If the atoms of *law* do not lie on the levels ``chain`` and ``draw``.
    """
    if not _has_chains(law):
        raise ValueError(
            f"{law.label!r} has no chains: its draws are indexed by "
            f"{list(law.atoms.level_names)}, but chain diagnostics need the levels "
            f"{list(_CHAIN_LEVELS)} of a posterior from an MCMC method"
        )
    chains, draws = law.atoms.batch_shape
    rows = law._rows
    if isinstance(rows, dict):
        return {
            path: jnp.reshape(column, (chains, draws, *column.shape[1:]))
            for path, column in rows.items()
        }
    (component,) = law.event_spec.components
    return {component: jnp.reshape(rows, (chains, draws, *rows.shape[1:]))}


def _flat_chains(law: EmpiricalDistribution) -> Array:
    """The draws of *law* as one array ``(chains, draws, d)`` in its target's flat layout.

    The columns follow the leaf order of the result's components, nested groups
    included, as the method's chains did.

    Parameters
    ----------
    law : EmpiricalDistribution
        The inference result.

    Returns
    -------
    Array
        The columns of :func:`_chain_columns`, each flattened after the chain and
        draw axes and concatenated along the last axis.

    Raises
    ------
    ValueError
        If the atoms of *law* do not lie on the levels ``chain`` and ``draw``.
    """
    columns = _chain_columns(law)
    chains, draws = law.atoms.batch_shape
    return jnp.concatenate(
        [jnp.reshape(column, (chains, draws, -1)) for column in columns.values()], axis=-1
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def make_posterior(
    chains: list[Array],
    parents: tuple[Distribution, ...],
    method: str,
    *,
    annotations: DataTree | None = None,
    event_spec: OutputSpec | TermSpec | None = None,
    field_order: list[str] | None = None,
    weights: ArrayLike | Weights | None = None,
    label: str = "posterior",
    **meta: Any,
) -> EmpiricalDistribution:
    """The empirical law of an inference run's draws, with its record of the run.

    The result is an :class:`~probpipe.EmpiricalDistribution` labeled *label*,
    whose atoms are the draws on the levels ``chain`` and ``draw``, labeled by
    the law's components, as ``beta`` or ``(K, r, phi)``. Its event declaration is the target's: a whole-term target stays
    whole, and a record target keeps its fields' supports, scalar shapes, and
    nested groups. Its provenance names the method and the target. Its
    annotations are a ``DataTree`` whose root attribute ``method`` is *method*,
    and which stores the method's diagnostics, sample statistics, and warmup
    draws as ArviZ-compatible groups under ``arviz/``.

    Parameters
    ----------
    chains : list of Array
        Per-chain draws, each of shape ``(num_draws, *flat)`` in the target's
        flat layout; the chains have equal lengths.
    parents : tuple of Distribution
        The provenance's parents, usually the target alone.
    method : str
        The inference method's name, by which the ``method`` control selects it,
        such as ``"tfp_nuts"`` or ``"blackjax_rwmh"``.
    annotations : DataTree or None
        The method's diagnostics, sample statistics, and warmup draws. A
        ``posterior`` or ``warmup`` group that holds the flat draws as its one
        variable ``params``, as ``build_mcmc_datatree`` stores them, is stored
        with one variable per leaf of the result, named by its path with ``.``
        between the parts, since a ``DataTree`` variable has no ``/`` in its name.
    event_spec : OutputSpec, TermSpec, or None
        The target's declaration, usually the prior's ``event_spec``, which the
        result declares as its event. A bare ``RecordSpec`` exposes its fields,
        and any other term is a whole term. ``None`` makes each draw one array.
    field_order : list of str or None
        The field each contiguous column block of *chains* belongs to, in the
        order the blocks appear, for a backend whose columns follow another
        order than the target's components, such as one that sorts variable
        names. It requires *event_spec* and is a permutation of its components.
    weights : array-like, :class:`~probpipe.Weights`, or None
        Per-draw importance weights across all chains in chain order, as SMC-ABC
        returns them.
    label : str
        The result's label, which is also the component of a whole-term event
        that *event_spec* leaves unnamed.
    **meta
        Further metadata recorded in the provenance.

    Returns
    -------
    EmpiricalDistribution
        The posterior, with its atoms on the levels ``chain`` and ``draw``,
        labeled by its components.

    Raises
    ------
    ValueError
        If *chains* is empty or the chains differ in length, *field_order* is
        given without *event_spec* or is not a permutation of its components, or
        a draw's flat width is not the target's flat size.
    TypeError
        If the target has a leaf without a flat size.
    """
    if not chains:
        raise ValueError("chains must hold at least one chain; got none")
    declaration = None if event_spec is None else _complete_event_spec(event_spec, label)
    flat_chains = [jnp.asarray(chain) for chain in chains]

    # When the chain columns follow another field order than the target's, as for
    # a backend whose trace sorts variable names, permute them into the target's
    # order, so each column is read by name.
    if field_order is not None:
        if declaration is None:
            raise ValueError("field_order requires an event_spec, whose components it orders")
        record = _components_record(declaration)
        perm = _column_permutation(record, field_order)
        # The width is checked before the gather, which would otherwise drop
        # extra columns or clamp out-of-bounds indices.
        for chain in flat_chains:
            if chain.shape[-1] != len(perm):
                raise ValueError(_width_mismatch(chain.shape[-1], len(perm), record.fields))
        if len(record.fields) > 1:
            flat_chains = [chain[..., perm] for chain in flat_chains]

    lengths = sorted({int(chain.shape[0]) for chain in flat_chains})
    if len(lengths) > 1:
        raise ValueError(f"all chains must have the same length; got lengths {lengths}")
    components = (label,) if declaration is None else declaration.components
    atoms = _chain_atoms(format_components(components), jnp.stack(flat_chains), declaration)
    result = EmpiricalDistribution(label, atoms, weights, event_spec=declaration)
    if annotations is not None:
        annotations = _named_draw_groups(annotations, result)
    return _record_run(result, parents, method, annotations=annotations, **meta)


def _leaf_variables(flat: Any, layout: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """The flat draws *flat* ``(chains, draws, *flat)`` as one ArviZ variable per leaf of *layout*.

    *layout* maps each leaf path to its shape, in the flat layout's order. A
    variable is named by its leaf's path with ``.`` between the parts, and its
    event axes are named ``<name>_dim_<i>``, as ArviZ names them.
    """
    import xarray as xr

    chains, draws = flat.shape[:2]
    flat = np.reshape(np.asarray(flat), (chains, draws, -1))
    coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    variables, offset = {}, 0
    for path, shape in layout.items():
        name = path.replace(_PATH_SEP, ".")
        width = prod(shape)
        values = np.reshape(flat[..., offset : offset + width], (chains, draws, *shape))
        dims = ["chain", "draw", *(f"{name}_dim_{i}" for i in range(len(shape)))]
        variables[name] = xr.DataArray(values, dims=dims, coords=coords)
        offset += width
    return variables


def _named_draw_groups(annotations: Any, result: EmpiricalDistribution) -> dict[str, Any]:
    """The groups of *annotations* with the flat draws of each draw group named by leaf of *result*.

    A ``posterior`` or ``warmup`` group whose one variable is ``params`` holds
    the run's flat draws in the target's flat layout, which the leaves of
    *result* split, so ArviZ reports each component by its name.
    """
    import xarray as xr

    layout = {path: tuple(column.shape[2:]) for path, column in _chain_columns(result).items()}
    groups = {}
    for path, node in annotations.items():
        group = node.to_dataset() if isinstance(node, xr.DataTree) else node
        if path.strip("/") in _DRAW_GROUPS and list(group.data_vars) == [_FLAT_DRAWS]:
            variables = _leaf_variables(group[_FLAT_DRAWS].values, layout)
            group = xr.Dataset(variables, attrs=group.attrs)
        groups[path] = group
    return groups


def _record_run(
    result: Distribution,
    parents: tuple[Distribution, ...],
    method: str,
    *,
    annotations: DataTree | None = None,
    **meta: Any,
) -> Distribution:
    """*result* with the record of the inference run that produced it.

    Its annotations become a ``DataTree`` whose root attribute ``method`` is
    *method*, with the method's ArviZ-compatible groups under ``arviz/``, and its
    provenance names the method and *parents*.
    """
    import xarray as xr

    # The root records the method, and the ArviZ-compatible groups are nested
    # under /arviz/, so the annotations can hold other subtrees, such as
    # /diagnostics/, alongside them.
    dicto: dict = {"/": xr.Dataset(attrs={"method": method})}
    if annotations is not None:
        for group_path, node in annotations.items():
            if group_path == "/":
                continue
            clean = group_path.lstrip("/")
            ds = node.to_dataset() if isinstance(node, xr.DataTree) else node
            dicto[f"arviz/{clean}"] = ds
    result._init_annotations(xr.DataTree.from_dict(dicto))

    result.with_provenance(
        Provenance.create(method, parents=list(parents), metadata={"method": method, **meta})
    )
    return result
