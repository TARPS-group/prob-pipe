"""EventTemplate — ProbPipe's universal structural schema.

An :class:`EventTemplate` describes the **structure** of a value, independent
of the data itself. A value is in general allowed to be a nested, tree-like
structure (in JAX terminology, a PyTree). An :class:`EventTemplate` encodes the
structure of the tree, including unique named paths to each tree leaf.

The structure of a leaf node is described by one of a fixed set of "spec"
objects:

| Leaf spec          | What the leaf holds                                  |
|--------------------|------------------------------------------------------|
| :class:`ArraySpec` | a numeric array (event ``shape``, optional dtype / support) |
| :class:`DistributionSpec` | a ``Distribution``                            |
| :class:`FunctionSpec` | a callable (with input / output sub-templates)     |
| :class:`OpaqueSpec`| Any other object (no structure assumed)             |

A field whose spec is itself an ``EventTemplate`` is an *internal node*, not a
leaf. The canonical leaf order is defined via insertion order at the top level,
descending depth-first into nested templates. Each leaf is keyed by a unique
leaf path, using ``/`` as the path separator. :attr:`EventTemplate.leaf_paths`
enumerates the leaf paths in the canonical order.

Numeric vs. mixed
-----------------

:class:`NumericEventTemplate` is the specialization in which all leaves are
``ArraySpec``\\ s. It describes a value that is a PyTree of arrays. This
sub-class exposes :attr:`~NumericEventTemplate.vector_size` and supports
conversion to a flat 1-D array representation via
:meth:`EventTemplate.to_vector`, and back to the structured representation via
:meth:`EventTemplate.from_vector`. ``EventTemplate(...)`` auto-promotes to
``NumericEventTemplate`` when every leaf is numeric.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, ArrayLike
from .constraints import Constraint

if TYPE_CHECKING:
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray
    from .record import Record

__all__ = [
    "ArraySpec",
    "DistributionSpec",
    "EventTemplate",
    "FunctionSpec",
    "LeafSpec",
    "NumericEventTemplate",
    "OpaqueSpec",
]

# Separator for nested leaf paths (``"outer/inner"``). The path convention is
# shared between a template and the values it describes; :mod:`probpipe.core.record`
# imports these from here so the constant has a single home.
_PATH_SEP = "/"


def _check_no_path_sep(name: str) -> None:
    if _PATH_SEP in name:
        raise ValueError(
            f"Field name {name!r} must not contain {_PATH_SEP!r} "
            f"(reserved as the nested-path separator)."
        )


@dataclass(frozen=True)
class ArraySpec:
    """A numeric array leaf: a fixed event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); the current
    auto-build path stores only shape, and distributions populate the extra
    fields in a later phase. Both must be hashable when set.
    """

    shape: tuple[int, ...]
    dtype: jnp.dtype | None = None
    support: Constraint | None = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise TypeError(
                f"ArraySpec.shape must be a tuple of non-negative ints, got {self.shape!r}"
            )
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class OpaqueSpec:
    """A non-array Python-object leaf (str, DataFrame, ...).

    ``meta`` is optional opaque metadata and must be hashable (or ``None``).
    """

    meta: Hashable = None


@dataclass(frozen=True)
class DistributionSpec:
    """A leaf whose value is a ``Distribution``.

    ``inner_template`` is the :class:`EventTemplate` of one draw from that
    distribution.
    """

    inner_template: EventTemplate


@dataclass(frozen=True)
class FunctionSpec:
    """A leaf whose value is a callable.

    ``input_template`` / ``output_template`` are the :class:`EventTemplate`\\ s
    of the callable's input and output.
    """

    input_template: EventTemplate
    output_template: EventTemplate


# ``_FieldSpec`` adds nested templates to the public leaf union;
# ``_FieldSpecInput`` also admits the construction-time sugar the constructor
# normalises (a bare shape tuple or ``None``).
type LeafSpec = ArraySpec | OpaqueSpec | DistributionSpec | FunctionSpec
type _FieldSpec = LeafSpec | EventTemplate
type _FieldSpecInput = _FieldSpec | tuple[int, ...] | None


def _to_spec(spec: _FieldSpecInput) -> _FieldSpec:
    """Normalise a constructor input to a stored field spec.

    Construction-time sugar (preserved): a bare shape ``tuple`` becomes an
    :class:`ArraySpec`, ``None`` becomes an :class:`OpaqueSpec`, and a nested
    :class:`EventTemplate` is kept as-is. Already-built specs pass through, so
    new code may supply explicit ``ArraySpec(...)`` / ``OpaqueSpec(...)`` etc.
    """
    # NB: an explicit class tuple rather than ``(LeafSpec, EventTemplate)`` —
    # pyright doesn't narrow ``spec`` through a union-alias inside isinstance,
    # which would leave the ``return spec`` typed as the wider input alias.
    if isinstance(spec, (ArraySpec, OpaqueSpec, DistributionSpec, FunctionSpec, EventTemplate)):
        return spec
    if spec is None:
        return OpaqueSpec()
    if isinstance(spec, tuple):
        return ArraySpec(shape=spec)
    raise TypeError(
        f"spec must be a shape tuple, None, a leaf spec "
        f"(ArraySpec/OpaqueSpec/DistributionSpec/FunctionSpec), or an "
        f"EventTemplate, got {type(spec).__name__}"
    )


def _is_numeric_spec(spec: Any) -> bool:
    """A numeric leaf: an :class:`ArraySpec` or a (nested) :class:`NumericEventTemplate`."""
    return isinstance(spec, (ArraySpec, NumericEventTemplate))


def _all_numeric(specs: Iterable[Any]) -> bool:
    """True iff every (raw, pre-normalisation) input spec is numeric.

    Drives the base-class auto-promotion hook so ``EventTemplate(x=(), y=(3,))``
    returns a ``NumericEventTemplate`` without opting in explicitly. Raw inputs
    also allow the shape-tuple sugar; ``None`` / ``OpaqueSpec`` /
    ``DistributionSpec`` / ``FunctionSpec`` / mixed nested templates and any
    unsupported type are non-numeric (``__init__`` rejects the latter).
    """
    return all(isinstance(s, tuple) or _is_numeric_spec(s) for s in specs)


# dtype.kind codes for numeric arrays: b=bool, i=int, u=uint, f=float, c=complex.
_NUMERIC_KINDS = frozenset("biufc")


def _full_array_shape_or_none(val: Any) -> tuple[int, ...] | None:
    """Return the shape of a numeric array-like value, or ``None``.

    A numeric scalar reports shape ``()`` and a numeric array reports its
    ``shape``. Anything else — strings, object arrays, Python lists/tuples, and
    any value without a numeric ``dtype`` / ``shape`` — reports ``None``.
    """
    if isinstance(val, (bool, int, float, complex, np.integer, np.floating, np.bool_)):
        return ()
    if (
        hasattr(val, "shape")
        and hasattr(val, "dtype")
        and getattr(val.dtype, "kind", None) in _NUMERIC_KINDS
    ):
        return tuple(val.shape)
    return None


# ---------------------------------------------------------------------------
# EventTemplate — structural skeleton
# ---------------------------------------------------------------------------


class EventTemplate:
    """Structural description of a value: its named, possibly-nested leaf structure.

    The word *event* follows probabilistic-programming usage and **generalizes**
    the ``event`` / ``event_shape`` notion from other PPLs (TensorFlow
    Probability, distrax, NumPyro). There, ``event_shape`` is the shape of a
    single draw of one array-valued random variable. ProbPipe supports
    distributions over general value types, not just arrays. The *event* in this
    context can thus be a structured Python object, with structure described by
    the ``EventTemplate``.

    Terminology
    -----------
    Used precisely throughout this class:

    - **field** — a *top-level* entry of the template: a ``name -> spec`` pair,
      where the spec is either a leaf or a nested ``EventTemplate``.
      :attr:`fields` lists these names in insertion order; it does **not**
      descend into nested sub-templates.
    - **leaf** — a *terminal* node: an :class:`ArraySpec` / :class:`OpaqueSpec`
      / :class:`DistributionSpec` / :class:`FunctionSpec`. A nested
      ``EventTemplate`` is an *internal node*, not a leaf.
    - **path** — the ``/``-delimited sequence of field names from the root to a
      node (e.g. ``"physics/mass"``); a top-level field is a single-segment
      path. These are the same path strings used to index into a value, e.g.
      ``record["physics/mass"]``.
    - **canonical leaf order** — the order in which leaves are traversed:
      depth-first, following each level's insertion order. This is the single
      ordering every leaf-wise operation uses. :attr:`leaf_paths` is its
      canonical definition — it returns the path to every leaf in this order;
      :meth:`to_vector` / :meth:`from_vector` lay out and read leaves in it, and
      :attr:`leaf_shapes` is keyed by it.

    PyTree contract
    ---------------
    The *values* an ``EventTemplate`` describes are JAX pytrees, and the template
    is the schema for that pytree — the structural analog of a JAX ``PyTreeDef``,
    enriched with each leaf's kind and shape. Nested ``EventTemplate``\\s mirror
    the value's internal nodes (their field names are the keys), and the terminal
    specs (:class:`ArraySpec` / :class:`OpaqueSpec` / :class:`DistributionSpec` /
    :class:`FunctionSpec`) mirror the value's leaves. A leaf's ``/``-delimited
    :attr:`path <leaf_paths>` is the string form of the value's JAX key path, and
    the canonical leaf order is JAX's left-to-right depth-first order.

    So the template lines up directly with JAX's pytree machinery on a value: for
    a value ``v`` matching this template, ``jax.tree_util.tree_leaves(v)`` returns
    the leaves in :attr:`leaf_paths` order, and ``jax.tree.map(f, v)`` rebuilds a
    value with the same structure. (The ``EventTemplate`` object is itself an
    opaque pytree *leaf*, not a node — it is structural metadata, like a
    ``PyTreeDef``, not a container JAX recurses into.)

    Parameters
    ----------
    **field_specs
        Named fields. Each value is one of:

        - ``tuple[int, ...]`` — shape of a numeric array leaf (e.g. ``()`` for a
          scalar, ``(3,)`` for a 3-vector); normalised to :class:`ArraySpec`.
        - ``None`` — opaque (non-array) leaf; normalised to :class:`OpaqueSpec`.
        - a leaf spec — :class:`ArraySpec` / :class:`OpaqueSpec` /
          :class:`DistributionSpec` / :class:`FunctionSpec` (passed through).
        - ``EventTemplate`` — a nested sub-structure (an internal node).

    Examples
    --------
    ::

        EventTemplate(x=(), y=(3,))                     # -> NumericEventTemplate
        EventTemplate(label=None, x=())                 # -> EventTemplate (mixed)
        EventTemplate(physics=EventTemplate(force=(), mass=()), obs=())

    Notes
    -----
    Inspired by JAX's ``PyTreeDef``: a template can reconstruct a value from its
    leaves and describes the expected structure for type-checking and
    vectorization. Leaves are stored as frozen, hashable spec objects, so a
    template is itself hashable (usable as a jit / treedef cache key).
    ``__getitem__`` returns the stored spec (or nested template); shape-shaped
    access lives on :attr:`leaf_shapes` / :attr:`event_shapes` /
    :meth:`field_event_shape`.

    Calling ``EventTemplate(...)`` directly auto-promotes to a
    :class:`NumericEventTemplate` when every spec is numeric (and every nested
    sub-template is itself all-numeric), so :attr:`vector_size` and
    :attr:`numeric_leaf_shapes` are reachable in the common all-numeric case
    without naming the subclass. Mixed templates (any opaque / ``None`` spec)
    stay plain ``EventTemplate`` and do not expose :attr:`vector_size` — it is
    not a meaningful quantity once opaque leaves are present.
    """

    __slots__ = ("_specs",)

    def __new__(
        cls,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericEventTemplate(...)`` calls bypass this path
        # and run their own strict validation.
        if cls is EventTemplate:
            specs = _dict if _dict is not None else field_specs
            if specs and _all_numeric(specs.values()):
                return object.__new__(NumericEventTemplate)
        return object.__new__(cls)

    def __init__(
        self,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        source: Mapping[str, _FieldSpecInput]
        if _dict is not None:
            if field_specs:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            source = _dict
        else:
            source = field_specs
        if not source:
            raise ValueError(f"{type(self).__name__} requires at least one field")
        specs: dict[str, _FieldSpec] = {}
        for name, spec in source.items():
            _check_no_path_sep(name)
            try:
                specs[name] = _to_spec(spec)
            except TypeError as exc:
                raise TypeError(f"Field {name!r}: {exc}") from None
        self._post_validate(specs)
        object.__setattr__(self, "_specs", specs)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __reduce__(self):
        return (_unpickle_event_template, (dict(self._specs),))

    # -- Field access -------------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Top-level field names, in insertion order.

        Returns the names of the template's *top-level* fields only (see the
        Terminology section); it does **not** descend into nested
        sub-templates, so a nested field contributes a single name here. For
        the path to every leaf in canonical leaf order, use :attr:`leaf_paths`;
        the two coincide for a flat template (one where every field is a leaf).
        """
        return tuple(self._specs.keys())

    @property
    def leaf_paths(self) -> tuple[str, ...]:
        """Path to every leaf, in canonical leaf order.

        Returns the ``/``-delimited path (see the Terminology section) to each
        leaf of the template, traversed depth-first in each level's insertion
        order. This property **is the canonical definition** of that order and
        of the leaf keys it yields: every leaf-wise operation in ProbPipe visits
        leaves in exactly this order and names them by exactly these paths —
        :meth:`to_vector` / :meth:`from_vector` lay out and read the per-leaf
        blocks in it, :meth:`~probpipe.Record.flatten` returns the value's
        leaves in it, and :attr:`leaf_shapes` (and
        :attr:`~NumericEventTemplate.numeric_leaf_shapes`) are keyed by these
        same paths. A nested field expands into one path per nested leaf; a flat
        template's ``leaf_paths`` equals its :attr:`fields`.

        Returns
        -------
        tuple of str
            One ``/``-delimited path per leaf, in canonical leaf order.
        """
        paths: list[str] = []
        for name, spec in self._specs.items():
            if isinstance(spec, EventTemplate):
                paths.extend(f"{name}{_PATH_SEP}{sub}" for sub in spec.leaf_paths)
            else:
                paths.append(name)
        return tuple(paths)

    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...] | None]:
        """Per-leaf shapes, keyed by :attr:`leaf_paths` (canonical leaf order).

        Maps each leaf's ``/``-delimited path to its array shape, or ``None``
        for an opaque (non-array) leaf. The keys are exactly :attr:`leaf_paths`,
        in canonical leaf order; a nested sub-template contributes one entry per
        nested leaf.
        """
        result: dict[str, tuple[int, ...] | None] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, EventTemplate):
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}{_PATH_SEP}{sub_name}"] = sub_shape
            elif isinstance(spec, ArraySpec):
                result[name] = spec.shape
            else:
                # Opaque / distribution / function leaves have no array shape.
                result[name] = None
        return result

    @property
    def event_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-field event shapes, keyed by :attr:`fields` (top-level).

        Emits one entry per *top-level* field — keyed by :attr:`fields`, not by
        :attr:`leaf_paths`. An :class:`ArraySpec` field returns its shape; a
        nested sub-template or opaque leaf collapses to ``()``. Contrast
        :attr:`leaf_shapes`, which descends and emits one entry per leaf. This
        is the view downstream Distribution code wants when answering "what is
        the per-field event shape of one draw?".
        """
        return {name: self.field_event_shape(name) for name in self._specs}

    def field_event_shape(self, name: str) -> tuple[int, ...]:
        """Event shape for one top-level field.

        :class:`ArraySpec` leaves return their ``shape`` verbatim; opaque /
        distribution / function leaves and nested ``EventTemplate``
        sub-structures collapse to ``()``. Raises ``KeyError`` if ``name`` is
        not a top-level field.
        """
        spec = self._specs[name]
        if isinstance(spec, ArraySpec):
            return spec.shape
        return ()

    def pack(self, **field_kwargs: Any) -> Record:
        """Build a :class:`Record` from named values matching this template.

        Validates that *field_kwargs* names exactly this template's top-level
        :attr:`fields` (no missing or unexpected names) and returns a
        ``Record`` keyed in field order. Values pass through unchanged (a
        nested-template field takes a sub-``Record``). Object form of
        :func:`probpipe.core.record._pack_fields`.
        """
        # Lazy import keeps the module graph one-way (record -> event_template).
        from .record import _pack_fields

        return _pack_fields(self.fields, field_kwargs, owner=type(self).__name__)

    # -- Numeric queries & projection ---------------------------------------

    @property
    def is_numeric(self) -> bool:
        """Whether every reachable leaf is an :class:`ArraySpec`.

        Recursive: descends into nested :class:`EventTemplate` fields and
        returns ``True`` only if *all* leaves (at every depth) are numeric
        array leaves. Any :class:`OpaqueSpec` / :class:`DistributionSpec` /
        :class:`FunctionSpec` leaf — or a nested sub-template that is not
        itself all-numeric — makes the whole template non-numeric.

        This is computed as an explicit recursive leaf check rather than
        ``isinstance(self, NumericEventTemplate)``. Under the ``__new__``
        auto-promotion invariant the two agree, but the recursive form is
        also correct for hand-built mixed nestings.

        Returns
        -------
        bool
            ``True`` iff every reachable leaf is an :class:`ArraySpec`.
        """
        for spec in self._specs.values():
            if isinstance(spec, ArraySpec):
                continue
            if isinstance(spec, EventTemplate):
                if not spec.is_numeric:
                    return False
                continue
            # Opaque / distribution / function leaf — not numeric.
            return False
        return True

    @property
    def is_multi_field(self) -> bool:
        """Whether this template describes more than one leaf.

        Counts *reachable* leaves recursively (descending into nested
        sub-templates), not top-level fields — so a single top-level field that
        nests several leaves is multi-field. For example
        ``EventTemplate(a=EventTemplate(b=(), c=()))`` has leaves ``a/b`` and
        ``a/c`` and is multi-field, whereas ``EventTemplate(a=EventTemplate(b=()))``
        describes the single leaf ``a/b`` and is not. Equivalent to
        ``len(self.leaf_shapes) > 1``.

        Returns
        -------
        bool
            ``True`` iff the template has more than one leaf; ``False`` iff it
            describes exactly one leaf.
        """
        return len(self.leaf_shapes) > 1

    def numeric_fields(self) -> tuple[str, ...]:
        """Top-level field names whose leaf is numeric.

        A top-level field is numeric if its spec is an :class:`ArraySpec` or a
        nested :class:`EventTemplate` that is itself all-numeric (see
        :attr:`is_numeric`). The exact complement of :meth:`non_numeric_fields`:
        together the two partition :attr:`fields` (each in insertion order).

        Note that this reports *top-level* fields only, so a field is numeric
        here iff its nested sub-template is *entirely* numeric — it does not
        descend to count partially-numeric nestings. Use :meth:`numeric_subset`
        for the recursive, path-stable projection to numeric leaves.

        Returns
        -------
        tuple of str
            Names of the numeric top-level fields, in insertion order. Empty
            when every top-level field is non-numeric.
        """
        result: list[str] = []
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec) or (isinstance(spec, EventTemplate) and spec.is_numeric):
                result.append(name)
        return tuple(result)

    def non_numeric_fields(self) -> tuple[str, ...]:
        """Top-level field names whose leaf is non-numeric.

        A top-level field is non-numeric if its spec is an
        :class:`OpaqueSpec` / :class:`DistributionSpec` / :class:`FunctionSpec`
        leaf, or a nested :class:`EventTemplate` that is not itself all-numeric
        (see :attr:`is_numeric`). The exact complement of :meth:`numeric_fields`:
        together the two partition :attr:`fields`. Used to build clear inference
        error messages when a template cannot be projected to a numeric subset.

        Returns
        -------
        tuple of str
            Names of the non-numeric top-level fields, in insertion order.
            Empty when the template :attr:`is_numeric`.
        """
        result: list[str] = []
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec):
                continue
            if isinstance(spec, EventTemplate) and spec.is_numeric:
                continue
            result.append(name)
        return tuple(result)

    def numeric_subset(self) -> NumericEventTemplate:
        """Project to the :class:`ArraySpec`-leaf sub-template.

        Keeps every numeric leaf, recursing into nested
        :class:`EventTemplate` fields (each contributes its own
        ``numeric_subset()``); drops :class:`OpaqueSpec` /
        :class:`DistributionSpec` / :class:`FunctionSpec` leaves; and prunes
        any nested template that becomes empty. Surviving leaves keep their
        ``/``-delimited paths (the projection is path-stable). Inference uses
        this to recover the numeric leaves of a mixed template.

        On an already-all-numeric template the result is an equal
        :class:`NumericEventTemplate` (the projection is idempotent).

        Returns
        -------
        NumericEventTemplate
            The numeric-leaf sub-template, so that :attr:`vector_size` and
            :attr:`numeric_leaf_shapes` are available.

        Raises
        ------
        ValueError
            If no numeric leaves survive — an :class:`EventTemplate` needs at
            least one field, so an empty numeric subset is meaningless. The
            message names the dropped (non-numeric) fields.
        """
        specs: dict[str, _FieldSpec] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, ArraySpec):
                specs[name] = spec
            elif isinstance(spec, EventTemplate):
                try:
                    specs[name] = spec.numeric_subset()
                except ValueError:
                    # The empty-projection guard below is the *only* ValueError
                    # numeric_subset() raises, so catching it here means the
                    # nested template had no numeric leaves — prune it. If a
                    # future change adds another ValueError path, narrow this
                    # catch so it can't mask an unrelated failure.
                    continue
            # Opaque / distribution / function leaves are dropped.
        if not specs:
            raise ValueError(
                f"numeric_subset() of {type(self).__name__} is empty: no "
                f"ArraySpec leaves survive. Dropped non-numeric fields: "
                f"{self.non_numeric_fields()}."
            )
        return NumericEventTemplate(specs)

    # -- 1-D numeric (de)serialization --------------------------------------

    def to_vector(self, value: NumericRecord | NumericRecordArray) -> Array:
        """Serialize the numeric leaves of *value* into its 1-D vector representation.

        ``to_vector`` / :meth:`from_vector` convert between the structured and
        vector representations of a **numeric** value. A numeric value is in
        general a structured tree of array-valued leaves, with
        :attr:`~NumericEventTemplate.vector_size` the total number of scalar
        values making up those arrays. ``to_vector`` converts the structured
        value to its unique flat representation: an array of shape
        ``(vector_size,)``. A batch of numeric values is flattened to a matrix:
        an array of shape ``(*batch_shape, vector_size)``.

        This method is distinct from ``flatten``. The latter follows the
        JAX-pytree terminology, implying the mapping ``value -> (leaves, aux)``
        in the sense of :func:`jax.tree_util.tree_flatten`. ``flatten`` returns
        the list of the leaves of the value object, and thus is defined for all
        values (not just numeric ones). ``to_vector`` applies only to numeric
        values, and goes further in that it ravels the leaf arrays to a 1-D
        representation.

        Leaves are visited in the value's pytree-leaf order, which for a *value*
        matching this template coincides with the template's **canonical leaf
        order** — the deterministic, depth-first, insertion-order traversal also
        used by :attr:`leaf_shapes`. The structural operation lives on the event
        template; a :class:`~probpipe.NumericRecord` inherits the functionality
        from the template.

        Parameters
        ----------
        value : NumericRecord or NumericRecordArray
            The value to serialize; its structure must match this template. A
            single :class:`~probpipe.NumericRecord` yields a 1-D vector; a
            batched :class:`~probpipe.NumericRecordArray` with
            ``batch_shape == B`` yields one vector per batch element.

        Returns
        -------
        jax.Array
            The concatenated numeric leaves, dtype-promoted by
            ``jnp.concatenate``. Shape ``(vector_size,)`` for a single
            ``value`` and ``(*B, vector_size)`` for a batched ``value`` with
            ``batch_shape == B``, where ``vector_size == self.vector_size``.

        Raises
        ------
        TypeError
            If this template is not :attr:`is_numeric` — it has non-numeric
            leaves, so there is no canonical numeric vector. The message names
            the offending fields and points at :meth:`numeric_subset` (which
            projects to the numeric leaves first); ``to_vector`` never silently
            drops non-numeric leaves. Also raised if *value* is not a
            :class:`~probpipe.NumericRecord` / :class:`~probpipe.NumericRecordArray`.

        See Also
        --------
        from_vector : Reconstruct a value from a flat vector (the inverse).
        numeric_subset : Project a mixed template to its numeric leaves.
        """
        if not self.is_numeric:
            raise TypeError(
                f"{type(self).__name__}.to_vector requires an all-numeric "
                f"template, but these fields are non-numeric: "
                f"{self.non_numeric_fields()}. Project to the numeric leaves "
                f"first with numeric_subset()."
            )
        from ._numeric_record import NumericRecord
        from ._record_array import NumericRecordArray

        if isinstance(value, NumericRecordArray):
            batch_shape = value.batch_shape
        elif isinstance(value, NumericRecord):
            batch_shape = ()
        else:
            raise TypeError(
                f"to_vector expects a NumericRecord (single) or "
                f"NumericRecordArray (batched), got {type(value).__name__}."
            )

        # Delegate leaf traversal to the JAX-pytree ``flatten``: it yields the
        # numeric leaves in canonical leaf order. Ravel each leaf's event
        # dimensions and concatenate along the trailing axis. Reshaping to
        # ``(*batch_shape, -1)`` preserves leading batch axes (``batch_shape ==
        # ()`` for a single value gives a 1-D result).
        leaves, _aux = value.flatten()
        return jnp.concatenate([jnp.reshape(leaf, (*batch_shape, -1)) for leaf in leaves], axis=-1)

    def from_vector(
        self,
        vec: ArrayLike,
        *,
        non_numeric: Mapping[str, Any] | None = None,
    ) -> Any:
        """Reconstruct a numeric value from its 1-D vector representation.

        :meth:`to_vector` / ``from_vector`` convert between the structured and
        vector representations of a **numeric** value (see :meth:`to_vector`).
        ``from_vector`` is the inverse: it splits *vec* along its trailing axis
        into the template's leaves — in the same canonical leaf order
        :meth:`to_vector` uses — reshapes each chunk to its event shape, and
        rebuilds the structured value. The **rank of** *vec* selects single vs.
        batched: a vector of shape ``(vector_size,)`` rebuilds a single value,
        while a matrix of shape ``(*batch_shape, vector_size)`` rebuilds a batch
        with that ``batch_shape``. Here ``vector_size`` is the total scalar
        count, :attr:`~NumericEventTemplate.vector_size` (for a mixed template,
        the ``vector_size`` of its :meth:`numeric_subset`).

        This method is distinct from ``unflatten``. The latter is the inverse of
        the JAX-pytree ``flatten`` (the mapping ``(leaves, aux) -> value`` in the
        sense of :func:`jax.tree_util.tree_unflatten`), rebuilding an arbitrary
        value from its leaves and structural ``aux``. ``from_vector`` applies
        only to numeric values, and rebuilds one from its dense 1-D
        representation alone — splitting and reshaping the vector using this
        template's leaf shapes.

        When this template is mixed (not :attr:`is_numeric`), *vec* carries only
        the numeric leaves that :meth:`numeric_subset` keeps; pass *non_numeric*
        to supply the dropped leaves and rebuild the full mixed value. Like
        :meth:`to_vector`, the structural operation lives on the event template.

        Parameters
        ----------
        vec : array-like
            The flat numeric vector; its trailing axis must have length
            ``vector_size``.
        non_numeric : Mapping[str, Any], optional
            Values for the leaves that :meth:`numeric_subset` dropped, keyed by
            their ``/``-delimited leaf path, used to rebuild a *full* mixed
            value from a numeric-only vector. Required — and only meaningful —
            when this template is **not** :attr:`is_numeric`; must be ``None``
            or empty when it is. For a batched result each supplied value must
            itself carry the leading ``batch_shape``.

        Returns
        -------
        NumericRecord or NumericRecordArray
            When this template :attr:`is_numeric`: the reconstructed numeric
            value (single → :class:`~probpipe.NumericRecord`; batched →
            :class:`~probpipe.NumericRecordArray`).
        Record or RecordArray
            When this template is mixed and *non_numeric* is supplied: the full
            mixed value, with numeric leaves taken from *vec* and the remaining
            leaves taken from *non_numeric* (single → :class:`~probpipe.Record`;
            batched → :class:`~probpipe.RecordArray`).

        Raises
        ------
        ValueError
            If *vec*'s trailing axis is not ``vector_size``; if *non_numeric*
            is ``None`` while this template is non-numeric (the dropped leaves
            cannot be supplied); if *non_numeric* is non-empty while this
            template is numeric (there are no dropped leaves); if a required
            dropped-leaf path is missing from *non_numeric*; or if a batched
            *non_numeric* value's leading axes do not match ``batch_shape``.

        Notes
        -----
        **Round-trip invariant.** ``self.from_vector(self.to_vector(v)) == v``
        for any numeric ``value`` ``v`` matching this (numeric) template. For a
        mixed template ``T`` whose numeric leaves form ``Tn = T.numeric_subset()``,
        the analogous round trip
        ``T.from_vector(Tn.to_vector(vn), non_numeric=dropped)`` rebuilds the
        full mixed value (a plain :class:`~probpipe.Record` /
        :class:`~probpipe.RecordArray`) from its numeric part ``vn`` and the
        dropped leaves.

        See Also
        --------
        to_vector : Serialize a value to a flat vector (the inverse).
        numeric_subset : Project a mixed template to its numeric leaves.
        """
        vec = jnp.asarray(vec)
        numeric = self.is_numeric

        if numeric and non_numeric:
            raise ValueError(
                f"{type(self).__name__}.from_vector: this template is numeric "
                f"(numeric_subset drops no leaves), so non_numeric must be None "
                f"or empty, got keys {sorted(non_numeric)}."
            )
        if not numeric and not non_numeric:
            raise ValueError(
                f"{type(self).__name__}.from_vector: this template is mixed; its "
                f"non-numeric leaves {self.non_numeric_fields()} were dropped by "
                f"numeric_subset and must be supplied via non_numeric to rebuild "
                f"a full value."
            )

        # vector_size is the total scalar count across the numeric leaves;
        # numeric_subset() is idempotent on an already-numeric template.
        num_tpl = self if isinstance(self, NumericEventTemplate) else self.numeric_subset()
        vector_size = num_tpl.vector_size
        if vec.shape[-1] != vector_size:
            raise ValueError(
                f"{type(self).__name__}.from_vector: vec trailing axis is "
                f"{vec.shape[-1]}, expected vector_size={vector_size}."
            )

        batched = vec.ndim > 1
        batch_shape = tuple(vec.shape[:-1]) if batched else ()
        non_numeric = dict(non_numeric) if non_numeric else {}

        def _check_batch(path: str, val: Any) -> None:
            shape = getattr(val, "shape", None)
            if shape is not None and tuple(shape[: len(batch_shape)]) != batch_shape:
                raise ValueError(
                    f"{type(self).__name__}.from_vector: non_numeric[{path!r}] has "
                    f"leading shape {tuple(shape[: len(batch_shape)])}, expected "
                    f"batch_shape {batch_shape}."
                )

        # Collect the reconstructed value's pytree leaves in canonical leaf order:
        # slice ``vec`` into each numeric leaf (reshaped to (*batch_shape,
        # *event_shape), one offset advancing along vec's trailing axis) and pull
        # each dropped non-numeric leaf (mixed templates only) from ``non_numeric``,
        # which consumes no vector. Leaf reconstruction (assembling these leaves
        # into the structured value) is then delegated to the JAX-pytree
        # ``unflatten``; the matching treedef is derived from this template and
        # batch_shape so the assembly logic lives in one place.
        offset = 0
        leaves: list[Any] = []

        def _collect(template: EventTemplate, prefix: str) -> None:
            nonlocal offset
            for name in template.fields:
                spec = template[name]
                path = f"{prefix}{name}"
                if isinstance(spec, EventTemplate):
                    _collect(spec, f"{path}{_PATH_SEP}")
                elif isinstance(spec, ArraySpec):
                    size = prod(spec.shape) if spec.shape else 1
                    chunk = vec[..., offset : offset + size]
                    offset += size
                    leaves.append(jnp.reshape(chunk, (*batch_shape, *spec.shape)))
                else:
                    # Opaque / distribution / function leaf — supplied by caller.
                    if path not in non_numeric:
                        raise ValueError(
                            f"{type(self).__name__}.from_vector: missing non_numeric "
                            f"value for dropped leaf {path!r}."
                        )
                    val = non_numeric[path]
                    if batched:
                        _check_batch(path, val)
                    leaves.append(val)

        _collect(self, "")
        treedef = _value_treedef(self, batch_shape, numeric=numeric)
        # Equivalent to ``Record.unflatten(leaves, treedef)``; called directly
        # to avoid importing the value type into this foundational module.
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> _FieldSpec:
        """Return the stored spec for *name*.

        Returns the leaf spec object (:class:`ArraySpec` / :class:`OpaqueSpec`
        / :class:`DistributionSpec` / :class:`FunctionSpec`) or, for a nested
        field, the nested :class:`EventTemplate`. For shape-shaped access use
        :attr:`leaf_shapes` / :attr:`event_shapes` / :meth:`field_event_shape`.
        """
        return self._specs[name]

    def __len__(self) -> int:
        return len(self._specs)

    # -- Equality and hashing -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventTemplate):
            return NotImplemented
        # Order-sensitive comparison so equality matches the
        # order-sensitive ``__hash__`` (insertion order is part of the
        # template's identity). dict.__eq__ alone would ignore order,
        # breaking the eq/hash contract.
        return tuple(self._specs.items()) == tuple(other._specs.items())

    def __hash__(self) -> int:
        # All field specs (leaf specs and nested templates) are hashable, so
        # the order-sensitive item tuple hashes directly. Insertion order is
        # part of the template's identity.
        return hash(tuple(self._specs.items()))

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def infer_from(
        cls,
        value: Record,
        *,
        batch_shape: tuple[int, ...] = (),
    ) -> EventTemplate:
        """Best-effort, **lossy** template inferred by inspecting a value.

        Walks *value*'s fields and reconstructs a schema from the runtime
        data: a numeric leaf becomes an :class:`ArraySpec` of its event shape
        (the leaf shape with the leading ``batch_shape`` stripped); a numeric
        Python scalar becomes a shape-``()`` ``ArraySpec``; anything else
        becomes a bare :class:`OpaqueSpec`. Auto-promotes to a
        :class:`NumericEventTemplate` when *value* is a
        :class:`~probpipe.NumericRecord`.

        This is a **fallback** for when no authoritative template is available
        (e.g. wrapping a raw value at a boundary); it is **not** the way to
        obtain a value's schema — read the stored ``value.event_template``
        instead, which is authoritative. Inference is lossy: it cannot recover
        an :class:`ArraySpec`'s ``dtype`` / ``support``, an
        :class:`OpaqueSpec`'s ``meta``, or a :class:`DistributionSpec` /
        :class:`FunctionSpec` — every non-array leaf collapses to a bare
        ``OpaqueSpec``.

        Parameters
        ----------
        value : Record
            Source value whose fields define the inferred structure.
        batch_shape : tuple of int
            Leading dimensions to strip from each leaf's shape to recover the
            event shape. For a single (unbatched) value use ``()`` (default).

        Returns
        -------
        EventTemplate
            The inferred schema; a :class:`NumericEventTemplate` when *value*
            is a ``NumericRecord``.

        Notes
        -----
        A Python ``list`` or ``tuple`` leaf has no ``.shape`` / ``.dtype`` and
        is treated as opaque even if it contains numbers. Wrap it in
        ``np.asarray(...)`` / ``jnp.asarray(...)`` before storing it if you
        want a numeric leaf; otherwise :meth:`from_vector` will later raise on
        the opaque field.
        """
        # Lazy import keeps the module graph one-way (record -> event_template).
        from .record import Record

        # Promote to ``NumericEventTemplate`` when the source signals it is
        # all-numeric (a ``NumericRecord``), so ``vector_size`` stays reachable
        # for the common all-numeric case without naming the subclass.
        target_cls = cls
        if cls is EventTemplate:
            from ._numeric_record import NumericRecord

            if isinstance(value, NumericRecord):
                target_cls = NumericEventTemplate
        n_batch = len(batch_shape)
        # Construction-time sugar; the constructor normalises to stored specs.
        specs: dict[str, _FieldSpecInput] = {}
        for name in value.fields:
            val = value[name]
            if isinstance(val, Record):
                specs[name] = target_cls.infer_from(val, batch_shape=batch_shape)
                continue
            # Numeric leaf → event shape (drop leading batch dims); opaque → None.
            full_shape = _full_array_shape_or_none(val)
            specs[name] = None if full_shape is None else full_shape[n_batch:]
        return target_cls(specs)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, spec in self._specs.items():
            if isinstance(spec, EventTemplate):
                parts.append(f"{name}={spec!r}")
            elif isinstance(spec, ArraySpec) and spec.dtype is None and spec.support is None:
                # Bare specs render as their sugar form (shape tuple / None).
                parts.append(f"{name}={spec.shape}")
            elif isinstance(spec, OpaqueSpec) and spec.meta is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}={spec!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NumericEventTemplate — all-numeric specialisation
# ---------------------------------------------------------------------------


class NumericEventTemplate(EventTemplate):
    """EventTemplate where every leaf is numeric.

    Extends :class:`EventTemplate` by requiring each spec to be a shape
    tuple (or a nested :class:`NumericEventTemplate`) — no opaque
    ``None`` leaves are allowed. That restriction is what makes
    :attr:`vector_size` and :meth:`numeric_leaf_shapes` meaningful:
    ``vector_size`` is the length of the per-element 1-D vector — the total
    number of scalar elements across every numeric leaf — and
    :meth:`~EventTemplate.from_vector` requires a template of this class so
    that every field can be reconstructed from a slice of that vector. A
    *batch* of such values is a matrix of shape ``(*batch_shape, vector_size)``,
    not a single vector.

    Use :meth:`EventTemplate.infer_from` on a :class:`NumericRecord`
    (it auto-promotes) or call this constructor directly when you have
    the shape specs in hand.
    """

    __slots__ = ("_vector_size",)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        for name, spec in field_specs.items():
            if _is_numeric_spec(spec):
                continue
            if isinstance(spec, EventTemplate):
                raise TypeError(
                    f"NumericEventTemplate: nested field {name!r} is a "
                    f"{type(spec).__name__}; nested sub-templates must "
                    f"themselves be NumericEventTemplate."
                )
            if isinstance(spec, OpaqueSpec):
                raise TypeError(
                    f"NumericEventTemplate: field {name!r} is opaque "
                    f"(OpaqueSpec); opaque leaves are not allowed — use "
                    f"EventTemplate if you need a mixed template."
                )
            # DistributionSpec / FunctionSpec — non-array leaves.
            raise TypeError(
                f"NumericEventTemplate: field {name!r} has a non-numeric leaf "
                f"({type(spec).__name__}); only ArraySpec leaves (or nested "
                f"NumericEventTemplate) are allowed — use EventTemplate if you "
                f"need a mixed template."
            )

    def __init__(
        self,
        _dict: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        super().__init__(_dict, **field_specs)
        object.__setattr__(self, "_vector_size", self._compute_vector_size())

    @property
    def numeric_leaf_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-leaf shapes, keyed by :attr:`leaf_paths` (canonical leaf order).

        On :class:`NumericEventTemplate` every leaf is numeric, so this equals
        :attr:`leaf_shapes` (no ``None`` entries). Kept as a distinct name for
        symmetry with historical callers that used it as a numeric filter.
        """
        return dict(self.leaf_shapes)

    def _compute_vector_size(self) -> int:
        """Total scalar count across all numeric leaves."""
        total = 0
        for spec in self._specs.values():
            if isinstance(spec, NumericEventTemplate):
                total += spec.vector_size
            else:
                # spec is an ArraySpec — validated by ``_post_validate``.
                total += prod(spec.shape) if spec.shape else 1
        return total

    @property
    def vector_size(self) -> int:
        """Length of the per-element 1-D vector (``to_vector`` / ``from_vector``).

        The total number of scalar elements across all numeric leaves — the
        trailing-axis length of :meth:`~EventTemplate.to_vector`'s output. A
        single value serializes to shape ``(vector_size,)``; a batch serializes
        to a matrix ``(*batch_shape, vector_size)``, not a single vector.
        """
        return self._vector_size


# ---------------------------------------------------------------------------
# Template walking helpers
# ---------------------------------------------------------------------------


def _spec_size(spec: _FieldSpec) -> int:
    """Number of scalar elements a leaf-spec stands for.

    Used when walking a template and slicing a flat vector into per-leaf chunks
    (e.g. sizing the numeric-leaf blocks of an approximate distribution). Nested
    specs must be :class:`NumericEventTemplate` so that ``.vector_size`` is
    defined; non-array leaves (:class:`OpaqueSpec` / :class:`DistributionSpec` /
    :class:`FunctionSpec`) have no vector size and raise.
    """
    if isinstance(spec, NumericEventTemplate):
        return spec.vector_size
    if isinstance(spec, EventTemplate):
        raise TypeError(
            f"nested {type(spec).__name__} contains non-numeric leaves; "
            f"vector (de)serialization requires a NumericEventTemplate."
        )
    if isinstance(spec, ArraySpec):
        return prod(spec.shape) if spec.shape else 1
    if isinstance(spec, OpaqueSpec):
        raise TypeError(
            "opaque template fields have no vector size; vector (de)serialization "
            "is only defined for numeric-leaf (ArraySpec) fields."
        )
    raise TypeError(
        f"non-array template field ({type(spec).__name__}) has no vector size; "
        f"vector (de)serialization is only defined for numeric-leaf (ArraySpec) fields."
    )


def _value_treedef(
    template: EventTemplate,
    batch_shape: tuple[int, ...],
    *,
    numeric: bool,
) -> jax.tree_util.PyTreeDef:
    """PyTreeDef of the value :meth:`EventTemplate.from_vector` reconstructs.

    Builds a throwaway skeleton that mirrors the structure ``from_vector``
    produces — ``NumericRecord`` / ``NumericRecordArray`` for a *numeric*
    template, plain ``Record`` / ``RecordArray`` for a mixed one — and returns
    its ``jax.tree_util.tree_structure``. The treedef depends only on the
    container structure (field names, nesting, ``batch_shape``, template), not
    on leaf values, so its placeholder leaves are cheap: zero-stride broadcast
    arrays for numeric leaves (no allocation) and a scalar sentinel for the
    opaque leaves of a mixed template. The returned treedef pairs with the real
    ordered leaves in :func:`jax.tree_util.tree_unflatten`, letting
    ``from_vector`` delegate leaf reconstruction to ``Record.unflatten``.
    """
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray, RecordArray
    from .record import Record

    single_cls = NumericRecord if numeric else Record
    batched_cls = NumericRecordArray if numeric else RecordArray
    numeric_fill = jnp.zeros((), dtype=jnp.float32)

    def _build(tpl: EventTemplate) -> Record:
        fields: dict[str, Any] = {}
        for name in tpl.fields:
            spec = tpl[name]
            if isinstance(spec, EventTemplate):
                fields[name] = _build(spec)
            elif isinstance(spec, ArraySpec):
                fields[name] = jnp.broadcast_to(numeric_fill, (*batch_shape, *spec.shape))
            else:
                # Opaque / distribution / function leaf (mixed templates only) —
                # any object is a valid pytree leaf; the value is irrelevant to
                # the treedef.
                fields[name] = 0
        if batch_shape:
            return batched_cls(fields, batch_shape=batch_shape, template=tpl)
        return single_cls(fields)

    return jax.tree_util.tree_structure(_build(template))


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------


def _unpickle_event_template(specs: dict) -> EventTemplate:
    return EventTemplate(specs)
