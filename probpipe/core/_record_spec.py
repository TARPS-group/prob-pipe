"""Record kind schemas, numeric refinement, and shared-scope unification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Self, cast

from ._immutable import Immutable
from ._spec_base import (
    NumericArraySpec,
    NumericSpec,
    OpaqueSpec,
    TermSpec,
    _full_array_shape_or_none,
    _require_hashable,
    _unify_specs,
)
from .named_tree import _PATH_SEP, NamedTree, _check_no_path_sep, _unflatten_paths


def _reshaped_template(
    template: RecordSpec,
    reshape: Callable[[tuple[int | str, ...]], tuple[int | str, ...]],
) -> RecordSpec:
    """*template* with every array field's shape mapped through *reshape*.

    The template-to-template counterpart of inferring one from data, for the
    operations that move a leading axis: a batch's rows become one element's, an
    empirical's atoms become its stacked samples, a replicate prepends its own.
    Deriving the result from the values instead would answer the shape correctly
    and lose everything else — ``dtype``, ``support``, an ``OpaqueSpec``'s
    ``meta``, a nested field's own kinds are exactly what inference cannot
    recover — so here only the shape moves and the rest rides through.

    A field with no shape to move is carried unchanged, and a nested template is
    reshaped in turn.
    """
    children: dict[str, TermSpec] = {}
    for path, spec in template.children.items():
        if isinstance(spec, RecordSpec):
            children[path] = _reshaped_template(spec, reshape)
        elif isinstance(spec, NumericArraySpec):
            children[path] = NumericArraySpec(
                tuple(reshape(tuple(spec.shape))), dtype=spec.dtype, support=spec.support
            )
        else:
            children[path] = spec
    return RecordSpec(children)


# Constructor inputs also admit nested mappings, shape tuples, and opaque shorthand.
type _FieldSpecInput = TermSpec | Mapping[str, Any] | tuple[int | str, ...] | None


def _to_spec(spec: _FieldSpecInput) -> TermSpec:
    """Normalise a constructor input to a stored field spec.

    Construction-time sugar (preserved): a bare shape ``tuple`` becomes an
    :class:`NumericArraySpec`, ``None`` becomes an :class:`OpaqueSpec`, and a nested
    :class:`RecordSpec` is kept as-is. Already-built specs pass through, so
    new code may supply explicit ``NumericArraySpec(...)`` / ``OpaqueSpec(...)`` etc.
    """
    if isinstance(spec, TermSpec):
        return spec
    if spec is None:
        return OpaqueSpec()
    if isinstance(spec, tuple):
        return NumericArraySpec(shape=spec)
    raise TypeError(f"spec must be a shape tuple, None, or a TermSpec, got {type(spec).__name__}")


def _is_numeric_spec(spec: Any) -> bool:
    """Whether a constructor input denotes a NumericSpec.

    Mappings and record schemas count when every child is numeric; empty
    subtrees have no non-numeric leaves and do not block a numeric parent.
    """
    if isinstance(spec, NumericSpec):
        return True
    if isinstance(spec, RecordSpec):
        return spec.is_numeric
    if isinstance(spec, Mapping):
        return _all_numeric(spec.values())
    return False


def _all_numeric(specs: Iterable[Any]) -> bool:
    """True iff every (raw, pre-normalisation) input spec is numeric.

    Drives the base-class auto-promotion hook so ``RecordSpec(x=(), y=(3,))``
    returns a ``NumericRecordSpec`` without opting in explicitly. Raw inputs
    also allow the shape-tuple sugar; ``None``, non-numeric leaf specs,
    mixed nested templates, and any unsupported type are non-numeric
    (``__init__`` rejects the latter).
    """
    return all(isinstance(s, tuple) or _is_numeric_spec(s) for s in specs)


class RecordSpec(NamedTree[TermSpec], Immutable, TermSpec):
    """Structural description of a value: its named, possibly-nested leaf structure.

    A ``RecordSpec`` describes the **structure** of a value as a **named
    tree** — an insertion-ordered map of named fields whose only internal node
    is a nested ``RecordSpec`` and whose leaves are value specs. It is the
    schema of a :class:`~probpipe.Record` (the value type with the same
    named-tree shape), **not** a description of an arbitrary JAX PyTree (see
    *Terminology* and *JAX pytree contract* below).

    The word *event* follows probabilistic-programming usage and **generalizes**
    the ``event`` / ``event_shape`` notion from other PPLs (TensorFlow
    Probability, distrax, NumPyro). There, ``event_shape`` is the shape of a
    single draw of one array-valued random variable. ProbPipe supports
    distributions over general value types, not just arrays. The *event* in this
    context can thus be a structured Python object, with structure described by
    the ``RecordSpec``.

    Terminology
    -----------
    Used precisely throughout this class:

    - **field** — one named object in the collection (here, a value spec),
      addressed by its full ``/``-delimited **key** (path from the root, e.g.
      ``"physics/mass"``; a single name for a flat template). The mapping
      protocol (:meth:`keys` / :meth:`values` / :meth:`items` / iteration /
      ``len`` / ``in`` / ``[]``) ranges over the fields, keyed by path.
    - **leaf** — a *terminal* node: a :class:`TermSpec`. A nested
      ``RecordSpec`` is an *internal node*, not a leaf; the fields are the
      leaves.
    - **key vs. path** — a **key** addresses a field (a leaf); a **path** may
      also address an interior node. The mapping operators (``[]`` / ``in`` /
      iteration) are leaf-keyed, so a partial path is *not* a member and
      ``template["physics"]`` (a subtree) raises ``KeyError`` — reach a subtree
      with :meth:`at_path`, and use :attr:`children` for the one-level view. The
      same path strings index a template or the value it describes
      (``template["physics/mass"]`` / ``record["physics/mass"]``); this
      collection protocol is shared with :class:`~probpipe.Record`.
    - **canonical leaf order** — the order in which leaves are traversed:
      depth-first, following each level's insertion order. This is the single
      ordering every leaf-wise operation uses. :meth:`keys` is its canonical
      definition — it returns the key (path) of every leaf in this order;
      the value-level ``to_vector`` and :meth:`from_vector` lay out and read
      leaves in it, and
      :attr:`~NumericRecordSpec.leaf_shapes` is keyed by it.

    JAX pytree contract
    -------------------
    A ``RecordSpec`` is **not** a registered JAX pytree node — its value specs
    are atomic, so ``jax.tree_util.tree_leaves(template) == [template]``. It is
    the *schema* of the value pytrees it describes, not a pytree itself (think of
    it as an enriched ``PyTreeDef`` that also carries each leaf's kind / shape).

    For a value ``v`` it describes (a :class:`~probpipe.Record`): a nested
    ``RecordSpec`` mirrors a nested ``Record`` (both internal nodes), and each
    value spec mirrors one field value. When every leaf is an array (the
    :class:`NumericRecordSpec` / :class:`~probpipe.NumericRecord` case),
    ``jax.tree_util.tree_leaves(v)`` returns the leaves in :meth:`keys`
    order. The one place the template's leaves and JAX's diverge is an
    :class:`OpaqueSpec` leaf whose value is *itself* a JAX container (a ``tuple``
    / ``list``; a ``dict`` is never a leaf — mappings denote tree structure):
    the template counts it as a single leaf while JAX descends into it. See
    :class:`~probpipe.Record` for the full statement.

    Parameters
    ----------
    **field_specs
        Fields with non-empty names. Each value is one of:

        - ``tuple[int | str, ...]`` — fixed or symbolic shape of a numeric array
          leaf (e.g. ``()`` for a scalar, ``(3,)`` for a 3-vector, or
          ``("obs", 3)``); normalised to :class:`NumericArraySpec`.
        - ``None`` — opaque (non-array) leaf; normalised to :class:`OpaqueSpec`.
        - a :class:`TermSpec` — an already-built spec (passed through).
        - ``RecordSpec`` — a nested sub-structure (an internal node).

    Examples
    --------
    ::

        RecordSpec(x=(), y=(3,))                     # -> NumericRecordSpec
        RecordSpec(label=None, x=())                 # -> RecordSpec (mixed)
        RecordSpec(physics=RecordSpec(force=(), mass=()), obs=())

    Notes
    -----
    Inspired by JAX's ``PyTreeDef``: a template can reconstruct a value from its
    leaves and describes the expected structure for type-checking and
    vectorization. Leaves are stored as frozen, hashable spec objects, so a
    template is itself hashable (usable as a jit / treedef cache key).
    ``__getitem__`` returns the stored value spec (and raises on an interior
    node — see *Terminology*); the enumeration of leaves is :meth:`keys`, and
    per-leaf array shapes (on a numeric template) live on
    :attr:`~NumericRecordSpec.leaf_shapes`.

    Symbolic array dimensions make a template polymorphic. :attr:`free_dims`
    returns their names and :attr:`is_concrete` is true only when all dimensions
    are fixed. Function invocation binds symbols in a call-local scope rather
    than mutating the declaration.

    Calling ``RecordSpec(...)`` directly auto-promotes to a
    :class:`NumericRecordSpec` when every spec is numeric (and every nested
    sub-template is itself all-numeric), so :attr:`vector_size` and
    :attr:`~NumericRecordSpec.leaf_shapes` are reachable in the common all-numeric case
    without naming the subclass. Mixed templates (any opaque / ``None`` spec)
    stay plain ``RecordSpec`` and do not expose :attr:`vector_size` — it is
    not a meaningful quantity once opaque leaves are present.
    Empty subtrees in a numeric parent are normalized to ``NumericRecordSpec``
    and contribute zero coordinates. A standalone ``RecordSpec()`` stays plain.
    """

    __slots__ = ("_tree",)

    def __new__(
        cls,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ) -> Self:
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericRecordSpec(...)`` calls bypass this path
        # and run their own strict validation.
        if _field_specs is not None and not isinstance(_field_specs, (Mapping, RecordSpec)):
            raise TypeError("RecordSpec fields must be a mapping or a RecordSpec")
        if cls is RecordSpec:
            specs = _field_specs if _field_specs is not None else field_specs
            if isinstance(specs, RecordSpec):
                specs = specs.children
            # No emptiness guard: an empty schema has no non-numeric leaf and
            # lays out flat at length zero, so it is numeric like any other
            # all-numeric schema. Withholding the class here is what let a
            # schema disagree with its own ``is_numeric``, and with the class
            # ``Record`` picks for the matching value.
            if _all_numeric(specs.values()):
                return cast(Self, object.__new__(NumericRecordSpec))
        return object.__new__(cls)

    def __init__(
        self,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        if _field_specs is not None:
            if field_specs:
                raise ValueError("Cannot pass both positional dict and keyword arguments")
            nested = (
                dict(_field_specs.children)
                if isinstance(_field_specs, RecordSpec)
                else _unflatten_paths(_field_specs)
            )
        else:
            for name in field_specs:
                # Same rule as the positional path form enforces in
                # ``_unflatten_paths``; the keyword form reaches neither it
                # nor its message.
                if not name:
                    raise ValueError("field key must be a non-empty string")
                _check_no_path_sep(name)
            nested = dict(field_specs)
        specs: dict[str, TermSpec] = {}
        for name, spec in nested.items():
            if isinstance(spec, RecordSpec):
                converted = spec
            elif isinstance(spec, Mapping):
                # A mapping spec is nested structure: materialise a subtree.
                converted = RecordSpec(spec)
            else:
                try:
                    converted = _to_spec(spec)
                except TypeError as exc:
                    raise TypeError(f"Field {name!r}: {exc}") from None
                if not isinstance(converted, RecordSpec):
                    self._check_leaf(name, converted)
            if (
                isinstance(self, NumericRecordSpec)
                and isinstance(converted, RecordSpec)
                and not isinstance(converted, NumericRecordSpec)
                and converted.is_numeric
            ):
                converted = NumericRecordSpec(converted)
            _require_hashable(converted, context=f"Field {name!r} spec")
            specs[name] = converted
        self._post_validate(specs)
        object.__setattr__(self, "_tree", specs)

    def _post_validate(self, field_specs: dict[str, TermSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Tree structure -----------------------------------------------------
    #
    # The mapping and path-navigation methods (``keys`` / ``values`` /
    # ``items`` / ``[]`` / ``at_path`` / ``children``) are inherited from
    # :class:`~probpipe.core.named_tree.NamedTree`. A leaf here is a
    # :class:`TermSpec`; an internal node is a nested ``RecordSpec``.

    @classmethod
    def _node_type(cls) -> type:
        return RecordSpec

    @classmethod
    def _leaf_type(cls) -> type:
        # Every leaf of a template is a value spec; construction converts
        # the shorthand forms (shapes, None, ...) via ``_to_spec`` first,
        # so the substrate check validates the converted leaf.
        return TermSpec

    @classmethod
    def _rebuild_class(cls) -> type:
        # Structural edits rebuild through the base class so ``__new__``
        # re-decides the numeric auto-promotion from the edited specs: an
        # all-numeric result promotes to ``NumericRecordSpec`` and a mixed
        # one stays (or becomes) a plain ``RecordSpec`` — replacing an array
        # spec with an opaque one must not be rejected by the original
        # subclass's validation.
        return RecordSpec

    # -- Numeric queries & projection ---------------------------------------

    @property
    def is_numeric(self) -> bool:
        """Whether every reachable leaf implements NumericSpec.

        Nested record schemas are checked recursively. Opaque, distribution,
        and callable leaves are non-numeric unless their spec implements the
        numeric mixin. The empty schema has no non-numeric leaves.
        """
        for spec in self._tree.values():
            if isinstance(spec, NumericSpec):
                continue
            if isinstance(spec, RecordSpec):
                if not spec.is_numeric:
                    return False
                continue
            # Opaque / record / distribution / function leaf — not numeric.
            return False
        return True

    @property
    def free_dims(self) -> frozenset[str]:
        """Symbolic dimension names declared anywhere in this template."""
        dimensions: set[str] = set()
        for spec in self._tree.values():
            dimensions.update(spec.free_dims)
        return frozenset(dimensions)

    @property
    def is_concrete(self) -> bool:
        """Whether every dimension this template declares has a fixed size."""
        return not self.free_dims

    def _substitute_dims(self, bindings: Mapping[str, int | str]) -> RecordSpec:
        """Substitute dimensions throughout the schema in one shared scope."""
        return type(self)(
            {name: spec._substitute_dims(bindings) for name, spec in self._tree.items()}
        )

    def _bind_dims_from_value(self, value: Any, bindings: dict[str, int], path: str) -> None:
        from .record import Record

        if isinstance(value, Record):
            children = value.children
        elif isinstance(value, Mapping):
            children = value
        else:
            raise ValueError(
                f"{path} does not match its RecordSpec: expected named fields, "
                f"got {type(value).__name__}"
            )
        if self._tree.keys() != children.keys():
            raise ValueError(
                f"{path} fields {sorted(children)} do not match template fields "
                f"{sorted(self._tree)}"
            )
        for name, spec in self._tree.items():
            child_path = f"{path}{_PATH_SEP}{name}" if path else name
            spec._bind_dims_from_value(children[name], bindings, child_path)

    def _bind_dims_from_spec(self, actual: TermSpec, bindings: dict[str, int], path: str) -> bool:
        if not isinstance(actual, RecordSpec):
            return False
        if self._tree.keys() != actual._tree.keys():
            raise ValueError(
                f"{path} fields {sorted(actual._tree)} do not match template fields "
                f"{sorted(self._tree)}"
            )
        for name, spec in self._tree.items():
            child_path = f"{path}{_PATH_SEP}{name}" if path else name
            _unify_specs(spec, actual._tree[name], bindings, child_path)
        return True

    def is_valid(self, value: Any) -> bool:
        """Whether a record or raw mapping satisfies the complete schema.

        Fields must match exactly. Nested structure, kinds, shapes and declared
        dtypes are checked, including agreement of repeated symbolic sizes.
        Structural mismatches return False; unexpected inspection errors propagate.
        """
        from .record import Record

        if not isinstance(value, (Record, Mapping)):
            return False
        try:
            self._bind_dims_from_value(value, {}, type(self).__name__)
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    def numeric_subset(self) -> NumericRecordSpec:
        """Project to the :class:`NumericSpec`-leaf sub-template.

        Keeps every numeric leaf and drops non-numeric leaves and empty
        subtrees. Surviving leaves retain their metadata, canonical order and
        ``/``-delimited paths.

        The projection is idempotent. An all-numeric template without empty
        subtrees produces an equal :class:`NumericRecordSpec`.

        Returns
        -------
        NumericRecordSpec
            The numeric-leaf sub-template, so that :attr:`vector_size` and
            :attr:`~NumericRecordSpec.leaf_shapes` are available.

        Raises
        ------
        ValueError
            If no numeric leaves survive. The message names the dropped fields.
        """
        specs = {path: spec for path, spec in self._walk_leaves() if isinstance(spec, NumericSpec)}
        if not specs:
            raise ValueError(
                f"numeric_subset() of {type(self).__name__} is empty: no "
                f"NumericSpec leaves survive. Dropped fields: {tuple(self._tree)}."
            )
        return NumericRecordSpec(specs)

    # -- Equality and hashing -----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordSpec):
            return NotImplemented
        # Order-sensitive comparison so equality matches the
        # order-sensitive ``__hash__`` (insertion order is part of the
        # template's identity). dict.__eq__ alone would ignore order,
        # breaking the eq/hash contract.
        return tuple(self._tree.items()) == tuple(other._tree.items())

    def __hash__(self) -> int:
        # All field specs (value specs and nested templates) are hashable, so
        # the order-sensitive item tuple hashes directly. Insertion order is
        # part of the template's identity.
        return hash(tuple(self._tree.items()))

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def infer_from(cls, value: Any) -> RecordSpec:
        """Best-effort, **lossy** schema inferred by inspecting a value.

        Two cases:

        - A :class:`~probpipe.Record` already carries its authoritative schema,
          so ``infer_from`` returns its :attr:`~probpipe.Record.event_template`
          unchanged.
        - A **mapping** of named fields (e.g. a ``Record``'s field dict) is
          inferred field by field: a nested ``Record`` field contributes its
          own schema; other tracked fields, distributions included, retain
          their specs. Callables become function specs;
          a numeric array or scalar becomes a :class:`NumericArraySpec` of its
          shape, and remaining raw values become :class:`OpaqueSpec`. The result
          auto-promotes to a :class:`NumericRecordSpec` when every field is numeric.

        This is the **fallback** for wrapping a raw value that has no template
        yet (e.g. at a workflow boundary); for a value you already hold, read
        its authoritative ``event_template`` directly. Inference is lossy — it
        cannot recover an untracked array's declared ``dtype`` / ``support``
        constraints or opaque metadata. A tracked field keeps its own spec.
        A Python ``list`` /
        ``tuple`` leaf (no ``.shape`` / ``.dtype``) is treated as opaque even if
        it holds numbers; wrap it in
        ``np.asarray`` / ``jnp.asarray`` first for a numeric leaf.

        Parameters
        ----------
        value : Any
            A :class:`~probpipe.Record`, or a mapping of field name to value
            (arrays / scalars / nested ``Record``\\ s).

        Returns
        -------
        RecordSpec
            The inferred schema (a :class:`NumericRecordSpec` when every
            field is numeric).

        Raises
        ------
        TypeError
            If *value* is neither a ``Record`` nor a mapping.
        """
        from .record import Record

        if isinstance(value, Record):
            return value.event_template
        if not isinstance(value, Mapping):
            raise TypeError(
                f"infer_from expects a Record or a mapping of fields, got {type(value).__name__}."
            )

        def _leaf_spec(val: Any) -> _FieldSpecInput:
            from ._kind_specs import FunctionSpec
            from .tracked import TrackedTerm

            if isinstance(val, TrackedTerm):
                spec = getattr(val, "spec", None)
                if isinstance(spec, TermSpec):
                    return spec
            if isinstance(val, Record):
                return val.event_template
            if callable(val):
                if isinstance(val, TrackedTerm):
                    return FunctionSpec(
                        getattr(val, "input_template", None), getattr(val, "output_template", None)
                    )
                return FunctionSpec()
            # A mapping is never a leaf: it denotes tree structure, so infer a
            # nested template from it rather than an (invalid) opaque-leaf spec
            # — matching the constructor and the workflow-output wrap.
            if isinstance(val, Mapping):
                return cls.infer_from(val)
            # Any numeric array-like — bare arrays and native containers
            # (xarray / pandas / registered backends) alike — infers an
            # ``NumericArraySpec``; leaves are stored in native form, so nothing is
            # lost by classing them numeric.
            return _full_array_shape_or_none(val)

        specs: dict[str, _FieldSpecInput] = {name: _leaf_spec(val) for name, val in value.items()}
        return RecordSpec(specs)

    # -- Repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, spec in self._tree.items():
            if isinstance(spec, RecordSpec):
                parts.append(f"{name}={spec!r}")
            elif isinstance(spec, NumericArraySpec) and spec.dtype is None and spec.support is None:
                # Bare specs render as their sugar form (shape tuple / None).
                parts.append(f"{name}={spec.shape}")
            elif isinstance(spec, OpaqueSpec) and spec.meta is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}={spec!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# NumericRecordSpec — all-numeric specialisation
# ---------------------------------------------------------------------------


class NumericRecordSpec(RecordSpec, NumericSpec):
    """RecordSpec where every leaf is numeric.

    Extends :class:`RecordSpec` by requiring every leaf to implement
    :class:`NumericSpec`. Shape tuples remain shorthand for array specs,
    and nested numeric records retain their structure. That restriction is what makes
    :attr:`vector_size` and :attr:`leaf_shapes` meaningful:
    ``vector_size`` is the length of the per-element 1-D vector — the total
    number of scalar elements across every numeric leaf — and
    :meth:`~probpipe.NumericRecord.from_vector` takes a template of this class
    so that every field can be reconstructed from a slice of that vector. A
    *batch* of such values is a matrix of shape ``(*batch_shape, vector_size)``,
    not a single vector.

    Use :meth:`RecordSpec.infer_from` on a :class:`NumericRecord`
    (it auto-promotes) or call this constructor directly when you have
    the shape specs in hand.
    """

    __slots__ = ("_cached_vector_size",)

    _cached_vector_size: int | None

    def _post_validate(self, field_specs: dict[str, TermSpec]) -> None:
        for name, spec in field_specs.items():
            if isinstance(spec, NumericSpec):
                continue
            if isinstance(spec, RecordSpec):
                raise TypeError(
                    f"NumericRecordSpec: nested field {name!r} is a "
                    f"{type(spec).__name__}; nested sub-templates must "
                    f"themselves be NumericRecordSpec."
                )
            # Any non-numeric leaf — OpaqueSpec, DistributionSpec, or FunctionSpec.
            raise TypeError(
                f"NumericRecordSpec: field {name!r} is a {type(spec).__name__}; "
                f"only NumericArraySpec or other NumericSpec leaves (including nested numeric records) are "
                f"allowed — use RecordSpec if you need a mixed template."
            )

    def __init__(
        self,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        super().__init__(_field_specs, **field_specs)
        size = None if self.free_dims else sum(spec.vector_size for spec in self._tree.values())
        object.__setattr__(self, "_cached_vector_size", size)

    @property
    def leaf_shapes(self) -> dict[str, tuple[int | str, ...]]:
        """Per-leaf array shapes, keyed by :meth:`keys` (canonical leaf order).

        Array leaves retain their declared shape, including symbolic entries.
        Nested records contribute one entry per leaf. Other numeric kinds
        contribute their flat coordinate shape, ``(vector_size,)``; they must
        be concrete for that size to be available. General mixed schemas have
        no numeric layout and expose only their structural keys.
        """
        result: dict[str, tuple[int | str, ...]] = {}
        for name, spec in self._tree.items():
            if isinstance(spec, NumericRecordSpec):
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}{_PATH_SEP}{sub_name}"] = sub_shape
            elif isinstance(spec, NumericArraySpec):
                result[name] = spec.shape
            else:
                result[name] = (spec.vector_size,)
        return result

    @property
    def is_concrete(self) -> bool:
        """Whether no symbolic dimension remains, as :class:`TermSpec` defines it.

        Read from the layout cached at construction, which is populated
        exactly when every dimension was concrete.
        """
        return self._cached_vector_size is not None

    def _vector_size(self) -> int:
        """The cached scalar count for this concrete record schema."""
        return cast(int, self._cached_vector_size)

    # 1-D numeric (de)serialization is a value operation and lives on the
    # value types: ``to_vector`` on :class:`~probpipe.NumericRecord` /
    # :class:`~probpipe.NumericRecordBatch`, and their ``from_vector``
    # classmethods (which take a template). A template describes structure
    # and does not depend on the value type, so it carries neither.


# ---------------------------------------------------------------------------
# Private symbolic-dimension unification
# ---------------------------------------------------------------------------


def _unify_record_spec_with_value(
    template: RecordSpec,
    value: Any,
    bindings: Mapping[str, int] | None = None,
    *,
    context: str = "value",
) -> tuple[RecordSpec, dict[str, int]]:
    """Return a refined copy of *template* unified with a concrete value.

    The declaration and the optional input bindings are never mutated. The
    returned binding dictionary can be threaded through several calls to give
    inputs and outputs one invocation-local symbolic-dimension scope.
    """
    resolved = dict(bindings or {})
    template._bind_dims_from_value(value, resolved, context)
    return template._substitute_dims(resolved), resolved


def _concretize_record_spec(
    template: RecordSpec,
    bindings: Mapping[str, int],
    *,
    context: str = "template",
) -> RecordSpec:
    """Substitute all symbolic dimensions, failing if any remain unbound."""
    missing = template.free_dims.difference(bindings)
    if missing:
        dimensions = ", ".join(sorted(missing))
        raise ValueError(f"{context} has unbound symbolic dimensions: {dimensions}")
    return template._substitute_dims(bindings)


def _check_kind_of(around: TermSpec, value: Any, spec: TermSpec, path: str) -> None:
    """Refuse *value* unless it satisfies *spec*'s kind.

    Only the sizes are deferred to the pass, so a value of the wrong kind is
    refused here as it would be for a concrete declaration. *around* is the spec
    rebuilt around the schema the value actually carries, which leaves the kind as
    the only thing its own ``is_valid`` still tests.
    """
    if not around.is_valid(value):
        raise ValueError(f"{path} does not conform to its field spec ({spec!r})")
