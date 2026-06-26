"""EventTemplate — ProbPipe's structural schema.

An :class:`EventTemplate` describes the **structure** of a value, independent of
the data itself. The structure is a **named tree** — an insertion-ordered map of
named fields — and it is exactly the shape of a :class:`~probpipe.Record`: an
``EventTemplate`` is the *schema* of a ``Record`` (or of one event / sample of a
:class:`~probpipe.core._distribution_base.Distribution`).

It is **not** a description of an arbitrary JAX PyTree. The only internal node is
a nested ``EventTemplate`` (named); positional containers (``tuple`` / ``list``)
and bare ``dict``\\ s are never structure. A Python container you do not model as
a nested template is a single opaque leaf (an :class:`OpaqueSpec`).

Tree structure
--------------
- **internal node** — *only* a nested ``EventTemplate`` (named, insertion-ordered).
- **leaf** — one of a fixed set of "spec" objects:

| Leaf spec | What the leaf describes |
|---|---|
| :class:`ArraySpec` | a numeric array (event ``shape``, optional dtype / support) |
| :class:`DistributionSpec` | a ``Distribution`` (carries the draw's sub-template) |
| :class:`FunctionSpec` | a callable (carries input / output sub-templates) |
| :class:`OpaqueSpec` | any other object — no structure assumed |

Field names are required and unique within a node; ``/`` is reserved as the path
separator, so every leaf has a unique ``/``-delimited string path. The
**canonical leaf order** is depth-first in insertion order;
:attr:`EventTemplate.leaf_paths` enumerates the leaf paths in that order.
``EventTemplate`` itself is **not** a registered JAX pytree node (its leaf specs
are atomic); it is the *schema* of the value pytrees it describes, not a pytree.

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

from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..custom_types import Array, ArrayLike
from .constraints import Constraint

if TYPE_CHECKING:
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray

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
# imports these from here so the constant has a single home. Docstrings spell the
# separator literally as ``/``; if it ever changes, update them (a guard test
# pins this value so the change can't pass silently).
_PATH_SEP = "/"


def _check_no_path_sep(name: str) -> None:
    if _PATH_SEP in name:
        raise ValueError(
            f"Field name {name!r} must not contain {_PATH_SEP!r} "
            f"(reserved as the nested-path separator)."
        )


class _PathSubtree(dict):
    """Marker for a nested mapping produced by path-key unflattening.

    A plain ``dict`` *value* is opaque leaf data and is never treated as
    structure; a ``_PathSubtree`` is the structural nesting that ``/``-delimited
    keys denote, so a constructor materialises it into a child collection while
    leaving plain-``dict`` values alone.
    """


def _unflatten_paths(source: Mapping[str, Any]) -> dict[str, Any]:
    """Explode a path-keyed mapping into a nested ordered mapping.

    Each key is a field key that may be *path-shaped* — a ``/``-delimited string
    denotes nesting (``{"a/b": x}`` becomes ``{"a": {"b": x}}``). Keys are read
    in order; the result preserves the **first-appearance order** of each
    distinct leading segment (recursively), which is the canonical field order.
    Structural nesting is tagged with :class:`_PathSubtree` so a constructor can
    tell it apart from a plain-``dict`` leaf value.

    Raises
    ------
    TypeError
        If a key is not a string.
    ValueError
        If a key is empty or has an empty segment (``""``, ``"a/"``, ``"/a"``,
        ``"a//b"``), or if one leading segment is used **both** as a complete key
        and as the prefix of a longer key (the field-versus-prefix collision).
    """
    groups: dict[str, Any] = {}
    is_prefix: dict[str, bool] = {}
    for key, value in source.items():
        if not isinstance(key, str):
            raise TypeError(f"field key must be a string, got {type(key).__name__}")
        if not key:
            raise ValueError("field key must be a non-empty string")
        segments = key.split(_PATH_SEP)
        if any(seg == "" for seg in segments):
            raise ValueError(
                f"malformed path key {key!r}: empty path segment "
                f"(no leading/trailing/doubled {_PATH_SEP!r})"
            )
        head, rest = segments[0], _PATH_SEP.join(segments[1:])
        if rest:
            if head in groups and not is_prefix[head]:
                raise ValueError(f"name {head!r} is used both as a field and as a path prefix")
            is_prefix[head] = True
            groups.setdefault(head, {})[rest] = value
        else:
            if head in groups:
                raise ValueError(
                    f"name {head!r} is used both as a field and as a path prefix"
                    if is_prefix.get(head)
                    else f"duplicate field key {head!r}"
                )
            groups[head] = value
            is_prefix[head] = False
    result: dict[str, Any] = {}
    for head, val in groups.items():
        result[head] = _PathSubtree(_unflatten_paths(val)) if is_prefix[head] else val
    return result


class _NamedTree:
    """Shared structural-access protocol for ProbPipe's named trees.

    Both :class:`EventTemplate` (a tree of specs) and
    :class:`~probpipe.Record` (a tree of values) are insertion-ordered maps of
    **fields** — top-level ``name -> value`` entries. A field's **value** is
    either a terminal **leaf** or a nested tree of the same kind (an **internal
    node**), addressed by a ``/``-delimited **path**. This mixin gives both the
    same way to name, address, enumerate, and iterate that structure;
    subclasses differ only in what their field values are (specs vs. data).

    A subclass supplies two hooks: :meth:`_field_map` (its ordered
    ``name -> value`` mapping) and :meth:`_node_type` (the type whose instances
    count as internal nodes). A field value is an internal node iff it is an
    instance of that type; everything else is a leaf.
    """

    __slots__ = ()

    def _field_map(self) -> Mapping[str, Any]:
        """The ordered ``field name -> field value`` mapping for this node (hook)."""
        raise NotImplementedError

    @classmethod
    def _node_type(cls) -> type:
        """The type whose instances are internal nodes of this tree (hook).

        A field value is an internal node iff it is an instance of this type;
        every other value is a leaf. Each named-tree family **narrows** this to
        itself (``Record`` → ``Record``, ``EventTemplate`` → ``EventTemplate``)
        so a tree never descends into a value of the *other* family. This keeps
        :attr:`leaf_paths` consistent with each type's own contract — in
        particular, a ``Record``'s ``leaf_paths`` always equals its
        ``event_template``'s ``leaf_paths``.
        """
        return _NamedTree

    @property
    def fields(self) -> tuple[str, ...]:
        """Top-level field names, in insertion order (does not descend)."""
        return tuple(self._field_map().keys())

    def __len__(self) -> int:
        return len(self._field_map())

    def __iter__(self) -> Iterator[str]:
        return iter(self._field_map())

    def keys(self) -> Iterator[str]:
        """Iterate top-level field names."""
        return iter(self._field_map())

    def values(self) -> Iterator[Any]:
        """Iterate top-level field values (one per field)."""
        return iter(self._field_map().values())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate ``(name, value)`` pairs for the top-level fields."""
        return iter(self._field_map().items())

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        """Return the field value at *key* — a field name, ``/``-path, or tuple path.

        A plain name selects a top-level field; a ``/``-delimited string (or a
        tuple of names) descends through internal nodes (``tree["a/b"]`` ==
        ``tree["a", "b"]``). Raises ``KeyError`` for a missing field or a path
        that descends through a leaf, and ``TypeError`` for a non-str/tuple key.
        """
        if isinstance(key, str):
            if _PATH_SEP in key:
                return self[tuple(key.split(_PATH_SEP))]
            field_map = self._field_map()
            if key not in field_map:
                raise KeyError(key)
            return field_map[key]
        if isinstance(key, tuple):
            node: Any = self
            node_type = self._node_type()
            for i, name in enumerate(key):
                if i > 0 and not isinstance(node, node_type):
                    raise KeyError(
                        f"path {_PATH_SEP.join(key)!r} descends through non-tree "
                        f"leaf {type(node).__name__} at {_PATH_SEP.join(key[:i])!r}"
                    )
                node = node[name]
            return node
        raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")

    def __contains__(self, key: object) -> bool:
        """Membership by field name or ``/``-path (missing paths are ``False``)."""
        if isinstance(key, str) and _PATH_SEP not in key:
            return key in self._field_map()
        try:
            self[key]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            return False
        return True

    @property
    def leaf_paths(self) -> tuple[str, ...]:
        """``/``-path to every leaf, in canonical leaf order.

        Canonical leaf order is the depth-first, insertion-order traversal:
        each internal node expands into one path per nested leaf; a flat tree's
        ``leaf_paths`` equals its :attr:`fields`. This is the single ordering
        every leaf-wise operation uses.
        """
        paths: list[str] = []
        node_type = self._node_type()
        for name, value in self._field_map().items():
            if isinstance(value, node_type):
                paths.extend(f"{name}{_PATH_SEP}{sub}" for sub in value.leaf_paths)
            else:
                paths.append(name)
        return tuple(paths)

    # -- Collection-of-objects surface --------------------------------------
    #
    # A collection is a named, ordered set of *fields* whose objects live at the
    # leaves of the storage tree, each addressed by its full path. The methods
    # below present that view: ``at_path`` navigates the whole tree (to a leaf or
    # an interior subtree), ``children`` is the one-level view, and the leaf walk
    # / rebuild primitives back the field-wise operations. They read the storage
    # tree through the ``_field_map`` / ``_node_type`` hooks, so a subclass need
    # not re-implement them.

    @staticmethod
    def _split_path(path: tuple[Any, ...]) -> tuple[str, ...]:
        """Normalise a path argument to a tuple of single-name segments.

        Accepts the forms used across the surface: a ``/``-delimited string
        (``"a/b"``), separate string segments (``"a", "b"``), or a single tuple
        of names (``("a", "b")``). Every segment must be a string; a non-string
        segment raises ``TypeError``.
        """
        if len(path) == 1 and isinstance(path[0], tuple):
            parts: tuple[Any, ...] = path[0]
        else:
            parts = path
        segments: list[str] = []
        for part in parts:
            if not isinstance(part, str):
                raise TypeError(f"path segments must be strings, got {type(part).__name__}")
            segments.extend(part.split(_PATH_SEP) if _PATH_SEP in part else [part])
        return tuple(segments)

    def at_path(self, *path: Any) -> Any:
        """Return the object at *path* — a field object, or an interior subtree.

        *path* addresses a position in the storage tree and may be written as a
        ``/``-delimited string, separate string segments, or a single tuple of
        names — these are equivalent::

            obj.at_path("physics/mass")
            obj.at_path("physics", "mass")
            obj.at_path(("physics", "mass"))

        A *key* (a path that ends at a leaf) returns the field object — a value
        for a :class:`~probpipe.Record`, a leaf spec for an ``EventTemplate``. A
        *partial path* (one that stops at an interior node) returns the subtree
        rooted there, a collection of the same class.

        This is the one operator that reaches interior nodes; the mapping
        operators (``[]`` / ``in`` / iteration) range only over fields.

        Raises
        ------
        KeyError
            If the path reaches nothing, or tries to descend through a leaf.
        TypeError
            If a path segment is not a string.
        """
        segments = self._split_path(path)
        node: Any = self
        node_type = self._node_type()
        for i, name in enumerate(segments):
            if i > 0 and not isinstance(node, node_type):
                raise KeyError(
                    f"path {_PATH_SEP.join(segments)!r} descends through non-tree "
                    f"leaf {type(node).__name__} at {_PATH_SEP.join(segments[:i])!r}"
                )
            field_map = node._field_map()
            if name not in field_map:
                raise KeyError(_PATH_SEP.join(segments))
            node = field_map[name]
        return node

    @property
    def children(self) -> Mapping[str, Any]:
        """Read-only one-level view of this node — ``local_name -> child``.

        Each value is an immediate child: a field object (leaf) or a subtree (a
        nested collection of the same class). Insertion-ordered. This is the
        labelled top-level view; the top-level names are ``tuple(obj.children)``.
        Unlike the leaf-keyed field view, ``children.values()`` includes
        subtrees.
        """
        return MappingProxyType(dict(self._field_map()))

    def is_leaf(self, *path: Any) -> bool:
        """Whether *path* resolves to a field (a leaf), not an interior subtree.

        Returns ``True`` only when *path* is navigable **and** ends at a leaf.
        Accepts the same path forms as :meth:`at_path`.
        """
        try:
            node = self.at_path(*path)
        except (KeyError, TypeError):
            return False
        return not isinstance(node, self._node_type())

    def to_nested_dict(self) -> dict[str, Any]:
        """Return a nested ``dict`` mirroring the storage tree.

        Each interior node becomes a nested ``dict``; each leaf maps to its
        field object (a value for a :class:`~probpipe.Record`, a spec for an
        ``EventTemplate``). This is the tree-shaped export, distinct from the
        flat ``dict(obj)`` view that is keyed by full path.
        """
        node_type = self._node_type()
        result: dict[str, Any] = {}
        for name, child in self._field_map().items():
            result[name] = child.to_nested_dict() if isinstance(child, node_type) else child
        return result

    def _walk_leaves(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(path, leaf_object)`` for every field, in canonical order.

        Canonical order is the depth-first, insertion-order traversal (§ the
        class docstring). This is the single traversal that the field-wise
        operations (the field mapping, ``map``, vector serialization, equality)
        are expressed on.
        """
        node_type = self._node_type()
        for name, child in self._field_map().items():
            if isinstance(child, node_type):
                for sub_path, leaf in child._walk_leaves():
                    yield f"{name}{_PATH_SEP}{sub_path}", leaf
            else:
                yield name, child

    def _rebuild_from_leaves(self, values: Iterable[Any]) -> Any:
        """Build a new collection of the same nested shape, with new leaf objects.

        *values* supplies the field objects in canonical order (one per leaf,
        the order of :meth:`_walk_leaves`); the structure (names and nesting) is
        taken from ``self``. Each interior node is rebuilt by constructing a new
        same-class collection, so a :class:`~probpipe.Record` re-infers its leaf
        specs and an ``EventTemplate`` re-normalises and re-promotes.

        Raises
        ------
        ValueError
            If *values* does not supply exactly one object per leaf.
        """
        it = iter(values)
        node_type = self._node_type()
        sentinel = object()

        def build(node: Any) -> Any:
            new_children: dict[str, Any] = {}
            for name, child in node._field_map().items():
                if isinstance(child, node_type):
                    new_children[name] = build(child)
                else:
                    nxt = next(it, sentinel)
                    if nxt is sentinel:
                        raise ValueError("_rebuild_from_leaves got fewer values than leaves")
                    new_children[name] = nxt
            return type(node)(new_children)

        result = build(self)
        if next(it, sentinel) is not sentinel:
            raise ValueError("_rebuild_from_leaves got more values than leaves")
        return result

    @classmethod
    def from_nested_dict(cls, data: Mapping[str, Any], *, event_template: Any | None = None) -> Any:
        """Build a collection from a **nested** ``dict`` — the inverse of :meth:`to_nested_dict`.

        Each nested ``dict`` level becomes a subtree and every other value a leaf.
        This is the opt-in way to read a nested ``dict`` as structure; the bare
        constructor instead treats a ``dict`` value as an opaque leaf.

        When *event_template* is given (``Record`` only), the template — not the
        Python type — decides structure at each position: a ``dict`` sitting where
        the template has an interior node is recursed into, while a ``dict`` where
        the template has a leaf spec is kept verbatim as opaque data. The template
        is then carried down exactly as in normal construction.

        Parameters
        ----------
        data
            A nested mapping of field names to values (or sub-mappings).
        event_template
            Optional schema deciding leaf-vs-structure and carried onto the result.

        Returns
        -------
        A new collection of this class.
        """
        flat: dict[str, Any] = {}

        def walk(sub: Mapping[str, Any], prefix: str) -> None:
            for name, val in sub.items():
                path = f"{prefix}{_PATH_SEP}{name}" if prefix else name
                structural = isinstance(val, Mapping)
                if structural and event_template is not None:
                    structural = not event_template.is_leaf(path)
                if structural:
                    walk(val, path)
                else:
                    flat[path] = val

        walk(data, "")
        if event_template is not None:
            return cls(flat, event_template=event_template)
        return cls(flat)


@dataclass(frozen=True)
class ArraySpec:
    """A numeric array leaf: a fixed event ``shape`` plus optional metadata.

    ``dtype`` and ``support`` are optional (default ``None``); when unset the
    leaf describes its shape only. Both must be hashable when set.
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
    """A numeric leaf: an :class:`ArraySpec` or a (nested) :class:`NumericEventTemplate`.

    A ``_PathSubtree`` (structural nesting from path-key unflattening, not yet
    materialised) counts as numeric iff all of its own values are — so
    auto-promotion sees through nesting introduced by ``/``-keys.
    """
    if isinstance(spec, _PathSubtree):
        return _all_numeric(spec.values())
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


class EventTemplate(_NamedTree):
    """Structural description of a value: its named, possibly-nested leaf structure.

    An ``EventTemplate`` describes the **structure** of a value as a **named
    tree** — an insertion-ordered map of named fields whose only internal node
    is a nested ``EventTemplate`` and whose leaves are leaf specs. It is the
    schema of a :class:`~probpipe.Record` (the value type with the same
    named-tree shape), **not** a description of an arbitrary JAX PyTree (see
    *Terminology* and *JAX pytree contract* below).

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
      path. The same path strings index a template or the value it describes
      (``template["physics/mass"]`` / ``record["physics/mass"]``) — the
      structural-access protocol (indexing, membership, iteration,
      :attr:`leaf_paths`) is shared with :class:`~probpipe.Record`.
    - **canonical leaf order** — the order in which leaves are traversed:
      depth-first, following each level's insertion order. This is the single
      ordering every leaf-wise operation uses. :attr:`leaf_paths` is its
      canonical definition — it returns the path to every leaf in this order;
      :meth:`to_vector` / :meth:`from_vector` lay out and read leaves in it, and
      :attr:`~NumericEventTemplate.leaf_shapes` is keyed by it.

    JAX pytree contract
    -------------------
    An ``EventTemplate`` is **not** a registered JAX pytree node — its leaf specs
    are atomic, so ``jax.tree_util.tree_leaves(template) == [template]``. It is
    the *schema* of the value pytrees it describes, not a pytree itself (think of
    it as an enriched ``PyTreeDef`` that also carries each leaf's kind / shape).

    For a value ``v`` it describes (a :class:`~probpipe.Record`): a nested
    ``EventTemplate`` mirrors a nested ``Record`` (both internal nodes), and each
    leaf spec mirrors one field value. When every leaf is an array (the
    :class:`NumericEventTemplate` / :class:`~probpipe.NumericRecord` case),
    ``jax.tree_util.tree_leaves(v)`` returns the leaves in :attr:`leaf_paths`
    order. The one place the template's leaves and JAX's diverge is an
    :class:`OpaqueSpec` leaf whose value is *itself* a JAX container (a ``tuple``
    / ``list`` / ``dict``): the template counts it as a single leaf while JAX
    descends into it. See :class:`~probpipe.Record` for the full statement.

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
    ``__getitem__`` returns the stored spec (or nested template); the
    enumeration of leaves is :attr:`leaf_paths`, and per-leaf array shapes (on a
    numeric template) live on :attr:`~NumericEventTemplate.leaf_shapes`.

    Calling ``EventTemplate(...)`` directly auto-promotes to a
    :class:`NumericEventTemplate` when every spec is numeric (and every nested
    sub-template is itself all-numeric), so :attr:`vector_size` and
    :attr:`~NumericEventTemplate.leaf_shapes` are reachable in the common all-numeric case
    without naming the subclass. Mixed templates (any opaque / ``None`` spec)
    stay plain ``EventTemplate`` and do not expose :attr:`vector_size` — it is
    not a meaningful quantity once opaque leaves are present.
    """

    __slots__ = ("_specs",)

    def __new__(
        cls,
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        # Only auto-promote when invoked directly on the base class —
        # explicit ``NumericEventTemplate(...)`` calls bypass this path
        # and run their own strict validation.
        if cls is EventTemplate:
            specs = _field_specs if _field_specs is not None else field_specs
            if specs and _all_numeric(specs.values()):
                return object.__new__(NumericEventTemplate)
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
            nested = _unflatten_paths(_field_specs)
        else:
            for name in field_specs:
                _check_no_path_sep(name)
            nested = dict(field_specs)
        if not nested:
            raise ValueError(f"{type(self).__name__} requires at least one field")
        specs: dict[str, _FieldSpec] = {}
        for name, spec in nested.items():
            if isinstance(spec, _PathSubtree):
                specs[name] = EventTemplate(spec)
            else:
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

    # -- Field access (structural protocol shared with ``Record``) ----------
    #
    # ``fields`` / ``leaf_paths`` / ``__getitem__`` (name / ``/``-path / tuple) /
    # ``__contains__`` / ``__iter__`` / ``keys`` / ``values`` / ``items`` /
    # ``__len__`` come from :class:`_NamedTree`. A leaf here is a leaf spec
    # (:class:`ArraySpec` / :class:`OpaqueSpec` / :class:`DistributionSpec` /
    # :class:`FunctionSpec`); an internal node is a nested ``EventTemplate``.

    def _field_map(self) -> Mapping[str, _FieldSpec]:
        return self._specs

    @classmethod
    def _node_type(cls) -> type:
        return EventTemplate

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
        ``len(self.leaf_paths) > 1``.

        Returns
        -------
        bool
            ``True`` iff the template has more than one leaf; ``False`` iff it
            describes exactly one leaf.
        """
        return len(self.leaf_paths) > 1

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
            :attr:`~NumericEventTemplate.leaf_shapes` are available.

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
            dropped = tuple(
                name
                for name, spec in self._specs.items()
                if not (
                    isinstance(spec, ArraySpec)
                    or (isinstance(spec, EventTemplate) and spec.is_numeric)
                )
            )
            raise ValueError(
                f"numeric_subset() of {type(self).__name__} is empty: no "
                f"ArraySpec leaves survive. Dropped non-numeric fields: {dropped}."
            )
        return NumericEventTemplate(specs)

    # -- Leaf-list (de)serialization (general; leaves kept whole) -----------

    def to_leaf_list(self, value: Any) -> list[Any]:
        """Flatten *value* to its leaves, in canonical leaf order.

        Returns *value*'s leaves — each kept *whole* (any type), one per
        :attr:`leaf_paths` entry — at this template's granularity: a
        container-valued opaque field (a ``tuple`` / ``list`` / ``dict``) is a
        single leaf, **not** descended into. Pairs with :attr:`leaf_paths`
        (``dict(zip(self.leaf_paths, leaves))`` is the keyed view); inverse of
        :meth:`from_leaf_list`.

        This is the general (any-leaf-type) counterpart of
        :meth:`~NumericEventTemplate.to_vector`, which goes further and ravels +
        concatenates the (numeric) leaves into a flat array. It is distinct from
        ``jax.tree_util.tree_flatten``, whose finer view descends into a
        container leaf (see :class:`~probpipe.Record`).
        """
        return [value[path] for path in self.leaf_paths]

    def from_leaf_list(self, leaves: Iterable[Any]) -> Any:
        """Rebuild a value from its *leaves* — the inverse of :meth:`to_leaf_list`.

        *leaves* are taken in :attr:`leaf_paths` (canonical) order, one per leaf
        and kept whole, and are placed at their paths to rebuild the nested
        value mirroring this template (a nested ``EventTemplate`` → a nested
        :class:`~probpipe.Record`; a numeric template → a
        :class:`~probpipe.NumericRecord`). Round-trip:
        ``tpl.from_leaf_list(tpl.to_leaf_list(v)) == v``.

        Single (unbatched) values only; batched reconstruction arrives with the
        batch abstractions (issue #235). Raises ``ValueError`` if the number of
        *leaves* is not ``len(self.leaf_paths)``.
        """
        leaves = list(leaves)
        n_leaves = len(self.leaf_paths)
        if len(leaves) != n_leaves:
            raise ValueError(
                f"{type(self).__name__}.from_leaf_list: got {len(leaves)} leaves, "
                f"expected {n_leaves} (one per leaf in leaf_paths)."
            )
        leaf_iter = iter(leaves)

        def _build(template: EventTemplate) -> Any:
            from ._numeric_record import NumericRecord
            from .record import Record

            fields = {
                name: (_build(spec) if isinstance(spec, EventTemplate) else next(leaf_iter))
                for name, spec in template._field_map().items()
            }
            cls = NumericRecord if isinstance(template, NumericEventTemplate) else Record
            return cls(fields, event_template=template)

        return _build(self)

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
    def infer_from(cls, value: Any) -> EventTemplate:
        """Best-effort, **lossy** schema inferred by inspecting a value.

        Two cases:

        - A :class:`~probpipe.Record` already carries its authoritative schema,
          so ``infer_from`` returns its :attr:`~probpipe.Record.event_template`
          unchanged.
        - A **mapping** of named fields (e.g. a ``Record``'s field dict) is
          inferred field by field: a nested ``Record`` field contributes its
          own ``event_template``; a numeric array or scalar becomes an
          :class:`ArraySpec` of its shape; anything else becomes a bare
          :class:`OpaqueSpec`. The result auto-promotes to a
          :class:`NumericEventTemplate` when every field is numeric.

        This is the **fallback** for wrapping a raw value that has no template
        yet (e.g. at a workflow boundary); for a value you already hold, read
        its authoritative ``event_template`` directly. Inference is lossy — it
        cannot recover an :class:`ArraySpec`'s ``dtype`` / ``support``, an
        :class:`OpaqueSpec`'s ``meta``, or a :class:`DistributionSpec` /
        :class:`FunctionSpec`. A Python ``list`` / ``tuple`` leaf (no ``.shape``
        / ``.dtype``) is treated as opaque even if it holds numbers; wrap it in
        ``np.asarray`` / ``jnp.asarray`` first for a numeric leaf.

        Parameters
        ----------
        value : Any
            A :class:`~probpipe.Record`, or a mapping of field name to value
            (arrays / scalars / nested ``Record``\\ s).

        Returns
        -------
        EventTemplate
            The inferred schema (a :class:`NumericEventTemplate` when every
            field is numeric).

        Raises
        ------
        TypeError
            If *value* is neither a ``Record`` nor a mapping.
        ValueError
            If *value* is an empty mapping (a template needs at least one field).
        """
        from .record import Record

        if isinstance(value, Record):
            return value.event_template
        if not isinstance(value, Mapping):
            raise TypeError(
                f"infer_from expects a Record or a mapping of fields, got {type(value).__name__}."
            )
        specs: dict[str, _FieldSpecInput] = {}
        for name, val in value.items():
            specs[name] = (
                val.event_template if isinstance(val, Record) else _full_array_shape_or_none(val)
            )
        return EventTemplate(specs)

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
    :attr:`vector_size` and :attr:`leaf_shapes` meaningful:
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
        _field_specs: Mapping[str, _FieldSpecInput] | None = None,
        /,
        **field_specs: _FieldSpecInput,
    ):
        super().__init__(_field_specs, **field_specs)
        object.__setattr__(self, "_vector_size", self._compute_vector_size())

    @property
    def leaf_shapes(self) -> dict[str, tuple[int, ...]]:
        """Per-leaf array shapes, keyed by :attr:`leaf_paths` (canonical leaf order).

        Maps each leaf's ``/``-delimited path to its array ``shape``. Defined
        only on :class:`NumericEventTemplate` — where every leaf is an
        :class:`ArraySpec` and therefore *has* a shape — because a shape is an
        array notion; on a general (mixed) :class:`EventTemplate` the leaves are
        a heterogeneous sum with no uniform shape, so the structural view there
        is :attr:`leaf_paths`. A nested sub-template contributes one entry per
        nested leaf.
        """
        result: dict[str, tuple[int, ...]] = {}
        for name, spec in self._specs.items():
            if isinstance(spec, NumericEventTemplate):
                for sub_name, sub_shape in spec.leaf_shapes.items():
                    result[f"{name}{_PATH_SEP}{sub_name}"] = sub_shape
            else:
                # ``_post_validate`` guarantees a non-nested spec is an ArraySpec.
                result[name] = spec.shape
        return result

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

    # -- 1-D numeric (de)serialization --------------------------------------

    def to_vector(self, value: NumericRecord | NumericRecordArray) -> Array:
        """Serialize *value*'s arrays into its flat 1-D vector representation.

        ``to_vector`` / :meth:`from_vector` convert between the structured and
        flat representations of a numeric value (a PyTree of arrays). A single
        :class:`~probpipe.NumericRecord` serializes to shape ``(vector_size,)``;
        a batched :class:`~probpipe.NumericRecordArray` with ``batch_shape == B``
        serializes to ``(*B, vector_size)``. Leaves are raveled and concatenated
        in this template's canonical leaf order (:attr:`~EventTemplate.leaf_paths`).

        This differs from :meth:`to_leaf_list` (which keeps each leaf whole, any
        type): ``to_vector`` is numeric-only and ravels the leaves into a single
        dense vector.

        Parameters
        ----------
        value : NumericRecord or NumericRecordArray
            The value to serialize; its structure must match this template.

        Returns
        -------
        jax.Array
            The concatenated, raveled numeric leaves: shape ``(vector_size,)``
            for a single value, ``(*B, vector_size)`` for a batch.

        Raises
        ------
        TypeError
            If *value* is not a ``NumericRecord`` / ``NumericRecordArray``.

        See Also
        --------
        from_vector : Reconstruct a value from a flat vector (the inverse).
        """
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
        leaves = self.to_leaf_list(value)
        return jnp.concatenate([jnp.reshape(leaf, (*batch_shape, -1)) for leaf in leaves], axis=-1)

    def from_vector(self, vec: ArrayLike) -> NumericRecord | NumericRecordArray:
        """Reconstruct a numeric value from its flat 1-D vector representation.

        The inverse of :meth:`to_vector`: splits *vec* along its trailing axis
        into this template's leaves (canonical leaf order), reshapes each chunk
        to its event shape, and rebuilds the structured value. The **rank** of
        *vec* selects single vs. batched — a vector of shape ``(vector_size,)``
        rebuilds a single :class:`~probpipe.NumericRecord`; a matrix of shape
        ``(*batch_shape, vector_size)`` rebuilds a
        :class:`~probpipe.NumericRecordArray` with that ``batch_shape``.

        This differs from :meth:`from_leaf_list` (which rebuilds from whole
        leaves, any type): ``from_vector`` is numeric-only and rebuilds from a
        dense vector alone, using this template's leaf shapes.

        Parameters
        ----------
        vec : array-like
            The flat numeric vector; its trailing axis must have length
            :attr:`vector_size`.

        Returns
        -------
        NumericRecord or NumericRecordArray
            Single when *vec* is 1-D, batched otherwise.

        Raises
        ------
        ValueError
            If *vec*'s trailing axis is not :attr:`vector_size`.

        Notes
        -----
        Round-trip: ``self.from_vector(self.to_vector(v)) == v`` for any numeric
        value ``v`` matching this template.

        See Also
        --------
        to_vector : Serialize a value to a flat vector (the inverse).
        """
        vec = jnp.asarray(vec)
        if vec.shape[-1] != self.vector_size:
            raise ValueError(
                f"{type(self).__name__}.from_vector: vec trailing axis is "
                f"{vec.shape[-1]}, expected vector_size={self.vector_size}."
            )
        batch_shape = tuple(vec.shape[:-1])

        offset = 0
        leaves: list[Any] = []

        def _collect(template: NumericEventTemplate) -> None:
            nonlocal offset
            for spec in template._specs.values():
                if isinstance(spec, NumericEventTemplate):
                    _collect(spec)
                else:
                    size = prod(spec.shape) if spec.shape else 1
                    chunk = vec[..., offset : offset + size]
                    offset += size
                    leaves.append(jnp.reshape(chunk, (*batch_shape, *spec.shape)))

        _collect(self)
        treedef = _value_treedef(self, batch_shape)
        return jax.tree_util.tree_unflatten(treedef, leaves)


# ---------------------------------------------------------------------------
# Template walking helpers
# ---------------------------------------------------------------------------


def _value_treedef(
    template: NumericEventTemplate,
    batch_shape: tuple[int, ...],
) -> jax.tree_util.PyTreeDef:
    """PyTreeDef of the value :meth:`NumericEventTemplate.from_vector` reconstructs.

    Builds a throwaway ``NumericRecord`` (or ``NumericRecordArray`` when
    ``batch_shape`` is non-empty) skeleton mirroring *template* and returns its
    ``jax.tree_util.tree_structure``. The treedef depends only on the container
    structure (field names, nesting, ``batch_shape``, template), not on leaf
    values, so its placeholder leaves are cheap zero-stride broadcast arrays.
    Pairing this treedef with the real ordered leaves in
    :func:`jax.tree_util.tree_unflatten` lets ``from_vector`` delegate the value
    assembly to one place.
    """
    from ._numeric_record import NumericRecord
    from ._record_array import NumericRecordArray

    numeric_fill = jnp.zeros((), dtype=jnp.float32)

    def _build(tpl: NumericEventTemplate) -> NumericRecord:
        fields: dict[str, Any] = {}
        for name in tpl.fields:
            spec = tpl[name]
            if isinstance(spec, NumericEventTemplate):
                fields[name] = _build(spec)
            else:
                # ``_post_validate`` guarantees a non-nested spec is an ArraySpec.
                fields[name] = jnp.broadcast_to(numeric_fill, (*batch_shape, *spec.shape))
        if batch_shape:
            return NumericRecordArray(fields, batch_shape=batch_shape, template=tpl)
        return NumericRecord(fields)

    return jax.tree_util.tree_structure(_build(template))


# ---------------------------------------------------------------------------
# Pickle helper
# ---------------------------------------------------------------------------


def _unpickle_event_template(specs: dict) -> EventTemplate:
    return EventTemplate(specs)
