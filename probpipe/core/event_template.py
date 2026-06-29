"""EventTemplate — ProbPipe's structural schema.

An :class:`EventTemplate` describes the **structure** of a value, independent of
the data itself. In particular, an event template describes the structure of a
Python object that takes the form of a named tree: a nested object with unique
string key paths to each leaf. We refer to the leaves of the tree as *fields*.
Thus, an event template can be thought of as the schema for a set of ordered
named fields, where the fields are allowed to be stored in an object with nested
structure.

An event template is designed to be quite general, able to describe the
structure of a single array or a complicated nested object storing arbitrary
Python objects. The restriction is that there must be a sequence of strings
forming a unique path to each field. This follows ProbPipe's convention of
working with *names* in most cases to avoid ambiguity. The event template for
a single object with no nested structure corresponds to a trivial tree with
only a root node. This node is still required to have a name (though ProbPipe
constructors will often auto-generate one when it would be inconvenient for the
user to supply it).

Field names are required and unique within a node; ``/`` is reserved as the path
separator, so every leaf has a unique ``/``-delimited string path
(e.g., "a/b/c"). The canonical leaf order is depth-first in insertion order.

In order to define the structure of trees consistently, :class:`EventTemplate`
clearly defines which objects are considered leaves and which are considered
internal nodes in the tree. The rule is intentionally restrictive for clarity:
- **non-leaf node**: an ``EventTemplate``.
- **leaf node (field)**: one of a fixed set of "field spec" objects.

A field spec is an object that says: "the object at this path is a leaf of the
tree, and it has this structure.". The specs for certain field types may
contain lots of useful structure (e.g., shape and dtype for arrays), while
others may expose no structure at all (e.g., an opaque Python object).
The full set of spec objects are as follows:
- :class:`ArraySpec`: describes a numeric array (shape, optional dtype/support)
- :class:`DistributionSpec`: describes a ``Distribution``. Carries a sub-template
  describing the structure of one sample from the distribution.
- :class:`FunctionSpec`: describes a callable. Carries two sub-templates,
  describing the structure of the inputs and outputs of the function.
- :class:`OpaqueSpec`: fallback for any other object (no structure exposed).

Numeric vs. Mixed
-----------------

:class:`NumericEventTemplate` is the specialization in which all fields are
``ArraySpec``\\ s. In JAX terminology, it describes a value that is a PyTree
of arrays. This sub-class exposes :attr:`~NumericEventTemplate.vector_size`,
which is the number of scalar array elements making up the whole value.
:meth:`NumericEventTemplate.to_vector` converts a concrete value of this
form to a canonical 1-D array representation, with shape `(vector_size,)`.
:meth:`NumericEventTemplate.from_vector` converts back to the structured
representation. ``EventTemplate(...)`` auto-promotes to
``NumericEventTemplate`` when every field is numeric.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
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
    """Mixin for named, ordered collection of fields with nested structure.

    A mixin for functionality that enables an object to be treated as a named,
    ordered collection of fields, where the fields may be stored in a nested,
    tree-like structure. In other words, the fields are the leaves of the tree.
    This allows the object to be treated as a standard Python dictionary, with
    the keys corresponding to the unique string paths identifying each field.
    The key paths reserve ``/`` as the field path separator. Therefore,
    `x["a/b/c"]` selects a field, ``len(x)`` gives the number of fields,
    `"a/b/c" in x` checks a field exists,
    ``x.keys()``/``x.values()``/``x.items()`` give the set of key
    paths/field values/path-value tuples. Iterating over the object iterates
    over the keys in insertion order, following the convention of a Python
    dictionary.

    While this API allows the object to be treated as a flat dictionary, the
    object is in general allowed to have nested structure. :class:`_NamedTree`
    differentiates between leaves and non-leaf nodes by defining the latter
    as an object of the class defined by :meth:`_node_type`; a
    child is an interior node iff it is an instance of that type. All other
     objects are treated as leaves. All nodes (including leaves) can be
     accessed via :meth:`at_path`. In particular,
    `x.at_path("a/b/c")` will return either a field, a sub-tree, or raise
    an error if the path does not exist. The attribute :attr:`children`
    gives local access to the immediate children of the node.

    This mixin does not provide constructor logic; it leaves this up to
    inheriting sub-classes.
    """

    __slots__ = ()

    @classmethod
    def _node_type(cls) -> type:
        """The type whose instances are internal nodes of this tree (hook).

        A field value is an internal node iff it is an instance of this type;
        every other value is a leaf. Each concrete family **narrows** this to
        itself, so a tree descends only into nodes of its own family and never
        into a value of another. Subclasses must override this (and supply the
        ``_tree`` storage attribute); the base returns ``_NamedTree`` only as a
        safe default.
        """
        return _NamedTree

    # -- Mapping API (leaf-keyed) -------------------------------------------

    def __len__(self) -> int:
        return sum(1 for _ in self._walk_leaves())

    def __iter__(self) -> Iterator[str]:
        return (path for path, _ in self._walk_leaves())

    def keys(self) -> Iterator[str]:
        """Iterate the field keys (``/``-paths to the leaves), in canonical order.

        Mirrors :meth:`__iter__` (the field keys *are* what the collection
        iterates), exposed under the mapping name.
        """
        return iter(self)

    def values(self) -> Iterator[Any]:
        """Iterate the field objects (one per leaf), in canonical order."""
        return (leaf for _, leaf in self._walk_leaves())

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate ``(key, field_object)`` pairs, in canonical order."""
        return self._walk_leaves()

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        """Return the field object at *key* — leaf access only.

        *key* is a field key: a ``/``-delimited string or a tuple of names
        (``obj["a/b"]`` == ``obj["a", "b"]``). It must address a leaf; a missing
        key, or a partial path that stops at a subtree, raises ``KeyError`` (use
        :meth:`at_path` to reach a subtree). A non-str/tuple key raises
        ``TypeError``.
        """
        if not isinstance(key, (str, tuple)):
            raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")
        node = self.at_path(key)
        if isinstance(node, self._node_type()):
            raise KeyError(f"{key!r} is a subtree, not a field; use at_path() to navigate to it")
        return node

    def __contains__(self, key: object) -> bool:
        """Whether *key* is a field key (a leaf). Partial paths are not members."""
        if not isinstance(key, (str, tuple)):
            return False
        return self.is_leaf(key)

    @property
    def fields(self) -> tuple[str, ...]:
        """Top-level field names, in insertion order — ``tuple(self.children)``.

        Temporary, retained during the migration to the collection vocabulary;
        new code should use :attr:`children` (one-level mapping) or :meth:`keys`
        (leaf keys). It is scheduled for removal once that vocabulary is carried
        onto the distribution layer, and should not be relied on in new code.
        """
        return tuple(self._tree.keys())

    # -- Tree structure -----------------------------------------------------

    def at_path(self, *path: Any) -> Any:
        """Return the object at *path* — a field object, or an interior subtree.

        *path* addresses a position in the storage tree and may be written as a
        ``/``-delimited string, separate string segments, or a single tuple of
        names — these are equivalent::

            obj.at_path("physics/mass")
            obj.at_path("physics", "mass")
            obj.at_path(("physics", "mass"))

        A *key* (a path that ends at a leaf) returns the field object. A
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
            field_map = node._tree
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
        return MappingProxyType(dict(self._tree))

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

    def is_field(self, *path: Any) -> bool:
        """Whether *path* resolves to a field (a leaf) — alias of :meth:`is_leaf`."""
        return self.is_leaf(*path)

    def to_nested_dict(self) -> dict[str, Any]:
        """Return a nested ``dict`` mirroring the storage tree.

        Each interior node becomes a nested ``dict``; each leaf maps to its
        field object. This is the tree-shaped export, distinct from the flat
        ``dict(obj)`` view that is keyed by full path.
        """
        node_type = self._node_type()
        result: dict[str, Any] = {}
        for name, child in self._tree.items():
            result[name] = child.to_nested_dict() if isinstance(child, node_type) else child
        return result

    # -- Leaf traversal primitives ------------------------------------------

    def _walk_leaves(self) -> Iterator[tuple[str, Any]]:
        """Yield ``(path, leaf_object)`` for every field, in canonical order.

        Canonical order is the depth-first, insertion-order traversal (§ the
        class docstring). This is the single traversal that the field-wise
        operations (the field mapping, ``map``, vector serialization, equality)
        are expressed on.
        """
        node_type = self._node_type()
        for name, child in self._tree.items():
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
        same-class collection, so whatever normalisation that constructor applies
        to its leaves is re-applied here.

        Raises
        ------
        ValueError
            If *values* does not supply exactly one object per leaf.
        """
        it = iter(values)
        node_type = self._node_type()
        sentinel = object()

        def build(node: Any, prefix: str) -> Any:
            new_children: dict[str, Any] = {}
            for name, child in node._tree.items():
                path = f"{prefix}{name}"
                if isinstance(child, node_type):
                    new_children[name] = build(child, f"{path}{_PATH_SEP}")
                else:
                    nxt = next(it, sentinel)
                    if nxt is sentinel:
                        raise ValueError("_rebuild_from_leaves got fewer values than leaves")
                    if isinstance(nxt, node_type):
                        raise ValueError(
                            f"cannot place a {node_type.__name__} at field {path!r}: "
                            f"that would introduce nesting and change the structure"
                        )
                    new_children[name] = nxt
            return type(node)(new_children)

        result = build(self, "")
        if next(it, sentinel) is not sentinel:
            raise ValueError("_rebuild_from_leaves got more values than leaves")
        return result

    # -- Structural transforms ----------------------------------------------

    def map(self, f: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Apply *f* to every field object, returning a same-shape collection.

        Returns a new collection of the **same class and the same structure**
        (identical names and nesting) whose fields are ``f``'s outputs. *f* is
        called as ``f(field_object, *args, **kwargs)`` for each field in canonical
        order. Any extra *args* / *kwargs* are forwarded to *f* unchanged and are
        **constant across fields** (not varied per field).

        The structure is preserved exactly. *f* must return a leaf object, not an
        instance of the node type (the class returned by :meth:`_node_type`); such
        a return would introduce nesting and raises ``ValueError`` naming the
        field. Each output is placed back through the subclass's own construction,
        so whatever normalisation that constructor applies to a leaf applies here
        too (see the concrete class's docstring).

        Raises
        ------
        ValueError
            If *f* returns a node-type instance at any field.
        """
        return self._rebuild_from_leaves(
            f(leaf, *args, **kwargs) for _, leaf in self._walk_leaves()
        )

    def map_with_keys(self, f: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Like :meth:`map`, but *f* also receives each field's key (path).

        *f* is called as ``f(key, field_object, *args, **kwargs)``, where *key* is
        the field's full ``/``-path. Everything else — structure preservation,
        the node-type-return guard, per-class result handling — matches :meth:`map`.
        """
        return self._rebuild_from_leaves(
            f(key, leaf, *args, **kwargs) for key, leaf in self._walk_leaves()
        )

    # -- Structural edits ---------------------------------------------------
    #
    # The leaf-map computations below are pure structure: they read the storage
    # tree and return a new flat ``path -> object`` mapping, with no knowledge of
    # leaf specs or an authoritative template. ``without`` / ``merge`` / ``replace``
    # rebuild ``type(self)`` from that map (correct for a tree of specs, where the
    # edited specs *are* the schema); a value type that carries a separate
    # authoritative schema overrides those three to thread it, reusing these helpers.

    def _leaves_without(self, paths: tuple[str, ...]) -> dict[str, Any]:
        """Flat leaf-map with the fields/subtrees at *paths* dropped."""
        for path in paths:
            self.at_path(path)  # KeyError if the path does not exist
        drops = [self._norm_path(p) for p in paths]

        def is_dropped(key: str) -> bool:
            return any(key == d or key.startswith(f"{d}{_PATH_SEP}") for d in drops)

        kept = {key: leaf for key, leaf in self._walk_leaves() if not is_dropped(key)}
        if not kept:
            raise ValueError("Cannot remove all fields from a collection")
        return kept

    def without(self, *paths: str) -> Any:
        """Return a new collection with the fields/subtrees at *paths* removed.

        Each path is a key (drop one leaf) or a partial path (drop a whole
        subtree). A missing path raises ``KeyError``; removing every field raises
        ``ValueError``. Surviving fields keep their order and their specs.
        """
        return type(self)(self._leaves_without(paths))

    def _leaves_merged(self, other: Any) -> dict[str, Any]:
        """Flat leaf-map unioning ``self``'s then *other*'s leaves, keyed by path."""
        left = dict(self._walk_leaves())
        right = dict(other._walk_leaves())
        overlap = set(left) & set(right)
        if overlap:
            raise ValueError(f"Overlapping field keys: {sorted(overlap)}")
        return {**left, **right}

    def merge(self, other: Any) -> Any:
        """Return the union of two collections, merged by leaf key (deep).

        The merge is by field key: ``{"a/x": ...}`` and ``{"a/y": ...}`` combine
        under a shared ``a``. A field key present in both, or a field-versus-prefix
        clash between them, raises ``ValueError``. ``self``'s fields come first,
        then ``other``'s.
        """
        return type(self)(self._leaves_merged(other))

    @staticmethod
    def _resolve_replace_updates(
        _updates: Mapping[str, Any] | None, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Normalise ``replace``'s positional-mapping XOR keyword inputs to one dict."""
        if _updates is not None:
            if updates:
                raise ValueError("Cannot pass both positional mapping and keyword arguments")
            return dict(_updates)
        return dict(updates)

    def _leaves_replaced(self, resolved: Mapping[str, Any]) -> dict[str, Any]:
        """Flat leaf-map with the object at each path replaced *in place*.

        Each old leaf (or whole subtree) at a path is dropped and *new_value* put
        where the first of those leaves was, so a replaced subtree keeps its
        position (canonical order is part of the collection's identity).
        """
        for path in resolved:
            self.at_path(path)  # KeyError if the path does not exist
        flat = dict(self._walk_leaves())
        for path, new_value in resolved.items():
            norm = self._norm_path(path)
            rebuilt: dict[str, Any] = {}
            placed = False
            for key, leaf in flat.items():
                if key == norm or key.startswith(f"{norm}{_PATH_SEP}"):
                    if not placed:
                        rebuilt[norm] = new_value
                        placed = True
                else:
                    rebuilt[key] = leaf
            flat = rebuilt
        return flat

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Any:
        """Return a new collection with the objects at the given paths replaced.

        Updates are given as a path-keyed positional mapping
        (``r.replace({"physics/mass": m})``) or flat keywords (``r.replace(obs=y)``),
        not both. Every path must already exist (``KeyError`` otherwise — ``replace``
        edits, it does not add). A partial path replaces a whole subtree. Untouched
        fields keep their specs; a replaced field takes the new value's spec.
        """
        resolved = self._resolve_replace_updates(_updates, updates)
        if not resolved:
            return self
        return type(self)(self._leaves_replaced(resolved))

    # -- Construction from a nested mapping ---------------------------------

    @staticmethod
    def _explode_nested(
        data: Mapping[str, Any],
        recurse_into: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """Flatten a nested mapping into a flat ``path -> value`` map.

        A ``Mapping`` value denotes nesting and is recursed into; every other value
        is a leaf. When *recurse_into* is given it is consulted (with the value's
        full ``/``-path) for each ``Mapping`` value and may veto the recursion,
        keeping that mapping as an opaque leaf.
        """
        flat: dict[str, Any] = {}

        def walk(sub: Mapping[str, Any], prefix: str) -> None:
            for name, val in sub.items():
                path = f"{prefix}{_PATH_SEP}{name}" if prefix else name
                structural = isinstance(val, Mapping)
                if structural and recurse_into is not None:
                    structural = recurse_into(path)
                if structural:
                    walk(val, path)
                else:
                    flat[path] = val

        walk(data, "")
        return flat

    @classmethod
    def from_nested_dict(cls, data: Mapping[str, Any]) -> Any:
        """Build a collection from a **nested** ``dict`` — the inverse of :meth:`to_nested_dict`.

        Each nested ``dict`` level becomes a subtree and every other value a leaf.
        This is the opt-in way to read a nested ``dict`` as structure; the bare
        constructor instead treats a ``dict`` value as an opaque leaf.

        Parameters
        ----------
        data
            A nested mapping of field names to values (or sub-mappings).

        Returns
        -------
        A new collection of this class.
        """
        return cls(cls._explode_nested(data))

    # -- Internal utilities -------------------------------------------------

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

    def _norm_path(self, path: Any) -> str:
        """Normalise a path argument to its canonical ``/``-string form."""
        return _PATH_SEP.join(self._split_path((path,)))


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

    - **field** — one named object in the collection (here, a leaf spec),
      addressed by its full ``/``-delimited **key** (path from the root, e.g.
      ``"physics/mass"``; a single name for a flat template). The mapping
      protocol (:meth:`keys` / :meth:`values` / :meth:`items` / iteration /
      ``len`` / ``in`` / ``[]``) ranges over the fields, keyed by path.
    - **leaf** — a *terminal* node: an :class:`ArraySpec` / :class:`OpaqueSpec`
      / :class:`DistributionSpec` / :class:`FunctionSpec`. A nested
      ``EventTemplate`` is an *internal node*, not a leaf; the fields are the
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
    ``jax.tree_util.tree_leaves(v)`` returns the leaves in :meth:`keys`
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
    enumeration of leaves is :meth:`keys`, and per-leaf array shapes (on a
    numeric template) live on :attr:`~NumericEventTemplate.leaf_shapes`.

    Calling ``EventTemplate(...)`` directly auto-promotes to a
    :class:`NumericEventTemplate` when every spec is numeric (and every nested
    sub-template is itself all-numeric), so :attr:`vector_size` and
    :attr:`~NumericEventTemplate.leaf_shapes` are reachable in the common all-numeric case
    without naming the subclass. Mixed templates (any opaque / ``None`` spec)
    stay plain ``EventTemplate`` and do not expose :attr:`vector_size` — it is
    not a meaningful quantity once opaque leaves are present.
    """

    __slots__ = ("_tree",)

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
        object.__setattr__(self, "_tree", specs)

    def _post_validate(self, field_specs: dict[str, _FieldSpec]) -> None:
        """Subclass hook for stricter spec validation. No-op on the base."""
        return

    # -- Immutability -------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("EventTemplate is immutable")

    def __reduce__(self):
        return (_unpickle_event_template, (dict(self._tree),))

    # -- Tree structure -----------------------------------------------------
    #
    # The mapping / navigation surface is inherited from :class:`_NamedTree`. A
    # leaf here is a leaf spec (:class:`ArraySpec` / :class:`OpaqueSpec` /
    # :class:`DistributionSpec` / :class:`FunctionSpec`); an internal node is a
    # nested ``EventTemplate``.

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
        for spec in self._tree.values():
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
        ``len(self) > 1``.

        Returns
        -------
        bool
            ``True`` iff the template has more than one leaf; ``False`` iff it
            describes exactly one leaf.
        """
        return len(self) > 1

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
        for name, spec in self._tree.items():
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
                for name, spec in self._tree.items()
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

    def from_field_values(self, values: Iterable[Any]) -> Any:
        """Reconstruct a value from an ordered sequence of field values.

        *values* supplies one object per leaf field, in canonical leaf order (the
        order of :meth:`keys`); the names and tree shape are taken from this template, so
        the result mirrors it — a nested template builds a nested
        :class:`~probpipe.Record`, a :class:`NumericEventTemplate` builds a
        :class:`~probpipe.NumericRecord`. The result carries this template as its
        **authoritative** :attr:`~probpipe.Record.event_template` (nothing is
        inferred), so the round-trip is faithful:
        ``tpl.from_field_values(list(r.values())) == r`` when
        ``r.event_template == tpl``. The export side is just ``list(r.values())``.

        Single (unbatched) values only; batched reconstruction arrives with the
        batch abstractions. Per-leaf shape/dtype is not checked (only the count).

        Raises
        ------
        ValueError
            If the number of *values* is not the number of fields (``len(self)``).
        """
        values = list(values)
        n_leaves = len(self)
        if len(values) != n_leaves:
            raise ValueError(
                f"{type(self).__name__}.from_field_values: got {len(values)} values, "
                f"expected {n_leaves} (one per field)."
            )
        leaf_iter = iter(values)

        def _build(template: EventTemplate) -> Any:
            from ._numeric_record import NumericRecord
            from .record import Record

            fields = {
                name: (_build(spec) if isinstance(spec, EventTemplate) else next(leaf_iter))
                for name, spec in template._tree.items()
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
        return tuple(self._tree.items()) == tuple(other._tree.items())

    def __hash__(self) -> int:
        # All field specs (leaf specs and nested templates) are hashable, so
        # the order-sensitive item tuple hashes directly. Insertion order is
        # part of the template's identity.
        return hash(tuple(self._tree.items()))

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
        for name, spec in self._tree.items():
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
        """Per-leaf array shapes, keyed by :meth:`keys` (canonical leaf order).

        Maps each leaf's ``/``-delimited path to its array ``shape``. Defined
        only on :class:`NumericEventTemplate` — where every leaf is an
        :class:`ArraySpec` and therefore *has* a shape — because a shape is an
        array notion; on a general (mixed) :class:`EventTemplate` the leaves are
        a heterogeneous sum with no uniform shape, so the structural view there
        is :meth:`keys`. A nested sub-template contributes one entry per
        nested leaf.
        """
        result: dict[str, tuple[int, ...]] = {}
        for name, spec in self._tree.items():
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
        for spec in self._tree.values():
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
        in this template's canonical leaf order (:meth:`~EventTemplate.keys`).

        This differs from ``list(record.values())`` (which keeps each leaf whole, any
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
        # The value's leaves in canonical leaf order (each kept whole).
        leaves = [value[key] for key in self.keys()]
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

        This differs from :meth:`from_field_values` (which rebuilds from whole
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
            for spec in template._tree.values():
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
        for name, spec in tpl.children.items():
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
