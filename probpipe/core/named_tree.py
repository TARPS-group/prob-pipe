"""The named, ordered tree substrate shared by ProbPipe's structured objects.

Values and schemas in ProbPipe are represented as **named, ordered trees**:
nested objects with a unique string path to each leaf. This module provides
:class:`NamedTree`, the shared substrate those families are built on — the one
place the leaf-keyed mapping contract, tree navigation, and the
structure-preserving transforms are defined. It is reused by
:class:`~probpipe.core._specs.RecordSpec` (a tree of value specs)
and :class:`~probpipe.Record` (a tree of values), so the type- and value-level
structures cannot drift apart on how a field is named or a path is resolved.

See :class:`NamedTree` for the specifics: the field/path/key terminology, the
mapping contract and navigation surface, the leaf-versus-node rule, and the
transforms.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Self, cast

__all__ = ["NamedTree"]

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


def _unflatten_paths(source: Mapping[str, Any]) -> dict[str, Any]:
    """Explode a path-keyed mapping into a nested ordered mapping.

    Each key is a field key that may be *path-shaped* — a ``/``-delimited string
    denotes nesting (``{"a/b": x}`` becomes ``{"a": {"b": x}}``). Keys are read
    in order; the result preserves the **first-appearance order** of each
    distinct leading segment (recursively), which is the canonical field order.
    Nesting is a plain nested ``dict``; a constructor materialises any mapping
    value into a child collection (a mapping is never a leaf).

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
        result[head] = _unflatten_paths(val) if is_prefix[head] else val
    return result


# -- Moving nodes to new exact paths ---------------------------------------------
#
# ``with_path_names`` rebuilds a plain nested-dict copy of a tree, in which the
# dicts are the interior nodes, since a mapping is never a leaf. Each node of the
# copy records its original path, so the rebuild reports where every node went.


class _Moving:
    """A node of the tree that a rename rebuilds, with its path in the original tree.

    *origin* is ``None`` for a group that a move creates. *held* records whether
    an original group held a node, since only such a group can be vacated.
    """

    __slots__ = ("children", "held", "leaf", "origin")

    def __init__(
        self,
        origin: tuple[str, ...] | None,
        *,
        leaf: Any = None,
        children: dict[str, _Moving] | None = None,
        held: bool = False,
    ) -> None:
        self.origin = origin
        self.leaf = leaf
        self.children = children
        self.held = held


def _moving(tree: Mapping[str, Any], prefix: tuple[str, ...]) -> dict[str, _Moving]:
    """The nodes of the nested dict *tree*, found at *prefix*, as nodes to rebuild."""
    nodes: dict[str, _Moving] = {}
    for name, value in tree.items():
        path = (*prefix, name)
        if isinstance(value, dict):
            nodes[name] = _Moving(path, children=_moving(value, path), held=bool(value))
        else:
            nodes[name] = _Moving(path, leaf=value)
    return nodes


def _is_vacant(node: _Moving) -> bool:
    """Whether *node* is a group left without a field by the moves.

    A group that was empty in the original tree is not vacant, so it stays.
    """
    if node.children is None or (node.origin is not None and not node.held):
        return False
    return all(_is_vacant(child) for child in node.children.values())


def _collision(path: tuple[str, ...], name: str, *, prefix_clash: bool) -> ValueError:
    message = f"with_path_names() target {_PATH_SEP.join(path)!r} collides with an existing node"
    if prefix_clash:
        message += f"; the name {name!r} would be used both as a field and as a path prefix"
    return ValueError(message)


def _place(group: dict[str, _Moving], name: str, node: _Moving, path: tuple[str, ...]) -> None:
    """Put *node* under *name* at the end of *group*, where a vacant node gives way.

    Raises
    ------
    ValueError
        If a node that is not vacant holds *name*.
    """
    held = group.get(name)
    if held is not None:
        if _is_vacant(held):
            del group[name]
        elif _is_vacant(node):
            return
        else:
            raise _collision(
                path, name, prefix_clash=(held.children is None) != (node.children is None)
            )
    group[name] = node


def _kept(
    nodes: dict[str, _Moving],
    moves: Mapping[tuple[str, ...], tuple[str, ...]],
    in_place: set[tuple[str, ...]],
    detached: dict[tuple[str, ...], _Moving],
) -> dict[str, _Moving]:
    """*nodes* with each moved node detached into *detached* and each in-place rename applied.

    The children of every node are treated first, so a node below a moved node
    that moves itself is detached from it.
    """
    kept: dict[str, _Moving] = {}
    for name, node in nodes.items():
        if node.children is not None:
            node.children = _kept(node.children, moves, in_place, detached)
        origin = cast(tuple[str, ...], node.origin)
        target = moves.get(origin)
        if target is None:
            _place(kept, name, node, origin)
        elif origin in in_place:
            _place(kept, target[-1], node, target)
        else:
            detached[origin] = node
    return kept


def _stripped(
    nodes: dict[str, _Moving], prefix: tuple[str, ...], origins: dict[str, str]
) -> dict[str, Any]:
    """The nested dict of *nodes* without vacant groups, recording each node's origin."""
    tree: dict[str, Any] = {}
    for name, node in nodes.items():
        if _is_vacant(node):
            continue
        path = (*prefix, name)
        if node.origin is not None:
            origins[_PATH_SEP.join(path)] = _PATH_SEP.join(node.origin)
        tree[name] = node.leaf if node.children is None else _stripped(node.children, path, origins)
    return tree


def _moved_tree(
    tree: Mapping[str, Any], moves: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """The nested dict *tree* with the node at each key of *moves* moved to its target.

    *moves* maps the exact paths of nodes to their new exact paths, as
    :meth:`NamedTree._resolve_path_renames` validates them. A node whose target
    has the parent of its source, and none of whose ancestors moves, is renamed
    in place; every other node is appended to its target's parent, in the order
    of *moves*, and a missing parent is created the same way. A group that the
    moves leave without a field is removed.

    Returns
    -------
    tuple of dict and dict
        The rebuilt nested dict, and the original path of each of its nodes that
        is a node of *tree*, keyed by the node's new path.

    Raises
    ------
    ValueError
        If a target collides with a node that remains.
    """
    split = {
        tuple(source.split(_PATH_SEP)): tuple(target.split(_PATH_SEP))
        for source, target in moves.items()
    }
    in_place = {
        source
        for source, target in split.items()
        if source[:-1] == target[:-1]
        and not any(source[:end] in split for end in range(1, len(source)))
    }
    detached: dict[tuple[str, ...], _Moving] = {}
    root = _kept(_moving(tree, ()), split, in_place, detached)
    for source, target in split.items():
        if source in in_place:
            continue
        group = root
        for segment in target[:-1]:
            parent = group.get(segment)
            if parent is None:
                parent = group[segment] = _Moving(None, children={})
            elif parent.children is None:
                raise _collision(target, segment, prefix_clash=True)
            group = parent.children
        _place(group, target[-1], detached[source], target)
    origins: dict[str, str] = {}
    return _stripped(root, (), origins), origins


class NamedTree[L]:
    """The named, ordered tree addressed by path — the shared collection substrate.

    A ``NamedTree`` is a named, ordered collection of **fields** stored in a
    nested, tree-like structure: a field is one named leaf, and the leaf value
    is nameless on its own — the name the tree gives it makes it a field. A
    **path** is a ``/``-joined sequence addressing a field or an interior
    node; a **key** is a path that addresses a field; a **child** of a node is
    an entry directly under it; and the **canonical order** is the depth-first
    walk visiting children in insertion order.

    The class is a ``Mapping`` whose keys are exactly its field keys, in
    canonical order, so ``keys()`` / ``values()`` / ``items()`` / ``len`` /
    ``in`` / ``[]`` all agree on that one key set, as for a plain ``dict``:
    ``x["a/b/c"]`` selects a field, ``len(x)`` counts the fields, and
    ``"a/b/c" in x`` tests membership. A path may be written equivalently as
    a ``/``-delimited string or a tuple of names, so
    ``x["a/b/c"] == x["a", "b", "c"]``. Since interior nodes are *not* keys,
    ``[]`` raises on an interior path; interior nodes are reached via the
    one-level :attr:`children` view of the storage or :meth:`at_path`, which
    can access any field or subtree, so ``x.children["a"].children["b"]`` is
    the node that ``x.at_path("a", "b")`` and ``x.at_path("a/b")`` address.
    Sibling names are distinct, so every path identifies at most one node;
    distinct subtrees may reuse a name (``a/c`` and ``b/c``).

    A child is an interior node if and only if it is an instance of the
    family's own node class (the hook :meth:`_node_type`); every other value
    is a leaf, validated against the family's declared :meth:`_leaf_type` at
    construction. The static counterpart of that leaf type is the class type
    parameter ``L`` — :class:`~probpipe.core._specs.RecordSpec`
    binds :class:`~probpipe.core._specs.TermSpec`, :class:`~probpipe.Record`
    binds ``Any`` — which the leaf-trafficking accessors (``[]``,
    :meth:`values`, :meth:`items`, :meth:`map`) carry through to typed
    consumers. Those accessors return each leaf as the family presents it
    (the hook :meth:`_present`), which is the stored leaf unless the family
    overrides the hook. Mappings are never leaves: a mapping value denotes tree
    structure (see :meth:`_check_leaf`).
    Navigation yields views into the same storage; the structure-preserving
    transforms (:meth:`map`, :meth:`replace`, :meth:`merge`, :meth:`without`,
    :meth:`with_path_names`) return a tree of the same family, rebuilt through
    :meth:`_rebuild_class`.

    This substrate provides no constructor logic; each family's constructor
    owns storage (the ``_tree`` attribute) and validation policy.
    """

    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Abstract substrate: an instance only ever exists as a concrete
        # family (``Record`` / ``RecordSpec`` / the batch types). A direct
        # ``NamedTree()`` would build an object with no ``_tree`` store, so
        # every later method would fail with a cryptic ``AttributeError``;
        # fail loudly here instead. Concrete families define their own
        # ``__new__`` (calling ``object.__new__`` directly) or inherit this
        # pass-through, so the guard only trips a literal ``NamedTree(...)``.
        if cls is NamedTree:
            raise TypeError(
                "NamedTree is an abstract substrate and cannot be instantiated "
                "directly; construct a concrete family such as Record or "
                "RecordSpec."
            )
        return super().__new__(cls)

    @classmethod
    def _node_type(cls) -> type:
        """The type whose instances are internal nodes of this tree (hook).

        A field value is an internal node iff it is an instance of this type;
        every other value is a leaf. Each concrete family **narrows** this to
        itself, so a tree descends only into nodes of its own family and never
        into a value of another. Subclasses must override this (and supply the
        ``_tree`` storage attribute); the base returns ``NamedTree`` only as a
        safe default.
        """
        return NamedTree

    @classmethod
    def _rebuild_class(cls) -> type:
        """The class the structural rebuilds construct through (hook).

        ``without`` / ``merge`` / ``replace`` / ``map`` build their results by
        calling this class. The default is ``cls`` itself, so a family whose
        concrete class is chosen *explicitly* (``Record`` vs ``NumericRecord``)
        keeps it through an edit. A family whose base constructor **selects**
        the concrete class (auto-promotion) overrides this with that base
        class, so an edit re-decides the promotion instead of forcing the
        result into the original subclass — e.g. replacing an array spec with
        an opaque one turns a ``NumericRecordSpec`` into a mixed
        ``RecordSpec``, and removing the last opaque spec promotes.
        """
        return cls

    def _rebuild_node(self, leaves: Mapping[str, Any], *, node_name: str | None) -> Any:
        """Construct one rebuilt node from a flat leaf map (hook).

        The structure-preserving transforms build their results through this
        hook. *node_name* is the field key the node sits under for a nested
        node and ``None`` for the root. The default constructs through
        :meth:`_rebuild_class` and ignores *node_name*; a family whose
        constructor requires a name (the value types) overrides this to
        supply it — a nested node is named by its field key, and the root
        follows the transform's identity rule.
        """
        return self._rebuild_class()(leaves)

    @classmethod
    def _leaf_type(cls) -> type | tuple[type, ...]:
        """The family's declared leaf type (hook).

        Construction validates every leaf against this type via
        :meth:`_check_leaf`, so a malformed tree fails at construction rather
        than at first navigation. A family whose leaves are arbitrary values
        declares ``object``, making the isinstance check vacuous; the
        mappings-are-never-leaves rule still applies.
        """
        return object

    @classmethod
    def _check_leaf(cls, path: str, value: Any) -> None:
        """Validate one leaf at construction time.

        Enforces two rules. First, the substrate-wide invariant that a
        **mapping is never a leaf**: a mapping denotes tree structure, so a
        constructor is expected to materialise any mapping value into a child
        collection *before* leaf validation (see the constructors); a mapping
        reaching this method is therefore a
        contract violation and raises. Second, the leaf must be an instance of
        the family's declared :meth:`_leaf_type`. Codifying the mapping rule
        here keeps it in one place rather than re-derived by each family; how a
        constructor materialises a mapping into the node type stays the family's
        own concern.

        Parameters
        ----------
        path : str
            The field path the value is being stored at (for the error).
        value
            The candidate leaf value.

        Raises
        ------
        TypeError
            If *value* is a mapping, or does not satisfy :meth:`_leaf_type`.
        """
        if isinstance(value, Mapping):
            raise TypeError(
                f"field {path!r} is a mapping ({type(value).__name__}); a mapping "
                f"denotes tree structure and is never a leaf — pass it as a nested "
                f"mapping so the constructor materialises it into a subtree instead"
            )
        leaf_type = cls._leaf_type()
        if leaf_type is not object and not isinstance(value, leaf_type):
            raise TypeError(
                f"field {path!r} must be a "
                f"{getattr(leaf_type, '__name__', leaf_type)}, "
                f"got {type(value).__name__}"
            )

    # -- Mapping API (leaf-keyed) -------------------------------------------

    def __len__(self) -> int:
        return sum(1 for _ in self._walk_leaves())

    def __iter__(self) -> Iterator[str]:
        return (path for path, _ in self._walk_leaves())

    def keys(self) -> tuple[str, ...]:
        """The field keys (``/``-paths to the leaves), in canonical order.

        Returns a reusable (materialised) tuple, matching the dict-like
        contract; iteration over the collection yields the same keys.
        """
        return tuple(path for path, _ in self._walk_leaves())

    def values(self) -> tuple[L, ...]:
        """The field objects (one per leaf), in canonical order (materialised)."""
        return tuple(self._present(key, leaf) for key, leaf in self._walk_leaves())

    def items(self) -> tuple[tuple[str, L], ...]:
        """``(key, field_object)`` pairs, in canonical order (materialised)."""
        return tuple((key, self._present(key, leaf)) for key, leaf in self._walk_leaves())

    def __getitem__(self, key: str | tuple[str, ...]) -> L:
        """Return the field object at *key* — leaf access only.

        *key* is a field key: a ``/``-delimited string or a tuple of names
        (``obj["a/b"]`` == ``obj["a", "b"]``). It must address a leaf; a missing
        key, or a partial path that stops at a subtree, raises ``KeyError`` (use
        :meth:`at_path` to reach a subtree). A non-str/tuple key raises
        ``TypeError``.
        """
        if not isinstance(key, (str, tuple)):
            raise TypeError(f"key must be str or tuple[str, ...], got {type(key).__name__}")
        node = self._node_at(key)
        if isinstance(node, self._node_type()):
            raise KeyError(f"{key!r} is a subtree, not a field; use at_path() to navigate to it")
        # Past the node-type guard the value is a leaf; ``_node_type()`` is a
        # runtime ``type`` a checker cannot use to narrow the ``L | Self`` union.
        return self._present(self._norm_path(key), cast(L, node))

    def __contains__(self, key: object) -> bool:
        """Whether *key* is a field key (a leaf). Partial paths are not members."""
        if not isinstance(key, (str, tuple)):
            return False
        return self.is_field(key)

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

    def at_path(self, *path: Any) -> L | Self:
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
        operators (``[]`` / ``in`` / iteration) range only over fields. A field
        object is the leaf as the family presents it (:meth:`_present`).

        Raises
        ------
        KeyError
            If the path reaches nothing, or tries to descend through a leaf.
        TypeError
            If a path segment is not a string.
        """
        node = self._node_at(*path)
        if isinstance(node, self._node_type()):
            return node
        return self._present(_PATH_SEP.join(self._split_path(path)), node)

    def _node_at(self, *path: Any) -> L | Self:
        """The stored node at *path*: the leaf as stored, or the subtree.

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
    def children(self) -> Mapping[str, L | Self]:
        """Read-only one-level view of this node — ``local_name -> child``.

        Each value is an immediate child: a field object (leaf) or a subtree (a
        nested collection of the same class). Insertion-ordered. This is the
        labelled top-level view; the top-level names are ``tuple(obj.children)``.
        Unlike the leaf-keyed field view, ``children.values()`` includes
        subtrees.
        """
        # Instances are immutable after construction, so the proxy can wrap
        # the storage dict directly (no per-access copy).
        return MappingProxyType(self._tree)

    def is_field(self, *path: Any) -> bool:
        """Whether *path* resolves to a field (a leaf), not an interior subtree.

        Returns ``True`` only when *path* is navigable **and** ends at a leaf.
        Accepts the same path forms as :meth:`at_path`.
        """
        try:
            node = self._node_at(*path)
        except (KeyError, TypeError):
            return False
        return not isinstance(node, self._node_type())

    @property
    def is_multi_field(self) -> bool:
        """Whether this tree holds more than one field.

        Counts *reachable* fields recursively (descending into nested
        subtrees), not top-level children — a single top-level child that
        nests several fields is multi-field. Equivalent to ``len(self) > 1``.
        """
        walker = self._walk_leaves()
        next(walker)  # a tree always has at least one field
        return next(walker, None) is not None

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

    def _present(self, key: str, leaf: L) -> L:
        """The field object the leaf-keyed accessors return for the stored *leaf* at *key* (hook).

        The stored leaf itself; a family whose access returns views overrides
        it. The structure-preserving transforms read the stored leaves.
        """
        return leaf

    def _node_paths(self) -> Iterator[str]:
        """Every node's path, interior nodes included, in canonical order."""
        node_type = self._node_type()
        for name, child in self._tree.items():
            yield name
            if isinstance(child, node_type):
                for sub_path in child._node_paths():
                    yield f"{name}{_PATH_SEP}{sub_path}"

    def _walk_leaves(self) -> Iterator[tuple[str, L]]:
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
            return node._rebuild_node(
                new_children, node_name=prefix.rstrip(_PATH_SEP).rsplit(_PATH_SEP, 1)[-1] or None
            )

        result = build(self, "")
        if next(it, sentinel) is not sentinel:
            raise ValueError("_rebuild_from_leaves got more values than leaves")
        return result

    # -- Structural transforms ----------------------------------------------

    def map(self, f: Callable[..., L], /, *args: Any, **kwargs: Any) -> Self:
        """Apply *f* to every field object, returning a same-shape collection.

        Returns a new collection of the **same structure** (identical names and
        nesting) whose fields are ``f``'s outputs, rebuilt through the class's
        constructor via :meth:`_rebuild_class` — for the value types this is the
        same class; a base ``RecordSpec`` may auto-promote to (or demote
        from) :class:`NumericRecordSpec` when the mapped specs change
        numericness. *f* is
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

    def map_with_keys(self, f: Callable[..., L], /, *args: Any, **kwargs: Any) -> Self:
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
    # value specs or an authoritative template. ``without`` / ``merge`` / ``replace``
    # rebuild ``type(self)`` from that map (correct for a tree of specs, where the
    # edited specs *are* the schema); a value type that carries a separate
    # authoritative schema overrides those three to thread it, reusing these helpers.

    def _leaves_without(self, paths: tuple[str, ...]) -> dict[str, Any]:
        """Flat leaf-map with the fields/subtrees at *paths* dropped."""
        for path in paths:
            self._node_at(path)  # KeyError if the path does not exist
        drops = [self._norm_path(p) for p in paths]

        def is_dropped(key: str) -> bool:
            return any(key == d or key.startswith(f"{d}{_PATH_SEP}") for d in drops)

        kept = {key: leaf for key, leaf in self._walk_leaves() if not is_dropped(key)}
        if not kept:
            raise ValueError("Cannot remove all fields from a collection")
        return kept

    def without(self, *paths: str) -> Self:
        """Return a new collection with the fields/subtrees at *paths* removed.

        Each path is a key (drop one leaf) or a partial path (drop a whole
        subtree). A missing path raises ``KeyError``; removing every field raises
        ``ValueError``. Surviving fields keep their order and their specs.
        """
        return self._rebuild_node(self._leaves_without(paths), node_name=None)

    def _leaves_merged(self, other: Any) -> dict[str, Any]:
        """Flat leaf-map unioning ``self``'s then *other*'s leaves, keyed by path."""
        left = dict(self._walk_leaves())
        right = dict(other._walk_leaves())
        overlap = set(left) & set(right)
        if overlap:
            raise ValueError(f"Overlapping field keys: {sorted(overlap)}")
        return {**left, **right}

    def merge(self, other: Any) -> Self:
        """Return the union of two collections, merged by leaf key (deep).

        The merge is by field key: ``{"a/x": ...}`` and ``{"a/y": ...}`` combine
        under a shared ``a``. A field key present in both, or a field-versus-prefix
        clash between them, raises ``ValueError``. ``self``'s fields come first,
        then ``other``'s.
        """
        return self._rebuild_node(self._leaves_merged(other), node_name=None)

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

        Raises
        ------
        KeyError
            If a path does not exist.
        ValueError
            If one update path is an ancestor of another — applying the
            ancestor first would silently swallow the descendant's update, so
            overlapping paths are rejected outright.
        """
        for path in resolved:
            self._node_at(path)  # KeyError if the path does not exist
        norms = [self._norm_path(p) for p in resolved]
        for i, a in enumerate(norms):
            for b in norms[i + 1 :]:
                if a == b or a.startswith(f"{b}{_PATH_SEP}") or b.startswith(f"{a}{_PATH_SEP}"):
                    raise ValueError(
                        f"replace() update paths overlap: {a!r} and {b!r} address "
                        f"the same subtree; replace the enclosing path once instead"
                    )
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

    def replace(self, _updates: Mapping[str, Any] | None = None, /, **updates: Any) -> Self:
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
        return self._rebuild_node(self._leaves_replaced(resolved), node_name=None)

    # -- Field renaming -------------------------------------------------------

    def _resolve_path_renames(
        self, mapping: Mapping[str, str] | None, kwargs: Mapping[str, str]
    ) -> dict[str, str]:
        """Resolve :meth:`with_path_names` inputs to ``{source path: target path}``.

        Each key is the exact path of a node, and each target is the node's new
        exact path.

        Raises
        ------
        KeyError
            If a key is not the path of a node, naming the paths the tree has.
        ValueError
            If a target is empty or has an empty segment, two keys resolve to the
            same node, no renames are given, a target lies inside its own node,
            or two targets are equal or one lies inside the other.
        """
        pairs: dict[str, str] = {}
        for source in (mapping or {}), kwargs:
            for old, new in source.items():
                if not isinstance(new, str) or not new:
                    raise ValueError(f"new path for {old!r} must be a non-empty string")
                if "" in new.split(_PATH_SEP):
                    raise ValueError(
                        f"new path {new!r} for {old!r} has an empty segment "
                        f"(no leading, trailing, or doubled {_PATH_SEP!r})"
                    )
                segments = self._split_path((old,))
                resolved = _PATH_SEP.join(segments)
                try:
                    self._node_at(segments)
                except KeyError:
                    raise KeyError(
                        f"with_path_names(): {resolved!r} is not the path of a node; the paths "
                        f"are {list(self._node_paths())}"
                    ) from None
                if resolved in pairs:
                    raise ValueError(f"node {resolved!r} is renamed more than once")
                pairs[resolved] = new
        if not pairs:
            raise ValueError("with_path_names() requires at least one rename")
        for resolved, new in pairs.items():
            if new.startswith(resolved + _PATH_SEP):
                raise ValueError(f"{resolved!r} cannot move into its own subtree, to {new!r}")
        targets = list(pairs.values())
        for index, target in enumerate(targets):
            for other in targets[index + 1 :]:
                if target == other:
                    raise ValueError(f"two nodes move to {target!r}, so they collide")
                if other.startswith(target + _PATH_SEP) or target.startswith(other + _PATH_SEP):
                    raise ValueError(f"the targets {target!r} and {other!r} overlap")
        return pairs

    def _renamed_tree(self, renames: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
        """This tree as a nested dict with *renames* applied, and each node's original path.

        *renames* is the output of :meth:`_resolve_path_renames`. The second
        result maps the new path of each node that the tree held to its
        original path; a group that a move creates has none.

        Raises
        ------
        ValueError
            If a target collides with a node that remains.
        """
        return _moved_tree(self.to_nested_dict(), renames)

    def _moved_leaf_paths(self, renames: Mapping[str, str]) -> dict[str, str]:
        """The new key of each field under *renames*, keyed by its key in canonical order.

        Raises
        ------
        ValueError
            If a target collides with a node that remains.
        """
        _, origins = self._renamed_tree(renames)
        moved = {origin: path for path, origin in origins.items()}
        return {key: moved[key] for key in self.keys()}

    def with_path_names(self, mapping: Mapping[str, str] | None = None, /, **kwargs: str) -> Self:
        """Return a same-family tree with the given nodes renamed or moved, ``old -> new``.

        Acts on the nodes *within* the tree, leaves or whole subtrees; the object
        itself is renamed by ``with_label`` on the tracked value types. Each key is
        the exact path of a node, so a keyword addresses a top-level node and the
        positional mapping any node. Each target is the node's new exact path,
        so a bare name is a top-level path::

            t.with_path_names(mu="loc")                  # rename a top-level node
            t.with_path_names({"g/mu": "g/loc"})         # rename a node within g
            t.with_path_names({"g/mu": "mu"})            # move it out of g
            t.with_path_names({"mu": "g/mu"})            # move it into g

        The renames apply simultaneously: every key is resolved against the
        original tree, so swaps are well-defined, and a node below a moved node
        moves with it unless it has a target of its own. A node whose target has
        its source's parent, and none of whose ancestors moves, is renamed in
        place and keeps its position. Every other node is moved: it is appended
        at the end of its new parent's children, in the order the renames are
        given, and a missing parent is created and appended the same way. A group
        that the moves leave without a field is removed. The leaf objects are
        unchanged, and the mapping interface stays keyed by full path.

        Raises
        ------
        KeyError
            If a key is not the path of a node.
        ValueError
            If a target is empty or has an empty segment, two keys rename the
            same node, no renames are given, a node moves into its own subtree,
            two targets are equal or one lies inside the other, or a target
            collides with a node that remains.
        """
        tree, _ = self._renamed_tree(self._resolve_path_renames(mapping, kwargs))
        return self._rebuild_node(tree, node_name=None)

    # -- Internal utilities -------------------------------------------------

    @staticmethod
    def _split_path(path: tuple[Any, ...]) -> tuple[str, ...]:
        """Normalise a path argument to a tuple of single-name segments.

        Accepts the forms the path-taking methods (``at_path`` / ``replace`` /
        ``with_path_names``) allow: a ``/``-delimited string (``"a/b"``),
        separate string segments (``"a", "b"``), or a single tuple of names
        (``("a", "b")``). Every segment must be a string; a non-string segment
        raises ``TypeError``.
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
