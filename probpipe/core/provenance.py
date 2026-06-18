"""Provenance tracking and graph utilities for distribution lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._record_array import RecordArray
    from .distribution import Distribution
    from .record import Record

    ProvenanceNode = Distribution | Record | RecordArray

from .config import ProvenanceMode, provenance_config  # noqa: E402

__all__ = [
    "ParentInfo",
    "Provenance",
    "ProvenanceMode",
    "provenance_ancestors",
    "provenance_dag",
]


# ---------------------------------------------------------------------------
# ParentInfo descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParentInfo:
    """Descriptor for a provenance parent, used in all non-OFF modes.

    Always carries enough information to describe lineage and traverse the
    ancestry DAG.  In FULL mode it additionally retains a live reference to
    the parent object via ``obj``.

    Attributes
    ----------
    type_name : str
        Class name of the parent (e.g. ``"EmpiricalDistribution"``).
    name : str or None
        Distribution or record name of the parent.  ``None`` for unnamed
        parents (uncommon; most framework objects carry a name).
    source : Provenance or None
        The parent's own provenance node.  Kept in both LIGHTWEIGHT and
        FULL modes so the ancestry DAG remains traversable without holding
        the parent's data arrays alive.  Excluded from hashing (``Provenance``
        holds an unhashable ``metadata`` dict) but included in equality so
        that two descriptors for the same ancestor compare equal.
    fingerprint : str or None
        Optional stable content hash of the parent's inputs.  Intended for
        a future workflow-result caching layer; currently always ``None``.
    obj : ProvenanceNode or None
        The live parent object.  Set in FULL mode; ``None`` in LIGHTWEIGHT
        so the parent's data can be garbage-collected.  Excluded from
        equality and hashing.
    """

    type_name: str
    name: str | None
    source: Provenance | None = field(default=None, hash=False)
    fingerprint: str | None = None
    obj: ProvenanceNode | None = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Tracks how a distribution was created."""

    operation: str
    parents: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parent_names = ", ".join(
            p.name or type(p).__name__ for p in self.parents
        )
        return f"Provenance({self.operation!r}, parents=[{parent_names}])"

    # -- Serialization -----------------------------------------------------

    def to_dict(self, *, recurse: bool = True) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Parameters
        ----------
        recurse : bool
            If True, recursively serialize parent provenance chains via
            each parent's ``.source``.
        """
        parent_dicts = []
        for p in self.parents:
            entry: dict[str, Any] = {
                "type": p.type_name,
                "name": p.name,
            }
            if p.fingerprint is not None:
                entry["fingerprint"] = p.fingerprint
            if recurse and p.source is not None:
                entry["source"] = p.source.to_dict(recurse=True)
            parent_dicts.append(entry)

        safe_metadata = {}
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                safe_metadata[k] = v
            else:
                safe_metadata[k] = str(v)

        return {
            "operation": self.operation,
            "parents": parent_dicts,
            "metadata": safe_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        """Reconstruct from a dict produced by :meth:`to_dict`.

        Parent distributions are not available at deserialization time, so
        ``parents`` will be an empty tuple.  The parent information is
        preserved in the dict under ``"parents"`` for inspection.
        """
        return cls(
            operation=d["operation"],
            parents=(),
            metadata={**d.get("metadata", {}), "_parents_info": d.get("parents", [])},
        )

    @classmethod
    def create(
        cls,
        operation: str,
        parents: tuple | list = (),
        metadata: dict[str, Any] | None = None,
    ) -> Provenance | None:
        """Build provenance respecting the global :attr:`~probpipe.provenance_config` mode.

        Returns ``None`` when the mode is :attr:`ProvenanceMode.OFF` so that
        call sites can pass the result directly to ``with_source()`` without
        an extra guard — ``with_source(None)`` is a no-op.

        Parameters
        ----------
        operation:
            Provenance operation label (e.g. ``"broadcast"``).
        parents:
            Raw parent objects that carry a ``.source`` attribute (already
            filtered and deduplicated by the caller).
        metadata:
            Optional mapping of scalar/string metadata.
        """
        mode = provenance_config.mode
        if mode is ProvenanceMode.OFF:
            return None
        keep = mode is ProvenanceMode.FULL
        refs = tuple(
            ParentInfo(
                type_name=type(p).__name__,
                name=getattr(p, "name", None),
                source=getattr(p, "source", None),
                obj=p if keep else None,
            )
            for p in parents
        )
        return cls(operation, parents=refs, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def _parent_key(p: Any) -> Any:
    """Stable dedup key for a parent node.

    Uses live-object identity in FULL mode (``p.obj`` is set), and a
    ``(type_name, name, id(source))`` tuple in LIGHTWEIGHT mode.  The
    parent's ``.source`` Provenance node is the same object on every path
    to the same ancestor, so its id is stable even though each path holds
    a distinct ``ParentInfo`` instance.
    """
    if isinstance(p, ParentInfo):
        if p.obj is not None:
            return id(p.obj)
        return (p.type_name, p.name, id(p.source))
    return id(p)


def provenance_ancestors(node: "ProvenanceNode") -> list[Any]:
    """Return all ancestor nodes reachable via provenance chains.

    Traverses ``node.source.parents`` recursively (breadth-first) and
    returns a flat list of unique ancestors, ordered by discovery.
    The input *node* is **not** included in the result.

    Returns :class:`ParentInfo` descriptors.  In FULL mode the live parent
    object is accessible via ``ancestor.obj``; in LIGHTWEIGHT ``ancestor.obj``
    is ``None``.
    """
    visited: set = {id(node)}
    ancestors: list = []
    queue: list = []

    def _enqueue(p: Any) -> None:
        key = _parent_key(p)
        if key not in visited:
            visited.add(key)
            queue.append(p)
            ancestors.append(p)

    if node.source is not None:
        for p in node.source.parents:
            _enqueue(p)

    while queue:
        current = queue.pop(0)
        current_source = current.source
        if current_source is not None:
            for p in current_source.parents:
                _enqueue(p)

    return ancestors


def provenance_dag(dist: "Distribution"):
    """Build a Graphviz ``Digraph`` of the provenance chain rooted at *dist*.

    Each node is labelled with its type and name.  Edges point from parent
    to child and are labelled with the operation that produced the child.
    Works in all modes that attach provenance (FULL and LIGHTWEIGHT).

    Requires the ``graphviz`` package.  Returns a ``graphviz.Digraph``
    instance that can be rendered or displayed in a notebook.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "graphviz is required for provenance_dag(). "
            "Install it with: pip install graphviz"
        )

    dot = Digraph(comment="Provenance DAG")
    dot.attr(rankdir="BT")  # bottom-to-top: parents below children

    visited: set = set()

    def _label(type_name: str, name: str) -> str:
        return f"{type_name}\n'{name}'" if name else type_name

    def _stable_nid(p: Any) -> str:
        """Graphviz node ID that is the same for all ParentInfo of the same ancestor."""
        key = _parent_key(p)
        if isinstance(key, tuple):
            return ":".join(str(x) for x in key)
        return str(key)

    def _visit_parent(p: Any, child_nid: str, operation: str) -> None:
        """Render a parent (ParentInfo or live object) and recurse."""
        key = _parent_key(p)
        nid = _stable_nid(p)
        if key not in visited:
            visited.add(key)
            dot.node(nid, _label(p.type_name, p.name))
            if p.source is not None:
                for pp in p.source.parents:
                    _visit_parent(pp, nid, p.source.operation)
        dot.edge(nid, child_nid, label=operation)

    def _visit_dist(d: "Distribution") -> str:
        nid = str(id(d))
        if id(d) in visited:
            return nid
        visited.add(id(d))
        dot.node(nid, _label(type(d).__name__, d.name or ""))
        if d.source is not None:
            for p in d.source.parents:
                _visit_parent(p, nid, d.source.operation)
        return nid

    _visit_dist(dist)
    return dot
