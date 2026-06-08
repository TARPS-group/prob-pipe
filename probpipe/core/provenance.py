"""Provenance tracking and graph utilities for distribution lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._record_array import RecordArray
    from .distribution import Distribution
    from .record import Record

    ProvenanceNode = Distribution | Record | RecordArray

__all__ = [
    "ParentInfo",
    "Provenance",
    "ProvenanceMode",
    "provenance_ancestors",
    "provenance_dag",
]


# ---------------------------------------------------------------------------
# ProvenanceMode enum
# ---------------------------------------------------------------------------

class ProvenanceMode(Enum):
    """Controls how much history is retained in provenance chains.

    Members
    -------
    FULL
        Store live references to parent Distribution / Record /
        RecordArray objects.  The entire ancestry chain stays in memory
        as long as the final result is alive.  Good for debugging and
        small test workflows where full graph traversal is useful.
    LIGHTWEIGHT
        Store only lightweight :class:`ParentInfo` descriptors — type
        name, distribution name, and an optional fingerprint.  Parent
        objects are free to be garbage-collected once a workflow step
        completes.  This is the default and scales to larger workflows.
    OFF
        Attach no provenance at all.  Minimises overhead when lineage
        tracking is not needed.
    """

    FULL = "full"
    LIGHTWEIGHT = "lightweight"
    OFF = "off"


# ---------------------------------------------------------------------------
# ParentInfo descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParentInfo:
    """Lightweight descriptor for a provenance parent.

    Used by :attr:`ProvenanceMode.LIGHTWEIGHT` in place of a live object
    reference.  Stores just enough information to describe lineage without
    keeping the parent's data alive.

    Attributes
    ----------
    type_name : str
        Class name of the parent (e.g. ``"EmpiricalDistribution"``).
    name : str
        Distribution or record name of the parent.
    fingerprint : str or None
        Optional stable hash of the parent's inputs.  Reserved for future
        use by the Prefect caching layer (PR 2).
    """

    type_name: str
    name: str
    fingerprint: str | None = None


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
            p.name if isinstance(p, ParentInfo) else (p.name or type(p).__name__)
            for p in self.parents
        )
        return f"Provenance({self.operation!r}, parents=[{parent_names}])"

    # -- Serialization -----------------------------------------------------

    def to_dict(self, *, recurse: bool = True) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Parameters
        ----------
        recurse : bool
            If True, recursively serialize parent provenance chains.
            Only applies to live-reference parents (FULL mode); ParentInfo
            descriptors are always serialized shallowly.
        """
        parent_dicts = []
        for p in self.parents:
            if isinstance(p, ParentInfo):
                entry: dict[str, Any] = {
                    "type": p.type_name,
                    "name": p.name,
                }
                if p.fingerprint is not None:
                    entry["fingerprint"] = p.fingerprint
            else:
                entry = {
                    "type": type(p).__name__,
                    "name": p.name,
                }
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


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def provenance_ancestors(node: "ProvenanceNode") -> list["ProvenanceNode"]:
    """Return all ancestor nodes reachable via provenance chains.

    Traverses ``node.source.parents`` recursively (breadth-first) and
    returns a flat list of unique ancestors, ordered by discovery.
    The input *node* is **not** included in the result.

    Only works in :attr:`ProvenanceMode.FULL` mode where live object
    references are stored.  Returns an empty list when parents are
    :class:`ParentInfo` descriptors (LIGHTWEIGHT mode).
    """
    visited: set[int] = {id(node)}
    ancestors: list = []
    queue: list = []

    if node.source is not None:
        for p in node.source.parents:
            if isinstance(p, ParentInfo):
                continue
            if id(p) not in visited:
                visited.add(id(p))
                queue.append(p)
                ancestors.append(p)

    while queue:
        current = queue.pop(0)
        if current.source is not None:
            for p in current.source.parents:
                if isinstance(p, ParentInfo):
                    continue
                if id(p) not in visited:
                    visited.add(id(p))
                    queue.append(p)
                    ancestors.append(p)

    return ancestors


def provenance_dag(dist: "Distribution"):
    """Build a Graphviz ``Digraph`` of the provenance chain rooted at *dist*.

    Each node is a distribution (labelled with type and name).  Edges point
    from parent to child and are labelled with the operation that produced
    the child.

    Only works in :attr:`ProvenanceMode.FULL` mode.  Requires the
    ``graphviz`` package.  Returns a ``graphviz.Digraph`` instance that can
    be rendered or displayed in a notebook.
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

    visited: set[int] = set()

    def _label(d: "Distribution") -> str:
        name = d.name or ""
        typename = type(d).__name__
        if name:
            return f"{typename}\n'{name}'"
        return typename

    def _visit(d: "Distribution") -> str:
        nid = str(id(d))
        if id(d) in visited:
            return nid
        visited.add(id(d))
        dot.node(nid, _label(d))

        if d.source is not None:
            for p in d.source.parents:
                if isinstance(p, ParentInfo):
                    continue
                pid = _visit(p)
                dot.edge(pid, nid, label=d.source.operation)

        return nid

    _visit(dist)
    return dot