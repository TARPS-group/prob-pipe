"""Provenance tracking and graph utilities for distribution lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .distribution import Distribution

__all__ = ["Provenance", "provenance_ancestors", "provenance_dag"]


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    """Tracks how a distribution was created."""

    operation: str
    parents: tuple[Distribution, ...] = ()
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
            If True, recursively serialize parent provenance chains.
            If False, only include parent type/name references.
        """
        parent_dicts = []
        for p in self.parents:
            entry: dict[str, Any] = {
                "type": type(p).__name__,
                "name": p.name,
            }
            if recurse and p.source is not None:
                entry["source"] = p.source.to_dict(recurse=True)
            parent_dicts.append(entry)

        # Filter metadata to JSON-serializable values
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

def provenance_ancestors(dist: Distribution) -> list[Distribution]:
    """Return all ancestor distributions reachable via provenance chains.

    Traverses ``dist.source.parents`` recursively (breadth-first) and
    returns a flat list of unique ancestor distributions, ordered by
    discovery.  The input *dist* is **not** included in the result.
    """
    visited: set[int] = {id(dist)}
    ancestors: list[Distribution] = []
    queue: list[Distribution] = []

    if dist.source is not None:
        for p in dist.source.parents:
            if id(p) not in visited:
                visited.add(id(p))
                queue.append(p)
                ancestors.append(p)

    while queue:
        current = queue.pop(0)
        if current.source is not None:
            for p in current.source.parents:
                if id(p) not in visited:
                    visited.add(id(p))
                    queue.append(p)
                    ancestors.append(p)

    return ancestors


def provenance_dag(dist: Distribution):
    """Build a Graphviz ``Digraph`` of the provenance chain rooted at *dist*.

    Each node is a distribution (labelled with type and name).  Edges point
    from parent to child and are labelled with the operation that produced
    the child.

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

    visited: set[int] = set()

    def _label(d: Distribution) -> str:
        name = d.name or ""
        typename = type(d).__name__
        if name:
            return f"{typename}\n'{name}'"
        return typename

    def _visit(d: Distribution) -> str:
        nid = str(id(d))
        if id(d) in visited:
            return nid
        visited.add(id(d))
        dot.node(nid, _label(d))

        if d.source is not None:
            for p in d.source.parents:
                pid = _visit(p)
                dot.edge(pid, nid, label=d.source.operation)

        return nid

    _visit(dist)
    return dot
