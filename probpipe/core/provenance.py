"""Provenance graph utilities for tracing distribution lineage."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .distribution import Distribution

__all__ = ["provenance_ancestors", "provenance_dag"]


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
