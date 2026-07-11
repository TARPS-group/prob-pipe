"""Provenance tracking and graph utilities for tracked-term lineage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .config import ProvenanceMode, provenance_config

logger = logging.getLogger(__name__)

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
# ParentInfo descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentInfo:
    """Descriptor for a provenance parent, used in all non-OFF modes.

    Always carries enough information to describe lineage and traverse the
    ancestry DAG.  In FULL mode it additionally retains a live reference to
    the parent object via ``parent``.

    Attributes
    ----------
    type_name : str
        Class name of the parent (e.g. ``"EmpiricalDistribution"``).
    name : str or None
        Name of the parent tracked term.  ``None`` for unnamed parents
        (uncommon; most framework objects carry a name).
    provenance : Provenance or None
        The parent's own provenance node.  Kept in both LIGHTWEIGHT and
        FULL modes so the ancestry DAG remains traversable without holding
        the parent's data arrays alive.  Excluded from hashing (``Provenance``
        holds an unhashable ``metadata`` dict) but included in equality so
        that two descriptors for the same ancestor compare equal.
    fingerprint : str or None
        Stable 16-character hex digest of the parent's content, populated
        by :meth:`Provenance.create`.  ``None`` only when fingerprinting
        raises an unexpected error.  Intended as the foundation for a future
        Prefect ``cache_key_fn``.  Excluded from equality and hashing: descriptor
        identity is structural (``type_name`` / ``name`` / ``provenance``), so a
        content digest must not perturb ancestor-set dedup.
    parent : ProvenanceNode or None
        The live parent object.  Set in FULL mode; ``None`` in LIGHTWEIGHT
        so the parent's data can be garbage-collected.  Excluded from
        equality and hashing.
    """

    type_name: str
    name: str | None
    provenance: Provenance | None = field(default=None, hash=False)
    fingerprint: str | None = field(default=None, compare=False)
    parent: ProvenanceNode | None = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """How a tracked term was produced: an operation plus parent descriptors.

    Attached write-once to a tracked term via ``with_provenance``.

    Attributes
    ----------
    operation : str
        The operation that produced the object (e.g. ``"broadcast"``,
        ``"condition_on"``, ``"with_name"``).
    parents : tuple of ParentInfo
        Descriptors of the inputs the operation consumed.
    metadata : dict
        Optional scalar/string metadata about the operation (e.g. the old
        and new names of a rename). Serialized alongside the operation by
        :meth:`to_dict`.
    """

    operation: str
    parents: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parent_names = ", ".join(p.name or p.type_name for p in self.parents)
        return f"Provenance({self.operation!r}, parents=[{parent_names}])"

    # -- Serialization -----------------------------------------------------

    def to_dict(self, *, recurse: bool = True) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Parameters
        ----------
        recurse : bool
            If True, recursively serialize parent provenance chains via
            each parent's ``.provenance``.
        """
        parent_dicts = []
        for p in self.parents:
            entry: dict[str, Any] = {
                "type": p.type_name,
                "name": p.name,
            }
            if p.fingerprint is not None:
                entry["fingerprint"] = p.fingerprint
            if recurse and p.provenance is not None:
                entry["provenance"] = p.provenance.to_dict(recurse=True)
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
        call sites can pass the result directly to ``with_provenance()``
        without an extra guard — ``with_provenance(None)`` is a no-op.

        Parameters
        ----------
        operation:
            Provenance operation label (e.g. ``"broadcast"``).
        parents:
            Raw parent objects that carry a ``.provenance`` attribute
            (already filtered and deduplicated by the caller).
        metadata:
            Optional mapping of scalar/string metadata.
        """
        mode = provenance_config.mode
        if mode is ProvenanceMode.OFF:
            return None
        keep = mode is ProvenanceMode.FULL

        from ._fingerprint import fingerprint as _fingerprint

        def _make_parent(p: Any) -> ParentInfo:
            fp: str | None
            try:
                fp = _fingerprint(p)
            except Exception as exc:
                logger.warning(
                    "fingerprint() failed for %s %r: %s",
                    type(p).__name__,
                    getattr(p, "name", None),
                    exc,
                )
                fp = None
            return ParentInfo(
                type_name=type(p).__name__,
                name=getattr(p, "name", None),
                provenance=getattr(p, "provenance", None),
                fingerprint=fp,
                parent=p if keep else None,
            )

        refs = tuple(_make_parent(p) for p in parents)
        return cls(operation, parents=refs, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------


def _parent_key(p: Any) -> Any:
    """Stable dedup key for a parent node.

    Uses live-object identity in FULL mode (``p.parent`` is set), and a
    ``(type_name, name, id(provenance))`` tuple in LIGHTWEIGHT mode.  The
    parent's ``.provenance`` node is the same object on every path to the
    same ancestor, so its id is stable even though each path holds a
    distinct ``ParentInfo`` instance.

    Two distinct *root* parents (``provenance is None``) that share a type
    and name collapse to one key in LIGHTWEIGHT — an accepted limitation of
    dropping object identity; FULL keeps them distinct via ``id(p.parent)``.
    """
    if isinstance(p, ParentInfo):
        if p.parent is not None:
            return id(p.parent)
        return (p.type_name, p.name, id(p.provenance))
    return id(p)


def provenance_ancestors(node: ProvenanceNode) -> list[Any]:
    """Return all ancestor nodes reachable via provenance chains.

    Traverses ``node.provenance.parents`` recursively (breadth-first) and
    returns a flat list of unique ancestors, ordered by discovery.
    The input *node* is **not** included in the result.

    Returns :class:`ParentInfo` descriptors.  In FULL mode the live parent
    object is accessible via ``ancestor.parent``; in LIGHTWEIGHT
    ``ancestor.parent`` is ``None``.
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

    if node.provenance is not None:
        for p in node.provenance.parents:
            _enqueue(p)

    while queue:
        current = queue.pop(0)
        current_provenance = current.provenance
        if current_provenance is not None:
            for p in current_provenance.parents:
                _enqueue(p)

    return ancestors


def provenance_dag(dist: Distribution):
    """Build a Graphviz ``Digraph`` of the provenance chain rooted at *dist*.

    Each node is labelled with its type and name.  Edges point from parent
    to child and are labelled with the operation that produced the child.
    Works in all modes that attach provenance (FULL and LIGHTWEIGHT).

    Requires the ``graphviz`` package.  Returns a ``graphviz.Digraph``
    instance that can be rendered or displayed in a notebook.
    """
    try:
        from graphviz import Digraph
    except ImportError as e:
        raise ImportError(
            "graphviz is required for provenance_dag(). Install it with: pip install graphviz"
        ) from e

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
            if p.provenance is not None:
                for pp in p.provenance.parents:
                    _visit_parent(pp, nid, p.provenance.operation)
        dot.edge(nid, child_nid, label=operation)

    def _visit_dist(d: Distribution) -> str:
        nid = str(id(d))
        if id(d) in visited:
            return nid
        visited.add(id(d))
        dot.node(nid, _label(type(d).__name__, d.name or ""))
        if d.provenance is not None:
            for p in d.provenance.parents:
                _visit_parent(p, nid, d.provenance.operation)
        return nid

    _visit_dist(dist)
    return dot
