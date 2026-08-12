"""Registry catalog: discovery layer for ProbPipe's dispatch surfaces.

Provides :class:`RegistryCatalog`, a singleton index of every registry in
the process, and :class:`SupportsRegistryCataloging`, the protocol a
registry must satisfy to appear in it.

The catalog supplements dispatch — it does not replace per-registry
module-level singletons (``inference_method_registry``,
``converter_registry``, …).  Users who know which registry they want
keep using direct imports; the catalog is for discovery and indirection
at the REPL or in tooling that walks every registry uniformly.

Notes
-----
Satisfying :class:`SupportsRegistryCataloging` (a structural shape check)
is distinct from *being in* the catalog (the runtime fact that
:meth:`RegistryCatalog.register` was called): any
:class:`~probpipe.core._registry.BaseDispatchRegistry` passes the shape
check structurally, but conforming dispatch registries only self-register
when given a ``name`` and non-conforming registries (converters, the
bijector facade) register explicitly at module load.  :meth:`register`
enforces a stronger condition than the protocol — empty / duplicate names
raise :class:`ValueError` — so a structurally-conforming registry without
a name can never land in the catalog by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._registry import OPT_IN_ONLY_PRIORITY

__all__ = [
    "EntrySummary",
    "RegistryCatalog",
    "RegistryInfo",
    "SupportsRegistryCataloging",
    "registry_catalog",
]


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntrySummary:
    """One catalog entry's record.

    An *entry* is one registered item — an inference method, a converter,
    or a bijector factory, depending on the registry.  Returned by
    :meth:`SupportsRegistryCataloging.entry_summaries` and
    :meth:`SupportsRegistryCataloging.describe_entry`.

    Parameters
    ----------
    name
        The entry's unique name within its registry.
    priority
        Effective priority used by the registry's auto-dispatch walk.
        ``0`` (i.e. :data:`~probpipe.core._registry.OPT_IN_ONLY_PRIORITY`)
        means opt-in only.  ``None`` is used by factory-style registries
        (e.g. the bijector facade) that have no priority semantics.
    supported_types
        A registry-specific representation of the types this entry
        accepts.  Unary dispatch registries pass a ``tuple[type, ...]``;
        binary dispatch registries pass ``(left_types, right_types)``;
        the converter adapter passes ``(source_types, target_types)``;
        the bijector facade passes the constraint key (type or instance)
        wrapped in a 1-tuple.  Rendering for human consumption belongs to
        :meth:`RegistryCatalog.describe`.
    description
        Optional one-line blurb from the entry itself (read from
        :attr:`BaseDispatchMethod.description` for conforming methods).
    module_path
        ``type(entry).__module__``, for "go to source" navigation.
    """

    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""

    @property
    def is_opt_in_only(self) -> bool:
        """``True`` if this entry is excluded from auto-dispatch."""
        return self.priority == OPT_IN_ONLY_PRIORITY


@dataclass(frozen=True)
class RegistryInfo:
    """One registry's catalog-level identity record.

    Returned by :meth:`RegistryCatalog.list`.  Carries the *outside* view
    of a registry — the per-entry detail lives in :class:`EntrySummary`.
    """

    name: str
    description: str
    kind: str
    entry_count: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    """Contract a registry satisfies to appear in :class:`RegistryCatalog`.

    An *entry* is one registered item — an inference method, a converter,
    or a bijector factory, depending on the registry.  A registry
    participates in the catalog by exposing three identity attributes and
    two introspection methods over its entries:

    Attributes
    ----------
    name : str
        Unique identifier within the catalog.
    description : str
        Short human-readable purpose statement.
    kind : str
        One of ``"dispatch"``, ``"factory"``, ``"converter"``, ``"other"``
        — used by the catalog's pretty-print.  A plain string (not an
        enum) so plugins can introduce new kinds freely.

    Methods
    -------
    entry_summaries() -> list[EntrySummary]
        Per-entry catalog records, in the registry's canonical order
        (priority-descending for dispatch; name/insertion order for
        factories).
    describe_entry(name) -> EntrySummary
        Single entry lookup; raises ``KeyError`` if *name* is unknown.

    Notes
    -----
    The protocol is intentionally narrower than the dispatch contract on
    :class:`~probpipe.core._registry.BaseDispatchRegistry`, so registries
    with a different dispatch shape — the converter registry's
    ``(source_type, target_type)`` lookup, the bijector facade's
    instance-first / MRO-fallback factory — satisfy it without changing
    their dispatch mechanics.  It deliberately does **not** require
    ``list_methods()``: that is a dispatch-side convenience, and the
    catalog derives names from ``entry_summaries()`` when needed.
    """

    name: str
    description: str
    kind: str

    def entry_summaries(self) -> list[EntrySummary]: ...

    def describe_entry(self, name: str) -> EntrySummary: ...


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class RegistryCatalog:
    """Name-indexed catalog of every registry in the process.

    Discoverability only — the catalog does not dispatch and owns no
    dispatch state.  Per-registry singletons (``inference_method_registry``,
    ``converter_registry``, …) stay the canonical entry points for code
    that knows which registry it wants.

    Examples
    --------
    >>> import probpipe
    >>> probpipe.registry_catalog.names()
    ['bijectors', 'converters', 'inference']
    >>> print(probpipe.registry_catalog.describe("inference"))
    inference  (dispatch) — Inference-method dispatch for condition_on.
    ...

    Notes
    -----
    Auto-populated as :class:`~probpipe.core._registry.BaseDispatchRegistry`
    subclasses are constructed with a ``name`` (the base self-registers in
    ``__init__``); non-conforming registries register explicitly at module
    load via :meth:`register`.
    """

    def __init__(self) -> None:
        self._registries: dict[str, SupportsRegistryCataloging] = {}

    # -- registration -------------------------------------------------------

    def register(self, registry: SupportsRegistryCataloging) -> None:
        """Add *registry* to the catalog.

        Raises
        ------
        ValueError
            If *registry* has an empty ``name`` or a name already
            registered in the catalog.
        """
        name = registry.name
        if not name:
            raise ValueError(
                "Cannot register a registry without a non-empty name; "
                f"got {name!r} for {type(registry).__name__}."
            )
        if name in self._registries:
            existing = type(self._registries[name]).__name__
            new = type(registry).__name__
            raise ValueError(
                f"Registry name {name!r} is already registered (existing: {existing}, new: {new})."
            )
        self._registries[name] = registry

    # -- query --------------------------------------------------------------

    def __getitem__(self, name: str) -> SupportsRegistryCataloging:
        """Look up a registry by name.  ``KeyError`` if missing."""
        try:
            return self._registries[name]
        except KeyError:
            available = ", ".join(sorted(self._registries)) or "(none)"
            raise KeyError(f"No registry named {name!r}. Available: {available}") from None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._registries

    def names(self) -> list[str]:
        """Return all registered names in sorted order."""
        return sorted(self._registries)

    def list(self) -> list[RegistryInfo]:
        """Return one :class:`RegistryInfo` per registered registry.

        Sorted by name for deterministic output.
        """
        return [
            RegistryInfo(
                name=r.name,
                description=r.description,
                kind=r.kind,
                entry_count=len(r.entry_summaries()),
            )
            for r in (self._registries[n] for n in self.names())
        ]

    def describe(self, name: str) -> str:
        """Return a pretty-printed multi-line description of *name*.

        Opt-in-only entries (priority ``0``) are rendered in their own
        section after the auto-dispatchable ones, so a reader sees they
        exist but isn't surprised when they don't auto-fire.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        reg = self[name]
        summaries = reg.entry_summaries()
        # "Opt-in only" only makes sense when priority semantics apply.
        # Factory-style registries (priority=None) have no opt-in concept;
        # everything they hold goes under the main "Entries" section.
        auto = [s for s in summaries if not s.is_opt_in_only]
        opt_in = [s for s in summaries if s.is_opt_in_only]

        header = f"{reg.name}  ({reg.kind})"
        if reg.description:
            header += f" — {reg.description}"
        lines = [header, ""]
        if auto:
            main_label = (
                "  Entries:" if reg.kind == "factory" else "  Auto-dispatched (by priority):"
            )
            lines.append(main_label)
            for s in auto:
                lines.append(_format_entry_line(s))
            lines.append("")
        if opt_in:
            lines.append("  Opt-in only (reachable via method=...):")
            for s in opt_in:
                lines.append(_format_entry_line(s))
            lines.append("")
        if not auto and not opt_in:
            lines.append("  (no entries registered)")
        return "\n".join(lines).rstrip()

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._registries:
            return "RegistryCatalog(empty)"
        infos = self.list()
        name_w = max(len(i.name) for i in infos)
        kind_w = max(len(i.kind) for i in infos)
        rows = [
            f"  {i.name:<{name_w}}  {i.kind:<{kind_w}}  "
            f"{i.entry_count:>3} {'entry' if i.entry_count == 1 else 'entries'}"
            + (f"  — {i.description}" if i.description else "")
            for i in infos
        ]
        return "RegistryCatalog (\n" + "\n".join(rows) + "\n)"

    def _repr_html_(self) -> str:
        if not self._registries:
            return "<i>RegistryCatalog (empty)</i>"
        rows = "\n".join(
            f"<tr><td><code>{i.name}</code></td>"
            f"<td>{i.kind}</td>"
            f"<td>{i.entry_count}</td>"
            f"<td>{i.description}</td></tr>"
            for i in self.list()
        )
        return (
            "<table>\n"
            "<thead><tr><th>Registry</th><th>Kind</th>"
            "<th>Entries</th><th>Description</th></tr></thead>\n"
            f"<tbody>\n{rows}\n</tbody>\n</table>"
        )


def _format_entry_line(s: EntrySummary) -> str:
    """One-line rendering of an :class:`EntrySummary` for ``describe()``.

    Priority is shown numerically (or ``"-"`` for factory-style),
    followed by the name, the description if any, and the module path in
    parentheses.
    """
    prio = "  -" if s.priority is None else f"{s.priority:>3}"
    line = f"    {prio}  {s.name}"
    if s.description:
        line += f"  — {s.description}"
    if s.module_path:
        line += f"  ({s.module_path})"
    return line


# Module-level singleton — the canonical entry point.
registry_catalog = RegistryCatalog()
