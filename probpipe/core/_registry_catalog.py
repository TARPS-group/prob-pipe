"""Registry catalog: discovery layer for ProbPipe's dispatch surfaces.

Provides :class:`RegistryCatalog`, a singleton index of every registry in
the process, and :class:`SupportsRegistryCataloging`, the protocol a
registry must satisfy to appear in it.

The catalog supplements dispatch — it does not replace per-registry
module-level singletons (``inference_method_registry``,
``converter_registry``, …).  Users who know which registry they want
keep using direct imports; the catalog is for discovery and indirection
at the REPL or in tooling that walks every registry uniformly.

Two checks, not one — *satisfying the protocol* vs *being in the catalog*:

- Satisfying :class:`SupportsRegistryCataloging` is a structural shape
  check (does the object have ``name: str``, ``description: str``,
  ``kind: str``, ``method_summaries(...)``, ``describe_method(...)``).
  Any :class:`~probpipe.core._registry.BaseDispatchRegistry` instance
  passes structurally, including a bare unnamed one.
- Being *in* the catalog is the runtime fact of whether
  :meth:`RegistryCatalog.register` was called.  Conforming dispatch
  registries call it from ``BaseDispatchRegistry.__init__`` iff a
  ``name`` was supplied.  Non-conforming registries (converters, the
  bijector facade) call it explicitly at module load.

The catalog enforces a stronger condition on :meth:`register` than the
protocol does — empty / duplicate names raise :class:`ValueError` — so a
structurally-conforming registry without a name can never be in the
catalog by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._registry import OPT_IN_ONLY_PRIORITY

__all__ = [
    "MethodSummary",
    "RegistryCatalog",
    "RegistryInfo",
    "SupportsRegistryCataloging",
    "registry_catalog",
]


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSummary:
    """One method's catalog record.

    Returned by :meth:`SupportsRegistryCataloging.method_summaries` and
    :meth:`SupportsRegistryCataloging.describe_method`.

    Parameters
    ----------
    name
        The method's unique name within its registry.
    priority
        Effective priority used by the registry's auto-dispatch walk.
        ``0`` (i.e. :data:`~probpipe.core._registry.OPT_IN_ONLY_PRIORITY`)
        means opt-in only.  ``None`` is used by factory-style registries
        (e.g. the bijector facade) that have no priority semantics.
    supported_types
        A registry-specific representation of the types this method
        accepts.  Unary dispatch registries pass a ``tuple[type, ...]``;
        binary dispatch registries pass ``(left_types, right_types)``;
        the converter adapter passes ``(source_types, target_types)``;
        the bijector facade passes the constraint key (type or instance)
        wrapped in a 1-tuple.  Rendering for human consumption belongs to
        :meth:`RegistryCatalog.describe`.
    description
        Optional one-line blurb from the method itself (read from
        :attr:`BaseDispatchMethod.description` for conforming methods).
    module_path
        ``type(method).__module__``, for "go to source" navigation.
    """

    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""

    @property
    def is_opt_in_only(self) -> bool:
        """``True`` if this method is excluded from auto-dispatch."""
        return self.priority == OPT_IN_ONLY_PRIORITY


@dataclass(frozen=True)
class RegistryInfo:
    """One registry's catalog-level identity record.

    Returned by :meth:`RegistryCatalog.list`.  Carries the *outside* view
    of a registry — the per-method detail lives in :class:`MethodSummary`.
    """

    name: str
    description: str
    kind: str
    method_count: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    """Contract a registry must satisfy to appear in :class:`RegistryCatalog`.

    Intentionally narrower than the dispatch contract carried by
    :class:`~probpipe.core._registry.BaseDispatchRegistry` so that
    *non-conforming* registries (the converter registry's
    ``(source_type, target_type)`` dispatch, the bijector facade's
    instance-first / MRO-fallback factory lookup) can satisfy it without
    changing their dispatch mechanics.

    Required surface
    ----------------
    Attributes:

    - ``name: str`` — unique identifier within the catalog.
    - ``description: str`` — short human-readable purpose statement.
    - ``kind: str`` — one of ``"dispatch"``, ``"factory"``,
      ``"converter"``, ``"other"``.  Used by the catalog's pretty-print
      and (eventually) by docs auto-generation.  Plain string, not enum,
      to leave room for plugin-introduced kinds without an enum extension.

    Methods:

    - ``method_summaries() -> list[MethodSummary]`` — full per-method
      catalog records, in whatever ordering the registry considers
      canonical (priority-descending for dispatch; insertion order for
      factories).
    - ``describe_method(name) -> MethodSummary`` — single lookup; raise
      ``KeyError`` if unknown.

    Notes
    -----
    The protocol intentionally does **not** require ``list_methods() ->
    list[str]``.  That method exists on
    :class:`~probpipe.core._registry.BaseDispatchRegistry` for dispatch
    users who need a cheap names list to pass as ``method="..."``, but it
    is a dispatch concern, not a cataloging concern — the catalog can
    derive names from ``method_summaries()`` whenever it needs them.
    """

    name: str
    description: str
    kind: str

    def method_summaries(self) -> list[MethodSummary]: ...

    def describe_method(self, name: str) -> MethodSummary: ...


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class RegistryCatalog:
    """Name-indexed catalog of every registry in the process.

    Auto-populated as registries that inherit
    :class:`~probpipe.core._registry.BaseDispatchRegistry` are instantiated
    with a ``name`` argument (the base self-registers in its ``__init__``).
    Non-conforming registries (converters, the bijector facade) register
    themselves explicitly at module load via :meth:`register`.

    The catalog is for discoverability only.  It does not dispatch and
    does not own dispatch state.  Per-registry module-level singletons
    remain the canonical entry points for code that knows which registry
    it wants.
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
                method_count=len(r.method_summaries()),
            )
            for r in (self._registries[n] for n in self.names())
        ]

    def describe(self, name: str) -> str:
        """Return a pretty-printed multi-line description of *name*.

        Opt-in-only methods (priority ``0``) are rendered in their own
        section after the auto-dispatchable ones, so a reader sees they
        exist but isn't surprised when they don't auto-fire.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        reg = self[name]
        summaries = reg.method_summaries()
        # "Opt-in only" only makes sense when priority semantics apply.
        # Factory-style registries (priority=None) have no opt-in concept;
        # everything they hold goes under the main "Methods" section.
        auto = [s for s in summaries if not s.is_opt_in_only]
        opt_in = [s for s in summaries if s.is_opt_in_only]

        header = f"{reg.name}  ({reg.kind})"
        if reg.description:
            header += f" — {reg.description}"
        lines = [header, ""]
        if auto:
            main_label = (
                "  Methods:" if reg.kind == "factory" else "  Auto-dispatched (by priority):"
            )
            lines.append(main_label)
            for s in auto:
                lines.append(_format_method_line(s))
            lines.append("")
        if opt_in:
            lines.append("  Opt-in only (reachable via method=...):")
            for s in opt_in:
                lines.append(_format_method_line(s))
            lines.append("")
        if not auto and not opt_in:
            lines.append("  (no methods registered)")
        return "\n".join(lines).rstrip()

    # -- delegation ---------------------------------------------------------

    def register_method(self, registry_name: str, method: Any) -> None:
        """Convenience indirection: ``self[registry_name].register(method)``.

        Lets tooling register a method without taking a direct dependency
        on the underlying registry's module.
        """
        self[registry_name].register(method)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._registries:
            return "RegistryCatalog(empty)"
        infos = self.list()
        name_w = max(len(i.name) for i in infos)
        kind_w = max(len(i.kind) for i in infos)
        rows = [
            f"  {i.name:<{name_w}}  {i.kind:<{kind_w}}  "
            f"{i.method_count:>3} method(s)" + (f"  — {i.description}" if i.description else "")
            for i in infos
        ]
        return "RegistryCatalog (\n" + "\n".join(rows) + "\n)"

    def _repr_html_(self) -> str:
        if not self._registries:
            return "<i>RegistryCatalog (empty)</i>"
        rows = "\n".join(
            f"<tr><td><code>{i.name}</code></td>"
            f"<td>{i.kind}</td>"
            f"<td>{i.method_count}</td>"
            f"<td>{i.description}</td></tr>"
            for i in self.list()
        )
        return (
            "<table>\n"
            "<thead><tr><th>Registry</th><th>Kind</th>"
            "<th>Methods</th><th>Description</th></tr></thead>\n"
            f"<tbody>\n{rows}\n</tbody>\n</table>"
        )


def _format_method_line(s: MethodSummary) -> str:
    """One-line rendering of a :class:`MethodSummary` for ``describe()``.

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
