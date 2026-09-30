"""The registry catalog: one place to discover every registry.

:data:`registry_catalog` lists the registries in the process, their entries
with each entry's exactness and priority, and a one-line description of
each, so a user can see which implementations exist and how a call will
resolve. An **entry** is one registered item within a registry: an
inference method, a converter, or a bijector factory, depending on the
registry.

The catalog supplements the per-registry singletons
(``inference_method_registry``, ``converter_registry``, ...), which stay the
entry points for code that knows which registry it wants; it never
dispatches.

A registry can be cataloged if it implements
:class:`SupportsRegistryCataloging`. Satisfying the protocol is structural,
and membership requires an explicit :meth:`RegistryCatalog.register`, which
rejects an empty or duplicate name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "EntrySummary",
    "RegistryCatalog",
    "RegistryInfo",
    "SupportsRegistryCataloging",
    "registry_catalog",
]


@dataclass(frozen=True)
class EntrySummary:
    """What the catalog records about one entry of a registry.

    An *entry* is one registered item: an inference method, a converter, or
    a bijector factory, depending on the registry. Returned by
    :meth:`SupportsRegistryCataloging.entry_summaries` and
    :meth:`SupportsRegistryCataloging.describe_entry`.

    Attributes
    ----------
    name : str
        The entry's name, unique within its registry.
    priority : int or None
        The entry's effective rank among entries of the same exactness,
        higher first. ``None`` in a dispatch registry is opt-in-only; a
        registry that does not rank its entries, such as a factory, reports
        ``None`` for every entry.
    supported_types : tuple
        The registry's representation of what the entry admits: a tuple of
        classes for a unary dispatch registry, a ``(left_types,
        right_types)`` pair for a binary one, ``(source_types,
        target_types)`` for the converters, and the constraint key in a
        1-tuple for the bijectors.
    description : str
        A one-line description of the entry, empty when it declares none.
    module_path : str
        The module defining the entry's class.
    exact : bool or None
        The entry's declared exactness, or ``None`` when the registry does
        not declare exactness per entry.
    """

    name: str
    priority: int | None
    supported_types: tuple[Any, ...] = ()
    description: str = ""
    module_path: str = ""
    exact: bool | None = None

    @property
    def is_opt_in_only(self) -> bool:
        """``True`` when ``priority`` is ``None``.

        In a dispatch registry such an entry is skipped by automatic
        selection and reachable only by ``method="..."``.
        """
        return self.priority is None


@dataclass(frozen=True)
class RegistryInfo:
    """What the catalog records about one registry.

    Returned by :meth:`RegistryCatalog.list`; the per-entry detail is in
    :class:`EntrySummary`.

    Attributes
    ----------
    name : str
        The registry's name, unique within the catalog.
    description : str
        A one-line description of the registry.
    kind : str
        The registry's kind, such as ``"dispatch"``, ``"factory"``, or
        ``"converter"``.
    entry_count : int
        The number of entries the registry holds.
    """

    name: str
    description: str
    kind: str
    entry_count: int


@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    """What a registry implements to be cataloged.

    Attributes
    ----------
    name : str
        The registry's name, unique within the catalog.
    description : str
        A one-line description of the registry.
    kind : str
        The registry's kind, such as ``"dispatch"``, ``"factory"``, or
        ``"converter"``. A plain string, so a plugin can introduce a kind.

    Methods
    -------
    entry_summaries() -> list[EntrySummary]
        One summary per entry, in the registry's own order: selection order
        before type specificity for a dispatch registry, name order for a
        factory.
    describe_entry(name) -> EntrySummary
        The summary of one entry; raises ``KeyError`` for an unknown name.

    Notes
    -----
    The protocol states identity and introspection only, not dispatch, so a
    registry whose lookup does not fit
    :class:`~probpipe.core._dispatch.BaseDispatchRegistry`, such as the
    converter registry's ``(source, target_type)`` lookup or the bijector
    factory's instance-first lookup, is cataloged without changing how it
    dispatches. Satisfying the protocol does not place a registry in the
    catalog; :meth:`RegistryCatalog.register` does.
    """

    name: str
    description: str
    kind: str

    def entry_summaries(self) -> list[EntrySummary]: ...

    def describe_entry(self, name: str) -> EntrySummary: ...


class RegistryCatalog:
    """A name-indexed catalog of registries.

    The catalog records registries and describes them; it holds no dispatch
    state.

    Examples
    --------
    >>> import probpipe
    >>> probpipe.registry_catalog.names()
    ['bijectors', 'converters', 'inference']
    >>> print(probpipe.registry_catalog.describe("inference"))  # doctest: +SKIP
    inference  (dispatch) — Inference-method dispatch for condition_on.
    ...
    """

    def __init__(self) -> None:
        self._registries: dict[str, SupportsRegistryCataloging] = {}

    def register(self, registry: SupportsRegistryCataloging) -> None:
        """Add a registry to the catalog.

        Parameters
        ----------
        registry : SupportsRegistryCataloging
            The registry to add, under its ``name``.

        Raises
        ------
        ValueError
            If ``registry.name`` is empty or already registered.
        """
        name = registry.name
        if not name:
            raise ValueError(
                f"Cannot catalog a registry without a name; got {name!r} "
                f"for {type(registry).__name__}"
            )
        if name in self._registries:
            existing = type(self._registries[name]).__name__
            raise ValueError(
                f"Registry name {name!r} is already registered "
                f"(existing: {existing}, new: {type(registry).__name__})"
            )
        self._registries[name] = registry

    def __getitem__(self, name: str) -> SupportsRegistryCataloging:
        """The registry named ``name``; ``KeyError`` if there is none."""
        try:
            return self._registries[name]
        except KeyError:
            available = ", ".join(sorted(self._registries)) or "(none)"
            raise KeyError(f"No registry named {name!r}. Available: {available}") from None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._registries

    def names(self) -> list[str]:
        """Every registry name, sorted."""
        return sorted(self._registries)

    def list(self) -> list[RegistryInfo]:
        """One :class:`RegistryInfo` per registry, sorted by name."""
        return [
            RegistryInfo(
                name=registry.name,
                description=registry.description,
                kind=registry.kind,
                entry_count=len(registry.entry_summaries()),
            )
            for registry in (self._registries[name] for name in self.names())
        ]

    def describe(self, name: str) -> str:
        """A readable summary of the registry named ``name``.

        A factory's entries are listed together. Any other registry's are
        listed in its own order with opt-in-only entries in a section of
        their own, so a reader sees that they exist and that automatic
        selection skips them. An entry's exactness is shown when its
        registry declares one.

        Parameters
        ----------
        name : str
            A registered registry name.

        Returns
        -------
        str
            The summary, one line per entry under a header line.

        Raises
        ------
        KeyError
            If no registry is registered under ``name``.
        """
        registry = self[name]
        summaries = registry.entry_summaries()
        show_exactness = any(summary.exact is not None for summary in summaries)
        if registry.kind == "factory":
            sections = [("Entries:", summaries)]
        else:
            sections = [
                (
                    "Auto-dispatched (exact first, then by priority):"
                    if show_exactness
                    else "Auto-dispatched (by priority):",
                    [s for s in summaries if not s.is_opt_in_only],
                ),
                (
                    "Opt-in only (reachable via method=...):",
                    [s for s in summaries if s.is_opt_in_only],
                ),
            ]
        header = f"{registry.name}  ({registry.kind})"
        if registry.description:
            header += f" — {registry.description}"
        lines = [header, ""]
        for label, entries in sections:
            if not entries:
                continue
            lines.append(f"  {label}")
            lines.extend(_format_entry_line(entry, show_exactness) for entry in entries)
            lines.append("")
        if not summaries:
            lines.append("  (no entries registered)")
        return "\n".join(lines).rstrip()

    def __repr__(self) -> str:
        if not self._registries:
            return "RegistryCatalog(empty)"
        infos = self.list()
        name_width = max(len(info.name) for info in infos)
        kind_width = max(len(info.kind) for info in infos)
        rows = [
            f"  {info.name:<{name_width}}  {info.kind:<{kind_width}}  "
            f"{info.entry_count:>3} {'entry' if info.entry_count == 1 else 'entries'}"
            + (f"  — {info.description}" if info.description else "")
            for info in infos
        ]
        return "RegistryCatalog (\n" + "\n".join(rows) + "\n)"

    def _repr_html_(self) -> str:
        if not self._registries:
            return "<i>RegistryCatalog (empty)</i>"
        rows = "\n".join(
            f"<tr><td><code>{info.name}</code></td>"
            f"<td>{info.kind}</td>"
            f"<td>{info.entry_count}</td>"
            f"<td>{info.description}</td></tr>"
            for info in self.list()
        )
        return (
            "<table>\n"
            "<thead><tr><th>Registry</th><th>Kind</th>"
            "<th>Entries</th><th>Description</th></tr></thead>\n"
            f"<tbody>\n{rows}\n</tbody>\n</table>"
        )


def _format_entry_line(summary: EntrySummary, show_exactness: bool) -> str:
    """One entry as :meth:`RegistryCatalog.describe` prints it.

    The priority, or ``-`` for ``None``; the exactness when the registry
    declares one; the name; the description; and the module in parentheses.
    """
    priority = "  -" if summary.priority is None else f"{summary.priority:>3}"
    line = f"    {priority}  "
    if show_exactness:
        exactness = "" if summary.exact is None else ("exact" if summary.exact else "approx")
        line += f"{exactness:<6}  "
    line += summary.name
    if summary.description:
        line += f"  — {summary.description}"
    if summary.module_path:
        line += f"  ({summary.module_path})"
    return line


registry_catalog = RegistryCatalog()
