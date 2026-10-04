"""The registry catalog, which lists every registry and its entries.

:data:`registry_catalog` lists the registries in the process, their entries
with each entry's exactness and priority, and a one-line description of each,
so a user can see which implementations exist and how a call resolves. An
**entry** is one registered item of a registry, such as an inference method, a
converter, an operation, or a bijector factory.

The catalog describes registries and selects nothing. A call goes to the
registry itself, such as ``inference_method_registry`` or
``converter_registry``.

A registry can be cataloged if it implements
:class:`SupportsRegistryCataloging`. Satisfying the protocol is structural, and
membership requires an explicit :meth:`RegistryCatalog.register`, which rejects
an empty or duplicate name.
"""

from __future__ import annotations

import html
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
    """The catalog's record of one entry of a registry.

    An *entry* is one registered item, such as an inference method, a
    converter, an operation, or a bijector factory. A registry returns these
    from :meth:`SupportsRegistryCataloging.entry_summaries` and
    :meth:`SupportsRegistryCataloging.describe_entry`.

    Attributes
    ----------
    name : str
        The entry's name, unique within its registry.
    priority : int or None
        The entry's effective rank among the entries of the same exactness,
        higher first. In a dispatch registry ``None`` is opt-in-only. A
        registry that does not rank its entries, such as the bijector factory
        or the operation registry, reports ``None`` for every entry.
    supported_types : tuple
        What the entry admits, in its registry's form: a tuple of classes for a
        unary dispatch registry, a ``(left_types, right_types)`` pair for a
        binary one, the kinds of the first operand for an operation, and the
        constraint key in a 1-tuple for a bijector factory.
    description : str
        A one-line description of the entry, empty when it declares none.
    module_path : str
        The module that defines the entry.
    exact : bool or None
        The entry's declared exactness, or ``None`` when its registry declares
        no exactness per entry.
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

        The flag applies to a registry that ranks its entries: a dispatch
        registry skips such an entry in automatic selection and runs it only
        when a call names it with ``method="..."``. A registry that does not
        rank its entries reports ``None`` for every entry and has no opt-in
        entries; :meth:`RegistryCatalog.describe` lists its entries together.
        """
        return self.priority is None


@dataclass(frozen=True)
class RegistryInfo:
    """The catalog's record of one registry.

    :meth:`RegistryCatalog.list` returns these; :class:`EntrySummary` holds
    the detail of each entry.

    Attributes
    ----------
    name : str
        The registry's name, unique within the catalog.
    description : str
        A one-line description of the registry.
    kind : str
        The registry's kind, such as ``"dispatch"``, ``"operation"``, or
        ``"factory"``.
    entry_count : int
        The number of entries the registry holds.
    """

    name: str
    description: str
    kind: str
    entry_count: int


@runtime_checkable
class SupportsRegistryCataloging(Protocol):
    """The members a registry implements to be cataloged.

    Attributes
    ----------
    name : str
        The registry's name, unique within the catalog.
    description : str
        A one-line description of the registry.
    kind : str
        The registry's kind, such as ``"dispatch"``, ``"operation"``, or
        ``"factory"``. The kind is a string, so a plugin can introduce one.

    Methods
    -------
    entry_summaries() -> list[EntrySummary]
        One summary per entry, in the registry's order: selection order before
        type specificity for a dispatch registry, registration order for the
        operation registry, and name order for the bijector factory.
    describe_entry(name) -> EntrySummary
        The summary of one entry; raises ``KeyError`` for a name the registry
        does not hold.

    Notes
    -----
    The protocol states a registry's identity and its entries and states
    nothing about selection, so a registry that selects by other means, such
    as the bijector factory's lookup by constraint instance and then by
    constraint class, is cataloged as it is. Satisfying the protocol does not
    place a registry in the catalog; :meth:`RegistryCatalog.register` does.
    """

    name: str
    description: str
    kind: str

    def entry_summaries(self) -> list[EntrySummary]: ...

    def describe_entry(self, name: str) -> EntrySummary: ...


class RegistryCatalog:
    """A catalog of registries, keyed by registry name.

    The catalog records registries and describes them, and it holds no
    selection state.

    Examples
    --------
    >>> import probpipe
    >>> "inference" in probpipe.registry_catalog
    True
    >>> print(probpipe.registry_catalog.describe("inference"))  # doctest: +SKIP
    inference  (dispatch) — Inference methods for condition_on, keyed on the target's type.
    ...
    """

    def __init__(self) -> None:
        self._registries: dict[str, SupportsRegistryCataloging] = {}

    def register(self, registry: SupportsRegistryCataloging) -> None:
        """Add *registry* to the catalog under its ``name``.

        Parameters
        ----------
        registry : SupportsRegistryCataloging
            The registry to add.

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
        """The registry named *name*.

        Raises
        ------
        KeyError
            If no registry is registered under *name*.
        """
        try:
            return self._registries[name]
        except KeyError:
            available = ", ".join(sorted(self._registries)) or "(none)"
            raise KeyError(f"No registry named {name!r}. Available: {available}") from None

    def __contains__(self, name: object) -> bool:
        """``True`` when *name* is a registered registry name."""
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
        """A readable summary of the registry named *name*.

        The first line names the registry, its kind, and its description, and
        each entry follows on a line of its own: its priority, or ``-`` for
        ``None``; its exactness, where the registry declares one; its name; its
        description; and its module. The entries follow the registry's
        :meth:`~SupportsRegistryCataloging.entry_summaries` order, which for a
        dispatch registry is its selection order before type specificity. A
        dispatch registry's opt-in-only entries are listed in a section of
        their own, and any other registry's entries are listed together.

        Parameters
        ----------
        name : str
            A registered registry name.

        Returns
        -------
        str
            The summary.

        Raises
        ------
        KeyError
            If no registry is registered under *name*.
        """
        registry = self[name]
        summaries = registry.entry_summaries()
        show_exactness = any(summary.exact is not None for summary in summaries)
        priority_width = max((len(_priority_text(s)) for s in summaries), default=1)
        if registry.kind == "dispatch":
            sections = [
                (
                    "Auto-dispatched, in selection order:",
                    [s for s in summaries if not s.is_opt_in_only],
                ),
                (
                    "Opt-in only (run by method=...):",
                    [s for s in summaries if s.is_opt_in_only],
                ),
            ]
        else:
            sections = [("Entries:", summaries)]
        header = f"{registry.name}  ({registry.kind})"
        if registry.description:
            header += f" — {registry.description}"
        lines = [header, ""]
        for label, entries in sections:
            if not entries:
                continue
            lines.append(f"  {label}")
            lines.extend(
                _format_entry_line(entry, show_exactness, priority_width) for entry in entries
            )
            lines.append("")
        if not summaries:
            lines.append("  (no entries registered)")
        return "\n".join(lines).rstrip()

    def __repr__(self) -> str:
        """One row per registry: its name, its kind, its entry count, and its description."""
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
        """The rows of :meth:`__repr__` as an HTML table, with each string escaped."""
        if not self._registries:
            return "<i>RegistryCatalog (empty)</i>"
        rows = "\n".join(
            f"<tr><td><code>{html.escape(info.name)}</code></td>"
            f"<td>{html.escape(info.kind)}</td>"
            f"<td>{info.entry_count}</td>"
            f"<td>{html.escape(info.description)}</td></tr>"
            for info in self.list()
        )
        return (
            "<table>\n"
            "<thead><tr><th>Registry</th><th>Kind</th>"
            "<th>Entries</th><th>Description</th></tr></thead>\n"
            f"<tbody>\n{rows}\n</tbody>\n</table>"
        )


def _priority_text(summary: EntrySummary) -> str:
    """The entry's priority as :meth:`RegistryCatalog.describe` prints it, ``-`` for ``None``."""
    return "-" if summary.priority is None else str(summary.priority)


def _format_entry_line(summary: EntrySummary, show_exactness: bool, priority_width: int) -> str:
    """One entry as :meth:`RegistryCatalog.describe` prints it, its priority right-aligned."""
    line = f"    {_priority_text(summary):>{priority_width}}  "
    if show_exactness:
        exactness = "" if summary.exact is None else ("exact" if summary.exact else "approx")
        line += f"{exactness:<6}  "
    line += summary.name
    if summary.description:
        line += f"  — {summary.description}"
    if summary.module_path:
        line += f"  ({summary.module_path})"
    return line


registry_catalog: RegistryCatalog = RegistryCatalog()
"""The global registry catalog."""
