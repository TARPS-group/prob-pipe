"""Converter registry and conversion metadata types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ..core._registry_catalog import EntrySummary, registry_catalog


class ConversionMethod(Enum):
    """How a conversion is performed."""

    EXACT = "exact"
    MOMENT_MATCH = "moment_match"
    SAMPLE = "sample"


@dataclass(frozen=True)
class ConversionInfo:
    """Metadata describing a potential conversion.

    Returned by :meth:`Converter.check` and
    :meth:`ConverterRegistry.check`.
    """

    feasible: bool
    method: ConversionMethod | None = None
    estimated_time: float = 0.0
    source_type: type | None = None
    target_type: type | None = None
    description: str = ""


# Sentinel for "no conversion possible"
_NOT_FEASIBLE = ConversionInfo(feasible=False, description="No converter found")


class Converter(ABC):
    """Base class for distribution converters.

    Subclasses declare which types they can convert between via
    ``source_types()`` and ``target_types()``, provide a cheap
    ``check()`` probe, and implement ``convert()`` for the actual work.
    """

    @abstractmethod
    def source_types(self) -> tuple[type, ...]:
        """Types this converter can convert FROM."""
        ...

    @abstractmethod
    def target_types(self) -> tuple[type, ...]:
        """Types this converter can convert TO."""
        ...

    @abstractmethod
    def check(self, source: Any, target_type: type) -> ConversionInfo:
        """Inspect feasibility and cost without performing conversion.

        Must be cheap (no sampling, no heavy computation).
        """
        ...

    @abstractmethod
    def convert(
        self, source: Any, target_type: type, *, key: Any | None = None, **kwargs: Any
    ) -> Any:
        """Perform the actual conversion.

        Returns an instance of *target_type* (or a compatible subclass).
        """
        ...

    @property
    def priority(self) -> int:
        """Higher priority converters are tried first. Default 0."""
        return 0


class ConverterRegistry:
    """Global registry of distribution converters.

    Converters are tried in descending priority order.  The first
    converter whose ``check()`` returns ``feasible=True`` wins.

    Cataloging surface
    ------------------
    The class carries the
    :class:`~probpipe.core._registry_catalog.SupportsRegistryCataloging`
    identity attributes (``name``, ``description``, ``kind``) as
    class-level fields and implements :meth:`entry_summaries` and
    :meth:`describe_entry` so the registry can be discovered via the
    global :data:`registry_catalog` ("converters" key).  Dispatch
    mechanics are unchanged — this is a *non-conforming* registry from
    the dispatch perspective (the ``(source_type, target_type)`` shape
    with ``target_type`` passed as a value-type doesn't fit
    :class:`~probpipe.core._registry.BinaryDispatchRegistry`), but the
    catalog only needs the introspection surface.
    """

    #: Catalog identity attributes (class-level so they are stable
    #: across the module-level singleton's lifetime).
    name: ClassVar[str] = "converters"
    description: ClassVar[str] = (
        "Cross-type distribution converters (TFP, scipy, ProbPipe-internal, protocol-based)."
    )
    kind: ClassVar[str] = "converter"

    def __init__(self) -> None:
        self._converters: list[Converter] = []
        self._type_cache: dict[type, list[Converter]] = {}

    # -- registration -------------------------------------------------------

    def register(self, converter: Converter) -> None:
        """Register a converter (invalidates the lookup cache)."""
        self._converters.append(converter)
        self._converters.sort(key=lambda c: c.priority, reverse=True)
        self._type_cache.clear()

    # -- query --------------------------------------------------------------

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        """Return conversion metadata for *source* → *target_type*.

        Tries converters in priority order; returns the first feasible
        result.  Returns a non-feasible ``ConversionInfo`` if no
        converter can handle the pair.
        """
        for conv in self._find_converters(type(source)):
            info = conv.check(source, target_type)
            if info.feasible:
                return info
        return ConversionInfo(
            feasible=False,
            source_type=type(source),
            target_type=target_type,
            description="No converter found",
        )

    def convert(
        self,
        source: Any,
        target_type: type,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Convert *source* to *target_type* using the best converter.

        Raises ``TypeError`` if no converter can handle the pair.
        """
        for conv in self._find_converters(type(source)):
            info = conv.check(source, target_type)
            if info.feasible:
                return conv.convert(source, target_type, key=key, **kwargs)
        raise TypeError(
            f"No converter registered for {type(source).__name__} -> {target_type.__name__}"
        )

    def is_distribution_type(self, obj: Any) -> bool:
        """Return ``True`` if *obj* is a recognized distribution-like object.

        This includes any ProbPipe ``Distribution`` subclass as well as
        external distribution types (e.g., TFP, scipy.stats) for which
        a registered converter declares support.
        """
        from ..core.distribution import Distribution

        if isinstance(obj, Distribution):
            return True
        return any(isinstance(obj, tuple(c.source_types())) for c in self._converters)

    # -- catalog surface ----------------------------------------------------

    def entry_summaries(self) -> list[EntrySummary]:
        """Return one :class:`EntrySummary` per registered converter.

        Walks ``self._converters`` in priority-sorted order (the same
        order :meth:`check` uses).  ``supported_types`` is encoded as
        ``(source_types, target_types)`` to match the converter's
        :meth:`Converter.source_types` / :meth:`Converter.target_types`
        contract.
        """
        return [
            EntrySummary(
                name=type(c).__name__,
                priority=c.priority,
                supported_types=(c.source_types(), c.target_types()),
                description=(type(c).__doc__ or "").strip().splitlines()[0]
                if type(c).__doc__
                else "",
                module_path=type(c).__module__,
            )
            for c in self._converters
        ]

    def describe_entry(self, name: str) -> EntrySummary:
        """Return the :class:`EntrySummary` for the named converter.

        *name* matches ``type(converter).__name__``.

        Raises
        ------
        KeyError
            If no registered converter has that class name.
        """
        for s in self.entry_summaries():
            if s.name == name:
                return s
        available = ", ".join(sorted(s.name for s in self.entry_summaries())) or "(none)"
        raise KeyError(f"No converter named {name!r}. Available: {available}")

    # -- internals ----------------------------------------------------------

    def _find_converters(self, source_type: type) -> list[Converter]:
        """Return converters that handle *source_type* (cached)."""
        if source_type not in self._type_cache:
            self._type_cache[source_type] = [
                c
                for c in self._converters
                if any(issubclass(source_type, st) for st in c.source_types())
            ]
        return self._type_cache[source_type]


# Module-level singleton.  Registered with the global registry catalog
# so it is discoverable via ``probpipe.registry_catalog["converters"]``.
converter_registry = ConverterRegistry()
registry_catalog.register(converter_registry)
