"""Converter registry and conversion metadata types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ._protocol import Converter


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
    estimated_error: float = 0.0
    cost: float = 0.0
    source_type: type | None = None
    target_type: type | None = None
    description: str = ""


# Sentinel for "no conversion possible"
_NOT_FEASIBLE = ConversionInfo(feasible=False, description="No converter found")


class ConverterRegistry:
    """Global registry of distribution converters.

    Converters are tried in descending priority order.  The first
    converter whose ``check()`` returns ``feasible=True`` wins.
    """

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
            f"No converter registered for "
            f"{type(source).__name__} -> {target_type.__name__}"
        )

    def is_distribution_type(self, obj: Any) -> bool:
        """Return ``True`` if *obj* is any recognized distribution type.

        Checks ProbPipe ``Distribution`` first, then falls back to
        asking registered converters whether *obj* matches any of their
        declared source types.
        """
        from ..distributions.distribution import Distribution

        if isinstance(obj, Distribution):
            return True
        return any(
            isinstance(obj, tuple(c.source_types()))
            for c in self._converters
        )

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


# Module-level singleton
converter_registry = ConverterRegistry()
