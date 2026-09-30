"""Cross-type conversion: moving a law between representations.

A **converter** is a binary dispatch method declaring the source types it
converts from and the targets it converts to, and whether it is exact. The
**converter registry** is keyed on the source's type and the requested target,
which is a distribution class or a capability protocol. A conversion changes
the representation and nothing else: the result carries the source's event
declaration and realizes the same law up to the converter's fidelity.

Provides:
  - ``ConversionInfo`` – what a converter's non-executing check promises.
  - ``Converter`` – the base of every converter.
  - ``ConverterRegistry`` and its global instance ``converter_registry``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from ..core._dispatch import (
    BinaryDispatchMethod,
    BinaryDispatchRegistry,
    Feasibility,
    MethodInfo,
    _mro_distance,
    _Registration,
)
from ._distribution import Distribution, DistributionSpec

__all__ = ["ConversionInfo", "Converter", "ConverterRegistry", "converter_registry"]


@dataclass(frozen=True)
class ConversionInfo(MethodInfo):
    """What a converter's ``check`` promises for one conversion, without executing it.

    Attributes
    ----------
    target_spec : DistributionSpec or None
        The declaration the result will carry, ``None`` while unresolved.
    target_class : type or None
        The representation class of the result, when known.
    capabilities : tuple of type
        The capability protocols guaranteed on the result.

    Notes
    -----
    The pending requirements are those of
    :class:`~probpipe.core._dispatch.Feasibility`, and the converter's name and
    exactness are those of :class:`~probpipe.core._dispatch.MethodInfo`, which a
    converter fills from its own declaration.
    """

    target_spec: DistributionSpec | None = None
    target_class: type | None = None
    capabilities: tuple[type, ...] = ()


class Converter(BinaryDispatchMethod):
    """A binary dispatch method that converts a law to a requested representation.

    A subclass declares ``name``, ``exact``, and the ``(source, target)`` types
    it supports, and implements a non-executing :meth:`check` and an
    :meth:`execute`. A target type is a distribution class or a capability
    protocol.
    """

    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """The source types this converter reads and the target types it produces."""

    @abstractmethod
    def check(self, source: Any, target_type: type, *, exact_only: bool = False) -> ConversionInfo:
        """Promise the conversion of *source* to *target_type* without executing it.

        It never samples, fits, or evaluates a density, and it reports as
        pending whatever it cannot settle without the source's values.
        """

    @abstractmethod
    def execute(self, source: Any, target_type: type) -> Distribution:
        """Construct the converted law, carrying *source*'s event declaration."""


class ConverterRegistry(BinaryDispatchRegistry[Converter]):
    """The binary dispatch registry of converters, keyed on the source type and the target.

    The target enters the key as itself, not as its type. A converter admits a
    requested target when a target it declares is that class or protocol or a
    subclass of it, so a converter producing a more specific representation or
    a refined capability serves the request.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> tuple[type, type]:
        if len(args) < 2:
            raise TypeError(
                f"ConverterRegistry requires a source and a target type; got {len(args)} "
                f"positional arguments"
            )
        source, target = args[0], args[1]
        if not isinstance(target, type):
            raise TypeError(f"a conversion target is a class or a protocol, got {target!r}")
        return (type(source), target)

    def _distance(
        self, supported_types: tuple[tuple[type, ...], tuple[type, ...]], key: tuple[type, type]
    ) -> int | None:
        supported_sources, supported_targets = supported_types
        source_distance = _mro_distance(key[0], supported_sources)
        target_distances = [
            distance
            for declared in supported_targets
            if (distance := _mro_distance(declared, (key[1],))) is not None
        ]
        if source_distance is None or not target_distances:
            return None
        return source_distance + min(target_distances)

    @staticmethod
    def _report(registration: _Registration[Converter], feasibility: Feasibility) -> MethodInfo:
        """The converter's report with its registered name and exactness, promises kept."""
        if isinstance(feasibility, ConversionInfo):
            return replace(feasibility, method_name=registration.name, exact=registration.exact)
        return BinaryDispatchRegistry._report(registration, feasibility)

    def convert(
        self,
        source: Any,
        target_type: type,
        method: str | None = None,
        exact_only: bool = False,
    ) -> Distribution:
        """Convert *source* to *target_type* with the selected converter.

        Raises
        ------
        ResolutionError
            If no converter is feasible under the controls, or *method* names one
            that is not registered or not applicable.
        """
        return self.execute(source, target_type, method=method, exact_only=exact_only)


converter_registry: ConverterRegistry = ConverterRegistry()
