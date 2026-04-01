"""Converter registry and conversion metadata types.

Provides a global :data:`converter_registry` that handles bidirectional
conversion between distribution types and protocol-based conversion
(e.g., converting a sampling-only distribution to one that supports
log-probability evaluation).
"""

from __future__ import annotations

from collections.abc import Callable
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
    estimated_time: float = 0.0
    source_type: type | None = None
    target_type: type | None = None
    description: str = ""


# Sentinel for "no conversion possible"
_NOT_FEASIBLE = ConversionInfo(feasible=False, description="No converter found")


# ---------------------------------------------------------------------------
# Protocol target resolvers
# ---------------------------------------------------------------------------

def _is_protocol(tp: Any) -> bool:
    """Return ``True`` if *tp* is a ``@runtime_checkable`` Protocol class.

    Concrete classes that *inherit* from a protocol have
    ``_is_runtime_protocol`` set to ``True`` on the class itself, so we
    additionally check that the attribute is defined directly on *tp*
    (not inherited) via ``__dict__``.
    """
    return "_is_runtime_protocol" in getattr(tp, "__dict__", {}) and tp.__dict__["_is_runtime_protocol"]


def _resolve_target_for_log_prob(dist: Any) -> type:
    """Choose Normal or MultivariateNormal based on event dimensionality."""
    from ..distributions.continuous import Normal
    from ..distributions.multivariate import MultivariateNormal

    try:
        event_shape = dist.event_shape
    except AttributeError:
        # If no event_shape, try dim
        try:
            if dist.dim <= 1:
                return Normal
            return MultivariateNormal
        except AttributeError:
            return Normal

    if len(event_shape) == 0 or (len(event_shape) == 1 and event_shape[0] == 1):
        return Normal
    return MultivariateNormal


def _resolve_target_for_sampling(dist: Any) -> type:
    """Choose ArrayEmpiricalDistribution for sampling conversion."""
    from ..core.distribution import ArrayEmpiricalDistribution

    return ArrayEmpiricalDistribution


class ConverterRegistry:
    """Global registry of distribution converters.

    Converters are tried in descending priority order.  The first
    converter whose ``check()`` returns ``feasible=True`` wins.

    In addition to type-to-type conversion, the registry supports
    **protocol-based conversion**: passing a protocol (e.g.,
    ``SupportsLogProb``) as the *target_type* to :meth:`convert` or
    :meth:`check` will resolve to a concrete target type via a
    registered resolver function.
    """

    def __init__(self) -> None:
        self._converters: list[Converter] = []
        self._type_cache: dict[type, list[Converter]] = {}
        self._protocol_resolvers: dict[type, Callable] = {}

    # -- registration -------------------------------------------------------

    def register(self, converter: Converter) -> None:
        """Register a converter (invalidates the lookup cache)."""
        self._converters.append(converter)
        self._converters.sort(key=lambda c: c.priority, reverse=True)
        self._type_cache.clear()

    def register_protocol_target(
        self,
        protocol: type,
        resolver: Callable[[Any], type],
    ) -> None:
        """Register a resolver that maps a protocol to a concrete target type.

        Parameters
        ----------
        protocol : type
            A ``@runtime_checkable`` Protocol class (e.g.,
            ``SupportsLogProb``).
        resolver : callable
            A function ``(source_distribution) -> target_type`` that
            inspects the source distribution and returns the concrete
            distribution class to convert to.
        """
        self._protocol_resolvers[protocol] = resolver

    # -- protocol resolution ------------------------------------------------

    def _resolve_protocol(
        self, source: Any, target_type: type
    ) -> type | None:
        """If *target_type* is a protocol, resolve to a concrete type.

        Returns ``None`` if *target_type* is not a protocol or if the
        source already satisfies it.
        """
        if not _is_protocol(target_type):
            return None
        if isinstance(source, target_type):
            return None  # Already satisfies the protocol
        resolver = self._protocol_resolvers.get(target_type)
        if resolver is None:
            raise TypeError(
                f"No protocol resolver registered for {target_type.__name__}. "
                f"{type(source).__name__} does not satisfy {target_type.__name__} "
                f"and no automatic conversion is available."
            )
        return resolver(source)

    # -- query --------------------------------------------------------------

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        """Return conversion metadata for *source* → *target_type*.

        *target_type* may be a concrete distribution class or a
        ``@runtime_checkable`` Protocol.  In the latter case the
        registry resolves to a concrete target via a registered
        resolver function.

        Tries converters in priority order; returns the first feasible
        result.  Returns a non-feasible ``ConversionInfo`` if no
        converter can handle the pair.
        """
        if _is_protocol(target_type):
            if isinstance(source, target_type):
                return ConversionInfo(
                    feasible=True,
                    method=ConversionMethod.EXACT,
                    estimated_time=0.0,
                    source_type=type(source),
                    target_type=target_type,
                    description=f"{type(source).__name__} already satisfies {target_type.__name__}",
                )
            resolved = self._resolve_protocol(source, target_type)
            if resolved is not None:
                target_type = resolved

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

        *target_type* may be a concrete distribution class or a
        ``@runtime_checkable`` Protocol.  Protocol targets are resolved
        to a concrete class via :meth:`register_protocol_target`
        resolvers.

        Raises ``TypeError`` if no converter can handle the pair.
        """
        if _is_protocol(target_type):
            if isinstance(source, target_type):
                return source
            resolved = self._resolve_protocol(source, target_type)
            if resolved is not None:
                target_type = resolved

        for conv in self._find_converters(type(source)):
            info = conv.check(source, target_type)
            if info.feasible:
                return conv.convert(source, target_type, key=key, **kwargs)
        raise TypeError(
            f"No converter registered for "
            f"{type(source).__name__} -> {target_type.__name__}"
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
