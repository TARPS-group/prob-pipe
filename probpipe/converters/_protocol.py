"""Protocol-based distribution converter.

Converts a distribution that lacks a required protocol (e.g.,
``SupportsLogProb``) into one that satisfies it by resolving the
protocol to a concrete target type and delegating to the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._registry import ConversionInfo, ConversionMethod, Converter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_protocol(tp: Any) -> bool:
    """Return ``True`` if *tp* is a ``@runtime_checkable`` Protocol class.

    Concrete classes that *inherit* from a protocol have
    ``_is_runtime_protocol`` inherited, so we check that the attribute
    is defined directly on *tp* (not inherited) via ``__dict__``.
    """
    return (
        "_is_runtime_protocol" in getattr(tp, "__dict__", {})
        and tp.__dict__["_is_runtime_protocol"]
    )


def _resolve_target_for_log_prob(dist: Any) -> type:
    """Choose Normal or MultivariateNormal based on event dimensionality."""
    from ..distributions.continuous import Normal
    from ..distributions.multivariate import MultivariateNormal

    try:
        event_shape = dist.event_shape
    except AttributeError:
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


# ---------------------------------------------------------------------------
# ProtocolConverter
# ---------------------------------------------------------------------------

class ProtocolConverter(Converter):
    """Converter that resolves ``@runtime_checkable`` protocol targets.

    When the *target_type* passed to :meth:`check` or :meth:`convert`
    is a protocol (e.g., ``SupportsLogProb``) rather than a concrete
    distribution class, this converter:

    1. Returns the source unchanged if it already satisfies the protocol.
    2. Resolves the protocol to a concrete target type via a registered
       resolver function.
    3. Delegates the concrete conversion back to the *registry*.

    Resolvers are registered with :meth:`register_protocol_target`.

    Parameters
    ----------
    registry : ConverterRegistry
        The registry instance to delegate concrete conversions to.
    """

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._resolvers: dict[type, Callable[[Any], type]] = {}

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
        self._resolvers[protocol] = resolver

    # -- Converter interface ------------------------------------------------

    def source_types(self) -> tuple[type, ...]:
        from ..core.distribution import Distribution

        return (Distribution,)

    def target_types(self) -> tuple[type, ...]:
        return ()  # Targets are protocols, not concrete types

    def check(self, source: Any, target_type: type) -> ConversionInfo:
        if not _is_protocol(target_type):
            return ConversionInfo(feasible=False)

        if isinstance(source, target_type):
            return ConversionInfo(
                feasible=True,
                method=ConversionMethod.EXACT,
                estimated_time=0.0,
                source_type=type(source),
                target_type=target_type,
                description=(
                    f"{type(source).__name__} already satisfies "
                    f"{target_type.__name__}"
                ),
            )

        resolver = self._resolvers.get(target_type)
        if resolver is None:
            return ConversionInfo(
                feasible=False,
                source_type=type(source),
                target_type=target_type,
                description=(
                    f"No protocol resolver registered for "
                    f"{target_type.__name__}"
                ),
            )

        # Resolve and probe the concrete conversion
        concrete_type = resolver(source)
        return self._registry.check(source, concrete_type)

    def convert(
        self,
        source: Any,
        target_type: type,
        *,
        key: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if not _is_protocol(target_type):
            raise TypeError(
                f"ProtocolConverter only handles protocol targets, "
                f"got {target_type!r}"
            )

        if isinstance(source, target_type):
            return source

        resolver = self._resolvers.get(target_type)
        if resolver is None:
            raise TypeError(
                f"No protocol resolver registered for "
                f"{target_type.__name__}. "
                f"{type(source).__name__} does not satisfy "
                f"{target_type.__name__} and no automatic conversion "
                f"is available."
            )

        concrete_type = resolver(source)
        return self._registry.convert(
            source, concrete_type, key=key, **kwargs
        )

    @property
    def priority(self) -> int:
        # Highest priority so protocol targets are intercepted before
        # concrete converters (which would reject them).
        return 200


def _make_protocol_converter(registry: Any) -> ProtocolConverter:
    """Create and configure a :class:`ProtocolConverter` with built-in resolvers."""
    from ..core.protocols import SupportsLogProb, SupportsSampling

    converter = ProtocolConverter(registry)
    converter.register_protocol_target(SupportsLogProb, _resolve_target_for_log_prob)
    converter.register_protocol_target(SupportsSampling, _resolve_target_for_sampling)
    return converter
