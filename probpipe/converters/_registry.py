"""Converter registry and conversion metadata types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from ..core import _workflow_broker

_ConversionExecutionMode = Literal[
    "exact",
    "analytic",
    "sampled",
    "conditional",
    "delegated",
]

_PROBPIPE_PROVIDER_ABI = "probpipe.distribution/v1"
_TFP_PROVIDER_ABI = "tensorflow_probability.substrates.jax/v1"
_SCIPY_PROVIDER_ABI = "scipy.stats.seedsequence-pcg64/v1"
_DECLARED_CONVERTER_ABI = "probpipe.converter.declared/v1"


@dataclass(frozen=True)
class _ConversionExecutionPlan:
    """Private stochastic contract for one selected conversion."""

    execution_mode: _ConversionExecutionMode
    sample_shape: tuple[int, ...] | None
    provider_abi: str
    automatic_key_certified: bool


def _validate_conversion_sample_count(value: Any) -> int:
    """Validate and return a conversion sample count before RNG commit."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"num_samples must be an integer; got {value!r}")
    if value <= 0:
        raise ValueError(f"num_samples must be positive; got {value!r}")
    return value


def _sampled_conversion_plan(
    num_samples: Any,
    *,
    provider_abi: str,
    automatic_key_certified: bool = True,
) -> _ConversionExecutionPlan:
    """Build a validated sampled-conversion execution plan."""
    count = _validate_conversion_sample_count(num_samples)
    return _ConversionExecutionPlan(
        execution_mode="sampled",
        sample_shape=(count,),
        provider_abi=provider_abi,
        automatic_key_certified=automatic_key_certified,
    )


def _resolve_conversion_key(
    key: Any | None,
    plan: _ConversionExecutionPlan,
) -> Any:
    """Preserve a caller key or claim the singleton conversion event."""
    if key is not None:
        return key
    if plan.execution_mode not in ("sampled", "conditional"):
        raise TypeError("a non-sampling conversion cannot request an automatic key")
    if not plan.automatic_key_certified:
        raise TypeError("an uncertified converter cannot request an automatic key")
    return _workflow_broker._resolve_automatic_key(
        None,
        _workflow_broker._singleton_effect_plan(
            operation_kind="conversion",
            execution_mode=plan.execution_mode,
            sample_shape=plan.sample_shape,
            provider_abi=plan.provider_abi,
        ),
    )


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
        with _workflow_broker._managed_stochastic_scope():
            for conv in self._find_converters(type(source)):
                info = conv.check(source, target_type)
                if not info.feasible:
                    continue
                plan = self._plan_conversion(
                    conv,
                    info,
                    source,
                    target_type,
                    kwargs,
                    validate_declared_sample=key is None,
                )
                resolved_key = key
                if key is None and plan.execution_mode == "sampled":
                    if not plan.automatic_key_certified:
                        raise TypeError(
                            f"{type(conv).__name__} declares a sampled conversion but "
                            "is not certified for workflow-owned randomness; pass an "
                            "explicit key= value."
                        )
                    resolved_key = _resolve_conversion_key(None, plan)
                return conv.convert(source, target_type, key=resolved_key, **kwargs)
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

    @staticmethod
    def _plan_conversion(
        converter: Converter,
        info: ConversionInfo,
        source: Any,
        target_type: type,
        kwargs: dict[str, Any],
        *,
        validate_declared_sample: bool,
    ) -> _ConversionExecutionPlan:
        """Resolve one converter's private execution contract exactly once."""
        planner = getattr(converter, "_workflow_plan_conversion", None)
        if planner is not None:
            plan = planner(source, target_type, dict(kwargs))
            if not isinstance(plan, _ConversionExecutionPlan):
                raise TypeError(
                    f"{type(converter).__name__} returned an invalid private "
                    "conversion execution plan"
                )
            return plan

        if info.method is ConversionMethod.SAMPLE:
            if validate_declared_sample:
                return _sampled_conversion_plan(
                    kwargs.get("num_samples", 1024),
                    provider_abi=_DECLARED_CONVERTER_ABI,
                    automatic_key_certified=False,
                )
            return _ConversionExecutionPlan(
                execution_mode="sampled",
                sample_shape=None,
                provider_abi=_DECLARED_CONVERTER_ABI,
                automatic_key_certified=False,
            )
        return _ConversionExecutionPlan(
            execution_mode=("exact" if info.method is ConversionMethod.EXACT else "analytic"),
            sample_shape=None,
            provider_abi=_DECLARED_CONVERTER_ABI,
            automatic_key_certified=True,
        )


# Module-level singleton
converter_registry = ConverterRegistry()
