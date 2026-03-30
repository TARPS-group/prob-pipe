"""Pushforward dispatch registry.

Mirrors the structure of :mod:`probpipe.converters`: rules are objects
with ``check()`` / ``apply()`` methods, stored in a priority-ordered
registry.  A convenience decorator (:meth:`PushforwardRegistry.rule`)
allows simple function-based rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from ..core.distribution import Distribution
    from ..custom_types import PRNGKey


# ── Metadata types ────────────────────────────────────────────────────


class PushforwardMethod(Enum):
    """How a pushforward is computed."""

    CLOSED_FORM = "closed_form"
    CHANGE_OF_VARIABLES = "change_of_variables"
    SAMPLE = "sample"


@dataclass(frozen=True)
class PushforwardInfo:
    """Metadata describing a potential pushforward.

    Mirrors :class:`ConversionInfo` in the converter registry.
    Returned by :meth:`PushforwardRule.check` and
    :meth:`PushforwardRegistry.check`.
    """

    feasible: bool
    method: PushforwardMethod | None = None
    description: str = ""


# Sentinel for "no rule applies"
_NOT_FEASIBLE = PushforwardInfo(feasible=False, description="No rule found")


# ── Rule protocol ─────────────────────────────────────────────────────


class PushforwardRule(ABC):
    """Base class for pushforward dispatch rules.

    Mirrors the :class:`Converter` protocol: subclasses declare which
    types they handle via ``map_types()`` / ``dist_types()``, provide a
    cheap ``check()`` probe, and implement ``apply()`` for the actual
    computation.
    """

    @abstractmethod
    def map_types(self) -> tuple[type, ...]:
        """TransportMap types this rule can handle."""
        ...

    @abstractmethod
    def dist_types(self) -> tuple[type, ...]:
        """Distribution types this rule can handle."""
        ...

    @abstractmethod
    def check(self, transport_map: Any, dist: Any) -> PushforwardInfo:
        """Inspect feasibility without performing the pushforward.

        Must be cheap (no sampling, no heavy computation).
        """
        ...

    @abstractmethod
    def apply(
        self,
        transport_map: Any,
        dist: Any,
        *,
        return_joint: bool = False,
        **kwargs: Any,
    ) -> Distribution:
        """Perform the pushforward.

        Returns a :class:`Distribution` (concrete type depends on the
        rule).
        """
        ...

    @property
    def priority(self) -> int:
        """Higher priority rules are tried first.  Default ``0``."""
        return 0


# ── Registry ──────────────────────────────────────────────────────────


class PushforwardRegistry:
    """Registry of pushforward dispatch rules.

    Rules are tried in descending priority order.  The first rule whose
    ``check()`` returns ``feasible=True`` wins.

    Structurally parallel to :class:`ConverterRegistry`.
    """

    def __init__(self) -> None:
        self._rules: list[PushforwardRule] = []
        self._type_cache: dict[tuple[type, type], list[PushforwardRule]] = {}

    # -- registration -------------------------------------------------------

    def register(self, rule: PushforwardRule) -> None:
        """Register a pushforward rule (invalidates the lookup cache)."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        self._type_cache.clear()

    def rule(
        self,
        map_type: type,
        dist_type: type,
        *,
        method: PushforwardMethod = PushforwardMethod.CLOSED_FORM,
        priority: int = 10,
        description: str = "",
    ) -> Callable:
        """Decorator to register a function as a pushforward rule.

        Use this when a (map_type, dist_type) pair has an **unconditionally
        feasible** closed-form pushforward — i.e., whenever the types match,
        the formula applies.  The decorator wraps the function into a
        :class:`PushforwardRule` and registers it automatically.

        For rules with **conditional feasibility** (where ``check()`` depends
        on runtime values, not just types), subclass :class:`PushforwardRule`
        directly and call :meth:`register` instead.

        The decorated function should have signature
        ``(transport_map, dist, **kwargs) -> Distribution``.

        Parameters
        ----------
        map_type : type
            TransportMap type this rule handles.
        dist_type : type
            Distribution type this rule handles.
        method : PushforwardMethod
            How the pushforward is computed.  Default ``CLOSED_FORM``.
        priority : int
            Higher = tried first.  Default 10 (above built-in rules at 0
            and -100).
        description : str
            Human-readable description.

        Examples
        --------
        >>> @pushforward_registry.rule(
        ...     MyMap, Normal,
        ...     method=PushforwardMethod.CLOSED_FORM,
        ... )
        ... def _(m, d, **kw):
        ...     return LogNormal(loc=d.loc, scale=d.scale)
        """

        def decorator(fn: Callable) -> Callable:
            rule = _FunctionalRule(
                fn,
                map_type=map_type,
                dist_type=dist_type,
                method=method,
                _priority=priority,
                description=description,
            )
            self.register(rule)
            return fn

        return decorator

    # -- query --------------------------------------------------------------

    def check(
        self,
        transport_map: Any,
        dist: Any,
        *,
        strategy: str | None = None,
    ) -> PushforwardInfo:
        """Return metadata for the best matching rule.

        Parameters
        ----------
        transport_map, dist
            The map and distribution to push forward.
        strategy : str, optional
            If given, only consider rules whose method matches.

        Returns
        -------
        PushforwardInfo
        """
        method_filter = _parse_strategy(strategy)
        for rule in self._find_rules(type(transport_map), type(dist)):
            info = rule.check(transport_map, dist)
            if info.feasible:
                if method_filter is None or info.method == method_filter:
                    return info
        return _NOT_FEASIBLE

    def apply(
        self,
        transport_map: Any,
        dist: Any,
        *,
        strategy: str | None = None,
        key: PRNGKey | None = None,
        num_samples: int | None = None,
        return_joint: bool = False,
    ) -> Distribution:
        """Dispatch to the best matching rule and execute.

        Parameters
        ----------
        transport_map, dist
            The map and distribution to push forward.
        strategy : str, optional
            ``"closed_form"``, ``"change_of_variables"``,
            ``"sampling"``, or ``None`` (auto).
        key : PRNGKey, optional
            For sampling fallback.
        num_samples : int, optional
            Number of samples for sampling fallback.
        return_joint : bool
            If ``True``, return a joint distribution over inputs and
            outputs.

        Returns
        -------
        Distribution
            The pushforward distribution.

        Raises
        ------
        ValueError
            If the requested strategy has no matching rule.
        """
        from ..core.distribution import Provenance

        method_filter = _parse_strategy(strategy)

        for rule in self._find_rules(type(transport_map), type(dist)):
            info = rule.check(transport_map, dist)
            if info.feasible:
                if method_filter is None or info.method == method_filter:
                    result = rule.apply(
                        transport_map,
                        dist,
                        key=key,
                        num_samples=num_samples,
                        return_joint=return_joint,
                    )
                    # Attach provenance if not already set
                    if result.source is None:
                        result.with_source(
                            Provenance(
                                "pushforward",
                                parents=(dist,),
                                metadata={
                                    "map": repr(transport_map),
                                    "method": info.method.value
                                    if info.method
                                    else "unknown",
                                },
                            )
                        )
                    return result

        if strategy is not None:
            raise ValueError(
                f"No pushforward rule with strategy={strategy!r} for "
                f"({type(transport_map).__name__}, {type(dist).__name__})"
            )
        raise TypeError(
            f"No pushforward rule for "
            f"({type(transport_map).__name__}, {type(dist).__name__})"
        )

    # -- internals ----------------------------------------------------------

    def _find_rules(
        self, map_type: type, dist_type: type
    ) -> list[PushforwardRule]:
        """Return matching rules for the given types (cached)."""
        key = (map_type, dist_type)
        if key not in self._type_cache:
            self._type_cache[key] = [
                r
                for r in self._rules
                if any(issubclass(map_type, mt) for mt in r.map_types())
                and any(issubclass(dist_type, dt) for dt in r.dist_types())
            ]
        return self._type_cache[key]


# ── Built-in rules ────────────────────────────────────────────────────


class _BijectorChangeOfVariablesRule(PushforwardRule):
    """Built-in: apply change-of-variables when map is a Bijector."""

    def map_types(self):
        from .bijector import Bijector

        return (Bijector,)

    def dist_types(self):
        from ..core.distribution import ArrayDistribution

        return (ArrayDistribution,)

    def check(self, transport_map, dist):
        return PushforwardInfo(
            feasible=True,
            method=PushforwardMethod.CHANGE_OF_VARIABLES,
            description="Bijector change-of-variables formula",
        )

    def apply(self, transport_map, dist, *, return_joint=False, **kwargs):
        from .transformed_distribution import BijectorTransformedDistribution
        from ..core.distribution import (
            BroadcastDistribution,
            Provenance,
            _auto_key,
        )
        from ..core.node import WorkflowFunction

        exact_output = BijectorTransformedDistribution(dist, transport_map)

        if not return_joint:
            return exact_output

        # return_joint=True: sample paired (input, output) for joint,
        # and store exact output marginal reference.
        key = kwargs.get("key")
        num_samples = kwargs.get("num_samples")
        if key is None:
            key = _auto_key()
        n = num_samples if num_samples is not None else WorkflowFunction.DEFAULT_N_BROADCAST_SAMPLES

        input_samples = dist._sample(key, sample_shape=(n,))
        output_samples = jax.vmap(transport_map.forward)(input_samples)

        result = BroadcastDistribution(
            input_samples={"input": input_samples},
            output_samples=output_samples,
            weights=None,
            broadcast_args=["input"],
        )
        result._exact_output_marginal = exact_output
        result.with_source(
            Provenance(
                "pushforward",
                parents=(dist,),
                metadata={
                    "map": repr(transport_map),
                    "method": "change_of_variables",
                    "return_joint": True,
                },
            )
        )
        return result

    @property
    def priority(self):
        return 0


class _SamplingFallbackRule(PushforwardRule):
    """Built-in: sample from base distribution and apply forward map."""

    def map_types(self):
        from .transport_map import TransportMap

        return (TransportMap, _CallableTransportMap)

    def dist_types(self):
        from ..core.distribution import Distribution

        return (Distribution,)

    def check(self, transport_map, dist):
        return PushforwardInfo(
            feasible=True,
            method=PushforwardMethod.SAMPLE,
            description="Sampling-based pushforward",
        )

    def apply(self, transport_map, dist, *, return_joint=False, **kwargs):
        from ..core.distribution import (
            EmpiricalDistribution,
            BroadcastDistribution,
            Provenance,
            _auto_key,
        )
        from ..core.node import WorkflowFunction

        key = kwargs.get("key")
        num_samples = kwargs.get("num_samples")

        if key is None:
            key = _auto_key()
        n = num_samples if num_samples is not None else WorkflowFunction.DEFAULT_N_BROADCAST_SAMPLES

        samples = dist._sample(key, sample_shape=(n,))
        transformed = jax.vmap(transport_map.forward)(samples)

        if return_joint:
            result = BroadcastDistribution(
                input_samples={"input": samples},
                output_samples=transformed,
                weights=None,
                broadcast_args=["input"],
            )
            result.with_source(
                Provenance(
                    "pushforward",
                    parents=(dist,),
                    metadata={
                        "map": repr(transport_map),
                        "method": "sample",
                        "num_samples": n,
                        "return_joint": True,
                    },
                )
            )
            return result

        result = EmpiricalDistribution(transformed)
        result.with_source(
            Provenance(
                "pushforward",
                parents=(dist,),
                metadata={
                    "map": repr(transport_map),
                    "method": "sample",
                    "num_samples": n,
                },
            )
        )
        return result

    @property
    def priority(self):
        return -100  # lowest priority: last resort


# ── Module-level singleton ────────────────────────────────────────────

pushforward_registry = PushforwardRegistry()

# Register built-in rules
pushforward_registry.register(_BijectorChangeOfVariablesRule())
pushforward_registry.register(_SamplingFallbackRule())


# ── Helpers ───────────────────────────────────────────────────────────


class _FunctionalRule(PushforwardRule):
    """Rule created by the ``rule`` decorator."""

    def __init__(
        self,
        fn: Callable,
        *,
        map_type: type,
        dist_type: type,
        method: PushforwardMethod,
        _priority: int,
        description: str,
    ) -> None:
        self._fn = fn
        self._map_type = map_type
        self._dist_type = dist_type
        self._method = method
        self._priority = _priority
        self._description = description

    def map_types(self):
        return (self._map_type,)

    def dist_types(self):
        return (self._dist_type,)

    def check(self, transport_map, dist):
        return PushforwardInfo(
            feasible=True,
            method=self._method,
            description=self._description,
        )

    def apply(self, transport_map, dist, *, return_joint=False, **kwargs):
        return self._fn(transport_map, dist, **kwargs)

    @property
    def priority(self):
        return self._priority


class _CallableTransportMap:
    """Wraps a plain callable as a TransportMap-like object for pushforward dispatch.

    Uses duck-typing rather than inheriting from TransportMap to avoid
    circular import issues and because plain callables don't need the
    full TransportMap interface.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def forward(self, value):
        return self._fn(value)

    def __call__(self, value):
        return self._fn(value)

    def __repr__(self) -> str:
        name = getattr(self._fn, "__name__", repr(self._fn))
        return f"_CallableTransportMap({name})"


def _parse_strategy(strategy: str | None) -> PushforwardMethod | None:
    """Convert a strategy string to a PushforwardMethod, or None."""
    if strategy is None:
        return None
    mapping = {
        "closed_form": PushforwardMethod.CLOSED_FORM,
        "change_of_variables": PushforwardMethod.CHANGE_OF_VARIABLES,
        "sampling": PushforwardMethod.SAMPLE,
    }
    if strategy not in mapping:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Choose from: {list(mapping.keys())}"
        )
    return mapping[strategy]
