"""Generic priority-based method registry.

Provides a reusable base for registries that dispatch operations
to pluggable methods based on priority and feasibility.  Concrete
registries (e.g., ``InferenceMethodRegistry``) extend this with
domain-specific ``check``/``execute`` signatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodInfo:
    """Metadata describing whether a method is applicable.

    Returned by a method's ``check()`` to describe feasibility
    without performing the actual computation.
    """

    feasible: bool
    method_name: str = ""
    description: str = ""


# Sentinel for "no method found"
_NOT_FEASIBLE = MethodInfo(feasible=False, description="No applicable method found")


class Method(ABC):
    """Abstract base for a pluggable method in a registry.

    Subclasses declare a unique ``name``, which ``supported_types``
    they handle (for fast filtering), and a ``priority`` (higher =
    tried first).  The concrete ``check`` and ``execute`` signatures
    are defined by each registry's method subclass.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this method (e.g., ``'tfp_nuts'``)."""
        ...

    @abstractmethod
    def supported_types(self) -> tuple[type, ...]:
        """Types this method can operate on (fast pre-filter)."""
        ...

    @property
    def priority(self) -> int:
        """Higher priority methods are tried first.  Default 0."""
        return 0


class MethodRegistry[M: Method]:
    """Generic priority-based method registry.

    Methods are tried in descending priority order.  The first method
    whose ``check()`` returns ``feasible=True`` wins.  Users can also
    select a specific method by name.

    Subclasses should add a domain-specific ``execute``-style method
    (e.g., ``condition``, ``compute``) that delegates to the matched
    method.
    """

    def __init__(self) -> None:
        self._methods: list[M] = []
        self._name_index: dict[str, M] = {}
        self._type_cache: dict[type, list[M]] = {}

    # -- registration -------------------------------------------------------

    def register(self, method: M) -> None:
        """Register a method (invalidates the lookup cache)."""
        if method.name in self._name_index:
            raise ValueError(
                f"Method name {method.name!r} is already registered"
            )
        self._methods.append(method)
        self._methods.sort(key=lambda m: m.priority, reverse=True)
        self._name_index[method.name] = method
        self._type_cache.clear()

    # -- query --------------------------------------------------------------

    def get_method(self, name: str) -> M:
        """Look up a method by name.  Raises ``KeyError`` if not found."""
        try:
            return self._name_index[name]
        except KeyError:
            available = ", ".join(sorted(self._name_index)) or "(none)"
            raise KeyError(
                f"No method named {name!r}. Available: {available}"
            ) from None

    def list_methods(self) -> list[str]:
        """Return method names in priority order (highest first)."""
        return [m.name for m in self._methods]

    # -- internals ----------------------------------------------------------

    def _find_methods(self, key_type: type) -> list[M]:
        """Return methods whose ``supported_types`` match *key_type* (cached)."""
        if key_type not in self._type_cache:
            self._type_cache[key_type] = [
                m for m in self._methods
                if any(issubclass(key_type, st) for st in m.supported_types())
            ]
        return self._type_cache[key_type]
