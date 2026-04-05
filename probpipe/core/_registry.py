"""Generic priority-based method registry.

Provides a reusable base for registries that dispatch operations
to pluggable methods based on priority and feasibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["Method", "MethodInfo", "MethodRegistry"]


@dataclass(frozen=True)
class MethodInfo:
    """Metadata describing whether a method is applicable.

    Returned by a method's ``check()`` to describe feasibility
    without performing the actual computation.
    """

    feasible: bool
    method_name: str = ""
    description: str = ""


class Method(ABC):
    """Abstract base for a pluggable method in a registry.

    Subclasses declare a unique ``name``, which ``supported_types``
    they handle (for fast filtering), a ``priority`` (higher =
    tried first), and implement ``check``/``execute``.
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

    @abstractmethod
    def check(self, *args: Any, **kwargs: Any) -> MethodInfo:
        """Probe whether this method is applicable (must be cheap)."""
        ...

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the method and return the result."""
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
    """

    def __init__(self) -> None:
        self._methods: list[M] = []
        self._name_index: dict[str, M] = {}
        self._priority_overrides: dict[str, int] = {}
        self._type_cache: dict[type, list[M]] = {}

    # -- registration -------------------------------------------------------

    def register(self, method: M) -> None:
        """Register a method (invalidates the lookup cache)."""
        if method.name in self._name_index:
            raise ValueError(
                f"Method name {method.name!r} is already registered"
            )
        self._methods.append(method)
        self._name_index[method.name] = method
        self._sort_methods()

    # -- priority management ------------------------------------------------

    def _effective_priority(self, method: M) -> int:
        """Return the effective priority (override if set, else default)."""
        return self._priority_overrides.get(method.name, method.priority)

    def _sort_methods(self) -> None:
        """Re-sort methods by effective priority and clear caches."""
        self._methods.sort(key=self._effective_priority, reverse=True)
        self._type_cache.clear()

    def set_priorities(self, **name_to_priority: int) -> None:
        """Override the priority of one or more methods.

        Higher priority methods are tried first during auto-selection.

        Parameters
        ----------
        **name_to_priority
            Keyword arguments mapping method names to new priorities.
            e.g., ``set_priorities(tfp_rwmh=200, tfp_nuts=50)``

        Raises
        ------
        KeyError
            If a method name is not registered.
        """
        for name in name_to_priority:
            if name not in self._name_index:
                available = ", ".join(sorted(self._name_index)) or "(none)"
                raise KeyError(
                    f"No method named {name!r}. Available: {available}"
                )
        self._priority_overrides.update(name_to_priority)
        self._sort_methods()

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

    def check(
        self, *args: Any, method: str | None = None, **kwargs: Any
    ) -> MethodInfo:
        """Check feasibility.  Auto-selects or uses the named method."""
        if method is not None:
            m = self.get_method(method)
            return m.check(*args, **kwargs)

        key_type = type(args[0]) if args else object
        for m in self._find_methods(key_type):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return info

        return MethodInfo(
            feasible=False,
            description=f"No applicable method for {key_type.__name__}",
        )

    def execute(
        self, *args: Any, method: str | None = None, **kwargs: Any
    ) -> Any:
        """Execute using the best (or named) method.

        Raises ``TypeError`` if no method is applicable.
        Raises ``KeyError`` if *method* is not registered.
        """
        if method is not None:
            m = self.get_method(method)
            info = m.check(*args, **kwargs)
            if not info.feasible:
                raise TypeError(
                    f"Method {method!r} is not applicable: {info.description}"
                )
            return m.execute(*args, **kwargs)

        key_type = type(args[0]) if args else object
        for m in self._find_methods(key_type):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return m.execute(*args, **kwargs)

        raise TypeError(
            f"No method registered for {key_type.__name__}. "
            f"Available: {self.list_methods()}"
        )

    # -- internals ----------------------------------------------------------

    def _find_methods(self, key_type: type) -> list[M]:
        """Return methods whose ``supported_types`` match *key_type* (cached).

        Uses ``issubclass`` rather than ``isinstance`` because we are
        comparing *types*, not instances.  ``supported_types()`` must
        return concrete classes (not protocols with non-method members),
        since ``issubclass`` does not work reliably with such protocols.
        """
        if key_type not in self._type_cache:
            self._type_cache[key_type] = [
                m for m in self._methods
                if any(issubclass(key_type, st) for st in m.supported_types())
            ]
        return self._type_cache[key_type]
