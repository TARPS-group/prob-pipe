"""Generic priority-based method registry.

Provides a reusable base for registries that dispatch operations
to pluggable methods based on priority and feasibility.

Priority semantics (see issue #189):

- ``priority > 50`` — *exact* methods. Auto-dispatched in descending
  priority order.
- ``0 < priority <= 50`` — *inexact* methods. Auto-dispatched in
  descending priority order; the ``50`` break is documentary, not
  behavioural, so the registry walks every positive priority uniformly.
- ``priority == 0`` — *opt-in only.* Skipped during auto-dispatch;
  selectable by name via ``method="..."``. This is the default value
  inherited from :class:`Method`, so a newly-registered method gets
  the safe behaviour (opt-in) until a contributor classifies it
  explicitly.

See ``docs/api/extending.md`` for the tier criteria a contributor
should use when choosing a priority for a new method.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = ["Method", "MethodInfo", "MethodRegistry"]

# Priority value for methods that should not auto-dispatch. The registry
# excludes any method whose effective priority equals this sentinel from
# the auto-selection walk; such a method is reachable only by name via
# ``method=...``.
OPT_IN_ONLY_PRIORITY: int = 0


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
        """Auto-dispatch ordering.

        Higher priority methods are tried first during auto-selection.
        The default of ``0`` is the opt-in-only sentinel — a method
        that doesn't override this property is reachable only by name
        via ``method="..."``. See the module docstring for the
        ``> 50`` / ``<= 50`` / ``== 0`` convention contributors should
        follow when choosing a number.
        """
        return OPT_IN_ONLY_PRIORITY


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
        Overrides are unrestricted: the new value can be any integer,
        including the opt-in-only sentinel ``0``. When an override
        moves a method *into* or *out of* ``0``, the registry emits a
        :class:`UserWarning` because that crossing changes whether the
        method participates in auto-dispatch at all.

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
        for name, new_priority in name_to_priority.items():
            old_priority = self._effective_priority(self._name_index[name])
            if (old_priority == OPT_IN_ONLY_PRIORITY) != (
                new_priority == OPT_IN_ONLY_PRIORITY
            ):
                direction = (
                    "out of opt-in-only"
                    if old_priority == OPT_IN_ONLY_PRIORITY
                    else "into opt-in-only"
                )
                warnings.warn(
                    f"Priority override for {name!r} moves it {direction} "
                    f"({old_priority} -> {new_priority}); auto-dispatch "
                    f"participation changes accordingly.",
                    UserWarning,
                    stacklevel=2,
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
        """Return auto-dispatchable methods matching *key_type* (cached).

        Filters by:

        - ``supported_types`` containing *key_type* (fast pre-filter).
        - effective priority not equal to ``OPT_IN_ONLY_PRIORITY`` —
          opt-in-only methods are excluded from the auto-dispatch
          walk and are only reachable by explicit ``method="..."``.

        Uses ``issubclass`` rather than ``isinstance`` because we are
        comparing *types*, not instances.  ``supported_types()`` must
        return concrete classes (not protocols with non-method members),
        since ``issubclass`` does not work reliably with such protocols.
        """
        if key_type not in self._type_cache:
            self._type_cache[key_type] = [
                m for m in self._methods
                if self._effective_priority(m) != OPT_IN_ONLY_PRIORITY
                and any(issubclass(key_type, st) for st in m.supported_types())
            ]
        return self._type_cache[key_type]
