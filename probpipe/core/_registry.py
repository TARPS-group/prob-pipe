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
  inherited from :class:`BaseDispatchMethod`, so a newly-registered
  method gets the safe behaviour (opt-in) until a contributor
  classifies it explicitly.

See ``docs/api/extending.md`` for the tier criteria a contributor
should use when choosing a priority for a new method.

**Arity-typed subclasses.** :class:`UnaryDispatchRegistry` and
:class:`BinaryDispatchRegistry` differ only in ``supported_types()``
signature, cache-key construction, and the pre-filter predicate.  Arity
is a type-level property: ``BinaryDispatchRegistry[KLMethod].execute(p)``
is a static type error rather than a runtime one.

``Method`` and ``MethodRegistry`` are aliases for the unary subclasses
and preserve the existing public API without change.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BaseDispatchMethod",
    "UnaryDispatchMethod",
    "BinaryDispatchMethod",
    "BaseDispatchRegistry",
    "UnaryDispatchRegistry",
    "BinaryDispatchRegistry",
    "Method",
    "MethodInfo",
    "MethodRegistry",
    "OPT_IN_ONLY_PRIORITY",
]

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


# ---------------------------------------------------------------------------
# Abstract method bases
# ---------------------------------------------------------------------------


class BaseDispatchMethod(ABC):
    """Abstract base for all pluggable dispatch methods.

    Subclasses declare a unique ``name``, a ``priority``, and implement
    ``check``/``execute``.  The ``supported_types`` method is **not**
    defined here — its return shape is arity-dependent and is declared
    by :class:`UnaryDispatchMethod` and :class:`BinaryDispatchMethod`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this method (e.g., ``'tfp_nuts'``)."""
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


class UnaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for single-argument dispatch methods.

    Adds :meth:`supported_types`, which returns the nominal types this
    method handles.  Used by :class:`UnaryDispatchRegistry` as a fast
    ``issubclass`` pre-filter before ``check()`` is called.
    """

    @abstractmethod
    def supported_types(self) -> tuple[type, ...]:
        """Types this method can operate on (fast pre-filter).

        Must return concrete classes, not protocols with non-method
        members — ``issubclass`` does not work reliably with such
        protocols.  Protocol-based feasibility checks belong in
        ``check()``.
        """
        ...


class BinaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for two-argument dispatch methods.

    Adds :meth:`supported_types`, which returns a pair of type-tuples
    ``((left_types, ...), (right_types, ...))`` used by
    :class:`BinaryDispatchRegistry` as a fast ``issubclass`` pre-filter.
    """

    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """Left and right type-tuples for the pre-filter.

        Returns a 2-tuple of type-tuples: the first covers the left
        argument, the second the right.  Example::

            def supported_types(self):
                return ((Normal,), (Normal,))

        Must contain concrete classes, not protocols — see the note on
        :meth:`UnaryDispatchMethod.supported_types`.
        """
        ...


# ---------------------------------------------------------------------------
# Registry base and arity-typed subclasses
# ---------------------------------------------------------------------------


class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    """Priority-based dispatch registry (arity-independent base).

    Holds all arity-independent logic: registration, priority
    management, opt-in filtering, and the ``check``/``execute`` loop.
    Arity-specific subclasses override :meth:`_cache_key`,
    :meth:`_find_methods`, and :meth:`_format_key`.

    Methods whose effective priority equals :data:`OPT_IN_ONLY_PRIORITY`
    (``0``) are excluded from auto-dispatch and are reachable only by
    name via ``method="..."``.
    """

    def __init__(self) -> None:
        self._methods: list[M] = []
        self._name_index: dict[str, M] = {}
        self._priority_overrides: dict[str, int] = {}
        self._type_cache: dict[Any, list[M]] = {}

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

    def _is_auto_dispatchable(self, m: M) -> bool:
        """True if *m* participates in auto-dispatch.

        A method whose effective priority equals
        :data:`OPT_IN_ONLY_PRIORITY` is excluded from the auto-dispatch
        walk.  It remains reachable by explicit ``method="..."``
        regardless.  Subclasses must call this inside
        :meth:`_find_methods` to keep filtering consistent.
        """
        return self._effective_priority(m) != OPT_IN_ONLY_PRIORITY

    def check(
        self, *args: Any, method: str | None = None, **kwargs: Any
    ) -> MethodInfo:
        """Check feasibility.  Auto-selects or uses the named method."""
        if method is not None:
            m = self.get_method(method)
            return m.check(*args, **kwargs)

        if not args:
            return MethodInfo(feasible=False, description="No arguments provided")

        key = self._cache_key(args)
        for m in self._find_methods(key):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return info

        return MethodInfo(
            feasible=False,
            description=f"No applicable method for {self._format_key(key)}",
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

        if not args:
            raise TypeError("No arguments provided for dispatch")

        key = self._cache_key(args)
        for m in self._find_methods(key):
            info = m.check(*args, **kwargs)
            if info.feasible:
                return m.execute(*args, **kwargs)

        raise TypeError(
            f"No method registered for {self._format_key(key)}. "
            f"Available: {self.list_methods()}"
        )

    # -- internals (arity-specific overrides) --------------------------------

    @abstractmethod
    def _cache_key(self, args: tuple) -> Any:
        """Compute the dispatch cache key from positional arguments."""
        ...

    @abstractmethod
    def _find_methods(self, key: Any) -> list[M]:
        """Return auto-dispatchable methods matching *key* (cached).

        Implementations must:

        1. Consult ``self._type_cache`` to avoid recomputation.
        2. Call ``self._is_auto_dispatchable(m)`` to exclude opt-in-only
           methods (Approach A: single point of truth for the filter).
        3. Apply an ``issubclass`` pre-filter via ``supported_types()``.
        """
        ...

    @abstractmethod
    def _format_key(self, key: Any) -> str:
        """Format *key* as a human-readable string for error messages."""
        ...


class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]):
    """Priority-based registry for single-argument dispatch.

    Dispatches on the type of the first positional argument.
    ``MethodRegistry`` is an alias for this class.
    """

    def _cache_key(self, args: tuple) -> type:
        return type(args[0])

    def _find_methods(self, key: type) -> list[M]:
        """Return auto-dispatchable methods matching *key* (cached).

        Applies an ``issubclass`` pre-filter on ``supported_types()``
        and excludes opt-in-only methods via
        :meth:`~BaseDispatchRegistry._is_auto_dispatchable`.

        Uses ``issubclass`` rather than ``isinstance`` because we are
        comparing *types*, not instances.  ``supported_types()`` must
        return concrete classes (not protocols with non-method members),
        since ``issubclass`` does not work reliably with such protocols.
        """
        if key not in self._type_cache:
            self._type_cache[key] = [
                m for m in self._methods
                if self._is_auto_dispatchable(m)
                and any(issubclass(key, st) for st in m.supported_types())
            ]
        return self._type_cache[key]

    def _format_key(self, key: type) -> str:
        return key.__name__


class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]):
    """Priority-based registry for two-argument dispatch.

    Dispatches on the joint type of the first two positional arguments.
    Each registered method's ``supported_types()`` must return a 2-tuple
    of type-tuples: ``((left_types, ...), (right_types, ...))``.
    """

    def _cache_key(self, args: tuple) -> tuple[type, type]:
        return (type(args[0]), type(args[1]))

    def _find_methods(self, key: tuple[type, type]) -> list[M]:
        """Return auto-dispatchable methods matching the type pair *key* (cached).

        Both left and right type slots must pass an ``issubclass`` check
        against ``supported_types()[0]`` and ``supported_types()[1]``
        respectively.  Opt-in-only methods are excluded via
        :meth:`~BaseDispatchRegistry._is_auto_dispatchable`.
        """
        if key not in self._type_cache:
            tl, tr = key
            self._type_cache[key] = [
                m for m in self._methods
                if self._is_auto_dispatchable(m)
                and any(issubclass(tl, lt) for lt in m.supported_types()[0])
                and any(issubclass(tr, rt) for rt in m.supported_types()[1])
            ]
        return self._type_cache[key]

    def _format_key(self, key: tuple[type, type]) -> str:
        return f"({key[0].__name__}, {key[1].__name__})"


# ---------------------------------------------------------------------------
# Backward-compatible aliases — preserve the existing public API
# ---------------------------------------------------------------------------

#: Alias for :class:`UnaryDispatchMethod`.
Method = UnaryDispatchMethod

#: Alias for :class:`UnaryDispatchRegistry`.
MethodRegistry = UnaryDispatchRegistry
