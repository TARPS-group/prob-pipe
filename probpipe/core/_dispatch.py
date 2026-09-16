"""Dispatch methods and registries.

A dispatch registry holds named implementations of one operation and
selects among them by the types of the arguments. Every method declares,
at registration and for its whole life, whether it is **exact**: whether
its result denotes the requested mathematical object or stands in for it.
Selection order is the same in every registry:

1. exact methods before approximate ones;
2. priority among methods of the same exactness, higher first;
3. registration order.

A method whose ``priority`` is ``None`` is **opt-in-only**: the auto walk
skips it and it runs only when named through ``method="..."``. That is the
default, so registering a method never changes what runs until a
contributor ranks it. ``set_priorities`` re-ranks at runtime; it cannot
change whether a method is exact, since exactness is not one of its inputs.

Two failures are distinct. :class:`ResolutionError` means no available
implementation under the requested controls. :class:`MathematicalDomainError`
means the mathematical operation is known to be undefined, a ``ValueError``
a method raises itself; the registry never converts one into the other.

The two arity subclasses differ only in how they compute the dispatch key
from the arguments and how they pre-filter methods by ``supported_types()``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BaseDispatchMethod",
    "BaseDispatchRegistry",
    "BinaryDispatchMethod",
    "BinaryDispatchRegistry",
    "MathematicalDomainError",
    "MethodInfo",
    "ResolutionError",
    "UnaryDispatchMethod",
    "UnaryDispatchRegistry",
]


class ResolutionError(Exception):
    """No available implementation under the requested controls.

    Raised by a registry when no registered method is feasible for the
    arguments, when the first candidate that is not infeasible is still
    unresolved, or when a method selected by name is infeasible. The
    message names the methods tried and what each was missing.
    """


class MathematicalDomainError(ValueError):
    """The mathematical operation is known to be undefined.

    Raised by a method that can establish nonexistence, for example a
    requested mean that does not exist. Failure to establish existence is
    not nonexistence: that is a :class:`ResolutionError`.
    """


@dataclass(frozen=True)
class MethodInfo:
    """What a method's ``check`` reports about one call.

    ``feasible`` is ``True`` when the method applies, ``False`` when it
    does not, and ``None`` when the declarations the probe needs are not
    yet available, in which case ``pending`` names them. ``exact`` is the
    method's local guarantee for this call, ``None`` while undetermined.
    """

    feasible: bool | None
    method_name: str = ""
    description: str = ""
    exact: bool | None = None
    pending: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.feasible is True and self.pending:
            raise ValueError("a feasible MethodInfo cannot carry pending requirements")
        if self.feasible is None and not self.pending:
            raise ValueError("an unresolved MethodInfo must name its pending requirements")

    @property
    def unresolved(self) -> bool:
        """``True`` when feasibility awaits declarations not yet available."""
        return self.feasible is None


# ---------------------------------------------------------------------------
# Abstract method bases
# ---------------------------------------------------------------------------


class BaseDispatchMethod(ABC):
    """Abstract base for all pluggable dispatch methods.

    A subclass declares a unique ``name``, whether it is ``exact``, and
    ``check`` / ``execute``. ``supported_types`` is declared by the arity
    subclasses, since its shape depends on the arity.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this method, for example ``'blackjax_nuts'``."""
        ...

    @property
    @abstractmethod
    def exact(self) -> bool:
        """Whether the result denotes the requested mathematical object.

        ``True`` for an equivalent representation or an exact draw in law;
        ``False`` for a stand-in, such as a finite Monte Carlo output or a
        moment-matched family. Declared once and fixed for the method's life.
        """
        ...

    @abstractmethod
    def check(self, *args: Any, **kwargs: Any) -> MethodInfo:
        """Probe feasibility without significant computation."""
        ...

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Run the method and return the result."""
        ...

    @property
    def priority(self) -> int | None:
        """Rank among methods of the same exactness, higher first.

        ``None``, the default, is opt-in-only: excluded from the auto walk
        and reachable only by name.
        """
        return None


class UnaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for single-argument dispatch methods."""

    @abstractmethod
    def supported_types(self) -> tuple[type, ...]:
        """Types this method can operate on, used as an ``issubclass`` pre-filter.

        Must return concrete classes, since ``issubclass`` does not work
        reliably with protocols carrying non-method members. Protocol-based
        feasibility belongs in ``check``.
        """
        ...


class BinaryDispatchMethod(BaseDispatchMethod):
    """Abstract base for two-argument dispatch methods."""

    @abstractmethod
    def supported_types(self) -> tuple[tuple[type, ...], tuple[type, ...]]:
        """``((left_types, ...), (right_types, ...))`` for the ``issubclass`` pre-filter."""
        ...


# ---------------------------------------------------------------------------
# Registry base
# ---------------------------------------------------------------------------


class BaseDispatchRegistry[M: BaseDispatchMethod](ABC):
    """Arity-independent registry logic.

    Holds registration, ranking, the opt-in filter, and the ``check`` /
    ``execute`` walk. The arity subclasses supply :meth:`_cache_key`,
    :meth:`_find_methods`, and :meth:`_format_key`.

    ``check`` with no positional arguments returns an infeasible
    :class:`MethodInfo`, the natural "is anything dispatchable here?"
    probe. Fewer arguments than the arity requires is a contract violation
    and raises ``TypeError`` from both ``check`` and ``execute``.
    """

    def __init__(self) -> None:
        self._methods: list[M] = []
        self._name_index: dict[str, M] = {}
        self._priority_overrides: dict[str, int | None] = {}
        self._registration_order: dict[str, int] = {}
        self._type_cache: dict[Any, list[M]] = {}

    # -- registration -------------------------------------------------------

    def register(self, method: M) -> None:
        """Register a method.

        Raises ``ValueError`` for an empty or duplicate name and ``TypeError``
        when the method does not declare a boolean ``exact``.
        """
        if not method.name:
            raise ValueError(f"Method.name must be a non-empty string; got {method.name!r}")
        if method.name in self._name_index:
            raise ValueError(f"Method name {method.name!r} is already registered")
        exact = method.exact
        if type(exact) is not bool:
            raise TypeError(f"Method {method.name!r} must declare exact as a bool; got {exact!r}")
        self._registration_order[method.name] = len(self._registration_order)
        self._methods.append(method)
        self._name_index[method.name] = method
        self._sort_methods()

    # -- ranking ------------------------------------------------------------

    def _effective_priority(self, method: M) -> int | None:
        return self._priority_overrides.get(method.name, method.priority)

    def _sort_key(self, method: M) -> tuple[int, int, int, int]:
        priority = self._effective_priority(method)
        return (
            0 if method.exact else 1,
            1 if priority is None else 0,
            -(priority or 0),
            self._registration_order[method.name],
        )

    def _sort_methods(self) -> None:
        self._methods.sort(key=self._sort_key)
        self._type_cache.clear()

    def set_priorities(
        self,
        priorities: Mapping[str, int | None] | None = None,
        /,
        **kwargs: int | None,
    ) -> None:
        """Override the rank of one or more methods.

        Accepts a mapping, keywords, or both; the mapping form is for method
        names that are not Python identifiers. A name in both raises
        ``ValueError``, an unknown name raises ``KeyError``, and in either
        case nothing is applied. Overrides are recorded on the registry and
        never mutate a method. A move between ``None`` and an integer emits
        a ``UserWarning``, since it changes whether the method participates
        in the auto walk. Exactness is not an input, so an override never
        lifts an approximate method above an exact one.
        """
        merged: dict[str, int | None] = dict(priorities or {})
        for name, value in kwargs.items():
            if name in merged:
                raise ValueError(f"Method {name!r} given both in the mapping and as a keyword")
            merged[name] = value
        for name in merged:
            if name not in self._name_index:
                available = ", ".join(sorted(self._name_index)) or "(none)"
                raise KeyError(f"No method named {name!r}. Available: {available}")
        for name, new_priority in merged.items():
            old_priority = self._effective_priority(self._name_index[name])
            if (old_priority is None) != (new_priority is None):
                direction = "out of opt-in-only" if old_priority is None else "into opt-in-only"
                warnings.warn(
                    f"Priority override for {name!r} moves it {direction} "
                    f"({old_priority} -> {new_priority}); auto-dispatch "
                    f"participation changes accordingly.",
                    UserWarning,
                    stacklevel=2,
                )
        self._priority_overrides.update(merged)
        self._sort_methods()

    # -- query --------------------------------------------------------------

    def get_method(self, name: str) -> M:
        """Look up a method by name; ``KeyError`` when it is not registered."""
        try:
            return self._name_index[name]
        except KeyError:
            available = ", ".join(sorted(self._name_index)) or "(none)"
            raise KeyError(f"No method named {name!r}. Available: {available}") from None

    def list_methods(self) -> list[str]:
        """Method names in selection order."""
        return [m.name for m in self._methods]

    def _is_auto_dispatchable(self, m: M) -> bool:
        return self._effective_priority(m) is not None

    @staticmethod
    def _passes_floor(m: M, exact_only: bool) -> bool:
        return m.exact or not exact_only

    def _candidates(self, args: tuple[Any, ...], exact_only: bool) -> list[M]:
        key = self._cache_key(args)
        found = self._find_methods(key)
        return [m for m in found if self._passes_floor(m, exact_only)]

    def check(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **kwargs: Any,
    ) -> MethodInfo:
        """Report which method would run, without running anything.

        Returns the first candidate in selection order whose ``check`` is
        feasible or unresolved; an unresolved candidate above a feasible one
        is reported as unresolved, since a probe may not claim a method that
        may not run. When no candidate is feasible the result is infeasible
        and its description names every method tried.
        """
        if method is not None:
            m = self.get_method(method)
            if not self._passes_floor(m, exact_only):
                return MethodInfo(
                    feasible=False,
                    method_name=method,
                    description=f"Method {method!r} is approximate and exact_only was requested",
                    exact=m.exact,
                )
            return self._with_exact(m, m.check(*args, **kwargs))
        if not args:
            return MethodInfo(feasible=False, description="No arguments provided")
        tried: list[str] = []
        for m in self._candidates(args, exact_only):
            info = self._with_exact(m, m.check(*args, **kwargs))
            if info.feasible is not False:
                return info
            tried.append(f"{m.name}: {info.description or 'infeasible'}")
        return MethodInfo(
            feasible=False,
            description=self._no_method_message(args, tried, exact_only),
        )

    def execute(
        self,
        *args: Any,
        method: str | None = None,
        exact_only: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run the selected method.

        Auto-selection runs the first feasible candidate in selection order.
        A candidate that is unresolved raises :class:`ResolutionError`
        naming its pending requirements, as does the absence of any feasible
        candidate. A named method that is infeasible or below the requested
        exactness raises :class:`ResolutionError`; an unknown name raises
        ``KeyError``. An exception raised by the method itself propagates
        unchanged, and no other method is tried after it.
        """
        if method is not None:
            m = self.get_method(method)
            if not self._passes_floor(m, exact_only):
                raise ResolutionError(
                    f"Method {method!r} is approximate and exact_only was requested"
                )
            info = m.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {method!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if not info.feasible:
                raise ResolutionError(f"Method {method!r} is not applicable: {info.description}")
            return m.execute(*args, **kwargs)
        if not args:
            raise TypeError("No arguments provided for dispatch")
        tried: list[str] = []
        for m in self._candidates(args, exact_only):
            info = m.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {m.name!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if info.feasible:
                return m.execute(*args, **kwargs)
            tried.append(f"{m.name}: {info.description or 'infeasible'}")
        raise ResolutionError(self._no_method_message(args, tried, exact_only))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _with_exact(m: M, info: MethodInfo) -> MethodInfo:
        """Fill ``method_name`` and ``exact`` from the method when the check left them."""
        if info.method_name and info.exact is not None:
            return info
        return MethodInfo(
            feasible=info.feasible,
            method_name=info.method_name or m.name,
            description=info.description,
            exact=m.exact if info.exact is None else info.exact,
            pending=info.pending,
        )

    def _no_method_message(self, args: tuple[Any, ...], tried: list[str], exact_only: bool) -> str:
        key = self._format_key(self._cache_key(args))
        floor = " with exact_only" if exact_only else ""
        if tried:
            return f"No feasible method for {key}{floor}. Tried: " + "; ".join(tried)
        return f"No method registered for {key}{floor}. Available: {self.list_methods()}"

    @abstractmethod
    def _cache_key(self, args: tuple[Any, ...]) -> Any:
        """The dispatch key for the positional arguments."""
        ...

    @abstractmethod
    def _find_methods(self, key: Any) -> list[M]:
        """Auto-dispatchable methods matching the key, in selection order, cached."""
        ...

    @abstractmethod
    def _format_key(self, key: Any) -> str:
        """The key as it appears in error messages."""
        ...


class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]):
    """Dispatches on the type of the first positional argument."""

    def _cache_key(self, args: tuple[Any, ...]) -> type:
        return type(args[0])

    def _find_methods(self, key: type) -> list[M]:
        if key not in self._type_cache:
            self._type_cache[key] = [
                m
                for m in self._methods
                if self._is_auto_dispatchable(m)
                and any(issubclass(key, st) for st in m.supported_types())
            ]
        return self._type_cache[key]

    def _format_key(self, key: type) -> str:
        return key.__name__


class BinaryDispatchRegistry[M: BinaryDispatchMethod](BaseDispatchRegistry[M]):
    """Dispatches on the joint type of the first two positional arguments.

    A method matches when the left type is a subclass of one of its left
    types and the right type of one of its right types.
    """

    def _cache_key(self, args: tuple[Any, ...]) -> tuple[type, type]:
        if len(args) < 2:
            raise TypeError(
                "BinaryDispatchRegistry requires at least two positional "
                f"arguments; got {len(args)}"
            )
        return (type(args[0]), type(args[1]))

    def _find_methods(self, key: tuple[type, type]) -> list[M]:
        if key not in self._type_cache:
            tl, tr = key
            matches: list[M] = []
            for m in self._methods:
                if not self._is_auto_dispatchable(m):
                    continue
                st = m.supported_types()
                if any(issubclass(tl, lt) for lt in st[0]) and any(
                    issubclass(tr, rt) for rt in st[1]
                ):
                    matches.append(m)
            self._type_cache[key] = matches
        return self._type_cache[key]

    def _format_key(self, key: tuple[type, type]) -> str:
        return f"({key[0].__name__}, {key[1].__name__})"
