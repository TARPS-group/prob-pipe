"""Dispatch methods and registries.

A dispatch registry holds named implementations of one operation and
selects among them by the types of the arguments. Every method declares,
at registration and for its whole life, whether it is **exact**: whether
its result denotes the requested mathematical object or stands in for it.
Selection order is the same in every dispatch registry:

1. exact methods before approximate ones;
2. priority among methods of the same exactness, higher first;
3. registration order.

A method whose ``priority`` is ``None`` is **opt-in-only**: automatic
selection skips it and it runs only when named through ``method="..."``.
That is the default, so registering a method never changes what runs until
a contributor ranks it. ``set_priorities`` re-ranks at runtime; it cannot
change whether a method is exact, since exactness is not one of its inputs.

Two failures are distinct. :class:`ResolutionError` means no available
implementation under the requested controls; it is a ``LookupError``,
like the ``KeyError`` an unknown method name raises.
:class:`MathematicalDomainError` means the mathematical operation is known
to be undefined, a ``ValueError`` a method raises itself; the registry never
converts one into the other.

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


class ResolutionError(LookupError):
    """No available implementation under the requested controls.

    Raised by a registry when no registered method is feasible for the
    arguments, when the first candidate that is not infeasible is still
    unresolved, or when a method selected by name is infeasible. The
    message names the methods tried and what each was missing.

    A ``LookupError``, like the ``KeyError`` an unknown method name
    raises, so ``except LookupError`` catches both ways a dispatch can
    fail to select a method. It is not a ``TypeError``: well-typed
    arguments can still have no applicable method, as a model that no
    registered inference method supports does.
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

    ``feasible`` has three values. ``True``: the method applies. ``False``:
    it does not, and ``description`` says why, for example ``"needs a
    density"``. ``None``: the probe could not decide, because a declaration
    it reads is not yet available, and ``pending`` names those declarations.

    ``pending`` holds one entry per missing declaration, phrased as the
    thing whose arrival would settle the verdict: ``"output spec of f"``
    for a result declaration the return will complete, ``"dimension obs"``
    for a symbolic dimension no value has bound, ``"conversion plan for
    theta"`` for a converter not yet resolved. It is non-empty exactly when
    ``feasible`` is ``None``; a feasible or infeasible verdict carries no
    pending entries, so a reader never has to decide which of the two
    fields is authoritative.

    ``exact`` is the method's declared exactness. The registry sets it from
    the declaration whatever a ``check`` returned, so it is ``None`` only in
    a report that a method's ``check`` returned directly.
    """

    feasible: bool | None
    method_name: str = ""
    description: str = ""
    exact: bool | None = None
    pending: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.feasible is None and not self.pending:
            raise ValueError("an unresolved MethodInfo must name its pending declarations")
        if self.feasible is not None and self.pending:
            raise ValueError("only an unresolved MethodInfo carries pending declarations")

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

        ``None``, the default, is opt-in-only: excluded from automatic selection
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

    Everything that does not depend on how many arguments select the method
    is implemented here: registration, ranking, the opt-in filter,
    ``set_priorities``, and the ``check`` / ``execute`` path. Three hooks are
    left to the arity subclasses, and they are the only place arity enters:

    - :meth:`_cache_key` turns the positional arguments into the **dispatch
      key**, the type or types a method's ``supported_types`` is matched
      against, and raises ``TypeError`` when there are too few arguments;
    - :meth:`_find_methods` returns the auto-dispatchable methods whose
      ``supported_types`` admit a key, in selection order, memoized per key
      in ``_type_cache``, which :meth:`_sort_methods` clears whenever the
      order can change;
    - :meth:`_format_key` renders a key for error messages.

    Selection reads only the list ``_find_methods`` returns, so a subclass
    reimplements none of ranking, the opt-in filter, or the errors.

    ``check`` with no positional arguments returns an infeasible
    :class:`MethodInfo` rather than raising, since there is nothing to
    dispatch on; ``execute`` raises ``TypeError`` in the same case. Fewer
    arguments than the arity requires, for example one argument to a binary
    registry, is a contract violation and raises ``TypeError`` from both.
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
        in automatic selection. Exactness is not an input, so an override never
        lifts an approximate method above an exact one.
        """
        overrides: dict[str, int | None] = dict(priorities or {})
        for name, value in kwargs.items():
            if name in overrides:
                raise ValueError(f"Method {name!r} given both in the mapping and as a keyword")
            overrides[name] = value
        for name in overrides:
            if name not in self._name_index:
                available = ", ".join(sorted(self._name_index)) or "(none)"
                raise KeyError(f"No method named {name!r}. Available: {available}")
        for name, new_priority in overrides.items():
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
        self._priority_overrides.update(overrides)
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
        """Every registered method name, in selection order.

        Exact before approximate, then priority, then registration order.
        The listing ranks the methods; it does not say what a call runs. An
        opt-in-only method keeps its position although automatic selection
        skips it, so an exact opt-in-only method can be listed above the
        approximate method a call would pick. :meth:`check` reports what
        would run.
        """
        return [method.name for method in self._methods]

    def _is_auto_dispatchable(self, method: M) -> bool:
        return self._effective_priority(method) is not None

    @staticmethod
    def _passes_exact_only(method: M, exact_only: bool) -> bool:
        return method.exact or not exact_only

    def _candidates(self, args: tuple[Any, ...], exact_only: bool) -> list[M]:
        admitting = self._find_methods(self._cache_key(args))
        return [method for method in admitting if self._passes_exact_only(method, exact_only)]

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
            named = self.get_method(method)
            if not self._passes_exact_only(named, exact_only):
                return MethodInfo(
                    feasible=False,
                    method_name=method,
                    description=f"Method {method!r} is approximate and exact_only was requested",
                    exact=named.exact,
                )
            return self._with_exact(named, named.check(*args, **kwargs))
        if not args:
            return MethodInfo(feasible=False, description="No arguments provided")
        tried: list[str] = []
        for candidate in self._candidates(args, exact_only):
            info = self._with_exact(candidate, candidate.check(*args, **kwargs))
            if info.feasible is not False:
                return info
            tried.append(f"{candidate.name}: {info.description or 'infeasible'}")
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
            named = self.get_method(method)
            if not self._passes_exact_only(named, exact_only):
                raise ResolutionError(
                    f"Method {method!r} is approximate and exact_only was requested"
                )
            info = named.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {method!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if not info.feasible:
                raise ResolutionError(f"Method {method!r} is not applicable: {info.description}")
            return named.execute(*args, **kwargs)
        if not args:
            raise TypeError("No arguments provided for dispatch")
        tried: list[str] = []
        for candidate in self._candidates(args, exact_only):
            info = candidate.check(*args, **kwargs)
            if info.feasible is None:
                raise ResolutionError(
                    f"Method {candidate.name!r} is unresolved; pending: {', '.join(info.pending)}"
                )
            if info.feasible:
                return candidate.execute(*args, **kwargs)
            tried.append(f"{candidate.name}: {info.description or 'infeasible'}")
        raise ResolutionError(self._no_method_message(args, tried, exact_only))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _with_exact(method: M, info: MethodInfo) -> MethodInfo:
        """Name the method and set ``exact`` from its declaration.

        ``exact`` is taken from the method, never from the check, so a
        report cannot disagree with what ``exact_only`` filters on.
        """
        return MethodInfo(
            feasible=info.feasible,
            method_name=info.method_name or method.name,
            description=info.description,
            exact=method.exact,
            pending=info.pending,
        )

    def _no_method_message(self, args: tuple[Any, ...], tried: list[str], exact_only: bool) -> str:
        key = self._format_key(self._cache_key(args))
        restriction = " with exact_only" if exact_only else ""
        if tried:
            return f"No feasible method for {key}{restriction}. Tried: " + "; ".join(tried)
        return f"No method registered for {key}{restriction}. Available: {self.list_methods()}"

    @abstractmethod
    def _cache_key(self, args: tuple[Any, ...]) -> Any:
        """The dispatch key for the positional arguments.

        A type for a unary registry, a pair of types for a binary one; the
        key is what ``supported_types`` is matched against and what the
        method cache is indexed by. Raises ``TypeError`` when ``args`` has
        fewer entries than the arity needs.
        """
        ...

    @abstractmethod
    def _find_methods(self, key: Any) -> list[M]:
        """The auto-dispatchable methods admitting ``key``, in selection order.

        Filters ``self._methods``, which :meth:`_sort_methods` keeps in
        selection order, to those passing :meth:`_is_auto_dispatchable` and
        whose ``supported_types`` admit the key by ``issubclass``, and
        memoizes the result in ``self._type_cache[key]``.
        """
        ...

    @abstractmethod
    def _format_key(self, key: Any) -> str:
        """``key`` as it appears in error messages, such as ``(Left, Right)``."""
        ...


class UnaryDispatchRegistry[M: UnaryDispatchMethod](BaseDispatchRegistry[M]):
    """Dispatches on the type of the first positional argument."""

    def _cache_key(self, args: tuple[Any, ...]) -> type:
        return type(args[0])

    def _find_methods(self, key: type) -> list[M]:
        if key not in self._type_cache:
            self._type_cache[key] = [
                method
                for method in self._methods
                if self._is_auto_dispatchable(method)
                and any(issubclass(key, supported) for supported in method.supported_types())
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
            left_type, right_type = key
            matches: list[M] = []
            for method in self._methods:
                if not self._is_auto_dispatchable(method):
                    continue
                supported_left, supported_right = method.supported_types()
                if any(issubclass(left_type, left) for left in supported_left) and any(
                    issubclass(right_type, right) for right in supported_right
                ):
                    matches.append(method)
            self._type_cache[key] = matches
        return self._type_cache[key]

    def _format_key(self, key: tuple[type, type]) -> str:
        return f"({key[0].__name__}, {key[1].__name__})"
